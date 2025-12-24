# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Megatron benchmark evaluation system.

This module provides a unified framework for evaluating language models on downstream tasks
during training. It supports two main task types:
1. Multiple Choice: Tasks with discrete answer choices (e.g., MMLU, ARC, HellaSwag)
2. Perplexity: Generative tasks evaluated by perplexity (e.g., WikiText, LAMBADA)

The design follows Megatron-Core patterns with:
- Configuration-driven task definitions using dataclasses
- Clear class hierarchies with abstract base classes
- Minimal global state
- Modular and extensible architecture
"""

import os
import re
import gzip
import json
import math
import torch
import logging

from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass, field
from functools import partial
from typing import Any, Dict, List, Optional, Tuple

from megatron.core import parallel_state
from megatron.core.pipeline_parallel import get_forward_backward_func

from .global_vars import get_args, get_tokenizer, get_tensorboard_writer, get_wandb_writer
from .utils import get_batch_on_this_tp_rank, get_ltor_masks_and_position_ids, print_rank_0

log = logging.getLogger(__name__)

# ============================================================================
# Helper Functions
# ============================================================================

ANSWER_CHOICES = ["A", "B", "C", "D"]

def _is_logging_rank() -> bool:
    """
    Check if this is the logging rank.
    """
    return (
        parallel_state.is_pipeline_last_stage() and
        parallel_state.get_data_parallel_rank() == 0
    )

def _load_benchmark(path: str) -> List[Dict[str, Any]]:
    """Load benchmark requests from jsonl.gz file."""
    requests = []

    open_fn = gzip.open if path.endswith(".gz") else open

    try:
        with open_fn(path, "rt", encoding="utf-8") as f:
            for line in f:
                record = json.loads(line)
                requests.append(record)
    except (OSError, json.JSONDecodeError) as e:
        if _is_logging_rank():
            log.error(f"Failed to load benchmark requests from {path}: {e}")
        return []

    return requests

def _normalize_label(value, num_options: int) -> int:
    """Normalize label value to integer index.

    Handles various formats:
    - int: return as-is
    - float: convert to int
    - str: parse digit or match A/B/C/D choice
    """
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        val = value.strip()
        if val.isdigit():
            return int(val)
        val_upper = val.upper()
        for idx, choice in enumerate(ANSWER_CHOICES[:num_options or len(ANSWER_CHOICES)]):
            if val_upper == choice:
                return idx
    return 0


# ============================================================================
# Configuration Classes
# ============================================================================


@dataclass
class BenchmarkConfig:
    """Configuration for a benchmark evaluation task.

    This config defines all parameters needed to load and evaluate a single task.
    """

    task_name: str
    """Unique identifier for this task."""

    task_type: str
    """Type of task: 'multiple_choice' or 'perplexity'."""

    data_path: str
    """Path to task data relative to benchmark root directory."""

    split: str = "validation"
    """Data split to evaluate on (e.g., 'validation', 'test')."""

    num_fewshot: int = 0
    """Number of few-shot examples to include in prompts."""

    batch_size: Optional[int] = None
    """Maximum batch size for this task. If None, uses default from args."""

    metric_for_best_model: str = "acc"
    """Primary metric to report (e.g., 'acc', 'perplexity', 'acc_norm')."""

    choices: Optional[List[str]] = None
    """Answer choices for multiple choice tasks (e.g., ['A', 'B', 'C', 'D'])."""

    description: Optional[str] = None
    """Human-readable description of the task."""


@dataclass
class BenchmarkResult:
    """Results from evaluating a single task."""

    task_name: str
    """Name of the evaluated task."""

    metrics: Dict[str, float]
    """Dictionary of metric names to values."""

    num_samples: int = 0
    """Total number of samples evaluated."""

    metadata: Dict[str, Any] = field(default_factory=dict)
    """Additional task-specific metadata."""


# ============================================================================
# Abstract Base Task
# ============================================================================


class BenchmarkTask(ABC):
    """Abstract base class for benchmark tasks.

    Subclasses must implement:
    - load_data(): Load and prepare task data
    - format_sample(): Convert a data sample to evaluation format
    - compute_metrics(): Aggregate predictions into metrics
    """

    def __init__(self, config: BenchmarkConfig, args):
        """Initialize task.

        Args:
            config: Task configuration
            args: Global training arguments
        """
        self.config = config
        self.args = args
        self.samples = []
        self.metadata = {}

    @abstractmethod
    def load_data(self) -> None:
        """Load task data from disk and store in self.samples.

        Each sample should be a dict with at least:
        - 'context': str, the prompt/context text
        - 'target': Any, the gold answer
        """
        pass

    @abstractmethod
    def format_sample(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """Format a sample for evaluation.

        Args:
            sample: Raw data sample

        Returns:
            Formatted sample dict with fields needed for scoring:
            - 'context': str or List[str]
            - 'continuations': List[str]
            - 'target': Any (gold answer)
        """
        pass

    @abstractmethod
    def compute_metrics(
        self,
        predictions: List[Any],
        targets: List[Any]
    ) -> Dict[str, float]:
        """Compute evaluation metrics from predictions and targets.

        Args:
            predictions: Model predictions
            targets: Gold targets

        Returns:
            Dictionary of metric names to values
        """
        pass

    def __len__(self) -> int:
        """Return number of samples in this task."""
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get a formatted sample by index."""
        return self.format_sample(self.samples[idx])

    def _resolve_data_path(self) -> str:
        """Resolve full path to data file."""
        # Get benchmark root relative to this code file
        code_dir = os.path.dirname(os.path.abspath(__file__))
        repo_root = os.path.dirname(os.path.dirname(code_dir))  # Go up to repo root
        benchmark_root = os.path.join(repo_root, "benchmark")

        if not os.path.exists(benchmark_root):
            if _is_logging_rank():
                log.error(f"Benchmark root directory not found: {benchmark_root}")
            return None

        return os.path.join(benchmark_root, self.config.data_path, "requests.jsonl.gz")


# ============================================================================
# Multiple Choice Task
# ============================================================================


class MultipleChoiceTask(BenchmarkTask):
    """Task for multiple choice questions.

    This handles tasks where the model selects from a discrete set of choices
    (e.g., MMLU, ARC, HellaSwag, COPA).

    Metrics computed:
    - acc: Accuracy using raw log probabilities
    - acc_norm: Accuracy using length-normalized log probabilities
    """

    def load_data(self) -> None:
        """Load multiple choice benchmark data from requests.jsonl.gz format."""
        data_path = self._resolve_data_path()

        if not os.path.exists(data_path):
            if _is_logging_rank():
                log.warning(f"Benchmark data file not found: {data_path}")
            return

        # Load benchmark requests
        benchmark_data = _load_benchmark(data_path)

        # Group by doc_id to reconstruct original questions
        docs_by_id = defaultdict(list)
        for req in benchmark_data:
            doc_id = req.get("doc_id", req.get("idx", 0))
            docs_by_id[doc_id].append(req)

        # Create samples
        for doc_id, requests in docs_by_id.items():
            if not requests:
                continue

            first_req = requests[0]

            # Collect contexts from each request (like original code)
            # Each request has its own context for its continuation
            contexts_list = []
            continuations = []
            lengths_list = []

            for req in requests:
                request_data = req.get("request", {})
                context = request_data.get("context", "")
                continuation = request_data.get("continuation", "")
                length = request_data.get("length", -1)

                contexts_list.append(context)
                continuations.append(continuation)
                lengths_list.append(length if isinstance(length, int) else -1)

            # Use default context (first one) if all contexts are the same
            default_context = contexts_list[0] if contexts_list else ""
            all_same_context = all(c == default_context for c in contexts_list)

            if all_same_context:
                # All contexts are the same, use single string
                context_value = default_context
            else:
                # Different contexts for each option
                context_value = contexts_list

            raw_label = first_req.get("label", 0)
            # Normalize label (handle int/float/str formats)
            label = _normalize_label(raw_label, len(continuations))

            # Store original doc info
            doc_info = first_req.get("doc", {})

            # Get choices if provided (alternative source for length normalization)
            choices = doc_info.get("choices") if doc_info else None

            self.samples.append({
                "context": context_value,  # Can be str or List[str]
                "continuations": continuations,
                "target": label,
                "doc_id": doc_id,
                "doc_info": doc_info,
                "lengths": lengths_list,  # Collected from each request
                "choices": choices,  # Optional: answer choices (for length calculation)
            })

        if _is_logging_rank():
            log.info(
                f"[{self.config.task_name}] Loaded {len(self.samples)} samples "
                f"from {len(benchmark_data)} requests"
            )

    def format_sample(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """Format multiple choice sample.

        Returns dict with:
        - context: str or List[str]
        - continuations: List[str]
        - target: int (correct choice index)
        - lengths: Optional[List[int]] (predefined lengths for normalization)
        - choices: Optional[List[str]] (answer choices for length calculation)
        """
        return {
            "context": sample["context"],
            "continuations": sample["continuations"],
            "target": sample["target"],
            "lengths": sample.get("lengths"),
            "choices": sample.get("choices"),
        }

    def compute_metrics(
        self,
        predictions: List[Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]],
        targets: List[int]
    ) -> Dict[str, float]:
        """Compute accuracy metrics.

        Args:
            predictions: List of (log_probs, lengths, greedy_flags) tuples
                        (greedy_flags is always None for multiple choice)
            targets: List of correct choice indices

        Returns:
            Dictionary with 'acc', 'acc_norm', and 'perplexity' metrics
        """
        correct = 0
        correct_norm = 0
        total = len(targets)

        # For perplexity of correct answers
        total_log_prob = 0.0
        total_tokens = 0

        for (log_probs, lengths, _), target in zip(predictions, targets):
            # Raw accuracy: argmax of log probabilities
            pred = int(torch.argmax(log_probs).item())
            correct += int(pred == target)

            # Length-normalized accuracy
            normalized_logits = log_probs / lengths
            pred_norm = int(torch.argmax(normalized_logits).item())
            correct_norm += int(pred_norm == target)

            # Accumulate log_prob and token count for correct answer's perplexity
            correct_log_prob = float(log_probs[target].item())
            correct_token_count = int(lengths[target].item())
            total_log_prob += correct_log_prob
            total_tokens += correct_token_count

        avg_log_prob = total_log_prob / total_tokens if total_tokens > 0 else 0.0

        return {
            "acc": correct / total if total > 0 else 0.0,
            "acc_norm": correct_norm / total if total > 0 else 0.0,
            "perplexity": math.exp(-avg_log_prob) if total_tokens > 0 else float("inf"),
        }


# ============================================================================
# Perplexity Task
# ============================================================================


class PerplexityTask(BenchmarkTask):
    """Task for perplexity evaluation.

    This handles generative tasks evaluated by perplexity or log-likelihood
    (e.g., WikiText, LAMBADA, Penn Treebank).

    Metrics computed:
    - perplexity: exp(-avg_log_prob)
    - bits_per_byte: -avg_log_prob / log(2)
    """

    def load_data(self) -> None:
        """Load perplexity data from requests.jsonl.gz format."""
        data_path = self._resolve_data_path()

        if not os.path.exists(data_path):
            if _is_logging_rank():
                log.warning(f"Benchmark data file not found: {data_path}")
            return

        # Load benchmark requests
        benchmark_data = _load_benchmark(data_path)

        # Each request is a separate sample for perplexity tasks
        for req in benchmark_data:
            context = req["request"]["context"]
            continuation = req["request"]["continuation"]

            # Length hint for scoring (if provided)
            length = req["request"].get("length", -1)

            self.samples.append({
                "context": context,
                "continuation": continuation,
                "length": length,
                "doc_id": req.get("doc_id", req.get("idx", 0)),
            })

        if _is_logging_rank():
            log.info(f"[{self.config.task_name}] Loaded {len(self.samples)} samples")

    def format_sample(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """Format perplexity sample.

        Returns dict with:
        - context: str
        - continuations: List[str] (single element)
        - target: None (no discrete target)
        - length: int (optional scoring length)
        """
        return {
            "context": sample["context"],
            "continuations": [sample["continuation"]],
            "target": None,
            "length": sample.get("length", -1),
        }

    def compute_metrics(
        self,
        predictions: List[Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]],
        targets: List[Any]  # Unused for perplexity
    ) -> Dict[str, float]:
        """Compute perplexity metrics.

        Args:
            predictions: List of (log_prob, token_count, greedy_flag) tuples
            targets: Unused

        Returns:
            Dictionary with 'acc', 'acc_norm', and 'perplexity' metrics
        """
        total_log_prob = 0.0
        total_tokens = 0
        greedy_correct = 0
        have_greedy = False

        for log_probs, token_counts, greedy_flag in predictions:
            # log_probs is scalar for perplexity tasks
            log_prob = float(log_probs.item()) if log_probs.numel() > 0 else 0.0

            # Get token count, with fallback like original code
            if token_counts is not None and token_counts.numel() > 0:
                token_count = int(token_counts.item())
            else:
                token_count = 0

            # If token_count is 0 or negative, use fallback
            if token_count <= 0:
                token_count = 1  # Default fallback

            total_log_prob += log_prob
            total_tokens += token_count

            # Track greedy accuracy if available
            if greedy_flag is not None:
                have_greedy = True
                greedy_correct += int(bool(greedy_flag.item()))

        avg_log_prob = total_log_prob / total_tokens if total_tokens > 0 else 0.0
        acc = greedy_correct / len(predictions) if have_greedy and len(predictions) > 0 else 0.0

        return {
            "acc": acc,
            "acc_norm": acc,  # Same as acc for perplexity tasks
            "perplexity": math.exp(-avg_log_prob) if total_tokens > 0 else float("inf"),
        }


# ============================================================================
# Benchmark Runner
# ============================================================================


class BenchmarkRunner:
    """Runner for executing benchmark evaluations.

    This class handles:
    - Model forward passes
    - Batch processing and optimization
    - Log probability collection
    - Coordination with pipeline parallelism
    """

    def __init__(self, model, args):
        """Initialize runner.

        Args:
            model: Model to evaluate
            args: Global training arguments
        """
        self.model = model
        self.args = args

        self.tokenizer = get_tokenizer()
        self.forward_backward = get_forward_backward_func()

        # Sequence parameters
        self.seq_length = int(getattr(args, "benchmark_sequence_length", args.seq_length))
        self.pad_token_id = int(getattr(self.tokenizer, "pad_token_id", 0))
        self.eod_token_id = int(getattr(self.tokenizer, "eod_token_id", 0))

    def evaluate_task(self, task: BenchmarkTask) -> BenchmarkResult:
        """Evaluate a single task.

        Args:
            task: Task to evaluate

        Returns:
            BenchmarkResult with metrics
        """
        if len(task) == 0:
            if _is_logging_rank():
                log.warning(f"[{task.config.task_name}] No benchmark samples to evaluate")
            return BenchmarkResult(
                task_name=task.config.task_name,
                metrics={},
                num_samples=0
            )

        # Determine total samples to evaluate (for quick evaluation, can limit with global-batch)
        global_batch = getattr(self.args, "benchmark_global_batch", None)
        num_samples = len(task)
        if global_batch is not None and global_batch > 0:
            num_samples = min(num_samples, global_batch)
            if _is_logging_rank():
                log.info(
                    f"[{task.config.task_name}] Evaluating {num_samples}/{len(task)} samples "
                    f"(limited by --benchmark-global-batch {global_batch})"
                )

        # Collect all samples and targets
        all_samples = [task[i] for i in range(num_samples)]
        targets = [sample["target"] for sample in all_samples]

        # Score all samples (internally chunks by micro-batch)
        predictions = self._score_batch(all_samples, task.config.task_type)

        # Compute metrics
        metrics = task.compute_metrics(predictions, targets)

        return BenchmarkResult(
            task_name=task.config.task_name,
            metrics=metrics,
            num_samples=num_samples,
            metadata=task.metadata
        )

    def _score_batch(
        self,
        samples: List[Dict[str, Any]],
        task_type: str
    ) -> List[Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]]:
        """Score a batch of samples.

        Args:
            samples: List of formatted samples
            task_type: 'multiple_choice' or 'perplexity'

        Returns:
            List of (log_probs, token_counts, greedy_flags) tuples
            - For multiple choice: greedy_flags is None
            - For perplexity: greedy_flags indicates if all tokens were predicted correctly
        """
        all_sequences = []
        all_masks = []
        all_metadata = []

        # Determine if we should compute greedy (only for perplexity tasks with single continuation)
        compute_greedy = task_type == "perplexity"

        # Build sequences for all samples and continuations
        for sample in samples:
            context = sample["context"]
            continuations = sample["continuations"]

            # Handle both single context (str) and per-option contexts (List[str])
            if isinstance(context, (list, tuple)):
                # Multiple contexts (one per continuation)
                context_list = list(context)
                default_context = str(context_list[0]) if context_list else ""
            else:
                # Single shared context
                context_list = None
                default_context = str(context)

            for option_idx, continuation in enumerate(continuations):
                # Get the appropriate context for this option
                option_context = (
                    str(context_list[option_idx])
                    if context_list is not None and option_idx < len(context_list)
                    else default_context
                )

                # Get target length if provided (for perplexity)
                target_length = sample.get("length", -1) if task_type == "perplexity" else -1

                # Build sequence
                tokens, mask = self._build_sequence(option_context, continuation, target_length)

                all_sequences.append(tokens)
                all_masks.append(mask)
                all_metadata.append((len(samples), len(continuations)))  # Track structure

        # Run model on all sequences with chunking based on micro-batch size
        # micro-batch controls the number of sequences per forward pass
        micro_batch = getattr(self.args, "benchmark_micro_batch", None)
        if micro_batch is None or micro_batch <= 0:
            # Use training micro_batch_size
            micro_batch = getattr(self.args, "micro_batch_size", 1)

        total_sequences = len(all_sequences)

        if total_sequences > 0:
            if _is_logging_rank():
                log.info(
                    f"[{samples[0].get('task', 'task')}] Processing {len(samples)} samples, {total_sequences} sequences (micro-batch={micro_batch})"
                )

        log_probs = []
        token_counts = []
        greedy_flags = []

        # If total sequences fit in one forward pass, do it directly
        if total_sequences <= micro_batch:
            chunk_log_probs, chunk_token_counts, chunk_greedy = self._run_forward(
                all_sequences, all_masks, compute_greedy=compute_greedy
            )
            log_probs = chunk_log_probs
            token_counts = chunk_token_counts
            greedy_flags = chunk_greedy if chunk_greedy is not None else []
        else:
            # Otherwise, chunk into multiple forward passes
            for i in range(0, total_sequences, micro_batch):
                chunk_sequences = all_sequences[i:i + micro_batch]
                chunk_masks = all_masks[i:i + micro_batch]
                chunk_log_probs, chunk_token_counts, chunk_greedy = self._run_forward(
                    chunk_sequences, chunk_masks, compute_greedy=compute_greedy
                )
                log_probs.extend(chunk_log_probs)
                token_counts.extend(chunk_token_counts)
                if chunk_greedy is not None:
                    greedy_flags.extend(chunk_greedy)

        # Group results by sample
        results = []
        offset = 0
        for sample in samples:
            num_continuations = len(sample["continuations"])

            # Gather log probs and counts for this sample
            sample_log_probs = log_probs[offset:offset + num_continuations]
            sample_token_counts = token_counts[offset:offset + num_continuations]
            sample_greedy = greedy_flags[offset:offset + num_continuations] if greedy_flags else None

            if task_type == "multiple_choice":
                # Stack into tensors for argmax
                stacked_log_probs = torch.stack(sample_log_probs)
                stacked_token_counts = torch.stack(sample_token_counts)
                results.append((stacked_log_probs, stacked_token_counts, None))
            else:
                # Return as-is for perplexity
                greedy_flag = sample_greedy[0] if sample_greedy else None
                results.append((sample_log_probs[0], sample_token_counts[0], greedy_flag))

            offset += num_continuations

        return results

    def _build_sequence(
        self,
        context: str,
        continuation: str,
        target_length: int = -1
    ) -> Tuple[List[int], List[float]]:
        """Build token sequence and loss mask.

        Args:
            context: Prompt text
            continuation: Completion text
            target_length: Optional target length for scoring

        Returns:
            tokens: List of token IDs
            mask: List of loss weights (1.0 for scored tokens, 0.0 otherwise)
        """
        # Tokenize
        context_tokens = self._encode_text(context)
        continuation_tokens = self._encode_text(continuation)

        # Concatenate
        tokens = context_tokens + continuation_tokens

        # Truncate if needed
        if len(tokens) > self.seq_length:
            tokens = tokens[-self.seq_length:]

        # Compute mask
        total_len = len(tokens)
        cont_len = min(len(continuation_tokens), total_len)

        # Effective length to score
        if target_length > 0:
            effective_len = min(cont_len, target_length)
        else:
            effective_len = cont_len

        # Build mask
        mask = [0.0] * total_len
        start_idx = max(0, total_len - cont_len)
        end_idx = total_len

        # Mark last effective_len tokens for scoring
        for i in range(max(start_idx, end_idx - effective_len), end_idx):
            mask[i] = 1.0

        # Ensure at least one token is scored
        # Fallback: mark all continuation tokens for scoring (matches downstream_eval.py)
        if sum(mask) == 0 and total_len > 0:
            num_to_mark = max(cont_len, 1)
            mask[-num_to_mark:] = [1.0] * num_to_mark

        return tokens, mask

    def _run_forward(
        self,
        sequences: List[List[int]],
        masks: List[List[float]],
        compute_greedy: bool = False
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor], Optional[List[torch.Tensor]]]:
        """Run model forward pass on sequences.

        Args:
            sequences: List of token sequences
            masks: List of loss masks
            compute_greedy: If True, also compute greedy decoding accuracy

        Returns:
            log_probs: List of sequence log probabilities
            token_counts: List of token counts
            greedy_flags: List of greedy accuracy flags (None if compute_greedy=False)
        """
        if not sequences:
            return [], [], None

        # Prepare batch
        batch = self._prepare_batch(sequences, masks)
        data_iterator = iter([batch])
        storage = {}

        def forward_step(data_iter, model, checkpoint_activations_microbatch=None):
            del checkpoint_activations_microbatch
            batch_data = get_batch_on_this_tp_rank(data_iter)
            output = model(
                batch_data["tokens"],
                batch_data["position_ids"],
                batch_data["attention_mask"],
                labels=batch_data["labels"],
                loss_mask=batch_data["loss_mask"],
            )
            return output, partial(self._collect_log_probs, storage, batch_data["loss_mask"])

        # Run forward pass without gradients
        with torch.no_grad(), self._temporary_eval():
            self.forward_backward(
                forward_step_func=forward_step,
                data_iterator=data_iterator,
                model=self.model,
                num_microbatches=1,
                seq_length=batch["tokens"].shape[1],
                micro_batch_size=batch["tokens"].shape[0],
                decoder_seq_length=batch["tokens"].shape[1],
                forward_only=True,
                collect_non_loss_data=False,
            )

        log_probs = storage["log_probs"]
        token_counts = storage["token_counts"]

        # Convert to list of tensors
        log_probs_list = [log_probs[i] for i in range(len(sequences))]
        token_counts_list = [token_counts[i] for i in range(len(sequences))]

        # Optionally compute greedy decoding accuracy
        greedy_flags_list = None
        if compute_greedy:
            greedy_storage = {}
            greedy_iterator = iter([batch])

            def greedy_forward_step(data_iter, model, checkpoint_activations_microbatch=None):
                del checkpoint_activations_microbatch
                batch_data = get_batch_on_this_tp_rank(data_iter)
                output = model(
                    batch_data["tokens"],
                    batch_data["position_ids"],
                    batch_data["attention_mask"],
                )
                return output, partial(
                    self._collect_greedy_flags,
                    greedy_storage,
                    batch_data["labels"],
                    batch_data["loss_mask"],
                )

            with torch.no_grad(), self._temporary_eval():
                self.forward_backward(
                    forward_step_func=greedy_forward_step,
                    data_iterator=greedy_iterator,
                    model=self.model,
                    num_microbatches=1,
                    seq_length=batch["tokens"].shape[1],
                    micro_batch_size=batch["tokens"].shape[0],
                    decoder_seq_length=batch["tokens"].shape[1],
                    forward_only=True,
                    collect_non_loss_data=True,
                )

            greedy_flags = greedy_storage.get("is_greedy")
            if greedy_flags is not None:
                greedy_flags_list = [greedy_flags[i] for i in range(len(sequences))]

        return log_probs_list, token_counts_list, greedy_flags_list

    def _prepare_batch(
        self,
        sequences: List[List[int]],
        masks: List[List[float]]
    ) -> Dict[str, torch.Tensor]:
        """Prepare batch tensors for model input.

        Args:
            sequences: List of token sequences
            masks: List of loss masks

        Returns:
            Dictionary with tensors: tokens, labels, loss_mask, attention_mask, position_ids
        """
        batch_size = len(sequences)
        max_len = max(len(seq) for seq in sequences)

        # Initialize tensors
        tokens = torch.full((batch_size, max_len), self.pad_token_id, dtype=torch.long)
        labels = torch.full((batch_size, max_len), self.pad_token_id, dtype=torch.long)
        loss_mask = torch.zeros((batch_size, max_len), dtype=torch.float32)

        # Fill tensors
        for i, (seq, mask) in enumerate(zip(sequences, masks)):
            seq_len = len(seq)
            tokens[i, :seq_len] = torch.tensor(seq, dtype=torch.long)
            if seq_len > 1:
                labels[i, :seq_len - 1] = torch.tensor(seq[1:], dtype=torch.long)
                loss_mask[i, :seq_len - 1] = torch.tensor(mask[1:], dtype=torch.float32)

        # Get attention mask and position IDs
        attention_mask, _, position_ids = get_ltor_masks_and_position_ids(
            tokens.clone(),
            self.eod_token_id,
            self.pad_token_id,
            self.args.reset_position_ids,
            self.args.reset_attention_mask,
            self.args.eod_mask_loss,
            False,  # pad_mask_loss: don't mask padding in benchmark eval
        )

        return {
            "tokens": tokens,
            "labels": labels,
            "loss_mask": loss_mask,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
        }

    def _collect_log_probs(
        self,
        storage: Dict[str, torch.Tensor],
        loss_mask: torch.Tensor,
        output_tensor: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        """Collect log probabilities from model output.

        This function is called as a callback during forward pass.
        """
        zeros = torch.zeros(1, device=output_tensor.device, dtype=output_tensor.dtype)

        if not parallel_state.is_pipeline_last_stage():
            token_count = loss_mask.sum().to(torch.int)
            return zeros, token_count, {}

        # Compute sequence log probabilities
        losses = output_tensor.float()
        mask = loss_mask.to(losses.device)

        # Sum losses over sequence dimension
        seq_log_probs = -(losses * mask).sum(dim=1)
        token_counts = mask.sum(dim=1)

        # Store results
        storage["log_probs"] = seq_log_probs.detach().cpu()
        storage["token_counts"] = token_counts.detach().cpu()

        token_count = mask.sum().to(torch.int)
        return zeros, token_count, {}

    def _collect_greedy_flags(
        self,
        storage: Dict[str, torch.Tensor],
        labels: torch.Tensor,
        loss_mask: torch.Tensor,
        logits: torch.Tensor,
        non_loss_data: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        """Collect greedy decoding accuracy flags from model output.

        This checks if the greedy prediction (argmax) matches the ground truth for all tokens.

        Args:
            storage: Dictionary to store results
            labels: Ground truth token IDs [batch_size, seq_len]
            loss_mask: Mask indicating which tokens to evaluate [batch_size, seq_len]
            logits: Model output logits [batch_size, seq_len, vocab_size]
            non_loss_data: Flag passed by Megatron when collect_non_loss_data=True (ignored)

        Returns:
            Tuple of (zeros, token_count, {})
        """
        zeros = torch.zeros(1, device=logits.device, dtype=logits.dtype)

        if not parallel_state.is_pipeline_last_stage():
            token_count = loss_mask.sum().to(torch.int)
            return zeros, token_count, {}

        # Get greedy predictions (argmax over vocabulary)
        predictions = torch.argmax(logits, dim=-1)

        # Create boolean mask for valid positions
        mask = loss_mask.to(predictions.device) > 0.0
        labels = labels.to(predictions.device)

        # Check if prediction matches label OR position is masked (not evaluated)
        correct = torch.logical_or(~mask, predictions == labels)

        # Check if ALL tokens in the sequence are correct
        greedy_correct = correct.all(dim=1).to(torch.float32)

        # Store results
        storage["is_greedy"] = greedy_correct.detach().cpu()

        token_count = mask.sum().to(torch.int)
        return zeros, token_count, {}

    def _encode_text(self, text: str) -> List[int]:
        """Encode text to token IDs."""
        tokens = self.tokenizer.tokenize(text)
        if isinstance(tokens, dict):
            tokens = tokens.get("input_ids", [])
        return list(tokens)

    def _temporary_eval(self):
        """Context manager to temporarily set model to eval mode."""
        modules = self.model if isinstance(self.model, list) else [self.model]
        prev_states = [m.training for m in modules]

        for m in modules:
            m.eval()

        class _Context:
            def __enter__(self):
                return None

            def __exit__(self, *args):
                for m, prev in zip(modules, prev_states):
                    m.train(prev)

        return _Context()


# ============================================================================
# Top-Level Evaluator
# ============================================================================


class BenchmarkEvaluator:
    """Top-level benchmark evaluator.

    This class manages:
    - Task registration and configuration
    - Scheduling evaluations during training
    - Logging results to TensorBoard/WandB
    """

    def __init__(self, args):
        """Initialize evaluator.

        Args:
            args: Global training arguments
        """
        self.args = args
        self.tasks = self._load_task_configs()

    def run(self, model, iteration: int) -> None:
        """Run benchmark evaluation.

        Args:
            model: Model to evaluate
            iteration: Current training iteration
        """

        # Create runner with model
        runner = BenchmarkRunner(model, self.args)

        # Evaluate each task
        for task_config in self.tasks:
            if _is_logging_rank():
                log.info(f"[Benchmark] Evaluating task: {task_config.task_name}")

            # Create task instance
            task = self._create_task(task_config)

            # Load data
            task.load_data()

            # Evaluate
            result = runner.evaluate_task(task)

            # Log results
            self._log_result(result, iteration)

            # Clear CUDA cache
            self._clear_cuda_cache()

    def _load_task_configs(self) -> List[BenchmarkConfig]:
        """Load task configurations from args."""
        task_list = getattr(self.args, "benchmark_tasks", None)
        if not task_list:
            return []

        # Parse task specifications
        if isinstance(task_list, str):
            task_specs = [s.strip() for s in task_list.split(",") if s.strip()]
        else:
            task_specs = []
            for item in task_list:
                if isinstance(item, str):
                    task_specs.extend([s.strip() for s in item.split(",") if s.strip()])

        # Create configs from specs
        configs = []
        for spec in task_specs:
            config = self._parse_task_spec(spec)
            if config:
                configs.append(config)

        return configs

    def _parse_task_spec(self, spec: str) -> Optional[BenchmarkConfig]:
        """Parse task specification string.

        Examples:
            'copa_mc_0shot' -> multiple choice task
            'lambada_ppl' -> perplexity task
            'unknown_task' -> auto-detect from data
        """
        # Determine task type from suffix
        if "_rc_" in spec or "_mc_" in spec:
            task_type = "multiple_choice"
        elif "_ppl_" in spec:
            task_type = "perplexity"
        else:
            if _is_logging_rank():
                log.error(f"Unknown benchmark task type: {spec}")
            return None

        # Extract few-shot count
        num_fewshot = 0
        match = re.search(r'_(\d+)shot', spec)
        if match:
            num_fewshot = int(match.group(1))

        # Construct data path
        # Example: copa_mc_0shot -> copa/mc_0shot
        # Example: arc_easy_rc_0shot -> arc_easy/rc_0shot
        # Strategy: Find known variant patterns (mc_, rc_, ppl, etc.) and split there
        variant_patterns = ["_mc_", "_rc_", "_ppl_"]
        task_name = spec
        variant = None

        for pattern in variant_patterns:
            if pattern in spec:
                # Find the last occurrence of the pattern
                idx = spec.rfind(pattern)
                task_name = spec[:idx]
                variant = spec[idx + 1:]  # Skip the leading underscore
                break

        if variant:
            data_path = f"{task_name}/{variant}"
        else:
            # Fallback: assume entire spec is the variant path
            data_path = f"{spec}"

        return BenchmarkConfig(
            task_name=spec,
            task_type=task_type,
            data_path=data_path,
            num_fewshot=num_fewshot,
        )

    def _create_task(self, config: BenchmarkConfig) -> BenchmarkTask:
        """
        Create task instance from config.
        """

        # Auto-detect task type if needed
        task_type = config.task_type
        if task_type == "multiple_choice":
            return MultipleChoiceTask(config, self.args)
        elif task_type == "perplexity":
            return PerplexityTask(config, self.args)

    def _log_result(self, result: BenchmarkResult, iteration: int) -> None:
        """Log evaluation result."""
        if not result.metrics:
            return

        # Console logging (on logging rank)
        if _is_logging_rank():
            metric_str = ", ".join(
                f"{name}={value:.4f}" for name, value in result.metrics.items()
            )
            print_rank_0(
                f"[Benchmark:{result.task_name}] {metric_str} "
                f"(n={result.num_samples}, iter={iteration})"
            )

        # TensorBoard and WandB logging (with distributed coordination)
        self._log_to_writers(result, iteration)

    def _log_to_writers(self, result: BenchmarkResult, iteration: int) -> None:
        """Log to TensorBoard and WandB with distributed coordination.
        
        Note: Both TensorBoard and WandB writers are initialized on rank (world_size - 1),
        but _is_logging_rank() returns True on a different rank (pipeline_last_stage and
        data_parallel_rank == 0). We need to gather results to the writer rank.
        """
        # Prepare log entry
        log_entry = None
        if _is_logging_rank():
            log_entry = (result.task_name, result.metrics, iteration)

        # Gather across ranks
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            gathered = [None] * torch.distributed.get_world_size()
            torch.distributed.all_gather_object(gathered, log_entry)
            rank = torch.distributed.get_rank()
            world_size = torch.distributed.get_world_size()
        else:
            gathered = [log_entry]
            rank = 0
            world_size = 1

        # Log from designated rank (tensorboard/wandb writers are initialized on world_size - 1)
        writer_rank = world_size - 1
        if rank != writer_rank:
            return

        # Process gathered entries
        for entry in gathered:
            if entry is None:
                continue
            task_name, metrics, iter_num = entry
            
            # TensorBoard logging
            tb_writer = get_tensorboard_writer()
            if tb_writer:
                for name, value in metrics.items():
                    tb_writer.add_scalar(
                        f"benchmark/{task_name}/{name}", value, iter_num
                    )
            
            # WandB logging
            if getattr(self.args, "wandb_project", ""):
                wandb_writer = get_wandb_writer()
                if wandb_writer:
                    log_payload = {
                        f"benchmark/{task_name}/{name}": value
                        for name, value in metrics.items()
                    }
                    wandb_writer.log(log_payload, step=iter_num)

    def _clear_cuda_cache(self) -> None:
        """Clear CUDA cache to prevent OOM."""
        if not torch.cuda.is_available():
            return
        torch.cuda.empty_cache()
        if hasattr(torch.cuda, "ipc_collect"):
            try:
                torch.cuda.ipc_collect()
            except RuntimeError:
                pass

# ============================================================================
# Public API
# ============================================================================


def run_benchmark(model: torch.nn.Module, iteration: int) -> None:
    """Run benchmark evaluation (public API).

    This is the main entry point called from the training loop.

    Args:
        model: Model to evaluate
        iteration: Current training iteration
    """
    args = get_args()

    # Create evaluator without model (model is passed to run())
    evaluator = BenchmarkEvaluator(args)
    evaluator.run(model, iteration)
