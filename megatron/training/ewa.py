"""Exponential Weight Averaging (EWA) for Megatron-LM.

Maintains shadow parameters ξ that track an exponential moving average of
the model weights θ:

    ξ_{t+1} = τ · ξ_t + (1 − τ) · θ_t

Evaluation uses ξ (the averaged weights) instead of θ.  This is the
"Constant + EWA" strategy from critical batch size literature.
"""

import os
from contextlib import contextmanager
from typing import List

import torch

from megatron.core.utils import log_single_rank
from megatron.training.utils import print_rank_0, unwrap_model

import logging

logger = logging.getLogger(__name__)


class EWAState:
    """Manages shadow (EWA-averaged) copies of model parameters.

    Args:
        model: List of model chunks (DDP-wrapped or raw).
        decay: EWA decay rate τ ∈ (0, 1).  Higher = slower averaging.
        start_iter: Iteration at which EWA updates begin.
            Before this, shadow params track live params exactly.
    """

    def __init__(self, model: list, decay: float, start_iter: int = 0):
        self.decay = decay
        self.start_iter = start_iter
        self._shadow: List[dict] = []

        for model_chunk in unwrap_model(model):
            shadow_dict = {}
            for name, param in model_chunk.named_parameters():
                if param.requires_grad:
                    shadow_dict[name] = param.data.clone()
            self._shadow.append(shadow_dict)

        log_single_rank(
            logger, logging.INFO,
            f'[EWA] Initialized shadow params (decay={decay}, '
            f'start_iter={start_iter}, '
            f'chunks={len(self._shadow)}, '
            f'params={sum(len(s) for s in self._shadow)})'
        )

    @torch.no_grad()
    def update(self, model: list, iteration: int):
        """Update shadow params: ξ = τ·ξ + (1−τ)·θ."""
        if iteration < self.start_iter:
            # Before start_iter, copy live params directly so shadow
            # tracks the model exactly.
            for shadow_dict, model_chunk in zip(
                self._shadow, unwrap_model(model)
            ):
                for name, param in model_chunk.named_parameters():
                    if name in shadow_dict:
                        shadow_dict[name].copy_(param.data)
            return

        tau = self.decay
        for shadow_dict, model_chunk in zip(
            self._shadow, unwrap_model(model)
        ):
            for name, param in model_chunk.named_parameters():
                if name in shadow_dict:
                    shadow_dict[name].lerp_(param.data, 1.0 - tau)

    @contextmanager
    def swap_for_eval(self, model: list):
        """Context manager that swaps model weights with EWA shadow for eval.

        On enter: model gets shadow weights (ξ).
        On exit:  model gets its original weights back (θ).
        """
        unwrapped = unwrap_model(model)
        saved = []
        try:
            for shadow_dict, model_chunk in zip(self._shadow, unwrapped):
                chunk_saved = {}
                for name, param in model_chunk.named_parameters():
                    if name in shadow_dict:
                        chunk_saved[name] = param.data.clone()
                        param.data.copy_(shadow_dict[name])
                saved.append(chunk_saved)
            yield
        finally:
            for chunk_saved, model_chunk in zip(saved, unwrapped):
                for name, param in model_chunk.named_parameters():
                    if name in chunk_saved:
                        param.data.copy_(chunk_saved[name])

    def state_dict(self):
        """Return serializable state for checkpointing."""
        return {
            'decay': self.decay,
            'start_iter': self.start_iter,
            'shadow': [
                {name: tensor.cpu() for name, tensor in sd.items()}
                for sd in self._shadow
            ],
        }

    def load_state_dict(self, state_dict):
        """Load shadow params from a checkpoint."""
        for sd_saved, sd_live in zip(state_dict['shadow'], self._shadow):
            for name in sd_live:
                if name in sd_saved:
                    sd_live[name].copy_(sd_saved[name])
        log_single_rank(
            logger, logging.INFO, '[EWA] Loaded shadow params from checkpoint'
        )


def save_ewa_checkpoint(ewa_state, checkpoint_dir, iteration):
    """Save EWA state as a separate file next to the main checkpoint."""
    if ewa_state is None:
        return
    if torch.distributed.get_rank() == 0:
        path = os.path.join(checkpoint_dir, f'ewa_state_iter_{iteration:07d}.pt')
        torch.save(ewa_state.state_dict(), path)
        latest_path = os.path.join(checkpoint_dir, 'latest_ewa_state.pt')
        torch.save(ewa_state.state_dict(), latest_path)
        print_rank_0(f'[EWA] Saved shadow params to {path}')


def load_ewa_checkpoint(ewa_state, checkpoint_dir):
    """Load EWA state from the latest saved file."""
    if ewa_state is None:
        return
    path = os.path.join(checkpoint_dir, 'latest_ewa_state.pt')
    if os.path.exists(path):
        state_dict = torch.load(path, map_location='cpu', weights_only=True)
        ewa_state.load_state_dict(state_dict)
    else:
        print_rank_0(f'[EWA] No saved shadow params found at {path}, starting fresh')
