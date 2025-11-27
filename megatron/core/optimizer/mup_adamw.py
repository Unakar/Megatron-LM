# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

"""MupAdamW optimizer with spectral mup learning rate scaling."""

import logging
import torch
from typing import Optional

logger = logging.getLogger(__name__)

try:
    from emerging_optimizers.orthogonalized_optimizers.spectral_ball_utils import (
        get_spectral_ball_scale_factor
    )
    HAVE_SPECTRAL_BALL_UTILS = True
except ImportError:
    HAVE_SPECTRAL_BALL_UTILS = False
    logger.warning(
        "Cannot import get_spectral_ball_scale_factor from emerging_optimizers. "
        "MupAdamW will not be available."
    )


class MupAdamW(torch.optim.Optimizer):
    """AdamW wrapper with per-parameter spectral mup learning rate scaling.

    This optimizer wraps a standard AdamW optimizer and applies spectral mup
    scaling to gradients before the optimizer step. The scaling is computed as:
        scale = sqrt(n_out / n_in)  for 2D weight matrices (except embeddings/output)
        scale = 1.0                  for biases, norms, embeddings, and output layers

    This allows different learning rates for different weight matrices based on
    their shapes, following the spectral mup principle used in Muon and
    SpectralBall optimizers.

    Embeddings and output layer (LM head) are explicitly excluded from MuP scaling
    by checking the `is_embedding_or_output_parameter` attribute, consistent with
    how SpectralBall and Muon handle these parameters.

    Example:
        For a weight matrix of shape (4096, 2048):
            scale = sqrt(4096 / 2048) = sqrt(2) ≈ 1.414
        For a weight matrix of shape (2048, 4096):
            scale = sqrt(2048 / 4096) = sqrt(0.5) ≈ 0.707
        For embeddings/output layer:
            scale = 1.0 (no MuP scaling)

    This means wider layers (larger n_out) get higher effective learning rates,
    which is the core idea of spectral mup.
    """

    def __init__(self, base_optimizer: torch.optim.Optimizer):
        """Initialize MupAdamW wrapper.

        Args:
            base_optimizer: The underlying AdamW optimizer to wrap.
        """
        if not HAVE_SPECTRAL_BALL_UTILS:
            raise ImportError(
                "MupAdamW requires emerging_optimizers to be installed. "
                "Please install it to use spectral mup learning rate scaling."
            )

        self.base_optimizer = base_optimizer

        # Cache for scale factors (shape -> scale)
        # This avoids recomputing sqrt for the same shapes
        self._scale_cache = {}

        # For logging: track which shapes we've encountered
        self._logged_shapes = set()

    def _get_scale_factor(self, param: torch.nn.Parameter) -> float:
        """Get scale factor for a parameter (with caching).

        Args:
            param: Parameter tensor

        Returns:
            Scale factor: sqrt(n_out / n_in) for 2D tensors (except embeddings/output), 1.0 otherwise
        """
        # Skip MuP scaling for embeddings and LM head (output layer)
        # This matches the behavior of SpectralBall and Muon optimizers
        if getattr(param, 'is_embedding_or_output_parameter', False):
            param_name = getattr(param, 'param_name', 'unknown')
            if param_name not in self._logged_shapes:
                logger.debug(
                    f"MupAdamW: parameter '{param_name}' (embedding/output) -> scale factor 1.0 (no MuP scaling)"
                )
                self._logged_shapes.add(param_name)
            return 1.0

        shape = param.shape
        if shape not in self._scale_cache:
            if len(shape) == 2:
                n_out, n_in = shape
                # Use spectral_mup mode: sqrt(n_out / n_in)
                scale = get_spectral_ball_scale_factor(
                    n_out, n_in, mode="spectral_mup"
                )
                self._scale_cache[shape] = scale

                # Log once per shape for debugging
                if shape not in self._logged_shapes:
                    logger.debug(
                        f"MupAdamW: shape {shape} -> scale factor {scale:.4f} "
                        f"(sqrt({n_out}/{n_in}))"
                    )
                    self._logged_shapes.add(shape)
            else:
                # For non-2D tensors (biases, norms), use scale=1.0
                self._scale_cache[shape] = 1.0

                if shape not in self._logged_shapes:
                    logger.debug(
                        f"MupAdamW: shape {shape} (non-2D) -> scale factor 1.0 (no scaling)"
                    )
                    self._logged_shapes.add(shape)

        return self._scale_cache[shape]

    @torch.no_grad()
    def step(self, closure=None):
        """Apply spectral mup scaling to gradients, then step the base optimizer.

        This method:
        1. Iterates over all parameters with gradients
        2. Multiplies each gradient by its scale factor (based on parameter shape and type)
        3. Calls the underlying AdamW optimizer's step

        Embeddings and output layer parameters are skipped (scale = 1.0).

        Args:
            closure: Optional closure for computing loss (passed to base optimizer)

        Returns:
            Loss value from closure (if provided)
        """
        # Apply scale factors to gradients
        for group in self.base_optimizer.param_groups:
            for p in group['params']:
                if p.grad is not None:
                    scale = self._get_scale_factor(p)
                    if scale != 1.0:
                        # In-place multiplication to save memory
                        p.grad.mul_(scale)

        # Call the underlying AdamW step
        loss = self.base_optimizer.step(closure)
        return loss

    # =========================================================================
    # Delegate all other methods to base_optimizer
    # =========================================================================

    def zero_grad(self, set_to_none: bool = True):
        """Zero out gradients. Delegates to base optimizer."""
        return self.base_optimizer.zero_grad(set_to_none)

    @property
    def state(self):
        """Access optimizer state. Delegates to base optimizer."""
        return self.base_optimizer.state

    @property
    def param_groups(self):
        """Access parameter groups. Delegates to base optimizer."""
        return self.base_optimizer.param_groups

    @param_groups.setter
    def param_groups(self, value):
        """Set parameter groups. Delegates to base optimizer."""
        self.base_optimizer.param_groups = value

    def state_dict(self):
        """Return optimizer state dict. Delegates to base optimizer."""
        return self.base_optimizer.state_dict()

    def load_state_dict(self, state_dict):
        """Load optimizer state dict. Delegates to base optimizer."""
        return self.base_optimizer.load_state_dict(state_dict)

    def __repr__(self):
        """String representation of the optimizer."""
        return f"MupAdamW(spectral_mup, base={repr(self.base_optimizer)})"
