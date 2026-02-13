# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Hyperball Adam optimizer: Adam with Frobenius-norm hypersphere projection.

Update rule for each 2D weight matrix W:

    u_t  = Adam_update(grad)           # standard Adam direction
    d_t  = u_t / ||u_t||_F            # normalize the update
    W_t' = W_t - lr * R * d_t         # step on the tangent space
    W_{t+1} = R * W_t' / ||W_t'||_F  # retract back to the sphere

where R = ||W_0||_F is the initial Frobenius norm, captured on the first step.
"""

from typing import List, Tuple, Union

import torch
from torch.optim import Optimizer

from emerging_optimizers.scalar_optimizers.adam import calculate_adam_update


__all__ = ["HyperballAdam"]


class HyperballAdam(Optimizer):
    """Adam optimizer with Frobenius-norm hypersphere constraint.

    After computing the standard Adam update direction, the weight matrix
    is projected back onto the Frobenius-norm sphere of radius R = ||W_0||_F.

    Args:
        params: Iterable of parameters or param groups.
        lr: Learning rate. Default: 1e-3.
        betas: Coefficients for computing running averages of gradient
            and its square. Default: (0.9, 0.999).
        eps: Term added to the denominator for numerical stability. Default: 1e-8.
        weight_decay: Decoupled weight decay coefficient. Default: 0.0.
        bias_correction: Whether to apply bias correction to Adam moments. Default: True.
    """

    def __init__(
        self,
        params,
        lr: float = 1e-3,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        bias_correction: bool = True,
    ):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if eps < 0.0:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")

        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            bias_correction=bias_correction,
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.

        Args:
            closure: A closure that reevaluates the model and returns the loss.

        Returns:
            Optional loss from the closure.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            betas = group["betas"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]
            bias_correction = group["bias_correction"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad

                # Initialize state on first step
                state = self.state[p]
                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p.data)
                    state["exp_avg_sq"] = torch.zeros_like(p.data)
                    # Capture the initial Frobenius norm as the target radius
                    state["initial_frobenius_norm"] = torch.norm(
                        p.data.float(), p="fro"
                    ).item()

                state["step"] += 1
                step = state["step"]
                exp_avg = state["exp_avg"]
                exp_avg_sq = state["exp_avg_sq"]
                R = state["initial_frobenius_norm"]

                # Apply decoupled weight decay (AdamW style)
                if weight_decay != 0.0:
                    p.data.mul_(1.0 - lr * weight_decay)

                # Compute Adam update direction u_t
                u_t = calculate_adam_update(
                    grad=grad,
                    exp_avg=exp_avg,
                    exp_avg_sq=exp_avg_sq,
                    betas=betas,
                    correct_bias=bias_correction,
                    use_nesterov=False,
                    step=step,
                    eps=eps,
                )

                # Normalize the update direction: d_t = u_t / ||u_t||_F
                u_norm = torch.norm(u_t, p="fro")
                # Guard against zero-norm updates
                if u_norm.item() > 0:
                    d_t = u_t / u_norm
                else:
                    d_t = u_t

                # Tangent step: W_raw = W_t - lr * R * d_t
                p.data.add_(d_t, alpha=-lr * R)

                # Retract to sphere: W_{t+1} = R * W_raw / ||W_raw||_F
                w_norm = torch.norm(p.data.float(), p="fro")
                if w_norm.item() > 0:
                    p.data.mul_(R / w_norm.item())

        return loss
