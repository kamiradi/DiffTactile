"""
PyTorch optimizer utilities for Taichi-based FEM optimization.

This module provides:
1. Reparameterization functions for constrained optimization
2. Adam optimizer wrapper that bridges PyTorch and Taichi gradients

Constraints handled via reparameterization:
- E >= 100: E = 100 + softplus(raw_E)
- nu in [0.01, 0.49]: nu = 0.01 + 0.48 * sigmoid(raw_nu)
"""

import torch
import torch.nn.functional as F
from typing import Callable, Any
import logging

logger = logging.getLogger(__name__)


def physical_to_raw(E: float, nu: float) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert physical params to unbounded raw params for optimization.

    Inverse transforms:
    - E = 100 + softplus(raw_E) => raw_E = softplus_inv(E - 100)
    - nu = 0.01 + 0.48 * sigmoid(raw_nu) => raw_nu = logit((nu - 0.01) / 0.48)

    Args:
        E: Young's modulus (must be >= 100)
        nu: Poisson's ratio (must be in [0.01, 0.49])

    Returns:
        Tuple of (raw_E, raw_nu) as torch tensors
    """
    # Clip values away from boundaries for numerical stability
    E_clipped = max(E, 101.0)
    nu_clipped = max(0.02, min(0.48, nu))

    # Inverse of softplus: softplus_inv(y) = log(exp(y) - 1)
    # For numerical stability, use: log(exp(y) - 1) = y + log(1 - exp(-y)) for large y
    y_E = E_clipped - 100.0
    if y_E > 20:
        raw_E = torch.tensor(y_E, dtype=torch.float32)
    else:
        raw_E = torch.log(torch.exp(torch.tensor(y_E, dtype=torch.float32)) - 1.0 + 1e-8)

    # Inverse of scaled sigmoid: logit((nu - 0.01) / 0.48)
    nu_normalized = (nu_clipped - 0.01) / 0.48
    raw_nu = torch.log(
        torch.tensor(nu_normalized / (1.0 - nu_normalized), dtype=torch.float32)
    )

    return raw_E, raw_nu


def raw_to_physical(raw_params: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert raw params to physical params (E, nu).

    Transforms:
    - E = 100 + softplus(raw_E)
    - nu = 0.01 + 0.48 * sigmoid(raw_nu)

    Args:
        raw_params: Tensor of shape (2,) containing [raw_E, raw_nu]

    Returns:
        Tuple of (E, nu) as torch tensors
    """
    E = 100.0 + F.softplus(raw_params[0])
    nu = 0.01 + 0.48 * torch.sigmoid(raw_params[1])
    return E, nu


def compute_raw_gradients(
    raw_params: torch.Tensor, grad_E: float, grad_nu: float
) -> torch.Tensor:
    """Apply chain rule to convert physical gradients to raw gradients.

    Chain rule:
    - dL/d(raw_E) = dL/dE * dE/d(raw_E) = grad_E * sigmoid(raw_E)
    - dL/d(raw_nu) = dL/dnu * dnu/d(raw_nu) = grad_nu * 0.48 * sigmoid(raw_nu) * (1 - sigmoid(raw_nu))

    Args:
        raw_params: Tensor of shape (2,) containing [raw_E, raw_nu]
        grad_E: Gradient of loss w.r.t. E (from Taichi)
        grad_nu: Gradient of loss w.r.t. nu (from Taichi)

    Returns:
        Tensor of shape (2,) containing [grad_raw_E, grad_raw_nu]
    """
    sig_E = torch.sigmoid(raw_params[0])
    sig_nu = torch.sigmoid(raw_params[1])

    grad_raw_E = grad_E * sig_E
    grad_raw_nu = grad_nu * 0.48 * sig_nu * (1 - sig_nu)

    return torch.tensor([grad_raw_E.item(), grad_raw_nu.item()], dtype=torch.float32)


class TaichiAdamOptimizer:
    """Adam optimizer wrapper for Taichi-based parameter estimation.

    This class bridges PyTorch's Adam optimizer with Taichi's autodiff system.
    It uses reparameterization to handle parameter constraints.

    Example:
        optimizer = TaichiAdamOptimizer(E_init=9000.0, nu_init=0.3, lr=0.01)

        for iteration in range(num_iters):
            # Get current physical params
            E, nu = optimizer.get_physical_params()

            # Run Taichi forward/backward (returns gradients)
            estimator.set_target_params(E, nu)
            result = estimator.train_step()

            # Update with Taichi gradients
            optimizer.step(result["grad_E"], result["grad_nu"])
    """

    def __init__(
        self,
        E_init: float,
        nu_init: float,
        lr: float = 0.01,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        amsgrad: bool = False,
    ):
        """Initialize the optimizer.

        Args:
            E_init: Initial Young's modulus
            nu_init: Initial Poisson's ratio
            lr: Learning rate for Adam
            betas: Coefficients for computing running averages (momentum)
            eps: Term added for numerical stability
            amsgrad: Whether to use AMSGrad variant
        """
        # Initialize raw parameters from physical values
        raw_E, raw_nu = physical_to_raw(E_init, nu_init)
        self.raw_params = torch.tensor(
            [raw_E.item(), raw_nu.item()],
            dtype=torch.float32,
            requires_grad=True,
        )

        # Create Adam optimizer
        self.optimizer = torch.optim.Adam(
            [self.raw_params], lr=lr, betas=betas, eps=eps, amsgrad=amsgrad
        )

        self._nan_count = 0

    def get_physical_params(self) -> tuple[float, float]:
        """Get current physical parameters (E, nu).

        Returns:
            Tuple of (E, nu) as Python floats
        """
        with torch.no_grad():
            E, nu = raw_to_physical(self.raw_params)
        return E.item(), nu.item()

    def step(self, grad_E: float, grad_nu: float) -> bool:
        """Perform one optimization step.

        Args:
            grad_E: Gradient of loss w.r.t. E (from Taichi autodiff)
            grad_nu: Gradient of loss w.r.t. nu (from Taichi autodiff)

        Returns:
            True if step was successful, False if skipped due to NaN
        """
        import numpy as np

        # Check for NaN gradients
        if np.isnan(grad_E) or np.isnan(grad_nu):
            self._nan_count += 1
            logger.warning(
                f"NaN gradient detected (count={self._nan_count}), skipping step"
            )
            return False

        # Zero gradients
        self.optimizer.zero_grad()

        # Compute gradients w.r.t. raw params via chain rule
        self.raw_params.grad = compute_raw_gradients(self.raw_params, grad_E, grad_nu)

        # Optimizer step
        self.optimizer.step()

        return True

    def get_state(self) -> dict:
        """Get optimizer state for checkpointing.

        Returns:
            Dictionary containing raw_params and optimizer state
        """
        return {
            "raw_params": self.raw_params.detach().clone(),
            "optimizer_state": self.optimizer.state_dict(),
            "nan_count": self._nan_count,
        }

    def load_state(self, state: dict):
        """Load optimizer state from checkpoint.

        Args:
            state: Dictionary from get_state()
        """
        self.raw_params.data.copy_(state["raw_params"])
        self.optimizer.load_state_dict(state["optimizer_state"])
        self._nan_count = state["nan_count"]
