import torch
import torch.nn as nn
import numpy as np
from .config import Config
from .models import FCN

class BurtonPINN(nn.Module):
    """
    PINN wrapper for the Burton Equation: dy/dt = a*q(t) - y/tau.
    Learns the solution 'y' and parameters 'a' and 'tau'.
    """
    def __init__(self, config: Config, u_0: float, t_0: float,
                 tau_init_norm: float, a_init_norm: float):
        super().__init__()
        self.config = config
        self.net = FCN(config, u_0, t_0)

        # -- Learnable Physics Parameters --
        # Tau is parametrized in log-space to enforce positivity (tau > 0)
        self.log_tau = nn.Parameter(torch.tensor(np.log(tau_init_norm), dtype=torch.float32))
        self.a_param = nn.Parameter(torch.tensor(a_init_norm, dtype=torch.float32))

        # Adaptive Loss Weight (Lambda)
        if config.adaptive_lambda:
            self.log_lambda = nn.Parameter(torch.tensor(np.log(config.lambda_init), dtype=torch.float32))
        else:
            self.register_buffer('fixed_lambda', torch.tensor(config.lambda_init, dtype=torch.float32))

    @property
    def tau(self):
        return torch.exp(self.log_tau)

    @property
    def lambda_val(self):
        return torch.exp(self.log_lambda) if self.config.adaptive_lambda else self.fixed_lambda

    def forward(self, t):
        return self.net(t)

    def compute_physics_loss(self, t_p: torch.Tensor, q_p: torch.Tensor) -> torch.Tensor:
        """Calculates ODE residual: R = du/dt + u/tau - a*q"""
        t_p.requires_grad_(True)
        u_pred = self.forward(t_p)

        dudt = torch.autograd.grad(
            outputs=u_pred, inputs=t_p,
            grad_outputs=torch.ones_like(u_pred), create_graph=True
        )[0]

        residual = dudt + (u_pred / (self.tau + 1e-9)) - (self.a_param * q_p)
        return torch.mean(residual ** 2)