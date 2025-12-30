import torch
import torch.nn as nn
from .config import Config

class RFFLayer(nn.Module):
    """Random Fourier Features layer for high-frequency mapping."""
    def __init__(self, input_dim: int, mapping_size: int, sigma: float):
        super().__init__()
        self.linear = nn.Linear(input_dim, mapping_size // 2, bias=False)
        B = torch.normal(0., sigma, (mapping_size // 2, input_dim))
        self.linear.weight = nn.Parameter(B, requires_grad=False)
        self.two_pi = 2. * torch.pi

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        proj = self.linear(x)
        return torch.cat([torch.sin(self.two_pi * proj), torch.cos(self.two_pi * proj)], dim=-1)

class FCN(nn.Module):
    """Fully Connected Network with hard initial condition constraint."""
    def __init__(self, config: Config, u_0: float, t_0: float):
        super().__init__()
        # Hard Constraint: u(t) = u0 + tanh(t - t0) * NN(t)
        self.register_buffer('u_0', torch.tensor(u_0, dtype=torch.float32))
        self.register_buffer('t_0', torch.tensor(t_0, dtype=torch.float32))

        layers = []
        input_dim = config.layer_sizes[0]
        current_dim = input_dim

        if config.n_rff > 0:
            rff = RFFLayer(input_dim, config.n_rff, config.sigma)
            layers.append(rff)
            current_dim = config.n_rff

        for hidden_dim in config.layer_sizes[1:-1]:
            layers.extend([nn.Linear(current_dim, hidden_dim), nn.Tanh()])
            current_dim = hidden_dim

        layers.append(nn.Linear(current_dim, config.layer_sizes[-1]))
        self.net = nn.Sequential(*layers)
        self.net.apply(self._init_weights)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        nn_out = self.net(t)
        return self.u_0 + torch.tanh(t - self.t_0) * nn_out

    @staticmethod
    def _init_weights(m: nn.Module):
        if isinstance(m, nn.Linear) and m.weight.requires_grad:
            gain = nn.init.calculate_gain('tanh')
            nn.init.xavier_uniform_(m.weight, gain=gain)
            if m.bias is not None: nn.init.constant_(m.bias, 0)