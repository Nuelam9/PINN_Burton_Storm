from dataclasses import dataclass, field
from typing import List
from pathlib import Path
import torch

@dataclass
class Config:
    """Configuration settings for the Burton PINN training and physics parameters."""

    # --- Paths & Identifiers ---
    file_path: str = "./data/data.csv"
    result_path: str = "results/"
    storm_id: int = 176

    # --- Data Processing ---
    window_size: int = 20  # Rolling mean window (minutes) for smoothing
    y_col: str = "smr"     # Target: SuperMAG Ring Current index
    q_col: str = "vBs"

    # --- Training Hyperparameters ---
    random_seed: int = 42
    train_steps: int = 35000
    ensemble_size: int = 1
    
    # Learning Rates
    lrate: float = 2e-4
    tau_lrate: float = 1e-3
    a_lrate: float = 2e-2
    lambda_lrate: float = 1e-4
    eps: float = 1e-8
    patience: int = 2000

    # --- Adaptive Loss Balancing ---
    adaptive_lambda: bool = True
    lambda_init: float = 0.0125

    # --- Network Architecture ---
    layer_sizes: List[int] = field(default_factory=lambda: [1, 32, 32, 32, 32, 32, 1])
    n_rff: int = 64
    sigma: float = 15.0

    # --- Physics / ODE Parameter Guesses ---
    tau_init_range: List[float] = field(default_factory=lambda: [0.0, 1.0])
    a_init_range: List[float] = field(default_factory=lambda: [-25.0, -5.0])

    # --- Collocation Point Sampling ---
    n_collocations: int = 15000

    # --- Logging & Plotting ---
    plot_every: int = 5000
    plot_final_only: bool = True
    log_steps: int = 1
    save_plots: bool = True
    verbose: bool = False

    # --- Device ---
    device: str = field(default_factory=lambda: "cuda" if torch.cuda.is_available() else "cpu")

    def __post_init__(self):
        self.result_path = Path(self.result_path)
        self.result_path.mkdir(parents=True, exist_ok=True)
        self.device = torch.device(self.device)