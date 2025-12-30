from dataclasses import dataclass, field
from typing import List, Dict, Optional
from pathlib import Path
import torch

@dataclass
class Config:
    """Configuration settings for the Burton PINN training and physics parameters."""

    # --- Paths & Identifiers ---
    file_path: str = "./data/data.csv"
    result_path: Path = Path("results/")
    storm_id: int = 176

    # --- Data Processing ---
    window_size: int = 20  # Rolling mean window (minutes) for smoothing
    y_col: str = "smr"     # Target: SuperMAG index
    q_col: str = "vBs"     # Input: Solar wind coupling function

    # --- Training Hyperparameters ---
    random_seed: int = 42
    train_steps: int = 35000
    ensemble_size: int = 100  # Number of runs for uncertainty quantification
    
    # Learning Rates (Separated for stability)
    lrate: float = 2e-4           # Network weights
    tau_lrate: float = 1e-3       # Decay parameter tau
    a_lrate: float = 2e-2         # Coupling parameter a
    lambda_lrate: float = 1e-4    # Adaptive loss weight
    eps: float = 1e-8
    patience: int = 2000          # Early stopping patience

    # --- Loss Balancing ---
    adaptive_lambda: bool = True  # If True, learns weights via Gradient Ascent
    lambda_init: float = 0.0125   # Initial weight for physics loss

    # --- Network Architecture ---
    # [Input, Hidden..., Output]
    layer_sizes: List[int] = field(default_factory=lambda: [1, 32, 32, 32, 32, 32, 1])
    n_rff: int = 64       # Fourier features mapping size (2m)
    sigma: float = 15.0   # sigma for Fourier features

    # --- Physics Parameter Initialization (Guesses) ---
    tau_init_range: List[float] = field(default_factory=lambda: [1e-5, 1.0]) # Normalized range
    a_init_range: List[float] = field(default_factory=lambda: [-25.0, -5.0]) # Normalized range

    param_overrides: Dict[str, List[float]] = field(default_factory=lambda: {
            'BZ_GSM': [10.0, 20.0],
            'V': [-5.0, 5.0]
        })

    # --- Collocation Point Sampling ---
    n_collocations: int = 15000   # Points sampled for ODE residual calculation

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
