import numpy as np
import torch
import random
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d
from sklearn.metrics import r2_score
from typing import Dict, Optional, Any
from .config import Config
from .data import DataScaler

def set_seed(seed: int):
    """Enforces reproducibility across numpy, random, and torch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def validate_rk45(best_params_norm: Dict, t_eval: np.ndarray, y0: float,
                  q_vals: np.ndarray, scaler: DataScaler) -> np.ndarray:
    """
    Validates learned parameters by solving the Burton ODE with a standard 
    Runge-Kutta (RK45) solver. Serves as Ground Truth verification.
    """
    t_64 = t_eval.astype(np.float64)
    q_64 = q_vals.astype(np.float64)
    y0_64 = float(y0)

    a_norm = float(best_params_norm['a'])
    tau_norm = float(best_params_norm['tau'])

    # Interpolate input driver q(t) for the ODE solver
    q_interp = interp1d(t_64, q_64, kind='linear', fill_value="extrapolate")

    def ode_func(t, y):
        return a_norm * q_interp(t) - y / (tau_norm + 1e-9)

    sol = solve_ivp(
        ode_func, [t_64.min(), t_64.max()], [y0_64],
        t_eval=t_64, method='RK45', rtol=1e-5, atol=1e-8
    )
    return sol.y[0] if sol.success else np.zeros_like(t_eval)

def plot_results(history: Dict, model: torch.nn.Module,
                 t_obs_tensor: torch.Tensor, u_obs_tensor: torch.Tensor,
                 q_obs_tensor: torch.Tensor, scaler: DataScaler,
                 config: Config, ens_id: int, step: int,
                 save: bool = False, best_params: Optional[Dict] = None):
    
    model.eval()
    with torch.no_grad():
        u_pinn_norm = model(t_obs_tensor.to(config.device)).cpu().numpy().flatten()

    t_norm = t_obs_tensor.cpu().numpy().flatten()
    q_obs_norm = q_obs_tensor.cpu().numpy().flatten()
    u_0_val = u_obs_tensor[0].item()

    if best_params is None:
        best_params = {'tau': history['tau_norm'][-1], 'a': history['a_norm'][-1]}

    y_rk45_norm = validate_rk45(best_params, t_norm, u_0_val, q_obs_norm, scaler)
    y_rk45_phys = scaler.unscale_y(y_rk45_norm)
    u_obs_phys = scaler.unscale_y(u_obs_tensor.cpu().numpy().flatten().astype(np.float64))
    u_pinn_phys = scaler.unscale_y(u_pinn_norm.astype(np.float64))
    t_phys_mins = scaler.unscale_time(t_norm)

    r2_pinn = r2_score(u_obs_phys, u_pinn_phys)
    r2_rk45 = r2_score(u_obs_phys, y_rk45_phys)

    fig, axs = plt.subplots(3, 2, figsize=(15, 12), constrained_layout=True)
    fig.suptitle(f"PINN Validation: {config.y_col} | Step {step} | Ens {ens_id}")

    # Plot 1: Fit
    ax = axs[0, 0]
    ax.plot(t_phys_mins, u_obs_phys, 'k.', alpha=0.3, label='Data')
    ax.plot(t_phys_mins, u_pinn_phys, 'r-', alpha=0.8, label=f'PINN (R2={r2_pinn:.2f})')
    ax.plot(t_phys_mins, y_rk45_phys, 'g--', label=f'RK45 (R2={r2_rk45:.2f})')
    ax.legend()
    ax.set_title("Time Series Prediction")

    # Plot 2: Losses
    ax = axs[0, 1]
    ax.semilogy(history['step'], history['loss_phys'], label='L_phys')
    ax.semilogy(history['step'], history['loss_data'], label='L_data')
    ax.set_title("Loss Components")
    ax.legend()

    # Plot 3: Parameters (Tau, a, Lambda)
    axs[1, 1].plot(history['step'], np.array(history['tau_norm']) * scaler.t_max_minutes)
    axs[1, 1].set_title("Tau (min)")
    
    axs[2, 1].plot(history['step'], history['a_norm']) # Note: Keeps normalized for trend viewing
    axs[2, 1].set_title("Parameter 'a' (Normalized)")

    axs[2, 0].plot(history['step'], history['lambda'])
    axs[2, 0].set_yscale('log')
    axs[2, 0].set_title("Adaptive Lambda")

    if save:
        q_clean = config.q_col.replace('/', '_')
        plot_dir = config.result_path / q_clean / f"storm_{config.storm_id}" / "plots"
        plot_dir.mkdir(parents=True, exist_ok=True)
        plt.savefig(plot_dir / f"pinn_ens{ens_id}_step{step}.png", dpi=100)
        plt.close()
    else:
        plt.show()