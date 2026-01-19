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
    seed = int(seed)
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
                 t_obs_tensor: torch.Tensor, y_obs_tensor: torch.Tensor,
                 q_obs_tensor: torch.Tensor, scaler: DataScaler,
                 config: Config, ens_id: int, step: int,
                 save: bool = False, best_params: Optional[Dict] = None):
    
    model.eval()
    
    # --- Predictions & Metrics (Unscaled to Physical Units) ---
    with torch.no_grad():
        y_pinn_norm = model(t_obs_tensor.to(config.device)).cpu().numpy().flatten()
    
    t_norm = t_obs_tensor.cpu().numpy().flatten()
    y_obs_phys = scaler.unscale_y(y_obs_tensor.cpu().numpy().flatten())
    y_pinn_phys = scaler.unscale_y(y_pinn_norm)
    t_phys_mins = scaler.unscale_time(t_norm)
    
    # Calculate RMSE & R2 for PINN
    rmse_pinn = np.sqrt(np.mean((y_pinn_phys - y_obs_phys) ** 2))
    r2_pinn = r2_score(y_obs_phys, y_pinn_phys)

    # RK45 Validation
    if best_params is None:
        best_params = {'tau': history['tau_norm'][-1], 'a': history['a_norm'][-1]}
            
    q_obs_norm = q_obs_tensor.cpu().numpy().flatten()
    y_rk45_norm = validate_rk45(best_params, t_norm, y_obs_tensor[0].item(), q_obs_norm, scaler)
    y_rk45_phys = scaler.unscale_y(y_rk45_norm)
    
    # Calculate RMSE & R2 for RK45
    rmse_rk45 = np.sqrt(np.mean((y_rk45_phys - y_obs_phys) ** 2))
    r2_rk45 = r2_score(y_obs_phys, y_rk45_phys)

    # --- Title Generation ---
    rff_str = ""
    if config.n_rff > 0:
        rff_str = f"RFF ({config.n_rff}, {config.sigma})"

    layer_str = '-'.join(map(str, config.layer_sizes))
    arch_parts = [f"Layers: {layer_str}"]
    if rff_str:
        arch_parts.append(rff_str)
    arch_str = f"({', '.join(arch_parts)})"

    collnum = config.n_collocations
    
    title = (f"PINN Training: Target: {config.y_col}, Q: {config.q_col}\n"
             f"Step: {step}, Arch: {arch_str}, "
             f"Collnum: {collnum}, CollSampling: Uniform, Ens: {ens_id}/{config.ensemble_size}")

    # --- Plotting ---
    fig, axs = plt.subplots(3, 2, figsize=(15, 12), constrained_layout=True)
    fig.suptitle(title, fontsize=14)
    
    # Panel 1: Time Series
    ax = axs[0, 0]
    ax.plot(t_phys_mins, y_obs_phys, 'k-', alpha=0.3, label='Data', lw=1, zorder=1)
    ax.plot(t_phys_mins, y_pinn_phys, 'r-', linewidth=1.5, alpha=0.8, 
            label=f'PINN (R²={r2_pinn:.3f}, RMSE={rmse_pinn:.2f})', zorder=2)
    ax.plot(t_phys_mins, y_rk45_phys, 'g--', linewidth=1.5, 
            label=f'RK45 (R²={r2_rk45:.3f}, RMSE={rmse_rk45:.2f})', zorder=3)
    
    ax.set_title(f"Fit vs Observations ({config.y_col})")
    ax.set_ylabel(f"{config.y_col} (nT)")
    ax.set_xlabel("Time (minutes)")
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)

    # Helper for vertical lines
    def add_vline(ax_obj):
        ax_obj.axvline(x=step, color='gray', linestyle='--', alpha=0.5, label='Best Step' if not ax_obj.get_legend() else "")

    # Panel 2: Losses
    ax = axs[0, 1]
    steps = history['step']
    loss_phys = np.array(history['loss_phys']) * np.array(history['lambda']) # Weighted
    loss_data = np.array(history['loss_data'])
    
    ax.semilogy(steps, loss_phys, 'r', alpha=0.7, label=r'$\lambda \cdot \mathcal{L}_r$')
    ax.semilogy(steps, loss_data, 'b', alpha=0.7, label=r'$\mathcal{L}_d$')
    add_vline(ax)
    ax.set_title("Component Losses")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Panel 3: Total Loss
    ax = axs[1, 0]
    ax.semilogy(steps, history['loss_total'], 'k-')
    add_vline(ax)
    ax.set_title("Total Loss")
    ax.grid(True, alpha=0.3)

    # Panel 4: Tau (Physical)
    ax = axs[1, 1]
    tau_phys = np.array(history['tau_norm']) * scaler.t_max_minutes
    ax.plot(steps, tau_phys, 'g-')
    add_vline(ax)
    ax.set_title(r"$\tau$ (Minutes)")
    ax.grid(True, alpha=0.3)

    # Panel 5: Lambda
    ax = axs[2, 0]
    ax.plot(steps, history['lambda'], 'purple')
    add_vline(ax)
    ax.set_title(r"$\lambda$ (Adaptive Weight)")
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)

    # Panel 6: 'a' (Physical)
    ax = axs[2, 1]
    a_conv = (scaler.y_std / (scaler.q_std * scaler.t_max_minutes))
    a_phys = np.array(history['a_norm']) * a_conv
    ax.plot(steps, a_phys, 'orange')
    add_vline(ax)
    ax.set_title(f"Coupling 'a' (Coeff of {config.q_col})")
    ax.grid(True, alpha=0.3)

    if save:
        q_clean = config.q_col.replace('/', '_')
        plot_dir = config.result_path / q_clean / f"storm_{config.storm_id}" / "plots"
        plot_dir.mkdir(parents=True, exist_ok=True)
        # Fix filename format
        filename = f"pinn_{config.y_col}_{q_clean}_storm{config.storm_id}_ens{ens_id}_step{step}.png"
        plt.savefig(plot_dir / filename, dpi=150)
        plt.close()
    else:
        plt.show()
