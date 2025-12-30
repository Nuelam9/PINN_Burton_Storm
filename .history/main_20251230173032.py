import numpy as np
import torch
import pandas as pd
from scipy.interpolate import interp1d
from sklearn.metrics import r2_score

from src.config import Config
from src.data import load_and_preprocess_data
from src.physics import BurtonPINN
from src.trainer import PINNTrainer
from src.utils import set_seed, validate_rk45, plot_results

def run_experiment(q_col_target: str):
    # Initialize Configuration
    config = Config()
    config.q_col = q_col_target

    # Specific parameter initialization ranges for Bz and V
    overrides = {'BZ_GSM': [10.0, 20.0], 'V': [-5.0, 5.0]}
    if config.q_col in overrides:
        config.a_init_range = overrides[config.q_col]
        print(f"Override: {config.q_col} init range set to {config.a_init_range}")

    # Load Data 
    df_norm, scaler = load_and_preprocess_data(config)
    
    # Prepare Tensors (Data for Loss)
    t_obs_np = df_norm['time_norm'].values
    u_obs_np = df_norm[config.y_col].values
    q_obs_np = df_norm[config.q_col].values

    t_obs = torch.tensor(t_obs_np, dtype=torch.float32).view(-1, 1)
    u_obs = torch.tensor(u_obs_np, dtype=torch.float32).view(-1, 1)
    q_obs = torch.tensor(q_obs_np, dtype=torch.float32).view(-1, 1)

    # Ensemble Training Loop
    master_rng = np.random.default_rng(config.random_seed)
    ensemble_seeds = master_rng.integers(0, 2**32 - 1, size=config.ensemble_size)
    print(f"Starting Training for {q_col_target} with {config.ensemble_size} ensemble members.")

    for i, seed in enumerate(ensemble_seeds):
        set_seed(seed)
        ens_id = i + 1

        # Collocation Sampling
        t_phys_np = np.random.uniform(0, 1, config.n_collocations)
        f_interp = interp1d(t_obs_np, q_obs_np, kind='linear', fill_value="extrapolate")
        q_phys_np = f_interp(t_phys_np)

        t_phys = torch.tensor(t_phys_np, dtype=torch.float32).view(-1, 1)
        q_phys = torch.tensor(q_phys_np, dtype=torch.float32).view(-1, 1)

        # Initialize Model & Trainer
        u0, t0 = u_obs[0].item(), t_obs[0].item()
        tau_init = np.random.uniform(*config.tau_init_range)
        a_init = np.random.uniform(*config.a_init_range)

        model = BurtonPINN(config, u0, t0, tau_init, a_init)
        trainer = PINNTrainer(model, config, scaler)

        try:
            hist, best_state = trainer.train(t_obs, u_obs, t_phys, q_phys, ens_id)
            trainer.save_checkpoint(ens_id)

            # Save CSV
            q_clean = config.q_col.replace('/', '_')
            hist_path = config.result_path / q_clean / f"storm_{config.storm_id}" / "history"
            hist_path.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(hist).to_csv(hist_path / f"hist_ens{ens_id}.csv", index=False)

            # Final Validation
            best_params = best_state['params_norm']
            y_val_norm = validate_rk45(best_params, t_obs_np, u0, q_obs_np, scaler)
            y_val_phys = scaler.unscale_y(y_val_norm)
            u_obs_phys = scaler.unscale_y(u_obs_np)
            
            final_r2 = r2_score(u_obs_phys, y_val_phys)
            phys_params = scaler.unscale_params(best_params)
            
            print(f"Ens {ens_id} Result: a={phys_params['a']:.2f}, tau={phys_params['tau']:.2f} | R2={final_r2:.3f}")

            # Plot Best State
            model.load_state_dict(best_state['model_state'])
            plot_results(hist, model, t_obs, u_obs, q_obs, scaler, config, ens_id, 
                         step=best_state['step'], save=config.save_plots, best_params=best_params)

        except Exception as e:
            print(f"Ens {ens_id} Failed: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    # List of drivers to test
    target_drivers = [
        'sw_E', 'vBs','BZ_GSM', 'V', 'Np', 'p',
        'Bs', 'epsilon', 'epsilon_2', 'epsilon_3', 'E_KL', 'E_KL_1_2',
        'E_KLV', 'E_WAV', 'E_WAV_2', 'E_WAV_1_2', 'E_WV',
        'E_SR', 'E_TL', 'dPhi_MP_dt', 'p_1_2_dPhi_MP_dt', 'E_B'
             ]
    
    for driver in target_drivers:
        run_experiment(driver)