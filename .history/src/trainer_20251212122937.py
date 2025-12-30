import torch
import numpy as np
from tqdm import tqdm
from dataclasses import asdict
from typing import Dict, Optional
from .config import Config
from .physics import BurtonPINN
from .data import DataScaler
from .utils import plot_results

class PINNTrainer:
    def __init__(self, model: BurtonPINN, config: Config, scaler: DataScaler):
        self.model = model.to(config.device)
        self.config = config
        self.scaler = scaler

        self.optimizer = torch.optim.Adam([
            {'params': self.model.net.parameters(), 'lr': config.lrate},
            {'params': [self.model.log_tau], 'lr': config.tau_lrate},
            {'params': [self.model.a_param], 'lr': config.a_lrate}
        ], eps=config.eps)

        self.opt_lambda = None
        if config.adaptive_lambda:
            self.opt_lambda = torch.optim.Adam([self.model.log_lambda], lr=config.lambda_lrate)

        self.history = {k: [] for k in ['step', 'loss_phys', 'loss_data', 'tau_norm', 'a_norm', 'lambda', 'loss_total']}
        self.best_loss = float('inf')
        self.best_state = None
        self.patience_counter = 0

    def train(self, t_obs, u_obs, t_phys, q_phys, ens_id: int):
        dev = self.config.device
        t_obs, u_obs = t_obs.to(dev), u_obs.to(dev)
        t_phys, q_phys = t_phys.to(dev), q_phys.to(dev)

        pbar = tqdm(range(self.config.train_steps), desc=f"Ens {ens_id}", leave=False)

        for step in pbar:
            self.model.train()

            # Forward & Loss
            loss_phys = self.model.compute_physics_loss(t_phys, q_phys)
            u_pred = self.model(t_obs)
            loss_data = torch.mean((u_pred - u_obs) ** 2)
            
            loss_total = loss_data + (self.model.lambda_val * loss_phys)

            if not torch.isfinite(loss_total):
                print(f"Ens {ens_id}: Loss infinite at step {step}. Breaking.")
                break

            # Optimization
            self.optimizer.zero_grad()
            if self.opt_lambda: self.opt_lambda.zero_grad()

            loss_total.backward(retain_graph=self.config.adaptive_lambda)
            self.optimizer.step()

            if self.opt_lambda: # Gradient Ascent for lambda
                self.model.log_lambda.grad *= -1.0
                self.opt_lambda.step()

            # Logging & Early Stopping
            if step % self.config.log_steps == 0:
                self._log_history(step, loss_total, loss_phys, loss_data)
                
            if loss_total.item() < self.best_loss:
                self.best_loss = loss_total.item()
                self.best_state = self._get_state(step)
                self.patience_counter = 0
            else:
                self.patience_counter += 1
                if self.patience_counter > self.config.patience:
                    print(f"Ens {ens_id}: Early stopping at step {step}")
                    break
            
            # Intermediate Plotting
            if not self.config.plot_final_only and step % self.config.plot_every == 0 and step > 0:
                 plot_results(self.history, self.model, t_obs, u_obs, q_phys, 
                              self.scaler, self.config, ens_id, step, save=self.config.save_plots)

        if self.best_state:
            self.model.load_state_dict(self.best_state['model_state'])

        return self.history, self.best_state

    def save_checkpoint(self, ens_id: int):
        if not self.best_state: return
        q_clean = self.config.q_col.replace('/', '_')
        ckpt_dir = self.config.result_path / q_clean / f"storm_{self.config.storm_id}" / "checkpoints"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        path = ckpt_dir / f"pinn_{self.config.y_col}_{q_clean}_storm{self.config.storm_id}_ens{ens_id}_best.pth"
        
        save_dict = self.best_state.copy()
        save_dict['config'] = asdict(self.config)
        torch.save(save_dict, path)

    def _log_history(self, step, total, phys, data):
        self.history['step'].append(step + 1)
        self.history['loss_total'].append(total.item())
        self.history['loss_phys'].append(phys.item())
        self.history['loss_data'].append(data.item())
        self.history['tau_norm'].append(self.model.tau.item())
        self.history['a_norm'].append(self.model.a_param.item())
        self.history['lambda'].append(self.model.lambda_val.item())

    def _get_state(self, step):
        return {
            'step': step + 1,
            'model_state': self.model.state_dict(),
            'params_norm': {'tau': self.model.tau.item(), 'a': self.model.a_param.item()}
        }