import pandas as pd
import numpy as np
import torch
from typing import Tuple, Dict
from .config import Config

class DataScaler:
    """Manages normalization/unscaling for time, physics variables, and parameters."""
    def __init__(self, df: pd.DataFrame, config: Config):
        self.y_col = config.y_col
        self.q_col = config.q_col

        # Time Scaling: Normalize time to [0, 1]
        self.t_start = df["Epoch"].min()
        self.t_max_minutes = (df["Epoch"].max() - self.t_start).total_seconds() / 60.0

        # Feature Scaling: Unit Variance
        self.y_std = float(df[self.y_col].std())
        self.q_std = float(df[self.q_col].std())

    def normalize_data(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        time_mins = (df["Epoch"] - self.t_start).dt.total_seconds() / 60.0
        df['time_norm'] = time_mins / self.t_max_minutes
        df[self.y_col] = df[self.y_col] / self.y_std
        df[self.q_col] = df[self.q_col] / self.q_std
        return df

    def unscale_time(self, t_norm: np.ndarray) -> np.ndarray:
        return t_norm * self.t_max_minutes

    def unscale_y(self, y_norm: np.ndarray) -> np.ndarray:
        return y_norm * self.y_std

    def unscale_params(self, params_norm: Dict[str, float]) -> Dict[str, float]:
        """Convert normalized parameters (tau, a) to physical units."""
        unscaled = {}
        for k, v in params_norm.items():
            if k == 'tau':
                # Time constant scales with time domain
                unscaled[k] = v * self.t_max_minutes
            elif k == 'a':
                # Conversion factor derived from ODE dimensional analysis
                val = v * (self.y_std / (self.q_std * self.t_max_minutes))
                unscaled[k] = val
            else:
                 unscaled[k] = v
        return unscaled

def load_and_preprocess_data(config: Config) -> Tuple[pd.DataFrame, DataScaler]:
    """Loads CSV, handles missing values, applies smoothing, and normalizes."""
    cols = ["Epoch", config.q_col, config.y_col]
    df = pd.read_csv(config.file_path, parse_dates=["Epoch"], usecols=cols)

    # Linear Interpolation
    df = df.interpolate(method="linear", limit_direction="forward", limit_area="inside")

    # Rolling Mean Smoothing
    df[config.q_col] = df[config.q_col].rolling(window=config.window_size, center=True, min_periods=1).mean()
    df[config.y_col] = df[config.y_col].rolling(window=config.window_size, center=True, min_periods=1).mean()

    scaler = DataScaler(df, config)
    df_norm = scaler.normalize_data(df)

    return df_norm, scaler