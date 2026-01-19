# Physics-Informed Neural Network (PINN) for Burton Equation Parameter Estimation

This repository contains a PyTorch implementation of a Physics-Informed Neural Network (PINN) designed to model geomagnetic storms (specifically the SMR SuperMAG index) and estimate the physical parameters of the Burton equation.

The model simultaneously learns the solution to the differential equation and identifies the unknown physical parameters: the decay time ($\tau$) and the solar wind coupling coefficient ($a$).

## Scientific Context

The project models the evolution of the ring current using the **Burton Equation**:

$$\frac{dSMR}{dt} = a \cdot q(t) - \frac{SMR}{\tau}$$

Where:
* $SMR$ is the SuperMAG index.
* $q(t)$ is the solar wind-magnetosphere coupling function.
* $a$ is the injection efficiency parameter.
* $\tau$ (tau) is the ring current decay time.

### Key Features
* **Hard Constraints:** Implements initial condition enforcement via ansatz $u(t) = u_0 + \tanh(t-t_0) \cdot \mathcal{N}(t)$.
* **Adaptive Loss Balancing:** Uses a learnable Lagrange multiplier ($\lambda$) to automatically balance data loss and physics residuals during training.
* **RK45 Validation:** Automatically validates learned parameters by solving the ODE with a standard Runge-Kutta (RK45) solver and comparing it to the PINN prediction and observed data.
* **Ensemble Training:** Supports training multiple ensemble members with different seeds to quantify uncertainty.

## Project Structure

```text
.
├── data/
│   └── data.csv        # OMNI solar wind drivers
├── src/
│   ├── __init__.py
│   ├── config.py       # Hyperparameters, file paths, and physics constants
│   ├── data.py         # Data loading, smoothing, and normalization
│   ├── models.py       # Neural Network architectures (FCN, RFF)
│   ├── physics.py      # Physics-Informed logic (BurtonPINN class)
│   ├── trainer.py      # Training loop, optimization, and early stopping
│   └── utils.py        # Visualization, RK45 validation, and seeding tools
├── main.py             # Entry point to run experiments
├── requirements.txt    # Python dependencies
└── README.md           
````

## Environment
- **Python Version:** 3.11.6
- **CUDA Version:** 11.8.0 (Optional, code runs on CPU as well)
- **Operating System:** Tested on Linux (Broadwell architecture).

Note: It is recommended to use a virtual environment.

## Installation

1.  **Clone the repository:**

    ```bash
    git clone https://github.com/Nuelam9/PINN_Burton_Storm.git
    cd PINN_Burton_Storm    
    ```

2.  **Install dependencies:**
    It is recommended to use a virtual environment.

    ```bash
    pip install -r requirements.txt
    ```


## Data Availability

This repository contains a sample subset of the driving data (Solar Wind from OMNI) in the `data/` folder. However, **the target geomagnetic index (SMR) is not included** in `data.csv` due to redistribution policies of the SuperMAG collaboration.

To reproduce the experiments, you must:

1.  **Register:** Obtain a free account at [SuperMAG](https://supermag.jhuapl.edu/).
2.  **Download:** Download the **SMR index** (in minutes) for the time interval corresponding to the storm in this study.
3.  **Merge:** Add the SMR data as a new column named `smr` (case-sensitive) to the `data/data.csv` file.

**Note on Code Execution:** The code will raise a `KeyError: 'smr'` if this column is missing from the input CSV.

## Configuration

All hyperparameters are centralized in `src/config.py`. You can modify the following key settings:

  * **Data Paths:** Update `file_path` to point to your OMNI/SuperMag CSV file.
  * **Physics:** Change `y_col` (target) and `q_col` (driver).
  * **Training:** Adjust `train_steps`, `lrate`, and `patience`.
  * **Architecture:** Modify `layer_sizes` or `n_rff` (Random Fourier Features).

## Usage

To train the model on a specific solar wind driver (e.g., `vBs`), run:

```bash
python main.py
```

By default, the script in `main.py` iterates over a list of drivers defined in the `__main__` block.

### Output

Results are saved in the `results/` directory, organized by driver and storm ID:

  * `plots/`: Visualization of the time series fit, parameter evolution ($\tau, a$), and loss curves.
  * `history/`: CSV files containing the training history for each ensemble member.
  * `checkpoints/`: PyTorch model states (`.pth`) for the best performing epoch.

## Methodology Visualization

The training process produces a composite plot for validation:

1.  **Fit vs Observations:** Compares Ground Truth, PINN prediction, and RK45 integration (using learned params).
2.  **Loss Components:** Tracks Data Loss vs. Weighted Physics Loss.
3.  **Parameter Recovery:** Shows the convergence of $\tau$ and $a$ over training steps.

## Citation

If you use this code for your research, please cite the software archive via Zenodo:

**BibTeX:**

```bibtex
@software{lacal2025pinn,
  author       = {Lacal, Manuel},
  title        = {pinn-burton-storm: A Physics-Informed Neural Network Approach to the Gannon Storm},
  year         = 2026,
  publisher    = {Zenodo},
  version      = {v1.0.2},
  doi          = {10.5281/zenodo.18098818},
  url          = {[https://doi.org/10.5281/zenodo.18098818](https://doi.org/10.5281/zenodo.18098818)}
}

```

### Associated Paper (Under Review)

This software implements the methodology described in the manuscript:

> Lacal, M., et al. (2025). "A Physics-Informed Neural Network Approach to the Gannon Storm". *Submitted to Geophysical Research Letters*.

## License

This project is licensed under the **MIT License**. This applies to all code and scripts within this repository. See the [LICENSE](LICENSE) file for the full text. 

*Note: The data used in this project are subject to the terms of use of their respective providers (OMNI/CDAWeb and SuperMAG).*
