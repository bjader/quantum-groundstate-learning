# Quantum Ground State Learning

Code to reproduce "Learning ground state observables from quantum computing experiments"

https://arxiv.org/abs/2606.15983

## Repository Structure

```
quantum-groundstate-learning/
├── requirements.txt          # Python dependencies (pip)
├── environment.yml           # Conda environment
├── qaml/                     # Core library
│   ├── analysis/             # Data analysis utilities
│   ├── diagonalisation/      # DMRG implementations
│   ├── graph/                # Graph and lattice structures
│   ├── ml/                   # Machine learning modules
│   ├── observables/          # Observable calculations
│   ├── skqd/                 # Sample-based Krylov Quantum Diagonalization
│   ├── trotter_circuits/     # Trotterized circuit generation
│   └── visualisation/        # Plotting and visualization
├── utils/                    # Utility functions
├── cmaps/                    # Heavy-hex lattice connectivity maps
├── figures/                  # Figure generation scripts
│   ├── fig1_overview/        # Figure 1 scripts
│   ├── fig2_data_accuracy/   # Figure 2 scripts
│   ├── fig3_in_distribution_ml/  # Figure 3 scripts
│   ├── fig4_phase_boundary_ml/   # Figure 4 scripts
│   └── supplementary/        # Supplementary figure scripts
├── ml_training/              # ML model training scripts
├── data_generation/          # Data generation pipeline
│   ├── basis_optimization/   # Basis optimization scripts
│   │   └── saved_opt_res_qpu/ # Basis optimization results (needs to be generated)
│   ├── classical_simulation/ # Classical SKQD simulations
│   ├── dmrg/                 # DMRG reference calculations
│   ├── observables/          # Observable computation from SKQD
│   └── qpu/                  # QPU experiment submission and processing
├── data/                     # Experimental data
│   ├── spins_57/             # 57-spin system data
│   └── spins_115/            # 115-spin system data
└── ml_models/                # Trained ML models (needs to be generated)
```

## Installation

**Python 3.11 required**

### Option 1: pip (Recommended)

```bash
python3.11 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### Option 2: conda/conda-forge

```bash
conda env create -f environment.yml
conda activate qgl
```

### Option 3: Poetry

```bash
poetry install
```

### System Dependencies (Optional, for graph layout rendering)

```bash
# macOS
brew install graphviz

# Linux (Debian/Ubuntu)
sudo apt install graphviz
```

## Data and Models

Data containing ground state observables from hardware experiments is available under `/data/`.

However, trained models and optimized basis parameters are not included in this repository due to their size. They can
be generated using these instructions.

### ML models

The repository provides pipelines for training the two distinct neural network architectures evaluated in the paper. The scripts automatically loop across the core observables (`z_basis_opt`, `2_nn_corr_funcs_zz_basis_opt`, and `z_loop`) and adapt to available hardware acceleration (CUDA/MPS/CPU). 

Target system sizes (`n_qubits = 57` or `115`) can be modified via configuration variables at the top of the respective execution scripts.

#### Architecture 2: Local-to-Global Heavy-Hex Network (Main Text Architecture)
This is the geometry-aware local neural network, originally defined in [Wanner et al. ](https://arxiv.org/abs/2405.18489), applied to the heavy-hex lattice. This incorporates the geometric structure of the system through the construction of local regions associated with each observable.

Execute the pipeline sequentially:
1. **Hyperparameter Cross-Validation (5-Fold):**
   ```bash
   python ml_training/train_k_cross_val2_eff.py
   ```
2. **In-Distribution Training & Testing:**
   ```bash
   python ml_training/train2_all_obs.py
   ```
3. **Phase Boundary Extrapolation Training:**
   ```bash
   python ml_training/train_boundary2.py
   ```

#### Architecture 1: Global Dense Baseline Network (Supplementary Baseline)
This is the lightweight per-observable neural network evaluated as a baseline, which relies solely on the global coupling parameter without incorporating any information about the underlying lattice geometry. Each observable is modeled independently as a function of the scalar anisotropy parameter using a simple multilayer perceptron.

Execute the pipeline sequentially:
1. **Hyperparameter Grid Search:**
   ```bash
   python ml_training/train_k_cross_val.py
   ```
2. **In-Distribution Evaluation:**
   ```bash
   python ml_training/train.py
   ```
3. **Phase Boundary Profile Training:**
   ```bash
   python ml_training/train_boundary.py
   ```

### Basis optimization results

The basis-optimised observables that the ML model is trained on are already stored in `/data/`, which is sufficient for 
many plots. However, to get certain plots from the paper (like the energy panel in Figure 2) you need to run basis 
optimization to get the final energy.

Run the two-stage variational parameter refinement using the provided optimization execution script:

```bash
python data_generation/basis_optimization/energy_two_method_optimization_wt_betas.py
```

#### Key Script Arguments
* `--edges_path`: Path to connectivity map text file (e.g., use `heavy_hex_edges_d_5.txt` for 57 qubits, `heavy_hex_edges_d_7.txt` for 115 qubits).
* `--jz` : Target Hamiltonian anisotropy.
* `--num_bs_opt`: Number of highly-weighted bitstrings filtered for Stage 1 optimization.
* `--num_bs_opt_2`: Total bitstring sample space size evaluated during Stage 2 diagonalization.

### Structure after generating models and running basis optimization

After running the ML workflows and executing the basis optimization pipeline for each Jz, your local workspace directory tree will be structured as follows:

```
quantum-groundstate-learning/
├── data/                    
│   ├── spins_57/
│   └── spins_115/
├── ml_models/               # Generated by training pipelines
│   ├── model/               # Global Baseline Network (Architecture 1)
│   ├── model2/              # Local-to-Global Network (Architecture 2)
│   ├── model_boundary/      # Phase Boundary (Architecture 1)
│   └── model_boundary2/     # Phase Boundary (Architecture 2)
└── data_generation/
    └── basis_optimization/
        └── saved_opt_res_qpu/  # Generated by optimization script (optional)
```

## Citation

```bibtex
@article{jaderberg2026learning,
  title={Learning ground state observables from quantum computing experiments},
  author={Jaderberg, Ben and Shah, Freya and Jeon, Minjun and Sahin, M Emre and Zoufal, Christa and Sharma, Kunal},
  journal={arXiv preprint arXiv:2606.15983},
  year={2026}
}
```

## Data availability

Models and data not included in the repository can be made avaialble upon reasonable request.

## License

This project uses the Apache License 2.0 - see [LICENSE](LICENSE).