# Quantum Ground State Learning

Code to reproduce "Deep learning of ground state observables from quantum computing experiments."
arxiv.TODO

## Repository Structure

```
quantum-groundstate-learning/
├── requirements.txt          # Python dependencies (pip)
├── environment.yml           # Conda environment
├── qaml/                     # Core library
│   ├── analysis/             # Data analysis utilities
│   ├── diagonalisation/      # DMRG implementations
│   ├── experiment/           # Experiment utilities
│   ├── graph/                # Graph and lattice structures
│   ├── ipr/                  # Inverse participation ratio
│   ├── ml/                   # Machine learning modules
│   ├── observables/          # Observable calculations
│   ├── quantum_info/         # Quantum information tools
│   ├── shadows/              # Classical shadow methods
│   ├── shadows_rdm/          # Reduced density matrix shadows
│   ├── skqd/                 # Shadow Krylov Quantum Diagonalization
│   ├── trotter_circuits/     # Trotterized circuit generation
│   └── visualisation/        # Plotting and visualization
├── utils/                    # Experiment utilities
├── figures/                  # Figure generation scripts
├── ml_training/              # ML model training scripts
├── ml_models/                # Trained ML models (downloaded from Box)
└── data_generation/          # Data generation pipeline
    ├── basis_optimization/   # Basis optimization scripts
    ├── classical_simulation/ # Classical SKQD simulations
    ├── dmrg/                 # DMRG reference calculations
    ├── observables/          # Observable computation from SKQD
    └── qpu/                  # QPU experiment submission and processing
```

## Installation

**Python 3.11 required**

### Option 1: pip (Recommended)

```bash
python3.11 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### Option 2: conda/mamba

```bash
conda env create -f environment.yml
conda activate qgl
```

### Option 3: Poetry

```bash
poetry install
```

### System Dependencies

```bash
# macOS
brew install graphviz

# Ubuntu/Debian
sudo apt-get install graphviz graphviz-dev

# Fedora/RHEL
sudo dnf install graphviz graphviz-devel
```

## Data and Models

**Data and trained models are NOT included in this repository** due to their size. They must be downloaded separately from Box.

### Downloading from Box

All required data and models are hosted in the **QAML** IBM Box folder (owned by Christa Zoufal and Kunal Sharma).

Download the following from the Box folder:

1. **`data/`** folder - Contains all experimental and simulation data
   - `data/spins_57/` - 57-spin system data
   - `data/spins_115/` - 115-spin system data
   
2. **`ml_models/`** folder - Pre-trained ML models
   - Architecture 1 and architecture 2 models
   - Models for Z, ZZ, and Z-loop observables
   - Both 57 and 115 spin systems

3. **`data_generation/basis_optimization/saved_opt_res_qpu/`** - Basis optimization results from QPU experiments
   - Required for reproducing Figure 2 and supplementary figures

**Box folder**: QAML (IBM Box, owned by Christa Zoufal and Kunal Sharma)

### Setup After Download

After downloading, place the folders in the repository root:

```
quantum-groundstate-learning/
├── data/                    # Downloaded from Box
│   ├── spins_57/
│   └── spins_115/
├── ml_models/               # Downloaded from Box
│   ├── spins_57/
│   └── spins_115/
└── data_generation/
    └── basis_optimization/
        └── saved_opt_res_qpu/  # Downloaded from Box
```

### Creating Unified Structure for Basis Optimization Results

**Important**: Before running Figure 2 or supplementary figure scripts that use basis optimization data, you must run:

```bash
cd data_generation/basis_optimization
python create_unified_structure.py
```

This script creates a unified directory structure (`saved_opt_res_qpu_unified/`) with symbolic links that organize the basis optimization results with consistent naming conventions. This is required for:
- `figures/fig2_data_accuracy/energy_vs_gap_combined_n57_n115.py`
- `figures/fig2_data_accuracy/unified_obs_vs_dmrg_basis_opt.py`
- `figures/fig2_data_accuracy/obs_vs_dmrg_z_loop_basis_opt_per_jz_vminmax.py`
- `figures/supplementary/energy_vs_gap_combined_n57_n115_recovery_basis_opt_comparison.py`
- Other scripts that reference basis optimization results

## Citation

```bibtex
[Add citation here]
```

## License

[Add license]

## Contact

[Add contact information]

## Acknowledgments

This work was performed using IBM Quantum systems.
