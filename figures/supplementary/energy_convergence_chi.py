"""
Creates supplementary Figure 1.
Plot DMRG energy extrapolation to chi -> infinity for n=57 and n=115.
"""

import logging
import sys
from pathlib import Path

import dill as pickle
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

# Fix for unpickling objects from original /qaml/ repo
import qaml.diagonalisation
import qaml.diagonalisation.twod
import qaml.diagonalisation.twod.dmrg
sys.modules['diagonalisation'] = qaml.diagonalisation
sys.modules['diagonalisation.twod'] = qaml.diagonalisation.twod
sys.modules['diagonalisation.twod.dmrg'] = qaml.diagonalisation.twod.dmrg

# Configure matplotlib to use LaTeX for text rendering (manuscript style)
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman"],
    "font.size": 10,
    "axes.labelsize": 10,
    "axes.titlesize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.titlesize": 10,
    "text.latex.preamble": r"\usepackage{amsmath}",
})

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

def load_pickle(path: Path):
    """Load a pickle file."""
    with path.open("rb") as f:
        return pickle.load(f)

def ensure_dir(path: Path):
    """Create directory if it doesn't exist."""
    path.mkdir(parents=True, exist_ok=True)
    return path

def exponential_decay(chi, E_inf, A, alpha):
    """
    Model: E(chi) = E_inf + A * exp(-alpha * chi)
    where E_inf is the extrapolated energy at chi -> infinity
    """
    return E_inf + A * np.exp(-alpha * chi)

def extrapolate_to_infinite_chi(chi_list, energy_list, min_chi_for_fit=40):
    """
    Fit energies to exponential decay and extrapolate to chi -> infinity.
    
    Returns:
        E_inf: extrapolated energy at infinite chi
        fit_params: (E_inf, A, alpha)
        fit_success: whether fit converged
    """
    chi_arr = np.array(chi_list)
    energy_arr = np.array(energy_list)
    
    mask = chi_arr >= min_chi_for_fit
    if np.sum(mask) < 3:  # Need at least 3 points
        return None, None, False
    
    chi_fit = chi_arr[mask]
    energy_fit = energy_arr[mask]
    
    # Initial guess
    E_guess = energy_fit[-1]
    A_guess = energy_fit[0] - energy_fit[-1]
    alpha_guess = 0.01
    
    try:
        popt, _ = curve_fit(
            exponential_decay,
            chi_fit,
            energy_fit,
            p0=[E_guess, A_guess, alpha_guess],
            maxfev=10000
        )
        return popt[0], popt, True
    except Exception:
        return None, None, False

def load_dmrg_energies(n_spins, chi_values, jz_values, trunc_cut=1e-6):
    """Load DMRG energies for different chi values."""
    base_data = Path("../../data")
    energies_by_chi = {}
    
    for chi in chi_values:
        dmrg_dir = base_data / f"spins_{n_spins}" / "dmrg" / f"chi_max_{chi}_trunc_{trunc_cut:.0e}"
        
        if not dmrg_dir.exists():
            logging.warning(f"DMRG directory not found for n={n_spins}, chi={chi}")
            continue
            
        energies = {}
        for jz in jz_values:
            dmrg_file = dmrg_dir / f"XXZ_2d_jz_{jz:.1f}.pkl"
            
            if not dmrg_file.exists():
                continue
                
            try:
                dmrg_res = load_pickle(dmrg_file)
                energy = float(dmrg_res["result"].e)
                energies[jz] = energy
            except Exception as ex:
                logging.warning(f"Failed to load DMRG for n={n_spins}, chi={chi}, Jz={jz:.1f}: {ex}")
                continue
        
        if energies:
            energies_by_chi[chi] = energies
            logging.info(f"Loaded {len(energies)} Jz values for n={n_spins}, chi={chi}")
    
    return energies_by_chi

def plot_extrapolation(n_spins, chi_values, jz_values, save_dir):
    """Plot energy extrapolation to chi -> infinity."""
    energies_by_chi = load_dmrg_energies(n_spins, chi_values, jz_values)
    
    if not energies_by_chi:
        logging.error(f"No data found for n={n_spins}")
        return
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(5, 3))
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(jz_values)))
    
    extrapolated_energies = {}
    extrapolation_errors = {}
    ref_chi = 20
    
    for i, jz in enumerate(jz_values):
        chi_list = []
        energy_list = []
        
        for chi in sorted(energies_by_chi.keys()):
            if jz in energies_by_chi[chi]:
                chi_list.append(chi)
                energy_list.append(energies_by_chi[chi][jz])
        
        if len(chi_list) < 4:
            continue
        
        # Get reference energy at chi=20
        if ref_chi not in energies_by_chi or jz not in energies_by_chi[ref_chi]:
            continue
        
        E_ref = energies_by_chi[ref_chi][jz]
        
        # Perform extrapolation on absolute energies
        E_inf, fit_params, success = extrapolate_to_infinite_chi(chi_list, energy_list)
        
        if success and E_inf is not None:
            # Plot data points (absolute value of energy difference from chi=20)
            energy_rel = np.abs(np.array(energy_list) - E_ref)
            ax1.plot(chi_list, energy_rel, 'o', color=colors[i], markersize=6,
                    label=f'$J_z={jz:.1f}$')
            
            # Plot fitted curve (absolute value of energy difference from chi=20)
            chi_dense = np.linspace(min(chi_list), max(chi_list) * 1.5, 100)
            energy_fit = np.abs(exponential_decay(chi_dense, *fit_params) - E_ref)
            ax1.plot(chi_dense, energy_fit, '-', color=colors[i], alpha=0.5, linewidth=1.5)
            
            # Mark extrapolated value (absolute value of energy difference from chi=20)
            E_inf_rel = abs(E_inf - E_ref)
            ax1.axhline(y=E_inf_rel, color=colors[i], linestyle='--', alpha=0.3, linewidth=1)
            
            # Store results (absolute energies)
            extrapolated_energies[jz] = E_inf
            if 320 in energies_by_chi and jz in energies_by_chi[320]:
                chi_320_energies = energies_by_chi[320][jz]
                error = abs(chi_320_energies - E_inf)
                extrapolation_errors[jz] = error
    
    ax1.set_xlabel(r'Bond dimension $\chi$')
    ax1.set_ylabel(rf'$|E(\chi) - E(\chi={ref_chi})|$')
    ax1.set_xscale('log')
    ax1.set_yscale('log')

    # Set x-axis ticks to show actual chi values including those we used
    chi_ticks = [20, 40, 80, 160, 320, 640]
    ax1.set_xticks(chi_ticks)
    ax1.set_xticklabels([str(chi) for chi in chi_ticks])

    ax1.grid(True, alpha=0.3, which='both')

    # Store legend handles and labels for separate legend file
    handles1, labels1 = ax1.get_legend_handles_labels()

    # Plot error: |E(320) - E(inf)|
    if extrapolation_errors:
        jz_list = sorted(extrapolation_errors.keys())
        errors = [extrapolation_errors[jz] for jz in jz_list]
        rel_errors = [abs(extrapolation_errors[jz] / extrapolated_energies[jz])
                     for jz in jz_list]

        ax2.plot(jz_list, errors, 'o-', markersize=8, linewidth=2, color='darkblue',
                label='Absolute error')
        ax2.set_xlabel(r'$J_z$')
        ax2.set_ylabel(r'$|E(\chi=320) - E(\chi \to \infty)|$')
        ax2.set_yscale('log')
        ax2.grid(True, alpha=0.3, which='both')

    fig.tight_layout()
    fig.savefig(save_dir / f"energy_extrapolation_chi_inf_n{n_spins}.pdf", bbox_inches='tight')
    plt.close(fig)

    # Save legend separately
    if handles1:
        fig_legend = plt.figure(figsize=(8, 2))
        fig_legend.legend(handles1, labels1, loc='center', ncol=5, frameon=False)
        fig_legend.savefig(save_dir / f"energy_extrapolation_legend_n{n_spins}.pdf", bbox_inches='tight')
        plt.close(fig_legend)

    logging.info(f"Saved extrapolation plot and legend for n={n_spins}")
    
    # Print summary statistics
    if extrapolation_errors:
        print(f"\n=== Extrapolation Summary for n={n_spins} ===")
        print(f"Number of Jz values successfully extrapolated: {len(extrapolation_errors)}")
        print(f"Median absolute error |E(320) - E(∞)|: {np.median(errors):.2e}")
        print(f"Median relative error: {np.median(rel_errors):.2e} ({np.median(rel_errors)*100:.4f}%)")
        print(f"Max absolute error: {np.max(errors):.2e}")
        print(f"Max relative error: {np.max(rel_errors):.2e} ({np.max(rel_errors)*100:.4f}%)")

if __name__ == "__main__":
    # Configuration
    chi_values = [20, 40, 80, 160, 320]
    jz_values = [1.1] + [round(jz, 1) for jz in np.arange(1.5, 5.51, 0.5)]
    
    save_dir = ensure_dir(Path("plots"))
    
    # Plot for n=57
    plot_extrapolation(57, chi_values, jz_values, save_dir)
    
    # Plot for n=115
    plot_extrapolation(115, chi_values, jz_values, save_dir)
    
    logging.info("Extrapolation plots completed successfully")
