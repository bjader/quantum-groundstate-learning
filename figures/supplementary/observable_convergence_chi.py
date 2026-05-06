"""
Creates supplementary Figures 2, 3 and 4
Plot DMRG observable extrapolation to chi -> infinity for n=57 and n=115.
Analyzes Z, ZZ, and Z_loop observables.
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

def exponential_decay(chi, O_inf, A, alpha):
    """Model: O(chi) = O_inf + A * exp(-alpha * chi)"""
    return O_inf + A * np.exp(-alpha * chi)

def remove_outliers_iqr(chi_arr, obs_arr, iqr_multiplier=3.0):
    """
    Remove outliers using the Interquartile Range (IQR) method.
    
    Parameters:
    - chi_arr: array of chi values
    - obs_arr: array of observable values
    - iqr_multiplier: multiplier for IQR (default 10.0 for very extreme outliers only)
    
    Returns:
    - chi_clean, obs_clean: arrays with outliers removed
    - outlier_mask: boolean mask indicating outliers
    """
    if len(obs_arr) < 4:
        return chi_arr, obs_arr, np.zeros(len(obs_arr), dtype=bool)
    
    Q1 = np.percentile(obs_arr, 25)
    Q3 = np.percentile(obs_arr, 75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - iqr_multiplier * IQR
    upper_bound = Q3 + iqr_multiplier * IQR
    
    outlier_mask = (obs_arr < lower_bound) | (obs_arr > upper_bound)
    inlier_mask = ~outlier_mask
    
    return chi_arr[inlier_mask], obs_arr[inlier_mask], outlier_mask

def extrapolate_to_infinite_chi(chi_list, obs_list, min_chi_for_fit=40):
    """Fit observable to exponential decay and extrapolate to chi -> infinity."""
    chi_arr = np.array(chi_list)
    obs_arr = np.array(obs_list)
    
    # Remove outliers before fitting
    chi_clean, obs_clean, outlier_mask = remove_outliers_iqr(chi_arr, obs_arr)
    
    if np.any(outlier_mask):
        outlier_chis = chi_arr[outlier_mask]
        outlier_vals = obs_arr[outlier_mask]
        print(f"⚠️  Removed {np.sum(outlier_mask)} outlier(s): chi={outlier_chis}, values={outlier_vals}")
    
    mask = chi_clean >= min_chi_for_fit
    if np.sum(mask) < 3:
        return None, None, False
    
    chi_fit = chi_clean[mask]
    obs_fit = obs_clean[mask]
    
    # Initial guess
    O_guess = obs_fit[-1]
    A_guess = obs_fit[0] - obs_fit[-1]
    alpha_guess = 0.01
    
    try:
        popt, _ = curve_fit(
            exponential_decay,
            chi_fit,
            obs_fit,
            p0=[O_guess, A_guess, alpha_guess],
            maxfev=10000
        )
        return popt[0], popt, True
    except Exception:
        return None, None, False

def load_dmrg_observables(n_spins, chi_values, jz_values, obs_type, trunc_cut=1e-6):
    """Load DMRG observables for different chi values."""
    base_data = Path("../../data")
    obs_by_chi = {}
    
    for chi in chi_values:
        dmrg_dir = base_data / f"spins_{n_spins}" / "dmrg" / f"chi_max_{chi}_trunc_{trunc_cut:.0e}"
        
        if not dmrg_dir.exists():
            logging.warning(f"DMRG directory not found for n={n_spins}, chi={chi}")
            continue
            
        obs_data = {}
        for jz in jz_values:
            dmrg_file = dmrg_dir / f"XXZ_2d_jz_{jz:.1f}.pkl"
            
            if not dmrg_file.exists():
                continue
                
            try:
                dmrg_res = load_pickle(dmrg_file)
                dmrg_result = dmrg_res["result"]
                
                if obs_type == 'z':
                    obs = dmrg_result.psi.expectation_value("Sz") * 2
                    # Use mean absolute value as scalar
                    obs_data[jz] = np.mean(np.abs(obs))
                    
                elif obs_type == 'zz':
                    if "2_nn_corr_zz" in dmrg_res:
                        obs = np.array(dmrg_res["2_nn_corr_zz"])
                        obs_data[jz] = np.mean(np.abs(obs))
                    else:
                        print(f"✗ MISSING 2_nn_corr_zz: n={n_spins}, chi={chi}, Jz={jz:.1f}")
                        
                elif obs_type == 'z_loop':
                    if "z_loop" in dmrg_res:
                        obs = np.array(dmrg_res["z_loop"])
                        obs_data[jz] = np.mean(np.abs(obs))
                    else:
                        print(f"✗ MISSING z_loop: n={n_spins}, chi={chi}, Jz={jz:.1f}")
                        
            except Exception as ex:
                logging.warning(f"Failed to load {obs_type} for n={n_spins}, chi={chi}, Jz={jz:.1f}: {ex}")
                continue
        
        if obs_data:
            obs_by_chi[chi] = obs_data
            logging.info(f"Loaded {len(obs_data)} Jz values for n={n_spins}, chi={chi}, obs={obs_type}")
    
    return obs_by_chi

def plot_observable_extrapolation(n_spins, chi_values, jz_values, obs_type, save_dir):
    """Plot observable extrapolation to chi -> infinity."""
    obs_by_chi = load_dmrg_observables(n_spins, chi_values, jz_values, obs_type)
    
    if not obs_by_chi:
        logging.error(f"No data found for n={n_spins}, obs={obs_type}")
        return
    
    obs_labels = {
        'z': r'\langle |Z_i| \rangle',
        'zz': r'\langle |Z_i Z_j| \rangle',
        'z_loop': r'\langle |Z^{\otimes 12}| \rangle'
    }
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(5, 3))
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(jz_values)))
    
    extrapolated_obs = {}
    extrapolation_errors = {}
    ref_chi = 20
    
    for i, jz in enumerate(jz_values):
        chi_list = []
        obs_list = []
        
        for chi in sorted(obs_by_chi.keys()):
            if jz in obs_by_chi[chi]:
                chi_list.append(chi)
                obs_list.append(obs_by_chi[chi][jz])
        
        if len(chi_list) < 4:
            continue
        
        # Get reference observable at chi=20
        if ref_chi not in obs_by_chi or jz not in obs_by_chi[ref_chi]:
            continue
        
        O_ref = obs_by_chi[ref_chi][jz]
        
        # Perform extrapolation on absolute observables
        O_inf, fit_params, success = extrapolate_to_infinite_chi(chi_list, obs_list)
        
        if success and O_inf is not None:
            # Plot data points (absolute value of difference from chi=20)
            obs_rel = np.abs(np.array(obs_list) - O_ref)
            ax1.plot(chi_list, obs_rel, 'o', color=colors[i], markersize=6,
                    label=f'$J_z={jz:.1f}$')
            
            # Plot fitted curve
            chi_dense = np.linspace(min(chi_list), max(chi_list) * 1.5, 100)
            obs_fit = np.abs(exponential_decay(chi_dense, *fit_params) - O_ref)
            ax1.plot(chi_dense, obs_fit, '-', color=colors[i], alpha=0.5, linewidth=1.5)
            
            # Mark extrapolated value
            O_inf_rel = abs(O_inf - O_ref)
            ax1.axhline(y=O_inf_rel, color=colors[i], linestyle='--', alpha=0.3, linewidth=1)
            
            # Store results
            extrapolated_obs[jz] = O_inf
            if 320 in obs_by_chi and jz in obs_by_chi[320]:
                chi_320_obs = obs_by_chi[320][jz]
                error = abs(chi_320_obs - O_inf)
                extrapolation_errors[jz] = error
    
    ax1.set_xlabel(r'Bond dimension $\chi$')
    ax1.set_ylabel(r'$|' + obs_labels[obs_type] + r'(\chi) - ' + obs_labels[obs_type] + r'(\chi=' + str(ref_chi) + r')|$')
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    
    # Set x-axis ticks
    chi_ticks = [20, 40, 80, 160, 320, 640]
    ax1.set_xticks(chi_ticks)
    ax1.set_xticklabels([str(chi) for chi in chi_ticks])
    
    ax1.grid(True, alpha=0.3, which='both')
    
    # Store legend handles and labels for separate legend file
    handles1, labels1 = ax1.get_legend_handles_labels()
    
    # Plot error
    if extrapolation_errors:
        jz_list = sorted(extrapolation_errors.keys())
        errors = [extrapolation_errors[jz] for jz in jz_list]
        
        ax2.plot(jz_list, errors, 'o-', markersize=8, linewidth=2, color='darkblue',
                label='Absolute error')
        ax2.set_xlabel(r'$J_z$')
        ax2.set_ylabel(r'$|' + obs_labels[obs_type] + r'(\chi=320) - ' + obs_labels[obs_type] + r'(\chi \to \infty)|$')
        ax2.set_yscale('log')
        ax2.grid(True, alpha=0.3, which='both')
    
    fig.tight_layout()
    fig.savefig(save_dir / f"{obs_type}_extrapolation_chi_inf_n{n_spins}.pdf", bbox_inches='tight')
    plt.close(fig)
    
    # Save legend separately
    if handles1:
        fig_legend = plt.figure(figsize=(8, 2))
        fig_legend.legend(handles1, labels1, loc='center', ncol=5, frameon=False)
        fig_legend.savefig(save_dir / f"{obs_type}_extrapolation_legend_n{n_spins}.pdf", bbox_inches='tight')
        plt.close(fig_legend)
    
    logging.info(f"Saved {obs_type} extrapolation plot and legend for n={n_spins}")

if __name__ == "__main__":
    # Configuration
    chi_values = [20, 40, 80, 160, 320]
    jz_values = [1.1] + [round(jz, 1) for jz in np.arange(1.5, 5.51, 0.5)]
    
    save_dir = ensure_dir(Path("plots"))
    
    # Plot for each observable type and system size
    for n_spins in [57, 115]:
        for obs_type in ['z', 'zz', 'z_loop']:
            plot_observable_extrapolation(n_spins, chi_values, jz_values, obs_type, save_dir)
    
    logging.info("All observable extrapolation plots completed successfully")
