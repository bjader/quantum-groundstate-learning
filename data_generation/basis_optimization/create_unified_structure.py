#!/usr/bin/env python3
"""
Create a unified directory structure for basis optimization results.

This script creates symbolic links to organize optimization results with a consistent
naming convention that includes:
- Number of qubits (n)
- Timestamp of the original run
- jz value
- Stage information
- Postprocessing method (post_select or random_flip)

For n=57:
  - timestamp 1773299045 with post_select
  - timestamp 1773299045 with random_flip
For n=115:
  - timestamp 1773150437 (except jz=3.3 which uses 1773854302) with post_select
  - timestamps 1773150437_1773854302 (mixed) with random_flip
"""

import os
from pathlib import Path

# Get the script's directory and set base directories
SCRIPT_DIR = Path(__file__).parent.resolve()
BASE_DIR = SCRIPT_DIR
UNIFIED_DIR = BASE_DIR / "saved_opt_res_qpu_unified"

# Source directories
N57_SOURCE = BASE_DIR / "saved_opt_res_qpu/n_57_50_pts_vopt"
N115_SOURCE = BASE_DIR / "saved_opt_res_qpu"

# Timestamps
N57_TIMESTAMP = "1773299045"
N115_TIMESTAMP_DEFAULT = "1773150437"
N115_TIMESTAMP_JZ33 = "1773854302"

def create_unified_structure():
    """Create unified directory structure with symbolic links."""
    
    # Create unified directory and subdirectories if they don't exist
    UNIFIED_DIR.mkdir(exist_ok=True)
    (UNIFIED_DIR / "n_57").mkdir(exist_ok=True)
    (UNIFIED_DIR / "n_115").mkdir(exist_ok=True)
    
    print(f"Creating unified structure in: {UNIFIED_DIR}")
    print("=" * 80)
    
    # Clean up old symlinks without suffixes
    print("\nCleaning up old symlinks without suffixes...")
    cleanup_old_symlinks()
    
    # Process n=57 data
    print("\nProcessing n=57 data...")
    process_n57_data()
    
    # Process n=115 data
    print("\nProcessing n=115 data...")
    process_n115_data()
    
    print("\n" + "=" * 80)
    print("Unified structure created successfully!")
    print(f"Location: {UNIFIED_DIR}")
    print(f"  - n_57 data: {UNIFIED_DIR / 'n_57'}")
    print(f"  - n_115 data: {UNIFIED_DIR / 'n_115'}")

def process_n57_data():
    """Process n=57 optimization results from both old and new data sources."""
    
    n57_dir = UNIFIED_DIR / "n_57"
    
    # Process old n=57 data (random_flip, from colleague, different pattern)
    print("  Processing old n=57 data (random_flip)...")
    if N57_SOURCE.exists():
        process_n57_old_data(n57_dir)
    else:
        print(f"  Warning: Old source directory not found: {N57_SOURCE}")
    
    # Process new n=57 data (post_select, same pattern as n=115)
    print("  Processing new n=57 data (post_select)...")
    process_n57_new_data(n57_dir)

def process_n57_old_data(n57_dir):
    """Process old n=57 optimization results (random_flip, from colleague)."""
    
    # Find all jz directories
    jz_dirs = sorted([d for d in N57_SOURCE.iterdir() if d.is_dir() and d.name.startswith("idea_basis_opt")])
    
    for jz_dir in jz_dirs:
        # Extract jz value from directory name
        # Format: idea_basis_opt_all_points_57_0.1_jz_X.X
        parts = jz_dir.name.split("_jz_")
        if len(parts) != 2:
            print(f"    Skipping unexpected directory: {jz_dir.name}")
            continue
        
        jz_value = parts[1]
        
        # Process stage1 and stage2
        for stage in ["stage1", "stage2"]:
            stage_dir = jz_dir / stage
            if not stage_dir.exists():
                continue
            
            # Find the optimization result directory
            opt_dirs = list(stage_dir.iterdir())
            if not opt_dirs:
                continue
            
            # Prefer iter_30_30 if multiple directories exist
            if len(opt_dirs) > 1:
                iter_30_30_dirs = [d for d in opt_dirs if 'iter_30_30' in d.name]
                if iter_30_30_dirs:
                    opt_dir = iter_30_30_dirs[0]
                else:
                    opt_dir = opt_dirs[0]
            else:
                opt_dir = opt_dirs[0]
            
            # Create unified name with random_flip postprocessing indicator
            unified_name = f"ts_{N57_TIMESTAMP}_jz_{jz_value}_{stage}_random_flip"
            unified_path = n57_dir / unified_name
            
            # Create relative symlink
            rel_path = os.path.relpath(opt_dir, n57_dir)
            
            # Check if symlink exists and points to the correct target
            if unified_path.exists() or unified_path.is_symlink():
                if unified_path.is_symlink():
                    current_target = os.readlink(unified_path)
                    if current_target != rel_path:
                        # Remove old symlink and create new one
                        unified_path.unlink()
                        unified_path.symlink_to(rel_path)
                        print(f"    Updated: {unified_name} (now points to {opt_dir.name})")
                    else:
                        print(f"    Skipping (exists): {unified_name}")
                else:
                    print(f"    Skipping (exists as file): {unified_name}")
            else:
                unified_path.symlink_to(rel_path)
                print(f"    Created: {unified_name}")

def process_n57_new_data(n57_dir):
    """Process new n=57 optimization results (post_select, same pattern as n=115)."""
    
    # Find all n_57 directories with the correct timestamp and post_select
    all_dirs = sorted([d for d in N115_SOURCE.iterdir()
                      if d.is_dir() and d.name.startswith("n_57") and N57_TIMESTAMP in d.name
                      and "recovery_post_select" in d.name])
    
    for dir_path in all_dirs:
        dir_name = dir_path.name
        
        # Extract jz value
        if "_jz_" not in dir_name:
            continue
        
        jz_part = dir_name.split("_jz_")[1]
        jz_value = jz_part.split("_")[0]
        
        # Determine stage
        if "_stage1" in dir_name:
            stage = "stage1"
        elif "_stage2" in dir_name:
            stage = "stage2"
        else:
            # Skip directories without stage information
            continue
        
        # Create unified name with post_select postprocessing indicator
        unified_name = f"ts_{N57_TIMESTAMP}_jz_{jz_value}_{stage}_post_select"
        unified_path = n57_dir / unified_name
        
        # Create relative symlink
        rel_path = os.path.relpath(dir_path, n57_dir)
        
        if unified_path.exists():
            print(f"    Skipping (exists): {unified_name}")
        else:
            unified_path.symlink_to(rel_path)
            print(f"    Created: {unified_name}")

def process_n115_data():
    """Process n=115 optimization results."""
    
    if not N115_SOURCE.exists():
        print(f"Warning: Source directory not found: {N115_SOURCE}")
        return
    
    n115_dir = UNIFIED_DIR / "n_115"
    
    # Process post_select data (old pattern: n_115_jz_...)
    print("  Processing n=115 post_select data...")
    process_n115_post_select(n115_dir)
    
    # Process random_flip data (new pattern: qpu_115n_...)
    print("  Processing n=115 random_flip data...")
    process_n115_random_flip(n115_dir)

def process_n115_post_select(n115_dir):
    """Process n=115 post_select optimization results (old pattern: n_115_jz_...)."""
    
    # Find all n_115 directories with the correct timestamps
    all_dirs = sorted([d for d in N115_SOURCE.iterdir() if d.is_dir() and d.name.startswith("n_115")])
    
    for dir_path in all_dirs:
        dir_name = dir_path.name
        
        # Extract jz value
        if "_jz_" not in dir_name:
            continue
        
        jz_part = dir_name.split("_jz_")[1]
        jz_value = jz_part.split("_")[0]
        
        # Determine expected timestamp (jz=3.3 uses special timestamp)
        if jz_value == "3.3":
            expected_ts = N115_TIMESTAMP_JZ33
        else:
            expected_ts = N115_TIMESTAMP_DEFAULT
        
        # Check if this directory has the correct timestamp
        if expected_ts not in dir_name:
            continue
        
        # Determine stage
        if "_stage1" in dir_name:
            stage = "stage1"
        elif "_stage2" in dir_name:
            stage = "stage2"
        else:
            # Skip directories without stage information
            continue
        
        # Create unified name with post_select suffix
        unified_name = f"ts_{expected_ts}_jz_{jz_value}_{stage}_post_select"
        unified_path = n115_dir / unified_name
        
        # Create relative symlink
        rel_path = os.path.relpath(dir_path, n115_dir)
        
        if unified_path.exists():
            print(f"    Skipping (exists): {unified_name}")
        else:
            unified_path.symlink_to(rel_path)
            print(f"    Created: {unified_name}")

def process_n115_random_flip(n115_dir):
    """Process n=115 random_flip optimization results (new pattern: qpu_115n_...)."""
    
    # Find all qpu_115n directories with random_flip
    all_dirs = sorted([d for d in N115_SOURCE.iterdir()
                      if d.is_dir() and d.name.startswith("qpu_115n")
                      and "random_flip" in d.name])
    
    for dir_path in all_dirs:
        dir_name = dir_path.name
        
        # Extract jz value
        # Format: qpu_115n_115_jz_X.X_ts_1_kd_11_shots_100k_ibm_boston_TIMESTAMP1_TIMESTAMP2_mixed_recovery_random_flip_...
        if "_jz_" not in dir_name:
            continue
        
        jz_part = dir_name.split("_jz_")[1]
        jz_value = jz_part.split("_")[0]
        
        # Extract timestamps (there are two: 1773150437_1773854302)
        # We'll use the first one as the primary timestamp
        ts_part = dir_name.split("_ibm_boston_")[1]
        timestamps = ts_part.split("_")[0:2]  # Get both timestamps
        primary_ts = timestamps[0]
        
        # Determine stage
        if "_stage1" in dir_name:
            stage = "stage1"
        elif "_stage2" in dir_name:
            stage = "stage2"
        else:
            # Skip directories without stage information
            continue
        
        # Create unified name with random_flip suffix
        unified_name = f"ts_{primary_ts}_jz_{jz_value}_{stage}_random_flip"
        unified_path = n115_dir / unified_name
        
        # Create relative symlink
        rel_path = os.path.relpath(dir_path, n115_dir)
        
        if unified_path.exists():
            print(f"    Skipping (exists): {unified_name}")
        else:
            unified_path.symlink_to(rel_path)
            print(f"    Created: {unified_name}")

def cleanup_old_symlinks():
    """Remove old symlinks without post_select or random_flip suffixes."""
    
    for n_dir in ["n_57", "n_115"]:
        target_dir = UNIFIED_DIR / n_dir
        if not target_dir.exists():
            continue
        
        print(f"  Checking {n_dir}...")
        removed_count = 0
        
        for item in target_dir.iterdir():
            # Check if it's a symlink and doesn't have a suffix
            if item.is_symlink():
                name = item.name
                # Old pattern: ts_TIMESTAMP_jz_X.X_stageN (without _post_select or _random_flip)
                if (name.startswith("ts_") and
                    ("_stage1" in name or "_stage2" in name) and
                    not name.endswith("_post_select") and
                    not name.endswith("_random_flip")):
                    item.unlink()
                    removed_count += 1
                    print(f"    Removed old symlink: {name}")
        
        if removed_count == 0:
            print(f"    No old symlinks found in {n_dir}")

if __name__ == "__main__":
    create_unified_structure()
