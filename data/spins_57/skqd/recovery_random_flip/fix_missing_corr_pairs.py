#!/usr/bin/env python3
"""
Script to fix missing correlation pair keys in SKQD data files.

This script:
1. Scans all pickle files in the directory
2. Finds files that contain the correlation pair keys
3. Verifies they're all the same across files
4. Adds those keys to files that are missing them
"""

import os
from pathlib import Path
import dill as pickle
import numpy as np

# Configuration
DATA_DIR = Path(__file__).parent
PAIR_KEYS = ['2_nn_corr_pairs_zz_basis_opt', '2_nn_corr_pairs_xx_basis_opt']


def main():
    print("=" * 80)
    print("Fixing missing correlation pair keys in SKQD data files")
    print("=" * 80)
    print(f"\nData directory: {DATA_DIR}")
    print(f"Looking for keys: {PAIR_KEYS}\n")

    # Get all pickle files
    pkl_files = sorted(DATA_DIR.glob("XXZ_2d_jz_*.pkl"))
    print(f"Found {len(pkl_files)} pickle files\n")

    # Step 1: Find files with the pair keys and collect the pairs
    files_with_pairs = {}
    collected_pairs = {key: [] for key in PAIR_KEYS}

    print("Step 1: Scanning files for correlation pair keys...")
    print("-" * 80)

    for pkl_file in pkl_files:
        jz_value = pkl_file.stem.split('_')[-1]

        try:
            with open(pkl_file, 'rb') as f:
                data = pickle.load(f)

            has_all_keys = all(key in data for key in PAIR_KEYS)

            if has_all_keys:
                files_with_pairs[pkl_file] = jz_value
                for key in PAIR_KEYS:
                    collected_pairs[key].append((jz_value, data[key]))
                print(f"✓ {pkl_file.name}: Has all pair keys")
            else:
                missing_keys = [key for key in PAIR_KEYS if key not in data]
                print(f"✗ {pkl_file.name}: Missing keys: {missing_keys}")

        except Exception as e:
            print(f"✗ {pkl_file.name}: Error loading file: {e}")

    print(f"\nFound {len(files_with_pairs)} files with all pair keys")

    if len(files_with_pairs) == 0:
        print("\nERROR: No files found with the required pair keys!")
        return

    # Step 2: Verify all pairs are the same
    print("\nStep 2: Verifying all correlation pairs are identical...")
    print("-" * 80)

    canonical_pairs = {}
    all_identical = True

    for key in PAIR_KEYS:
        if len(collected_pairs[key]) == 0:
            print(f"✗ {key}: No data found")
            all_identical = False
            continue

        # Use the first file's pairs as reference
        reference_jz, reference_pairs = collected_pairs[key][0]
        canonical_pairs[key] = reference_pairs

        # Convert to list for comparison
        reference_list = list(reference_pairs)

        # Check all other files
        mismatches = []
        for jz, pairs in collected_pairs[key][1:]:
            pairs_list = list(pairs)
            if len(pairs_list) != len(reference_list):
                mismatches.append(
                    f"Jz={jz} (length mismatch: {len(pairs_list)} vs {len(reference_list)})")
            elif not all(np.array_equal(p1, p2) for p1, p2 in zip(pairs_list, reference_list)):
                mismatches.append(f"Jz={jz} (content mismatch)")

        if mismatches:
            print(f"✗ {key}: NOT identical across files!")
            print(f"  Reference: Jz={reference_jz}")
            print(f"  Mismatches: {', '.join(mismatches)}")
            all_identical = False
        else:
            print(f"✓ {key}: Identical across all {len(collected_pairs[key])} files")
            print(f"  Number of pairs: {len(reference_list)}")

    if not all_identical:
        print("\nERROR: Correlation pairs are not identical across all files!")
        print("Cannot proceed with fixing missing keys.")
        return

    print("\n✓ All correlation pairs are identical across files!")

    # Step 3: Add pairs to files that are missing them
    print("\nStep 3: Adding missing correlation pairs to files...")
    print("-" * 80)

    files_updated = 0
    files_skipped = 0

    for pkl_file in pkl_files:
        jz_value = pkl_file.stem.split('_')[-1]

        try:
            with open(pkl_file, 'rb') as f:
                data = pickle.load(f)

            # Check if any keys are missing
            missing_keys = [key for key in PAIR_KEYS if key not in data]

            if not missing_keys:
                files_skipped += 1
                continue

            # Add the missing keys
            for key in missing_keys:
                data[key] = canonical_pairs[key]

            # Save the updated file
            with open(pkl_file, 'wb') as f:
                pickle.dump(data, f)

            print(f"✓ {pkl_file.name}: Added {len(missing_keys)} missing key(s): {missing_keys}")
            files_updated += 1

        except Exception as e:
            print(f"✗ {pkl_file.name}: Error updating file: {e}")

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Total files processed: {len(pkl_files)}")
    print(f"Files updated: {files_updated}")
    print(f"Files skipped (already had keys): {files_skipped}")
    print(f"Files with errors: {len(pkl_files) - files_updated - files_skipped}")

    if files_updated > 0:
        print("\n✓ Successfully fixed missing correlation pair keys!")
    else:
        print("\nNo files needed updating.")

    print("=" * 80)


if __name__ == "__main__":
    main()

# Made with Bob
