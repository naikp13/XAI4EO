from pathlib import Path
import numpy as np
import pandas as pd

def load_ground_truth(csv_path):
    """Load ground truth data from CSV."""
    gt_df = pd.read_csv(csv_path)
    C_ref = gt_df.values[:, 1:]  # Remove index column
    return C_ref

def load_npz_files(directory, num_files, file_type="HSI"):
    """Load .npz files from a directory."""
    data = []
    for i in range(num_files):
        file_name = f"{i:04d}.npz"
        file_path = directory / file_name
        if file_path.exists():
            with np.load(file_path) as npz:
                arr = np.ma.MaskedArray(**npz)
                data.append(arr)
                print(f"Loaded {file_type} {file_name}: shape {arr.shape}")
        else:
            print(f"{file_type} file {file_name} not found.")
            data.append(None)
    print(f"Loaded {len([x for x in data if x is not None])} {file_type} files")
    return data