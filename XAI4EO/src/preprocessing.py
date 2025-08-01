import numpy as np
from pathlib import Path

def remove_water_bands(data, water_bands=(list(range(95, 120)) + list(range(140, 180)))):
    """Remove water absorption bands from HSI data."""
    keep_bands = [i for i in range(data[0].shape[0]) if i not in water_bands]
    processed_data = []
    for i, arr in enumerate(data):
        if arr is not None:
            arr_filtered = arr[keep_bands, :, :]
            processed_data.append(arr_filtered)
            print(f"HSI {i:04d}.npz: original shape {arr.shape}, filtered shape {arr_filtered.shape}")
        else:
            print(f"HSI {i:04d}.npz not found.")
            processed_data.append(None)
    print(f"Loaded {len([x for x in processed_data if x is not None])} HSI-RWB files")
    return processed_data

def prepare_features(hsi_data, msi_data, pansharpen_fn):
    """Prepare fused features using pansharpening."""
    fused_data = []
    for hsi, msi in zip(hsi_data, msi_data):
        if hsi is not None and msi is not None:
            try:
                fused_hsi = pansharpen_fn(hsi, msi)
                fused_mean = fused_hsi.mean(axis=(1, 2))
            except Exception as e:
                print(f"Fusion error: {e}")
                fused_mean = np.full(hsi.shape[0], np.nan)
        else:
            fused_mean = np.full(hsi.shape[0] if hsi is not None else 165, np.nan)
        fused_data.append(fused_mean)
    X = np.stack(fused_data)
    X = np.nan_to_num(X)
    print(f"Fused X shape: {X.shape}")
    return X