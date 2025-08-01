# XAI4EO

A Python package for automated machine learning on hyperspectral and multispectral imaging data, focusing on pansharpening and regression modeling for geochemical analysis.

## Installation

```bash
pip install XAI4EO
```

## Requirements

See `requirements.txt` for dependencies.

## Usage

### Basic Example with SVR
```python
from src import load_ground_truth, load_npz_files, remove_water_bands, prepare_features, gram_schmidt_pansharpen, optimize_svr, predict_targets
from pathlib import Path

# Define paths
DATASET_DIR = Path('/content/drive/MyDrive/hyperview/eotdl/datasets/HYPERVIEW2/')
RESULTS_DIR = Path('/content/drive/MyDrive/hyperview/results/')
GT_TRAIN_CSV = DATASET_DIR / 'train_gt.csv'
HSI_SATELLITE_TRAIN_DIR = DATASET_DIR / 'train/hsi_satellite'
MSI_SATELLITE_TRAIN_DIR = DATASET_DIR / 'train/msi_satellite'

# Load data
C_ref = load_ground_truth(GT_TRAIN_CSV)
hsi_train = load_npz_files(HSI_SATELLITE_TRAIN_DIR, 1876, "HSI")
msi_train = load_npz_files(MSI_SATELLITE_TRAIN_DIR, 1876, "MSI")
hsi_train_rwb = remove_water_bands(hsi_train)
X_train = prepare_features(hsi_train_rwb, msi_train, gram_schmidt_pansharpen)

# Optimize model
optimize_svr(X_train, C_ref, RESULTS_DIR, n_trials=2000)

# Predict on test data
X_test = load_data('/content/drive/MyDrive/hyperview/processed_data/X1_test_hsi_rwb_msi_gs.npy')
predictions = predict_targets(X_test, RESULTS_DIR, model_type="svr")
```

### H2O AutoML Example
```python
from src import load_ground_truth, load_npz_files, remove_water_bands, prepare_features, gram_schmidt_pansharpen, optimize_h2o_automl, predict_targets
from pathlib import Path

# Define paths
DATASET_DIR = Path('/content/drive/MyDrive/hyperview/eotdl/datasets/HYPERVIEW2/')
RESULTS_DIR = Path('/content/drive/MyDrive/hyperview/results/')
GT_TRAIN_CSV = DATASET_DIR / 'train_gt.csv'
HSI_SATELLITE_TRAIN_DIR = DATASET_DIR / 'train/hsi_satellite'
MSI_SATELLITE_TRAIN_DIR = DATASET_DIR / 'train/msi_satellite'

# Load data
C_ref = load_ground_truth(GT_TRAIN_CSV)
hsi_train = load_npz_files(HSI_SATELLITE_TRAIN_DIR, 1876, "HSI")
msi_train = load_npz_files(MSI_SATELLITE_TRAIN_DIR, 1876, "MSI")
hsi_train_rwb = remove_water_bands(hsi_train)
X_train = prepare_features(hsi_train_rwb, msi_train, gram_schmidt_pansharpen)

# Optimize H2O AutoML
optimize_h2o_automl(X_train, C_ref, RESULTS_DIR, n_trials=3000)

# Predict on test data
X_test = load_data('/content/drive/MyDrive/hyperview/processed_data/X1_test_hsi_rwb_msi_gs.npy')
 непод

predictions = predict_targets(X_test, RESULTS_DIR, model_type="h2o_automl")
```

## License

MIT License. See `LICENSE` for details.