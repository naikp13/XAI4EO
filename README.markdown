# XAI4EO

A Python based focusing on AutoML driven modeling and Explainable AI for EO applications. A part of the package was used to conduct experiments for end-to-end automated forest analysis published in the paper accessible at - https://doi.org/10.1109/jstars.2022.3232583 . A part of the package is also developed for contesting at the HYPERVIEW2 Challenge implemented as part of the Explainable AI in Space (EASi) workshop - https://ai4eo.eu/portfolio/easi-workshop-hyperview2/

<img src="https://raw.githubusercontent.com/naikp13/XAI4EO/main/XAI_image.gif" alt="XAI" width="800"/>

## Installation

```bash
from XAI4EO.src import *
```

## Requirements

See `requirements.txt` for dependencies.

## Usage

### Basic Example with SVR
```python
from src import load_ground_truth, load_npz_files, remove_water_bands, prepare_features, gram_schmidt_pansharpen, optimize_svr, predict_targets
from pathlib import Path

# Define paths
DATASET_DIR = Path('/dataset/directory/')
RESULTS_DIR = Path('/results/directory/')
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

### AutoML Example
```python
from src import load_ground_truth, load_npz_files, remove_water_bands, prepare_features, gram_schmidt_pansharpen, optimize_h2o_automl, predict_targets
from pathlib import Path

# Define paths
DATASET_DIR = Path('/dataset/directory/')
RESULTS_DIR = Path('/results/directory')
GT_TRAIN_CSV = DATASET_DIR / 'train_gt.csv'
HSI_SATELLITE_TRAIN_DIR = DATASET_DIR / 'train/hsi_satellite'
MSI_SATELLITE_TRAIN_DIR = DATASET_DIR / 'train/msi_satellite'

# Load data
n_samples_train = 
C_ref = load_ground_truth(GT_TRAIN_CSV)
hsi_train = load_npz_files(HSI_SATELLITE_TRAIN_DIR, n_samples_train, "HSI")
msi_train = load_npz_files(MSI_SATELLITE_TRAIN_DIR, n_samples_train, "MSI")
hsi_train_rwb = remove_water_bands(hsi_train)
X_train = prepare_features(hsi_train_rwb, msi_train, gram_schmidt_pansharpen)

# Optimize H2O AutoML
optimize_h2o_automl(X_train, C_ref, RESULTS_DIR, n_trials=3000)

# Predict on test data
X_test = load_data('/test_data/test.npy')

predictions = predict_targets(X_test, RESULTS_DIR, model_type="h2o_automl")

```

## Cite this work

Please cite this article in case this method was helpful for your research or used for your work,

```Citation
Naik, P., Dalponte, M., & Bruzzone, L. (2023). Automated Machine Learning Driven Stacked Ensemble Modeling for Forest Aboveground Biomass Prediction Using Multitemporal Sentinel-2 Data. IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing, 16, 3442–3454. https://doi.org/10.1109/jstars.2022.3232583
```

## Contact

For issues or questions, open an issue on GitHub or contact [parthnaik1993@gmail.com](mailto:parthnaik1993@gmail.com).

## Contributing
Contributions are welcome! Please open an issue or submit a pull request for any improvements or bug fixes.


## License

MIT License. See `LICENSE` for details.
