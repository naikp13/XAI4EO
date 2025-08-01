import numpy as np
import json
import joblib
import h2o
from pathlib import Path
from src.optimization import make_features

def predict_targets(X_test, results_dir, num_targets=6, model_type="svr", num_bands=14):
    """Predict targets using trained models."""
    if model_type in ["h2o_glm", "h2o_xgboost", "h2o_automl"]:
        h2o.init()
    
    predictions = np.zeros((X_test.shape[0], num_targets))
    best_band_indices = []
    for i in range(num_targets):
        if model_type == "svr":
            suffix = f"t2000_{i}"
            json_path = results_dir / f"best_sub3_bands_{suffix}.json"
        elif model_type == "ridge":
            suffix = f"t20000_{i}"
            json_path = results_dir / f"best_sub2_bands_{suffix}.json"
        elif model_type == "xgb":
            suffix = f"t1500_{i}"
            json_path = results_dir / f"best_sub2_bands_{suffix}.json"
        elif model_type == "lgbm":
            suffix = f"t5000_{i}"
            json_path = results_dir / f"best_s5_bands_{suffix}.json"
        else:  # h2o_glm, h2o_xgboost, h2o_automl
            suffix = f"t2000_{i}"
            json_path = results_dir / f"best_h2o_bands_{suffix}.json"
        
        with open(json_path, 'r') as f:
            B = json.load(f)
            best_band_indices.append(B)
    
    for t in range(num_targets):
        print(f"Predicting target {t}")
        B = best_band_indices[t]
        features = make_features(X_test, B[:num_bands] if model_type == "svr" else B)
        features = np.nan_to_num(features)
        
        if model_type in ["h2o_glm", "h2o_xgboost", "h2o_automl"]:
            test_data = h2o.H2OFrame(features)
            test_data.columns = [f'feature_{i}' for i in range(features.shape[1])]
            model_folder = results_dir / f"best_h2o_model_t2000_{t}"
            model_files = list(model_folder.glob("*"))
            if not model_files:
                raise FileNotFoundError(f"No model file or directory found in {model_folder}")
            if len(model_files) > 1:
                raise ValueError(f"Multiple files/directories found in {model_folder}. Please ensure only one model is present.")
            model_path = str(model_files[0])
            model = h2o.load_model(model_path)
            pred = model.predict(test_data).as_data_frame(use_multi_thread=True).values.flatten()
        else:
            model = joblib.load(results_dir / f"best_sub3_{model_type}_{suffix}.pkl")
            pred = model.predict(features)
        
        predictions[:, t] = pred
    
    if model_type in ["h2o_glm", "h2o_xgboost", "h2o_automl"]:
        h2o.cluster().shutdown()
    
    return predictions