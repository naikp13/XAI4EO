import numpy as np
import optuna
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.svm import SVR
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
import joblib
import json
from pathlib import Path
import h2o
from h2o.estimators.glm import H2OGeneralizedLinearEstimator
from h2o.estimators import H2OXGBoostEstimator
from h2o.automl import H2OAutoML

def make_features(X, B):
    """Create feature indices for optimization."""
    band_index1 = (X[:, B[0]] - X[:, B[1]]) / (X[:, B[0]] + X[:, B[1]] + 1e-4)
    band_index2 = (X[:, B[2]] - X[:, B[3]]) / (X[:, B[4]] + 1e-4)
    band_index3 = X[:, B[5]] / (X[:, B[6]] - X[:, B[7]] + 1e-4)
    band_index4 = X[:, B[8]] / (X[:, B[9]] + X[:, B[10]] + 1e-4)
    band_index5 = X[:, B[11]] / (X[:, B[12]] + 1e-4)
    band_index6 = X[:, B[13]]
    band_index7 = X[:, B[14]]
    band_index8 = X[:, B[15]]
    band_index9 = X[:, B[16]]
    band_index10 = X[:, B[17]]
    return np.stack((band_index1, band_index2, band_index3, band_index4, band_index5, 
                     band_index6, band_index7, band_index8, band_index9, band_index10), axis=1)

def optimize_svr(X, C_ref, results_dir, n_trials=2000):
    """Optimize SVR model with NSGA-III."""
    def make_objective(target_index):
        def objective(trial):
            num_features = X.shape[1]
            B = [trial.suggest_int(f'B{i+1}', 0, num_features - 1) for i in range(14)]
            features = make_features(X, B)
            y = np.nan_to_num(C_ref[:, target_index])
            X_train, X_test, y_train, y_test = train_test_split(features, y, test_size=0.3, random_state=42)
            X_train, X_test, y_train, y_test = [np.nan_to_num(x) for x in [X_train, X_test, y_train, y_test]]
            model = SVR(
                kernel='rbf',
                C=trial.suggest_float('C', 0.1, 5.0, log=True),
                epsilon=trial.suggest_float('epsilon', 0.01, 0.3),
                gamma='scale',
                tol=5e-2,
                max_iter=20000
            )
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            r_squared = r2_score(y_test, y_pred)
            best_score = trial.study.user_attrs.get("best_score", -np.inf)
            if r_squared > best_score:
                trial.study.set_user_attr("best_score", r_squared)
                joblib.dump(model, results_dir / f"best_sub3_svr_t2000_{target_index}.pkl")
                with open(results_dir / f"best_sub3_bands_t2000_{target_index}.json", "w") as f:
                    json.dump(B, f)
            return r_squared
        return objective

    for i in range(C_ref.shape[1]):
        print(f"Optimizing SVR model for target index {i}")
        sampler = optuna.samplers.NSGAIIISampler(seed=42)
        study = optuna.create_study(direction='maximize', sampler=sampler)
        study.optimize(make_objective(i), n_trials=n_trials)
        print(f"Best R² for target {i}: {study.best_value}")

def optimize_ridge(X, C_ref, results_dir, n_trials=20000):
    """Optimize Ridge Regression model with NSGA-III."""
    def make_objective(target_index):
        def objective(trial):
            num_features = X.shape[1]
            B = [trial.suggest_int(f'B{i+1}', 0, num_features - 1) for i in range(18)]
            if len(set(B)) < 18:
                return -float("inf")
            features = make_features(X, B)
            scaler = StandardScaler()
            features = scaler.fit_transform(features)
            y = np.nan_to_num(C_ref[:, target_index])
            X_train, X_test, y_train, y_test = train_test_split(features, y, test_size=0.3, random_state=42)
            X_train, X_test, y_train, y_test = [np.nan_to_num(x) for x in [X_train, X_test, y_train, y_test]]
            model = Ridge(alpha=0.9, random_state=42, solver='svd')
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            r_squared = r2_score(y_test, y_pred)
            best_score = trial.study.user_attrs.get("best_score", -np.inf)
            if r_squared > best_score:
                trial.study.set_user_attr("best_score", r_squared)
                joblib.dump(model, results_dir / f"best_sub2_ridge_t20000_{target_index}.pkl")
                with open(results_dir / f"best_sub2_bands_t20000_{target_index}.json", "w") as f:
                    json.dump(B, f)
                joblib.dump(scaler, results_dir / f"best_sub2_scaler_t20000_{target_index}.pkl")
            return r_squared
        return objective

    for i in range(C_ref.shape[1]):
        print(f"Optimizing Ridge Regression model for target index {i}")
        sampler = optuna.samplers.NSGAIIISampler(seed=42)
        study = optuna.create_study(direction='maximize', sampler=sampler)
        study.optimize(make_objective(i), n_trials=n_trials)
        print(f"Best R² for target {i}: {study.best_value}")

def optimize_xgboost(X, C_ref, results_dir, n_trials=1500):
    """Optimize XGBoost model with NSGA-III."""
    def make_objective(target_index):
        def objective(trial):
            num_features = X.shape[1]
            B = [trial.suggest_int(f'B{i+1}', 0, num_features - 1) for i in range(18)]
            features = make_features(X, B)
            y = np.nan_to_num(C_ref[:, target_index])
            X_train, X_test, y_train, y_test = train_test_split(features, y, test_size=0.3, random_state=42)
            X_train, X_test, y_train, y_test = [np.nan_to_num(x) for x in [X_train, X_test, y_train, y_test]]
            model = XGBRegressor(
                max_depth=trial.suggest_int('max_depth', 3, 8),
                learning_rate=trial.suggest_float('learning_rate', 0.01, 0.3),
                n_estimators=trial.suggest_int('n_estimators', 50, 200),
                subsample=trial.suggest_float('subsample', 0.5, 1.0),
                colsample_bytree=trial.suggest_float('colsample_bytree', 0.5, 1.0),
                reg_alpha=trial.suggest_float('reg_alpha', 0, 1.0),
                reg_lambda=trial.suggest_float('reg_lambda', 0, 1.0),
                random_state=42,
                verbosity=0
            )
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            r_squared = r2_score(y_test, y_pred)
            best_score = trial.study.user_attrs.get("best_score", -np.inf)
            if r_squared > best_score:
                trial.study.set_user_attr("best_score", r_squared)
                joblib.dump(model, results_dir / f"best_sub2_xgb_t1500_{target_index}.pkl")
                with open(results_dir / f"best_sub2_bands_t1500_{target_index}.json", "w") as f:
                    json.dump(B, f)
            return r_squared
        return objective

    for i in range(C_ref.shape[1]):
        print(f"Optimizing XGBoost model for target index {i}")
        sampler = optuna.samplers.NSGAIIISampler(seed=42)
        study = optuna.create_study(direction='maximize', sampler=sampler)
        study.optimize(make_objective(i), n_trials=n_trials)
        print(f"Best R² for target {i}: {study.best_value}")

def optimize_lightgbm(X, C_ref, results_dir, n_trials=5000):
    """Optimize LightGBM model with NSGA-III."""
    def make_objective(target_index):
        def objective(trial):
            num_features = X.shape[1]
            B = [trial.suggest_int(f'B{i+1}', 0, num_features - 1) for i in range(18)]
            features = make_features(X, B)
            y = np.nan_to_num(C_ref[:, target_index])
            X_train, X_test, y_train, y_test = train_test_split(features, y, test_size=0.3, random_state=42)
            X_train, X_test, y_train, y_test = [np.nan_to_num(x) for x in [X_train, X_test, y_train, y_test]]
            model = LGBMRegressor(
                max_depth=trial.suggest_int('max_depth', 3, 8),
                learning_rate=trial.suggest_float('learning_rate', 0.01, 0.3),
                n_estimators=trial.suggest_int('n_estimators', 50, 200),
                subsample=trial.suggest_float('subsample', 0.5, 1.0),
                colsample_bytree=trial.suggest_float('colsample_bytree', 0.5, 1.0),
                reg_alpha=trial.suggest_float('reg_alpha', 0, 1.0),
                reg_lambda=trial.suggest_float('reg_lambda', 0, 1.0),
                random_state=42,
                verbose=-1
            )
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            r_squared = r2_score(y_test, y_pred)
            best_score = trial.study.user_attrs.get("best_score", -np.inf)
            if r_squared > best_score:
                trial.study.set_user_attr("best_score", r_squared)
                joblib.dump(model, results_dir / f"best_s5_lgbm_t5000_{target_index}.pkl")
                with open(results_dir / f"best_s5_bands_t5000_{target_index}.json", "w") as f:
                    json.dump(B, f)
            return r_squared
        return objective

    for i in range(C_ref.shape[1]):
        print(f"Optimizing LightGBM model for target index {i}")
        sampler = optuna.samplers.NSGAIIISampler(seed=42)
        study = optuna.create_study(direction='maximize', sampler=sampler)
        study.optimize(make_objective(i), n_trials=n_trials)
        print(f"Best R² for target {i}: {study.best_value}")

def optimize_glm(X, C_ref, results_dir, n_trials=3000):
    """Optimize H2O GLM model with NSGA-III for band selection."""
    h2o.init()
    def make_objective(target_index):
        def objective(trial):
            num_features = X.shape[1]
            B = [trial.suggest_int(f'B{i+1}', 0, num_features - 1) for i in range(18)]
            features = make_features(X, B)
            y = np.nan_to_num(C_ref[:, target_index])
            X_train, X_test, y_train, y_test = train_test_split(features, y, test_size=0.3, random_state=42)
            X_train, X_test, y_train, y_test = [np.nan_to_num(x) for x in [X_train, X_test, y_train, y_test]]
            train_data = h2o.H2OFrame(np.column_stack((X_train, y_train)))
            x = [f'feature_{i}' for i in range(X_train.shape[1])]
            y_col = 'target'
            train_data.columns = x + [y_col]
            test_data = h2o.H2OFrame(np.column_stack((X_test, y_test)))
            test_data.columns = x + [y_col]
            glm = H2OGeneralizedLinearEstimator(
                family='gaussian',
                lambda_search=True,
                seed=42,
                max_iterations=2000
            )
            glm.train(x=x, y=y_col, training_frame=train_data)
            y_pred = glm.predict(test_data).as_data_frame(use_multi_thread=True).values.flatten()
            r_squared = r2_score(y_test, y_pred)
            best_score = trial.study.user_attrs.get("best_score", -np.inf)
            if r_squared > best_score:
                trial.study.set_user_attr("best_score", r_squared)
                with open(results_dir / f"best_h2o_bands_t2000_{target_index}.json", "w") as f:
                    json.dump(B, f)
            return r_squared
        return objective

    for i in range(C_ref.shape[1]):
        print(f"Optimizing GLM model for target index {i}")
        sampler = optuna.samplers.NSGAIIISampler(seed=42)
        study = optuna.create_study(direction='maximize', sampler=sampler)
        study.optimize(make_objective(i), n_trials=n_trials)
        print(f"Best R² for target {i}: {study.best_value}")
    h2o.cluster().shutdown()

def optimize_h2o_xgboost(X, C_ref, results_dir, n_trials=3000):
    """Optimize H2O XGBoost model with NSGA-III for band selection."""
    h2o.init()
    def make_objective(target_index):
        def objective(trial):
            num_features = X.shape[1]
            B = [trial.suggest_int(f'B{i+1}', 0, num_features - 1) for i in range(18)]
            features = make_features(X, B)
            y = np.nan_to_num(C_ref[:, target_index])
            X_train, X_test, y_train, y_test = train_test_split(features, y, test_size=0.3, random_state=42)
            X_train, X_test, y_train, y_test = [np.nan_to_num(x) for x in [X_train, X_test, y_train, y_test]]
            train_data = h2o.H2OFrame(np.column_stack((X_train, y_train)))
            x = [f'feature_{i}' for i in range(X_train.shape[1])]
            y_col = 'target'
            train_data.columns = x + [y_col]
            test_data = h2o.H2OFrame(np.column_stack((X_test, y_test)))
            test_data.columns = x + [y_col]
            xgb = H2OXGBoostEstimator(
                ntrees=100,
                max_depth=6,
                learn_rate=0.1,
                seed=42
            )
            xgb.train(x=x, y=y_col, training_frame=train_data)
            y_pred = xgb.predict(test_data).as_data_frame(use_multi_thread=True).values.flatten()
            r_squared = r2_score(y_test, y_pred)
            best_score = trial.study.user_attrs.get("best_score", -np.inf)
            if r_squared > best_score:
                trial.study.set_user_attr("best_score", r_squared)
                with open(results_dir / f"best_h2o_bands_t2000_{target_index}.json", "w") as f:
                    json.dump(B, f)
            return r_squared
        return objective

    for i in range(C_ref.shape[1]):
        print(f"Optimizing H2O XGBoost model for target index {i}")
        sampler = optuna.samplers.NSGAIIISampler(seed=42)
        study = optuna.create_study(direction='maximize', sampler=sampler)
        study.optimize(make_objective(i), n_trials=n_trials)
        print(f"Best R² for target {i}: {study.best_value}")
    h2o.cluster().shutdown()

def optimize_h2o_automl(X, C_ref, results_dir, n_trials=3000):
    """Optimize H2O AutoML with NSGA-III for band selection and train ensemble."""
    h2o.init()
    def make_objective(target_index):
        def objective(trial):
            num_features = X.shape[1]
            B = [trial.suggest_int(f'B{i+1}', 0, num_features - 1) for i in range(18)]
            features = make_features(X, B)
            y = np.nan_to_num(C_ref[:, target_index])
            X_train, X_test, y_train, y_test = train_test_split(features, y, test_size=0.3, random_state=42)
            X_train, X_test, y_train, y_test = [np.nan_to_num(x) for x in [X_train, X_test, y_train, y_test]]
            train_data = h2o.H2OFrame(np.column_stack((X_train, y_train)))
            x = [f'feature_{i}' for i in range(X_train.shape[1])]
            y_col = 'target'
            train_data.columns = x + [y_col]
            test_data = h2o.H2OFrame(np.column_stack((X_test, y_test)))
            test_data.columns = x + [y_col]
            aml = H2OAutoML(
                max_runtime_secs=2500,
                seed=42,
                sort_metric='r2',
                max_models=5,
                include_algos=['XGBoost', 'GLM', 'DeepLearning'],
                nfolds=0,
                keep_cross_validation_predictions=False,
                keep_cross_validation_models=False
            )
            aml.train(x=x, y=y_col, training_frame=train_data)
            y_pred = aml.leader.predict(test_data).as_data_frame(use_multi_thread=True).values.flatten()
            r_squared = r2_score(y_test, y_pred)
            best_score = trial.study.user_attrs.get("best_score", -np.inf)
            if r_squared > best_score:
                trial.study.set_user_attr("best_score", r_squared)
                models_dir = results_dir / f"models_target_{target_index}"
                models_dir.mkdir(exist_ok=True)
                leaderboard = aml.leaderboard.as_data_frame(use_multi_thread=True)
                for model_id in leaderboard['model_id']:
                    model = h2o.get_model(model_id)
                    model_path = str(models_dir / f"model_{model_id}")
                    h2o.save_model(model=model, path=model_path, force=True)
                best_model_path = str(results_dir / f"best_h2o_model_t2000_{target_index}")
                h2o.save_model(model=aml.leader, path=best_model_path, force=True)
                with open(results_dir / f"best_h2o_bands_t2000_{target_index}.json", "w") as f:
                    json.dump(B, f)
            return r_squared
        return objective

    for i in range(C_ref.shape[1]):
        print(f"Optimizing H2O AutoML for target index {i}")
        sampler = optuna.samplers.NSGAIIISampler(seed=42)
        study = optuna.create_study(direction='maximize', sampler=sampler)
        study.optimize(make_objective(i), n_trials=n_trials)
        print(f"Best R² for target {i}: {study.best_value}")
    h2o.cluster().shutdown()