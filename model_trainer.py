import lightgbm as lgb
import optuna
from sklearn.metrics import precision_score
import pandas as pd
import numpy as np

class QuantModel:
    def __init__(self, df, feature_cols):
        self.df = df
        self.feature_cols = feature_cols
        self.model = None
        self.best_params = {}

    def prepare_split(self, test_size=0.2):
        # Time Series Split
        split_idx = int(len(self.df) * (1 - test_size))
        train_df = self.df.iloc[:split_idx]
        test_df = self.df.iloc[split_idx:]
        
        X_train = train_df[self.feature_cols]
        y_train = train_df['Target_Direction']
        X_test = test_df[self.feature_cols]
        y_test = test_df['Target_Direction']
        
        return X_train, y_train, X_test, y_test, test_df

    def objective(self, trial, X_train, y_train, X_test, y_test):
        param = {
            'objective': 'binary',
            'metric': 'binary_logloss',
            'verbosity': -1,
            'boosting_type': 'gbdt',
            'lambda_l1': trial.suggest_float('lambda_l1', 1e-8, 10.0, log=True),
            'lambda_l2': trial.suggest_float('lambda_l2', 1e-8, 10.0, log=True),
            'num_leaves': trial.suggest_int('num_leaves', 2, 256),
            'feature_fraction': trial.suggest_float('feature_fraction', 0.4, 1.0),
            'bagging_fraction': trial.suggest_float('bagging_fraction', 0.4, 1.0),
            'bagging_freq': trial.suggest_int('bagging_freq', 1, 7),
            'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
            'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.1),
            'n_estimators': 500
        }

        model = lgb.LGBMClassifier(**param)
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        
        accuracy = precision_score(y_test, preds, zero_division=0)
        return accuracy

    def optimize_hyperparameters(self, X_train, y_train, X_test, y_test, n_trials=20):
        print("Mulai Hyperparameter Tuning dengan Optuna...")
        study = optuna.create_study(direction='maximize') 
        
        func = lambda trial: self.objective(trial, X_train, y_train, X_test, y_test)
        study.optimize(func, n_trials=n_trials)

        print(f"Trial Terbaik: {study.best_trial.value}")
        print("Parameter Terbaik:")
        for key, value in study.best_trial.params.items():
            print(f"  {key}: {value}")
        
        self.best_params = study.best_trial.params
        return self.best_params

    def train_final_model(self, X_train, y_train):
        print("Training model final dengan parameter terbaik...")
        # Tambahkan n_estimators jika belum ada di best_params
        if 'n_estimators' not in self.best_params:
            self.best_params['n_estimators'] = 500
            
        self.model = lgb.LGBMClassifier(**self.best_params, random_state=42)
        self.model.fit(X_train, y_train)

    def evaluate(self, X_test, y_test):
        # Sama seperti sebelumnya...
        preds = self.model.predict(X_test)
        probs = self.model.predict_proba(X_test)[:, 1]
        print(f"Final Precision: {precision_score(y_test, preds):.4f}")
        return preds, probs