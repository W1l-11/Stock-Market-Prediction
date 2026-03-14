import warnings
from typing import Tuple

import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd
from sklearn.metrics import (f1_score, precision_score, recall_score,
                              roc_auc_score)
from sklearn.model_selection import TimeSeriesSplit

warnings.filterwarnings("ignore", category=optuna.exceptions.ExperimentalWarning)
optuna.logging.set_verbosity(optuna.logging.WARNING)

def _compute_sharpe(returns: np.ndarray, periods_per_year: int = 252) -> float:
    if len(returns) < 20 or returns.std() == 0:
        return -999.0
    mean_r  = returns.mean() * periods_per_year
    std_r   = returns.std()  * np.sqrt(periods_per_year)
    return mean_r / std_r


def _compute_max_drawdown(equity_curve: np.ndarray) -> float:
    rolling_max = np.maximum.accumulate(equity_curve)
    drawdowns   = (equity_curve - rolling_max) / rolling_max
    return float(-drawdowns.min())


def _compute_sortino(returns: np.ndarray, periods_per_year: int = 252) -> float:
    downside = returns[returns < 0]
    if len(downside) < 5:
        return -999.0
    downside_std = downside.std() * np.sqrt(periods_per_year)
    if downside_std == 0:
        return -999.0
    return (returns.mean() * periods_per_year) / downside_std


def _compute_calmar(equity_curve: np.ndarray, periods_per_year: int = 252) -> float:
    n_periods   = len(equity_curve)
    total_ret   = equity_curve[-1] / equity_curve[0] - 1
    ann_ret     = (1 + total_ret) ** (periods_per_year / n_periods) - 1
    mdd         = _compute_max_drawdown(equity_curve)
    return ann_ret / mdd if mdd > 0 else -999.0


def _simulate_strategy_returns(
    probs: np.ndarray,
    log_returns: np.ndarray,
    entry_threshold: float = 0.55,
    exit_threshold: float  = 0.45,
    tc: float = 0.0015,          # 0.15% per side (IDX buy cost)
) -> Tuple[np.ndarray, np.ndarray]:
    n = len(probs)
    positions = np.zeros(n)
    current   = 0

    for i in range(n):
        if current == 0 and probs[i] > entry_threshold:
            current = 1
        elif current == 1 and probs[i] < exit_threshold:
            current = 0
        positions[i] = current

    shifted_pos = np.roll(positions, 1)
    shifted_pos[0] = 0   

    trades        = np.abs(np.diff(shifted_pos, prepend=0))
    strategy_ret  = shifted_pos * log_returns
    strategy_ret_net = strategy_ret - trades * tc

    equity_curve = np.exp(np.cumsum(strategy_ret_net))
    return strategy_ret_net, equity_curve

def _walk_forward_sharpe_objective(
    trial: optuna.Trial,
    df: pd.DataFrame,
    feature_cols: list,
    n_splits: int = 5,
    entry_threshold: float = 0.55,
    exit_threshold: float  = 0.45,
) -> float:
    params = {
        "objective":        "binary",
        "metric":           "binary_logloss",
        "verbosity":        -1,
        "boosting_type":    "gbdt",
        "n_estimators":     trial.suggest_int("n_estimators", 200, 1000),
        "learning_rate":    trial.suggest_float("learning_rate", 5e-4, 0.1, log=True),
        "num_leaves":       trial.suggest_int("num_leaves", 8, 128),
        "max_depth":        trial.suggest_int("max_depth", 3, 10),
        "lambda_l1":        trial.suggest_float("lambda_l1", 1e-8, 5.0, log=True),
        "lambda_l2":        trial.suggest_float("lambda_l2", 1e-8, 5.0, log=True),
        "feature_fraction": trial.suggest_float("feature_fraction", 0.5, 1.0),
        "bagging_fraction": trial.suggest_float("bagging_fraction", 0.5, 1.0),
        "bagging_freq":     trial.suggest_int("bagging_freq", 1, 7),
        "min_child_samples":trial.suggest_int("min_child_samples", 10, 100),
        "random_state":     42,
    }

    tscv        = TimeSeriesSplit(n_splits=n_splits)
    fold_sharpes = []

    for fold_idx, (train_idx, val_idx) in enumerate(tscv.split(df)):
        train_fold = df.iloc[train_idx].dropna(subset=feature_cols + ["Target_Direction"])
        val_fold   = df.iloc[val_idx].dropna(subset=feature_cols + ["Log_Return"])

        if len(train_fold) < 100 or len(val_fold) < 30:
            continue  

        X_tr = train_fold[feature_cols]
        y_tr = train_fold["Target_Direction"]
        X_va = val_fold[feature_cols]

        # Class imbalance handling 
        pos_rate   = y_tr.mean()
        neg_rate   = 1 - pos_rate
        scale_pos  = neg_rate / pos_rate if pos_rate > 0 else 1.0
        params["scale_pos_weight"] = scale_pos

        model = lgb.LGBMClassifier(**params)
        model.fit(X_tr, y_tr)

        probs       = model.predict_proba(X_va)[:, 1]
        log_returns = val_fold["Log_Return"].values

        _, equity_curve = _simulate_strategy_returns(
            probs, log_returns, entry_threshold, exit_threshold
        )
        strategy_returns = np.diff(np.log(equity_curve), prepend=0)
        sharpe           = _compute_sharpe(strategy_returns)

        fold_sharpes.append(sharpe)

    if not fold_sharpes:
        return -999.0

    return float(np.mean(fold_sharpes))

class QuantModel:
    def __init__(self, df: pd.DataFrame, feature_cols: list):
        self.df           = df
        self.feature_cols = feature_cols
        self.model        = None
        self.best_params  = {}
        self.entry_threshold = 0.55
        self.exit_threshold  = 0.45

    def prepare_train_test_split(
        self, test_size: float = 0.2
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        split_idx  = int(len(self.df) * (1 - test_size))
        train_df   = self.df.iloc[:split_idx]
        test_df    = self.df.iloc[split_idx:]

        print(f"  [Split] Train: {train_df.index[0].date()} → {train_df.index[-1].date()} "
              f"({len(train_df)} rows)")
        print(f"  [Split] Test:  {test_df.index[0].date()} → {test_df.index[-1].date()} "
              f"({len(test_df)} rows)")
        return train_df, test_df

    def optimize_hyperparameters(
        self,
        train_df: pd.DataFrame,
        n_trials: int = 50,
        n_cv_folds: int = 5,
        entry_threshold: float = 0.55,
        exit_threshold: float  = 0.45,
    ) -> dict:
        self.entry_threshold = entry_threshold
        self.exit_threshold  = exit_threshold

        print(f"[HPO] Walk-forward Sharpe optimization | {n_trials} trials | {n_cv_folds} folds")

        study = optuna.create_study(
            direction  = "maximize",
            sampler    = optuna.samplers.TPESampler(seed=42),
            pruner     = optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=2),
        )

        objective = lambda trial: _walk_forward_sharpe_objective(
            trial, train_df, self.feature_cols, n_cv_folds,
            entry_threshold, exit_threshold,
        )
        study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

        self.best_params = study.best_trial.params
        best_sharpe      = study.best_trial.value

        print(f"[HPO] Best walk-forward Sharpe: {best_sharpe:.4f}")
        print(f"[HPO] Best params: {self.best_params}")
        return self.best_params

    def train_final_model(self, train_df: pd.DataFrame) -> None:
        clean_train = train_df.dropna(subset=self.feature_cols + ["Target_Direction"])

        X_train = clean_train[self.feature_cols]
        y_train = clean_train["Target_Direction"]

        pos_rate         = y_train.mean()
        scale_pos        = (1 - pos_rate) / pos_rate if pos_rate > 0 else 1.0
        final_params     = {**self.best_params,
                             "objective":      "binary",
                             "verbosity":      -1,
                             "random_state":   42,
                             "scale_pos_weight": scale_pos}

        self.model = lgb.LGBMClassifier(**final_params)
        self.model.fit(X_train, y_train)

        print(f"[Train] Final model trained on {len(X_train)} samples | "
              f"Features: {len(self.feature_cols)} | "
              f"Positive class rate: {pos_rate:.2%}")

    def evaluate(self, test_df: pd.DataFrame) -> dict:
        if self.model is None:
            raise RuntimeError("Call train_final_model() before evaluate().")

        # dropna inside test fold 
        clean_test  = test_df.dropna(subset=self.feature_cols + ["Target_Direction", "Log_Return"])
        X_test      = clean_test[self.feature_cols]
        y_test      = clean_test["Target_Direction"]
        log_returns = clean_test["Log_Return"].values

        preds = self.model.predict(X_test)
        probs = self.model.predict_proba(X_test)[:, 1]

        # Threshold-dependent classification metrics
        self.entry_threshold = max(np.percentile(probs, 80), 0.505)
        self.exit_threshold  = np.percentile(probs, 40)

        signal_preds = (probs > self.entry_threshold).astype(int)

        # Backtest simulation 
        strategy_ret_net, equity_curve = _simulate_strategy_returns(
            probs, log_returns, self.entry_threshold, self.exit_threshold
        )
        benchmark_equity = np.exp(np.cumsum(log_returns))

        # Risk metrics
        sharpe  = _compute_sharpe(strategy_ret_net)
        sortino = _compute_sortino(strategy_ret_net)
        mdd     = _compute_max_drawdown(equity_curve)
        calmar  = _compute_calmar(equity_curve)

        # Win/loss analysis
        winning_trades  = strategy_ret_net[strategy_ret_net > 0]
        losing_trades   = strategy_ret_net[strategy_ret_net < 0]
        win_rate        = len(winning_trades) / max(len(strategy_ret_net[strategy_ret_net != 0]), 1)
        avg_win         = winning_trades.mean() if len(winning_trades) > 0 else 0
        avg_loss        = losing_trades.mean()  if len(losing_trades)  > 0 else 0
        profit_factor   = abs(winning_trades.sum() / losing_trades.sum()) if losing_trades.sum() != 0 else np.inf

        total_return_strat = (equity_curve[-1] - 1) * 100
        total_return_bm    = (benchmark_equity[-1] - 1) * 100

        metrics = {
            # Return
            "total_return_strategy_pct": round(total_return_strat, 2),
            "total_return_benchmark_pct": round(total_return_bm, 2),
            "alpha_pct": round(total_return_strat - total_return_bm, 2),
            # Risk-adjusted
            "sharpe_ratio":     round(sharpe, 4),
            "sortino_ratio":    round(sortino, 4),
            "max_drawdown_pct": round(mdd * 100, 2),
            "calmar_ratio":     round(calmar, 4),
            # Classification
            "roc_auc":          round(roc_auc_score(y_test, probs), 4),
            "precision":        round(precision_score(y_test, signal_preds, zero_division=0), 4),
            "recall":           round(recall_score(y_test, signal_preds, zero_division=0), 4),
            "f1_score":         round(f1_score(y_test, signal_preds, zero_division=0), 4),
            # Trade stats
            "win_rate_pct":     round(win_rate * 100, 2),
            "avg_win_pct":      round(avg_win * 100, 4),
            "avg_loss_pct":     round(avg_loss * 100, 4),
            "profit_factor":    round(profit_factor, 4),
            # Thresholds used
            "entry_threshold":  round(self.entry_threshold, 4),
            "exit_threshold":   round(self.exit_threshold, 4),
        }

        print("\n" + "=" * 50)
        print("  EVALUATION RESULTS")
        print("=" * 50)
        for k, v in metrics.items():
            print(f"  {k:<35} {v}")
        print("=" * 50 + "\n")

        return metrics, probs

    def get_feature_importance(self, top_n: int = 20) -> pd.DataFrame:
        if self.model is None:
            raise RuntimeError("Train the model first.")
        imp = pd.DataFrame({
            "feature":    self.feature_cols,
            "importance": self.model.feature_importances_,
        }).sort_values("importance", ascending=False).head(top_n)
        return imp


if __name__ == "__main__":
    np.random.seed(42)
    n = 2000
    idx = pd.date_range("2015-01-01", periods=n, freq="B")
    df  = pd.DataFrame({
        "RSI":             np.random.uniform(20, 80, n),
        "MACD_Hist":       np.random.randn(n) * 0.5,
        "BB_PctB":         np.random.uniform(0, 1, n),
        "BB_Width":        np.random.uniform(0.01, 0.05, n),
        "ATR_Pct":         np.random.uniform(0.005, 0.02, n),
        "Realized_Vol_20": np.random.uniform(0.1, 0.4, n),
        "Return_Lag_1":    np.random.randn(n) * 0.01,
        "Log_Return":      np.random.randn(n) * 0.01,
        "Target_Direction": np.random.randint(0, 2, n),
        "Volume_ZScore":   np.random.randn(n),
        "OBV_Change":      np.random.randn(n) * 1e6,
    }, index=idx)

    features = ["RSI", "MACD_Hist", "BB_PctB", "BB_Width", "ATR_Pct",
                "Realized_Vol_20", "Return_Lag_1", "Volume_ZScore", "OBV_Change"]

    qm = QuantModel(df, features)
    train_df, test_df = qm.prepare_train_test_split()
    qm.optimize_hyperparameters(train_df, n_trials=5, n_cv_folds=3)
    qm.train_final_model(train_df)
    metrics, probs = qm.evaluate(test_df)
    print(qm.get_feature_importance())