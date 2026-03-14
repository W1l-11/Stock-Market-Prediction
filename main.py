import joblib
import numpy as np

from backtester import VectorizedBacktester
from data_loader import DataLoader
from feature_engineering import FeatureEngineer
from model_trainer import QuantModel


def main():
    TICKER     = "BBRI.JK"
    START      = "2020-01-01"
    END        = "2026-02-28"
    MODEL_PATH = "quant_model_bbri.pkl"
    TARGET_VOL = 0.15      

    print("\n--- 1. Data Loading ---")
    loader   = DataLoader(TICKER, START, END)
    raw_data = loader.get_historical_data()

    print("\n--- 2. Feature Engineering ---")
    fe = FeatureEngineer(raw_data)
    processed = (fe
                 .add_technical_indicators()
                 .add_volume_features()
                 .add_macro_features()
                 .add_lags()
                 .add_target(forward_days=1)
                 .get_df())  

    features = fe.get_feature_names()
    print(f"  Feature count: {len(features)}")
    print(f"  Features: {features}")

    print("\n--- 3. Model Training ---")
    qm = QuantModel(processed, features)
    train_df, test_df = qm.prepare_train_test_split(test_size=0.2)

    qm.optimize_hyperparameters(train_df, n_trials=30, n_cv_folds=5)
    qm.train_final_model(train_df)

    print("\n--- 4. Evaluation ---")
    metrics, probs = qm.evaluate(test_df)

    print("\nTop 10 Feature Importances:")
    print(qm.get_feature_importance(top_n=10).to_string(index=False))

    print("\n--- 5. Backtest ---")
    clean_test = test_df.dropna(subset=features + ["Log_Return"])

    bt = VectorizedBacktester(
        clean_test, probs,
        target_vol=TARGET_VOL,
    )
    bt.run(
        entry_threshold=qm.entry_threshold,
        exit_threshold=qm.exit_threshold,
    )
    bt.print_metrics()
    bt.plot_results(TICKER)

    model_export = {
        "model":             qm.model,
        "features":          features,
        "entry_threshold":   qm.entry_threshold,
        "exit_threshold":    qm.exit_threshold,
        "target_vol":        TARGET_VOL,
        "macro_cols_required": ["USDIDR_Ret", "SPX_Ret", "OIL_Ret", "JKSE_Ret",
                                 "USDIDR", "SPX", "OIL", "JKSE"],
    }
    joblib.dump(model_export, MODEL_PATH)
    print(f"\n[SUCCESS] Model saved to {MODEL_PATH}")
    print(f"  Entry threshold: {qm.entry_threshold:.4f}")
    print(f"  Exit  threshold: {qm.exit_threshold:.4f}")


if __name__ == "__main__":
    main()