"""
train_model.py
----------------
Standalone training script for RoadGuard AI.

Run this manually:
    python train_model.py

...or just launch the app — app.py calls the same functions
automatically the first time if models/trained_model.pkl is missing.

What it does:
1. Loads data/accidents.csv
2. Engineers leakage-safe features
3. Trains a RandomForestRegressor to predict risk_score (0-100 scale)
   using ONLY pre-event features (see src/data_utils.py docstring for
   the full data-leakage explanation)
4. Saves model + preprocessor + metrics to models/
5. Builds geographic hotspot clusters + aggregated stats and saves
   them to models/hotspots.csv (used by the Hotspot Map / Top 10 pages)
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.data_utils import ALL_FEATURES, MODELS_DIR, build_hotspots, load_processed_data
from src.model_utils import predict_risk_score, train_and_save, load_artifacts


def main():
    print("Loading and processing dataset...")
    df = load_processed_data()
    print(f"  {len(df)} rows loaded.")

    print("Training model (RandomForestRegressor, pre-event features only)...")
    metrics = train_and_save(df)
    print("  Training complete. Evaluation metrics:")
    for k, v in metrics.items():
        print(f"    {k}: {v}")

    print("Scoring full dataset with trained model (for hotspot aggregation)...")
    preprocessor, model, feature_names, _ = load_artifacts()
    df["predicted_risk"] = predict_risk_score(df, preprocessor, model)

    print("Building geographic hotspot clusters...")
    hotspots = build_hotspots(df, predicted_risk_col="predicted_risk")
    os.makedirs(MODELS_DIR, exist_ok=True)
    hotspots_path = os.path.join(MODELS_DIR, "hotspots.csv")
    hotspots.to_csv(hotspots_path, index=False)
    print(f"  {len(hotspots)} hotspots saved to {hotspots_path}")

    print("\nDone. You can now run: streamlit run app.py")


if __name__ == "__main__":
    main()
