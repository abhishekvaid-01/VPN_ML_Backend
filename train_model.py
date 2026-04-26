import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "consolidated_traffic_data.csv"
MODEL_PATH = PROJECT_ROOT / "artifacts" / "vpn_model_bundle.joblib"


def prepare_dataset(df_raw: pd.DataFrame):
    df = df_raw.copy()
    df["label"] = df["traffic_type"].apply(lambda x: 1 if str(x).startswith("VPN") else 0)

    num_cols = [c for c in df.select_dtypes(include=[np.number]).columns if c != "label"]

    # Match your notebook logic: replace -1, mode imputation, IQR capping, feature engineering.
    df[num_cols] = df[num_cols].replace(-1, np.nan)

    imputer = SimpleImputer(strategy="most_frequent")
    x_imp = imputer.fit_transform(df[num_cols])
    df_imp = pd.DataFrame(x_imp, columns=num_cols)

    x_cap = df_imp.copy()
    for col in num_cols:
        q1 = x_cap[col].quantile(0.25)
        q3 = x_cap[col].quantile(0.75)
        iqr = q3 - q1
        low = q1 - 1.5 * iqr
        high = q3 + 1.5 * iqr
        x_cap[col] = x_cap[col].clip(lower=low, upper=high)

    x_cap["fiat_biat_ratio"] = x_cap["total_fiat"] / (x_cap["total_biat"] + 1)
    x_cap["flow_iat_cv"] = x_cap["std_flowiat"] / (x_cap["mean_flowiat"] + 1)
    x_cap["active_idle_ratio"] = x_cap["mean_active"] / (x_cap["mean_idle"] + 1)
    x_cap["pkt_byte_ratio"] = x_cap["flowPktsPerSecond"] / (x_cap["flowBytesPerSecond"] + 1)

    feature_cols = list(x_cap.columns)

    corr_mat = x_cap[feature_cols].corr().abs()
    upper = corr_mat.where(np.triu(np.ones(corr_mat.shape), k=1).astype(bool))
    to_drop = {
        col
        for col in upper.columns
        for row in upper.index
        if pd.notna(upper.at[row, col]) and upper.at[row, col] > 0.95
    }
    final_features = [f for f in feature_cols if f not in to_drop]

    x_final = x_cap[final_features].copy()
    y = df["label"].values

    return x_final, y, imputer, num_cols, final_features


def train_and_save():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found at: {DATA_PATH}")

    df_raw = pd.read_csv(DATA_PATH)
    x_final, y, imputer, num_cols, final_features = prepare_dataset(df_raw)

    # Oversampling to balance classes, as in your script.
    rng = np.random.RandomState(42)
    cls0 = np.where(y == 0)[0]
    cls1 = np.where(y == 1)[0]
    minor, major = (cls1, cls0) if len(cls1) < len(cls0) else (cls0, cls1)
    extra = rng.choice(minor, size=len(major) - len(minor), replace=True)
    idx_b = np.concatenate([cls0, cls1, extra])
    rng.shuffle(idx_b)

    x_bal = x_final.iloc[idx_b].values
    y_bal = y[idx_b]

    x_train, _, y_train, _ = train_test_split(
        x_bal, y_bal, test_size=0.2, random_state=42, stratify=y_bal
    )

    scaler = RobustScaler()
    x_train_scaled = scaler.fit_transform(x_train)

    model = ExtraTreesClassifier(n_estimators=200, random_state=42, n_jobs=-1)
    model.fit(x_train_scaled, y_train)

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "model": model,
            "scaler": scaler,
            "imputer": imputer,
            "num_cols": num_cols,
            "final_features": final_features,
        },
        MODEL_PATH,
    )

    print(f"Saved model bundle at: {MODEL_PATH}")
    print(f"Final features count: {len(final_features)}")


if __name__ == "__main__":
    train_and_save()
