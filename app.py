import os
from pathlib import Path

import joblib
import numpy as np
from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS


APP_DIR = Path(__file__).resolve().parent
MODEL_PATH = Path(os.getenv("MODEL_PATH", APP_DIR / "artifacts" / "vpn_model_bundle.joblib"))
PLOTS_PATH = Path(os.getenv("PLOTS_PATH", APP_DIR / "plots"))

app = Flask(__name__)
cors_origins = os.getenv("CORS_ORIGINS", "*")
CORS(app, resources={r"/api/*": {"origins": cors_origins.split(",") if cors_origins != "*" else "*"}})

bundle = None


def load_bundle():
    global bundle
    if not MODEL_PATH.exists():
        bundle = None
        return False

    if bundle is None:
        bundle = joblib.load(MODEL_PATH)

    return True


def ensure_bundle_loaded():
    try:
        return load_bundle()
    except MemoryError:
        raise RuntimeError("Insufficient memory to load model on current Render instance")


def build_feature_vector(payload: dict):
    num_cols = bundle["num_cols"]
    final_features = bundle["final_features"]

    raw_values = {}
    missing = []

    for col in num_cols:
        if col not in payload:
            missing.append(col)
            continue
        try:
            raw_values[col] = float(payload[col])
        except (TypeError, ValueError):
            raise ValueError(f"Invalid value for feature '{col}'")

    if missing:
        raise ValueError(f"Missing required features: {', '.join(missing[:8])}")

    # Build engineered features exactly like training.
    fv = dict(raw_values)
    fv["fiat_biat_ratio"] = fv["total_fiat"] / (fv["total_biat"] + 1)
    fv["flow_iat_cv"] = fv["std_flowiat"] / (fv["mean_flowiat"] + 1)
    fv["active_idle_ratio"] = fv["mean_active"] / (fv["mean_idle"] + 1)
    fv["pkt_byte_ratio"] = fv["flowPktsPerSecond"] / (fv["flowBytesPerSecond"] + 1)

    vector = np.array([fv[c] for c in final_features], dtype=float).reshape(1, -1)
    return vector


@app.get("/api/health")
def health():
    if not MODEL_PATH.exists():
        return jsonify({"status": "error", "message": f"Model bundle not found at {MODEL_PATH}"}), 500
    return jsonify({"status": "ok", "model_path": str(MODEL_PATH)})


@app.get("/api/features")
def features():
    try:
        if not ensure_bundle_loaded():
            return jsonify({"error": "Model bundle not found"}), 500
    except RuntimeError as exc:
        return jsonify({"error": str(exc)}), 503

    return jsonify(
        {
            "input_features": bundle["num_cols"],
            "final_feature_count": len(bundle["final_features"]),
        }
    )


@app.get("/api/plots")
def plots():
    if not PLOTS_PATH.exists():
        return jsonify({"images": []})

    files = sorted(
        [p.name for p in PLOTS_PATH.iterdir() if p.is_file() and p.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}]
    )
    return jsonify({"images": [f"/api/plots/{name}" for name in files]})


@app.get("/api/plots/<path:filename>")
def plot_file(filename):
    return send_from_directory(PLOTS_PATH, filename)


@app.post("/api/predict")
def predict():
    try:
        if not ensure_bundle_loaded():
            return jsonify({"error": "Model bundle not found"}), 500
    except RuntimeError as exc:
        return jsonify({"error": str(exc)}), 503

    payload = request.get_json(silent=True) or {}

    try:
        x = build_feature_vector(payload)
        x_scaled = bundle["scaler"].transform(x)
        proba_vpn = float(bundle["model"].predict_proba(x_scaled)[0][1])
        pred = 1 if proba_vpn >= 0.5 else 0
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except Exception as exc:
        return jsonify({"error": f"Prediction failed: {exc}"}), 500

    response = jsonify(
        {
            "prediction": "VPN" if pred == 1 else "Non-VPN",
            "vpn_probability": round(proba_vpn, 6),
            "non_vpn_probability": round(1.0 - proba_vpn, 6),
        }
    )

    # Free model from memory after each prediction on low-memory instances.
    if os.getenv("UNLOAD_MODEL_AFTER_PREDICT", "1") == "1":
        global bundle
        bundle = None

    return response


if __name__ == "__main__":
    port = int(os.getenv("PORT", "5000"))
    app.run(host="0.0.0.0", port=port, debug=False)
