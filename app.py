"""
app.py -- Flask Backend for Stock Price Trend Prediction

Routes:
  GET  /           -> Render the main dashboard UI
  POST /predict    -> Predict stock trend from form input
  GET  /metrics    -> Return model comparison JSON
  GET  /health     -> Health check for deployment testing
"""

import json
import os
import logging
import sys
import numpy as np
import joblib
from flask import Flask, render_template, request, jsonify

# ── Logging ───────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

# ── Flask app ─────────────────────────────────────────────────
app = Flask(__name__)
app.config["SECRET_KEY"] = os.environ.get("SECRET_KEY", "stock-prediction-secret-key")

# ── Paths ─────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "scaler.pkl")
ENCODER_PATH = os.path.join(BASE_DIR, "label_encoder.pkl")
METRICS_PATH = os.path.join(BASE_DIR, "metrics.json")

# ── Load artifacts at startup ─────────────────────────────────
try:
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    label_encoder = joblib.load(ENCODER_PATH)
    with open(METRICS_PATH, encoding="utf-8") as f:
        metrics_data = json.load(f)
    logger.info("[STARTUP] Model artifacts loaded successfully.")
except FileNotFoundError as e:
    logger.warning(f"[STARTUP] Artifact missing: {e}. Run train_model.py first.")
    model = scaler = label_encoder = None
    metrics_data = {}

# Feature order must match training
FEATURE_COLS = [
    "Open", "High", "Low", "Close", "Volume",
    "SMA_10", "SMA_50", "RSI", "MACD", "Prev_Close",
]


# ── Routes ────────────────────────────────────────────────────
@app.route("/")
def index():
    """Render the main dashboard UI."""
    return render_template("index.html", metrics=metrics_data)


@app.route("/predict", methods=["POST"])
def predict():
    """Accept form input, run prediction, return JSON result."""
    if model is None:
        return jsonify({"error": "Model not loaded. Run train_model.py first."}), 503

    try:
        # Parse all 10 feature inputs from the form
        features = []
        for col in FEATURE_COLS:
            value = float(request.form[col])
            features.append(value)

        features_array = np.array([features])

        # Scale and predict
        features_scaled = scaler.transform(features_array)
        prediction_encoded = model.predict(features_scaled)
        label = label_encoder.inverse_transform(prediction_encoded)[0]

        # Confidence via predict_proba
        confidence = 0.0
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(features_scaled)[0]
            confidence = round(float(max(proba)) * 100, 1)

        model_name = metrics_data.get("best_model", "Unknown")

        logger.info(
            f"[PREDICT] {label} (confidence={confidence}%) "
            f"using {model_name}"
        )

        return jsonify({
            "prediction": label,
            "confidence": confidence,
            "model_used": model_name,
        })

    except KeyError as e:
        return jsonify({"error": f"Missing input field: {e}"}), 400
    except ValueError as e:
        return jsonify({"error": f"Invalid input value: {e}"}), 400
    except Exception as e:
        logger.error(f"[ERROR] Prediction failed: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/metrics")
def metrics():
    """Return model comparison results as JSON."""
    return jsonify(metrics_data)


@app.route("/health")
def health():
    """Health check endpoint for deployment verification."""
    status = {
        "status": "healthy",
        "model_loaded": model is not None,
        "best_model": metrics_data.get("best_model", "N/A"),
        "best_accuracy": metrics_data.get("best_accuracy", "N/A"),
    }
    return jsonify(status), 200


# ── Entry point ───────────────────────────────────────────────
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    logger.info(f"[SERVER] Flask running at http://localhost:{port}")
    app.run(debug=True, host="0.0.0.0", port=port)
