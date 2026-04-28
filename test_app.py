"""
test_app.py -- Unit tests for the Flask stock prediction app.

Tests:
  - Health check returns 200
  - Metrics endpoint returns JSON with model data
  - Index page renders successfully
  - Predict endpoint returns valid prediction
  - Predict endpoint handles missing fields
"""

import json
import os
import sys
import pytest

# Ensure project root is on the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app import app


@pytest.fixture
def client():
    """Create a Flask test client."""
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client


# ── Health check ──────────────────────────────────────────────
def test_health(client):
    """GET /health should return 200 with status=healthy."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.get_json()
    assert data["status"] == "healthy"
    assert "model_loaded" in data


# ── Metrics ───────────────────────────────────────────────────
def test_metrics(client):
    """GET /metrics should return JSON with best_model key."""
    response = client.get("/metrics")
    assert response.status_code == 200
    data = response.get_json()
    assert "best_model" in data
    assert "models" in data


# ── Index page ────────────────────────────────────────────────
def test_index(client):
    """GET / should return 200 and contain the page title."""
    response = client.get("/")
    assert response.status_code == 200
    assert b"Stock" in response.data or b"stock" in response.data


# ── Prediction ────────────────────────────────────────────────
def test_predict_valid(client):
    """POST /predict with valid data should return UP or DOWN."""
    form_data = {
        "Open": "150.0",
        "High": "155.0",
        "Low": "148.0",
        "Close": "153.0",
        "Volume": "5000000",
        "SMA_10": "151.5",
        "SMA_50": "149.0",
        "RSI": "55.0",
        "MACD": "1.2",
        "Prev_Close": "150.0",
    }
    response = client.post("/predict", data=form_data)
    assert response.status_code == 200
    data = response.get_json()
    assert "prediction" in data
    assert data["prediction"] in ["UP", "DOWN"]
    assert "confidence" in data
    assert 0 <= data["confidence"] <= 100


# ── Missing field ─────────────────────────────────────────────
def test_predict_missing_field(client):
    """POST /predict with missing fields should return 400."""
    form_data = {
        "Open": "150.0",
        "High": "155.0",
        # Missing many fields
    }
    response = client.post("/predict", data=form_data)
    assert response.status_code == 400


# ── Invalid value ─────────────────────────────────────────────
def test_predict_invalid_value(client):
    """POST /predict with non-numeric value should return 400."""
    form_data = {
        "Open": "not_a_number",
        "High": "155.0",
        "Low": "148.0",
        "Close": "153.0",
        "Volume": "5000000",
        "SMA_10": "151.5",
        "SMA_50": "149.0",
        "RSI": "55.0",
        "MACD": "1.2",
        "Prev_Close": "150.0",
    }
    response = client.post("/predict", data=form_data)
    assert response.status_code == 400
