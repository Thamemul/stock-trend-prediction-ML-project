# StockSense AI — Stock Price Trend Prediction with CI/CD Pipeline

![Python](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)
![Flask](https://img.shields.io/badge/Flask-3.1-000000?logo=flask)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.6-F7931E?logo=scikitlearn)
![XGBoost](https://img.shields.io/badge/XGBoost-2.1-blue)
![CI/CD](https://img.shields.io/badge/CI%2FCD-GitHub%20Actions-2088FF?logo=githubactions&logoColor=white)
![Deploy](https://img.shields.io/badge/Deploy-Render-46E3B7?logo=render)

> An end-to-end Machine Learning project that predicts whether a stock's price will go **UP** or **DOWN** the next day, with a fully automated CI/CD pipeline that trains, tests, and deploys on every push.

---

## Features

- **5 ML Models** — Random Forest, Gradient Boosting, XGBoost, Voting Ensemble, Stacking Ensemble
- **Auto-selects best model** based on accuracy, precision, recall, F1 & ROC AUC
- **Flask dashboard** with prediction form, confidence scores, and interactive charts
- **CI/CD pipeline** via GitHub Actions — trains model, runs tests, deploys to Render
- **Health check endpoint** for deployment monitoring
- **Responsive UI** with glassmorphism dark theme, Chart.js visualizations
- **Production-ready** — logging, error handling, environment variables, gunicorn

---

## Project Structure

```
project/
├── app.py                    # Flask backend (routes & prediction API)
├── train_model.py            # ML training pipeline (5 models)
├── generate_dataset.py       # Synthetic dataset generator
├── test_app.py               # Pytest test suite
├── dataset.csv               # Stock market dataset
├── model.pkl                 # Trained best model (auto-generated)
├── scaler.pkl                # StandardScaler (auto-generated)
├── label_encoder.pkl         # LabelEncoder (auto-generated)
├── metrics.json              # Model comparison metrics (auto-generated)
├── requirements.txt          # Python dependencies
├── Procfile                  # Render/gunicorn entry point
├── runtime.txt               # Python version for Render
├── README.md                 # This file
│
├── templates/
│   └── index.html            # Dashboard UI template
│
├── static/
│   └── style.css             # Custom CSS (dark theme)
│
└── .github/
    └── workflows/
        └── deploy.yml        # CI/CD pipeline
```

---

## Quick Start (Local)

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/stock-trend-prediction.git
cd stock-trend-prediction
```

### 2. Create virtual environment

```bash
python -m venv venv
venv\Scripts\activate       # Windows
# source venv/bin/activate  # macOS/Linux
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Generate dataset (optional — dataset.csv is included)

```bash
python generate_dataset.py
```

### 5. Train the model

```bash
python train_model.py
```

This will produce: `model.pkl`, `scaler.pkl`, `label_encoder.pkl`, `metrics.json`

### 6. Run the Flask app

```bash
python app.py
```

Open **http://localhost:5000** in your browser.

### 7. Run tests

```bash
pytest test_app.py -v
```

---

## CI/CD Pipeline

The GitHub Actions workflow (`.github/workflows/deploy.yml`) runs on every push to `main`:

```
Push Code
  → Checkout repository
  → Setup Python 3.11
  → Install dependencies
  → Train model (train_model.py)
  → Verify artifacts exist
  → Run pytest suite
  → Verify /health endpoint
  → Deploy to Render via API
```

### Pipeline Badge

Add this to your repo (replace with your GitHub username/repo):

```markdown
![CI/CD](https://github.com/YOUR_USERNAME/stock-trend-prediction/actions/workflows/deploy.yml/badge.svg)
```

---

## Deployment to Render

### Step-by-step:

1. **Create account** at [render.com](https://render.com)
2. **New Web Service** → Connect your GitHub repo
3. **Settings:**
   - **Build Command:** `pip install -r requirements.txt && python train_model.py`
   - **Start Command:** `gunicorn app:app --bind 0.0.0.0:$PORT --workers 2`
   - **Python Version:** 3.11
4. **Get API Key:** Dashboard → Account Settings → API Keys → Create
5. **Get Service ID:** From the Render dashboard URL: `https://dashboard.render.com/web/srv-XXXXXX` → `srv-XXXXXX` is your Service ID
6. **Add GitHub Secrets** in your repo → Settings → Secrets → Actions:
   - `RENDER_API_KEY` — your Render API key
   - `RENDER_SERVICE_ID` — your Render service ID (e.g., `srv-xxxxx`)
7. Push to `main` → GitHub Actions will auto-deploy!

---

## API Endpoints

| Method | Endpoint    | Description                        |
|--------|-------------|------------------------------------|
| GET    | `/`         | Main dashboard UI                  |
| POST   | `/predict`  | Predict stock trend from form data |
| GET    | `/metrics`  | Model comparison results (JSON)    |
| GET    | `/health`   | Health check for monitoring        |

### Prediction Input Fields

| Field      | Description              | Example  |
|------------|--------------------------|----------|
| Open       | Opening price            | 150.25   |
| High       | Highest price            | 155.80   |
| Low        | Lowest price             | 148.10   |
| Close      | Closing price            | 153.40   |
| Volume     | Trading volume           | 5000000  |
| SMA_10     | 10-day moving average    | 151.50   |
| SMA_50     | 50-day moving average    | 149.00   |
| RSI        | Relative Strength Index  | 55.00    |
| MACD       | MACD indicator           | 1.20     |
| Prev_Close | Previous day close price | 150.00   |

---

## How to Update & Retrain

### Update the dataset:
1. Replace `dataset.csv` with your new data (same column format)
2. Run `python train_model.py` locally to verify
3. Push to GitHub → CI/CD will auto-retrain and deploy

### Use real stock data:
- Download OHLCV data from Yahoo Finance, Alpha Vantage, or similar
- Ensure columns match: Open, High, Low, Close, Volume, SMA_10, SMA_50, RSI, MACD, Prev_Close, Trend
- Compute technical indicators using `pandas` / `ta-lib`

---

## Screenshots

> Add screenshots of your dashboard here after running the app.

| Dashboard | Prediction Result |
|-----------|-------------------|
| *screenshot* | *screenshot* |

---

## Tech Stack

| Category       | Technology                                         |
|----------------|----------------------------------------------------|
| Language       | Python 3.11                                        |
| ML Libraries   | scikit-learn, XGBoost, pandas, numpy               |
| Web Framework  | Flask 3.1                                          |
| Frontend       | HTML5, CSS3, Bootstrap 5, Chart.js                 |
| Server         | Gunicorn                                           |
| CI/CD          | GitHub Actions                                     |
| Deployment     | Render                                             |
| Testing        | Pytest                                             |

---

## Future Improvements

- [ ] Add LSTM / Transformer models for time-series prediction
- [ ] Integrate live stock data via Yahoo Finance API
- [ ] Add model versioning with MLflow or DVC
- [ ] Implement A/B testing between model versions
- [ ] Add WebSocket for real-time prediction updates
- [ ] Containerize with Docker for multi-platform deployment
- [ ] Add Prometheus + Grafana monitoring
- [ ] Implement feature importance visualization
- [ ] Add backtesting module for strategy evaluation

---

## License

MIT License — see [LICENSE](LICENSE) for details.

---

**Built with Python, Flask, scikit-learn & XGBoost | CI/CD with GitHub Actions | Deployed on Render**
