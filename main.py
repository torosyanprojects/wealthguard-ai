
import json
import joblib
import numpy as np
import pandas as pd

from pathlib import Path
from functools import lru_cache
from typing import Dict, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from wealthguard_engine import analyze_company, analyze_finances


# ==================================================
# FILE LOCATIONS
# ==================================================

BASE_DIR = Path(__file__).resolve().parent


def find_asset(filename, folders=("", "models", "data")):
    for folder in folders:
        path = BASE_DIR / folder / filename
        if path.is_file():
            return path
    return None


def required_asset(filename):
    path = find_asset(filename)
    if path is None:
        raise FileNotFoundError(
            f"Missing required asset: {filename}"
        )
    return path


# ==================================================
# LOAD TRAINED MODELS
# ==================================================

fundex_model = joblib.load(
    required_asset("fundex_pipeline.joblib")
)

fundex_features = list(
    joblib.load(required_asset("fundex_features.joblib"))
)

config_path = find_asset("model_config.json")

if config_path:
    with open(config_path, encoding="utf-8") as f:
        config = json.load(f)
else:
    config = {}

warning_threshold = float(
    config.get("warning_threshold", 0.5)
)


@lru_cache(maxsize=2)
def load_regional_model(region):
    if region not in ("us", "poland"):
        raise ValueError("Unsupported model")

    model = joblib.load(
        required_asset(f"{region}_model.joblib")
    )

    imputer = joblib.load(
        required_asset(f"{region}_imputer.joblib")
    )

    features = list(
        joblib.load(
            required_asset(f"{region}_features.joblib")
        )
    )

    return model, imputer, features


# ==================================================
# FASTAPI APPLICATION
# ==================================================

app = FastAPI(
    title="WealthGuard AI API",
    version="2.1.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://wealthguard-frontend-iota.vercel.app"
    ],
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization"]
)


class FinancialRequest(BaseModel):
    financial_data: Dict[str, Optional[float]] = Field(...)


# ==================================================
# DATA PROCESSING
# ==================================================

def clean_values(data, features):
    row = pd.DataFrame([
        {
            name: data.get(name, np.nan)
            for name in features
        }
    ])

    return (
        row.apply(pd.to_numeric, errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
    )


def feature_coverage(row):
    return float(
        row.notna().mean(axis=1).iloc[0]
    )


def financial_report(data):
    return analyze_finances(data)


# ==================================================
# HEALTH CHECK
# ==================================================

@app.get("/health")
def health():
    return {
        "status": "healthy",
        "project": "WealthGuard AI",
        "version": "2.1.0",
        "main_model": "fundex",
        "available_models": [
            "fundex",
            "us",
            "poland"
        ]
    }


# ==================================================
# MAIN WEBSITE ANALYSIS
# ==================================================

@app.post("/analyze")
def analyze(request: FinancialRequest):

    data = request.financial_data

    row = clean_values(
        data,
        fundex_features
    )

    coverage = feature_coverage(row)

    if coverage < 0.50:
        return {
            "project": "WealthGuard AI",
            "model_score": None,
            "prediction_status": "insufficient_data",
            "feature_coverage": round(coverage, 4),
            **financial_report(data),
            "message": (
                "More financial data is needed "
                "to generate a model score."
            )
        }

    try:
        report = analyze_company(
            data,
            fundex_model,
            fundex_features,
            warning_threshold
        )

        report["prediction_status"] = "provisional"
        report["feature_coverage"] = round(coverage, 4)

        return report

    except Exception as exc:
        raise HTTPException(
            status_code=500,
           detail=f"Financial analysis failed: {type(exc).__name__}: {str(exc)}"
        )


# ==================================================
# FUNDEX MODEL ANALYSIS
# ==================================================

@app.post("/analyze/fundex")
def analyze_fundex(request: FinancialRequest):

    data = request.financial_data

    row = clean_values(
        data,
        fundex_features
    )

    coverage = feature_coverage(row)

    if coverage < 0.50:
        return {
            "project": "WealthGuard AI",
            "model": "fundex",
            "model_score": None,
            "prediction_status": "insufficient_data",
            "feature_coverage": round(coverage, 4),
            **financial_report(data)
        }

    try:
        score = float(
            fundex_model.predict_proba(row)[0, 1]
        )

        return {
            "project": "WealthGuard AI",
            "model": "fundex",
            "model_score": round(score, 4),
            "score_type": "Uncalibrated model output",
            "prediction_status": "provisional",
            "feature_coverage": round(coverage, 4),
            **financial_report(data),
            "disclaimer": (
                "Experimental financial screening only. "
                "The score is not a calibrated "
                "bankruptcy probability."
            )
        }

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"FundEx prediction failed: {type(exc).__name__}"
        )


# ==================================================
# REGIONAL MODEL ANALYSIS
# ==================================================

@app.post("/analyze/{region}")
def analyze_region(
    region: str,
    request: FinancialRequest
):

    if region not in ("us", "poland"):
        raise HTTPException(
            status_code=404,
            detail="Unknown model"
        )

    try:
        model, imputer, features = load_regional_model(region)

        data = request.financial_data

        row = clean_values(data, features)

        coverage = feature_coverage(row)

        if coverage < 0.50:
            return {
                "project": "WealthGuard AI",
                "model": region,
                "model_score": None,
                "prediction_status": "insufficient_data",
                "feature_coverage": round(coverage, 4),
                **financial_report(data)
            }

        processed = imputer.transform(row)

        score = float(
            model.predict_proba(processed)[0, 1]
        )

        return {
            "project": "WealthGuard AI",
            "model": region,
            "model_score": round(score, 4),
            "score_type": "Uncalibrated model output",
            "prediction_status": "provisional",
            "feature_coverage": round(coverage, 4),
            **financial_report(data),
            "disclaimer": (
                "Experimental financial screening only. "
                "Not a calibrated bankruptcy probability."
            )
        }

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Regional prediction failed: {type(exc).__name__}"
        )


# ==================================================
# AVAILABLE MODEL INFORMATION
# ==================================================

@app.get("/models")
def available_models():
    return {
        "project": "WealthGuard AI",
        "models": [
            {
                "id": "fundex",
                "endpoint": "/analyze/fundex",
                "feature_count": len(fundex_features)
            },
            {
                "id": "us",
                "endpoint": "/analyze/us"
            },
            {
                "id": "poland",
                "endpoint": "/analyze/poland"
            }
        ],
        "default_endpoint": "/analyze"
    }


# ==================================================
# DASHBOARD DATA
# ==================================================

def read_dashboard(filename):

    path = find_asset(
        filename,
        ("", "data")
    )

    if path is None:
        raise HTTPException(
            status_code=503,
            detail=(
                "Dashboard data unavailable: "
                + filename
            )
        )

    with open(path, encoding="utf-8") as f:
        return json.load(f)


@app.get("/api/risk-patterns")
def risk_patterns():
    return read_dashboard("risk_patterns.json")


@app.get("/api/bankruptcy-trends")
def bankruptcy_trends():
    return read_dashboard("bankruptcy_trends.json")


@app.get("/api/bankruptcy-cases")
def bankruptcy_cases():
    return read_dashboard("bankruptcy_cases.json")
