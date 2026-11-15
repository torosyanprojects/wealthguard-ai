
import json
import joblib
import numpy as np
import pandas as pd

from pathlib import Path
from functools import lru_cache
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Dict, Optional

from wealthguard_engine import analyze_company, analyze_finances

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
        raise FileNotFoundError(f"Missing asset: {filename}")
    return path

# Preserve the original model and its configuration.
legacy_model = joblib.load(required_asset("fundex_pipeline.joblib"))

with open(required_asset("feature_names.json")) as f:
    legacy_features = joblib.load(required_asset("fundex_features.joblib"))

with open(required_asset("model_config.json")) as f:
    config = json.load(f)

# Load the OpenFundEx pipeline and features.
fundex_model = joblib.load(required_asset("fundex_pipeline.joblib"))
fundex_features = joblib.load(required_asset("fundex_features.joblib"))

# Load additional models.
@lru_cache(maxsize=2)
def load_regional_model(region):
    if region not in ("us", "poland"):
        raise ValueError("Unsupported model")

    model = joblib.load(required_asset(f"{region}_model.joblib"))
    imputer = joblib.load(required_asset(f"{region}_imputer.joblib"))
    features = joblib.load(required_asset(f"{region}_features.joblib"))

    return model, imputer, features

app = FastAPI(
    title="WealthGuard AI API",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://wealthguard-frontend-iota.vercel.app"
    ],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"]
)

class FinancialRequest(BaseModel):
    financial_data: Dict[str, Optional[float]] = Field(...)

def clean_values(data, features):
    row = pd.DataFrame([{
        name: data.get(name, np.nan)
        for name in features
    }])
    return row.apply(
        pd.to_numeric, errors="coerce"
    ).replace([np.inf, -np.inf], np.nan)

@app.get("/health")
def health():
    return {
        "status": "healthy",
        "project": "WealthGuard AI",
        "version": "2.0.0"
    }

# Keep the original endpoint compatible with the frontend.
@app.post("/analyze")
def analyze(request: FinancialRequest):
    data = request.financial_data

    allowed = set(legacy_features) | {"previous_revenue"}
    unknown = sorted(set(data) - allowed)

    if unknown:
        raise HTTPException(
            status_code=422,
            detail={"unsupported_indicators": unknown}
        )

    row = clean_values(data, legacy_features)
    coverage = float(row.notna().mean(axis=1).iloc[0])

    if coverage < 0.50:
        return {
            "project": "WealthGuard AI",
            "model_score": None,
            "prediction_status": "insufficient_data",
            "feature_coverage": round(coverage, 4),
            **analyze_finances(data),
            "message": "More financial data is needed to generate a model score."
        }

    report = analyze_company(
        data,
        legacy_model,
        legacy_features,
        float(config["warning_threshold"])
    )

    report["prediction_status"] = "provisional"
    report["feature_coverage"] = round(coverage, 4)
    return report

@app.post("/analyze/fundex")
def analyze_fundex(request: FinancialRequest):
    data = request.financial_data

    allowed = set(fundex_features) | {
        "previous_revenue", "operating_cash_flow"
    }
    unknown = sorted(set(data) - allowed)

    if unknown:
        raise HTTPException(
            status_code=422,
            detail={"unsupported_indicators": unknown}
        )

    row = clean_values(data, fundex_features)
    coverage = float(row.notna().mean(axis=1).iloc[0])

    if coverage < 0.50:
        return {
            "project": "WealthGuard AI",
            "model_score": None,
            "prediction_status": "insufficient_data",
            "feature_coverage": round(coverage, 4),
            **analyze_finances(data)
        }

    score = float(fundex_model.predict_proba(row)[0, 1])

    return {
        "project": "WealthGuard AI",
        "model": "OpenFundEx",
        "model_score": round(score, 4),
        "score_type": "Uncalibrated model output",
        "prediction_status": "provisional",
        "feature_coverage": round(coverage, 4),
        **analyze_finances(data),
        "disclaimer": (
            "Experimental financial screening only. "
            "The score is not a calibrated bankruptcy probability."
        )
    }

@app.post("/analyze/{region}")
def analyze_region(region: str, request: FinancialRequest):
    if region not in ("us", "poland"):
        raise HTTPException(status_code=404, detail="Unknown model")

    model, imputer, features = load_regional_model(region)
    data = request.financial_data

    unknown = sorted(set(data) - set(features))
    if unknown:
        raise HTTPException(
            status_code=422,
            detail={"unsupported_indicators": unknown}
        )

    row = clean_values(data, features)
    coverage = float(row.notna().mean(axis=1).iloc[0])

    if coverage < 0.50:
        return {
            "model": region,
            "model_score": None,
            "prediction_status": "insufficient_data",
            "feature_coverage": round(coverage, 4)
        }

    processed = imputer.transform(row)
    score = float(model.predict_proba(processed)[0, 1])

    return {
        "model": region,
        "model_score": round(score, 4),
        "score_type": "Uncalibrated model output",
        "prediction_status": "provisional",
        "feature_coverage": round(coverage, 4),
        "disclaimer": (
            "Experimental model output. "
            "Not a calibrated bankruptcy probability."
        )
    }

def read_dashboard(filename):
    path = find_asset(filename, ("", "data"))
    if path is None:
        raise HTTPException(
            status_code=503,
            detail=f"Dashboard data unavailable: {filename}"
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
