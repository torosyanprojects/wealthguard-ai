
import json
import joblib
import numpy as np
import pandas as pd

from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Dict, Optional

from wealthguard_engine import analyze_company, analyze_finances

BASE_DIR = Path(__file__).resolve().parent

model = joblib.load(
    BASE_DIR / "wealthguard_model.joblib"
)

with open(BASE_DIR / "feature_names.json") as f:
    feature_names = json.load(f)

with open(BASE_DIR / "model_config.json") as f:
    config = json.load(f)

app = FastAPI(
    title="WealthGuard AI API",
    version="1.0.0"
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://wealthguard-frontend-iota.vercel.app"
    ],
    allow_credentials=False,
    allow_methods=["POST", "GET"],
    allow_headers=["Content-Type"],
)
class FinancialRequest(BaseModel):
    financial_data: Dict[str, Optional[float]] = Field(...)

@app.get("/health")
def health():
    return {
        "status": "healthy",
        "project": "WealthGuard AI"
    }

@app.post("/analyze")
def analyze(request: FinancialRequest):

    data = request.financial_data

    allowed = set(feature_names) | {"previous_revenue"}
    unknown = sorted(set(data) - allowed)

    if unknown:
        raise HTTPException(
            status_code=422,
            detail={"unsupported_indicators": unknown}
        )

    values = pd.to_numeric(
        pd.Series({
            name: data.get(name, np.nan)
            for name in feature_names
        }),
        errors="coerce"
    ).replace([np.inf, -np.inf], np.nan)

    coverage = float(values.notna().mean())

    if coverage < 0.50:
        return {
            "project": "WealthGuard AI",
            "model_score": None,
            "prediction_status": "insufficient_data",
            "feature_coverage": round(coverage, 4),
            **analyze_finances(data),
            "message": (
                "More financial data is needed "
                "to generate a model score."
            )
        }

    report = analyze_company(
        data,
        model,
        feature_names,
        float(config["warning_threshold"])
    )

    report["prediction_status"] = "provisional"
    report["feature_coverage"] = round(coverage, 4)

    return report
