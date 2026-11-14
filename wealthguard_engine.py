
import numpy as np
import pandas as pd

def _number(data, key):
    try:
        value = float(data.get(key))
        return value if np.isfinite(value) else None
    except (TypeError, ValueError):
        return None

def _ratio(a, b):
    if a is None or b is None or b <= 0:
        return None
    return a / b

def analyze_finances(data):
    current_ratio = _ratio(
        _number(data, "current_assets"),
        _number(data, "current_liabilities")
    )

    liabilities_to_assets = _ratio(
        _number(data, "total_liabilities"),
        _number(data, "total_assets")
    )

    net_profit_margin = _ratio(
        _number(data, "net_income"),
        _number(data, "revenue")
    )

    operating_cash_flow = _number(
        data, "operating_cash_flow"
    )

    revenue = _number(data, "revenue")
    previous_revenue = _number(
        data, "previous_revenue"
    )

    revenue_growth = (
        (revenue - previous_revenue) / previous_revenue
        if revenue is not None
        and previous_revenue is not None
        and previous_revenue > 0
        else None
    )

    indicators = {
        "current_ratio": current_ratio,
        "liabilities_to_assets": liabilities_to_assets,
        "net_profit_margin": net_profit_margin,
        "operating_cash_flow": operating_cash_flow,
        "revenue_growth": revenue_growth
    }

    issues = []

    def add(problem, explanation, actions, monitor):
        issues.append({
            "problem": problem,
            "explanation": explanation,
            "recommended_actions": actions,
            "monitor": monitor
        })

    if current_ratio is not None and current_ratio < 1:
        add(
            "Weak Liquidity",
            "Current liabilities exceed current assets.",
            [
                "Prepare a short-term cash-flow forecast.",
                "Review upcoming payment obligations.",
                "Investigate delayed customer payments."
            ],
            "Current ratio"
        )

    if (
        liabilities_to_assets is not None
        and liabilities_to_assets > 0.70
    ):
        add(
            "High Liabilities Relative to Assets",
            "Total liabilities exceed 70% of total assets.",
            [
                "Review liabilities and repayment schedules.",
                "Assess the ability to meet obligations.",
                "Consider discussing financing options with an adviser."
            ],
            "Liabilities-to-assets ratio"
        )

    if net_profit_margin is not None and net_profit_margin < 0:
        add(
            "Negative Profitability",
            "The company reports a net loss relative to revenue.",
            [
                "Review operating expenses.",
                "Investigate low-margin products or services.",
                "Evaluate pricing and cost structure."
            ],
            "Net profit margin"
        )

    if operating_cash_flow is not None and operating_cash_flow < 0:
        add(
            "Negative Operating Cash Flow",
            "Operating activities are consuming cash.",
            [
                "Prepare a rolling cash-flow forecast.",
                "Review receivables and inventory.",
                "Assess near-term cash requirements."
            ],
            "Operating cash flow"
        )

    if revenue_growth is not None and revenue_growth < -0.10:
        add(
            "Declining Revenue",
            "Revenue has declined by more than 10%.",
            [
                "Investigate changes in customer demand.",
                "Review sales and customer retention.",
                "Update revenue forecasts."
            ],
            "Revenue growth"
        )

    missing = [
        name for name, value in indicators.items()
        if value is None
    ]

    return {
        "financial_indicators": indicators,
        "missing_indicators": missing,
        "issues_identified": len(issues),
        "recommendations": issues
    }

def analyze_company(data, model, feature_names,
                    warning_threshold=0.85):

    if not isinstance(data, dict):
        raise TypeError("Company data must be a dictionary.")

    if not any(
        key in data and _number(data, key) is not None
        for key in feature_names
    ):
        raise ValueError(
            "No valid model financial indicators provided."
        )

    row = pd.DataFrame([data]).reindex(
        columns=feature_names
    )

    row = row.apply(
        pd.to_numeric, errors="coerce"
    ).replace([np.inf, -np.inf], np.nan)

    score = float(model.predict_proba(row)[0, 1])

    financial_report = analyze_finances(data)

    return {
        "project": "WealthGuard AI",
        "model_score": round(score, 4),
        "score_type": "Uncalibrated model output",
        "warning_threshold": warning_threshold,
        "warning_triggered": bool(
            score >= warning_threshold
        ),
        **financial_report,
        "disclaimer": (
            "Educational financial screening only. "
            "Model score is not a calibrated bankruptcy "
            "probability. Recommendations are preliminary."
        )
    }
