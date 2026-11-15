
from pathlib import Path

BASE = Path(__file__).resolve().parent

required = [
    "fundex_pipeline.joblib",
    "fundex_features.joblib",
    "us_model.joblib",
    "us_imputer.joblib",
    "us_features.joblib",
    "poland_model.joblib",
    "poland_imputer.joblib",
    "poland_features.joblib",
    "risk_patterns.json",
    "bankruptcy_trends.json",
    "bankruptcy_cases.json"
]

for filename in required:
    matches = [
        str(path.relative_to(BASE))
        for path in BASE.rglob(filename)
        if path.is_file()
    ]
    print(filename, ":", matches or "MISSING")
