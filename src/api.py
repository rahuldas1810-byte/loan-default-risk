from typing import Optional

import pandas as pd
from pydantic import BaseModel, Field

from features import add_features
import joblib
import shap
from fastapi import FastAPI

# Open the lunchbox 🍱
bundle = joblib.load("../models/loan_model.joblib")
model = bundle["model"]
FORM_COLS = bundle["form_cols"]
MEDIANS = bundle["medians"]
THRESHOLD = bundle["threshold"]
REVIEW_THRESHOLD = bundle["review_threshold"]
explainer = shap.TreeExplainer(model)
FRIENDLY = {
    "EXT_SOURCE_1": "Credit score 1", "EXT_SOURCE_2": "Credit score 2",
    "EXT_SOURCE_3": "Credit score 3", "EXT_MEAN": "Average credit score",
    "EXT_MIN": "Lowest credit score", "CREDIT_TERM": "Loan repayment speed",
    "GOODS_CREDIT_RATIO": "Item price vs. loan amount",
    "ANNUITY_INCOME_RATIO": "EMI as a share of income",
    "EMPLOYED_AGE_RATIO": "Job stability", "INCOME_PER_PERSON": "Income per family member",
    "AMT_INCOME_TOTAL": "Yearly income", "AMT_CREDIT": "Loan amount",
    "AMT_ANNUITY": "Yearly EMI", "AMT_GOODS_PRICE": "Item price",
    "DAYS_BIRTH": "Age", "DAYS_EMPLOYED": "Years at current job",
    "DAYS_ID_PUBLISH": "Time since ID was issued",
    "DAYS_REGISTRATION": "Time since registration change",
    "DAYS_LAST_PHONE_CHANGE": "Time since phone number change",
    "CNT_FAM_MEMBERS": "Family size", "FLAG_OWN_CAR": "Car ownership",
    "NAME_EDUCATION_TYPE_Higher_education": "Higher education",
}
app = FastAPI(title="Loan Default Risk API")

@app.get("/")
def home():
    return {
        "status": "API is running ✅",
        "threshold": THRESHOLD,
        "review_threshold": REVIEW_THRESHOLD,
        "inputs": FORM_COLS,
    }
class Applicant(BaseModel):
    income: float = Field(gt=0)
    loan_amount: float = Field(gt=0)
    yearly_emi: float = Field(gt=0)
    goods_price: float = Field(gt=0)
    age_years: float = Field(ge=18, le=100)
    years_employed: Optional[float] = Field(default=None, ge=0)
    years_since_id_issued: float = Field(ge=0)
    years_since_registration_change: float = Field(ge=0)
    days_since_phone_change: float = Field(ge=0)
    family_members: int = Field(ge=1)
    credit_score_1: Optional[float] = Field(default=None, ge=0, le=1)
    credit_score_2: Optional[float] = Field(default=None, ge=0, le=1)
    credit_score_3: Optional[float] = Field(default=None, ge=0, le=1)
    owns_car: bool
    higher_education: bool
@app.post("/predict")
def predict(a: Applicant):
    # 1. Translate human inputs → model format
    row = {
        "AMT_INCOME_TOTAL": a.income,
        "AMT_CREDIT": a.loan_amount,
        "AMT_ANNUITY": a.yearly_emi,
        "AMT_GOODS_PRICE": a.goods_price,
        "DAYS_BIRTH": -a.age_years * 365,
        "DAYS_EMPLOYED": -a.years_employed * 365 if a.years_employed is not None else None,
        "DAYS_ID_PUBLISH": -a.years_since_id_issued * 365,
        "DAYS_REGISTRATION": -a.years_since_registration_change * 365,
        "DAYS_LAST_PHONE_CHANGE": -a.days_since_phone_change,
        "CNT_FAM_MEMBERS": a.family_members,
        "EXT_SOURCE_1": a.credit_score_1,
        "EXT_SOURCE_2": a.credit_score_2,
        "EXT_SOURCE_3": a.credit_score_3,
        "FLAG_OWN_CAR": int(a.owns_car),
        "NAME_EDUCATION_TYPE_Higher_education": int(a.higher_education),
    }

    # 2. Make a 1-row table in the right column order, fill blanks
    X = pd.DataFrame([row])[FORM_COLS].astype(float).fillna(MEDIANS)

    # 3. Add the 7 smart features
    X = add_features(X)

    # 4. Predict and decide
    risk = float(model.predict_proba(X)[0, 1])
    if risk >= THRESHOLD:
        decision = "REJECT"
    elif risk >= REVIEW_THRESHOLD:
        decision = "REVIEW"
    else:
        decision = "APPROVE"

    # 5. Explain with SHAP: top 4 reasons
    impacts = explainer(X).values[0]
    top = sorted(zip(X.columns, impacts), key=lambda t: abs(t[1]), reverse=True)[:4]
    reasons = [
        {"factor": FRIENDLY.get(f, f),
         "effect": "increased risk" if v > 0 else "lowered risk",
         "strength": round(abs(float(v)), 3)}
        for f, v in top
    ]

    return {"risk_score": round(risk * 100, 1), "decision": decision, "reasons": reasons}