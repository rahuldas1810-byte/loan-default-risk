import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from fastapi.testclient import TestClient
from api import app

client = TestClient(app)

GOOD_APPLICANT = {
    "income": 270000, "loan_amount": 600000, "yearly_emi": 30000,
    "goods_price": 500000, "age_years": 28, "years_employed": 2,
    "years_since_id_issued": 3, "years_since_registration_change": 5,
    "days_since_phone_change": 200, "family_members": 3,
    "credit_score_1": 0.4, "credit_score_2": 0.5, "credit_score_3": 0.35,
    "owns_car": False, "higher_education": False,
}


def test_health():
    r = client.get("/health")
    assert r.status_code == 200


def test_predict_returns_valid_answer():
    r = client.post("/predict", json=GOOD_APPLICANT)
    assert r.status_code == 200
    data = r.json()
    assert 0 <= data["risk_score"] <= 100
    assert data["decision"] in ["APPROVE", "REVIEW", "REJECT"]
    assert len(data["reasons"]) == 4


def test_rejects_impossible_age():
    bad = {**GOOD_APPLICANT, "age_years": -5}
    r = client.post("/predict", json=bad)
    assert r.status_code == 422


def test_rejects_credit_score_above_one():
    bad = {**GOOD_APPLICANT, "credit_score_3": 7}
    r = client.post("/predict", json=bad)
    assert r.status_code == 422


def test_blank_optional_fields_are_allowed():
    blanks = {**GOOD_APPLICANT, "years_employed": None,
              "credit_score_1": None, "credit_score_2": None}
    r = client.post("/predict", json=blanks)
    assert r.status_code == 200


def test_higher_credit_score_lowers_risk():
    low = client.post("/predict", json={**GOOD_APPLICANT, "credit_score_3": 0.2}).json()
    high = client.post("/predict", json={**GOOD_APPLICANT, "credit_score_3": 0.9}).json()
    assert high["risk_score"] < low["risk_score"]