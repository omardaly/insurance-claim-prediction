"""Scoring service for the frequency model (GBM v2): validation, annual rate, risk tier, explanation.

Annual rate = expected claims per policy-year at Exposure = 1, i.e. the claim rate of policies observed for a
full year. Frequency only: there is no severity model, so nothing here is a premium.
"""
from pathlib import Path
import joblib
import numpy as np
import pandas as pd

from src import gbm

MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "gbm_v2.joblib"
NUM_INPUTS = ["VehPower", "VehAge", "DrivAge", "BonusMalus", "Density"]
CAT_INPUTS = ["VehBrand", "VehGas", "Region"]
INPUTS = NUM_INPUTS + CAT_INPUTS


def load(path=MODEL_PATH):
    b = joblib.load(path)
    for key in ("model", "levels", "features", "reference_annual_rate", "tier_edges", "ranges", "portfolio_rate"):
        assert key in b, f"bundle is missing '{key}': run the metadata cell first"
    return b


def check_inputs(b, d):
    """Vectorised validation. Returns (errors, warnings): one list of messages per row.
    Errors: the model cannot score the row. Warnings: scored, but treat with caution."""
    missing = [c for c in INPUTS if c not in d.columns]
    if missing:
        raise ValueError(f"missing columns: {missing}")
    n = len(d)
    errors, warnings = [[] for _ in range(n)], [[] for _ in range(n)]

    def flag(store, mask, msg):
        for i in np.flatnonzero(np.asarray(mask)):
            store[i].append(msg)

    for c in CAT_INPUTS:
        s = d[c]
        empty = s.isna()
        unseen = ~empty & ~s.astype(str).str.strip().isin(b["levels"][c])
        flag(errors, empty, f"{c} is empty")
        flag(errors, unseen, f"{c} was not seen in training")

    for c in NUM_INPUTS:
        x = pd.to_numeric(d[c], errors="coerce")
        flag(errors, x.isna(), f"{c} is empty or not a number")
        flag(errors, x.notna() & (x != np.floor(x)), f"{c} must be a whole number")
        lo, hi = b["ranges"][c]
        # outside the training range a tree model just returns its boundary value, so we refuse to score
        flag(errors, x.notna() & ((x < lo) | (x > hi)), f"{c} is outside the training range [{lo}, {hi}]")

    veh = pd.to_numeric(d["VehAge"], errors="coerce")
    flag(warnings, veh == 0, "Brand-new vehicle: calibration for this group is less certain (see the model card)")
    flag(warnings, veh > 25, "Vehicle older than 25 years: very few training policies, treat the rate with caution")

    if "Exposure" in d.columns:
        e = pd.to_numeric(d["Exposure"], errors="coerce")
        flag(warnings, e.isna() | (e <= 0) | (e > 1), "Exposure outside (0, 1]: no expected-claims figure for this row")
    return errors, warnings


def score(b, d):
    """Annual rate (Exposure = 1) for every valid row. If the file has an Exposure column, also the expected
    claims at the policy's own exposure. Rows with errors get no numbers."""
    d = d.reset_index(drop=True).copy()
    errors, warnings = check_inputs(b, d)
    ok = np.array([len(e) == 0 for e in errors])
    out = pd.DataFrame({"annual_rate": np.nan, "relativity": np.nan, "risk_tier": np.nan,
                        "expected_claims": np.nan}, index=d.index)
    if ok.any():
        sub = d.loc[ok, INPUTS].copy()
        for c in NUM_INPUTS:
            sub[c] = pd.to_numeric(sub[c]).astype(int)
        for c in CAT_INPUTS:
            sub[c] = sub[c].astype(str).str.strip()
        rate = b["model"].predict(gbm.make_X(sub.assign(Exposure=1.0), b["levels"], b["features"]))
        out.loc[ok, "annual_rate"] = rate
        out.loc[ok, "relativity"] = rate / b["reference_annual_rate"]
        out.loc[ok, "risk_tier"] = np.searchsorted(b["tier_edges"], rate, side="right") + 1

        if "Exposure" in d.columns:
            e = pd.to_numeric(d.loc[ok, "Exposure"], errors="coerce").to_numpy()
            valid = (e > 0) & (e <= 1)                       # NaN compares False, so it is excluded
            if valid.any():
                s2 = sub.loc[valid].assign(Exposure=e[valid])
                mu = b["model"].predict(gbm.make_X(s2, b["levels"], b["features"])) * e[valid]
                col = np.full(int(ok.sum()), np.nan)
                col[valid] = mu
                out.loc[ok, "expected_claims"] = col
    out["errors"] = ["; ".join(e) for e in errors]
    out["warnings"] = ["; ".join(w) for w in warnings]
    return out


def poisson_deviance(y, mu):
    y, mu = np.asarray(y, float), np.asarray(mu, float)
    t = np.zeros_like(y)
    p = y > 0
    t[p] = y[p] * np.log(y[p] / mu[p])
    return 2 * np.mean(t - (y - mu))


def evaluate(b, out):
    """If the scored file has ClaimNb and Exposure: compare the model with what actually happened."""
    m = out["expected_claims"].notna() & pd.to_numeric(out["ClaimNb"], errors="coerce").notna()
    if not m.any():
        return None
    y = pd.to_numeric(out.loc[m, "ClaimNb"]).to_numpy(float)
    expo = pd.to_numeric(out.loc[m, "Exposure"]).to_numpy(float)
    mu = out.loc[m, "expected_claims"].to_numpy(float)
    base = b["portfolio_rate"] * expo                       # same rate for everyone, scaled by exposure
    band = pd.cut(expo, [0, 0.1, 0.25, 0.5, 0.75, 1.0])
    cal = (pd.DataFrame({"band": band, "actual": y, "predicted": mu}).groupby("band", observed=True)
           .agg(policies=("actual", "size"), actual=("actual", "sum"), predicted=("predicted", "sum")))
    cal["actual / predicted"] = (cal["actual"] / cal["predicted"]).round(3)
    return {"rows": int(m.sum()), "deviance_model": poisson_deviance(y, mu),
            "deviance_baseline": poisson_deviance(y, base), "actual_over_predicted": y.sum() / mu.sum(),
            "calibration": cal}


def explain(b, policy, k=4):
    """Top-k drivers for one policy. base_level x all factors = annual rate.
    base_level already includes the full-year exposure effect; 'other' is the product of the factors not shown.
    These describe how the prediction is composed, not what would happen if a characteristic changed."""
    d = pd.DataFrame([policy])
    errors, _ = check_inputs(b, d)
    if errors[0]:
        raise ValueError("; ".join(errors[0]))
    for c in NUM_INPUTS:
        d[c] = pd.to_numeric(d[c]).astype(int)
    for c in CAT_INPUTS:
        d[c] = d[c].astype(str).str.strip()
    X = gbm.make_X(d.assign(Exposure=1.0), b["levels"], b["features"])
    contrib = b["model"].predict(X, pred_contrib=True)[0]
    rate = float(b["model"].predict(X)[0])
    by = dict(zip(X.columns, contrib[:-1]))
    base = float(np.exp(contrib[-1] + by.pop("Exposure")))        # base level at Exposure = 1
    factors = [(n, X.iloc[0][n], float(np.exp(v))) for n, v in by.items()]
    factors.sort(key=lambda t: -abs(np.log(t[2])))
    shown, rest = factors[:k], factors[k:]
    other = float(np.prod([f for _, _, f in rest])) if rest else 1.0
    assert np.isclose(base * np.prod([f for _, _, f in factors]), rate, rtol=1e-6)
    return {"base_level": base, "drivers": shown, "other_factor": other, "annual_rate": rate}