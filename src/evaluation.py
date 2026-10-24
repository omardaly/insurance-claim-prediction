"""Shared helpers: training data with folds, Poisson deviance, grouped CV, calibration."""
from pathlib import Path

import numpy as np
import pandas as pd

PROCESSED = Path(__file__).resolve().parents[1] / "data" / "processed"
SEED = 42
N_FOLDS = 5
EXPOSURE_EDGES = [0, 0.1, 0.25, 0.5, 0.75, 1.0]
FEATURES = ["VehPower", "VehAge", "DrivAge", "BonusMalus", "VehBrand", "VehGas", "Area", "Density", "Region"]


def _load_base_train():
    """Training policies (from the saved split) with their feature-profile group."""
    df = pd.read_csv(PROCESSED / "policies_clean.csv")
    ids = pd.read_csv(PROCESSED / "train_ids.csv")["IDpol"]
    groups = pd.read_csv(PROCESSED / "policy_groups.csv")
    train = df[df["IDpol"].isin(ids)].merge(groups, on="IDpol", how="left").reset_index(drop=True)
    assert len(train) == len(ids) and train["Group"].notna().all()
    return train


def make_folds(seed=SEED, n_folds=N_FOLDS):
    """Assign every whole feature-profile group to one fold. Returns a DataFrame [IDpol, Fold]."""
    train = _load_base_train()
    rng = np.random.default_rng(seed)
    ug = train["Group"].unique()
    fold_of_group = pd.Series(rng.integers(0, n_folds, len(ug)), index=ug)
    folds = pd.DataFrame({"IDpol": train["IDpol"], "Fold": train["Group"].map(fold_of_group)})

    # leak check: no feature profile may span several folds
    profile = train.groupby(FEATURES, sort=False).ngroup()
    assert (folds["Fold"].groupby(profile).nunique() == 1).all(), "a feature profile spans several folds"
    return folds


def load_train():
    """Training policies with the frozen Fold column (read from train_folds.csv)."""
    folds = pd.read_csv(PROCESSED / "train_folds.csv")
    train = _load_base_train().merge(folds, on="IDpol", how="left")
    assert train["Fold"].notna().all(), "some training policies have no fold"
    train["Fold"] = train["Fold"].astype(int)
    return train


def poisson_deviance(y, mu):
    """Mean of 2*[y*log(y/mu) - (y - mu)]; the log term is 0 when y = 0."""
    y, mu = np.asarray(y, dtype=float), np.asarray(mu, dtype=float)
    term = np.zeros_like(y)
    pos = y > 0
    term[pos] = y[pos] * np.log(y[pos] / mu[pos])
    return 2 * np.mean(term - (y - mu))

assert np.isclose(poisson_deviance([0], [0.5]), 1.0)     # 2 * (0 - (0 - 0.5)) = 1
assert np.isclose(poisson_deviance([2], [2]), 0.0)
assert np.isclose(poisson_deviance([1], [2]), 2 * (np.log(0.5) + 1))     # = 0.6137, tests the log term

def cross_validate(train, fit_predict, check=True):
    """fit_predict(train_part, valid_part) -> expected claim counts for valid_part.
    Returns (out-of-fold predictions, array of per-fold deviances)."""
    ref = None
    if check:   # independent reference implementation, used only if scikit-learn can be imported
        try:
            from sklearn.metrics import mean_poisson_deviance as ref
        except ImportError:
            print("note: sklearn.metrics unavailable on this machine; skipping the scikit-learn cross-check")

    oof, scores = np.zeros(len(train)), []
    for k in range(N_FOLDS):
        va = (train["Fold"] == k).to_numpy()
        oof[va] = np.asarray(fit_predict(train.loc[~va], train.loc[va]), dtype=float)
        y = train.loc[va, "ClaimNb"]
        d = poisson_deviance(y, oof[va])
        if ref is not None:
            assert np.isclose(d, ref(y, oof[va])), f"fold {k}: deviance mismatch"
        scores.append(d)
    return oof, np.array(scores)

def exposure_band_reference(tr, va):
    """Diagnostic yardstick, NOT a model: exposure is unknown when pricing a new policy.
    Rate per exposure band learned on tr, applied to va."""
    k = len(EXPOSURE_EDGES) - 1
    code_tr = pd.cut(tr["Exposure"], EXPOSURE_EDGES, labels=False)
    g = tr.groupby(code_tr)[["ClaimNb", "Exposure"]].sum()
    rate = (g["ClaimNb"] / g["Exposure"]).reindex(range(k)).to_numpy()
    code_va = pd.cut(va["Exposure"], EXPOSURE_EDGES, labels=False).astype(int).to_numpy()
    return va["Exposure"].to_numpy() * rate[code_va]

def calibration_by_exposure(train, pred, edges=EXPOSURE_EDGES):
    """Actual vs predicted claims per exposure band."""
    band = pd.cut(train["Exposure"], edges)
    cal = train.assign(pred=pred).groupby(band, observed=True).agg(
        policies=("IDpol", "size"), actual=("ClaimNb", "sum"), predicted=("pred", "sum"))
    cal["actual/predicted"] = (cal["actual"] / cal["predicted"]).round(3)
    return cal.round({"actual": 1, "predicted": 1})