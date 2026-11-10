"""Poisson GLM (log link) for claim frequency.

Target = frequency (ClaimNb / Exposure) with Exposure as sample weight, which is equivalent to modelling claim counts
with an exposure offset. Predictions are returned as EXPECTED CLAIMS (frequency x exposure), the convention of
src.evaluation.cross_validate.

v1 = the shared features only (extras=()).
v2 = v1 + four interaction terms for NEW vehicles (age 0): brand B12, petrol, off the bonus-malus ladder, bonus-malus 100.
"""
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import PoissonRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.features import add_features, make_preprocessor, NUM, FLAGS, CAT

ALPHA = 1e-4          # ridge penalty (tuned in the GLM notebook: nothing beat it)

# Interaction terms: 0/1 flags that are on only for new vehicles in a given subgroup, added on top of the main effects.
_INTERACTIONS = {
    "age0_B12":        lambda d, X: ((d["VehAge"] == 0) & (d["VehBrand"] == "B12")).astype(int),
    "age0_Regular":    lambda d, X: ((d["VehAge"] == 0) & (d["VehGas"] == "Regular")).astype(int),
    "age0_off_ladder": lambda d, X: ((d["VehAge"] == 0) & (X["bm_off_ladder"] == 1)).astype(int),
    "age0_bm100":      lambda d, X: ((d["VehAge"] == 0) & (X["bm_is_100"] == 1)).astype(int),
}
V2_EXTRAS = ["age0_B12", "age0_Regular", "age0_off_ladder", "age0_bm100"]


def features(d, extras=()):
    """Shared row-wise features + the requested interaction flags."""
    X = add_features(d)
    for name in extras:
        X[name] = _INTERACTIONS[name](d, X)
    return X


def make_glm(alpha=ALPHA, extras=()):
    if extras:
        pre = ColumnTransformer([
            ("num",  StandardScaler(), NUM),
            ("flag", "passthrough", FLAGS + list(extras)),          # 0/1 flags pass through unscaled
            # same rare-level rule as src.features.make_preprocessor (keep the two in sync)
            ("cat",  OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=0.0037), CAT),
        ])
    else:
        pre = make_preprocessor()                                    # exactly the v1 preprocessing
    return Pipeline([("pre", pre), ("glm", PoissonRegressor(alpha=alpha, max_iter=1000))])


def fit(tr, alpha=ALPHA, extras=()):
    """Fit on a training part only."""
    model = make_glm(alpha, extras)
    model.fit(features(tr, extras), tr["ClaimNb"] / tr["Exposure"], glm__sample_weight=tr["Exposure"])
    return model


def fit_predict(tr, va, alpha=ALPHA, extras=()):
    """Expected claims for `va`, from a model fitted on `tr` only (the signature cross_validate expects)."""
    return fit(tr, alpha, extras).predict(features(va, extras)) * va["Exposure"].to_numpy()


def fit_predict_v2(tr, va, alpha=ALPHA):
    """GLM v2: the model compared with boosting."""
    return fit_predict(tr, va, alpha, V2_EXTRAS)