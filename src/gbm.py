"""Gradient boosting with Poisson loss for claim frequency: features, parameters, fitting.

Target = frequency (ClaimNb / Exposure) with Exposure as sample weight (equivalent to counts with an exposure offset).
Predictions are EXPECTED CLAIMS (frequency x exposure), the convention of src.evaluation.cross_validate.

v1: no Exposure feature -> calibrated overall, but about 20% too high for near-full-year policies (age 1+).
v2: Exposure is a feature -> calibrated in every exposure band. Annual rates are predicted at Exposure = 1 and
    mean "claim rate of policies observed for a full year" (may understate risk for new policies if short
    records follow claims; cancellation reasons are not in the data).
"""
import lightgbm as lgb
import pandas as pd

from src.features import LADDER

NUM_RAW = ["VehPower", "VehAge", "DrivAge", "BonusMalus", "Density"]
CAT_RAW = ["VehBrand", "VehGas", "Region"]
FEATURES_V1 = NUM_RAW + CAT_RAW + ["bm_off_ladder"]      # no Area (binned Density), no IDs, no targets
FEATURES_V2 = FEATURES_V1 + ["Exposure"]
FEATURES = FEATURES_V1                                   # kept so existing notebooks keep working

PARAMS = dict(objective="poisson", learning_rate=0.05, n_estimators=700, num_leaves=16, min_child_samples=500,
              subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0, random_state=42, verbose=-1)


def make_X(d, levels, features=FEATURES_V1):
    """Model inputs. `levels` = category level names of the FITTING part (names only, no target information)."""
    X = d[NUM_RAW + CAT_RAW].copy()
    X["bm_off_ladder"] = (~d["BonusMalus"].isin(LADDER)).astype(int)
    if "Exposure" in features:
        X["Exposure"] = d["Exposure"]
    for c in CAT_RAW:
        X[c] = pd.Categorical(X[c], categories=levels[c])
    return X[features]


def fit(tr, params=PARAMS, features=FEATURES_V1):
    """Fit on a training part only. Returns (model, levels)."""
    levels = {c: sorted(tr[c].unique()) for c in CAT_RAW}
    model = lgb.LGBMRegressor(**params)
    model.fit(make_X(tr, levels, features), tr["ClaimNb"] / tr["Exposure"], sample_weight=tr["Exposure"])
    return model, levels


def fit_predict(tr, va, params=PARAMS, features=FEATURES_V1):
    """Expected claims for `va`, from a model fitted on `tr` only (the signature cross_validate expects)."""
    model, levels = fit(tr, params, features)
    return model.predict(make_X(va, levels, features)) * va["Exposure"].to_numpy()


def fit_predict_v2(tr, va, params=PARAMS):
    return fit_predict(tr, va, params, FEATURES_V2)


def predict_annual_rate(model, levels, d, features=FEATURES_V2):
    """Claims per policy-year for a policy observed for a full year (Exposure forced to 1): the quoting setting."""
    return model.predict(make_X(d.assign(Exposure=1.0), levels, features))