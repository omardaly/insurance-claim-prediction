"""Feature engineering for the claims models.

add_features: row-wise features with fixed rules (nothing is learned from the data).
make_preprocessor: the parts that learn from data (scaling, one-hot levels, rare-level merging).
It must always be fitted on a training part only (inside cross-validation, never on all data).
"""
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

DRIVAGE_EDGES = [17, 21, 25, 30, 40, 50, 60, 70, 100]            # right-closed: (17, 21] = ages 18-21
DRIVAGE_LABELS = ["18-21", "22-25", "26-30", "31-40", "41-50", "51-60", "61-70", "71-100"]
VEHAGE_EDGES = [-1, 0, 1, 2, 5, 10, 15, 100]
VEHAGE_LABELS = ["00", "01", "02", "03-05", "06-10", "11-15", "16+"]


def ladder(start=100, floor=50):
    """Claim-free-year ladder of the bonus-malus scale: -5% per year, truncated, never below the floor."""
    vals, v = [start], start
    while v > floor:
        v = max(floor, v * 95 // 100)          # integer arithmetic avoids floating-point surprises
        vals.append(v)
    return vals


LADDER = ladder()
assert LADDER == [100, 95, 90, 85, 80, 76, 72, 68, 64, 60, 57, 54, 51, 50]

NUM = ["logBonusMalus", "logDensity"]
FLAGS = ["bm_is_50", "bm_is_100", "bm_off_ladder"]
CAT = ["DrivAgeBand", "VehAgeBand", "VehGas", "VehBrand", "Region", "VehPower"]


def _band(s, edges, labels):
    b = pd.cut(s, edges, labels=labels)
    assert b.notna().all(), f"{s.name}: values outside the band edges"      # fail loudly, never create 'nan' levels
    return b.astype(str)


def add_features(d):
    """Model inputs from the cleaned policy table. Never includes Area, IDs, targets or Exposure."""
    bm = d["BonusMalus"]
    return pd.DataFrame({
        "DrivAgeBand":   _band(d["DrivAge"], DRIVAGE_EDGES, DRIVAGE_LABELS),
        "VehAgeBand":    _band(d["VehAge"], VEHAGE_EDGES, VEHAGE_LABELS),
        "logBonusMalus": np.log(bm),
        "bm_is_50":      (bm == 50).astype(int),                  # maximum bonus
        "bm_is_100":     (bm == 100).astype(int),                 # neutral starting value
        "bm_off_ladder": (~bm.isin(LADDER)).astype(int),          # not a claim-free rung: likely a recent claim
        "logDensity":    np.log(d["Density"]),
        "VehGas":        d["VehGas"],
        "VehBrand":      d["VehBrand"],
        "Region":        d["Region"],
        "VehPower":      d["VehPower"].astype(str),               # treated as a category
    }, index=d.index)[NUM + FLAGS + CAT]


def make_preprocessor(min_frequency=0.0037):
    """min_frequency as a fraction of the training rows (0.0037 is about 2000 of 541k policies),
    so the same relative rule applies to every fold and to the final model."""
    return ColumnTransformer([
        ("num",  StandardScaler(), NUM),
        ("flag", "passthrough", FLAGS),
        ("cat",  OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=min_frequency), CAT),
    ])