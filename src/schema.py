"""Input contract for scoring: what the expert's CSV must contain and which values are acceptable.

Hard problems (errors) make a row NOT scoreable. Soft problems (warnings) are reported but the row is still scored,
because the model can handle them (for example a vehicle brand never seen in training, or a rare value).
"""
from dataclasses import dataclass
from io import StringIO

import numpy as np
import pandas as pd

REQUIRED_NUMERIC = ["VehPower", "VehAge", "DrivAge", "BonusMalus", "Density"]
REQUIRED_CATEGORICAL = ["VehBrand", "VehGas", "Region"]
REQUIRED = REQUIRED_NUMERIC + REQUIRED_CATEGORICAL
OPTIONAL = ["IDpol", "Exposure"]          # IDpol defaults to the row number, Exposure to 1.0 (an annual contract)
IGNORED = ["Area", "ClaimNb"]             # present in freMTPL2freq.csv; not used for scoring (Area is binned Density;
                                          # ClaimNb is the OUTCOME: if present it can be used later to compare with predictions)

COLUMN_HELP = {
    "IDpol": "Policy identifier (optional). Default: the row number.",
    "Exposure": "Fraction of a year the policy was or will be in force (optional, default 1.0).",
    "VehPower": "Vehicle power, ordinal integer (4 to 15 in the training data).",
    "VehAge": "Vehicle age in years, integer part (0 = new vehicle).",
    "DrivAge": "Driver age in years, integer part (18 or more).",
    "BonusMalus": "Bonus-malus coefficient, 50 to 350 (below 100 = bonus, above 100 = malus).",
    "VehBrand": "Anonymised vehicle brand code (B1, B2, ..., B14).",
    "VehGas": "Fuel type: Diesel or Regular.",
    "Density": "Inhabitants per km2 where the driver lives.",
    "Region": "French region (1982-2015 regions), for example Ile-de-France, Bretagne.",
    "Area": "Not used (it is a binned version of Density).",
    "ClaimNb": "Not used for scoring (it is the outcome). Kept if present, for comparison with the predictions.",
}

# two real rows of freMTPL2freq.csv (a public, anonymised dataset), as an example for the expert
TEMPLATE_ROWS = [
    {"IDpol": 1, "Exposure": 0.10, "VehPower": 5, "VehAge": 0, "DrivAge": 55, "BonusMalus": 50, "VehBrand": "B12",
     "VehGas": "Regular", "Density": 1217, "Region": "Rhone-Alpes"},
    {"IDpol": 5, "Exposure": 0.75, "VehPower": 6, "VehAge": 2, "DrivAge": 52, "BonusMalus": 50, "VehBrand": "B12",
     "VehGas": "Diesel", "Density": 54, "Region": "Picardie"},
]


def template_csv():
    """A ready-to-fill CSV (header + two example rows) to give to the expert."""
    cols = ["IDpol", "Exposure"] + REQUIRED
    return pd.DataFrame(TEMPLATE_ROWS)[cols].to_csv(index=False)


# Category levels seen in training (names only)
KNOWN = {
    "VehBrand": ["B1", "B10", "B11", "B12", "B13", "B14", "B2", "B3", "B4", "B5", "B6"],
    "VehGas": ["Diesel", "Regular"],
    "Region": ["Alsace", "Aquitaine", "Auvergne", "Basse-Normandie", "Bourgogne", "Bretagne", "Centre",
               "Champagne-Ardenne", "Corse", "Franche-Comte", "Haute-Normandie", "Ile-de-France",
               "Languedoc-Roussillon", "Limousin", "Midi-Pyrenees", "Nord-Pas-de-Calais", "Pays-de-la-Loire",
               "Picardie", "Poitou-Charentes", "Provence-Alpes-Cotes-D'Azur", "Rhone-Alpes"],
}
# Hard limits (outside = error) and rare-in-training limits (outside = warning)
HARD = {"DrivAge": (18, 100), "VehAge": (0, 100), "BonusMalus": (50, 350), "Density": (1, None), "VehPower": (1, None)}
SOFT = {"VehPower": (4, 15), "VehAge": (0, 30), "Density": (1, 27000)}


class SchemaError(Exception):
    """The file as a whole cannot be used (for example a required column is missing)."""


@dataclass
class ValidationResult:
    data: pd.DataFrame          # typed copy with row_id, IDpol, Exposure and a boolean `valid` column
    issues: pd.DataFrame        # one row per problem: row_id, column, severity, message

    def summary(self):
        n = len(self.data)
        n_ok = int(self.data["valid"].sum())
        n_warn_rows = self.issues.loc[self.issues["severity"] == "warning", "row_id"].nunique()
        return {"rows": n, "scoreable": n_ok, "rejected": n - n_ok, "rows_with_warnings": n_warn_rows}


def read_expert_csv(file_or_path):
    """Read an expert-supplied CSV: UTF-8 or Latin-1, delimiter ',' ';' or tab. Everything is read as text on purpose:
    validate() does the type conversion and reports what it could not convert (decimal comma included)."""
    if hasattr(file_or_path, "read"):
        raw = file_or_path.read()
    else:
        with open(file_or_path, "rb") as f:
            raw = f.read()
    if isinstance(raw, str):
        text = raw
    else:
        for enc in ("utf-8-sig", "latin-1"):
            try:
                text = raw.decode(enc)
                break
            except UnicodeDecodeError:
                continue
    lines = text.splitlines()
    if not lines:
        raise SchemaError("the file is empty")
    delimiter = max([",", ";", "\t"], key=lines[0].count)
    df = pd.read_csv(StringIO(text), sep=delimiter, dtype=str, keep_default_na=False)
    df.columns = [c.strip() for c in df.columns]
    return df


def validate(raw):
    """Check and type a raw table. Returns a ValidationResult; raises SchemaError if a required column is missing."""
    missing_cols = [c for c in REQUIRED if c not in raw.columns]
    if missing_cols:
        raise SchemaError(f"missing required column(s): {missing_cols}")

    df = raw.copy()
    df["row_id"] = np.arange(1, len(df) + 1)                    # 1 = first data row of the file
    if "IDpol" not in df.columns:
        df["IDpol"] = df["row_id"]
    if "Exposure" not in df.columns:
        df["Exposure"] = 1.0
    ids = pd.to_numeric(df["IDpol"].astype(str).str.strip(), errors="coerce")
    if ids.notna().all() and (ids % 1 == 0).all():
        df["IDpol"] = ids.astype("int64")                          # freMTPL2freq.csv stores ids as floats ("1.0")
    issues = []

    def add(mask, column, severity, message):
        mask = np.asarray(mask, dtype=bool)
        if mask.any():
            issues.append(pd.DataFrame({"row_id": df.loc[mask, "row_id"].to_numpy(), "column": column,
                                        "severity": severity, "message": message}))

    add(df["IDpol"].duplicated(keep=False).to_numpy(), "IDpol", "warning", "duplicate identifier")

    # numeric columns: convert (decimal comma accepted), report missing and non-numeric values
    for c in REQUIRED_NUMERIC + ["Exposure"]:
        s = df[c]
        text = s.astype(str).str.strip()
        missing = (s.isna() | text.eq("") | text.str.lower().isin(["nan", "none", "null"])).to_numpy()
        num = pd.to_numeric(text.str.replace(",", ".", regex=False), errors="coerce")
        add(missing, c, "error", "missing value")
        add(num.isna().to_numpy() & ~missing, c, "error", "not a number")
        df[c] = num

    for c, (lo, hi) in HARD.items():
        if lo is not None:
            add((df[c] < lo).to_numpy(), c, "error", f"below the allowed minimum {lo}")
        if hi is not None:
            add((df[c] > hi).to_numpy(), c, "error", f"above the allowed maximum {hi}")
    add((df["Exposure"] <= 0).to_numpy(), "Exposure", "error", "exposure must be positive")
    add((df["Exposure"] > 1).to_numpy(), "Exposure", "warning", "exposure above 1 year (training policies were at most 1)")
    for c, (lo, hi) in SOFT.items():
        add(((df[c] < lo) | (df[c] > hi)).to_numpy(), c, "warning", f"outside the range common in training data ({lo} to {hi})")
    for c in ["VehPower", "VehAge", "DrivAge"]:
        add((df[c].notna() & (df[c] % 1 != 0)).to_numpy(), c, "warning", "not an integer")

    # categorical columns: normalise case and spaces, then check against the levels seen in training
    for c in REQUIRED_CATEGORICAL:
        text = df[c].astype(str).str.strip()
        missing = (df[c].isna() | text.eq("")).to_numpy()
        canon = {k.lower(): k for k in KNOWN[c]}
        mapped = text.str.lower().map(canon)
        unknown = mapped.isna().to_numpy() & ~missing
        add(missing, c, "error", "missing value")
        if c == "VehGas":
            add(unknown, c, "error", f"unknown fuel type (expected one of {KNOWN[c]})")
        else:
            add(unknown, c, "warning", "level never seen in training: the model treats it as unknown")
        df[c] = mapped.where(mapped.notna(), text)

    issues = pd.concat(issues, ignore_index=True) if issues else pd.DataFrame(columns=["row_id", "column", "severity", "message"])
    bad_rows = set(issues.loc[issues["severity"] == "error", "row_id"])
    df["valid"] = ~df["row_id"].isin(bad_rows)
    return ValidationResult(data=df, issues=issues)