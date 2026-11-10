import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))
sys.path.append(str(Path(__file__).resolve().parent))

import pandas as pd
import streamlit as st
from src import scoring
from src import auth
import ui 
import auth_ui

st.set_page_config(page_title="Claim frequency model", page_icon="📈", layout="wide")
ui.apply()
user = auth_ui.require_login()          # nothing below runs until someone is signed in


@st.cache_resource
def get_bundle():
    return scoring.load()


b = get_bundle()

ui.show('<div class="brand"><div class="name">Claim frequency model</div>'
        '<div class="tag">Motor insurance · GBM v2</div></div>', sidebar=True)
ICONS = {"Score one policy": "🔎  Score one policy", "Score a CSV": "📄  Score a CSV", "Model card": "📊  Model card"}
if user["role"] == "admin":
    ICONS["Users"] = "👥  Users"
page = st.sidebar.radio("Navigation", list(ICONS), format_func=lambda p: ICONS[p], label_visibility="collapsed")
auth_ui.sidebar_user(user)
st.sidebar.caption(f"Model version: {b['version']}")
st.sidebar.caption("Claim frequency only. This is not a premium.")


# ---------------------------------------------------------------- page 1
if page == "Score one policy":
    ui.show(ui.hero("Score one policy",
                    "Expected claims per year for a policy observed for a full year (Exposure = 1)"))
    lv, rg = b["levels"], b["ranges"]

    with st.form("policy"):
        c1, c2, c3 = st.columns(3)
        veh_age = c1.number_input("Vehicle age (years)", rg["VehAge"][0], rg["VehAge"][1], 5, 1)
        veh_power = c1.number_input("Vehicle power", rg["VehPower"][0], rg["VehPower"][1], 6, 1)
        veh_brand = c1.selectbox("Vehicle brand", lv["VehBrand"])
        driv_age = c2.number_input("Driver age", rg["DrivAge"][0], rg["DrivAge"][1], 40, 1)
        bonus = c2.number_input("Bonus-malus", rg["BonusMalus"][0], rg["BonusMalus"][1], 50, 1)
        veh_gas = c2.selectbox("Fuel", lv["VehGas"])
        density = c3.number_input("Density (inhabitants/km2)", rg["Density"][0], rg["Density"][1], 500, 1)
        region = c3.selectbox("Region", lv["Region"])
        submitted = st.form_submit_button("Score this policy")

    if submitted:
        policy = {"VehPower": veh_power, "VehAge": veh_age, "DrivAge": driv_age, "BonusMalus": bonus,
                  "Density": density, "VehBrand": veh_brand, "VehGas": veh_gas, "Region": region}
        res = scoring.score(b, pd.DataFrame([policy])).iloc[0]

        if res["errors"]:
            st.error(res["errors"])
        else:
            tier = int(res["risk_tier"])
            rate, rel = res["annual_rate"], res["relativity"]
            ui.show(ui.kpi_row([
                ui.kpi("Annual claim rate", f"{rate:.3f}", f"about {rate * 100:.1f} claims per 100 policies per year", "#0B6E99"),
                ui.kpi("Relative to portfolio", f"{rel:.2f}×", f"portfolio average {b['reference_annual_rate']:.3f}", "#14A3A3"),
                ui.kpi("Risk tier", f"{tier} / 10", ui.tier_label(tier), ui.TIER_COLORS[tier - 1]),
            ]))
            ui.show(ui.tier_scale(tier))
            st.caption("A risk level for a group of similar policies, observed for a full year. It is not a statement "
                       "about one customer.")

            for w in [w for w in res["warnings"].split("; ") if w]:
                st.warning(w)

            ui.show(ui.factors_card(scoring.explain(b, policy)))


# ---------------------------------------------------------------- page 2
elif page == "Score a CSV":
    ui.show(ui.hero("Score a CSV file",
                    "One policy per row. Upload a file like freMTPL2freq.csv, or just the model inputs."))
    st.caption("Required columns: " + ", ".join(scoring.INPUTS) +
               ". Other columns (IDpol, Area, ...) are kept and ignored by the model. "
               "If the file also has Exposure and ClaimNb, the app compares the model with the actual claims.")

    template = pd.DataFrame([{"VehPower": 6, "VehAge": 5, "DrivAge": 40, "BonusMalus": 50, "Density": 500,
                              "VehBrand": b["levels"]["VehBrand"][0], "VehGas": b["levels"]["VehGas"][0],
                              "Region": b["levels"]["Region"][0]}])
    st.download_button("Download an example file", template.to_csv(index=False), "example_policies.csv", "text/csv")

    up = st.file_uploader("Upload a CSV (up to 50,000 rows)", type="csv")
    if up is not None:
        d = pd.read_csv(up)
        missing = [c for c in scoring.INPUTS if c not in d.columns]
        if missing:
            st.error(f"Missing columns: {missing}")
        elif len(d) > 50_000:
            st.error("Please upload at most 50,000 rows at a time.")
        else:
            with st.spinner("Scoring..."):
                res = scoring.score(b, d)
            out = pd.concat([d.reset_index(drop=True), res], axis=1)
            if "Exposure" not in d.columns:
                out = out.drop(columns=["expected_claims"])
            n_err = int((res["errors"] != "").sum())
            n_warn = int((res["warnings"] != "").sum())

            ui.show(ui.kpi_row([
                ui.kpi("Rows", f"{len(out):,}", "in the file", "#0B6E99"),
                ui.kpi("Scored", f"{len(out) - n_err:,}", "got an annual rate", "#1B9E77"),
                ui.kpi("Errors", f"{n_err:,}", "no rate, see the errors column", "#C93A2E" if n_err else "#1B9E77"),
                ui.kpi("Warnings", f"{n_warn:,}", "scored, read with care", "#F2A33C" if n_warn else "#1B9E77"),
            ]))
            if n_err:
                st.warning("Rows with errors get no rate. See the 'errors' column.")

            ok = res["annual_rate"].notna()
            tab_sum, tab_eval, tab_tbl = st.tabs(["Summary", "Model vs actual", "Results table"])

            with tab_sum:
                if ok.any():
                    ui.show(ui.kpi_row([
                        ui.kpi("Mean annual rate", f"{res.loc[ok, 'annual_rate'].mean():.4f}", "this file", "#0B6E99"),
                        ui.kpi("Portfolio reference", f"{b['reference_annual_rate']:.4f}", "training portfolio", "#14A3A3"),
                    ]))
                    ui.show(ui.tier_bars(res.loc[ok, "risk_tier"].astype(int).value_counts().to_dict()))
                else:
                    st.info("No rows could be scored.")

            with tab_eval:
                ev = scoring.evaluate(b, out) if {"ClaimNb", "Exposure"}.issubset(d.columns) else None
                if ev:
                    gain = (ev["deviance_baseline"] - ev["deviance_model"]) / ev["deviance_baseline"]
                    ui.show(ui.kpi_row([
                        ui.kpi("Model deviance", f"{ev['deviance_model']:.4f}", "lower is better", "#14A3A3"),
                        ui.kpi("Baseline deviance", f"{ev['deviance_baseline']:.4f}", "same rate for everyone", "#8A99AD"),
                        ui.kpi("Improvement", f"{100 * gain:.1f}%", "model vs baseline", "#1B9E77"),
                        ui.kpi("Actual / predicted", f"{ev['actual_over_predicted']:.3f}", "1.000 = perfect level", "#0B6E99"),
                    ]))
                    st.dataframe(ev["calibration"])
                    st.caption(f"{ev['rows']:,} rows with a valid exposure in (0, 1]. **If this file contains policies the "
                               "model was trained on (for example the full freMTPL2freq file), these figures are "
                               "optimistic.** The unbiased test-set figures are on the model card.")
                else:
                    st.info("Add the columns Exposure and ClaimNb to compare the model with the actual claims.")

            with tab_tbl:
                st.dataframe(out.head(200))
                st.caption("Showing the first 200 rows. The download has all rows.")
                st.download_button("Download all results", out.to_csv(index=False), "scored_policies.csv", "text/csv")

# ---------------------------------------------------------------- admin page
elif page == "Users":
    if user["role"] != "admin":
        st.error("Administrators only.")
        st.stop()
    ui.show(ui.hero("Users", "Approve new accounts and manage access"))

    users = auth.list_users()
    with st.form("manage"):
        target = st.selectbox("User", [u["username"] for u in users])
        action = st.selectbox("Action", ["Approve / enable", "Disable", "Make administrator", "Make expert"])
        go = st.form_submit_button("Apply")
    if go:
        me = user["username"]
        if action == "Approve / enable":
            ok, msg = auth.set_status(target, "active", me)
        elif action == "Disable":
            ok, msg = auth.set_status(target, "disabled", me)
        elif action == "Make administrator":
            ok, msg = auth.set_role(target, "admin", me)
        else:
            ok, msg = auth.set_role(target, "expert", me)
        (st.success if ok else st.error)(msg)

    users = auth.list_users()                                   # fresh list after the action
    pending = sum(u["status"] == "pending" for u in users)
    ui.show(ui.kpi_row([
        ui.kpi("Users", f"{len(users)}", "accounts", "#0B6E99"),
        ui.kpi("Waiting for approval", f"{pending}", "need an administrator", "#F2A33C" if pending else "#1B9E77"),
    ]))
    st.dataframe(pd.DataFrame(users), hide_index=True)
# ---------------------------------------------------------------- page 3
else:
    ui.show(ui.hero("Model card", "What the model predicts, how well it works, and where to be careful"))
    st.markdown("**What it predicts.** The expected number of claims per policy-year (claim frequency), from vehicle, "
                "driver and region characteristics. A gradient boosting model with Poisson loss. "
                "There is **no severity model**, so nothing here is a claim cost or a premium.")

    st.subheader("Performance on the held-out test set (evaluated once)")
    ui.show(ui.kpi_row([
        ui.kpi("Reference: same rate for everyone", "0.3307", "mean Poisson deviance", "#8A99AD"),
        ui.kpi("This model", "0.2892", "mean Poisson deviance", "#14A3A3"),
        ui.kpi("Improvement", "12.5%", "lower deviance than the reference", "#1B9E77"),
    ]))
    ui.show(ui.compare_bars(
        [("Reference", 0.3307, "#8A99AD"), ("This model", 0.2892, "#14A3A3")],
        "Mean Poisson deviance (lower is better)",
        "95% bootstrap interval for the gain over the reference: 0.0386 to 0.0449. "
        "Cross-validation on the training set gave 0.2907 +/- 0.0028."))

    st.subheader("Calibration: actual vs predicted claims (test set)")
    cal = pd.DataFrame({
        "Exposure band": ["0-0.1", "0.1-0.25", "0.25-0.5", "0.5-0.75", "0.75-1.0"],
        "Policies": [25542, 19162, 26194, 18527, 45932],
        "Actual claims": [675, 781, 1269, 1146, 3345],
        "Predicted claims": [683.6, 727.9, 1298.6, 1187.9, 3293.5],
        "Actual / predicted": [0.987, 1.073, 0.977, 0.965, 1.016]})
    st.dataframe(cal, hide_index=True)
    near = pd.DataFrame({
        "Group": ["Vehicle age 1+, exposure >= 0.75", "Vehicle age 1+, exposure >= 0.90",
                  "Vehicle age 0, exposure >= 0.75", "Vehicle age 0, exposure >= 0.90"],
        "Claims": [3318, 2725, 133, 30],
        "Actual / predicted": [1.014, 1.010, 1.182, 1.008]})
    st.dataframe(near, hide_index=True)

    st.subheader("How to read the annual rate")
    st.markdown(
        "- The model uses the observation period (exposure) as an input. The annual rate is predicted at exposure = 1, "
        "that is, the claim rate of policies **observed for a full year**.\n"
        "- If short records often follow a claim (for example a cancellation), full-year records are a selected group "
        "and the rate may **understate** the risk of a brand-new policy. The data has no cancellation reason, so this "
        "cannot be checked.\n"
        "- The exposure input is what keeps the annual rate calibrated: without it, near-full-year policies were "
        "over-predicted by about 15% in testing.")

    st.subheader("Known limitations")
    st.markdown(
        "- **Brand-new vehicles** (age 0): calibration is less certain. The test ratios (1.18 and 1.01) rest on only "
        "133 and 30 claims, and the training folds gave 0.90 to 0.94.\n"
        "- **Ranking power is moderate.** Claims are rare and largely random: use the tier and rate for groups of "
        "policies, not to say whether one customer will claim. Cross-validated Gini (exposure-weighted, out-of-fold): "
        "0.399 when policies are ranked at their own exposure, which mostly reflects the exposure input, and 0.324 "
        "when ranked by the annual rate at Exposure = 1, which is how the risk tiers are built.\n"
        "- **Associations, not causes.** The factors explain how a prediction is composed.\n"
        "- **Tied records.** Many policies appear as split records with identical features. Evaluation used a split "
        "that keeps identical profiles together.\n"
        "- **Inputs outside the training data** (unknown brand or region, very old vehicles) are rejected or flagged."
    )
    st.caption(f"Model version: {b['version']} | code tag: gbm_v2_frozen | figures come from the single test-set "
               "evaluation and cross-validation.")