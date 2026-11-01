"""
app_tariff.py - Streamlit front end: "Will this household exceed 180 kWh next month?"
Run:  streamlit run app_tariff.py
Needs results_cls/ebm_cls_model.pkl and results_cls/ebm_cls_meta.json (created by 04_tariff_block_classifier.py)
"""
import json
import numpy as np, pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import joblib

st.set_page_config(page_title="Tariff-block risk checker", page_icon="⚡", layout="wide")

@st.cache_resource
def load():
    return json.load(open("results_cls/ebm_cls_meta.json")), joblib.load("results_cls/ebm_cls_model.pkl")

meta, model = load()
F, CAT, NS, CM, THR = meta["features"], meta["cat_features"], meta["num_stats"], meta["cat_mode"], int(meta["threshold"])

def prob(row):
    return float(model.predict_proba(pd.DataFrame([row])[F])[0, 1])

st.title("⚡ Will this household exceed 180 kWh next month?")
st.caption("180 kWh is the top block of the Sri Lankan domestic electricity tariff (highest unit rate). The model uses "
           "the household's last three months of consumption plus a few household characteristics. "
           "Model: Explainable Boosting Machine trained on RECON-SL.")

row = {c: NS[c]["median"] for c in meta["num_features"]}
row.update({c: CM[c] for c in CAT})
with st.sidebar:
    st.header("Recent consumption (kWh)")
    row["lag1"] = st.number_input("Last month", 0, 1500, 120, step=5)
    row["lag2"] = st.number_input("2 months ago", 0, 1500, 115, step=5)
    row["lag3"] = st.number_input("3 months ago", 0, 1500, 110, step=5)
    row["roll3"] = (row["lag1"] + row["lag2"] + row["lag3"]) / 3
    months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    row["cal_month"] = months.index(st.selectbox("Month being predicted", months, index=3)) + 1
    st.header("Household")
    row["hh_size"] = st.slider("Household members", 1, 12, int(NS["hh_size"]["median"]))
    row["n_rooms"] = st.slider("Rooms", 2, 25, int(NS["n_rooms"]["median"]))
    row["n_bedrooms"] = st.slider("Bedrooms", 0, 10, int(NS["n_bedrooms"]["median"]))
    row["floor_area"] = st.number_input("Floor area (sq ft)", 100, 9000, int(NS["floor_area"]["median"]), step=50)
    row["n_fans"] = st.slider("Fans", 0, 15, int(NS["n_fans"]["median"]))
    row["n_ac"] = st.slider("Air conditioners", 0, 6, 0); row["has_ac"] = int(row["n_ac"] > 0)
    row["n_appliances"] = st.slider("Total electrical appliances", 0, 80, int(NS["n_appliances"]["median"]))
    row["app_hours_week"] = st.slider("Appliance-hours used last week", 0, 1500, int(NS["app_hours_week"]["median"]), step=10)
    row["app_refrigerator"] = st.number_input("Refrigerators", 0, 5, int(NS["app_refrigerator"]["median"]))
    row["app_washing_machine"] = st.number_input("Washing machines", 0, 4, int(NS["app_washing_machine"]["median"]))
    row["mean_hours_home"] = st.slider("Average hours at home per week", 0, 168, int(NS["mean_hours_home"]["median"]))
    row["sec"] = st.selectbox("Socio-economic class", meta["cat_levels"]["sec"], index=meta["cat_levels"]["sec"].index(CM["sec"]))
    row["meter_type"] = st.selectbox("Meter type", meta["cat_levels"]["meter_type"], index=meta["cat_levels"]["meter_type"].index(CM["meter_type"]))

p = prob(row)
c1, c2 = st.columns([1, 2])
with c1:
    st.metric(f"Probability of exceeding {THR} kWh", f"{p:.0%}")
    if p >= 0.5:
        st.error(f"⚠️ Likely to land in the top tariff block (more than {THR} units).")
    elif p >= 0.25:
        st.warning("Borderline: usage could tip into the top block.")
    else:
        st.success(f"Unlikely to exceed {THR} kWh next month.")
    st.subheader("What-if")
    w = dict(row); w["n_ac"] = row["n_ac"] + 1; w["has_ac"] = 1
    st.write(f"Add one air conditioner → **{prob(w):.0%}** ({prob(w) - p:+.0%})")
    for pct in (10, 20):
        w = dict(row)
        for k in ("lag1", "lag2", "lag3"): w[k] = row[k] * (1 - pct / 100)
        w["roll3"] = (w["lag1"] + w["lag2"] + w["lag3"]) / 3
        st.write(f"Usage {pct}% lower in recent months → **{prob(w):.0%}** ({prob(w) - p:+.0%})")
    st.caption(f"On unseen months the model is right about {meta['test_accuracy']:.1%} of the time. A simple rule "
               "('same as last month') scores about the same: most of the signal is that heavy users stay heavy. "
               "The model adds calibrated probabilities and explanations. It is for awareness and planning, not billing.")
with c2:
    st.subheader("Why this probability? (EBM term contributions)")
    X1 = pd.DataFrame([row])[F]
    contrib = pd.Series(model.eval_terms(X1)[0], index=model.term_names_)
    contrib = contrib.reindex(contrib.abs().sort_values(ascending=False).index)[:10].iloc[::-1]
    fig, ax = plt.subplots(figsize=(6.5, 4))
    ax.barh(contrib.index, contrib.values, color=np.where(contrib.values > 0, "#c44e52", "#4c72b0"))
    ax.axvline(0, c="k", lw=.6); ax.set_xlabel("effect on log-odds of exceeding 180 kWh")
    plt.tight_layout(); st.pyplot(fig, clear_figure=True)
    st.caption("Red bars raise the risk, blue bars lower it. The model is additive, so these contributions plus the "
               "baseline add up exactly to the prediction.")

with st.expander("About this model & limitations"):
    st.markdown(f"""
- Trained on about 4,000 households in one utility service area (RECON-SL); it may not generalise to other regions.
- Needs the last three months of consumption; survey traits add explanation but little extra accuracy.
- Probabilities are estimates for planning and awareness, **not** a billing tool or a guarantee.
- Check the current tariff blocks with the Public Utilities Commission of Sri Lanka (PUCSL); the {THR} kWh boundary was the top block in the tariff structure used for this project.
""")
