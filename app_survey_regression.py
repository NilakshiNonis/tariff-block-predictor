"""
app.py - Streamlit front end for the household electricity model.
Run:  streamlit run app_survey_regression.py        (expects results/ebm_model.pkl and results/ebm_meta.json)
"""
import json
import numpy as np, pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import joblib

st.set_page_config(page_title="Household Electricity Predictor", page_icon="⚡", layout="wide")

@st.cache_resource
def load():
    meta = json.load(open("results/ebm_meta.json"))
    model = joblib.load("results/ebm_model.pkl")
    return meta, model

meta, model = load()
F, CAT = meta["features"], meta["cat_features"]
NS, CM = meta["num_stats"], meta["cat_mode"]

def default_row():
    row = {c: NS[c]["median"] for c in meta["num_features"]}
    row.update({c: CM[c] for c in CAT})
    return row

def predict(row):
    df = pd.DataFrame([row])[F]
    return float(np.expm1(model.predict(df)[0]))

st.title("⚡ Household Electricity Consumption Predictor")
st.caption("Predicts a Sri Lankan household's typical monthly electricity use (kWh) from housing, appliance and "
           "household characteristics, and explains what drives the estimate. Model: Explainable Boosting Machine trained on RECON-SL.")

row = default_row()
with st.sidebar:
    st.header("Describe the household")
    row["hh_size"] = st.slider("Household members", 1, 12, int(NS["hh_size"]["median"]))
    row["n_rooms"] = st.slider("Number of rooms", 2, 25, int(NS["n_rooms"]["median"]))
    row["n_bedrooms"] = st.slider("Bedrooms", 0, 10, int(NS["n_bedrooms"]["median"]))
    row["floor_area"] = st.number_input("Floor area (sq ft)", 100, 9000, int(NS["floor_area"]["median"]), step=50)
    st.subheader("Cooling & lighting")
    row["n_fans"] = st.slider("Fans", 0, 15, int(NS["n_fans"]["median"]))
    row["n_ac"] = st.slider("Air conditioners", 0, 6, 0)
    row["has_ac"] = int(row["n_ac"] > 0)
    row["n_bulbs"] = st.slider("Light bulbs", 0, 60, int(NS["n_bulbs"]["median"]))
    st.subheader("Appliances")
    for key, label in [("app_refrigerator", "Refrigerator"), ("app_washing_machine", "Washing machine"),
                       ("app_microwave", "Microwave"), ("app_electric_kettle", "Electric kettle"),
                       ("app_laptops", "Laptops")]:
        if key in F:
            row[key] = st.number_input(label, 0, 6, int(NS[key]["median"]))
    row["n_appliances"] = st.slider("Total electrical appliances", 0, 80, int(NS["n_appliances"]["median"]))
    row["app_hours_week"] = st.slider("Total appliance-hours used last week", 0, 1500,
                                      int(NS["app_hours_week"]["median"]), step=10)
    st.subheader("Household profile")
    row["mean_hours_home"] = st.slider("Average hours at home per week", 0, 168, int(NS["mean_hours_home"]["median"]))
    row["monthly_expenditure"] = st.number_input("Total monthly expenditure (LKR)", 5000, 300000,
                                                 int(NS["monthly_expenditure"]["median"]), step=5000)
    row["sec"] = st.selectbox("Socio-economic class", meta["cat_levels"]["sec"],
                              index=meta["cat_levels"]["sec"].index(CM["sec"]))
    row["type_of_house"] = st.selectbox("Type of house", meta["cat_levels"]["type_of_house"],
                                        index=meta["cat_levels"]["type_of_house"].index(CM["type_of_house"]))
    row["csc_area"] = st.selectbox("Service area (CSC)", meta["cat_levels"]["csc_area"],
                                   index=meta["cat_levels"]["csc_area"].index(CM["csc_area"]))
    row["persons_per_room"] = row["hh_size"] / max(row["n_rooms"], 1)

pred = predict(row)
c1, c2 = st.columns([1, 2])
with c1:
    st.metric("Predicted monthly consumption", f"{pred:,.0f} kWh")
    st.caption("Typical error of the model is about ±48 kWh/month (MAE on unseen households). "
               "It estimates what *similar* households use; it cannot see this household's actual habits.")
    st.subheader("What-if")
    w = dict(row); w["n_ac"] = row["n_ac"] + 1; w["has_ac"] = 1
    st.write(f"Add one air conditioner → **{predict(w):,.0f} kWh** ({predict(w) - pred:+.0f})")
    if row["n_fans"] > 0:
        w = dict(row); w["n_fans"] = row["n_fans"] - 1
        st.write(f"Remove one fan → **{predict(w):,.0f} kWh** ({predict(w) - pred:+.0f})")
    w = dict(row); w["app_hours_week"] = row["app_hours_week"] * 0.8
    st.write(f"Use appliances 20% fewer hours → **{predict(w):,.0f} kWh** ({predict(w) - pred:+.0f})")

with c2:
    st.subheader("Why this prediction? (EBM term contributions)")
    X1 = pd.DataFrame([row])[F]
    contrib = pd.Series(model.eval_terms(X1)[0], index=model.term_names_)
    contrib = contrib.reindex(contrib.abs().sort_values(ascending=False).index)[:10].iloc[::-1]
    fig, ax = plt.subplots(figsize=(6.5, 4))
    ax.barh(contrib.index, contrib.values, color=np.where(contrib.values > 0, "#c44e52", "#4c72b0"))
    ax.axvline(0, c="k", lw=.6); ax.set_xlabel("effect on log(kWh) relative to the average household")
    plt.tight_layout(); st.pyplot(fig, clear_figure=True)
    st.caption("Red bars push the prediction up, blue bars push it down. The EBM is an additive model, so these "
               "contributions add up exactly to the prediction. Features you did not set use the median / most common value.")

with st.expander("About this model & limitations"):
    st.markdown("""
- Model: Explainable Boosting Machine (a glass-box additive model), so every prediction is fully explainable.
- Trained on ~4,000 households in the LECO service area (RECON-SL dataset); may not generalise to other regions.
- Uses survey data only: it captures a household's *capacity* to use electricity, not daily behaviour (R² ≈ 0.4 in cross-validation).
- Predictions are estimates for planning and awareness, **not** a billing tool.
""")
