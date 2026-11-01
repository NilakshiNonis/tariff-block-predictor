"""
01_build_dataset.py
Builds a modelling table (one row per household-month) from RECON-SL:
  monthly_consumption + Wave-1 household info, rooms, appliances, demographics.
Usage:  python 01_build_dataset.py --data_dir <folder with the CSVs> --out model_table.csv
"""
import argparse, re
import numpy as np, pandas as pd

p = argparse.ArgumentParser()
p.add_argument("--data_dir", default=".")
p.add_argument("--out", default="model_table.csv")
a = p.parse_args()
rd = lambda f: pd.read_csv(f"{a.data_dir}/{f}", low_memory=False)

# ---------- 1. target: monthly consumption ----------
m = rd("monthly_consumption.csv")
m["consumption"] = pd.to_numeric(m["consumption"].astype(str).str.replace(",", ""), errors="coerce")
m["month"] = pd.to_datetime(m["month"])
n0 = len(m)
m = m.dropna(subset=["consumption"])            # 590 missing readings
m = m[m["consumption"] > 0]                     # zero = vacant / disconnected, not real usage
m = m[m["consumption"] <= m["consumption"].quantile(0.999)]   # drop extreme outliers (likely commercial/errors)
print(f"consumption rows: {n0} -> {len(m)}")
m["cal_month"] = m["month"].dt.month
m["month_idx"] = (m["month"].dt.year - 2022) * 12 + m["month"].dt.month - 10   # time trend

# ---------- 2. household information ----------
h = rd("w1_household_information_and_history.csv")
keep = ["household_ID", "electricity_provider_csc_area", "own_the_house_or_living_on_rent",
        "built_year_of_the_house", "type_of_house", "floor_area", "no_of_household_members",
        "is_there_business_carried_out_in_the_household", "main_material_used_for_walls_of_the_house",
        "any_constructions_or_renovations_in_the_household", "highest_level_of_education_of_the_chief_wage_earner",
        "occupation_of_the_chief_wage_earner", "socio_economic_class",
        "total_monthly_expenditure_of_last_month", "type_of_electricity_meter", "no_of_electricity_meters"]
h = h[keep].rename(columns={
    "electricity_provider_csc_area": "csc_area", "own_the_house_or_living_on_rent": "ownership",
    "built_year_of_the_house": "built_year", "no_of_household_members": "hh_size",
    "is_there_business_carried_out_in_the_household": "has_business",
    "main_material_used_for_walls_of_the_house": "wall_material",
    "any_constructions_or_renovations_in_the_household": "renovations",
    "highest_level_of_education_of_the_chief_wage_earner": "head_education",
    "occupation_of_the_chief_wage_earner": "head_occupation", "socio_economic_class": "sec",
    "total_monthly_expenditure_of_last_month": "monthly_expenditure", "type_of_electricity_meter": "meter_type"})
# group rare CSC areas
vc = h["csc_area"].value_counts()
h["csc_area"] = h["csc_area"].where(h["csc_area"].isin(vc[vc >= 50].index), "OTHER")
h["ownership"] = h["ownership"].str.startswith("Yes").map({True: "Owner", False: "Rent/other"})

# ---------- 3. rooms ----------
r = rd("w1_room_roster.csv")
r["household_ID"] = r["household_ID"].str.strip()
r["is_bedroom"] = (r["main_purpose_of_the_room"] == "Bedrooms").astype(int)
rooms = r.groupby("household_ID").agg(
    n_rooms=("room_ID", "count"), n_bedrooms=("is_bedroom", "sum"),
    n_bulbs=("no_of_bulbs_in_the_room", "sum"), n_fans=("no_of_fans_in_the_room", "sum"),
    n_ac=("no_of_ACs_in_the_room", "sum"), n_windows=("no_of_windows", "sum")).reset_index()
rooms["has_ac"] = (rooms["n_ac"] > 0).astype(int)

# ---------- 4. appliances (counts of the most common / high-load types + total weekly hours) ----------
ap = rd("w1_appliances.csv")
ap["household_ID"] = ap["household_ID"].str.strip()
def slug(s):
    return re.sub(r"[^a-z0-9]+", "_", s.lower()).strip("_")[:25]
top = ap["appliance_type"].value_counts()
top = top[top >= 300].index                      # appliances owned by >=300 records
cnt = (ap[ap["appliance_type"].isin(top)].assign(t="app_" + ap["appliance_type"].map(slug))
       .pivot_table(index="household_ID", columns="t", values="appliance_ID", aggfunc="count", fill_value=0))
tot = ap.groupby("household_ID").agg(n_appliances=("appliance_ID", "count"),
                                     app_hours_week=("no_of_hours_used_during_last_week", "sum"))
appl = cnt.join(tot, how="outer").fillna(0).reset_index()

# ---------- 5. demographics (household-level summaries) ----------
d = rd("w1_demographics.csv")
demo = d.groupby("household_ID").agg(
    mean_age=("age", "mean"),
    n_children=("age", lambda s: (s < 15).sum()),
    n_elderly=("age", lambda s: (s >= 60).sum()),
    mean_hours_home=("no_of_hours_stayed_at_home_during_last_week", "mean")).reset_index()
emp = d["main_activity_engaged_in"].fillna("").str.contains("economic activity", case=False) & \
      ~d["main_activity_engaged_in"].fillna("").str.contains("not engaged in economic", case=False)
demo = demo.merge(d.assign(emp=emp).groupby("household_ID")["emp"].sum().rename("n_employed").reset_index())

# ---------- 6. merge ----------
df = (m.merge(h, on="household_ID", how="left").merge(rooms, on="household_ID", how="left")
        .merge(appl, on="household_ID", how="left").merge(demo, on="household_ID", how="left"))
app_cols = [c for c in df.columns if c.startswith("app_") or c in ("n_appliances", "app_hours_week")]
df[app_cols] = df[app_cols].fillna(0)            # no appliance record = none owned
df["persons_per_room"] = df["hh_size"] / df["n_rooms"].replace(0, np.nan)
df["target_log"] = np.log1p(df["consumption"])
print("final table:", df.shape, "| households:", df["household_ID"].nunique())
print("missing (top):\n", df.isna().mean().sort_values(ascending=False).head(8).round(3).to_string())
df.to_csv(a.out, index=False)
