"""
02_train_evaluate.py
Household-level regression: predict average monthly electricity consumption (kWh)
from Wave-1 survey characteristics (RECON-SL).  Main model: CatBoost.
Baselines taught in lectures: Ridge, Random Forest, XGBoost.
Explainability: SHAP (global + local) and LIME (local).

Usage: python 02_train_evaluate.py --table model_table.csv --out results
"""
import argparse, json, os, warnings
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from itertools import product
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import RidgeCV
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor, Pool
import shap
from lime.lime_tabular import LimeTabularExplainer

warnings.filterwarnings("ignore")
p = argparse.ArgumentParser()
p.add_argument("--table", default="model_table.csv")
p.add_argument("--out", default="results")
a = p.parse_args()
os.makedirs(a.out, exist_ok=True)
SEED = 42

# ------------------------------------------------------------------ 1. data
df = pd.read_csv(a.table)
static_cols = [c for c in df.columns if c not in
               ["month", "consumption", "target_log", "cal_month", "month_idx"]]
hh = (df.groupby("household_ID").agg(avg_kwh=("consumption", "mean")).reset_index()
        .merge(df[static_cols].drop_duplicates("household_ID"), on="household_ID"))
y_kwh = hh["avg_kwh"].values
y = np.log1p(y_kwh)                                   # model the log (consumption is right-skewed)
X = hh.drop(columns=["household_ID", "avg_kwh"])
cat = [c for c in X if not pd.api.types.is_numeric_dtype(X[c])]
num = [c for c in X if c not in cat]
X[cat] = X[cat].fillna("NA").astype(str)
print(f"households: {len(X)} | features: {X.shape[1]} ({len(cat)} categorical, {len(num)} numeric)")

# train / validation / test = 70 / 15 / 15
idx = np.arange(len(X))
i_tr, i_tmp = train_test_split(idx, test_size=0.30, random_state=SEED)
i_va, i_te = train_test_split(i_tmp, test_size=0.50, random_state=SEED)
Xtr, Xva, Xte = X.iloc[i_tr], X.iloc[i_va], X.iloc[i_te]
ytr, yva, yte = y[i_tr], y[i_va], y[i_te]
print("split sizes (train/val/test):", len(i_tr), len(i_va), len(i_te))

def metrics(y_log_true, y_log_pred):
    t, pr = np.expm1(y_log_true), np.expm1(y_log_pred)
    return dict(RMSE_kWh=mean_squared_error(t, pr) ** .5, MAE_kWh=mean_absolute_error(t, pr),
                R2_kWh=r2_score(t, pr), R2_log=r2_score(y_log_true, y_log_pred))

prep = ColumnTransformer([
    ("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())]), num),
    ("cat", Pipeline([("imp", SimpleImputer(strategy="constant", fill_value="NA")),
                      ("oh", OneHotEncoder(handle_unknown="ignore", min_frequency=20))]), cat)])
rows, preds = [], {}

# ------------------------------------------------------------------ 2. baselines
# 2a Ridge (L2-regularised linear regression; lambda chosen by internal CV)
ridge = Pipeline([("prep", prep), ("m", RidgeCV(alphas=np.logspace(-2, 3, 20)))]).fit(Xtr, ytr)
preds["Ridge"] = ridge.predict(Xte)

# 2b Random Forest (Breiman's regression default m = p/3, 500 trees, min leaf 3)
rf = Pipeline([("prep", prep), ("m", RandomForestRegressor(
    n_estimators=500, max_features=1 / 3, min_samples_leaf=3, n_jobs=-1, random_state=SEED))]).fit(Xtr, ytr)
preds["Random Forest"] = rf.predict(Xte)

# 2c XGBoost - small grid, chosen on the validation set
prep_x = prep.fit(Xtr)
Ttr, Tva, Tte = prep_x.transform(Xtr), prep_x.transform(Xva), prep_x.transform(Xte)
best = (1e9, None)
for d, lr in product([3, 4, 6], [0.03, 0.06]):
    m = XGBRegressor(n_estimators=1500, learning_rate=lr, max_depth=d, subsample=0.8,
                     colsample_bytree=0.8, reg_lambda=3, early_stopping_rounds=60,
                     random_state=SEED, n_jobs=-1).fit(Ttr, ytr, eval_set=[(Tva, yva)], verbose=False)
    if m.best_score < best[0]:
        best = (m.best_score, m, dict(depth=d, lr=lr))
xgb = best[1]; xgb_params = best[2]
preds["XGBoost"] = xgb.predict(Tte)

# ------------------------------------------------------------------ 3. CatBoost (main model)
grid = list(product([4, 6, 8], [0.03, 0.06], [1, 3, 10]))     # depth, learning rate, L2 leaf reg
search = []
best_cb, best_val = None, 1e9
for d, lr, l2 in grid:
    m = CatBoostRegressor(iterations=3000, learning_rate=lr, depth=d, l2_leaf_reg=l2, loss_function="RMSE",
                          random_seed=SEED, verbose=0, cat_features=cat, early_stopping_rounds=100)
    m.fit(Xtr, ytr, eval_set=(Xva, yva))
    v = m.get_best_score()["validation"]["RMSE"]
    search.append(dict(depth=d, learning_rate=lr, l2_leaf_reg=l2, val_RMSE_log=v, best_iter=m.get_best_iteration()))
    if v < best_val:
        best_val, best_cb, cb_params = v, m, dict(depth=d, learning_rate=lr, l2_leaf_reg=l2)
pd.DataFrame(search).sort_values("val_RMSE_log").to_csv(f"{a.out}/catboost_tuning.csv", index=False)
cb = best_cb
preds["CatBoost"] = cb.predict(Xte)
print("CatBoost best params:", cb_params, "| best iteration:", cb.get_best_iteration())
print("XGBoost  best params:", xgb_params)

# ------------------------------------------------------------------ 4. results table
base = np.full(len(yte), ytr.mean())
rows.append(dict(model="Mean baseline", **metrics(yte, base)))
for k in ["Ridge", "Random Forest", "XGBoost", "CatBoost"]:
    rows.append(dict(model=k, **metrics(yte, preds[k])))
res = pd.DataFrame(rows).round(3)
res.to_csv(f"{a.out}/test_metrics.csv", index=False)
print("\nTEST SET RESULTS\n", res.to_string(index=False))

# 5-fold CV of the tuned CatBoost on train+val (stability of the estimate)
Xtv, ytv = pd.concat([Xtr, Xva]), np.concatenate([ytr, yva])
cv = []
for k, (tr, va) in enumerate(KFold(5, shuffle=True, random_state=SEED).split(Xtv)):
    m = CatBoostRegressor(iterations=cb.get_best_iteration() + 1, random_seed=SEED, verbose=0,
                          cat_features=cat, **cb_params).fit(Xtv.iloc[tr], ytv[tr])
    cv.append(metrics(ytv[va], m.predict(Xtv.iloc[va])))
cv = pd.DataFrame(cv)
cv_summary = cv.agg(["mean", "std"]).round(3)
cv_summary.to_csv(f"{a.out}/catboost_5fold_cv.csv")
print("\n5-FOLD CV (CatBoost)\n", cv_summary.to_string())

# ablation: does total monthly expenditure (which includes the electricity bill) leak the target?
drop = [c for c in ["monthly_expenditure"] if c in X]
m = CatBoostRegressor(iterations=cb.get_best_iteration() + 1, random_seed=SEED, verbose=0,
                      cat_features=cat, **cb_params).fit(Xtr.drop(columns=drop), ytr)
abl = metrics(yte, m.predict(Xte.drop(columns=drop)))
pd.DataFrame([dict(setting="without monthly_expenditure", **abl)]).round(3).to_csv(f"{a.out}/ablation.csv", index=False)
print("\nAblation (no expenditure):", {k: round(v, 3) for k, v in abl.items()})

# ------------------------------------------------------------------ 6. plots
plt.rcParams.update({"figure.dpi": 130, "font.size": 9})
# 6a model comparison
fig, ax = plt.subplots(1, 2, figsize=(9, 3.2))
r = res[res.model != "Mean baseline"]
ax[0].bar(r.model, r.RMSE_kWh, color="#4c72b0"); ax[0].set_title("Test RMSE (kWh/month) - lower is better")
ax[1].bar(r.model, r.R2_kWh, color="#55a868"); ax[1].set_title("Test R2 (kWh scale) - higher is better")
for x in ax: x.tick_params(axis="x", rotation=20)
plt.tight_layout(); plt.savefig(f"{a.out}/fig_model_comparison.png"); plt.close()

# 6b learning curve (train vs validation RMSE over boosting iterations)
ev = cb.get_evals_result()
fig, ax = plt.subplots(figsize=(5, 3.3))
ax.plot(ev["learn"]["RMSE"], label="train"); ax.plot(ev["validation"]["RMSE"], label="validation")
ax.axvline(cb.get_best_iteration(), ls="--", c="grey", label="early stop")
ax.set_xlabel("iteration"); ax.set_ylabel("RMSE (log kWh)"); ax.legend(); ax.set_title("CatBoost learning curve")
plt.tight_layout(); plt.savefig(f"{a.out}/fig_learning_curve.png"); plt.close()

# 6c predicted vs actual + residuals
pt, pa = np.expm1(preds["CatBoost"]), np.expm1(yte)
fig, ax = plt.subplots(1, 2, figsize=(9, 3.6))
ax[0].scatter(pa, pt, s=6, alpha=.4); lim = [0, np.percentile(pa, 99.5)]
ax[0].plot(lim, lim, "r--"); ax[0].set_xlim(lim); ax[0].set_ylim(lim)
ax[0].set_xlabel("actual kWh/month"); ax[0].set_ylabel("predicted"); ax[0].set_title("Predicted vs actual (test)")
resid = yte - preds["CatBoost"]
ax[1].scatter(preds["CatBoost"], resid, s=6, alpha=.4); ax[1].axhline(0, c="r", ls="--")
ax[1].set_xlabel("predicted (log kWh)"); ax[1].set_ylabel("residual (log kWh)"); ax[1].set_title("Residual plot")
plt.tight_layout(); plt.savefig(f"{a.out}/fig_pred_vs_actual_residuals.png"); plt.close()

# 6d error by group (fairness / critical discussion)
te_df = Xte.copy(); te_df["abs_err"] = np.abs(pt - pa); te_df["actual"] = pa; te_df["pred"] = pt
te_df["signed_err"] = pt - pa
grp = []
for g in ["sec", "meter_type", "has_ac"]:
    t = te_df.groupby(g).agg(n=("abs_err", "size"), MAE=("abs_err", "mean"),
                             mean_actual=("actual", "mean"), mean_bias=("signed_err", "mean")).round(1)
    t.insert(0, "group_by", g); grp.append(t.reset_index().rename(columns={g: "group"}))
pd.concat(grp).to_csv(f"{a.out}/error_by_group.csv", index=False)

# ------------------------------------------------------------------ 7. SHAP (CatBoost native TreeSHAP)
sv_all = cb.get_feature_importance(Pool(Xte, cat_features=cat), type="ShapValues")
sv, base_val = sv_all[:, :-1], sv_all[0, -1]
disp = Xte.copy()
for c in cat: disp[c] = disp[c].astype("category").cat.codes
expl = shap.Explanation(values=sv, base_values=np.full(len(Xte), base_val),
                        data=disp.values.astype(float), feature_names=list(X.columns))
plt.figure(); shap.plots.bar(expl, max_display=15, show=False)
plt.tight_layout(); plt.savefig(f"{a.out}/fig_shap_bar.png", bbox_inches="tight"); plt.close()
plt.figure(); shap.plots.beeswarm(expl, max_display=15, show=False)
plt.tight_layout(); plt.savefig(f"{a.out}/fig_shap_beeswarm.png", bbox_inches="tight"); plt.close()
# one local waterfall (a household near the median prediction)
j = int(np.argsort(np.abs(preds["CatBoost"] - np.median(preds["CatBoost"])))[0])
e1 = shap.Explanation(values=sv[j], base_values=base_val, data=Xte.iloc[j].values, feature_names=list(X.columns))
plt.figure(); shap.plots.waterfall(e1, max_display=12, show=False)
plt.savefig(f"{a.out}/fig_shap_waterfall.png", bbox_inches="tight"); plt.close()
imp = pd.DataFrame({"feature": X.columns, "mean_abs_shap": np.abs(sv).mean(0)}).sort_values("mean_abs_shap", ascending=False)
imp.to_csv(f"{a.out}/shap_importance.csv", index=False)
print("\nTop SHAP features:\n", imp.head(10).round(3).to_string(index=False))

# ------------------------------------------------------------------ 8. LIME (local)
cat_levels = {c: sorted(X[c].unique()) for c in cat}
enc = X.copy()
for c in cat: enc[c] = enc[c].map({v: i for i, v in enumerate(cat_levels[c])})
enc = enc.fillna(X[num].median())
cat_idx = [list(X.columns).index(c) for c in cat]
def predict_fn(arr):
    d = pd.DataFrame(arr, columns=X.columns)
    for c in cat: d[c] = d[c].round().astype(int).map(dict(enumerate(cat_levels[c]))).fillna("NA")
    return cb.predict(d)
lime_exp = LimeTabularExplainer(enc.loc[Xtr.index.map(lambda i: i)].values if False else enc.iloc[i_tr].values,
                                feature_names=list(X.columns), categorical_features=cat_idx,
                                categorical_names={list(X.columns).index(c): cat_levels[c] for c in cat},
                                mode="regression", random_state=SEED, kernel_width=None)
for name, jj in [("typical", j), ("highest_pred", int(np.argmax(preds["CatBoost"]))),
                 ("lowest_pred", int(np.argmin(preds["CatBoost"])))]:
    ex = lime_exp.explain_instance(enc.iloc[i_te[jj]].values, predict_fn, num_features=10, num_samples=3000)
    fig = ex.as_pyplot_figure(); fig.set_size_inches(7, 3.8); plt.tight_layout()
    plt.savefig(f"{a.out}/fig_lime_{name}.png"); plt.close()
    print(f"LIME {name}: local pred={np.expm1(preds['CatBoost'][jj]):.0f} kWh, actual={pa[jj]:.0f} kWh,",
          "surrogate R2 =", round(ex.score, 2))

# ------------------------------------------------------------------ 9. save for the app
cb.save_model(f"{a.out}/catboost_model.cbm")
meta = dict(features=list(X.columns), cat_features=cat, num_features=num,
            cat_levels={c: cat_levels[c] for c in cat},
            num_stats={c: dict(min=float(X[c].min()), max=float(X[c].max()), median=float(X[c].median()))
                       for c in num},
            cat_mode={c: X[c].mode()[0] for c in cat}, shap_base_log=float(base_val),
            top_features=imp.feature.head(12).tolist())
json.dump(meta, open(f"{a.out}/model_meta.json", "w"), indent=1)
print("\nsaved model + metadata to", a.out)
