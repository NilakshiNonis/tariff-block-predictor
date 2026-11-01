"""
03_ebm_main.py
MAIN MODEL: Explainable Boosting Machine (EBM / GA2M) - a glass-box additive model
    log(1 + kWh) = b0 + sum_j f_j(x_j) + sum_{(j,k)} f_jk(x_j, x_k)
trained by cyclic round-robin gradient boosting (one feature at a time, tiny learning rate, bagging).
Not covered in the lectures. CatBoost / XGBoost / Random Forest / Ridge (from 02_train_evaluate.py) are
kept as black-box benchmarks.

Usage: python 03_ebm_main.py --table model_table.csv --out results
(run 02_train_evaluate.py first so results/test_metrics.csv and results/shap_importance.csv exist)
"""
import argparse, json, os, warnings, joblib
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from itertools import product
from scipy.stats import spearmanr
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from interpret.glassbox import ExplainableBoostingRegressor as EBM
from lime.lime_tabular import LimeTabularExplainer

warnings.filterwarnings("ignore")
p = argparse.ArgumentParser()
p.add_argument("--table", default="model_table.csv")
p.add_argument("--out", default="results")
a = p.parse_args()
os.makedirs(a.out, exist_ok=True)
SEED = 42

# ------------------------------------------------ data + identical 70/15/15 split as script 02
df = pd.read_csv(a.table)
static_cols = [c for c in df.columns if c not in ["month", "consumption", "target_log", "cal_month", "month_idx"]]
hh = (df.groupby("household_ID").agg(avg_kwh=("consumption", "mean")).reset_index()
        .merge(df[static_cols].drop_duplicates("household_ID"), on="household_ID"))
y = np.log1p(hh["avg_kwh"].values)
X = hh.drop(columns=["household_ID", "avg_kwh"])
cat = [c for c in X if not pd.api.types.is_numeric_dtype(X[c])]
num = [c for c in X if c not in cat]
X[cat] = X[cat].fillna("NA").astype(str)
idx = np.arange(len(X))
i_tr, i_tmp = train_test_split(idx, test_size=0.30, random_state=SEED)
i_va, i_te = train_test_split(i_tmp, test_size=0.50, random_state=SEED)
Xtr, Xva, Xte = X.iloc[i_tr], X.iloc[i_va], X.iloc[i_te]
ytr, yva, yte = y[i_tr], y[i_va], y[i_te]

def metrics(yt_log, yp_log):
    t, pr = np.expm1(yt_log), np.expm1(yp_log)
    return dict(RMSE_kWh=mean_squared_error(t, pr) ** .5, MAE_kWh=mean_absolute_error(t, pr),
                R2_kWh=r2_score(t, pr), R2_log=r2_score(yt_log, yp_log))

def make(inter=0, lr=0.02, leaves=3, rounds=3000, bags=8):
    return EBM(interactions=inter, learning_rate=lr, max_leaves=leaves, max_rounds=rounds,
               early_stopping_rounds=60, outer_bags=bags, random_state=SEED, n_jobs=-1)

# ------------------------------------------------ hyperparameter search (selected on the VALIDATION set)
rows = []
best, best_v = None, 1e9
for inter, lr, leaves in product([0, 10], [0.01, 0.03], [2, 3]):
    m = make(inter, lr, leaves, bags=4).fit(Xtr, ytr)
    v = mean_squared_error(yva, m.predict(Xva)) ** .5
    rows.append(dict(interactions=inter, learning_rate=lr, max_leaves=leaves, val_RMSE_log=v))
    if v < best_v:
        best_v, best, params = v, m, dict(inter=inter, lr=lr, leaves=leaves)
pd.DataFrame(rows).sort_values("val_RMSE_log").round(4).to_csv(f"{a.out}/ebm_tuning.csv", index=False)
ebm = make(**params).fit(Xtr, ytr)      # refit the chosen configuration with the full 8 bags
print("EBM best params:", params, "| val RMSE (log):", round(best_v, 4))

# ------------------------------------------------ test results vs benchmarks
pte = ebm.predict(Xte)
res = pd.read_csv(f"{a.out}/test_metrics.csv")
res = res[res.model != "EBM (main)"]
res = pd.concat([res, pd.DataFrame([dict(model="EBM (main)", **metrics(yte, pte))])]).round(3)
res.to_csv(f"{a.out}/test_metrics_with_ebm.csv", index=False)
print("\nTEST SET RESULTS\n", res.to_string(index=False))

cv = []
Xtv, ytv = pd.concat([Xtr, Xva]), np.concatenate([ytr, yva])
for tr, va in KFold(5, shuffle=True, random_state=SEED).split(Xtv):
    cv.append(metrics(ytv[va], make(**params).fit(Xtv.iloc[tr], ytv[tr]).predict(Xtv.iloc[va])))
cvs = pd.DataFrame(cv).agg(["mean", "std"]).round(3)
cvs.to_csv(f"{a.out}/ebm_5fold_cv.csv")
print("\n5-FOLD CV (EBM)\n", cvs.to_string())

# ablation: drop expenditure (it includes the electricity bill) to rule out target leakage
drop = ["monthly_expenditure"]
ab = metrics(yte, make(**params).fit(Xtr.drop(columns=drop), ytr).predict(Xte.drop(columns=drop)))
pd.DataFrame([dict(setting="EBM without monthly_expenditure", **ab)]).round(3).to_csv(f"{a.out}/ebm_ablation.csv", index=False)
print("Ablation (no expenditure):", {k: round(v, 3) for k, v in ab.items()})

# ------------------------------------------------ plots
plt.rcParams.update({"figure.dpi": 130, "font.size": 9})
r = res[res.model != "Mean baseline"]
fig, ax = plt.subplots(1, 2, figsize=(9.5, 3.3))
cols = ["#c44e52" if m == "EBM (main)" else "#4c72b0" for m in r.model]
ax[0].bar(r.model, r.RMSE_kWh, color=cols); ax[0].set_title("Test RMSE (kWh/month) - lower is better")
ax[1].bar(r.model, r.R2_kWh, color=cols); ax[1].set_title("Test R2 (kWh scale) - higher is better")
for x in ax: x.tick_params(axis="x", rotation=25)
plt.tight_layout(); plt.savefig(f"{a.out}/fig_ebm_model_comparison.png"); plt.close()

# learning curve vs training-set size (train vs validation error -> over/underfitting diagnosis)
fr, trE, vaE = [0.2, 0.4, 0.6, 0.8, 1.0], [], []
rng = np.random.RandomState(SEED)
for f in fr:
    sub = rng.permutation(len(Xtr))[: int(f * len(Xtr))]
    m = make(**params).fit(Xtr.iloc[sub], ytr[sub])
    trE.append(mean_squared_error(ytr[sub], m.predict(Xtr.iloc[sub])) ** .5)
    vaE.append(mean_squared_error(yva, m.predict(Xva)) ** .5)
fig, ax = plt.subplots(figsize=(5, 3.3))
ax.plot([int(f * len(Xtr)) for f in fr], trE, "o-", label="train"); ax.plot([int(f * len(Xtr)) for f in fr], vaE, "o-", label="validation")
ax.set_xlabel("training households"); ax.set_ylabel("RMSE (log kWh)"); ax.legend(); ax.set_title("EBM learning curve")
plt.tight_layout(); plt.savefig(f"{a.out}/fig_ebm_learning_curve.png"); plt.close()

pt, pa = np.expm1(pte), np.expm1(yte)
fig, ax = plt.subplots(1, 2, figsize=(9, 3.6))
ax[0].scatter(pa, pt, s=6, alpha=.4); lim = [0, np.percentile(pa, 99.5)]
ax[0].plot(lim, lim, "r--"); ax[0].set_xlim(lim); ax[0].set_ylim(lim)
ax[0].set_xlabel("actual kWh/month"); ax[0].set_ylabel("predicted"); ax[0].set_title("Predicted vs actual (test)")
ax[1].scatter(pte, yte - pte, s=6, alpha=.4); ax[1].axhline(0, c="r", ls="--")
ax[1].set_xlabel("predicted (log kWh)"); ax[1].set_ylabel("residual (log kWh)"); ax[1].set_title("Residual plot")
plt.tight_layout(); plt.savefig(f"{a.out}/fig_ebm_pred_vs_actual_residuals.png"); plt.close()

# error by group (fairness)
g = Xte.copy(); g["abs_err"] = np.abs(pt - pa); g["actual"] = pa; g["bias"] = pt - pa
out = []
for c in ["sec", "meter_type", "has_ac"]:
    t = g.groupby(c).agg(n=("abs_err", "size"), MAE=("abs_err", "mean"), mean_actual=("actual", "mean"),
                         mean_bias=("bias", "mean")).round(1)
    t.insert(0, "group_by", c); out.append(t.reset_index().rename(columns={c: "group"}))
pd.concat(out).to_csv(f"{a.out}/ebm_error_by_group.csv", index=False)

# ------------------------------------------------ INTERPRETATION 1: global importance + shape functions (= PDP-style)
glob = ebm.explain_global()
imp = pd.DataFrame({"term": ebm.term_names_, "importance": ebm.term_importances()}).sort_values("importance", ascending=False)
imp.to_csv(f"{a.out}/ebm_term_importance.csv", index=False)
top = imp.head(15).iloc[::-1]
plt.figure(figsize=(6, 4.5)); plt.barh(top.term, top.importance, color="#c44e52")
plt.xlabel("mean |contribution| (log kWh)"); plt.title("EBM global feature importance"); plt.tight_layout()
plt.savefig(f"{a.out}/fig_ebm_importance.png"); plt.close()

single = [t for t in imp.term if " & " not in t][:9]
fig, axs = plt.subplots(3, 3, figsize=(11, 8.5)); axs = axs.ravel()
for ax, t in zip(axs, single):
    d = glob.data(ebm.term_names_.index(t))
    sc = np.array(d["scores"], float)
    if t in cat:
        ax.bar(range(len(sc)), sc); ax.set_xticks(range(len(sc))); ax.set_xticklabels([str(n)[:12] for n in d["names"]], rotation=60, ha="right", fontsize=6)
    else:
        edges = np.array(d["names"], float)
        ax.step(edges, np.append(sc, sc[-1]), where="post", color="#c44e52")
        ax.fill_between(edges, np.append(d["lower_bounds"], d["lower_bounds"][-1]),
                        np.append(d["upper_bounds"], d["upper_bounds"][-1]), step="post", alpha=.25)
    ax.axhline(0, c="grey", lw=.6); ax.set_title(t, fontsize=9); ax.set_ylabel("effect on log kWh", fontsize=7)
plt.tight_layout(); plt.savefig(f"{a.out}/fig_ebm_shape_functions.png"); plt.close()

# ------------------------------------------------ INTERPRETATION 2: local explanations (exact additive contributions)
contrib = ebm.eval_terms(Xte)                      # n x terms, in log-kWh, exact decomposition
def local_plot(j, name):
    c = pd.Series(contrib[j], index=ebm.term_names_)
    c = c.reindex(c.abs().sort_values(ascending=False).index)[:10].iloc[::-1]
    plt.figure(figsize=(6.5, 3.8)); plt.barh(c.index, c.values, color=np.where(c.values > 0, "#c44e52", "#4c72b0"))
    plt.axvline(0, c="k", lw=.6)
    plt.title(f"{name}: pred {pt[j]:.0f} kWh (actual {pa[j]:.0f}); intercept {ebm.intercept_:.2f}", fontsize=9)
    plt.xlabel("contribution to log kWh"); plt.tight_layout(); plt.savefig(f"{a.out}/fig_ebm_local_{name}.png"); plt.close()
jt = int(np.argsort(np.abs(pte - np.median(pte)))[0])
for name, j in [("typical", jt), ("highest_pred", int(np.argmax(pte))), ("lowest_pred", int(np.argmin(pte)))]:
    local_plot(j, name)

# ------------------------------------------------ INTERPRETATION 3: LIME (post-hoc) to cross-check the native explanation
levels = {c: sorted(X[c].unique()) for c in cat}
enc = X.copy()
for c in cat: enc[c] = enc[c].map({v: i for i, v in enumerate(levels[c])})
enc = enc.fillna(X[num].median())
cidx = [list(X.columns).index(c) for c in cat]
def pfn(arr):
    d = pd.DataFrame(arr, columns=X.columns)
    for c in cat: d[c] = d[c].round().astype(int).map(dict(enumerate(levels[c]))).fillna("NA")
    return ebm.predict(d)
lx = LimeTabularExplainer(enc.iloc[i_tr].values, feature_names=list(X.columns), categorical_features=cidx,
                          categorical_names={list(X.columns).index(c): levels[c] for c in cat},
                          mode="regression", random_state=SEED)
ex = lx.explain_instance(enc.iloc[i_te[jt]].values, pfn, num_features=10, num_samples=3000)
fig = ex.as_pyplot_figure(); fig.set_size_inches(7, 3.8); plt.tight_layout()
plt.savefig(f"{a.out}/fig_ebm_lime_typical.png"); plt.close()
print("LIME surrogate R2 (typical household):", round(ex.score, 2))

# ------------------------------------------------ agreement between explainers (does the story hold?)
try:
    cb = pd.read_csv(f"{a.out}/shap_importance.csv").set_index("feature").mean_abs_shap
    e = imp.set_index("term").importance
    common = [t for t in cb.index if t in e.index]
    rho = spearmanr(cb[common], e[common]).correlation
    top10 = len(set(cb.sort_values(ascending=False).head(10).index) & set(e.sort_values(ascending=False).head(10).index))
    print(f"\nEBM vs CatBoost-SHAP importance: Spearman rho = {rho:.2f}; overlap in top-10 = {top10}/10")
except Exception as ex_:
    print("agreement check skipped:", ex_)
print("\nTop EBM terms:\n", imp.head(10).round(3).to_string(index=False))

# ------------------------------------------------ save for the app
joblib.dump(ebm, f"{a.out}/ebm_model.pkl")
meta = dict(features=list(X.columns), cat_features=cat, num_features=num,
            cat_levels=levels,
            num_stats={c: dict(min=float(X[c].min()), max=float(X[c].max()), median=float(X[c].median())) for c in num},
            cat_mode={c: X[c].mode()[0] for c in cat}, top_features=imp.term.head(12).tolist(),
            intercept=float(ebm.intercept_), model="EBM")
json.dump(meta, open(f"{a.out}/ebm_meta.json", "w"), indent=1)
print("saved ebm_model.pkl and ebm_meta.json")
