"""
04_tariff_block_classifier.py
PROBLEM: Will a household's electricity consumption NEXT month exceed 180 kWh (the top Sri Lankan domestic
tariff block)?   Inputs: last 3 months of consumption, calendar month, Wave-1 survey traits.
MAIN MODEL: Explainable Boosting Machine (not covered in lectures).
BENCHMARKS: logistic regression, Random Forest, XGBoost (all taught) + two naive baselines.
Time-based split: train = earliest months, validation = next 3 months, test = final 3 months.

Usage: python 04_tariff_block_classifier.py --table model_table.csv --out results_cls
"""
import argparse, json, os, warnings, joblib
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from itertools import product
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, average_precision_score, confusion_matrix, roc_curve,
                             precision_recall_curve, log_loss)
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from interpret.glassbox import ExplainableBoostingClassifier as EBC
from lime.lime_tabular import LimeTabularExplainer

warnings.filterwarnings("ignore")
p = argparse.ArgumentParser()
p.add_argument("--table", default="model_table.csv")
p.add_argument("--out", default="results_cls")
p.add_argument("--threshold", type=float, default=180.0)
a = p.parse_args()
os.makedirs(a.out, exist_ok=True)
SEED, THR = 42, a.threshold
plt.rcParams.update({"figure.dpi": 130, "font.size": 9})

# ------------------------------------------------------------------ 1. build lag features (past months only)
df = pd.read_csv(a.table)
df["m"] = pd.to_datetime(df["month"]).dt.to_period("M").astype(int)
piv = df.pivot_table(index="household_ID", columns="m", values="consumption")
def lag(k):
    s = piv.stack().rename("c").reset_index(); s["m"] = s["m"] + k
    return s.rename(columns={"c": f"lag{k}"})
for k in (1, 2, 3):
    df = df.merge(lag(k), on=["household_ID", "m"], how="left")
df["roll3"] = df[["lag1", "lag2", "lag3"]].mean(axis=1)
d = df.dropna(subset=["lag1"]).copy().reset_index(drop=True)
d["y"] = (d["consumption"] > THR).astype(int)

LAGS = ["lag1", "lag2", "lag3", "roll3"]
SURVEY_NUM = ["hh_size", "n_ac", "has_ac", "n_fans", "n_appliances", "app_hours_week", "floor_area", "n_rooms",
              "n_bedrooms", "app_refrigerator", "app_washing_machine", "mean_hours_home"]
SURVEY_CAT = ["sec", "meter_type"]
FEATS = LAGS + ["cal_month"] + SURVEY_NUM + SURVEY_CAT
NUM = [f for f in FEATS if f not in SURVEY_CAT]
for c in SURVEY_CAT:
    d[c] = d[c].fillna("NA").astype(str)

mx = d["m"].max()
tr = d[d.m <= mx - 6]; va = d[(d.m > mx - 6) & (d.m <= mx - 3)]; te = d[d.m > mx - 3]
print(f"rows train/val/test = {len(tr)}/{len(va)}/{len(te)} | positive share = "
      f"{tr.y.mean():.3f}/{va.y.mean():.3f}/{te.y.mean():.3f}")
Xtr, Xva, Xte = tr[FEATS], va[FEATS], te[FEATS]
ytr, yva, yte = tr.y.values, va.y.values, te.y.values

def full_metrics(y, pr, thr=0.5):
    pred = (pr >= thr).astype(int)
    return dict(accuracy=accuracy_score(y, pred), balanced_acc=balanced_accuracy_score(y, pred),
                precision=precision_score(y, pred, zero_division=0), recall=recall_score(y, pred),
                F1=f1_score(y, pred), ROC_AUC=roc_auc_score(y, pr), PR_AUC=average_precision_score(y, pr))

# ------------------------------------------------------------------ 2. naive baselines
baseline_rows = []
for name, fn in [("Majority class (always 'below')", lambda s: np.zeros(len(s))),
                 ("Persistence ('same as last month')", lambda s: (s.lag1 > THR).astype(float).values)]:
    row = dict(model=name)
    for sp, s in [("train", tr), ("val", va), ("test", te)]:
        row[f"acc_{sp}"] = accuracy_score(s.y, fn(s))
    pt = fn(te)
    row.update(balanced_acc=balanced_accuracy_score(yte, pt), precision=precision_score(yte, pt, zero_division=0),
               recall=recall_score(yte, pt), F1=f1_score(yte, pt, zero_division=0), ROC_AUC=np.nan, PR_AUC=np.nan)
    baseline_rows.append(row)

# ------------------------------------------------------------------ 3. benchmark models (selected on validation log-loss)
prep = ColumnTransformer([
    ("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())]), NUM),
    ("cat", OneHotEncoder(handle_unknown="ignore"), SURVEY_CAT)])
probs = {}   # name -> dict(train, val, test probabilities)

# Logistic regression (L2; C chosen on validation)
best = (1e9, None)
for C in [0.01, 0.1, 1, 10]:
    m = Pipeline([("prep", prep), ("m", LogisticRegression(C=C, max_iter=2000))]).fit(Xtr, ytr)
    ll = log_loss(yva, m.predict_proba(Xva)[:, 1])
    if ll < best[0]: best = (ll, m, C)
lr = best[1]; probs["Logistic Regression"] = {s: lr.predict_proba(X)[:, 1] for s, X in [("train", Xtr), ("val", Xva), ("test", Xte)]}
print("LogReg C =", best[2])

# Random Forest (Breiman default m = sqrt(p) for classification)
rf = Pipeline([("prep", prep), ("m", RandomForestClassifier(n_estimators=300, max_features="sqrt", min_samples_leaf=5,
                                                          n_jobs=-1, random_state=SEED))]).fit(Xtr, ytr)
probs["Random Forest"] = {s: rf.predict_proba(X)[:, 1] for s, X in [("train", Xtr), ("val", Xva), ("test", Xte)]}

# XGBoost (small grid; early stopping on validation)
Ttr = prep.fit(Xtr).transform(Xtr); Tva = prep.transform(Xva); Tte = prep.transform(Xte)
best = (1e9, None)
for dpt, lrt in product([3, 5], [0.05, 0.1]):
    m = XGBClassifier(n_estimators=1500, learning_rate=lrt, max_depth=dpt, subsample=0.8, colsample_bytree=0.8,
                      reg_lambda=3, early_stopping_rounds=50, eval_metric="logloss", random_state=SEED, n_jobs=-1)
    m.fit(Ttr, ytr, eval_set=[(Tva, yva)], verbose=False)
    if m.best_score < best[0]: best = (m.best_score, m, (dpt, lrt))
xgb = best[1]; print("XGBoost depth/lr =", best[2])
probs["XGBoost"] = {s: xgb.predict_proba(T)[:, 1] for s, T in [("train", Ttr), ("val", Tva), ("test", Tte)]}

# ------------------------------------------------------------------ 4. EBM (main model)
def make_ebm(inter=0, lr_=0.02, leaves=3, bags=4, rounds=2000):
    return EBC(interactions=inter, learning_rate=lr_, max_leaves=leaves, max_rounds=rounds, early_stopping_rounds=50,
               outer_bags=bags, random_state=SEED, n_jobs=-1)
grid, best = [], (1e9, None)
for inter, lr_, leaves in product([0, 5], [0.02, 0.05], [2, 3]):
    m = make_ebm(inter, lr_, leaves).fit(Xtr, ytr)
    ll = log_loss(yva, m.predict_proba(Xva)[:, 1])
    grid.append(dict(interactions=inter, learning_rate=lr_, max_leaves=leaves, val_log_loss=ll,
                     val_accuracy=accuracy_score(yva, m.predict(Xva))))
    if ll < best[0]: best = (ll, m, dict(inter=inter, lr_=lr_, leaves=leaves))
pd.DataFrame(grid).sort_values("val_log_loss").round(4).to_csv(f"{a.out}/ebm_tuning.csv", index=False)
ebm_params = best[2]; print("EBM best params:", ebm_params)
ebm = make_ebm(**ebm_params, bags=8).fit(Xtr, ytr)
probs["EBM (main)"] = {s: ebm.predict_proba(X)[:, 1] for s, X in [("train", Xtr), ("val", Xva), ("test", Xte)]}

# ------------------------------------------------------------------ 5. results tables
rows = []
for name, pr in probs.items():
    mt = full_metrics(yte, pr["test"])
    rows.append(dict(model=name, acc_train=accuracy_score(ytr, pr["train"] >= .5), acc_val=accuracy_score(yva, pr["val"] >= .5),
                     acc_test=mt["accuracy"], **{k: v for k, v in mt.items() if k != "accuracy"}))
res = pd.concat([pd.DataFrame(baseline_rows), pd.DataFrame(rows)]).round(4)
res.to_csv(f"{a.out}/test_metrics.csv", index=False)
print("\nRESULTS\n", res.to_string(index=False))

pe = probs["EBM (main)"]["test"]; pred_e = (pe >= .5).astype(int)
cm = confusion_matrix(yte, pred_e); print("\nConfusion matrix (EBM, test):\n", cm)

# status-change analysis: rows whose class differs from last month
pers = (te.lag1 > THR).astype(int).values
chg = pers != yte
ch = pd.DataFrame([dict(model=n, share_rows_changed=chg.mean(), acc_changed_rows=((pr["test"] >= .5)[chg] == yte[chg]).mean(),
                        acc_unchanged_rows=((pr["test"] >= .5)[~chg] == yte[~chg]).mean()) for n, pr in probs.items()]
                  + [dict(model="Persistence", share_rows_changed=chg.mean(), acc_changed_rows=0.0, acc_unchanged_rows=1.0)]).round(3)
ch.to_csv(f"{a.out}/status_change_analysis.csv", index=False)

# bootstrap CI (resampling households) for EBM and persistence test accuracy
rng = np.random.RandomState(SEED)
hh_ids = te.household_ID.values; uniq = np.unique(hh_ids)
idx_by = {h: np.where(hh_ids == h)[0] for h in uniq}
corr_e, corr_p = (pred_e == yte).astype(int), (pers == yte).astype(int)
be, bp = [], []
for _ in range(400):
    ii = np.concatenate([idx_by[h] for h in rng.choice(uniq, len(uniq))])
    be.append(corr_e[ii].mean()); bp.append(corr_p[ii].mean())
ci = pd.DataFrame([dict(model="EBM (main)", acc=corr_e.mean(), ci_low=np.percentile(be, 2.5), ci_high=np.percentile(be, 97.5)),
                   dict(model="Persistence", acc=corr_p.mean(), ci_low=np.percentile(bp, 2.5), ci_high=np.percentile(bp, 97.5))]).round(4)
ci.to_csv(f"{a.out}/test_accuracy_bootstrap_ci.csv", index=False); print("\nBootstrap CI:\n", ci.to_string(index=False))

# rolling-origin stability: 4 successive 3-month test windows, always training on everything before
roll = []
for k, end in enumerate([mx - 9, mx - 6, mx - 3, mx]):
    trk = d[d.m <= end - 3]; tek = d[(d.m > end - 3) & (d.m <= end)]
    m = make_ebm(**ebm_params, bags=4).fit(trk[FEATS], trk.y.values)
    prk = m.predict_proba(tek[FEATS])[:, 1]
    roll.append(dict(window=f"months {end - 2}..{end}", n_test=len(tek), pos_share=tek.y.mean(),
                     acc_train=accuracy_score(trk.y, m.predict(trk[FEATS])), acc_test=accuracy_score(tek.y, prk >= .5),
                     acc_persistence=((tek.lag1 > THR).astype(int) == tek.y).mean(), AUC=roc_auc_score(tek.y, prk),
                     F1=f1_score(tek.y, prk >= .5)))
roll = pd.DataFrame(roll).round(4); roll.to_csv(f"{a.out}/rolling_origin_stability.csv", index=False)
print("\nRolling-origin:\n", roll.to_string(index=False))

# fairness / error analysis on the test set
g = te.copy(); g["pred"] = pred_e; g["prob"] = pe
out = []
for c in ["sec", "meter_type", "has_ac"]:
    t = g.groupby(c).apply(lambda s: pd.Series(dict(n=len(s), positive_rate=s.y.mean(), accuracy=(s.pred == s.y).mean(),
                                                    recall=((s.pred == 1) & (s.y == 1)).sum() / max((s.y == 1).sum(), 1),
                                                    mean_prob=s.prob.mean()))).round(3)
    t.insert(0, "group_by", c); out.append(t.reset_index().rename(columns={c: "group"}))
pd.concat(out).to_csv(f"{a.out}/error_by_group.csv", index=False)

# ------------------------------------------------------------------ 6. figures
# 6a model comparison
names = list(res.model); acc = res.acc_test.values
fig, ax = plt.subplots(1, 3, figsize=(12, 3.6))
cols = ["#999999"] * 2 + ["#4c72b0"] * 3 + ["#c44e52"]
for x, (col, title) in zip(ax, [("acc_test", "Test accuracy"), ("F1", "Test F1 (positive = above 180 kWh)"), ("ROC_AUC", "Test ROC-AUC")]):
    x.barh(names, res[col].fillna(0).values, color=cols); x.set_title(title); x.invert_yaxis()
    x.set_xlim(0.5 if col != "F1" else 0.5, 1.0)
for x in ax[1:]: x.set_yticklabels([])
plt.tight_layout(); plt.savefig(f"{a.out}/fig_model_comparison.png"); plt.close()

# 6b confusion matrix
fig, ax = plt.subplots(figsize=(3.8, 3.4)); ax.imshow(cm, cmap="Blues")
for (i, j), v in np.ndenumerate(cm): ax.text(j, i, f"{v:,}", ha="center", va="center", color="black" if v < cm.max() / 2 else "white")
ax.set_xticks([0, 1]); ax.set_yticks([0, 1]); ax.set_xticklabels(["<=180", ">180"]); ax.set_yticklabels(["<=180", ">180"])
ax.set_xlabel("predicted"); ax.set_ylabel("actual"); ax.set_title("EBM confusion matrix (test)")
plt.tight_layout(); plt.savefig(f"{a.out}/fig_confusion_matrix.png"); plt.close()

# 6c ROC and PR
fig, ax = plt.subplots(1, 2, figsize=(9, 3.6))
for name, pr in probs.items():
    f_, t_, _ = roc_curve(yte, pr["test"]); ax[0].plot(f_, t_, label=f"{name} ({roc_auc_score(yte, pr['test']):.3f})")
    pp, rr, _ = precision_recall_curve(yte, pr["test"]); ax[1].plot(rr, pp, label=f"{name} ({average_precision_score(yte, pr['test']):.3f})")
ax[0].plot([0, 1], [0, 1], "k:", lw=.8); ax[0].set_xlabel("false positive rate"); ax[0].set_ylabel("true positive rate"); ax[0].set_title("ROC (test)")
ax[1].set_xlabel("recall"); ax[1].set_ylabel("precision"); ax[1].set_title("Precision-recall (test)")
ax[0].legend(fontsize=7, loc="lower right"); ax[1].legend(fontsize=7, loc="lower left")
plt.tight_layout(); plt.savefig(f"{a.out}/fig_roc_pr.png"); plt.close()

# 6d learning curve (train vs validation accuracy against training size)
sizes = [0.05, 0.15, 0.3, 0.6, 1.0]; trA, vaA = [], []
rs = np.random.RandomState(SEED)
for f in sizes:
    sub = rs.permutation(len(Xtr))[: int(f * len(Xtr))]
    m = make_ebm(**ebm_params, bags=4).fit(Xtr.iloc[sub], ytr[sub])
    trA.append(accuracy_score(ytr[sub], m.predict(Xtr.iloc[sub]))); vaA.append(accuracy_score(yva, m.predict(Xva)))
fig, ax = plt.subplots(figsize=(5, 3.3)); xs = [int(f * len(Xtr)) for f in sizes]
ax.plot(xs, trA, "o-", label="train"); ax.plot(xs, vaA, "o-", label="validation")
ax.set_xlabel("training rows"); ax.set_ylabel("accuracy"); ax.set_title("EBM learning curve"); ax.legend()
plt.tight_layout(); plt.savefig(f"{a.out}/fig_learning_curve.png"); plt.close()

# 6e rolling-origin
fig, ax = plt.subplots(figsize=(6, 3.3)); xs = np.arange(len(roll))
ax.bar(xs - .2, roll.acc_test, .4, label="EBM", color="#c44e52"); ax.bar(xs + .2, roll.acc_persistence, .4, label="Persistence", color="#999999")
ax.set_xticks(xs); ax.set_xticklabels(roll.window, fontsize=7); ax.set_ylim(.9, 1); ax.set_ylabel("test accuracy"); ax.legend()
ax.set_title("Stability across four test windows"); plt.tight_layout(); plt.savefig(f"{a.out}/fig_rolling_origin.png"); plt.close()

# ------------------------------------------------------------------ 7. EBM interpretation
glob = ebm.explain_global()
imp = pd.DataFrame({"term": ebm.term_names_, "importance": ebm.term_importances()}).sort_values("importance", ascending=False)
imp.to_csv(f"{a.out}/ebm_term_importance.csv", index=False)
top = imp.head(15).iloc[::-1]
plt.figure(figsize=(6, 4.5)); plt.barh(top.term, top.importance, color="#c44e52")
plt.xlabel("mean |contribution| (log-odds)"); plt.title("EBM global feature importance"); plt.tight_layout()
plt.savefig(f"{a.out}/fig_ebm_importance.png"); plt.close()

single = [t for t in imp.term if " & " not in t][:9]
fig, axs = plt.subplots(3, 3, figsize=(11, 8.5)); axs = axs.ravel()
for ax, t in zip(axs, single):
    dd = glob.data(ebm.term_names_.index(t)); sc = np.array(dd["scores"], float)
    if t in SURVEY_CAT:
        ax.bar(range(len(sc)), sc); ax.set_xticks(range(len(sc))); ax.set_xticklabels([str(n)[:12] for n in dd["names"]], rotation=60, ha="right", fontsize=7)
    else:
        e = np.array(dd["names"], float)
        ax.step(e, np.append(sc, sc[-1]), where="post", color="#c44e52")
        ax.fill_between(e, np.append(dd["lower_bounds"], dd["lower_bounds"][-1]), np.append(dd["upper_bounds"], dd["upper_bounds"][-1]), step="post", alpha=.25)
    ax.axhline(0, c="grey", lw=.6); ax.set_title(t, fontsize=9); ax.set_ylabel("effect on log-odds", fontsize=7)
plt.tight_layout(); plt.savefig(f"{a.out}/fig_ebm_shape_functions.png"); plt.close()

contrib = ebm.eval_terms(Xte)
def local_plot(j, name):
    c = pd.Series(contrib[j], index=ebm.term_names_); c = c.reindex(c.abs().sort_values(ascending=False).index)[:10].iloc[::-1]
    plt.figure(figsize=(6.5, 3.8)); plt.barh(c.index, c.values, color=np.where(c.values > 0, "#c44e52", "#4c72b0")); plt.axvline(0, c="k", lw=.6)
    plt.title(f"{name}: P(>180 kWh) = {pe[j]:.2f} | last month {te.lag1.values[j]:.0f} kWh | actual {te.consumption.values[j]:.0f}", fontsize=8.5)
    plt.xlabel("contribution to log-odds of exceeding 180 kWh"); plt.tight_layout(); plt.savefig(f"{a.out}/fig_ebm_local_{name}.png"); plt.close()
jh, jb, jl = int(np.argmax(pe)), int(np.argmin(np.abs(pe - .5))), int(np.argmin(pe))
for n, j in [("high_risk", jh), ("borderline", jb), ("low_risk", jl)]: local_plot(j, n)

# LIME cross-check
levels = {c: sorted(d[c].unique()) for c in SURVEY_CAT}
enc = d[FEATS].copy()
for c in SURVEY_CAT: enc[c] = enc[c].map({v: i for i, v in enumerate(levels[c])})
med = enc.median(); enc = enc.fillna(med)
cidx = [FEATS.index(c) for c in SURVEY_CAT]
def pfn(arr):
    z = pd.DataFrame(arr, columns=FEATS)
    for c in SURVEY_CAT: z[c] = z[c].round().astype(int).clip(0, len(levels[c]) - 1).map(dict(enumerate(levels[c])))
    return ebm.predict_proba(z)
lx = LimeTabularExplainer(enc.loc[tr.index].sample(20000, random_state=SEED).values, feature_names=FEATS, categorical_features=cidx,
                          categorical_names={FEATS.index(c): levels[c] for c in SURVEY_CAT}, class_names=["<=180", ">180"],
                          mode="classification", random_state=SEED)
lime_scores = []
for n, j in [("high_risk", jh), ("borderline", jb), ("low_risk", jl)]:
    ex = lx.explain_instance(enc.loc[te.index[j]].values, pfn, num_features=10, num_samples=3000)
    fig = ex.as_pyplot_figure(); fig.set_size_inches(7, 3.8); plt.tight_layout()
    plt.savefig(f"{a.out}/fig_lime_{n}.png"); plt.close(); lime_scores.append((n, round(ex.score, 3)))
print("LIME local-fit R2:", lime_scores)
print("\nTop EBM terms:\n", imp.head(10).round(3).to_string(index=False))

# ------------------------------------------------------------------ 8. save for app
joblib.dump(ebm, f"{a.out}/ebm_cls_model.pkl")
json.dump(dict(features=FEATS, num_features=NUM, cat_features=SURVEY_CAT, cat_levels=levels, threshold=THR,
               num_stats={c: dict(min=float(d[c].min()), max=float(d[c].max()), median=float(d[c].median())) for c in NUM},
               cat_mode={c: d[c].mode()[0] for c in SURVEY_CAT}, test_accuracy=float(res.loc[res.model == "EBM (main)", "acc_test"].iloc[0])),
          open(f"{a.out}/ebm_cls_meta.json", "w"), indent=1)
print("saved model + metadata")
