"""
Evaluation of the ablations, the grounding test, self-consistency uncertainty, the tuned and joint classifiers and the
zero-shot reference (run after ablations.py and zeroshot.py; missing variants are skipped).

Outputs (results/paper/): ablations.csv, grounding.csv, uncertainty.csv, direct_tuned.csv, direct_joint.csv,
figB6_ablations, figB7_grounding, figB8_uncertainty (.pdf/.png) and numbers_b2.tex (macros \\NBB{a}{b}).
"""
import json, logging
import numpy as np, pandas as pd
from scipy.stats import spearmanr
from common import RES, WORK, auc, boot_auc_ci, calibration
from report_scoring import parse, per_patient, summarize, boot, mcnemar, TERR

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)
plt.rcParams.update({"font.family": ["Arial", "Liberation Sans", "DejaVu Sans"], "font.size": 7, "axes.linewidth": 0.6,
                     "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False, "pdf.fonttype": 42,
                     "xtick.major.width": 0.6, "ytick.major.width": 0.6, "xtick.major.size": 2.5, "ytick.major.size": 2.5})
V = RES / "vlm"; A = V / "abl"; OUT = RES / "paper"; OUT.mkdir(parents=True, exist_ok=True)
R = pd.read_csv(V / "reports.csv").set_index("participant_id")
coh = pd.read_csv(RES / "cohort_all.csv").set_index("participant_id")
L = coh[coh.labelled]


def load(pattern):
    fs = [A / pattern.format(k=k) for k in range(5)]
    fs = [f for f in fs if f.exists()]
    return pd.concat([pd.read_csv(f) for f in fs]).set_index("participant_id") if len(fs) == 5 else None


main = pd.concat([pd.read_csv(V / f"gen_fold{k}.csv") for k in range(5)]).set_index("participant_id")
VARIANTS = [("Full model", main), ("Full model, seed 1", load("gen_seed1_fold{k}.csv")),
            ("Full model, seed 2", load("gen_seed2_fold{k}.csv")),
            ("No visual tokens", load("gen_noimage_fold{k}.csv")), ("No clinical sentence", load("gen_noclin_fold{k}.csv")),
            ("Pooled token only", load("gen_pooled_fold{k}.csv")), ("Ungrounded ImageNet features", load("gen_imnet_fold{k}.csv")),
            ("Qwen2.5-0.5B", load("gen_llm05b_fold{k}.csv")),
            ("No language pretraining", load("gen_scratch_fold{k}.csv")), ("Qwen2.5-3B", load("gen_llm3b_fold{k}.csv")),
            ("Zero-shot Qwen2.5-VL-3B", load("gen_zeroshot_qwen25-vl-3b-instruct_fold{k}.csv"))]
VARIANTS = [(n, g) for n, g in VARIANTS if g is not None]
print("variants found:", [n for n, _ in VARIANTS])
ids = main[main.has_report].index
T = R.loc[ids].copy(); T["radc"] = coh.loc[ids, "acute_radc_mean"].values
Ps = {n: pd.DataFrame([parse(t) for t in g.reindex(ids).generated.fillna("")], index=ids) for n, g in VARIANTS}
Qs = {n: per_patient(P, T) for n, P in Ps.items()}
BT = boot(Qs, B=500, ref="Full model")
rows = []
lab = L.index
y = L.y_poor.values.astype(int)
for n, g in VARIANTS:
    P = Ps[n]; r = {"variant": n}
    sub = BT[BT.source == n].set_index("metric")
    for c in ["acc_side", "acc_vol_bin", "acc_restr", "terr_f1", "terr_exact", "halluc", "omission", "all_correct"]:
        r[c], r[c + "_lo"], r[c + "_hi"] = sub.loc[c, "value"], sub.loc[c, "lo"], sub.loc[c, "hi"]
        if "d_ref" in sub:
            r[c + "_d"], r[c + "_dlo"], r[c + "_dhi"] = sub.loc[c, "d_ref"], sub.loc[c, "d_lo"], sub.loc[c, "d_hi"]
    ok = P.vol_ml.notna(); r["vol_rho"] = spearmanr(P.vol_ml[ok], T.vol_ml[ok]).correlation if ok.sum() > 10 else np.nan
    okr = P.radc.notna() & T.radc.notna(); r["radc_rho"] = spearmanr(P.radc[okr], T.radc[okr]).correlation if okr.sum() > 10 else np.nan
    r["wellformed"] = P.wellformed.mean()
    if n != "Full model":
        r["mcn_full_better"], r["mcn_var_better"], r["mcn_p"] = mcnemar(Qs["Full model"]["all"], Qs[n]["all"])
    p = g.reindex(lab).p_poor
    if p.notna().all():
        r["auc"], (r["auc_lo"], r["auc_hi"]) = auc(y, p.values), boot_auc_ci(y, p.values)
        r["cal_slope"] = calibration(y, p.values)[1]
    if "p_poor_direct" in g and g.reindex(lab).p_poor_direct.notna().all():
        r["auc_direct"] = auc(y, g.reindex(lab).p_poor_direct.values)
    rows.append(r)
AB = pd.DataFrame(rows); AB.round(4).to_csv(OUT / "ablations.csv", index=False)
print(AB[[c for c in ["variant", "acc_side", "terr_f1", "halluc", "all_correct", "vol_rho", "radc_rho", "wellformed", "auc"] if c in AB]].round(3).to_string())

# ------------------------------------------------------------------------------------------- tuned direct classifier
D = pd.concat([pd.read_csv(f) for f in sorted(V.glob("direct_fold*.csv"))]).set_index("participant_id").reindex(ids)
fold = R.loc[ids, "vlm_fold"].values
Qt = pd.DataFrame(index=ids)
Qt["side"] = np.array(list("LRB"))[D[["p_side_L", "p_side_R", "p_side_B"]].values.argmax(1)]
Qt["vol_bin"] = D[[f"p_vol_{i}" for i in range(4)]].values.argmax(1)
Qt["restr"] = D[[f"p_restr_{i}" for i in range(3)]].values.argmax(1)
grid = np.round(np.arange(0.05, 0.96, 0.05), 2)
for col, tcol, kind in [("multiple", "p_mult", "acc"), ("chronic", "p_chron", "acc")] + [(f"terr_{t}", f"p_terr_{t}", "f1") for t in TERR]:
    pred = np.zeros(len(ids), int); truth = T[col].values.astype(int)
    for k in range(5):
        tr, te = fold != k, fold == k
        def score(th):
            p = (D[tcol].values[tr] > th).astype(int); t = truth[tr]
            if kind == "acc":
                return (p == t).mean()
            tp = (p & t).sum(); return 2 * tp / max(1, 2 * tp + (p & (1 - t)).sum() + ((1 - p) & t).sum())
        th = grid[int(np.argmax([score(g) for g in grid]))]
        pred[te] = (D[tcol].values[te] > th).astype(int)
    Qt[col] = pred
Qtun = per_patient(Qt, T)
DT = boot({"Full model": Qs["Full model"], "Direct, tuned thresholds": Qtun}, B=500, ref="Full model")
DT.round(4).to_csv(OUT / "direct_tuned.csv", index=False)
Qdef = pd.DataFrame(index=ids)
Qdef["side"], Qdef["vol_bin"], Qdef["restr"] = Qt["side"], Qt["vol_bin"], Qt["restr"]
Qdef["multiple"] = (D.p_mult > 0.5).astype(int).values; Qdef["chronic"] = (D.p_chron > 0.5).astype(int).values
for t in TERR:
    Qdef[f"terr_{t}"] = (D[f"p_terr_{t}"] > 0.5).astype(int).values
MC = {"default": mcnemar(Qs["Full model"]["all"], per_patient(Qdef, T)["all"]), "tuned": mcnemar(Qs["Full model"]["all"], Qtun["all"])}

# ------------------------------------------------------------------------------------------- joint (set-level) classifier
# A non-language model that also makes one joint decision: multinomial logistic regression on the same pooled encoder
# vector, predicting the combination of side and territory set as one class (combinations seen at least 5 times in
# the training folds; a scan predicted as "rare" falls back to the tuned classifier). Volume class, multiplicity,
# restriction and chronic infarct come from the tuned classifier.
JT = None
try:
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    cfile = WORK / "cache" / "index.csv"
    cidx = pd.read_csv(cfile if cfile.exists() else V / "cache_index.csv").set_index("participant_id")
    rows_ = cidx.loc[ids, "row"].values.astype(int)
    combo = T["side"].astype(str) + "|" + T[[f"terr_{t}" for t in TERR]].astype(int).astype(str).agg("".join, axis=1)
    Qj = Qt.copy()
    for k in range(5):
        pf = V / f"pooled_fold{k}.npy"       # pooled token saved by analysis.py --pool (same values as feats[:, -1])
        Fk = np.load(pf) if pf.exists() else np.load(WORK / "vlm" / f"feats_fold{k}.npy", mmap_mode="r")[:, -1]
        X = np.asarray(Fk[rows_], dtype=np.float32)
        tr, te = fold != k, fold == k
        lab_ = combo.values.copy(); vc = pd.Series(lab_[tr]).value_counts(); rare = ~pd.Series(lab_).isin(vc[vc >= 5].index).values
        lab_[rare] = "rare"
        sc = StandardScaler().fit(X[tr]); m = LogisticRegression(C=0.1, max_iter=3000).fit(sc.transform(X[tr]), lab_[tr])
        pr = m.predict(sc.transform(X[te]))
        for i, c in zip(np.where(te)[0], pr):
            if c != "rare":
                sd, bits = c.split("|"); Qj.iloc[i, Qj.columns.get_loc("side")] = sd
                for t, b in zip(TERR, bits):
                    Qj.iloc[i, Qj.columns.get_loc(f"terr_{t}")] = int(b)
    Qjoint = per_patient(Qj, T)
    JT = boot({"Full model": Qs["Full model"], "Joint set classifier": Qjoint}, B=500, ref="Full model")
    JT.round(4).to_csv(OUT / "direct_joint.csv", index=False)
    MC["joint"] = mcnemar(Qs["Full model"]["all"], Qjoint["all"])
    print("joint classifier:", JT[JT.source == "Joint set classifier"][["metric", "value", "d_ref", "d_lo", "d_hi"]].round(3).to_string())
except FileNotFoundError as e:
    print("joint classifier skipped:", e)

# ------------------------------------------------------------------------------------------- grounding test
GR = None
sh = load("gen_full_shuffle_fold{k}.csv")
if sh is not None:
    sh = sh.reindex(ids)
    Psh = pd.DataFrame([parse(t) for t in sh.generated.fillna("")], index=ids)
    Td = R.loc[sh.donor.values].copy(); Td.index = ids
    Pfull = Ps["Full model"]
    conds = {"Own image, own reference": per_patient(Pfull, T), "Other image, that image's reference": per_patient(Psh, Td),
             "Other image, own reference": per_patient(Psh, T), "Own image, other scan's reference (chance)": per_patient(Pfull, Td)}
    gb = boot(conds, B=500, ref="Own image, own reference")
    GR = gb; gb.round(4).to_csv(OUT / "grounding.csv", index=False)
    changed = {f: float((Psh[f].values != Pfull[f].values).mean()) for f in ["side", "vol_bin", "restr"]}
    changed["terr"] = float((Psh[[f"terr_{t}" for t in TERR]].values != Pfull[[f"terr_{t}" for t in TERR]].values).any(1).mean())
    pd.Series(changed).to_csv(OUT / "grounding_changed.csv")
    ps = sh.reindex(lab).p_poor if sh.reindex(lab).p_poor.notna().all() else None

# ------------------------------------------------------------------------------------------- self-consistency
UN = None
smp = load("gen_full_fold{k}.csv")
if smp is not None and smp.samples.notna().any():
    smp = smp.reindex(ids)
    key = lambda d: (d["side"], d["vol_bin"], tuple(d[f"terr_{t}"] for t in TERR))
    agree = []
    for g_txt, s_json in zip(smp.generated.fillna(""), smp.samples.fillna("[]")):
        ss = json.loads(s_json) if isinstance(s_json, str) and s_json.startswith("[") else []
        kg = key(parse(g_txt)); ks = [key(parse(s)) for s in ss]
        agree.append(np.mean([k == kg for k in ks]) if ks else np.nan)
    U = pd.DataFrame({"agreement": agree, "correct": per_patient(pd.DataFrame([parse(t) for t in smp.generated.fillna("")], index=ids), T)["all"].values}, index=ids)
    U = U.dropna(); U.to_csv(OUT / "uncertainty.csv")
    order = np.argsort(-U.agreement.values + 1e-9 * np.random.default_rng(0).random(len(U)))
    cov = np.arange(1, len(U) + 1) / len(U); acc = np.cumsum(U.correct.values[order]) / np.arange(1, len(U) + 1)
    _lo, _hi = boot_auc_ci(1 - U.correct.values.astype(int), 1 - U.agreement.values)
    UN = dict(cov=cov, acc=acc, auroc=auc(1 - U.correct.values, 1 - U.agreement.values), auroc_lo=_lo, auroc_hi=_hi,
              acc50=float(acc[int(0.5 * len(U)) - 1]), acc80=float(acc[int(0.8 * len(U)) - 1]), acc100=float(acc[-1]),
              full_agree=float((U.agreement == 1).mean()), acc_full_agree=float(U.correct[U.agreement == 1].mean()),
              acc_low_agree=float(U.correct[U.agreement < 0.6].mean()) if (U.agreement < 0.6).any() else np.nan)

# ------------------------------------------------------------------------------------------- figures
def save(fig, name):
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight", pad_inches=0.02)
    fig.savefig(OUT / f"{name}.png", dpi=300, bbox_inches="tight", pad_inches=0.02); plt.close(fig)


if len(AB) > 1:
    pan = [("all_correct", "All key fields correct"), ("terr_f1", "Territory F1"), ("halluc", "Hallucinated territories"),
           ("auc", "Outcome AUC")]
    # non-language references on the same encoder, shown below the language-model variants
    ABp = AB.copy()
    for nm_, tab_, src_ in [("Direct classifier, tuned", DT, "Direct, tuned thresholds"), ("Joint set classifier", JT, "Joint set classifier")]:
        if tab_ is not None:
            sub_ = tab_[tab_.source == src_].set_index("metric"); row_ = {"variant": nm_}
            for c_ in ["all_correct", "terr_f1", "halluc"]:
                if c_ in sub_.index:
                    row_[c_], row_[c_ + "_lo"], row_[c_ + "_hi"] = sub_.loc[c_, "value"], sub_.loc[c_, "lo"], sub_.loc[c_, "hi"]
            ABp = pd.concat([ABp, pd.DataFrame([row_])], ignore_index=True)
    fig, ax = plt.subplots(1, 4, figsize=(7.16, 0.22 * len(ABp) + 0.9), sharey=True, gridspec_kw=dict(wspace=0.12))
    AB_full = AB; AB = ABp
    yy = np.arange(len(AB))
    for a, (c, t) in zip(ax, pan):
        if c not in AB:
            a.set_visible(False); continue
        v = AB[c].values.astype(float); lo = AB.get(c + "_lo", AB[c]).values.astype(float); hi = AB.get(c + "_hi", AB[c]).values.astype(float)
        col = ["#d95f02" if n == "Full model" else "#7570b3" if n.startswith("Zero") else "#1f78b4" if n in ("Direct classifier, tuned", "Joint set classifier") else "#4d4d4d" for n in AB.variant]
        for i in range(len(AB)):
            if np.isfinite(v[i]):
                a.errorbar(v[i], yy[i], xerr=[[v[i] - lo[i]], [hi[i] - v[i]]], fmt="o", ms=3, color=col[i], lw=0.8, capsize=0)
        if np.isfinite(v[0]):
            a.axvline(v[0], color="#d95f02", lw=0.6, ls=":")
        a.set_title(t, fontsize=7)
    ax[0].set_yticks(yy); ax[0].set_yticklabels(AB.variant); ax[0].invert_yaxis()
    save(fig, "figB6_ablations")
    AB = AB_full

if GR is not None:
    mets = [("acc_side", "Side"), ("acc_vol_bin", "Volume class"), ("acc_restr", "Restriction"), ("terr_f1", "Territory F1"), ("all_correct", "All key fields")]
    conds = list(dict.fromkeys(GR.source))
    cols = ["#d95f02", "#1f78b4", "#a6cee3", "#bdbdbd"]
    fig, ax = plt.subplots(figsize=(7.16, 2.1)); w = 0.2
    for j, c in enumerate(conds):
        sub = GR[GR.source == c].set_index("metric")
        x = np.arange(len(mets)) + (j - 1.5) * w
        v = [sub.loc[m, "value"] for m, _ in mets]
        ax.bar(x, v, w * 0.92, color=cols[j], label=c)
        ax.errorbar(x, v, yerr=[np.array(v) - [sub.loc[m, "lo"] for m, _ in mets], np.array([sub.loc[m, "hi"] for m, _ in mets]) - v],
                    fmt="none", ecolor="black", elinewidth=0.6)
    ax.set_xticks(range(len(mets))); ax.set_xticklabels([l for _, l in mets]); ax.set_ylim(0, 1); ax.set_ylabel("Agreement (95% CI)")
    ax.legend(ncol=2, fontsize=6, loc="lower center", bbox_to_anchor=(0.5, 1.0))
    save(fig, "figB7_grounding")

if UN is not None:
    fig, ax = plt.subplots(1, 2, figsize=(7.16, 2.2), gridspec_kw=dict(wspace=0.3))
    ax[0].plot(UN["cov"], UN["acc"], color="#d95f02", lw=1.4); ax[0].axhline(UN["acc100"], color="#777777", lw=0.6, ls=":")
    ax[0].set(xlabel="Coverage (most consistent reports first)", ylabel="All key fields correct", xlim=(0, 1), ylim=(0, 1))
    bins = [0, 0.4, 0.6, 0.8, 0.99, 1.0]; U2 = pd.read_csv(OUT / "uncertainty.csv")
    cut = pd.cut(U2.agreement, [-0.01, 0.39, 0.59, 0.79, 0.99, 1.0], labels=["0 to 0.2", "0.4", "0.6", "0.8", "1.0"])
    g = U2.groupby(cut, observed=False).correct.agg(["mean", "size"])
    ax[1].bar(range(len(g)), g["mean"], color="#d95f02", width=0.6)
    ax[1].set_xticks(range(len(g))); ax[1].set_xticklabels([f"{i}\n({n})" for i, n in zip(g.index, g["size"])], fontsize=6)
    ax[1].set(xlabel="Agreement of 5 sampled reports with the greedy report (scans)", ylabel="All key fields correct", ylim=(0, 1))
    for a, s in zip(ax, "ab"):
        a.text(-0.16, 1.04, s, transform=a.transAxes, fontsize=8, fontweight="bold")
    save(fig, "figB8_uncertainty")

# ------------------------------------------------------------------------------------------- macros
mac = ["% generated by src/ablation_eval.py",
       "\\providecommand{\\NBB}[2]{\\ifcsname nbb@#1@#2\\endcsname\\csname nbb@#1@#2\\endcsname\\else\\textbf{\\textcolor{red}{??}}\\fi}"]
put = lambda a, b, v, f="{:.3f}": mac.append(f"\\expandafter\\def\\csname nbb@{a}@{b}\\endcsname{{{f.format(v)}}}") if v == v else None
short = {"Full model": "full", "Full model, seed 1": "seedone", "Full model, seed 2": "seedtwo", "No visual tokens": "noimg",
         "No clinical sentence": "noclin", "Pooled token only": "pooled", "Ungrounded ImageNet features": "imnet",
         "Qwen2.5-0.5B": "small", "No language pretraining": "scratch", "Qwen2.5-3B": "large", "Zero-shot Qwen2.5-VL-3B": "zeroshot"}
for _, r in AB.iterrows():
    s = short[r.variant]
    for c, v in r.items():
        if c != "variant" and isinstance(v, (int, float, np.floating)) and pd.notna(v):
            put(s, c.replace("_", ""), float(v), "{:.0f}" if c.startswith("mcn_") and not c.endswith("p") else "{:.3f}")
sub = DT[DT.source == "Direct, tuned thresholds"].set_index("metric")
for c in ["acc_side", "terr_f1", "terr_exact", "halluc", "omission", "all_correct"]:
    put("tuned", c.replace("_", ""), sub.loc[c, "value"]); put("tuned", c.replace("_", "") + "d", sub.loc[c, "d_ref"])
    put("tuned", c.replace("_", "") + "dlo", sub.loc[c, "d_lo"]); put("tuned", c.replace("_", "") + "dhi", sub.loc[c, "d_hi"])
if JT is not None:
    sub = JT[JT.source == "Joint set classifier"].set_index("metric")
    for c in ["acc_side", "terr_f1", "halluc", "omission", "terr_exact", "all_correct"]:
        if c in sub.index:
            put("joint", c.replace("_", ""), sub.loc[c, "value"]); put("joint", c.replace("_", "") + "d", sub.loc[c, "d_ref"])
            put("joint", c.replace("_", "") + "dlo", sub.loc[c, "d_lo"]); put("joint", c.replace("_", "") + "dhi", sub.loc[c, "d_hi"])
for nm, (b10, b01, pv) in MC.items():
    put("mcn" + nm, "vlmonly", b10, "{:.0f}"); put("mcn" + nm, "dironly", b01, "{:.0f}"); put("mcn" + nm, "p", max(pv, 1e-4), "{:.4f}")
if GR is not None:
    cs = {"Own image, own reference": "own", "Other image, that image's reference": "donor",
          "Other image, own reference": "shownref", "Own image, other scan's reference (chance)": "chance"}
    for _, r in GR.iterrows():
        put("gr" + cs[r.source], r.metric.replace("_", ""), r.value)
    for f, v in changed.items():
        put("grchanged", f, 100 * v, "{:.0f}")
    if ps is not None:
        put("grshuffle", "auc", auc(y, ps.values))
if UN is not None:
    for kk in ["auroc", "auroc_lo", "auroc_hi", "acc50", "acc80", "acc100", "full_agree", "acc_full_agree", "acc_low_agree"]:
        put("unc", kk.replace("_", ""), UN[kk])
(OUT / "numbers_b2.tex").write_text("\n".join(m for m in mac if m) + "\n")
print("written", OUT)
