"""
Main evaluation of the grounded vision-language model (run after evaluate_reports.py).

  * patient-level bootstrap 95% CIs for every factuality metric and paired differences VLM - direct classifier
  * nearest-neighbor retrieval baseline (report of the most similar training scan in the encoder space)
  * text-overlap metrics (BLEU-4, ROUGE-L) next to the clinical-fact metrics
  * numeric fidelity of lesion volume and relative ADC; factuality by lesion size, field strength and sex
  * outcome through the report: clinical variables + facts parsed from the generated report (in-fold logistic model),
    compared with the reference models and with CALM-Stroke and 2.5D MIL from the companion study (results/oof)
Inputs: results/vlm/{reports,gen_fold*,direct_fold*}.csv, results/cohort_all.csv, results/oof/*.csv,
        results/vlm/pooled_fold{k}.npy (pooled encoder features, written by --pool)
Outputs: results/paper/ (figures, LaTeX tables, numbers_b.tex, csv)
Usage: python src/analysis.py            (analysis)
       python src/analysis.py --pool     (only extract pooled features from <work>/vlm/feats_fold*.npy)
"""
import re, sys, logging
import numpy as np, pandas as pd
from common import RES, WORK, auc, boot_auc_ci, paired_boot_delta, calibration

V = RES / "vlm"; OUT = RES / "paper"
if "--pool" in sys.argv:
    for k in range(5):
        F = np.load(WORK / "vlm" / f"feats_fold{k}.npy", mmap_mode="r")
        np.save(V / f"pooled_fold{k}.npy", np.asarray(F[:, -1, :], dtype=np.float16)); print("pooled", k, F.shape)
    import shutil
    shutil.copy(WORK / "cache" / "index.csv", V / "cache_index.csv")   # row of every scan in the pooled arrays
    sys.exit(0)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)
plt.rcParams.update({"font.family": ["Arial", "Liberation Sans", "DejaVu Sans"], "font.size": 7, "axes.linewidth": 0.6,
                     "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False, "pdf.fonttype": 42,
                     "xtick.major.width": 0.6, "ytick.major.width": 0.6, "xtick.major.size": 2.5, "ytick.major.size": 2.5})
OUT.mkdir(parents=True, exist_ok=True)
TERR = ["ACA", "MCA", "DEEP", "PCA", "THAL", "BS", "CEREB"]
TNAME = {"ACA": "ACA", "MCA": "MCA cortical", "DEEP": "Lenticulostriate", "PCA": "PCA", "THAL": "Thalamic",
         "BS": "Brainstem", "CEREB": "Cerebellar"}
KEY = {"ACA": "anterior cerebral", "MCA": "middle cerebral", "DEEP": "lenticulostriate", "PCA": "posterior cerebral",
       "THAL": "thalamic", "BS": "brainstem", "CEREB": "cerebellar"}
EDGES = [2, 20, 70]
COL = {"VLM": "#d95f02", "Direct": "#1f78b4", "Retrieval": "#7570b3", "Majority": "#9e9e9e"}


def parse(t):
    t = t if isinstance(t, str) else ""
    side = "L" if "the left side" in t else "R" if "the right side" in t else "B" if "both sides" in t else None
    m = re.search(r"volume ([\d.]+) mL", t); vol = float(m.group(1)) if m else np.nan
    m = re.search(r"relative ADC ([\d.]+)\)", t); radc = float(m.group(1)) if m else np.nan
    restr = 2 if "Marked diffusion" in t else 1 if "Mild diffusion" in t else 0 if "No clear diffusion" in t else np.nan
    chron = 1 if "Chronic infarct also present" in t else 0 if "No chronic infarct" in t else np.nan
    d = dict(side=side, vol_ml=vol, radc=radc, vol_bin=int(np.searchsorted(EDGES, vol, side="right")) if vol == vol else np.nan,
             multiple=int("Multiple acute" in t), chronic=chron, restr=restr,
             wellformed=int(t.startswith("FINDINGS:") and side is not None and vol == vol and restr == restr and chron == chron))
    d.update({f"terr_{k}": int(KEY[k] in t) for k in TERR})
    return d


FIELDS = ["side", "vol_bin", "multiple", "chronic", "restr"]


def per_patient(P, T):
    """Per-patient correctness indicators and territory counts (so metrics can be bootstrapped)."""
    Q = pd.DataFrame(index=T.index)
    for f in FIELDS:
        Q[f] = (P[f].values == T[f].values).astype(float)
    pt = P[[f"terr_{k}" for k in TERR]].values.astype(int); tt = T[[f"terr_{k}" for k in TERR]].values.astype(int)
    Q["tp"], Q["fp"], Q["fn"] = (pt & tt).sum(1), (pt & (1 - tt)).sum(1), ((1 - pt) & tt).sum(1)
    Q["npred"], Q["ntrue"] = pt.sum(1), tt.sum(1)
    Q["exact"] = (pt == tt).all(1).astype(float)
    Q["all"] = ((P.side.values == T.side.values) & (P.vol_bin.values == T.vol_bin.values) & (pt == tt).all(1)).astype(float)
    for i, k in enumerate(TERR):
        Q[f"tp_{k}"], Q[f"fp_{k}"], Q[f"fn_{k}"] = pt[:, i] & tt[:, i], pt[:, i] & (1 - tt[:, i]), (1 - pt[:, i]) & tt[:, i]
    return Q


def summarize(Q):
    r = {f"acc_{f}": Q[f].mean() for f in FIELDS}
    r["terr_f1"] = 2 * Q.tp.sum() / max(1, 2 * Q.tp.sum() + Q.fp.sum() + Q.fn.sum())
    r["terr_exact"] = Q.exact.mean(); r["all_correct"] = Q["all"].mean()
    r["halluc"] = Q.fp.sum() / max(1, Q.npred.sum()); r["omission"] = Q.fn.sum() / max(1, Q.ntrue.sum())
    for k in TERR:
        r[f"f1_{k}"] = 2 * Q[f"tp_{k}"].sum() / max(1, 2 * Q[f"tp_{k}"].sum() + Q[f"fp_{k}"].sum() + Q[f"fn_{k}"].sum())
    return r


def boot(Qs, B=1000, seed=0):
    """Qs: dict name -> per-patient frame on the same index. Returns point, CI and paired differences vs 'VLM'."""
    rng = np.random.default_rng(seed); n = len(next(iter(Qs.values())))
    pts = {m: summarize(Q) for m, Q in Qs.items()}; bs = {m: [] for m in Qs}
    for _ in range(B):
        i = rng.integers(0, n, n)
        for m, Q in Qs.items():
            bs[m].append(summarize(Q.iloc[i]))
    rows = []
    for m in Qs:
        Bm = pd.DataFrame(bs[m])
        for c, v in pts[m].items():
            r = dict(source=m, metric=c, value=v, lo=Bm[c].quantile(0.025), hi=Bm[c].quantile(0.975))
            if m != "VLM" and "VLM" in Qs:
                d = pd.DataFrame(bs["VLM"])[c] - Bm[c]
                r.update(d_vlm=pts["VLM"][c] - v, d_lo=d.quantile(0.025), d_hi=d.quantile(0.975))
            rows.append(r)
    return pd.DataFrame(rows)


# ------------------------------------------------------------------------------------------- load
R = pd.read_csv(V / "reports.csv").set_index("participant_id")
G = pd.concat([pd.read_csv(f) for f in sorted(V.glob("gen_fold[0-9].csv"))]).set_index("participant_id")
coh = pd.read_csv(RES / "cohort_all.csv").set_index("participant_id")
gr = G[G.has_report]
P = pd.DataFrame([parse(t) for t in gr.generated], index=gr.index)
Tt = R.loc[P.index].copy()
Tt["radc"] = coh.loc[P.index, "acute_radc_mean"].values
D = pd.concat([pd.read_csv(f) for f in sorted(V.glob("direct_fold*.csv"))]).set_index("participant_id").loc[P.index]
Qd = pd.DataFrame(index=D.index)
Qd["side"] = np.array(list("LRB"))[D[["p_side_L", "p_side_R", "p_side_B"]].values.argmax(1)]
Qd["vol_bin"] = D[[f"p_vol_{i}" for i in range(4)]].values.argmax(1)
Qd["multiple"] = (D.p_mult > 0.5).astype(int); Qd["chronic"] = (D.p_chron > 0.5).astype(int)
Qd["restr"] = D[[f"p_restr_{i}" for i in range(3)]].values.argmax(1)
for k in TERR:
    Qd[f"terr_{k}"] = (D[f"p_terr_{k}"] > 0.5).astype(int)
M = pd.DataFrame(index=Tt.index)
for f in FIELDS:
    M[f] = Tt[f].mode()[0]
for k in TERR:
    M[f"terr_{k}"] = int(Tt[f"terr_{k}"].mean() > 0.5)

# retrieval baseline: nearest training scan (cosine, pooled encoder features of the same fold's encoder)
SRC = {"VLM": P, "Direct": Qd, "Majority": M}
idx = pd.read_csv(WORK / "cache" / "index.csv") if (WORK / "cache" / "index.csv").exists() else \
    (pd.read_csv(V / "cache_index.csv") if (V / "cache_index.csv").exists() else None)
pooled = [V / f"pooled_fold{k}.npy" for k in range(5)]
ret_txt = {}
if idx is not None and all(p.exists() for p in pooled):
    row_of = dict(zip(idx.participant_id, idx.row))
    for k in range(5):
        F = np.load(pooled[k]).astype(np.float32); F /= np.linalg.norm(F, axis=1, keepdims=True) + 1e-8
        tr = R[R.vlm_fold != k]; te = R[(R.vlm_fold == k) & R.index.isin(P.index)]
        Ftr = F[[row_of[s] for s in tr.index]]; Fte = F[[row_of[s] for s in te.index]]
        nn_i = (Fte @ Ftr.T).argmax(1)
        for s, j in zip(te.index, nn_i):
            ret_txt[s] = tr.report.iloc[j]
    Rt = pd.DataFrame([parse(ret_txt[s]) for s in P.index], index=P.index)
    SRC = {"VLM": P, "Direct": Qd, "Retrieval": Rt, "Majority": M}
else:
    print("pooled features not found: retrieval baseline skipped (run analysis.py --pool after encoder.py)")

Qs = {m: per_patient(S.loc[Tt.index], Tt) for m, S in SRC.items()}
BT = boot(Qs); BT.round(4).to_csv(OUT / "factuality_ci.csv", index=False)
print(BT[BT.metric.isin(["acc_side", "acc_vol_bin", "terr_f1", "all_correct", "halluc", "omission"])].round(3).to_string())

# ------------------------------------------------------------------------------------------- text metrics
def tokens(s):
    return re.findall(r"[a-z]+|\d+(?:\.\d+)?|[()/.,:]", (s or "").lower())


def bleu4(c, r):
    from collections import Counter
    c, r = tokens(c), tokens(r)
    if not c:
        return 0.0
    ps = []
    for n in range(1, 5):
        cn = Counter(tuple(c[i:i + n]) for i in range(len(c) - n + 1)); rn = Counter(tuple(r[i:i + n]) for i in range(len(r) - n + 1))
        ps.append((sum(min(v, rn[g]) for g, v in cn.items()) + 1) / (max(1, sum(cn.values())) + 1))   # add-one smoothing
    bp = 1.0 if len(c) > len(r) else np.exp(1 - len(r) / max(1, len(c)))
    return float(bp * np.exp(np.mean(np.log(ps))))


def rougel(c, r):
    a, b = tokens(c), tokens(r)
    if not a or not b:
        return 0.0
    L = np.zeros((len(a) + 1, len(b) + 1), int)
    for i in range(len(a)):
        for j in range(len(b)):
            L[i + 1, j + 1] = L[i, j] + 1 if a[i] == b[j] else max(L[i, j + 1], L[i + 1, j])
    l = L[-1, -1]; p, rr = l / len(a), l / len(b)
    return 0.0 if l == 0 else 2 * p * rr / (p + rr)


txt = {"VLM": gr.generated.reindex(P.index)}
if ret_txt:
    txt["Retrieval"] = pd.Series(ret_txt).reindex(P.index)
TM = []
for m, s in txt.items():
    b = np.array([bleu4(c, r) for c, r in zip(s, R.loc[P.index, "report"])])
    rl = np.array([rougel(c, r) for c, r in zip(s, R.loc[P.index, "report"])])
    TM.append(dict(source=m, bleu4=b.mean(), rougeL=rl.mean()))
TM = pd.DataFrame(TM); TM.round(4).to_csv(OUT / "text_metrics.csv", index=False); print(TM.round(3))

# ------------------------------------------------------------------------------------------- numeric fidelity, strata
from scipy.stats import spearmanr
ok = P.vol_ml.notna()
vol_rho = spearmanr(P.vol_ml[ok], Tt.vol_ml[ok]).correlation
okr = P.radc.notna() & Tt.radc.notna()
radc_rho = spearmanr(P.radc[okr], Tt.radc[okr]).correlation
radc_mode = P.radc.round(2).value_counts(normalize=True).iloc[0]
ratio = np.log2((P.vol_ml[ok] + 0.1) / (Tt.vol_ml[ok] + 0.1))
strata = []
for name, msk in [("< 2 mL", Tt.vol_bin == 0), ("2 to 20 mL", Tt.vol_bin == 1), ("20 to 70 mL", Tt.vol_bin == 2),
                  ("≥ 70 mL", Tt.vol_bin == 3), ("1.5 T", coh.loc[Tt.index, "field"] < 3), ("3 T", coh.loc[Tt.index, "field"] >= 3),
                  ("Female", coh.loc[Tt.index, "sex"] == "F"), ("Male", coh.loc[Tt.index, "sex"] == "M")]:
    for m, Q in Qs.items():
        if m in ("VLM", "Direct"):
            s = summarize(Q[msk.values]); strata.append(dict(stratum=name, n=int(msk.sum()), source=m, **s))
ST = pd.DataFrame(strata); ST.round(4).to_csv(OUT / "factuality_strata.csv", index=False)

# ------------------------------------------------------------------------------------------- outcome through the report
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_curve
L = coh[coh.labelled].copy()
gl = G.loc[G.index.intersection(L.index)]
Pall = pd.DataFrame([parse(t) if h else parse("") for t, h in zip(gl.generated, gl.has_report)], index=gl.index)
X = pd.DataFrame(index=L.index)
for c in ["age", "nihss", "priorstroke", "bmi"]:
    X[c] = L[c]
X["bmi_miss"] = X.bmi.isna().astype(float)
X["gen_logvol"] = np.log1p(Pall.vol_ml.reindex(L.index).fillna(0))
X["gen_has"] = Pall.vol_ml.reindex(L.index).notna().astype(float)
X["gen_restr"] = Pall.restr.reindex(L.index).fillna(0); X["gen_chronic"] = Pall.chronic.reindex(L.index).fillna(0)
X["gen_mult"] = Pall.multiple.reindex(L.index).fillna(0)
for k in TERR:
    X[f"gen_{k}"] = Pall[f"terr_{k}"].reindex(L.index).fillna(0)
X["gen_left"] = (Pall.side.reindex(L.index) == "L").astype(float); X["gen_bilat"] = (Pall.side.reindex(L.index) == "B").astype(float)
y = L.y_poor.values.astype(int); prep = np.zeros(len(L))
for k in range(5):
    tr, te = (L.fold != k).values, (L.fold == k).values
    Xtr, Xte = X[tr].copy(), X[te].copy(); med = Xtr.bmi.median(); Xtr["bmi"] = Xtr.bmi.fillna(med); Xte["bmi"] = Xte.bmi.fillna(med)
    sc = StandardScaler().fit(Xtr); m = LogisticRegression(C=0.3, max_iter=3000).fit(sc.transform(Xtr), y[tr])
    prep[te] = m.predict_proba(sc.transform(Xte))[:, 1]
ids = L.index
OC = {"VLM (outcome token after report)": G.loc[ids, "p_poor"].values, "Clinical + generated report findings": prep}
for nm, f in [("Clinical (R0)", "baseline_R0_clinical__cv5.csv"), ("Clinical + expert volume (R1)", "baseline_R1_clin_volume__cv5.csv"),
              ("Clinical + expert lesion (R1b)", "baseline_R1b_clin_vol_chronic_radc__cv5.csv")]:
    if (RES / "oof" / f).exists():
        OC[nm] = pd.read_csv(RES / "oof" / f).set_index("participant_id").loc[ids, "p_gt2"].values
for nm, stem in [("2.5D MIL + anchor", "slice25d"), ("CALM-Stroke", "calm25d")]:
    fs = sorted((RES / "oof").glob(f"{stem}__cv5__s*.csv"))
    ds = [pd.read_csv(f).set_index("participant_id") for f in fs]
    ds = [d for d in ds if set(ids) <= set(d.index)]
    if ds:
        OC[nm] = np.mean([d.loc[ids, "p_gt2"].values for d in ds], 0)
rows = []
ref = OC.get("Clinical (R0)")
for nm, p in OC.items():
    lo, hi = boot_auc_ci(y, p); ci, cs = calibration(y, p)
    r = dict(model=nm, auc=auc(y, p), lo=lo, hi=hi, cal_int=ci, cal_slope=cs)
    if ref is not None and nm != "Clinical (R0)":
        d, (dl, dh), pv = paired_boot_delta(y, p, ref); r.update(d_r0=d, d_lo=dl, d_hi=dh, p_r0=pv)
    rows.append(r)
OT = pd.DataFrame(rows); OT.round(4).to_csv(OUT / "outcome.csv", index=False); print(OT.round(3).to_string())

# ------------------------------------------------------------------------------------------- figures
def save(fig, name):
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight", pad_inches=0.02)
    fig.savefig(OUT / f"{name}.png", dpi=300, bbox_inches="tight", pad_inches=0.02); plt.close(fig)


srcs = list(SRC)
mets = [("acc_side", "Side"), ("acc_vol_bin", "Volume class"), ("acc_multiple", "Multiplicity"), ("acc_restr", "Diffusion\nrestriction"),
        ("acc_chronic", "Chronic\ninfarct"), ("terr_f1", "Territory F1"), ("terr_exact", "Territory set\nexact"), ("all_correct", "All key\nfields correct")]
fig, ax = plt.subplots(figsize=(7.16, 2.2))
w = 0.8 / len(srcs)
for j, m in enumerate(srcs):
    sub = BT[BT.source == m].set_index("metric")
    x = np.arange(len(mets)) + (j - (len(srcs) - 1) / 2) * w
    v = [sub.loc[c, "value"] for c, _ in mets]; lo = [sub.loc[c, "lo"] for c, _ in mets]; hi = [sub.loc[c, "hi"] for c, _ in mets]
    ax.bar(x, v, w * 0.92, color=COL[m], label={"VLM": "Grounded VLM", "Direct": "Direct classifier", "Retrieval": "Nearest-neighbor retrieval", "Majority": "Majority class"}[m])
    ax.errorbar(x, v, yerr=[np.array(v) - lo, np.array(hi) - v], fmt="none", ecolor="black", elinewidth=0.6, capsize=0)
ax.set_xticks(range(len(mets))); ax.set_xticklabels([l for _, l in mets]); ax.set_ylim(0, 1); ax.set_ylabel("Accuracy or F1 (95% CI)")
ax.legend(ncol=4, loc="lower center", bbox_to_anchor=(0.5, 1.0), fontsize=6.2)
save(fig, "figB3_factuality")

fig, ax = plt.subplots(1, 3, figsize=(7.16, 2.3), gridspec_kw=dict(wspace=0.42, width_ratios=[1.25, 1, 1]))
y0 = np.arange(len(TERR)); prev = [Tt[f"terr_{k}"].sum() for k in TERR]
for j, m in enumerate([s for s in srcs if s != "Majority"]):
    sub = BT[BT.source == m].set_index("metric")
    ax[0].barh(y0 + (j - 1) * 0.27, [sub.loc[f"f1_{k}", "value"] for k in TERR], 0.25, color=COL[m],
               label={"VLM": "Grounded VLM", "Direct": "Direct classifier", "Retrieval": "Retrieval"}[m])
ax[0].set_yticks(y0); ax[0].set_yticklabels([f"{TNAME[k]} ({p})" for k, p in zip(TERR, prev)]); ax[0].invert_yaxis()
ax[0].set_xlabel("F1 per territory"); ax[0].set_xlim(0, 1)
ax[0].legend(fontsize=5.6, loc="lower left", bbox_to_anchor=(-0.62, 1.0), ncol=3, handlelength=0.9, columnspacing=0.5, handletextpad=0.3)
okv = ok & (Tt.vol_ml > 0)
ax[1].scatter(Tt.vol_ml[okv] + 0.1, P.vol_ml[okv] + 0.1, s=3, alpha=0.35, color=COL["VLM"], lw=0)
ax[1].plot([0.1, 400], [0.1, 400], ls=":", color="#777777", lw=0.7); ax[1].set_xscale("log"); ax[1].set_yscale("log")
ax[1].set(xlabel="Expert lesion volume, mL", ylabel="Volume in generated report, mL")
ax[1].text(0.04, 0.93, f"Spearman ρ = {vol_rho:.2f}", transform=ax[1].transAxes, fontsize=6)
ax[2].scatter(Tt.radc[okr], P.radc[okr], s=3, alpha=0.35, color=COL["VLM"], lw=0)
ax[2].plot([0.3, 1.6], [0.3, 1.6], ls=":", color="#777777", lw=0.7); ax[2].set_xlim(0.3, 1.6); ax[2].set_ylim(0.3, 1.6)
ax[2].set(xlabel="Expert-mask mean rADC", ylabel="rADC in generated report")
ax[2].text(0.04, 0.93, f"Spearman ρ = {radc_rho:.2f}", transform=ax[2].transAxes, fontsize=6)
for a, s in zip(ax, "abc"):
    a.text(-0.55 if s == "a" else -0.22, 1.1 if s == "a" else 1.04, s, transform=a.transAxes, fontsize=8, fontweight="bold", va="bottom")
save(fig, "figB4_territory_numeric")

fig, ax = plt.subplots(1, 2, figsize=(7.16, 2.4), gridspec_kw=dict(wspace=0.3, width_ratios=[1, 1.25]))
oc_col = {"VLM (outcome token after report)": COL["VLM"], "Clinical + generated report findings": "#e7298a", "Clinical (R0)": "#9e9e9e",
          "Clinical + expert lesion (R1b)": "#3c3c3c", "2.5D MIL + anchor": "#1f78b4", "CALM-Stroke": "#7570b3",
          "Clinical + expert volume (R1)": "#bdbdbd"}
for nm, p in OC.items():
    if nm == "Clinical + expert volume (R1)":
        continue
    fpr, tpr, _ = roc_curve(y, p); ax[0].plot(fpr, tpr, lw=1.5 if nm.startswith("VLM") else 1.0, color=oc_col[nm], label=f"{nm}  {auc(y, p):.3f}")
ax[0].plot([0, 1], [0, 1], ls=":", color="#777777", lw=0.7); ax[0].set(xlabel="1 − specificity", ylabel="Sensitivity")
ax[0].legend(fontsize=5.6, loc="lower right", title="AUC", title_fontsize=5.6)
S2 = ST.pivot_table(index="stratum", columns="source", values="all_correct").reindex(["< 2 mL", "2 to 20 mL", "20 to 70 mL", "≥ 70 mL", "1.5 T", "3 T"])
xx = np.arange(len(S2))
for j, m in enumerate(["VLM", "Direct"]):
    ax[1].bar(xx + (j - 0.5) * 0.38, S2[m], 0.36, color=COL[m], label={"VLM": "Grounded VLM", "Direct": "Direct classifier"}[m])
nlab = ST.drop_duplicates("stratum").set_index("stratum").n.reindex(S2.index)
disp = {"< 2 mL": "below\n2 mL", "2 to 20 mL": "2 to 20\nmL", "20 to 70 mL": "20 to 70\nmL", "\u2265 70 mL": "70 mL\nor more", "1.5 T": "1.5 T\n", "3 T": "3 T\n"}
ax[1].set_xticks(xx); ax[1].set_xticklabels([f"{disp[s]}\n({n})" for s, n in zip(S2.index, nlab)], fontsize=5.8)
ax[1].set_xlabel("Expert lesion volume or field strength (number of reports)", fontsize=6.2)
ax[1].set_ylabel("Reports with all key fields correct"); ax[1].legend(fontsize=5.8); ax[1].set_ylim(0, 0.8)
ax[0].text(-0.2, 1.04, "a", transform=ax[0].transAxes, fontsize=8, fontweight="bold"); ax[1].text(-0.16, 1.04, "b", transform=ax[1].transAxes, fontsize=8, fontweight="bold")
save(fig, "figB5_outcome_strata")

# single-column versions used in the manuscript: outcome ROC, and accuracy by lesion size and field strength
fig, ax0 = plt.subplots(figsize=(3.45, 2.9))
for nm, p in OC.items():
    if nm == "Clinical + expert volume (R1)":
        continue
    fpr, tpr, _ = roc_curve(y, p); ax0.plot(fpr, tpr, lw=1.5 if nm.startswith("VLM") else 1.0, color=oc_col[nm], label=f"{nm}  {auc(y, p):.3f}")
ax0.plot([0, 1], [0, 1], ls=":", color="#777777", lw=0.7); ax0.set(xlabel="1 − specificity", ylabel="Sensitivity", xlim=(0, 1), ylim=(0, 1.01))
ax0.legend(fontsize=5.8, loc="lower right", title="AUC", title_fontsize=5.8)
save(fig, "figB5a_outcome_roc")
fig, ax1 = plt.subplots(figsize=(3.45, 2.5))
for j, m in enumerate(["VLM", "Direct"]):
    ax1.bar(xx + (j - 0.5) * 0.38, S2[m], 0.36, color=COL[m], label={"VLM": "Grounded VLM", "Direct": "Direct classifier"}[m])
ax1.set_xticks(xx); ax1.set_xticklabels([f"{disp[s]}\n({n})" for s, n in zip(S2.index, nlab)], fontsize=6)
ax1.axvline(3.5, color="#cccccc", lw=0.6)
ax1.set_xlabel("Expert lesion volume or field strength (reports)", fontsize=6.5)
ax1.set_ylabel("All key fields correct"); ax1.legend(fontsize=6); ax1.set_ylim(0, 0.8)
save(fig, "figB5b_strata")

# ------------------------------------------------------------------------------------------- numbers for the manuscript
mac = ["% generated by src/analysis.py",
       "\\providecommand{\\NB}[2]{\\ifcsname nb@#1@#2\\endcsname\\csname nb@#1@#2\\endcsname\\else\\textbf{\\textcolor{red}{??}}\\fi}"]
tex_num = lambda s: "$-$" + s[1:] if s.startswith("-") else s          # typeset minus signs
put = lambda a, b, v, f="{:.3f}": mac.append(f"\\expandafter\\def\\csname nb@{a}@{b}\\endcsname{{{tex_num(f.format(v))}}}")
short = {"VLM": "vlm", "Direct": "dir", "Retrieval": "ret", "Majority": "maj"}
for _, r in BT.iterrows():
    m, c = short[r.source], r.metric.replace("_", "")
    put(m, c, r.value); put(m, c + "lo", r.lo); put(m, c + "hi", r.hi)
    if pd.notna(r.get("d_vlm", np.nan)):
        put(m, c + "d", r.d_vlm); put(m, c + "dlo", r.d_lo); put(m, c + "dhi", r.d_hi)
for _, r in TM.iterrows():
    put(short[r.source], "bleu", r.bleu4); put(short[r.source], "rouge", r.rougeL)
put("num", "volrho", vol_rho, "{:.2f}"); put("num", "radcrho", radc_rho, "{:.2f}"); put("num", "radcmode", 100 * radc_mode, "{:.0f}")
put("num", "volratio", float(np.median(np.abs(ratio))), "{:.2f}")
put("num", "wellformed", 100 * P.wellformed.mean(), "{:.1f}"); put("num", "nrep", len(P), "{:.0f}")
okb = ok & (Tt.vol_ml > 0)
put("num", "volwithin2x", 100 * float((np.abs(ratio[okb.values[ok.values]]) <= 1).mean()), "{:.0f}")
okey = {"VLM (outcome token after report)": "vlm", "Clinical + generated report findings": "facts", "Clinical (R0)": "R0",
        "Clinical + expert volume (R1)": "R1", "Clinical + expert lesion (R1b)": "R1b", "2.5D MIL + anchor": "mil", "CALM-Stroke": "calm"}
for _, r in OT.iterrows():
    k = okey[r.model]
    for c in ["auc", "lo", "hi", "cal_slope", "cal_int", "d_r0", "d_lo", "d_hi"]:
        if c in r and pd.notna(r[c]):
            put("out" + k, c.replace("_", ""), r[c], "{:.2f}" if c.startswith("cal") else "{:.3f}")
for _, r in ST.iterrows():
    put("st" + short[r.source], re.sub(r"[^a-z0-9]", "", r.stratum.lower().replace("≥", "ge")), r.all_correct)
    put("stn", re.sub(r"[^a-z0-9]", "", r.stratum.lower().replace("≥", "ge")), r.n, "{:,.0f}")
(OUT / "numbers_b.tex").write_text("\n".join(mac) + "\n")
print("written", OUT)
