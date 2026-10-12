"""
First scoring of the generated reports against the mask-derived truth, and of the outcome predicted through language.
Also scores the direct structured classifier (encoder.py) as the reference.

Outputs: results/vlm/table_factuality.csv, table_outcome.csv, report.md, per-scan parsed_*.csv
"""
import re
import numpy as np, pandas as pd
from scipy.stats import spearmanr
from common import RES, auc, boot_auc_ci, paired_boot_delta, calibration

TERR = ["ACA", "MCA", "DEEP", "PCA", "THAL", "BS", "CEREB"]
KEY = {"ACA": "anterior cerebral", "MCA": "middle cerebral", "DEEP": "lenticulostriate", "PCA": "posterior cerebral",
       "THAL": "thalamic", "BS": "brainstem", "CEREB": "cerebellar"}
EDGES = [2, 20, 70]
V = RES / "vlm"


def parse(t):
    t = t if isinstance(t, str) else ""
    side = "L" if "the left side" in t else "R" if "the right side" in t else "B" if "both sides" in t else None
    m = re.search(r"volume ([\d.]+) mL", t)
    vol = float(m.group(1)) if m else np.nan
    restr = 2 if "Marked diffusion" in t else 1 if "Mild diffusion" in t else 0 if "No clear diffusion" in t else np.nan
    chron = 1 if "Chronic infarct also present" in t else 0 if "No chronic infarct" in t else np.nan
    d = dict(side=side, vol_ml=vol, vol_bin=int(np.searchsorted(EDGES, vol, side="right")) if vol == vol else np.nan,
             multiple=int("Multiple acute" in t), chronic=chron, restr=restr)
    d.update({f"terr_{k}": int(KEY[k] in t) for k in TERR})
    return d


def factuality(P, T, name):
    r = dict(source=name, n=len(T))
    for f in ["side", "vol_bin", "multiple", "chronic", "restr"]:
        r[f"acc_{f}"] = float((P[f].values == T[f].values).mean())
    pt, tt = P[[f"terr_{k}" for k in TERR]].values.astype(int), T[[f"terr_{k}" for k in TERR]].values.astype(int)
    tp, fp, fn = (pt & tt).sum(), (pt & (1 - tt)).sum(), ((1 - pt) & tt).sum()
    r["terr_micro_f1"] = float(2 * tp / max(1, 2 * tp + fp + fn))
    r["terr_exact_set"] = float((pt == tt).all(1).mean())
    r["terr_hallucination_rate"] = float(fp / max(1, pt.sum()))          # mentioned territories that are wrong
    r["terr_omission_rate"] = float(fn / max(1, tt.sum()))               # true territories not mentioned
    for i, k in enumerate(TERR):
        tpk, fpk, fnk = (pt[:, i] & tt[:, i]).sum(), (pt[:, i] & (1 - tt[:, i])).sum(), ((1 - pt[:, i]) & tt[:, i]).sum()
        r[f"f1_{k}"] = float(2 * tpk / max(1, 2 * tpk + fpk + fnk))
    if "vol_ml" in P and P.vol_ml.notna().any():
        ok = P.vol_ml.notna().values
        r["vol_spearman"] = float(spearmanr(P.vol_ml.values[ok], T.vol_ml.values[ok]).correlation)
        r["vol_mae_ml"] = float(np.abs(P.vol_ml.values[ok] - T.vol_ml.values[ok]).mean())
    r["all_fields_correct"] = float(((P.side.values == T.side.values) & (P.vol_bin.values == T.vol_bin.values)
                                     & (pt == tt).all(1)).mean())
    return r


R = pd.read_csv(V / "reports.csv").set_index("participant_id")
G = pd.concat([pd.read_csv(f) for f in sorted(V.glob("gen_fold[0-9].csv"))]).set_index("participant_id")
rows, md = [], ["# VLM-Stroke results (auto-generated)\n"]

# ---------------- factuality: VLM text vs direct structured heads, on the same held-out scans
gr = G[G.has_report]
P = pd.DataFrame([parse(t) for t in gr.generated], index=gr.index)
P.to_csv(V / "parsed_vlm.csv"); T = R.loc[P.index]
rows.append(factuality(P, T, "VLM generated report"))
Dfiles = sorted(V.glob("direct_fold*.csv"))
if Dfiles:
    D = pd.concat([pd.read_csv(f) for f in Dfiles]).set_index("participant_id")
    D = D.loc[D.index.intersection(P.index)]
    Q = pd.DataFrame(index=D.index)
    Q["side"] = np.array(list("LRB"))[D[["p_side_L", "p_side_R", "p_side_B"]].values.argmax(1)]
    Q["vol_bin"] = D[[f"p_vol_{i}" for i in range(4)]].values.argmax(1)
    Q["multiple"] = (D.p_mult > 0.5).astype(int); Q["chronic"] = (D.p_chron > 0.5).astype(int)
    Q["restr"] = D[[f"p_restr_{i}" for i in range(3)]].values.argmax(1)
    for k in TERR:
        Q[f"terr_{k}"] = (D[f"p_terr_{k}"] > 0.5).astype(int)
    rows.append(factuality(Q, R.loc[Q.index], "Direct structured classifier"))
    rows.append(factuality(P.loc[Q.index], R.loc[Q.index], "VLM (same scans as direct)"))
# majority-class reference
M = pd.DataFrame(index=T.index)
for f in ["side", "vol_bin", "multiple", "chronic", "restr"]:
    M[f] = T[f].mode()[0]
for k in TERR:
    M[f"terr_{k}"] = int(T[f"terr_{k}"].mean() > 0.5)
rows.append(factuality(M, T, "Majority-class reference"))
FT = pd.DataFrame(rows).set_index("source").round(3)
FT.to_csv(V / "table_factuality.csv")
md += ["## Report factuality (held-out folds)\n", "```", FT.T.to_string(), "```\n"]

# ---------------- outcome through language vs the outcome models, same patients
gl = G[G.labelled & G.p_poor.notna()]
y = gl.y_poor.values.astype(int); p = gl.p_poor.values
out = [dict(model="VLM (image + clinical text)", n=len(y), auc=auc(y, p), ci=tuple(np.round(boot_auc_ci(y, p), 3)),
            cal_slope=calibration(y, p)[1])]
for name, f in [("Clinical logistic (R0)", "baseline_R0_clinical__cv5.csv"),
                ("Clinical + expert lesion (R1b)", "baseline_R1b_clin_vol_chronic_radc__cv5.csv")] + \
        [(f"2.5D anchored, {s}", f.name) for s, f in [("seed 0", RES / "oof" / "slice25d__cv5__s0.csv")] if f.exists()]:
    fp = RES / "oof" / f
    if fp.exists():
        B = pd.read_csv(fp).set_index("participant_id").loc[gl.index, "p_gt2"].values
        dlt, (lo, hi), pv = paired_boot_delta(y, p, B)
        out.append(dict(model=name, n=len(y), auc=auc(y, B), ci=tuple(np.round(boot_auc_ci(y, B), 3)),
                        cal_slope=calibration(y, B)[1], vlm_minus_this=dlt, delta_lo=lo, delta_hi=hi))
OT = pd.DataFrame(out).set_index("model").round(3)
OT.to_csv(V / "table_outcome.csv")
md += ["## Outcome (discharge mRS 3-6) predicted through language\n", "```", OT.to_string(), "```\n",
       "## Example generated vs reference reports\n"]
for pid in P.index[:5]:
    md += [f"**{pid}**", f"- generated: {G.loc[pid, 'generated']}", f"- reference: {R.loc[pid, 'report']}", ""]
(V / "report.md").write_text("\n".join(md))
print("\n".join(md))
