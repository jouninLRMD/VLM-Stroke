"""
Additional analyses of the saved outputs (run after analysis.py and ablation_eval.py).

  * equivalence of the primary endpoint (all key fields correct) with a margin of 0.03: two one-sided tests on the
    paired bootstrap (90% CI of the difference inside the margin)
  * image-swap test field by field, including multiplicity and chronic infarct
  * prior stroke in the clinical sentence against the chronic part of the expert masks and the generated reports
  * reference audit: diffusion restriction against vendor, "both sides" in brainstem and single-component lesions,
    left-right confusion of the generated side
  * results by circulation (anterior only, posterior only, both) and by lesion visibility on the 17 sampled MNI levels
  * encoder confidence against self-consistency for detecting report errors
  * outcome: seed spread, patients with and without an expert mask
  * subgroup results with bootstrap CIs

Inputs: results/vlm/*, results/paper/{outcome_predictions,uncertainty}.csv, results/cohort_all.csv and
        results/vlm/visibility.csv (written by `python src/review_analyses.py --visibility`, which reads the cache).
Outputs: results/paper/review_*.csv and numbers_b3.tex (macros \\NBR{a}{b}).
"""
import sys, json
import numpy as np, pandas as pd
from common import RES, WORK, auc, boot_auc_ci, paired_boot_delta
from report_scoring import parse, per_patient, summarize, mcnemar, TERR, FIELDS

V = RES / "vlm"; A = V / "abl"; OUT = RES / "paper"
NZ_SAMPLED = None

if "--visibility" in sys.argv:
    # lesion voxels on the 17 axial MNI levels that the encoder sees (same sampling as encoder.py)
    C = WORK / "cache"
    M = np.load(C / "M.npy", mmap_mode="r"); atl = np.load(C / "atlas.npy") > 0
    nz = M.shape[-1]; z = sorted(set(np.linspace(0.17 * nz, 0.85 * nz, 17).astype(int).tolist()))
    idx = pd.read_csv(C / "index.csv"); rows = []
    for r, pid in zip(idx.row, idx.participant_id):
        m = np.asarray(M[r, 0]) > 0; tot = int(m.sum())
        if tot == 0:
            rows.append((pid, 0, 0, 0, 0, 0)); continue
        zs = m.sum((0, 1))
        rows.append((pid, tot, int(zs[z].sum()), int((zs[z] > 0).sum()), int(zs[:z[0]].sum()), int((m & atl).sum())))
    pd.DataFrame(rows, columns=["participant_id", "vox", "vox_sampled", "levels_hit", "vox_below", "vox_in_atlas"]
                 ).to_csv(V / "visibility.csv", index=False)
    print("written", V / "visibility.csv"); sys.exit(0)

R = pd.read_csv(V / "reports.csv").set_index("participant_id")
coh = pd.read_csv(RES / "cohort_all.csv").set_index("participant_id")


def load(pattern, base=A):
    fs = [base / pattern.format(k=k) for k in range(5)]
    return pd.concat([pd.read_csv(f) for f in fs]).set_index("participant_id") if all(f.exists() for f in fs) else None


G = load("gen_fold{k}.csv", V)
ids = G[G.has_report].index
T = R.loc[ids].copy()
P = pd.DataFrame([parse(t) for t in G.loc[ids, "generated"].fillna("")], index=ids)
fold = R.loc[ids, "vlm_fold"].values

# ------------------------------------------------------------------------------------------- comparators
D = load("direct_fold{k}.csv", V).reindex(ids)
pt = D[[f"p_terr_{t}" for t in TERR]].values


def at_least_one(Q):
    e = Q[[f"terr_{t}" for t in TERR]].sum(axis=1).values == 0
    for j, t in enumerate(TERR):
        Q.loc[e, f"terr_{t}"] = (pt[e].argmax(1) == j).astype(int)
    return Q


Qd = pd.DataFrame(index=ids)
Qd["side"] = np.array(list("LRB"))[D[["p_side_L", "p_side_R", "p_side_B"]].values.argmax(1)]
Qd["vol_bin"] = D[[f"p_vol_{i}" for i in range(4)]].values.argmax(1)
Qd["restr"] = D[[f"p_restr_{i}" for i in range(3)]].values.argmax(1)
Qd["multiple"] = (D.p_mult > 0.5).astype(int).values; Qd["chronic"] = (D.p_chron > 0.5).astype(int).values
for t in TERR:
    Qd[f"terr_{t}"] = (D[f"p_terr_{t}"] > 0.5).astype(int).values
n_empty = int((Qd[[f"terr_{t}" for t in TERR]].sum(axis=1) == 0).sum())
Qd = at_least_one(Qd)

SRC = {"VLM": P, "Direct": Qd}
for nm, pat in [("Seed 1", "gen_seed1_fold{k}.csv"), ("Seed 2", "gen_seed2_fold{k}.csv"),
                ("No clinical sentence", "gen_noclin_fold{k}.csv"), ("No prior stroke in prompt", "gen_noprior_fold{k}.csv"),
                ("No language pretraining", "gen_scratch_fold{k}.csv"), ("Qwen2.5-0.5B", "gen_llm05b_fold{k}.csv"),
                ("Qwen2.5-3B", "gen_llm3b_fold{k}.csv"), ("Qwen2.5-3B, effective batch 8", "gen_llm3bacc_fold{k}.csv"),
                ("Pooled token only", "gen_pooled_fold{k}.csv"), ("ImageNet features, left-right tokens", "gen_imnetsp_fold{k}.csv")]:
    g = load(pat)
    if g is not None:
        SRC[nm] = pd.DataFrame([parse(t) for t in g.reindex(ids).generated.fillna("")], index=ids)
Q = {m: per_patient(S, T) for m, S in SRC.items()}

# joint classifier predictions are rebuilt exactly as in ablation_eval.py (same seed-free logistic model)
try:
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    cidx = pd.read_csv(V / "cache_index.csv").set_index("participant_id"); rows_ = cidx.loc[ids, "row"].values.astype(int)
    grid = np.round(np.arange(0.05, 0.96, 0.05), 2); Qt = Qd[["side", "vol_bin", "restr"]].copy()
    for col, tcol, kind in [("multiple", "p_mult", "acc"), ("chronic", "p_chron", "acc")] + [(f"terr_{t}", f"p_terr_{t}", "f1") for t in TERR]:
        pred = np.zeros(len(ids), int); truth = T[col].values.astype(int)
        for k in range(5):
            tr, te = fold != k, fold == k
            def score(th):
                p = (D[tcol].values[tr] > th).astype(int); t_ = truth[tr]
                if kind == "acc":
                    return (p == t_).mean()
                tp = (p & t_).sum(); return 2 * tp / max(1, 2 * tp + (p & (1 - t_)).sum() + ((1 - p) & t_).sum())
            th = grid[int(np.argmax([score(g) for g in grid]))]; pred[te] = (D[tcol].values[te] > th).astype(int)
        Qt[col] = pred
    Qt = at_least_one(Qt); Q["Direct, tuned"] = per_patient(Qt, T)
    combo = T["side"].astype(str) + "|" + T[[f"terr_{t}" for t in TERR]].astype(int).astype(str).agg("".join, axis=1)
    Qj = Qt.copy()
    for k in range(5):
        X = np.asarray(np.load(V / f"pooled_fold{k}.npy")[rows_], dtype=np.float32); tr, te = fold != k, fold == k
        lab_ = combo.values.copy(); vc = pd.Series(lab_[tr]).value_counts(); lab_[~pd.Series(lab_).isin(vc[vc >= 5].index).values] = "rare"
        sc = StandardScaler().fit(X[tr]); m = LogisticRegression(C=0.1, max_iter=3000).fit(sc.transform(X[tr]), lab_[tr])
        for i, c in zip(np.where(te)[0], m.predict(sc.transform(X[te]))):
            if c != "rare":
                sd, bits = c.split("|"); Qj.iloc[i, Qj.columns.get_loc("side")] = sd
                for t, b in zip(TERR, bits):
                    Qj.iloc[i, Qj.columns.get_loc(f"terr_{t}")] = int(b)
    Q["Joint set classifier"] = per_patient(Qj, T); SRC["Joint set classifier"] = Qj
except FileNotFoundError as e:
    print("joint classifier skipped:", e)

# ------------------------------------------------------------------------------------------- equivalence (TOST)
MARGIN = 0.03; B = 2000; rng = np.random.default_rng(0); n = len(ids)
BI = rng.integers(0, n, (B, n))
ref = Q["VLM"]["all"].values
EQ = []
for m, q in Q.items():
    if m == "VLM":
        continue
    a = q["all"].values; d = ref - a; bd = d[BI].mean(1)
    lo90, hi90 = np.quantile(bd, [0.05, 0.95]); lo95, hi95 = np.quantile(bd, [0.025, 0.975])
    se = bd.std(); from scipy.stats import norm
    p_tost = max(norm.sf((d.mean() + MARGIN) / se), norm.sf((MARGIN - d.mean()) / se))
    EQ.append(dict(source=m, value=a.mean(), d=d.mean(), lo95=lo95, hi95=hi95, lo90=lo90, hi90=hi90, p_tost=p_tost,
                   equivalent=bool(lo90 > -MARGIN and hi90 < MARGIN), mcn=mcnemar(ref, a)[2]))
EQ = pd.DataFrame(EQ); EQ.round(4).to_csv(OUT / "review_equivalence.csv", index=False); print(EQ.round(3).to_string())

# ------------------------------------------------------------------------------------------- image swap, every field
sh = load("gen_full_shuffle_fold{k}.csv").reindex(ids)
Psh = pd.DataFrame([parse(t) for t in sh.generated.fillna("")], index=ids)
Td = R.loc[sh.donor.values].copy(); Td.index = ids
SW = []
for f in FIELDS + ["terr_exact"]:
    if f == "terr_exact":
        own = (Psh[[f"terr_{t}" for t in TERR]].values == T[[f"terr_{t}" for t in TERR]].values).all(1)
        don = (Psh[[f"terr_{t}" for t in TERR]].values == Td[[f"terr_{t}" for t in TERR]].values).all(1)
        base = (P[[f"terr_{t}" for t in TERR]].values == T[[f"terr_{t}" for t in TERR]].values).all(1)
        chance = (P[[f"terr_{t}" for t in TERR]].values == Td[[f"terr_{t}" for t in TERR]].values).all(1)
        changed = (Psh[[f"terr_{t}" for t in TERR]].values != P[[f"terr_{t}" for t in TERR]].values).any(1)
    else:
        own, don = Psh[f].values == T[f].values, Psh[f].values == Td[f].values
        base, chance = P[f].values == T[f].values, P[f].values == Td[f].values
        changed = Psh[f].values != P[f].values
    SW.append(dict(field=f, normal=base.mean(), swap_donor=don.mean(), swap_own=own.mean(), chance=chance.mean(),
                   changed=changed.mean(), donor_differs=float((T[f].values != Td[f].values).mean()) if f != "terr_exact" else
                   float((T[[f"terr_{t}" for t in TERR]].values != Td[[f"terr_{t}" for t in TERR]].values).any(1).mean())))
SW = pd.DataFrame(SW); SW.round(4).to_csv(OUT / "review_swap_fields.csv", index=False); print(SW.round(3).to_string())
# the outcome token under the swap, restricted to patients whose image was swapped (scans with a mask)
lab = coh[coh.labelled].index
lm = [s for s in lab if s in ids]
swap_auc_masked = auc(coh.loc[lm, "y_poor"].values.astype(int), sh.loc[lm, "p_poor"].values) if sh.loc[lm, "p_poor"].notna().all() else np.nan

# ------------------------------------------------------------------------------------------- prior stroke and chronic infarct
ps = coh.loc[ids, "priorstroke"]
CH = dict(n_ps0=int((ps == 0).sum()), n_ps1=int((ps == 1).sum()),
          ref_chron_ps0=int(T.chronic[ps == 0].sum()), ref_chron_ps1=int(T.chronic[ps == 1].sum()),
          gen_chron_ps0=int((P.chronic == 1)[ps == 0].sum()), gen_chron_ps1=int((P.chronic == 1)[ps == 1].sum()))
allm = coh[coh.has_mask == 1]
CH.update(all_ps0=int((allm.priorstroke == 0).sum()), all_ps1=int((allm.priorstroke == 1).sum()),
          all_chron_ps0=int(((allm.chronic_ml_filled >= 1) & (allm.priorstroke == 0)).sum()),
          all_chron_ps1=int(((allm.chronic_ml_filled >= 1) & (allm.priorstroke == 1)).sum()))
CH["acc_chron_prior_rule"] = float(((ps == 1).astype(int).values == T.chronic.values).mean())
if "No prior stroke in prompt" in Q:
    CH["acc_chron_noprior"] = float(Q["No prior stroke in prompt"]["chronic"].mean())
    Pn = SRC["No prior stroke in prompt"]
    CH["noprior_gen_chron"] = int((Pn.chronic == 1).sum()); CH["noprior_gen_chron_tp"] = int(((Pn.chronic == 1) & (T.chronic == 1)).sum())
    CH["ref_chron"] = int(T.chronic.sum())
print(CH)

# ------------------------------------------------------------------------------------------- reference audit
mf = coh.loc[ids, "manufacturer"].astype(str).str.lower()
vend = np.where(mf.str.contains("philips"), "Philips", np.where(mf.str.contains("siemens"), "Siemens",
                np.where(mf.str.contains("ge|general"), "GE", "Other")))
radc = coh.loc[ids, "acute_radc_mean"]
AU = dict(n=len(ids), norestr=int((T.restr == 0).sum()), radc12=int((radc >= 1.2).sum()),
          multiple=int(T.multiple.sum()), both=int((T.side == "B").sum()),
          bs=int(T.terr_BS.sum()), bs_both=int(((T.side == "B") & (T.terr_BS == 1)).sum()),
          single_both=int(((T.side == "B") & (coh.loc[ids, "acute_n_components"] == 1)).sum()))
for v in ["Philips", "Siemens", "GE"]:
    mv = vend == v; AU[f"n_{v}"] = int(mv.sum()); AU[f"norestr_{v}"] = int((T.restr[mv] == 0).sum())
uni = T.side.isin(["L", "R"]) & P.side.isin(["L", "R"])
AU["lr_n"] = int(T.side.isin(["L", "R"]).sum()); AU["lr_swap"] = int((uni & (T.side != P.side)).sum())
AU["acc_side_unilat"] = float((P.side[T.side.isin(["L", "R"])] == T.side[T.side.isin(["L", "R"])]).mean())
AU["acc_side_both"] = float((P.side[T.side == "B"] == "B").mean())
AU["dir_lr_swap"] = int((T.side.isin(["L", "R"]) & Qd.side.isin(["L", "R"]) & (T.side != Qd.side)).sum())
conf = pd.crosstab(T.side, P.side).reindex(index=list("LRB"), columns=list("LRB")).fillna(0).astype(int)
conf.to_csv(OUT / "review_side_confusion.csv"); print(conf)
nm_ = {"L": "Left", "R": "Right", "B": "Both"}
tl = ["\\begin{table}[!htbp]", "\\caption{Side in the generated reports of the VLM against the side in the reference (1,451 scans).}",
      "\\label{tab:sideconf}", "\\centering", "\\footnotesize", "\\begin{tabular}{@{}lccc@{}}", "\\toprule",
      " & \\multicolumn{3}{c}{Generated side} \\\\", "Reference side & Left & Right & Both \\\\", "\\midrule"]
for a in "LRB":
    tl.append(f"{nm_[a]} & " + " & ".join(f"{conf.loc[a, b]:,}" for b in "LRB") + " \\\\")
tl += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
(OUT / "tab_sideconf.tex").write_text("\n".join(tl) + "\n")
print(AU)

# ------------------------------------------------------------------------------------------- circulation and visibility
ant = T[["terr_ACA", "terr_MCA", "terr_DEEP"]].sum(axis=1) > 0
post = T[["terr_PCA", "terr_THAL", "terr_BS", "terr_CEREB"]].sum(axis=1) > 0
circ = pd.Series(np.where(ant & ~post, "Anterior only", np.where(post & ~ant, "Posterior only", "Both circulations")), index=ids)
vis = pd.read_csv(V / "visibility.csv").set_index("participant_id").reindex(ids)
vis_frac = vis.vox_sampled / vis.vox.replace(0, np.nan)
vgrp = pd.Series(np.where(vis.levels_hit == 0, "Not on any sampled level",
                          np.where(vis_frac < 0.05, "Under 5% of voxels sampled", "5% or more sampled")), index=ids)
VIS = dict(none=int((vis.levels_hit == 0).sum()), below=int((vis.vox_below > 0).sum()),
           mostly_below=int((vis.vox_below / vis.vox > 0.5).sum()),
           none_small=int(((vis.levels_hit == 0) & (T.vol_bin == 0)).sum()), n_small=int((T.vol_bin == 0).sum()),
           none_post=int(((vis.levels_hit == 0) & post).sum()), n_post_only=int((circ == "Posterior only").sum()),
           outside_atlas=int(((vis.vox_in_atlas / vis.vox) < 0.8).sum()), median_frac=float(vis_frac.median()))
print(VIS)


def ci_mean(x, B=1000, seed=1):
    x = np.asarray(x, float); r = np.random.default_rng(seed); bs = x[r.integers(0, len(x), (B, len(x)))].mean(1)
    return x.mean(), *np.quantile(bs, [0.025, 0.975])


STR = []
groups = [("circ", g, circ == g) for g in ["Anterior only", "Posterior only", "Both circulations"]] + \
         [("vis", g, vgrp == g) for g in ["Not on any sampled level", "Under 5% of voxels sampled", "5% or more sampled"]] + \
         [("vol", g, T.vol_bin == i) for i, g in enumerate(["< 2 mL", "2 to 20 mL", "20 to 70 mL", "70 mL or more"])] + \
         [("field", "1.5 T", coh.loc[ids, "field"] < 3), ("field", "3 T", coh.loc[ids, "field"] >= 3),
          ("sex", "Female", coh.loc[ids, "sex"] == "F"), ("sex", "Male", coh.loc[ids, "sex"] == "M")]
for kind, g, msk in groups:
    msk = np.asarray(msk, bool)
    for m in ["VLM", "Direct", "Joint set classifier"]:
        if m not in Q:
            continue
        q = Q[m][msk]; v, lo, hi = ci_mean(q["all"]); s = summarize(q)
        STR.append(dict(kind=kind, group=g, source=m, n=int(msk.sum()), all_correct=v, lo=lo, hi=hi, acc_side=s["acc_side"],
                        terr_f1=s["terr_f1"], halluc=s["halluc"]))
    d = Q["VLM"]["all"].values[msk] - Q["Direct"]["all"].values[msk]; dv, dlo, dhi = ci_mean(d)
    STR.append(dict(kind=kind, group=g, source="VLM minus Direct", n=int(msk.sum()), all_correct=dv, lo=dlo, hi=dhi))
STR = pd.DataFrame(STR); STR.round(4).to_csv(OUT / "review_strata.csv", index=False)
print(STR[STR.source.isin(["VLM", "Direct", "VLM minus Direct"])].round(3).to_string())

# ------------------------------------------------------------------------------------------- confidence for error detection
U = pd.read_csv(OUT / "uncertainty.csv").set_index("participant_id").reindex(ids)
err = 1 - Q["VLM"]["all"].values
encc = D[["p_side_L", "p_side_R", "p_side_B"]].max(axis=1).values * D[[f"p_vol_{i}" for i in range(4)]].max(axis=1).values \
    * np.prod(np.maximum(pt, 1 - pt), 1)
okU = U.agreement.notna().values
UC = dict(auc_sc=auc(err[okU], 1 - U.agreement.values[okU]), auc_enc=auc(err, 1 - encc),
          auc_vol=auc(err, -np.log1p(P.vol_ml.fillna(0).values)))
r2 = np.random.default_rng(2); dd = []
for _ in range(2000):
    i = r2.integers(0, okU.sum(), okU.sum()); e_, a_, c_ = err[okU][i], U.agreement.values[okU][i], encc[okU][i]
    if 0 < e_.sum() < len(e_):
        dd.append(auc(e_, 1 - c_) - auc(e_, 1 - a_))
UC["d_enc_sc"], UC["d_lo"], UC["d_hi"] = auc(err[okU], 1 - encc[okU]) - UC["auc_sc"], *np.quantile(dd, [0.025, 0.975])
order = np.argsort(-encc + 1e-9 * np.random.default_rng(0).random(len(encc)))
UC["enc_acc50"] = float((1 - err[order][: len(err) // 2]).mean())
small = (T.vol_bin == 0).values & okU
UC["auc_sc_small"] = auc(err[small], 1 - U.agreement.values[small]); UC["auc_enc_small"] = auc(err[small], 1 - encc[small])
print(UC)

# ------------------------------------------------------------------------------------------- outcome
OP = pd.read_csv(OUT / "outcome_predictions.csv").set_index("participant_id")
y = OP.y_poor.values.astype(int); hm = OP.index.isin(ids)          # patients with a mask-derived report
OCR = dict(n_mask=int(hm.sum()), n_nomask=int((~hm).sum()), poor_mask=float(y[hm].mean()), poor_nomask=float(y[~hm].mean()))
seeds = {"s0": OP["VLM (outcome token after report)"].values}
for s in (1, 2):
    g = load(f"gen_seed{s}_fold{{k}}.csv")
    if g is not None:
        seeds[f"s{s}"] = g.loc[OP.index, "p_poor"].values
for k_, p in seeds.items():
    OCR[f"auc_{k_}"] = auc(y, p)
for nm, col in [("vlm", "VLM (outcome token after report)"), ("vlmseed", "VLM, mean of seeds"), ("r0", "Clinical (R0)"),
                ("facts", "Clinical + generated report findings"), ("r1b", "Clinical + expert lesion (R1b)")]:
    if col in OP:
        OCR[f"{nm}_mask"] = auc(y[hm], OP[col].values[hm]); OCR[f"{nm}_nomask"] = auc(y[~hm], OP[col].values[~hm])
for nm, col in [("vlmseed", "VLM, mean of seeds"), ("facts", "Clinical + generated report findings")]:
    d, (lo, hi), pv = paired_boot_delta(y[hm], OP[col].values[hm], OP["Clinical (R0)"].values[hm])
    OCR[f"{nm}_mask_d"], OCR[f"{nm}_mask_dlo"], OCR[f"{nm}_mask_dhi"] = d, lo, hi
OCR["swap_auc_masked"] = swap_auc_masked
print(OCR)

# ------------------------------------------------------------------------------------------- output diversity
DIV = {}
for nm, S in [("full", P)] + [(k_, pd.DataFrame([parse(t) for t in g.reindex(ids).generated.fillna("")], index=ids))
                              for k_, g in [("imnet", load("gen_imnet_fold{k}.csv")), ("imnetsp", load("gen_imnetsp_fold{k}.csv"))] if g is not None]:
    DIV[f"{nm}_nsets"] = int(S[[f"terr_{t}" for t in TERR]].astype(int).astype(str).agg("".join, axis=1).nunique())
    DIV[f"{nm}_small"] = float((S.vol_bin == 0).mean()); DIV[f"{nm}_nboth"] = float((S.side == "B").mean())
print(DIV)

# ------------------------------------------------------------------------------------------- macros
mac = ["% generated by src/review_analyses.py",
       "\\providecommand{\\NBR}[2]{\\ifcsname nbr@#1@#2\\endcsname\\csname nbr@#1@#2\\endcsname\\else\\textbf{\\textcolor{red}{??}}\\fi}"]
tex = lambda s: "$-$" + s[1:] if s.startswith("-") else s
put = lambda a, b, v, f="{:.3f}": mac.append(f"\\expandafter\\def\\csname nbr@{a}@{b}\\endcsname{{{tex(f.format(v))}}}")
key = lambda s: "".join(ch for ch in s.lower() if ch.isalnum())
put("dir", "nempty", n_empty, "{:,.0f}"); put("dir", "pempty", 100 * n_empty / len(ids), "{:.0f}")
put("eq", "margin", MARGIN, "{:.2f}")
for _, r in EQ.iterrows():
    k = key(r.source)
    for c in ["value", "d", "lo95", "hi95", "lo90", "hi90"]:
        put("eq" + k, c, r[c])
    put("eq" + k, "ptost", r.p_tost, "{:.3f}" if r.p_tost >= 0.001 else "{:.0e}"); put("eq" + k, "pmcn", r.mcn, "{:.3f}" if r.mcn >= 0.001 else "{:.0e}")
    put("eq" + k, "equiv", r.equivalent, "{}")
for _, r in SW.iterrows():
    for c in ["normal", "swap_donor", "swap_own", "chance", "changed", "donor_differs"]:
        put("sw" + key(r.field), key(c), r[c])
for k, v in CH.items():
    put("ch", key(k), v, "{:,.0f}" if isinstance(v, int) else "{:.3f}")
for k, v in AU.items():
    put("au", key(k), v, "{:,.0f}" if isinstance(v, int) else "{:.3f}")
put("au", "lrswappct", 100 * AU["lr_swap"] / AU["lr_n"], "{:.1f}"); put("au", "dirlrswappct", 100 * AU["dir_lr_swap"] / AU["lr_n"], "{:.1f}")
put("au", "multiplepct", 100 * AU["multiple"] / AU["n"], "{:.0f}"); put("au", "norestrpct", 100 * AU["norestr"] / AU["n"], "{:.0f}")
for k, v in VIS.items():
    put("vis", key(k), v, "{:,.0f}" if isinstance(v, int) else "{:.2f}")
for _, r in STR.iterrows():
    k = key(r.group); s = {"VLM": "vlm", "Direct": "dir", "Joint set classifier": "joint", "VLM minus Direct": "d"}[r.source]
    put("st" + s, k, r.all_correct); put("st" + s, k + "lo", r.lo); put("st" + s, k + "hi", r.hi); put("stn", k, r.n, "{:,.0f}")
    if s != "d":
        put("st" + s, k + "side", r.acc_side); put("st" + s, k + "halluc", r.halluc)
for k, v in UC.items():
    put("uc", key(k), v)
for k, v in OCR.items():
    put("oc", key(k), v, "{:,.0f}" if isinstance(v, int) else "{:.3f}")
for k, v in DIV.items():
    put("div", key(k), v, "{:,.0f}" if isinstance(v, int) else "{:.3f}")
put("oc", "poormaskpct", 100 * OCR["poor_mask"], "{:.0f}"); put("oc", "poornomaskpct", 100 * OCR["poor_nomask"], "{:.0f}")
(OUT / "numbers_b3.tex").write_text("\n".join(mac) + "\n")
print("written", OUT / "numbers_b3.tex")
