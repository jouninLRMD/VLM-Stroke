r"""
Additional analyses of the grounded VLM, run on the outputs in results/ and the repository's own scoring code
(report_scoring.py, ablation_eval.py), so that every definition matches the paper.

1. Bayesian bootstrap posterior of paired differences in the proportion of reports with all key findings
   correct, with a region of practical equivalence (ROPE, +/-0.03, the margin of the equivalence tests;
   sensitivity +/-0.02 and +/-0.05).
2. Probability of a fully correct report as a smooth function of expert lesion volume: Bayesian logistic
   P-spline (cubic B-splines on log volume, second-order random-walk prior, Laplace approximation,
   smoothing chosen by maximum marginal likelihood).
3. A no-cost uncertainty score for each report: the probability that the encoder heads assign to the key
   findings written in the VLM report, compared with self-consistency.

Usage (from the repository root): python src/bayes_extra.py [repo root] [output dir]
Outputs: results/paper/numbers_b4.tex (macros \NBX{key}{field}), bayes_rope.csv, volume_curves.csv,
         uncertainty_encoder.csv, figB9_volume_curves.pdf/.png
"""
import sys, os, json
import numpy as np, pandas as pd
from pathlib import Path

REPO = Path(sys.argv[1] if len(sys.argv) > 1 else ".").resolve()
OUT = Path(sys.argv[2]).resolve() if len(sys.argv) > 2 else REPO / "results" / "paper"
OUT.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(REPO / "src"))
os.chdir(REPO / "src")

# run the repository evaluation up to (not including) its figures, to obtain the per-scan scores
src = (REPO / "src" / "ablation_eval.py").read_text()
src = src.split("# ------------------------------------------------------------------------------------------- figures")[0]
ns = {"__name__": "ablation_eval_partial"}
exec(compile(src, "ablation_eval.py", "exec"), ns)
Qs, T, ids, D, Ps, conds = ns["Qs"], ns["T"], ns["ids"], ns["D"], ns["Ps"], None
from report_scoring import per_patient, TERR, parse
Qdef = per_patient(ns["Qdef"], T); Qtun = ns["Qtun"]; Qjoint = ns["Qjoint"]
sh = ns["sh"]; Psh = ns["Psh"]; Td = ns["Td"]
conds = {"own": per_patient(Ps["Full model"], T), "donor": per_patient(Psh, Td),
         "swap_own": per_patient(Psh, T), "chance": per_patient(Ps["Full model"], Td)}

S = pd.DataFrame({"Full model": Qs["Full model"]["all"].values, "Direct (default)": Qdef["all"].values,
                  "Direct (tuned)": Qtun["all"].values, "Joint set classifier": Qjoint["all"].values}, index=ids)
for n in ["Full model, seed 1", "Full model, seed 2", "No language pretraining", "Qwen2.5-0.5B",
          "No clinical sentence", "Pooled token only", "Qwen2.5-3B"]:
    S[n] = Qs[n]["all"].values
for k, v in conds.items():
    S["swap:" + k] = v["all"].values
print("proportions of reports with all key findings correct:")
print(S.mean().round(3).to_string())

# ---------------------------------------------------------------- 1. Bayesian bootstrap + ROPE
NDRAW, ROPES, ROPE = 4000, (0.02, 0.03, 0.05), 0.03
rng = np.random.default_rng(20261117)
W = rng.dirichlet(np.ones(len(S)), size=NDRAW)
M = W @ S.values                                     # NDRAW x systems: posterior of each proportion
post = pd.DataFrame(M, columns=S.columns)
pairs = [("Full model", "Direct (default)", "dir"), ("Full model", "Direct (tuned)", "tuned"),
         ("Full model", "Joint set classifier", "joint"), ("Full model", "No language pretraining", "scratch"),
         ("Full model", "Qwen2.5-0.5B", "small"), ("Full model", "Full model, seed 1", "seedone"),
         ("Full model", "Full model, seed 2", "seedtwo"), ("Full model", "No clinical sentence", "noclin"),
         ("Full model", "Pooled token only", "pooled"),
         ("swap:own", "swap:donor", "grdonor"),          # normal vs swapped-against-donor
         ("swap:swap_own", "swap:chance", "grchance")]   # swapped-against-own vs chance
rows = []
for a, b, key in pairs:
    d = post[a] - post[b]
    r = dict(key=key, a=a, b=b, pa=S[a].mean(), pb=S[b].mean(), d=d.mean(), lo=d.quantile(.025), hi=d.quantile(.975),
             pgt0=(d > 0).mean())
    for m in ROPES:
        r[f"prope{m}"], r[f"pa_better{m}"], r[f"pb_better{m}"] = (d.abs() <= m).mean(), (d > m).mean(), (d < -m).mean()
    rows.append(r)
R1 = pd.DataFrame(rows); R1.to_csv(OUT / "bayes_rope.csv", index=False, float_format="%.4f")
print(R1[["key", "pa", "pb", "d", "lo", "hi", "prope0.03", "pa_better0.03", "pb_better0.03", "prope0.02", "prope0.05"]].round(3).to_string())

# ---------------------------------------------------------------- 2. accuracy vs volume (Bayesian P-spline)
from scipy.interpolate import BSpline
from scipy.optimize import minimize
vol = T.vol_ml.values.astype(float)
x = np.log10(np.clip(vol, 0.05, None))
lo_, hi_ = np.quantile(x, [0.005, 0.995])
nk = 10
inner = np.linspace(lo_, hi_, nk)
kn = np.r_[[lo_] * 3, inner, [hi_] * 3]
def basis(xx):
    xx = np.clip(xx, lo_, hi_)
    return BSpline.design_matrix(xx, kn, 3).toarray()
B = basis(x); K = B.shape[1]
Dm = np.diff(np.eye(K), 2, axis=0); Pen = Dm.T @ Dm

def laplace(y, lam):
    def nlp(b):
        eta = B @ b
        return -(y * eta - np.logaddexp(0, eta)).sum() + 0.5 * lam * b @ Pen @ b + 0.5e-6 * b @ b
    def grad(b):
        p = 1 / (1 + np.exp(-(B @ b)))
        return -B.T @ (y - p) + lam * Pen @ b + 1e-6 * b
    b = minimize(nlp, np.zeros(K), jac=grad, method="L-BFGS-B").x
    p = 1 / (1 + np.exp(-(B @ b)))
    H = B.T @ (B * (p * (1 - p))[:, None]) + lam * Pen + 1e-6 * np.eye(K)
    # log marginal likelihood (Laplace), prior rank K-2
    ev = np.linalg.eigvalsh(lam * Pen + 1e-6 * np.eye(K))
    lml = -nlp(b) + 0.5 * np.sum(np.log(ev)) - 0.5 * np.linalg.slogdet(H)[1]
    return b, np.linalg.inv(H), lml

grid = np.linspace(lo_, hi_, 200); Bg = basis(grid)
curves = {}
for name in ["Full model", "Joint set classifier", "Direct (default)"]:
    y = S[name].values.astype(float)
    best = max((laplace(y, l) + (l,) for l in 10.0 ** np.arange(-3, 3.1, 0.25)), key=lambda t: t[2])
    b, V, _, lam = best
    draws = rng.multivariate_normal(b, V, size=4000)
    pg = 1 / (1 + np.exp(-(draws @ Bg.T)))
    curves[name] = dict(mean=pg.mean(0), lo=np.quantile(pg, .025, 0), hi=np.quantile(pg, .975, 0), draws=pg, lam=lam)
# posterior of difference VLM - joint along volume
dj = curves["Full model"]["draws"] - curves["Joint set classifier"]["draws"]
pd.DataFrame({"log10_vol": grid, "vol_ml": 10 ** grid,
              **{f"{n}_mean": c["mean"] for n, c in curves.items()},
              **{f"{n}_lo": c["lo"] for n, c in curves.items()},
              **{f"{n}_hi": c["hi"] for n, c in curves.items()},
              "d_vlm_joint_mean": dj.mean(0), "d_vlm_joint_lo": np.quantile(dj, .025, 0), "d_vlm_joint_hi": np.quantile(dj, .975, 0),
              "p_vlm_better": (dj > 0).mean(0)}).to_csv(OUT / "volume_curves.csv", index=False, float_format="%.4f")
def at(name, ml):
    i = np.argmin(np.abs(grid - np.log10(ml))); c = curves[name]; return c["mean"][i], c["lo"][i], c["hi"][i]

# ---------------------------------------------------------------- 3. encoder-probability uncertainty
from sklearn.metrics import roc_auc_score
U = pd.read_csv(REPO / "results" / "paper" / "uncertainty.csv").set_index("participant_id")
P = Ps["Full model"]
side_idx = {"L": 0, "R": 1, "B": 2}
conf = []
for pid in ids:
    p, dd = P.loc[pid], D.loc[pid]
    if p.side not in side_idx or p.vol_bin != p.vol_bin:
        conf.append(np.nan); continue
    c = [dd["p_side_L"], dd["p_side_R"], dd["p_side_B"]][side_idx[p.side]] * dd[f"p_vol_{int(p.vol_bin)}"]
    for t in TERR:
        q = dd[f"p_terr_{t}"]; c *= q if p[f"terr_{t}"] == 1 else 1 - q
    conf.append(c)
E = pd.DataFrame({"enc_conf": conf}, index=ids).join(U, how="inner").dropna()
err = 1 - E.correct.values
a_sc, a_en = roc_auc_score(err, 1 - E.agreement.values), roc_auc_score(err, -E.enc_conf.values)
bs = []
for _ in range(2000):
    i = rng.integers(0, len(E), len(E))
    if err[i].min() == err[i].max(): continue
    bs.append((roc_auc_score(err[i], 1 - E.agreement.values[i]), roc_auc_score(err[i], -E.enc_conf.values[i])))
bs = np.array(bs)
# combined: rank average
comb = pd.Series(1 - E.agreement.values).rank().values + pd.Series(-E.enc_conf.values).rank().values
a_cb = roc_auc_score(err, comb)
unc = dict(n=len(E), auroc_sc=a_sc, auroc_enc=a_en, auroc_enc_lo=np.quantile(bs[:, 1], .025), auroc_enc_hi=np.quantile(bs[:, 1], .975),
           d=a_sc - a_en, d_lo=np.quantile(bs[:, 0] - bs[:, 1], .025), d_hi=np.quantile(bs[:, 0] - bs[:, 1], .975), auroc_comb=a_cb)
# lesions below 2 mL (volume class 0), where errors matter most
sm = (T.loc[E.index, "vol_bin"] == 0).values
if err[sm].min() != err[sm].max():
    unc["auroc_enc_small"] = roc_auc_score(err[sm], -E.enc_conf.values[sm])
    unc["auroc_sc_small"] = roc_auc_score(err[sm], 1 - E.agreement.values[sm])
# retention at 50% by encoder confidence
o = np.argsort(-E.enc_conf.values); unc["acc50_enc"] = E.correct.values[o][: len(E) // 2].mean()
pd.Series(unc).to_csv(OUT / "uncertainty_encoder.csv", float_format="%.4f")
print(pd.Series(unc).round(3).to_string())

# ---------------------------------------------------------------- macros \NBX{key}{field}
with open(OUT / "numbers_b4.tex", "w") as f:
    f.write("%% generated by src/bayes_extra.py (Bayesian bootstrap %d draws, ROPE +/-%.2f)\n" % (NDRAW, ROPE))
    f.write("\\providecommand{\\NBX}[2]{\\ifcsname nbx@#1@#2\\endcsname\\csname nbx@#1@#2\\endcsname\\else\\textbf{\\textcolor{red}{??}}\\fi}\n")
    def w(k, fld, v, fmt="%.3f"):
        s = (fmt % v).replace("-", "$-$")
        f.write("\\expandafter\\def\\csname nbx@%s@%s\\endcsname{%s}\n" % (k, fld, s))
    for _, r in R1.iterrows():
        for fld in ["d", "lo", "hi"]: w(r.key, fld, r[fld])
        w(r.key, "prope", r[f"prope{ROPE}"], "%.2f"); w(r.key, "propetwo", r["prope0.02"], "%.2f"); w(r.key, "propefive", r["prope0.05"], "%.2f")
        w(r.key, "pabetter", r[f"pa_better{ROPE}"], "%.2f"); w(r.key, "pbbetter", r[f"pb_better{ROPE}"], "%.2f")
        w(r.key, "pgtzero", r.pgt0, "%.3f")
    for nm, k in [("Full model", "vlm"), ("Joint set classifier", "joint"), ("Direct (default)", "dir")]:
        for ml, tag in [(1, "one"), (5, "five"), (20, "twenty"), (100, "hundred")]:
            m, l, h = at(nm, ml); w(k, "vol" + tag, m); w(k, "vol" + tag + "lo", l); w(k, "vol" + tag + "hi", h)
    # volume ranges in which the VLM or the joint classifier is more often fully correct (posterior probability > 0.9)
    pb = (dj > 0).mean(0)
    for tag, sel in [("vlmbetter", pb > 0.9), ("jointbetter", pb < 0.1)]:
        if sel.any():
            w("vc", tag + "min", 10 ** grid[sel].min(), "%.1f"); w("vc", tag + "max", 10 ** grid[sel].max(), "%.1f")
    for k2, v in unc.items():
        if k2 != "n": w("unc", k2.replace("_", ""), v)

# ---------------------------------------------------------------- figure: accuracy vs volume
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 7, "axes.linewidth": 0.6, "axes.edgecolor": "#52514e",
                     "xtick.color": "#52514e", "ytick.color": "#52514e", "pdf.fonttype": 42})
cols = {"Full model": "#eb6834", "Joint set classifier": "#2a78d6", "Direct (default)": "#1baf7a"}
labs = {"Full model": "Grounded VLM", "Joint set classifier": "Joint set classifier", "Direct (default)": "Direct classifier"}
fig, (ax, ax2) = plt.subplots(2, 1, figsize=(3.5, 3.9), sharex=True, gridspec_kw=dict(height_ratios=[2.2, 1], hspace=0.12))
for n, c in curves.items():
    ax.fill_between(10 ** grid, c["lo"], c["hi"], color=cols[n], alpha=0.18, lw=0)
    ax.plot(10 ** grid, c["mean"], color=cols[n], lw=1.6, label=labs[n])
# rug of volumes
ax.plot(vol, np.full_like(vol, -0.03), "|", color="#9b9a97", ms=3, alpha=0.25, mew=0.5)
for v_ in (2, 20, 70):
    ax.axvline(v_, color="#c3c2b7", lw=0.5, ls=":"); ax2.axvline(v_, color="#c3c2b7", lw=0.5, ls=":")
ax.set_xscale("log"); ax.set_ylim(-0.06, 0.85); ax.set_ylabel("P(all key findings correct)")
for s in ["top", "right"]: ax.spines[s].set_visible(False); ax2.spines[s].set_visible(False)
ax.legend(frameon=False, loc="upper left", fontsize=6.5)
ax2.fill_between(10 ** grid, np.quantile(dj, .025, 0), np.quantile(dj, .975, 0), color="#9b9a97", alpha=0.3, lw=0)
ax2.plot(10 ** grid, dj.mean(0), color="#0b0b0b", lw=1.2)
ax2.axhline(0, color="#52514e", lw=0.5); ax2.axhspan(-ROPE, ROPE, color="#9b9a97", alpha=0.12, lw=0)
ax2.set_ylabel("VLM $-$ joint"); ax2.set_xlabel("Expert acute lesion volume (mL, log scale)")
ax2.set_xlim(10 ** lo_, 10 ** hi_)
fig.subplots_adjust(left=0.16, right=0.97, top=0.98, bottom=0.11)
fig.savefig(OUT / "figB9_volume_curves.pdf"); fig.savefig(OUT / "figB9_volume_curves.png", dpi=300)
print("lambda:", {n: c["lam"] for n, c in curves.items()})
