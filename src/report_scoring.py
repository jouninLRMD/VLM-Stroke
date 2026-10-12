"""Shared parsing and fact-level scoring of generated stroke reports (used by analysis.py and ablation_eval.py)."""
import re
import numpy as np, pandas as pd

TERR = ["ACA", "MCA", "DEEP", "PCA", "THAL", "BS", "CEREB"]
KEY = {"ACA": "anterior cerebral", "MCA": "middle cerebral", "DEEP": "lenticulostriate", "PCA": "posterior cerebral",
       "THAL": "thalamic", "BS": "brainstem", "CEREB": "cerebellar"}
EDGES = [2, 20, 70]
FIELDS = ["side", "vol_bin", "multiple", "chronic", "restr"]


def parse(t):
    t = t if isinstance(t, str) else ""
    side = "L" if "the left side" in t else "R" if "the right side" in t else "B" if "both sides" in t else None
    m = re.search(r"volume ([\d.]+) mL", t); vol = float(m.group(1)) if m else np.nan
    m = re.search(r"relative ADC ([\d.]+)\)", t); radc = float(m.group(1)) if m else np.nan
    restr = 2 if "Marked diffusion" in t else 1 if "Mild diffusion" in t else 0 if "No clear diffusion" in t else np.nan
    chron = 1 if "Chronic infarct also present" in t else 0 if "No chronic infarct" in t else np.nan
    br = re.search(r"MCA branches: ([a-z, ]+)\)", t)
    d = dict(side=side, vol_ml=vol, radc=radc, vol_bin=int(np.searchsorted(EDGES, vol, side="right")) if vol == vol else np.nan,
             multiple=int("Multiple acute" in t), chronic=chron, restr=restr,
             branches=",".join(sorted(br.group(1).split(", "))) if br else "",
             wellformed=int(t.startswith("FINDINGS:") and side is not None and vol == vol and restr == restr and chron == chron))
    d.update({f"terr_{k}": int(KEY[k] in t) for k in TERR})
    return d


def per_patient(P, T):
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


def boot(Qs, B=1000, seed=0, ref="VLM"):
    """Point estimates, bootstrap 95% CIs and paired differences (ref minus each source) on the same resamples."""
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
            if m != ref and ref in Qs:
                dd = pd.DataFrame(bs[ref])[c] - Bm[c]
                r.update(d_ref=pts[ref][c] - v, d_lo=dd.quantile(0.025), d_hi=dd.quantile(0.975))
            rows.append(r)
    return pd.DataFrame(rows)


def mcnemar(a, b):
    """Exact McNemar test on paired binary outcomes (two-sided)."""
    from scipy.stats import binomtest
    a, b = np.asarray(a, bool), np.asarray(b, bool)
    n01, n10 = int((~a & b).sum()), int((a & ~b).sum())
    return n10, n01, (binomtest(n10, n10 + n01, 0.5).pvalue if n10 + n01 else 1.0)
