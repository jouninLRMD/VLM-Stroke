"""Shared paths, clinical-feature preparation, ordinal helpers and metrics for the SOOP pipeline."""
import json, os
from pathlib import Path
import numpy as np
import pandas as pd

CODE = Path(__file__).resolve().parent
REPO = CODE.parent
# Data folder (SOOP download, preprocessed volumes, atlas). Override with the CALM_DATA environment variable.
DATA = Path(os.environ.get("CALM_DATA", str(REPO / "data")))
BIDS = DATA / "ds004889"                             # OpenNeuro ds004889 v1.1.2
PREP = DATA / "derivatives" / "calm_prep"            # outputs of preprocess_dwi.py
RESOURCES = DATA / "resources"                       # arterial territory atlas (scripts/fetch_atlas.py)
# Tables, out-of-fold predictions and figures (small files). Override with CALM_RESULTS.
RES = Path(os.environ.get("CALM_RESULTS", str(REPO / "results")))
# Large binary caches and model weights. Override with CALM_WORK.
WORK = Path(os.environ.get("CALM_WORK", str(DATA / "work")))

CLIN = ["age", "nihss", "priorstroke", "bmi"]        # variables of the clinical anchor and of R0
N_ORD = 6                                            # thresholds of mRS 0..6 -> P(mRS > k), k = 0..5
POOR_K = 2                                           # poor outcome = mRS >= 3 = P(mRS > 2)


def load_cohort():
    return pd.read_csv(RES / "cohort_all.csv")


def clinical_matrix(df, fit_df=None, cols=CLIN):
    """Median-impute (fit on fit_df) + missing indicators. Returns np.array; scaling is done by the models."""
    fit_df = df if fit_df is None else fit_df
    X = []
    for c in cols:
        med = float(np.nanmedian(fit_df[c].astype(float))) if fit_df[c].notna().any() else 0.0
        v = df[c].astype(float)
        X.append(v.fillna(med).values)
        if fit_df[c].isna().any():
            X.append(v.isna().astype(float).values)
    return np.vstack(X).T


def fit_anchor(train_df, apply_df, cols=CLIN):
    """In-fold clinical anchor: one logistic model per threshold P(mRS > k). Returns logits (n_apply, 6)."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    tr = train_df[train_df.mrs.notna()]
    Xtr = clinical_matrix(tr, tr, cols); Xap = clinical_matrix(apply_df, tr, cols)
    sc = StandardScaler().fit(Xtr); Xtr, Xap = sc.transform(Xtr), sc.transform(Xap)
    out = np.zeros((len(apply_df), N_ORD), dtype=np.float32)
    for k in range(N_ORD):
        yk = (tr.mrs.values > k).astype(int)
        if yk.min() == yk.max():
            q = np.clip(yk.mean(), 1e-3, 1 - 1e-3); out[:, k] = np.log(q / (1 - q)); continue
        m = LogisticRegression(max_iter=2000, C=1.0).fit(Xtr, yk)
        out[:, k] = m.decision_function(Xap)
    return out


def expected_mrs(p_gt):
    """E[mRS] = sum_k P(mRS > k)."""
    return np.asarray(p_gt).sum(1)


def auc(y, p):
    from sklearn.metrics import roc_auc_score
    return float(roc_auc_score(y, p))


def boot_auc_ci(y, p, B=2000, seed=0):
    rng = np.random.default_rng(seed); y, p = np.asarray(y), np.asarray(p); n = len(y); a = []
    for _ in range(B):
        i = rng.integers(0, n, n)
        if y[i].min() == y[i].max():
            continue
        a.append(auc(y[i], p[i]))
    return np.percentile(a, [2.5, 97.5])


def paired_boot_delta(y, p_new, p_ref, B=2000, seed=0):
    rng = np.random.default_rng(seed); y = np.asarray(y); n = len(y); d = []
    for _ in range(B):
        i = rng.integers(0, n, n)
        if y[i].min() == y[i].max():
            continue
        d.append(auc(y[i], p_new[i]) - auc(y[i], p_ref[i]))
    d = np.array(d)
    return float(auc(y, p_new) - auc(y, p_ref)), np.percentile(d, [2.5, 97.5]), float((d <= 0).mean())


def calibration(y, p):
    """Calibration intercept (calibration-in-the-large) and slope via logistic recalibration."""
    from sklearn.linear_model import LogisticRegression
    p = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6); lp = np.log(p / (1 - p)).reshape(-1, 1)
    slope = LogisticRegression(C=1e6, max_iter=1000).fit(lp, y).coef_[0, 0]
    # intercept with slope fixed at 1: solve via offset Newton iterations
    a = 0.0
    for _ in range(50):
        q = 1 / (1 + np.exp(-(lp[:, 0] + a))); a += (np.sum(y - q)) / max(np.sum(q * (1 - q)), 1e-9)
    return float(a), float(slope)


def harrell_c(mrs, score):
    """Concordance between predicted expected mRS and observed ordinal mRS (ties in mrs ignored)."""
    mrs, score = np.asarray(mrs), np.asarray(score); num = den = 0.0
    for i in range(len(mrs)):
        d = mrs[i] - mrs[i + 1:]; s = score[i] - score[i + 1:]; m = d != 0
        den += m.sum(); num += ((np.sign(d[m]) == np.sign(s[m])).sum() + 0.5 * (s[m] == 0).sum())
    return float(num / max(den, 1))


def save_json(obj, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(obj, indent=2, default=float))
