"""
Reference models R0, R1, R1b, R2a, R2b, I1 and I2 on the frozen splits (CPU, a few minutes).

Every model predicts P(mRS > k) for k = 0..5 with one L2-regularized logistic regression per threshold, so it can be
compared with the networks on both the binary (mRS 3 to 6) and the ordinal endpoints.
Outputs:
  results/oof/baseline_<model>__<scheme>.csv   out-of-fold predictions (scheme = cv5 | loso | field)
  results/baselines_repeats.csv                AUC over 20 x 5-fold repeats (mean, SD) per model
"""
import numpy as np, pandas as pd, warnings
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from common import RES, WORK, CLIN, N_ORD, clinical_matrix, auc

warnings.filterwarnings("ignore")
(RES / "oof").mkdir(parents=True, exist_ok=True)
coh = pd.read_csv(RES / "cohort_all.csv")
L = coh[coh.labelled].reset_index(drop=True)
F = pd.read_csv(RES / "features_atlas.csv")
L = L.merge(F, on="participant_id", how="left")
grp_cols = [c for c in F.columns if c.startswith("grp") and c.endswith("_ml")]
for c in grp_cols:
    L["log_" + c] = np.log1p(L[c].fillna(0.0))
L["frac_left_f"] = L.frac_left.fillna(0.5)
L["has_mask_f"] = L.has_mask.astype(float)
L["log_chronic_ml"] = np.log1p(L.chronic_ml_filled)
L["radc_f"] = L.acute_radc_mean.fillna(L.acute_radc_mean.median())
# 4 mm lesion maps for the topography baseline (zeros for patients without a mask)
Z = np.load(WORK / "lesion_maps_4mm.npz"); idx = {s: i for i, s in enumerate(Z["ids"])}
shape = Z["X"].shape[1:]
MAPS = np.stack([Z["X"][idx[s]].astype(np.float32) if s in idx else np.zeros(shape, np.float32)
                 for s in L.participant_id]).reshape(len(L), -1)

LES = ["log_acute_ml", "has_mask_f"]
MODELS = {   # name: (extra tabular columns, use PCA of lesion maps, include clinical, C)
    "R0_clinical": ([], False, True, 1.0),
    "I1_volume_only": (LES, False, False, 1.0),
    "R1_clin_volume": (LES, False, True, 1.0),
    "R1b_clin_vol_chronic_radc": (LES + ["log_chronic_ml", "radc_f"], False, True, 1.0),
    "R2a_clin_vol_territory": (LES + ["frac_left_f"] + ["log_" + c for c in grp_cols], False, True, 0.3),
    "I2_topography_only": (["has_mask_f"], True, False, 0.1),
    "R2b_clin_vol_topography": (LES, True, True, 0.1),
}


def fit_predict(tr, te, spec):
    extra, use_pca, use_clin, C = spec
    parts_tr, parts_te = [], []
    if use_clin:
        parts_tr.append(clinical_matrix(L.iloc[tr], L.iloc[tr])); parts_te.append(clinical_matrix(L.iloc[te], L.iloc[tr]))
    if extra:
        parts_tr.append(L.iloc[tr][extra].values.astype(float)); parts_te.append(L.iloc[te][extra].values.astype(float))
    if use_pca:
        pca = PCA(20, random_state=0).fit(MAPS[tr])
        parts_tr.append(pca.transform(MAPS[tr])); parts_te.append(pca.transform(MAPS[te]))
    Xtr, Xte = np.hstack(parts_tr), np.hstack(parts_te)
    sc = StandardScaler().fit(Xtr); Xtr, Xte = sc.transform(Xtr), sc.transform(Xte)
    P = np.zeros((len(te), N_ORD))
    for k in range(N_ORD):
        yk = (L.mrs.values[tr] > k).astype(int)
        if yk.min() == yk.max():                     # degenerate threshold in a small training split
            P[:, k] = yk.mean(); continue
        P[:, k] = LogisticRegression(max_iter=3000, C=C).fit(Xtr, yk).predict_proba(Xte)[:, 1]
    return P


def splits(scheme):
    if scheme == "cv5":
        for k in range(5):
            yield f"fold{k}", np.where(L.fold != k)[0], np.where(L.fold == k)[0]
    elif scheme == "loso":
        vc = L.scanner_group.value_counts()
        for g in vc[vc >= 30].index:
            yield str(g), np.where(L.scanner_group != g)[0], np.where(L.scanner_group == g)[0]
    elif scheme == "field":
        yield "train1.5T_test3T", np.where(L.field < 3)[0], np.where(L.field >= 3)[0]


for scheme in ["cv5", "loso", "field"]:
    for name, spec in MODELS.items():
        out = []
        for unit, tr, te in splits(scheme):
            P = fit_predict(tr, te, spec)
            o = pd.DataFrame(P, columns=[f"p_gt{k}" for k in range(N_ORD)])
            o.insert(0, "test_unit", unit); o.insert(0, "participant_id", L.participant_id.values[te])
            out.append(o)
        pd.concat(out).to_csv(RES / "oof" / f"baseline_{name}__{scheme}.csv", index=False)
        print(f"{scheme:6s} {name:30s} done")

R = pd.read_csv(RES / "folds_repeats.csv")
rows = []
for name, spec in MODELS.items():
    aucs = []
    for r in range(20):
        fr = L[["participant_id"]].merge(R[R.rep == r], on="participant_id", how="left").fold.values
        for k in range(5):
            tr, te = np.where(fr != k)[0], np.where(fr == k)[0]
            aucs.append(auc(L.y_poor.values[te], fit_predict(tr, te, spec)[:, 2]))
    rows.append(dict(model=name, auc_mean=np.mean(aucs), auc_sd=np.std(aucs), n=len(L)))
    print(f"{name:30s} 20x5 CV AUC {np.mean(aucs):.3f} +/- {np.std(aucs):.3f}")
pd.DataFrame(rows).to_csv(RES / "baselines_repeats.csv", index=False)
