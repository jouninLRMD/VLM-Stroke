"""
Mask-grounded structured findings and reference report text for every scan with an expert acute lesion mask.

Every statement in a report is computed from the expert acute lesion mask (native volumes from preprocess_dwi.py,
locations from the MNI-warped mask and the arterial atlas of atlas_features.py), so the text is true by construction
and its facts can be scored.

Output: results/vlm/reports.csv with, per scan:
  vlm_fold (labeled patients keep their cv5 fold; other masked scans get a seeded random fold), labelled, y_poor
  truth fields: side (L/R/B), terr_<T> (7 vascular territories, 0/1), vol_ml, vol_bin (0-3), multiple (>= 2 lesions
  of >= 0.2 mL), chronic (>= 1 mL), restr (0 none / 1 mild / 2 marked, from mean relative ADC)
  clinical_text (prompt), report (target text)
"""
import numpy as np, pandas as pd, nibabel as nib
from scipy import ndimage
from common import RES, PREP, BIDS

TERR = {  # territory -> arterial-atlas region ids (odd = left, even = right)
    "ACA": [1, 2], "MCA": list(range(7, 17)), "DEEP": [3, 4, 5, 6, 23, 24], "PCA": [17, 18, 19, 20],
    "THAL": [21, 22], "BS": [25, 26], "CEREB": [27, 28, 29, 30]}
PHRASE = {"ACA": "anterior cerebral artery (ACA)", "MCA": "middle cerebral artery (MCA) cortical",
          "DEEP": "deep lenticulostriate/anterior choroidal", "PCA": "posterior cerebral artery (PCA)",
          "THAL": "thalamic (thalamoperforator)", "BS": "brainstem (basilar perforator)", "CEREB": "cerebellar"}
MCA_BRANCH = {7: "frontal", 8: "frontal", 9: "parietal", 10: "parietal", 11: "temporal", 12: "temporal",
              13: "occipital", 14: "occipital", 15: "insular", 16: "insular"}
VOL_EDGES = [2, 20, 70]                       # mL: small < 2 <= medium < 20 <= large < 70 <= very large
VOL_WORD = ["small", "medium", "large", "very large"]

coh = pd.read_csv(RES / "cohort_all.csv")
F = pd.read_csv(RES / "features_atlas.csv")
A = np.rint(nib.load(str(PREP / "atlas_mni2.nii.gz")).get_fdata()).astype(int)
vox_ml = abs(np.linalg.det(nib.load(str(PREP / "atlas_mni2.nii.gz")).affine[:3, :3])) / 1000
reg_ml = {k: (A == k).sum() * vox_ml for k in range(1, 33)}
load_col = {int(c.split("_")[-1]): c for c in F.columns if c.startswith("load_")}

d = coh.merge(F, on="participant_id", how="inner")
d = d[d.has_mask & (d.acute_ml.fillna(0) > 0)].copy()
rng = np.random.default_rng(2026)
d["vlm_fold"] = np.where(d.labelled, d.fold, rng.integers(0, 5, len(d)))


def n_lesions(sub, min_ml=0.2):
    """Separate acute lesions >= min_ml in the native expert mask (26-connectivity), so specks do not count."""
    p = BIDS / "derivatives" / "lesion_masks" / sub / "dwi" / f"{sub}_space-TRACE_desc-lesionAcute_mask.nii.gz"
    if not p.exists():
        return 1
    img = nib.load(str(p)); m = np.asarray(img.dataobj) > 0.5
    while m.ndim > 3:
        m = m[..., 0]
    lab, n = ndimage.label(m, structure=np.ones((3, 3, 3)))
    if n == 0:
        return 0
    sizes = ndimage.sum(m, lab, range(1, n + 1)) * np.prod(img.header.get_zooms()[:3]) / 1000
    return max(1, int((sizes >= min_ml).sum()))


def join(words):
    return words[0] if len(words) == 1 else ", ".join(words[:-1]) + " and " + words[-1]


rows = []
for _, r in d.iterrows():
    ml = {k: r[load_col[k]] * reg_ml[k] for k in load_col if k <= 30}
    tot = max(sum(ml.values()), 1e-6)
    inv = [k for k, v in ml.items() if v >= 0.25 and v / tot >= 0.10]
    if not inv:
        inv = [max(ml, key=ml.get)]
    terr = [t for t, ks in TERR.items() if any(k in inv for k in ks)]
    fl = r.frac_left if pd.notna(r.frac_left) else 0.5
    side = "L" if fl >= 0.9 else "R" if fl <= 0.1 else "B"
    side_phrase = {"L": "the left side", "R": "the right side", "B": "both sides"}[side]
    vol = float(r.acute_ml); vbin = int(np.searchsorted(VOL_EDGES, vol, side="right"))
    multiple = int(n_lesions(r.participant_id) >= 2)
    chronic = int((r.chronic_ml if pd.notna(r.chronic_ml) else 0) >= 1.0)
    radc = r.acute_radc_mean
    restr = 0 if pd.isna(radc) else (2 if radc < 0.8 else 1 if radc < 1.0 else 0)   # 2 marked, 1 mild, 0 none
    branches = sorted({MCA_BRANCH[k] for k in inv if k in MCA_BRANCH}, key=["frontal", "parietal", "temporal", "occipital", "insular"].index)
    terr_txt = join([PHRASE[t] for t in terr]) + (" territories" if len(terr) > 1 else " territory")
    if branches:
        terr_txt += f" (MCA branches: {', '.join(branches)})"
    rep = (f"FINDINGS: {'Multiple acute infarcts' if multiple else 'Acute infarct'} on {side_phrase}, involving the "
           f"{terr_txt}. Acute lesion volume {vol:.1f} mL ({VOL_WORD[vbin]}). "
           f"{['No clear diffusion restriction', 'Mild diffusion restriction', 'Marked diffusion restriction'][restr]}"
           f"{f' (mean relative ADC {radc:.2f})' if pd.notna(radc) else ''}. "
           f"{'Chronic infarct also present.' if chronic else 'No chronic infarct.'}")
    sex = {"M": "man", "F": "woman"}.get(r.sex, "patient")
    clin = (f"{int(r.age)}-year-old {sex}" if pd.notna(r.age) else f"Adult {sex}") + \
           (f", admission NIHSS {int(r.nihss)}" if pd.notna(r.nihss) else ", admission NIHSS unknown") + \
           (f", prior stroke {'yes' if r.priorstroke == 1 else 'no'}" if pd.notna(r.priorstroke) else "") + "."
    row = dict(participant_id=r.participant_id, vlm_fold=int(r.vlm_fold), labelled=bool(r.labelled),
               y_poor=r.y_poor, side=side, vol_ml=vol, vol_bin=vbin, multiple=multiple, chronic=chronic,
               restr=restr, mca_branches=",".join(branches), clinical_text=clin, report=rep)
    row.update({f"terr_{t}": int(t in terr) for t in TERR})
    rows.append(row)

out = RES / "vlm"; out.mkdir(parents=True, exist_ok=True)
R = pd.DataFrame(rows); R.to_csv(out / "reports.csv", index=False)
print(f"Reports: {len(R)} subjects ({R.labelled.sum()} labelled) | sides {R.side.value_counts().to_dict()}")
print("Territories:", {t: int(R[f'terr_{t}'].sum()) for t in TERR})
print("Volume bins:", R.vol_bin.value_counts().sort_index().to_dict(), "| multiple", int(R.multiple.sum()),
      "| chronic", int(R.chronic.sum()), "| restriction none/mild/marked", R.restr.value_counts().sort_index().to_dict())
print("\nExample:\n" + R.report.iloc[0] + "\n" + R.clinical_text.iloc[0])
