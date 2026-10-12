"""
Lesion location features in MNI space, computed from the warped expert acute lesion masks.

Outputs:
  data/derivatives/calm_prep/atlas_mni2.nii.gz   arterial atlas (32 level-1 regions) on the MNI 2 mm grid
  results/features_atlas.csv                     per subject: lesion load per region (fraction), volume per
                                                 territory group (mL), left fraction, posterior fraction
  <work>/lesion_maps_4mm.npz                      4 mm lesion maps for the topography (PCA) reference model
"""
import numpy as np, pandas as pd, nibabel as nib
from scipy import ndimage
from nilearn.image import resample_to_img
from common import PREP, RES, RESOURCES, WORK

WORK.mkdir(parents=True, exist_ok=True)
tmpl = nib.load(str(PREP / "MNI152_2mm_T1.nii.gz"))
atl = resample_to_img(nib.load(str(RESOURCES / "ArterialAtlas136.nii.gz")), tmpl, interpolation="nearest",
                      force_resample=True, copy_header=True)
A = np.rint(atl.get_fdata()).astype(int)
nib.save(nib.Nifti1Image(A.astype(np.int16), tmpl.affine), str(PREP / "atlas_mni2.nii.gz"))
names, groups = {}, {}
for line in open(RESOURCES / "ArterialAtlas136.txt"):
    if line.strip():
        c = line.strip().split("|"); k = int(c[0])
        names[k] = f"{c[1]}_{k}"; groups[k] = int(c[3])
labels = [k for k in sorted(names) if (A == k).any()]
# world x of every voxel (MNI: x < 0 is left)
ii = np.arange(A.shape[0]); xw = tmpl.affine[0, 0] * ii + tmpl.affine[0, 3]
left = (xw < 0)[:, None, None] * np.ones(A.shape, bool)
POST = [5, 6, 7, 8]                                   # level-2 groups: PCA/choroidal-thalamic + basilar/cerebellar
vox_ml = abs(np.linalg.det(tmpl.affine[:3, :3])) / 1000.0

coh = pd.read_csv(RES / "cohort_all.csv")
rows, maps, ids = [], [], []
for s in coh.participant_id[coh.has_mask]:
    fp = PREP / s / f"{s}_space-MNI2_lesionAcute.nii.gz"
    if not fp.exists():
        fp = PREP / s / f"{s}_space-MNI2_lesion.nii.gz"
    if not fp.exists():
        continue
    m = nib.load(str(fp)).get_fdata() > 0.5
    r = dict(participant_id=s, mni_acute_ml=m.sum() * vox_ml)
    n = max(1, m.sum())
    r["frac_left"] = (m & left).sum() / n if m.any() else np.nan
    for k in labels:
        r[f"load_{names[k]}"] = (m & (A == k)).sum() / max(1, (A == k).sum())
    for g in sorted(set(groups.values())):
        gm = np.isin(A, [k for k in labels if groups[k] == g])
        r[f"grp{g}_ml"] = (m & gm).sum() * vox_ml
    r["post_frac"] = sum(r[f"grp{g}_ml"] for g in POST) / max(r["mni_acute_ml"], 1e-6) if m.any() else np.nan
    rows.append(r)
    maps.append(ndimage.zoom(m.astype(np.float32), 0.5, order=1).astype(np.float16)); ids.append(s)
F = pd.DataFrame(rows); F.to_csv(RES / "features_atlas.csv", index=False)
np.savez_compressed(WORK / "lesion_maps_4mm.npz", ids=np.array(ids), X=np.stack(maps))
print(f"Atlas regions on MNI2 grid: {len(labels)} | subjects with lesion maps: {len(F)} | "
      f"median MNI acute volume {F.mni_acute_ml.median():.1f} mL | posterior-predominant: {(F.post_frac > 0.5).sum()}")
