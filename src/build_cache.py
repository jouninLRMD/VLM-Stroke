"""
Packs the MNI 2 mm volumes into memory-mapped arrays for GPU training (about 7 GB, written to <work>/cache).
Run once after preprocess_dwi.py, cohort_splits.py and atlas_features.py.

  <work>/cache/X.npy      float16 (N, 2, 96, 112, 96): z-scored trace, relative ADC
  <work>/cache/M.npy      uint8   (N, 1, 96, 112, 96): acute lesion mask (zeros if no expert mask)
  <work>/cache/index.csv  row -> participant_id, has_mask
  <work>/cache/atlas.npy  int16   (96, 112, 96) arterial atlas labels; centroids.npy (K, 3)
"""
import numpy as np, pandas as pd, nibabel as nib
from common import PREP, RES, WORK

GRID = (96, 112, 96)
C = WORK / "cache"; C.mkdir(parents=True, exist_ok=True)


def pad(v):
    out = np.zeros(GRID, v.dtype)
    s = [(g - n) // 2 for g, n in zip(GRID, v.shape)]
    sl_out = tuple(slice(max(0, a), max(0, a) + min(n, g)) for a, n, g in zip(s, v.shape, GRID))
    sl_in = tuple(slice(max(0, -a), max(0, -a) + min(n, g)) for a, n, g in zip(s, v.shape, GRID))
    out[sl_out] = v[sl_in]
    return out


coh = pd.read_csv(RES / "cohort_all.csv")
ok = coh[coh.dwi_ok].reset_index(drop=True)
N = len(ok)
X = np.lib.format.open_memmap(C / "X.npy", mode="w+", dtype=np.float16, shape=(N, 2) + GRID)
M = np.lib.format.open_memmap(C / "M.npy", mode="w+", dtype=np.uint8, shape=(N, 1) + GRID)
hasm = []
for i, s in enumerate(ok.participant_id):
    d = PREP / s
    tr = nib.load(str(d / f"{s}_space-MNI2_trace.nii.gz")).get_fdata()
    ra = nib.load(str(d / f"{s}_space-MNI2_radc.nii.gz")).get_fdata()
    X[i, 0] = pad(np.clip(tr, -5, 10)).astype(np.float16)
    X[i, 1] = pad(np.clip(ra, 0, 4)).astype(np.float16)
    mp = d / f"{s}_space-MNI2_lesionAcute.nii.gz"
    if not mp.exists():
        mp = d / f"{s}_space-MNI2_lesion.nii.gz"
    hasm.append(mp.exists())
    M[i, 0] = pad((nib.load(str(mp)).get_fdata() > 0.5).astype(np.uint8)) if mp.exists() else 0
    if (i + 1) % 100 == 0 or i + 1 == N:
        print(f"  cached {i+1}/{N}", flush=True)
X.flush(); M.flush()
pd.DataFrame(dict(row=np.arange(N), participant_id=ok.participant_id, has_mask=hasm)).to_csv(C / "index.csv", index=False)

A = pad(np.rint(nib.load(str(PREP / "atlas_mni2.nii.gz")).get_fdata()).astype(np.int16))
labels = [k for k in np.unique(A) if k > 0]
cent = np.array([np.argwhere(A == k).mean(0) for k in labels], dtype=np.float32)
cent = ((cent - np.array(GRID) / 2) / (np.array(GRID) / 2)).astype(np.float32)   # normalised to [-1, 1]
np.save(C / "atlas.npy", A); np.save(C / "atlas_labels.npy", np.array(labels)); np.save(C / "centroids.npy", cent)
print(f"Cache ready in {C}: {N} subjects, {len(labels)} atlas regions, {X.nbytes/1e9:.1f} GB")
