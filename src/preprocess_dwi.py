"""
DWI preprocessing for every SOOP subject.

For each subject with a DWI trace image and an ADC map:
  1. load, squeeze and reorient to RAS
  2. brain mask on the trace image (Otsu threshold, morphology, largest component)
  3. relative ADC: ADC divided by the median ADC of brain parenchyma (removes vendor scaling)
  4. trace image z-scored inside the brain
  5. affine registration of the trace image to the MNI152 2 mm template (nilearn template, no download);
     --syn adds a SyN stage
  6. trace, rADC, brain mask and acute and total lesion masks warped to MNI 2 mm (91 x 109 x 91)
  7. native-space lesion features (volumes in mL from the true voxel size, chronic = total - acute, lesion rADC)
  8. QC: Dice of the warped brain mask with the template mask and a PNG montage with the lesion outline

Outputs:  <out>/sub-X/sub-X_space-MNI2_{trace,radc,brainmask,lesionAcute,lesion}.nii.gz, sub-X_affine.mat
          <out>/features_dwi.csv   (one row per subject)
          <out>/qc/sub-X.png
Usage:    python src/preprocess_dwi.py --workers 6
          (re-runnable: finished subjects are skipped; add --subjects sub-1 sub-10 to test a few)
"""
import argparse, json, os, sys, time, traceback
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

os.environ.setdefault("ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS", "2")
import numpy as np
import nibabel as nib
import pandas as pd
from scipy import ndimage


def load_ras(path):
    img = nib.as_closest_canonical(nib.load(str(path)))
    d = np.asarray(img.dataobj, dtype=np.float32)
    while d.ndim > 3:
        d = d[..., 0]
    return nib.Nifti1Image(d, img.affine), d


def otsu(x, bins=256):
    h, e = np.histogram(x, bins=bins)
    c = (e[:-1] + e[1:]) / 2
    w0 = np.cumsum(h); w1 = w0[-1] - w0
    m0 = np.cumsum(h * c) / np.maximum(w0, 1); m1 = (np.sum(h * c) - np.cumsum(h * c)) / np.maximum(w1, 1)
    return c[np.argmax(w0 * w1 * (m0 - m1) ** 2)]


def brain_mask(trace):
    v = trace[trace > 0]
    if v.size < 1000:
        return trace > 0
    lo, hi = np.percentile(v, [1, 99.5])
    t = otsu(np.clip(v, lo, hi)) * 0.6
    m = trace > t
    st = ndimage.generate_binary_structure(3, 1)
    m = ndimage.binary_opening(m, structure=st, iterations=1)
    lab, n = ndimage.label(m)
    if n > 1:
        sizes = ndimage.sum(m, lab, range(1, n + 1)); m = lab == (1 + int(np.argmax(sizes)))
    m = ndimage.binary_closing(m, structure=st, iterations=2)
    for z in range(m.shape[2]):                      # fill holes slice-wise (thick 2D DWI slices)
        m[:, :, z] = ndimage.binary_fill_holes(m[:, :, z])
    return m


def dice(a, b):
    a, b = a > 0.5, b > 0.5
    return 2 * (a & b).sum() / max(1, a.sum() + b.sum())


def qc_png(path, trace_mni, les_mni, tmpl, sub, meta):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    zs = np.linspace(20, 70, 6).astype(int)
    fig, ax = plt.subplots(1, 6, figsize=(15, 3.2), facecolor="black")
    for a, z in zip(ax, zs):
        a.imshow(np.rot90(trace_mni[:, :, z]), cmap="gray", vmin=-2, vmax=4)
        a.contour(np.rot90(tmpl[:, :, z]), levels=[0.5], colors="cyan", linewidths=0.5)
        if les_mni[:, :, z].any():
            a.contour(np.rot90(les_mni[:, :, z]), levels=[0.5], colors="red", linewidths=0.8)
        a.set_title(f"z={z}", color="white", fontsize=8); a.axis("off")
    fig.suptitle(f"{sub} | {meta}", color="white", fontsize=9)
    fig.savefig(path, dpi=70, bbox_inches="tight", facecolor="black"); plt.close(fig)


def process(sub, root, out, syn, template_path, tmask_path):
    import ants
    t0 = time.time()
    root, out = Path(root), Path(out)
    od = out / sub; od.mkdir(parents=True, exist_ok=True)
    done = od / "done.json"
    if done.exists():
        return json.loads(done.read_text())
    tr_p = root / sub / "dwi" / f"{sub}_rec-TRACE_dwi.nii.gz"
    adc_p = root / sub / "dwi" / f"{sub}_rec-ADC_dwi.nii.gz"
    if not (tr_p.exists() and adc_p.exists()):
        return dict(participant_id=sub, status="missing_dwi")
    tr_img, tr = load_ras(tr_p)
    adc_img, adc = load_ras(adc_p)
    if adc.shape != tr.shape:                        # rare: resample ADC onto TRACE grid
        from nilearn.image import resample_to_img
        adc = resample_to_img(adc_img, tr_img, interpolation="linear", force_resample=True, copy_header=True).get_fdata().astype(np.float32)
    zooms = np.array(tr_img.header.get_zooms()[:3], dtype=float); vox_ml = float(np.prod(zooms)) / 1000.0
    bm = brain_mask(tr)
    # relative ADC: divide by median ADC of brain voxels (vendor-independent)
    a_in = adc[bm & (adc > 0)]
    if a_in.size:                                    # parenchyma reference: exclude CSF (top 30% of ADC)
        a_in = a_in[a_in <= np.percentile(a_in, 70)]
    adc_med = float(np.median(a_in)) if a_in.size else 1.0
    radc = np.clip(adc / max(adc_med, 1e-6), 0, 5).astype(np.float32)
    t_in = tr[bm]
    trz = ((tr - t_in.mean()) / (t_in.std() + 1e-6)).astype(np.float32); trz[~bm] = 0

    les = {}
    for k in ["lesionAcute", "lesion"]:
        p = root / "derivatives" / "lesion_masks" / sub / "dwi" / f"{sub}_space-TRACE_desc-{k}_mask.nii.gz"
        if p.exists():
            m_img, m = load_ras(p)
            les[k] = (m > 0.5) if m.shape == tr.shape else None
    acute, total = les.get("lesionAcute"), les.get("lesion")
    feat = dict(participant_id=sub, status="ok", nx=tr.shape[0], ny=tr.shape[1], nz=tr.shape[2],
                vx=zooms[0], vy=zooms[1], vz=zooms[2], adc_brain_median_raw=adc_med,
                brain_ml=float(bm.sum() * vox_ml), has_mask=acute is not None or total is not None)
    if acute is not None:
        feat.update(acute_ml=float(acute.sum() * vox_ml),
                    acute_n_components=int(ndimage.label(acute)[1]),
                    acute_radc_mean=float(radc[acute].mean()) if acute.any() else np.nan,
                    acute_radc_p10=float(np.percentile(radc[acute], 10)) if acute.any() else np.nan,
                    acute_slices=int(acute.any(axis=(0, 1)).sum()))
    if total is not None:
        feat["total_lesion_ml"] = float(total.sum() * vox_ml)
        if acute is not None:
            feat["chronic_ml"] = float((total & ~acute).sum() * vox_ml)

    # ---- registration TRACE -> MNI 2 mm
    fixed = ants.image_read(str(template_path))
    tmask = ants.image_read(str(tmask_path))
    tmpd = od / "_tmp"; tmpd.mkdir(exist_ok=True)

    def to_ants(arr, name):                          # via a temp file: works on every ANTsPy version
        p = tmpd / f"{name}.nii.gz"
        nib.save(nib.Nifti1Image(arr.astype(np.float32), tr_img.affine), str(p))
        return ants.image_read(str(p))

    mov = to_ants(np.where(bm, tr, 0), "moving")
    reg = ants.registration(fixed=fixed, moving=mov, type_of_transform="SyN" if syn else "Affine",
                            aff_metric="mattes", syn_metric="mattes", verbose=False)
    tfm = reg["fwdtransforms"]
    import shutil
    for i, f in enumerate(tfm):
        shutil.copy(f, od / (f"{sub}_affine.mat" if f.endswith(".mat") else f"{sub}_warp{i}.nii.gz"))

    def warp(arr, interp):
        im = to_ants(arr, "warp_in")
        return ants.apply_transforms(fixed=fixed, moving=im, transformlist=tfm, interpolator=interp).numpy()

    outs = {"trace": warp(trz, "linear"), "radc": warp(radc, "linear"),
            "brainmask": (warp(bm, "linear") > 0.5).astype(np.float32)}
    if acute is not None:
        outs["lesionAcute"] = (warp(acute, "linear") > 0.5).astype(np.float32)
    if total is not None:
        outs["lesion"] = (warp(total, "linear") > 0.5).astype(np.float32)
    aff = nib.load(str(template_path)).affine
    for k, v in outs.items():
        nib.save(nib.Nifti1Image(v.astype(np.float32), aff), str(od / f"{sub}_space-MNI2_{k}.nii.gz"))
    shutil.rmtree(tmpd, ignore_errors=True)
    tm = tmask.numpy() > 0.5
    feat["qc_brain_dice_mni"] = float(dice(outs["brainmask"], tm))
    feat["seconds"] = round(time.time() - t0, 1)
    les_png = outs.get("lesionAcute", outs.get("lesion", np.zeros_like(outs["trace"])))
    qc_png(out / "qc" / f"{sub}.png", outs["trace"], les_png, tm, sub,
           f"acute {feat.get('acute_ml', float('nan')):.1f} mL | brain Dice {feat['qc_brain_dice_mni']:.2f}")
    done.write_text(json.dumps(feat))
    return feat


def main():
    ap = argparse.ArgumentParser()
    from common import BIDS, PREP
    ap.add_argument("--root", default=str(BIDS))
    ap.add_argument("--out", default=str(PREP))
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) // 2))
    ap.add_argument("--subjects", nargs="*")
    ap.add_argument("--syn", action="store_true", help="affine + SyN instead of affine only")
    a = ap.parse_args()
    root, out = Path(a.root), Path(a.out)
    (out / "qc").mkdir(parents=True, exist_ok=True)

    # MNI152 2 mm template + brain mask from nilearn (bundled, no download)
    from nilearn.datasets import load_mni152_template, load_mni152_brain_mask
    tp, mp = out / "MNI152_2mm_T1.nii.gz", out / "MNI152_2mm_brainmask.nii.gz"
    if not tp.exists():
        t = load_mni152_template(resolution=2); m = load_mni152_brain_mask(resolution=2)
        nib.save(nib.Nifti1Image(t.get_fdata().astype(np.float32) * (m.get_fdata() > 0), t.affine), str(tp))
        nib.save(nib.Nifti1Image((m.get_fdata() > 0).astype(np.float32), m.affine), str(mp))

    subs = a.subjects or sorted([d.name for d in root.iterdir() if d.name.startswith("sub-")],
                                key=lambda s: int(s.split("-")[1]))
    print(f"{len(subs)} subjects | workers {a.workers} | {'SyN' if a.syn else 'Affine'} -> {out}", flush=True)
    rows, fails, t0 = [], [], time.time()
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        futs = {ex.submit(process, s, root, out, a.syn, tp, mp): s for s in subs}
        for i, f in enumerate(as_completed(futs), 1):
            s = futs[f]
            try:
                rows.append(f.result())
            except Exception as e:
                fails.append((s, repr(e))); rows.append(dict(participant_id=s, status="error", error=repr(e)[:300]))
                (out / "errors.log").open("a").write(f"{s}\n{traceback.format_exc()}\n")
            if i % 20 == 0 or i == len(subs):
                el = time.time() - t0
                print(f"  {i}/{len(subs)} done | {el/60:.1f} min | ETA {el/i*(len(subs)-i)/60:.0f} min | errors {len(fails)}", flush=True)
                pd.DataFrame(rows).to_csv(out / "features_dwi.csv", index=False)
    df = pd.DataFrame(rows); df.to_csv(out / "features_dwi.csv", index=False)
    ok = df[df.status == "ok"]
    print(f"\nFinished: {len(ok)} ok, {len(fails)} errors, {(df.status=='missing_dwi').sum()} missing DWI.")
    if len(ok) and "qc_brain_dice_mni" in ok:
        print(f"Registration QC: median brain Dice {ok.qc_brain_dice_mni.median():.3f}; "
              f"{(ok.qc_brain_dice_mni < 0.85).sum()} subjects below 0.85 (check their PNGs in {out/'qc'})")


if __name__ == "__main__":
    main()
