"""
Cohort table for all 1,715 SOOP sessions and the frozen validation splits.

Outputs (in results/):
  cohort_all.csv     one row per subject: clinical variables, discharge mRS, scanner, DWI features, label flags, fold
  folds_repeats.csv  20 repeats x 5 folds (for the mean and SD of the reference models)
  cohort_summary.txt counts used in the cohort flowchart (Fig. 2a)
Validation schemes (columns in cohort_all.csv):
  fold          0..4 stratified by poor outcome x field strength (seed 2026) -> scheme "cv5"
  scanner_group held-out scanner model                                    -> scheme "loso"
  field         1.5 T train / 3 T test                                     -> scheme "field"
"""
import json
import numpy as np, pandas as pd
from sklearn.model_selection import StratifiedKFold
from common import BIDS, PREP, RES

RES.mkdir(parents=True, exist_ok=True)
p = pd.read_csv(BIDS / "participants.tsv", sep="\t", na_values=["n/a"])
for c in ["age", "nihss", "bmi", "priorstroke", "gs_rankin_6isdeath", "acuteischaemicstroke"]:
    p[c] = pd.to_numeric(p[c], errors="coerce")
p = p.rename(columns={"gs_rankin_6isdeath": "mrs"})
p["toast"] = pd.to_numeric(p.etiology.astype(str).str.extract(r"^(\d)")[0], errors="coerce")
p["sex_m"] = (p.sex == "M").astype(float).where(p.sex.notna())
p["race_b"] = (p.race == "b").astype(float).where(p.race.notna())

subs = sorted([d.name for d in BIDS.iterdir() if d.is_dir() and d.name.startswith("sub-")], key=lambda s: int(s[4:]))
rows = []
for s in subs:
    j = BIDS / s / "dwi" / f"{s}_rec-TRACE_dwi.json"
    meta = json.loads(j.read_text()) if j.exists() else {}
    rows.append(dict(participant_id=s, field=meta.get("MagneticFieldStrength"),
                     manufacturer=meta.get("Manufacturer"), scanner_group=meta.get("ManufacturersModelName")))
sc = pd.DataFrame(rows)
f = pd.read_csv(PREP / "features_dwi.csv")
d = sc.merge(p, on="participant_id", how="left").merge(f, on="participant_id", how="left")
d["dwi_ok"] = d.status.eq("ok")
d["labelled"] = d.mrs.notna() & d.age.notna() & d.nihss.notna() & d.dwi_ok
d["y_poor"] = np.where(d.mrs.notna(), (d.mrs >= 3).astype(float), np.nan)
d["has_mask"] = d.has_mask.fillna(False).astype(bool)
for c in ["acute_ml", "total_lesion_ml", "chronic_ml"]:
    d[c + "_filled"] = d[c].fillna(0.0)                    # no expert mask -> 0 mL + has_mask indicator
d["log_acute_ml"] = np.log1p(d.acute_ml_filled)
d["qc_fail"] = d.qc_brain_dice_mni < 0.85

lab = d[d.labelled].copy()
strata = (lab.y_poor.astype(int) * 2 + (lab.field >= 3).astype(int)).values
d["fold"] = -1
skf = StratifiedKFold(5, shuffle=True, random_state=2026)
for k, (_, te) in enumerate(skf.split(lab, strata)):
    d.loc[lab.index[te], "fold"] = k
reps = []
for r in range(20):
    skf = StratifiedKFold(5, shuffle=True, random_state=1000 + r)
    for k, (_, te) in enumerate(skf.split(lab, strata)):
        reps.append(pd.DataFrame(dict(participant_id=lab.participant_id.values[te], rep=r, fold=k)))
pd.concat(reps).to_csv(RES / "folds_repeats.csv", index=False)
d.to_csv(RES / "cohort_all.csv", index=False)

L = d[d.labelled]
lines = [
    f"Subjects with MRI: {len(d)}",
    f"DWI preprocessing ok: {d.dwi_ok.sum()} | QC brain Dice < 0.85: {int(d.qc_fail.sum())}",
    f"With participants.tsv metadata (age): {d.age.notna().sum()} | NIHSS: {d.nihss.notna().sum()} | TOAST: {d.toast.notna().sum()}",
    f"With discharge mRS: {d.mrs.notna().sum()} | labelled cohort (mRS + age + NIHSS + DWI ok): {len(L)}",
    f"Labelled: poor outcome {int(L.y_poor.sum())} ({L.y_poor.mean():.1%}) | with expert mask {int(L.has_mask.sum())}",
    f"Labelled by field: {L.field.value_counts().to_dict()}",
    f"Labelled by scanner: {L.scanner_group.value_counts().to_dict()}",
    f"Fold sizes: {L.fold.value_counts().sort_index().to_dict()}",
    f"Unlabelled scans usable for auxiliary training: {int((~d.labelled & d.dwi_ok).sum())}",
]
(RES / "cohort_summary.txt").write_text("\n".join(lines))
print("\n".join(lines))
