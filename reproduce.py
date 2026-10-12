"""
Reproduces every result of the VLM-Stroke paper, from the public SOOP download to the tables, figures and numbers.

Stages, in order:
  download    SOOP metadata, lesion masks and DWI from OpenNeuro (about 2.8 GB)
  atlas       arterial territory atlas (Liu et al. 2023)
  preprocess  brain mask, relative ADC, affine registration to MNI152 2 mm, lesion features, QC images
  cohort      cohort table and frozen folds
  features    lesion load per arterial territory (used to derive the territories of the reports)
  baselines   clinical and lesion reference models for the outcome comparison
  cache       memory-mapped image arrays
  reports     mask-grounded findings and reference reports for the 1,451 scans with an expert mask
  encoder     grounded 2.5D encoder per fold, direct classifier predictions and visual tokens (GPU)
  pool        pooled encoder features for the retrieval baseline and the joint classifier
  vlm         grounded vision-language model per fold: training and report generation (GPU)
  evaluate    first scoring of the generated reports
  ablations   grounding test, sampled reports, ablations, seeds, model sizes, decoder without pretraining (GPU)
  zeroshot    zero-shot general vision-language model (GPU)
  analysis    main evaluation, figures and number macros (results/paper)
  ablation_eval  evaluation of the ablations, grounding, uncertainty, tuned and joint classifiers (results/paper)

Usage (from the repository root):
  python reproduce.py                          # everything
  python reproduce.py --dry-run                # print the commands only
  python reproduce.py --from vlm               # start at a stage
  python reproduce.py --only analysis ablation_eval   # recompute the paper's numbers from the saved predictions
Finished runs of the language models are skipped; use --force to run them again.
"""
import argparse, os, subprocess, sys, time
from pathlib import Path

REPO = Path(__file__).resolve().parent
SRC, SCRIPTS = REPO / "src", REPO / "scripts"
RES = Path(os.environ.get("CALM_RESULTS", str(REPO / "results")))
STAGES = ["download", "atlas", "preprocess", "cohort", "features", "baselines", "cache", "reports", "encoder", "pool",
          "vlm", "evaluate", "ablations", "zeroshot", "analysis", "ablation_eval"]

# (tag, arguments of src/ablations.py), in the order used for the paper
ABLATIONS = [
    (None,     ["--dump-imnet"]),
    ("full_samples", ["--tag", "full", "--eval-only", "--samples", "5"]),
    ("full_shuffle", ["--tag", "full", "--eval-only", "--shuffle"]),
    ("noimage", ["--tag", "noimage", "--no-image"]),
    ("noclin",  ["--tag", "noclin", "--no-clin"]),
    ("pooled",  ["--tag", "pooled", "--pooled-only"]),
    ("imnet",   ["--tag", "imnet", "--feats", "imnet"]),
    ("seed1",   ["--tag", "seed1", "--seed", "1"]),
    ("seed2",   ["--tag", "seed2", "--seed", "2"]),
    ("llm05b",  ["--tag", "llm05b", "--llm", "Qwen/Qwen2.5-0.5B-Instruct"]),
    ("llm3b",   ["--tag", "llm3b", "--llm", "Qwen/Qwen2.5-3B-Instruct", "--batch", "4"]),
    ("scratch", ["--tag", "scratch", "--scratch", "--lr", "5e-4", "--epochs", "10"]),
]


def done(pattern):
    return len(list((RES / "vlm").glob(pattern))) >= 5


def run(cmd, log=None, dry=False):
    print("  $ " + " ".join(str(c) for c in cmd) + (f"  > {log}" if log else ""), flush=True)
    if dry:
        return
    t0 = time.time()
    if log:
        log.parent.mkdir(parents=True, exist_ok=True)
        with open(log, "w") as fh:
            rc = subprocess.call(cmd, stdout=fh, stderr=subprocess.STDOUT, cwd=REPO)
    else:
        rc = subprocess.call(cmd, cwd=REPO)
    if rc != 0:
        sys.exit(f"Command failed (exit code {rc})" + (f"; see {log}" if log else ""))
    print(f"    done in {(time.time() - t0) / 60:.1f} min", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", nargs="+", choices=STAGES)
    ap.add_argument("--from", dest="start", choices=STAGES)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) // 2), help="CPU workers")
    ap.add_argument("--force", action="store_true", help="rerun language-model stages whose outputs already exist")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    if a.only:
        stages = [s for s in STAGES if s in a.only]
    elif a.start:
        stages = STAGES[STAGES.index(a.start):]
    else:
        stages = STAGES
    py, L = sys.executable, RES / "logs"
    for st in stages:
        print(f"\n=== {st}", flush=True)
        if st == "download":
            run([py, SCRIPTS / "download_soop.py"], dry=a.dry_run)
        elif st == "atlas":
            run([py, SCRIPTS / "fetch_atlas.py"], dry=a.dry_run)
        elif st == "preprocess":
            run([py, SRC / "preprocess_dwi.py", "--workers", str(a.workers)], dry=a.dry_run)
        elif st == "cohort":
            run([py, SRC / "cohort_splits.py"], dry=a.dry_run)
        elif st == "features":
            run([py, SRC / "atlas_features.py"], dry=a.dry_run)
        elif st == "baselines":
            run([py, SRC / "baselines.py"], dry=a.dry_run)
        elif st == "cache":
            run([py, SRC / "build_cache.py"], dry=a.dry_run)
        elif st == "reports":
            run([py, SRC / "make_reports.py"], log=L / "make_reports.txt", dry=a.dry_run)
        elif st == "encoder":
            if done("direct_fold*.csv") and not a.force:
                print("  encoder outputs exist, skipped")
            else:
                run([py, SRC / "encoder.py"], log=L / "encoder.txt", dry=a.dry_run)
        elif st == "pool":
            run([py, SRC / "analysis.py", "--pool"], log=L / "pool.txt", dry=a.dry_run)
        elif st == "vlm":
            if done("gen_fold?.csv") and not a.force:
                print("  generated reports exist, skipped")
            else:
                run([py, SRC / "vlm.py"], log=L / "vlm.txt", dry=a.dry_run)
        elif st == "evaluate":
            run([py, SRC / "evaluate_reports.py"], log=L / "evaluate_reports.txt", dry=a.dry_run)
        elif st == "ablations":
            for tag, args in ABLATIONS:
                if tag and not a.force:
                    pat = {"full_samples": "abl/gen_full_fold?.csv", "full_shuffle": "abl/gen_full_shuffle_fold?.csv"}.get(
                        tag, f"abl/gen_{tag}_fold?.csv")
                    if done(pat):
                        print(f"  {tag}: outputs exist, skipped")
                        continue
                run([py, SRC / "ablations.py"] + args, log=L / f"ablation_{tag or 'imnet_dump'}.txt", dry=a.dry_run)
        elif st == "zeroshot":
            if done("abl/gen_zeroshot_*_fold?.csv") and not a.force:
                print("  zero-shot outputs exist, skipped")
            else:
                run([py, SRC / "zeroshot.py"], log=L / "zeroshot.txt", dry=a.dry_run)
        elif st == "analysis":
            run([py, SRC / "analysis.py"], log=L / "analysis.txt", dry=a.dry_run)
        elif st == "ablation_eval":
            run([py, SRC / "ablation_eval.py"], log=L / "ablation_eval.txt", dry=a.dry_run)
    print("\nFinished. Manuscript outputs: results/paper")


if __name__ == "__main__":
    main()
