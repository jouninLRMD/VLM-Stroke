# VLM-Stroke

Code and results for **"Checkable lesion reports from diffusion MRI in acute ischemic stroke: a grounded vision-language model trained and evaluated on mask-derived findings"** (L. R. Mercado-Diaz, C. Barrera-Causil, L. S. Donado Pinto, H. F. Posada-Quintero; submitted to Artificial Intelligence in Medicine).

There are no public radiology reports for stroke MRI, and no simple way to check whether what a vision-language model writes is true or comes from the image. This repository builds a setting in which every statement can be checked. From the expert lesion masks of 1,451 diffusion MRI scans in the public [Stroke Outcome Optimization Project (SOOP)](https://openneuro.org/datasets/ds004889/versions/1.1.2) and an arterial territory atlas, seven findings are derived and written as template reports that are true by construction. A 2.5D image encoder trained on these findings feeds a small instruction-tuned language model adapted with LoRA, which writes the report and predicts the discharge outcome. Every generated statement is then scored against the mask-derived reference.

## What is in the repository

```
reproduce.py                runs the whole study, stage by stage
scripts/download_soop.py    downloads SOOP from OpenNeuro and checks every file against data/soop_manifest.csv
scripts/fetch_atlas.py      downloads the arterial territory atlas (Liu et al., 2023)
src/preprocess_dwi.py       brain mask, relative ADC, affine registration to MNI152 2 mm, lesion features, QC images
src/cohort_splits.py        cohort table and frozen folds
src/atlas_features.py       lesion load per arterial territory
src/baselines.py            clinical and lesion reference models for the outcome comparison
src/build_cache.py          memory-mapped image arrays
src/augment.py              3D augmentation used to train the encoder
src/make_reports.py         mask-grounded findings and reference reports
src/encoder.py              grounded 2.5D encoder, direct classifier and visual tokens
src/vlm.py                  grounded vision-language model (projector + Qwen2.5-1.5B-Instruct with LoRA)
src/evaluate_reports.py     first scoring of the generated reports
src/report_scoring.py       report parsing and fact-level scoring shared by the evaluation scripts
src/ablations.py            grounding test, sampled reports, ablations, seeds, model sizes, decoder without pretraining
src/zeroshot.py             zero-shot general vision-language model (Qwen2.5-VL-3B-Instruct)
src/analysis.py             main evaluation: factuality with bootstrap CIs, retrieval, text overlap, numeric fidelity,
                            strata, outcome through language; figures and number macros
src/ablation_eval.py        ablations, grounding, uncertainty, tuned and joint classifiers, zero-shot reference
results/                    reference reports, generated reports of every model and variant, predictions and the
                            manuscript outputs
```

The preprocessing, cohort and reference-model scripts are shared with the companion repository [CALM-Stroke](https://github.com/jouninLRMD/CALM-Stroke), so both studies use exactly the same images, patients and folds. If you already ran CALM-Stroke, point `CALM_DATA` to the same data folder and start this pipeline at the `reports` stage.

## Installation

Python 3.11 and a CUDA GPU with at least 16 GB of memory are needed for the encoder and the language models (the 3B model and the zero-shot model need about 20 GB); the evaluation runs on a CPU. The paper used PyTorch 2.11 (CUDA 12.8) on one NVIDIA RTX 4000 Ada (20 GB).

```bash
git clone https://github.com/jouninLRMD/VLM-Stroke.git
cd VLM-Stroke
conda env create -f environment.yml
conda activate vlm-stroke
# install the PyTorch build that matches your CUDA version, for example:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
```

The language models are downloaded from Hugging Face the first time they are used: Qwen2.5-1.5B-Instruct (about 3 GB), Qwen2.5-0.5B-Instruct and Qwen2.5-3B-Instruct for the model-size comparison, and Qwen2.5-VL-3B-Instruct (about 7 GB) for the zero-shot reference.

## Reproducing the paper

From the repository root:

```bash
python reproduce.py            # everything, from the download to the manuscript outputs
python reproduce.py --dry-run  # list the commands without running them
```

| Stage | What it does | Time (approximate) |
|---|---|---|
| `download`, `atlas` | SOOP metadata, lesion masks and DWI (2.8 GB); atlas | depends on the connection |
| `preprocess` | preprocessing and registration of 1,715 sessions | 1 to 2 h with 6 CPU workers |
| `cohort`, `features`, `baselines`, `cache` | cohort, folds, territory loads, reference models, image arrays | 30 min |
| `reports` | findings and reference reports of the 1,451 scans with an expert mask | 2 min |
| `encoder` | grounded encoder, five folds | 20 min |
| `pool` | pooled encoder features for retrieval and the joint classifier | 1 min |
| `vlm` | language model training and report generation, five folds | 1 h |
| `evaluate` | first scoring | 1 min |
| `ablations` | image-swap test, sampled reports, 9 trained variants | 8 to 9 h |
| `zeroshot` | zero-shot reference | 2 h |
| `analysis`, `ablation_eval` | all metrics, figures and the number macros of the manuscript (`results/paper`) | 15 min |

Each stage can be run alone with `--only <stage>`, or the pipeline can be resumed with `--from <stage>`. Stages whose outputs already exist are skipped unless `--force` is given.

### Without a GPU

The reference reports, the generated reports of every model and variant, the direct classifier predictions, the pooled encoder features and the outcome predictions are included in `results/`. To recompute every table, figure and number of the paper from them:

```bash
python reproduce.py --only analysis ablation_eval
```

`results/oof` holds the out-of-fold outcome predictions of the reference models and, for the comparison in the outcome analysis, of CALM-Stroke and 2.5D MIL from the CALM-Stroke repository.

## Data

SOOP (OpenNeuro ds004889, version 1.1.2) is released under a CC0 license by Absher et al., *Scientific Data* 11, 839 (2024), doi:[10.1038/s41597-024-03667-5](https://doi.org/10.1038/s41597-024-03667-5). The arterial territory atlas is from Liu et al., *Scientific Data* 10, 74 (2023), doi:[10.1038/s41597-022-01923-0](https://doi.org/10.1038/s41597-022-01923-0), as distributed in the [SOOP demonstration repository](https://github.com/neurolabusc/StrokeOutcomeOptimizationProjectDemo) under a BSD 2-Clause license; it is downloaded, not redistributed here. The language models are used under the licenses listed on their Hugging Face model pages.

By default the data go to `data/` inside the repository. The environment variables `CALM_DATA` (download, preprocessed volumes, atlas), `CALM_WORK` (image cache, visual tokens and model weights) and `CALM_RESULTS` (tables and predictions) select other folders.

## Citation

If you use this code, please cite the paper (the reference will be updated after publication):

```
L. R. Mercado-Diaz, C. Barrera-Causil, L. S. Donado Pinto and H. F. Posada-Quintero,
"Checkable lesion reports from diffusion MRI in acute ischemic stroke: a grounded vision-language model trained
and evaluated on mask-derived findings," submitted to Artificial Intelligence in Medicine, 2026.
```

## License

The code is released under the GNU General Public License v3.0 (see `LICENSE`).

## Contact

Luis R. Mercado-Diaz, Department of Biomedical Engineering, University of Connecticut.
Questions and issues are welcome through the GitHub issue tracker.
