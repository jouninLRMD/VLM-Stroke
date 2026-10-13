# VLM-Stroke results (auto-generated)

## Report factuality (held-out folds)

```
source                   VLM generated report  Direct structured classifier  VLM (same scans as direct)  Majority-class reference
n                                    1451.000                      1451.000                    1451.000                  1451.000
acc_side                                0.817                         0.802                       0.817                     0.421
acc_vol_bin                             0.711                         0.713                       0.711                     0.365
acc_multiple                            0.695                         0.683                       0.695                     0.509
acc_chronic                             0.895                         0.880                       0.895                     0.894
acc_restr                               0.607                         0.553                       0.607                     0.427
terr_micro_f1                           0.719                         0.675                       0.719                     0.515
terr_exact_set                          0.499                         0.408                       0.499                     0.323
terr_hallucination_rate                 0.231                         0.133                       0.231                     0.363
terr_omission_rate                      0.325                         0.448                       0.325                     0.568
f1_ACA                                  0.599                         0.385                       0.599                     0.000
f1_MCA                                  0.861                         0.855                       0.861                     0.778
f1_DEEP                                 0.647                         0.635                       0.647                     0.000
f1_PCA                                  0.573                         0.539                       0.573                     0.000
f1_THAL                                 0.409                         0.224                       0.409                     0.000
f1_BS                                   0.457                         0.000                       0.457                     0.000
f1_CEREB                                0.664                         0.574                       0.664                     0.000
vol_spearman                            0.881                           NaN                       0.881                       NaN
vol_mae_ml                             12.261                           NaN                      12.261                       NaN
all_fields_correct                      0.324                         0.272                       0.324                     0.061
```

## Outcome (discharge mRS 3-6) predicted through language

```
                                  n    auc              ci  cal_slope  vlm_minus_this  delta_lo  delta_hi
model                                                                                                    
VLM (image + clinical text)     718  0.773  (0.739, 0.806)      0.774             NaN       NaN       NaN
Clinical logistic (R0)          718  0.756   (0.721, 0.79)      0.957           0.017    -0.015     0.050
Clinical + expert lesion (R1b)  718  0.798  (0.765, 0.829)      0.929          -0.025    -0.049    -0.002
2.5D anchored, seed 0           718  0.808  (0.774, 0.839)      0.818          -0.035    -0.062    -0.010
```

## Example generated vs reference reports

**sub-2**
- generated: FINDINGS: Multiple acute infarcts on the left side, involving the anterior cerebral artery (ACA) and middle cerebral artery (MCA) cortical territories (MCA branches: frontal). Acute lesion volume 52.0 mL (large). Mild diffusion restriction (mean relative ADC 0.86). No chronic infarct.
- reference: FINDINGS: Multiple acute infarcts on both sides, involving the middle cerebral artery (MCA) cortical and posterior cerebral artery (PCA) territories (MCA branches: frontal, parietal, occipital). Acute lesion volume 82.9 mL (very large). Mild diffusion restriction (mean relative ADC 0.93). No chronic infarct.

**sub-8**
- generated: FINDINGS: Multiple acute infarcts on both sides, involving the anterior cerebral artery (ACA) and middle cerebral artery (MCA) cortical territories (MCA branches: frontal). Acute lesion volume 31.0 mL (large). No clear diffusion restriction (mean relative ADC 1.24). No chronic infarct.
- reference: FINDINGS: Multiple acute infarcts on both sides, involving the anterior cerebral artery (ACA) and middle cerebral artery (MCA) cortical territories (MCA branches: frontal). Acute lesion volume 79.7 mL (very large). No clear diffusion restriction (mean relative ADC 1.05). No chronic infarct.

**sub-13**
- generated: FINDINGS: Acute infarct on the right side, involving the anterior cerebral artery (ACA) territory. Acute lesion volume 10.2 mL (medium). Mild diffusion restriction (mean relative ADC 0.86). No chronic infarct.
- reference: FINDINGS: Multiple acute infarcts on the right side, involving the anterior cerebral artery (ACA) and posterior cerebral artery (PCA) territories. Acute lesion volume 1.1 mL (small). No clear diffusion restriction (mean relative ADC 1.14). No chronic infarct.

**sub-20**
- generated: FINDINGS: Acute infarct on the left side, involving the middle cerebral artery (MCA) cortical territory (MCA branches: temporal). Acute lesion volume 0.4 mL (small). Mild diffusion restriction (mean relative ADC 0.86). No chronic infarct.
- reference: FINDINGS: Acute infarct on both sides, involving the anterior cerebral artery (ACA) territory. Acute lesion volume 0.9 mL (small). Mild diffusion restriction (mean relative ADC 0.87). No chronic infarct.

**sub-24**
- generated: FINDINGS: Multiple acute infarcts on the right side, involving the middle cerebral artery (MCA) cortical territory (MCA branches: parietal). Acute lesion volume 1.4 mL (small). Mild diffusion restriction (mean relative ADC 0.86). No chronic infarct.
- reference: FINDINGS: Multiple acute infarcts on the right side, involving the middle cerebral artery (MCA) cortical territory (MCA branches: frontal, parietal, occipital). Acute lesion volume 6.7 mL (medium). Mild diffusion restriction (mean relative ADC 0.87). No chronic infarct.
