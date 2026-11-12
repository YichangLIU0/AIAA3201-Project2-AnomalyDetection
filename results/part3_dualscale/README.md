# Part 3 direction A: dual-scale AnomalyDINO (448 + 672)

Protocol `visa6_sorted20_official_refs_native_v2`: 6 VisA categories, shots {1,2,4,8}, seeds {0,1,2}, giving 72 runs per configuration.
The code is in `methods/part3/`. `run_dualscale.py` was frozen before the grid ran, with sha256 `cfa34135…737f3`; see `frozen_config.json`.

| config | what it is |
|---|---|
| `s448` | The Part 2 formal AnomalyDINO (vits14, 448), recomputed in the same runner as a control |
| `s672` | The same pipeline at a smaller edge of 672. It uses the 448 PCA mask projected to the 672 grid |
| `fuse` | The 448 distance grid is bilinearly resampled to the 672 grid and averaged 0.5/0.5 with the 672 grid, followed by one `dists2map` |

Nothing was tuned on test data. The 0.5/0.5 weight and 672 were fixed before the grid. The threshold rule is unchanged: the 95th percentile of the 20 sorted calibration normals.

## Baseline check (`verify_baseline.json`)
`s448` matches the formal Part 2 runs for all 72 runs:
- the maximum difference in test and calibration scores is 0.0;
- every metric is exactly equal;
- the regenerated maps equal the saved formal maps on 24 runs (maximum difference 0.0).

## Macro means over 6 categories and 3 seeds

| config | shots | I-AUROC | P-AUROC | I-AP | P-AP | AUPRO@0.3 | FPR | FNR |
|---|---|---|---|---|---|---|---|---|
| s448 | 1 | 0.9461 | 0.9776 | 0.9677 | 0.4435 | 0.9403 | 0.1356 | 0.1583 |
| s448 | 2 | 0.9533 | 0.9792 | 0.9716 | 0.4586 | 0.9459 | 0.1430 | 0.1278 |
| s448 | 4 | 0.9615 | 0.9809 | 0.9765 | 0.4793 | 0.9481 | 0.1407 | 0.1172 |
| s448 | 8 | 0.9662 | 0.9821 | 0.9787 | 0.4864 | 0.9514 | 0.1596 | 0.0939 |
| s672 | 1 | 0.9502 | 0.9774 | 0.9686 | 0.4695 | 0.9477 | 0.1600 | 0.1383 |
| s672 | 2 | 0.9584 | 0.9791 | 0.9737 | 0.4850 | 0.9528 | 0.1783 | 0.1100 |
| s672 | 4 | 0.9641 | 0.9809 | 0.9770 | 0.5020 | 0.9555 | 0.1717 | 0.1039 |
| s672 | 8 | 0.9688 | 0.9822 | 0.9797 | 0.5126 | 0.9582 | 0.1709 | 0.0828 |
| fuse | 1 | 0.9572 | 0.9811 | 0.9736 | 0.4886 | 0.9528 | 0.1572 | 0.1322 |
| fuse | 2 | 0.9634 | 0.9825 | 0.9772 | 0.5014 | 0.9573 | 0.1874 | 0.0967 |
| fuse | 4 | 0.9686 | 0.9840 | 0.9803 | 0.5179 | 0.9593 | 0.1763 | 0.0956 |
| fuse | 8 | 0.9731 | 0.9850 | 0.9827 | 0.5251 | 0.9615 | 0.1893 | 0.0717 |

Per-category and per-seed results are in `analysis/aggregate_*.json` and `analysis/per_run.csv`.

## Paired differences over all 72 runs (`analysis/paired.json`)

| metric | fuse − s448 (wins/losses) | s672 − s448 | fuse − s672 |
|---|---|---|---|
| I-AUROC | +0.0088 (53/19) | +0.0037 (45/27) | +0.0052 (60/11) |
| P-AUROC | +0.0032 (72/0) | −0.0001 (38/34) | +0.0033 (70/2) |
| P-AP | +0.0413 (72/0) | +0.0253 (57/15) | +0.0160 (63/9) |
| AUPRO@0.3 | +0.0113 (72/0) | +0.0071 (69/3) | +0.0042 (62/10) |
| FPR | +0.0328 (worse in 44, better in 19) | +0.0255 | +0.0073 |
| FNR | −0.0253 (better in 42, worse in 17) | −0.0156 | −0.0097 |

The runs share categories and reference images, so they are not independent samples. Treat the sign-test p-values in the JSON as a rough guide only.

## Findings
- Fusion improves every ranking metric (AUROC, AP and AUPRO) at every shot count, and it beats 672 alone. The gain is therefore not just the effect of a finer grid.
- The gain is largest for small defects (`analysis/defect_size.csv`). For defects of 0–1 patches at 448, 1-shot detection rises from 0.669 to 0.720, and the map-peak-in-GT rate rises from 0.258 to 0.425.
- **Trade-off:** at the unchanged 95th-percentile threshold, fusion misses fewer defects (FNR down) but flags more normal images (FPR up by about 3 points). This trade-off is already visible in a check that used only held-out normal images, before the grid ran (`heldout/candle.json`, candle, seed 0):

  | config | share of held-out normals above the threshold, 1-shot | 8-shot |
  |---|---|---|
  | s448 | 0.067 | 0.133 |
  | s672 | 0.20 | 0.27 |

  Twenty calibration normals give a noisy 95th-percentile estimate; see the Part 2 threshold diagnostic.
- FN/FP flips relative to s448 are in `analysis/flips.json`. Example panels (8-shot, seed 0) are in `analysis/panels/`, and still need human inspection.

## Cost (`timing/timing.json`)
These timings were taken while the RTX A6000 was otherwise idle. They cover 30 candle test images, end to end per image, including the map at native size.

| config | median ms/img, 1-shot | median ms/img, 8-shot | peak GPU allocated, 8-shot |
|---|---|---|---|
| s448 | ~183 | ~185 | ~0.67 GB |
| s672 | ~313 | ~317 | ~1.18 GB |
| fuse | ~328 | ~320 | ~1.18 GB |

## Deviations and limitations
- **Shared disk:** the server disk was nearly full (under 100 MB free) for most of the run. Native-resolution maps were therefore not saved:
  - metrics were computed in memory using `shared_protocol.metrics`;
  - distance grids were kept as float16 for analysis only, and every score and metric comes from float32;
  - grids are not in git; they are backed up off-repo.
- **Disk plan:** the disk plan was not agreed with A beforehand.
- **Scope:** only one fusion weight (0.5) and one second scale (672) were tested, by design. No weight search was run.
