# Part 2 results (member B)

Protocol `visa6_sorted20_official_refs_native_v2`, `shared_protocol` from `d07aae3` (unchanged on this branch). All formal runs ran
on the shared RTX A6000 server on 2026-11-06 with the code now in `methods/part2/` (see `methods/part2/PROVENANCE.md`).
This folder holds every run's `run.json`, `metrics.json` and `metrics_reeval.json`, the summaries, the server logs, and the
upstream parity evidence. Failure-analysis panels and tables are in `results/part2_failure_analysis/`; AnomalyCLIP records are
in `results/part2_anomalyclip/`.

**Map files are not in Git.** Native-resolution float32 maps are about 0.9–1.2 GB per run. The maps of DINO seed 0 (24 runs)
and all 6 AnomalyCLIP runs (27 GB) are kept on member B's machine and on the server, and all 4,800 of them match the sha256
recorded in `run.json`. The maps of DINO seeds 1 and 2 were deleted on 2026-11-08 when the server disk filled up; their scores,
metrics and map hashes remain in the records. Any map can be regenerated deterministically with `methods/part2/run_dino_formal.py`
and checked against the hash in `run.json`. The `map` paths inside `run.json` (`maps/0000.npy`, ...) refer to those local files.

**Status:** runs and evaluation are complete. Manual inspection of the hard samples (section 4) and the Part 1/2 cross-check
with A are still open.

## 1. Evidence that the results are sound

| Check | Result |
|---|---|
| `check_ready` with full hashes | passed, six categories |
| Shared tests / `part2_official` tests | 26 of 26 / 7 of 7 |
| DINO runs | 72 of 72 complete; every `metrics.json` reproduced exactly by re-evaluation |
| AnomalyCLIP runs | 6 of 6 complete; every `metrics.json` reproduced exactly by re-evaluation |
| DINO, server vs laptop (22 finished laptop runs) | worst image score difference 1.25e-6; largest image AUROC change 1e-4 (candle 4-shot seed 2) |
| CLIP, server vs provisional laptop run (6 categories) | worst image score difference 1.07e-6; image AUROC identical to 4 decimals |
| `dino_interface.py` vs saved run (5 images, same server) | score and map difference 0 |

Map files are not byte-identical between laptop and server, so compare them numerically rather than by hash.

## 2. Results

Six-category equal-weight means. DINO std is across reference seeds 0/1/2 (`ddof=0`). CLIP has one run per category, so
no seed stability was measured.

| Method | Image AUROC | Pixel AUROC | AUPRO@0.30 | Image FPR | Image FNR |
|---|---|---|---|---|---|
| DINO 1-shot | 0.9461 ± 0.0084 | 0.9776 ± 0.0006 | 0.9403 | 0.136 | 0.158 |
| DINO 2-shot | 0.9533 ± 0.0046 | 0.9792 ± 0.0007 | 0.9459 | 0.143 | 0.128 |
| DINO 4-shot | 0.9615 ± 0.0040 | 0.9809 ± 0.0003 | 0.9481 | 0.141 | 0.117 |
| DINO 8-shot | 0.9662 ± 0.0057 | 0.9821 ± 0.0002 | 0.9514 | 0.160 | 0.094 |
| AnomalyCLIP zero-shot | 0.8698 | 0.9660 | 0.8888 | 0.188 | 0.253 |

Per category (DINO values are means across seeds):

| Category | DINO 1-shot I-AUROC | DINO 8-shot I-AUROC | CLIP I-AUROC | DINO 8-shot FPR / FNR | CLIP FPR / FNR |
|---|---|---|---|---|---|
| candle | 0.900 | 0.935 | 0.8094 | 0.163 / 0.100 | 0.340 / 0.200 |
| capsules | 0.977 | 0.981 | 0.8278 | 0.094 / 0.100 | 0.067 / 0.550 |
| cashew | 0.936 | 0.946 | 0.7596 | 0.033 / 0.297 | 0.040 / 0.480 |
| chewinggum | 0.985 | 0.989 | 0.9722 | 0.187 / 0.017 | 0.160 / 0.060 |
| fryum | 0.957 | 0.981 | 0.9270 | 0.127 / 0.033 | 0.140 / 0.190 |
| pipe_fryum | 0.921 | 0.965 | 0.9228 | 0.353 / 0.017 | 0.380 / 0.040 |

Full tables: `results/part2_dino/summary/` and `results/part2_anomalyclip/summary/aggregate.json`.

CLIP checkpoint: `9_12_4_multiscale_visa/epoch_15.pth`. Despite the name it is trained on the MVTec AD **test split**: upstream
`train.sh` trains that folder with `--dataset mvtec`, `train.py` builds its `Dataset` with the default `mode='test'` (the test
entries of `meta.json`), and the paper (arXiv:2310.18961, Sec. 4.1) says "we fine-tune AnomalyCLIP using the test data on MVTec
AD". `test.sh` tests it with `--dataset visa`. The file is tracked in the upstream repository at commit `3911738` and is unchanged
(sha256 `415c5dcb…`). Nothing was fitted on VisA. Report it as "zero-shot inference, with 20 extra normal images for threshold
calibration".

## 3. Resources

| Method | Time per image | Peak GPU allocated / reserved | Setup | Disk per run |
|---|---|---|---|---|
| DINO 1-shot | 265 ms | 156 / 374 MB | bank 0.89 s | 0.9–1.2 GB |
| DINO 2-shot | 266 ms | 218 / 664 MB | bank 2.03 s | 0.9–1.2 GB |
| DINO 4-shot | 256 ms | 342 / 1168 MB | bank 4.01 s | 0.9–1.2 GB |
| DINO 8-shot | 268 ms | 590 / 2472 MB | bank 8.45 s | 0.9–1.2 GB |
| AnomalyCLIP | 329–341 ms | 2029 / 2188 MB | model load not timed | about 0.9 GB |

Peak values are the maximum across runs; DINO times and bank builds are means across runs.

**Measurement scope.** Device: one NVIDIA RTX A6000 (CUDA, fp32, batch size 1). Input: DINO smaller edge 448
(about 448×490 to 448×672 after cropping to multiples of 14); CLIP 518×518. Time per image is the wall time of one
`predict_one` call with CUDA synchronisation before and after, after 25 (DINO) or 10 (CLIP) warm-up passes. It **includes**
image reading, preprocessing, the backbone forward pass (feature extraction), PCA masking, 1-NN search, the image score and
the anomaly map at the native image resolution (DINO: upstream `dists2map`, bilinear plus Gaussian σ=4 on about 1.5 MP on
the CPU; CLIP: Gaussian σ=4 at 518 then bilinear resize). It **excludes** model loading, memory-bank construction (reported
separately as "Setup"), threshold calibration, writing map files and metric computation. Upstream `run_anomalydino.py` stops
its timer before map generation, so our DINO times are not comparable to the paper's runtime tables. Peak GPU memory is
`torch.cuda.max_memory_allocated/reserved` over the calibration and test pass with the memory bank resident. It excludes
the CUDA context and other processes.

**Treat the timings as indicative.** Other users' jobs were sharing the GPU (7–14 GB in use at about 50% utilisation) and the
CPU, and the laptop's A500 measured 180 ms against about 287 ms here on the same run. The cause was not investigated. If timing is
a claim in the report, re-measure when the server is quiet, and measure Part 1 and Part 3 on the same device with the same scope.

## 4. Failure analysis

Files: `results/part2_failure_analysis/` (`panels/`, `samples.csv`, `defect_size.csv`, `normal_shift.csv`). Test labels
were used only to explain errors after evaluation.

**Status of the manual check:** the plan requires manual inspection of at least five hard samples. The notes below are Claude's
first pass over 8 of the 18 panels. **They are not the required manual inspection.** Member B must look at the panels and confirm or
correct them. Panel colour scales are for display only.

### Defect size (all anomalous test images, seeds 0/1/2 for DINO)

Size is the defect area in DINO patches (14 px at short side 448). "Peak in GT" means the single highest map pixel falls inside
the mask, which is a strict test.

| Defect size (patches) | n per seed set | DINO 1-shot detected | DINO 8-shot detected | DINO 8-shot peak in GT | CLIP detected (of n/3) | CLIP peak in GT |
|---|---|---|---|---|---|---|
| under 1 | 489 | 0.669 | 0.796 | 0.286 | 0.656 | 0.294 |
| 1–4 | 672 | 0.841 | 0.906 | 0.500 | 0.723 | 0.598 |
| 4–16 | 363 | 0.975 | 0.994 | 0.711 | 0.901 | 0.702 |
| 16–64 | 168 | 0.958 | 0.976 | 0.881 | 0.786 | 0.679 |
| 64–256 | 69 | 1.000 | 1.000 | 0.855 | 0.826 | 0.826 |
| over 256 | 39 | 1.000 | 1.000 | 0.769 | 0.538 (n=13) | 0.846 |

What this shows:

1. **Coarse patches limit small defects.** 65% of anomalous images (1161 of 1800 across seeds) have defects smaller than
   4 patches, and that is where detection and localisation both fall. More references help detection (under 1 patch: 0.67 to
   0.80) but barely help localisation (0.26 to 0.29). This supports trying direction A (multi-scale) on this failure.
2. **Many detections are not localised.** 45% of DINO 1-shot, 43% of DINO 8-shot and 40% of CLIP correct image decisions have the
   map peak outside the defect (676 of 1515, 699 of 1631, 179 of 448). An image AUROC or NG decision alone does not show the
   defect was found. The strict single-pixel test overstates this somewhat, because a peak beside a tiny defect also counts
   as a miss.
3. **Most misses are also mislocalised.** Of DINO 1-shot misses, 75 of 285 had the peak inside the defect; for 8-shot it is 39 of 169.
   So the image score or threshold is the main problem in only a minority of misses. The 8 panels overrepresent this case.
4. **CLIP is weak on large defects at image level:** 7 of 13 detected above 256 patches, though the map peak is in the defect in 11
   of 13. The sample is small, and this has not been investigated further.

### False alarms

Calibration FPR is 0.05 by construction, but test FPR averages 0.136–0.160 for DINO and 0.188 for CLIP, and varies strongly by
category (DINO 8-shot: cashew 0.03, pipe_fryum 0.35).

- **The fair baseline is about 9%, not 5%.** With 20 calibration images, the linear 95th percentile sits close to the second-highest
  score. For exchangeable scores, the expected exceedance is about 0.095 × 0.95 + 0.048 × 0.05 = 9.3%. This is an analytic
  estimate, not a measurement.
- **Distribution shift explains part of the rest at most.** Normal test medians sit above calibration medians for pipe_fryum
  (+0.019 at 8-shot) and candle (+0.015), the categories with high FPR. Chewinggum has FPR 0.19 with a shift of only +0.003, so a median
  shift does not explain it.
- **Open:** the remaining excess is unexplained. Do not change the threshold rule using test results. The rule belongs to A, and
  test labels must not choose it.

### First-pass notes on panels (Claude, not yet confirmed)

| Panel | Observation |
|---|---|
| `candle_fn_039` | Long thin defect near the wick. Both DINO maps show scattered wax-texture activity with no clear response on it. CLIP says NG, but its brightest spot is on a different candle, and its wick responses also appear on the normal `candle_fp_0488`. Its correct decision looks accidental. |
| `candle_fp_0488` | Normal. DINO shows scattered activity on the wax surface and rims, with no visible defect. |
| `capsules_fn_087` | Missed by all three. DINO's peak coincides with the tiny marked region, but scores stay below threshold. |
| `cashew_small_055` | Defect is about 0.1 patch and barely visible. No method responds at it; peaks are elsewhere. |
| `chewinggum_fp_467` | Normal, flagged by all three. Strong response near the left end of the piece and on surface marks. |
| `fryum_fn_031` | Peak on the small defect in both DINO maps, but the score stays below threshold. An example of the minority case in point 3. |
| `pipe_fryum_fp_400` | Normal, flagged by all three. DINO's response follows the tube edges, which looks like a pose or shape difference from the references, and 8 shots do not remove it. Likely useful for direction B or A, but unproven. |
| `pipe_fryum_fn_050` | Clear localised peak. 1-shot misses it with a threshold of 0.298, well above its calibration median of 0.197. 8-shot catches it. |

Not yet viewed by anyone: `candle_small_059`, `capsules_fp_363`, `capsules_small_038`, `cashew_fn_063`, `cashew_fp_130`,
`chewinggum_fn_033`, `chewinggum_small_070`, `fryum_fp_239`, `fryum_small_038`, `pipe_fryum_small_039`.

## 5. Other points for the report

- **Source hash:** the 22 laptop runs recorded an older `shared_protocol` hash (`76ad3e6b…`). Formal results should use the server's
  72 runs (`10b6d973…`). Laptop `capsules/8shot_seed1` is incomplete and unused.
- **Server changes to B's code:** the disk guard in `run_dino_formal.py` now checks the disk holding the outputs.
- **Environment:** matches the laptop for all compared packages except `pip`. `thop` was added after it was first missed.
- **Provisional AUROC comparison:** the aggregator's comparison with the provisional DINO runs was not run, because those folders
  are not on the server. The laptop comparison in section 1 covers the same question on 22 runs. An independent audit
  (2026-11-08) also compared all 72 server runs with the laptop's provisional `run_dino_official.py` runs (same references):
  worst image score difference 1.7e-6, largest image AUROC change 1e-4.

## 6. Disclosures for the final report (Methods / Experimental setup)

These came out of an independent audit on 2026-11-08. Items 1–9 are acceptable deviations that only need stating; item 10 is
about claims, not code. None needs a rerun.

1. **Reference selection.** "Following the course protocol, the first 20 lexicographically sorted normal training images of
   each category are reserved for threshold calibration and removed from the reference pool before sampling. References are
   then chosen with the upstream deterministic rule `sorted(pool)[seed·k:(seed+1)·k]` (seeds 0/1/2 are block indices, not
   random draws) on the remaining pool. Our reference sets therefore differ from those of Damm et al.: their seed-0 references
   are among our calibration images, and none of their references for any k and seed occurs in ours."
2. **Nearest-neighbour search.** "Instead of FAISS `IndexFlatL2` (`--faiss_on_cpu`), exact 1-NN search is implemented in PyTorch
   (fp32, TF32 disabled) with the upstream `L2_normalized` metric, distance = ‖a−b‖²/2 = 1 − cos. On candle 1-shot seed 0,
   per-image scores differ from unmodified upstream by at most 5×10⁻⁶ (the upstream CSV is rounded to 5 decimals)."
3. **Backbone loading.** "DINOv2 ViT-S/14 is built from a pinned local copy of facebookresearch/dinov2 (commit `7764ea0`) and
   loaded with the released `dinov2_vits14_pretrain.pth` (sha256 `b938bf1b…`, strict state-dict load), instead of
   `torch.hub.load('facebookresearch/dinov2')`, which follows the unpinned main branch. Inference is fp32 as in upstream. xFormers
   is not installed, so DINOv2 uses its standard attention." (Meta publishes no checksum for this file; the hash is ours.)
4. **What is reused unchanged.** "Image reading, `prepare_image`, feature extraction, PCA masking, rotation augmentation,
   `mean_top1p` and `dists2map` are imported unchanged from the official AnomalyDINO repository (commit `b9d1c26`); only the
   driver loop is ours. VisA `agnostic` = test-image masking on, 8 reference rotations, references not masked."
5. **Category subset and aggregation.** "We evaluate the six required categories through shared manifests instead of editing
   `get_dataset_info`; means are equal-weight over six categories, whereas the paper's VisA means cover all twelve. For each
   shot count we report the mean over seeds and the population standard deviation (ddof = 0) of the three per-seed
   six-category means."
6. **Metrics implementation.** "Pixel metrics are computed at the native image resolution, including normal test images with
   all-zero masks, without per-image normalisation. AUPRO uses a shared exact streaming implementation (8-connectivity, FPR
   limit 0.30) instead of upstream `compute_pro`; on a sample run (cashew 1-shot seed 0) the two agree to 5×10⁻¹¹."
7. **Efficiency.** State the scope in section 3 above: device, input size, that feature extraction and native-resolution
   map generation are included, that bank construction is reported separately, and that the GPU was shared.
8. **AnomalyCLIP checkpoint.** "We use the authors' released prompt checkpoint `9_12_4_multiscale_visa/epoch_15.pth`, which the
   authors trained on the MVTec AD test set (auxiliary data). Nothing is trained or tuned on VisA, and no prompt is chosen with
   VisA labels; 20 normal training images per category only set the OK/NG threshold."
9. **AnomalyCLIP evaluation resolution.** "Inference follows the official `test.py --dataset visa` settings (ViT-L/14@336px,
   518×518, features_list [24], depth 9, n_ctx 12, t_n_ctx 4, DPAM layer 20, σ = 4, fp32). Unlike `test.py`, which scores
   518×518 maps against squashed masks, the 518×518 map is bilinearly resized to the native resolution and scored against
   the original masks, so that all methods share one evaluation. Our pixel metrics are therefore not directly comparable to
   the paper's."
10. **Comparison with the paper.** Present it as indicative only: per-category results in the paper (Table 7) are at
    resolution 672; at 448 the paper reports only twelve-category means (Table 3: 1/2/4/8-shot image AUROC 85.6/88.3/91.3/92.6,
    pixel AUROC 97.5/97.8/98.0/98.2); its references include our calibration images; and its PRO uses upstream code. These
    values were checked against the arXiv v3 PDF. Do not write that we "reproduced the paper's numbers".
