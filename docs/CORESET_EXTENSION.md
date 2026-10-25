# Coverage-preserving memory compression extension

Problem: random quarter-size memory lowers storage and query cost, but may leave some nominal reference features poorly represented. Existing errors include benign boundaries and texture triggering anomaly decisions. The link between compression coverage and those visual errors is a hypothesis, not an established cause.

Fixed design before reading extension outcomes:
- Eight reference images, original three reference seeds, all six required classes and all 960 test images.
- Same frozen DINO features, rotations, PCA query masks, nearest-neighbor scoring, smoothing, metrics and disjoint 40-normal calibration split.
- Same 4096-vector / 6 MiB storage budget for random25, pooled25 and coreset25; full bank remains a 24 MiB reference.
- Coreset: normalize original features; Gaussian random projection from 384 to 128 dimensions (seed = reference seed + 2000); first center furthest from projected mean; repeatedly select the point furthest from its nearest selected center. Store original 384-dimensional selected vectors, not projections. No test labels used in selection.
- This is a PatchCore-inspired projected greedy k-center baseline, not the complete PatchCore detector or a novel selection algorithm.

Evidence chain:
1. Measure mean, 95th percentile and maximum nearest-bank cosine distance for all original reference features in ORIGINAL 384-dimensional space. No defect labels are required.
2. Evaluate image/pixel ranking and the normal-calibrated operating point; count FP/FN fixed and introduced relative to each existing method.
3. Report offline selection cost as well as bank size and matching time. Selection memory is measured separately from GPU feature extraction.
4. Preserve per-category/seed results and paired comparisons. A better reference-coverage radius alone does not prove better detection or real-world robustness.

Interpretation rule: retain negative findings. Do not tune projection dimension, memory budget or thresholds using test outcomes. If coverage improves but false alarms or missed defects do not, describe that mismatch explicitly. High test accuracy or a prettier heatmap is not required for a valid course exploration.
