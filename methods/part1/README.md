# Part 1: fixed Template Difference baseline

This is a formal baseline, separate from the earlier `integration_only=true` smoke test.
The shared protocol and its release are unchanged. ECC and Autoencoder are separate,
optional methods and are not implemented or claimed here.

## Frozen design

Before running the formal test grid, `template.py` fixes all parameters: shared
K normal references, eight reference rotations, short edge 448 with bottom/right
crop to multiples of 14, nearest template by 28x28 RGB descriptor MSE, absolute
RGB difference, Gaussian sigma 2, and 3x3 grayscale closing. The image score is
the mean of the largest 1% of processed-map pixels. No learned weights, test-label
parameter search, per-image normalization, or test-time augmentation are used.
Linear interpolation restores the map to the original image size for evaluation;
the original masks are never resized. See `frozen_config.json` for exact settings
and SHA256 hashes of both inference and runner source files.

The 20 fixed calibration normals set the linear 95th-percentile threshold.
Every official test image is evaluated. The full grid has 6 categories x 4 shot
counts (1/2/4/8) x 3 reference seeds (0/1/2) = 72 runs. Seeds select the shared
reference blocks; they are not three identical reruns.

## Run from the repository root

Install `methods/part1/requirements.txt` in a project environment. Provide the
directory containing the six prepared category folders:

```powershell
python -m unittest discover -s methods/part1/tests -v
python -u -m methods.part1.run_template --data-root 'E:/path/to/VisA_pytorch/1cls'
```

`--categories candle --shots 1 --seeds 0` runs one genuine formal cell as a
pipeline check. Rerunning the full command skips completed matching cells.
Do not change frozen sources/settings while filling the same output directory.
If interrupted after prediction, completed `run.json` files can be evaluated on
resume. Incomplete prediction directories must be moved aside before retrying;
the writer deliberately refuses to overwrite partial predictions.

Raw, native float32 maps, run metadata, checksums and timings go to
`outputs/part1_template_v1/` (ignored by Git; about 65 GiB for the entire grid).
Compact metrics and image-level predictions go to `results/part1_template/`.
The macro summary is produced only after all 72 cells pass shared evaluation.
Image AUROC, native pixel AUROC, AP, AUPRO at FPR 0.3, calibrated FPR/FNR, and
seed population standard deviation are retained. Visualization is post hoc and
must never alter these predictions or select a revised model on test labels.

## Resource measurements

This classical implementation runs on CPU with four OpenCV threads. Each image
timer covers JPG reading, preprocessing, selection, difference, postprocessing
and native-map interpolation. It excludes reference setup, three calibration
warmup images, map saving and metric evaluation. Reference setup is timed
separately. GPU memory is 0 because no GPU is used; this is not a measurement of
a GPU implementation. Hardware and dependencies are recorded. Compare latency
with neural methods only while disclosing their different execution devices.
