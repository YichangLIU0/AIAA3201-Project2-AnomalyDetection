# AIAA3201 Project 2: Few-shot industrial anomaly detection

Anonymous public release. Contributor names and student IDs are intentionally omitted.

## Deliverables
- `reports/main.pdf`: CVPR-style English course report; source and bibliography are alongside it.
- `deliverables/anomaly_demo.zip`: recorded actual predictions, panels, and decision records.
- `deliverables/project2_code.zip`: source, reproducibility records, compact results, and report.
- `reference_results/all12_summary.csv`: scope-separated aggregate metrics; per-run metrics and manifests are retained.
- `docs/FAILURE_ANALYSIS.md`: 24 visually inspected FP/FN cases.

Public repository: https://github.com/YichangLIU0/AIAA3201-Project2-AnomalyDetection . The public files are anonymous. No arXiv or course-platform submission has been made.

## Protocol
All twelve VisA categories and every official test image. Standard inputs are 224x224; the fixed-budget high-resolution extension reads original images at 448x448. All metric masks are 224x224. DINO full memory: 1/2/4/8 references, three seeds. Compressed banks, classical baselines, resolution/fusion/peak ablations: eight references, three seeds. AnomalyCLIP: one released fixed checkpoint. Every operating threshold uses 40 additional disjoint normal calibration images per category, so this is not a strict K-total-normal-images protocol. No threshold is fitted to test labels. AUPRO is normalized over background FPR [0, 0.3]. See docs/LOW_MISS_PROTOCOL.md for the frozen choices and exploratory additions.

## Updated twelve-category results

All 2,162 official test images, three reference seeds, eight references per category, and 40 extra disjoint normal calibration images. Numbers below are category-macro means. FNR and FPR use each variant's own normal-calibrated threshold, not test-optimized or empirically matched-FPR thresholds.

| Method | Image AUROC | FNR | FPR | Bank |
|---|---:|---:|---:|---:|
| coreset25 | 0.8725 | 45.36% | 4.84% | 6 MiB |
| hi448 | 0.9252 | 31.56% | 5.56% | 6 MiB |
| multiscale | 0.9013 | 41.92% | 4.79% | 6 MiB |
| hi448_peak | 0.9321 | 26.92% | 5.96% | 6 MiB |

Higher resolution does not improve every category: macaroni1 and macaroni2 have higher FNR with hi448 than coreset25. Peak scoring is an exploratory addition. The uniform hi448_peak demo default reflects this observed aggregate tradeoff, not independent post-selection validation. REVIEW defers an item and is never counted as successful detection.

The report includes 36 inspected errors, size-stratified misses, paired corrected/introduced decisions, half-bank controls and measured online costs. Four new videos contain 96 fresh predictions checked against formal outputs; two original videos add 24.

The source is a resource-constrained adaptation of upstream methods, not an exact published-protocol reproduction. The original smoke test is separate from the formal results.

## Run
Use a Python 3.11 environment with CUDA-enabled PyTorch. The commands below use a Windows virtual environment named `.venv`. See fresh installation below. Full evaluation requires downloading the dataset and model weights.

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests
.\.venv\Scripts\python.exe scripts/run_all12_baselines.py
.\.venv\Scripts\python.exe coursework/run_lowmiss.py
.\.venv\Scripts\python.exe coursework/run_peak.py
.\.venv\Scripts\python.exe scripts/analyze_all12.py
.\.venv\Scripts\python.exe scripts/analyze_lowmiss_pairs.py
.\.venv\Scripts\python.exe scripts/build_lowmiss_examples.py
.\.venv\Scripts\python.exe scripts/build_demo.py
.\.venv\Scripts\python.exe scripts/build_demo_coreset.py
.\.venv\Scripts\python.exe scripts/build_demo_lowmiss.py
.\.venv\Scripts\python.exe scripts/benchmark_lowmiss.py
.\.venv\Scripts\python.exe coursework/run_robustness.py
.\.venv\Scripts\python.exe scripts/build_all12_report.py
# Compile reports/main.tex before the artifact audit.
.\.venv\Scripts\python.exe scripts/audit_project.py
```

GPU experiments should run sequentially for meaningful timings. DINO and CLIP skip completed metric files. To change the protocol, use a separate output directory and manifests rather than mixing runs. The recorded scripts define the exact parameters; command-line configurability is intentionally limited.

## Fresh installation (tested on Windows)
Use Python 3.11 and an isolated environment. Install CUDA-enabled PyTorch/torchvision compatible with the machine, then `pip install -r requirements-coursework.txt`. Core dependency versions are recorded in `requirements-coursework.txt`. Clone the exact revisions in `docs/source_versions.txt` into `third_party/`:

- https://github.com/dammsi/AnomalyDINO
- https://github.com/facebookresearch/dinov2
- https://github.com/amazon-science/spot-diff
- https://github.com/zqhang/AnomalyCLIP
- https://github.com/cvpr-org/author-kit

Keep upstream licenses. Download VisA from its official project and run its official one-class split preparation script. Required layout: `data/VisA_pytorch/1cls/<category>/train/good`, `test/good`, `test/bad`, and `ground_truth/bad`. Expected counts are checked by the audit. Use the official VisA archive and split CSV.

Place official `dinov2_vits14_pretrain.pth` and `ViT-L-14-336px.pt` in `checkpoints/`. The CLIP backbone hash is validated by the inference script. The AnomalyCLIP target-VisA prompt checkpoint is `third_party/AnomalyCLIP/checkpoints/9_12_4_multiscale_visa/epoch_15.pth`. Consult upstream download instructions and licensing. Do not replace it with the target-MVTec checkpoint. Model weights, datasets, and a virtual environment are excluded from the code ZIP.

## Demo
After generating eight-shot banks, run `python coursework/demo.py` to choose a category and upload an image. To save a prediction without a GUI:

```powershell
python coursework/demo.py --image path/to/image.JPG --category candle --method hi448_peak --output prediction.png
```

Output: image score, normal-calibrated threshold, OK/NG, heatmap, overlay, and a calibrated pixel mask. `--method` accepts full/random25/pooled25/coreset25/hi448/multiscale/core224_peak/hi448_peak. Optional `--review-quantile 0.9` introduces REVIEW using only calibration data; a deferred item is not counted as a detected defect. The category must match the input. The recorded video is an image-inference demonstration, not a camera acquisition recording; all regenerated predictions are checked against the formal saved outputs. A wrong prediction remains visible in the video and is not silently relabeled.

## Archive versus local output
The code ZIP stores historical metrics and manifests under `reference_results/`. It intentionally has no completed-run markers in `outputs/coursework/`, so a new run will not skip experiments whose pixel predictions were excluded from the ZIP. The desktop project retains full predictions in `outputs/coursework/`. Use all12_summary.json/csv for the updated study; summary.json/csv retain the original-six snapshot. Compare new summaries with the matching historical scope; bitwise cross-machine equality is not promised.

After extracting VisA into `data/VisA`, prepare the official split:

```powershell
python third_party/spot-diff/utils/prepare_data.py --split-type 1cls --data-folder data/VisA --save-folder data/VisA_pytorch --split-file third_party/spot-diff/split_csv/1cls.csv
```

The full audit requires regenerated predictions and demo artifacts. Qualitative review records belong to the reported study and fall back to reference_results/; a fresh rerun does not constitute a new visual review of its errors. Metrics unit tests require no dataset or GPU. FFmpeg is required only to build the video. The existing demo builder uses this machine's FFmpeg and Arial font paths; adjust them on another machine.

## Report build
Use TeX with the official CVPR style, or the provided local Tectonic installation. Run from `reports/`: `tectonic main.tex` (XeTeX; fontspec and TeX Gyre Termes). Tectonic may need internet access for its TeX resource cache. The already-built PDF can be read without any software installation.

## Interpretation and attribution
Random25 reduces an eight-shot bank from 24 to 6 MiB with a small measured ranking loss; Pooled25 performs worse at the same budget. Matching time excludes feature extraction and is not end-to-end latency. Three seeds quantify reference variation only. This coursework does not establish CVPR-level novelty. See `docs/SOURCES.md`, the report, and upstream licenses for attribution. AI assistance was used for code, experiment execution, and writing; authors must review and comply with course disclosure requirements.

## Anonymous public artifacts
The report and source archive omit contributor names and student IDs. The public repository owner remains visible. `deliverables/project2_paper_source.zip` contains the anonymous manuscript source. `deliverables/anomaly_demo.zip` contains the recorded actual predictions. No arXiv submission is implied.

## Coverage-based extension
See `docs/CORESET_EXTENSION.md`. The added coreset is a projected greedy k-center selection inspired by PatchCore, not an original algorithm. It stores 4096 original vectors after selection in a fixed 128-dimensional projection. Run it after `run_dino.py` has generated the shared feature cache. `python scripts/analyze_coreset.py` generates reference-coverage statistics, per-category results, paired FP/FN transitions and example images. Test labels never enter bank selection or threshold calibration. Offline selection cost is reported separately from online matching. Brightness stress tests still cover only full/random/pooled banks.

Historical compression-stage results (original six-category macro average, three seeds): at 6 MiB, random25 versus coreset25 image AUROC is 0.92084 versus 0.92938, pixel AP 0.41184 versus 0.42748, FPR 0.05111 versus 0.04185, and FNR 0.28778 versus 0.30333. Coreset improves ranking and lowers false alarms here, but misses more anomalies at the normal-calibrated operating point. Offline selection costs about 2.95 seconds. The average worst reference cosine distance decreases from 0.28776 to 0.11104. This is not a novel algorithm or proof of general superiority.

The demo defaults to hi448_peak for its lower observed aggregate miss rate. This is an exploratory choice made after the reported comparison, not an independently validated deployment selection. It is applied uniformly across categories; there is no test-fitted category-specific routing.
