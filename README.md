# AIAA3201 Project 2: Few-shot industrial anomaly detection

Anonymous public release. Contributor names and student IDs are intentionally omitted.

## Deliverables
- `reports/main.pdf`: CVPR-style English course report; source and bibliography are alongside it.
- `deliverables/anomaly_demo.zip`: two actual-prediction videos, 24 frames, and decision records.
- `deliverables/project2_code.zip`: source, reproducibility records, compact results, and report.
- `reference_results/summary.csv`: recorded aggregate metrics; per-run metrics and manifests are retained.
- `docs/FAILURE_ANALYSIS.md`: 24 visually inspected FP/FN cases.

Public repository: https://github.com/YichangLIU0/AIAA3201-Project2-AnomalyDetection . No arXiv submission has been made.

## Protocol
Six categories, 960 official VisA test images, 224x224 input/masks. DINO full memory: 1/2/4/8 reference images x seeds 0/1/2 x six classes. Random25 and Pooled25: eight-shot, three seeds, six classes. Template ECC and normal-only compact AE: the same eight-shot references and seeds. Coreset25: eight-shot, three seeds, six classes, same 6 MiB budget as random25/pooled25. AnomalyCLIP: one released fixed checkpoint, no target fitting. Every operating threshold uses 40 additional normal calibration images/class, excluded from reference/training images. Thus this is not a strict K-total-normal-images protocol. Test masks are evaluation-only. AUPRO is normalized over FPR [0, 0.3].

The source is a resource-constrained adaptation of upstream methods, not an exact published-protocol reproduction. The original smoke test is separate from the formal results.

## Run
Use a Python 3.11 environment with CUDA-enabled PyTorch. The commands below use a Windows virtual environment named `.venv`. See fresh installation below. Full evaluation requires downloading the dataset and model weights.

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests
.\.venv\Scripts\python.exe coursework/run_dino.py
.\.venv\Scripts\python.exe coursework/run_coreset.py
.\.venv\Scripts\python.exe coursework/run_baselines.py
.\.venv\Scripts\python.exe coursework/run_anomalyclip.py
.\.venv\Scripts\python.exe coursework/run_robustness.py
.\.venv\Scripts\python.exe coursework/summarize.py
.\.venv\Scripts\python.exe scripts/build_tables.py
.\.venv\Scripts\python.exe scripts/build_demo.py
.\.venv\Scripts\python.exe scripts/build_demo_coreset.py
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
python coursework/demo.py --image path/to/image.JPG --category candle --method random25 --output prediction.png
```

Output: image score, normal-calibrated threshold, OK/NG, heatmap, overlay, and a calibrated pixel mask. `--method` accepts full/random25/pooled25/coreset25. The category must match the input. The recorded video is an image-inference demonstration, not a camera acquisition recording; all 24 regenerated predictions are checked against the formal saved outputs. A wrong prediction remains visible in the video and is not silently relabeled.

## Archive versus local output
The code ZIP stores historical metrics and manifests under `reference_results/`. It intentionally has no completed-run markers in `outputs/coursework/`, so a new run will not skip experiments whose pixel predictions were excluded from the ZIP. The desktop project retains full predictions in `outputs/coursework/`. Compare new summaries with the historical results; bitwise cross-machine equality is not promised.

After extracting VisA into `data/VisA`, prepare the official split:

```powershell
python third_party/spot-diff/utils/prepare_data.py --split-type 1cls --data-folder data/VisA --save-folder data/VisA_pytorch --split-file third_party/spot-diff/split_csv/1cls.csv
```

The full audit requires regenerated predictions and demo artifacts. Metrics unit tests require no dataset or GPU. FFmpeg is required only to build the video. The existing demo builder uses this machine's FFmpeg and Arial font paths; adjust them on another machine.

## Report build
Use TeX with the official CVPR style, or the provided local Tectonic installation. Run from `reports/`: `tectonic main.tex` (XeTeX; fontspec and TeX Gyre Termes). Tectonic may need internet access for its TeX resource cache. The already-built PDF can be read without any software installation.

## Interpretation and attribution
Random25 reduces an eight-shot bank from 24 to 6 MiB with a small measured ranking loss; Pooled25 performs worse at the same budget. Matching time excludes feature extraction and is not end-to-end latency. Three seeds quantify reference variation only. This coursework does not establish CVPR-level novelty. See `docs/SOURCES.md`, the report, and upstream licenses for attribution. AI assistance was used for code, experiment execution, and writing; authors must review and comply with course disclosure requirements.

## Anonymous public artifacts
The report and source archive omit contributor names and student IDs. The public repository owner remains visible. `deliverables/project2_paper_source.zip` contains the anonymous manuscript source. `deliverables/anomaly_demo.zip` contains two actual-prediction recordings. No arXiv submission is implied.

## Coverage-based extension
See `docs/CORESET_EXTENSION.md`. The added coreset is a projected greedy k-center selection inspired by PatchCore, not an original algorithm. It stores 4096 original vectors after selection in a fixed 128-dimensional projection. Run it after `run_dino.py` has generated the shared feature cache. `python scripts/analyze_coreset.py` generates reference-coverage statistics, per-category results, paired FP/FN transitions and example images. Test labels never enter bank selection or threshold calibration. Offline selection cost is reported separately from online matching. Brightness stress tests still cover only full/random/pooled banks.

Latest extension results (six-category macro average, three seeds): at 6 MiB, random25 versus coreset25 image AUROC is 0.92084 versus 0.92938, pixel AP 0.41184 versus 0.42748, FPR 0.05111 versus 0.04185, and FNR 0.28778 versus 0.30333. Coreset improves ranking and lowers false alarms here, but misses more anomalies at the normal-calibrated operating point. Offline selection costs about 2.95 seconds. The average worst reference cosine distance decreases from 0.28776 to 0.11104. This is not a novel algorithm or proof of general superiority.
