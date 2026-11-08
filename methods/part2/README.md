# Part 2: official AnomalyDINO baseline and AnomalyCLIP (member B)

Formal results: `results/part2_dino/` (72 runs), `results/part2_anomalyclip/` (6 zero-shot runs) and
`results/part2_failure_analysis/`. The write-up, including the deviations to disclose, is
[`results/part2_dino/PART2_REPORT.md`](../../results/part2_dino/PART2_REPORT.md). How the code relates to the files that produced
those results is in [`PROVENANCE.md`](PROVENANCE.md).

## Configuration

- **AnomalyDINO**: `dinov2_vits14`, smaller edge 448, preprocess `agnostic` (VisA: test-image PCA masking on, 8 reference
  rotations, references unmasked), 1/2/4/8 shots × reference seeds 0/1/2 × 6 categories = 72 runs. Image reading,
  `prepare_image`, feature extraction, PCA masking, rotations, `mean_top1p` and `dists2map` are imported unchanged from
  upstream AnomalyDINO `b9d1c26`. Exact 1-NN is done in PyTorch instead of FAISS (same metric, 1 − cos).
- **AnomalyCLIP**: authors' checkpoint `9_12_4_multiscale_visa/epoch_15.pth` (trained by the authors on the MVTec AD test set),
  settings of the official `test.py --dataset visa`; zero-shot (`shots=0`, no references). The 518×518 map is resized to the
  native resolution for evaluation.
- References, the 20 calibration images, the test set, the threshold and all metrics come only from `shared_protocol`.

## Files

| File | Role |
|---|---|
| `run_dino_formal.py` | Formal DINO grid (writes through `shared_protocol.run_detector`) |
| `run_anomalyclip_formal.py` | Formal AnomalyCLIP zero-shot runs |
| `run_dino_official.py` | Upstream-based building blocks used by the formal runner (its own `main` is the earlier provisional runner) |
| `run_anomalyclip_official.py` | CLIP model loading used by the formal runner (its own `main` is provisional) |
| `aggregate_dino_formal.py` | CPU post-processing: re-evaluates all 72 runs and aggregates them |
| `failure_analysis.py` | CPU post-analysis: defect size, false alarms, hard-sample panels (test labels used only here) |
| `dino_interface.py` | Fixed DINO interface for member A's Part 3 directions C/F (`DinoDetector`) |
| `run_upstream_reference.py`, `compare_with_upstream.py` | Parity check against unmodified upstream `run_anomalydino.py` |
| `make_temp_manifest.py` | Provisional manifests used before `shared_protocol` existed (kept for the tests) |
| `tests/test_part2_official.py` | CPU-only unit tests |

## Setup (not stored in Git)

1. Data: prepare `data/VisA_pytorch/1cls/` with `shared_protocol` (see the top-level README).
2. Upstream code in `third_party/` at these commits:
   ```powershell
   git clone https://github.com/dammsi/AnomalyDINO third_party/AnomalyDINO; git -C third_party/AnomalyDINO checkout b9d1c2648e3a5247437d4d953d907a8f3d994457
   git clone https://github.com/facebookresearch/dinov2 third_party/dinov2; git -C third_party/dinov2 checkout 7764ea0f912e53c92e82eb78a2a1631e92725fc8
   git clone https://github.com/zqhang/AnomalyCLIP third_party/AnomalyCLIP; git -C third_party/AnomalyCLIP checkout 3911738c0867544f545a076ad78f3f11d9ecbfdf
   ```
   The AnomalyCLIP repository already contains the prompt checkpoint.
3. Weights in `checkpoints/`: `dinov2_vits14_pretrain.pth` from `https://dl.fbaipublicfiles.com/dinov2/dinov2_vits14/` and
   `ViT-L-14-336px.pt` from OpenAI CLIP. Their sha256 values are in `PROVENANCE.md`; the CLIP runner checks its hash.
4. Environment: the exact server environment is `requirements-server.txt` (Python 3.11, PyTorch 2.8.0+cu128, CUDA GPU).

## Run from the repository root

```powershell
python -m unittest discover -s methods/part2/tests -v
python methods/part2/run_dino_formal.py --categories candle --shots 1 --seeds 0   # one cell as a check
python methods/part2/run_dino_formal.py                                          # all 72 runs, skips finished ones
python methods/part2/run_anomalyclip_formal.py
python methods/part2/aggregate_dino_formal.py
python methods/part2/failure_analysis.py
python methods/part2/dino_interface.py --run outputs/dino/candle/1shot_seed0 --images 5
```

Outputs go to `outputs/` (ignored by Git). A DINO run writes about 0.9–1.2 GB of native-resolution maps and the runner stops if
less than 20 GB would remain. Copy `run.json`, `metrics.json` and summaries into `results/` by hand; never the maps.

Timing scope, device and the other points that must be stated in the final report are listed in sections 3 and 6 of
`PART2_REPORT.md`.
