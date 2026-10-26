# Low-miss extension: fixed protocol before new test outcomes

Motivation: existing six-category coreset evaluations miss 512/1317 tiny-defect image-seed events (area <=0.5%), versus 1/141 events with area >2%. This is a diagnostic association, not proof of causation.

Frozen design:
- Extend the original 224-pixel baselines to all 12 VisA classes with the same official split and three reference seeds. The six previously untested categories provide an additional generalization check; this is not a blinded clinical-style confirmation.
- Compare original 224 coreset25 (4096 vectors, 6 MiB) with native-image 448 coreset (4096 vectors, 6 MiB) and a two-scale fusion (2048 original vectors from each resolution, total 4096 vectors / 6 MiB).
- Resolution 448 gives a 32x32 patch grid; read the ORIGINAL image before resizing, never upsample the cached 224 image.
- Same eight reference images, eight rotations, official PCA query masking, frozen DINOv2-S/14, FP32, Gaussian smoothing and 224 metric masks. Mask postprocessing stays inherited to avoid conflating a mask change with resolution.
- Select high-resolution memory with the same Gaussian 128-dimensional projected greedy k-center rule and seed+2000. Half banks are deterministic prefixes of the greedy order.
- Fuse scores/maps by maximum after dividing each scale by its median leave-one-reference-image-out score. Fit these scale factors on the eight reference images only; remove every selected vector originating from the held-out reference image. Calibrate the final fused score on the unchanged 40 disjoint normal calibration images.
- Keep 224-half and 448-half score-only ablations; no test-label tuning of weights, projection, budgets or thresholds.
- Target operating FPR is nominal 5% using the existing finite-sample normal quantile; actual held-out FPR may differ. Report BOTH FPR and FNR, all classes/seeds, tiny-defect performance, costs and negative outcomes.
- Test labels are only for evaluation and retrospective failure inspection. No checkpoint/model/threshold selection will be justified using test F1.
- Main goal: lower misses with a defensible false-alarm tradeoff. Equal stored-bank bytes do not imply equal feature-extraction cost; two-scale fusion invokes the backbone twice.

Original public artifacts remain intact until new evidence is verified. Any later public update must retain the existing anonymity rules.

Selective-review analysis (fixed before the new resolution outcomes): use normal-calibration quantile levels 0.80, 0.85, 0.90 and the existing nominal 0.95 hard-reject threshold. Keep the hard NG threshold unchanged. Scores between the lower review threshold and NG threshold become REVIEW; only lower scores are automatically OK. Report automatic decision coverage, review load, anomalous items sent to review, and anomalous items incorrectly auto-passed. REVIEW is not counted as a correct defect detection; no assumed human-review accuracy. Curve points are predefined and will not be selected by test F1.

Additional aggregation ablation, motivated during development on the original categories: compare the existing top-1%-patch image score with the maximum of the smoothed pixel anomaly map. Keep each parent map, memory bank and all localization metrics identical; derive a NEW image threshold from that variant's own 40 normal calibration maxima. Apply to 224 coreset and 448 coreset. This tests score dilution without new model training or test-label threshold fitting. It is an exploratory development addition, not a preregistered discovery claim; additional-category results remain a generalization check rather than a blinded holdout.
