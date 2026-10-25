# Sources and attribution

This repository implements the AIAA3201 Project 2 coursework. The course instruction PDF is not redistributed.

Implementation sources:
- AnomalyDINO official repository https://github.com/dammsi/AnomalyDINO : pretrained backbone wrapper, rotation augmentation, PCA query masking, normalized nearest-neighbor scoring and smoothing conventions. Local cache/seed/data/metric code is an adaptation, not an unmodified official benchmark run.
- DINOv2 https://github.com/facebookresearch/dinov2 : pretrained ViT-S/14.
- VisA https://github.com/amazon-science/spot-diff : official split CSV, preparation script, images and masks (CC BY 4.0).
- AnomalyCLIP https://github.com/zqhang/AnomalyCLIP : released model and target-VisA prompt checkpoint; upstream README states VisA evaluation uses auxiliary training on MVTec AD. Any quantitative comparison must use actual local outputs.
- CVPR author kit https://github.com/cvpr-org/author-kit : official LaTeX style in camera-ready mode.

Assigned reading list is covered by reports/references.bib. Related-work prose is paraphrased from primary papers or official pages, not copied abstracts. Literature results and our local experiments must remain separate.

Our Part 3 is a course exploration of memory/latency tradeoffs, not a claim that spatial pooling or random subsampling is a new research method. Neither publication acceptance nor general superiority is asserted.

AI assistance: code construction, environment diagnosis, experiment orchestration, result visualization, literature summary, report drafting and packaging were assisted by Codex. Authors should review the code, results and prose before submission/publication.

## Coreset extension
`coursework/coreset.py` is a local implementation of projected greedy k-center selection inspired by PatchCore, not a reproduction of its complete detector. Source: Roth et al., CVPR 2022, https://openaccess.thecvf.com/content/CVPR2022/papers/Roth_Towards_Total_Recall_in_Industrial_Anomaly_Detection_CVPR_2022_paper.pdf . Our stated differences include a fixed dense Gaussian 128-dimensional projection and a first center furthest from the projected mean. Original DINO features and detection/calibration protocol are retained. No novelty is claimed for selection.
