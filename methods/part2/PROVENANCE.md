# Part 2 code provenance

The formal results in `results/part2_dino/` and `results/part2_anomalyclip/` were produced on 2026-11-06 by these files when
they lived in `part2_official/` at the project root (the `runner` field in each `run.json` still says `part2_official/...`).
On 2026-11-06 the server copies had exactly the "original sha256" below, and the scripts were last modified before the
grids started (`run_dino_formal.py` 12:27:18, grid start 12:27:51; `run_anomalyclip_formal.py` 14:25:21, grid start 14:29).

To fit the repository layout, only path lines were changed when the files moved to `methods/part2/`:

- `ROOT = Path(__file__).resolve().parents[1]` became `parents[2]`, so `ROOT` is still the repository root;
- usage strings, the provisional-manifest default folder and the `runner` label changed from `part2_official/` to `methods/part2/`;
- the test file moved to `methods/part2/tests/` and adds the parent folder to `sys.path` instead of its own folder.

No computation changed. Every difference is visible with `git diff` against the original files, and the "original sha256" lets
anyone check a copy of the original folder.

| File | Original sha256 (as run) | sha256 in this repository |
|---|---|---|
| `methods/part2/aggregate_dino_formal.py` | `171c924c0aeaa042cda6ff6027a238b993daf657c24bdc470078f7fa13c5fd64` | `21856c52c90a32e47018c4c4b84610b7f310fe82cc93766954c6b47fdfa99043` |
| `methods/part2/compare_with_upstream.py` | `607b3348649c49eead14f058b8460d62bc3819f89cf6242e3d3fc2a95bac10e5` | `1e0c8cd0abedc9a97ef47334d4d24211820485f7aa2ab68470e157288057f143` |
| `methods/part2/dino_interface.py` | `54225c7a40a809cf7d838bbc2d34658e2ab9d9c91e349b7c175cf8e38d9ae9de` | `fff9aa88d591336315819a1d5b67d2fc8c1a2feaf380e3d770f7ce8afd96a1ae` |
| `methods/part2/failure_analysis.py` | `c16bfb64bbb38c3953e147c4466e1f01e1d8bea5927ad4d82ed81d73dd201329` | `c6f8cc0a7a20f5f848cb8b85e705a41d2ea4d97bb273d88ad1610ec6a57ddb7a` |
| `methods/part2/make_temp_manifest.py` | `b14639cf44ac9b0e3011775736c2e878315b45f021e373e5758c5a5c59c91f39` | `25b72e0d1a42c52489965fed4ca666fda0b5f2f5478bb7117f10ed09c38c495a` |
| `methods/part2/run_anomalyclip_formal.py` | `b6dcb5330533654df4cbb764272cd60cbe5ce69ef956b0c53aa6a0b2d1971e21` | `5e223a791cb1fa464d9bf929452ac8038afb250a79d1b56a0d1bf4936b8aa6b8` |
| `methods/part2/run_anomalyclip_official.py` | `b362dd9a822d172774eeb783dc0c6a75ec53fc1d47f6e465c59dab72e3eab83e` | `0dcb41d36a79308d4ddabec44b80896938e9eb289177ddc00f1eb0cc94007248` |
| `methods/part2/run_dino_formal.py` | `30720149cd75a5d72caec0a80c9262cd14279f17a8441e48b76c0aaab5c3efda` | `cf2f4edb492f79ee77dd36b5eb5522e45cc26554d29af98fae3e876ab8492b6a` |
| `methods/part2/run_dino_official.py` | `26ed61cd665e7d8a262d291fdf3ad43374ee9248dbec2a58607a7451bf5bbf5e` | `06c00abc7124b33e9d5c8aaf6e28db7f76f65260bde99d7366a51f7b47a91766` |
| `methods/part2/run_upstream_reference.py` | `4950f29124c11440cc5f6d95cf77f81d44d09477d89023044784ec2113829dc2` | `043effe07ad8a14d6d8ad6c07c68a3e7667b876f11481a036b11a0983c1b4dd8` |
| `methods/part2/tests/test_part2_official.py` | `29cd4265573f84ffafd746b5aa1643f583f18f8b00291f0ff7eb8b8ec8c9a464` | `0d78a246977c5f90013b0ba17b70cd7c7b80dedb65f4b5693da96118eb2b4bc3` |

Upstream code and weights used (not stored in Git):

| Item | Version / sha256 |
|---|---|
| AnomalyDINO (github.com/dammsi/AnomalyDINO) | commit `b9d1c2648e3a5247437d4d953d907a8f3d994457` |
| DINOv2 (github.com/facebookresearch/dinov2) | commit `7764ea0f912e53c92e82eb78a2a1631e92725fc8` |
| AnomalyCLIP (github.com/zqhang/AnomalyCLIP) | commit `3911738c0867544f545a076ad78f3f11d9ecbfdf` |
| `checkpoints/dinov2_vits14_pretrain.pth` | `b938bf1bc15cd2ec0feacfe3a1bb553fe8ea9ca46a7e1d8d00217f29aef60cd9` |
| `checkpoints/ViT-L-14-336px.pt` (OpenAI CLIP) | `3035c92b350959924f9f00213499208652fc7ea050643e8b385c2dac08641f02` |
| `third_party/AnomalyCLIP/checkpoints/9_12_4_multiscale_visa/epoch_15.pth` | `415c5dcb52668b8c33fb9c1a351c686d632b919df5b384d63fa9ce7a2338ced4` |
