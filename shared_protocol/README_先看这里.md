# 公共准备使用说明（已修正版）

当前版本：`visa6_sorted20_official_refs_native_v2`。

已补充零样本兼容：AnomalyCLIP 可使用 `shots=0`、空参考图、`reference_seed=null`，默认按六类各一次汇总。现有数据、校准清单和 DINO 少样本流程保持兼容。接法见交接说明第 8 节。

给 B 的完整接入说明请看 [给B同学_从这里开始.md](给B同学_从这里开始.md)。它是当前交接入口，包含文件范围、依赖安装、验收命令、官方 DINO 参考图选择接法，以及保存和评价原尺寸预测的示例。

本次修复保留原六类数据和全部参考/校准选择，修改了像素评价及验证规则：

- 像素评价使用原始标注尺寸，标注不缩小。
- AUPRO 与官方一样使用八邻域，FPR 上限为 0.30。
- 清单读取同时检查官方 CSV、冻结哈希、JSON/TXT 一致性，以及校准与参考图隔离。
- 必做 AUROC 缺失、运行失败、预测不完整或联调专用结果不能进入正式汇总。
- 提供 `PredictionWriter` 和 `run_detector`，可保存、评价不同原尺寸的逐图预测。
- `release.json` 记录代码、文档和清单冻结版本；`requirements-lock.txt` 记录独立公共工具验证环境。

## 快速检查

在项目根目录运行：

```powershell
python -m shared_protocol.check_ready
python -m unittest discover -s shared_protocol/tests -v
python -m shared_protocol.example_usage --category candle --shots 1 --seed 0
```

换电脑或数据位置时可给检查/运行接口传 `--data-root` 或 `data_root=...`，指向六类目录的上一级 `VisA_pytorch/1cls`。

## 数据与清单

| 类别 | 正常训练总数 | 固定校准 | 可用正常训练池 | 正常测试 | 异常测试 |
|---|---:|---:|---:|---:|---:|
| candle | 900 | 20 | 880 | 100 | 100 |
| capsules | 542 | 20 | 522 | 60 | 100 |
| cashew | 450 | 20 | 430 | 50 | 100 |
| chewinggum | 453 | 20 | 433 | 50 | 100 |
| fryum | 450 | 20 | 430 | 50 | 100 |
| pipe_fryum | 450 | 20 | 430 | 50 | 100 |
| 合计 | 3245 | 120 | 3125 | 360 | 600 |

六类训练/测试划分来自官方 1cls CSV；保留全部 960 张测试图。正常测试图由程序生成全零标注。校准取每类正常训练文件排序后的前 20 张；参考图从排除这 20 张后的训练池按官方分段方式选择。六类、四 shot、三 seed 共 72 份参考清单。

清单位于 `manifests/calibration/`、`manifests/references/`，完整 JSON 含原尺寸和内容哈希。`reports/data_preview.png` 是图片和标准标注的预览，不是模型热图。

## 重建与复查

```powershell
python -m shared_protocol.download_visa
python -m shared_protocol.prepare
```

已有数据仅重新审计：

```powershell
python -m shared_protocol.prepare --audit-only
```

这会再次解码、核查数据并验证冻结清单；正常使用不需要重复下载。读取程序不会自动接受被改动的清单或重新冻结源码。维护者只有在完成修改和验证后，才运行 `python -m shared_protocol.release --freeze` 生成新发布快照。

## 验证与范围

修复后增加了官方 AUPRO、原尺寸变长图片、清单误改、缺失指标、完整运行接口等测试。candle 的 20 张校准图和完整 200 张测试图已运行真实 Template 联调，并在实际原尺寸预测上交叉核对 sklearn/官方像素指标。安装也在全新独立环境中验证。

完整进度与证据见 [当前完成情况.md](当前完成情况.md)。`reports/audit_20261105/复核汇报.md` 保留的是修复前检查记录，不是当前仍未修复的问题清单。

本次交接不包含 B 的 DINO/CLIP 实现和正式实验。Template 联调被标记为 `integration_only=true`，不能充当 A 的六类正式 Part 1 结果。
