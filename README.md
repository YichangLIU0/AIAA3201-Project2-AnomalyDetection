# AIAA3201 Project 2 — Few-Shot Industrial Visual Anomaly Detection

当前主分支是按课程要求重新建立的实验起点：提供六类 VisA 数据准备、固定清单和公共评估工具。**这里尚未完成新的 DINO、AnomalyCLIP 和全部 Part 1/3 正式实验。**

旧版代码、旧报告、Demo 和实验结果保存在 [legacy-v1 分支](https://github.com/YichangLIU0/AIAA3201-Project2-AnomalyDetection/tree/legacy-v1)，不与新协议结果混用。旧分支固定起点为 `568f8c588c44880c05b6434c79ba2dc66650f144`。

## 当前包含什么

- `shared_protocol/`：数据准备、读取、预测保存、统一评价与运行接口。
- `shared_protocol/manifests/`：六类完整清单；每类 20 张校准图，以及四 shot、三个 seed 的 72 组参考选择。
- `shared_protocol/tests/`：26 项公共工具测试，包含零样本完整接口测试。
- `shared_protocol/upstream/`：固定版本的外部实现参考、来源与许可证。
- `shared_protocol/reports/`：数据和工具验证记录；不代表新模型正式实验成绩。

交接入口：[给 B 同学的接入说明](shared_protocol/给B同学_从这里开始.md)。当前数据协议为 `visa6_sorted20_official_refs_native_v2`；该版本同时支持 few-shot 和 zero-shot 实验记录。

## 安装公共工具

以下是在 Windows 上新建独立公共工具环境的示例。本次验证使用 Python 3.13.5；此环境不安装 DINO/CLIP 权重或 CUDA/PyTorch。

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r shared_protocol/requirements-lock.txt
.\.venv\Scripts\python.exe -m unittest discover -s shared_protocol/tests -v
```

如果已经安装了 DINO 环境，先阅读接入说明，再按需补充 `shared_protocol/requirements.txt` 中缺失的包，避免强行替换既有 PyTorch/FAISS/NumPy 组合。

## 数据准备

六类为 `candle`、`capsules`、`cashew`、`chewinggum`、`fryum`、`pipe_fryum`。图片数据不保存在 Git 中。

```powershell
.\.venv\Scripts\python.exe -m shared_protocol.download_visa
.\.venv\Scripts\python.exe -m shared_protocol.prepare
.\.venv\Scripts\python.exe -m shared_protocol.check_ready
```

准备好的目录为 `data/VisA_pytorch/1cls/<category>/...`。已收到相同六类数据时，直接放到该位置并运行检查；数据在其他位置时给检查程序传入 `--data-root`。

正常测试图使用全零标注并参加像素评价。每类正常训练文件按文件名排序，前 20 张仅用于阈值校准，不进入参考库或模型训练。测试集保留全部 960 张官方测试图。

## 实验接口与评价规则

- DINO / Template few-shot：1/2/4/8 张参考图、参考 seed 0/1/2；相同配置复用共享清单。
- AnomalyCLIP zero-shot：`shots=0`、`reference_paths=[]`、`reference_seed=null`；默认六类各一次，不能伪装成 one-shot。
- 额外的 20 张正常图仅校准 OK/NG 阈值，不能将其作为零样本模型的训练或参考图。
- Image AUROC 使用原始图像分数；Pixel AUROC 使用原始标注尺寸，标注不缩小，预测分数不做逐图 min–max 归一化。
- AUPRO 为选做指标，使用官方八邻域和 0.30 FPR 上限。
- 六类等权平均；few-shot 按三 seed 报告均值/标准差；zero-shot 单次运行不伪报 seed 稳定性。

接入、原尺寸预测保存、CUDA 计时说明和零样本示例见 [交接说明](shared_protocol/给B同学_从这里开始.md)。

## 已验证与尚未完成

公共工具在原环境和独立安装环境中各通过 26 项测试；真实 candle 的 20 张校准图与完整 200 张测试图完成过 Template 联调，原尺寸指标与可信实现进行了交叉核对。

这些是工具验证，**不是完整的正式模型复现或提交结果**。AnomalyDINO / AnomalyCLIP 的真实模型实验、A 的六类正式 Template、改进、消融、资源及困难案例分析仍按分工推进。联调结果带有 `integration_only=true`，不能进入正式结果汇总。

## 外部来源与历史

数据来自 [官方 VisA / SPot-Diff](https://github.com/amazon-science/spot-diff)；参考选择及指标对照使用固定版本 [AnomalyDINO](https://github.com/dammsi/AnomalyDINO)。具体修订、来源哈希和许可证见 `shared_protocol/upstream/`。AnomalyCLIP 的权重和辅助训练数据由运行者按上游说明披露。

旧代码可从 `legacy-v1` 分支检查、移植。旧标准实验使用 224 输入和 40 张随机正常校准图，不能直接作为当前要求的完整复现。主分支清理保留 Git 历史，不覆盖旧分支。

AI assistance was used in implementation, validation, and documentation. Project members must inspect the work and follow course disclosure requirements.
