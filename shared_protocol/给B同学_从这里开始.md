# B 同学的公共准备交接

当前可用版本：`visa6_sorted20_official_refs_native_v2`。

零样本兼容更新：当前源码同时支持 few-shot DINO 和 zero-shot AnomalyCLIP。数据及原有 72 组参考选择没有变化；更新时替换整个 `shared_protocol` 文件夹，包括新的 `release.json`，六类数据不用重新传输。AnomalyCLIP 的具体接法见本页第 8 节。

这份交接提供六类数据、固定参考/校准清单、读取程序、统一指标、预测保存和运行接口。B 可以开始接入 DINO；无需等待 A 完成 Template、ECC 或 AE。

## 1. 需要哪些文件

保持下面的相对目录结构：

```text
项目根目录/
  shared_protocol/             整个目录，包括 upstream、tests、manifests、release.json
  data/VisA_pytorch/1cls/      candle、capsules、cashew、chewinggum、fryum、pipe_fryum
```

`outputs/` 中的联调结果和 `.venv-shared-check` 都不用交给 B。六类数据合计 3245 张正常训练图、960 张测试图、600 份异常标注；训练图中的 120 张固定校准图已从可用训练/参考池中排除。

## 2. 在 B 的电脑上先运行

先在项目根目录运行：

```powershell
python -m shared_protocol.check_ready
python -m unittest discover -s shared_protocol/tests -v
python -m shared_protocol.example_usage --category candle --shots 1 --seed 0
```

检查会核对源码冻结版本、官方划分、清单内容、文件哈希和正常图全零标注。看到 `readiness.json` 的 `status=passed`，再进行正式模型运行。`--skip-hashes` 只会产生 `partial`，不能当完整验收。

如果缺公共工具依赖：

```powershell
python -m pip install -r shared_protocol/requirements.txt
```

B 已有 DINO 环境时，优先在原环境检查和补缺失依赖。`requirements-lock.txt` 是本次单独 CPU 公共工具环境验证过的精确版本，不要求强行替换 DINO 环境中的 NumPy/FAISS/PyTorch 组合。公共工具不安装模型权重或 CUDA/PyTorch。

本机已用全新 Python 3.13.5 环境按锁定依赖验证公共工具。其他 Python/模型环境应运行上面三个检查；不能把本次验证直接说成 B 的 CUDA 环境已经验证。

## 3. 固定清单怎样读

```python
from shared_protocol.data import load_manifest, reference_paths, calibration_paths

m = load_manifest('candle')
refs = reference_paths(m, shots=1, seed=0)
cal = calibration_paths(m)
```

- `refs`：仅用这些正常图建库；同一类别/shot/seed 的基准和改进必须复用。
- `cal`：固定 20 张正常图，仅用于计算当前配置的阈值，不进内存库、不训练模型。
- 测试清单：`m['test']`，必须全部跑完。
- 所有清单路径相对于 `VisA_pytorch/1cls`；图片文件名有缺号，不能按数字编号猜前 20 张。

参考图采用固定版本官方代码的排序分段选择。这里的 seed 是分段索引，不是 NumPy 随机抽样。共六类 × 四 shot × 三 seed = 72 组。

## 4. 接入官方 DINO 时要改的地方

官方 `src/detection.py` 默认从整个 `train/good` 自行选择参考图，会重新包含前 20 张校准图。需要把它的参考图选择分支换成读取共享清单，例如：

```python
from shared_protocol.data import load_manifest, reference_paths

# object_name、n_ref_samples、seed 和 data_root 使用当前官方运行的变量。
manifest = load_manifest(object_name)
img_ref_samples = [p.name for p in reference_paths(
    manifest, n_ref_samples, seed, data_root=data_root
)]
```

这只替换参考图文件名来源。保持官方 backbone、特征提取、旋转增强、agnostic 设置、邻居搜索和评分规则。运行入口及评估入口的 VisA 类别范围都要限制为六类，避免评估代码继续寻找其余六类。

必做配置仍为 `dinov2_vits14`、resolution 448、`agnostic`、shots 1/2/4/8、seeds 0/1/2。公共 RGB 读取默认保留原图，交给官方 `prepare_image` 按短边 448 缩放及 patch 倍数裁剪；不要提前拉成 448×448 方形。

## 5. 怎样保存预测和调用统一评价

推荐使用公共运行接口。B 先按 `refs` 建好正常内存库，然后提供一个单图预测函数：

```python
from shared_protocol.run_detector import run_detector

def predict_one(path):
    # 使用 B 的固定模型与正常内存库读取 path 并推理。
    # 输出原始图像分数，以及已对应原图尺寸的异常图。
    # DINO 异常图应使用官方 dists2map(patch_grid, original_rgb.shape)。
    # 图像分数保留官方基于 patch 距离的定义。
    return raw_image_score, native_anomaly_map

result = run_detector(
    predict_one,
    output='outputs/dino/candle/1shot_seed0',
    category='candle', shots=1, seed=0,
    method='official_dino', config_id='vits14_448_agnostic_v1',
    run_config={
        'model': 'dinov2_vits14', 'input_resolution': 448,
        'preprocess': 'agnostic',
        # 在这里继续记录实际上游版本、模型权重、运行人、计时范围等。
    },
    synchronize=torch.cuda.synchronize,
    resource_reader=lambda: {
        'device': str(device),
        'peak_gpu_allocated_mb': torch.cuda.max_memory_allocated() / 1024**2,
    },
)
```

上例中的 `predict_one` 是接入示意，不是已实现的 DINO 推理函数。B 应在自己的环境中先 warm up 并重置显存峰值计数；正常建库耗时单独记录。公共接口自动推理固定 20 张校准图和所有测试图，并按顺序保存。它不会向 `predict_one` 传入测试标签或标注。

如果 B 希望沿用自己的运行循环，可以直接使用 `PredictionWriter`，接口见 `predictions.py`：先写 20 个校准分数，再按 `m['test']` 顺序逐张 `write_test`，最后 `finish`。

文件格式是一个 `run.json` 加每张原尺寸异常图的 `.npy`。评价命令：

```powershell
python -m shared_protocol.evaluate_predictions --run outputs/dino/candle/1shot_seed0 --output outputs/dino/candle/1shot_seed0/metrics.json
```

`--required-only` 可跳过选做 AP/AUPRO。旧固定 448×448 的 NPZ 格式不属于 v2 正式结果。

## 6. 共同评价规则

- 模型输入 resolution 448；像素评价保留原始标注尺寸，标注不缩小。
- 预测图用官方流程对应到原图。保存、评价时均不做逐图 min–max 归一化。
- 正常测试图自动生成原尺寸全零标注，参与像素指标。
- 每类、每配置，20 张校准图原始分数的线性第 95 百分位为阈值；分数严格大于阈值才为 NG。
- Image/Pixel AUROC 必做；AP/AUPRO 选做；AUPRO 为官方八邻域、FPR 上限 0.30。
- 六类等权平均，三 seed 报均值和总体标准差 `ddof=0`。
- `aggregate_runs` 只接受成功且必做指标齐全的完整运行；拒绝联调专用结果。few-shot 默认要求 72 次；zero-shot 默认只要求六类各一次，二者不能混在同一张汇总中。
- 模型选择、权重、prompt、融合设置和阈值仅用训练/校准数据确定。

## 7. 已经实际验证了什么

公共代码已增加官方 AUPRO 对照和防错测试；真实 candle 的 20 张校准图和 200 张测试图完成 Template 联调，原尺寸预测又与 sklearn/官方指标交叉核对。新建环境按锁定依赖重新安装并验证。

这些证明公共基础能够连通使用。它们不代表 B 的 DINO 实验已完成，也不代表 A 的六类正式 Template 已完成。接收者仍需运行本页的环境与数据检查，并先跑通自己的一个类别。

## 8. AnomalyCLIP 使用真正的 zero-shot 入口

AnomalyCLIP 不建正常参考图内存库。不要为了适配接口填写 `shots=1`；方法名为 `anomalyclip` 时，公共工具会直接拒绝这种记录。

| 字段 | DINO few-shot | AnomalyCLIP zero-shot |
|---|---|---|
| `shots` | 1 / 2 / 4 / 8 | 0 |
| `experiment_mode` | `few_shot` | `zero_shot` |
| `reference_paths` | 共享清单中实际选取的参考图 | 空列表 `[]` |
| `reference_seed` | 0 / 1 / 2 | `null`，没有参考图选择 |
| `seed` / `run_seed` | 本次运行对应的参考索引和运行种子 | 默认 0，仅表示本次运行随机种子 |
| 默认汇总 | 六类 × 四 shot × 三 seed | 六类各一次，共六次 |

接入方式与 DINO 相同，单图预测函数使用 B 的 AnomalyCLIP 权重和推理实现：

```python
from shared_protocol.run_detector import run_detector

# predict_clip_one(path) 是 B 已实现的单图推理，返回原始图像分数、原图尺寸异常图。
result = run_detector(
    predict_clip_one,
    output='outputs/anomalyclip/candle/zero_shot_seed0',
    category='candle', shots=0, seed=0,
    method='anomalyclip', config_id='released_visa_checkpoint_v1',
    experiment_mode='zero_shot',
    run_config={
        'checkpoint': '填写实际使用的面向 VisA 评价的 checkpoint',
        'checkpoint_source_training_data': '按上游说明填写辅助训练数据',
        'input_preprocessing': '填写实际官方 CLIP 预处理配置',
        'target_reference_images': 0,
    },
    # CUDA 计时/显存的 synchronize、resource_reader 接法与第 5 节相同。
)
```

不需要调用 `reference_paths` 或读取参考图；若调用 `reference_paths(m, 0, 0)`，也会正确返回空列表。公共程序仍会推理固定的 20 张正常校准图，**只用于确定展示的 OK/NG 阈值，不训练模型、不选 prompt、不建立参考库**。报告应写清“零样本模型推理，额外 20 张正常图用于阈值校准”，不能声称完全没有使用目标数据的阈值校准。

像素评价、完整测试集、原始分数和标准标注规则与 DINO 共用。CLIP 的输入按官方方法处理，实际分辨率和专用处理写入配置；不要直接套用 DINO 的 patch 处理，也不要用测试标签选择 prompt 或模型设置。

六类完成后，读取各自的 `metrics.json`，调用：

```python
from shared_protocol.metrics import aggregate_runs
from shared_protocol.protocol import CATEGORIES

summary = aggregate_runs(six_clip_metric_records, CATEGORIES)
```

程序根据零样本记录自动选择六类、0-shot、运行 seed0 的网格，并输出六类等权平均。每类只有一次运行时，seed 标准差为 `null`，`seed_stability_measured=false`；不会把一个结果复制三次来假装测了稳定性。如果有理由研究多次运行随机性，可以显式传 `seeds=[0, 1, 2]`；这表示运行重复，仍不涉及参考图 seed。

本次新增的是公共接口及数值/文件流程测试，没有替 B 运行真实 AnomalyCLIP 模型；测试中构造的预测不是 CLIP 实验成绩。
