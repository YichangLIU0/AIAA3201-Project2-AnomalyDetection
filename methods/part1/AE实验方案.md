# AE：正常图训练扩展，不是 few-shot

这项是老师的可选 Part1 扩展，也是本组 A 的分工。它与 Template/ECC 分开保存，不修改共享协议，不把完整正常训练池写成 shots=1 或 shots=0。

固定方案：六类，各自训练 seed 0/1/2 三个独立初始化，共18组；每组使用该类 manifest 的全部 normal_training_pool，扣除20张校准图，训练12个完整 epoch。模型93,283个参数，无预训练；Adam，学习率0.001，batch8，RGB MSE，随机水平/垂直翻转。只使用最后一个 epoch，不根据测试成绩选择模型。

输入遵循已有 Template 的短边448、bicubic、右下裁成14倍数；网络内部临时反射补齐到16倍数，重建后裁回输入尺寸。误差是 RGB 绝对差均值、Gaussian sigma2，图像分数是最大的1%输入网格误差均值。原尺寸 float32 热图无损 NPZ 保存，不缩小 GT，不逐图归一化。校准图只用于20张分数的95%分位阈值，不参与训练或早停。

AE的训练数据量明显多于 K-shot 模板和 DINO，所以结果仅作为正常图训练扩展，不声称是相同训练预算下的优胜方法。三个 seed 表示权重初始化、增强和训练顺序，不是参考图选择。

训练与推理使用 RTX4060 Laptop GPU；所有 CUDA 操作计时前后同步。训练、推理 GPU 峰值分别记录。推理计时包括读图、预处理、网络前向、误差、平滑、分数和恢复原尺寸；不包括压缩、存储和指标计算。Template/ECC 使用 CPU，两者耗时不能当成同设备速度胜负。

在正式仓库根目录运行（GPU环境）：

```powershell
.venv-a-gpu/Scripts/python.exe -m methods.part1.run_ae --data-root ../data/VisA_pytorch/1cls
```

默认18组可恢复已完整保存的模型和预测，不覆盖热图。算法、源文件、依赖版本与配置一起冻结；修改它们后请换输出和结果根目录，避免混入一次实验。数据核查在 `results/part1_ae/data_check`。`methods.normal_training` 单独记录 normal_training 模式，并调用未修改的共享 image/pixel 评价函数。

```powershell
.venv-a-gpu/Scripts/python.exe -m methods.part1.analyze_ae --data-root ../data/VisA_pytorch/1cls
.venv-a-gpu/Scripts/python.exe -m methods.part3.run_selective --source results/part1_ae --output results/part3_f_ae
.venv-a-gpu/Scripts/python.exe -m methods.part3.analyze_selective
```

依赖见 `requirements-a-gpu.txt`。本次工作区虚拟环境复用已有 PyTorch2.5.1+cu121，不改电脑原有环境；其他依赖安装在仓库内、被 Git 忽略的 `.venv-a-gpu`。

其他电脑可新建 Python3.10 环境；以下 CUDA121 安装源对应 [PyTorch官方历史版本说明](https://pytorch.org/get-started/previous-versions/#v251)。AE本身不需要torchvision或FAISS；DINO的环境依赖由B的接口另行明确。

```powershell
py -3.10 -m venv .venv-a-gpu
.venv-a-gpu/Scripts/python.exe -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121
.venv-a-gpu/Scripts/python.exe -m pip install -r methods/part1/requirements-a-gpu.txt
```

训练完成后的规范顺序是：等所有训练进程退出，先串行 `benchmark_ae`，再 `analyze_ae`。前者在全部2880张测试预测上核对保存分数并测时，后者诊断训练退化、复制小模型、审计与生成报告。正常训练MSE差或黑色重建的seed也保留；不是只重跑/挑选好的seed。
