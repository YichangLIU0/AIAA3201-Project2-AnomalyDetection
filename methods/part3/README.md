# A 的 Part3：当前可独立完成的部分

## F：选择性预测

程序保持原检测器分数和热图不变，用20张正常校准图构造阈值与复核区间。主规则是对称区间；消融仅复核阈值下侧、区间宽度相同；另有无复核和同覆盖率的随机复核对照。主参数在查看本次 F 测试结果前固定，不根据测试风险调宽度。

输出完整 risk–coverage 曲线、AURC、各类/shot 三 seed 统计、原阈值误报/漏报、自动漏检、复核异常和正常数量。测试 ROC 的固定 FPR≤5% FNR 仅用于评价；它的测试标签阈值不会被写入部署策略。REVIEW 从不算正确检出，自动漏检分母使用全部异常图；另报仅在自动接收异常图里的漏检率。

```powershell
.venv-a-gpu/Scripts/python.exe -m methods.part3.run_selective --source results/part1_template --output results/part3_f_template
.venv-a-gpu/Scripts/python.exe -m methods.part3.run_selective --source results/part1_ecc_full --output results/part3_f_ecc
.venv-a-gpu/Scripts/python.exe -m methods.part3.run_selective --source results/part1_ae --output results/part3_f_ae
.venv-a-gpu/Scripts/python.exe -m methods.part3.analyze_selective
```

这三种方法的真实实验不等于原分工要求的 DINO–DINO+REVIEW 对照。B 提交兼容共享协议的 `prediction_manifest.json` 后，用同样入口读取其结果即可；若格式不同，应先明确转换，不能凭空补分数或虚构72组。

## C：仅准备完成，正式实验未完成

`synthetic.py` 提供 CutPaste-inspired、程序纹理两种可复现增强、逐像素真实变化掩码、patch 覆盖率软标签和384→128→1轻量评分头。`frozen_features` 关闭 backbone 梯度；测试验证一次优化后只有 head 权重更新。测试用 Conv2d 固件是梯度隔离检查，不是 DINO 实验。

人工看过训练图预览后发现，全图随机贴块常落在背景上。因此默认根据图像边缘正常背景的 Lab 色度距离粗估产品区域，暗背景另使用正亮度差，只改动估计区域内的像素；`placement='canvas'` 保留为后续消融。这个启发式不读取测试 GT，不是 B 的 DINO 背景抑制方法，也不保证准确分割产品。估计失败会报错，不偷偷改用测试掩码。

```powershell
.venv-a-gpu/Scripts/python.exe -m methods.part3.prepare_synthetic --data-root ../data/VisA_pytorch/1cls
.venv-a-gpu/Scripts/python.exe -m unittest methods.part3.test_synthetic methods.part3.test_selective -v
```

`results/part3_c_preparation` 只展示每类正常训练池首图上的两种合成结果和掩码，不含测试图，不说明真实缺陷迁移有效。借鉴来源：[CutPaste, CVPR2021](https://openaccess.thecvf.com/content/CVPR2021/html/Li_CutPaste_Self-Supervised_Learning_for_Anomaly_Detection_and_Localization_CVPR_2021_paper.html)。这里不声称复现该论文的完整训练与检测方法。

不能直接用旧仓库224方形、40张校准图、全12类的输出代替现在的六类448协议。需要 B 的真正冻结 DINO 特征提取接口后，才能在合成图上提特征、训练 head，并完成与未改 DINO 的同参考图对照、CutPaste/texture 消融及真实测试集迁移。

正式 C 实验方案须在拿到接口后、查看 C 测试结果前冻结：优先在每个 K-shot 的相同参考集合上生成合成图，避免把额外训练数据带来的收益伪装成仅改 head；如另外使用全正常训练池，必须作为训练预算不同的独立实验报告。
