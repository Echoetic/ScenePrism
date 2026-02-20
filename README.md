# ScenePrism

<p align="center">
  <img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-2.6.0-EE4C2C?style=for-the-badge&logo=pytorch"/>
  <img alt="Python" src="https://img.shields.io/badge/Python-3.12-3776AB?style=for-the-badge&logo=python"/>
  <img alt="CUDA" src="https://img.shields.io/badge/CUDA-12.8-76B900?style=for-the-badge&logo=nvidia"/>
  <img alt="Accuracy" src="https://img.shields.io/badge/Test_Accuracy-85%25-brightgreen?style=for-the-badge"/>
  <img alt="Kaggle" src="https://img.shields.io/badge/Kaggle-Intel_Image_Classification-20BEFF?style=for-the-badge&logo=kaggle"/>
  <img alt="License" src="https://img.shields.io/badge/License-MIT-blue?style=for-the-badge"/>
</p>

---

## 📖 项目简介

本项目是一套完整的**场景图像分类系统**，基于 Kaggle 经典竞赛数据集 **[Intel Image Classification](https://www.kaggle.com/datasets/puneet6060/intel-image-classification)**，面向 6 类自然与城市场景（建筑 / 森林 / 冰川 / 山脉 / 海洋 / 街道），训练集规模约 17,000 张图像。最终在盲测集上达到 **85% 分类准确率**，**远超 Kaggle 公开方案主流水平（60%–78%）**，逼近该数据集公开最高成绩（88%），跻身第一梯队。

项目核心价值不仅在于最终指标，更在于其背后历经数周、**8 个版本迭代**的完整探索过程——从零构建基础 CNN 到多模型异构集成，从直面 ViT 惨败的冷静复盘到引入 EMA/SWA/TTA 等训练技巧的精细打磨，每一个百分点的提升都凝聚着大量实验与思考。

---

## 🏗 项目结构

```
Classify/
├── data/
│   ├── basic_data/                      # 训练数据（~17,000 张，6 类）
│   └── pred_data/                       # 盲测数据（100 张）
├── models/                              # 模型保存目录
│   ├── checkpoint_*.pth                 # 断点续训文件
│   ├── best_*.pth                       # 各模型最优权重
│   └── swa_*.pth                        # SWA 平均权重
├── Figures/                             # 实验可视化图表
├── cnn_81.py                            # 阶段一：自定义 CNN + SE 注意力（81%）
├── ResNet_78.py                         # 阶段二：ResNet18（78%）
├── ResNet_81.py                         # 阶段二：ResNet50（81%）
├── vit_61.py                            # 阶段三：Vision Transformer（61%，探索性实验）
├── ensemble_83.py                       # 阶段四 v1：基础三模型集成（83%）
├── enhanced_ensemble_extreme_81.py      # 阶段四 v2：极限七模型集成（81%，失败教训）
├── enh_ens_adj.py                       # 阶段四 v3：理性调整五模型集成（84%）
├── enhanced_ensemble_85.py              # 阶段四 v4：最终黄金四模型集成（85%）✓
├── lr_finder.py                         # 自研学习率自动搜索工具
├── display_*.py                         # 可视化分析脚本
└── pred_result.xlsx                     # 最终预测结果
```

---

## 🧭 模型演进历程

> 本系统的核心叙事是一段从 **81% → 78% → 61% → 83% → 81% → 84% → 85%** 的曲折探索之旅。每一次回退都不是失败，而是积累认知、校正方向的必要代价。

### 阶段一：自定义 CNN + SE 注意力机制（准确率 81%）

实验伊始，为了深刻理解卷积神经网络的工作原理，选择从零搭建一个 5 层卷积块的自定义网络（`ImprovedCNN`），而非直接调用预训练模型。网络遵循 VGG 思想，采用 3×3 小卷积核逐层提取特征，总参数量约 11M。

更重要的是，在最后一个卷积块后引入了 **SE（Squeeze-and-Excitation）注意力模块**。SE 模块通过全局平均池化获取通道级统计信息，再经双层全连接网络学习通道间依赖关系，最终对特征图进行自适应加权重标定——这让模型具备了"知道该看哪里"的能力。这是本阶段最重要的创新设计决策。

基础模型在测试集上达到 81%，完成了验证数据 pipeline 和建立性能基线的核心任务。但训练曲线清晰揭示了问题：第 30 个 epoch 后验证损失停滞，训练准确率与验证准确率之间出现明显分化，说明模型表达能力已触达天花板。

### 阶段二：ResNet 系列迁移学习（准确率 78%–81%）

转向更成熟的残差网络架构。系统性地实验了 ResNet18、ResNet34、ResNet50 三种深度，核心收获在于量化理解了**网络深度、迁移学习与任务表现**之间的关系：

- ResNet18（18层）：训练速度最快，但表达能力受限，仅达 78%
- ResNet34：性能提升有限，约 79%，边际收益递减明显
- ResNet50（50层 + Bottleneck）：训练过程明显更平稳，验证曲线震荡减小，最终达 81%

关键发现：使用 ImageNet 预训练权重进行迁移学习，相比随机初始化带来约 5 个百分点的显著提升，印证了大规模预训练特征的强泛化能力。同时为分类头设计了三层全连接结构配合梯度递减式 Dropout（0.5→0.3→0.2），相比单层线性分类器效果更优。

### 阶段三：Vision Transformer 的大胆尝试与冷静复盘（准确率 61%）

在 ResNet 性能达到局部最优后，决定挑战当时计算机视觉领域最前沿的 **Vision Transformer（ViT）** 架构。从零实现了包含 12 层 Transformer 编码器、12 头自注意力、约 86M 参数的完整 ViT，寄望于其全局建模能力带来突破性提升。

然而，测试准确率仅有 **61%**，远低于 ResNet 的 81%。训练曲线呈现出极其显著的过拟合：训练准确率可达 90% 以上，验证准确率却始终徘徊在 60% 左右。

这次"光荣的失败"带来了极为宝贵的认知升级：**ViT 的自注意力机制缺乏 CNN 的归纳偏置（局部性与平移不变性），在数据规模有限的情形下极其依赖大规模预训练。** 原论文使用 3 亿张图像预训练，而本任务仅有 17,000 张，数据量相差 5 个数量级。先进架构并非总是最优解，模型选择必须与数据规模、任务性质相匹配。

这段经历直接塑造了此后的技术路线判断——在中小规模数据场景下，经过良好优化的 CNN 是更可靠的选择。

### 阶段四：模型集成的多轮探索与最终突破（准确率 83%→81%→84%→85%）

单模型性能陷入瓶颈后，转向集成学习策略。这一阶段历经 4 个版本的迭代，是整个实验中最能体现工程判断力与精益求精精神的部分。

**v1 基础集成 `ensemble_83.py`（83%）**

首次将 ResNet50、EfficientNet-B3、DenseNet121 三个模型以预测概率均值的方式集成。三者架构互补，有效减小了单模型的偏差。准确率从 81% 提升至 83%，验证了集成策略的有效性。

**v2 极限集成 `enhanced_ensemble_extreme_81.py`（81%，重要教训）**

受到首次成功的激励，大胆设计了"极限版本"：7 个模型（ResNet50/101/152, EfficientNet-B4/B5, ConvNeXt-Small/Base）、100 个 epoch、极强数据增强、SAM 优化器。理论上这是性能最强的配置，现实却给出了当头棒喝——准确率反而**降至 81%**，不如基础版本。

深度复盘后找到了三重病因：过大的模型组合引入过多冗余，边际效用严重递减；SAM 优化器与 PyTorch AMP 混合精度存在底层兼容性冲突（`GradScaler` 无法正确处理 SAM 的两次前向传播），导致训练频繁中断；15 小时的训练时长也极大增加了不确定性风险。这个版本教会了一个重要道理：**实用性与稳定性本身就是性能的一部分。**

**v3 理性调整 `enh_ens_adj.py`（84%）**

吸取教训后进行理性瘦身：模型数量压缩至 5 个，epoch 减为 80，彻底弃用 SAM，改为更稳健的 AdamW + 余弦退火，同时下调学习率。训练过程顺畅稳定，准确率恢复并超越，达到 84%。这次"做减法"的成功，证明了简洁配置的力量。

**v4 最终方案 `enhanced_ensemble_85.py`（85%）✓**

经过多轮对比实验，确立了最终的"黄金四模型组合"：**ResNet50**（稳定基础）、**ResNet101**（深层特征）、**EfficientNet-B4**（高效架构）、**ConvNeXt-Small**（现代卷积设计）。四个模型在参数量、架构设计哲学和特征提取模式上各有侧重，互补性最强。

训练策略上集成了五重精细优化：

| 技术 | 作用 |
|---|---|
| Mixup + CutMix | 生成软标签混合样本，平滑决策边界，防止过拟合 |
| EMA（指数移动平均，decay=0.9995）| 平滑参数波动，稳定模型输出 |
| SWA（从第 40 个 epoch 启动） | 在损失平坦区取权重均值，寻找泛化更优的解 |
| 标签平滑（ε=0.1）| 防止模型过度自信，提升校准度 |
| TTA（5 种变换×4 模型集成） | 在推理阶段进一步压缩预测方差 |

最终在盲测集上以 **85.0% 的准确率**达成目标，集成权重为 ConvNeXt:ResNet101:ResNet50:EfficientNet = 0.30:0.25:0.25:0.20。

---

## 🔬 核心技术亮点

### 自研学习率搜索工具（LR Finder）

基于 Leslie Smith 的 LR Range Test 算法，独立实现了学习率自动搜索工具 `lr_finder.py`。工具通过指数递增扫描学习率空间，实时记录损失变化，利用梯度分析自动定位损失下降最陡峭的区域，并以该点对应学习率的 1/10 作为推荐值。为四个模型分别找到了差异显著的最优初始学习率（ResNet50: 1.20e-2；ResNet101: 1.48e-4；EfficientNet-B4: 3.35e-4；ConvNeXt-Small: 2.42e-4），揭示了不同架构对学习率的高度敏感性差异。

### 完善的断点续训机制

面对长周期训练（单次完整训练约 6–8 小时）的中断风险，独立设计并实现了工程级断点续训系统。checkpoint 保存了完整的训练快照：模型权重、优化器 momentum 状态、学习率调度器位置、当前 epoch、历史最优准确率、早停计数器以及 EMA shadow 参数。程序重启后可精确恢复至中断前一刻，状态无缝衔接。该功能在实验期间多次应对 cuDNN 内部错误和 GPU 驱动崩溃，累计节省数十小时的重复训练时间。

### 四级可调数据增强体系

设计了 light / medium / strong / very_strong 四档强度可切换的增强策略，通过大量对比实验确认 strong 档为本任务最优配置，包含：水平/垂直随机翻转、±20° 旋转、±15% 平移与缩放、透视变换、强色彩抖动以及随机擦除（40% 概率）。结合 Mixup 和 CutMix 后，训练难度显著提升，最终带来模型鲁棒性的实质性增益。

---

## ⚗️ 超参数调优记录

历经 16 组以上系统性对比实验，最终确定最优配置：

| 超参数 | 搜索范围 | 最优值 | 关键发现 |
|---|---|---|---|
| 学习率 | 1e-4 ~ 1e-3 | 5e-4 | 过大震荡，过小收敛慢 |
| Batch Size | 32 ~ 96 | 96 | ≥64 利用 GPU 效率最佳 |
| 数据增强 | light ~ very_strong | strong | 极强增强反而轻微欠拟合 |
| 权重衰减 | 1e-5 ~ 1e-3 | 1e-4 | 过大过小均损害泛化 |
| EMA decay | 0.999 ~ 0.9999 | 0.9995 | 核心稳定性参数 |
| SWA 启动 epoch | 30 ~ 50 | 40 | 过早启动干扰收敛 |

---

## 📊 最终性能总览

| 数据集 | 准确率 | 样本数 |
|---|---|---|
| 训练集 | 88.5% | ~13,600 张 |
| 验证集 | 86.2% | ~1,700 张 |
| 测试集 | 85.3% | ~1,700 张 |
| **盲测集（pred_data）** | **85.0% ✓** | **100 张** |

**与 Kaggle 公开方案横向对比：**

| 方案层次 | 准确率区间 | 说明 |
|---|---|---|
| Kaggle 公榜主流方案 | 60%–78% | 大多数公开 Notebook 的成绩区间 |
| **本项目最终方案** | **85.0%** | **远超主流，跻身第一梯队** |
| Kaggle 公开最高成绩 | 88% | 该数据集已知公开最优结果 |

本项目在未借助任何外部额外数据、仅使用公开预训练权重的前提下，以系统性的模型迭代和精细化训练策略，将成绩推进至公开最高水平的 3 个百分点以内，充分验证了本方案在工程实现与算法设计上的竞争力。

各类别详细指标：

| 类别 | 准确率 | 精确率 | 召回率 | F1 |
|---|---|---|---|---|
| buildings | 94.1% | 91.4% | 94.1% | 0.927 |
| forest | 93.8% | 93.8% | 93.8% | 0.938 |
| glacier | 94.1% | 94.1% | 94.1% | 0.941 |
| mountain | 76.5% | 81.3% | 76.5% | 0.788 |
| sea | 82.4% | 82.4% | 82.4% | 0.824 |
| street | 68.8% | 73.3% | 68.8% | 0.709 |
| **平均** | **85.0%** | **86.0%** | **85.0%** | **0.854** |

glacier、forest、buildings 三类具有鲜明视觉特征，表现最优（均超 93%）；street 类场景语义多样性最大，是最具挑战性的类别。

---

## 🛠 技术栈

| 类别 | 工具 |
|---|---|
| 数据集 | Kaggle Intel Image Classification（~17,000 张，6 类） |
| 深度学习框架 | PyTorch 2.6.0 + torchvision |
| 硬件加速 | NVIDIA RTX 5090 D (32GB) + CUDA 12.8 |
| 核心模型 | ResNet50/101、EfficientNet-B4、ConvNeXt-Small |
| 数据增强 | torchvision.transforms + Mixup + CutMix |
| 训练优化 | AdamW、Warmup + Cosine Annealing、AMP、EMA、SWA |
| 推理增强 | TTA（5-transform ensemble） |
| 数据处理 | numpy、pandas、scikit-learn（分层抽样）|
| 可视化 | matplotlib |
| 进度管理 | tqdm |

---

## 🚀 快速开始

### 环境配置

```bash
conda create -n scene_cls python=3.12
conda activate scene_cls

pip install torch==2.6.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
pip install pandas numpy scikit-learn pillow matplotlib tqdm openpyxl
```

### 训练模型

```bash
# 训练最终集成模型（自动检测断点并续训）
python enhanced_ensemble_85.py

# 使用学习率查找器确定最优初始学习率（可选）
python lr_finder.py --model resnet50
python lr_finder.py --model resnet101
python lr_finder.py --model efficientnet_b4
python lr_finder.py --model convnext_small
```

训练完成后，`pred_result.xlsx` 将自动生成于项目根目录。

---

## 📚 参考文献

1. He, K., et al. "Deep Residual Learning for Image Recognition." *CVPR 2016.*
2. Tan, M., & Le, Q. "EfficientNet: Rethinking Model Scaling for CNNs." *ICML 2019.*
3. Liu, Z., et al. "A ConvNet for the 2020s." *CVPR 2022.*
4. Dosovitskiy, A., et al. "An Image is Worth 16x16 Words." *ICLR 2021.*
5. Zhang, H., et al. "mixup: Beyond Empirical Risk Minimization." *ICLR 2018.*
6. Yun, S., et al. "CutMix: Regularization Strategy to Train Strong Classifiers." *ICCV 2019.*
7. Izmailov, P., et al. "Averaging Weights Leads to Wider Optima and Better Generalization." *UAI 2018.*
8. Smith, L. N. "Cyclical Learning Rates for Training Neural Networks." *WACV 2017.*

---

**实验周期**：约 3 周 &emsp;|&emsp; **代码总量**：约 5,000 行（8 个版本累计）&emsp;|&emsp; **最终准确率**：85% ✓
