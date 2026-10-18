# video_data_verify

[English](README.md) | 中文

通过 finetune 来检验**自己生成的视频 3D 检测/跟踪数据是否正确**。这是 [data_verify](../../data_verify/data_verify) 的视频版本。

做法：以 SpatialEncoder v2 Small（lite）发布的 checkpoint 为起点，把标准数据（从 SpatialEncoder 训练集里均匀抽样的 10k 帧）和你的数据按一定比例混合后 finetune，然后看你的测试集有没有变好、标准测试集（CA-1M test）有没有退化。模型代码来自 [lite_model](https://github.com/qic999/spatial_encoder_v2_small)，本仓库只放自己的代码，不改它的源码。

```
1. 安装环境          ./setup_env.sh
2. 下载标准数据集     scripts/download_data.sh --accept-license
3. 准备自己的数据集   按 vdv/README_zh.md 的格式放进一个目录，用 scripts/validate.sh 校验
4. 启动训练          scripts/train.sh --custom <数据集目录> --ratio <混合比例>
5. 查看输出          --out 目录：results/summary.md、metrics/（loss、每轮验证）
```

## 模型和评估方式

lite 模型是一个**用 GT 2D 框做 prompt 的视频 3D 框估计和跟踪模型**：在 clip 第一帧给每个物体一个 GT 2D 框，模型在之后每一帧输出它的 3D 框（同时预测深度图和相机位姿）。它没有类别识别 / 文字 prompt，所以没有 AP，也没有 data_verify 里的 box / text 两种评估模式。

评估：测试集的每段视频切成不重叠的 8 帧 clip，每个 clip 用第一帧里最多 10 个物体的 GT 2D 框做 prompt，对每个可见的（物体, 帧）计算预测 3D 框和 GT 的 3D IoU。报告：

| 指标 | 含义 |
|---|---|
| `miou` | 平均 3D IoU（%） |
| `acc25` / `acc50` / `acc10` | 3D IoU ≥ 0.25 / 0.5 / 0.1 的比例（%） |
| `trans_err` / `depth_err` | 3D 框中心误差 / 深度误差（米） |
| `yaw_err` | 绕竖直轴的朝向误差（度），按长方体底面的对称性折叠到 0–45°（转 90° 再交换长宽是同一个框）；`yaw_err_raw` 是 lite_model 原始的未折叠值，取决于各数据集水平两轴的先后顺序（PCA 拟合的框即使完全正确也约 90°）；`scale_err` = 1 − 尺寸 IoU |
| `n` | 参与统计的（物体, 帧）数 |

都同时给出整体值和**每个类别**的值。标准测试集是 CA-1M test（2 个 CA-1M val capture），里面大多数框的类别是 `object`，所以它主要用来看整体有没有退化。目标类别是否变好看你自己的测试集。

---

## 1. 安装环境

```bash
mkdir -p <workspace> && cd <workspace>     # 工作区：lite_model、环境、权重、数据和输出都放在这里
git clone <本仓库地址> repo
hf auth login                              # 权重在 HuggingFace 的 gated 仓库 SpatialCoding/Data_verification
cd repo && ./setup_env.sh                  # 日志写到 <workspace>/envs/setup_env.log
```

这一步会：
- 在本仓库旁边克隆 lite_model；
- 用系统的 Python 3.10 建 venv `../envs/lite`，按 lite_model README 安装 torch 2.10（cu128）、Depth-Anything-3 等；
- 下载发布的 lite checkpoint 到 `../ckpts/se_small_lite_wds518_e100/checkpoint.pt`（2.4 GB，里面已经有 SAM3 tracker 的全部权重，不需要 `checkpoint_64.pt`）。

硬件：Linux + NVIDIA GPU。每张卡占用约 4 GB 显存，batch 为每卡 1 个 clip。

## 2. 下载标准数据集

```bash
scripts/download_data.sh --accept-license               # 默认下载到 ../data
scripts/download_data.sh /path/to/data --accept-license # 也可以指定目录（之后 export VDV_DATA=/path/to/data）
```

标准数据就是 lite 模型自己的训练集（HF `qicq1c/spatialencoder_full`，CA-1M + ADT + Hypersim 的 train 部分，约 217 万帧），从中**按帧均匀抽样 10k 帧**：每次按帧数比例选一个序列，在里面随机取一段不重叠的连续 100 帧（模型按视频片段训练），一共 100 段。所以各来源的比例和全集一致。抽样用固定种子，结果存在 `data/standard_10k_plan.json` 里，每次构建得到的数据完全相同。定义在 `data/standard_10k.json`。

CA-1M 的许可证是 CC-BY-NC-ND 4.0（仅限非商业用途），`--accept-license` 表示接受。已经完成的步骤会跳过，中断后重跑会接着下载。要下载约 90 GB 的 tar（ADT 每个序列一个 6.7 GB 的包），但只保留抽中的帧，tar 用完即删，完成后约 12 GB：

| 内容 | 路径（相对 DATA_DIR） | 用途 |
|---|---|---|
| 均匀抽样的 9,916 帧：CA-1M 9,216 帧（93 个 capture；其中一个 capture 的实际帧数比索引少，它的窗口只有 16 帧）、ADT 500 帧（5 个序列）、Hypersim 200 帧（2 个相机序列） | `standard_10k/{ca1m,adt,hypersim}/` | 标准训练数据，按帧数混合 |
| CA-1M val 的 2 个 capture（2,131 帧，270 个测试 clip，见 `data/ca1m_test.txt`） | `ca1m/raw/`，转换后 `ca1m/vdv_test/` | 遗忘检查 |
| 另外 3 个 CA-1M train capture 里的 `cushion` 框（训练 361 帧，测试 349 帧 / 45 个 clip） | `ca1m/raw_toy/`，转换后 `custom/toy/` | toy 自定义数据集：格式示例、冒烟测试 |

## 3. 准备自己的数据集

数据格式见 **[vdv/README_zh.md](vdv/README_zh.md)**：

```
/path/to/mydata/
├── index.json        [[场景 JSON, 帧数, "train"|"test"], ...]
├── scenes/*.json     每段视频的帧（图片、深度、内参、相机位姿）和物体（每帧的 3D 框、2D 框，跨帧同一个跟踪 id）
└── ...               图片、深度文件
```

先校验数据。所有 ERROR 都要改掉，然后看渲染图（连续几帧上的 3D 框）是否贴合物体：

```bash
scripts/validate.sh --custom /path/to/mydata    # 渲染图在 ../outputs/video_data_verify/vis/<数据集>/<split>/
```

除了格式，它还检查：投影后的 3D 框和 `camera_box_2d_proj` 是否一致（坐标轴、单位、内参）、框的局部 +Y 是否朝下（重力方向）、深度单位和深度与框中心是否一致、静止物体在世界坐标下是否漂移（位姿方向、单位）、有多少物体能通过 lite_model 的物体过滤。

再做 overfit 检查：只用你的训练集训练，并在训练集上评估。mIoU 应该明显上升；如果一直不动，说明标注之间不一致：

```bash
scripts/train.sh --custom /path/to/mydata --overfit
```

## 4. 启动训练

```bash
scripts/train.sh --custom /path/to/mydata --ratio 0.2 --out ../outputs/video_data_verify/mydata_r0.2
```

`--ratio` 是训练 clip 中来自你的数据的比例，其余来自 CA-1M。

| `--ratio` | 含义 |
|---|---|
| `0` | 只用标准数据，作为对照组（A1） |
| `0.2` | 80% 标准数据 + 20% 你的数据（A2，默认值） |
| `1` | 只用你的数据 |

不同 ratio 的 run 之间，epoch 数、每个 epoch 的 clip 数、学习率调度和 seed 都相同，所以结果的差别只来自数据。

**常用参数**（完整列表见 `scripts/train.sh -h`）：

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--gpus N` | 8 | GPU 数量（用 `CUDA_VISIBLE_DEVICES` 选卡） |
| `--epochs N` | 5 | 学习率在全部 epoch 上做 cosine 衰减到 1/100 |
| `--samples N` | 1600 | 每个 epoch 的训练 clip 总数。8 卡时为 200 步/epoch |
| `--lr X` | 3e-5 | head 的学习率（上游训练用 1e-4，在这里会让已收敛的 checkpoint 退化）；SAM3 backbone 和 spatial encoder 沿用上游的 1e-7 |
| `--train-split` / `--test-split` | `train` / `test` | `index.json` 里的 split 名 |
| `--depth-scale S` | 1000 | 16-bit PNG depth 每米对应的数值 |
| `--val-every N` | 1 | 每 N 个 epoch 验证一次；最后一个 epoch 之后一定验证 |
| `key=value` | | 其他 lite_model 配置，例如 `scratch.frame_num_range=[2,4]` |

每次验证都在标准测试集（CA-1M test）和你的测试集上进行。**最后一次验证用的就是最终权重，它的结果就是这次 run 的结果**（`results/summary.md`），不需要再单独评估。用同一个 `--out` 重跑会从断点继续训练。

**一次跑完整组对比**：zero-shot 基线，加上每个 ratio 的训练，最后汇总成一张表：

```bash
nohup scripts/compare.sh --custom /path/to/mydata --ratios "0 0.2" > compare.log 2>&1 &
```

**单独评估某个 checkpoint**：

```bash
scripts/eval.sh --ckpt released --custom /path/to/mydata                       # zero-shot（A0）
scripts/eval.sh --ckpt <run>/checkpoints/checkpoint.pt --custom /path/to/mydata
scripts/eval.sh --ckpt <run>/checkpoints/checkpoint.pt               # 只评估标准测试集（--no-standard 则只评估你的测试集）
```

## 5. 输出

```
<out>/
├── checkpoints/checkpoint.pt   最终权重（2.4 GB，也是断点续训的位置）
├── metrics/
│   ├── train.csv               每 10 步一行：总 loss、各项 loss 和学习率
│   ├── val.csv                 每次验证 × 每个测试集一行：miou、acc10/25/50、各项误差、n
│   └── val_per_category.csv    同上，按类别
├── results/
│   ├── summary.md              最终结果表（整体 + 每个类别）
│   └── summary.json
├── logs/                       train.log / eval.log（完整输出）、log.txt、train_stats.json、val_stats.json
├── tensorboard/                tensorboard --logdir <out>
└── config.yaml                 本次运行完整的 lite_model 配置
```

`compare.sh` 的输出目录里每个 run 一个子目录，另外有一个汇总表 `compare.md`：

```
| run      | custom miou | custom acc25 | custom acc50 | ca1m miou | ca1m acc25 | ca1m acc50 |
| zeroshot | ...                                                                 |   ← A0
| ratio0   | ...                                                                 |   ← A1（对照组）
| ratio0.2 | ...                                                                 |   ← A2
```

后面还有一张你的测试集上按类别的 mIoU 表。

**判断标准**：如果你的测试集上 ratio0.2 明显高于 ratio0，同时 CA-1M 测试集上两者持平，说明数据是对的，模型能从中学到东西。如果 ratio0.2 在你的测试集上反而更差，或者 CA-1M 明显下降，优先检查 `validate.sh` 报的 WARN（位姿方向、单位、2D/3D 框不一致）。

想重新导出指标或者对比任意几个 run：

```bash
source scripts/env.sh
python $VDV/tools/export_metrics.py run <out>
python $VDV/tools/export_metrics.py compare <run1> <run2> ... --to table.md
```

---

## 已验证的结果（发布的 lite checkpoint，8×H100）

**zero-shot**（`scripts/eval.sh --ckpt released --custom data/custom/toy`，约 1 分钟）：

| 测试集 | miou | acc10 | acc25 | acc50 | trans_err (m) | yaw_err (°) | n |
|---|---|---|---|---|---|---|---|
| CA-1M test（270 个 clip） | 14.7 | 45.4 | 26.3 | 4.7 | 0.489 | 31.5 | 4727 |
| toy test（cushion，45 个 clip） | 17.8 | 55.6 | 34.9 | 4.9 | 0.342 | 17.3 | 430 |

**toy overfit**（`scripts/train.sh --custom data/custom/toy --overfit`，10 epoch × 800 clip，约 20 分钟）：在 toy 训练集上 mIoU 24.6 → 46.2，acc25 46 → 81，acc50 12.5 → 48，基本单调上升。说明在一致的标注上，模型能学到东西，指标也能反映出来。

**学习率**：用上游的 1e-4 做短冒烟测试（`--epochs 2 --samples 320`），CA-1M 从 14.7 掉到 13.5；改成 3e-5 后，完整预算（5 epoch × 1600 clip，标准数据为只含 CA-1M 的早期版本）下 CA-1M 不再退化，所以默认值是 3e-5：

| run（lr 3e-5） | toy miou | toy acc25 | toy acc50 | CA-1M miou | CA-1M acc25 | CA-1M acc50 |
|---|---|---|---|---|---|---|
| zero-shot | 17.8 | 34.9 | 4.9 | 14.7 | 26.3 | 4.7 |
| ratio0 | 21.5 | 42.1 | 11.6 | 14.7 | 26.7 | 5.6 |
| ratio0.2 | 19.2 | 35.6 | 8.1 | 14.4 | 26.3 | 5.2 |

toy 上 ratio0.2 低于 ratio0，并不说明 toy 的标注有问题（它就是 CA-1M 官方标注）：toy 训练集只有一个 capture 的 361 帧，ratio0.2 在 5 个 epoch 里从中采了约 1600 个 clip，等于在过拟合一个场景；而 CA-1M 训练集里本来就有多个场景的 cushion。toy 测试集只有 45 个 clip，相邻 epoch 之间也会波动 1 个点左右。所以 toy 只用于冒烟测试。正式实验的自定义数据应该包含多个场景，类别最好是标准数据里没有、或 zero-shot 很差的。

---

## 目录结构

```
<workspace>/               工作区
├── lite_model/           上游仓库，未改动
├── envs/lite/            环境
├── ckpts/                发布的 lite checkpoint
├── data/                 标准数据和 toy 数据（默认的 VDV_DATA）
├── outputs/video_data_verify/  默认输出目录
└── repo/                 ← 本仓库
    ├── setup_env.sh
    ├── data/standard_10k.json       标准数据：训练来源和权重、测试集（VDV_STD 默认指向它）
    ├── data/standard_10k_plan.json  均匀抽样的 100 个窗口（来源、序列、起始帧）
    ├── data/ca1m_test.txt           测试集的 2 个 CA-1M val capture（ca1m_10k.txt 是早期只用 CA-1M 的标准集）
    ├── scripts/
    │   ├── env.sh            被其他脚本 source：设置 PATH、PYTHONPATH 和默认路径
    │   └── download_data.sh、validate.sh、train.sh、eval.sh、compare.sh
    ├── vdv/
    │   ├── README.md         数据格式
    │   ├── data.py           数据集（基于 lite_model 的通用视频 loader）和混合采样
    │   ├── config.py         在 lite_model 配置上替换数据、验证、指标、预算
    │   ├── trainer.py        支持多个验证集的 Trainer
    │   ├── metrics.py        3D IoU 指标（整体 + 每类）
    │   └── run.py            训练 / 评估入口（torchrun）
    └── tools/                标准集抽样（sample_standard.py）、CA-1M / Infinigen 转换、toy 数据、数据校验、dataloader 检查、指标导出
```

**运行机制**：脚本 source `scripts/env.sh` 后用 `torchrun` 启动 `python -m vdv.run`。`vdv/config.py` 读取 lite_model 的 `configs/depth/train_data_scaling_lite.yaml`（模型、loss、优化器和发布权重训练时的 HF/WDS lite 配置完全一致），只替换训练数据、验证集、指标、epoch 预算、学习率、初始权重和输出目录，然后交给 lite_model 自己的 `run_training`。

## 和上游的差异与注意事项

- **标准数据走通用 loader**：CA-1M 被转换成和自定义数据相同的格式，用 lite_model 的 `ObjectDetectionDataset` 读取（数据集 id 仍是 CA-1M 的 0），而不是需要 pickle + 原始 tar 的 `CA1MTrainIterableDataset`。两者的区别：CA-1M 原生 loader 按内参把 256×192 的深度重采样到 RGB，通用 loader 直接 resize；CA-1M 的深度内参正好是 RGB 的 1/4，两者相差不到 1.5 个 RGB 像素。原生 loader 带的 `T_gravity` 没有被 loss 或指标使用。ADT 和 Hypersim 在 HF 上本来就是通用格式的场景 JSON，只截取抽中的帧，分别以数据源名 `ADT` 和 `hyperism` 读取，lite_model 会对它们做自己的坐标轴和位姿转换。
- **评估协议**：沿用 lite_model 的 CA-1M 验证设置（8 帧、帧间隔 1、前 10 个物体），但 clip 不重叠（步长 8，CA-1M 测试集 270 个 clip），并且 padding 出来的样本不计分，所以每个 clip 只算一次。
- **关闭的上游评估**：ScanNet val88、WildDet3D 10 数据集、HF source probe、CA-1M 1000 cases 都依赖集群上的数据，这里不用；W&B 默认关闭。
- **学习率**：和上游一样做 cosine 衰减到 1/100，但 head 的起始学习率是 3e-5 而不是上游的 1e-4（backbone 和 spatial encoder 仍是 1e-7）；总 epoch 数缩短，没有 warmup。
- **`use_fa3=false`**：这台机器没有 FlashAttention 3。
- **磁盘**：每个 run 保存一份 2.4 GB 的 checkpoint（只保留最新的）。
