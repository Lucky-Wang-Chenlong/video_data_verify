# 视频数据集格式

[English](README.md) | 中文

自己的数据和标准数据（CA-1M 10k）用同一种格式，由 lite_model 的通用视频 loader（`ObjectDetectionDataset`）读取。一个数据集一个目录：

```
/path/to/mydata/
├── index.json          场景列表和 split
├── scenes/*.json       每个场景（一段视频）一个文件
└── ...                 图片、深度等文件，路径相对于数据集根目录，结构随意
```

写完后用 `scripts/validate.sh --custom /path/to/mydata` 校验，所有 ERROR 都要改掉，再看渲染图里的 3D 框是否贴合物体。参考实现见 `tools/prepare_ca1m.py`（CA-1M → 本格式）。

## index.json

```json
[
  ["scenes/kitchen_01.json", 300, "train"],
  ["scenes/kitchen_02.json", 240, "test"]
]
```

每行是 `[场景 JSON 路径（相对根目录）, 帧数, split]`。split 默认用 `train` 和 `test`（可以用 `--train-split` / `--test-split` 指定别的名字）。同一段视频不要同时出现在 train 和 test 里。

## 场景 JSON

```json
{
  "scene_id": "kitchen_01",
  "frames": {
    "000000": {
      "rgb_path": "images/kitchen_01/000000.jpg",
      "depth_path": "depth/kitchen_01/000000.png",
      "rgb_intrinsic": [[fx, 0, cx], [0, fy, cy], [0, 0, 1]],
      "camera_pose": [[r11, r12, r13, tx], [r21, r22, r23, ty], [r31, r32, r33, tz], [0, 0, 0, 1]],
      "original_width": 1920,
      "original_height": 1080
    }
  },
  "objects": {
    "17": {
      "3d_category": "chair",
      "000000": {
        "camera_center": [x, y, z],
        "camera_scale": [sx, sy, sz],
        "camera_R": [[...], [...], [...]],
        "camera_box_2d_rend": [x1, y1, x2, y2],
        "camera_box_2d_proj": [x1, y1, x2, y2]
      }
    }
  }
}
```

**frames**：key 是帧 id，loader 按**字符串**排序后当作时间顺序，所以要补零（`000123`，不要用 `123`）。连续的帧应该是视频里相邻的帧。

| 字段 | 说明 |
|---|---|
| `rgb_path` | RGB 图片，相对根目录 |
| `depth_path` | 深度图，可选（没有就省略，深度 loss 会忽略这一帧）。16-bit PNG 时值 / `--depth-scale`（默认 1000，即毫米）为米；`.npy` / `.npz`（key `depth`）/ `.exr` 直接是米。必须是相机 z 深度（不是到光心的距离），并和 RGB 视场对齐；分辨率可以更低，会被 resize 到 RGB 大小。无效值填 0 |
| `rgb_intrinsic` | 3×3 内参，像素单位，对应原始 RGB 分辨率 |
| `camera_pose` | 4×4 **相机到世界**（C2W）变换，米。所有帧在同一个世界坐标系下。没有位姿时可以省略（当作单位阵，此时相机位姿 loss 没有意义，时序一致性检查会跳过）|
| `original_width` / `original_height` | RGB 原始尺寸。物体过滤要用；不写时按 1024×1024 处理 |

**objects**：key 是物体的**跟踪 id**，同一个物体在所有帧里用同一个 id（模型在第一帧用 GT 2D 框提示它，然后在后面的帧里跟踪它）。`3d_category` 是类别名，评估按它分类统计。之后每个 key 是该物体可见的帧 id，值是这一帧的框：

| 字段 | 说明 |
|---|---|
| `camera_center` | 3D 框中心，**这一帧的相机坐标系**（OpenCV：x 右、y 下、z 前），米 |
| `camera_scale` | 3D 框三条边的长度，米，顺序和 `camera_R` 的三列一一对应 |
| `camera_R` | 3×3 旋转矩阵（行列式 +1），第 i 列是第 i 条边在相机坐标系下的方向。**局部 +Y 必须是最接近竖直、朝下（重力方向）的那根轴**（和 CA-1M、Infinigen 一致）：模型的朝向（绕竖直轴的 yaw）和尺寸目标都按这个约定计算。如果你的框是 Z 朝上，改成 `R' = [X, −Z, Y]`、`scale' = [sx, sz, sy]`；竖直轴随物体变化的（如 PCA 拟合的框）要逐个物体换：在世界坐标下找出最接近竖直的那根轴 k，令 Y' = 指向下方的 ±轴 k，X' = 轴 (k+1)%3，Z' = X' × Y'，三条边长按同样顺序重排。`validate.sh` 会检查这一点 |
| `camera_box_2d_rend` | 物体**可见部分**的 2D 框，像素 `x1, y1, x2, y2`，在图像内 |
| `camera_box_2d_proj` | 3D 框 8 个角点投影后的外接 2D 框，可以超出图像 |

8 个角点 = `camera_center + camera_R @ (±sx/2, ±sy/2, ±sz/2)`。框在相机坐标系下给出，所以每帧都要写；物体静止时，`camera_pose @ camera_center` 在各帧应该基本不变（`validate.sh` 会检查）。

## 哪些物体会被用到

每个训练/测试 clip 从视频里取连续若干帧（测试固定 8 帧、不重叠），只使用在 **clip 第一帧**满足以下条件的物体（lite_model 的规则，`validate.sh` 会统计可用比例）：

- `camera_center` 的 z > 0.1 m；
- 最长边 > 0.3 m，最短边 > 0.1 m；
- `camera_box_2d_proj` 的中心在图像内（离边缘 > 10 px）；
- `camera_box_2d_rend` 宽高都 ≥ 30 px，且没有大部分截断在图外。

训练时每个 clip 最多随机取 12 个物体，测试时取前 10 个。
