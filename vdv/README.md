# Video dataset format

English | [中文](README_zh.md)

Your data and the standard data (CA-1M 10k) use the same format, read by lite_model's generic video loader (`ObjectDetectionDataset`). One directory per dataset:

```
/path/to/mydata/
├── index.json          scene list and splits
├── scenes/*.json       one file per scene (one video)
└── ...                 images, depth, ...; paths are relative to the dataset root, any layout
```

Check it with `scripts/validate.sh --custom /path/to/mydata`: fix every ERROR, then look at the renders to see whether the 3D boxes fit the objects. Reference converter: `tools/prepare_ca1m.py` (CA-1M → this format).

## index.json

```json
[
  ["scenes/kitchen_01.json", 300, "train"],
  ["scenes/kitchen_02.json", 240, "test"]
]
```

Each row is `[scene JSON path (relative to the root), number of frames, split]`. Splits default to `train` and `test` (other names via `--train-split` / `--test-split`). Do not put the same video in both splits.

## Scene JSON

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

**frames**: keys are frame ids. The loader sorts them **as strings** to get the time order, so zero-pad them (`000123`, not `123`). Consecutive frames should be neighbouring video frames.

| field | meaning |
|---|---|
| `rgb_path` | RGB image, relative to the root |
| `depth_path` | depth map, optional (omit it and the depth loss ignores the frame). 16-bit PNG: value / `--depth-scale` (default 1000 = millimeters) is meters; `.npy` / `.npz` (key `depth`) / `.exr` are meters. Must be camera z-depth (not distance to the optical center) with the RGB field of view; a lower resolution is resized to the RGB size. 0 = invalid |
| `rgb_intrinsic` | 3×3 intrinsics in pixels of the original RGB image |
| `camera_pose` | 4×4 **camera-to-world** (C2W) transform in meters, all frames in one world frame. May be omitted (identity; the camera-pose loss is then meaningless and the temporal check is skipped) |
| `original_width` / `original_height` | original RGB size, used by the object filter; 1024×1024 is assumed otherwise |

**objects**: keys are object **track ids**; an object keeps its id in every frame (the model is prompted with the object's GT 2D box in the first frame and tracks it through the clip). `3d_category` is the category name the evaluation groups by. Every other key is a frame id where the object is visible, holding that frame's box:

| field | meaning |
|---|---|
| `camera_center` | 3D box center in **this frame's camera coordinates** (OpenCV: x right, y down, z forward), meters |
| `camera_scale` | edge lengths of the 3D box in meters, in the order of the columns of `camera_R` |
| `camera_R` | 3×3 rotation (det +1); column i is the direction of edge i in camera coordinates. **Local +Y must be the axis closest to vertical, pointing down (gravity)**, as in CA-1M and Infinigen: the model's orientation (yaw about the vertical) and size targets assume it. Z-up boxes become `R' = [X, -Z, Y]`, `scale' = [sx, sz, sy]`; if the vertical axis varies per object (e.g. PCA-fitted boxes), convert each object: find the axis k closest to world vertical, set Y' = ±axis k pointing down, X' = axis (k+1)%3, Z' = X' × Y', and reorder the sizes alike. `validate.sh` checks this |
| `camera_box_2d_rend` | 2D box of the **visible part** of the object, pixels `x1, y1, x2, y2`, inside the image |
| `camera_box_2d_proj` | 2D bounding box of the 8 projected 3D box corners; may extend beyond the image |

The 8 corners are `camera_center + camera_R @ (±sx/2, ±sy/2, ±sz/2)`. Boxes are given in camera coordinates, so every frame needs its own; for a static object `camera_pose @ camera_center` should stay (nearly) constant across frames (`validate.sh` checks this).

## Which objects are used

Each training / test clip takes consecutive frames of one video (test: 8 frames, non-overlapping). Only objects that pass lite_model's filter in the **first frame of the clip** are used (`validate.sh` reports the usable fraction):

- `camera_center` z > 0.1 m;
- longest edge > 0.3 m, shortest edge > 0.1 m;
- center of `camera_box_2d_proj` inside the image (more than 10 px from the border);
- `camera_box_2d_rend` at least 30 px wide and high, and not mostly truncated by the image border.

Training samples up to 12 objects per clip at random; testing uses the first 10.
