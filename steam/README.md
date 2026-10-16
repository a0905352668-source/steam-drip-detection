# 水蒸气检测

X3D-M 视频二分类，类别 0 为 `no_steam`，类别 1 为 `steam`。默认使用 v8.1 PyTorch checkpoint，输入为 RGB、48 帧、256×256，mean 为 `(0.5,0.5,0.5)`，std 为 `(0.25,0.25,0.25)`。

本页所有命令从仓库根目录运行。环境和模型获取参见 [总说明](../README.md)。

## 文件用途

| 文件 | 作用 |
|---|---|
| `preprocess.py` | 帧差热度、网格切片、Top-K、可选紧致裁剪及背景净化 |
| `train.py` | 视频分类训练、时域抖动、正负样本差异增强、可选 EMA |
| `inference.py` | PyTorch／TorchScript／ONNX 分类、阈值判断及 CSV |
| `core/` | 原项目中的公共模型／数据集／工具参考模块 |
| `deploy/converter/` | PyTorch → ONNX、ONNX → TensorRT |
| `deploy/steam/` | C++ 部署参考代码与独立预处理／推理程序 |

当前训练脚本仍保留自己的模型和数据集实现；`core/` 并非所有入口都调用的唯一实现。

## 预处理与推理

```bash
python steam/preprocess.py --input data/raw/steam \
  --output outputs/steam/preprocessed --topk 5 --tile-size 512 --overlap 0.2
python steam/inference.py --input outputs/steam/preprocessed \
  --output outputs/steam/results.csv --device cpu
```

预处理目录结构：`输出目录/原视频名/group_0000/tile_000.mp4`，同级 `tiles_meta.json` 记录切片位置；默认输出完整视频为一组，不强制固定帧数，推理时再均匀采样 48 帧。输入目录会递归读取；不同目录里有同名视频时请先改名，避免输出覆盖。长视频宜先分段以控制内存占用。

常用可选参数：

- `--coverage all`：保留全部网格区域，替代 Top-K。
- `--tighten --bg-mode blur`：按运动掩膜紧致裁剪，并模糊背景；默认关闭。
- `--no-tta`：关闭推理时水平翻转增强。
- `--threshold 0.5 --high-threshold 0.65`：默认区域概率与汇总阈值。

原视频汇总规则：同一 group 中至少 2 个 tile 概率 ≥ `threshold`，或至少 1 个 tile 概率 ≥ `high-threshold`，即认为该组命中；任意组命中则原视频命中。该结果打印到控制台；CSV 保存的是逐 tile 结果。

已有 `checkpoints/best.onnx` 和 `best.engine` 来自 v7.2，与默认 v8.1 `.pt` 不同。切换版本请记录实际模型路径，不要把两个版本的差异当成后端数值误差。

## 训练

先按原始视频或采集场景划分训练／验证集，再分别预处理，避免同一视频的不同 tile 同时进入训练与验证。按类别收集预处理后的视频文件，可放子目录，训练脚本递归读取。

```text
data/steam/
├── train/
│   ├── no_steam/**/*.mp4
│   └── steam/**/*.mp4
└── val/
    ├── no_steam/**/*.mp4
    └── steam/**/*.mp4
```

```bash
python steam/train.py --root data/steam --arch x3d_m \
  --frames 48 --size 256 --size_h 256 --size_w 256 \
  --mean 0.5,0.5,0.5 --std 0.25,0.25,0.25 \
  --epochs 25 --batch 4 --workers 4 --freeze_epochs 0 \
  --tta_flip 1 --ema 1 --ckpt outputs/steam-training/best.pt
```

`--pretrained 1` 会下载 PyTorchVideo 的基础预训练权重，需要联网；`--pretrained 0` 从随机权重开始训练。`--ckpt` 是**最佳模型输出路径**，不是加载已有模型或恢复训练的参数。逐 epoch 检查点保存在其同级 `checkpoints_v2/`，再次训练请使用新的输出目录。

本次仅验证命令行及推理运行，没有重新进行完整训练。部署模型导出见 [部署说明](../docs/DEPLOYMENT.md)。
