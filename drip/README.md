# 水滴／滴水检测

X3D-M 视频二分类。当前默认模型为 `checkpoints_v4.28.1/OKepoch_40.pt`；完整处理工具默认使用同版本 ONNX。输入为 36 帧、480×224，mean `(0.05,0.05,0.05)`，std `(0.15,0.15,0.15)`。模型的输入是预处理后的运动带通图视频。

训练数据负类目录必须叫 `nodrip`，正类叫 `drip`；推理展示负类名称为 `no_drip`。类别顺序没有改变。

本页所有命令从仓库根目录运行。安装环境、下载模型参见 [总说明](../README.md)。

## 完整视频处理

```bash
python drip/tools/run_drip_pipeline.py data/raw/drip/example.mp4 \
  --output outputs/drip/example --device cpu --num-strips 3 --threshold 0.5
```

该工具按 5 秒分段（≤5.2 秒的源视频直接复制为一个片段），对每段生成竖条运动视频，再分类和汇总。长视频分段依赖 `ffmpeg`。输出目录已经存在时会报错；只有明确需要删除并重建该输出目录时才使用 `--overwrite`。

| 输出 | 内容 |
|---|---|
| `split/` | 分段后的原视频 |
| `preprocessed/` | 每段对应的竖条运动视频 |
| `strip_results_threshold_0.5.csv` | 每个竖条的概率及类别 |
| `segment_results_threshold_0.5.csv` | 每段最高概率、平均概率、阳性条数与汇总 |
| `visualized_strip_boxes/` | 原画面上的阳性竖条区域可视化 |

片段中任意一个竖条的滴水概率 ≥ 阈值，即判定该片段为 `drip`。可视化是分类区域的覆盖框，不能视为精确水滴定位。

## 分步处理

```bash
python drip/preprocess.py --input data/raw/drip/example.mp4 \
  --output outputs/drip/preprocessed --num-strips 3 --short-window 10 \
  --long-window 30 --target-height 480 --frame-stride 1 --overlap 0.2
python drip/inference.py --input outputs/drip/preprocessed \
  --output outputs/drip/results.csv --device cpu
```

预处理目录结构为 `输出目录/原视频名/strip_00.mp4`。输入可为单视频或目录，目录只处理直接子文件。灰度帧差经过短、长 EMA 相减及归一化，转换为三通道灰度视频。推理再 resize 到 480×224，均匀采样 36 帧。

`preprocess_fast.py` 将 resize 提前到帧差和 EMA 之前以减少计算；由于运算顺序不同，与基础版并非逐像素等价。当前完整处理工具默认保留基础版，请用现场数据对比后再切换。

```bash
python drip/preprocess_fast.py --input data/raw/drip/example.mp4 \
  --output outputs/drip/preprocessed-fast
python drip/inference.py --input outputs/drip/preprocessed-fast \
  --model drip/checkpoints_v4.28.1/OKepoch_40.onnx \
  --output outputs/drip/fast_results.csv --device cpu
```

默认不再复制少数类视频到输入目录。阈值影响 CSV 的 `drip@阈值` 列，`pred_label` 始终表示概率最大类别。

## 训练

请先按原视频或采集场景划分，再生成各集合的竖条视频，避免同一原视频的数据泄漏到两个集合。

```text
data/drip/
├── train/
│   ├── nodrip/**/*.mp4
│   └── drip/**/*.mp4
└── val/
    ├── nodrip/**/*.mp4
    └── drip/**/*.mp4
```

```bash
python drip/train.py --root data/drip --arch x3d_m \
  --frames 36 --size_h 480 --size_w 224 \
  --mean 0.05,0.05,0.05 --std 0.15,0.15,0.15 \
  --epochs 40 --batch 4 --workers 4 --freeze_epochs 0 \
  --ckpt outputs/drip-training/best.pt
```

`--ckpt` 是最佳模型保存路径，不能用来指定恢复训练；逐 epoch 检查点在其同级 `checkpoints_v2/`。`--pretrained 1` 会下载基础预训练权重；离线或随机初始化请指定 `--pretrained 0`。

`train_new.py` 是保留的实验训练分支：采用“当前灰度／运动／前一帧灰度”的通道组合假设，mean/std 及增强方法不同。当前 `preprocess.py` 导出的是三通道灰度带通视频，不能直接据此认为实验分支与已上传模型配套；本次没有把该分支设为默认流程。

## 历史模型与工具

历史 `checkpoints/best.pt` 输入宽度为 **226**，mean=0、std=1，不能沿用默认参数：

```bash
python scripts/download_models.py --task drip --include-historical
python drip/inference.py --input outputs/drip/preprocessed \
  --model drip/checkpoints/best.pt --size-w 226 \
  --mean 0,0,0 --std 1,1,1 --output outputs/drip/historical.csv
```

该命令只设置历史模型的输入规格；历史训练数据和预处理配置未完全追溯，不能据此保证现场性能。

- `tools/split_videos.py`：长视频分段。
- `tools/split_train_val.py`：数据集划分参考工具；用于生成的竖条时需自行确保同源分组。
- `tools/visualize_drip_strips.py`：区域画框，可查看 `--help`。
- `band.py VIDEO`：检查运动带通强度的诊断工具。
- `pt2onnx.py`、`onnx2engine.py`：模型转换，参见 [部署说明](../docs/DEPLOYMENT.md)。
