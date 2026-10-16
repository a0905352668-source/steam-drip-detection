# 模型清单与下载

本次随 [Release v1.0.0](https://github.com/a0905352668-source/steam-drip-detection/releases/tag/v1.0.0) 上传 11 个原始模型文件，总计 328,088,662 字节（约 313 MiB）。所有模型保持原始字节，未重新训练或改写权重。

## 下载

```bash
gh auth login
python3 scripts/download_models.py                    # 当前推理和既有部署文件
python3 scripts/download_models.py --task steam        # 仅水蒸气
python3 scripts/download_models.py --include-historical # 含所有历史权重
python3 scripts/download_models.py --verify-only --include-historical
```

默认下载水蒸气 v8.1 PT、v7.2 ONNX／engine，以及滴水 v4.28.1 PT／ONNX，共 5 个文件。历史文件需要 `--include-historical`。也可以从 Release 手动下载，按下表放入本地路径。完整机器可读清单为 [manifest.json](manifest.json)，校验值为 [SHA256SUMS](SHA256SUMS)。

## 文件对应关系

| Release 附件 | 下载后的仓库相对路径 | 大小 MiB | 用途 |
|---|---|---:|---|
| `steam-v8.1.pt` | `steam/checkpoints/best.pt` | 34.66 | 默认推理 |
| `steam-v7.2.onnx` | `steam/checkpoints/best.onnx` | 11.39 | 历史 v7.2 ONNX 部署 |
| `steam-v7.2.engine` | `steam/checkpoints/best.engine` | 12.84 | 历史 v7.2 engine，需匹配硬件 |
| `steam-x3d.pth` | `steam/weights/steam_x3d.pth` | 34.66 | 历史权重 |
| `steam-x3d-v2.pth` | `steam/weights/steam_x3d_v2.pth` | 34.66 | 历史权重 |
| `drip-v4.28.1-epoch40.pt` | `drip/checkpoints_v4.28.1/OKepoch_40.pt` | 34.66 | 默认推理 |
| `drip-v4.28.1-epoch40.onnx` | `drip/checkpoints_v4.28.1/OKepoch_40.onnx` | 11.39 | 默认推理 |
| `drip-historical-best.pt` | `drip/checkpoints/best.pt` | 34.66 | 历史权重，尺寸与归一化不同 |
| `drip-x3d-v2.pth` | `drip/weights/drip_x3d_v2.pth` | 34.66 | 历史权重 |
| `steam-v7.2-epoch25.pt` | `steam/checkpoints_v7.2/epoch_25.pt` | 34.66 | 历史权重 |
| `steam-v7.3-epoch25.pt` | `steam/checkpoints_v7.3/epoch_25.pt` | 34.66 | 历史权重 |

## 输入规格

| 模型 | C×T×H×W | mean | std | 类别顺序 |
|---|---|---|---|---|
| 水蒸气 v8.1／v7.2／v7.3 及两份历史 pth | 3×48×256×256 | 0.5,0.5,0.5 | 0.25,0.25,0.25 | no_steam, steam |
| 滴水 v4.28.1／v2 pth | 3×36×480×224 | 0.05,0.05,0.05 | 0.15,0.15,0.15 | nodrip, drip |
| 滴水历史 best.pt | 3×36×480×226 | 0,0,0 | 1,1,1 | nodrip, drip |

水蒸气原目录里：`best.pt` 与 `X3D_video/model/steam_check_v8.1/epoch_25.pt` 的 SHA-256 相同；`best.onnx`、`best.engine` 则分别与 v7.2 对应文件相同。这是原项目已有的版本差异，本次没有把它们当作同一模型。Release 附件名称带上版本，以避免误用；本地恢复原相对路径供脚本使用。

滴水模型 checkpoint 的负类名称是 `nodrip`，推理输出展示为 `no_drip`，均为类别 0。PyTorch checkpoint 的训练参数可在 manifest 中查看；ONNX 的帧数为采样约定，模型时间维本身为动态维度。

SHA-256 只能验证字节是否与归档一致，不代表模型精度或硬件兼容性。历史权重的完整训练来源尚未逐一重建，不建议只凭文件名认定性能优劣。

`.engine` 是硬件相关部署产物，不能由当前 Python 通用推理入口加载；推荐在目标机器从相应 ONNX 重新构建。当前业务 checkpoint 含原训练元数据，`--pretrained 1` 所需的公共基础模型仍由 PyTorchVideo 下载。

## 未上传的中间模型

原服务器滴水项目保留多个版本的所有 epoch 检查点，合计约 6 GiB。本次上传当前使用的 epoch 40、历史最佳及相关权重；没有重复上传每个 epoch。原服务器文件均未删除或修改。
