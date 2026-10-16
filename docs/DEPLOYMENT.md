# 模型导出与 C++ 部署

## Python／ONNX

Python 推理支持 `.pt`／`.pth` checkpoint、TorchScript 和 `.onnx`。`.engine` 由 TensorRT／C++ 部署流程读取，不能传给当前 `inference.py`。

水蒸气的默认 PyTorch 模型是 v8.1，已有 ONNX／engine 是 v7.2。若需要与 v8.1 对应的 ONNX，请重新导出到新文件：

```bash
python steam/deploy/converter/pt2onnx.py \
  --checkpoint steam/checkpoints/best.pt \
  --output outputs/steam/v8.1.onnx --frames 48 --size-h 256 --size-w 256

python drip/pt2onnx.py \
  --checkpoint drip/checkpoints_v4.28.1/OKepoch_40.pt \
  --output outputs/drip/v4.28.1.onnx --frames 36 --size-h 480 --size-w 224
```

导出输入名为 `video`，输出名为 `logits`，默认 ONNX opset 17，batch／时间维动态，H／W 固定；可用 `--static` 固定全部维度。转换时会严格检查 state_dict，并在安装 ONNX 的情况下执行结构检查。转换完成后还应在代表性数据上比较后端概率和阈值结果。本次发布保留原始模型文件，未重新转换训练权重。

## TensorRT engine

需在目标 NVIDIA 环境中安装 TensorRT Python 包。原转换实现基于 TensorRT 8 系列；新环境需自行验证兼容性。示例：

```bash
python steam/deploy/converter/onnx2engine.py \
  --input outputs/steam/v8.1.onnx --output outputs/steam/v8.1.engine \
  --frames 48 --size-h 256 --size-w 256

python drip/onnx2engine.py \
  --input drip/checkpoints_v4.28.1/OKepoch_40.onnx \
  --output outputs/drip/v4.28.1.engine --frames 36 --size-h 480 --size-w 224
```

默认允许 FP16；`--fp32` 关闭该优化。命令行模式固定时间维，batch profile 为 1～4。模型文件内置的 H／W 必须与参数一致。归档的 `steam/checkpoints/best.engine` 仅作为原部署产物保留，engine 与生成时的硬件、TensorRT／CUDA 环境相关，建议在目标设备重新生成。

## 水蒸气 C++ 代码

| 路径 | 用途 |
|---|---|
| `steam/deploy/steam/steam.h`、`steam.cpp` | 原业务集成接口参考，需要接入宿主工程 |
| `steam/deploy/steam/steam_pre/steam_pre.cpp` | 独立原视频预处理程序 |
| `steam/deploy/steam/steam_infer/steam_infer.cpp` | 独立 TensorRT 分类程序 |
| 两个目录中其余 `.cpp` | 原项目保留的参考版本；不编入默认目标 |

CMake 默认 TensorRT 目录为 `/opt/TensorRT-8.6.1.6`，可通过 `-DTENSORRT_ROOT=...` 覆盖；OpenCV 不能自动找到时传入 `-DOpenCV_DIR=...`。本次将推理 CMake 的默认源文件修正为 `steam_infer.cpp`。

```bash
cmake -S steam/deploy/steam/steam_pre -B outputs/build-steam-pre \
  -DCMAKE_BUILD_TYPE=Release -DTENSORRT_ROOT=/opt/TensorRT-8.6.1.6 \
  -DCUDAToolkit_ROOT=/usr/local/cuda
cmake --build outputs/build-steam-pre -j4

cmake -S steam/deploy/steam/steam_infer -B outputs/build-steam-infer \
  -DCMAKE_BUILD_TYPE=Release -DTENSORRT_ROOT=/opt/TensorRT-8.6.1.6 \
  -DCUDAToolkit_ROOT=/usr/local/cuda
cmake --build outputs/build-steam-infer -j4

outputs/build-steam-pre/cpp --input data/raw/steam/example.mp4 \
  --output outputs/steam/cpp-pre
outputs/build-steam-infer/cpp steam/checkpoints/best.engine \
  outputs/steam/cpp-pre 1 48 256 256 0.5
```

C++ 预处理和 Python 预处理来自不同演进版本，部分切片参数及评分实现有区别；切换前应逐项对齐并使用实际业务视频回归。C++ 推理程序的单区域阈值也不等同于 Python 控制台中的多区域聚合规则。当前滴水项目未找到独立 C++ 实现；已提供 Python／ONNX 及 engine 构建工具。

本次已在原服务器完成上述两个 C++ 程序的编译，并用合成视频跑通预处理和既有 v7.2 engine 推理；这不代表生产现场精度验证。具体环境和边界见 [验证记录](VALIDATION.md)。
