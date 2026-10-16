# 环境安装与兼容性

## Python 环境

推荐 Python 3.10。原服务器的现有运行环境为 Linux x86_64、Python 3.9.23、PyTorch 2.5.0+cu121、torchvision 0.20.0+cu121、PyTorchVideo 0.1.5、NumPy 2.0.2、OpenCV 4.12.0、PyAV 15.1.0、ONNX 1.19.1、ONNX Runtime 1.19.2。

根目录 `requirements.txt` 根据实际可用环境固定了主要依赖；PyTorch 的 CPU／CUDA 构建由所用安装源决定，清单不包含系统显卡驱动、CUDA 工具链或 TensorRT。

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Windows 使用 `.venv\Scripts\activate`。Mac 推理使用 `--device cpu`；本次没有在 Mac 新建并验证完整机器学习环境。

## NVIDIA GPU

原服务器验证的是 PyTorch CUDA 12.1 构建。若目标机器驱动支持该构建，可以先安装对应 torch／torchvision，再装其余依赖：

```bash
python -m pip install torch==2.5.0 torchvision==0.20.0 \
  --index-url https://download.pytorch.org/whl/cu121
python -m pip install -r requirements.txt
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

`--device auto` 在 PyTorch CUDA 可用时选择显卡，否则使用 CPU。ONNX 后端单独检查自己的 provider；PyTorch 有 CUDA 不代表 ONNX Runtime 也有 CUDA。

基础依赖默认安装 ONNX Runtime CPU 版本。如果需要 ONNX CUDA，请在安装基础依赖后移除相互冲突的包，再安装 GPU 构建，并确认目标机器的 CUDA／cuDNN 动态库满足其要求：

```bash
python -m pip uninstall -y onnxruntime onnxruntime-gpu
python -m pip install onnxruntime-gpu==1.19.2
python -c "import onnxruntime as ort; print(ort.get_available_providers())"
```

列表中应有 `CUDAExecutionProvider`。显式指定 `--device cuda` 时，ONNX CUDA provider 缺失或初始化失败会报错；`--device auto` 允许 CPU 回退，并打印实际 provider。不要在 GPU 包安装后再次直接安装根清单，否则可能重新安装 CPU 包。

## 系统工具

- `gh`：访问私有仓库和模型附件。
- `ffmpeg`：滴水长视频分段，需出现在 PATH 中；Ubuntu 可安装 `sudo apt install ffmpeg`，Mac 可用 `brew install ffmpeg`。
- C++ 部署额外需要 CMake、OpenCV 开发库、CUDA Toolkit 和 TensorRT；详见 [部署说明](DEPLOYMENT.md)。

## 常见问题

| 现象 | 处理 |
|---|---|
| GitHub 404 或 Release 下载失败 | 私有仓库需要有访问权限，并先运行 `gh auth login` |
| 模型文件不存在 | 运行 `python scripts/download_models.py`，不要只下载仓库源码 ZIP |
| `torchvision` 算子／导入报错 | 确保 torch 2.5.0 与 torchvision 0.20.0 使用相匹配的构建 |
| `av` 或视频读取失败 | 安装清单中的 PyAV，确认 OpenCV 能解码输入视频 |
| 历史滴水模型尺寸报错或结果异常 | 旧模型宽度 226、mean=0、std=1，不能沿用当前参数 |
| 长视频内存占用高 | 先分段；水蒸气预处理及视频读取会将片段加载到内存 |
| 缺少 TensorRT 或 `.engine` 无法加载 | Python 常规推理请使用 `.pt` 或 `.onnx`；engine 需匹配 NVIDIA 环境 |

训练使用 `--pretrained 1` 时仍会联网下载基础预训练权重；已上传的业务 checkpoint 用于推理或归档，不是训练脚本的自动恢复参数。
