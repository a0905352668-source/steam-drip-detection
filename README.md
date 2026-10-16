# 水蒸气与滴水视频检测

基于 X3D-M 的工业视频二分类代码，包含水蒸气检测和水滴／滴水检测两条流程：原视频预处理、模型训练、PyTorch／ONNX 推理、模型导出，以及水蒸气 C++ 部署参考实现。

这是现有项目的整理版。原始训练权重保持不变；本次工作主要补充命令行入口、路径整理、模型版本说明和使用文档。模型输出是视频区域的分类结果，画框表示切片区域，并非逐滴实例检测框。未重新评估现场检出率或误报率。

新增：[料袋密封检测的实时试验流程](docs/REALTIME_PACKING.md)，支持海康机器人 MVS 适配、检测周期信号、同袋背景参考和区域分类。现有模型仍是水蒸气权重，对微弱检漏烟雾的现场检出率尚未验证。

| 项目 | 类别顺序 | 默认输入 C×T×H×W | 归一化 mean / std | 默认模型 |
|---|---|---|---|---|
| 水蒸气 | `no_steam`, `steam` | 3×48×256×256 | 0.5 / 0.25（各通道） | v8.1 PyTorch |
| 滴水 | `no_drip`, `drip` | 3×36×480×224 | 0.05 / 0.15（各通道） | v4.28.1 epoch 40 |

滴水训练集的负类目录名称为 **`nodrip`**，推理展示名称为 `no_drip`，二者都对应类别 0。

## 目录

```text
steam/                 水蒸气：预处理、训练、推理、模型转换、C++ 源码
drip/                  滴水：预处理、训练、推理、完整视频处理工具
models/manifest.json   模型文件名、来源、输入参数、大小和 SHA-256
scripts/               模型下载和最小运行验证
docs/                  环境、部署、来源和验证记录
requirements.txt       共享 Python 依赖
```

模型二进制放在 [GitHub Release v1.0.0](https://github.com/a0905352668-source/steam-drip-detection/releases/tag/v1.0.0)，不写入 Git 历史。只下载仓库 ZIP 不包含模型；下载工具会把模型恢复到脚本所需的位置。

## 1. 获取代码和模型

仓库已公开，可直接浏览。下述下载工具使用 [GitHub CLI](https://cli.github.com/)，命令行使用前按提示登录 GitHub。

```bash
gh auth login
gh repo clone a0905352668-source/steam-drip-detection
cd steam-drip-detection
python3 scripts/download_models.py
# 需要历史权重时：
python3 scripts/download_models.py --include-historical
# 校验已经下载的所有模型：
python3 scripts/download_models.py --verify-only --include-historical
```

下载工具只使用 Python 标准库和已登录的 `gh`，不需要先装深度学习依赖。每个文件下载后会校验 SHA-256。

**版本提醒：** 水蒸气目录里原名都为 `best` 的文件，实际上来自不同版本：`best.pt` 是 v8.1，而 `best.onnx`／`best.engine` 是 v7.2。这些文件已分别按真实版本命名为 Release 附件，二者不能作为同一套权重比较。详见 [模型清单](models/README.md)。

## 2. 安装环境

建议新建 Python 3.10 环境。所有命令从仓库根目录执行。

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Windows 激活命令为 `.venv\Scripts\activate`。Mac 可使用 CPU 推理；仓库未启用 MPS。CUDA／TensorRT 在具有 NVIDIA 显卡的 Linux 环境运行。滴水长视频分段还需要系统安装 `ffmpeg`。GPU 依赖安装和服务器实际验证环境见 [环境说明](docs/ENVIRONMENT.md)。

## 3. 水蒸气检测

先对原视频切片，再把切片送入分类模型。支持单个视频或包含视频的目录。

```bash
python steam/preprocess.py \
  --input data/raw/steam/example.mp4 \
  --output outputs/steam/preprocessed

python steam/inference.py \
  --input outputs/steam/preprocessed \
  --output outputs/steam/results.csv
```

默认 Top-5、512 像素切片、20% 重叠、导出 256×256 视频。紧致裁剪与背景净化需要显式传入 `--tighten --bg-mode blur`；保持与具体模型训练时的预处理一致。长视频会整体读入内存，建议先切成较短片段。

使用已有 v7.2 ONNX 部署模型：

```bash
python steam/inference.py --input outputs/steam/preprocessed \
  --model steam/checkpoints/best.onnx --output outputs/steam/onnx_results.csv
```

详细目录、参数、训练命令和结果规则见 [水蒸气使用说明](steam/README.md)。

## 4. 水滴／滴水检测

已有完整视频时，推荐使用完整处理工具：按 5 秒分段、运动带通预处理、竖条分类、按片段汇总、生成区域可视化。

```bash
python drip/tools/run_drip_pipeline.py data/raw/drip/example.mp4 \
  --output outputs/drip/example --device cpu
```

该工具默认使用已归档的 v4.28.1 ONNX；GPU 环境可改为 `--device cuda`。输出包括 `strip_results_threshold_0.5.csv`、`segment_results_threshold_0.5.csv` 和 `visualized_strip_boxes/`。

分步执行：

```bash
python drip/preprocess.py --input data/raw/drip/example.mp4 \
  --output outputs/drip/preprocessed --num-strips 3
python drip/inference.py --input outputs/drip/preprocessed \
  --output outputs/drip/results.csv --device cpu
```

默认生成 3 个竖条，先做灰度、帧差、短长 EMA 带通，再生成灰度三通道视频。**请勿直接把原始彩色视频当作预处理后的模型输入。** 加速预处理版本和实验训练分支见 [滴水使用说明](drip/README.md)。

## 5. 结果与参数

`inference.py --help` 可查看全部命令行参数；两条流程都支持 `--model`、`--device`、`--threshold`、`--high-threshold`、`--frames`、`--size-h`、`--size-w`、`--mean`、`--std`、`--no-tta`。

CSV 中 `pred_label` 是概率最大类别；`steam@阈值`／`drip@阈值` 才是自定义阈值的判定，因此修改阈值后两者可能不同。概率字段为 `prob_no_steam`／`prob_steam` 或 `prob_no_drip`／`prob_drip`。发生视频读取或推理错误时保留 `ERROR` 行并以非零状态退出。

## 6. 训练、导出与验证

- [水蒸气训练](steam/README.md#训练)及[滴水训练](drip/README.md#训练)：数据目录、输入尺寸、增强和保存位置。
- [模型清单](models/README.md)：当前权重、历史权重、版本差异和 SHA-256。
- [部署说明](docs/DEPLOYMENT.md)：ONNX 导出、TensorRT engine 构建、C++ 编译。
- [来源与整理范围](docs/PROVENANCE.md)：与服务器项目的对应关系。
- [验证记录](docs/VALIDATION.md)：已执行的检查与验证边界。

```bash
python scripts/smoke_test.py --device cpu
```

该命令生成合成视频，检查两条预处理流程及当前 PyTorch／已有 ONNX 模型能否产生有效概率。它不衡量实际水蒸气或滴水识别准确率。

训练集、现场视频、凭据和每个 epoch 的全部中间模型不在本仓库和 Release 中。未附加开源授权；如需对外公开或分发，请由项目权利人确定授权范围。
