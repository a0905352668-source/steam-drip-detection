# 验证记录

日期：2026-10-16（北京时间）。验证使用原服务器的临时独立目录，不改动原项目、数据集或模型。

## 已通过

| 项目 | 检查内容 |
|---|---|
| Python 静态检查 | 27 个 Python 文件语法解析通过，Git diff 无空白格式错误 |
| 命令行 | 两套预处理、推理、训练、实验训练、模型转换及滴水完整处理入口的帮助参数可用 |
| 模型归档 | 11 个文件 SHA-256 与清单一致，总字节数 328,088,662 |
| 水蒸气预处理 | 合成视频生成切片、元数据及视频输出 |
| 滴水预处理 | 基础版与加速版均生成 3 个可读取的竖条视频 |
| PyTorch 推理 | 水蒸气 v8.1、滴水 v4.28.1 在 CUDA 上输出有效概率；两个模型另以 CPU 完成单区域推理 |
| ONNX 推理 | 水蒸气 v7.2、滴水 v4.28.1 在 ONNX Runtime CPU provider 上输出有效概率 |
| 滴水后端比较 | 同一合成输入上，PyTorch CUDA 与 ONNX CPU 最大概率差约 9.44×10⁻⁵（关闭 TTA） |
| 滴水完整流程 | 6 秒合成视频经 ffmpeg 分为 2 段，输出 2 条片段汇总与 2 个可解码的区域可视化视频 |
| 失败路径 | 无效输入以非零状态退出 |
| 水蒸气 C++ | 两个独立程序 CMake 配置与编译通过，合成视频预处理及既有 v7.2 TensorRT engine 推理成功 |
| 敏感文本 | 整理后的代码与文档未发现私钥、GitHub 令牌或密码赋值 |

## 实际环境

- Linux x86_64，Python 3.9.23。
- PyTorch 2.5.0+cu121，torchvision 0.20.0+cu121，PyTorchVideo 0.1.5。
- PyAV 15.1.0，NumPy 2.0.2，OpenCV Python 4.12.0，ONNX 1.19.1。
- ONNX Runtime 1.19.2 在该环境实际仅提供 CPU／Azure provider；**没有验证 ONNX CUDA 推理**。代码会打印实际 provider，显式要求 CUDA 时不可用会报错。
- C++：GCC 11.4、OpenCV 4.11.0、CUDA Toolkit 12.1、TensorRT 8.6.1.6，NVIDIA GeForce RTX 3080。

已有 TensorRT engine 加载时提示跨设备模型可能影响性能或兼容性；本机合成输入运行成功，不代表可以在任意 NVIDIA 设备直接复用。原 C++ 使用的 TensorRT 8 binding API 有弃用警告，尚未迁移至新版本 API。

## 如何复现

```bash
python scripts/download_models.py
python scripts/smoke_test.py --device auto
python scripts/download_models.py --verify-only --include-historical
```

最后一条要求先使用 `--include-historical` 下载全部权重。`outputs/smoke/report.json` 记录最小运行验证结果；输出和合成视频不提交到 Git。C++ 编译命令见 [部署说明](DEPLOYMENT.md)。

## 验证边界

以上合成数据用于验证程序连接、视频解码和数值输出，不包含真实水蒸气／滴水标注，不能证明检出率、误报率或现场泛化能力。本次没有重新训练模型、完整复现历史训练，或验证 Mac／Windows 上的完整依赖环境。TensorRT Python 转换没有执行（原测试环境未安装其 Python 包），已有 ONNX 文件通过实际推理验证，新导出流程尚未执行完整转换回归。历史模型仅核对文件与配置，未逐个进行精度比较。
