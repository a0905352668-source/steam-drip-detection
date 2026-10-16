import os
import time
import shutil
from pathlib import Path
from typing import List, Tuple
import numpy as np
import torch
from torchvision.io import read_video
from torchvision.transforms import functional as TF

# --------- 可选依赖（OpenCV 仅用于读视频回退） ----------
try:
    import cv2
except Exception:
    cv2 = None

# ======================= 用户可配置区 =======================
PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_PATH = str(PROJECT_ROOT / "checkpoints_v4.28.1/OKepoch_40.pt")
INPUT_PATH = ""  # 通过 --input 传入预处理后的视频或目录
CLASSES = ("no_drip", "drip")

# 输入规格（与你训练/预处理一致）
FRAMES = 36
SIZE_H = 480
SIZE_W = 224

# 归一化（建议用你统计值）
MEAN_TUP = (0.05, 0.05, 0.05)
STD_TUP = (0.15, 0.15, 0.15)

# 推理增强
NUM_CLIPS = 1
TTA_FLIP = True
THRESHOLD = 0.50    # 多区域聚合所用阈值
HIGH_THRESHOLD = 0.50  # 单区域触发阈值

# 设备
USE_CUDA_TORCH = torch.cuda.is_available()
REQUIRE_CUDA = False
USE_CUDA_ONNX = False  # 需要已安装 onnxruntime-gpu；否则自动回落CPU
SAVE_CSV = ""  # 如 r"./infer_result.csv"

# 是否将少数类清单另存
SAVE_MINORITY_TXT = ""  # 如 r"./minority_list.txt"；为空则不落盘

# 是否移动少数类视频到新文件夹
MOVE_MINORITY_VIDEOS = False  # True: 移动少数类视频到单独文件夹; False: 不移动

# 日志输出模式：'verbose'=全部输出, 'drip_only'=只输出drip视频, 'compact'=简洁模式(仅drip+统计)
LOG_MODE = "drip_only"  # 推荐: drip_only

# ===========================================================


# ------------------ 通用工具 ------------------
def list_videos(root: Path, exts=(".mp4", ".avi", ".mov", ".mkv")) -> List[Path]:
    files = []
    for ext in exts:
        files += list(root.rglob(f"*{ext}"))
    return sorted(files)


def uniform_indices(total: int, num: int) -> np.ndarray:
    if total <= 0:
        return np.zeros((num,), dtype=np.int64)
    if total >= num:
        idx = np.linspace(0, total - 1, num=num)
        return np.round(idx).astype(np.int64)
    base = list(range(total))
    base += [total - 1] * (num - total)
    return np.array(base, dtype=np.int64)


def segment_indices(total: int, num_frames: int, clip_idx: int, num_clips: int) -> np.ndarray:
    if num_clips <= 1:
        return uniform_indices(total, num_frames)
    start = int(np.floor(clip_idx * total / num_clips))
    end = int(np.floor((clip_idx + 1) * total / num_clips)) - 1
    end = max(start, end)
    seg_len = end - start + 1
    if seg_len >= num_frames:
        idx = np.linspace(start, end, num=num_frames)
        return np.round(idx).astype(np.int64)
    idx = list(range(start, end + 1))
    idx += [end] * (num_frames - seg_len)
    return np.array(idx, dtype=np.int64)


def read_video_fallback(path: str):
    """优先 torchvision.io.read_video；失败时用 OpenCV。返回 (T,C,H,W) uint8 tensor。"""
    try:
        vframes, _, _ = read_video(path, pts_unit="sec", output_format="TCHW")
        if vframes.numel() == 0:
            raise RuntimeError("Empty video")
        return vframes
    except Exception as e:
        if cv2 is None:
            raise RuntimeError(f"read_video失败且无OpenCV可用: {e}")
        cap = cv2.VideoCapture(path)
        frames = []
        while True:
            ok, f = cap.read()
            if not ok:
                break
            f = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
            frames.append(torch.from_numpy(f))
        cap.release()
        if not frames:
            raise RuntimeError("OpenCV读取失败或空视频")
        arr = torch.stack(frames, 0).permute(0, 3, 1, 2).contiguous()  # T,C,H,W
        return arr


def softmax_np(x, axis=-1):
    x = x - np.max(x, axis=axis, keepdims=True)
    ex = np.exp(x)
    return ex / np.sum(ex, axis=axis, keepdims=True)


# ------------------ 预处理 ------------------
@torch.no_grad()
def preprocess_clip_tensor(vframes: torch.Tensor, frames: int, size_h: int, size_w: int,
                           idxs: np.ndarray, mean_t: torch.Tensor, std_t: torch.Tensor) -> torch.Tensor:
    clip = vframes[idxs]  # (T,C,H,W)
    if clip.shape[1] == 1:
        clip = clip.repeat(1, 3, 1, 1)
    clip = TF.resize(clip, [size_h, size_w], antialias=True)
    clip = clip.permute(1, 0, 2, 3).contiguous().float() / 255.0  # (C,T,H,W)
    clip = (clip - mean_t) / std_t
    return clip.unsqueeze(0)  # (1,3,T,H,W)


def preprocess_clip_numpy(vframes: torch.Tensor, frames: int, size_h: int, size_w: int,
                          idxs: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    clip = vframes[idxs]
    if clip.shape[1] == 1:
        clip = clip.repeat(1, 3, 1, 1)
    clip = TF.resize(clip, [size_h, size_w], antialias=True)
    clip = clip.permute(1, 0, 2, 3).contiguous().float() / 255.0  # (C,T,H,W)
    x = clip.numpy()
    x = (x - mean) / std
    return np.expand_dims(x, 0).astype(np.float32)  # (1,3,T,H,W)


# -------------- 后端适配：Torch / TorchScript / ONNX --------------
class TorchModel:
    """PyTorch eager: 构建X3D并load_state_dict"""

    def __init__(self, ckpt_path: str, device: str, num_classes: int):
        from pytorchvideo.models import hub as pv_hub
        # 从ckpt推断arch（若有）
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        args = ckpt.get("args", {}) or {}
        arch = args.get("arch", "x3d_m").lower()
        model = getattr(pv_hub, arch)(pretrained=False)
        # 替换分类头
        import torch.nn as nn
        if hasattr(model.blocks[-1], "proj") and isinstance(model.blocks[-1].proj, nn.Linear):
            in_feats = model.blocks[-1].proj.in_features
            model.blocks[-1].proj = nn.Linear(in_feats, num_classes)
        else:
            head = model.blocks[-1]
            proj = None
            for _, m in head.named_modules():
                if isinstance(m, nn.Linear):
                    proj = m
                    break
            if proj is None:
                raise RuntimeError("未找到X3D分类头线性层")
            in_feats = proj.in_features
            model.blocks[-1].proj = nn.Linear(in_feats, num_classes)
        # 加权
        missing, unexpected = model.load_state_dict(ckpt["state_dict"], strict=True)
        if missing or unexpected:
            print(f"[load_state_dict] missing={missing} unexpected={unexpected}")
        self.model = model.to(device).eval()
        self.device = device

    @torch.no_grad()
    def forward_logits(self, x: torch.Tensor, tta_flip: bool) -> torch.Tensor:
        # x: (1,3,T,H,W) on device
        logits = self.model(x)
        if tta_flip:
            logits = 0.5 * (logits + self.model(torch.flip(x, dims=[-1])))
        return logits.float()


class TorchScriptModel:
    """TorchScript .pt：直接 torch.jit.load"""

    def __init__(self, script_path: str, device: str):
        self.model = torch.jit.load(script_path, map_location=device)
        self.model.eval()
        self.device = device

    @torch.no_grad()
    def forward_logits(self, x: torch.Tensor, tta_flip: bool) -> torch.Tensor:
        logits = self.model(x)
        if tta_flip:
            logits = 0.5 * (logits + self.model(torch.flip(x, dims=[-1])))
        return logits.float()


class OnnxModel:
    """ONNX Runtime 推理"""

    def __init__(self, onnx_path: str, use_cuda: bool):
        try:
            import onnxruntime as ort
        except Exception as e:
            raise RuntimeError(f"需要安装 onnxruntime 或 onnxruntime-gpu，错误：{e}")
        available = ort.get_available_providers()
        if use_cuda and "CUDAExecutionProvider" not in available:
            if REQUIRE_CUDA:
                raise RuntimeError("当前 ONNX Runtime 不支持 CUDA；安装 onnxruntime-gpu 或选择 --device cpu/auto")
            print("[ONNX] CUDA provider 不可用，使用 CPU")
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if use_cuda and "CUDAExecutionProvider" in available else ["CPUExecutionProvider"]
        self.session = ort.InferenceSession(onnx_path, providers=providers)
        actual = self.session.get_providers()
        if REQUIRE_CUDA and "CUDAExecutionProvider" not in actual:
            raise RuntimeError("ONNX CUDA provider 初始化失败，请检查 CUDA/cuDNN 依赖")
        self.device = "cuda" if "CUDAExecutionProvider" in actual else "cpu"
        print(f"[ONNX] actual providers: {actual}")
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        print(f"[ONNX] input={self.input_name} shape={self.session.get_inputs()[0].shape}  "
              f"output={self.output_name} shape={self.session.get_outputs()[0].shape}")

    def forward_logits(self, x: np.ndarray, tta_flip: bool) -> np.ndarray:
        out = self.session.run([self.output_name], {self.input_name: x})[0]  # (B,2)
        if tta_flip:
            x_flip = x[..., :, ::-1]  # 水平翻转W
            out2 = self.session.run([self.output_name], {self.input_name: x_flip})[0]
            out = 0.5 * (out + out2)
        return out.astype(np.float32)


# -------------- 推理统一流程 --------------
def infer_one_video(model_backend, is_onnx: bool, path: Path,
                    frames: int, size_h: int, size_w: int,
                    num_clips: int, tta_flip: bool,
                    mean_t: torch.Tensor, std_t: torch.Tensor,
                    mean_np: np.ndarray, std_np: np.ndarray,
                    device: str) -> Tuple[int, np.ndarray, dict]:
    # 初始化计时字典
    timing = {
        'total': 0.0,
        'read': 0.0,
        'preprocess': 0.0,
        'inference': 0.0,
        'postprocess': 0.0,
        'frames': 0
    }

    start_total = time.time()

    # 读取视频
    start_read = time.time()
    vframes = read_video_fallback(str(path))
    if vframes.shape[1] == 1:
        vframes = vframes.repeat(1, 3, 1, 1)
    T = vframes.shape[0]
    timing['read'] = time.time() - start_read
    timing['frames'] = T

    logits_sum = np.zeros((1, len(CLASSES)), dtype=np.float32)
    num_clips_used = max(1, num_clips)

    # 处理每个clip
    start_preprocess = time.time()
    if is_onnx:
        for ci in range(num_clips_used):
            idxs = segment_indices(T, frames, ci, num_clips)
            x_np = preprocess_clip_numpy(vframes, frames, size_h, size_w, idxs, mean_np, std_np)  # (1,3,T,H,W)
            timing['preprocess'] += time.time() - start_preprocess

            # 推理
            start_inference = time.time()
            logits = model_backend.forward_logits(x_np, tta_flip)  # (1,2)
            timing['inference'] += time.time() - start_inference

            logits_sum += logits
            start_preprocess = time.time()  # 重置预处理计时
    else:
        amp = torch.amp.autocast(device_type='cuda', dtype=torch.float16, enabled=device.startswith("cuda"))
        for ci in range(num_clips_used):
            idxs = segment_indices(T, frames, ci, num_clips)
            x = preprocess_clip_tensor(vframes, frames, size_h, size_w, idxs, mean_t, std_t).to(
                device, non_blocking=True
            )
            timing['preprocess'] += time.time() - start_preprocess

            # 推理
            start_inference = time.time()
            with amp:
                logits = model_backend.forward_logits(x, tta_flip)  # torch.Tensor (1,2)
            timing['inference'] += time.time() - start_inference

            logits_sum += logits.cpu().numpy()
            start_preprocess = time.time()  # 重置预处理计时

    # 后处理
    start_postprocess = time.time()
    logits_avg = logits_sum / num_clips_used
    probs = softmax_np(logits_avg, axis=1).squeeze(0)  # (2,)
    pred = int(np.argmax(probs))
    timing['postprocess'] = time.time() - start_postprocess

    # 总时间
    timing['total'] = time.time() - start_total

    return pred, probs, timing


def format_time(seconds: float) -> str:
    """格式化时间为易读的字符串"""
    if seconds < 1e-3:
        return f"{seconds * 1e6:.1f} μs"
    elif seconds < 1:
        return f"{seconds * 1000:.1f} ms"
    else:
        return f"{seconds:.2f} s"


def main():
    if not Path(MODEL_PATH).is_file():
        raise FileNotFoundError(f"模型不存在: {MODEL_PATH}; 请先运行 python scripts/download_models.py")
    if not INPUT_PATH or not Path(INPUT_PATH).exists():
        raise FileNotFoundError(f"输入不存在: {INPUT_PATH}; 使用 --input 指定预处理后的视频")
    if Path(MODEL_PATH).suffix.lower() not in {".pt", ".pth", ".onnx"}:
        raise ValueError("Python 推理支持 .pt/.pth/.onnx；.engine 请使用 TensorRT/C++")
    print(f"[Config] MODEL={MODEL_PATH}\n        INPUT={INPUT_PATH}\n        FRAMES={FRAMES} SIZE={SIZE_H}x{SIZE_W}")
    print(f"[TTA]    clips={NUM_CLIPS} flip={TTA_FLIP} threshold={THRESHOLD}")
    device = "cuda" if USE_CUDA_TORCH else "cpu"
    if device == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True

    # 选择后端
    ext = Path(MODEL_PATH).suffix.lower()
    is_onnx = (ext == ".onnx")
    if is_onnx:
        backend = OnnxModel(MODEL_PATH, use_cuda=USE_CUDA_ONNX and USE_CUDA_TORCH)
        backend_type = "ONNX"
        device = backend.device
    else:
        # 尝试把 .pt 作为 TorchScript 加载；不行则当作 state_dict checkpoint 加载
        try:
            backend = TorchScriptModel(MODEL_PATH, device=device)
            backend_type = "TorchScript"
            print("[Backend] TorchScript (.pt) loaded.")
        except Exception:
            # 作为传统 checkpoint 构建X3D并load_state_dict
            backend = TorchModel(MODEL_PATH, device=device, num_classes=len(CLASSES))
            backend_type = "PyTorch"
            print("[Backend] PyTorch checkpoint (.pth/.pt state_dict) loaded.")

    mean_t = torch.tensor(MEAN_TUP).view(3, 1, 1, 1)
    std_t = torch.tensor(STD_TUP).view(3, 1, 1, 1)
    mean_np = np.array(MEAN_TUP, dtype=np.float32).reshape(3, 1, 1, 1)
    std_np = np.array(STD_TUP, dtype=np.float32).reshape(3, 1, 1, 1)

    inp = Path(INPUT_PATH)
    paths = list_videos(inp) if inp.is_dir() else [inp]
    if not paths or not paths[0].exists():
        raise FileNotFoundError(f"未找到视频：{INPUT_PATH}")

    rows = []
    total_timing = {
        'total': 0.0,
        'read': 0.0,
        'preprocess': 0.0,
        'inference': 0.0,
        'postprocess': 0.0,
        'frames': 0,
        'videos': 0
    }

    # 统计容器
    names_by_threshold = {0: [], 1: []}   # 0=no_( prob_drip<th ), 1=drip(>=th)
    names_by_label = {CLASSES[0]: [], CLASSES[1]: []}

    print("\n开始处理视频:")
    print("=" * 80)

    for i, p in enumerate(paths, 1):
        try:
            start_time = time.time()
            pred, probs, timing = infer_one_video(
                backend, is_onnx, p,
                FRAMES, SIZE_H, SIZE_W,
                NUM_CLIPS, TTA_FLIP,
                mean_t, std_t, mean_np, std_np,
                device
            )
            elapsed = time.time() - start_time

            # 更新总计时
            for key in ['total', 'read', 'preprocess', 'inference', 'postprocess']:
                total_timing[key] += timing[key]
            total_timing['frames'] += timing['frames']
            total_timing['videos'] += 1

            label = CLASSES[pred] if pred < len(CLASSES) else str(pred)
            prob_drip = float(probs[1]) if len(probs) > 1 else float('nan')
            is_drip = int(prob_drip >= THRESHOLD)

            # 统计：label 和 阈值判定
            names_by_label[label].append(str(p))
            names_by_threshold[is_drip].append(str(p))

            # 速度
            fps = timing['frames'] / timing['total'] if timing['total'] > 0 else 0
            inference_fps = timing['frames'] / timing['inference'] if timing['inference'] > 0 else 0

            # ===== 根据 LOG_MODE 控制输出 =====
            is_drip_video = (label == "drip")

            if LOG_MODE == "drip_only":
                # 只输出 drip 视频（带醒目提示）
                if is_drip_video:
                    print(f"\n{'='*80}")
                    print(f"🔴 [{i}/{len(paths)}] ⚠️ DRIP DETECTED ⚠️")
                    print(f"{'='*80}")
                    print(f"  文件: {p.name}")
                    print(f"  完整路径: {p}")
                    print(f"  预测: {label} | 概率: no_drip={probs[0]:.4f}, drip={probs[1]:.4f}")
                    print(f"  耗时: {format_time(timing['total'])} | FPS: {fps:.2f}")
                    print(f"{'='*80}\n")
                else:
                    # 正常视频只显示进度点，不刷屏
                    if i % 100 == 0 or i == len(paths):
                        print(f"[{i}/{len(paths)}] 处理中... (已发现 {len(names_by_threshold[1])} 个drip)")

            elif LOG_MODE == "compact":
                # 简洁模式：drip详细输出，正常视频简略输出
                if is_drip_video:
                    print(f"\n{'='*80}")
                    print(f"🔴 [{i}/{len(paths)}] ⚠️ DRIP DETECTED ⚠️")
                    print(f"{'='*80}")
                    print(f"  文件: {p.name}")
                    print(f"  完整路径: {p}")
                    print(f"  预测: {label} | 概率: no_drip={probs[0]:.4f}, drip={probs[1]:.4f}")
                    print(f"  耗时: {format_time(timing['total'])} | FPS: {fps:.2f}")
                    print(f"{'='*80}\n")
                else:
                    # 正常视频简略输出（每5个或最后一个才显示）
                    if i % 5 == 0 or i == len(paths):
                        print(f"[{i}/{len(paths)}] {p.name} | {label} (drip={probs[1]:.3f}) | {format_time(timing['total'])}")

            else:  # verbose 模式（原始逻辑）
                print(f"[{i}/{len(paths)}] {p.name}")
                print(f"  预测: {label} | 概率: no_drip={probs[0]:.4f}, drip={probs[1]:.4f} | drip@{THRESHOLD}: {is_drip}")
                print(f"  总耗时: {format_time(timing['total'])} | FPS: {fps:.2f} | 推理FPS: {inference_fps:.2f}")
                print(f"  帧数: {timing['frames']} | 分段数: {max(1, NUM_CLIPS)}")
                print(f"  读取: {format_time(timing['read'])} | 预处理: {format_time(timing['preprocess'])}")
                print(f"  推理: {format_time(timing['inference'])} | 后处理: {format_time(timing['postprocess'])}")
                print("-" * 80)

            rows.append((str(p), label, probs[0], probs[1] if len(probs) > 1 else float('nan'), is_drip))
        except Exception as e:
            print(f"[Error] {p}: {e}")
            rows.append((str(p), "ERROR", float('nan'), float('nan'), -1))

    # 打印总体统计
    if total_timing['videos'] > 0:
        avg_total = total_timing['total'] / total_timing['videos']
        avg_read = total_timing['read'] / total_timing['videos']
        avg_preprocess = total_timing['preprocess'] / total_timing['videos']
        avg_inference = total_timing['inference'] / total_timing['videos']
        avg_postprocess = total_timing['postprocess'] / total_timing['videos']

        total_fps = total_timing['frames'] / total_timing['total'] if total_timing['total'] > 0 else 0
        inference_fps = total_timing['frames'] / total_timing['inference'] if total_timing['inference'] > 0 else 0

        print("\n总体统计:")
        print("=" * 80)
        print(f"处理视频数: {total_timing['videos']} | 总帧数: {total_timing['frames']}")
        print(f"总耗时: {format_time(total_timing['total'])} | 平均FPS: {total_fps:.2f} | 平均推理FPS: {inference_fps:.2f}")
        print(f"平均耗时: {format_time(avg_total)}")
        print(f"平均读取: {format_time(avg_read)} | 平均预处理: {format_time(avg_preprocess)}")
        print(f"平均推理: {format_time(avg_inference)} | 平均后处理: {format_time(avg_postprocess)}")
        print(f"后端类型: {backend_type} | 设备: {device}")

        # =============== 按 group_XXXX 规则判定原始视频 drip 与否 ===============
        # key: (raw_dir_str, group_name)  value: [prob_drip, ...]
        from collections import defaultdict
        group_probs = defaultdict(list)

        for path_str, label, prob_no, prob_drip, is_drip in rows:
            # 跳过推理失败或无效结果
            if prob_drip != prob_drip:  # NaN 检查
                continue

            p = Path(path_str)
            group_dir = p.parent           # 如 .../group_0000
            raw_dir   = group_dir          # 每个原视频对应一个 strip 输出目录

            key = (str(raw_dir), group_dir.name)
            group_probs[key].append(float(prob_drip))

        # 根据规则对每个 group 判定：是否"满足 drip 条件"
        # 规则：
        #   1) 一个 group 下有 >=2 个 tile prob_drip >= THRESHOLD (0.45)
        #   2) 一个 group 下有任意 1 个 tile prob_drip >= HIGH_THRESHOLD (0.65)
        # 满足任意一条，就认为该 raw_dir 为 drip（只要有一个 group 满足即可）
        video_drip_by_rule = defaultdict(bool)

        for (raw_dir_str, group_name), probs_list in group_probs.items():
            threshold_count = sum(1 for v in probs_list if v >= THRESHOLD)
            has_high  = any(v >= HIGH_THRESHOLD for v in probs_list)

            if threshold_count >= 2 or has_high:
                video_drip_by_rule[raw_dir_str] = True

        if video_drip_by_rule:
            print("\n基于 group_XXXX 规则的原始视频 drip 判定：")
            print("=" * 80)
            drip_raw_list = []
            for raw_dir_str in sorted(video_drip_by_rule.keys()):
                status = "drip" if video_drip_by_rule[raw_dir_str] else "no_drip"
                print(f"  {raw_dir_str}  =>  {status}")
                if status == "drip":
                    drip_raw_list.append(raw_dir_str)

            # 这里就是你要的：统计最终被判定为 drip 的原始视频数量
            print("\n统计：满足 group 规则的 drip 原始视频数量 = "
                  f"{len(drip_raw_list)}")
        else:
            print("\n[Info] 没有任何原始视频满足 group 规则。")

        # ==================== 少数类统计和按路径分类输出 ====================
        # 类别数量统计（基于概率阈值）
        n_no_drip_by_th = len(names_by_threshold[0])
        n_drip_by_th = len(names_by_threshold[1])
        print("\n基于阈值的类别计数 (drip 概率>=阈值 视为 drip):")
        print(f"  no_drip  : {n_no_drip_by_th}")
        print(f"  drip     : {n_drip_by_th}")

        minority_key = None
        if n_no_drip_by_th < n_drip_by_th:
            minority_key = 0
        elif n_drip_by_th < n_no_drip_by_th:
            minority_key = 1

        if minority_key is None:
            print("\n两类数量相同，没有明显的少数类。")
        else:
            minority_name = "drip" if minority_key == 1 else "no_drip"
            minority_list = names_by_threshold[minority_key]

            print(f"\n少数类: {minority_name}  | 视频数: {len(minority_list)}")

            # 按父目录分组：{parent_dir: [(path_str, prob_drip), ...]}
            groups_by_dir = defaultdict(list)
            for full_path in minority_list:
                # 从 rows中查找对应的prob_drip
                prob_drip = next((prob for path, lbl, p_no, prob, is_s in rows if path == full_path), 0.0)
                p = Path(full_path)
                parent_dir = str(p.parent)
                groups_by_dir[parent_dir].append((full_path, prob_drip))

            print("\n少数类统计结果（按路径分类，每个路径下按置信度从大到小排序）:")
            print("=" * 80)

            total_minority_count = 0

            for parent_dir in sorted(groups_by_dir.keys()):
                videos = groups_by_dir[parent_dir]
                # 按置信度从大到小排序
                videos_sorted = sorted(videos, key=lambda x: x[1], reverse=True)

                print(f"\n目录: {parent_dir}")
                print(f"  视频数: {len(videos_sorted)}")
                print("-" * 80)

                for idx, (full_path, prob_drip) in enumerate(videos_sorted, 1):
                    p = Path(full_path)
                    total_minority_count += 1
                    print(f"  [{idx}] {p.name}  | drip置信度: {prob_drip:.4f}")
                print("-" * 80)

            print(f"\n少数类总计: {total_minority_count} 个视频")
            print("-" * 80)

            # 如果设置了 SAVE_MINORITY_TXT，则写入文件
            if SAVE_MINORITY_TXT:
                try:
                    with open(SAVE_MINORITY_TXT, "w", encoding="utf-8") as f:
                        f.write(f"少数类: {minority_name}\n")
                        f.write(f"视频数: {len(minority_list)}\n\n")
                        f.write("少数类统计结果（按路径分类，每个路径下按置信度从大到小排序）\n")
                        f.write("=" * 80 + "\n\n")

                        for parent_dir in sorted(groups_by_dir.keys()):
                            videos = groups_by_dir[parent_dir]
                            videos_sorted = sorted(videos, key=lambda x: x[1], reverse=True)

                            f.write(f"目录: {parent_dir}\n")
                            f.write(f"  视频数: {len(videos_sorted)}\n")
                            f.write("-" * 80 + "\n")

                            for idx, (full_path, prob_drip) in enumerate(videos_sorted, 1):
                                p = Path(full_path)
                                f.write(f"  [{idx}] {p.name}  | drip置信度: {prob_drip:.4f}\n")
                            f.write("-" * 80 + "\n\n")

                        f.write(f"\n少数类总计: {total_minority_count} 个视频\n")
                    print(f"少数类视频名单已保存到: {SAVE_MINORITY_TXT}")
                except Exception as e:
                    print(f"[Error] 保存少数类名单失败: {e}")

            # 如果启用了 MOVE_MINORITY_VIDEOS，移动少数类视频
            if MOVE_MINORITY_VIDEOS:
                try:
                    # 在输入路径下创建少数类文件夹
                    minority_folder = inp / f"minority_{minority_name}"
                    minority_folder.mkdir(parents=True, exist_ok=True)
                    print(f"\n开始移动少数类视频到: {minority_folder}")
                    print("=" * 80)

                    moved_count = 0
                    for full_path in minority_list:
                        src = Path(full_path)
                        # 避免文件名重复：加上父文件夹名作为前缀
                        parent_name = src.parent.name
                        new_name = f"{parent_name}_{src.name}"
                        dst = minority_folder / new_name

                        # 如果目标文件已存在，添加数字后缀
                        if dst.exists():
                            counter = 1
                            while dst.exists():
                                dst = minority_folder / f"{parent_name}_{src.stem}_{counter}{src.suffix}"
                                counter += 1

                        # 复制文件（保留原文件）
                        shutil.copy2(str(src), str(dst))
                        moved_count += 1
                        print(f"  [{moved_count}/{len(minority_list)}] {src.name} -> {new_name}")

                    print("=" * 80)
                    print(f"少数类视频移动完成: 共 {moved_count} 个视频")
                    print(f"目标文件夹: {minority_folder}")
                except Exception as e:
                    print(f"[Error] 移动少数类视频失败: {e}")
        print("=" * 80)

    if SAVE_CSV:
        Path(SAVE_CSV).parent.mkdir(parents=True, exist_ok=True)
        import csv
        with open(SAVE_CSV, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["path", "pred_label", "prob_no_drip", "prob_drip", f"drip@{THRESHOLD}"])
            w.writerows(rows)
        print(f"保存结果到: {SAVE_CSV}")

    if any(row[1] == "ERROR" for row in rows):
        raise RuntimeError("部分视频推理失败，请检查日志和 CSV 中的 ERROR 行")


def parse_cli():
    """命令行只覆盖配置，保持原有推理和阈值聚合逻辑。"""
    import argparse
    global MODEL_PATH, INPUT_PATH, SAVE_CSV, FRAMES, SIZE_H, SIZE_W
    global MEAN_TUP, STD_TUP, THRESHOLD, HIGH_THRESHOLD, TTA_FLIP
    global USE_CUDA_TORCH, USE_CUDA_ONNX, REQUIRE_CUDA
    parser = argparse.ArgumentParser(description="drip X3D 视频分类；输入应先完成预处理")
    parser.add_argument("--input", required=True, help="预处理后的视频或目录（递归读取）")
    parser.add_argument("--model", default=MODEL_PATH, help=".pt/.pth/.onnx 模型")
    parser.add_argument("--output", default="outputs/drip/results.csv", help="CSV 输出文件")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--threshold", type=float, default=THRESHOLD)
    parser.add_argument("--high-threshold", type=float, default=HIGH_THRESHOLD)
    parser.add_argument("--frames", type=int, default=FRAMES)
    parser.add_argument("--size-h", type=int, default=SIZE_H)
    parser.add_argument("--size-w", type=int, default=SIZE_W)
    parser.add_argument("--mean", default=",".join(map(str, MEAN_TUP)))
    parser.add_argument("--std", default=",".join(map(str, STD_TUP)))
    parser.add_argument("--no-tta", action="store_true", help="关闭水平翻转增强以加快推理")
    args = parser.parse_args()
    def triple(value):
        try:
            values = tuple(float(v) for v in value.split(","))
        except ValueError:
            parser.error("mean/std 必须是三个逗号分隔的数值")
        if len(values) != 3 or not all(np.isfinite(v) for v in values):
            parser.error("mean/std 必须是三个有限数值")
        return values
    MEAN_TUP, STD_TUP = triple(args.mean), triple(args.std)
    if min(STD_TUP) <= 0:
        parser.error("std 必须大于 0")
    if min(args.frames, args.size_h, args.size_w) <= 0:
        parser.error("帧数和尺寸必须大于 0")
    if not (0 <= args.threshold <= 1 and 0 <= args.high_threshold <= 1):
        parser.error("阈值必须在 [0,1] 范围")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA 不可用，请选择 --device cpu")
    REQUIRE_CUDA = args.device == "cuda"
    USE_CUDA_TORCH = args.device != "cpu" and torch.cuda.is_available()
    USE_CUDA_ONNX = USE_CUDA_TORCH
    MODEL_PATH = str(Path(args.model).expanduser().resolve())
    INPUT_PATH = str(Path(args.input).expanduser().resolve())
    SAVE_CSV = str(Path(args.output).expanduser().resolve())
    FRAMES, SIZE_H, SIZE_W = args.frames, args.size_h, args.size_w
    THRESHOLD, HIGH_THRESHOLD = args.threshold, args.high_threshold
    TTA_FLIP = not args.no_tta


if __name__ == "__main__":
    parse_cli()
    main_start = time.time()
    main()
    print(f"\n程序总运行时间: {format_time(time.time() - main_start)}")
