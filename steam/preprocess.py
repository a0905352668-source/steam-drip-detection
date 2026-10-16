# -*- coding: utf-8 -*-
"""
视频预处理（Windows 可参数化）v3 —— 不抽帧 + 全覆盖切片 + **自动紧致裁剪 + 背景净化**

新增能力（解决“裁剪小视频背景太杂”的问题）：
1) **自动紧致裁剪(AUTO_TIGHTEN)**：对每个 tile 的小视频，先做帧差热度→二值掩膜→最大连通域→
   取外接框并按比例扩边，二次裁剪得到“更贴近蒸汽”的区域；
2) **背景净化(BG_MODE)**：在紧致区域之外，把背景做模糊/置黑/置灰/均值填充，减少纹理干扰；
3) 全部参数在 CONFIG 顶部即可调整；保持“不抽帧”，并兼容全覆盖/Top-K 两种切片策略。

依赖：opencv-python, numpy（tqdm 可选）。
"""
from __future__ import annotations
import os, json, math, time
from pathlib import Path
from typing import List, Tuple, Dict
import numpy as np
import cv2

# 可选进度条
try:
    from tqdm import tqdm
except Exception:
    def tqdm(it, **kwargs):
        return it

# ====================== 配置（按需修改） ======================
CONFIG: Dict = {
    # 输入/输出
    "INPUT_PATH": "",  # 由 --input 指定
    "OUTPUT_DIR": "outputs/steam/preprocessed",

    # 帧读取
    "USE_NATIVE_FPS": True,    # 不抽帧，逐帧读取

    # 是否把整段分组（不影响是否抽帧，仅用于组织输出体量）
    "GROUP_TO_CLIPS": False,   # False=整段一组；True=按窗口分组
    "CLIP_LEN": 16,
    "WINDOW_STRIDE_FRAMES": 8,

    # 尺度与网格
    "RESIZE_SHORT_SIDE": 0,    # 0=不缩放；>0 则把短边缩放到该值（如 720/864/960）
    "TILE_SIZE": 512,
    "OVERLAP": 0.2,           # stride = tile*(1-overlap)

    # 切片覆盖策略
    "COVERAGE_MODE": "topk",    # "all"=全覆盖；"topk"=热度Top-K
    "TOPK_TILES": 5,
    "USE_DIFF_HEAT": True,

    # === 新增：自动紧致裁剪 & 背景净化 ===
    "AUTO_TIGHTEN": False,         # 开启自动二次裁剪（更贴蒸汽）
    "TIGHTEN_THR_MODE": "percentile",  # "percentile" | "otsu"
    "TIGHTEN_THR_PCT": 85,        # 百分位阈（当 MODE=percentile 时生效）
    "TIGHTEN_MIN_AREA": 300,      # 掩膜最小联通域面积（像素），小于则视为无效
    "TIGHTEN_MARGIN": 0.2,       # 外接框按比例扩边
    "BG_MODE": "blur",           # 背景处理："keep"|"blur"|"black"|"gray"|"mean"
    "BG_BLUR_K": 21,             # 背景模糊核（奇数）

    # 导出
    "EXPORT_MODE": "video",       # "json" | "frames" | "video"
    "RESIZE_OUT": 256,
    "JPEG_QUALITY": 95,
    "SAVE_VIS": True,
    "SKIP_EXISTING": True,

    "VIDEO_EXTS": [".mp4", ".avi", ".mov", ".mkv", ".flv", ".webm"],
}
# =============================================================

# -------------------- 工具函数 --------------------
def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def list_videos(input_path: Path, exts: List[str]) -> List[Path]:
    exts = {e.lower() for e in exts}
    if input_path.is_file():
        return [input_path]
    files: List[Path] = []
    for ext in exts:
        files += list(input_path.rglob(f"*{ext}"))
    return sorted(files)

# 读取**全部**帧（不抽帧）
def read_all_frames(path: Path) -> List[np.ndarray]:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"无法打开视频: {path}")
    frames: List[np.ndarray] = []
    while True:
        ret, bgr = cap.read()
        if not ret:
            break
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        frames.append(rgb)
    cap.release()
    if not frames:
        raise RuntimeError(f"读取到 0 帧: {path}")
    return frames

# 缩放
def resize_short_side(frames: List[np.ndarray], short_side: int) -> List[np.ndarray]:
    if short_side is None or short_side <= 0:
        return frames
    h, w = frames[0].shape[:2]
    if min(h, w) == short_side:
        return frames
    if h < w:
        new_h = short_side; new_w = int(round(w * short_side / h))
    else:
        new_w = short_side; new_h = int(round(h * short_side / w))
    return [cv2.resize(im, (new_w, new_h), interpolation=cv2.INTER_AREA) for im in frames]

# 全覆盖网格（锚定右/下缘，确保覆盖）
def make_fullcover_tiles(h: int, w: int, tile: int, overlap: float) -> List[Tuple[int,int,int,int]]:
    if tile <= 0:
        raise ValueError("TILE_SIZE 必须>0")
    stride = max(1, int(round(tile * (1.0 - overlap))))
    def starts(L: int) -> List[int]:
        if L <= tile:
            return [0]
        xs = list(range(0, L - tile + 1, stride))
        last = L - tile
        if xs[-1] != last:
            xs.append(last)
        return xs
    ys = starts(h)
    xs = starts(w)
    return [(y, x, min(h, y+tile), min(w, x+tile)) for y in ys for x in xs]

# 帧差热度（0~1）
def diff_heatmap(frames: List[np.ndarray]) -> np.ndarray:
    T = len(frames)
    gray = [cv2.cvtColor(im, cv2.COLOR_RGB2GRAY).astype(np.float32) for im in frames]
    diffs = [np.abs(gray[t]-gray[t-1]) for t in range(1, T)]
    if not diffs: heat = np.zeros_like(gray[0])
    else: heat = np.mean(diffs, axis=0)
    mn, mx = float(heat.min()), float(heat.max())
    if mx > mn: heat = (heat - mn) / (mx - mn)
    else: heat = np.zeros_like(heat)
    return cv2.GaussianBlur(heat.astype(np.float32), (5,5), 0)

# Top-K by heat
def select_topk_by_heat(heat: np.ndarray, tile: int, overlap: float, k: int) -> Tuple[List[Tuple[int,int,int,int]], List[float]]:
    h, w = heat.shape[:2]
    tiles = make_fullcover_tiles(h, w, tile, overlap)
    scores = [float(heat[y0:y1, x0:x1].mean()) for (y0,x0,y1,x1) in tiles]
    order = np.argsort(scores)[::-1][:min(k, len(scores))]
    return [tiles[i] for i in order], [scores[i] for i in order]

# 裁剪+缩放
def crop_resize_frame(im: np.ndarray, box: Tuple[int,int,int,int], out_size: int) -> np.ndarray:
    y0,x0,y1,x1 = box
    crop = im[y0:y1, x0:x1]
    return cv2.resize(crop, (out_size, out_size), interpolation=cv2.INTER_AREA)

# === 新增：从小视频生成运动掩膜 → 最大连通域 → 外接框（带扩边） ===
def motion_mask_from_clip(frames: List[np.ndarray], thr_mode: str, thr_pct: int,
                          min_area: int) -> Tuple[np.ndarray, Tuple[int,int,int,int] | None, np.ndarray]:
    """
    返回: (mask uint8 0/255, bbox or None, heat float32[0..1])
    """
    heat = diff_heatmap(frames)  # 0..1
    if thr_mode.lower() == "otsu":
        # Otsu on 0..255
        heat_255 = (heat * 255).astype(np.uint8)
        _, th = cv2.threshold(heat_255, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        mask = (heat_255 >= th).astype(np.uint8) * 255
    else:
        thr = np.percentile(heat, np.clip(thr_pct, 1, 99))
        mask = (heat >= float(thr)).astype(np.uint8) * 255
    # 形态学净化
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3,3), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5,5), np.uint8))
    # 最大连通域
    num, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if num <= 1:
        return mask, None, heat
    areas = stats[1:, cv2.CC_STAT_AREA]
    idx = 1 + int(np.argmax(areas))
    if stats[idx, cv2.CC_STAT_AREA] < max(1, int(min_area)):
        return mask, None, heat
    y, x, w, h = stats[idx, cv2.CC_STAT_TOP], stats[idx, cv2.CC_STAT_LEFT], stats[idx, cv2.CC_STAT_WIDTH], stats[idx, cv2.CC_STAT_HEIGHT]
    return mask, (y, x, y+h, x+w), heat

# 扩边 & 裁剪到边界
def expand_bbox(box: Tuple[int,int,int,int], H: int, W: int, margin: float) -> Tuple[int,int,int,int]:
    y0,x0,y1,x1 = box
    h = y1 - y0; w = x1 - x0
    dy = int(round(h * margin)); dx = int(round(w * margin))
    y0 = max(0, y0 - dy); x0 = max(0, x0 - dx)
    y1 = min(H, y1 + dy); x1 = min(W, x1 + dx)
    return y0,x0,y1,x1

# 背景净化（keep/blur/black/gray/mean）
def apply_bg_mode(frames: List[np.ndarray], mask: np.ndarray, mode: str, blur_k: int) -> List[np.ndarray]:
    if mode == "keep":
        return frames
    blur_k = int(blur_k) | 1
    inv = (255 - mask).astype(np.float32) / 255.0  # H×W
    inv3 = inv[..., None]
    out: List[np.ndarray] = []
    for im in frames:
        if mode == "black":
            bg = np.zeros_like(im)
        elif mode == "gray":
            bg = np.full_like(im, 128)
        elif mode == "mean":
            m = im.mean(axis=(0,1), keepdims=True).astype(im.dtype)
            bg = np.tile(m, (im.shape[0], im.shape[1], 1))
        else:  # blur
            bg = cv2.GaussianBlur(im, (blur_k, blur_k), 0)
        im2 = (im.astype(np.float32) * (1.0 - inv3) + bg.astype(np.float32) * inv3)
        out.append(np.clip(im2, 0, 255).astype(np.uint8))
    return out

# 可视化框
def draw_tiles_vis(im_rgb: np.ndarray, tiles: List[Tuple[int,int,int,int]], scores: List[float]|None=None) -> np.ndarray:
    vis = im_rgb[:, :, ::-1].copy()
    for i,(y0,x0,y1,x1) in enumerate(tiles):
        s = 0.0 if scores is None or i>=len(scores) else float(scores[i])
        color = (int(255*(1-s)), int(120*(1-s)), int(255*s))
        cv2.rectangle(vis, (x0,y0), (x1,y1), color, 2)
        if scores is not None:
            cv2.putText(vis, f"{s:.2f}", (x0+4, y0+18), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    return vis

# 保存函数
def save_frames_dir(out_dir: Path, frames: List[np.ndarray], jpeg_quality: int):
    ensure_dir(out_dir)
    for i, im in enumerate(frames):
        cv2.imwrite(str(out_dir / f"frame_{i:06d}.jpg"), im[:, :, ::-1], [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_quality)])

def save_video(out_path: Path, frames: List[np.ndarray], fps: float|int):
    ensure_dir(out_path.parent)
    h, w = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    vw = cv2.VideoWriter(str(out_path), fourcc, float(fps), (w, h))
    if not vw.isOpened():
        raise RuntimeError(f"VideoWriter 打开失败: {out_path}")
    for im in frames:
        vw.write(im[:, :, ::-1])
    vw.release()

def uniform_sample_indices(n: int, k: int) -> List[int]:
    if n <= 0: return [0]*k
    if k <= 1: return [min(n-1, 0)]
    xs = np.linspace(0, max(0, n-1), num=k)
    return [int(round(x)) for x in xs]

# 组 clip（仅在 GROUP_TO_CLIPS=True 时用）
def make_clips(frames: List[np.ndarray], clip_len: int, stride: int) -> List[List[np.ndarray]]:
    if clip_len <= 0:
        return [frames]
    clips: List[List[np.ndarray]] = []
    T = len(frames)
    for s in range(0, max(1, T-clip_len+1), max(1, stride)):
        seg = frames[s:s+clip_len]
        if len(seg) < clip_len:
            seg = seg + [seg[-1]]*(clip_len-len(seg))
        clips.append(seg)
    return clips if clips else [frames]

# -------------------- 主处理逻辑 --------------------

def process_video(path: Path, cfg: Dict):
    rel = path.stem
    out_dir_video = Path(cfg["OUTPUT_DIR"]) / rel
    ensure_dir(out_dir_video)

    # 读取全部帧
    frames = read_all_frames(path)

    # 统一缩放（可选）
    if cfg["RESIZE_SHORT_SIDE"] and cfg["RESIZE_SHORT_SIDE"] > 0:
        frames = resize_short_side(frames, cfg["RESIZE_SHORT_SIDE"])

    # 选择 tiles（全覆盖 or Top-K）
    h, w = frames[0].shape[:2]
    if cfg["COVERAGE_MODE"].lower() == "topk":
        heat = diff_heatmap(frames) if cfg["USE_DIFF_HEAT"] else np.zeros((h,w), np.float32)
        sel_tiles, sel_scores = select_topk_by_heat(heat, cfg["TILE_SIZE"], cfg["OVERLAP"], cfg["TOPK_TILES"])
    else:
        sel_tiles = make_fullcover_tiles(h, w, cfg["TILE_SIZE"], cfg["OVERLAP"])
        sel_scores = [0.0]*len(sel_tiles)

    # 可视化（首帧）
    if cfg["SAVE_VIS"]:
        vis = draw_tiles_vis(frames[0], sel_tiles, sel_scores if cfg["COVERAGE_MODE"].lower()=="topk" else None)
        cv2.imwrite(str(out_dir_video / "vis.jpg"), vis)

    # 保存元数据（tile 坐标）
    meta = {
        "video": str(path),
        "size_hw": [int(h), int(w)],
        "tile_size": int(cfg["TILE_SIZE"]),
        "overlap": float(cfg["OVERLAP"]),
        "coverage_mode": cfg["COVERAGE_MODE"],
        "tiles": [[int(a) for a in b] for b in sel_tiles],
        "scores": [float(s) for s in sel_scores],
    }
    with open(out_dir_video / "tiles_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    # 若只要坐标与可视化
    if cfg["EXPORT_MODE"].lower() == "json":
        return

    # 组织分组
    groups: List[List[np.ndarray]]
    if cfg["GROUP_TO_CLIPS"]:
        groups = make_clips(frames, cfg["CLIP_LEN"], cfg["WINDOW_STRIDE_FRAMES"])
    else:
        groups = [frames]

    # 取源FPS用于 mp4 写出
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    cap.release()

    # 遍历分组与 tiles
    for gi, g in enumerate(tqdm(groups, desc=rel, ncols=100)):
        base = out_dir_video / ("group_%04d" % gi)
        ensure_dir(base)
        for ti, box in enumerate(sel_tiles):
            # 初次裁剪到 tile
            y0,x0,y1,x1 = box
            crops = [im[y0:y1, x0:x1] for im in g]

            # === 自动紧致裁剪 + 背景净化 ===
            if cfg["AUTO_TIGHTEN"]:
                mask, bbox, heat = motion_mask_from_clip(
                    crops,
                    thr_mode=cfg["TIGHTEN_THR_MODE"],
                    thr_pct=cfg["TIGHTEN_THR_PCT"],
                    min_area=cfg["TIGHTEN_MIN_AREA"],
                )
                if bbox is not None:
                    Ht, Wt = crops[0].shape[:2]
                    y0b,x0b,y1b,x1b = expand_bbox(bbox, Ht, Wt, cfg["TIGHTEN_MARGIN"])
                    crops = [im[y0b:y1b, x0b:x1b] for im in crops]
                    # 同步缩放掩膜到输出尺寸以便背景处理
                    mask_resized = cv2.resize(mask[y0b:y1b, x0b:x1b], (cfg["RESIZE_OUT"], cfg["RESIZE_OUT"]), interpolation=cv2.INTER_NEAREST)
                else:
                    mask_resized = cv2.resize(mask, (cfg["RESIZE_OUT"], cfg["RESIZE_OUT"]), interpolation=cv2.INTER_NEAREST)
                # 背景净化
                crops = [cv2.resize(im, (cfg["RESIZE_OUT"], cfg["RESIZE_OUT"]), interpolation=cv2.INTER_AREA) for im in crops]
                crops = apply_bg_mode(crops, mask_resized, cfg["BG_MODE"], cfg["BG_BLUR_K"])
            else:
                # 仅缩放到输出尺寸
                crops = [cv2.resize(im, (cfg["RESIZE_OUT"], cfg["RESIZE_OUT"]), interpolation=cv2.INTER_AREA) for im in crops]

            # 写出
            exp = cfg["EXPORT_MODE"].lower()
            if exp == "frames":
                out_dir = base / f"tile_{ti:03d}"
                save_frames_dir(out_dir, crops, cfg["JPEG_QUALITY"])
            elif exp == "video":
                out_mp4 = base / f"tile_{ti:03d}.mp4"
                if cfg["SKIP_EXISTING"] and out_mp4.exists():
                    continue
                save_video(out_mp4, crops, fps)

# -------------------- 入口 --------------------
def main():
    import argparse
    parser = argparse.ArgumentParser(description="水蒸气视频切片预处理")
    parser.add_argument("--input", required=True, help="原始视频或目录")
    parser.add_argument("--output", required=True)
    parser.add_argument("--topk", type=int, default=CONFIG["TOPK_TILES"])
    parser.add_argument("--tile-size", type=int, default=CONFIG["TILE_SIZE"])
    parser.add_argument("--overlap", type=float, default=CONFIG["OVERLAP"])
    parser.add_argument("--coverage", choices=("topk", "all"), default=CONFIG["COVERAGE_MODE"])
    parser.add_argument("--tighten", action="store_true", help="启用紧致裁剪与背景净化")
    parser.add_argument("--bg-mode", choices=("keep", "blur", "black", "gray", "mean"), default=CONFIG["BG_MODE"])
    args = parser.parse_args()
    if args.topk <= 0 or args.tile_size <= 0 or not 0 <= args.overlap < 1:
        parser.error("topk/tile-size 必须大于 0，overlap 必须在 [0,1)")
    cfg = dict(CONFIG)
    cfg.update(INPUT_PATH=str(Path(args.input).expanduser().resolve()),
               OUTPUT_DIR=str(Path(args.output).expanduser().resolve()),
               TOPK_TILES=args.topk, TILE_SIZE=args.tile_size, OVERLAP=args.overlap,
               COVERAGE_MODE=args.coverage, AUTO_TIGHTEN=args.tighten, BG_MODE=args.bg_mode)
    in_path = Path(cfg["INPUT_PATH"])
    videos = list_videos(in_path, cfg["VIDEO_EXTS"])
    if not videos:
        parser.error(f"未找到视频: {in_path}")
    stems = [v.stem for v in videos]
    if len(stems) != len(set(stems)):
        parser.error("输入含有同名视频，输出目录会冲突；请先重命名或分批处理")
    ensure_dir(Path(cfg["OUTPUT_DIR"]))
    t0 = time.time()
    for video in videos:
        process_video(video, cfg)
    print(f"完成，用时 {time.time()-t0:.1f}s；输出目录: {cfg['OUTPUT_DIR']}")


if __name__ == "__main__":
    main()
