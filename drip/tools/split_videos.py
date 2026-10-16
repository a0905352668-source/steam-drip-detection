import os
import subprocess
from pathlib import Path
from tqdm import tqdm
import argparse

# ======================== 配置区 ========================
# 支持的视频扩展名（包含大小写）
VIDEO_EXTENSIONS = {
    '.mp4', '.MP4', '.avi', '.AVI', '.mkv', '.MKV',
    '.mov', '.MOV', '.wmv', '.WMV', '.flv', '.FLV',
    '.webm', '.WEBM', '.m4v', '.M4V', '.mpg', '.MPG',
    '.mpeg', '.MPEG', '.3gp', '.3GP', '.ts', '.TS'
}

# 默认切割时长（秒）
DEFAULT_DURATION = 5
# ========================================================


def get_video_duration(video_path):
    """
    获取视频时长（秒）

    参数:
        video_path: 视频文件路径

    返回:
        视频时长（秒），失败返回 None
    """
    try:
        cmd = [
            'ffprobe', '-v', 'error',
            '-show_entries', 'format=duration',
            '-of', 'default=noprint_wrappers=1:nokey=1',
            str(video_path)
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)

        if result.returncode != 0:
            print(f"⚠️  ffprobe 失败 {video_path.name}: {result.stderr[:200]}")
            return None

        duration = float(result.stdout.strip())
        return duration
    except subprocess.TimeoutExpired:
        print(f"⚠️  获取时长超时 {video_path.name}")
        return None
    except Exception as e:
        print(f"⚠️  获取视频时长失败 {video_path.name}: {e}")
        return None


def split_video(video_path, output_dir, duration=5):
    """
    将单个视频按固定时长切割为多个小视频

    参数:
        video_path: 源视频文件路径
        output_dir: 输出文件夹路径（该视频的所有小视频保存到这里）
        duration: 每个小视频的时长（秒）

    返回:
        (成功数量, 失败数量)
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 获取视频时长
    total_duration = get_video_duration(video_path)
    if total_duration is None:
        return 0, 1

    # 计算需要切割的段数
    num_segments = int(total_duration / duration)
    if total_duration % duration > 0:
        num_segments += 1

    success_count = 0
    fail_count = 0

    # 逐段切割
    for i in range(num_segments):
        start_time = i * duration
        output_file = output_dir / f"{video_path.stem}_part_{i+1:03d}{video_path.suffix}"

        # 构建 ffmpeg 命令 - 使用重新编码模式更稳定
        cmd = [
            'ffmpeg', '-y',
            '-ss', str(start_time),
            '-i', str(video_path),
            '-t', str(duration),
            '-c:v', 'libx264',
            '-preset', 'fast',
            '-crf', '23',
            '-c:a', 'aac',
            '-b:a', '128k',
            '-avoid_negative_ts', 'make_zero',
            str(output_file)
        ]

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300  # 5分钟超时
            )

            # 检查返回码
            if result.returncode != 0:
                print(f"\n⚠️  切割失败 {video_path.name} 第 {i+1} 段:")
                print(f"    错误: {result.stderr[-200:]}")  # 只显示最后200字符
                fail_count += 1
                continue

            # 验证输出文件是否存在且大小大于0
            if output_file.exists() and output_file.stat().st_size > 0:
                success_count += 1
            else:
                print(f"\n⚠️  输出文件为空 {video_path.name} 第 {i+1} 段")
                fail_count += 1
        except subprocess.TimeoutExpired:
            print(f"\n⚠️  切割超时 {video_path.name} 第 {i+1} 段")
            fail_count += 1
        except Exception as e:
            print(f"\n⚠️  切割失败 {video_path.name} 第 {i+1} 段: {e}")
            fail_count += 1

    return success_count, fail_count


def split_all_videos(src_dir, output_dir, duration=5, overwrite=False):
    """
    批量切割文件夹下的所有视频

    参数:
        src_dir: 源视频文件夹路径
        output_dir: 输出根目录
        duration: 每个小视频的时长（秒）
        overwrite: 是否覆盖已存在的输出文件夹
    """
    src_dir = Path(src_dir).resolve()
    output_dir = Path(output_dir).resolve()

    # 校验源文件夹
    if not src_dir.exists():
        print(f"❌ 源文件夹不存在: {src_dir}")
        return

    if not src_dir.is_dir():
        print(f"❌ 源路径不是文件夹: {src_dir}")
        return

    # 查找所有视频文件（递归查找子目录）
    video_files = []
    for ext in VIDEO_EXTENSIONS:
        video_files.extend(list(src_dir.rglob(f"*{ext}")))

    # 去重并排序
    video_files = sorted(list(set(video_files)))

    if len(video_files) == 0:
        print(f"❌ 未找到任何视频文件！")
        return

    print(f"{'='*60}")
    print(f"源文件夹: {src_dir}")
    print(f"输出目录: {output_dir}")
    print(f"切割时长: {duration} 秒/段")
    print(f"视频数量: {len(video_files)}")
    print(f"{'='*60}\n")

    # 创建输出根目录
    output_dir.mkdir(parents=True, exist_ok=True)

    total_success = 0
    total_fail = 0
    total_segments = 0

    # 处理每个视频
    for video_file in tqdm(video_files, desc="处理视频"):
        # 为每个视频创建独立的输出文件夹
        video_output_dir = output_dir / video_file.stem

        # 如果文件夹已存在
        if video_output_dir.exists():
            if overwrite:
                # 删除已存在的文件夹
                import shutil
                shutil.rmtree(video_output_dir)
            else:
                print(f"\n⚠️  跳过已存在的文件夹: {video_output_dir.name}")
                print(f"    如需覆盖，请使用 --overwrite 参数")
                continue

        # 获取视频时长
        duration_sec = get_video_duration(video_file)
        if duration_sec is None:
            total_fail += 1
            continue

        num_segments = int(duration_sec / duration)
        if duration_sec % duration > 0:
            num_segments += 1

        print(f"\n📹 {video_file.name} (时长: {duration_sec:.1f}秒, 将切割为 {num_segments} 段)")

        # 切割视频
        success, fail = split_video(video_file, video_output_dir, duration)

        total_success += success
        total_fail += fail
        total_segments += (success + fail)

        if fail == 0:
            print(f"✅ 成功切割 {success} 段 → {video_output_dir}")
        else:
            print(f"⚠️  完成: 成功 {success} 段, 失败 {fail} 段 → {video_output_dir}")

    # 汇总统计
    print(f"\n{'='*60}")
    print(f"✅ 全部完成！")
    print(f"📊 处理视频: {len(video_files)} 个")
    print(f"📊 总片段数: {total_segments} 段")
    print(f"📊 成功: {total_success} 段")
    if total_fail > 0:
        print(f"⚠️  失败: {total_fail} 段")
    print(f"📁 输出目录: {output_dir}")
    print(f"{'='*60}")


def check_ffmpeg():
    """检查 ffmpeg 是否安装"""
    try:
        result = subprocess.run(
            ['ffmpeg', '-version'],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0:
            return True
    except Exception:
        pass
    return False


# ======================== 主入口 ========================
if __name__ == '__main__':
    # 检查 ffmpeg
    if not check_ffmpeg():
        print("❌ 错误: 未找到 ffmpeg！")
        print("\n请先安装 ffmpeg:")
        print("  Ubuntu/Debian: sudo apt install ffmpeg")
        print("  CentOS/RHEL:   sudo yum install ffmpeg")
        print("  macOS:         brew install ffmpeg")
        print("  Windows:       从 https://ffmpeg.org/download.html 下载并添加到PATH")
        exit(1)

    parser = argparse.ArgumentParser(
        description='将大视频切割为多个固定时长的小视频'
    )
    parser.add_argument('src', help='源视频文件夹路径')
    parser.add_argument('dst', help='输出目录路径')
    parser.add_argument('--duration', type=float, default=DEFAULT_DURATION,
                        help=f'每个小视频的时长（秒，默认 {DEFAULT_DURATION}）')
    parser.add_argument('--overwrite', action='store_true',
                        help='覆盖已存在的输出文件夹')

    args = parser.parse_args()

    split_all_videos(
        args.src,
        args.dst,
        duration=args.duration,
        overwrite=args.overwrite
    )

# ======================== 使用示例 ========================
# 基本用法（5秒一段）
# python split_videos.py /path/to/videos /path/to/output
#
# 自定义时长（10秒一段）
# python split_videos.py /path/to/videos /path/to/output --duration 10
#
# 覆盖已存在的文件夹
# python split_videos.py /path/to/videos /path/to/output --overwrite
#
# 输出结构：
# output/
# ├── video1/
# │   ├── video1_part_001.mp4
# │   ├── video1_part_002.mp4
# │   └── video1_part_003.mp4
# ├── video2/
# │   ├── video2_part_001.mp4
# │   └── video2_part_002.mp4
# └── video3/
#     └── ...
