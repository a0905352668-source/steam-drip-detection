import os
import shutil
import random
import argparse
from pathlib import Path
from tqdm import tqdm

# ======================== 配置区 ========================
# 支持的视频扩展名
VIDEO_EXTENSIONS = {
    '.mp4', '.avi', '.mkv', '.mov', '.wmv', '.flv',
    '.webm', '.m4v', '.mpg', '.mpeg', '.3gp', '.ts',
    '.rmvb', '.rm', '.vob', '.ogv', '.m2ts', '.mts'
}
# ========================================================


def split_videos(src_dir, output_dir, val_ratio=0.2, seed=42, copy_mode=True):
    """
    将视频数据集随机划分为训练集和验证集

    参数:
        src_dir: 源视频文件夹路径（会递归查找所有视频）
        output_dir: 输出目录（会自动创建 train 和 val 子文件夹）
        val_ratio: 验证集比例（默认 0.2，即 20%）
        seed: 随机种子（保证每次划分结果可复现）
        copy_mode: True=复制模式, False=移动模式（默认复制）
    """
    src_dir = Path(src_dir).resolve()
    output_dir = Path(output_dir).resolve()

    # 校验源文件夹
    if not src_dir.exists():
        print(f"❌ 源文件夹不存在: {src_dir}")
        return

    # 创建输出目录结构
    train_dir = output_dir / "train"
    val_dir = output_dir / "val"
    train_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)

    action = "复制" if copy_mode else "移动"
    print(f"{'='*60}")
    print(f"操作模式: {action}")
    print(f"源文件夹: {src_dir}")
    print(f"输出目录: {output_dir}")
    print(f"验证集比例: {val_ratio*100:.0f}%")
    print(f"{'='*60}\n")

    # 递归查找所有视频文件
    video_files = []
    for ext in VIDEO_EXTENSIONS:
        video_files.extend(list(src_dir.rglob(f"*{ext}")))

    # 去重并排序
    video_files = sorted(list(set(video_files)))
    total_count = len(video_files)

    if total_count == 0:
        print("❌ 错误：未找到任何视频文件！")
        return

    # 随机打乱
    random.seed(seed)
    random.shuffle(video_files)

    # 计算划分点
    val_count = int(total_count * val_ratio)
    train_count = total_count - val_count

    train_files = video_files[:train_count]
    val_files = video_files[train_count:]

    print(f"📊 视频总量: {total_count}")
    print(f"🔥 训练集: {len(train_files)} ({(1-val_ratio)*100:.0f}%)")
    print(f"🧪 验证集: {len(val_files)} ({val_ratio*100:.0f}%)")
    print("-" * 60)

    # 复制/移动文件函数
    def process_files(file_list, split_type, target_dir):
        print(f"🚀 正在生成 {split_type} 集...")
        success_count = 0
        skip_count = 0

        for src_file in tqdm(file_list, desc=f"{split_type}"):
            # 构建目标文件名：使用相对路径作为前缀避免重名
            rel_path = src_file.relative_to(src_dir)
            # 将路径分隔符替换为下划线
            folder_prefix = str(rel_path.parent).replace(os.sep, '_')

            if folder_prefix and folder_prefix != '.':
                new_filename = f"{folder_prefix}_{src_file.name}"
            else:
                new_filename = src_file.name

            dst_file = target_dir / new_filename

            # 处理重名
            if dst_file.exists():
                counter = 1
                stem = dst_file.stem
                suffix = dst_file.suffix
                while dst_file.exists():
                    dst_file = target_dir / f"{stem}_{counter}{suffix}"
                    counter += 1

            # 执行复制或移动
            try:
                if copy_mode:
                    shutil.copy2(str(src_file), str(dst_file))
                else:
                    shutil.move(str(src_file), str(dst_file))
                success_count += 1
            except Exception as e:
                print(f"\n⚠️ 失败: {src_file.name} - {e}")
                skip_count += 1

        print(f"✅ {split_type} 完成: 成功 {success_count}, 失败 {skip_count}")
        return success_count, skip_count

    # 处理训练集和验证集
    train_success, train_skip = process_files(train_files, "训练集", train_dir)
    print()
    val_success, val_skip = process_files(val_files, "验证集", val_dir)

    # 汇总统计
    total_success = train_success + val_success
    total_skip = train_skip + val_skip

    print(f"\n{'='*60}")
    print(f"✅ 划分完成！")
    print(f"📁 训练集路径: {train_dir}")
    print(f"📁 验证集路径: {val_dir}")
    print(f"📊 成功处理: {total_success} 个视频")
    if total_skip > 0:
        print(f"⚠️  失败/跳过: {total_skip} 个视频")
    print(f"{'='*60}")


def split_by_classes(src_dir, output_dir, val_ratio=0.2, seed=42, copy_mode=True):
    """
    按类别划分：每个类别文件夹内的视频按比例划分为 train 和 val
    输出结构：
        output/
        ├── train/
        │   ├── class1/
        │   └── class2/
        └── val/
            ├── class1/
            └── class2/

    参数:
        src_dir: 源文件夹（包含多个类别子文件夹，每个子文件夹是一个类别）
        output_dir: 输出目录
        val_ratio: 验证集比例
        seed: 随机种子
        copy_mode: True=复制, False=移动
    """
    src_dir = Path(src_dir).resolve()
    output_dir = Path(output_dir).resolve()

    if not src_dir.exists():
        print(f"❌ 源文件夹不存在: {src_dir}")
        return

    # 创建输出目录结构
    train_dir = output_dir / "train"
    val_dir = output_dir / "val"

    # 获取所有一级子文件夹（类别）
    class_folders = [d for d in src_dir.iterdir() if d.is_dir()]

    if not class_folders:
        print("❌ 错误：源文件夹下没有子文件夹（类别）！")
        return

    # 为每个类别创建 train 和 val 子文件夹
    for class_folder in class_folders:
        (train_dir / class_folder.name).mkdir(parents=True, exist_ok=True)
        (val_dir / class_folder.name).mkdir(parents=True, exist_ok=True)

    action = "复制" if copy_mode else "移动"
    print(f"{'='*60}")
    print(f"操作模式: {action} (按类别划分)")
    print(f"源文件夹: {src_dir}")
    print(f"输出目录: {output_dir}")
    print(f"验证集比例: {val_ratio*100:.0f}%")
    print(f"找到 {len(class_folders)} 个类别: {[f.name for f in class_folders]}")
    print(f"{'='*60}\n")

    total_train = 0
    total_val = 0

    # 处理每个类别
    for class_folder in tqdm(class_folders, desc="处理类别"):
        # 查找类别文件夹内的所有视频
        video_files = []
        for ext in VIDEO_EXTENSIONS:
            video_files.extend(list(class_folder.rglob(f"*{ext}")))

        video_files = sorted(list(set(video_files)))

        if len(video_files) == 0:
            print(f"⚠️  类别 '{class_folder.name}' 中没有视频，跳过")
            continue

        # 随机打乱
        random.seed(seed)
        random.shuffle(video_files)

        # 划分
        val_count = int(len(video_files) * val_ratio)
        train_count = len(video_files) - val_count

        train_files = video_files[:train_count]
        val_files = video_files[train_count:]

        # 目标文件夹
        class_train_dir = train_dir / class_folder.name
        class_val_dir = val_dir / class_folder.name

        # 处理训练集
        train_success = 0
        for src_file in train_files:
            dst_file = class_train_dir / src_file.name
            # 处理重名
            if dst_file.exists():
                counter = 1
                stem = dst_file.stem
                suffix = dst_file.suffix
                while dst_file.exists():
                    dst_file = class_train_dir / f"{stem}_{counter}{suffix}"
                    counter += 1

            try:
                if copy_mode:
                    shutil.copy2(str(src_file), str(dst_file))
                else:
                    shutil.move(str(src_file), str(dst_file))
                train_success += 1
            except Exception as e:
                print(f"\n⚠️  失败: {src_file.name} - {e}")

        # 处理验证集
        val_success = 0
        for src_file in val_files:
            dst_file = class_val_dir / src_file.name
            # 处理重名
            if dst_file.exists():
                counter = 1
                stem = dst_file.stem
                suffix = dst_file.suffix
                while dst_file.exists():
                    dst_file = class_val_dir / f"{stem}_{counter}{suffix}"
                    counter += 1

            try:
                if copy_mode:
                    shutil.copy2(str(src_file), str(dst_file))
                else:
                    shutil.move(str(src_file), str(dst_file))
                val_success += 1
            except Exception as e:
                print(f"\n⚠️  失败: {src_file.name} - {e}")

        total_train += train_success
        total_val += val_success

        print(f"\n✅ 类别 '{class_folder.name}': 训练集 {train_success}, 验证集 {val_success}")

    print(f"\n{'='*60}")
    print(f"✅ 划分完成！")
    print(f"📁 训练集路径: {train_dir}")
    print(f"📁 验证集路径: {val_dir}")
    print(f"📊 训练集总数: {total_train} 个视频")
    print(f"📊 验证集总数: {total_val} 个视频")
    print(f"{'='*60}")

    # 显示每个类别的统计
    print(f"\n📈 各类别详细统计:")
    print(f"{'-'*60}")
    for class_folder in class_folders:
        train_count = len(list((train_dir / class_folder.name).glob("*"))) if (train_dir / class_folder.name).exists() else 0
        val_count = len(list((val_dir / class_folder.name).glob("*"))) if (val_dir / class_folder.name).exists() else 0
        print(f"  {class_folder.name:20s} - 训练集: {train_count:4d}, 验证集: {val_count:4d}, 总计: {train_count + val_count:4d}")
    print(f"{'-'*60}")


# ======================== 主入口 ========================
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='将视频数据集划分为训练集和验证集'
    )
    parser.add_argument('src', help='源视频文件夹路径')
    parser.add_argument('dst', help='输出目录路径')
    parser.add_argument('--val_ratio', type=float, default=0.2,
                        help='验证集比例（默认 0.2，即 20%%）')
    parser.add_argument('--seed', type=int, default=42,
                        help='随机种子（默认 42）')
    parser.add_argument('--copy', action='store_true', default=True,
                        help='使用复制模式（默认）')
    parser.add_argument('--move', action='store_true',
                        help='使用移动模式')
    parser.add_argument('--by_class', action='store_true',
                        help='按类别划分（每个类别文件夹内独立划分 train/val，推荐）')

    args = parser.parse_args()

    copy_mode = not args.move

    if args.by_class:
        split_by_classes(
            args.src, args.dst,
            val_ratio=args.val_ratio,
            seed=args.seed,
            copy_mode=copy_mode
        )
    else:
        split_videos(
            args.src, args.dst,
            val_ratio=args.val_ratio,
            seed=args.seed,
            copy_mode=copy_mode
        )

# ======================== 使用示例 ========================
# 方式1：将所有视频混合后随机划分
# python split_train_val.py /path/to/videos /path/to/output --val_ratio 0.2
#
# 方式2：按类别划分（推荐！每个类别独立划分 train/val）
# 假设源文件夹结构：
#   videos/
#   ├── drip/
#   │   ├── video1.mp4
#   │   └── video2.mp4
#   └── no_drip/
#       ├── video3.mp4
#       └── video4.mp4
#
# 命令：
# python split_train_val.py /path/to/videos /path/to/output --by_class --val_ratio 0.2
#
# 输出结构：
#   output/
#   ├── train/
#   │   ├── drip/
#   │   │   ├── video1.mp4
#   │   │   └── ...
#   │   └── no_drip/
#   │       ├── video3.mp4
#   │       └── ...
#   └── val/
#       ├── drip/
#       │   └── ...
#       └── no_drip/
#           └── ...
#
# 方式3：移动模式（不保留原文件）
# python split_train_val.py /path/to/videos /path/to/output --by_class --move
#
# 方式4：自定义随机种子和验证集比例
# python split_train_val.py /path/to/videos /path/to/output --by_class --val_ratio 0.3 --seed 123
