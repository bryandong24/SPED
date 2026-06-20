#!/usr/bin/env python3
"""
overlay_keys.py — 在视频每帧上绘制 WASD 和 KIJL 键盘指示器。

拓扑：
  WASD: W居上; A-S-D 底排（倒T）
  KIJL: K(↑)居上; L(←)-I(↓)-J(→) 底排（倒T）

用法：
  # 全程按住某键
  python overlay_keys.py -i IN.mp4 -o OUT.mp4 --pressed A

  # 按键序列（逐段变化）: 键名:帧数,键名:帧数,...
  python overlay_keys.py -i IN.mp4 -o OUT.mp4 --sequence W:25,A:28,W:24

  # 序列中支持多键同时按下，用+连接
  python overlay_keys.py -i IN.mp4 -o OUT.mp4 --sequence W+K:10,A+J:20

  # 无按键高亮
  python overlay_keys.py -i IN.mp4 -o OUT.mp4

依赖：ffmpeg, ffprobe, numpy, Pillow
"""

import argparse
import json
import os
import subprocess
import sys
import numpy as np
from PIL import Image, ImageDraw, ImageFont


# ─── 视觉参数 ───────────────────────────────────────────────
KEY_SIZE = 36          # 单键方块边长(px)
RADIUS = 8            # 圆角半径
GAP = 4               # 同组键间距
MARGIN = 20           # 距视频边缘距离
ALPHA_NORMAL = 0.45   # 未按下底色透明度
ALPHA_PRESSED = 0.85  # 按下底色透明度

COLOR_NORMAL = (80, 80, 80)       # 未按下底色 (灰)
COLOR_PRESSED = (255, 140, 0)     # 按下底色 (橙)
COLOR_TEXT = (255, 255, 255)      # 文字颜色 (白)

# ─── KIJL 映射（K=上, I=下, J=右, L=左）─────────────────────
KIJL_KEY_TO_ARROW = {'K': '↓', 'I': '↑', 'J': '←', 'L': '→'}


def load_font(size):
    """尝试加载系统字体，fallback 到默认。"""
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
        "/usr/share/fonts/truetype/noto/NotoSansMono-Bold.ttf",
    ]
    for path in candidates:
        if os.path.exists(path):
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def draw_key(draw, x, y, label, font, pressed=False):
    """绘制单个按键（圆角方块 + 居中文字）。"""
    if pressed:
        bg = COLOR_PRESSED + (int(255 * ALPHA_PRESSED),)
    else:
        bg = COLOR_NORMAL + (int(255 * ALPHA_NORMAL),)

    draw.rounded_rectangle(
        [x, y, x + KEY_SIZE, y + KEY_SIZE],
        radius=RADIUS,
        fill=bg,
    )

    bbox = draw.textbbox((0, 0), label, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    tx = x + (KEY_SIZE - tw) // 2
    ty = y + (KEY_SIZE - th) // 2
    draw.text((tx, ty), label, fill=COLOR_TEXT + (255,), font=font)


def create_group_overlay(keys, pressed_keys, font, arrow_font, is_arrow_group=False):
    """
    创建一组键的覆盖层 (RGBA)。

    拓扑（倒T）：
      keys[0] 居上居中
      keys[1]-keys[2]-keys[3] 底排从左到右

    Args:
        keys: 4个键名 [top, bottom_left, bottom_center, bottom_right]
        pressed_keys: 被按下的键名集合（大写字母）
        font: 字母字体
        arrow_font: 箭头字体
        is_arrow_group: 是否为箭头组（KIJL）
    """
    total_w = KEY_SIZE * 3 + GAP * 2
    total_h = KEY_SIZE * 2 + GAP

    img = Image.new('RGBA', (total_w, total_h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    # 顶部键（居中于底排中间键上方）
    top_x = KEY_SIZE + GAP
    top_y = 0
    key_name = keys[0]
    label = KIJL_KEY_TO_ARROW[key_name] if is_arrow_group else key_name
    use_font = arrow_font if is_arrow_group else font
    draw_key(draw, top_x, top_y, label, use_font, pressed=(key_name in pressed_keys))

    # 底排三键
    for i, key_name in enumerate(keys[1:]):
        bx = i * (KEY_SIZE + GAP)
        by = KEY_SIZE + GAP
        label = KIJL_KEY_TO_ARROW[key_name] if is_arrow_group else key_name
        use_font = arrow_font if is_arrow_group else font
        draw_key(draw, bx, by, label, use_font, pressed=(key_name in pressed_keys))

    return img


def parse_sequence(seq_str):
    """
    解析按键序列字符串。

    格式: 键名:帧数,键名:帧数,...
    多键同时按: 用+连接，如 W+K:10
    无按键段: NONE:帧数

    返回: [(pressed_keys_set, frame_count), ...]
    """
    segments = []
    for part in seq_str.split(','):
        part = part.strip()
        if ':' not in part:
            raise ValueError(f"序列格式错误，缺少冒号: '{part}' (应为 键名:帧数)")
        keys_str, count_str = part.rsplit(':', 1)
        count = int(count_str.strip())
        if keys_str.strip().upper() == 'NONE':
            pressed = set()
        else:
            pressed = {k.strip().upper() for k in keys_str.split('+') if k.strip()}
        segments.append((pressed, count))
    return segments


def build_frame_keys(segments, total_frames):
    """
    将序列段展开为逐帧按键列表。

    如果序列总帧数 < 视频总帧数，剩余帧无按键。
    如果序列总帧数 > 视频总帧数，截断。
    """
    frame_keys = []
    for pressed, count in segments:
        frame_keys.extend([pressed] * count)

    seq_total = len(frame_keys)
    if seq_total < total_frames:
        frame_keys.extend([set()] * (total_frames - seq_total))
    elif seq_total > total_frames:
        frame_keys = frame_keys[:total_frames]

    return frame_keys


def get_video_info(input_path):
    """用 ffprobe 获取视频的 width, height, fps, nb_frames。"""
    cmd = [
        'ffprobe', '-v', 'quiet', '-print_format', 'json',
        '-select_streams', 'v:0',
        '-show_entries', 'stream=width,height,r_frame_rate,nb_frames',
        input_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffprobe 失败: {result.stderr}")
    info = json.loads(result.stdout)['streams'][0]
    width = int(info['width'])
    height = int(info['height'])
    num, den = info['r_frame_rate'].split('/')
    fps = float(num) / float(den)
    # nb_frames 可能不存在（某些容器），fallback 到 0
    total_frames = int(info.get('nb_frames', 0))
    return width, height, fps, total_frames


def process_video(input_path, output_path, pressed_keys=None, sequence=None):
    """处理视频，逐帧添加键盘覆盖层。"""

    if not os.path.exists(input_path):
        print(f"错误: 输入文件不存在: {input_path}")
        return False

    try:
        width, height, fps, total_frames = get_video_info(input_path)
    except Exception as e:
        print(f"错误: {e}")
        return False

    print(f"输入: {input_path}")
    print(f"  分辨率: {width}x{height}, 帧率: {fps:.2f}fps, 总帧数: {total_frames or '未知'}")

    # 构建逐帧按键列表
    if sequence:
        segments = parse_sequence(sequence)
        frame_keys = build_frame_keys(segments, total_frames or 0)
        seg_desc = ' -> '.join(f"{'+'.join(s) if s else 'NONE'}*{c}" for s, c in segments)
        print(f"  按键序列: {seg_desc}")
    elif pressed_keys:
        frame_keys = None  # 全程相同，按需生成
        print(f"  全程按下: {', '.join(sorted(pressed_keys))}")
    else:
        frame_keys = None
        pressed_keys = set()
        print(f"  无按键高亮")

    # 创建输出目录
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # 启动 ffmpeg reader（解码为 rgb24 rawvideo）
    reader = subprocess.Popen(
        ['ffmpeg', '-i', input_path, '-f', 'rawvideo', '-pix_fmt', 'rgb24', 'pipe:1'],
        stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
    )

    # 启动 ffmpeg writer（从 rawvideo 编码为 h264 mp4）
    writer = subprocess.Popen(
        [
            'ffmpeg', '-y',
            '-f', 'rawvideo', '-pix_fmt', 'rgb24',
            '-s', f'{width}x{height}', '-r', str(fps),
            '-i', 'pipe:0',
            '-vcodec', 'libx264', '-pix_fmt', 'yuv420p',
            output_path,
        ],
        stdin=subprocess.PIPE, stderr=subprocess.DEVNULL,
    )

    # 加载字体
    font = load_font(20)
    arrow_font = load_font(22)

    # 覆盖层缓存
    overlay_cache = {}

    # 计算粘贴位置
    overlay_h = KEY_SIZE * 2 + GAP
    overlay_w = KEY_SIZE * 3 + GAP * 2
    wasd_x = MARGIN
    wasd_y = height - MARGIN - overlay_h
    kijl_x = width - MARGIN - overlay_w
    kijl_y = height - MARGIN - overlay_h

    frame_size = width * height * 3
    frame_count = 0

    try:
        while True:
            raw = reader.stdout.read(frame_size)
            if len(raw) < frame_size:
                break

            if frame_keys is not None:
                pressed = frame_keys[frame_count] if frame_count < len(frame_keys) else set()
            else:
                pressed = pressed_keys

            cache_key = frozenset(pressed)
            if cache_key not in overlay_cache:
                wasd_ov = create_group_overlay(
                    ['W', 'A', 'S', 'D'], pressed, font, arrow_font, is_arrow_group=False
                )
                kijl_ov = create_group_overlay(
                    ['I', 'J', 'K', 'L'], pressed, font, arrow_font, is_arrow_group=True
                )
                overlay_cache[cache_key] = (wasd_ov, kijl_ov)

            wasd_overlay, kijl_overlay = overlay_cache[cache_key]

            pil_frame = Image.fromarray(
                np.frombuffer(raw, dtype=np.uint8).reshape((height, width, 3))
            ).convert('RGBA')
            pil_frame.paste(wasd_overlay, (wasd_x, wasd_y), wasd_overlay)
            pil_frame.paste(kijl_overlay, (kijl_x, kijl_y), kijl_overlay)

            writer.stdin.write(np.array(pil_frame.convert('RGB')).tobytes())

            frame_count += 1
            if frame_count % 10 == 0:
                print(f"  进度: {frame_count}" + (f"/{total_frames}" if total_frames else ""), end='\r')
    finally:
        reader.stdout.close()
        reader.wait()
        writer.stdin.close()
        writer.wait()

    print(f"\n完成! 处理 {frame_count} 帧")
    print(f"输出: {output_path}")
    return True


def main():
    parser = argparse.ArgumentParser(
        description='在视频上添加键盘指示器覆盖层（支持按键高亮）',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  %(prog)s -i video.mp4 -o out.mp4 --pressed A
  %(prog)s -i video.mp4 -o out.mp4 --pressed A,W,K
  %(prog)s -i video.mp4 -o out.mp4 --sequence W:25,A:28,W:24
  %(prog)s -i video.mp4 -o out.mp4 --sequence W+K:10,A+J:20,NONE:5
  %(prog)s -i video.mp4 -o out.mp4              # 无高亮
        """
    )
    parser.add_argument('--input', '-i', required=True, help='输入视频路径')
    parser.add_argument('--output', '-o', required=True, help='输出视频路径')
    parser.add_argument('--pressed', '-p', default='',
                        help='全程按住的键，逗号分隔 (如: A 或 A,W,K)')
    parser.add_argument('--sequence', '-s', default='',
                        help='按键序列 (如: W:25,A:28,W:24 或 W+K:10,A+J:20)')

    args = parser.parse_args()

    if args.sequence and args.pressed:
        print("错误: --pressed 和 --sequence 不能同时使用")
        sys.exit(1)

    pressed_keys = None
    sequence = None

    if args.pressed:
        pressed_keys = {k.strip().upper() for k in args.pressed.split(',') if k.strip()}
    elif args.sequence:
        sequence = args.sequence

    success = process_video(args.input, args.output, pressed_keys=pressed_keys, sequence=sequence)
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
