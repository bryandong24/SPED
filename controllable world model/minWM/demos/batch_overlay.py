#!/usr/bin/env python3
"""
批量为 step_3000 目录下的 MP4 添加键盘可视化。

文件名格式: {name}_{action1}{num1}-{action2}{num2}-...mp4
帧数规则:
  - 第一个 action: 1 + 4*(num-1)
  - 中间 action:   num * 4
  - 最后一个 action: 4 + num * 4
"""

import argparse
import os
import re
import sys

sys.path.insert(0, os.path.dirname(__file__))
from overlay_keys import process_video


def parse_sequence_from_filename(stem):
    """从文件名 stem 解析按键序列字符串。"""
    # stem 格式: {name}_{action1}{num1}-{action2}{num2}-...
    # 找最后一个 _ 后面的 action 序列部分
    action_part = stem.rsplit('_', 1)[-1]
    segments = re.findall(r'([A-Za-z]+)(\d+)', action_part)
    if not segments:
        return None

    result = []
    for i, (action, num_str) in enumerate(segments):
        num = int(num_str)
        key = action.upper()
        if len(segments) == 1:
            frames = 1 + 4 * (num - 1) + 4
        elif i == 0:
            frames = 1 + 4 * (num - 1)
        elif i == len(segments) - 1:
            frames = 4 + num * 4
        else:
            frames = num * 4
        result.append(f"{key}:{frames}")

    return ','.join(result)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', '-i', required=True, help='输入视频目录')
    parser.add_argument('--output', '-o', required=True, help='输出视频目录')
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    mp4_files = [f for f in os.listdir(args.input) if f.endswith('.mp4')]
    print(f"共 {len(mp4_files)} 个文件")

    for fname in sorted(mp4_files):
        stem = fname[:-4]
        seq = parse_sequence_from_filename(stem)
        input_path = os.path.join(args.input, fname)
        output_path = os.path.join(args.output, fname)

        print(f"\n[{fname}]")
        print(f"  序列: {seq}")

        process_video(input_path, output_path, sequence=seq)


if __name__ == '__main__':
    main()
