#!/usr/bin/env python3
"""
将 step_3000_overlay 目录下所有 MP4 按文件名排序后时间拼接为一个视频。
使用 ffmpeg concat demuxer，无需任何视频处理库。
"""

import argparse
import os
import subprocess
import tempfile


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', '-i', required=True, help='输入视频目录')
    parser.add_argument('--output', '-o', required=True, help='输出视频路径')
    args = parser.parse_args()

    mp4_files = sorted(f for f in os.listdir(args.input) if f.endswith('.mp4'))
    if not mp4_files:
        print("没有找到 MP4 文件")
        return

    print(f"共 {len(mp4_files)} 个文件，开始拼接...")

    with tempfile.NamedTemporaryFile('w', suffix='.txt', delete=False) as f:
        list_path = f.name
        for fname in mp4_files:
            # ffmpeg concat list 要求路径中单引号转义
            path = os.path.join(args.input, fname).replace("'", "'\\''")
            f.write(f"file '{path}'\n")

    try:
        subprocess.run(
            [
                'ffmpeg', '-y',
                '-f', 'concat', '-safe', '0',
                '-i', list_path,
                '-c', 'copy',
                args.output,
            ],
            check=True,
        )
        print(f"完成: {args.output}")
    finally:
        os.unlink(list_path)


if __name__ == '__main__':
    main()
