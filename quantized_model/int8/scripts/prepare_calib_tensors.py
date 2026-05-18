#!/usr/bin/env python3
from __future__ import annotations

import argparse
import random
from pathlib import Path

import cv2
import numpy as np


def list_videos(dataset_root: Path, limit: int) -> list[Path]:
    videos = sorted(dataset_root.rglob("*.mp4"))
    if not videos:
        raise RuntimeError(f"No mp4 files found under {dataset_root}")
    random.seed(0)
    random.shuffle(videos)
    return videos[:limit]


def preprocess_video(video_path: Path, num_frames: int, height: int, width: int) -> np.ndarray:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        cap.release()
        raise RuntimeError(f"No frames in video: {video_path}")

    indices = np.linspace(0, total_frames - 1, num_frames, dtype=np.int32)
    frames = []
    last_valid = np.zeros((height, width, 3), dtype=np.uint8)

    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, frame = cap.read()
        if ok:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (width, height))
            last_valid = frame
        frames.append(last_valid.copy())

    cap.release()

    tensor = np.asarray(frames, dtype=np.float32).transpose(3, 0, 1, 2)
    tensor = tensor / 255.0
    tensor = (tensor - 0.45) / 0.225
    return np.ascontiguousarray(tensor[np.newaxis, ...], dtype=np.float32)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare calibration tensors for C++ TensorRT calibrator.")
    parser.add_argument("--dataset-root", type=Path, default=Path("/root/data_with_weight_file/dataset/train/clips"))
    parser.add_argument("--output-dir", type=Path, default=Path("/root/data_with_weight_file/quantized/calib_tensors"))
    parser.add_argument("--list-file", type=Path, default=Path("/root/data_with_weight_file/quantized/calib_tensors/list.txt"))
    parser.add_argument("--num-videos", type=int, default=100)
    parser.add_argument("--num-frames", type=int, default=16)
    parser.add_argument("--height", type=int, default=112)
    parser.add_argument("--width", type=int, default=112)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.list_file.parent.mkdir(parents=True, exist_ok=True)

    videos = list_videos(args.dataset_root, args.num_videos)
    written = []
    for idx, video_path in enumerate(videos):
        try:
            tensor = preprocess_video(video_path, args.num_frames, args.height, args.width)
        except RuntimeError as exc:
            print(f"[WARN] {exc}")
            continue

        output_path = args.output_dir / f"calib_{idx:05d}.bin"
        tensor.tofile(output_path)
        written.append(output_path)
        print(f"[OK] {output_path} <- {video_path}")

    if not written:
        raise RuntimeError("No calibration tensors were written.")

    args.list_file.write_text("\n".join(str(path) for path in written) + "\n", encoding="utf-8")
    print(f"[OK] Wrote tensor list: {args.list_file}")


if __name__ == "__main__":
    main()
