# inference.py
import argparse
import re
import shutil
import subprocess
import tempfile
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import torch
import torch.nn as nn
import cv2
import numpy as np

from config import CONFIG
from dataset import CLASS_MAP


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_VAL_DIR = '../../data_with_weight_file/dataset/val'
DEFAULT_CHECKPOINT = '../../data_with_weight_file/checkpoints/best_model.pth'
_TORCHVISION_STUB_LIB = None


class ResourceMonitor:
    def __init__(self, interval_ms=1000):
        self.interval_ms = interval_ms
        self.log_path = None
        self.log_file = None
        self.process = None

    def start(self):
        if shutil.which('tegrastats') is None:
            return

        self.log_file = tempfile.NamedTemporaryFile(
            prefix='python_action_tegrastats.',
            suffix='.log',
            delete=False,
            mode='w',
            encoding='utf-8',
        )
        self.log_path = self.log_file.name
        self.process = subprocess.Popen(
            ['tegrastats', '--interval', str(self.interval_ms)],
            stdout=self.log_file,
            stderr=subprocess.DEVNULL,
        )

    def stop(self):
        if self.process is not None:
            self.process.terminate()
            try:
                self.process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait()
            self.process = None

        if self.log_file is not None:
            self.log_file.close()
            self.log_file = None

    def summary_lines(self):
        if not self.log_path or not Path(self.log_path).is_file() or Path(self.log_path).stat().st_size == 0:
            return [
                '평균 CPU 사용률: unavailable',
                '평균 GPU 사용률: unavailable',
                '평균 메모리 사용량: unavailable',
                '평균 전력 사용량: unavailable',
            ]

        samples = 0
        cpu_sum = 0.0
        cpu_samples = 0
        gpu_sum = 0.0
        gpu_samples = 0
        ram_sum = 0.0
        ram_total = None
        ram_samples = 0
        power_sum = 0.0
        power_samples = 0

        with open(self.log_path, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                samples += 1

                ram = re.search(r'RAM\s+(\d+)/(\d+)MB', line)
                if ram:
                    ram_sum += int(ram.group(1))
                    ram_total = int(ram.group(2))
                    ram_samples += 1

                cpu = re.search(r'CPU\s+\[([^\]]+)\]', line)
                if cpu:
                    values = [int(v) for v in re.findall(r'(\d+)%', cpu.group(1))]
                    if values:
                        cpu_sum += sum(values) / len(values)
                        cpu_samples += 1

                gpu = re.search(r'GR3D_FREQ\s+(\d+)%', line)
                if gpu:
                    gpu_sum += int(gpu.group(1))
                    gpu_samples += 1

                power = re.search(r'(?:VDD_IN|POM_5V_IN|VIN_SYS_5V0)\s+(\d+)mW', line)
                if power:
                    power_sum += int(power.group(1))
                    power_samples += 1

        def avg(total, count):
            return total / count if count else None

        cpu_avg = avg(cpu_sum, cpu_samples)
        gpu_avg = avg(gpu_sum, gpu_samples)
        ram_avg = avg(ram_sum, ram_samples)
        power_avg = avg(power_sum, power_samples)

        lines = [f'리소스 샘플 수: {samples}']
        lines.append(f'평균 CPU 사용률: {cpu_avg:.2f}%' if cpu_avg is not None else '평균 CPU 사용률: unavailable')
        lines.append(f'평균 GPU 사용률: {gpu_avg:.2f}%' if gpu_avg is not None else '평균 GPU 사용률: unavailable')
        if ram_avg is not None:
            suffix = f'/{ram_total}MB' if ram_total is not None else 'MB'
            lines.append(f'평균 메모리 사용량: {ram_avg:.0f}{suffix}')
        else:
            lines.append('평균 메모리 사용량: unavailable')
        lines.append(f'평균 전력 사용량: {power_avg:.0f}mW' if power_avg is not None else '평균 전력 사용량: unavailable')
        lines.append(f'리소스 로그: {self.log_path}')
        return lines


def resolve_path(path):
    path = Path(path)
    if path.is_absolute():
        return path
    return (BASE_DIR / path).resolve()


def normalize_label(label):
    return 'broken' if label == 'brocken' else label


def model_precision_from_path(path):
    name = Path(path).name.lower()
    if 'int8' in name:
        return 'INT8'
    if 'fp16' in name:
        return 'FP16'
    if 'fp32' in name:
        return 'FP32'
    return 'FP32'


def load_model(checkpoint_path, device):
    global _TORCHVISION_STUB_LIB

    # Some torch/torchvision builds fail while registering fake torchvision::nms,
    # even though video models do not use NMS. Defining the op first avoids that
    # import-time failure without changing inference behavior.
    try:
        _TORCHVISION_STUB_LIB = torch.library.Library("torchvision", "DEF")
        _TORCHVISION_STUB_LIB.define(
            "nms(Tensor dets, Tensor scores, float iou_threshold) -> Tensor"
        )
    except RuntimeError:
        pass

    from torchvision.models.video import r2plus1d_18

    model    = r2plus1d_18(weights=None)
    model.fc = nn.Linear(512, CONFIG['num_classes'])

    checkpoint = torch.load(resolve_path(checkpoint_path), map_location=device)
    model.load_state_dict(checkpoint['model'])
    model = model.to(device)
    model.eval()
    return model


def read_frame(cap, frame_idx):
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
    ret, frame = cap.read()
    if not ret:
        return None
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return cv2.resize(frame, (112, 112))


def load_frame_window(video_path, start_frame=None, end_frame=None, num_frames=None):
    num_frames = num_frames or CONFIG['num_frames']

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"⚠️ 영상 열기 실패: {video_path}")
        return None

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        print(f"⚠️ 프레임 수 없음: {video_path}")
        cap.release()
        return None

    if start_frame is None:
        start_frame = 0
    if end_frame is None:
        end_frame = total_frames - 1

    start_frame = max(0, int(start_frame))
    end_frame = min(total_frames - 1, int(end_frame))
    if end_frame < start_frame:
        cap.release()
        return None

    indices = np.linspace(start_frame, end_frame, num_frames, dtype=int)
    frames = []
    last_valid = np.zeros((112, 112, 3), dtype=np.uint8)

    for idx in indices:
        frame = read_frame(cap, idx)
        if frame is not None:
            last_valid = frame
        frames.append(last_valid.copy())

    cap.release()

    frames = np.array(frames).transpose(3, 0, 1, 2)
    frames = frames.astype(np.float32) / 255.0
    mean = np.array([0.45, 0.45, 0.45]).reshape(3, 1, 1, 1)
    std = np.array([0.225, 0.225, 0.225]).reshape(3, 1, 1, 1)
    frames = (frames - mean) / std
    return torch.FloatTensor(frames)


def load_video_frame_cache(video_path):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"⚠️ 영상 열기 실패: {video_path}")
        return None

    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (112, 112))
        frames.append(frame)

    cap.release()

    if not frames:
        print(f"⚠️ 프레임 수 없음: {video_path}")
        return None

    frames = np.asarray(frames, dtype=np.float32) / 255.0
    frames = frames.transpose(3, 0, 1, 2)
    mean = np.array([0.45, 0.45, 0.45], dtype=np.float32).reshape(3, 1, 1, 1)
    std = np.array([0.225, 0.225, 0.225], dtype=np.float32).reshape(3, 1, 1, 1)
    frames = (frames - mean) / std
    return torch.from_numpy(frames)


def cached_contiguous_window(frame_cache, start_frame, num_frames=None):
    num_frames = num_frames or CONFIG['num_frames']
    end_frame = start_frame + num_frames
    if start_frame < 0 or end_frame > frame_cache.shape[1]:
        return None
    return frame_cache[:, start_frame:end_frame, :, :]


def load_contiguous_window(video_path, start_frame, num_frames=None):
    num_frames = num_frames or CONFIG['num_frames']
    return load_frame_window(
        video_path,
        start_frame=start_frame,
        end_frame=start_frame + num_frames - 1,
        num_frames=num_frames,
    )


def get_total_frames(video_path):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return 0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return total_frames


def label_for_frame(frame_idx, event_label, ranges):
    for start_frame, end_frame in ranges:
        if start_frame <= frame_idx <= end_frame:
            return event_label
    return 'unlabeled'


def predict_tensor(model, frames, device):
    idx_to_cls = {v: k for k, v in CLASS_MAP.items()}
    input_tensor = frames.contiguous().unsqueeze(0).to(device)

    with torch.no_grad():
        outputs     = model(input_tensor)
        probs       = torch.softmax(outputs, dim=1)[0]
        pred_idx    = probs.argmax().item()
        pred_class  = idx_to_cls[pred_idx]
        confidence  = probs[pred_idx].item() * 100

    return pred_class, confidence, probs.detach().cpu()


def predict_video(video_path, checkpoint_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = load_model(checkpoint_path, device)
    frames = load_frame_window(video_path)
    if frames is None:
        return None

    pred_class, confidence, probs = predict_tensor(model, frames, device)
    idx_to_cls = {v: k for k, v in CLASS_MAP.items()}

    # 결과 출력
    print(f"\n영상: {video_path}")
    print(f"예측: {pred_class} ({confidence:.1f}%)")
    print(f"\n클래스별 확률:")
    for idx, cls in idx_to_cls.items():
        print(f"  {cls:>10}: {probs[idx].item()*100:.1f}%")

    return pred_class, confidence


def xml_for_video(val_dir, event_label, video_path):
    stem = Path(video_path).stem
    return Path(val_dir) / 'labels' / event_label / f'{stem}.xml'


def parse_event_ranges(xml_path, event_label):
    if not Path(xml_path).is_file():
        return []

    tree = ET.parse(xml_path)
    ranges = []
    pending_start = None
    markers = []

    for track in tree.findall('.//track'):
        label = track.attrib.get('label', '')
        if label not in {f'{event_label}_start', f'{event_label}_end'}:
            continue

        kind = 'start' if label.endswith('_start') else 'end'
        for node in list(track):
            frame = node.attrib.get('frame')
            outside = node.attrib.get('outside')
            if frame is None or outside != '0':
                continue
            markers.append((int(frame), kind))

    for frame, kind in sorted(markers):
        if kind == 'start':
            pending_start = frame
        elif kind == 'end' and pending_start is not None:
            ranges.append((pending_start, frame))
            pending_start = None

    return ranges


def iter_validation_videos(val_dir):
    clips_dir = Path(val_dir) / 'clips'
    if not clips_dir.is_dir():
        return

    for video_path in sorted(clips_dir.glob('*/*.mp4')):
        event_label = normalize_label(video_path.parent.name)
        if event_label == 'normal':
            continue
        yield video_path, event_label


def validate_event_only(val_dir, checkpoint_path):
    start_time = time.perf_counter()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    val_dir = resolve_path(val_dir)
    model_precision = model_precision_from_path(checkpoint_path)
    monitor = ResourceMonitor()

    print()
    print("========== 검증 시작 ==========")
    print(f"검증 데이터셋: {val_dir}")
    print(f"모델 정밀도: {model_precision}")
    print("평가 방식: XML 이벤트 구간만 추론/평가")
    print("================================")
    print()

    monitor.start()
    try:
        model = load_model(checkpoint_path, device)

        total_windows = 0
        correct_windows = 0
        correct_videos = 0
        skipped_videos = 0
        skipped_windows = 0
        failed = 0
        num_frames = CONFIG['num_frames']
        total_videos = 0
        inference_time = 0.0

        for video_idx, (video_path, event_label) in enumerate(iter_validation_videos(val_dir), 1):
            total_videos = video_idx
            xml_path = xml_for_video(val_dir, event_label, video_path)
            ranges = parse_event_ranges(xml_path, event_label)
            last_pred = 'none'

            if not ranges:
                skipped_videos += 1
                failed += 1
                print(
                    f"[{video_idx}] {video_path.name}  이벤트={event_label}  "
                    f"마지막예측=none  이벤트프레임정답=0/0  스킵=0  MISS"
                )
                continue

            video_correct = 0
            video_total = 0
            video_skipped = 0
            frame_cache = load_video_frame_cache(video_path)
            if frame_cache is None:
                failed += 1
                print(
                    f"[{video_idx}] {video_path.name}  이벤트={event_label}  "
                    f"마지막예측=none  이벤트프레임정답=0/0  스킵=0  MISS"
                )
                continue
            total_frames = frame_cache.shape[1]

            for window_end in range(num_frames - 1, total_frames):
                window_start = window_end - num_frames + 1
                frames = cached_contiguous_window(frame_cache, window_start, num_frames)
                if frames is None:
                    video_skipped += 1
                    skipped_windows += 1
                    print(
                        f"    frame={window_end:03d} 정답=unlabeled 예측=none SKIP"
                    )
                    continue

                if device.type == 'cuda':
                    torch.cuda.synchronize()
                infer_start = time.perf_counter()
                pred_class, confidence, _ = predict_tensor(model, frames, device)
                if device.type == 'cuda':
                    torch.cuda.synchronize()
                inference_time += time.perf_counter() - infer_start

                expected = label_for_frame(window_end, event_label, ranges)
                last_pred = pred_class

                if expected == 'unlabeled':
                    video_skipped += 1
                    skipped_windows += 1
                    print(
                        f"    frame={window_end:03d} 정답=unlabeled 예측={pred_class} SKIP"
                    )
                    continue

                total_windows += 1
                video_total += 1

                is_correct = pred_class == expected
                if is_correct:
                    correct_windows += 1
                    video_correct += 1

                result = 'OK' if is_correct else 'MISS'
                print(
                    f"    frame={window_end:03d} 정답={expected} "
                    f"예측={pred_class} {result}"
                )

            result = 'MISS'
            if video_total > 0 and video_correct == video_total:
                correct_videos += 1
                result = 'OK'

            print(
                f"[{video_idx}] {video_path.name}  이벤트={event_label}  "
                f"마지막예측={last_pred}  이벤트프레임정답={video_correct}/{video_total}  "
                f"스킵={video_skipped}  {result}"
            )
    finally:
        monitor.stop()

    accuracy = (correct_windows / total_windows * 100.0) if total_windows else 0.0
    total_time = time.perf_counter() - start_time
    fps = (total_windows / inference_time) if inference_time > 0 else 0.0
    avg_infer_ms = (inference_time * 1000.0 / total_windows) if total_windows else 0.0

    print("\n========== 검증 결과 ==========")
    print(f"이벤트 프레임 전체 정답 영상 수: {correct_videos}/{total_videos}")
    print(f"이벤트 프레임/윈도우 정답 수: {correct_windows}/{total_windows}")
    print(f"스킵한 unlabeled 프레임/윈도우 수: {skipped_windows}")
    print(f"실패/예측없음: {failed}")
    print(f"Event-only Accuracy: {accuracy:.2f}%")
    print(f"모델 정밀도: {model_precision}")
    print(f"총 실행 시간: {total_time:.3f}초")
    print(f"추론 실행 시간: {inference_time:.3f}초")
    print(f"평가 윈도우 처리 FPS: {fps:.2f}")
    print(f"윈도우당 평균 추론 시간: {avg_infer_ms:.2f}ms")
    for line in monitor.summary_lines():
        print(line)
    print("================================")

    return accuracy


def parse_args():
    parser = argparse.ArgumentParser(description='R(2+1)D action inference/evaluation')
    parser.add_argument('--video', help='단일 영상 경로')
    parser.add_argument('--checkpoint', default=DEFAULT_CHECKPOINT, help='체크포인트 경로')
    parser.add_argument('--validate', action='store_true', help='val 디렉터리 이벤트 구간 평가')
    parser.add_argument('--val-dir', default=DEFAULT_VAL_DIR, help='validation 디렉터리')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    if args.validate or not args.video:
        validate_event_only(args.val_dir, args.checkpoint)
    else:
        predict_video(args.video, args.checkpoint)
