# inference.py
import argparse
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


def resolve_path(path):
    path = Path(path)
    if path.is_absolute():
        return path
    return (BASE_DIR / path).resolve()


def normalize_label(label):
    return 'broken' if label == 'brocken' else label


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
    input_tensor = frames.unsqueeze(0).to(device)

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
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    val_dir = resolve_path(val_dir)

    print()
    print("========== 검증 시작 ==========")
    print(f"검증 데이터셋: {val_dir}")
    print("평가 방식: XML 이벤트 구간만 추론/평가")
    print("================================")
    print()

    model = load_model(checkpoint_path, device)

    total_windows = 0
    correct_windows = 0
    correct_videos = 0
    skipped_videos = 0
    skipped_windows = 0
    failed = 0
    num_frames = CONFIG['num_frames']
    total_videos = 0

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
        total_frames = get_total_frames(video_path)

        for window_end in range(num_frames - 1, total_frames):
            window_start = window_end - num_frames + 1
            frames = load_contiguous_window(video_path, window_start, num_frames)
            if frames is None:
                video_skipped += 1
                skipped_windows += 1
                print(
                    f"    frame={window_end:03d} 정답=unlabeled 예측=none SKIP"
                )
                continue

            pred_class, confidence, _ = predict_tensor(model, frames, device)
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

    accuracy = (correct_windows / total_windows * 100.0) if total_windows else 0.0
    print("\n========== 검증 결과 ==========")
    print(f"이벤트 프레임 전체 정답 영상 수: {correct_videos}/{total_videos}")
    print(f"이벤트 프레임/윈도우 정답 수: {correct_windows}/{total_windows}")
    print(f"스킵한 unlabeled 프레임/윈도우 수: {skipped_windows}")
    print(f"실패/예측없음: {failed}")
    print(f"Event-only Accuracy: {accuracy:.2f}%")
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
