import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

CLASS_MAP = {
    'abandon': 0,
    'fight'  : 1,
    'broken' : 2,
    'theft'  : 3,
    'normal' : 4
}

class ActionDataset(Dataset):
    def __init__(self, data_root, split='train', num_frames=16):
        self.num_frames = num_frames
        self.split      = split
        self.clips      = []
        self.labels     = []

        split_dir = os.path.join(data_root, split, 'clips')

        for cls_name, cls_idx in CLASS_MAP.items():
            cls_dir = os.path.join(split_dir, cls_name)

            if not os.path.exists(cls_dir):
                print(f"⚠️ 폴더 없음: {cls_dir}")
                continue

            clips = [f for f in os.listdir(cls_dir)
                    if f.endswith('.mp4')]

            for clip in clips:
                self.clips.append(os.path.join(cls_dir, clip))
                self.labels.append(cls_idx)

            print(f"[{split}] {cls_name}: {len(clips)}개")

        print(f"→ 총 {len(self.clips)}개\n")

    def load_frames(self, video_path):
        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            print(f"⚠️ 영상 열기 실패: {video_path}")
            return torch.zeros(3, self.num_frames, 112, 112)

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if total_frames <= 0:
            print(f"⚠️ 프레임 수 없음: {video_path}")
            cap.release()
            return torch.zeros(3, self.num_frames, 112, 112)

        indices    = np.linspace(0, total_frames - 1,
                                self.num_frames, dtype=int)

        frames     = []
        last_valid = np.zeros((112, 112, 3), dtype=np.uint8)

        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()

            if ret:
                frame      = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame      = cv2.resize(frame, (112, 112))
                last_valid = frame

            frames.append(last_valid.copy())

        cap.release()

        frames = np.array(frames).transpose(3, 0, 1, 2)
        frames = frames.astype(np.float32) / 255.0

        mean   = np.array([0.45, 0.45, 0.45]).reshape(3, 1, 1, 1)
        std    = np.array([0.225, 0.225, 0.225]).reshape(3, 1, 1, 1)
        frames = (frames - mean) / std

        return torch.FloatTensor(frames)

    def __len__(self):
        return len(self.clips)

    def __getitem__(self, idx):
        try:
            clip  = self.load_frames(self.clips[idx])
            label = self.labels[idx]
            return clip, label
        except Exception as e:
            print(f"⚠️ 로딩 실패 [{idx}]: {e}")
            return torch.zeros(3, self.num_frames, 112, 112), self.labels[idx]