# inference.py
import torch
import torch.nn as nn
import cv2
import numpy as np
from torchvision.models.video import r2plus1d_18

from config import CONFIG
from dataset import CLASS_MAP

def predict_video(video_path, checkpoint_path):
    
    device = torch.device('cuda' if torch.cuda.is_available()
                         else 'cpu')
    
    # 모델 로드
    model    = r2plus1d_18(pretrained=False)
    model.fc = nn.Linear(512, CONFIG['num_classes'])
    
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model'])
    model = model.to(device)
    model.eval()
    
    idx_to_cls = {v: k for k, v in CLASS_MAP.items()}
    
    # 영상 로드
    cap          = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if not cap.isOpened():
        print(f"⚠️ 영상 열기 실패: {video_path}")
        return
    
    # 프레임 추출
    indices    = np.linspace(0, total_frames - 1,
                            CONFIG['num_frames'], dtype=int)
    target_set = set(indices)
    all_frames = {}
    frame_idx  = 0
    
    while frame_idx <= indices[-1]:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx in target_set:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (112, 112))
            all_frames[frame_idx] = frame
        frame_idx += 1
    cap.release()
    
    # 텐서 변환
    frames     = []
    last_valid = np.zeros((112, 112, 3), dtype=np.uint8)
    for idx in indices:
        if idx in all_frames:
            last_valid = all_frames[idx]
        frames.append(last_valid.copy())
    
    frames = np.array(frames).transpose(3, 0, 1, 2)
    frames = frames.astype(np.float32) / 255.0
    mean   = np.array([0.45, 0.45, 0.45]).reshape(3, 1, 1, 1)
    std    = np.array([0.225, 0.225, 0.225]).reshape(3, 1, 1, 1)
    frames = (frames - mean) / std
    
    # 추론
    input_tensor = torch.FloatTensor(frames).unsqueeze(0).to(device)
    
    with torch.no_grad():
        outputs     = model(input_tensor)
        probs       = torch.softmax(outputs, dim=1)[0]
        pred_idx    = probs.argmax().item()
        pred_class  = idx_to_cls[pred_idx]
        confidence  = probs[pred_idx].item() * 100
    
    # 결과 출력
    print(f"\n영상: {video_path}")
    print(f"예측: {pred_class} ({confidence:.1f}%)")
    print(f"\n클래스별 확률:")
    for idx, cls in idx_to_cls.items():
        print(f"  {cls:>10}: {probs[idx].item()*100:.1f}%")
    
    return pred_class, confidence


if __name__ == '__main__':
    # predict_video(
    #     video_path      = './test.mp4',       # ← 테스트할 영상 경로
    #     checkpoint_path = './checkpoints/best_model.pth'
    # )
    import os
    import random

    # test 폴더에서 랜덤으로 영상 하나 뽑기
    CLASSES = ['abandon', 'theft', 'fight', 'broken'] 
    cls_name  = random.choice(CLASSES)
    test_dir    = f'./dataset/test/clips/{cls_name}'
    
    videos     = [f for f in os.listdir(test_dir) 
                  if f.endswith('.mp4')]
    random_video = random.choice(videos)
    video_path   = os.path.join(test_dir, random_video)
    
    predict_video(
        video_path      = video_path,
        checkpoint_path = '/root/data_with_weight_file/checkpoints/best_model.pth'
    )