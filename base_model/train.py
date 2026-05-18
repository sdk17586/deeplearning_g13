import os
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision.models.video import r2plus1d_18
from tqdm import tqdm

from config import CONFIG
from dataset import ActionDataset, CLASS_MAP

def get_sampler(dataset):
    class_counts = [0] * len(CLASS_MAP)
    for label in dataset.labels:
        class_counts[label] += 1

    print("클래스별 샘플 수:")
    for cls, idx in CLASS_MAP.items():
        print(f"  {cls:>10}: {class_counts[idx]}개")

    weights = [1.0 / class_counts[label]
               for label in dataset.labels]

    return WeightedRandomSampler(
        weights,
        num_samples=len(dataset),
        replacement=True
    )

def train():

    device = torch.device('cuda' if torch.cuda.is_available()
                         else 'cpu')
    print(f"사용 디바이스: {device}\n")

    train_dataset = ActionDataset(CONFIG['data_root'],
                                  split='train',
                                  num_frames=CONFIG['num_frames'])
    val_dataset   = ActionDataset(CONFIG['data_root'],
                                  split='val',
                                  num_frames=CONFIG['num_frames'])

    sampler      = get_sampler(train_dataset)
    train_loader = DataLoader(train_dataset,
                              batch_size=CONFIG['batch_size'],
                              sampler=sampler,
                              num_workers=CONFIG['num_workers'],
                              pin_memory=True)
    val_loader   = DataLoader(val_dataset,
                              batch_size=CONFIG['batch_size'],
                              shuffle=False,
                              num_workers=CONFIG['num_workers'],
                              pin_memory=True)

    model    = r2plus1d_18(pretrained=True)
    model.fc = nn.Linear(512, CONFIG['num_classes'])
    model    = model.to(device)
    print(f"모델 로드 완료 (Kinetics pretrained)\n")

    optimizer = torch.optim.Adam([
        {'params': model.stem.parameters(),   'lr': CONFIG['lr'] * 0.1},
        {'params': model.layer1.parameters(), 'lr': CONFIG['lr'] * 0.1},
        {'params': model.layer2.parameters(), 'lr': CONFIG['lr'] * 0.1},
        {'params': model.layer3.parameters(), 'lr': CONFIG['lr'] * 0.1},
        {'params': model.layer4.parameters(), 'lr': CONFIG['lr']},
        {'params': model.fc.parameters(),     'lr': CONFIG['lr'] * 10},
    ])

    criterion = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=10, gamma=0.1
    )

    os.makedirs(CONFIG['save_dir'], exist_ok=True)

    best_val_acc  = 0
    best_val_loss = float('inf')
    idx_to_cls    = {v: k for k, v in CLASS_MAP.items()}

    # 히스토리 초기화
    history = {
        'train_loss'      : [],
        'val_loss'        : [],
        'train_acc'       : [],
        'val_acc'         : [],
        'class_val_acc'   : [],   # 에폭별 클래스별 val 정확도
    }

    for epoch in range(CONFIG['epochs']):

        # ── Train ──────────────────────────────────────────
        model.train()
        train_loss    = 0
        train_correct = 0

        pbar = tqdm(train_loader,
                   desc=f"Epoch [{epoch+1}/{CONFIG['epochs']}] Train")

        for clips, labels in pbar:
            clips  = clips.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(clips)
            loss    = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss    += loss.item()
            _, predicted   = outputs.max(1)
            train_correct += predicted.eq(labels).sum().item()

            pbar.set_postfix({'loss': f"{loss.item():.4f}"})

        avg_train_loss = train_loss / len(train_loader)
        train_acc      = train_correct / len(train_dataset) * 100

        # ── Validation ─────────────────────────────────────
        model.eval()
        val_loss      = 0
        val_correct   = 0
        class_correct = {cls: 0 for cls in CLASS_MAP}
        class_total   = {cls: 0 for cls in CLASS_MAP}

        with torch.no_grad():
            for clips, labels in tqdm(val_loader, desc="Val"):
                clips   = clips.to(device)
                labels  = labels.to(device)
                outputs = model(clips)
                loss    = criterion(outputs, labels)   # ← val_loss 계산

                val_loss    += loss.item()
                _, predicted = outputs.max(1)
                val_correct += predicted.eq(labels).sum().item()

                for pred, label in zip(predicted, labels):
                    cls_name = idx_to_cls[label.item()]
                    class_total[cls_name]   += 1
                    if pred == label:
                        class_correct[cls_name] += 1

        avg_val_loss = val_loss / len(val_loader)
        val_acc      = val_correct / len(val_dataset) * 100

        # 클래스별 val 정확도
        class_val_acc = {}
        for cls in CLASS_MAP:
            if class_total[cls] > 0:
                class_val_acc[cls] = round(
                    class_correct[cls] / class_total[cls] * 100, 2
                )

        # ── 출력 ───────────────────────────────────────────
        print(f"\nEpoch [{epoch+1}/{CONFIG['epochs']}]")
        print(f"  Train Loss : {avg_train_loss:.4f}  |  Train Acc: {train_acc:.2f}%")
        print(f"  Val   Loss : {avg_val_loss:.4f}  |  Val   Acc: {val_acc:.2f}%")

        # 과적합 경고
        if avg_val_loss > avg_train_loss * 1.5:
            print(f"  ⚠️  과적합 의심 (train_loss: {avg_train_loss:.4f} / val_loss: {avg_val_loss:.4f})")

        print(f"  클래스별 Val 정확도:")
        for cls, acc in class_val_acc.items():
            print(f"    {cls:>10}: {acc:.2f}%")

        # ── 히스토리 누적 ───────────────────────────────────
        history['train_loss'].append(round(avg_train_loss, 4))
        history['val_loss'].append(round(avg_val_loss, 4))
        history['train_acc'].append(round(train_acc, 2))
        history['val_acc'].append(round(val_acc, 2))
        history['class_val_acc'].append(class_val_acc)

        # ── 모델 저장 (val_acc 기준) ────────────────────────
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_path    = os.path.join(CONFIG['save_dir'], 'best_model50.pth')
            torch.save({
                'epoch'      : epoch + 1,
                'model'      : model.state_dict(),
                'optimizer'  : optimizer.state_dict(),
                'train_loss' : avg_train_loss,
                'val_loss'   : avg_val_loss,
                'train_acc'  : train_acc,
                'val_acc'    : val_acc,
                'config'     : CONFIG
            }, save_path)
            print(f"  ✅ 최고 모델 저장 → Val Acc: {val_acc:.2f}%  Val Loss: {avg_val_loss:.4f}")

        scheduler.step()
        print()

    # ── 학습 완료 후 히스토리 저장 ──────────────────────────
    history_path = os.path.join(CONFIG['save_dir'], 'history50.json')
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2, ensure_ascii=False)

    print(f"학습 히스토리 저장 완료 → {history_path}")
    print(f"학습 완료 | 최고 Val Acc: {best_val_acc:.2f}%")


if __name__ == '__main__':
    train()