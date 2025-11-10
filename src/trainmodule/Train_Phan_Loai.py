#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
train_full.py
Phiên bản hoàn chỉnh tích hợp:
- ResNet50 training (2 lớp: good / bad)
- Mô phỏng & thực thi các thuật toán: CNN (mô tả), NMS, Confidence Thresholding, Counting
- Tuỳ chọn tích hợp YOLOv8 (nếu cài ultralytics) cho detection + counting thực tế
Mục tiêu: file có thể nộp báo cáo, minh hoạ rõ ràng các thuật toán trong/ngoài CNN.
"""

import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
import json
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
from PIL import Image
from pathlib import Path
import logging
import argparse

# Thử import ultralytics (YOLOv8) nếu có; nếu không, chương trình vẫn chạy ResNet.
try:
    from ultralytics import YOLO
    HAS_ULTRALYTICS = True
except Exception:
    HAS_ULTRALYTICS = False

# ===================== CẤU HÌNH LOGGING =====================
def setup_logging(output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    log_file = output_dir / f'training_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s',
                        handlers=[logging.FileHandler(log_file), logging.StreamHandler()])
    logger = logging.getLogger(__name__)
    return logger

# ===================== DATASET =====================
class PotatoQualityDataset(Dataset):
    """Dataset phân loại chất lượng khoai tây: Good (Tốt) / Bad (Kém)"""
    def __init__(self, root_dir, transform=None, split='train'):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.split = split
        # Chỉ 2 lớp
        self.classes = ['good', 'bad']
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        self.samples = self._load_samples()

    def _load_samples(self):
        samples = []
        data_dir = self.root_dir / self.split
        for class_name in self.classes:
            class_dir = data_dir / class_name
            if class_dir.exists():
                for ext in ('*.jpg', '*.png', '*.jpeg'):
                    for img_path in class_dir.glob(ext):
                        samples.append((str(img_path), self.class_to_idx[class_name]))
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = cv2.imread(img_path)
        if image is None:
            # placeholder image nếu đọc lỗi
            image = np.zeros((224,224,3), dtype=np.uint8)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image)
        if self.transform:
            image = self.transform(image)
        return image, label

# ===================== MODEL =====================
class PotatoQualityModel(nn.Module):
    """ResNet50 classification (2 classes)"""
    def __init__(self, num_classes=2, pretrained=True):
        super(PotatoQualityModel, self).__init__()
        # Dùng torchvision models
        self.backbone = models.resnet50(pretrained=pretrained)
        num_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_features, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
    def forward(self, x):
        return self.backbone(x)

# ===================== THUẬT TOÁN (MÔ PHỎNG + THỰC THI) =====================
# 1) Mô tả / mô phỏng Convolution operation (để minh hoạ thuật toán CNN)
def convolution_operation(input_image, kernel):
    """
    Mô phỏng phép convolution (dùng OpenCV filter2D).
    Đây là phần minh hoạ thuật toán CNN (core idea: y = f(W * x + b)).
    """
    return cv2.filter2D(input_image, -1, kernel)

# 2) Non-Maximum Suppression (NMS) - loại bỏ box trùng lặp
def non_max_suppression(boxes, scores, iou_threshold=0.45):
    """
    boxes: list of [x1,y1,x2,y2]
    scores: list of confidence scores
    return indices_to_keep
    """
    if len(boxes) == 0:
        return []
    boxes = np.array(boxes, dtype=float)
    scores = np.array(scores, dtype=float)
    idxs = np.argsort(scores)[::-1]
    keep = []
    while len(idxs) > 0:
        i = idxs[0]
        keep.append(i)
        if len(idxs) == 1:
            break
        rest = idxs[1:]
        ious = []
        for j in rest:
            x1 = max(boxes[i][0], boxes[j][0])
            y1 = max(boxes[i][1], boxes[j][1])
            x2 = min(boxes[i][2], boxes[j][2])
            y2 = min(boxes[i][3], boxes[j][3])
            inter_w = max(0, x2 - x1)
            inter_h = max(0, y2 - y1)
            inter = inter_w * inter_h
            area_i = (boxes[i][2] - boxes[i][0]) * (boxes[i][3] - boxes[i][1])
            area_j = (boxes[j][2] - boxes[j][0]) * (boxes[j][3] - boxes[j][1])
            union = area_i + area_j - inter
            iou = inter / union if union > 0 else 0
            ious.append(iou)
        # keep those with IoU < threshold
        idxs = [rest[k] for k, v in enumerate(ious) if v < iou_threshold]
    return keep

# 3) Confidence Thresholding
def confidence_filter(predictions, threshold=0.5):
    """
    predictions: list of dict {'confidence': float, 'label': int, 'box': [x1,y1,x2,y2] (opt)}
    return filtered list
    """
    return [p for p in predictions if p.get('confidence', 0) >= threshold]

# 4) Counting Algorithm
def count_objects(predictions, threshold=0.5):
    """
    Count số lượng objects có confidence >= threshold
    """
    return sum(1 for p in predictions if p.get('confidence', 0) >= threshold)

# ===================== TRAINER =====================
class PotatoQualityTrainer:
    def __init__(self, data_root, output_dir='./potato_quality_models', logger=None):
        self.data_root = Path(data_root)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logger or setup_logging(self.output_dir)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.logger.info(f"Using device: {self.device}")

        # config
        self.config = {
            'batch_size': 32,
            'learning_rate': 1e-4,
            'num_epochs': 50,
            'weight_decay': 1e-4,
            'step_size': 10,
            'gamma': 0.1,
            'num_classes': 2,
            'input_size': (224, 224),
            'early_stopping_patience': 8
        }
        self.model = None
        self.train_loader = None
        self.val_loader = None
        self.test_loader = None
        self.criterion = None
        self.optimizer = None
        self.scheduler = None

    def prepare_data(self):
        self.logger.info("Preparing data...")
        train_transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomCrop(self.config['input_size']),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
        val_transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.CenterCrop(self.config['input_size']),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
        train_dataset = PotatoQualityDataset(self.data_root, train_transform, 'train')
        val_dataset = PotatoQualityDataset(self.data_root, val_transform, 'val')
        test_dataset = PotatoQualityDataset(self.data_root, val_transform, 'test')
        self.train_loader = DataLoader(train_dataset, batch_size=self.config['batch_size'],
                                       shuffle=True, num_workers=4, pin_memory=True)
        self.val_loader = DataLoader(val_dataset, batch_size=self.config['batch_size'],
                                     shuffle=False, num_workers=4, pin_memory=True)
        self.test_loader = DataLoader(test_dataset, batch_size=self.config['batch_size'],
                                      shuffle=False, num_workers=4, pin_memory=True)
        self.logger.info(f"Train samples: {len(train_dataset)} | Val samples: {len(val_dataset)} | Test samples: {len(test_dataset)}")

    def build_model(self):
        self.logger.info("Building model (ResNet50)...")
        self.model = PotatoQualityModel(num_classes=self.config['num_classes'], pretrained=True).to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.config['learning_rate'],
                                    weight_decay=self.config['weight_decay'])
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=self.config['step_size'], gamma=self.config['gamma'])

    def train_epoch(self):
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        for images, labels in self.train_loader:
            images, labels = images.to(self.device), labels.to(self.device)
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()
            running_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
        return running_loss / max(1, len(self.train_loader)), correct / max(1, total)

    def validate_epoch(self):
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in self.val_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                running_loss += loss.item()
                _, preds = torch.max(outputs, 1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
        return running_loss / max(1, len(self.val_loader)), correct / max(1, total)

    def train(self):
        self.logger.info("Start training...")
        best_val_acc = 0.0
        patience = 0
        train_losses, val_losses, train_accs, val_accs = [], [], [], []
        for epoch in range(self.config['num_epochs']):
            self.logger.info(f"Epoch {epoch+1}/{self.config['num_epochs']}")
            tr_loss, tr_acc = self.train_epoch()
            val_loss, val_acc = self.validate_epoch()
            self.scheduler.step()
            train_losses.append(tr_loss); val_losses.append(val_loss)
            train_accs.append(tr_acc); val_accs.append(val_acc)
            self.logger.info(f"Train Loss: {tr_loss:.4f}, Acc: {tr_acc:.4f} | Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f}")
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                patience = 0
                self.save_model('best_model.pth', epoch, val_acc)
            else:
                patience += 1
            if patience >= self.config['early_stopping_patience']:
                self.logger.info("Early stopping triggered.")
                break
        self.save_model('final_model.pth', epoch, val_acc)
        # Vẽ đồ thị
        self.plot_training_curves(train_losses, val_losses, train_accs, val_accs)
        self.logger.info(f"Training finished. Best val acc: {best_val_acc:.4f}")

    def save_model(self, filename, epoch, val_acc):
        path = self.output_dir / filename
        torch.save({'epoch': epoch, 'state_dict': self.model.state_dict(), 'val_acc': val_acc, 'config': self.config}, path)
        # lưu info
        info = {'epoch': int(epoch), 'val_acc': float(val_acc), 'timestamp': datetime.now().isoformat()}
        with open(self.output_dir / f"{filename.replace('.pth','')}_info.json", 'w') as f:
            json.dump(info, f, indent=2)
        self.logger.info(f"Saved model: {path}")

    def plot_training_curves(self, train_losses, val_losses, train_accs, val_accs):
        plt.figure(figsize=(12,5))
        plt.subplot(1,2,1)
        plt.plot(train_losses, label='Train Loss'); plt.plot(val_losses, label='Val Loss')
        plt.title('Loss'); plt.legend()
        plt.subplot(1,2,2)
        plt.plot(train_accs, label='Train Acc'); plt.plot(val_accs, label='Val Acc')
        plt.title('Accuracy'); plt.legend()
        plt.tight_layout()
        plt.savefig(self.output_dir / 'training_curves.png', dpi=200)
        plt.close()

    def evaluate_model(self, model_path=None, conf_threshold=0.5, do_algo_post=True):
        if model_path:
            ckpt = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(ckpt['state_dict'])
            self.logger.info(f"Loaded model from {model_path}")
        self.model.eval()
        preds, labels_all, confidences = [], [], []
        with torch.no_grad():
            for imgs, labels in self.test_loader:
                imgs, labels = imgs.to(self.device), labels.to(self.device)
                out = self.model(imgs)
                probs = torch.softmax(out, dim=1)
                conf, predicted = torch.max(probs, 1)
                preds.extend(predicted.cpu().numpy().tolist())
                confidences.extend(conf.cpu().numpy().tolist())
                labels_all.extend(labels.cpu().numpy().tolist())
        # Báo cáo classification
        acc = float(np.mean(np.array(preds) == np.array(labels_all)))
        class_names = ['Good','Bad']
        report = classification_report(labels_all, preds, target_names=class_names)
        cm = confusion_matrix(labels_all, preds)
        self.logger.info("Classification Report:\n" + report)
        # vẽ confusion matrix
        plt.figure(figsize=(6,5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
        plt.title('Confusion Matrix'); plt.savefig(self.output_dir / 'confusion_matrix.png'); plt.close()
        # Nếu yêu cầu, áp dụng thuật toán hậu xử lý: confidence thresholding & counting
        predictions = [{'confidence': c, 'label': p} for c,p in zip(confidences, preds)]
        if do_algo_post:
            filtered = confidence_filter(predictions, threshold=conf_threshold)
            total_all = count_objects(predictions, threshold=conf_threshold)
            total_filtered = len(filtered)
            self.logger.info(f"Total test samples: {len(predictions)}")
            self.logger.info(f"Total with confidence >= {conf_threshold}: {total_filtered}")
        return acc, report, cm

# ===================== YOLOv8 UTILITY (TÙY CHỌN) =====================
def run_yolo_demo(image_path, yolo_model_path=None, conf=0.25, iou=0.45, use_ultralytics=HAS_ULTRALYTICS):
    """
    Nếu ultralytics có sẵn, dùng YOLOv8 để detect và count.
    Nếu không, hàm trả về thông báo ko cài đặt.
    """
    if not use_ultralytics:
        print("Ultralytics (YOLOv8) không được cài. Cài bằng 'pip install ultralytics' để dùng chức năng này.")
        return None
    # load model - nếu có model checkpoint dùng model_path, ngược lại dùng pretrain tiny
    model = YOLO(yolo_model_path or 'yolov8n.pt')
    results = model.predict(source=image_path, conf=conf, iou=iou, save=False)
    # results là list, mỗi item có .boxes, .masks (nếu segmentation)
    r = results[0]
    boxes = []
    scores = []
    labels = []
    try:
        boxes_xyxy = r.boxes.xyxy.cpu().numpy()  # shape (N,4)
        scores = r.boxes.conf.cpu().numpy().tolist()
        labels = r.boxes.cls.cpu().numpy().tolist()
        boxes = boxes_xyxy.tolist()
    except Exception:
        boxes = []; scores = []; labels = []
    # apply NMS (r đã có nms nội bộ nhưng ta minh hoạ)
    keep_idx = non_max_suppression(boxes, scores, iou_threshold=iou)
    kept = [{'box': boxes[i], 'score': scores[i], 'label': labels[i]} for i in keep_idx]
    count = len(kept)
    print(f"YOLO: phát hiện {len(boxes)} boxes, sau NMS giữ {count} boxes")
    return kept

# ===================== DATASET STRUCTURE CREATOR =====================
def create_sample_dataset_structure(base_path):
    base_path = Path(base_path)
    for split in ['train','val','test']:
        for cls in ['good','bad']:
            (base_path / split / cls).mkdir(parents=True, exist_ok=True)
    with open(base_path / 'README.md','w',encoding='utf-8') as f:
        f.write("""
# Potato Quality Dataset (2 Classes)
- good/: Khoai tây tốt
- bad/: Khoai tây kém
Recommended split: 70% train, 15% val, 15% test
Image format: .jpg / .png / .jpeg
""")
    print(f"Dataset folders created at: {base_path}")

# ===================== MAIN CLI =====================
def main():
    parser = argparse.ArgumentParser(description="Train & demo đầy đủ (ResNet + thuật toán YOLO-like)")
    parser.add_argument('--data_root', type=str, required=True, help='Path to dataset root')
    parser.add_argument('--output_dir', type=str, default='./potato_quality_models', help='Output dir')
    parser.add_argument('--create_structure', action='store_true', help='Create sample dataset structure')
    parser.add_argument('--train_resnet', action='store_true', help='Train ResNet classifier')
    parser.add_argument('--evaluate', action='store_true', help='Evaluate ResNet on test set')
    parser.add_argument('--yolo_demo', type=str, default=None, help='Run YOLOv8 demo on image (path). Requires ultralytics.')
    parser.add_argument('--yolo_model', type=str, default=None, help='YOLOv8 model path (optional)')
    parser.add_argument('--conf_thresh', type=float, default=0.5, help='Confidence threshold for post-processing/counting')
    args = parser.parse_args()

    logger = setup_logging(args.output_dir)

    if args.create_structure:
        create_sample_dataset_structure(args.data_root)
        return

    trainer = PotatoQualityTrainer(args.data_root, args.output_dir, logger=logger)
    trainer.prepare_data()
    trainer.build_model()

    if args.train_resnet:
        trainer.train()

    if args.evaluate:
        best_model = Path(args.output_dir) / 'best_model.pth'
        model_to_eval = str(best_model) if best_model.exists() else None
        acc, report, cm = trainer.evaluate_model(model_to_eval, conf_threshold=args.conf_thresh)
        print(f"Eval acc: {acc:.4f}")

    if args.yolo_demo:
        kept = run_yolo_demo(args.yolo_demo, yolo_model_path=args.yolo_model, conf=args.conf_thresh)
        if kept is not None:
            for i, item in enumerate(kept):
                box = item['box']
                score = item['score']
                print(f"#{i+1}: box={box}, score={score:.3f}")

if __name__ == "__main__":
    main()
