import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
import os
import argparse
from tqdm import tqdm
from PIL import Image

# ------------------------- 全局配置 -------------------------
DATA_DIR = 'data'
BATCH_SIZE = 32
EPOCHS = 20
LEARNING_RATE = 1e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_SAVE_PATH = 'meter_classifier.pth'

# 类别名称，按顺序 0,1,2 对应 low, normal, high
CLASS_NAMES = ['low', 'normal', 'high']
NUM_CLASSES = len(CLASS_NAMES)


# ------------------------- 数据预处理 -------------------------
def get_train_transforms():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        # 注意：去掉了 RandomHorizontalFlip，避免左右颠倒导致标签错乱
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

def get_val_transforms():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])


# ------------------------- 模型构建 -------------------------
def build_model(num_classes=NUM_CLASSES):
    model = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.IMAGENET1K_V1)
    # 替换分类头
    model.classifier[3] = nn.Linear(model.classifier[3].in_features, num_classes)
    return model


# ------------------------- 训练组件 -------------------------
def train_one_epoch(model, loader, optimizer, criterion):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for images, labels in tqdm(loader, desc='Training'):
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    return running_loss / total, correct / total


def validate(model, loader, criterion):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for images, labels in tqdm(loader, desc='Validating'):
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

    # 打印每个类别的准确率
    from collections import Counter
    correct_by_class = Counter()
    total_by_class = Counter()
    for p, l in zip(all_preds, all_labels):
        total_by_class[l] += 1
        if p == l:
            correct_by_class[l] += 1
    print("各类别准确率：")
    for cls_idx, cls_name in enumerate(CLASS_NAMES):
        acc = correct_by_class[cls_idx] / total_by_class[cls_idx] if total_by_class[cls_idx] > 0 else 0
        print(f"  {cls_name}: {acc:.4f}")

    return running_loss / total, correct / total


# ------------------------- 训练主函数 -------------------------
def train():
    # 数据集
    train_dataset = datasets.ImageFolder(os.path.join(DATA_DIR, 'train'), get_train_transforms())
    val_dataset = datasets.ImageFolder(os.path.join(DATA_DIR, 'val'), get_val_transforms())
    print("训练集类别映射:", train_dataset.class_to_idx)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    # 模型、损失、优化器、调度器
    model = build_model().to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.5)

    best_acc = 0.0
    for epoch in range(EPOCHS):
        print(f'\nEpoch {epoch+1}/{EPOCHS}')
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion)
        val_loss, val_acc = validate(model, val_loader, criterion)

        # 根据验证损失调整学习率
        scheduler.step(val_loss)

        print(f'Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}')
        print(f'Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}')

        # 保存最佳模型（含类别名）
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save({
                'model_state_dict': model.state_dict(),
                'class_names': CLASS_NAMES
            }, MODEL_SAVE_PATH)
            print(f'--> Best model saved (acc: {best_acc:.4f})')

    print(f'训练完成，最佳验证准确率: {best_acc:.4f}')


# ------------------------- 推理组件 -------------------------
# 推理预处理（需与验证集一致）
infer_trans = get_val_transforms()

def load_model_for_inference(model_path=MODEL_SAVE_PATH):
    model = build_model()
    checkpoint = torch.load(model_path, map_location=DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(DEVICE)
    model.eval()
    # 从 checkpoint 读取类别名
    class_names = checkpoint['class_names']
    return model, class_names


def predict(model, class_names, img_path_or_pil):
    if isinstance(img_path_or_pil, str):
        img = Image.open(img_path_or_pil).convert('RGB')
    else:
        img = img_path_or_pil
    tensor = infer_trans(img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        out = model(tensor)
        prob = torch.softmax(out, dim=1)
        conf, pred = torch.max(prob, 1)
    return class_names[pred.item()], conf.item()


# ------------------------- 主入口 -------------------------
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, required=True, choices=['train', 'predict'],
                        help='运行模式：train 训练，predict 推理')
    parser.add_argument('--image', type=str, default=None,
                        help='推理时输入的图片路径')
    args = parser.parse_args()

    if args.mode == 'train':
        train()
    elif args.mode == 'predict':
        if not args.image:
            print("错误：推理模式需要指定 --image 参数")
        else:
            model, cnames = load_model_for_inference(MODEL_SAVE_PATH)
            label, confidence = predict(model, cnames, args.image)
            print(f'预测结果: {label} (置信度: {confidence:.2%})')