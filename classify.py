import torch
from ultralytics import YOLO
from config import tag_config, pestv3_config
from commons.dataset import ClassifyDataset
from old_module.DisNet import DisNet
from torch import nn
from torch.utils.data import DataLoader, random_split, WeightedRandomSampler
from sklearn.utils.class_weight import compute_class_weight
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from sklearn.model_selection import KFold
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from torchvision import transforms
from torchvision.models import shufflenet_v2_x0_5, ShuffleNet_V2_X0_5_Weights, shufflenet_v2_x1_0, ShuffleNet_V2_X1_0_Weights
# Calculate weights for each class based on the frequency of samples
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
import os
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR                                                                      
from sklearn.metrics import classification_report, confusion_matrix


dataset = 'pestv3'
ratio = 70
img_dir = os.path.join('out', f'{dataset}', f'{ratio}', 'trainval')

def get_model(id, device):
    if id == 0:
        cls = DisNet().to(device)
    elif id == 1:
        cls = shufflenet_v2_x0_5(weights=ShuffleNet_V2_X0_5_Weights.DEFAULT)
        cls.fc = torch.nn.Linear(cls.fc.in_features, 2)
        cls.to(device)
    else:
        cls = shufflenet_v2_x1_0(weights=ShuffleNet_V2_X1_0_Weights.DEFAULT)
        cls.fc = torch.nn.Linear(cls.fc.in_features, 2)
        cls.to(device)
    return cls

def calculate_accuracy(outputs, labels):
    _, predicted = torch.max(outputs, 1)
    correct = (predicted == labels).sum().item()
    total = labels.size(0)
    return correct / total

def freeze_layer(model, num):
    freeze = [f'model.{x}.' for x in range(num)]
    for k, v in model.named_parameters():
        v.requires_grad = True
        if any(x in k for x in freeze):
            print(f'freezing {k}')
            v.requires_grad = False
        print(f"{num} layers are freezer.")

def calculate_accuracy(outputs, labels):
    _, predicted = torch.max(outputs, 1)
    correct = (predicted == labels).sum().item()
    total = labels.size(0)
    return correct / total

# Combined loss function
class CombinedLoss(nn.Module):
    def __init__(self, weight=None, gamma=2.0, smoothing=0.1):
        super(CombinedLoss, self).__init__()
        self.ce_loss = nn.CrossEntropyLoss(weight=weight)
        self.gamma = gamma
        self.smoothing = smoothing

    def focal_loss(self, logits, labels):
        ce_loss = nn.CrossEntropyLoss(weight=self.ce_loss.weight, reduction='none')(logits, labels)
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        return focal_loss

    def label_smoothing_loss(self, logits, labels):
        num_classes = logits.size(1)
        log_probs = torch.log_softmax(logits, dim=1)
        labels_one_hot = F.one_hot(labels, num_classes).float()
        smoothed_labels = (1 - self.smoothing) * labels_one_hot + self.smoothing / num_classes
        loss = -(smoothed_labels * log_probs).sum(dim=1)
        return loss

    def forward(self, logits, labels):
        loss_ce = self.ce_loss(logits, labels)
        loss_focal = self.focal_loss(logits, labels).mean()
        loss_smooth = self.label_smoothing_loss(logits, labels).mean()
        return 0.4 * loss_ce + 0.4 * loss_focal + 0.2 * loss_smooth

def evaluate_model_accuracy(model_path, dataset='pestv3', ratio=70):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    
    # 加载模型
    model = get_model(1, device)  # 使用训练时的相同模型ID（1表示shufflenet_v2_x0_5）
    model.load_state_dict(torch.load(model_path))
    model.eval()
    
    # 准备验证集
    val_transform = transforms.Compose([
        transforms.Resize((640, 640)),
        transforms.ToTensor(),
    ])
    val_dataset = ClassifyDataset(img_dir=os.path.join('out', dataset, str(ratio), 'trainval'), 
                                transform=val_transform, train=False)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=16)
    
    # 计算分类精度
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)  # 获取预测类别
            
            correct += (preds == labels).sum().item()  # 统计正确预测数
            total += labels.size(0)  # 统计总样本数
    
    accuracy = correct / total  # 计算分类精度
    print(f"Validation Accuracy: {accuracy:.4f} ({correct}/{total})")
    return accuracy


def main():
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # Define transformations with data augmentation
    train_transform = transforms.Compose([
        transforms.Resize((640, 640)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(p=0.3),  # 随机垂直翻转
        # transforms.RandomGrayscale(p=0.1),  # 随机灰度处理
        # transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.RandomRotation(15),
        # transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        transforms.ToTensor(),
        # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # 标准化
    ])

    val_transform = transforms.Compose([
        transforms.Resize((640, 640)),
        transforms.ToTensor(),
    ])

    cls = get_model(1, device)

    # Create datasets
    train_dataset = ClassifyDataset(img_dir=img_dir, transform=train_transform, train=True)
    val_dataset = ClassifyDataset(img_dir=img_dir, transform=val_transform, train=False)

    # Compute class weights and use WeightedRandomSampler
    easy_label = np.array(train_dataset.n_easy * [0])
    diff_label = np.array(train_dataset.n_diff * [1])
    all_label = np.concat((easy_label, diff_label), axis=None) 
    class_weights = compute_class_weight('balanced', classes=np.unique(all_label), y=all_label)

    sample_weights = np.array([class_weights[label] for label in all_label])
    class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)
    sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)

    trainloader = DataLoader(train_dataset, batch_size=64, sampler=sampler, num_workers=16)
    valloader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=16)

    # Loss function and optimizer
    criterion = CombinedLoss(weight=class_weights)
    optimizer = Adam(cls.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=100)

    writer = SummaryWriter(log_dir='./logs')
    num_epochs = 200
    best_val_f1 = 0
    

    for epoch in range(num_epochs):
        cls.train()
        running_loss, running_corrects, running_total = 0.0, 0, 0

        for images, labels in trainloader:
            images, labels = images.to(device), labels.to(device)
            outputs = cls(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            running_corrects += (outputs.argmax(dim=1) == labels).sum().item()
            running_total += labels.size(0)

        avg_train_loss = running_loss / len(trainloader)
        train_acc = running_corrects / running_total
        writer.add_scalar('Training Loss', avg_train_loss, epoch)
        writer.add_scalar('Training Accuracy', train_acc, epoch)

        # Validation
        cls.eval()
        val_loss, val_corrects, val_total = 0.0, 0, 0
        all_preds, all_labels = [], []

        with torch.no_grad():
            for images, labels in valloader:
                images, labels = images.to(device), labels.to(device)
                outputs = cls(images)
                loss = criterion(outputs, labels)

                val_loss += loss.item()
                val_corrects += (outputs.argmax(dim=1) == labels).sum().item()
                val_total += labels.size(0)

                _, preds = torch.max(outputs, 1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        avg_val_loss = val_loss / len(valloader)
        val_acc = val_corrects / val_total

        report = classification_report(all_labels, all_preds, output_dict=True)
        val_f1 = report['macro avg']['f1-score']

        writer.add_scalar('Validation Loss', avg_val_loss, epoch)
        writer.add_scalar('Validation Accuracy', val_acc, epoch)
        writer.add_scalar('Validation F1', val_f1, epoch)

        print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.4f}, Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc:.4f}, Val F1: {val_f1:.4f}")

        scheduler.step()

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            os.makedirs('checkpoint/classifier', exist_ok=True)
            torch.save(cls.state_dict(), os.path.join('checkpoint/classifier', f'{dataset}', f'{ratio}', 'best_model.pth'))

    writer.close()

if __name__ == '__main__':
    # main()
    for r in [30, 40, 50, 60, 70]:
        model_path = os.path.join('checkpoint/classifier', 'pestv3', f'{r}', 'best_model.pth')
        val_accuracy = evaluate_model_accuracy(model_path)
        print(f"ratio: {r}, acc: {val_accuracy}")