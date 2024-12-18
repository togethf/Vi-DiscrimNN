import torch
from ultralytics import YOLO
from config import tag_config, pestv3_config
from commons.dataset import ClassifyDataset
from torch.utils.data import DataLoader, random_split
from sklearn.utils.class_weight import compute_class_weight
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from torchvision import transforms
from torchvision.models import shufflenet_v2_x0_5, ShuffleNet_V2_X0_5_Weights, shufflenet_v2_x1_0, ShuffleNet_V2_X1_0_Weights
# Calculate weights for each class based on the frequency of samples
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
import os
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR                                                                      
from sklearn.metrics import classification_report, confusion_matrix

class DiscrimNN(torch.nn.Module):
    def __init__(self, yolo):
        super(DiscrimNN, self).__init__()
        self.backbone = yolo.model.model[:8]  # 提取YOLO的前8层作为骨干网络
        
        # 分类头
        self.conv1 = torch.nn.Conv2d(256, 128, kernel_size=3, padding=1)
        self.bn1 = torch.nn.BatchNorm2d(128)
        self.conv2 = torch.nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.bn2 = torch.nn.BatchNorm2d(64)
        self.conv3 = torch.nn.Conv2d(64, 20, kernel_size=1)  # 1x1 卷积层，输出通道数为类别数
        self.activation = torch.nn.ReLU()
        self.dropout = torch.nn.Dropout2d(0.5)

        # self.linear1 = torch.nn.Linear(256, 128)
        # self.activation = torch.nn.ReLU()
        # self.dropout = torch.nn.Dropout(0.5)
        # self.linear2 = torch.nn.Linear(128, 64)
        # self.linear3 = torch.nn.Linear(64, 20)

    def forward(self, x):
        x = self.backbone(x)
        
        # 分类头
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.activation(x)
        x = self.dropout(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.activation(x)
        # x = self.dropout(x)
        
        x = self.conv3(x)
        
        # 全局平均池化
        x = torch.nn.functional.adaptive_avg_pool2d(x, (1, 1))
        x = x.view(x.size(0), -1)  # 展平以适应全连接层

        # x = self.linear1(x)
        # x = self.activation(x)
        # # x = self.dropout(x)
        # x = self.linear2(x)
        # x = self.activation(x)
        # x = self.linear3(x)
        
        return x

def get_model(id, device):
    if id == 0:
        yolo = YOLO(pestv3_config['strong_detector'])
        # 冻结骨干网络参数
        # freeze_layer(yolo, 11)
        cls = DiscrimNN(yolo).to(device)
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

def main():
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    # Define transformations for training and validation
    train_transform = transforms.Compose([
        transforms.Resize((640, 640)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    val_transform = transforms.Compose([
        transforms.Resize((640, 640)),
        transforms.ToTensor(),
        # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    cls = get_model(1, device)
    # Create datasets
    train_dataset = ClassifyDataset(img_dir=tag_config['output_dir'], transform=train_transform, train=True)
    val_dataset = ClassifyDataset(img_dir=tag_config['output_dir'], transform=val_transform, train=False)

    trainloader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=8)
    valloader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=8)


    # 损失函数与优化器
    # Assuming you have labels as a list or numpy array
    # class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    class_weights = np.array([1, 4])
    class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)
    criterion = CrossEntropyLoss(weight=class_weights)
    optimizer = AdamW(cls.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=10)

    writer = SummaryWriter(log_dir='./logs')
    num_epochs = 300
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

        # 验证
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
            torch.save(cls.state_dict(), os.path.join('checkpoint', 'classifier', 'best_model.pth'))

    writer.close()
if __name__ == '__main__':
    main()
