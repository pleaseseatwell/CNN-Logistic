import torch
from torchvision import transforms, datasets
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from PIL import Image
train_transforms = transforms.Compose([
    transforms.Resize((224, 224)),  # 统一尺寸
    transforms.RandomRotation(10),  # 数据增强
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],  # ImageNet统计参数
                         std=[0.229, 0.224, 0.225])
])

test_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# 加载数据集（假设目录结构为：data/train/class_x/*.jpg）
dataset = datasets.ImageFolder(root='data/train/TestA', transform=train_transforms)

# 划分训练集和验证集
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

# 创建DataLoader
batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)  # 修改num_workers为0
val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=0)  # 修改num_workers为0


class LeukocyteCNN(nn.Module):
    def __init__(self, num_classes=5):
        super(LeukocyteCNN, self).__init__()

        # 特征提取层（保持不变）
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )

        # 修改后的分类层（Logistic回归）
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512 * 14 * 14, num_classes)  # 直接映射到类别数
        )

        self.feature_outputs = []

    def forward(self, x):
        x = self.features(x)
        self.feature_outputs.append(x.detach().cpu().numpy())
        x = self.classifier(x)  # 输出原始logits
        return x

    # 其他方法保持不变...
    def save_model(self, path):
        """保存模型参数"""
        torch.save(self.state_dict(), path)
        print(f"模型参数已保存到 {path}")

    def load_model(self, path):
        """加载模型参数"""
        self.load_state_dict(torch.load(path))
        print(f"模型参数已从 {path} 加载")
    def predict(self, image_input, device=None):
        """
        预测输入图像的类别

        参数:
            image_input: 可以是以下形式之一:
                - PIL.Image 对象
                - 图像文件路径 (str)
                - 已经预处理过的torch.Tensor (形状为 [C, H, W] 或 [B, C, H, W])
            device: 指定运行设备 (None表示自动选择)

        返回:
            如果是单张图像: 返回预测类别(int)和类别概率(torch.Tensor)
            如果是批量图像: 返回预测类别列表(List[int])和类别概率(torch.Tensor)
        """
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(device)
        self.eval()

        # 预处理输入
        if isinstance(image_input, str):  # 文件路径
            image = Image.open(image_input).convert('RGB')
            input_tensor = test_transforms(image).unsqueeze(0)  # 增加batch维度
        elif isinstance(image_input, Image.Image):  # PIL图像
            input_tensor = test_transforms(image_input).unsqueeze(0)
        elif isinstance(image_input, torch.Tensor):
            if image_input.dim() == 3:  # 单张图像 [C, H, W]
                input_tensor = image_input.unsqueeze(0)
            else:  # 已经是批量 [B, C, H, W]
                input_tensor = image_input
        else:
            raise ValueError("不支持的输入类型，请提供PIL图像、文件路径或Tensor")

        input_tensor = input_tensor.to(device)

        with torch.no_grad():
            outputs = self(input_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            _, predicted = torch.max(probabilities, 1)

        probs = probabilities[0] if input_tensor.shape[0] == 1 else probabilities
        # 根据输入类型返回适当格式
        if input_tensor.shape[0] == 1:  # 单张图像
            return predicted.item(), probs[0]
        else:  # 批量图像
            return predicted.cpu().numpy().tolist(), probs
