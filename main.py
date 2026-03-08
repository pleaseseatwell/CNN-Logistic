import torch
from torchvision import transforms, datasets
import torch.nn as nn   #神经网络模块
from torch.utils.data import DataLoader, random_split
from PIL import Image   #打开处理图像文件
train_transforms = transforms.Compose([    #训练集
    transforms.Resize((224, 224)),  # 统一尺寸，将图像调整为固定的224*224
    transforms.RandomRotation(10),  # 数据增强（随机旋转图像十度，增加多样性）
    transforms.RandomHorizontalFlip(),    #以50%的概率翻转图像
    transforms.ToTensor(),   #转化为Pytorch张量，便于使用GPU加强计算
    transforms.Normalize(mean=[0.485, 0.456, 0.406],  # ImageNet统计参数
                         std=[0.229, 0.224, 0.225])
])

test_transforms = transforms.Compose([    #验证集，与训练集的差别在于没有数据增强
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# 加载数据集
dataset = datasets.ImageFolder(root='data/train/TestA', transform=train_transforms)

# 划分训练集和验证集
train_size = int(0.8 * len(dataset))             #80%的训练集
val_size = len(dataset) - train_size		#20%的验证集
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

# 创建DataLoader，用于批量加载数据
batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)  # 修改num_workers为0
val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=0)  # 修改num_workers为0


class LeukocyteCNN(nn.Module):    #定义模型
    def __init__(self, num_classes=5):    #默认输出5个WBC类别
        super(LeukocyteCNN, self).__init__()

        # 特征提取层（保持不变）
        self.features = nn.Sequential(    #特征提取层，包含4个卷积块
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
        )  #将特征展开为一维向量输入到logistic模型中

        self.feature_outputs = []

    def forward(self, x):
        x = self.features(x)
        self.feature_outputs.append(x.detach().cpu().numpy())
        x = self.classifier(x)  # 输出原始logits
        return x

    # 其他方法保持不变...
    def save_model(self, path):
        #保存模型参数
        torch.save(self.state_dict(), path)
        print(f"模型参数已保存到 {path}")

    def load_model(self, path):
        #加载模型参数
        self.load_state_dict(torch.load(path))
        print(f"模型参数已从 {path} 加载")
    def predict(self, image_input, device=None):    #预测函数
 
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
