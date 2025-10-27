import torch
from torch.utils.data import Dataset, random_split, DataLoader
import pandas as pd
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import matplotlib.pyplot as plt


# 定义乳腺癌数据集类
class BreastCancerDataset(Dataset):
    def __init__(self, csv_path):
        # 读取 CSV 文件（请替换为你的数据实际路径）
        data = pd.read_csv(csv_path)

        # 排除 id 列和 Unnamed: 32 列
        data = data.drop(['id', 'Unnamed: 32'], axis=1, errors='ignore')

        # 将 diagnosis 列进行编码，M（恶性）为 0，B（良性）为 1
        data['diagnosis'] = data['diagnosis'].map({'M': 0, 'B': 1})

        # 清洗数据，删除包含空值的行
        data = data.dropna()

        # 提取特征和标签
        self.features = torch.tensor(data.drop('diagnosis', axis=1).values, dtype=torch.float32)
        self.labels = torch.tensor(data['diagnosis'].values, dtype=torch.long)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


# 定义神经网络模型
class BreastCancerNet(nn.Module):
    def __init__(self, input_size):
        super(BreastCancerNet, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.3)  # 添加dropout防止过拟合
        self.fc2 = nn.Linear(64, 32)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(0.3)
        self.fc3 = nn.Linear(32, 2)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu1(out)
        out = self.dropout1(out)
        out = self.fc2(out)
        out = self.relu2(out)
        out = self.dropout2(out)
        out = self.fc3(out)
        return out


# 计算准确率的函数
def calculate_accuracy(loader, model):
    model.eval()  # 切换到评估模式
    correct = 0
    total = 0
    with torch.no_grad():
        for features, labels in loader:
            outputs = model(features)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total
    model.train()  # 切换回训练模式
    return accuracy


# 加载数据集（请替换为你的数据路径）
dataset = BreastCancerDataset('data.csv')  # 假设数据文件与代码同目录

# 拆分训练集和测试集
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

# 构建数据加载器
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# 初始化模型
input_size = dataset.features.shape[1]
model = BreastCancerNet(input_size)

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = StepLR(optimizer, step_size=10, gamma=0.1)

# 训练过程记录
num_epochs = 30
train_losses = []
train_accuracies = []
test_accuracies = []

# 开始训练
model.train()
for epoch in range(num_epochs):
    running_loss = 0.0
    for batch_features, batch_labels in train_loader:
        optimizer.zero_grad()

        outputs = model(batch_features)
        loss = criterion(outputs, batch_labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    # 计算当前epoch的损失和准确率
    epoch_loss = running_loss / len(train_loader)
    train_acc = calculate_accuracy(train_loader, model)
    test_acc = calculate_accuracy(test_loader, model)

    # 记录指标
    train_losses.append(epoch_loss)
    train_accuracies.append(train_acc)
    test_accuracies.append(test_acc)

    # 打印信息
    print(f'Epoch {epoch + 1}/{num_epochs}')
    print(f'Loss: {epoch_loss:.4f} | 训练准确率: {train_acc:.2f}% | 测试准确率: {test_acc:.2f}%')
    scheduler.step()

# 最终测试集准确率
final_test_acc = calculate_accuracy(test_loader, model)
print(f'\n最终测试集准确率: {final_test_acc:.2f}%')

# 绘制损失曲线和准确率曲线
plt.figure(figsize=(12, 5))

# 损失曲线
plt.subplot(1, 2, 1)
plt.plot(range(1, num_epochs + 1), train_losses, label='训练损失')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('训练损失曲线')
plt.legend()

# 准确率曲线
plt.subplot(1, 2, 2)
plt.plot(range(1, num_epochs + 1), train_accuracies, label='训练准确率')
plt.plot(range(1, num_epochs + 1), test_accuracies, label='测试准确率')
plt.xlabel('Epoch')
plt.ylabel('准确率 (%)')
plt.title('准确率曲线')
plt.legend()

plt.tight_layout()
plt.show()