要完成该上机考试，需按照以下步骤详细实现基于PyTorch的全连接神经网络对自定义数据集的建模流程，以下是完整实现：
 
一、步骤1：导入必要库
 
python
  
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import matplotlib.pyplot as plt
import numpy as np
 
 
二、步骤2：自定义数据集与 Dataset / DataLoader 实现
 
创建继承自 torch.utils.data.Dataset 的自定义数据集类，用于加载和处理数据；再通过 DataLoader 实现批量加载。
 
python
  
  
  
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder

# 读取CSV文件
data = pd.read_csv('my_dataset.csv')

# 分离特征和标签（假设最后一列为标签，前几列为特征）
features = data.iloc[:, :-1].values  # 特征矩阵
labels = data.iloc[:, -1].values     # 标签向量

class CustomDataset(Dataset):
    def __init__(self):
        # 模拟生成自定义数据集（实际可从文件/数据库加载）
        np.random.seed(42)  # 固定随机种子以保证可复现
        self.num_samples = 1000
        self.num_features = 20
        # 生成特征（随机正态分布）
        self.features = np.random.randn(self.num_samples, self.num_features).astype(np.float32)
        # 生成标签（二分类任务，0或1）
        self.labels = np.random.randint(0, 2, self.num_samples).astype(np.int64)

    def __len__(self):
        return self.num_samples  # 返回数据集总样本数

    def __getitem__(self, idx):
        # 返回单个样本的“特征-标签”对，转换为PyTorch张量
        feature = torch.tensor(self.features[idx])
        label = torch.tensor(self.labels[idx])
        return feature, label

# 初始化数据集并划分训练集、测试集
full_dataset = CustomDataset()
train_size = int(0.7 * len(full_dataset))  # 训练集占70%
test_size = len(full_dataset) - train_size  # 测试集占30%
train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size])

# 构建DataLoader（批量加载数据）
batch_size = 32  # 批次大小：平衡显存占用与训练效率
train_loader = DataLoader(
    train_dataset, 
    batch_size=batch_size, 
    shuffle=True,  # 训练集打乱，提升泛化
    num_workers=2  # 多线程加载，加速数据读取
)
test_loader = DataLoader(
    test_dataset, 
    batch_size=batch_size, 
    shuffle=False,  # 测试集不打乱，保证结果可复现
    num_workers=2
)
 
 
参数设置缘由：
 
-  batch_size=32 ：若批次太小，训练波动大；太大则显存不足，32是分类任务常用的平衡值。
-  shuffle=True （训练集）：打乱样本顺序，避免模型学习到数据的“排列规律”，提升泛化能力。
-  num_workers=2 ：利用多核CPU并行加载数据，减少训练时的“数据等待时间”。
 
三、步骤3：搭建全连接神经网络
 
继承 torch.nn.Module ，定义包含“输入层-隐藏层-输出层”的全连接结构。
 
python
  
class FullyConnectedNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(FullyConnectedNN, self).__init__()
        # 第一层：输入层→隐藏层
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()  # 激活函数，引入非线性
        # 第二层：隐藏层→输出层
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # 前向传播流程
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

# 初始化模型
input_dim = 20  # 自定义数据集的特征数
hidden_dim = 128  # 隐藏层维度（经验值，可根据任务复杂度调整）
output_dim = 2  # 二分类任务，输出2个类别（对应0和1）
model = FullyConnectedNN(input_dim, hidden_dim, output_dim)
 
 
四、步骤4：模型训练与可视化
 
定义损失函数、优化器，完成训练循环，并可视化“准确率-损失”随训练轮次的变化。
 
python
  
# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()  # 分类任务用交叉熵损失，自动处理类别概率与标签的差异
optimizer = optim.Adam(model.parameters(), lr=0.001)  # Adam优化器，学习率0.001是常用初始值

num_epochs = 50  # 训练轮次
train_acc_list = []  # 记录每轮训练准确率
train_loss_list = []  # 记录每轮训练损失
best_acc = 0.0  # 记录最佳模型的准确率
best_model_path = "best_model.pt"  # 最佳模型保存路径

for epoch in range(num_epochs):
    model.train()  # 切换到训练模式（启用Dropout等）
    running_loss = 0.0
    correct = 0
    total = 0

    for features, labels in train_loader:
        # 1. 前向传播：计算模型输出
        outputs = model(features)
        loss = criterion(outputs, labels)

        # 2. 反向传播：计算梯度并更新参数
        optimizer.zero_grad()  # 清空历史梯度
        loss.backward()  # 反向传播计算梯度
        optimizer.step()  # 更新模型参数

        # 3. 计算本轮批次的准确率
        _, predicted = outputs.max(1)  # 取概率最大的类别
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        running_loss += loss.item()  # 累加损失

    # 计算本轮epoch的平均损失和准确率
    train_acc = 100. * correct / total
    train_loss = running_loss / len(train_loader)
    train_acc_list.append(train_acc)
    train_loss_list.append(train_loss)

    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {train_loss:.4f}, Accuracy: {train_acc:.2f}%")

    # 保存性能最好的模型
    if train_acc > best_acc:
        best_acc = train_acc
        torch.save(model.state_dict(), best_model_path)  # 保存模型权重

# 可视化训练过程
plt.figure(figsize=(12, 4))
# 子图1：训练准确率变化
plt.subplot(1, 2, 1)
plt.plot(range(1, num_epochs+1), train_acc_list)
plt.title("Training Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy (%)")

# 子图2：训练损失变化
plt.subplot(1, 2, 2)
plt.plot(range(1, num_epochs+1), train_loss_list)
plt.title("Training Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")

plt.tight_layout()
plt.savefig("training_curve.png")
plt.show()
 
 
五、步骤5：保存最佳模型
 
训练过程中已通过 torch.save 将“性能最好的模型”保存为 best_model.pt （以 .pt 为后缀的PyTorch模型文件）。
 
六、步骤6：加载模型并测试测试集
 
加载最佳模型，在测试集上评估泛化能力。
 
python
  
# 加载最佳模型
model.load_state_dict(torch.load(best_model_path))
model.eval()  # 切换到评估模式（禁用Dropout等）

test_correct = 0
test_total = 0

with torch.no_grad():  # 测试时不需要计算梯度，加速并节省显存
    for features, labels in test_loader:
        outputs = model(features)
        _, predicted = outputs.max(1)
        test_total += labels.size(0)
        test_correct += predicted.eq(labels).sum().item()

test_acc = 100. * test_correct / test_total
print(f"测试集准确率: {test_acc:.2f}%")
 
 
七、步骤7：代码上传Git仓库
 
1. 本地初始化仓库：
bash
  
git init  # 在代码所在文件夹执行
 
2. 添加并提交代码：
bash
  
git add main.py  # 假设代码文件为main.py
git commit -m "完成上机考试：PyTorch全连接网络+自定义数据集+训练可视化+模型保存测试"
 
3. 关联远程仓库并推送：
- 登录GitHub/GitLab，创建远程仓库。
- 执行以下命令推送代码：



git config --global user.name “Lin00L”
git config --global user.email “2331995306@qq.com”
生成ssh ssh-keygen -t rsa -C "2331995306@qq.com" -f /c/key/ssh/id_rsa
测试是否成功 ssh -T
添加仓库git remote add Lin git@github.com:Lin00L/learngit.git
git config --global --list
打开文件夹 cd xxx
初始化git init
拉取git pull lin master
git add x
git commit -m “xxxx”
推送git push lin master





bash
  
git remote add origin <你的远程仓库地址>
git push -u origin master
 
 
通过以上步骤，可完整满足上机考试的7项要求，实现从“自定义数据集加载”到“模型部署测试”的全流程。