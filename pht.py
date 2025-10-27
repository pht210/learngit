import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
import os
from tqdm import tqdm
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
# 设置随机种子，保证结果可复现
torch.manual_seed(42)
np.random.seed(42)


# 1. 定义数据集类
class StrokeDataset(Dataset):
    def __init__(self, features, labels):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.float32).unsqueeze(1)  # 增加维度以匹配输出

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


# 2. 定义神经网络模型
class StrokePredictor(nn.Module):
    def __init__(self, input_size):
        super(StrokePredictor, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.BatchNorm1d(32),
            nn.Dropout(0.2),
            nn.Linear(32, 1),
            nn.Sigmoid()  # 二分类问题使用sigmoid激活
        )

    def forward(self, x):
        return self.layers(x)


# 3. 数据预处理函数
def preprocess_data(df):
    # 分离特征和目标变量
    X = df.drop(['id', 'stroke'], axis=1)
    y = df['stroke'].values

    # 定义数值特征和分类特征
    numerical_features = ['age', 'avg_glucose_level', 'bmi']
    categorical_features = ['gender', 'hypertension', 'heart_disease',
                            'ever_married', 'work_type', 'Residence_type', 'smoking_status']

    # 创建预处理管道
    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),  # 用中位数填充缺失值
        ('scaler', StandardScaler())  # 标准化
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),  # 用最频繁值填充缺失值
        ('onehot', OneHotEncoder(drop='first', handle_unknown='ignore'))  # 独热编码
    ])

    # 组合所有预处理器
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features)
        ])

    # 应用预处理
    X_processed = preprocessor.fit_transform(X)
    return X_processed, y, preprocessor


# 4. 计算准确率
def calculate_accuracy(y_pred, y_true):
    y_pred_tag = torch.round(y_pred)  # 将概率转换为0或1
    correct_results_sum = (y_pred_tag == y_true).sum().float()
    acc = correct_results_sum / y_true.shape[0]
    acc = torch.round(acc * 100)  # 转换为百分比
    return acc


# 5. 训练函数
def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, device):
    # 初始化记录变量
    best_val_acc = 0.0
    train_losses = []
    train_accs = []
    val_losses = []
    val_accs = []

    # 训练循环
    for epoch in range(num_epochs):
        model.train()
        train_running_loss = 0.0
        train_running_acc = 0.0

        # 训练集迭代
        for features, labels in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}"):
            features = features.to(device)
            labels = labels.to(device)

            # 前向传播
            outputs = model(features)
            loss = criterion(outputs, labels)

            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # 计算训练损失和准确率
            train_running_loss += loss.item()
            train_running_acc += calculate_accuracy(outputs, labels).item()

        # 计算平均训练损失和准确率
        avg_train_loss = train_running_loss / len(train_loader)
        avg_train_acc = train_running_acc / len(train_loader)
        train_losses.append(avg_train_loss)
        train_accs.append(avg_train_acc)

        # 在验证集上评估
        model.eval()
        val_running_loss = 0.0
        val_running_acc = 0.0

        with torch.no_grad():  # 不计算梯度
            for features, labels in val_loader:
                features = features.to(device)
                labels = labels.to(device)

                outputs = model(features)
                loss = criterion(outputs, labels)

                val_running_loss += loss.item()
                val_running_acc += calculate_accuracy(outputs, labels).item()

        # 计算平均验证损失和准确率
        avg_val_loss = val_running_loss / len(val_loader)
        avg_val_acc = val_running_acc / len(val_loader)
        val_losses.append(avg_val_loss)
        val_accs.append(avg_val_acc)

        # 打印 epoch 结果
        print(f'Epoch {epoch + 1}/{num_epochs}')
        print(f'Train Loss: {avg_train_loss:.4f}, Train Acc: {avg_train_acc:.2f}%')
        print(f'Val Loss: {avg_val_loss:.4f}, Val Acc: {avg_val_acc:.2f}%')
        print('-' * 50)

        # 保存性能最好的模型
        if avg_val_acc > best_val_acc:
            best_val_acc = avg_val_acc
            torch.save(model.state_dict(), 'best_model.pt')
            print(f"Saved best model with validation accuracy: {best_val_acc:.2f}%")

    return model, train_losses, train_accs, val_losses, val_accs, best_val_acc


# 6. 可视化训练过程
def plot_training_results(train_losses, train_accs, val_losses, val_accs):
    plt.figure(figsize=(12, 5))

    # 绘制损失曲线
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Loss vs Epochs')
    plt.legend()

    # 绘制准确率曲线
    plt.subplot(1, 2, 2)
    plt.plot(train_accs, label='Training Accuracy')
    plt.plot(val_accs, label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.title('Accuracy vs Epochs')
    plt.legend()

    plt.tight_layout()
    plt.savefig('training_results.png')
    plt.show()


# 7. 测试模型
def test_model(model, test_loader, criterion, device):
    model.eval()
    test_running_loss = 0.0
    test_running_acc = 0.0

    with torch.no_grad():
        for features, labels in test_loader:
            features = features.to(device)
            labels = labels.to(device)

            outputs = model(features)
            loss = criterion(outputs, labels)

            test_running_loss += loss.item()
            test_running_acc += calculate_accuracy(outputs, labels).item()

    avg_test_loss = test_running_loss / len(test_loader)
    avg_test_acc = test_running_acc / len(test_loader)

    print(f'Test Loss: {avg_test_loss:.4f}, Test Accuracy: {avg_test_acc:.2f}%')
    return avg_test_acc


# 主函数
def main():
    # 检查GPU是否可用
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    # 加载数据
    print("加载并预处理数据...")
    df = pd.read_csv('healthcare-dataset-stroke-data.csv')  # 假设数据保存在此文件中
    X, y, preprocessor = preprocess_data(df)
    input_size = X.shape[1]
    print(f"特征处理完成，输入特征维度: {input_size}")

    # 划分训练集和测试集
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 创建数据集实例
    train_val_dataset = StrokeDataset(X_train_val, y_train_val)
    test_dataset = StrokeDataset(X_test, y_test)

    # 进一步将训练集划分为训练集和验证集
    train_size = int(0.8 * len(train_val_dataset))
    val_size = len(train_val_dataset) - train_size
    train_dataset, val_dataset = random_split(
        train_val_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )

    # 定义DataLoader参数
    batch_size = 32  # 批大小设置为32，平衡内存使用和训练效率
    shuffle = True  # 训练时打乱数据顺序，提高模型泛化能力
    num_workers = 4  # 使用4个进程加载数据，加速数据预处理

    # 创建DataLoader
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    print(f"数据集划分完成: 训练集 {len(train_dataset)}, 验证集 {len(val_dataset)}, 测试集 {len(test_dataset)}")

    # 初始化模型、损失函数和优化器
    model = StrokePredictor(input_size).to(device)
    criterion = nn.BCELoss()  # 二分类问题使用二元交叉熵损失
    optimizer = optim.Adam(model.parameters(), lr=0.001)  # Adam优化器，学习率0.001

    # 训练模型
    print("开始训练模型...")
    num_epochs = 10  # 训练轮次设置为50
    model, train_losses, train_accs, val_losses, val_accs, best_val_acc = train_model(
        model, train_loader, val_loader, criterion, optimizer, num_epochs, device
    )

    # 可视化训练结果
    plot_training_results(train_losses, train_accs, val_losses, val_accs)

    # 加载最佳模型并在测试集上评估
    print("加载最佳模型并在测试集上评估...")
    best_model = StrokePredictor(input_size).to(device)
    best_model.load_state_dict(torch.load('best_model.pt'))
    test_acc = test_model(best_model, test_loader, criterion, device)
    print(f"测试集上的准确率: {test_acc:.2f}%")

if __name__ == "__main__":
    main()