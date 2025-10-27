import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt


# 定义数据集类
class StudentDataset(Dataset):
    def __init__(self, features, labels):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


# 定义神经网络模型
class StudentPerformanceModel(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(StudentPerformanceModel, self).__init__()
        self.layer1 = nn.Linear(input_dim, 64)
        self.layer2 = nn.Linear(64, 32)
        self.layer3 = nn.Linear(32, num_classes)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = self.relu(self.layer1(x))
        x = self.dropout(x)
        x = self.relu(self.layer2(x))
        x = self.layer3(x)
        return x


# 数据预处理函数
def preprocess_data(file_path):
    # 读取数据
    df = pd.read_csv(file_path)

    # 假设标签列为'G3'（最终成绩），根据实际数据调整
    label_column = 'G3'

    # 检查并处理类别样本数不足的问题
    print("原始标签分布：")
    class_distribution = df[label_column].value_counts()
    print(class_distribution)

    # 过滤掉样本数少于2的类别
    valid_classes = class_distribution[class_distribution >= 2].index
    df_filtered = df[df[label_column].isin(valid_classes)].copy()

    # 检查过滤后的数据量
    print(f"\n过滤后保留的样本数：{len(df_filtered)}")
    print("过滤后的标签分布：")
    print(df_filtered[label_column].value_counts())

    # 如果过滤后数据为空，抛出异常
    if len(df_filtered) == 0:
        raise ValueError("过滤后数据为空，请检查标签列或调整过滤条件")

    # 定义特征列（排除标签列）
    feature_columns = [col for col in df_filtered.columns if col != label_column]

    # 区分数值特征和类别特征
    numeric_features = df_filtered[feature_columns].select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_features = df_filtered[feature_columns].select_dtypes(include=['object', 'category']).columns.tolist()

    # 创建预处理管道
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])

    # 划分特征和标签
    X = df_filtered[feature_columns]
    y = df_filtered[label_column]

    # 将标签映射为连续的整数（避免类别编号不连续的问题）
    unique_labels = sorted(y.unique())
    label_mapping = {label: i for i, label in enumerate(unique_labels)}
    y_encoded = y.map(label_mapping)
    num_classes = len(unique_labels)

    # 划分训练集和测试集（如果类别数足够，使用分层抽样；否则普通抽样）
    if num_classes >= 2:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
        )
    else:
        # 如果只剩一个类别，使用普通抽样
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, test_size=0.2, random_state=42
        )

    # 预处理特征
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)

    # 获取输入维度
    input_dim = X_train_processed.shape[1]

    # 创建数据集和数据加载器
    train_dataset = StudentDataset(X_train_processed, y_train.values)
    test_dataset = StudentDataset(X_test_processed, y_test.values)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    return train_loader, test_loader, input_dim, num_classes


# 计算准确率的辅助函数
def calculate_accuracy(model, data_loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for features, labels in data_loader:
            features, labels = features.to(device), labels.to(device)
            outputs = model(features)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return 100 * correct / total


# 训练模型函数（增加准确率记录）
def train_model(model, train_loader, test_loader, criterion, optimizer, device, epochs=50):
    train_losses = []
    train_accuracies = []  # 记录训练准确率
    test_accuracies = []  # 记录测试准确率

    for epoch in range(epochs):
        # 训练阶段
        model.train()
        total_loss = 0.0
        for features, labels in train_loader:
            features, labels = features.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            total_loss += loss.item() * features.size(0)

        # 计算本轮的损失和准确率
        avg_loss = total_loss / len(train_loader.dataset)
        train_acc = calculate_accuracy(model, train_loader, device)
        test_acc = calculate_accuracy(model, test_loader, device)

        # 保存记录
        train_losses.append(avg_loss)
        train_accuracies.append(train_acc)
        test_accuracies.append(test_acc)

        # 每5轮打印一次信息
        if (epoch + 1) % 5 == 0:
            print(f'Epoch [{epoch + 1}/{epochs}]')
            print(f'  Loss: {avg_loss:.4f} | Train Acc: {train_acc:.2f}% | Test Acc: {test_acc:.2f}%')

    return train_losses, train_accuracies, test_accuracies


# 主函数
def main():
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    # 预处理数据（请替换为你的CSV文件路径）
    train_loader, test_loader, input_dim, num_classes = preprocess_data("student-por.csv")

    # 确保至少有两个类别
    if num_classes < 2:
        print(f"警告：只有{num_classes}个类别，不适合分类任务")
        return

    # 初始化模型、损失函数和优化器
    model = StudentPerformanceModel(input_dim, num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 训练模型（获取损失和准确率记录）
    print("\n开始训练模型...")
    train_losses, train_accuracies, test_accuracies = train_model(
        model, train_loader, test_loader, criterion, optimizer, device, epochs=50
    )

    # 最终评估
    final_test_acc = calculate_accuracy(model, test_loader, device)
    print(f"\n最终测试准确率: {final_test_acc:.2f}%")

    # 绘制损失曲线和准确率曲线
    plt.figure(figsize=(14, 6))

    # 子图1：损失曲线
    plt.subplot(1, 2, 1)
    plt.plot(range(1, len(train_losses) + 1), train_losses, label='Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss Curve')
    plt.legend()
    plt.grid(True)

    # 子图2：准确率曲线（新增）
    plt.subplot(1, 2, 2)
    plt.plot(range(1, len(train_accuracies) + 1), train_accuracies, label='Training Accuracy')
    plt.plot(range(1, len(test_accuracies) + 1), test_accuracies, label='Test Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('Training vs Test Accuracy Curve')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()  # 调整子图间距
    plt.show()


if __name__ == "__main__":
    main()