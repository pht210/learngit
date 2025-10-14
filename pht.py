import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re

# 解决matplotlib相关冲突和中文显示问题
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import matplotlib
matplotlib.use('TkAgg')
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


# 1. 优化模型结构（增加复杂度+Dropout防过拟合）
class Net(nn.Module):
    def __init__(self, input_size):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input_size, 16)
        self.drop1 = nn.Dropout(0.2)
        self.fc2 = nn.Linear(16, 12)
        self.drop2 = nn.Dropout(0.2)
        self.fc3 = nn.Linear(12, 8)
        self.fc4 = nn.Linear(8, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.drop1(x)
        x = torch.relu(self.fc2(x))
        x = self.drop2(x)
        x = torch.relu(self.fc3(x))
        x = self.fc4(x)
        return x


# 2. 数据集类
class CountryDataset(Dataset):
    def __init__(self, features, targets):
        self.x = torch.tensor(features, dtype=torch.float32)
        self.y = torch.tensor(targets, dtype=torch.float32).unsqueeze(1)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


def preprocess_data(data_path):
    data = pd.read_csv(data_path)
    print("原始数据列名：", data.columns.tolist())
    print(f"原始数据样本数：{len(data)}")

    target_col = "GDP per Capita"
    initial_features_col = [col for col in data.columns
                           if col not in ["Country", "Region", target_col]]
    print("初始特征列：", initial_features_col)

    # 处理特征列（转为数值型）
    features_df = data[initial_features_col].copy()
    for col in features_df.columns:
        features_df[col] = pd.to_numeric(features_df[col], errors='coerce')

    # 处理目标列（清洗特殊字符+异常值）
    raw_target = data[target_col].astype(str).copy()
    raw_target = raw_target.apply(lambda x: re.sub(r'[^\d.-]', '', x))
    target = pd.to_numeric(raw_target, errors='coerce')

    # IQR删除目标列异常值
    Q1 = np.percentile(target[~np.isnan(target)], 25)
    Q3 = np.percentile(target[~np.isnan(target)], 75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    valid_mask = (target >= lower_bound) & (target <= upper_bound) & (~np.isnan(target))
    features_df_clean = features_df[valid_mask].copy()
    target_clean = target[valid_mask]

    print(f"删除异常值后样本数：{len(target_clean)}")
    print(f"目标列（{target_col}）范围：[{target_clean.min():.2f}, {target_clean.max():.2f}]")

    # 筛选高相关性特征
    corr_with_target = features_df_clean.corrwith(target_clean)
    print("\n特征与目标的相关系数：")
    print(corr_with_target.sort_values(ascending=False).round(3))

    high_corr_features = corr_with_target[abs(corr_with_target) > 0.3].index.tolist()
    if not high_corr_features:
        high_corr_features = initial_features_col
    features_df_final = features_df_clean[high_corr_features]
    print(f"\n筛选后最终特征列：{high_corr_features}")

    return features_df_final, target_clean, high_corr_features, target_col


# 4. 数据标准化
def scale_data(X_train, X_test, y_train, y_test):
    # 特征标准化
    scaler_X = StandardScaler()
    X_train_scaled = scaler_X.fit_transform(X_train)
    X_test_scaled = scaler_X.transform(X_test)

    # 目标变量标准化
    scaler_y = StandardScaler()
    y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1)).flatten()
    y_test_scaled = scaler_y.transform(y_test.reshape(-1, 1)).flatten()

    return X_train_scaled, X_test_scaled, y_train_scaled, y_test_scaled, scaler_X, scaler_y


if __name__ == "__main__":
    # 5. 配置参数
    data_path = r"C:\Users\陆小虎\Desktop\上机课\神经网络\神经网络\日常\countries.csv"
    test_size = 0.2
    random_state = 42
    batch_size = 16
    num_workers = 0
    learning_rate = 0.01
    num_epochs = 200
    best_model_path = "best_model_optimized.pt"

    # 6. 数据流程：加载→预处理→分割→标准化
    features_df, target, features_col, target_col = preprocess_data(data_path)

    # 分割训练集/测试集
    X_train, X_test, y_train, y_test = train_test_split(
        features_df.values,
        target.values,
        test_size=test_size,
        random_state=random_state
    )
    print(f"\n训练集样本数：{len(X_train)}，测试集样本数：{len(X_test)}")

    # 填充缺失值
    train_mean = np.nanmean(X_train, axis=0)
    X_train = np.where(np.isnan(X_train), train_mean, X_train)
    X_test = np.where(np.isnan(X_test), train_mean, X_test)

    # 标准化
    X_train_scaled, X_test_scaled, y_train_scaled, y_test_scaled, scaler_X, scaler_y = scale_data(
        X_train, X_test, y_train, y_test
    )
    print(f"标准化后特征尺度：均值≈{X_train_scaled.mean():.4f}，标准差≈{X_train_scaled.std():.4f}")
    print(f"标准化后目标尺度：均值≈{y_train_scaled.mean():.4f}，标准差≈{y_train_scaled.std():.4f}")

    # 创建DataLoader
    train_dataset = CountryDataset(X_train_scaled, y_train_scaled)
    test_dataset = CountryDataset(X_test_scaled, y_test_scaled)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )

    # 7. 模型训练与测试
    input_size = X_train_scaled.shape[1]
    model = Net(input_size)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # 记录损失
    train_losses_scaled = []
    test_losses_scaled = []
    train_losses_original = []
    test_losses_original = []
    best_test_loss = float('inf')

    print(f"\n开始训练（共{num_epochs}轮）...")
    for epoch in range(num_epochs):
        # 训练阶段
        model.train()
        train_running_loss_scaled = 0.0
        for inputs, targets in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            train_running_loss_scaled += loss.item() * inputs.size(0)

        # 计算训练损失
        train_epoch_loss_scaled = train_running_loss_scaled / len(train_dataset)
        train_losses_scaled.append(train_epoch_loss_scaled)
        train_epoch_loss_original = train_epoch_loss_scaled * (scaler_y.scale_[0] ** 2)
        train_losses_original.append(train_epoch_loss_original)

        # 测试阶段
        model.eval()
        test_running_loss_scaled = 0.0
        with torch.no_grad():
            for inputs, targets in test_loader:
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                test_running_loss_scaled += loss.item() * inputs.size(0)

        # 计算测试损失
        test_epoch_loss_scaled = test_running_loss_scaled / len(test_dataset)
        test_losses_scaled.append(test_epoch_loss_scaled)
        test_epoch_loss_original = test_epoch_loss_scaled * (scaler_y.scale_[0] ** 2)
        test_losses_original.append(test_epoch_loss_original)

        # 打印日志（每10轮）
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{num_epochs} | "
                  f"训练损失：{train_epoch_loss_scaled:.4f} | "
                  f"测试损失：{test_epoch_loss_scaled:.4f} | "
                  )

        # 保存最佳模型
        if test_epoch_loss_scaled < best_test_loss:
            best_test_loss = test_epoch_loss_scaled
            torch.save(model.state_dict(), best_model_path)

    # ----------------------
    # 8. 修正可视化（删除特征逆标准化，用标准化数据绘图）
    # ----------------------
    plt.figure(figsize=(16, 14))

    # 8.1 损失曲线（双y轴：标准化+原始尺度）
    ax1 = plt.subplot(2, 2, 1)
    line1 = ax1.plot(range(1, num_epochs+1), train_losses_scaled, 'b-', label='训练损失（标准化）', linewidth=2)
    line2 = ax1.plot(range(1, num_epochs+1), test_losses_scaled, 'r--', label='测试损失（标准化）', linewidth=2)
    ax1.set_xlabel('训练轮次（Epoch）', fontsize=10)
    ax1.set_ylabel('损失值（标准化MSE）', fontsize=10, color='black')
    ax1.tick_params(axis='y', labelcolor='black')
    ax1.grid(alpha=0.3)
    ax1.legend(loc='upper left')

    ax2 = ax1.twinx()
    line3 = ax2.plot(range(1, num_epochs+1), train_losses_original, 'g-', label='训练损失（原始）', linewidth=1, alpha=0.7)
    line4 = ax2.plot(range(1, num_epochs+1), test_losses_original, 'orange', linestyle='--', label='测试损失（原始）', linewidth=1, alpha=0.7)
    ax2.set_ylabel(f'损失值（原始MSE，{target_col}²）', fontsize=10, color='darkred')
    ax2.tick_params(axis='y', labelcolor='darkred')
    ax2.legend(loc='upper right')
    plt.title('训练损失 vs 测试损失（双尺度对比）', fontsize=12)

    # 8.2 训练集：真实值 vs 预测值（原始尺度）
    plt.subplot(2, 2, 2)
    model.eval()
    with torch.no_grad():
        train_preds_scaled = []
        train_true_scaled = []
        for inputs, targets in train_loader:
            preds = model(inputs)
            train_preds_scaled.extend(preds.numpy())
            train_true_scaled.extend(targets.numpy())
        # 目标变量逆标准化（仅目标需要，特征无需）
        train_preds_original = scaler_y.inverse_transform(np.array(train_preds_scaled).reshape(-1, 1)).flatten()
        train_true_original = scaler_y.inverse_transform(np.array(train_true_scaled).reshape(-1, 1)).flatten()

    plt.scatter(train_true_original, train_preds_original, c='blue', alpha=0.6, label='训练集')
    min_val = min(min(train_true_original), min(train_preds_original))
    max_val = max(max(train_true_original), max(train_preds_original))
    plt.plot([min_val, max_val], [min_val, max_val], 'k--', label='理想预测线（y=x）')
    plt.xlabel(f'真实{target_col}', fontsize=10)
    plt.ylabel(f'预测{target_col}', fontsize=10)
    plt.title('训练集：真实值 vs 预测值（原始尺度）', fontsize=12)
    plt.legend()
    plt.grid(alpha=0.3)

    # 8.3 测试集：真实值 vs 预测值（原始尺度）
    plt.subplot(2, 2, 3)
    with torch.no_grad():
        test_preds_scaled = []
        test_true_scaled = []
        for inputs, targets in test_loader:
            preds = model(inputs)
            test_preds_scaled.extend(preds.numpy())
            test_true_scaled.extend(targets.numpy())
        test_preds_original = scaler_y.inverse_transform(np.array(test_preds_scaled).reshape(-1, 1)).flatten()
        test_true_original = scaler_y.inverse_transform(np.array(test_true_scaled).reshape(-1, 1)).flatten()

    plt.scatter(test_true_original, test_preds_original, c='orange', alpha=0.6, label='测试集')
    min_val = min(min(test_true_original), min(test_preds_original))
    max_val = max(max(test_true_original), max(test_preds_original))
    plt.plot([min_val, max_val], [min_val, max_val], 'k--', label='理想预测线（y=x）')
    plt.xlabel(f'真实{target_col}', fontsize=10)
    plt.ylabel(f'预测{target_col}', fontsize=10)
    plt.title('测试集：真实值 vs 预测值（原始尺度）', fontsize=12)
    plt.legend()
    plt.grid(alpha=0.3)

    # 8.4 特征与目标相关性（用标准化数据，避免维度不匹配）
    plt.subplot(2, 2, 4)
    if len(features_col) > 0:
        # 选择相关性最高的特征（而非硬编码HDI）
        corr_with_target_scaled = np.corrcoef(X_train_scaled.T, y_train_scaled)[-1, :-1]  # 标准化后特征与目标的相关系数
        best_feature_idx = np.argmax(abs(corr_with_target_scaled))  # 相关性绝对值最大的特征索引
        best_feature_name = features_col[best_feature_idx]
        best_corr = corr_with_target_scaled[best_feature_idx]

        # 直接用标准化后的特征和目标绘图
        plt.scatter(X_train_scaled[:, best_feature_idx], y_train_scaled, c='green', alpha=0.6)
        plt.xlabel(f'{best_feature_name}（标准化尺度）', fontsize=10)
        plt.ylabel(f'{target_col}（标准化尺度）', fontsize=10)
        plt.title(f'{best_feature_name}与{target_col}的相关性（r={best_corr:.2f}）', fontsize=12)
        plt.grid(alpha=0.3)

    # 保存图像
    plt.tight_layout()
    plt.savefig('train_test_visualizations_optimized.png', dpi=300, bbox_inches='tight')

    # 输出结果
    print(f"\n训练完成！")
    print(f"最佳模型已保存为：{best_model_path}")
    print(f"可视化图像已保存为：train_test_visualizations_optimized.png")
    print(f"最终测试损失（标准化）：{test_losses_scaled[-1]:.4f}")
    plt.show()