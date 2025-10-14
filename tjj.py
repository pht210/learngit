import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os


# 生态足迹数据集类
class EcoFootprintDataset(Dataset):
    def __init__(self, csv_path):
        # 读取CSV文件
        df = pd.read_csv(csv_path)

        # 清洗GDP数据
        def clean_gdp(value):
            if isinstance(value, str):
                # 提取数字和小数点
                cleaned = ''.join([c for c in value if c.isdigit() or c == '.'])
                return float(cleaned) if cleaned else 0.0
            return float(value) if not pd.isna(value) else 0.0

        df['GDP per Capita'] = df['GDP per Capita'].apply(clean_gdp)

        # 定义特征列和目标列
        feature_columns = [
            'Population (millions)', 'HDI', 'GDP per Capita',
            'Cropland Footprint', 'Grazing Footprint',
            'Forest Footprint', 'Carbon Footprint', 'Fish Footprint'
        ]

        # 填充缺失值
        for col in feature_columns:
            median_val = df[col].median()
            df.loc[:, col] = df[col].fillna(median_val)

        # 处理异常值
        for col in feature_columns:
            q5, q95 = df[col].quantile(0.05), df[col].quantile(0.95)
            df.loc[:, col] = df[col].clip(lower=q5, upper=q95)

        # 提取特征和目标变量
        self.features = df[feature_columns].values.astype(np.float32)
        self.target = df['Total Ecological Footprint'].values.astype(np.float32)

        # 计算标准化参数
        self.feature_mean = np.nanmean(self.features, axis=0)
        self.feature_std = np.nanstd(self.features, axis=0)
        self.feature_std[self.feature_std == 0] = 1.0  # 防止除零

        self.target_mean = np.nanmean(self.target)
        self.target_std = np.nanstd(self.target)
        self.target_std = self.target_std if self.target_std != 0 else 1.0

        # 标准化处理
        self.features = (self.features - self.feature_mean) / self.feature_std
        self.target = (self.target - self.target_mean) / self.target_std

        # 数据完整性检查
        assert not np.isnan(self.features).any(), "特征中存在NaN值!"
        assert not np.isnan(self.target).any(), "目标中存在NaN值!"

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.features[idx], dtype=torch.float32),
            torch.tensor(self.target[idx], dtype=torch.float32)
        )


# 生态足迹预测模型
class EcoFootprintModel(nn.Module):
    def __init__(self, input_size):
        super(EcoFootprintModel, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, 7),
            nn.BatchNorm1d(7),
            nn.ReLU(),

            nn.Linear(7, 6),
            nn.BatchNorm1d(6),
            nn.ReLU(),

            nn.Linear(6, 5),
            nn.BatchNorm1d(5),
            nn.ReLU(),

            nn.Linear(5, 1)
        )

        # 初始化权重
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                nn.init.constant_(m.bias, 0.1)

    def forward(self, x):
        return self.network(x)


# 安全的MSE损失函数
class SafeMSELoss(nn.Module):
    def __init__(self):
        super(SafeMSELoss, self).__init__()
        self.epsilon = 1e-6

    def forward(self, predictions, targets):
        loss = (predictions - targets) ** 2
        loss[torch.isnan(loss)] = self.epsilon
        return loss.mean()


# 主训练函数
def train_model():
    # 加载数据集
    dataset = EcoFootprintDataset('countries.csv')

    # 打印数据统计信息
    print("特征统计信息:")
    print(f"样本数量: {len(dataset)}")
    print(f"特征均值: {dataset.feature_mean}")
    print(f"特征标准差: {dataset.feature_std}")
    print(f"目标均值: {dataset.target_mean}")
    print(f"目标标准差: {dataset.target_std}")

    # 创建数据加载器
    train_loader = DataLoader(
        dataset,
        batch_size=32,
        shuffle=True,
        num_workers=0
    )

    # 初始化模型
    model = EcoFootprintModel(input_size=8)

    # 定义损失函数和优化器
    criterion = SafeMSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    # 学习率调度器
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )

    # 训练参数
    max_epochs = 300
    max_grad_norm = 1.0
    train_losses = []
    best_loss = float('inf')

    # 训练循环
    for epoch in range(max_epochs):
        model.train()
        epoch_total_loss = 0.0

        for batch_idx, (features, target) in enumerate(train_loader):
            # 清零梯度
            optimizer.zero_grad()

            # 前向传播
            outputs = model(features)
            loss = criterion(outputs, target.unsqueeze(1))

            # 检查NaN损失
            if torch.isnan(loss):
                print(f"检测到NaN损失! 批次: {batch_idx}")
                continue

            # 反向传播和优化
            loss.backward()

            # 梯度裁剪
            nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

            optimizer.step()

            # 累积损失
            epoch_total_loss += loss.item()

            # 打印批次信息
            if batch_idx % 5 == 0:
                print(
                    f'Epoch [{epoch + 1}/{max_epochs}], Batch [{batch_idx}/{len(train_loader)}], Loss: {loss.item():.4f}')

        # 计算平均损失
        avg_loss = epoch_total_loss / len(train_loader)
        train_losses.append(avg_loss)

        # 调整学习率
        scheduler.step(avg_loss)

        # 保存最佳模型
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), 'best_model.pt')
            print(f'保存最佳模型，损失: {best_loss:.4f}')

        # 打印 epoch 信息
        print(
            f'Epoch [{epoch + 1}/{max_epochs}], 平均损失: {avg_loss:.4f}, 学习率: {optimizer.param_groups[0]["lr"]:.8e}')

    # 绘制训练损失曲线
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False

    plt.figure(figsize=(12, 8))
    plt.plot(train_losses, label='训练损失')
    plt.title('训练损失曲线', fontsize=14)
    plt.xlabel('Epochs', fontsize=12)
    plt.ylabel('MSE损失', fontsize=12)
    plt.yscale('log')
    plt.legend()
    plt.grid(True, which="both", ls="--")
    plt.savefig('training_loss.png', dpi=300, bbox_inches='tight')
    plt.close()

    print("训练完成! 最佳模型已保存为 'best_model.pt'")


if __name__ == '__main__':
    train_model()