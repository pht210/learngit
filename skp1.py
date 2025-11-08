import torch
from torch import nn, optim
import torch.nn.functional as F
import torchvision
import os

# 解决KMP库冲突问题
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

# 配置matplotlib支持中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# -------------------------- 配置参数 --------------------------
TRAIN_BATCH_SIZE = 64
TEST_BATCH_SIZE = 1000
IMG_SIZE = 28
LR = 0.01
EPOCHS = 6
SAVE_DIR = "cnn_results"

# 创建模型保存目录（若不存在）
os.makedirs(SAVE_DIR, exist_ok=True)


# -------------------------- 数据加载 --------------------------
def get_dataloader(train=True):
    """
    获取MNIST数据集的DataLoader
    Args:
        train: bool，True返回训练集，False返回测试集
    Returns:
        DataLoader实例
    """
    assert isinstance(train, bool), "train 必须是bool类型"

    dataset = torchvision.datasets.MNIST(
        root='/data',
        train=train,
        download=True,
        transform=torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.1307,), (0.3081,))
        ])
    )

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=TRAIN_BATCH_SIZE if train else TEST_BATCH_SIZE,
        shuffle=train
    )

    return dataloader


# -------------------------- CNN模型定义 --------------------------
class CNNMnistNet(nn.Module):
    def __init__(self):
        super(CNNMnistNet, self).__init__()
        # 卷积层1：输入1通道，输出32通道，3x3卷积核，步长1，填充1
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        # 卷积层2：输入32通道，输出64通道，3x3卷积核，步长1，填充1
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        # 最大池化层：2x2池化核，步长2
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        # Dropout层：防止过拟合
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        # 全连接层1：输入64*7*7，输出128
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        # 全连接层2：输入128，输出10（对应10个数字类别）
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        """前向传播过程"""
        # 卷积1 → ReLU激活 → 池化
        x = self.pool(F.relu(self.conv1(x)))
        # 卷积2 → ReLU激活 → 池化
        x = self.pool(F.relu(self.conv2(x)))
        # 展平特征图：(batch_size, 64, 7, 7) → (batch_size, 64*7*7)
        x = x.view(-1, 64 * 7 * 7)
        # 全连接1 → ReLU激活
        x = F.relu(self.fc1(x))
        # Dropout1
        x = self.dropout1(x)
        # 全连接2
        x = self.fc2(x)
        # Dropout2
        x = self.dropout2(x)
        # 输出层：log_softmax（配合nll_loss使用）
        return F.log_softmax(x, dim=-1)


# -------------------------- 初始化组件 --------------------------
# 初始化模型
model = CNNMnistNet()
# 初始化优化器（Adam）
optimizer = optim.Adam(model.parameters(), lr=LR)

# 记录训练/测试过程的指标（长度=EPOCHS）
train_loss_list = []
train_acc_list = []
test_loss_list = []
test_acc_list = []


# -------------------------- 训练函数 --------------------------
def train(epoch):
    """训练单个epoch"""
    model.train()  # 设为训练模式
    train_dataloader = get_dataloader(train=True)

    total_correct = 0  # 总正确数
    total_samples = 0  # 总样本数
    total_loss = 0.0  # 总损失

    for batch_idx, (data, target) in enumerate(train_dataloader):
        # 梯度清零
        optimizer.zero_grad()
        # 前向传播
        output = model(data)
        # 计算损失
        loss = F.nll_loss(output, target)
        # 反向传播
        loss.backward()
        # 参数更新
        optimizer.step()

        # 累计损失和样本数
        total_loss += loss.item() * target.size(0)
        # 预测结果
        pred = output.argmax(dim=1)
        # 累计正确数
        total_correct += pred.eq(target).sum().item()
        total_samples += target.size(0)

        # 每10个batch打印一次进度
        if batch_idx % 10 == 0:
            print(
                f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_dataloader.dataset)} '
                f'({100. * batch_idx / len(train_dataloader):.0f}%)]\tBatch Loss: {loss.item():.6f}'
            )

    # 计算当前epoch的平均损失和准确率
    train_avg_loss = total_loss / total_samples
    train_acc = 100. * total_correct / total_samples

    # 记录指标
    train_loss_list.append(train_avg_loss)
    train_acc_list.append(train_acc)

    # 打印当前epoch训练总结
    print(
        f'Train Epoch: {epoch} 总结 → Avg Loss: {train_avg_loss:.6f}\tTrain Accuracy: {train_acc:.2f}%\n'
    )

    # 保存模型参数
    torch.save(model.state_dict(), os.path.join(SAVE_DIR, "cnn_mnist_net.pth"))


# -------------------------- 测试函数 --------------------------
def test(epoch):
    """测试单个epoch"""
    model.eval()  # 设为评估模式
    test_loss = 0  # 测试总损失
    correct = 0  # 测试正确数
    test_dataloader = get_dataloader(train=False)

    # 禁用梯度计算（测试阶段不需要）
    with torch.no_grad():
        for data, target in test_dataloader:
            # 前向传播
            output = model(data)
            # 累计损失（reduction='sum'表示求和）
            test_loss += F.nll_loss(output, target, reduction='sum').item()
            # 预测结果（保持维度）
            pred = output.argmax(dim=1, keepdim=True)
            # 累计正确数
            correct += pred.eq(target.view_as(pred)).sum().item()

    # 计算平均损失和准确率
    test_avg_loss = test_loss / len(test_dataloader.dataset)
    test_acc = 100. * correct / len(test_dataloader.dataset)

    # 记录指标
    test_loss_list.append(test_avg_loss)
    test_acc_list.append(test_acc)

    # 打印测试结果
    print(
        f'Test Epoch: {epoch} → Avg Loss: {test_avg_loss:.4f}\tTest Accuracy: {test_acc:.2f}%\n'
    )

    return test_acc


# -------------------------- 可视化函数 --------------------------
def plot_metrics():
    """绘制训练/测试指标曲线"""
    plt.figure(figsize=(12, 5))

    # 子图1：损失曲线
    plt.subplot(1, 2, 1)
    plt.plot(
        range(1, EPOCHS + 1),
        train_loss_list,
        label='Train Loss',
        marker='o',
        linewidth=2
    )
    plt.plot(
        range(1, EPOCHS + 1),
        test_loss_list,
        label='Test Loss',
        marker='s',
        linewidth=2
    )
    plt.xlabel('Epoch')
    plt.ylabel('Average Loss')
    plt.title('CNN Model - Loss Curve')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xticks(range(1, EPOCHS + 1))

    # 子图2：准确率曲线
    plt.subplot(1, 2, 2)
    plt.plot(
        range(1, EPOCHS + 1),
        train_acc_list,
        label='Train Accuracy',
        marker='o',
        color='green',
        linewidth=2
    )
    plt.plot(
        range(1, EPOCHS + 1),
        test_acc_list,
        label='Test Accuracy',
        marker='s',
        color='red',
        linewidth=2
    )
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('CNN Model - Accuracy Curve')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xticks(range(1, EPOCHS + 1))
    plt.ylim(10, 100)  # CNN准确率较高，限制y轴范围便于观察

    # 调整子图间距
    plt.tight_layout()
    # 保存图片
    plt.savefig(os.path.join(SAVE_DIR, "cnn_model_metrics.png"), dpi=100)
    # 显示图片
    plt.show()

    print(f'指标曲线已保存至 {os.path.join(SAVE_DIR, "cnn_model_metrics.png")}')


# -------------------------- 主函数 --------------------------
if __name__ == '__main__':
    print("=" * 60)
    print("开始训练卷积神经网络（CNN）模型...")
    print("=" * 60)

    # 迭代训练和测试
    for epoch in range(1, EPOCHS + 1):
        print(f"\n----- Epoch {epoch}/{EPOCHS} -----")
        train(epoch)
        test(epoch)

    # 绘制指标曲线
    plot_metrics()

    # 打印训练总结
    print("=" * 60)
    print("CNN模型训练完成！")
    print(f'最终测试准确率：{test_acc_list[-1]:.2f}%')
    print(f'最高测试准确率：{max(test_acc_list):.2f}%')
    print("=" * 60)