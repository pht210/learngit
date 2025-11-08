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
SAVE_DIR = "fc_results"

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


# -------------------------- 全连接模型定义 --------------------------
class FCMnistNet(nn.Module):
    def __init__(self):
        super(FCMnistNet, self).__init__()
        # 全连接层1：输入28*28（图片展平后），输出256
        self.fc1 = nn.Linear(IMG_SIZE * IMG_SIZE, 256)
        # 全连接层2：输入256，输出128
        self.fc2 = nn.Linear(256, 128)
        # 全连接层3：输入128，输出64
        self.fc3 = nn.Linear(128, 64)
        # 全连接层4：输入64，输出10（对应10个数字类别）
        self.fc4 = nn.Linear(64, 10)
        # Dropout层：防止过拟合（ dropout rate=0.2 ）
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        """前向传播过程"""
        # 展平输入：(batch_size, 1, 28, 28) → (batch_size, 28*28)
        x = x.view(-1, IMG_SIZE * IMG_SIZE)
        # 全连接1 → ReLU激活 → Dropout
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        # 全连接2 → ReLU激活 → Dropout
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        # 全连接3 → ReLU激活
        x = F.relu(self.fc3(x))
        # 全连接4（输出层）
        x = self.fc4(x)
        # 输出层：log_softmax（配合nll_loss使用）
        return F.log_softmax(x, dim=-1)


# -------------------------- 初始化组件 --------------------------
# 初始化全连接模型
model = FCMnistNet()
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
        # 计算损失（nll_loss配合log_softmax）
        loss = F.nll_loss(output, target)
        # 反向传播
        loss.backward()
        # 参数更新
        optimizer.step()

        # 累计损失（乘以batch大小得到总损失）
        total_loss += loss.item() * target.size(0)
        # 预测结果（取概率最大的类别）
        pred = output.argmax(dim=1)
        # 累计正确样本数
        total_correct += pred.eq(target).sum().item()
        total_samples += target.size(0)

        # 每10个batch打印一次训练进度
        if batch_idx % 10 == 0:
            print(
                f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_dataloader.dataset)} '
                f'({100. * batch_idx / len(train_dataloader):.0f}%)]\tBatch Loss: {loss.item():.6f}'
            )

    # 计算当前epoch的平均损失和准确率
    train_avg_loss = total_loss / total_samples
    train_acc = 100. * total_correct / total_samples

    # 记录指标到列表
    train_loss_list.append(train_avg_loss)
    train_acc_list.append(train_acc)

    # 打印当前epoch训练总结
    print(
        f'Train Epoch: {epoch} 总结 → Avg Loss: {train_avg_loss:.6f}\tTrain Accuracy: {train_acc:.2f}%\n'
    )

    # 保存模型参数
    torch.save(model.state_dict(), os.path.join(SAVE_DIR, "fc_mnist_net.pth"))


# -------------------------- 测试函数 --------------------------
def test(epoch):
    """测试单个epoch"""
    model.eval()  # 设为评估模式
    test_loss = 0  # 测试总损失
    correct = 0  # 测试正确数
    test_dataloader = get_dataloader(train=False)

    # 禁用梯度计算（测试阶段无需反向传播）
    with torch.no_grad():
        for data, target in test_dataloader:
            # 前向传播
            output = model(data)
            # 累计损失（reduction='sum'表示求和所有样本损失）
            test_loss += F.nll_loss(output, target, reduction='sum').item()
            # 预测结果（保持维度一致，便于后续比较）
            pred = output.argmax(dim=1, keepdim=True)
            # 累计正确样本数
            correct += pred.eq(target.view_as(pred)).sum().item()

    # 计算测试集平均损失和准确率
    test_avg_loss = test_loss / len(test_dataloader.dataset)
    test_acc = 100. * correct / len(test_dataloader.dataset)

    # 记录指标到列表
    test_loss_list.append(test_avg_loss)
    test_acc_list.append(test_acc)

    # 打印测试结果
    print(
        f'Test Epoch: {epoch} → Avg Loss: {test_avg_loss:.4f}\tTest Accuracy: {test_acc:.2f}%\n'
    )

    return test_acc


# -------------------------- 可视化函数 --------------------------
def plot_metrics():
    """绘制训练/测试指标曲线（确保x和y维度一致）"""
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
    plt.title('FC Model - Loss Curve')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xticks(range(1, EPOCHS + 1))  # 强制x轴显示整数epoch

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
    plt.title('FC Model - Accuracy Curve')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xticks(range(1, EPOCHS + 1))  # 强制x轴显示整数epoch
    plt.ylim(90, 100)  # 限制准确率范围，更直观观察变化

    # 调整子图间距，避免重叠
    plt.tight_layout()
    # 保存指标曲线图片
    plt.savefig(os.path.join(SAVE_DIR, "fc_model_metrics.png"), dpi=100)
    # 显示图片
    plt.show()

    print(f'指标曲线已保存至 {os.path.join(SAVE_DIR, "fc_model_metrics.png")}')


# -------------------------- 主函数 --------------------------
if __name__ == '__main__':
    print("=" * 60)
    print("开始训练全连接（FC）模型...")
    print("=" * 60)

    # 迭代训练和测试（每个epoch先训练后测试）
    for epoch in range(1, EPOCHS + 1):
        print(f"\n----- Epoch {epoch}/{EPOCHS} -----")
        train(epoch)
        test(epoch)

    # 训练完成后绘制指标曲线
    plot_metrics()

    # 打印训练总结信息
    print("=" * 60)
    print("FC模型训练完成！")
    print(f'最终测试准确率：{test_acc_list[-1]:.2f}%')
    print(f'最高测试准确率：{max(test_acc_list):.2f}%')
    print("=" * 60)