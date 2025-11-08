import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
import torchvision
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
# -------------------------- 配置参数 --------------------------
train_batch_size = 64
test_batch_size = 1000
img_size = 28
lr = 0.001
epochs = 8
save_dir = "fc_model_results"
os.makedirs(save_dir, exist_ok=True)

# -------------------------- 数据加载 --------------------------
def get_dataloader(train=True):
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
        batch_size=train_batch_size if train else test_batch_size,
        shuffle=train
    )
    return dataloader

# -------------------------- 全连接模型定义 --------------------------
class FCMnistNet(nn.Module):
    def __init__(self):
        super(FCMnistNet, self).__init__()
        self.fc1 = nn.Linear(img_size * img_size, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 10)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = x.view(-1, img_size * img_size)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return F.log_softmax(x, dim=-1)

# -------------------------- 初始化组件 --------------------------
model = FCMnistNet()
optimizer = optim.Adam(model.parameters(), lr=lr)

# 记录训练/测试过程（长度=epochs）
train_loss_list = []
train_acc_list = []
test_loss_list = []
test_acc_list = []

# -------------------------- 训练函数 --------------------------
def train(epoch):
    model.train()
    train_dataloader = get_dataloader(train=True)
    total_correct = 0
    total_samples = 0
    total_loss = 0.0

    for batch_idx, (data, target) in enumerate(train_dataloader):
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * target.size(0)
        pred = output.argmax(dim=1)
        total_correct += pred.eq(target).sum().item()
        total_samples += target.size(0)

        if batch_idx % 10 == 0:
            print(
                f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_dataloader.dataset)} '
                f'({100. * batch_idx / len(train_dataloader):.0f}%)]\tBatch Loss: {loss.item():.6f}'
            )

    # 计算本轮平均指标
    train_avg_loss = total_loss / total_samples
    train_acc = 100. * total_correct / total_samples
    train_loss_list.append(train_avg_loss)
    train_acc_list.append(train_acc)

    print(
        f'Train Epoch: {epoch} 总结 → Avg Loss: {train_avg_loss:.6f}\tTrain Accuracy: {train_acc:.2f}%\n'
    )

    # 保存模型
    torch.save(model.state_dict(), os.path.join(save_dir, "fc_mnist_net.pth"))

# -------------------------- 测试函数 --------------------------
def test(epoch):
    model.eval()
    test_loss = 0
    correct = 0
    test_dataloader = get_dataloader(train=False)

    with torch.no_grad():
        for data, target in test_dataloader:
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_avg_loss = test_loss / len(test_dataloader.dataset)
    test_acc = 100. * correct / len(test_dataloader.dataset)

    test_loss_list.append(test_avg_loss)
    test_acc_list.append(test_acc)

    print(
        f'Test Epoch: {epoch} → Avg Loss: {test_avg_loss:.4f}\tTest Accuracy: {test_acc:.2f}%\n'
    )
    return test_acc

# -------------------------- 可视化函数（确保x和y维度一致）--------------------------
def plot_metrics():
    plt.figure(figsize=(12, 5))

    # 子图1：损失曲线（x=epochs，y=8个值）
    plt.subplot(1, 2, 1)
    plt.plot(range(1, epochs + 1), train_loss_list, label='Train Loss', marker='o', linewidth=2)
    plt.plot(range(1, epochs + 1), test_loss_list, label='Test Loss', marker='s', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Average Loss')
    plt.title('FC Model - Loss Curve')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xticks(range(1, epochs + 1))  # 强制x轴显示整数epoch

    # 子图2：准确率曲线（x=epochs，y=8个值）
    plt.subplot(1, 2, 2)
    plt.plot(range(1, epochs + 1), train_acc_list, label='Train Accuracy', marker='o', color='green', linewidth=2)
    plt.plot(range(1, epochs + 1), test_acc_list, label='Test Accuracy', marker='s', color='red', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('FC Model - Accuracy Curve')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xticks(range(1, epochs + 1))  # 强制x轴显示整数epoch
    plt.ylim(90, 100)  # 准确率范围限制在90-100%，更直观

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "fc_model_metrics.png"), dpi=100)
    plt.show()
    print(f'指标曲线已保存至 {os.path.join(save_dir, "fc_model_metrics.png")}')

# -------------------------- 主函数（去掉初始测试）--------------------------
if __name__ == '__main__':
    print("="*60)
    print("开始训练全连接（FC）模型...")
    print("="*60)

    # 直接训练+测试（无初始测试，避免列表长度多余）
    for epoch in range(1, epochs + 1):
        print(f"\n----- Epoch {epoch}/{epochs} -----")
        train(epoch)
        test(epoch)

    # 绘制指标曲线（此时所有列表长度都是8，与x轴匹配）
    plot_metrics()

    # 打印最终结果
    print("="*60)
    print("FC模型训练完成！")
    print(f'最终测试准确率：{test_acc_list[-1]:.2f}%')
    print(f'最高测试准确率：{max(test_acc_list):.2f}%')
    print("="*60)