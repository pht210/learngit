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
save_dir = "cnn_model_results"
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

# -------------------------- CNN模型定义 --------------------------
class CNNMnistNet(nn.Module):
    def __init__(self):
        super(CNNMnistNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 7 * 7)
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = self.fc2(x)
        x = self.dropout2(x)
        return F.log_softmax(x, dim=-1)

# -------------------------- 初始化组件 --------------------------
model = CNNMnistNet()
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

    train_avg_loss = total_loss / total_samples
    train_acc = 100. * total_correct / total_samples
    train_loss_list.append(train_avg_loss)
    train_acc_list.append(train_acc)

    print(
        f'Train Epoch: {epoch} 总结 → Avg Loss: {train_avg_loss:.6f}\tTrain Accuracy: {train_acc:.2f}%\n'
    )

    torch.save(model.state_dict(), os.path.join(save_dir, "cnn_mnist_net.pth"))

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

# -------------------------- 可视化函数 --------------------------
def plot_metrics():
    plt.figure(figsize=(12, 5))

    # 子图1：损失曲线
    plt.subplot(1, 2, 1)
    plt.plot(range(1, epochs + 1), train_loss_list, label='Train Loss', marker='o', linewidth=2)
    plt.plot(range(1, epochs + 1), test_loss_list, label='Test Loss', marker='s', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Average Loss')
    plt.title('CNN Model - Loss Curve')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xticks(range(1, epochs + 1))

    # 子图2：准确率曲线
    plt.subplot(1, 2, 2)
    plt.plot(range(1, epochs + 1), train_acc_list, label='Train Accuracy', marker='o', color='green', linewidth=2)
    plt.plot(range(1, epochs + 1), test_acc_list, label='Test Accuracy', marker='s', color='red', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('CNN Model - Accuracy Curve')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xticks(range(1, epochs + 1))
    plt.ylim(95, 100)  # CNN准确率更高，限制在95-100%

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "cnn_model_metrics.png"), dpi=100)
    plt.show()
    print(f'指标曲线已保存至 {os.path.join(save_dir, "cnn_model_metrics.png")}')

# -------------------------- 主函数（去掉初始测试）--------------------------
if __name__ == '__main__':
    print("="*60)
    print("开始训练卷积神经网络（CNN）模型...")
    print("="*60)

    for epoch in range(1, epochs + 1):
        print(f"\n----- Epoch {epoch}/{epochs} -----")
        train(epoch)
        test(epoch)

    plot_metrics()

    print("="*60)
    print("CNN模型训练完成！")
    print(f'最终测试准确率：{test_acc_list[-1]:.2f}%')
    print(f'最高测试准确率：{max(test_acc_list):.2f}%')
    print("="*60)