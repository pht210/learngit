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
fc_save_dir = "fc_model"
cnn_save_dir = "cnn_model"
os.makedirs(fc_save_dir, exist_ok=True)
os.makedirs(cnn_save_dir, exist_ok=True)


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


# -------------------------- 模型定义 --------------------------
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


# -------------------------- 训练和测试通用函数 --------------------------
def train_model(model, optimizer, epoch, save_dir):
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

    print(
        f'Train Epoch: {epoch} 总结 → Avg Loss: {train_avg_loss:.6f}\tTrain Accuracy: {train_acc:.2f}%\n'
    )

    torch.save(model.state_dict(), os.path.join(save_dir, f"{model.__class__.__name__}.pth"))
    return train_avg_loss, train_acc


def test_model(model, epoch):
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

    print(
        f'Test Epoch: {epoch} → Avg Loss: {test_avg_loss:.4f}\tTest Accuracy: {test_acc:.2f}%\n'
    )
    return test_avg_loss, test_acc


# -------------------------- 可视化函数 --------------------------
def plot_combined_metrics(fc_metrics, cnn_metrics):
    # 解析指标数据
    fc_train_loss, fc_train_acc, fc_test_loss, fc_test_acc = fc_metrics
    cnn_train_loss, cnn_train_acc, cnn_test_loss, cnn_test_acc = cnn_metrics

    plt.figure(figsize=(14, 10))

    # 子图1：训练损失对比
    plt.subplot(2, 2, 1)
    plt.plot(range(1, epochs + 1), fc_train_loss, label='FC Train Loss', marker='o')
    plt.plot(range(1, epochs + 1), cnn_train_loss, label='CNN Train Loss', marker='s')
    plt.xlabel('Epoch')
    plt.ylabel('Average Loss')
    plt.title('训练损失对比')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xticks(range(1, epochs + 1))

    # 子图2：测试损失对比
    plt.subplot(2, 2, 2)
    plt.plot(range(1, epochs + 1), fc_test_loss, label='FC Test Loss', marker='o')
    plt.plot(range(1, epochs + 1), cnn_test_loss, label='CNN Test Loss', marker='s')
    plt.xlabel('Epoch')
    plt.ylabel('Average Loss')
    plt.title('测试损失对比')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xticks(range(1, epochs + 1))

    # 子图3：训练准确率对比
    plt.subplot(2, 2, 3)
    plt.plot(range(1, epochs + 1), fc_train_acc, label='FC Train Accuracy', marker='o')
    plt.plot(range(1, epochs + 1), cnn_train_acc, label='CNN Train Accuracy', marker='s')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('训练准确率对比')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xticks(range(1, epochs + 1))
    plt.ylim(90, 100)

    # 子图4：测试准确率对比（重点对比图）
    plt.subplot(2, 2, 4)
    plt.plot(range(1, epochs + 1), fc_test_acc, label='FC Test Accuracy', marker='o', linewidth=2)
    plt.plot(range(1, epochs + 1), cnn_test_acc, label='CNN Test Accuracy', marker='s', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('测试准确率对比')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xticks(range(1, epochs + 1))
    plt.ylim(90, 100)

    plt.tight_layout()
    plt.savefig("model_comparison_metrics.png", dpi=100)
    plt.show()
    print(f'模型对比指标曲线已保存至 model_comparison_metrics.png')


# -------------------------- 主函数 --------------------------
if __name__ == '__main__':
    print("=" * 60)
    print("开始训练FC和CNN模型并进行对比...")
    print("=" * 60)

    # 初始化FC模型
    fc_model = FCMnistNet()
    fc_optimizer = optim.Adam(fc_model.parameters(), lr=lr)

    # 初始化CNN模型
    cnn_model = CNNMnistNet()
    cnn_optimizer = optim.Adam(cnn_model.parameters(), lr=lr)

    # 记录两个模型的指标
    fc_train_loss = []
    fc_train_acc = []
    fc_test_loss = []
    fc_test_acc = []

    cnn_train_loss = []
    cnn_train_acc = []
    cnn_test_loss = []
    cnn_test_acc = []

    # 训练FC模型
    print("\n" + "=" * 40)
    print("开始训练全连接（FC）模型...")
    print("=" * 40)
    for epoch in range(1, epochs + 1):
        print(f"\n----- Epoch {epoch}/{epochs} -----")
        train_loss, train_acc = train_model(fc_model, fc_optimizer, epoch, fc_save_dir)
        test_loss, test_acc = test_model(fc_model, epoch)

        fc_train_loss.append(train_loss)
        fc_train_acc.append(train_acc)
        fc_test_loss.append(test_loss)
        fc_test_acc.append(test_acc)

    # 训练CNN模型
    print("\n" + "=" * 40)
    print("开始训练卷积神经网络（CNN）模型...")
    print("=" * 40)
    for epoch in range(1, epochs + 1):
        print(f"\n----- Epoch {epoch}/{epochs} -----")
        train_loss, train_acc = train_model(cnn_model, cnn_optimizer, epoch, cnn_save_dir)
        test_loss, test_acc = test_model(cnn_model, epoch)

        cnn_train_loss.append(train_loss)
        cnn_train_acc.append(train_acc)
        cnn_test_loss.append(test_loss)
        cnn_test_acc.append(test_acc)

    # 绘制对比图表
    fc_metrics = (fc_train_loss, fc_train_acc, fc_test_loss, fc_test_acc)
    cnn_metrics = (cnn_train_loss, cnn_train_acc, cnn_test_loss, cnn_test_acc)
    plot_combined_metrics(fc_metrics, cnn_metrics)

    # 打印最终结果对比
    print("\n" + "=" * 60)
    print("模型训练完成！最终结果对比：")
    print(f'FC模型最终测试准确率：{fc_test_acc[-1]:.2f}%，最高测试准确率：{max(fc_test_acc):.2f}%')
    print(f'CNN模型最终测试准确率：{cnn_test_acc[-1]:.2f}%，最高测试准确率：{max(cnn_test_acc):.2f}%')
    print("=" * 60)