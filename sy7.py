import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import os
import matplotlib

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

TRAIN_BATCH_SIZE = 64
TEST_BATCH_SIZE = 1000
IMG_SIZE = 28

LEARNING_RATE = 0.001
EPOCHS = 8

FC_SAVE_DIR = "fc_model"
CNN_SAVE_DIR = "cnn_model"

os.makedirs(FC_SAVE_DIR, exist_ok=True)
os.makedirs(CNN_SAVE_DIR, exist_ok=True)

def get_dataloader(train: bool = True) -> torch.utils.data.DataLoader:

    assert isinstance(train, bool), "train参数必须是bool类型"

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

class FCMnistNet(nn.Module):

    def __init__(self):
        super(FCMnistNet, self).__init__()
        self.fc1 = nn.Linear(IMG_SIZE * IMG_SIZE, 256)
        输入层→隐藏层1
        self.fc2 = nn.Linear(256, 128)
        隐藏层1→隐藏层2
        self.fc3 = nn.Linear(128, 64)
        隐藏层2→隐藏层3
        self.fc4 = nn.Linear(64, 10)
        隐藏层3→输出层（10
        个类别）
        self.dropout = nn.Dropout(0.2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(-1, IMG_SIZE * IMG_SIZE)
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
        输入1通道→32
        通道
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)

        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 7 * 7)
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = self.fc2(x)
        x = self.dropout2(x)
        return F.log_softmax(x, dim=-1)

def train_model(
        model: nn.Module,
        optimizer: optim.Optimizer,
        epoch: int,
        save_dir: str
) -> tuple[float, float]:
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
        pred = output.argmax(dim=1)  # 预测类别
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

    torch.save(
        model.state_dict(),
        os.path.join(save_dir, f"{model.__class__.__name__}.pth")
    )

    return train_avg_loss, train_acc


def test_model(model: nn.Module, epoch: int) -> tuple[float, float]:
    model.eval()
    test_dataloader = get_dataloader(train=False)

    test_loss = 0
    correct = 0

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

def plot_combined_metrics(
        fc_metrics: tuple[list[float], list[float], list[float], list[float]],
        cnn_metrics: tuple[list[float], list[float], list[float], list[float]]
) -> None:
    fc_train_loss, fc_train_acc, fc_test_loss, fc_test_acc = fc_metrics
    cnn_train_loss, cnn_train_acc, cnn_test_loss, cnn_test_acc = cnn_metrics

    plt.figure(figsize=(14, 10))

    plt.subplot(2, 2, 1)
    plt.plot(range(1, EPOCHS + 1), fc_train_loss, label='FC Train Loss', marker='o')
    plt.plot(range(1, EPOCHS + 1), cnn_train_loss, label='CNN Train Loss', marker='s')
    plt.xlabel('Epoch')
    plt.ylabel('Average Loss')
    plt.title('训练损失对比')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xticks(range(1, EPOCHS + 1))

    plt.subplot(2, 2, 2)
    plt.plot(range(1, EPOCHS + 1), fc_test_loss, label='FC Test Loss', marker='o')
    plt.plot(range(1, EPOCHS + 1), cnn_test_loss, label='CNN Test Loss', marker='s')
    plt.xlabel('Epoch')
    plt.ylabel('Average Loss')
    plt.title('测试损失对比')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xticks(range(1, EPOCHS + 1))

    plt.subplot(2, 2, 3)
    plt.plot(range(1, EPOCHS + 1), fc_train_acc, label='FC Train Accuracy', marker='o')
    plt.plot(range(1, EPOCHS + 1), cnn_train_acc, label='CNN Train Accuracy', marker='s')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('训练准确率对比')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xticks(range(1, EPOCHS + 1))
    plt.ylim(90, 100)

    plt.subplot(2, 2, 4)
    plt.plot(range(1, EPOCHS + 1), fc_test_acc, label='FC Test Accuracy', marker='o', linewidth=2)
    plt.plot(range(1, EPOCHS + 1), cnn_test_acc, label='CNN Test Accuracy', marker='s', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('测试准确率对比')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xticks(range(1, EPOCHS + 1))
    plt.ylim(90, 100)

    plt.tight_layout()
    plt.savefig("model_comparison_metrics.png", dpi=100)
    plt.show()
    print(f'模型对比指标曲线已保存至 model_comparison_metrics.png')

if __name__ == '__main__':
    print("=" * 60)
    print("开始训练FC和CNN模型并进行对比...")
    print("=" * 60)

    fc_model = FCMnistNet()
    fc_optimizer = optim.Adam(fc_model.parameters(), lr=LEARNING_RATE)

    cnn_model = CNNMnistNet()
    cnn_optimizer = optim.Adam(cnn_model.parameters(), lr=LEARNING_RATE)

    fc_train_loss = []
    fc_train_acc = []
    fc_test_loss = []
    fc_test_acc = []

    cnn_train_loss = []
    cnn_train_acc = []
    cnn_test_loss = []
    cnn_test_acc = []

    print("\n" + "=" * 40)
    print("开始训练全连接（FC）模型...")
    print("=" * 40)
    for epoch in range(1, EPOCHS + 1):
        print(f"\n----- Epoch {epoch}/{EPOCHS} -----")
        train_loss, train_acc = train_model(fc_model, fc_optimizer, epoch, FC_SAVE_DIR)
        test_loss, test_acc = test_model(fc_model, epoch)
        fc_train_loss.append(train_loss)
        fc_train_acc.append(train_acc)
        fc_test_loss.append(test_loss)
        fc_test_acc.append(test_acc)

    print("\n" + "=" * 40)
    print("开始训练卷积神经网络（CNN）模型...")
    print("=" * 40)
    for epoch in range(1, EPOCHS + 1):
        print(f"\n----- Epoch {epoch}/{EPOCHS} -----")
        # 训练一轮
        train_loss, train_acc = train_model(cnn_model, cnn_optimizer, epoch, CNN_SAVE_DIR)
        # 测试一轮
        test_loss, test_acc = test_model(cnn_model, epoch)
        # 记录指标
        cnn_train_loss.append(train_loss)
        cnn_train_acc.append(train_acc)
        cnn_test_loss.append(test_loss)
        cnn_test_acc.append(test_acc)

    fc_metrics = (fc_train_loss, fc_train_acc, fc_test_loss, fc_test_acc)
    cnn_metrics = (cnn_train_loss, cnn_train_acc, cnn_test_loss, cnn_test_acc)
    plot_combined_metrics(fc_metrics, cnn_metrics)

    print("\n" + "=" * 60)
    print("模型训练完成！最终结果对比：")
    print(f'FC模型最终测试准确率：{fc_test_acc[-1]:.2f}%，最高测试准确率：{max(fc_test_acc):.2f}%')
    print(f'CNN模型最终测试准确率：{cnn_test_acc[-1]:.2f}%，最高测试准确率：{max(cnn_test_acc):.2f}%')
    print("=" * 60)