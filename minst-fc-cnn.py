import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
from torchvision import transforms
import os
import matplotlib.pyplot as plt

# 解决中文显示和库冲突问题
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]
plt.rcParams["axes.unicode_minus"] = False

# 模型训练配置参数
TRAIN_BATCH = 64
TEST_BATCH = 1000
IMAGE_DIM = 28
LEARNING_RATE = 0.001
TRAIN_EPOCHS = 8
FC_MODEL_SAVE = "fc_checkpoints"
CNN_MODEL_SAVE = "cnn_checkpoints"

# 创建模型保存目录
os.makedirs(FC_MODEL_SAVE, exist_ok=True)
os.makedirs(CNN_MODEL_SAVE, exist_ok=True)


def load_dataset(is_training):
    """加载MNIST数据集并返回数据加载器"""
    data_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    mnist_dataset = torchvision.datasets.MNIST(
        root="/data",
        train=is_training,
        download=True,
        transform=data_transform
    )
    
    data_loader = torch.utils.data.DataLoader(
        dataset=mnist_dataset,
        batch_size=TRAIN_BATCH if is_training else TEST_BATCH,
        shuffle=is_training
    )
    
    return data_loader


class FCNet(nn.Module):
    """全连接神经网络模型"""
    def __init__(self):
        super(FCNet, self).__init__()
        self.layer1 = nn.Linear(IMAGE_DIM * IMAGE_DIM, 256)
        self.layer2 = nn.Linear(256, 128)
        self.layer3 = nn.Linear(128, 64)
        self.output = nn.Linear(64, 10)
        self.drop = nn.Dropout(0.2)

    def forward(self, x):
        x = x.reshape(-1, IMAGE_DIM * IMAGE_DIM)
        x = F.relu(self.layer1(x))
        x = self.drop(x)
        x = F.relu(self.layer2(x))
        x = self.drop(x)
        x = F.relu(self.layer3(x))
        x = self.output(x)
        return F.log_softmax(x, dim=1)


class ConvNet(nn.Module):
    """卷积神经网络模型"""
    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv_layer1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv_layer2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.max_pool = nn.MaxPool2d(2, 2)
        self.drop_layer1 = nn.Dropout(0.25)
        self.drop_layer2 = nn.Dropout(0.5)
        self.fc_layer1 = nn.Linear(64 * 7 * 7, 128)
        self.fc_layer2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.max_pool(F.relu(self.conv_layer1(x)))
        x = self.max_pool(F.relu(self.conv_layer2(x)))
        x = x.view(-1, 64 * 7 * 7)
        x = F.relu(self.fc_layer1(x))
        x = self.drop_layer1(x)
        x = self.fc_layer2(x)
        x = self.drop_layer2(x)
        return F.log_softmax(x, dim=1)


def model_training(model, optimizer, epoch_num, save_path):
    """模型训练函数"""
    model.train()
    train_loader = load_dataset(is_training=True)
    total_loss = 0.0
    correct_count = 0
    total_samples = 0

    for batch_idx, (images, labels) in enumerate(train_loader):
        optimizer.zero_grad()
        outputs = model(images)
        loss = F.nll_loss(outputs, labels)
        loss.backward()
        optimizer.step()

        # 累计损失和准确率计算
        total_loss += loss.item() * labels.size(0)
        predicted = outputs.argmax(dim=1)
        correct_count += predicted.eq(labels).sum().item()
        total_samples += labels.size(0)

        # 打印训练进度
        if batch_idx % 10 == 0:
            progress = 100. * batch_idx / len(train_loader)
            print(f"训练轮次: {epoch_num} [{batch_idx * len(images)}/{len(train_loader.dataset)} "
                  f"({progress:.0f}%)]\t批次损失: {loss.item():.6f}")

    # 计算平均损失和准确率
    avg_loss = total_loss / total_samples
    accuracy = 100. * correct_count / total_samples
    print(f"训练轮次: {epoch_num} 总结 → 平均损失: {avg_loss:.6f}\t训练准确率: {accuracy:.2f}%\n")

    # 保存模型
    torch.save(model.state_dict(), os.path.join(save_path, f"{model.__class__.__name__}.pth"))
    return avg_loss, accuracy


def model_evaluation(model, epoch_num):
    """模型评估函数"""
    model.eval()
    test_loader = load_dataset(is_training=False)
    test_loss = 0.0
    correct_count = 0

    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images)
            test_loss += F.nll_loss(outputs, labels, reduction="sum").item()
            predicted = outputs.argmax(dim=1, keepdim=True)
            correct_count += predicted.eq(labels.view_as(predicted)).sum().item()

    # 计算评估指标
    avg_loss = test_loss / len(test_loader.dataset)
    accuracy = 100. * correct_count / len(test_loader.dataset)
    print(f"测试轮次: {epoch_num} → 平均损失: {avg_loss:.4f}\t测试准确率: {accuracy:.2f}%\n")
    return avg_loss, accuracy


def plot_performance(fc_data, cnn_data):
    """绘制模型性能对比图表"""
    fc_train_loss, fc_train_acc, fc_test_loss, fc_test_acc = fc_data
    cnn_train_loss, cnn_train_acc, cnn_test_loss, cnn_test_acc = cnn_data

    plt.figure(figsize=(16, 12))

    # 训练损失对比
    plt.subplot(2, 2, 1)
    plt.plot(range(1, TRAIN_EPOCHS + 1), fc_train_loss, "bo-", label="FC网络训练损失")
    plt.plot(range(1, TRAIN_EPOCHS + 1), cnn_train_loss, "gs-", label="CNN网络训练损失")
    plt.xlabel("轮次")
    plt.ylabel("平均损失")
    plt.title("训练损失对比")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.xticks(range(1, TRAIN_EPOCHS + 1))

    # 测试损失对比
    plt.subplot(2, 2, 2)
    plt.plot(range(1, TRAIN_EPOCHS + 1), fc_test_loss, "bo-", label="FC网络测试损失")
    plt.plot(range(1, TRAIN_EPOCHS + 1), cnn_test_loss, "gs-", label="CNN网络测试损失")
    plt.xlabel("轮次")
    plt.ylabel("平均损失")
    plt.title("测试损失对比")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.xticks(range(1, TRAIN_EPOCHS + 1))

    # 训练准确率对比
    plt.subplot(2, 2, 3)
    plt.plot(range(1, TRAIN_EPOCHS + 1), fc_train_acc, "bo-", label="FC网络训练准确率")
    plt.plot(range(1, TRAIN_EPOCHS + 1), cnn_train_acc, "gs-", label="CNN网络训练准确率")
    plt.xlabel("轮次")
    plt.ylabel("准确率 (%)")
    plt.title("训练准确率对比")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.xticks(range(1, TRAIN_EPOCHS + 1))
    plt.ylim(90, 100)

    # 测试准确率对比
    plt.subplot(2, 2, 4)
    plt.plot(range(1, TRAIN_EPOCHS + 1), fc_test_acc, "bo-", label="FC网络测试准确率")
    plt.plot(range(1, TRAIN_EPOCHS + 1), cnn_test_acc, "gs-", label="CNN网络测试准确率")
    plt.xlabel("轮次")
    plt.ylabel("准确率 (%)")
    plt.title("测试准确率对比")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.xticks(range(1, TRAIN_EPOCHS + 1))
    plt.ylim(90, 100)

    plt.tight_layout()
    plt.savefig("performance_comparison.png", dpi=100)
    plt.show()
    print("模型性能对比图表已保存为 performance_comparison.png")


def main():
    """主函数：训练并对比两个模型"""
    print("=" * 60)
    print("开始MNIST数据集上的FC与CNN模型训练及对比实验")
    print("=" * 60)

    # 初始化模型和优化器
    fc_network = FCNet()
    fc_optim = optim.Adam(fc_network.parameters(), lr=LEARNING_RATE)
    
    cnn_network = ConvNet()
    cnn_optim = optim.Adam(cnn_network.parameters(), lr=LEARNING_RATE)

    # 存储训练过程中的指标
    fc_train_losses, fc_train_accs, fc_test_losses, fc_test_accs = [], [], [], []
    cnn_train_losses, cnn_train_accs, cnn_test_losses, cnn_test_accs = [], [], [], []

    # 训练FC网络
    print("\n" + "=" * 40)
    print("开始训练全连接神经网络...")
    print("=" * 40)
    for epoch in range(1, TRAIN_EPOCHS + 1):
        print(f"\n----- 第 {epoch}/{TRAIN_EPOCHS} 轮 -----")
        train_loss, train_acc = model_training(fc_network, fc_optim, epoch, FC_MODEL_SAVE)
        test_loss, test_acc = model_evaluation(fc_network, epoch)
        
        fc_train_losses.append(train_loss)
        fc_train_accs.append(train_acc)
        fc_test_losses.append(test_loss)
        fc_test_accs.append(test_acc)

    # 训练CNN网络
    print("\n" + "=" * 40)
    print("开始训练卷积神经网络...")
    print("=" * 40)
    for epoch in range(1, TRAIN_EPOCHS + 1):
        print(f"\n----- 第 {epoch}/{TRAIN_EPOCHS} 轮 -----")
        train_loss, train_acc = model_training(cnn_network, cnn_optim, epoch, CNN_MODEL_SAVE)
        test_loss, test_acc = model_evaluation(cnn_network, epoch)
        
        cnn_train_losses.append(train_loss)
        cnn_train_accs.append(train_acc)
        cnn_test_losses.append(test_loss)
        cnn_test_accs.append(test_acc)

    # 可视化性能对比
    fc_metrics = (fc_train_losses, fc_train_accs, fc_test_losses, fc_test_accs)
    cnn_metrics = (cnn_train_losses, cnn_train_accs, cnn_test_losses, cnn_test_accs)
    plot_performance(fc_metrics, cnn_metrics)

    # 输出最终结果对比
    print("\n" + "=" * 60)
    print("模型训练完成！最终性能对比：")
    print(f"FC网络最终测试准确率：{fc_test_accs[-1]:.2f}%，最高测试准确率：{max(fc_test_accs):.2f}%")
    print(f"CNN网络最终测试准确率：{cnn_test_accs[-1]:.2f}%，最高测试准确率：{max(cnn_test_accs):.2f}%")
    print("=" * 60)


if __name__ == "__main__":
    main()