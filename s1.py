import os
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torch.optim import Adamax, RMSprop, Rprop
import warnings


class DataProcessor:
    """数据处理类，封装数据创建、加载、清理与标准化功能"""

    @staticmethod
    def create_clean_sample_data():
        np.random.seed(42)
        x = np.linspace(0, 5, 100)
        y = 2 * x + 1 + np.random.normal(0, 0.3, 100)
        x = np.nan_to_num(x, nan=0.0, posinf=5.0, neginf=0.0)
        y = np.nan_to_num(y, nan=1.0, posinf=10.0, neginf=1.0)
        data = pd.DataFrame({'x': x, 'y': y})
        data.to_csv('train.csv', index=False)
        print("创建了新的干净数据集")
        return data

    @staticmethod
    def load_and_clean_data():
        try:
            data = pd.read_csv('../日常/train.csv')
            print(f"成功加载数据集，大小: {len(data)}")
        except:
            print("无法加载train.csv，创建新数据集")
            return DataProcessor.create_clean_sample_data()

        print("数据基本信息:")
        print(data.info())
        print("\n数据统计描述:")
        print(data.describe())

        # 处理缺失值和无穷值
        nan_count = data.isnull().sum().sum()
        if nan_count > 0:
            print(f"发现 {nan_count} 个NaN值，进行清理...")
            data = data.dropna()
            print(f"清理后数据大小: {len(data)}")

        inf_count = np.isinf(data.values).sum()
        if inf_count > 0:
            print(f"发现 {inf_count} 个无穷大值，进行清理...")
            data = data.replace([np.inf, -np.inf], np.nan).dropna()
            print(f"清理后数据大小: {len(data)}")

        return data

    @staticmethod
    def safe_normalize(data):
        data = np.array(data, dtype=np.float32)
        data = np.nan_to_num(
            data,
            nan=np.nanmean(data),
            posinf=np.nanmax(data),
            neginf=np.nanmin(data)
        )
        mean_val = np.mean(data)
        std_val = np.std(data)
        if std_val < 1e-8:
            std_val = 1.0
        normalized = (data - mean_val) / std_val

        if np.isnan(normalized).any() or np.isinf(normalized).any():
            print("警告: 标准化后仍有无效值，使用原始数据")
            return data
        return normalized

    @staticmethod
    def prepare_data():
        """整合数据处理流程，返回处理后的张量和原始值"""
        data = DataProcessor.load_and_clean_data()
        x_values = data['x'].values.astype(np.float32)
        y_values = data['y'].values.astype(np.float32)

        # 数据有效性检查
        print(f"\n数据有效性检查:")
        print(f"x值范围: {x_values.min():.4f} ~ {x_values.max():.4f}")
        print(f"y值范围: {y_values.min():.4f} ~ {y_values.max():.4f}")
        print(f"x值NaN数量: {np.isnan(x_values).sum()}")
        print(f"y值NaN数量: {np.isnan(y_values).sum()}")
        print(f"x值无穷大数量: {np.isinf(x_values).sum()}")
        print(f"y值无穷大数量: {np.isinf(y_values).sum()}")

        # 标准化
        x_normalized = DataProcessor.safe_normalize(x_values)
        y_normalized = DataProcessor.safe_normalize(y_values)

        print(f"标准化后x范围: {x_normalized.min():.4f} ~ {x_normalized.max():.4f}")
        print(f"标准化后y范围: {y_normalized.min():.4f} ~ {y_normalized.max():.4f}")

        # 转换为张量
        x_data = torch.Tensor(x_normalized).reshape(-1, 1)
        y_data = torch.Tensor(y_normalized).reshape(-1, 1)
        print(f"最终数据形状: x_data {x_data.shape}, y_data {y_data.shape}")

        return x_data, y_data, x_values, y_values


class LinearModel(nn.Module):
    """线性回归模型类"""

    def __init__(self):
        super(LinearModel, self).__init__()
        self.linear = nn.Linear(1, 1)
        # 初始化参数
        nn.init.normal_(self.linear.weight, mean=0, std=0.01)
        nn.init.normal_(self.linear.bias, mean=0, std=0.01)

    def forward(self, x):
        return self.linear(x)


class Trainer:
    """训练器类，封装模型训练逻辑"""

    @staticmethod
    def train(optimizer_class, optimizer_name, x_data, y_data, lr=0.01, epochs=1000):
        model = LinearModel()
        criterion = nn.MSELoss()
        optimizer = optimizer_class(model.parameters(), lr=lr)

        losses = []
        weights = []
        biases = []
        print(f"\n使用 {optimizer_name} 训练...")

        # 记录初始参数
        initial_w = model.linear.weight.item()
        initial_b = model.linear.bias.item()
        print(f"初始参数: w={initial_w:.4f}, b={initial_b:.4f}")

        nan_detected = False
        for epoch in range(epochs):
            y_pred = model(x_data)
            loss = criterion(y_pred, y_data)

            # 检测无效损失
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"在epoch {epoch} 检测到无效损失，停止训练")
                nan_detected = True
                break

            # 反向传播与参数更新
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            # 记录训练过程
            losses.append(loss.item())
            weights.append(model.linear.weight.item())
            biases.append(model.linear.bias.item())

            # 定期打印日志
            if epoch % 200 == 0 and epoch > 0:
                print(f'Epoch {epoch}, Loss: {loss.item():.6f}')

        # 处理异常终止情况
        final_w = model.linear.weight.item()
        final_b = model.linear.bias.item()
        if nan_detected and len(weights) > 0:
            final_w = weights[-1]
            final_b = biases[-1]

        print(f"最终参数: w={final_w:.4f}, b={final_b:.4f}")
        print(f"参数变化: Δw={final_w - initial_w:.4f}, Δb={final_b - initial_b:.4f}")

        return {
            'name': optimizer_name,
            'losses': losses,
            'weights': weights,
            'biases': biases,
            'final_w': final_w,
            'final_b': final_b,
            'model': model
        }


class Visualizer:
    """可视化工具类，封装所有绘图功能（已移除热力图）"""

    def __init__(self, results, x_data, y_data, x_values, y_values):
        self.results = results
        self.x_data = x_data
        self.y_data = y_data
        self.x_values = x_values
        self.y_values = y_values

    def plot_optimizer_performance(self, subplot):
        """绘制优化器性能对比图"""
        plt.subplot(subplot)
        for result in self.results:
            if len(result['losses']) > 0:
                plt.plot(result['losses'][:500], label=result['name'], linewidth=2)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('不同优化器的损失函数变化对比', fontsize=14, fontweight='bold')
        plt.legend()
        plt.grid(True, alpha=0.3)

    def plot_loss_log_scale(self, subplot):
        """绘制对数尺度损失图"""
        plt.subplot(subplot)
        for result in self.results:
            if len(result['losses']) > 0:
                plt.semilogy(result['losses'][:500], label=result['name'], linewidth=2)
        plt.xlabel('Epoch')
        plt.ylabel('Loss (log scale)')
        plt.title('损失函数变化（对数尺度）', fontsize=14, fontweight='bold')
        plt.legend()
        plt.grid(True, alpha=0.3)

    def plot_final_losses(self, subplot):
        """绘制最终损失对比图"""
        plt.subplot(subplot)
        names = [result['name'] for result in self.results]
        final_losses = [result['losses'][-1] if len(result['losses']) > 0 else float('inf')
                        for result in self.results]
        colors = ['skyblue', 'lightcoral', 'lightgreen']
        bars = plt.bar(names, final_losses, color=colors, alpha=0.7)

        # 添加数值标签
        for bar, loss in zip(bars, final_losses):
            plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                     f'{loss:.4f}', ha='center', va='bottom')

        plt.ylabel('Final Loss')
        plt.title('各优化器最终损失值比较', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)

    def plot_weight_adjustment(self, subplot):
        """绘制权重参数调整过程"""
        plt.subplot(subplot)
        for result in self.results:
            if len(result['weights']) > 0:
                plt.plot(result['weights'][:500], label=f"{result['name']} - w", linewidth=2)
        plt.xlabel('Epoch')
        plt.ylabel('Weight (w)')
        plt.title('权重参数w的调节过程', fontsize=14, fontweight='bold')
        plt.legend()
        plt.grid(True, alpha=0.3)

    def plot_bias_adjustment(self, subplot):
        """绘制偏置参数调整过程"""
        plt.subplot(subplot)
        for result in self.results:
            if len(result['biases']) > 0:
                plt.plot(result['biases'][:500], label=f"{result['name']} - b", linewidth=2)
        plt.xlabel('Epoch')
        plt.ylabel('Bias (b)')
        plt.title('偏置参数b的调节过程', fontsize=14, fontweight='bold')
        plt.legend()
        plt.grid(True, alpha=0.3)

    def plot_parameter_trajectory(self, subplot):
        """绘制参数空间优化轨迹"""
        plt.subplot(subplot)
        colors = ['red', 'blue', 'green']
        for i, result in enumerate(self.results):
            if len(result['weights']) > 10 and len(result['biases']) > 10:
                plt.plot(result['weights'][:100], result['biases'][:100],
                         color=colors[i], label=result['name'], linewidth=2, alpha=0.7)
                plt.scatter(result['weights'][0], result['biases'][0],
                            color=colors[i], marker='o', s=100, label=f"{result['name']} Start")
                plt.scatter(result['weights'][-1], result['biases'][-1],
                            color=colors[i], marker='*', s=150, label=f"{result['name']} End")

        plt.xlabel('Weight (w)')
        plt.ylabel('Bias (b)')
        plt.title('参数空间中的优化轨迹', fontsize=14, fontweight='bold')
        plt.legend()
        plt.grid(True, alpha=0.3)

    def plot_learning_rate_study(self, subplot):
        """绘制学习率影响研究图"""
        plt.subplot(subplot)
        learning_rates = [0.001, 0.01, 0.1, 0.5]
        lr_results = []

        for lr in learning_rates:
            print(f"\n研究学习率: {lr}")
            try:
                result = Trainer.train(Adamax, f'Adamax_lr_{lr}',
                                       self.x_data, self.y_data, lr=lr, epochs=500)
                if len(result['losses']) > 0:
                    lr_results.append(result)
                    plt.plot(result['losses'][:200], label=f'lr={lr}', linewidth=2)
            except Exception as e:
                print(f"学习率 {lr} 训练失败: {e}")

        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('不同学习率对训练的影响', fontsize=14, fontweight='bold')
        plt.legend()
        plt.grid(True, alpha=0.3)

    def plot_epoch_study(self, subplot):
        """绘制训练轮数影响研究图"""
        plt.subplot(subplot)
        epoch_settings = [100, 500, 1000, 2000]
        epoch_results = []

        for epochs in epoch_settings:
            print(f"\n研究训练轮数: {epochs}")
            try:
                result = Trainer.train(Adamax, f'Adamax_epochs_{epochs}',
                                       self.x_data, self.y_data, lr=0.002, epochs=epochs)
                if len(result['losses']) > 0:
                    epoch_results.append(result)
                    normalized_epochs = np.linspace(0, 1, len(result['losses']))
                    plt.plot(normalized_epochs, result['losses'], label=f'epochs={epochs}', linewidth=2)
            except Exception as e:
                print(f"epoch数 {epochs} 训练失败: {e}")

        plt.xlabel('Normalized Training Progress')
        plt.ylabel('Loss')
        plt.title('不同训练轮数对收敛的影响', fontsize=14, fontweight='bold')
        plt.legend()
        plt.grid(True, alpha=0.3)

    def create_comprehensive_visualization(self):
        """整合所有可视化图表"""
        plt.figure(figsize=(20, 12))  # 调整画布大小适配8个子图
        # 第一行：优化器性能（3个子图）
        self.plot_optimizer_performance(241)
        self.plot_loss_log_scale(242)
        self.plot_final_losses(243)
        # 第二行：参数调整（3个子图）
        self.plot_weight_adjustment(245)
        self.plot_bias_adjustment(246)
        self.plot_parameter_trajectory(247)
        # 第三行：超参数研究（2个子图）
        self.plot_learning_rate_study(244)
        self.plot_epoch_study(248)

        plt.tight_layout()
        plt.savefig('comprehensive_analysis.png', dpi=300, bbox_inches='tight')
        print("\n完整分析图表已保存为 'comprehensive_analysis.png'")

    def plot_final_fit(self, model):
        """绘制最终模型拟合效果"""
        try:
            plt.figure(figsize=(10, 6))
            plt.scatter(self.x_values, self.y_values, alpha=0.6, label='真实数据', s=20)

            # 生成预测曲线
            x_range_original = np.linspace(self.x_values.min(), self.x_values.max(), 100)
            x_mean, x_std = np.mean(self.x_values), np.std(self.x_values)
            x_range_normalized = (x_range_original - x_mean) / (x_std if x_std > 1e-8 else 1.0)

            x_range_tensor = torch.Tensor(x_range_normalized).reshape(-1, 1)
            y_range_normalized = model(x_range_tensor).detach().numpy().flatten()

            y_mean, y_std = np.mean(self.y_values), np.std(self.y_values)
            y_range_original = y_range_normalized * (y_std if y_std > 1e-8 else 1.0) + y_mean

            plt.plot(x_range_original, y_range_original, 'r-', linewidth=2, label='模型拟合')
            plt.xlabel('x')
            plt.ylabel('y')
            plt.title('最终模型拟合效果')
            plt.legend()
            plt.grid(True)
            plt.savefig('final_fit.png', dpi=300, bbox_inches='tight')
            print("最终拟合图表已保存为 'final_fit.png'")
        except Exception as e:
            print(f"最终拟合绘图时发生错误: {e}")


def main():
    # 配置初始化
    warnings.filterwarnings('ignore')
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False

    # 1. 数据准备
    x_data, y_data, x_values, y_values = DataProcessor.prepare_data()

    # 2. 优化器配置与训练
    optimizers = [
        (Adamax, 'Adamax'),
        (RMSprop, 'RMSprop'),
        (Rprop, 'Rprop')
    ]

    results = []
    for optim_class, optim_name in optimizers:
        # 学习率设置（与原代码一致）
        lr = 0.01
        if optim_name == 'Adamax':
            lr = 0.002
        elif optim_name == 'RMSprop':
            lr = 0.001
        elif optim_name == 'Rprop':
            lr = 0.01

        result = Trainer.train(optim_class, optim_name, x_data, y_data, lr=lr, epochs=1000)
        results.append(result)

    # 3. 可视化分析
    visualizer = Visualizer(results, x_data, y_data, x_values, y_values)
    visualizer.create_comprehensive_visualization()

    # 4. 最终模型测试
    print("\n" + "=" * 60)
    print("最终模型测试")
    print("=" * 60)

    # 选择最佳模型
    best_result = None
    for result in results:
        if len(result['losses']) > 0:
            if best_result is None or result['losses'][-1] < best_result['losses'][-1]:
                best_result = result

    # 处理无有效模型情况
    if best_result is None:
        print("没有有效的训练模型，使用Adamax重新训练...")
        final_model = LinearModel()
        criterion = nn.MSELoss()
        optimizer = Adamax(final_model.parameters(), lr=0.002)

        for epoch in range(500):
            y_pred = final_model(x_data)
            loss = criterion(y_pred, y_data)
            if torch.isnan(loss) or torch.isinf(loss):
                break
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(final_model.parameters(), max_norm=1.0)
            optimizer.step()
            if epoch % 100 == 0:
                print(f'Epoch {epoch}, Loss: {loss.item():.6f}')
    else:
        final_model = best_result['model']
        print(f"使用最佳模型: {best_result['name']}")

    # 输出最终参数
    print(f'最终参数: w = {final_model.linear.weight.item():.4f}')
    print(f'最终参数: b = {final_model.linear.bias.item():.4f}')

    # 测试预测（反标准化）
    x_test_original = 3.0
    x_mean, x_std = np.mean(x_values), np.std(x_values)
    x_test_normalized = (x_test_original - x_mean) / (x_std if x_std > 1e-8 else 1.0)
    x_test_tensor = torch.Tensor([[x_test_normalized]])

    y_test_normalized = final_model(x_test_tensor)
    y_mean, y_std = np.mean(y_values), np.std(y_values)
    y_test_original = y_test_normalized.item() * (y_std if y_std > 1e-8 else 1.0) + y_mean

    print(f'输入 x={x_test_original} 的预测结果: y_pred = {y_test_original:.4f}')

    # 绘制最终拟合图
    visualizer.plot_final_fit(final_model)

    print("\n程序执行完成!")
    print("生成的图表文件:")
    print("- comprehensive_analysis.png (完整分析图表)")
    print("- final_fit.png (最终拟合效果)")


if __name__ == "__main__":
    main()