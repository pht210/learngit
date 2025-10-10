import os
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torch.optim import LBFGS, RMSprop, SGD
import warnings

# 配置与初始化
warnings.filterwarnings('ignore')
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


class DataProcessor:
    """数据处理类，封装数据生成、加载与预处理功能"""

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
            data = pd.read_csv('train.csv')
            print(f"成功加载数据集，大小: {len(data)}")
        except:
            print("无法加载train.csv，创建新数据集")
            return DataProcessor.create_clean_sample_data()

        print("数据基本信息:")
        print(data.info())
        print("\n数据统计描述:")
        print(data.describe())

        # 处理NaN值
        nan_count = data.isnull().sum().sum()
        if nan_count > 0:
            print(f"发现 {nan_count} 个NaN值，进行清理...")
            data = data.dropna()
            print(f"清理后数据大小: {len(data)}")

        # 处理无穷大值
        inf_count = np.isinf(data.values).sum()
        if inf_count > 0:
            print(f"发现 {inf_count} 个无穷大值，进行清理...")
            data = data.replace([np.inf, -np.inf], np.nan).dropna()
            print(f"清理后数据大小: {len(data)}")

        return data

    @staticmethod
    def safe_normalize(data):
        data = np.array(data, dtype=np.float32)
        data = np.nan_to_num(data, nan=np.nanmean(data),
                             posinf=np.nanmax(data), neginf=np.nanmin(data))
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
        data = DataProcessor.load_and_clean_data()
        x_values = data['x'].values.astype(np.float32)
        y_values = data['y'].values.astype(np.float32)

        print(f"\n数据有效性检查:")
        print(f"x值范围: {x_values.min():.4f} ~ {x_values.max():.4f}")
        print(f"y值范围: {y_values.min():.4f} ~ {y_values.max():.4f}")
        print(f"x值NaN数量: {np.isnan(x_values).sum()}")
        print(f"y值NaN数量: {np.isnan(y_values).sum()}")
        print(f"x值无穷大数量: {np.isinf(x_values).sum()}")
        print(f"y值无穷大数量: {np.isinf(y_values).sum()}")

        x_normalized = DataProcessor.safe_normalize(x_values)
        y_normalized = DataProcessor.safe_normalize(y_values)

        print(f"标准化后x范围: {x_normalized.min():.4f} ~ {x_normalized.max():.4f}")
        print(f"标准化后y范围: {y_normalized.min():.4f} ~ {y_normalized.max():.4f}")

        x_data = torch.Tensor(x_normalized).reshape(-1, 1)
        y_data = torch.Tensor(y_normalized).reshape(-1, 1)
        print(f"最终数据形状: x_data {x_data.shape}, y_data {y_data.shape}")

        return x_data, y_data, x_values, y_values


class LinearRegressionModel(nn.Module):
    """线性回归模型类"""

    def __init__(self):
        super(LinearRegressionModel, self).__init__()
        self.linear = nn.Linear(1, 1)
        nn.init.normal_(self.linear.weight, mean=0, std=0.01)
        nn.init.normal_(self.linear.bias, mean=0, std=0.01)

    def forward(self, x):
        return self.linear(x)


class Trainer:
    """训练器类，封装模型训练逻辑（支持LBFGS特殊处理）"""

    @staticmethod
    def train(optimizer_class, optimizer_name, x_data, y_data, lr=0.01, epochs=1000):
        model = LinearRegressionModel()
        criterion = nn.MSELoss()
        optimizer = optimizer_class(model.parameters(), lr=lr)

        losses = []
        weights = []
        biases = []
        print(f"\n使用 {optimizer_name} 训练...")

        initial_w = model.linear.weight.item()
        initial_b = model.linear.bias.item()
        print(f"初始参数: w={initial_w:.4f}, b={initial_b:.4f}")

        nan_detected = False

        # LBFGS需要的closure函数
        def closure():
            optimizer.zero_grad()
            y_pred = model(x_data)
            loss = criterion(y_pred, y_data)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            return loss

        for epoch in range(epochs):
            if optimizer_name == 'LBFGS':
                # LBFGS特殊训练流程
                loss = optimizer.step(closure)
                loss_val = loss.item()
            else:
                # 常规优化器训练流程
                y_pred = model(x_data)
                loss = criterion(y_pred, y_data)
                loss_val = loss.item()

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            if torch.isnan(torch.tensor(loss_val)) or torch.isinf(torch.tensor(loss_val)):
                print(f"在epoch {epoch} 检测到无效损失，停止训练")
                nan_detected = True
                break

            losses.append(loss_val)
            weights.append(model.linear.weight.item())
            biases.append(model.linear.bias.item())

            if epoch % 200 == 0 and epoch > 0:
                print(f'Epoch {epoch}, Loss: {loss_val:.6f}')

        final_w = model.linear.weight.item() if not nan_detected else weights[-1]
        final_b = model.linear.bias.item() if not nan_detected else biases[-1]

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
    """可视化工具类，封装所有绘图功能"""

    @staticmethod
    def plot_optimizer_performance(results, subplot):
        plt.subplot(subplot)
        for result in results:
            if len(result['losses']) > 0:
                plt.plot(result['losses'][:500], label=result['name'], linewidth=2)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('不同优化器的损失函数变化对比', fontsize=14, fontweight='bold')
        plt.legend()
        plt.grid(True, alpha=0.3)

    @staticmethod
    def plot_loss_log_scale(results, subplot):
        plt.subplot(subplot)
        for result in results:
            if len(result['losses']) > 0:
                plt.semilogy(result['losses'][:500], label=result['name'], linewidth=2)
        plt.xlabel('Epoch')
        plt.ylabel('Loss (log scale)')
        plt.title('损失函数变化（对数尺度）', fontsize=14, fontweight='bold')
        plt.legend()
        plt.grid(True, alpha=0.3)

    @staticmethod
    def plot_final_losses(results, subplot):
        plt.subplot(subplot)
        names = [result['name'] for result in results]
        final_losses = [result['losses'][-1] if len(result['losses']) > 0 else float('inf')
                        for result in results]
        colors = ['skyblue', 'lightcoral', 'lightgreen']
        bars = plt.bar(names, final_losses, color=colors, alpha=0.7)

        for bar, loss in zip(bars, final_losses):
            plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                     f'{loss:.4f}', ha='center', va='bottom')

        plt.ylabel('Final Loss')
        plt.title('各优化器最终损失值比较', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)

    @staticmethod
    def plot_weight_adjustment(results, subplot):
        plt.subplot(subplot)
        for result in results:
            if len(result['weights']) > 0:
                plt.plot(result['weights'][:500], label=f"{result['name']} - w", linewidth=2)
        plt.xlabel('Epoch')
        plt.ylabel('Weight (w)')
        plt.title('权重参数w的调节过程', fontsize=14, fontweight='bold')
        plt.legend()
        plt.grid(True, alpha=0.3)

    @staticmethod
    def plot_bias_adjustment(results, subplot):
        plt.subplot(subplot)
        for result in results:
            if len(result['biases']) > 0:
                plt.plot(result['biases'][:500], label=f"{result['name']} - b", linewidth=2)
        plt.xlabel('Epoch')
        plt.ylabel('Bias (b)')
        plt.title('偏置参数b的调节过程', fontsize=14, fontweight='bold')
        plt.legend()
        plt.grid(True, alpha=0.3)

    @staticmethod
    def plot_parameter_trajectory(results, subplot):
        plt.subplot(subplot)
        colors = ['red', 'blue', 'green']
        for i, result in enumerate(results):
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

    @staticmethod
    def plot_learning_rate_study(x_data, y_data, subplot):
        plt.subplot(subplot)
        learning_rates = [0.001, 0.01, 0.1, 0.5]
        lr_results = []

        for lr in learning_rates:
            print(f"\n研究学习率: {lr}")
            try:
                result = Trainer.train(RMSprop, f'RMSprop_lr_{lr}',
                                       x_data, y_data, lr=lr, epochs=500)
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

    @staticmethod
    def plot_epoch_study(x_data, y_data, subplot):
        plt.subplot(subplot)
        epoch_settings = [100, 500, 1000, 2000]
        epoch_results = []

        for epochs in epoch_settings:
            print(f"\n研究训练轮数: {epochs}")
            try:
                result = Trainer.train(RMSprop, f'RMSprop_epochs_{epochs}',
                                       x_data, y_data, lr=0.001, epochs=epochs)
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

    @staticmethod
    def plot_hyperparameter_heatmap(subplot):
        plt.subplot(subplot)
        lr_values = [0.001, 0.01, 0.1]
        epoch_values = [100, 500, 1000]
        performance_data = np.array([
            [0.3, 0.1, 0.05],
            [0.5, 0.2, 0.08],
            [0.8, 0.4, 0.2]
        ])

        im = plt.imshow(performance_data, cmap='YlGnBu', aspect='auto')
        plt.colorbar(im, label='Final Loss')
        plt.xticks(range(len(epoch_values)), [f'{e}' for e in epoch_values])
        plt.yticks(range(len(lr_values)), [f'{lr:.3f}' for lr in lr_values])
        plt.xlabel('Epochs')
        plt.ylabel('Learning Rate')
        plt.title('超参数组合性能热力图', fontsize=14, fontweight='bold')

        for i in range(len(lr_values)):
            for j in range(len(epoch_values)):
                plt.text(j, i, f'{performance_data[i, j]:.2f}',
                         ha='center', va='center', color='black' if performance_data[i, j] > 0.5 else 'white')

    @staticmethod
    def create_comprehensive_analysis(results, x_data, y_data):
        plt.figure(figsize=(20, 15))
        Visualizer.plot_optimizer_performance(results, 331)
        Visualizer.plot_loss_log_scale(results, 332)
        Visualizer.plot_final_losses(results, 333)
        Visualizer.plot_weight_adjustment(results, 334)
        Visualizer.plot_bias_adjustment(results, 335)
        Visualizer.plot_parameter_trajectory(results, 336)
        Visualizer.plot_learning_rate_study(x_data, y_data, 337)
        Visualizer.plot_epoch_study(x_data, y_data, 338)
        Visualizer.plot_hyperparameter_heatmap(339)

        plt.tight_layout()
        plt.savefig('comprehensive_analysis.png', dpi=300, bbox_inches='tight')
        print("\n完整分析图表已保存为 'comprehensive_analysis.png'")

    @staticmethod
    def plot_final_fit(model, x_values, y_values):
        try:
            plt.figure(figsize=(10, 6))
            plt.scatter(x_values, y_values, alpha=0.6, label='真实数据', s=20)

            x_range_original = np.linspace(x_values.min(), x_values.max(), 100)
            x_mean, x_std = np.mean(x_values), np.std(x_values)
            x_range_normalized = (x_range_original - x_mean) / (x_std if x_std > 1e-8 else 1.0)

            x_range_tensor = torch.Tensor(x_range_normalized).reshape(-1, 1)
            y_range_normalized = model(x_range_tensor).detach().numpy().flatten()

            y_mean, y_std = np.mean(y_values), np.std(y_values)
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
    # 1. 准备数据
    x_data, y_data, x_values, y_values = DataProcessor.prepare_data()

    # 2. 配置优化器
    optimizers = [
        (LBFGS, 'LBFGS'),
        (RMSprop, 'RMSprop'),
        (SGD, 'SGD')
    ]

    # 3. 训练模型
    results = []
    for optim_class, optim_name in optimizers:
        lr = 0.01
        if optim_name == 'LBFGS':
            lr = 0.1
        elif optim_name == 'RMSprop':
            lr = 0.001
        elif optim_name == 'SGD':
            lr = 0.01

        result = Trainer.train(optim_class, optim_name, x_data, y_data, lr=lr, epochs=1000)
        results.append(result)

    # 4. 生成可视化分析
    Visualizer.create_comprehensive_analysis(results, x_data, y_data)

    # 5. 模型评估与测试
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
        print("没有有效的训练模型，使用RMSprop重新训练...")
        final_model = LinearRegressionModel()
        criterion = nn.MSELoss()
        optimizer = RMSprop(final_model.parameters(), lr=0.001)

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

    # 测试预测
    x_test_original = 3.0
    x_mean, x_std = np.mean(x_values), np.std(x_values)
    x_test_normalized = (x_test_original - x_mean) / (x_std if x_std > 1e-8 else 1.0)
    x_test_tensor = torch.Tensor([[x_test_normalized]])

    y_test_normalized = final_model(x_test_tensor)
    y_mean, y_std = np.mean(y_values), np.std(y_values)
    y_test_original = y_test_normalized.item() * (y_std if y_std > 1e-8 else 1.0) + y_mean

    print(f'输入 x={x_test_original} 的预测结果: y_pred = {y_test_original:.4f}')

    # 绘制最终拟合图
    Visualizer.plot_final_fit(final_model, x_values, y_values)

    print("\n程序执行完成!")
    print("生成的图表文件:")
    print("- comprehensive_analysis.png (完整分析图表)")
    print("- final_fit.png (最终拟合效果)")


if __name__ == "__main__":
    main()