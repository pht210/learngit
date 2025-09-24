# ------------------------------------------------------
# 模块1：导入依赖库
# ------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# ------------------------------------------------------
# 模块2：数据加载与预处理
# ------------------------------------------------------
# 读取CSV文件
raw_data = pd.read_csv('train.csv')
# 移除缺失值
clean_data = raw_data.dropna()
# 提取特征与标签
feature = clean_data['x'].values
label = clean_data['y'].values
sample_count = len(feature)

# ------------------------------------------------------
# 模块3：定义模型核心计算
# ------------------------------------------------------
def linear_model(x, weight, bias):
    """线性模型：y = weight*x + bias"""
    return x * weight + bias

def mean_squared_error(x, y, w, b):
    """计算均方误差"""
    total_error = 0
    for xi, yi in zip(x, y):
        prediction = linear_model(xi, w, b)
        error = (prediction - yi) **2
        total_error += error
        # 打印单样本信息
        print(f'x_val={xi}, y_val={yi}, y_pred_val={prediction}, loss_val={error}')
    return total_error / sample_count

# ------------------------------------------------------
# 模块4：网格搜索参数空间
# ------------------------------------------------------
# 定义参数搜索范围
weight_range = np.arange(0.5, 1.51, 0.05)
bias_range = np.arange(-3.0, 3.01, 0.05)

# 记录所有参数组合的MSE
weight_log = []
bias_log = []
mse_log = []

# 遍历所有参数组合
for w in weight_range:
    for b in bias_range:
        current_mse = mean_squared_error(feature, label, w, b)
        print(f'MSE={current_mse}')
        weight_log.append(w)
        bias_log.append(b)
        mse_log.append(current_mse)

# ------------------------------------------------------
# 模块5：确定最优参数
# ------------------------------------------------------
# 找到MSE最小的索引
optimal_index = np.argmin(mse_log)
optimal_w = weight_log[optimal_index]
optimal_b = bias_log[optimal_index]
optimal_mse = mse_log[optimal_index]

# 输出最优结果
print(f"\n最优参数：w={optimal_w:.4f}, b={optimal_b:.4f}")
print(f"最优参数对应的MSE：{optimal_mse:.4f}")

# ------------------------------------------------------
# 模块6：结果可视化
# ------------------------------------------------------
# 设置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 创建画布
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# 第一个子图：w与MSE的关系（固定最优b）
mask_b = np.isclose(bias_log, optimal_b)
ax1.plot(np.array(weight_log)[mask_b], np.array(mse_log)[mask_b], color='#1f77b4')
ax1.scatter(optimal_w, optimal_mse, color='red', s=50, label=f'最优w：{optimal_w:.4f}')
ax1.set_xlabel('参数 w（斜率）')
ax1.set_ylabel('MSE（均方误差）')
ax1.set_title(f'w与MSE的关系（b={optimal_b:.4f}）')
ax1.legend()
ax1.grid(alpha=0.3)

# 第二个子图：b与MSE的关系（固定最优w）
mask_w = np.isclose(weight_log, optimal_w)
ax2.plot(np.array(bias_log)[mask_w], np.array(mse_log)[mask_w], color='#ff7f0e')
ax2.scatter(optimal_b, optimal_mse, color='red', s=50, label=f'最优b：{optimal_b:.4f}')
ax2.set_xlabel('参数 b（截距）')
ax2.set_ylabel('MSE（均方误差）')
ax2.set_title(f'b与MSE的关系（w={optimal_w:.4f}）')
ax2.legend()
ax2.grid(alpha=0.3)

# 保存图像
plt.tight_layout()
plt.savefig('w_b_loss_relation.png', dpi=300, bbox_inches='tight')
print("\n可视化图已保存为：w_b_loss_relation.png")