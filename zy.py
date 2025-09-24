import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# 数据加载
df = pd.read_csv('train.csv').dropna()
x = df['x'].values
y = df['y'].values

# 模型定义（向量版）
def predict_vector(x, w, b):
    return x * w + b

def compute_mse_vector(x, y, w, b):
    y_pred = predict_vector(x, w, b)
    losses = (y_pred - y)** 2
    # 打印每个样本的详细信息（保留原打印逻辑）
    for xi, yi, yp, loss in zip(x, y, y_pred, losses):
        print(f'x_val={xi}, y_val={yi}, y_pred_val={yp}, loss_val={loss}')
    return np.mean(losses)

# 网格搜索参数
w_vals = np.arange(0.5, 2.51, 0.05)
b_vals = np.arange(-3.0, 2.01, 0.05)

# 存储结果
results = []
for w in w_vals:
    for b in b_vals:
        mse = compute_mse_vector(x, y, w, b)
        print(f'MSE={mse}')
        results.append((w, b, mse))

# 最优参数
best_w, best_b, best_mse = min(results, key=lambda item: item[2])
print(f"\n最优参数：w={best_w:.4f}, b={best_b:.4f}")
print(f"最优MSE：{best_mse:.4f}")

# 可视化
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

w_list = [r[0] for r in results]
b_list = [r[1] for r in results]
mse_list = [r[2] for r in results]

# w-MSE图
mask_best_b = np.isclose(b_list, best_b)
plt.figure(figsize=(12, 5))
plt.subplot(121)
plt.plot(np.array(w_list)[mask_best_b], np.array(mse_list)[mask_best_b], '#1f77b4')
plt.scatter(best_w, best_mse, c='red', label=f'最优w：{best_w:.4f}')
plt.xlabel('w（斜率）')
plt.ylabel('MSE')
plt.title(f'w与MSE关系（b={best_b:.4f}）')
plt.legend()
plt.grid(alpha=0.3)

# b-MSE图
mask_best_w = np.isclose(w_list, best_w)
plt.subplot(122)
plt.plot(np.array(b_list)[mask_best_w], np.array(mse_list)[mask_best_w], '#ff7f0e')
plt.scatter(best_b, best_mse, c='red', label=f'最优b：{best_b:.4f}')
plt.xlabel('b（截距）')
plt.ylabel('MSE')
plt.title(f'b与MSE关系（w={best_w:.4f}）')
plt.legend()
plt.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('w_b_loss_relation.png', dpi=300)
print("可视化图已保存为：w_b_loss_relation.png")