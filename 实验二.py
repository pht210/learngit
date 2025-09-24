import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv('train.csv')
df.dropna(inplace=True)
x_data = df['x'].values
y_data = df['y'].values

def forward(x, w, b):
    return x * w + b

def loss(x, y, w, b):
    y_pred = forward(x, w, b)
    return (y_pred - y) ** 2


# 3. 网格搜索训练：遍历w、b计算MSE
w_range = np.arange(0.5, 1.51, 0.05)
b_range = np.arange(-3.0, 3.01, 0.05)

w_list = []
b_list = []
mse_list = []

for w in w_range:
    for b in b_range:
        l_sum = 0
        for x_val, y_val in zip(x_data, y_data):
            y_pred_val = forward(x_val, w, b)
            loss_val = loss(x_val, y_val, w, b)
            l_sum += loss_val
        mse = l_sum / len(x_data)
        print(f'w={w:.2f}, b={b:.2f}, MSE={mse:.4f}')  # 简化打印信息，避免冗余
        w_list.append(w)
        b_list.append(b)
        mse_list.append(mse)

min_mse_idx = np.argmin(mse_list)
best_w = w_list[min_mse_idx]
best_b = b_list[min_mse_idx]
best_mse = mse_list[min_mse_idx]
print(f"\n最优参数：w={best_w:.4f}, b={best_b:.4f}")
print(f"最优参数对应的MSE：{best_mse:.4f}")


plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


best_b_mask = np.isclose(b_list, best_b)
w_best_b = np.array(w_list)[best_b_mask]
mse_best_b = np.array(mse_list)[best_b_mask]

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(w_best_b, mse_best_b, color='#1f77b4', linewidth=2)
plt.scatter(best_w, best_mse, color='red', s=50, label=f'最优w：{best_w:.4f}')
plt.xlabel('参数 w（斜率）', fontsize=12)
plt.ylabel('MSE（均方误差）', fontsize=12)
plt.title(f'w与MSE的关系（固定b={best_b:.4f}）', fontsize=14)
plt.legend()
plt.grid(alpha=0.3)

best_w_mask = np.isclose(w_list, best_w)
b_best_w = np.array(b_list)[best_w_mask]
mse_best_w = np.array(mse_list)[best_w_mask]

plt.subplot(1, 2, 2)
plt.plot(b_best_w, mse_best_w, color='#ff7f0e', linewidth=2)
plt.scatter(best_b, best_mse, color='red', s=50, label=f'最优b：{best_b:.4f}')
plt.xlabel('参数 b（截距）', fontsize=12)
plt.ylabel('MSE（均方误差）', fontsize=12)
plt.title(f'b与MSE的关系（固定w={best_w:.4f}）', fontsize=14)
plt.legend()
plt.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('w_b_loss_relation.png', dpi=300, bbox_inches='tight')
print("\n可视化图已保存为：w_b_loss_relation.png")