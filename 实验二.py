import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# ----------------------
# 1. 数据处理块
# ----------------------
df = pd.read_csv('train.csv')
df.dropna(inplace=True)
x = df['x'].values
y = df['y'].values
n_samples = len(x)

# ----------------------
# 2. 参数与结果存储块
# ----------------------
w_candidates = np.arange(0.5, 1.51, 0.05)
b_candidates = np.arange(-3.0, 3.01, 0.05)
w_records, b_records, mse_records = [], [], []

# ----------------------
# 3. 网格搜索计算块
# ----------------------
for w in w_candidates:
    for b in b_candidates:
        total_loss = 0.0
        # 计算每个样本的损失
        for xi, yi in zip(x, y):
            y_pred = xi * w + b
            loss = (y_pred - yi) **2
            total_loss += loss
            print(f'x={xi}, y={yi}, pred={y_pred}, loss={loss}')
        # 计算MSE并记录
        current_mse = total_loss / n_samples
        print(f'MSE={current_mse}')
        w_records.append(w)
        b_records.append(b)
        mse_records.append(current_mse)

# ----------------------
# 4. 最优参数提取块
# ----------------------
min_idx = np.argmin(mse_records)
best_w = w_records[min_idx]
best_b = b_records[min_idx]
best_mse = mse_records[min_idx]
print(f"\n最优参数：w={best_w:.4f}, b={best_b:.4f}")
print(f"最优MSE：{best_mse:.4f}")

# ----------------------
# 5. 可视化块
# ----------------------
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 子图1：w与MSE（固定最优b）
mask_b = np.isclose(b_records, best_b)
plt.figure(figsize=(12, 5))
plt.subplot(121)
plt.plot(np.array(w_records)[mask_b], np.array(mse_records)[mask_b], '#1f77b4')
plt.scatter(best_w, best_mse, color='red', label=f'最优w：{best_w:.4f}')
plt.xlabel('w（斜率）')
plt.ylabel('MSE')
plt.title(f'w与MSE关系（b={best_b:.4f}）')
plt.legend()
plt.grid(alpha=0.3)

# 子图2：b与MSE（固定最优w）
mask_w = np.isclose(w_records, best_w)
plt.subplot(122)
plt.plot(np.array(b_records)[mask_w], np.array(mse_records)[mask_w], '#ff7f0e')
plt.scatter(best_b, best_mse, color='red', label=f'最优b：{best_b:.4f}')
plt.xlabel('b（截距）')
plt.ylabel('MSE')
plt.title(f'b与MSE关系（w={best_w:.4f}）')
plt.legend()
plt.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('w_b_loss_relation.png', dpi=300)
print("可视化图已保存为：w_b_loss_relation.png")