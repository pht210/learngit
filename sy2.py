import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


class LinearRegressionGridSearch:
    def __init__(self, filepath):
        self.x_data, self.y_data = self._load_data(filepath)
        self.w_range = np.arange(0.5, 1.51, 0.05)
        self.b_range = np.arange(-3.0, 3.01, 0.05)
        self.results = []  # 存储(w, b, mse)

    def _load_data(self, filepath):
        """内部数据加载方法"""
        df = pd.read_csv(filepath)
        df.dropna(inplace=True)
        return df['x'].values, df['y'].values

    def _predict(self, x, w, b):
        """内部预测方法"""
        return x * w + b

    def _compute_mse(self, w, b):
        """计算单个参数组合的MSE"""
        total_loss = 0
        for x, y in zip(self.x_data, self.y_data):
            pred = self._predict(x, w, b)
            loss = (pred - y) ** 2
            total_loss += loss
            print(f'x={x}, y={y}, pred={pred}, loss={loss}')
        mse = total_loss / len(self.x_data)
        print(f'MSE={mse}')
        return mse

    def train(self):
        """执行网格搜索训练"""
        for w in self.w_range:
            for b in self.b_range:
                mse = self._compute_mse(w, b)
                self.results.append((w, b, mse))

        # 提取最优参数
        best = min(self.results, key=lambda x: x[2])
        self.best_w, self.best_b, self.best_mse = best
        return self.best_w, self.best_b, self.best_mse

    def visualize(self):
        """可视化结果"""
        plt.rcParams['font.sans-serif'] = ['SimHei']
        plt.rcParams['axes.unicode_minus'] = False

        # 拆分结果为列表
        w_list = [r[0] for r in self.results]
        b_list = [r[1] for r in self.results]
        mse_list = [r[2] for r in self.results]

        # 绘制w-MSE图
        best_b_mask = np.isclose(b_list, self.best_b)
        plt.figure(figsize=(12, 5))
        plt.subplot(121)
        plt.plot(np.array(w_list)[best_b_mask], np.array(mse_list)[best_b_mask], '#1f77b4')
        plt.scatter(self.best_w, self.best_mse, color='red', label=f'最优w：{self.best_w:.4f}')
        plt.xlabel('w（斜率）')
        plt.ylabel('MSE')
        plt.title(f'w与MSE关系（b={self.best_b:.4f}）')
        plt.legend()
        plt.grid(alpha=0.3)

        # 绘制b-MSE图
        best_w_mask = np.isclose(w_list, self.best_w)
        plt.subplot(122)
        plt.plot(np.array(b_list)[best_w_mask], np.array(mse_list)[best_w_mask], '#ff7f0e')
        plt.scatter(self.best_b, self.best_mse, color='red', label=f'最优b：{self.best_b:.4f}')
        plt.xlabel('b（截距）')
        plt.ylabel('MSE')
        plt.title(f'b与MSE关系（w={self.best_w:.4f}）')
        plt.legend()
        plt.grid(alpha=0.3)

        plt.tight_layout()
        plt.savefig('w_b_loss_relation.png', dpi=300)
        print("可视化图已保存为：w_b_loss_relation.png")


# 主执行逻辑
if __name__ == "__main__":
    model = LinearRegressionGridSearch('train.csv')
    best_w, best_b, best_mse = model.train()
    print(f"\n最优参数：w={best_w:.4f}, b={best_b:.4f}")
    print(f"最优MSE：{best_mse:.4f}")
    model.visualize()