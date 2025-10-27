import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, \
    classification_report
import warnings

warnings.filterwarnings('ignore')

# 1. 数据加载（请替换为实际文件路径）
# 若从Kaggle下载，文件名为'diabetes_health_indicators.csv'
df = pd.read_csv('diabetes_dataset.csv')

# 2. 数据探索
print("数据集基本信息：")
print(f"样本量: {df.shape[0]}, 特征数: {df.shape[1]}")
print("\n目标变量分布：")
print(df['diagnosed_diabetes'].value_counts(normalize=True).round(3))

# 查看数值特征统计描述
numeric_features = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
numeric_features.remove('diagnosed_diabetes')  # 排除目标变量
print("\n数值特征统计描述：")
print(df[numeric_features].describe().round(2))

# 相关性分析（数值特征与目标变量）
corr = df[numeric_features + ['diagnosed_diabetes']].corr()
plt.figure(figsize=(12, 8))
sns.heatmap(corr[['diagnosed_diabetes']].sort_values(by='diagnosed_diabetes', ascending=False),
            annot=True, cmap='coolwarm', fmt='.2f')
plt.title('特征与糖尿病的相关性')
plt.tight_layout()
plt.show()

# 3. 数据预处理
# 区分数值特征和类别特征
categorical_features = df.select_dtypes(include=['object']).columns.tolist()

# 划分特征与目标变量
X = df.drop('diagnosed_diabetes', axis=1)
y = df['diagnosed_diabetes']

# 划分训练集和测试集（8:2）
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 构建预处理管道：数值特征标准化，类别特征独热编码
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(drop='first', sparse_output=False), categorical_features)
    ])


# 4. 模型训练与评估
def evaluate_model(model, X_train, X_test, y_train, y_test, model_name):
    """评估模型性能并输出指标"""
    # 训练模型
    model.fit(X_train, y_train)

    # 预测
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    # 计算指标
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_proba)

    # 输出结果
    print(f"\n{model_name} 性能指标：")
    print(f"准确率: {accuracy:.4f}")
    print(f"精确率: {precision:.4f}")
    print(f"召回率: {recall:.4f}")
    print(f"F1分数: {f1:.4f}")
    print(f"ROC-AUC: {roc_auc:.4f}")

    # 混淆矩阵
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['未患病', '患病'],
                yticklabels=['未患病', '患病'])
    plt.xlabel('预测标签')
    plt.ylabel('真实标签')
    plt.title(f'{model_name} 混淆矩阵')
    plt.tight_layout()
    plt.show()

    return model


# 构建带预处理的模型管道
models = [
    ('逻辑回归', LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42)),
    ('随机森林', RandomForestClassifier(class_weight='balanced', n_estimators=100, random_state=42)),
    ('梯度提升树', GradientBoostingClassifier(n_estimators=100, random_state=42))
]

# 训练并评估所有模型
trained_models = {}
for name, model in models:
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', model)
    ])
    trained_models[name] = evaluate_model(pipeline, X_train, X_test, y_train, y_test, name)

# 5. 特征重要性分析（针对树模型）
if '随机森林' in trained_models:
    rf_model = trained_models['随机森林']['classifier']
    # 获取独热编码后的特征名
    cat_encoder = preprocessor.named_transformers_['cat']
    cat_feature_names = list(cat_encoder.get_feature_names_out(categorical_features))
    all_feature_names = numeric_features + cat_feature_names

    # 提取特征重要性
    importances = rf_model.feature_importances_
    indices = np.argsort(importances)[-10:]  # 取Top10重要特征

    plt.figure(figsize=(10, 6))
    plt.barh(range(len(indices)), importances[indices], color='b', align='center')
    plt.yticks(range(len(indices)), [all_feature_names[i] for i in indices])
    plt.xlabel('特征重要性')
    plt.title('随机森林Top10重要特征')
    plt.tight_layout()
    plt.show()

import matplotlib.pyplot as plt
# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
# 解决负号显示问题
plt.rcParams['axes.unicode_minus'] = False

print("\n===== 模型评估总结 =====")
print(
    f"最佳模型（ROC-AUC）: {max(trained_models, key=lambda x: roc_auc_score(y_test, trained_models[x].predict_proba(X_test)[:, 1]))}")