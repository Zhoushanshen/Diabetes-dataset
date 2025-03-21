import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import StandardScaler
from imblearn.under_sampling import RandomUnderSampler
import seaborn as sns
import matplotlib
matplotlib.use('TkAgg')  # 设置 Matplotlib 后端
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置字体为 SimHei（黑体）
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题


# 加载数据集
file_path = "data/jiujing.csv"

# 读取CSV文件，指定编码格式为'gbk'
try:
    df = pd.read_csv(file_path, encoding='gbk', low_memory=False)
except UnicodeDecodeError:
    try:
        df = pd.read_csv(file_path, encoding='latin1', low_memory=False)
    except UnicodeDecodeError:
        df = pd.read_csv(file_path, encoding='ISO-8859-1', low_memory=False)

# 数据预处理
# 删除非数值列
if 'data' in df.columns:
    df = df.drop('data', axis=1)

# 强制转换所有列为数值类型（无法转换的设为NaN）
df = df.apply(pd.to_numeric, errors='coerce')

# 删除仍包含非数值的列
df = df.dropna(axis=1, how='any')

# 分离特征和标签
if 'DM' not in df.columns:
    raise KeyError("数据集中缺少目标列 'DM'，请检查列名")
X = df.drop('DM', axis=1)
y = df['DM']

# 处理缺失值
# 删除缺失值过多的列
missing_percent = X.isnull().mean()
X = X.loc[:, missing_percent < 0.5]  # 删除缺失率超过50%的列
X = X.fillna(X.mean())

# 计算皮尔森相关系数
correlation_matrix = X.corrwith(y)

# 删除与DM相关系数绝对值高于0.7的列
high_corr_columns = correlation_matrix[abs(correlation_matrix) > 0.7].index
X = X.drop(columns=high_corr_columns)

# 欠采样
rus = RandomUnderSampler(random_state=42)
X_resampled, y_resampled = rus.fit_resample(X, y)

# 特征标准化
scaler = StandardScaler()
X_resampled = scaler.fit_transform(X_resampled)

# 数据集划分
X_train, X_test, y_train, y_test = train_test_split(
    X_resampled, y_resampled, test_size=0.3, random_state=42, stratify=y_resampled
)

# 初始化模型
model = GaussianNB()

# 交叉验证
cv_scores = cross_val_score(model, X_resampled, y_resampled, cv=5, scoring='accuracy')
print(f"交叉验证准确率：{np.mean(cv_scores):.2%} (±{np.std(cv_scores):.2%})")

# 训练模型
model.fit(X_train, y_train)

# 预测
y_pred = model.predict(X_test)

# ROC曲线
fpr,tpr,thresholds = roc_curve(y_test,y_pred)
roc_auc = auc(fpr,tpr)

# 计算准确率
accuracy = accuracy_score(y_test, y_pred)
print(f"\nAccuracy: {accuracy:.2%}")

# 打印分类报告
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# 打印混淆矩阵
conf_matrix = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:")
print(conf_matrix)

# 绘制ROC曲线
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f' ROC 曲线 (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('假正率')
plt.ylabel('真正率')
plt.title(' ROC - AUC 曲线')
plt.legend(loc='lower right')
plt.show()
# 绘制混淆矩阵热力图
plt.figure(figsize=(8, 6))
vmin = 0
vmax = 120
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', vmin=vmin, vmax=vmax)
plt.xlabel('预测标签')
plt.ylabel('真实标签')
plt.title('混淆矩阵热力图')
plt.show()