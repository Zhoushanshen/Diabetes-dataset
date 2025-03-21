import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from imblearn.under_sampling import RandomUnderSampler
import matplotlib
matplotlib.use('TkAgg')  # 设置 Matplotlib 后端
import matplotlib.pyplot as plt
import seaborn as sns

# 加载糖尿病数据集
file_path = "data/jiujing.csv"

# 读取CSV文件，指定编码格式为'gbk'
try:
    df = pd.read_csv(file_path, encoding='gbk', low_memory=False)
except UnicodeDecodeError:
    try:
        df = pd.read_csv(file_path, encoding='latin1', low_memory=False)
    except UnicodeDecodeError:
        df = pd.read_csv(file_path, encoding='ISO-8859-1', low_memory=False)

# 检查数据是否加载成功
# 修复1：处理非数值列和 missing data
# 删除日期列
if 'data' in df.columns:
    df = df.drop('data', axis=1)

# 强制转换所有列为数值类型（无法转换的设为NaN）
df = df.apply(pd.to_numeric, errors='coerce')

# 删除仍包含非数值的列（可选）
df = df.dropna(axis=1, how='any')

# 分离特征和标签
if 'DM' not in df.columns:
    raise KeyError("数据集中缺少目标列 'DM'，请检查列名")
X = df.drop('DM', axis=1)
y = df['DM']

# 修复4：优化相关性分析
# 计算皮尔森相关性
correlation = X.corrwith(y)

# 选择相关性小于0.7的列
high_corr_features = [col for col in X.columns if abs(correlation[col]) <= 0.7]
X_filtered = X[high_corr_features]

# 修复2：处理缺失值
# 填充缺失值（先删除缺失值过多的列）
missing_percent = X_filtered.isnull().mean()
X_filtered = X_filtered.loc[:, missing_percent < 0.5]  # 删除缺失率超过50%的列
X_filtered = X_filtered.fillna(X_filtered.mean())

# 修复3：添加欠采样步骤
rus = RandomUnderSampler(random_state=42)
X_resampled, y_resampled = rus.fit_resample(X_filtered, y)

# 后续建模步骤
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.3, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

svm_classifier = SVC(kernel='linear', random_state=42)
svm_classifier.fit(X_train, y_train)

y_pred = svm_classifier.predict(X_test)

print(f"\nAccuracy: {accuracy_score(y_test, y_pred):.2%}")
print("Confusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
print(cm)
print("Classification Report:")
print(classification_report(y_test, y_pred))

# 混淆矩阵可视化
plt.figure(figsize=(8, 6))
vmin = 0
vmax = 120
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', vmin=vmin, vmax=vmax)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()