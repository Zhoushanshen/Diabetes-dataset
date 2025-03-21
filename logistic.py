import numpy as np
import pandas as pd
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import shap
import matplotlib

matplotlib.use('TkAgg')  # 设置 Matplotlib 后端
import matplotlib.pyplot as plt

# 1. 加载数据
file_path = "data/jiujing.csv"
df = pd.read_csv(file_path, encoding='gbk', low_memory=False)

# 数据预处理
# 删除非数值列（如果存在）
non_numeric_cols = df.select_dtypes(exclude=np.number).columns
df = df.drop(non_numeric_cols, axis=1)

# 强制转换所有列为数值类型并删除缺失值
df = df.apply(pd.to_numeric, errors='coerce').dropna(axis=1, how='any')


# 2. 用 VIF 去除多重共线性特征
def calculate_vif(df, threshold=10):
    vif_data = pd.DataFrame()
    vif_data["feature"] = df.columns
    vif_data["VIF"] = [variance_inflation_factor(df.values, i) for i in range(df.shape[1])]

    # 逐步删除高 VIF 特征
    while True:
        max_vif = vif_data["VIF"].max()
        if max_vif <= threshold:
            break
        remove_feature = vif_data.loc[vif_data["VIF"].idxmax(), "feature"]
        df = df.drop(remove_feature, axis=1)
        vif_data = vif_data[vif_data["feature"] != remove_feature]
    return df


# 分离特征和目标变量
X = df.drop("DM", axis=1)
y = df["DM"]

# 计算 VIF 并删除高共线性特征
X_vif_filtered = calculate_vif(X, threshold=10)

# 2. 计算皮尔森相关系数
# 删除与 DM 相关系数高于 0.7 的列
if 'DM' in df.columns:
    corr_matrix = df.corr().abs()
    high_corr_columns = corr_matrix[corr_matrix['DM'] > 0.7].index.tolist()
    # 排除 DM 列本身
    high_corr_columns = [col for col in high_corr_columns if col != 'DM']
    df = df.drop(columns=high_corr_columns, errors='ignore')


# 4. 欠采样（Undersampling）
def undersample(X, y):
    # 分离多数类和少数类
    majority_class = 0  # 假设多数类是 0
    minority_class = 1  # 假设少数类是 1

    # 获取各类别的样本数
    counts = {}
    for target_class in [majority_class, minority_class]:
        counts[target_class] = np.sum(y == target_class)

    # 确定欠采样的目标
    target_samples = counts[minority_class]

    # 欠采样多数类
    majority_indices = np.where(y == majority_class)[0]
    selected_majority = np.random.choice(majority_indices, size=target_samples, replace=False)

    minority_indices = np.where(y == minority_class)[0]

    # 合并欠采样后的多数类和少数类
    undersampled_indices = np.concatenate([selected_majority, minority_indices])
    X_undersampled = X.iloc[undersampled_indices]
    y_undersampled = y.iloc[undersampled_indices]

    return X_undersampled, y_undersampled


# 应用欠采样
X_undersampled, y_undersampled = undersample(X_vif_filtered, y)

# 5. 数据预处理
# 划分数据集（先划分再标准化！）
X_train, X_test, y_train, y_test = train_test_split(
    X_undersampled.values, y_undersampled.values, test_size=0.3, random_state=42, stratify=y_undersampled
)

# 标准化（仅用训练集统计量）
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 6. 构建并训练逻辑回归模型
model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(X_train, y_train)

# 7. 评估模型
y_pred = model.predict(X_test)

# 计算准确率
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2%}")

# 输出分类报告
print("分类报告：\n", classification_report(y_test, y_pred))

# 输出混淆矩阵
print("混淆矩阵：\n", confusion_matrix(y_test, y_pred))


# 8. SHAP 特征重要性分析
def shap_analysis(model, X_train):
    # 创建一个 SHAP 解释器
    explainer = shap.LinearExplainer(model, X_train)
    shap_values = explainer.shap_values(X_train)

    # 可视化特征重要性
    shap.summary_plot(shap_values, X_train, feature_names=X_vif_filtered.columns.tolist())


# 随机采样部分数据用于 SHAP（避免内存不足）
X_train_sample = X_train[:100]  # 取前 100 个样本
shap_analysis(model, X_train_sample)