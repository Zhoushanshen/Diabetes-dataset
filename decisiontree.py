# 基础库导入
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib
matplotlib.use('TkAgg')  # 设置 Matplotlib 后端
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (accuracy_score,
                             classification_report,
                             confusion_matrix,
                             roc_curve, auc)
from imblearn.under_sampling import RandomUnderSampler  # 添加随机欠采样库

# 设置 Matplotlib 支持中文
plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置字体为 SimHei（黑体）
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题


# 数据加载函数
def load_data(file_path):
    try:
        df = pd.read_csv(file_path, encoding='gbk', low_memory=False)
        # print("=" * 50)
        # print(f"成功加载数据集，共 {len(df)} 条记录，{len(df.columns)} 个特征")
        # print("=" * 50)
        return df
    except FileNotFoundError:
        print(f"错误：文件 {file_path} 未找到！")
        return None
    except Exception as e:
        print(f"加载数据时发生错误：{str(e)}")
        return None


# 数据完整性分析函数
def data_integrity_analysis(df):
    # print("\n" + "=" * 20 + " 数据完整性分析开始 " + "=" * 20)

    # 检查缺失值
    missing_values = df.isnull().sum()
    print("\n[缺失值统计]")
    print(missing_values.to_string())

    # 检查重复值
    duplicates = df.duplicated().sum()
    print(f"\n[重复记录] 共 {duplicates} 条重复记录")

    # print("\n" + "=" * 20 + " 分析完成 " + "=" * 20)
    return missing_values, duplicates


# 数据预处理函数
def preprocess_data(df):
    # print("\n" + "=" * 20 + " 数据预处理开始 " + "=" * 20)
    original_size = len(df)

    # 处理缺失值（中位数填充）
    if df.isnull().sum().sum() > 0:
        # print("\n[缺失值处理]")
        for col in df.columns:
            if df[col].isnull().sum() > 0:
                median_val = df[col].median()
                df[col].fillna(median_val, inplace=True)
                # print(f"特征 '{col}'：填充 {df[col].isnull().sum()} 个缺失值（中位数={median_val:.2f}）")

    # 删除非数值型特征（假设 'date_column' 是非数值型特征）
    if 'data' in df.columns:
        df = df.drop(columns=['data'])

    # 对所有非数值型特征进行独热编码
    df = pd.get_dummies(df, columns=None)  # 对所有非数值型列自动进行独热编码

    # print(f"\n数据量保持不变：{original_size} 条")
    # print("\n" + "=" * 20 + " 预处理完成 " + "=" * 20)
    return df


# 可视化目标列与其他列的皮尔森相关系数
def visualize_target_correlation(df, target_col='DM', threshold=0.7):
    # print("\n" + "=" * 40 + " 可视化目标列与其他列的皮尔森相关系数 " + "=" * 40)

    if target_col not in df.columns:
        print(f"错误：目标列 '{target_col}' 不存在！")
        print(f"当前列名：{df.columns.tolist()}")
        return None

    # 检查数据类型，确保所有列均为数值型
    non_numeric_cols = df.select_dtypes(exclude=['number']).columns
    if len(non_numeric_cols) > 0:
        # print(f"警告：以下列为非数值型，可能影响计算：{non_numeric_cols.tolist()}")
        # print("尝试删除非数值列...")
        df = df.select_dtypes(include=['number'])

    # 计算目标列与其他列的皮尔森相关系数

    try:
        # print(f"目标列 '{target_col}' 存在，继续计算相关系数...")
        target_corr = df.corr(method='pearson')[[target_col]].drop(target_col)
        # print("相关系数计算结果：")
        # print(target_corr)
    except Exception as e:
        print(f"计算相关系数时发生错误：{str(e)}")
        return None


    high_corr_features = target_corr[abs(target_corr[target_col]) > threshold].index.tolist()
    if high_corr_features:
        # print(f"\n以下特征与目标列 '{target_col}' 的相关系数绝对值大于 {threshold}，将被排除：")
        # print(high_corr_features)
        df = df.drop(columns=high_corr_features)
    else:
        print(f"\n没有特征与目标列 '{target_col}' 的相关系数绝对值大于 {threshold}。")
    return df, target_corr


# 建模评估函数
def train_evaluate_model(df):
    if 'DM' not in df.columns:
        print("错误：数据集中未找到目标变量 'DM'")
        return

    X = df.drop('DM', axis=1)
    y = df['DM']
    # 进行随机欠采样，使两类样本数量相同
    rus = RandomUnderSampler(random_state=42)
    X_resampled, y_resampled = rus.fit_resample(X, y)

    # 交叉验证
    model = DecisionTreeClassifier(
        max_depth=3,  # 控制树复杂度
        min_samples_split=10,
        random_state=42
    )
    cv_scores = cross_val_score(model, X_resampled, y_resampled, cv=5, scoring='accuracy')
    print(f"5 折交叉验证平均准确率：{np.mean(cv_scores):.2%}")

    # 数据集划分
    X_train, X_test, y_train, y_test = train_test_split(
        X_resampled, y_resampled,
        test_size=0.6,
        random_state=42,
        stratify=y_resampled  # 保持类别分布
    )

    # 模型训练
    model.fit(X_train, y_train)

    # 预测评估
    y_pred = model.predict(X_test)

    # print("\n" + "=" * 20 + " 模型评估 " + "=" * 20)
    print(f"测试集准确率：{accuracy_score(y_test, y_pred):.2%}")
    print("\n分类报告：")
    print(classification_report(y_test, y_pred))
    print("\n混淆矩阵：")
    conf_matrix = confusion_matrix(y_test, y_pred)
    print(conf_matrix)

    # 混淆矩阵可视化
    plt.figure(figsize=(8, 6))
    vmin = 0
    vmax = 200
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', vmin=vmin, vmax=vmax)
    plt.xlabel('预测标签')
    plt.ylabel('真实标签')
    plt.title('混淆矩阵热力图')
    plt.show()

    # 计算 AUC 和绘制 ROC 曲线
    y_score = model.predict_proba(X_test)[:, 1]
    fpr, tpr, thresholds = roc_curve(y_test, y_score)
    roc_auc = auc(fpr, tpr)

    print(f"AUC 值: {roc_auc:.2f}")

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f' ROC 曲线 (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('假正率')
    plt.ylabel('真正率')
    plt.title(' ROC - AUC 曲线')
    plt.legend(loc="lower right")
    plt.show()

    return model


# 主流程
if __name__ == "__main__":
    # 参数配置
    DATA_PATH = 'data/jiujing.csv'

    # 数据加载
    raw_df = load_data(DATA_PATH)
    if raw_df is None:
        exit()

    # 完整性分析
    missing, duplicates = data_integrity_analysis(raw_df)

    # 数据预处理
    processed_df = preprocess_data(raw_df.copy())

    # 可视化目标列与其他列的皮尔森相关系数，并排除高相关性列
    processed_df, target_corr = visualize_target_correlation(processed_df, target_col='DM', threshold=0.7)

    # 建模评估
    trained_model = train_evaluate_model(processed_df)