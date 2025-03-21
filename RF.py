import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib
matplotlib.use('TkAgg')  # 设置 Matplotlib 后端
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (accuracy_score,
                             classification_report,
                             confusion_matrix,
                             roc_curve, auc)
from sklearn.metrics import ConfusionMatrixDisplay  # 导入用于可视化混淆矩阵的类
from imblearn.under_sampling import RandomUnderSampler

# 设置字体为 SimHei（黑体）
plt.rcParams['font.sans-serif'] = ['SimHei']
# 解决负号显示问题
plt.rcParams['axes.unicode_minus'] = False

# 数据加载与预处理
def load_and_preprocess_data(file_path):
    try:
        # 加载数据
        df = pd.read_csv(file_path, encoding='gbk', low_memory=False)

        # 处理缺失值（假设存在缺失值）
        if df.isnull().sum().sum() > 0:
            print("\n[缺失值处理]")
            for col in df.columns:
                if df[col].isnull().sum() > 0:
                    median_val = df[col].median()
                    df[col].fillna(median_val, inplace=True)
                    print(f"列 '{col}': 填充 {df[col].isnull().sum()} 个缺失值（中位数={median_val:.2f}）")

        # 删除非数值列（根据实际情况调整）
        non_numeric = df.select_dtypes(exclude=['number']).columns
        if len(non_numeric) > 0:
            df = df.drop(columns=non_numeric)

        return df
    except Exception as e:
        print(f"数据加载失败: {str(e)}")
        return None

# 目标列与其他列的皮尔森相关系数
def visualize_target_correlation(df, target_col='DM', threshold=0.7):
    if target_col not in df.columns:
        print(f"错误：目标列 '{target_col}' 不存在！")
        print(f"当前列名：{df.columns.tolist()}")
        return None

    # 检查数据类型，确保所有列均为数值型
    non_numeric_cols = df.select_dtypes(exclude=['number']).columns
    if len(non_numeric_cols) > 0:
        df = df.select_dtypes(include=['number'])

    # 计算目标列与其他列的皮尔森相关系数
    try:
        target_corr = df.corr(method='pearson')[[target_col]].drop(target_col)
    except Exception as e:
        print(f"计算相关系数时发生错误：{str(e)}")
        return None

    # 筛选出高相关性列并排除
    high_corr_features = target_corr[abs(target_corr[target_col]) > threshold].index.tolist()
    if high_corr_features:
        df = df.drop(columns=high_corr_features)
    else:
        print(f"\n没有特征与目标列 '{target_col}' 的相关系数绝对值大于 {threshold}。")
    return df, target_corr

# 建模与评估
def train_and_evaluate(df, target_col='DM'):
    # 划分特征和目标
    X = df.drop(target_col, axis=1)
    y = df[target_col]

    # 处理样本不平衡
    rus = RandomUnderSampler(random_state=42)
    X_res, y_res = rus.fit_resample(X, y)
    # 初始化随机森林
    model = RandomForestClassifier(
        n_estimators=50,
        max_depth=2,
        min_samples_split=20,  # 增加分裂所需最小样本数
        min_samples_leaf=10,  # 增加叶节点最小样本数
        max_features='sqrt',  # 限制每棵树使用的特征数量
        class_weight='balanced',
        random_state=42
    )

    # 交叉验证
    cv_scores = cross_val_score(model, X_res, y_res, cv=5, scoring='accuracy')
    print(f"5 折交叉验证平均准确率: {np.mean(cv_scores):.2%}")

    # 划分训练测试集
    X_train, X_test, y_train, y_test = train_test_split(
        X_res, y_res,
        test_size=0.4,
        random_state=42,
        stratify=y_res,
        shuffle=True
    )

    # 训练模型
    model.fit(X_train, y_train)

    # 预测评估
    y_pred = model.predict(X_test)

    # 评估指标
    print(f"测试集准确率: {accuracy_score(y_test, y_pred):.2%}")
    print("\n分类报告:")
    print(classification_report(y_test, y_pred))

    # 计算混淆矩阵
    cm = confusion_matrix(y_test, y_pred)
    print(cm)

    # 可视化混淆矩阵
    vmin = 0
    vmax = 150
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', vmin=vmin, vmax=vmax)
    plt.xlabel('预测标签')
    plt.ylabel('真实标签')
    plt.title('混淆矩阵热力图')
    plt.show()

    # 计算 ROC - AUC 曲线
    y_score = model.predict_proba(X_test)[:, 1]
    fpr, tpr, thresholds = roc_curve(y_test, y_score)
    roc_auc = auc(fpr, tpr)

    print(f"AUC 值: {roc_auc:.2f}")

    # 绘制 ROC 曲线
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC Curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc="lower right")
    plt.show()

    return model

# 主流程
if __name__ == "__main__":
    # 参数配置
    DATA_PATH = 'data/jiujing.csv'  # 请替换为实际路径
    TARGET_COL = 'DM'  # 目标列名称

    # 数据加载与预处理
    df = load_and_preprocess_data(DATA_PATH)
    if df is None:
        exit()

    # 可视化目标列与其他列的皮尔森相关系数，并排除高相关性列
    processed_df, target_corr = visualize_target_correlation(df, target_col='DM', threshold=0.7)

    # 建模与评估
    trained_model = train_and_evaluate(processed_df, TARGET_COL)