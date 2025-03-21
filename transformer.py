import warnings
warnings.filterwarnings("ignore", category=UserWarning, message="unrecognized nn.Module: LayerNorm")
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, confusion_matrix, classification_report, roc_curve, auc
from statsmodels.stats.outliers_influence import variance_inflation_factor
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import shap
import matplotlib
matplotlib.use('TkAgg')  # 设置 Matplotlib 后端
import matplotlib.pyplot as plt
import seaborn as sns

# 设置 Matplotlib 支持中文
plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置字体为 SimHei（黑体）
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 欠采样（Undersampling）
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

# 定义简化后的Transformer模型
class DiabetesTransformer(nn.Module):
    def __init__(self, input_dim, d_model=16, nhead=2, num_layers=1):
        super().__init__()
        self.feature_embedding = nn.Linear(1, d_model)
        self.position_embedding = nn.Parameter(torch.randn(1, input_dim, d_model))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=64,
            batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.classifier = nn.Sequential(
            nn.Linear(d_model * input_dim, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, 2))

    def forward(self, x):
        x = x.unsqueeze(-1)  # (batch_size, input_dim, 1)
        x = self.feature_embedding(x)
        x = x + self.position_embedding  # 非原地操作
        x = self.transformer(x)
        x = x.flatten(start_dim=1)
        return self.classifier(x)

# SHAP 特征重要性分析
def shap_analysis(model, background_data, test_samples, feature_names):
    # 创建一个 SHAP 解释器
    explainer = shap.DeepExplainer(model, background_data)
    shap_values = explainer.shap_values(test_samples, check_additivity=False)

    # 假设 shap_values 是每个类别的 SHAP 值列表，取正类别（索引为 1）的 SHAP 值
    shap_values_positive = np.array(shap_values[1])
    shap_values_positive = shap_values_positive.reshape(test_samples.shape[0], -1)[:, :background_data.shape[1]]

    # 可视化特征重要性
    shap.summary_plot(
        shap_values_positive,
        test_samples.cpu().numpy(),
        feature_names=feature_names,
        plot_type="dot"
    )
    plt.show()

# 计算 VIF 并去除高共线性特征，同时打印高度共线性特征
def calculate_vif(X, threshold=10):
    vif_data = pd.DataFrame()
    vif_data["feature"] = X.columns
    vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]

    high_vif_features = []
    # 逐步删除高 VIF 特征
    while True:
        max_vif = vif_data["VIF"].max()
        if max_vif <= threshold:
            break
        remove_feature = vif_data.loc[vif_data["VIF"].idxmax(), "feature"]
        high_vif_features.append(remove_feature)
        X = X.drop(remove_feature, axis=1)
        vif_data = vif_data[vif_data["feature"] != remove_feature]

    return X

# 数据加载和预处理
file_path = 'data/jiujing.csv'
df = pd.read_csv(file_path, encoding='gbk', low_memory=False)

# 数据清洗流程
if 'data' in df.columns:
    df = df.drop('data', axis=1)

df = df.apply(pd.to_numeric, errors='coerce')
df = df.dropna(axis=1, how='any')

if 'DM' not in df.columns:
    raise KeyError("目标列 'DM' 不存在")

# 计算 DM 与其他列的皮尔森相关系数
corr_with_DM = df.corr()['DM'].abs().drop('DM')
high_corr_columns = corr_with_DM[corr_with_DM > 0.7].index.tolist()
df = df.drop(columns=high_corr_columns, errors='ignore')

# 分离特征和目标变量
X = df.drop('DM', axis=1)
y = df['DM']

# 计算 VIF 并去除高共线性特征
X = calculate_vif(X)

# 应用欠采样
X_undersampled, y_undersampled = undersample(X, y)

# 五折交叉验证
kf = KFold(n_splits=5, shuffle=True, random_state=42)
accuracies = []
aucs = []
f1_scores = []
confusion_matrices = []
classification_reports_list = []

for train_index, test_index in kf.split(X_undersampled):
    X_train, X_test = X_undersampled.iloc[train_index], X_undersampled.iloc[test_index]
    y_train, y_test = y_undersampled.iloc[train_index], y_undersampled.iloc[test_index]

    # 标准化处理
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 转换为PyTorch Tensor
    X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train.values, dtype=torch.long)
    y_test_tensor = torch.tensor(y_test.values, dtype=torch.long)

    # 创建数据加载器
    batch_size = 32
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # 模型初始化
    input_dim = X_train.shape[1]
    model = DiabetesTransformer(input_dim=input_dim)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # 训练配置
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    # 训练循环
    best_auc = 0
    for epoch in range(20):  # 减少训练轮数
        model.train()
        total_loss = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        # 验证过程
        model.eval()
        y_true, y_pred, y_probs = [], [], []
        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                X_batch = X_batch.to(device)
                outputs = model(X_batch)
                probs = torch.softmax(outputs, dim=1)[:, 1]
                y_probs.extend(probs.cpu().numpy())
                y_pred.extend(torch.argmax(outputs, dim=1).cpu().numpy())
                y_true.extend(y_batch.numpy())

        # 计算指标
        acc = accuracy_score(y_true, y_pred)
        current_auc = roc_auc_score(y_true, y_probs)
        f1 = f1_score(y_true, y_pred)

        # 早停机制
        if current_auc > best_auc:
            best_auc = current_auc
            torch.save(model.state_dict(), "best_model.pth")

    # 加载最佳模型
    model.load_state_dict(torch.load("best_model.pth"))

    # 最终评估
    model.eval()
    y_true, y_pred, y_probs = [], [], []
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(device)
            outputs = model(X_batch)
            probs = torch.softmax(outputs, dim=1)[:, 1]
            y_probs.extend(probs.cpu().numpy())
            y_pred.extend(torch.argmax(outputs, dim=1).cpu().numpy())
            y_true.extend(y_batch.numpy())

    # 计算指标
    acc = accuracy_score(y_true, y_pred)
    current_auc = roc_auc_score(y_true, y_probs)
    f1 = f1_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)
    class_report = classification_report(y_true, y_pred)

    accuracies.append(acc)
    aucs.append(current_auc)
    f1_scores.append(f1)
    confusion_matrices.append(cm)
    classification_reports_list.append(class_report)

# 打印五折交叉验证结果
print("五折交叉验证结果：")
print(f"平均准确率: {np.mean(accuracies):.2%}")

# 打印最后一次的分类报告
print("分类性能报告：")
print(classification_reports_list[-1])

# 打印最后一次的混淆矩阵
cm = confusion_matrices[-1]

# 混淆矩阵可视化（最后一次交叉验证）
plt.figure(figsize=(8, 6))
vmin = 0
vmax = 80
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', vmin=vmin, vmax=vmax)
plt.xlabel('预测标签')
plt.ylabel('真实标签')
plt.title('混淆矩阵热力图')
plt.show()

# 绘制 ROC - AUC 曲线
fpr, tpr, thresholds = roc_curve(y_true, y_probs)
roc_auc = auc(fpr, tpr)

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

# SHAP 分析
# 重新划分数据集用于 SHAP 分析
X_train, X_test, y_train, y_test = train_test_split(
    X_undersampled, y_undersampled, test_size=0.3, stratify=y_undersampled, random_state=42, shuffle=True)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
background_data = torch.tensor(X_train_scaled[:200], dtype=torch.float32).to(device)  # 减少背景数据量
test_samples = torch.tensor(X_test_scaled[:2], dtype=torch.float32).to(device)  # 减少测试数据量
feature_names = X_undersampled.columns.tolist()

shap_analysis(model, background_data, test_samples, feature_names)