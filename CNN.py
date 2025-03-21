import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_curve, auc
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import seaborn as sns
import matplotlib
matplotlib.use('TkAgg')  # 设置 Matplotlib 后端
import matplotlib.pyplot as plt

# 构建CNN模型
class CNNModel(nn.Module):
    def __init__(self, input_channels, input_length, output_units):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv1d(input_channels, 32, kernel_size=3)
        self.pool1 = nn.MaxPool1d(kernel_size=2)
        self.dropout1 = nn.Dropout(0.2)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=2)
        self.pool2 = nn.MaxPool1d(kernel_size=2)
        self.dropout2 = nn.Dropout(0.3)
        # 计算经过卷积和池化层后的输出大小
        self.fc1_input_size = self._calculate_fc1_input_size(input_length)

        self.fc1 = nn.Linear(self.fc1_input_size, 64)
        self.dropout3 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(64, output_units)

    def _calculate_fc1_input_size(self, input_length):
        # 计算经过 conv1 和 pool1 后的输出大小
        conv1_output = input_length - 3 + 1  # (input_length - kernel_size + 1)
        pool1_output = conv1_output // 2  # (conv1_output / pool_size)

        # 计算经过 conv2 和 pool2 后的输出大小
        conv2_output = pool1_output - 2 + 1  # (pool1_output - kernel_size + 1)
        pool2_output = conv2_output // 2  # (conv2_output / pool_size)

        # fc1 的输入大小是通道数乘以 pool2 后的长度
        return 64 * pool2_output

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = self.pool1(x)
        x = self.dropout1(x)
        x = torch.relu(self.conv2(x))
        x = self.pool2(x)
        x = self.dropout2(x)
        x = torch.flatten(x, start_dim=1)
        x = torch.relu(self.fc1(x))
        x = self.dropout3(x)
        x = torch.sigmoid(self.fc2(x))
        return x

# 1. 加载数据
file_path = "data/jiujing.csv"
# 读取CSV文件
df = pd.read_csv(file_path, encoding='gbk', low_memory=False)
# 数据预处理
# 删除非数值列
if 'data' in df.columns:
    df = df.drop('data', axis=1)

# 强制转换所有列为数值类型
df = df.apply(pd.to_numeric, errors='coerce')

# 删除仍包含非数值的列
df = df.dropna(axis=1, how='any')

# 2. 计算皮尔森相关系数
# 删除与 DM 相关系数高于 0.7 的列
if 'DM' in df.columns:
    corr_matrix = df.corr().abs()
    high_corr_columns = corr_matrix[corr_matrix['DM'] > 0.7].index.tolist()
    # 排除 DM 列本身
    high_corr_columns = [col for col in high_corr_columns if col != 'DM']
    df = df.drop(columns=high_corr_columns, errors='ignore')

# 3. 欠采样（Undersampling）
# 分离特征和标签
X = df.drop('DM', axis=1).values
y = df['DM'].values

# 获取多数类和少数类的索引
majority_class = 0
minority_class = 1

counts = {}
for target_class in [majority_class, minority_class]:
    counts[target_class] = np.sum(y == target_class)

target_samples = counts[minority_class]

# 欠采样多数类
majority_indices = np.where(y == majority_class)[0]
selected_majority = np.random.choice(majority_indices, size=target_samples, replace=False)

minority_indices = np.where(y == minority_class)[0]

# 合并欠采样后的多数类和少数类
undersampled_indices = np.concatenate([selected_majority, minority_indices])
X = X[undersampled_indices]
y = y[undersampled_indices]

# 转换为张量
X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.float32).unsqueeze(1)

# 五折交叉验证
kf = KFold(n_splits=5, shuffle=True, random_state=42)
accuracies = []
classification_reports = []
confusion_matrices = []
fprs = []
tprs = []
aucs = []

for train_index, test_index in kf.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    # 标准化数据
    scaler = StandardScaler()
    X_train = torch.tensor(
        scaler.fit_transform(X_train), dtype=torch.float32
    ).unsqueeze(2)
    X_test = torch.tensor(
        scaler.transform(X_test), dtype=torch.float32
    ).unsqueeze(2)

    # 调整为 PyTorch 的输入形状 (batch_size, channels, length)
    X_train = X_train.permute(0, 2, 1)
    X_test = X_test.permute(0, 2, 1)

    # 创建数据加载器
    batch_size = 16
    train_dataset = TensorDataset(X_train, y_train)
    test_dataset = TensorDataset(X_test, y_test)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # 获取输入通道数
    input_channels = X_train.shape[1]
    input_length = X_train.shape[2]

    # 初始化模型
    model = CNNModel(input_channels, input_length, 1)

    # 定义损失函数和优化器
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 训练模型
    num_epochs = 100
    for epoch in range(num_epochs):
        model.train()
        for inputs, labels in train_loader:
            # 前向传播
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    # 评估模型
    model.eval()
    with torch.no_grad():
        y_pred_prob = []
        y_true = []
        for inputs, labels in test_loader:
            outputs = model(inputs)
            y_pred_prob.extend(outputs.squeeze().numpy())
            y_true.extend(labels.squeeze().numpy())

        y_pred_prob = np.array(y_pred_prob)
        y_pred = (y_pred_prob > 0.5).astype(int)
        y_true = np.array(y_true)

        # 计算并记录Accuracy
        accuracy = accuracy_score(y_true, y_pred)
        accuracies.append(accuracy)

        # 记录其他评估指标
        classification_reports.append(classification_report(y_true, y_pred))
        confusion_matrices.append(confusion_matrix(y_true, y_pred))

        # 计算 ROC 曲线和 AUC
        fpr, tpr, _ = roc_curve(y_true, y_pred_prob)
        roc_auc = auc(fpr, tpr)
        fprs.append(fpr)
        tprs.append(tpr)
        aucs.append(roc_auc)

# 计算平均准确率
average_accuracy = np.mean(accuracies)
print(f"五折交叉验证平均准确率: {average_accuracy:.2%}")

# 打印最后一次折叠的分类报告和混淆矩阵
# print("最后一次折叠的 Classification Report:")
print(classification_reports[-1])
# print("最后一次折叠的 Confusion Matrix:")
print(confusion_matrices[-1])

# 混淆矩阵可视化（最后一次折叠）
cm = confusion_matrices[-1]
plt.figure(figsize=(8, 6))
vmin = 0
vmax = 100
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', vmin=vmin, vmax=vmax)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()

# ROC - AUC 曲线可视化（平均）
plt.figure(figsize=(8, 6))
mean_fpr = np.linspace(0, 1, 100)
tprs_interp = []
for i in range(len(fprs)):
    tprs_interp.append(np.interp(mean_fpr, fprs[i], tprs[i]))
    tprs_interp[-1][0] = 0.0
mean_tpr = np.mean(tprs_interp, axis=0)
mean_tpr[-1] = 1.0
mean_auc = auc(mean_fpr, mean_tpr)

plt.plot(mean_fpr, mean_tpr, color='darkorange', lw=2, label=f'ROC Curve (AUC = {mean_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc="lower right")
plt.show()