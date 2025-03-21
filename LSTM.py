import warnings
warnings.filterwarnings("ignore", category=UserWarning)
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, roc_curve, auc
from sklearn.impute import SimpleImputer
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import seaborn as sns
import matplotlib
matplotlib.use('TkAgg')  # 设置 Matplotlib 后端
import matplotlib.pyplot as plt

# 1. 加载数据
file_path = "data/jiujing.csv"
df = pd.read_csv(file_path, encoding='gbk', low_memory=False)

# 删除日期列（在数据划分之前）
if 'data' in df.columns:
    df = df.drop('data', axis=1)

# 强制转换所有列为数值类型（无法转换的设为NaN）
df = df.apply(pd.to_numeric, errors='coerce')

# 删除仍包含非数值的列（可选）
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
# 分离特征和目标变量
X = df.drop('DM', axis=1).values
y = df['DM'].values

# 获取多数类和少数类的索引
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
X = X[undersampled_indices]
y = y[undersampled_indices]

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

    # 4. 数据预处理
    # 处理缺失值（仅用训练集统计量填充）
    imputer = SimpleImputer(strategy='mean')
    X_train = imputer.fit_transform(X_train)
    X_test = imputer.transform(X_test)

    # 标准化（仅用训练集统计量）
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # 转换为 PyTorch 张量
    X_train = torch.tensor(X_train, dtype=torch.float32).unsqueeze(1)  # 形状: (samples, 1, features)
    X_test = torch.tensor(X_test, dtype=torch.float32).unsqueeze(1)
    y_train = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
    y_test = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)

    # 创建数据加载器
    batch_size = 32
    train_dataset = TensorDataset(X_train, y_train)
    test_dataset = TensorDataset(X_test, y_test)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # 5. 构建LSTM模型
    class LSTMModel(nn.Module):
        def __init__(self, input_size, hidden_size1, hidden_size2, output_size):
            super(LSTMModel, self).__init__()
            self.lstm1 = nn.LSTM(input_size, hidden_size1, batch_first=True)
            self.dropout1 = nn.Dropout(0.3)
            self.lstm2 = nn.LSTM(hidden_size1, hidden_size2, batch_first=True)
            self.dropout2 = nn.Dropout(0.3)
            self.fc = nn.Linear(hidden_size2, output_size)
            self.sigmoid = nn.Sigmoid()

        def forward(self, x):
            # 输入形状: (batch_size, 1, input_size)
            out, _ = self.lstm1(x)
            out = self.dropout1(out)
            out, _ = self.lstm2(out)
            out = self.dropout2(out)
            # 取最后一个时间步的输出
            out = out[:, -1, :]
            out = self.fc(out)
            out = self.sigmoid(out)
            return out

    input_size = X_train.shape[2]  # 特征数量
    hidden_size1 = 64
    hidden_size2 = 32
    output_size = 1

    model = LSTMModel(input_size, hidden_size1, hidden_size2, output_size)

    # 6. 定义损失函数和优化器
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 7. 定义早停机制
    class EarlyStopping:
        def __init__(self, patience=10, delta=0):
            self.patience = patience
            self.delta = delta
            self.best_loss = None
            self.early_stop = False
            self.counter = 0

        def __call__(self, val_loss):
            if self.best_loss is None:
                self.best_loss = val_loss
            elif val_loss > self.best_loss - self.delta:
                self.counter += 1
                if self.counter >= self.patience:
                    self.early_stop = True
            else:
                self.best_loss = val_loss
                self.counter = 0

    early_stop = EarlyStopping(patience=10, delta=0.01)  # 设置耐心和最小改进值

    # 8. 训练模型
    num_epochs = 50
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    for epoch in range(num_epochs):
        model.train()
        total_train_loss = 0.0  # 用于记录训练损失

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            # 前向传播
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            total_train_loss += loss.item()

            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # 计算平均训练损失
        avg_train_loss = total_train_loss / len(train_loader)

        # 验证阶段
        model.eval()
        total_val_loss = 0.0
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                total_val_loss += loss.item()

        # 计算平均验证损失
        avg_val_loss = total_val_loss / len(test_loader)

        # 更新早停机制
        early_stop(avg_val_loss)
        if early_stop.early_stop:
            print("Early stopping triggered")
            break

    # 9. 评估模型
    model.eval()
    with torch.no_grad():
        y_pred = []
        y_true = []
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            y_pred.extend(outputs.cpu().numpy().flatten().tolist())
            y_true.extend(labels.cpu().numpy().flatten().tolist())

        # 应用阈值获得二进制预测
        y_pred_binary = np.array(y_pred) > 0.5
        y_true = np.array(y_true)

        # 计算准确率
        accuracy = accuracy_score(y_true, y_pred_binary)
        accuracies.append(accuracy)

        # 输出分类报告
        classification_reports.append(classification_report(y_true, y_pred_binary))

        # 计算混淆矩阵
        cm = confusion_matrix(y_true, y_pred_binary)
        confusion_matrices.append(cm)

        # 计算 ROC 曲线和 AUC
        fpr, tpr, _ = roc_curve(y_true, y_pred)
        roc_auc = auc(fpr, tpr)
        fprs.append(fpr)
        tprs.append(tpr)
        aucs.append(roc_auc)

# 计算平均准确率
average_accuracy = np.mean(accuracies)
print(f"五折交叉验证平均准确率: {average_accuracy:.2%}")

# 混淆矩阵可视化（最后一次折叠）
cm = confusion_matrices[-1]
plt.figure(figsize=(8, 6))
vmin = 0
vmax = 70
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', vmin=vmin, vmax=vmax)
plt.xlabel('预测标签')
plt.ylabel('真实标签')
plt.title('混淆矩阵热力图')
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
