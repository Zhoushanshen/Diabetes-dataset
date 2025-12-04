import warnings
warnings.filterwarnings("ignore", category=UserWarning, message="unrecognized nn.Module: LayerNorm")
import pandas as pd
import matplotlib
matplotlib.use('TkAgg')  # 设置 Matplotlib 后端
import matplotlib.pyplot as plt
import seaborn as sns

plt.rcParams['font.family'] = 'Times New Roman'

# 1. 读取数据
file_path = r"D:\糖尿病数据\数据集.csv"
df = pd.read_csv(file_path, encoding='gbk', low_memory=False)

# 2. 选择关键连续变量
continuous_vars = ['BMI', 'FPG', 'PG2h', 'TG', 'CHOL', 'HDL', 'LDL', 'FINS']

# 3. 将这些列强制转换为数值型（无法转换的自动变 NaN）
for var in continuous_vars:
    df[var] = pd.to_numeric(df[var], errors='coerce')

# 4. 描述性统计
desc_stats = df[continuous_vars].describe().T
print("Descriptive Statistics:")
print(desc_stats)

# 5. 绘制直方图 + KDE
sns.set(style="white")
for var in continuous_vars:
    plt.figure(figsize=(8, 5))
    sns.histplot(
        df[var].dropna(),
        bins=30,
        color=(31 / 255, 122 / 255, 183 / 255),  # RGB 归一化到 0~1
        edgecolor=None,  # 去掉柱状边框
        kde=False
    )
    sns.kdeplot(df[var].dropna(), color='darkblue', linewidth=2)
#    plt.title(f'Distribution of {var}', fontsize=14)
    plt.xlabel(var, fontsize=12)
    # 仅对 PG2h 限制横坐标范围 0-30
    if var == 'PG2h':
        plt.xlim(0, 25)
    plt.ylabel('Number of patients', fontsize=12)
    plt.grid(axis='y', color='gray', linestyle='-', linewidth=0.5, alpha=0.5)
    plt.tight_layout()
    plt.show()
    plt.savefig(f'{var}_distribution.png', dpi=300)
    plt.close()
