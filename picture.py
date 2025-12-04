import warnings
warnings.filterwarnings("ignore", category=UserWarning)
import pandas as pd
import matplotlib

matplotlib.use('TkAgg')  # 设置 Matplotlib 后端
import matplotlib.pyplot as plt
import random
import numpy as np
from scipy.signal import medfilt
import seaborn as sns

plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 读取数据
df = pd.read_csv('data/jiujing.csv', encoding='gbk', low_memory=False)

# 查看数据基本信息
print(df.info())
# 统计每列缺失值数量
print(df.isnull().sum())
# 删除缺失值较多的列（假设删除缺失值超过一半的列）
df = df.dropna(axis=1, thresh=len(df) * 0.5)

# 排除 'NO' 列（数据集中未提及，若存在可按此排除）
if 'NO' in df.columns:
    df = df.drop(columns=['NO'])

# 第一组要绘制箱线图的列
group1_cols = [
    'height', 'weight', 'waist1', 'waist2', 'hip', 'HR',
    'Bpsys', 'Bpdia', 'FINS', 'CP2h', 'HGB'
]
valid_group1_cols = [col for col in group1_cols if col in df.columns]

# 第二组要绘制箱线图的列
group2_cols = ['GGT', 'PLT']
valid_group2_cols = [col for col in group2_cols if col in df.columns]

# 第三组要绘制箱线图的列
group3_cols = ['INS2h', 'ALT', 'AST', 'ALP']
valid_group3_cols = [col for col in group3_cols if col in df.columns]

# 第四组要绘制箱线图的列
group4_cols = [
    'BMI', 'WHR', 'FPG', 'PG2h', 'HbA1c', 'FCP',
    'CHOL', 'TG', 'HDL', 'LDL', 'WBC', 'RBC', 'RDWCV'
]
valid_group4_cols = [col for col in group4_cols if col in df.columns]

# 第五组要绘制箱线图的列
group5_cols = ['RDWSD', 'MCV']
valid_group5_cols = [col for col in group5_cols if col in df.columns]


# 确保所选列都是数值类型并删除包含 NaN 的行
def process_columns(df, cols):
    for col in cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    return df.dropna(subset=cols)


df = process_columns(df, valid_group1_cols)
df = process_columns(df, valid_group2_cols)
df = process_columns(df, valid_group3_cols)
df = process_columns(df, valid_group4_cols)
df = process_columns(df, valid_group5_cols)


# 将 RGB 值转换为 (r, g, b) 元组形式，取值范围 0 到 1
def rgb_to_tuple(rgb):
    r, g, b = [int(x) for x in rgb.split(',')]
    return r / 255, g / 255, b / 255


# 13 个新的颜色
colors = [
    rgb_to_tuple('238,140,125'),
    rgb_to_tuple('80,29,138'),
    rgb_to_tuple('63,171,71'),
    rgb_to_tuple('201,71,51'),
    rgb_to_tuple('253,223,139'),
    rgb_to_tuple('139,91,38'),
    rgb_to_tuple('82,185,216'),
    rgb_to_tuple('229,81,06'),
    rgb_to_tuple('0,148,255'),
    rgb_to_tuple('170,52,116'),
    rgb_to_tuple('255,146,0'),
    rgb_to_tuple('207,185,158'),
    rgb_to_tuple('46,95,161')
]


# 定义绘制箱线图的函数
def plot_boxplot(data, columns):
    # 对数据进行降噪处理
    denoised_data = data.copy()
    for col in columns:
        denoised_data[col] = medfilt(denoised_data[col], kernel_size=3)

    # 按照 RGB 顺序选取颜色
    num_boxes = len(columns)
    random_colors = colors[:num_boxes]

    # 使用 matplotlib 的 boxplot 函数绘制箱线图
    fig, ax = plt.subplots(figsize=(12, 6))
    bp = ax.boxplot(denoised_data[columns].values, patch_artist=True)

    # 遍历每个箱体并设置颜色
    for patch, color in zip(bp['boxes'], random_colors):
        patch.set_facecolor(color)
        patch.set_edgecolor(color)

    # 设置须线、中位数线等的颜色
    for element in ['whiskers', 'caps']:
        for line, color in zip(bp[element], random_colors * 2):
            line.set_color(color)

    for median, color in zip(bp['medians'], random_colors):
        median.set_color('black')

    # 设置异常值（fliers）的颜色为对应箱体的颜色
    for flier, color in zip(bp['fliers'], random_colors):
        flier.set(markerfacecolor=color, markeredgecolor=color)

    # 设置 x 轴标签
    ax.set_xticklabels(columns)
    plt.xlabel('Column names')
    plt.ylabel('Values')
    plt.xticks(rotation=90)
    plt.savefig('Figure_1.tif', dpi=300, format='tiff', bbox_inches='tight')
    plt.show()


# 定义绘制不同 DM 值的箱线图的函数
def plot_boxplots_by_dm(df, columns):
    if 'DM' in df.columns:
        df_dm_0 = df[df['DM'] == 0]
        df_dm_1 = df[df['DM'] == 1]

        if valid_columns := [col for col in columns if col in df_dm_0.columns]:
            # plot_boxplot(df_dm_0, valid_columns)
            pass

        if valid_columns := [col for col in columns if col in df_dm_1.columns]:
            # plot_boxplot(df_dm_1, valid_columns)
            pass


# 绘制第一组箱线图
if valid_group1_cols:
    # plot_boxplots_by_dm(df, valid_group1_cols)
    pass

# 绘制第二组箱线图
if valid_group2_cols:
    # plot_boxplots_by_dm(df, valid_group2_cols)
    pass

# 绘制第三组箱线图
if valid_group3_cols:
    # plot_boxplots_by_dm(df, valid_group3_cols)
    pass

# 绘制第四组箱线图
if valid_group4_cols:
    # plot_boxplots_by_dm(df, valid_group4_cols)
    pass

# 绘制第五组箱线图
if valid_group5_cols:
    # plot_boxplots_by_dm(df, valid_group5_cols)
    pass

# 绘制所有患者年龄直方图
if 'age' in df.columns:
    plt.figure(figsize=(10, 6))
    # 修改颜色为 rgb_to_tuple('46,95,161')
    color_all_patients = rgb_to_tuple('46,95,161')
    plt.hist(df['age'], bins=10, edgecolor='white', color=color_all_patients)
    plt.xlabel('Age')
    plt.ylabel('Number of patients')
    plt.savefig('Figure_2.tif', dpi=300, format='tiff', bbox_inches='tight')
    plt.show()

# 绘制 DM 为 1 的患者年龄直方图
if 'age' in df.columns and 'DM' in df.columns:
    df_dm_1 = df[df['DM'] == 1]
    plt.figure(figsize=(10, 6))
    # 修改颜色为 rgb_to_tuple('238,140,125')
    color_dm_1_patients = rgb_to_tuple('238,140,125')
    # 将边框颜色设置为无色
    plt.hist(df_dm_1['age'], bins=10, edgecolor='white', color=color_dm_1_patients)
    plt.xlabel('Age')
    plt.ylabel('Number of patients')
    plt.savefig('Figure_3.tif', dpi=300, format='tiff', bbox_inches='tight')
    plt.show()

# 绘制 BMI vs FPG 和 BMI vs PG2h 的散点图
if 'BMI' in df.columns and 'FPG' in df.columns and 'PG2h' in df.columns:
    # BMI vs FPG 散点图
    plt.figure(figsize=(10, 6))
    plt.scatter(df['BMI'], df['FPG'], alpha=0.5)
    plt.xlabel('BMI')
    plt.ylabel('FPG')
    plt.savefig('Figure_4.tif', dpi=300, format='tiff', bbox_inches='tight')
    plt.show()

    # BMI vs PG2h 散点图
    plt.figure(figsize=(10, 6))
    plt.scatter(df['BMI'], df['PG2h'], alpha=0.5)
    plt.xlabel('BMI')
    plt.ylabel('PG2h')
    plt.savefig('Figure_5.tif', dpi=300, format='tiff', bbox_inches='tight')
    plt.show()

# 绘制纵向血糖变化趋势图（仅 DM = 1）
if 'DM' in df.columns and ('FPG' in df.columns or 'PG2h' in df.columns or 'FPGover7' in df.columns):
    df_dm_1 = df[df['DM'] == 1]
    if not df_dm_1.empty:
        plt.figure(figsize=(10, 6))
        index = range(len(df_dm_1))
        if 'FPG' in df_dm_1.columns:
            plt.plot(index, df_dm_1['FPG'], label='Fasting Plasma Glucose (FPG)', marker='o')
        if 'PG2h' in df_dm_1.columns:
            plt.plot(index, df_dm_1['PG2h'], label='Postprandial Glucose 2 hours (PG2h)', marker='s')
        plt.xlabel('Sample order')
        plt.ylabel('Blood glucose value')
        plt.legend()
        plt.savefig('Figure_6.tif', dpi=300, format='tiff', bbox_inches='tight')
        plt.show()

# 绘制堆叠柱状图（不同肾功能分期的血糖指标对比）
if 'GFR5lev' in df.columns and ('FPG' in df.columns or 'PG2h' in df.columns):
    # 确保 FPG、PG2h 和 GFR5lev 列是数值类型
    df['FPG'] = pd.to_numeric(df['FPG'], errors='coerce')
    df['PG2h'] = pd.to_numeric(df['PG2h'], errors='coerce')
    df['GFR5lev'] = pd.to_numeric(df['GFR5lev'], errors='coerce')  # 新增：转换GFR5lev为数值

    # 删除 FPG、PG2h 或 GFR5lev 列中包含 NaN 的行
    df = df.dropna(subset=['FPG', 'PG2h', 'GFR5lev'])  # 修改：增加对GFR5lev的检查

    # 可选：筛选GFR5lev的有效范围（如1-5期）
    df = df[df['GFR5lev'].between(0, 4)]  # 新增：限制分期范围

    # 计算不同肾功能分期的血糖指标均值
    fpg_by_stage = df.groupby('GFR5lev')['FPG'].mean()
    pg2h_by_stage = df.groupby('GFR5lev')['PG2h'].mean()

    # 绘制堆叠柱状图
    plt.figure(figsize=(10, 6))
    width = 0.35
    ind = np.arange(len(fpg_by_stage))
    plt.bar(ind, fpg_by_stage.values, width, label='FPG')
    plt.bar(ind, pg2h_by_stage.values, width, bottom=fpg_by_stage.values, label='PG2h')

    # 修改：显式设置横坐标标签为分期1-5
    plt.xticks(ind, [f'{i}' for i in range(0, len(fpg_by_stage))])

    plt.xlabel('Renal Function Stages (GFR5lev)')
    plt.ylabel('Average Blood Glucose Value')
    plt.legend()
    plt.savefig('Figure_7.tif', dpi=300, format='tiff', bbox_inches='tight')
    plt.show()

# 绘制相关性矩阵图
if 'FINS' in df.columns and 'HomaIR' in df.columns:
    # 选择需要分析相关性的列
    selected_cols = ['FINS', 'HomaIR']
    df_selected = df[selected_cols]

    # 将空字符串替换为 NaN
    df_selected = df_selected.replace(' ', np.nan)

    # 删除包含 NaN 的行
    df_selected = df_selected.dropna()

    # 计算相关性矩阵
    corr_matrix = df_selected.corr()

    # 绘制相关性矩阵图
    plt.figure(figsize=(8, 6))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
    plt.savefig('Figure_8.tif', dpi=300, format='tiff', bbox_inches='tight')
    plt.show()