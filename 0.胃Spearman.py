import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import os
from scipy import stats
import warnings

# 忽略警告
warnings.filterwarnings("ignore")

# ==============================================================================
# 🛠️ 配置区域
# ==============================================================================
# 请确保你的文件名正确，且已安装 openpyxl: pip install openpyxl
FILE_PATH = "Stomach_final.xlsx"  # 替换为你的文件名
SAVE_PREFIX = "Stomach_Spearman"
NOW_TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")

# 设置全局字体 (Times New Roman 是 SCI 论文标配)
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
plt.rcParams['axes.unicode_minus'] = False

# ==============================================================================
# 1. 数据读取与预处理
# ==============================================================================
# 读取训练集数据
df_raw = pd.read_excel(FILE_PATH, sheet_name='Train')

# 复制一份用于计算，确保包含所有列（独热编码列 + 结局列）
df = df_raw.copy()

print(f"✅ 成功读取数据: {df.shape[1]} 个变量, {df.shape[0]} 个样本。")

# 计算斯皮尔曼相关系数矩阵
corr = df.corr(method='spearman')

# 计算 P 值矩阵的函数
def calculate_pvalues(df):
    cols = df.columns
    n = len(cols)
    pvalues = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i == j:
                pvalues[i, j] = 0
            else:
                _, p = stats.spearmanr(df.iloc[:, i], df.iloc[:, j])
                pvalues[i, j] = p
    return pd.DataFrame(pvalues, index=cols, columns=cols)

pvalue_matrix = calculate_pvalues(df)

# 定义显著性星号函数
def get_stars(p):
    if p < 0.001: return '***'
    elif p < 0.01: return '**'
    elif p < 0.05: return '*'
    else: return ''

# ==============================================================================
# 2. 绘图阶段 (对角线留白版)
# ==============================================================================
fig, ax = plt.subplots(figsize=(18, 16), dpi=300)

cmap = plt.cm.coolwarm
norm = plt.Normalize(vmin=-1, vmax=1)
n_vars = len(corr.columns)

for i in range(n_vars):
    for j in range(n_vars):
        corr_val = corr.iloc[i, j]
        p_val = pvalue_matrix.iloc[i, j]
        color = cmap(norm(corr_val))
        
        # --- 下三角 (i > j): 绘制气泡图 ---
        if i > j:
            size = np.abs(corr_val) * 2000  # 气泡大小与相关系数绝对值成正比
            ax.scatter(j, i, s=size, color=color, alpha=0.9, 
                       edgecolors='white', linewidths=0.5)
            
        # --- 上三角 (i < j): 显示数值和星号 ---
        elif i < j:
            stars = get_stars(p_val)
            label = f"{corr_val:.2f}\n{stars}" if stars else f"{corr_val:.2f}"
            ax.text(j, i, label, ha='center', va='center', 
                    color=color, fontsize=11, fontweight='bold')
            
        # --- 对角线 (i == j): 彻底留白 ---
        else:
            # 逻辑跳过，不绘制任何内容
            pass

# ==============================================================================
# 3. 图像修饰与格式化
# ==============================================================================
# --- 🚀 核心修正：解决 Age 压线问题 ---
# 1. 手动设置轴限制，给四周留出 0.75 的空间，确保第一个和最后一个变量都在"框内"
ax.set_xlim(-0.75, n_vars - 0.25)
ax.set_ylim(n_vars - 0.25, -0.75) # 注意这里直接在 ylim 里实现 invert 逻辑

# 2. 保留坐标轴线（左和下），删掉右和上的黑框
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['left'].set_linewidth(1.5)
ax.spines['bottom'].set_linewidth(1.5)

# 3. 设置刻度和标签
ax.set_xticks(range(n_vars))
ax.set_xticklabels(corr.columns, rotation=45, ha='right', fontsize=11, fontweight='bold')
ax.set_yticks(range(n_vars))
ax.set_yticklabels(corr.columns, fontsize=11, fontweight='bold')

# 隐藏刻度线短轴
ax.tick_params(axis='both', which='both', length=0)

# 添加网格线，帮助对齐
ax.set_axisbelow(True)
ax.grid(True, linestyle='--', alpha=0.25, color='gray')

# --- 侧边颜色条 ---
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
cbar = fig.colorbar(sm, ax=ax, shrink=0.7, aspect=30, pad=0.04)
cbar.outline.set_visible(False) # 颜色条也不要边框
cbar.set_label('Spearman Correlation Coefficient (Rho)', fontsize=13, labelpad=15)

# ==============================================================================
png_file = f"{SAVE_PREFIX}_Perfect_Bubble_{NOW_TIMESTAMP}.png"
pdf_file = f"{SAVE_PREFIX}_Perfect_Bubble_{NOW_TIMESTAMP}.pdf"
plt.savefig(png_file, dpi=300, facecolor='white')
plt.savefig(pdf_file, format='pdf')
plt.close()

print(f"\n✅ 任务完成！已修复 Age 压线问题。")
print(f"📸 PNG 请查看: {png_file}")
print(f"📄 PDF 请查看: {pdf_file}")