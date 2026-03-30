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
FILE_PATH = "Intestine_final.xlsx"  # 替换为你的文件名
SAVE_PREFIX = "Intestine_Spearman"
NOW_TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")

# 设置全局字体
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
plt.rcParams['axes.unicode_minus'] = False

# ==============================================================================
# 1. 数据读取与计算
# ==============================================================================
df_raw = pd.read_excel(FILE_PATH, sheet_name='Train')
df = df_raw.copy()

corr = df.corr(method='spearman')

def calculate_pvalues(df):
    cols = df.columns
    n = len(cols)
    pvalues = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i == j: pvalues[i, j] = 0
            else:
                _, p = stats.spearmanr(df.iloc[:, i], df.iloc[:, j])
                pvalues[i, j] = p
    return pd.DataFrame(pvalues, index=cols, columns=cols)

pvalue_matrix = calculate_pvalues(df)

def get_stars(p):
    if p < 0.001: return '***'
    elif p < 0.01: return '**'
    elif p < 0.05: return '*'
    else: return ''

# ==============================================================================
# 2. 绘图阶段 (解决“压线”问题的终极版)
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
        
        if i > j:  # 下三角：气泡
            size = np.abs(corr_val) * 2000 
            ax.scatter(j, i, s=size, color=color, alpha=0.9, edgecolors='white', linewidths=0.5)
        elif i < j:  # 上三角：数值
            stars = get_stars(p_val)
            label = f"{corr_val:.2f}\n{stars}" if stars else f"{corr_val:.2f}"
            ax.text(j, i, label, ha='center', va='center', color=color, fontsize=10, fontweight='bold')
        else:
            pass # 对角线留白

# --- 🚀 核心修正：解决 Age 压线问题 ---
# 1. 手动设置轴限制，给四周留出 0.75 的空间，确保第一个和最后一个变量都在“框内”
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

# --- 标题 ---
plt.title(f'Spearman Correlation Analysis\n({FILE_PATH.split("_")[0]} Cohort)', 
          fontsize=20, fontweight='bold', pad=40)

# 布局优化
plt.tight_layout()
# 手工调整边距确保不被剪裁
plt.subplots_adjust(top=0.88, bottom=0.18, left=0.15, right=0.91)

# ==============================================================================
# 3. 保存
# ==============================================================================
png_file = f"{SAVE_PREFIX}_Perfect_Bubble_{NOW_TIMESTAMP}.png"
pdf_file = f"{SAVE_PREFIX}_Perfect_Bubble_{NOW_TIMESTAMP}.pdf"
plt.savefig(png_file, dpi=300, facecolor='white')
plt.savefig(pdf_file, format='pdf')
plt.close()

print(f"\n✅ 任务完成！已修复 Age 压线问题。")
print(f"📸 PNG 请查看: {png_file}")
print(f"📄 PDF 请查看: {pdf_file}")