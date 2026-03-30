# ==============================================================================
# 小肠 RSF 模型 SHAP 依赖散点图 (智能 Cutoff 计算 + 临床数值还原版)
# ==============================================================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap
import pickle
import warnings
import os
from datetime import datetime
from scipy.interpolate import UnivariateSpline

warnings.filterwarnings('ignore')

# 字体设置 (SCI 标配)
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
plt.rcParams['axes.unicode_minus'] = False

print("=" * 80)
print("🏥 小肠 RSF 模型 SHAP 依赖散点图 (Dependence Scatter Plot)")
print("=" * 80)

# ===============================================================================
# 第1部分：加载模型和数据
# ===============================================================================
print("\n[1/4] 📊 加载模型和数据...")

model_file = 'Intestine.pkl'
if not os.path.exists(model_file):
    raise FileNotFoundError(f"❌ 找不到文件: {model_file}")

with open(model_file, 'rb') as f:
    model_pkg = pickle.load(f)

model = model_pkg['model']
X_train = model_pkg['X_train']
feature_cols = model_pkg['feature_cols']

# 加载测试数据
test_df = pd.read_excel('Intestine_final.xlsx', sheet_name='Test')
X_test = test_df[feature_cols]

print(f"✓ 训练集: {X_train.shape[0]} 样本 (用于基线)")
print(f"✓ 测试集: {len(X_test)} 样本 (用于绘图)")

# ===============================================================================
# 第2部分：特征标签和输出目录
# ===============================================================================
print("\n[2/4] 🏷️  准备特征标签 (精准对齐底层)...")

feature_labels = {
    'Age': 'Age (Years)',
    'Tumor.size': 'Tumor Size (cm)',
    'Mitotic.rate': 'Mitotic Rate (/50 HPF)',
    'Race.0': 'Race: White',
    'Race.1': 'Race: Black',
    'Race.2': 'Race: Asian/Pacific Islander',
    'Marital.status.0': 'Marital Status: Married',
    'Marital.status.1': 'Marital Status: Single/Unmarried',
    'Marital.status.2': 'Marital Status: Separated/Divorced/Widowed',
    'Gender': 'Gender (Male)',
    'Systemic.treatment': 'Systemic Treatment',
    'Liver.metastasis': 'Liver Metastasis'
}

shap_feature_names = [feature_labels.get(f, f) for f in feature_cols]

timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
output_dir = f"SHAP_小肠_scatter_{timestamp}"
os.makedirs(output_dir, exist_ok=True)

# 我们要重点绘制的连续变量 (对应原始列名)
target_raw_features = ['Age', 'Tumor.size', 'Mitotic.rate']

# ===============================================================================
# 第3部分：计算 SHAP 值 (降维提速)
# ===============================================================================
print("\n[3/4] 🧮 计算 SHAP 值 (应用降维提速技术)...")

def predict_model(X):
    return model.predict(X)

try:
    print("  尝试 TreeExplainer (快速模式)...")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test.values)
    print("✓ TreeExplainer 计算完成")
except Exception as e:
    print(f"  ⚠️ TreeExplainer 不支持，切换 KernelExplainer...")
    print("  - 正在使用 K-Means 浓缩背景数据 (提速几十倍)...")
    background_sample = shap.kmeans(X_train, 50)
    explainer = shap.KernelExplainer(predict_model, background_sample)
    
    print(f"  - 正在推演全部 {len(X_test)} 个测试样本的 SHAP 贡献度...")
    shap_values = explainer.shap_values(X_test.values, nsamples=150)
    print("✓ KernelExplainer 计算完成")

sv_for_plot = shap_values[0] if isinstance(shap_values, list) else shap_values
base_value = explainer.expected_value
if isinstance(base_value, (np.ndarray, list)):
    base_value = base_value[0]

# ===============================================================================
# 第4部分：数值还原与绘图 (带智能 Cutoff 标注)
# ===============================================================================
print("\n[4/4] ⚕️  临床数值还原与散点图绘制...")

X_test_display = X_test.copy()

# 🌟 核心：将核分裂象除以 10，恢复到病理真实数值
if 'Mitotic.rate' in feature_cols:
    X_test_display['Mitotic.rate'] = X_test_display['Mitotic.rate'] / 10.0
    print("  ✓ Mitotic.rate 已还原为真实数值 ( ÷ 10 )")

# 构建 Explanation 对象
shap_obj = shap.Explanation(
    values=sv_for_plot,
    base_values=base_value,
    data=X_test_display.values,
    feature_names=shap_feature_names
)

cutoff_results = {}

for raw_feat in target_raw_features:
    if raw_feat not in feature_cols:
        continue
    
    feat_idx = feature_cols.index(raw_feat)
    pretty_feat = shap_feature_names[feat_idx]
    print(f"\n  - 正在分析 [{pretty_feat}] ...")
    
    # 获取画图用的 X 值和 Y 值 (SHAP)
    x_vals = X_test_display[raw_feat].values
    y_vals = sv_for_plot[:, feat_idx]
    
    # 🌟 计算 SHAP=0 时的截断值 (Cutoff)
    cutoff_val = None
    try:
        unique_x, unique_indices = np.unique(x_vals, return_index=True)
        unique_y = y_vals[unique_indices]
        
        if len(unique_x) > 3:
            spline = UnivariateSpline(unique_x, unique_y, s=len(unique_x)*0.05)
            roots = spline.roots()
            if len(roots) > 0:
                # 寻找最靠近中位数的根作为主要 cutoff
                median_x = np.median(unique_x)
                cutoff_val = roots[np.argmin(np.abs(roots - median_x))]
    except Exception as e:
        pass
    
    # 如果样条插值没找到，用最接近 0 的点兜底
    if cutoff_val is None:
        closest_idx = np.argmin(np.abs(y_vals))
        cutoff_val = x_vals[closest_idx]
    
    cutoff_results[pretty_feat] = cutoff_val

    # --- 开始绘图 ---
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # hist=True开启底部直方图，color设置按自身值染色
    shap.plots.scatter(
        shap_obj[:, feat_idx], 
        hist=True,
        color=shap_obj[:, feat_idx], 
        ax=ax, 
        show=False
    )
    
    # 添加 SHAP=0 的水平基准线
    ax.axhline(0, color='gray', linestyle='--', linewidth=1.5, alpha=0.8, zorder=1)
    
    # 🌟 添加垂直于 Cutoff 点的标线
    ax.axvline(cutoff_val, color='#E74C3C', linestyle='-.', linewidth=2, alpha=0.9, zorder=2)
    
    # 动态调整 Cutoff 文字位置，防止遮挡
    y_min, y_max = ax.get_ylim()
    text_y_pos = y_max * 0.75 if y_max > 0 else y_min * 0.2
    
    # 在图上写出精准的截断值 (带白底)
    ax.text(
        cutoff_val * 1.02, 
        text_y_pos, 
        f"Cutoff ≈ {cutoff_val:.1f}", 
        color='#E74C3C', 
        fontsize=14, 
        fontweight='bold',
        bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', boxstyle='round,pad=0.3', zorder=3)
    )
    
    # 美化标题
    ax.set_title(f"SHAP Dependence: {pretty_feat}\n(SHAP > 0 indicates increased mortality risk)", 
                 fontsize=15, fontweight='bold', pad=20)
    
    # 文件名安全处理
    safe_name = pretty_feat.replace(' ', '_').replace('(', '').replace(')', '').replace('/', '').replace('__', '_')
    save_name = f"{output_dir}/Scatter_{safe_name}.pdf"
    
    plt.savefig(save_name, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"    ✓ 发现 Cutoff={cutoff_val:.1f}，已保存: {save_name}")

print(f"\n✅ 散点图及截断值已保存到: {output_dir}/")

# ===============================================================================
# 第5部分：生成解读报告
# ===============================================================================
report = f"""
{'='*80}
小肠 RSF 模型 SHAP 依赖散点图分析报告
{'='*80}

分析时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
输出目录: {output_dir}

【重点提醒】
Mitotic Rate 已在展示前自动除以 10，恢复为 SEER 真实的临床核分裂象数值。

【智能 Cutoff 计算结果】
基于单变量样条插值 (Univariate Spline) 计算横穿 SHAP=0 基准线时的临界特征值：
"""

for feat, val in cutoff_results.items():
    report += f"  - {feat:30s}: Cutoff ≈ {val:.2f}\n"

report += f"""
【散点图解读指南】
1. 灰色柱状图 (底部)
   - 表示该数值区间内的患者分布密度 (越高代表人群越集中)。
2. 虚线 (SHAP=0)
   - 散点在虚线【上方】: 该特征导致死亡风险显著上升 (高危区间)。
   - 散点在虚线【下方】: 该特征处于相对安全的低风险区间。
3. 红色垂直标线 (Cutoff)
   - 代表生存风险发生“质变”的临床截断点，对于制定随访和干预策略具有重大指导意义。

{'='*80}
"""

with open(f'{output_dir}/分析报告.txt', 'w', encoding='utf-8') as f:
    f.write(report)

print("=" * 80)
print("🚀 小肠 SHAP 依赖散点图 (Scatter Plot) 分析完美收官！")
print("=" * 80)