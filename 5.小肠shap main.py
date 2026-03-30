# ==============================================================================
# 🏥 小肠 RSF 模型 SHAP 主要图表分析 (精简纯净版：全局图 + 单样本图)
# ==============================================================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap
import pickle
import warnings
import os
from datetime import datetime

warnings.filterwarnings('ignore')

# 字体设置 (SCI 标配)
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
plt.rcParams['axes.unicode_minus'] = False

print("=" * 80)
print("🏥 小肠 RSF 模型 SHAP 主要图表 (Main Plots)")
print("=" * 80)

# ===============================================================================
# 第1部分：加载模型和数据
# ===============================================================================
print("\n[1/5] 📊 加载模型和数据...")

model_file = 'Intestine.pkl'
if not os.path.exists(model_file):
    raise FileNotFoundError(f"❌ 找不到文件: {model_file}")

with open(model_file, 'rb') as f:
    model_pkg = pickle.load(f)

model = model_pkg['model']
X_train = model_pkg['X_train']
feature_cols = model_pkg['feature_cols']
auc_results = model_pkg.get('auc_results', {})

# 加载测试数据 (仅展示测试集的 SHAP)
test_df = pd.read_excel('Intestine_final.xlsx', sheet_name='Test')
X_test = test_df[feature_cols].values

print(f"✓ 训练集: {X_train.shape[0]} 样本, {X_train.shape[1]} 特征 (用于构建基线)")
print(f"✓ 测试集: {X_test.shape[0]} 样本, {X_test.shape[1]} 特征 (用于 SHAP 解释)")

# ===============================================================================
# 第2部分：特征标签和输出目录
# ===============================================================================
print("\n[2/5] 🏷️  准备特征标签 (精准对齐底层 R 代码)...")

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
output_dir = f"SHAP_小肠_main_{timestamp}"
os.makedirs(output_dir, exist_ok=True)
print(f"✓ 特征马甲已准备，结果将保存至: {output_dir}/")

# ===============================================================================
# 第3部分：计算 SHAP 值 (应用降维提速技术)
# ===============================================================================
print("\n[3/5] 🧮 计算 SHAP 值...")

def predict_model(X):
    return model.predict(X)

try:
    print("  尝试 TreeExplainer (快速模式)...")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)
    print("✓ TreeExplainer 计算完成")
except Exception as e:
    print(f"  ⚠️ TreeExplainer 不支持，无缝切换 KernelExplainer...")
    # 🌟 用 kmeans 浓缩背景数据，提速几十倍
    print("  - 正在使用 K-Means 浓缩背景数据...")
    background_sample = shap.kmeans(X_train, 50)
    explainer = shap.KernelExplainer(predict_model, background_sample)
    
    # 🌟 nsamples=150 限制穷举次数，保证速度与精度
    print(f"  - 正在计算全队列 SHAP 值 (请稍候)...")
    shap_values = explainer.shap_values(X_test, nsamples=150)
    print("✓ KernelExplainer 计算完成")

# 提取SHAP值数组
sv_for_plot = shap_values[0] if isinstance(shap_values, list) else shap_values

# 获取基准值 (Base Value)
base_value = explainer.expected_value
if isinstance(base_value, (np.ndarray, list)):
    base_value = base_value[0]

# ===============================================================================
# 第4部分：临床数值还原与构建 Explanation 对象
# ===============================================================================
print("\n[4/5] ⚕️  还原临床真实数值...")

X_test_display = X_test.copy()

# 🌟 核心：将核分裂象除以 10，恢复病理报告真实数值
if 'Mitotic.rate' in feature_cols:
    mitotic_idx = feature_cols.index('Mitotic.rate')
    X_test_display[:, mitotic_idx] = X_test_display[:, mitotic_idx] / 10.0
    print("✓ Mitotic.rate 已还原为真实数值 ( ÷ 10 )")

# 构建大一统的 Explanation 对象
shap_obj = shap.Explanation(
    values=sv_for_plot,
    base_values=base_value,
    data=X_test_display,  
    feature_names=shap_feature_names
)

# ===============================================================================
# 第5部分：生成核心图表 (去掉 Decision 和 Scatter)
# ===============================================================================
print("\n[5/5] 📈 生成核心图表全家桶...")

# 1. Summary Plot (Beeswarm)
print("  ✓ 生成 Beeswarm Plot...")
plt.figure(figsize=(10, 8))
shap.summary_plot(sv_for_plot, X_test_display, feature_names=shap_feature_names, show=False)
plt.title("SHAP Summary Plot (Global Risk Impact)", fontsize=16, pad=20)
plt.tight_layout()
plt.savefig(f'{output_dir}/01_Summary_Beeswarm.pdf', dpi=300, bbox_inches='tight')
plt.close()

# 2. Feature Importance (Bar)
print("  ✓ 生成 Feature Importance (Bar)...")
plt.figure(figsize=(10, 8))
shap.summary_plot(sv_for_plot, X_test_display, feature_names=shap_feature_names, plot_type="bar", show=False, color='#4A90E2')
plt.title("Feature Importance (Mean |SHAP|)", fontsize=16, pad=20)
plt.tight_layout()
plt.savefig(f'{output_dir}/02_Feature_Importance_Bar.pdf', dpi=300, bbox_inches='tight')
plt.close()

# 3. Heatmap
print("  ✓ 生成 Heatmap...")
plt.figure(figsize=(12, 6))
shap.plots.heatmap(shap_obj[:min(1000, len(X_test))], max_display=12, show=False)
plt.savefig(f'{output_dir}/03_Heatmap.pdf', dpi=300, bbox_inches='tight')
plt.close()

# --- 单样本分析 (高低风险人群) ---
print("  ✓ 生成 典型病例精细分解 (Waterfall & Force Plot)...")
risk_scores = model.predict(X_test)
high_risk_idx = np.argmax(risk_scores)
low_risk_idx = np.argmin(risk_scores)

# 4. Waterfall Plot
def plot_waterfall(idx, title_prefix, filename):
    plt.figure(figsize=(10, 6))
    shap.plots.waterfall(shap_obj[idx], show=False, max_display=12)
    ax = plt.gca()
    xmin, xmax = ax.get_xlim()
    range_width = xmax - xmin
    ax.set_xlim(xmin - range_width * 0.15, xmax + range_width * 0.15)
    plt.title(f"{title_prefix} (Risk Score: {risk_scores[idx]:.2f})", fontsize=14, pad=20, fontweight='bold')
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()

plot_waterfall(high_risk_idx, "Waterfall: High Risk Patient", f'{output_dir}/04a_Waterfall_HighRisk.pdf')
plot_waterfall(low_risk_idx, "Waterfall: Low Risk Patient", f'{output_dir}/04b_Waterfall_LowRisk.pdf')

# 5. Force Plot (HTML 交互格式)
try:
    force_high = shap.force_plot(base_value, sv_for_plot[high_risk_idx], X_test_display[high_risk_idx],
                                 feature_names=shap_feature_names, matplotlib=False)
    shap.save_html(f'{output_dir}/05a_Force_HighRisk.html', force_high)
    
    force_low = shap.force_plot(base_value, sv_for_plot[low_risk_idx], X_test_display[low_risk_idx],
                                feature_names=shap_feature_names, matplotlib=False)
    shap.save_html(f'{output_dir}/05b_Force_LowRisk.html', force_low)
    print("  ✓ Force Plot (HTML) 生成成功")
except Exception as e:
    print(f"  ⚠️ Force Plot 生成失败: {e}")

# ===============================================================================
# 保存分析报告
# ===============================================================================
mean_abs_shap_values = np.mean(np.abs(sv_for_plot), axis=0)
feature_importance = pd.DataFrame({
    'Feature': shap_feature_names,
    'Mean |SHAP|': mean_abs_shap_values
}).sort_values('Mean |SHAP|', ascending=False)

report = f"""
{'='*80}
小肠 RSF 模型 SHAP 分析核心报告 (纯净版)
{'='*80}

分析时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
输出目录: {output_dir}

【核心提醒】
* 核分裂象 (Mitotic Rate) 在生成展示图表前，已自动除以 10 还原为临床真实数值。
* 图表中的 X 轴 (SHAP Value) 代表对“死亡风险评分 (Risk Score)”的贡献。SHAP > 0 代表推高死亡风险 (预后差)。

【生成的图表】
1. 01_Summary_Beeswarm.pdf        - 核心全局解释 (蜜蜂群图)
2. 02_Feature_Importance_Bar.pdf  - 特征重要性条形图
3. 03_Heatmap.pdf                 - 全样本热力图
4. 04a/b_Waterfall_*.pdf          - 典型极值患者瀑布图
5. 05a/b_Force_*.html             - 典型极值患者交互式力图

【特征重要性排序 (基于测试集平均绝对SHAP值)】
"""
for idx, row in feature_importance.iterrows():
    report += f"  {row['Feature']:40s}: {row['Mean |SHAP|']:10.6f}\n"

with open(f'{output_dir}/分析报告.txt', 'w', encoding='utf-8') as f:
    f.write(report)
feature_importance.to_csv(f'{output_dir}/特征重要性.csv', index=False, encoding='utf-8')

print(f"\n✅ 所有图表和报告已成功保存至: {output_dir}/")
print("=" * 80)