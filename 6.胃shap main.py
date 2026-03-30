# ==============================================================================
# 胃 RSF 模型 SHAP 主要图表分析 (参考 SHAP main.py)
# ==============================================================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap
import pickle
import warnings
from datetime import datetime

warnings.filterwarnings('ignore')

# 字体设置
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
plt.rcParams['axes.unicode_minus'] = False

print("=" * 80)
print("🏥 胃 RSF 模型 SHAP 主要图表 (Main Plots)")
print("=" * 80)

# ===============================================================================
# 第1部分：加载模型和数据
# ===============================================================================
print("\n[1/4] 📊 加载模型和数据...")

model_file = 'Stomach.pkl'
if not __import__('os').path.exists(model_file):
    raise FileNotFoundError(f"❌ 找不到文件: {model_file}")

with open(model_file, 'rb') as f:
    model_pkg = pickle.load(f)

model = model_pkg['model']
X_train = model_pkg['X_train']
feature_cols = model_pkg['feature_cols']
auc_results = model_pkg['auc_results']

# 加载测试数据
test_df = pd.read_excel('Stomach_final.xlsx', sheet_name='Test')

X_test = test_df[feature_cols].values
print(f"✓ 训练集: {X_train.shape[0]} 样本, {X_train.shape[1]} 特征")
print(f"✓ 测试集: {X_test.shape[0]} 样本, {X_test.shape[1]} 特征")

# ===============================================================================
# 第2部分：特征标签和输出目录
# ===============================================================================
print("\n[2/4] 🏷️  准备特征标签...")

feature_labels = {
    'Age': 'Age (Years)',
    'Tumor.size': 'Tumor Size (cm)',
    'Mitotic.rate': 'Mitotic Rate (/50 HPF)',
    'Race.0': 'Race: White',
    'Race.1': 'Race: Black',
    'Race.2': 'Race: Asian/Pacific Islander',
    'Marital.status.0': 'Marital: Married',
    'Marital.status.1': 'Marital: Single/Unmarried',
    'Marital.status.2': 'Marital: Separated/Divorced/Widowed',
    'Gender': 'Gender (Male=1)',
    'Liver.metastasis': 'Liver Metastasis'
}

shap_feature_names = [feature_labels.get(f, f) for f in feature_cols]
print(f"✓ {len(shap_feature_names)} 个特征已准备")

timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
output_dir = f"SHAP_胃_main_{timestamp}"
__import__('os').makedirs(output_dir, exist_ok=True)

# ===============================================================================
# 第3部分：计算 SHAP 值
# ===============================================================================
print("\n[3/4] 🧮 计算 SHAP 值...")

def predict_model(X):
    return model.predict(X)

# 先尝试速度更快的 TreeExplainer (像 SHAP main.py 一样)
try:
    print("  尝试 TreeExplainer (快速模式)...")
    explainer = shap.TreeExplainer(model)
    print("✓ TreeExplainer 初始化成功，计算中...")
    shap_values = explainer.shap_values(X_test)
    print("✓ SHAP 值已计算 (快速完成)")
except Exception as e:
    print(f"  ⚠️ TreeExplainer 不支持此模型，改用 KernelExplainer...")
    print(f"     原因: {type(e).__name__}")
    print(f"  使用 KernelExplainer (这可能需要几分钟)...")
    # 🌟 提速秘籍 1：使用 K-Means 浓缩背景数据 (提速 50 倍)
    print("  - 正在使用 K-Means 浓缩背景数据...")
    background_sample = shap.kmeans(X_train, 50)
    explainer = shap.KernelExplainer(predict_model, background_sample)
    print(f"  - 计算全部 {len(X_test)} 个测试样本的 SHAP 值...")
    # 🌟 提速秘籍 2：限制采样次数 nsamples=150
    shap_values = explainer.shap_values(X_test, nsamples=150)
    print("✓ SHAP 值已计算")

# 提取SHAP值
if isinstance(shap_values, list):
    sv_for_plot = shap_values[0]
else:
    sv_for_plot = shap_values

base_value = explainer.expected_value if not isinstance(explainer.expected_value, list) else explainer.expected_value[0]

# ===============================================================================
# 第4部分：生成图表
# ===============================================================================
print("\n[4/4] 📈 生成图表...")

# 1. Summary Plot (Beeswarm)
print("  ✓ 生成 Beeswarm Plot...")
plt.figure(figsize=(12, 8))
try:
    shap.summary_plot(sv_for_plot, X_test, feature_names=shap_feature_names, show=False)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/01_Summary_Beeswarm.pdf', dpi=300, bbox_inches='tight')
    plt.close()
    print("    ✓ 成功")
except Exception as e:
    print(f"    ⚠️ 失败: {e}")
    plt.close()

# 2. Summary Plot (Bar)
print("  ✓ 生成 Feature Importance (Bar)...")
plt.figure(figsize=(12, 8))
try:
    shap.summary_plot(sv_for_plot, X_test, feature_names=shap_feature_names, plot_type="bar", show=False)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/02_Feature_Importance_Bar.pdf', dpi=300, bbox_inches='tight')
    plt.close()
    print("    ✓ 成功")
except Exception as e:
    print(f"    ⚠️ 失败: {e}")
    plt.close()

# 3. Heatmap
print("  ✓ 生成 Heatmap...")
try:
    shap_obj = shap.Explanation(
        values=sv_for_plot,
        base_values=base_value,
        data=X_test,
        feature_names=shap_feature_names
    )
    plt.figure(figsize=(14, 8))
    shap.plots.heatmap(shap_obj, max_display=len(shap_feature_names), show=False)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/03_Heatmap.pdf', dpi=300, bbox_inches='tight')
    plt.close()
    print("    ✓ 成功")
except Exception as e:
    print(f"    ⚠️ 失败: {e}")
    plt.close()

# 4. Waterfall & Force - High/Low Risk Individuals
print("  ✓ 生成 Waterfall & Force (高低风险个体)...")

risk_scores = model.predict(X_test)
high_risk_idx = np.argmax(risk_scores)
low_risk_idx = np.argmin(risk_scores)

# Waterfall - High Risk
try:
    fig = plt.figure(figsize=(12, 8))
    shap.waterfall_plot(shap.Explanation(
        values=sv_for_plot[high_risk_idx],
        base_values=base_value,
        data=X_test[high_risk_idx],
        feature_names=shap_feature_names
    ), show=False, max_display=15)
    plt.title(f"Waterfall: High Risk Patient (Risk Score: {risk_scores[high_risk_idx]:.4f})", fontsize=12)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/05a_Waterfall_HighRisk.pdf', dpi=300, bbox_inches='tight')
    plt.close()
    print("    ✓ Waterfall High Risk 成功")
except Exception as e:
    print(f"    ⚠️ Waterfall High Risk 失败: {e}")
    plt.close()

# Waterfall - Low Risk
try:
    fig = plt.figure(figsize=(12, 8))
    shap.waterfall_plot(shap.Explanation(
        values=sv_for_plot[low_risk_idx],
        base_values=base_value,
        data=X_test[low_risk_idx],
        feature_names=shap_feature_names
    ), show=False, max_display=15)
    plt.title(f"Waterfall: Low Risk Patient (Risk Score: {risk_scores[low_risk_idx]:.4f})", fontsize=12)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/05b_Waterfall_LowRisk.pdf', dpi=300, bbox_inches='tight')
    plt.close()
    print("    ✓ Waterfall Low Risk 成功")
except Exception as e:
    print(f"    ⚠️ Waterfall Low Risk 失败: {e}")
    plt.close()

# Force - High Risk (HTML format)
try:
    force_high = shap.force_plot(base_value, sv_for_plot[high_risk_idx], X_test[high_risk_idx],
                               feature_names=shap_feature_names, matplotlib=False)
    shap.save_html(f'{output_dir}/06a_Force_HighRisk.html', force_high)
    print("    ✓ Force High Risk (HTML) 成功")
except Exception as e:
    print(f"    ⚠️ Force High Risk 失败: {e}")

# Force - Low Risk (HTML format)
try:
    force_low = shap.force_plot(base_value, sv_for_plot[low_risk_idx], X_test[low_risk_idx],
                              feature_names=shap_feature_names, matplotlib=False)
    shap.save_html(f'{output_dir}/06b_Force_LowRisk.html', force_low)
    print("    ✓ Force Low Risk (HTML) 成功")
except Exception as e:
    print(f"    ⚠️ Force Low Risk 失败: {e}")

print(f"\n✅ 所有图表已保存到: {output_dir}/")

# 保存分析报告
mean_abs_shap_values = np.mean(np.abs(sv_for_plot), axis=0)
feature_importance = pd.DataFrame({
    'Feature': shap_feature_names,
    'Mean |SHAP|': mean_abs_shap_values
}).sort_values('Mean |SHAP|', ascending=False)

report = f"""
{'='*80}
胃 RSF 模型 SHAP 主要图表分析报告
{'='*80}

分析时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
输出目录: {output_dir}

【模型性能】
测试集样本数: {len(X_test)}

测试集 AUC 指标:
"""

for t, result in auc_results.items():
    report += f"  {result['time_label']:10s}: {result['auc']:.4f}  (95% CI: {result['ci'][0]:.4f}-{result['ci'][1]:.4f})\n"

report += f"""
【特征重要性排序 (基于平均绝对SHAP值)】
"""

for idx, row in feature_importance.iterrows():
    report += f"  {row['Feature']:40s}: {row['Mean |SHAP|']:10.6f}\n"

report += f"""
【生成的图表】
1. 01_Summary_Beeswarm.pdf         - 所有样本的SHAP值分布 (蜜蜂群图)
2. 02_Feature_Importance_Bar.pdf   - 特征重要性排序 (条形图)
3. 03_Heatmap.pdf                  - 所有特征和样本的热力图
4. 05a_Waterfall_HighRisk.pdf      - 最高风险患者的预测分解
   05b_Waterfall_LowRisk.pdf       - 最低风险患者的预测分解
5. 06a_Force_HighRisk.html         - 最高风险患者的力图 (HTML交互)
   06b_Force_LowRisk.html          - 最低风险患者的力图 (HTML交互)

（注：依赖图/散点图请参考 胃SHAP_scatter.py）

{'='*80}
"""

report_file = f'{output_dir}/分析报告.txt'
with open(report_file, 'w', encoding='utf-8') as f:
    f.write(report)

importance_file = f'{output_dir}/特征重要性.csv'
feature_importance.to_csv(importance_file, index=False, encoding='utf-8')

print(f"✓ 分析报告已保存")
print("\n" + "=" * 80)
print("✅ 胃 SHAP 主要图表分析完成！")
print("=" * 80)
