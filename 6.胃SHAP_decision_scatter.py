# ==============================================================================
# 胃 RSF 模型 SHAP 决策图 + 散点图 (合一版本)
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
print("🏥 胃 RSF 模型 SHAP 决策图 + 散点图分析 (Combined Version)")
print("=" * 80)

# ===============================================================================
# 第1部分：加载模型和数据
# ===============================================================================
print("\n[1/5] 📊 加载模型和数据...")

model_file = 'Stomach.pkl'
if not os.path.exists(model_file):
    raise FileNotFoundError(f"❌ 找不到文件: {model_file}")

with open(model_file, 'rb') as f:
    model_pkg = pickle.load(f)

model = model_pkg['model']
X_train = model_pkg['X_train']
feature_cols = model_pkg['feature_cols']

# 加载测试数据
test_df = pd.read_excel('Stomach_final.xlsx', sheet_name='Test')
X_test = test_df[feature_cols]

print(f"✓ 训练集: {X_train.shape[0]} 样本 (用于SHAP背景)")
print(f"✓ 测试集: {len(X_test)} 样本 (用于绘图)")

# ===============================================================================
# 第2部分：特征标签和输出目录
# ===============================================================================
print("\n[2/5] 🏷️  准备特征标签...")

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
    'Liver.metastasis': 'Liver Metastasis'
}

shap_feature_names = [feature_labels.get(f, f) for f in feature_cols]

timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
output_dir = f"SHAP_胃_combined_{timestamp}"
os.makedirs(output_dir, exist_ok=True)
print(f"✓ {len(shap_feature_names)} 个特征已准备，输出目录: {output_dir}")

# ===============================================================================
# 第3部分：计算 SHAP 值 (共享，仅计算一次)
# ===============================================================================
print("\n[3/5] 🧮 计算 SHAP 值...")

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

# 构建 Explanation 对象（用于散点图）
shap_explanation = shap.Explanation(
    values=sv_for_plot,
    base_values=base_value,
    data=X_test.values,
    feature_names=shap_feature_names
)

# ===============================================================================
# 第4部分：生成 Decision Plot
# ===============================================================================
print("\n[4/5] 📈 生成 Decision Plot...")

# 临床数值还原
print("  - 正在将 Mitotic.rate 还原为真实临床记录数值 (除以 10)...")
X_test_display = X_test.copy()
if 'Mitotic.rate' in feature_cols:
    X_test_display['Mitotic.rate'] = X_test_display['Mitotic.rate'] / 10.0

# 计算特征总体重要性
feature_importance = np.abs(sv_for_plot).mean(0)
importance_df = pd.DataFrame({
    'Feature': shap_feature_names,
    'Mean |SHAP|': feature_importance
}).sort_values('Mean |SHAP|', ascending=False).reset_index(drop=True)
importance_df.index = importance_df.index + 1
importance_df.index.name = 'Rank'

# 采样以保持可视化清晰
n_display = min(600, len(X_test))
np.random.seed(42)
display_indices = np.random.choice(len(X_test), n_display, replace=False)

print(f"  - 正在绘制 Decision Plot (展示 {n_display}/{len(X_test)} 个样本轨迹)...")

plt.figure(figsize=(10, 10))
try:
    shap.decision_plot(
        base_value=base_value,
        shap_values=sv_for_plot[display_indices],
        features=X_test_display.values[display_indices],
        feature_names=shap_feature_names,
        feature_order='importance',
        alpha=0.15,
        show=False
    )
    plt.title("SHAP Decision Plot (Ordered by Importance)", fontsize=16, pad=20, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/01_Decision_Plot.pdf', dpi=300, bbox_inches='tight')
    plt.close()
    print("  ✓ Decision Plot 生成成功 (PDF)")
except Exception as e:
    print(f"  ⚠️ Decision Plot 生成失败: {e}")
    plt.close()

# ===============================================================================
# 第5部分：生成 Scatter Plot (Dependence Plots with Histogram)
# ===============================================================================
print("\n[5/5] 📊 生成 Scatter Plots...")

# 选择去连续变量或绘制所有特征的散点图
target_features = ['Age', 'Tumor.size', 'Mitotic.rate']

for feature in target_features:
    if feature not in feature_cols:
        continue

    print(f"  - 正在绘制 [{feature}] 的散点图...")

    # 获取特征值和SHAP值
    feat_idx = feature_cols.index(feature)
    x_vals = X_test[feature].values
    y_vals = sv_for_plot[:, feat_idx]

    # 创建画布
    fig, ax = plt.subplots(figsize=(10, 7))

    # 绘制带直方图的散点图
    shap.plots.scatter(
        shap_explanation[:, feature],
        hist=True,
        ax=ax,
        show=False,
        color=X_test[feature].values
    )

    # 添加辅助线 (SHAP=0)
    ax.axhline(0, color='gray', linestyle='--', linewidth=1, alpha=0.8, zorder=-1)

    # 美化标题
    ax.set_title(f"SHAP Dependence: {feature}\n(SHAP > 0 indicates Higher Risk)",
                fontsize=12, fontweight='bold', pad=15)

    # 保存
    save_name = f'{output_dir}/02_Scatter_{feature}.pdf'
    plt.savefig(save_name, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"    ✓ 已保存: Scatter_{feature}.pdf")

# ===============================================================================
# 生成报告
# ===============================================================================
importance_file = f'{output_dir}/特征重要性_排序.csv'
importance_df.to_csv(importance_file, encoding='utf-8')

report = f"""
{'='*80}
胃 RSF 模型 SHAP 决策图 + 散点图分析报告
{'='*80}

分析时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
输出目录: {output_dir}

【模型数据信息】
训练集样本数: {X_train.shape[0]}
测试集样本数: {len(X_test)}
分析特征数: {len(shap_feature_names)}

【生成的图表】
1. 01_Decision_Plot.pdf
   - 展示了模型预测的累积决策路径
   - 特征从下往上按重要性递减排列
   - 每条线代表一个患者的预测过程

2. 02_Scatter_*.pdf (Age, Tumor.size, Mitotic.rate)
   - X 轴: 特征值
   - Y 轴: SHAP 值 (该特征对风险的贡献)
   - 底部灰色直方图: 该数值区间内的样本分布
   - 虚线 (SHAP=0): 分界线，上方增加风险，下方降低风险

【特征重要性排序 (按标准 |SHAP| 值)】
"""

for idx, row in importance_df.iterrows():
    report += f"  {idx:2d}. {row['Feature']:40s}: {row['Mean |SHAP|']:10.6f}\n"

report += f"""
【计算优化】
- 使用 TreeExplainer (若支持) 或 KernelExplainer + K-Means 浓缩 (50 个代表点) + nsamples=150
- 这获得了显著的速度提升（接近实时计算）

{'='*80}
"""

report_file = f'{output_dir}/分析报告.txt'
with open(report_file, 'w', encoding='utf-8') as f:
    f.write(report)

print(f"\n✅ 分析报告已保存至 {output_dir}")
print("=" * 80)
print("🚀 胃 SHAP 决策图 + 散点图分析完美收官！")
print("=" * 80)
