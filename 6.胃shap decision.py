# ==============================================================================
# 胃 RSF 模型 SHAP 决策图 (极速提速 + 临床数值还原版)
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
print("🏥 胃 RSF 模型 SHAP 决策图 (Decision Plot)")
print("=" * 80)

# ===============================================================================
# 第1部分：加载模型和数据
# ===============================================================================
print("\n[1/3] 📊 加载模型和数据...")

model_file = 'Stomach.pkl'
if not os.path.exists(model_file):
    raise FileNotFoundError(f"❌ 找不到文件: {model_file}")

with open(model_file, 'rb') as f:
    model_pkg = pickle.load(f)

model = model_pkg['model']
X_train = model_pkg['X_train']
feature_cols = model_pkg['feature_cols']

# 加载测试数据 (用于绘图展示)
test_df = pd.read_excel('Stomach_final.xlsx', sheet_name='Test')
X_test = test_df[feature_cols].values

print(f"✓ 训练集: {X_train.shape[0]} 样本 (用于SHAP背景)")
print(f"✓ 测试集: {len(X_test)} 个样本 (用于决策图展示)")

# ===============================================================================
# 第2部分：特征标签和输出目录
# ===============================================================================
print("\n[2/3] 🏷️  准备特征标签 (精准对齐底层 R 代码)...")

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
output_dir = f"SHAP_胃_decision_{timestamp}"
os.makedirs(output_dir, exist_ok=True)
print(f"✓ {len(shap_feature_names)} 个特征马甲已准备，输出目录: {output_dir}")

# ===============================================================================
# 第3部分：计算 SHAP 值并生成决策图
# ===============================================================================
print("\n[3/3] 🧮 计算 SHAP 值并生成决策图 (应用降维提速技术)...")

def predict_model(X):
    return model.predict(X)

try:
    print("  尝试 TreeExplainer (快速模式)...")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)
    print("✓ TreeExplainer 计算完成")
except Exception as e:
    print(f"  ⚠️ TreeExplainer 不支持，无缝切换 KernelExplainer...")
    # 🌟 提速秘籍 1：使用 K-Means 浓缩背景数据 (提速 50 倍)
    print("  - 正在使用 K-Means 浓缩背景数据...")
    background_sample = shap.kmeans(X_train, 50)
    explainer = shap.KernelExplainer(predict_model, background_sample)

    # 🌟 提速秘籍 2：限制采样次数 nsamples=150
    print(f"  - 正在推演全部 {len(X_test)} 个测试样本的决策路径 (请稍候)...")
    shap_values = explainer.shap_values(X_test, nsamples=150)
    print("✓ KernelExplainer 计算完成")

# 提取SHAP值和基线值
sv_for_plot = shap_values[0] if isinstance(shap_values, list) else shap_values

base_value = explainer.expected_value
if isinstance(base_value, (np.ndarray, list)):
    base_value = base_value[0]

# ===============================================================================
# ⚕️ 核心步骤：临床数值还原
# ===============================================================================
print("  - 正在将 Mitotic.rate 还原为真实临床记录数值 (除以 10)...")
X_test_display = X_test.copy()
if 'Mitotic.rate' in feature_cols:
    mitotic_idx = feature_cols.index('Mitotic.rate')
    X_test_display[:, mitotic_idx] = X_test_display[:, mitotic_idx] / 10.0

# 计算特征总体重要性，用于生成报表
feature_importance = np.abs(sv_for_plot).mean(0)
importance_df = pd.DataFrame({
    'Feature': shap_feature_names,
    'Mean |SHAP|': feature_importance
}).sort_values('Mean |SHAP|', ascending=False).reset_index(drop=True)
importance_df.index = importance_df.index + 1
importance_df.index.name = 'Rank'

# 采样以保持可视化清晰 (防止几千根线挤在一起变成黑乎乎一片)
n_display = min(600, len(X_test))
np.random.seed(42)
display_indices = np.random.choice(len(X_test), n_display, replace=False)

print(f"  - 正在绘制 Decision Plot (展示 {n_display}/{len(X_test)} 个样本轨迹)...")

plt.figure(figsize=(10, 10))
try:
    # 直接喂全量特征，让 shap 自己根据 importance 排序
    shap.decision_plot(
        base_value=base_value,
        shap_values=sv_for_plot[display_indices],
        features=X_test_display[display_indices], # 使用除以 10 后的真实数据，确保颜色映射正确
        feature_names=shap_feature_names,
        feature_order='importance', # 🌟 核心：强制严格从大到小排序
        alpha=0.15,
        show=False
    )
    plt.title("SHAP Decision Plot (Ordered by Importance)", fontsize=16, pad=20, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/Decision_Plot.pdf', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Decision Plot 生成成功 (PDF)")
except Exception as e:
    print(f"⚠️ Decision Plot 生成失败: {e}")
    plt.close()

# ===============================================================================
# 生成和保存报告
# ===============================================================================
importance_file = f'{output_dir}/特征重要性_决策图排序依据.csv'
importance_df.to_csv(importance_file, encoding='utf-8')

report = f"""
{'='*80}
胃 RSF 模型 SHAP 决策图分析核心报告
{'='*80}

分析时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
输出目录: {output_dir}

【决策图说明 (Decision Plot)】
Decision Plot 展示了模型预测的累积决策路径：
- X 轴 (Bottom)：模型输出的最终"死亡风险评分 (Risk Score)"。越往右风险越高。
- X 轴 (Top)：与底层横坐标对齐的风险标尺。图底部的竖线代表人群平均基线 (Base Value)。
- Y 轴：特征名称，严格按照对模型预测的总体贡献度从大到小、自下而上排列。
- 轨迹线：每条实线代表一个测试集患者的预测历程。它从底部的基线开始，顺着Y轴向上，每遇到一个特征，该特征的 SHAP 值就会将轨迹向左 (降低风险) 或向右 (推高风险) 偏折，直至到达顶部的最终风险得分。
- 颜色映射：线条颜色反映了该特征本身数值的大小。由于我们在画图前已将核分裂象除以 10 还原为真实数值，因此红色代表该特征真实值偏高，蓝色代表该特征真实值偏低。

【特征重要性排序 (自上而下的决策依据)】
"""

for idx, row in importance_df.iterrows():
    report += f"  {idx:2d}. {row['Feature']:40s}: {row['Mean |SHAP|']:10.6f}\n"

report += f"""
【生成的图表】
- Decision_Plot.pdf: 核心矢量图，可直接用于 SCI 论文排版。
"""

report_file = f'{output_dir}/分析报告.txt'
with open(report_file, 'w', encoding='utf-8') as f:
    f.write(report)

print(f"\n✅ 分析报告已保存至 {output_dir}")
print("=" * 80)
print("🚀 胃 SHAP 决策图分析完美收官！")
print("=" * 80)
