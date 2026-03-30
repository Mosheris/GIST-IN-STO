# ==============================================================================
# 🏥 RSF 模型训练与封包脚本 (利用 intestine_final 和 Stomach_final)
# ==============================================================================
import pandas as pd
import numpy as np
import pickle
import warnings
from datetime import datetime
from tqdm import tqdm
from sksurv.ensemble import RandomSurvivalForest
from sksurv.metrics import concordance_index_censored, cumulative_dynamic_auc
from sksurv.nonparametric import kaplan_meier_estimator
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore')

# 🔥 锁死随机种子
np.random.seed(42)

print("=" * 80)
print("🏥 RSF 模型训练与封包系统 (小肠 vs 胃)")
print("=" * 80)

# ===============================================================================
# 第1部分：数据加载与预处理
# ===============================================================================
print("\n[1/4] 📊 加载数据...")

# 小肠数据 - 分离训练/测试集
intestine_train = pd.read_excel('Intestine_final.xlsx', sheet_name='Train')
intestine_test = pd.read_excel('Intestine_final.xlsx', sheet_name='Test')
print(f"✓ Intestine_final.xlsx - Train: {intestine_train.shape}, Test: {intestine_test.shape}")

# 胃部数据 - 分离训练/测试集
stomach_train = pd.read_excel('Stomach_final.xlsx', sheet_name='Train')
stomach_test = pd.read_excel('Stomach_final.xlsx', sheet_name='Test')
print(f"✓ Stomach_final.xlsx - Train: {stomach_train.shape}, Test: {stomach_test.shape}")

# 特征列表 (排除OS标签和outcome)
# 小肠：包含全部特征
intestine_features = ['Age', 'Tumor.size', 'Mitotic.rate', 'Race.0', 'Race.1', 'Race.2', 
                      'Marital.status.0', 'Marital.status.1', 'Marital.status.2', 'Gender', 
                      'Systemic.treatment', 'Liver.metastasis']

# 胃：不用Systemic.treatment特征
stomach_features = ['Age', 'Tumor.size', 'Mitotic.rate', 'Race.0', 'Race.1', 'Race.2', 
                    'Marital.status.0', 'Marital.status.1', 'Marital.status.2', 'Gender', 
                    'Liver.metastasis']

# ===============================================================================
# 第2部分：模型训练函数
# ===============================================================================
def train_rsf_model(data_train, data_test, cohort_name, n_estimators=100, feature_cols=None):
    """
    训练RSF模型并返回模型、特征和标签（使用分开的训练/测试集）
    """
    print(f"\n[2/4] 🚀 训练 {cohort_name} RSF 模型...")
    
    # 提取特征和标签
    X_train = data_train[feature_cols].values
    X_test = data_test[feature_cols].values
    
    y_train_array = np.array([(bool(s), t) for s, t in zip(data_train['OS.status'], data_train['OS.months'])],
                             dtype=[('Status', '?'), ('Survival_in_months', '<f8')])
    y_test_array = np.array([(bool(s), t) for s, t in zip(data_test['OS.status'], data_test['OS.months'])],
                            dtype=[('Status', '?'), ('Survival_in_months', '<f8')])
    
    # 模型配置
    rsf_model = RandomSurvivalForest(
        n_estimators=n_estimators,
        min_samples_split=15,
        min_samples_leaf=5,
        random_state=42,
        n_jobs=-1
    )
    
    # 训练模型
    rsf_model.fit(X_train, y_train_array)
    print(f"✓ {cohort_name} RSF 模型训练完成！")
    print(f"  - 训练集样本数: {len(X_train)}")
    print(f"  - 测试集样本数: {len(X_test)}")
    print(f"  - 特征数: {X_train.shape[1]}")
    print(f"  - n_estimators: {n_estimators}")
    
    return rsf_model, X_train, y_train_array, X_test, y_test_array, feature_cols

# ===============================================================================
# 第3部分：AUC 计算函数（使用 cumulative_dynamic_auc）
# ===============================================================================
def calculate_auc_at_times(model, y_train, y_test, X_train, X_test, times=[12, 36, 60]):
    """
    计算指定时间点的动态 AUC（在测试集上评估）
    """
    print(f"\n[3/4] 📈 计算测试集动态AUC (cumulative_dynamic_auc)...")
    
    # 获取测试集的风险分数
    risk_scores = model.predict(X_test)
    
    # 一次计算所有时间点的AUC（使用训练集和测试集）
    times_array = np.array(times)
    auc_points, _ = cumulative_dynamic_auc(y_train, y_test, risk_scores, times_array)
    
    auc_results = {}
    time_labels = {12: '1-Year', 36: '3-Year', 60: '5-Year'}
    
    # Bootstrap 置信区间计算
    rng = np.random.RandomState(42)
    auc_boot = {t: [] for t in times}
    
    print("  📊 计算 Bootstrap 置信区间 (1000 次迭代)...")
    for _ in tqdm(range(1000), desc="  Bootstrap", leave=False, ncols=60):
        idx = rng.randint(0, len(y_test), len(y_test))
        y_test_boot = y_test[idx]
        risk_scores_boot = risk_scores[idx]
        
        try:
            auc_boot_point, _ = cumulative_dynamic_auc(y_train, y_test_boot, risk_scores_boot, times_array)
            for i, t in enumerate(times):
                auc_boot[t].append(auc_boot_point[i])
        except:
            pass
    
    # 计算置信区间
    for i, t in enumerate(times):
        auc_boot[t].sort()
        ci_lower = auc_boot[t][25] if len(auc_boot[t]) > 25 else np.nan
        ci_upper = auc_boot[t][975] if len(auc_boot[t]) > 975 else np.nan
        
        auc_results[t] = {
            'time_months': t,
            'time_label': time_labels.get(t, f'{t}m'),
            'auc': float(auc_points[i]),
            'ci': (float(ci_lower), float(ci_upper))
        }
        print(f"  ✓ {time_labels.get(t, f'{t}m')}: AUC = {auc_points[i]:.4f} (95% CI: {ci_lower:.4f}-{ci_upper:.4f})")
    
    return auc_results

# ===============================================================================
# 第4部分：模型保存与汇总输出
# ===============================================================================
def save_model_package(model, X_data, y_data, feature_cols, cohort_name, auc_results):
    """
    将模型、数据和参数封包为PKL文件
    """
    print(f"\n[4/4] 💾 封包 {cohort_name} 模型为 PKL...")
    
    # 创建完整的模型包
    model_package = {
        'model': model,
        'X_train': X_data,
        'y_train': y_data,
        'feature_cols': feature_cols,
        'cohort': cohort_name,
        'timestamp': datetime.now().isoformat(),
        'model_type': 'RandomSurvivalForest',
        'hyperparameters': {
            'n_estimators': model.n_estimators,
            'min_samples_split': model.min_samples_split,
            'min_samples_leaf': model.min_samples_leaf,
            'random_state': model.random_state
        },
        'auc_results': auc_results
    }
    
    # 保存为PKL
    filename = f"RSF_{cohort_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl"
    with open(filename, 'wb') as f:
        pickle.dump(model_package, f)
    
    print(f"✓ 模型已保存: {filename}")
    
    return filename, model_package

# ===============================================================================
# 主程序执行
# ===============================================================================
print("\n" + "=" * 80)
print("开始训练小肠 RSF 模型...")
print("=" * 80)

# 小肠模型训练
intestine_model, X_intestine_train, y_intestine_train, X_intestine_test, y_intestine_test, features = train_rsf_model(
    intestine_train,
    intestine_test,
    cohort_name='小肠',
    n_estimators=100,
    feature_cols=intestine_features
)

# 小肠AUC计算（在测试集上评估）
intestine_auc = calculate_auc_at_times(
    intestine_model, 
    y_intestine_train, 
    y_intestine_test, 
    X_intestine_train,
    X_intestine_test,
    times=[12, 36, 60]
)

# 小肠模型保存
intestine_filename, intestine_package = save_model_package(
    intestine_model, 
    X_intestine_train, 
    y_intestine_train, 
    features, 
    '小肠', 
    intestine_auc
)

print("\n" + "=" * 80)
print("开始训练胃部 RSF 模型...")
print("=" * 80)

# 胃部模型训练
stomach_model, X_stomach_train, y_stomach_train, X_stomach_test, y_stomach_test, features = train_rsf_model(
    stomach_train,
    stomach_test,
    cohort_name='胃',
    n_estimators=150,
    feature_cols=stomach_features
)

# 胃部AUC计算（在测试集上评估）
stomach_auc = calculate_auc_at_times(
    stomach_model, 
    y_stomach_train, 
    y_stomach_test, 
    X_stomach_train,
    X_stomach_test,
    times=[12, 36, 60]
)

# 胃部模型保存
stomach_filename, stomach_package = save_model_package(
    stomach_model, 
    X_stomach_train, 
    y_stomach_train, 
    features, 
    '胃', 
    stomach_auc
)

# ===============================================================================
# 最终汇总输出
# ===============================================================================
print("\n" + "=" * 80)
print("📊 模型训练完成！最终汇总结果")
print("=" * 80)

print("\n【小肠 RSF 模型】")
print(f"  📁 模型文件: {intestine_filename}")
print(f"  📊 动态AUC指标:")
for t, result in intestine_auc.items():
    print(f"     {result['time_label']:10s}: AUC = {result['auc']:.4f}  (95% CI: {result['ci'][0]:.4f}-{result['ci'][1]:.4f})")

print("\n【胃部 RSF 模型】")
print(f"  📁 模型文件: {stomach_filename}")
print(f"  📊 动态AUC指标:")
for t, result in stomach_auc.items():
    print(f"     {result['time_label']:10s}: AUC = {result['auc']:.4f}  (95% CI: {result['ci'][0]:.4f}-{result['ci'][1]:.4f})")

print("\n" + "=" * 80)
print("✅ 所有模型已成功训练和封包！")
print("=" * 80)

# 保存汇总报告
summary_report = f"""
{'='*80}
RSF 模型训练汇总报告 (独立测试集评估)
{'='*80}

训练时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

【小肠 RSF 模型】
文件名: {intestine_filename}
训练集样本数: {len(X_intestine_train)}
测试集样本数: {len(X_intestine_test)}
特征数: {X_intestine_train.shape[1]}
超参数: n_estimators=100, min_samples_split=15, min_samples_leaf=5

测试集 AUC 指标:
"""

for t, result in intestine_auc.items():
    summary_report += f"  {result['time_label']:10s}: {result['auc']:.4f}  (95% CI: {result['ci'][0]:.4f}-{result['ci'][1]:.4f})\n"

summary_report += f"""
【胃部 RSF 模型】
文件名: {stomach_filename}
训练集样本数: {len(X_stomach_train)}
测试集样本数: {len(X_stomach_test)}
特征数: {X_stomach_train.shape[1]}
超参数: n_estimators=150, min_samples_split=15, min_samples_leaf=5

测试集 AUC 指标:
"""

for t, result in stomach_auc.items():
    summary_report += f"  {result['time_label']:10s}: {result['auc']:.4f}  (95% CI: {result['ci'][0]:.4f}-{result['ci'][1]:.4f})\n"

summary_report += f"""
{'='*80}
模型使用说明:
使用 pickle.load(open(model_filename, 'rb')) 来加载模型包
模型包包含: model, X_train, y_train, feature_cols, hyperparameters, auc_results
{'='*80}
"""

with open('RSF_模型训练汇总.txt', 'w', encoding='utf-8') as f:
    f.write(summary_report)

print("\n📄 汇总报告已保存: RSF_模型训练汇总.txt")
