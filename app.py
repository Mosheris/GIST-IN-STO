import pickle
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import shap
import streamlit as st
import streamlit.components.v1 as components


warnings.filterwarnings("ignore")


st.set_page_config(
    page_title="Gastric / Small Intestine RSF Survival Risk Calculator",
    page_icon="🧬",
    layout="wide",
)

st.markdown(
    """
<style>
    .main-header {
        font-size: 2.1rem;
        font-weight: 700;
        color: #15304a;
        text-align: center;
        padding: 0.9rem 0 0.4rem 0;
    }
    .sub-header {
        text-align: center;
        color: #4a5f73;
        margin-bottom: 1rem;
    }
    .prediction-box {
        padding: 1.5rem 1.6rem;
        border-radius: 14px;
        border-left: 6px solid #1f77b4;
        background: linear-gradient(180deg, rgba(255,255,255,0.96), rgba(245,248,252,0.96));
        box-shadow: 0 6px 24px rgba(22, 34, 51, 0.08);
        margin-bottom: 1rem;
    }
    .high-risk {
        border-left-color: #d32f2f;
        background: linear-gradient(180deg, #fff5f5, #ffecec);
    }
    .medium-risk {
        border-left-color: #f57c00;
        background: linear-gradient(180deg, #fffaf0, #fff4df);
    }
    .low-risk {
        border-left-color: #2e7d32;
        background: linear-gradient(180deg, #f4fff4, #eaf9ea);
    }
    .note-box {
        padding: 0.85rem 1rem;
        border-radius: 10px;
        background: #f6f8fb;
        border: 1px solid #dfe6ee;
        color: #334155;
        font-size: 0.95rem;
    }
</style>
""",
    unsafe_allow_html=True,
)


MODEL_CONFIGS = {
    "Gastric": {
        "title": "Gastric RSF Survival Risk Calculator",
        "file": "Stomach.pkl",
        "key_prefix": "stomach",
        "description": "Individual survival risk assessment for gastric RSF model.",
    },
    "Small Intestine": {
        "title": "Small Intestine RSF Survival Risk Calculator",
        "file": "Intestine.pkl",
        "key_prefix": "intestine",
        "description": "Individual survival risk assessment for small intestine RSF model.",
    },
}

BASE_FEATURE_LABELS = {
    "Age": "Age (Years)",
    "Tumor.size": "Tumor Size (cm)",
    "Mitotic.rate": "Mitotic Rate (/50 HPF)",
    "Race.0": "Race: White",
    "Race.1": "Race: Black",
    "Race.2": "Race: Asian/Pacific Islander",
    "Marital.status.0": "Marital Status: Married",
    "Marital.status.1": "Marital Status: Single/Unmarried",
    "Marital.status.2": "Marital Status: Separated/Divorced/Widowed",
    "Gender": "Gender (Male)",
    "Systemic.treatment": "Systemic Treatment",
    "Liver.metastasis": "Liver Metastasis",
}

RACE_OPTIONS = [
    "White",
    "Black",
    "Asian/Pacific Islander",
]

MARITAL_OPTIONS = [
    "Married",
    "Single / Unmarried",
    "Separated / Divorced / Widowed",
]

GENDER_OPTIONS = ["Female", "Male"]
BINARY_OPTIONS = ["No", "Yes"]


def binary_value(label: str) -> int:
    return 1 if label == "Yes" else 0


def encode_race(label: str) -> dict:
    if label == "White":
        return {"Race.0": 1, "Race.1": 0, "Race.2": 0}
    if label == "Black":
        return {"Race.0": 0, "Race.1": 1, "Race.2": 0}
    if label == "Asian/Pacific Islander":
        return {"Race.0": 0, "Race.1": 0, "Race.2": 1}
    return {"Race.0": 0, "Race.1": 0, "Race.2": 0}


def encode_marital(label: str) -> dict:
    if label == "Married":
        return {"Marital.status.0": 1, "Marital.status.1": 0, "Marital.status.2": 0}
    if label == "Single / Unmarried":
        return {"Marital.status.0": 0, "Marital.status.1": 1, "Marital.status.2": 0}
    if label == "Separated / Divorced / Widowed":
        return {"Marital.status.0": 0, "Marital.status.1": 0, "Marital.status.2": 1}
    return {"Marital.status.0": 0, "Marital.status.1": 0, "Marital.status.2": 0}


def load_model_package(model_file: str) -> dict:
    script_dir = Path(__file__).parent
    model_path = script_dir / model_file

    if not model_path.exists():
        st.error(f"❌ 找不到模型文件: {model_path}")
        st.stop()

    with model_path.open("rb") as f:
        return pickle.load(f)


@st.cache_resource
def cached_model_package(model_file: str) -> dict:
    return load_model_package(model_file)


@st.cache_resource
def cached_model_and_explainer(model_file: str):
    model_pkg = load_model_package(model_file)
    model = model_pkg["model"]
    x_train = np.asarray(model_pkg.get("X_train"))

    if x_train.ndim != 2 or len(x_train) == 0:
        raise ValueError("训练数据为空，无法构建 SHAP 背景样本")

    background_size = min(50, len(x_train))
    background = shap.sample(x_train, background_size, random_state=42)

    def predict_fn(x):
        return np.asarray(model.predict(np.asarray(x)))

    explainer = shap.KernelExplainer(predict_fn, background)
    return model_pkg, explainer


def shap_target_from_values(shap_values):
    if isinstance(shap_values, list):
        if len(shap_values) > 1:
            return shap_values[1][0]
        return shap_values[0][0]

    if isinstance(shap_values, np.ndarray):
        if shap_values.ndim == 3:
            return shap_values[0, :, 1] if shap_values.shape[2] > 1 else shap_values[0, :, 0]
        if shap_values.ndim == 2:
            return shap_values[0, :]
        return shap_values[0]

    return shap_values[0] if hasattr(shap_values, "__getitem__") else shap_values


def base_value_from_expected(expected_value):
    if isinstance(expected_value, (list, np.ndarray)):
        if len(expected_value) > 1:
            return expected_value[1]
        return expected_value[0]
    return expected_value


def build_input_row(
    feature_cols,
    age,
    tumor_size,
    mitotic_rate,
    race_label,
    marital_label,
    gender_label,
    liver_label,
    systemic_label=None,
):
    row = {feature: 0 for feature in feature_cols}

    if "Age" in row:
        row["Age"] = age
    if "Tumor.size" in row:
        row["Tumor.size"] = tumor_size * 10  # Convert cm to mm
    if "Mitotic.rate" in row:
        row["Mitotic.rate"] = mitotic_rate * 10
    if "Gender" in row:
        row["Gender"] = 1 if gender_label == "Male" else 0

    row.update(encode_race(race_label))
    row.update(encode_marital(marital_label))

    if "Systemic.treatment" in row and systemic_label is not None:
        row["Systemic.treatment"] = binary_value(systemic_label)
    if "Liver.metastasis" in row:
        row["Liver.metastasis"] = binary_value(liver_label)

    return pd.DataFrame([row], columns=feature_cols)


selected_model = st.sidebar.radio("Select Model", list(MODEL_CONFIGS.keys()))
config = MODEL_CONFIGS[selected_model]
model_pkg, explainer = cached_model_and_explainer(config["file"])
model = model_pkg["model"]
feature_cols = list(model_pkg["feature_cols"])
X_train = model_pkg.get("X_train")
cohort = model_pkg.get("cohort", selected_model)

feature_labels = {feature: BASE_FEATURE_LABELS.get(feature, feature) for feature in feature_cols}
shap_feature_names = [feature_labels[feature] for feature in feature_cols]

st.markdown(f'<div class="main-header">{config["title"]}</div>', unsafe_allow_html=True)
st.markdown(f'<div class="sub-header">{config["description"]}</div>', unsafe_allow_html=True)

with st.sidebar:
    st.markdown("---")
    st.header("📋 Patient Clinical Feature Input")
    with st.form("patient_data_form"):
        key_prefix = config["key_prefix"]
        age = st.number_input("Age (years)", min_value=0, max_value=120, value=60, step=1, key=f"{key_prefix}_age")
        tumor_size = st.number_input("Tumor Size (cm)", min_value=0.0, max_value=50.0, value=3.0, step=0.1, key=f"{key_prefix}_tumor")
        mitotic_rate = st.number_input("Mitotic Rate (/50 HPF)", min_value=0.0, max_value=100.0, value=5.0, step=0.1, key=f"{key_prefix}_mitotic")

        race_label = st.selectbox("Race", options=RACE_OPTIONS, key=f"{key_prefix}_race")
        marital_label = st.selectbox("Marital Status", options=MARITAL_OPTIONS, key=f"{key_prefix}_marital")

        col_a, col_b = st.columns(2)
        with col_a:
            gender_label = st.selectbox("Gender", options=GENDER_OPTIONS, index=1, key=f"{key_prefix}_gender")
        with col_b:
            liver_label = st.selectbox("Liver Metastasis", options=BINARY_OPTIONS, key=f"{key_prefix}_liver")

        systemic_label = None
        if "Systemic.treatment" in feature_cols:
            systemic_label = st.selectbox("Systemic Treatment", options=BINARY_OPTIONS, key=f"{key_prefix}_systemic")

        submit_button = st.form_submit_button("🔍 Start Prediction", type="primary")


col1, col2 = st.columns([1, 1.25])

if submit_button:
    input_df = build_input_row(
        feature_cols=feature_cols,
        age=age,
        tumor_size=tumor_size,
        mitotic_rate=mitotic_rate,
        race_label=race_label,
        marital_label=marital_label,
        gender_label=gender_label,
        liver_label=liver_label,
        systemic_label=systemic_label,
    )
    input_array = input_df.to_numpy()

    with col1:
        st.subheader("📊 Risk Prediction Results")
        try:
            risk_score = float(np.asarray(model.predict(input_array))[0])
            train_scores = np.asarray(model.predict(X_train)) if X_train is not None else np.asarray([risk_score])
            risk_percentile = float((train_scores <= risk_score).mean() * 100.0)

            if risk_percentile < 33:
                level, css, emoji = "Low Risk", "low-risk", "🟢"
                advice = "相对风险较低，可按常规随访节奏监测。<br><small style='color: #666;'>Relatively low risk. Follow-up monitoring at routine intervals is recommended.</small>"
            elif risk_percentile < 66:
                level, css, emoji = "Medium Risk", "medium-risk", "🟡"
                advice = "属于中等相对风险，建议缩短复查间隔并加强监测。<br><small style='color: #666;'>Moderate relative risk. Recommend shortening follow-up intervals and intensifying monitoring.</small>"
            else:
                level, css, emoji = "High Risk", "high-risk", "🔴"
                advice = "相对风险较高，建议进一步密切随访并结合临床综合判断。<br><small style='color: #666;'>Relatively high risk. Close follow-up monitoring and comprehensive clinical judgment are recommended.</small>"

            # 生存时间预测
            survival_info = ""
            try:
                surv_func = model.predict_survival_function(input_array)
                times = surv_func[0].x
                probs = surv_func[0].y
                
                # 1年（12个月）生存概率
                one_year_prob = None
                for t, p in zip(times, probs):
                    if t >= 12:
                        one_year_prob = p
                        break
                
                # 3年（36个月）生存概率
                three_year_prob = None
                for t, p in zip(times, probs):
                    if t >= 36:
                        three_year_prob = p
                        break
                
                # 5年（60个月）生存概率
                five_year_prob = None
                for t, p in zip(times, probs):
                    if t >= 60:
                        five_year_prob = p
                        break
                
                # 中位生存时间
                median_survival = None
                for t, p in zip(times, probs):
                    if p <= 0.5:
                        median_survival = t
                        break
                
                if one_year_prob is not None:
                    survival_info += f"<p style='margin-top:10px; font-size:0.98rem; color: #334155;'>1-Year Survival Rate: {one_year_prob*100:.1f}%</p>"
                
                if three_year_prob is not None:
                    survival_info += f"<p style='font-size:0.98rem; color: #334155;'>3-Year Survival Rate: {three_year_prob*100:.1f}%</p>"
                
                if five_year_prob is not None:
                    survival_info += f"<p style='font-size:0.98rem; color: #334155;'>5-Year Survival Rate: {five_year_prob*100:.1f}%</p>"
                
                if median_survival is not None:
                    survival_info += f"<p style='font-size:0.98rem; color: #334155;'>⏱️ Median Survival: {median_survival:.1f} months</p>"
            except Exception as surv_error:
                survival_info = ""

            st.markdown(
                f"""
                <div class="prediction-box {css}">
                    <h2 style="color: black; margin-bottom: 0.5rem;">{emoji} Risk Score: {risk_score:.4f}</h2>
                    <h3 style="color: black; margin-bottom: 0.4rem;">Relative Risk Level: {level}</h3>
                    <p style="margin-top:10px; font-size:1.02rem; color: black;">{advice}</p>
                    {survival_info}
                </div>
                """,
                unsafe_allow_html=True,
            )

            with st.expander("View Encoded Input Data"):
                st.dataframe(input_df.rename(columns=feature_labels), hide_index=True, use_container_width=True)

        except Exception as e:
            st.error(f"Prediction error: {e}")

    with col2:
        st.subheader("🔥 Feature Attribution Analysis (SHAP)")
        try:
            with st.spinner("Calculating feature contributions..."):
                shap_values = explainer.shap_values(input_array, nsamples=150)
                shap_target = shap_target_from_values(shap_values)
                expected_value = base_value_from_expected(explainer.expected_value)

                try:
                    shap_html = shap.force_plot(
                        expected_value,
                        shap_target,
                        input_df.iloc[0],
                        feature_names=shap_feature_names,
                        matplotlib=False,
                    )
                    components.html(shap.getjs() + shap_html.html(), height=360, scrolling=True)
                except Exception:
                    shap_df = pd.DataFrame(
                        {
                            "Feature": shap_feature_names,
                            "SHAP Value": shap_target,
                        }
                    ).sort_values("SHAP Value", key=abs, ascending=False).head(10)

                    st.warning("Force plot rendering failed, showing bar chart fallback.")
                    st.bar_chart(shap_df.set_index("Feature")["SHAP Value"])

                st.info(
                    """
                    **Interpretation**
                    - Red bars: factors pushing the risk score upward.
                    - Blue bars: factors pushing the risk score downward.
                    - The relative length reflects influence strength.
                    """
                )

        except Exception as e:
            st.error(f"SHAP analysis failed: {e}")

else:
    with col1:
        st.write("")
        st.write("")


st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #667085;'>Powered by Streamlit + scikit-survival + SHAP</div>",
    unsafe_allow_html=True,
)
