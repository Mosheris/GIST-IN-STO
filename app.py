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
    page_title="GIST RSF Overall Survival Calculator",
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
    .higher-risk {
        border-left-color: #b3261e;
        background: linear-gradient(180deg, #fff7f7, #fff0f0);
    }
    .intermediate-risk {
        border-left-color: #b26a00;
        background: linear-gradient(180deg, #fffaf2, #fff5e6);
    }
    .lower-risk {
        border-left-color: #2e7d32;
        background: linear-gradient(180deg, #f6fff6, #edf9ed);
    }
</style>
""",
    unsafe_allow_html=True,
)


MODEL_CONFIGS = {
    "Gastric GIST": {
        "title": "Gastric GIST Overall Survival Calculator",
        "file": "rsf_gastric_gist.pkl",
        "key_prefix": "gastric",
        "description": "Site-specific random survival forest model for overall survival estimation.",
    },
    "Small intestinal GIST": {
        "title": "Small Intestinal GIST Overall Survival Calculator",
        "file": "rsf_small_intestinal_gist.pkl",
        "key_prefix": "small_intestinal",
        "description": "Site-specific random survival forest model for overall survival estimation.",
    },
}

FEATURE_LABELS = {
    "Age": "Age (years)",
    "Tumor.size": "Tumor Size (mm)",
    "Mitotic.rate": "Mitotic Rate (/50 HPF)",
    "Race.1": "Race: Black",
    "Race.2": "Race: Asian/Pacific Islander",
    "Marital.status.1": "Marital Status: Single/Unmarried",
    "Marital.status.2": "Marital Status: Separated/Divorced/Widowed",
    "Gender": "Gender (Male)",
    "Systemic.treatment": "Systemic Treatment",
    "Liver.metastasis": "Liver Metastasis",
}

RACE_OPTIONS = ["White", "Black", "Asian/Pacific Islander"]
MARITAL_OPTIONS = [
    "Married",
    "Single / Unmarried",
    "Separated / Divorced / Widowed",
]
GENDER_OPTIONS = ["Female", "Male"]
BINARY_OPTIONS = ["No", "Yes"]


def binary_value(label: str) -> int:
    return int(label == "Yes")


def encode_race(label: str) -> dict[str, int]:
    return {
        "Race.1": int(label == "Black"),
        "Race.2": int(label == "Asian/Pacific Islander"),
    }


def encode_marital(label: str) -> dict[str, int]:
    return {
        "Marital.status.1": int(label == "Single / Unmarried"),
        "Marital.status.2": int(label == "Separated / Divorced / Widowed"),
    }


def load_model_package(model_file: str) -> dict:
    model_path = Path(__file__).resolve().parent / model_file
    if not model_path.exists():
        raise FileNotFoundError(f"Model package not found: {model_path.name}")

    with model_path.open("rb") as handle:
        package = pickle.load(handle)

    required = {
        "model",
        "feature_columns",
        "shap_background",
        "risk_score_cutoffs",
        "input_ranges",
    }
    missing = sorted(required.difference(package))
    if missing:
        raise KeyError(f"Model package is missing: {', '.join(missing)}")
    return package


@st.cache_resource
def load_model_and_explainer(model_file: str):
    package = load_model_package(model_file)
    model = package["model"]
    background = np.asarray(package["shap_background"], dtype=float)
    if background.ndim != 2 or background.shape[0] == 0:
        raise ValueError("The SHAP background sample is unavailable.")

    def predict_risk(values):
        return np.asarray(model.predict(np.asarray(values, dtype=float)))

    explainer = shap.KernelExplainer(predict_risk, background)
    return package, explainer


def extract_shap_vector(shap_values) -> np.ndarray:
    if isinstance(shap_values, list):
        shap_values = shap_values[0]
    values = np.asarray(shap_values)
    if values.ndim == 1:
        return values
    if values.ndim == 2:
        return values[0]
    return values.reshape(values.shape[0], -1)[0]


def extract_expected_value(expected_value) -> float:
    return float(np.asarray(expected_value).reshape(-1)[0])


def build_input_row(
    feature_columns,
    age,
    tumor_size_mm,
    mitotic_rate_per_50_hpf,
    race_label,
    marital_label,
    gender_label,
    liver_label,
    systemic_label=None,
) -> pd.DataFrame:
    row = {feature: 0.0 for feature in feature_columns}
    row["Age"] = float(age)
    row["Tumor.size"] = float(tumor_size_mm)

    # The SEER-derived modeling field stores 10 times the displayed /50 HPF value.
    row["Mitotic.rate"] = float(mitotic_rate_per_50_hpf) * 10.0
    row["Gender"] = float(gender_label == "Male")

    for feature, value in encode_race(race_label).items():
        if feature in row:
            row[feature] = float(value)
    for feature, value in encode_marital(marital_label).items():
        if feature in row:
            row[feature] = float(value)

    if "Systemic.treatment" in row and systemic_label is not None:
        row["Systemic.treatment"] = float(binary_value(systemic_label))
    if "Liver.metastasis" in row:
        row["Liver.metastasis"] = float(binary_value(liver_label))

    return pd.DataFrame([row], columns=feature_columns)


def build_display_row(model_input: pd.DataFrame) -> pd.DataFrame:
    display = model_input.copy()
    if "Mitotic.rate" in display.columns:
        display["Mitotic.rate"] = display["Mitotic.rate"] / 10.0
    return display


def survival_probability_at(model, input_array: np.ndarray, month: float) -> float:
    survival_function = model.predict_survival_function(input_array)[0]
    return float(survival_function(month))


def clipped_default(value: float, lower: float, upper: float) -> float:
    return float(min(max(value, lower), upper))


selected_cohort = st.sidebar.radio("Model", list(MODEL_CONFIGS))
config = MODEL_CONFIGS[selected_cohort]

try:
    model_package, explainer = load_model_and_explainer(config["file"])
except Exception as error:
    st.error(f"Model loading failed: {error}")
    st.stop()

model = model_package["model"]
feature_columns = list(model_package["feature_columns"])
input_ranges = model_package["input_ranges"]
risk_cutoffs = model_package["risk_score_cutoffs"]
feature_names = [FEATURE_LABELS.get(feature, feature) for feature in feature_columns]

age_min, age_max = [int(round(value)) for value in input_ranges["Age"]]
tumor_min, tumor_max = input_ranges["Tumor.size"]
mitotic_min, mitotic_max = input_ranges["Mitotic.rate_display"]

st.markdown(f'<div class="main-header">{config["title"]}</div>', unsafe_allow_html=True)
st.markdown(f'<div class="sub-header">{config["description"]}</div>', unsafe_allow_html=True)

with st.sidebar:
    st.markdown("---")
    st.header("Patient Characteristics")
    st.caption("Input ranges correspond to the model development cohort.")

    with st.form("patient_data_form"):
        key_prefix = config["key_prefix"]
        age = st.number_input(
            "Age (years)",
            min_value=age_min,
            max_value=age_max,
            value=int(round(clipped_default(60, age_min, age_max))),
            step=1,
            key=f"{key_prefix}_age",
        )
        tumor_size = st.number_input(
            "Tumor Size (mm)",
            min_value=float(tumor_min),
            max_value=float(tumor_max),
            value=clipped_default(50.0, tumor_min, tumor_max),
            step=1.0,
            key=f"{key_prefix}_tumor",
        )
        mitotic_rate = st.number_input(
            "Mitotic Rate (/50 HPF)",
            min_value=float(mitotic_min),
            max_value=float(mitotic_max),
            value=clipped_default(5.0, mitotic_min, mitotic_max),
            step=0.1,
            key=f"{key_prefix}_mitotic",
        )
        race_label = st.selectbox("Race", RACE_OPTIONS, key=f"{key_prefix}_race")
        marital_label = st.selectbox(
            "Marital Status", MARITAL_OPTIONS, key=f"{key_prefix}_marital"
        )

        col_a, col_b = st.columns(2)
        with col_a:
            gender_label = st.selectbox(
                "Gender", GENDER_OPTIONS, key=f"{key_prefix}_gender"
            )
        with col_b:
            liver_label = st.selectbox(
                "Liver Metastasis", BINARY_OPTIONS, key=f"{key_prefix}_liver"
            )

        systemic_label = None
        if "Systemic.treatment" in feature_columns:
            systemic_label = st.selectbox(
                "Systemic Treatment", BINARY_OPTIONS, key=f"{key_prefix}_systemic"
            )

        submitted = st.form_submit_button("Calculate", type="primary")


result_column, explanation_column = st.columns([1, 1.25])

if submitted:
    input_df = build_input_row(
        feature_columns=feature_columns,
        age=age,
        tumor_size_mm=tumor_size,
        mitotic_rate_per_50_hpf=mitotic_rate,
        race_label=race_label,
        marital_label=marital_label,
        gender_label=gender_label,
        liver_label=liver_label,
        systemic_label=systemic_label,
    )
    display_df = build_display_row(input_df)
    input_array = input_df.to_numpy(dtype=float)

    with result_column:
        st.subheader("Prediction Results")
        try:
            risk_score = float(np.asarray(model.predict(input_array))[0])
            lower_cutoff = float(risk_cutoffs["lower_to_intermediate"])
            upper_cutoff = float(risk_cutoffs["intermediate_to_higher"])

            if risk_score < lower_cutoff:
                risk_level = "Lower"
                css_class = "lower-risk"
                group_text = "Lower third of training-cohort risk scores"
            elif risk_score < upper_cutoff:
                risk_level = "Intermediate"
                css_class = "intermediate-risk"
                group_text = "Middle third of training-cohort risk scores"
            else:
                risk_level = "Higher"
                css_class = "higher-risk"
                group_text = "Upper third of training-cohort risk scores"

            probabilities = {
                "1-Year Overall Survival": survival_probability_at(model, input_array, 12.0),
                "3-Year Overall Survival": survival_probability_at(model, input_array, 36.0),
                "5-Year Overall Survival": survival_probability_at(model, input_array, 60.0),
            }
            survival_html = "".join(
                f"<p style='font-size:1rem; color:#334155;'>{label}: {value * 100:.1f}%</p>"
                for label, value in probabilities.items()
            )

            st.markdown(
                f"""
                <div class="prediction-box {css_class}">
                    <h2 style="color:black; margin-bottom:0.5rem;">Risk Score: {risk_score:.3f}</h2>
                    <h3 style="color:black; margin-bottom:0.4rem;">Relative Risk Group: {risk_level}</h3>
                    <p style="font-size:1rem; color:#334155;">{group_text}</p>
                    {survival_html}
                </div>
                """,
                unsafe_allow_html=True,
            )

            with st.expander("Encoded Model Input"):
                st.dataframe(
                    display_df.rename(columns=FEATURE_LABELS),
                    hide_index=True,
                    use_container_width=True,
                )
        except Exception as error:
            st.error(f"Prediction failed: {error}")

    with explanation_column:
        st.subheader("Feature Attribution")
        try:
            with st.spinner("Calculating feature contributions"):
                shap_values = explainer.shap_values(input_array, nsamples=200)
                shap_vector = extract_shap_vector(shap_values)
                expected_value = extract_expected_value(explainer.expected_value)

                try:
                    force_plot = shap.force_plot(
                        expected_value,
                        shap_vector,
                        display_df.iloc[0].to_numpy(dtype=float),
                        feature_names=feature_names,
                        matplotlib=False,
                    )
                    components.html(
                        shap.getjs() + force_plot.html(),
                        height=330,
                        scrolling=True,
                    )
                except Exception:
                    attribution = pd.DataFrame(
                        {"Feature": feature_names, "SHAP Value": shap_vector}
                    )
                    attribution = attribution.reindex(
                        attribution["SHAP Value"].abs().sort_values(ascending=False).index
                    ).head(10)
                    st.warning("Force plot unavailable; feature contributions are shown below.")
                    st.bar_chart(attribution.set_index("Feature")["SHAP Value"])

                st.markdown(
                    "Positive SHAP values increase the model risk score; negative values decrease it. "
                    "The absolute value represents the magnitude of the contribution."
                )
        except Exception as error:
            st.error(f"Feature attribution failed: {error}")


st.markdown("---")
st.markdown(
    "<div style='text-align:center; color:#667085;'>Random survival forest model with SHAP interpretation</div>",
    unsafe_allow_html=True,
)
