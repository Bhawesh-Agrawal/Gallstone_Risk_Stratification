import joblib
import json
import numpy as np
import pandas as pd
import shap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import base64
from io import BytesIO
import os
from utils.constant import RESULT_DIR

model = RESULT_DIR

def get_latest_model(base_name: str = "rf"):
    version = 1
    model_path = os.path.join(model, f"{base_name}_v{version}.pkl")

    # keep incrementing while file exists
    while os.path.exists(model_path):
        version += 1
        model_path = os.path.join(model, f"{base_name}_v{version}.pkl")

    # at this point version is at first missing
    # so latest existing is version - 1
    if version == 0:
        raise FileNotFoundError(f"No model found for base {base_name} in {model}")

    latest_version = version - 1
    latest_model_path = os.path.join(model, f"{base_name}_v{latest_version}.pkl")
    return latest_version


latest_version = get_latest_model()

MODEL_PATH = os.path.join(model, f"rf_v{latest_version}.pkl")
SCALER_PATH = model_path = os.path.join(model, f"scaler_v{latest_version}.pkl")
FEATURE_COLS_PATH = os.path.join(model, f"feature_cols_v{latest_version}.json")

model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)
with open(FEATURE_COLS_PATH, "r") as f:
    feature_cols = json.load(f)

numeric_cols = [
    'Height', 'Weight', 'Body Mass Index (BMI)', 'Total Body Water (TBW)',
    'Extracellular Water (ECW)', 'Intracellular Water (ICW)', 'Extracellular Fluid/Total Body Water (ECF/TBW)',
    'Total Body Fat Ratio (TBFR) (%)', 'Lean Mass (LM) (%)', 'Body Protein Content (Protein) (%)',
    'Visceral Fat Rating (VFR)', 'Bone Mass (BM)', 'Muscle Mass (MM)', 'Obesity (%)',
    'Total Fat Content (TFC)', 'Visceral Fat Area (VFA)', 'Visceral Muscle Area (VMA) (Kg)',
    'Hepatic Fat Accumulation (HFA)', 'Glucose', 'Total Cholesterol (TC)', 'Low Density Lipoprotein (LDL)',
    'High Density Lipoprotein (HDL)', 'Triglyceride', 'Aspartat Aminotransferaz (AST)',
    'Alanin Aminotransferaz (ALT)', 'Alkaline Phosphatase (ALP)', 'Creatinine',
    'Glomerular Filtration Rate (GFR)', 'C-Reactive Protein (CRP)', 'Hemoglobin (HGB)', 'Vitamin D'
]
categorical_cols = [
    "Age", "Gender", "Comorbidity", "Coronary Artery Disease (CAD)",
    "Hypothyroidism", "Hyperlipidemia", "Diabetes Mellitus (DM)"
]

def fig_to_base64(plt_figure) -> str:
    buf = BytesIO()
    plt_figure.savefig(buf, format= "png", bbox_inches = "tight")
    buf.seek(0)
    img_b64 = base64.b64encode(buf.read()).decode("utf-8")
    buf.close()
    return f"data:image/png;base64,{img_b64}"


def predict_gallstone(input_data: dict):
    df = pd.DataFrame([input_data])
    df = df[feature_cols]

    # Scale numeric columns
    df[numeric_cols] = scaler.transform(df[numeric_cols])

    # Prediction
    prediction_proba = model.predict_proba(df)[0]
    prediction = np.argmax(prediction_proba)

    # SHAP Explanation
    explainer = shap.TreeExplainer(model)
    shap_exp = explainer(df)

    # For binary classification → (1, n_features, 2)
    shap_values_class0 = shap_exp.values[0, :, 0]   # positive class (0)
    shap_values_class1 = shap_exp.values[0, :, 1]   # negative class (1)

    # Base values
    base_value_class0 = shap_exp.base_values[0, 0]
    base_value_class1 = shap_exp.base_values[0, 1]

    # DataFrames for sorting
    df_shap = pd.DataFrame({
        "feature": feature_cols,
        "shap_value_class_0": shap_values_class0,
        "shap_value_class_1": shap_values_class1
    })

    # Top contributors
    top_positive_contributors = df_shap.sort_values("shap_value_class_0", ascending=False).head(5)
    top_negative_contributors = df_shap.sort_values("shap_value_class_1", ascending=False).head(5)

    # ✅ Generate plots
    # Waterfall for single instance (positive class = 0)
    plt.figure()
    shap.plots.waterfall(shap_exp[0, :, 0], show=False)
    waterfall_b64 = fig_to_base64(plt)
    plt.close()

    # Bar summary (global view, positive class = 0)
    plt.figure()
    shap.plots.bar(shap_exp[:, :, 0], show=False)
    bar_b64 = fig_to_base64(plt)
    plt.close()

    # Build JSON output
    output = {
        "prediction_label": "Positive" if prediction == 0 else "Negative",
        "prediction_value": int(prediction),
        "probability": {
            "positive (class 0)": f"{prediction_proba[0]:.4f}",
            "negative (class 1)": f"{prediction_proba[1]:.4f}"
        },
        "shap_analysis": {
            "base_value_for_positive_class": float(base_value_class0),
            "base_value_for_negative_class": float(base_value_class1),
            "top_contributors_to_positive": {
                row.feature: f"{row.shap_value_class_0:.4f}" for _, row in top_positive_contributors.iterrows()
            },
            "top_contributors_to_negative": {
                row.feature: f"{row.shap_value_class_1:.4f}" for _, row in top_negative_contributors.iterrows()
            },
            "plots": {
                "waterfall_plot": waterfall_b64,
                "bar_summary_plot": bar_b64
            }
        }
    }

    return output



