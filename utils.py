import streamlit as st
import numpy as np
import pickle
import shap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from io import BytesIO
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from datetime import datetime
from reportlab.lib import colors

import pandas as pd
import dice_ml
from dice_ml import Dice
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# Feature lists
# ─────────────────────────────────────────────
FEATURES_DS3 = [
    "Age",
    "Systolic BP",
    "Diastolic",
    "BS",
    "Body Temp",
    "BMI",
    "Previous Complications",
    "Preexisting Diabetes",
    "Gestational Diabetes",
    "Mental Health",
    "Heart Rate",
]

DICE_ACTIONABLE_DS3 = [
    "Systolic BP",
    "Diastolic",
    "BS",
    "Body Temp",
    "BMI",
    "Mental Health",
    "Heart Rate",
]

# ─────────────────────────────────────────────
# Clinical advice per feature
# ─────────────────────────────────────────────
FEATURE_ADVICE_DS3 = {
    "Systolic BP": (
        "💓 Systolic Blood Pressure",
        "Elevated systolic BP is a significant maternal risk factor. Adopt a low-sodium diet, "
        "limit caffeine, and practice relaxation techniques such as deep breathing or meditation. "
        "Regular, moderate exercise (as advised by your doctor) helps. Take prescribed "
        "antihypertensive medication consistently and attend all follow-up appointments."
    ),
    "Diastolic": (
        "💓 Diastolic Blood Pressure",
        "High diastolic BP needs prompt attention. Reduce salt and processed food intake. "
        "Ensure adequate rest and avoid prolonged standing. Report any persistent headaches, "
        "visual disturbances, or oedema to your healthcare provider immediately. "
        "Medication compliance is important — do not stop prescribed drugs without advice."
    ),
    "BS": (
        "🍬 Blood Sugar Level (mmol/L)",
        "Your blood sugar is outside the normal range (4.0–7.8 mmol/L for pregnancy). "
        "Follow a low-glycaemic diet — reduce refined carbohydrates, sugary drinks, and white rice. "
        "Regular light activity (e.g., 30-minute walks) improves insulin sensitivity significantly. "
        "If prescribed insulin or oral hypoglycaemics, adhere strictly to the dosing schedule. "
        "Monitor glucose levels at home and keep a log to share with your doctor at each visit."
    ),
    "Body Temp": (
        "🌡️ Body Temperature",
        "Your body temperature is outside the normal range (97.0–99.0 °F). "
        "A raised temperature may indicate infection or fever, which poses risks during pregnancy. "
        "Stay well-hydrated and rest adequately. For fever, use paracetamol as directed — avoid aspirin or ibuprofen. "
        "Seek immediate medical attention if temperature exceeds 100.4 °F (38 °C) or is persistently subnormal."
    ),
    "BMI": (
        "⚖️ BMI — Body Mass Index",
        "An abnormal BMI (high or low) increases maternal risk. If overweight: focus on a nutrient-dense, "
        "calorie-controlled diet and regular safe exercise — aim for gradual improvement, not rapid loss. "
        "If underweight: increase intake of protein, healthy fats, and complex carbohydrates. "
        "Work with a dietitian to create a personalised plan that supports both mother and baby."
    ),
    "Mental Health": (
        "🧠 Mental Health & Emotional Wellbeing",
        "Mental health is as important as physical health during pregnancy. If experiencing anxiety, "
        "depression, or emotional distress: speak openly with your doctor or midwife. "
        "Consider professional counselling or cognitive behavioural therapy (CBT). "
        "Build a support network of trusted friends and family. Mindfulness, gentle yoga, "
        "and adequate sleep significantly improve mental wellbeing. You are not alone — help is available."
    ),
    "Heart Rate": (
        "❤️ Heart Rate",
        "An abnormal resting heart rate (too high or too low) warrants monitoring. "
        "For elevated heart rate: avoid caffeine and stimulants, practise slow deep breathing, "
        "and reduce physical or emotional stress. For low heart rate: ensure adequate hydration and nutrition. "
        "If palpitations, dizziness, or chest discomfort occur, seek medical review promptly."
    ),
    "Age": (
        "👩 Maternal Age",
        "Maternal age is a non-modifiable but important risk factor. Younger mothers (<18) and older mothers (>35) "
        "benefit from closer monitoring. Ensure more frequent antenatal visits, detailed screening, "
        "and open communication with your obstetric team about age-related risks."
    ),
    "Previous Complications": (
        "📋 Previous Pregnancy Complications",
        "A history of complications significantly increases current pregnancy risk. "
        "Share full obstetric history with your healthcare provider. More frequent monitoring, "
        "specialist referrals, and early intervention planning are strongly recommended."
    ),
    "Preexisting Diabetes": (
        "🩸 Preexisting Diabetes",
        "Preexisting diabetes requires careful management during pregnancy. Maintain strict blood sugar control, "
        "attend all diabetic clinic appointments, and ensure your insulin or medication is adjusted as pregnancy progresses. "
        "Work closely with both your obstetrician and endocrinologist."
    ),
    "Gestational Diabetes": (
        "🍬 Gestational Diabetes",
        "Gestational diabetes increases risk for both mother and baby. Follow a low-glycaemic diet, "
        "monitor blood sugar levels regularly, and attend all follow-up appointments. "
        "In many cases it resolves after delivery, but close monitoring continues postpartum."
    ),
}

# ─────────────────────────────────────────────
# Normal ranges
# ─────────────────────────────────────────────
NORMAL_RANGES_DS3 = {
    "Systolic BP":  (110.0, 130.0),
    "Diastolic":    (70.0, 90.0),
    "BS":           (3.9, 7.8),
    "Body Temp":    (97.0, 99.0),
    "BMI":          (18.5, 28.0),
    "Mental Health": (0.0, 0.0),
    "Heart Rate":   (60.0, 90.0),
}

# Features that are binary flags (1 = issue present)
BINARY_RISK_FLAGS_DS3 = {
    "Previous Complications",
    "Preexisting Diabetes",
    "Gestational Diabetes",
    "Mental Health",
}


def get_flagged_features(input_data: dict, normal_ranges: dict) -> list:
    flagged = []
    for feat, (lo, hi) in normal_ranges.items():
        val = input_data.get(feat, None)
        if val is None:
            continue
        if lo == hi == 0.0:
            if float(val) > 0:
                flagged.append(feat)
        else:
            if float(val) < lo or float(val) > hi:
                flagged.append(feat)
    return flagged


def get_shap_driven_advice_features(shap_values, feature_names, input_data, top_n=3) -> list:
    """
    Return features to show advice for based on SHAP magnitude,
    even when values are within normal range.
    Only returns features that have advice defined.
    """
    idx_sorted = np.argsort(np.abs(shap_values))[::-1]
    features = []
    for i in idx_sorted:
        feat = feature_names[i]
        if feat in FEATURE_ADVICE_DS3:
            features.append(feat)
        if len(features) >= top_n:
            break
    return features


# ─────────────────────────────────────────────
# CSS — refreshed MaternalCare design
# ─────────────────────────────────────────────
def apply_global_css():
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@400;600;700&family=Nunito:wght@300;400;500;600;700&display=swap');

        html, body, [class*="css"] {
            font-family: 'Nunito', sans-serif;
        }

        /* ── background ── */
        .stApp {
            background: linear-gradient(160deg, #fdf6f0 0%, #fef9f5 40%, #f5f0fa 100%);
        }

        .main-title {
            font-family: 'Playfair Display', serif;
            font-size: 46px;
            font-weight: 700;
            background: linear-gradient(135deg, #b5467a 0%, #7c3aed 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            text-align: center;
            margin-top: 0.2rem;
            margin-bottom: 0.1rem;
            letter-spacing: -0.5px;
        }
        .main-subtitle {
            font-size: 15px;
            font-weight: 400;
            text-align: center;
            color: #8b7b8e;
            margin-bottom: 2rem;
            letter-spacing: 0.3px;
        }

        /* ── cards ── */
        .card {
            padding: 1.5rem 1.7rem;
            border-radius: 20px;
            border: 1px solid rgba(181, 70, 122, 0.12);
            background: rgba(255,255,255,0.85);
            backdrop-filter: blur(8px);
            box-shadow: 0 4px 24px rgba(181,70,122,0.07), 0 1px 4px rgba(0,0,0,0.04);
            margin-bottom: 1.2rem;
            transition: box-shadow 0.2s;
        }
        .card:hover {
            box-shadow: 0 8px 32px rgba(181,70,122,0.12), 0 2px 8px rgba(0,0,0,0.06);
        }
        .card-header {
            font-family: 'Playfair Display', serif;
            font-size: 17px;
            font-weight: 600;
            margin-bottom: 0.7rem;
            display: flex;
            align-items: center;
            gap: 0.5rem;
            color: #5b2d7a;
        }
        .card-header span.icon { font-size: 20px; }

        .result-card {
            padding: 1.8rem 2rem;
            border-radius: 22px;
            background: linear-gradient(135deg, #ffffff 0%, #fdf0f8 50%, #f5eeff 100%);
            border: 1px solid rgba(181, 70, 122, 0.18);
            box-shadow: 0 10px 32px rgba(181,70,122,0.10);
            margin-top: 1rem;
            margin-bottom: 1rem;
        }
        .result-title {
            font-family: 'Playfair Display', serif;
            font-size: 16px;
            font-weight: 600;
            margin-bottom: 0.7rem;
            color: #5b2d7a;
        }

        /* ── risk badges ── */
        .risk-badge {
            display: inline-block;
            padding: 0.4rem 1.3rem;
            border-radius: 999px;
            font-size: 12.5px;
            font-weight: 700;
            color: white;
            letter-spacing: 0.1em;
            text-transform: uppercase;
        }
        .risk-low      { background: linear-gradient(135deg, #0f9b6e, #27ae60); box-shadow: 0 4px 12px rgba(15,155,110,0.3); }
        .risk-moderate { background: linear-gradient(135deg, #e6a817, #d68910); box-shadow: 0 4px 12px rgba(230,168,23,0.3); }
        .risk-high     { background: linear-gradient(135deg, #c0392b, #e74c3c); box-shadow: 0 4px 12px rgba(192,57,43,0.3); }

        .risk-main-value {
            font-family: 'Playfair Display', serif;
            font-size: 32px;
            font-weight: 700;
            margin-top: 0.5rem;
        }
        .risk-subtext { font-size: 12.5px; color: #8b7b8e; margin-top: 0.3rem; }

        .section-caption { font-size: 13px; color: #9a8a9d; margin-bottom: 0.7rem; }

        .shap-card {
            padding: 0.8rem 1rem 0.5rem 1rem;
            border-radius: 16px;
            background: rgba(255,255,255,0.9);
            border: 1px solid rgba(181,70,122,0.1);
            box-shadow: 0 2px 12px rgba(0,0,0,0.04);
            margin-top: 0.5rem;
        }

        /* ── advice cards ── */
        .advice-card {
            padding: 1.2rem 1.5rem;
            border-radius: 16px;
            background: linear-gradient(135deg, #fff9f5, #fff3ee);
            border-left: 5px solid #e67e22;
            margin-bottom: 1rem;
            box-shadow: 0 3px 12px rgba(230,126,34,0.09);
        }
        .advice-card.urgent {
            background: linear-gradient(135deg, #fff5f5, #fdecea);
            border-left-color: #c0392b;
            box-shadow: 0 3px 12px rgba(192,57,43,0.11);
        }
        .advice-card.shap-driven {
            background: linear-gradient(135deg, #f5f0ff, #ede8ff);
            border-left: 5px solid #7c3aed;
            box-shadow: 0 3px 12px rgba(124,58,237,0.09);
        }
        .advice-title {
            font-family: 'Playfair Display', serif;
            font-size: 15px;
            font-weight: 600;
            color: #3d1f5a;
            margin-bottom: 0.4rem;
        }
        .advice-dice {
            font-size: 12.5px;
            color: #4a3070;
            background: rgba(124,58,237,0.08);
            padding: 0.4rem 0.8rem;
            border-radius: 8px;
            margin-top: 0.5rem;
            display: inline-block;
            font-weight: 600;
        }
        .advice-source-tag {
            font-size: 10.5px;
            color: #9a8ab0;
            text-transform: uppercase;
            letter-spacing: 0.06em;
            margin-bottom: 0.3rem;
        }

        .dice-info {
            padding: 1.1rem 1.5rem;
            border-radius: 16px;
            background: linear-gradient(135deg, #eef4fb, #f8f0ff);
            border-left: 4px solid #7c3aed;
            margin-bottom: 1rem;
            font-size: 13.5px;
            color: #2d1a4b;
        }

        /* ── section headers ── */
        h3 {
            font-family: 'Playfair Display', serif !important;
            color: #5b2d7a !important;
        }

        /* ── tabs ── */
        div[data-testid="stTabs"] button {
            font-family: 'Nunito', sans-serif;
            font-size: 13.5px;
            font-weight: 600;
        }

        /* ── buttons ── */
        .stButton > button {
            font-family: 'Nunito', sans-serif;
            font-weight: 700;
            border-radius: 12px;
            letter-spacing: 0.03em;
            background: linear-gradient(135deg, #b5467a, #7c3aed);
            color: white;
            border: none;
            transition: opacity 0.2s, transform 0.15s;
        }
        .stButton > button:hover {
            opacity: 0.92;
            transform: translateY(-1px);
        }

        /* ── expander ── */
        .streamlit-expanderHeader {
            font-family: 'Nunito', sans-serif;
            font-weight: 600;
        }

        /* ── metric ── */
        [data-testid="metric-container"] {
            background: rgba(255,255,255,0.7);
            border-radius: 12px;
            padding: 0.6rem;
            border: 1px solid rgba(181,70,122,0.1);
        }

        /* ── hero strip ── */
        .hero-strip {
            background: linear-gradient(135deg, #b5467a 0%, #7c3aed 100%);
            border-radius: 22px;
            padding: 2rem 2.5rem;
            margin-bottom: 1.5rem;
            color: white;
            box-shadow: 0 8px 32px rgba(181,70,122,0.25);
        }
        .hero-strip .hero-title {
            font-family: 'Playfair Display', serif;
            font-size: 36px;
            font-weight: 700;
            margin-bottom: 0.3rem;
        }
        .hero-strip .hero-sub {
            font-size: 14px;
            opacity: 0.88;
        }

        /* ── feature pill ── */
        .feature-pill {
            display: inline-block;
            padding: 0.25rem 0.8rem;
            border-radius: 999px;
            background: rgba(181,70,122,0.1);
            color: #b5467a;
            font-size: 12px;
            font-weight: 600;
            margin: 0.2rem 0.15rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


# ─────────────────────────────────────────────
# Model loading
# ─────────────────────────────────────────────
@st.cache_resource
def load_model():
    with open("model-d3.pkl", "rb") as f:
        model = pickle.load(f)
    return model


@st.cache_resource(show_spinner=False)
def load_train_data():
    try:
        with open("train_df-d3.pkl", "rb") as f:
            df = pickle.load(f)
        X = df[FEATURES_DS3].values
        y = df["Risk Level"].values
        return X, y
    except Exception:
        return None, None


# ─────────────────────────────────────────────
# SHAP helpers
# ─────────────────────────────────────────────
def get_shap_values(model, x_array, predicted_class_index=None):
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(x_array)

    if isinstance(shap_values, list):
        if predicted_class_index is None:
            predicted_class_index = 0
        shap_instance = shap_values[predicted_class_index][0]
        base_value = (
            explainer.expected_value[predicted_class_index]
            if isinstance(explainer.expected_value, (list, np.ndarray))
            else explainer.expected_value
        )
    else:
        shap_instance = shap_values[0]
        base_value = explainer.expected_value

    return shap_instance, base_value


def plot_shap_bar(shap_values, feature_names, title):
    idx_sorted = np.argsort(np.abs(shap_values))
    shap_sorted = shap_values[idx_sorted]
    feat_sorted = np.array(feature_names)[idx_sorted]

    fig, ax = plt.subplots(figsize=(7, max(4, len(feature_names) * 0.58)))
    fig.patch.set_facecolor('#fdf9fc')
    ax.set_facecolor('#fdf9fc')

    colors_bar = ["#c0392b" if v > 0 else "#0f9b6e" for v in shap_sorted]
    ax.barh(feat_sorted, shap_sorted, color=colors_bar, edgecolor="white", height=0.62)
    ax.axvline(0, color="#888", linewidth=0.8, linestyle="--", alpha=0.5)
    ax.set_xlabel("SHAP value  (→ increases risk   ← decreases risk)", fontsize=10)
    ax.set_title(title, fontsize=11, fontweight="bold", pad=12, color="#3d1f5a")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#ddd")
    ax.spines["bottom"].set_color("#ddd")
    ax.tick_params(axis="y", labelsize=9)
    plt.tight_layout()
    return fig


def plot_shap_waterfall(shap_values, base_value, x_row, feature_names, title):
    explanation = shap.Explanation(
        values=shap_values,
        base_values=base_value,
        data=x_row,
        feature_names=feature_names,
    )
    fig, ax = plt.subplots(figsize=(7, max(4, len(feature_names) * 0.6)))
    fig.patch.set_facecolor('#fdf9fc')
    shap.plots.waterfall(explanation, show=False)
    plt.title(title, fontsize=11, fontweight="bold", color="#3d1f5a")
    plt.tight_layout()
    return fig


# ─────────────────────────────────────────────
# Advice renderer — now integrated into DiCE tab
# ─────────────────────────────────────────────
def render_advice_section(
    input_data: dict,
    normal_ranges: dict,
    feature_advice: dict,
    shap_values=None,
    feature_names=None,
    risk_label: str = "",
    dice_deltas: dict = None,
):
    """
    Render clinical advice cards strictly based on patient input.

    Rules:
    - Flagged features (outside normal range) → always show advice.
    - Binary risk flags set to 1 by the patient (diabetes, complications etc.) → always show advice.
    - SHAP fallback → ONLY for High/Moderate risk AND ONLY for features where the patient
      actually has a measurable problem (value outside range OR binary flag set).
      Never shown for Low Risk or when every value is clinically normal.
    - Low Risk with no flagged features → no advice rendered at all.
    """
    flagged = get_flagged_features(input_data, normal_ranges)

    # Flag binary clinical risk factors that the patient has set to Yes/1
    for feat in BINARY_RISK_FLAGS_DS3:
        if feat in input_data and float(input_data[feat]) > 0 and feat not in flagged:
            flagged.append(feat)

    is_low_risk = "low" in risk_label.lower()
    is_high_risk = "high" in risk_label.lower()

    # SHAP fallback: only for High/Moderate risk, only when there are actual flagged
    # features to cross-reference — never surfaces phantom advice for clean patients.
    shap_features = []
    if not is_low_risk and is_high_risk and flagged and shap_values is not None and feature_names is not None:
        # From the top SHAP drivers, only keep ones that are also flagged
        # (i.e. the model is weighting something the patient genuinely has wrong)
        shap_features = get_shap_driven_advice_features(shap_values, feature_names, input_data, top_n=5)
        shap_features = [f for f in shap_features if f in flagged]
        # Avoid duplicating what's already in flagged
        shap_features = [f for f in shap_features if f not in flagged[:len(flagged)]]

    all_advice_features = flagged  # shap_features only added as styling signal, not extra items

    if not all_advice_features:
        # Nothing flagged — show a clean all-clear message instead of silence
        st.markdown(
            "<div style='background:linear-gradient(135deg,#f0fff8,#e8f8f0); border-radius:14px; "
            "padding:1rem 1.4rem; border-left:5px solid #27ae60; margin-bottom:1rem; text-align:center;'>"
            "<b style='color:#0f9b6e; font-size:14px;'>✅ All monitored parameters are within normal range.</b><br>"
            "<span style='font-size:12.5px; color:#2d6a4f;'>Continue routine care, healthy lifestyle habits, "
            "and attend all scheduled appointments.</span>"
            "</div>",
            unsafe_allow_html=True,
        )
        return

    # Header banner based on risk level
    if is_high_risk:
        st.markdown(
            "<div style='background:linear-gradient(135deg,#fdecea,#fff5f5); border-radius:14px; "
            "padding:1rem 1.4rem; border-left:5px solid #c0392b; margin-bottom:1rem;'>"
            "<b style='color:#c0392b; font-size:14px;'>🚨 High Risk — Urgent Clinical Evaluation Recommended</b><br>"
            "<span style='font-size:13px; color:#5a2020;'>"
            "The following flagged parameters are key areas of concern. "
            "Where DiCE targets are available, quantitative improvement goals are shown.</span>"
            "</div>",
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f"<p style='font-size:13px; color:#7a3a1a; font-weight:600; margin-bottom:0.8rem;'>"
            f"⚠️ <b>{len(flagged)} parameter(s)</b> flagged outside expected range — "
            f"personalised advice shown below.</p>",
            unsafe_allow_html=True,
        )

    urgent_features = {"Body Temp", "Preexisting Diabetes", "Gestational Diabetes"}

    for feat in all_advice_features:
        if feat not in feature_advice:
            continue
        title_str, advice_str = feature_advice[feat]
        is_urgent = feat in urgent_features

        card_class = "advice-card urgent" if is_urgent else "advice-card"
        source_tag = "<div class='advice-source-tag'>⚠️ Out-of-range / flagged parameter</div>"

        dice_hint = ""
        if dice_deltas and feat in dice_deltas:
            delta = dice_deltas[feat]
            direction = "increase" if delta > 0 else "decrease"
            arrow = "↑" if delta > 0 else "↓"
            dice_hint = (
                f"<div class='advice-dice'>"
                f"🎯 DiCE target: {arrow} {direction} <b>{feat.replace('_', ' ')}</b> by "
                f"<b>{abs(delta):.2f}</b> units to potentially shift risk category."
                f"</div>"
            )

        st.markdown(
            f"<div class='{card_class}'>"
            f"{source_tag}"
            f"<div class='advice-title'>{title_str}</div>"
            f"<p style='font-size:13px; color:#3d2c50; margin:0.2rem 0 0.5rem 0; line-height:1.6;'>{advice_str}</p>"
            f"{dice_hint}"
            f"</div>",
            unsafe_allow_html=True,
        )


# ─────────────────────────────────────────────
# DiCE helpers
# ─────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def get_dice_explainer(_model, feature_cols: tuple, actionable_features: tuple,
                        _X_train: np.ndarray, _y_train: np.ndarray):
    feature_cols = list(feature_cols)
    actionable_features = list(actionable_features)

    X_train_df = pd.DataFrame(_X_train, columns=feature_cols)
    y_arr = np.asarray(_y_train).ravel()

    label_map = None
    if y_arr.dtype.kind in {"U", "S", "O"}:
        unique = sorted(set(y_arr.tolist()))
        label_map = {lbl: i for i, lbl in enumerate(unique)}
        y_arr = np.array([label_map[v] for v in y_arr])

    train_df = X_train_df.copy()
    train_df["Risk Level"] = y_arr

    d = dice_ml.Data(
        dataframe=train_df,
        continuous_features=feature_cols,
        outcome_name="Risk Level",
    )
    m = dice_ml.Model(model=_model, backend="sklearn", model_type="classifier")

    try:
        exp = Dice(d, m, method="genetic")
    except Exception:
        exp = Dice(d, m, method="random")

    permitted_range = {
        "Systolic BP":  [110.0, 130.0],
        "Diastolic":    [70.0, 90.0],
        "BS":           [3.0,  20.0],
        "Body Temp":    [96.0, 104.0],
        "BMI":          [18.5, 28.0],
        "Mental Health":[0.0,  1.0],
        "Heart Rate":   [60.0, 100.0],
    }

    return exp, permitted_range, label_map


def generate_counterfactuals(exp, query_row, feature_cols, actionable_features,
                              permitted_range, label_map, model, n_cfs=3):
    query_df = pd.DataFrame(query_row.reshape(1, -1), columns=feature_cols)
    try:
        cf = exp.generate_counterfactuals(
            query_df,
            total_CFs=n_cfs,
            desired_class="opposite",
            features_to_vary=actionable_features,
            permitted_range=permitted_range,
        )

        if not cf.cf_examples_list or cf.cf_examples_list[0].final_cfs_df is None:
            return None, None, "DiCE could not find valid counterfactuals for this instance."

        cf_df = cf.cf_examples_list[0].final_cfs_df.copy()
        cf_df = cf_df.drop(columns=["Risk Level"], errors="ignore")
        cf_df = cf_df[[c for c in feature_cols if c in cf_df.columns]]

        delta_df = cf_df.copy()
        for col in actionable_features:
            if col in cf_df.columns:
                delta_df[col] = cf_df[col].astype(float) - float(query_df[col].values[0])

        return cf_df, delta_df[actionable_features], None

    except Exception as e:
        return None, None, str(e)


def get_first_cf_deltas(delta_df, actionable_features):
    if delta_df is None or delta_df.empty:
        return {}
    row = delta_df.iloc[0]
    return {feat: float(row[feat]) for feat in actionable_features if feat in row}


def plot_dice_heatmap(delta_df, actionable_features, title):
    data = delta_df[actionable_features].values.astype(float)
    n_cfs, n_feats = data.shape
    vmax = np.abs(data).max()
    if vmax == 0:
        vmax = 1.0

    fig, ax = plt.subplots(figsize=(max(7, n_feats * 1.2), max(3, n_cfs * 1.2)))
    fig.patch.set_facecolor('#fdf9fc')
    ax.set_facecolor('#fdf9fc')

    im = ax.imshow(data, cmap="RdYlGn_r", aspect="auto", vmin=-vmax, vmax=vmax)
    ax.set_xticks(range(n_feats))
    ax.set_xticklabels(
        [f.replace("_", " ") for f in actionable_features],
        rotation=35, ha="right", fontsize=9
    )
    ax.set_yticks(range(n_cfs))
    ax.set_yticklabels([f"Scenario {i+1}" for i in range(n_cfs)], fontsize=9)
    ax.set_title(title, fontsize=11, fontweight="bold", pad=10, color="#3d1f5a")

    for r in range(n_cfs):
        for c_ in range(n_feats):
            val = data[r, c_]
            txt = f"{val:+.2f}" if val != 0 else "—"
            ax.text(c_, r, txt, ha="center", va="center", fontsize=8,
                    color="black", fontweight="bold")

    plt.colorbar(im, ax=ax, shrink=0.75, label="Change from original")
    plt.tight_layout()
    return fig


def render_dice_tab(model, x_row, feature_cols, actionable_features,
                    X_train, y_train, predicted_label,
                    input_data=None, normal_ranges=None,
                    shap_values=None, feature_names=None):
    """
    Renders DiCE counterfactuals AND the integrated advice section below.
    For Low Risk predictions, DiCE is skipped — only advice is shown if any
    parameters are actually flagged.
    """
    is_low_risk = "low" in predicted_label.lower()

    # ── Low Risk: skip DiCE entirely ──────────────────────────────────────────
    if is_low_risk:
        st.markdown(
            "<div style='background:linear-gradient(135deg,#f0fff8,#e8f8f0); border-radius:14px; "
            "padding:1.1rem 1.5rem; border-left:5px solid #27ae60; margin-bottom:1rem;'>"
            "<b style='color:#0f9b6e; font-size:14px;'>✅ Low Risk — Counterfactual Analysis Not Required</b><br>"
            "<span style='font-size:13px; color:#2d6a4f;'>"
            "DiCE counterfactuals are designed to identify changes that flip a high or moderate risk prediction. "
            "Since the model predicts <b>Low Risk</b>, no what-if scenarios are generated. "
            "Continue routine care and monitoring.</span>"
            "</div>",
            unsafe_allow_html=True,
        )
        # Still show advice if any parameters are genuinely flagged
        if input_data is not None and normal_ranges is not None:
            st.markdown("### 🩹 Clinical Advice")
            render_advice_section(
                input_data=input_data,
                normal_ranges=normal_ranges,
                feature_advice=FEATURE_ADVICE_DS3,
                shap_values=None,        # No SHAP fallback for Low Risk
                feature_names=None,
                risk_label=predicted_label,
                dice_deltas=None,
            )
        return None, None

    # ── High / Moderate Risk: run DiCE ───────────────────────────────────────
    st.markdown(
        "<div class='dice-info'>"
        "<b>💡 What is a Counterfactual?</b><br>"
        "A counterfactual shows the <em>minimum changes</em> to modifiable features "
        "that would flip the model's prediction to the opposite risk class. "
        "Think of it as a concrete <b>action plan</b>: "
        "<em>'If these values changed by this much, your estimated risk category would differ.'</em>"
        "</div>",
        unsafe_allow_html=True,
    )

    st.markdown(
        f"**Current prediction:** `{predicted_label}` &nbsp;|&nbsp; "
        f"**Modifiable features:** {', '.join([f.replace('_', ' ') for f in actionable_features])}",
        unsafe_allow_html=True,
    )

    with st.spinner("Generating counterfactuals… (first run may take ~15 s)"):
        try:
            exp, permitted_range, label_map = get_dice_explainer(
                model,
                tuple(feature_cols),
                tuple(actionable_features),
                X_train,
                y_train,
            )
        except Exception as e:
            st.error(f"Could not build DiCE explainer: {e}")
            return None, None

    cf_df, delta_df, err = generate_counterfactuals(
        exp, x_row, feature_cols, actionable_features,
        permitted_range, label_map, model,
    )

    dice_deltas = {}

    if err:
        st.warning(f"⚠️ {err}")
        st.info("Try adjusting inputs and predicting again.")
    else:
        st.markdown("#### 🔥 Counterfactual Change Heatmap")
        st.caption(
            "Each row is one alternative scenario. "
            "**Red** = feature needs to increase · **Green** = feature needs to decrease · "
            "**—** = no change needed."
        )
        fig = plot_dice_heatmap(delta_df, actionable_features, "Required changes per scenario")
        st.pyplot(fig)
        plt.close(fig)

        st.markdown("#### 📋 Scenario Details")
        orig_df = pd.DataFrame(x_row.reshape(1, -1), columns=feature_cols)
        for i, (_, cf_row) in enumerate(cf_df.iterrows()):
            with st.expander(f"Scenario {i+1} — What needs to change?", expanded=(i == 0)):
                rows = []
                for feat in actionable_features:
                    orig_val = float(orig_df[feat].values[0])
                    new_val = float(cf_row[feat])
                    delta = new_val - orig_val
                    arrow = "⬆️" if delta > 0.005 else ("⬇️" if delta < -0.005 else "➡️")
                    rows.append({
                        "Feature": feat.replace("_", " "),
                        "Current Value": f"{orig_val:.2f}",
                        "Suggested Value": f"{new_val:.2f}",
                        "Change": f"{arrow} {delta:+.2f}",
                    })
                st.dataframe(pd.DataFrame(rows).set_index("Feature"), use_container_width=True)

        st.caption("⚠️ These are model-derived suggestions only. Always apply clinical judgement.")

        dice_deltas = get_first_cf_deltas(delta_df, actionable_features)

    # ── Integrated Clinical Advice ──
    if input_data is not None and normal_ranges is not None:
        st.markdown("---")
        st.markdown("### 🩹 Personalised Clinical Advice")
        st.markdown(
            "<p class='section-caption'>"
            "Advice is generated for flagged parameters and model-identified risk drivers. "
            "Where available, DiCE-derived quantitative targets are shown inline.</p>",
            unsafe_allow_html=True,
        )
        render_advice_section(
            input_data=input_data,
            normal_ranges=normal_ranges,
            feature_advice=FEATURE_ADVICE_DS3,
            shap_values=shap_values,
            feature_names=feature_names,
            risk_label=predicted_label,
            dice_deltas=dice_deltas,
        )

    return delta_df, actionable_features


# ─────────────────────────────────────────────
# PDF report — includes full advice section
# ─────────────────────────────────────────────
def create_pdf_report(input_dict, pred_label, proba_dict=None,
                      shap_contribs=None, flagged_features=None,
                      feature_advice=None, dice_deltas=None,
                      shap_values=None, feature_names=None):
    buffer = BytesIO()
    c = canvas.Canvas(buffer, pagesize=A4)
    width, height = A4
    y = height - 50

    # ── Header ──
    c.setFillColor(colors.HexColor("#b5467a"))
    c.rect(0, height - 85, width, 85, fill=True, stroke=False)
    # accent strip
    c.setFillColor(colors.HexColor("#7c3aed"))
    c.rect(0, height - 85, 8, 85, fill=True, stroke=False)

    c.setFillColor(colors.white)
    c.setFont("Helvetica-Bold", 22)
    c.drawString(55, height - 42, "MaternalCare")
    c.setFont("Helvetica", 12)
    c.drawString(55, height - 60, "Maternal Risk Assessment Report")
    c.setFont("Helvetica", 9)
    c.drawString(55, height - 75, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    c.setFillColor(colors.black)

    y = height - 105
    c.setFont("Helvetica-Oblique", 8.5)
    c.setFillColor(colors.HexColor("#888888"))
    c.drawString(50, y, "This report is a clinical decision-support tool only. All findings must be reviewed by a qualified health professional.")
    c.setFillColor(colors.black)
    y -= 22

    lower_label = str(pred_label).lower()
    is_high = "high" in lower_label
    is_low = "low" in lower_label

    # ── 1. Risk Summary ──
    _pdf_section_header(c, "1. Risk Summary", y)
    y -= 20
    c.setFont("Helvetica-Bold", 13)
    if is_high:
        c.setFillColor(colors.HexColor("#c0392b"))
    elif is_low:
        c.setFillColor(colors.HexColor("#0f9b6e"))
    else:
        c.setFillColor(colors.HexColor("#e6a817"))
    c.drawString(60, y, f"Predicted Risk Level: {pred_label}")
    c.setFillColor(colors.black)
    y -= 16

    c.setFont("Helvetica", 10)
    if is_high:
        c.drawString(60, y, "The model suggests HIGH maternal risk.")
        y -= 14
        c.setFont("Helvetica-Bold", 10)
        c.setFillColor(colors.HexColor("#c0392b"))
        c.drawString(60, y, "URGENT: Please consult a qualified health professional as soon as possible.")
        c.setFillColor(colors.black)
        c.setFont("Helvetica", 10)
        y -= 14
    elif is_low:
        c.drawString(60, y, "The model suggests LOW maternal risk. Continue routine monitoring.")
        y -= 14
    else:
        c.drawString(60, y, "The model suggests MODERATE maternal risk. Close monitoring is advised.")
        y -= 14
    y -= 8

    # ── 2. Probabilities ──
    if proba_dict:
        y = _pdf_page_break(c, y, height)
        _pdf_section_header(c, "2. Class Probabilities", y)
        y -= 18
        c.setFont("Helvetica", 10)
        for cls, p in proba_dict.items():
            c.drawString(60, y, f"{cls}: {p:.3f}  ({p*100:.1f}%)")
            y -= 14
            y = _pdf_page_break(c, y, height)
    y -= 8

    # ── 3. Input Values ──
    y = _pdf_page_break(c, y, height)
    _pdf_section_header(c, "3. Input Values Used", y)
    y -= 18
    c.setFont("Helvetica", 10)
    for k, v in input_dict.items():
        c.drawString(60, y, f"{k.replace('_', ' ')}: {v}")
        y -= 14
        y = _pdf_page_break(c, y, height)
    y -= 8

    # ── 4. SHAP ──
    if shap_contribs:
        y = _pdf_page_break(c, y, height)
        _pdf_section_header(c, "4. Top Influencing Features (SHAP)", y)
        y -= 18
        c.setFont("Helvetica", 10)
        c.drawString(60, y, "Positive SHAP = increased risk   |   Negative SHAP = decreased risk")
        y -= 16
        for feat, val in shap_contribs:
            indicator = "▲" if val > 0 else "▼"
            c.drawString(60, y, f"{indicator}  {feat.replace('_', ' ')}: SHAP = {val:.4f}")
            y -= 14
            y = _pdf_page_break(c, y, height)
    y -= 8

    # ── 5. Clinical Advice ──
    # Only based on genuinely flagged parameters and binary flags set by the patient.
    # SHAP-driven fallback is intentionally excluded from the PDF to avoid phantom advice.
    all_advice_features = list(flagged_features) if flagged_features else []

    # Add binary clinical flags that the patient has set to Yes/1
    for feat in BINARY_RISK_FLAGS_DS3:
        if feat in input_dict and float(input_dict.get(feat, 0)) > 0 and feat not in all_advice_features:
            all_advice_features.append(feat)

    if all_advice_features and feature_advice:
        y = _pdf_page_break(c, y, height)
        _pdf_section_header(c, "5. Clinical Advice", y)
        y -= 18

        for feat in all_advice_features:
            if feat not in feature_advice:
                continue
            y = _pdf_page_break(c, y, height, min_y=130)
            title_str, advice_str = feature_advice[feat]
            clean_title = title_str.replace("**", "").replace("*", "")
            c.setFont("Helvetica-Bold", 10)
            c.setFillColor(colors.HexColor("#b5467a"))
            c.drawString(60, y, f"• {clean_title}")
            c.setFillColor(colors.black)
            y -= 14
            # x=70, right margin=40 → available width = width - 70 - 40
            y = _pdf_wrap_text(c, advice_str, 70, y, width - 110, height,
                               font_name="Helvetica", font_size=9)
            # DiCE hint — only for High/Moderate risk reports
            if not is_low and dice_deltas and feat in dice_deltas:
                delta = dice_deltas[feat]
                direction = "increase" if delta > 0 else "decrease"
                c.setFont("Helvetica-Oblique", 9)
                c.setFillColor(colors.HexColor("#7c3aed"))
                dice_hint_text = (f"DiCE target: {direction} {feat.replace('_',' ')} "
                                  f"by {abs(delta):.2f} units to potentially shift risk category.")
                y = _pdf_wrap_text(c, dice_hint_text, 70, y, width - 110, height,
                                   font_name="Helvetica-Oblique", font_size=9)
                c.setFillColor(colors.black)
            y -= 4
    elif is_low:
        # Low risk, no flagged params — note it explicitly in PDF
        y = _pdf_page_break(c, y, height)
        _pdf_section_header(c, "5. Clinical Advice", y)
        y -= 18
        c.setFont("Helvetica", 10)
        c.setFillColor(colors.HexColor("#0f9b6e"))
        c.drawString(60, y, "All monitored parameters are within normal range.")
        y -= 14
        c.setFillColor(colors.black)
        c.drawString(60, y, "Continue routine care, healthy lifestyle habits, and scheduled appointments.")
        y -= 14
    y -= 8

    # ── 6. DiCE Summary — skipped for Low Risk ──
    if not is_low and dice_deltas:
        y = _pdf_page_break(c, y, height)
        _pdf_section_header(c, "6. DiCE Counterfactual Targets (Scenario 1)", y)
        y -= 18
        c.setFont("Helvetica", 10)
        c.drawString(60, y, "Minimum changes to potentially shift risk classification:")
        y -= 14
        for feat, delta in dice_deltas.items():
            if abs(delta) < 0.001:
                continue
            arrow = "↑" if delta > 0 else "↓"
            c.drawString(70, y, f"{arrow}  {feat.replace('_',' ')}: {delta:+.3f}")
            y -= 13
            y = _pdf_page_break(c, y, height)
        y -= 8

    # ── 7. Notice ──
    y = _pdf_page_break(c, y, height, min_y=100)
    _pdf_section_header(c, "7. Important Notice", y)
    y -= 16
    notice1 = ("This report is generated by a machine learning model for use by trained "
               "health professionals only.")
    notice2 = ("It must not be used as the sole basis for any diagnosis or treatment decision.")
    y = _pdf_wrap_text(c, notice1, 60, y, width - 100, height, font_name="Helvetica", font_size=9)
    y = _pdf_wrap_text(c, notice2, 60, y, width - 100, height, font_name="Helvetica", font_size=9)

    c.showPage()
    c.save()
    buffer.seek(0)
    return buffer


def _pdf_section_header(c, text, y):
    c.setFont("Helvetica-Bold", 13)
    c.setFillColor(colors.HexColor("#b5467a"))
    c.drawString(50, y, text)
    c.setFillColor(colors.black)


def _pdf_page_break(c, y, height, min_y=80):
    if y < min_y:
        c.showPage()
        return height - 50
    return y


def _pdf_wrap_text(c, text, x, y, max_width_pts, page_height, font_name="Helvetica", font_size=9):
    """Wrap text to fit within max_width_pts using actual glyph-width measurement."""
    from reportlab.pdfbase.pdfmetrics import stringWidth
    c.setFont(font_name, font_size)
    words = text.split()
    line = ""
    line_height = font_size + 3

    for word in words:
        test_line = (line + " " + word).strip()
        if stringWidth(test_line, font_name, font_size) <= max_width_pts:
            line = test_line
        else:
            if line:
                c.drawString(x, y, line)
                y -= line_height
                if y < 80:
                    c.showPage()
                    y = page_height - 50
                    c.setFont(font_name, font_size)
            line = word
    if line:
        c.drawString(x, y, line)
        y -= line_height + 2
    return y


# ─────────────────────────────────────────────
# Label formatter
# ─────────────────────────────────────────────
def format_risk_label(raw_label: str) -> str:
    s = str(raw_label).strip()
    lower = s.lower()
    if lower in ["0", "low"]:
        return "Low Risk"
    if lower in ["1", "high"]:
        return "High Risk"
    if lower in ["2", "medium", "moderate"]:
        return "Moderate Risk"
    return s.capitalize()


def risk_color(label: str):
    label = label.lower()
    if "low" in label:  return "#0f9b6e"
    if "high" in label: return "#c0392b"
    return "#e6a817"