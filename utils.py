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
    "Systolic BP":   (110.0, 130.0),
    "Diastolic":     (70.0,  90.0),
    "BS":            (3.9,   7.8),
    "Body Temp":     (97.0,  99.0),
    "BMI":           (18.5,  28.0),
    "Mental Health": (0.0,   0.0),
    "Heart Rate":    (60.0,  90.0),
}

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
# CSS  — complete rewrite for reliable dark mode
# Strategy: use CSS custom properties defined
# under both :root (light) and a dark-mode block.
# Streamlit 1.x sets  data-theme="dark"  on <html>.
# ALL hardcoded colours in HTML strings are
# replaced with var(--xxx) references so they
# respond automatically.
# ─────────────────────────────────────────────
def apply_global_css():
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@400;600;700&family=Nunito:wght@300;400;500;600;700&display=swap');

        /* ══════════════════════════════════════════════
           LIGHT MODE — CSS custom properties
        ══════════════════════════════════════════════ */
        :root {
            --bg-page:            linear-gradient(160deg,#fdf6f0 0%,#fef9f5 40%,#f5f0fa 100%);
            --bg-card:            rgba(255,255,255,0.92);
            --bg-card-solid:      #ffffff;
            --bg-result-card:     linear-gradient(135deg,#ffffff 0%,#fdf0f8 50%,#f5eeff 100%);
            --bg-shap-card:       rgba(255,255,255,0.9);
            --bg-advice:          linear-gradient(135deg,#fff9f5,#fff3ee);
            --bg-advice-urgent:   linear-gradient(135deg,#fff5f5,#fdecea);
            --bg-advice-shap:     linear-gradient(135deg,#f5f0ff,#ede8ff);
            --bg-dice-info:       linear-gradient(135deg,#eef4fb,#f8f0ff);
            --bg-hero:            linear-gradient(135deg,#b5467a 0%,#7c3aed 100%);
            --bg-pill:            rgba(181,70,122,0.10);
            --bg-proba-track:     #f0e8f5;
            --bg-metric:          rgba(255,255,255,0.7);
            --bg-allclear:        linear-gradient(135deg,#f0fff8,#e8f8f0);
            --bg-highrisk:        linear-gradient(135deg,#fdecea,#fff5f5);
            --bg-moderate:        linear-gradient(135deg,#fff9f5,#fff3ee);

            --text-primary:       #3d2c50;
            --text-secondary:     #5a4a6a;
            --text-caption:       #7a6a8a;
            --text-card-header:   #5b2d7a;
            --text-advice-body:   #3d2c50;
            --text-advice-title:  #3d1f5a;
            --text-dice-info:     #2d1a4b;
            --text-range:         #8a6a9a;
            --text-proba-label:   #6a5a7a;
            --text-shap-driver:   #5b2d7a;
            --text-shap-hint:     #7a6a8a;
            --text-allclear:      #0f9b6e;
            --text-allclear-sub:  #1a6040;
            --text-highrisk:      #c0392b;
            --text-highrisk-sub:  #5a1010;
            --text-moderate-sub:  #5a2a10;

            --border-card:        rgba(181,70,122,0.15);
            --border-result:      rgba(181,70,122,0.20);
            --border-shap:        rgba(181,70,122,0.12);
            --border-metric:      rgba(181,70,122,0.12);

            --shadow-card:        0 4px 24px rgba(181,70,122,0.08),0 1px 4px rgba(0,0,0,0.04);
            --shadow-card-hov:    0 8px 32px rgba(181,70,122,0.14),0 2px 8px rgba(0,0,0,0.06);
            --shadow-result:      0 10px 32px rgba(181,70,122,0.10);
            --shadow-shap:        0 2px 12px rgba(0,0,0,0.04);
            --shadow-advice:      0 3px 12px rgba(230,126,34,0.09);
            --shadow-advice-urg:  0 3px 12px rgba(192,57,43,0.11);
            --shadow-advice-shp:  0 3px 12px rgba(124,58,237,0.09);

            --color-accent:       #b5467a;
            --color-purple:       #7c3aed;
            --color-purple-mid:   #5b2d7a;
            --color-green:        #0f9b6e;
            --color-red:          #c0392b;
            --color-amber:        #e6a817;

            --advice-dice-color:  #4a3070;
            --advice-dice-bg:     rgba(124,58,237,0.08);
            --dice-em-color:      #3a2060;
            --dice-b-color:       #3a2060;
        }

        /* ══════════════════════════════════════════════
           DARK MODE — override every custom property
           Streamlit sets  data-theme="dark"  on <html>
        ══════════════════════════════════════════════ */
        html[data-theme="dark"] {
            --bg-page:            linear-gradient(160deg,#0f0a18 0%,#130d20 40%,#0d0a1a 100%);
            --bg-card:            rgba(36,24,54,0.97);
            --bg-card-solid:      #221638;
            --bg-result-card:     linear-gradient(135deg,#1c1228 0%,#271335 50%,#191028 100%);
            --bg-shap-card:       rgba(36,24,54,0.95);
            --bg-advice:          linear-gradient(135deg,#2a1a0e,#2e1c10);
            --bg-advice-urgent:   linear-gradient(135deg,#2a1010,#300e0e);
            --bg-advice-shap:     linear-gradient(135deg,#1a1230,#201540);
            --bg-dice-info:       linear-gradient(135deg,#0e1420,#180f2e);
            --bg-hero:            linear-gradient(135deg,#8c2d58 0%,#5b28b8 100%);
            --bg-pill:            rgba(181,70,122,0.25);
            --bg-proba-track:     rgba(181,70,122,0.20);
            --bg-metric:          rgba(36,24,54,0.85);
            --bg-allclear:        linear-gradient(135deg,#0a2018,#0d2a1e);
            --bg-highrisk:        linear-gradient(135deg,#2a0a0a,#320e0e);
            --bg-moderate:        linear-gradient(135deg,#2a1508,#321808);

            --text-primary:       #ecddf8;
            --text-secondary:     #c8b8d8;
            --text-caption:       #b8a8c8;
            --text-card-header:   #ddb8f5;
            --text-advice-body:   #ddd0ec;
            --text-advice-title:  #eac8ff;
            --text-dice-info:     #cdbae8;
            --text-range:         #cc9ee0;
            --text-proba-label:   #c0b0d0;
            --text-shap-driver:   #ddb8f5;
            --text-shap-hint:     #b8a8c8;
            --text-allclear:      #4ecfa0;
            --text-allclear-sub:  #80e0b8;
            --text-highrisk:      #f07070;
            --text-highrisk-sub:  #f0a0a0;
            --text-moderate-sub:  #f0c090;

            --border-card:        rgba(181,70,122,0.35);
            --border-result:      rgba(181,70,122,0.40);
            --border-shap:        rgba(181,70,122,0.28);
            --border-metric:      rgba(181,70,122,0.28);

            --shadow-card:        0 4px 24px rgba(0,0,0,0.50),0 1px 4px rgba(0,0,0,0.35);
            --shadow-card-hov:    0 8px 32px rgba(0,0,0,0.65),0 2px 8px rgba(0,0,0,0.40);
            --shadow-result:      0 10px 32px rgba(0,0,0,0.50);
            --shadow-shap:        0 2px 12px rgba(0,0,0,0.40);
            --shadow-advice:      0 3px 12px rgba(0,0,0,0.35);
            --shadow-advice-urg:  0 3px 12px rgba(0,0,0,0.40);
            --shadow-advice-shp:  0 3px 12px rgba(0,0,0,0.35);

            --advice-dice-color:  #caa8f8;
            --advice-dice-bg:     rgba(124,58,237,0.30);
            --dice-em-color:      #c8a8f0;
            --dice-b-color:       #d8b8ff;
        }

        /* ══════════════════════════════════════════════
           BASE
        ══════════════════════════════════════════════ */
        html, body, [class*="css"] {
            font-family: 'Nunito', sans-serif;
        }
        .stApp {
            background: var(--bg-page) !important;
        }

        /* ══════════════════════════════════════════════
           STREAMLIT NATIVE WIDGET TEXT
           Must use !important — Streamlit injects its
           own colour rules at high specificity.
        ══════════════════════════════════════════════ */
        /* All labels / helper text */
        label, .stRadio label, .stNumberInput label,
        [data-testid="stWidgetLabel"] p,
        [data-testid="stWidgetLabel"] span,
        [data-baseweb="radio"] span,
        p, span, div {
            /* Don't override here globally — do it per-component below */
        }

        /* Number-input label */
        [data-testid="stNumberInput"] > label {
            color: var(--text-primary) !important;
        }
        [data-testid="stNumberInput"] > label p,
        [data-testid="stNumberInput"] > label span {
            color: var(--text-primary) !important;
        }
        /* Number-input box */
        [data-testid="stNumberInput"] input {
            color: var(--text-primary) !important;
            background-color: var(--bg-card-solid) !important;
            border-color: var(--border-card) !important;
        }
        /* Radio label */
        [data-testid="stRadio"] > label {
            color: var(--text-primary) !important;
        }
        [data-testid="stRadio"] > label p,
        [data-testid="stRadio"] > label span {
            color: var(--text-primary) !important;
        }
        /* Radio option text */
        [data-baseweb="radio"] p,
        [data-baseweb="radio"] span,
        [data-baseweb="radio"] label {
            color: var(--text-primary) !important;
        }
        /* Generic widget label catch-all */
        [data-testid="stWidgetLabel"] p {
            color: var(--text-primary) !important;
        }
        /* Markdown text */
        [data-testid="stMarkdownContainer"] p,
        [data-testid="stMarkdownContainer"] li,
        [data-testid="stMarkdownContainer"] span {
            color: var(--text-primary) !important;
        }
        /* Caption */
        [data-testid="stCaptionContainer"] p {
            color: var(--text-caption) !important;
        }

        /* ══════════════════════════════════════════════
           CARD  (custom HTML divs)
        ══════════════════════════════════════════════ */
        .card {
            padding: 1.5rem 1.7rem;
            border-radius: 20px;
            border: 1px solid var(--border-card);
            background: var(--bg-card) !important;
            backdrop-filter: blur(8px);
            box-shadow: var(--shadow-card);
            margin-bottom: 1.2rem;
            transition: box-shadow 0.2s;
        }
        .card:hover { box-shadow: var(--shadow-card-hov); }

        .card-header {
            font-family: 'Playfair Display', serif;
            font-size: 17px;
            font-weight: 600;
            margin-bottom: 0.7rem;
            display: flex;
            align-items: center;
            gap: 0.5rem;
            color: var(--text-card-header) !important;
        }
        .card-header span.icon { font-size: 20px; }

        .card-body {
            font-size: 13.5px;
            color: var(--text-primary) !important;
            line-height: 1.65;
            margin: 0;
        }
        .card-body strong, .card-body b { color: var(--text-card-header) !important; }
        .card-body em                   { color: var(--text-secondary) !important; }

        /* catch all <p> directly inside a card */
        .card p {
            color: var(--text-primary) !important;
        }
        .card b, .card strong {
            color: var(--text-card-header) !important;
        }

        /* ══════════════════════════════════════════════
           RESULT CARD
        ══════════════════════════════════════════════ */
        .result-card {
            padding: 1.8rem 2rem;
            border-radius: 22px;
            background: var(--bg-result-card) !important;
            border: 1px solid var(--border-result);
            box-shadow: var(--shadow-result);
            margin-top: 1rem;
            margin-bottom: 1rem;
        }
        .result-title {
            font-family: 'Playfair Display', serif;
            font-size: 16px;
            font-weight: 600;
            margin-bottom: 0.7rem;
            color: var(--text-card-header) !important;
        }

        /* ══════════════════════════════════════════════
           RISK BADGES
        ══════════════════════════════════════════════ */
        .risk-badge {
            display: inline-block;
            padding: 0.4rem 1.3rem;
            border-radius: 999px;
            font-size: 12.5px;
            font-weight: 700;
            color: white !important;
            letter-spacing: 0.1em;
            text-transform: uppercase;
        }
        .risk-low      { background: linear-gradient(135deg,#0f9b6e,#27ae60); box-shadow: 0 4px 12px rgba(15,155,110,0.3); }
        .risk-moderate { background: linear-gradient(135deg,#e6a817,#d68910); box-shadow: 0 4px 12px rgba(230,168,23,0.3); }
        .risk-high     { background: linear-gradient(135deg,#c0392b,#e74c3c); box-shadow: 0 4px 12px rgba(192,57,43,0.3); }

        .risk-main-value {
            font-family: 'Playfair Display', serif;
            font-size: 32px;
            font-weight: 700;
            margin-top: 0.5rem;
        }
        .risk-subtext {
            font-size: 12.5px;
            color: var(--text-secondary) !important;
            margin-top: 0.3rem;
        }

        /* ══════════════════════════════════════════════
           TYPOGRAPHY HELPERS
        ══════════════════════════════════════════════ */
        .main-title {
            font-family: 'Playfair Display', serif;
            font-size: 46px;
            font-weight: 700;
            background: linear-gradient(135deg,#b5467a 0%,#7c3aed 100%);
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
            text-align: center;
            color: var(--text-secondary) !important;
            margin-bottom: 2rem;
        }
        .section-caption {
            font-size: 13px;
            color: var(--text-caption) !important;
            margin-bottom: 0.7rem;
        }

        h3 {
            font-family: 'Playfair Display', serif !important;
            color: var(--text-card-header) !important;
        }

        /* ══════════════════════════════════════════════
           SHAP CARD
        ══════════════════════════════════════════════ */
        .shap-card {
            padding: 0.8rem 1rem 0.5rem 1rem;
            border-radius: 16px;
            background: var(--bg-shap-card) !important;
            border: 1px solid var(--border-shap);
            box-shadow: var(--shadow-shap);
            margin-top: 0.5rem;
        }
        .shap-card p { color: var(--text-shap-hint) !important; }

        /* ══════════════════════════════════════════════
           ADVICE CARDS
        ══════════════════════════════════════════════ */
        .advice-card {
            padding: 1.2rem 1.5rem;
            border-radius: 16px;
            background: var(--bg-advice) !important;
            border-left: 5px solid #e67e22;
            margin-bottom: 1rem;
            box-shadow: var(--shadow-advice);
        }
        .advice-card.urgent {
            background: var(--bg-advice-urgent) !important;
            border-left-color: #c0392b;
            box-shadow: var(--shadow-advice-urg);
        }
        .advice-card.shap-driven {
            background: var(--bg-advice-shap) !important;
            border-left: 5px solid #7c3aed;
            box-shadow: var(--shadow-advice-shp);
        }
        .advice-title {
            font-family: 'Playfair Display', serif;
            font-size: 15px;
            font-weight: 600;
            color: var(--text-advice-title) !important;
            margin-bottom: 0.4rem;
        }
        .advice-card p {
            color: var(--text-advice-body) !important;
            line-height: 1.6;
        }
        .advice-dice {
            font-size: 12.5px;
            color: var(--advice-dice-color) !important;
            background: var(--advice-dice-bg) !important;
            padding: 0.4rem 0.8rem;
            border-radius: 8px;
            margin-top: 0.5rem;
            display: inline-block;
            font-weight: 600;
        }
        .advice-source-tag {
            font-size: 10.5px;
            color: var(--text-caption) !important;
            text-transform: uppercase;
            letter-spacing: 0.06em;
            margin-bottom: 0.3rem;
        }

        /* ══════════════════════════════════════════════
           DICE INFO BOX
        ══════════════════════════════════════════════ */
        .dice-info {
            padding: 1.1rem 1.5rem;
            border-radius: 16px;
            background: var(--bg-dice-info) !important;
            border-left: 4px solid #7c3aed;
            margin-bottom: 1rem;
            font-size: 13.5px;
            color: var(--text-dice-info) !important;
        }
        .dice-info b  { color: var(--dice-b-color) !important; }
        .dice-info em { color: var(--dice-em-color) !important; }

        /* ══════════════════════════════════════════════
           ALL-CLEAR / RISK BANNER BOXES
        ══════════════════════════════════════════════ */
        .banner-allclear {
            background: var(--bg-allclear) !important;
            border-radius: 14px;
            padding: 1rem 1.4rem;
            border-left: 5px solid #27ae60;
            margin-bottom: 1rem;
            text-align: center;
        }
        .banner-allclear b  { color: var(--text-allclear) !important; font-size:14px; }
        .banner-allclear span { font-size:12.5px; color: var(--text-allclear-sub) !important; }

        .banner-highrisk {
            background: var(--bg-highrisk) !important;
            border-radius: 14px;
            padding: 1rem 1.4rem;
            border-left: 5px solid #c0392b;
            margin-bottom: 1rem;
        }
        .banner-highrisk b    { color: var(--text-highrisk) !important; font-size:14px; }
        .banner-highrisk span { font-size:13px; color: var(--text-highrisk-sub) !important; }

        .banner-moderate {
            background: var(--bg-moderate) !important;
            border-radius: 14px;
            padding: 1rem 1.4rem;
            border-left: 5px solid #e67e22;
            margin-bottom: 1rem;
        }
        .banner-moderate p { font-size:13px; color: var(--text-moderate-sub) !important; font-weight:600; margin-bottom:0.8rem; }

        /* ══════════════════════════════════════════════
           PROBA MINI BARS
        ══════════════════════════════════════════════ */
        .proba-track {
            flex: 1;
            background: var(--bg-proba-track) !important;
            border-radius: 999px;
            height: 8px;
        }
        .proba-label {
            width: 70px;
            font-size: 11px;
            color: var(--text-proba-label) !important;
        }
        .proba-row {
            display: flex;
            align-items: center;
            gap: 0.5rem;
            margin-bottom: 0.25rem;
        }

        /* ══════════════════════════════════════════════
           HERO STRIP
        ══════════════════════════════════════════════ */
        .hero-strip {
            background: var(--bg-hero) !important;
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
            color: white !important;
        }
        .hero-strip .hero-sub { font-size:14px; opacity:0.88; color:white !important; }

        /* ══════════════════════════════════════════════
           FEATURE PILL
        ══════════════════════════════════════════════ */
        .feature-pill {
            display: inline-block;
            padding: 0.25rem 0.8rem;
            border-radius: 999px;
            background: var(--bg-pill) !important;
            color: var(--color-accent) !important;
            font-size: 12px;
            font-weight: 600;
            margin: 0.2rem 0.15rem;
        }

        /* ══════════════════════════════════════════════
           BUTTONS
        ══════════════════════════════════════════════ */
        .stButton > button {
            font-family: 'Nunito', sans-serif;
            font-weight: 700;
            border-radius: 12px;
            letter-spacing: 0.03em;
            background: linear-gradient(135deg,#b5467a,#7c3aed) !important;
            color: white !important;
            border: none !important;
            transition: opacity 0.2s, transform 0.15s;
        }
        .stButton > button:hover { opacity:0.92; transform:translateY(-1px); }

        /* ══════════════════════════════════════════════
           TABS
        ══════════════════════════════════════════════ */
        div[data-testid="stTabs"] button {
            font-family: 'Nunito', sans-serif;
            font-size: 13.5px;
            font-weight: 600;
        }

        /* ══════════════════════════════════════════════
           EXPANDER
        ══════════════════════════════════════════════ */
        .streamlit-expanderHeader { font-family:'Nunito',sans-serif; font-weight:600; }
        [data-testid="stExpander"] {
            background: var(--bg-card) !important;
            border-color: var(--border-card) !important;
        }
        [data-testid="stExpander"] summary {
            color: var(--text-primary) !important;
        }

        /* ══════════════════════════════════════════════
           METRIC
        ══════════════════════════════════════════════ */
        [data-testid="metric-container"] {
            background: var(--bg-metric) !important;
            border-radius: 12px;
            padding: 0.6rem;
            border: 1px solid var(--border-metric);
        }

        /* ══════════════════════════════════════════════
           STEP CARDS (home page)
        ══════════════════════════════════════════════ */
        .step-card { text-align:center; padding:1.1rem 1rem; }
        .step-title {
            font-family:'Playfair Display',serif;
            font-weight:600;
            font-size:13px;
            color: var(--text-card-header) !important;
            margin-bottom:0.4rem;
        }
        .step-desc {
            font-size:11.5px !important;
            color: var(--text-secondary) !important;
            line-height:1.5;
        }

        /* ══════════════════════════════════════════════
           DISCLAIMER FOOTER
        ══════════════════════════════════════════════ */
        .disclaimer-text {
            font-size: 11.5px;
            color: var(--text-secondary) !important;
            text-align: center;
            margin-top: 1.5rem;
            border-top: 1px solid var(--border-card);
            padding-top: 1.2rem;
        }
        .disclaimer-text b { color: var(--text-card-header) !important; }

        /* ══════════════════════════════════════════════
           INLINE-STYLE COLOUR OVERRIDES
           For any hardcoded hex that slips through in
           f-strings — map to CSS-variable equivalents.
           Uses  html[data-theme="dark"]  for specificity.
        ══════════════════════════════════════════════ */
        html[data-theme="dark"] *[style*="color:#5b2d7a"],
        html[data-theme="dark"] *[style*="color: #5b2d7a"] { color:var(--text-card-header) !important; }
        html[data-theme="dark"] *[style*="color:#8b7b8e"],
        html[data-theme="dark"] *[style*="color: #8b7b8e"] { color:var(--text-secondary) !important; }
        html[data-theme="dark"] *[style*="color:#9a8ab0"],
        html[data-theme="dark"] *[style*="color: #9a8ab0"] { color:var(--text-caption) !important; }
        html[data-theme="dark"] *[style*="color:#9a8a9d"],
        html[data-theme="dark"] *[style*="color: #9a8a9d"] { color:var(--text-caption) !important; }
        html[data-theme="dark"] *[style*="color:#3d2c50"],
        html[data-theme="dark"] *[style*="color: #3d2c50"] { color:var(--text-primary) !important; }
        html[data-theme="dark"] *[style*="color:#3d1f5a"],
        html[data-theme="dark"] *[style*="color: #3d1f5a"] { color:var(--text-advice-title) !important; }
        html[data-theme="dark"] *[style*="color:#b08aba"],
        html[data-theme="dark"] *[style*="color: #b08aba"] { color:var(--text-range) !important; }
        html[data-theme="dark"] *[style*="color:#4a3070"],
        html[data-theme="dark"] *[style*="color: #4a3070"] { color:var(--advice-dice-color) !important; }
        html[data-theme="dark"] *[style*="color:#2d6a4f"],
        html[data-theme="dark"] *[style*="color: #2d6a4f"] { color:#6ecfaa !important; }
        html[data-theme="dark"] *[style*="color:#5a2020"],
        html[data-theme="dark"] *[style*="color: #5a2020"] { color:#f0a0a0 !important; }
        html[data-theme="dark"] *[style*="color:#7a3a1a"],
        html[data-theme="dark"] *[style*="color: #7a3a1a"] { color:#f0c090 !important; }
        html[data-theme="dark"] *[style*="color:#2d1a4b"],
        html[data-theme="dark"] *[style*="color: #2d1a4b"] { color:var(--text-dice-info) !important; }
        html[data-theme="dark"] *[style*="color:#0f9b6e"],
        html[data-theme="dark"] *[style*="color: #0f9b6e"] { color:#4ecfa0 !important; }
        html[data-theme="dark"] *[style*="color:#c0392b"],
        html[data-theme="dark"] *[style*="color: #c0392b"] { color:#f07070 !important; }
        html[data-theme="dark"] *[style*="color:#aaa"],
        html[data-theme="dark"] *[style*="color: #aaa"]    { color:#8870a0 !important; }

        /* background inline overrides */
        html[data-theme="dark"] *[style*="background:#f0e8f5"],
        html[data-theme="dark"] *[style*="background: #f0e8f5"] {
            background: rgba(181,70,122,0.20) !important;
        }
        html[data-theme="dark"] *[style*="background:rgba(255,255,255,0.2)"],
        html[data-theme="dark"] *[style*="background: rgba(255,255,255,0.2)"] {
            background: rgba(255,255,255,0.10) !important;
        }
        html[data-theme="dark"] *[style*="background:rgba(124,58,237,0.08)"],
        html[data-theme="dark"] *[style*="background: rgba(124,58,237,0.08)"] {
            background: rgba(124,58,237,0.30) !important;
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
    idx_sorted  = np.argsort(np.abs(shap_values))
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
# Advice renderer
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
    flagged = get_flagged_features(input_data, normal_ranges)

    for feat in BINARY_RISK_FLAGS_DS3:
        if feat in input_data and float(input_data[feat]) > 0 and feat not in flagged:
            flagged.append(feat)

    is_low_risk  = "low"  in risk_label.lower()
    is_high_risk = "high" in risk_label.lower()

    shap_features = []
    if not is_low_risk and is_high_risk and flagged and shap_values is not None and feature_names is not None:
        shap_features = get_shap_driven_advice_features(shap_values, feature_names, input_data, top_n=5)
        shap_features = [f for f in shap_features if f in flagged]

    all_advice_features = flagged

    if not all_advice_features:
        st.markdown(
            "<div class='banner-allclear'>"
            "<b>✅ All monitored parameters are within normal range.</b><br>"
            "<span>Continue routine care, healthy lifestyle habits, "
            "and attend all scheduled appointments.</span>"
            "</div>",
            unsafe_allow_html=True,
        )
        return

    if is_high_risk:
        st.markdown(
            "<div class='banner-highrisk'>"
            "<b>🚨 High Risk — Urgent Clinical Evaluation Recommended</b><br>"
            "<span>The following flagged parameters are key areas of concern. "
            "Where DiCE targets are available, quantitative improvement goals are shown.</span>"
            "</div>",
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f"<div class='banner-moderate'>"
            f"<p>⚠️ <b>{len(flagged)} parameter(s)</b> flagged outside expected range — "
            f"personalised advice shown below.</p>"
            f"</div>",
            unsafe_allow_html=True,
        )

    urgent_features = {"Body Temp", "Preexisting Diabetes", "Gestational Diabetes"}

    for feat in all_advice_features:
        if feat not in feature_advice:
            continue
        title_str, advice_str = feature_advice[feat]
        is_urgent  = feat in urgent_features
        card_class = "advice-card urgent" if is_urgent else "advice-card"
        source_tag = "<div class='advice-source-tag'>⚠️ Out-of-range / flagged parameter</div>"

        dice_hint = ""
        if dice_deltas and feat in dice_deltas:
            delta     = dice_deltas[feat]
            direction = "increase" if delta > 0 else "decrease"
            arrow     = "↑" if delta > 0 else "↓"
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
            f"<p style='font-size:13px; margin:0.2rem 0 0.5rem 0; line-height:1.6;'>{advice_str}</p>"
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
    feature_cols        = list(feature_cols)
    actionable_features = list(actionable_features)

    X_train_df = pd.DataFrame(_X_train, columns=feature_cols)
    y_arr      = np.asarray(_y_train).ravel()

    label_map = None
    if y_arr.dtype.kind in {"U", "S", "O"}:
        unique    = sorted(set(y_arr.tolist()))
        label_map = {lbl: i for i, lbl in enumerate(unique)}
        y_arr     = np.array([label_map[v] for v in y_arr])

    train_df               = X_train_df.copy()
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
        "Systolic BP":   [110.0, 130.0],
        "Diastolic":     [70.0,  90.0],
        "BS":            [3.0,   20.0],
        "Body Temp":     [96.0,  104.0],
        "BMI":           [18.5,  28.0],
        "Mental Health": [0.0,   1.0],
        "Heart Rate":    [60.0,  100.0],
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
    data           = delta_df[actionable_features].values.astype(float)
    n_cfs, n_feats = data.shape
    vmax           = np.abs(data).max()
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
    is_low_risk = "low" in predicted_label.lower()

    if is_low_risk:
        st.markdown(
            "<div class='banner-allclear' style='text-align:left;'>"
            "<b>✅ Low Risk — Counterfactual Analysis Not Required</b><br>"
            "<span>DiCE counterfactuals are designed to identify changes that flip a high or moderate "
            "risk prediction. Since the model predicts <b>Low Risk</b>, no what-if scenarios are "
            "generated. Continue routine care and monitoring.</span>"
            "</div>",
            unsafe_allow_html=True,
        )
        if input_data is not None and normal_ranges is not None:
            st.markdown("### 🩹 Clinical Advice")
            render_advice_section(
                input_data=input_data,
                normal_ranges=normal_ranges,
                feature_advice=FEATURE_ADVICE_DS3,
                shap_values=None,
                feature_names=None,
                risk_label=predicted_label,
                dice_deltas=None,
            )
        return None, None

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
                    new_val  = float(cf_row[feat])
                    delta    = new_val - orig_val
                    arrow    = "⬆️" if delta > 0.005 else ("⬇️" if delta < -0.005 else "➡️")
                    rows.append({
                        "Feature":         feat.replace("_", " "),
                        "Current Value":   f"{orig_val:.2f}",
                        "Suggested Value": f"{new_val:.2f}",
                        "Change":          f"{arrow} {delta:+.2f}",
                    })
                st.dataframe(pd.DataFrame(rows).set_index("Feature"), use_container_width=True)

        st.caption("⚠️ These are model-derived suggestions only. Always apply clinical judgement.")
        dice_deltas = get_first_cf_deltas(delta_df, actionable_features)

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
# PDF report
# ─────────────────────────────────────────────
def create_pdf_report(input_dict, pred_label, proba_dict=None,
                      shap_contribs=None, flagged_features=None,
                      feature_advice=None, dice_deltas=None,
                      shap_values=None, feature_names=None):
    buffer = BytesIO()
    c      = canvas.Canvas(buffer, pagesize=A4)
    width, height = A4
    y = height - 50

    c.setFillColor(colors.HexColor("#b5467a"))
    c.rect(0, height - 85, width, 85, fill=True, stroke=False)
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
    is_high     = "high" in lower_label
    is_low      = "low"  in lower_label

    _pdf_section_header(c, "1. Risk Summary", y);            y -= 20
    c.setFont("Helvetica-Bold", 13)
    if is_high:   c.setFillColor(colors.HexColor("#c0392b"))
    elif is_low:  c.setFillColor(colors.HexColor("#0f9b6e"))
    else:         c.setFillColor(colors.HexColor("#e6a817"))
    c.drawString(60, y, f"Predicted Risk Level: {pred_label}")
    c.setFillColor(colors.black);  y -= 16
    c.setFont("Helvetica", 10)
    if is_high:
        c.drawString(60, y, "The model suggests HIGH maternal risk.");  y -= 14
        c.setFont("Helvetica-Bold", 10)
        c.setFillColor(colors.HexColor("#c0392b"))
        c.drawString(60, y, "URGENT: Please consult a qualified health professional as soon as possible.")
        c.setFillColor(colors.black);  c.setFont("Helvetica", 10);  y -= 14
    elif is_low:
        c.drawString(60, y, "The model suggests LOW maternal risk. Continue routine monitoring.");  y -= 14
    else:
        c.drawString(60, y, "The model suggests MODERATE maternal risk. Close monitoring is advised.");  y -= 14
    y -= 8

    if proba_dict:
        y = _pdf_page_break(c, y, height)
        _pdf_section_header(c, "2. Class Probabilities", y);  y -= 18
        c.setFont("Helvetica", 10)
        for cls, p in proba_dict.items():
            c.drawString(60, y, f"{cls}: {p:.3f}  ({p*100:.1f}%)");  y -= 14
            y = _pdf_page_break(c, y, height)
    y -= 8

    y = _pdf_page_break(c, y, height)
    _pdf_section_header(c, "3. Input Values Used", y);  y -= 18
    c.setFont("Helvetica", 10)
    for k, v in input_dict.items():
        c.drawString(60, y, f"{k.replace('_', ' ')}: {v}");  y -= 14
        y = _pdf_page_break(c, y, height)
    y -= 8

    if shap_contribs:
        y = _pdf_page_break(c, y, height)
        _pdf_section_header(c, "4. Top Influencing Features (SHAP)", y);  y -= 18
        c.setFont("Helvetica", 10)
        c.drawString(60, y, "Positive SHAP = increased risk   |   Negative SHAP = decreased risk");  y -= 16
        for feat, val in shap_contribs:
            indicator = "▲" if val > 0 else "▼"
            c.drawString(60, y, f"{indicator}  {feat.replace('_', ' ')}: SHAP = {val:.4f}");  y -= 14
            y = _pdf_page_break(c, y, height)
    y -= 8

    all_advice_features = list(flagged_features) if flagged_features else []
    for feat in BINARY_RISK_FLAGS_DS3:
        if feat in input_dict and float(input_dict.get(feat, 0)) > 0 and feat not in all_advice_features:
            all_advice_features.append(feat)

    if all_advice_features and feature_advice:
        y = _pdf_page_break(c, y, height)
        _pdf_section_header(c, "5. Clinical Advice", y);  y -= 18
        for feat in all_advice_features:
            if feat not in feature_advice:
                continue
            y = _pdf_page_break(c, y, height, min_y=130)
            title_str, advice_str = feature_advice[feat]
            clean_title = title_str.replace("**", "").replace("*", "")
            c.setFont("Helvetica-Bold", 10)
            c.setFillColor(colors.HexColor("#b5467a"))
            c.drawString(60, y, f"• {clean_title}")
            c.setFillColor(colors.black);  y -= 14
            y = _pdf_wrap_text(c, advice_str, 70, y, width - 110, height,
                               font_name="Helvetica", font_size=9)
            if not is_low and dice_deltas and feat in dice_deltas:
                delta     = dice_deltas[feat]
                direction = "increase" if delta > 0 else "decrease"
                dice_hint_text = (f"DiCE target: {direction} {feat.replace('_',' ')} "
                                  f"by {abs(delta):.2f} units to potentially shift risk category.")
                c.setFont("Helvetica-Oblique", 9)
                c.setFillColor(colors.HexColor("#7c3aed"))
                y = _pdf_wrap_text(c, dice_hint_text, 70, y, width - 110, height,
                                   font_name="Helvetica-Oblique", font_size=9)
                c.setFillColor(colors.black)
            y -= 4
    elif is_low:
        y = _pdf_page_break(c, y, height)
        _pdf_section_header(c, "5. Clinical Advice", y);  y -= 18
        c.setFont("Helvetica", 10)
        c.setFillColor(colors.HexColor("#0f9b6e"))
        c.drawString(60, y, "All monitored parameters are within normal range.");  y -= 14
        c.setFillColor(colors.black)
        c.drawString(60, y, "Continue routine care, healthy lifestyle habits, and scheduled appointments.");  y -= 14
    y -= 8

    if not is_low and dice_deltas:
        y = _pdf_page_break(c, y, height)
        _pdf_section_header(c, "6. DiCE Counterfactual Targets (Scenario 1)", y);  y -= 18
        c.setFont("Helvetica", 10)
        c.drawString(60, y, "Minimum changes to potentially shift risk classification:");  y -= 14
        for feat, delta in dice_deltas.items():
            if abs(delta) < 0.001:
                continue
            arrow = "↑" if delta > 0 else "↓"
            c.drawString(70, y, f"{arrow}  {feat.replace('_',' ')}: {delta:+.3f}");  y -= 13
            y = _pdf_page_break(c, y, height)
        y -= 8

    y = _pdf_page_break(c, y, height, min_y=100)
    _pdf_section_header(c, "7. Important Notice", y);  y -= 16
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


def _pdf_wrap_text(c, text, x, y, max_width_pts, page_height,
                   font_name="Helvetica", font_size=9):
    from reportlab.pdfbase.pdfmetrics import stringWidth
    c.setFont(font_name, font_size)
    words       = text.split()
    line        = ""
    line_height = font_size + 3

    for word in words:
        test_line = (line + " " + word).strip()
        if stringWidth(test_line, font_name, font_size) <= max_width_pts:
            line = test_line
        else:
            if line:
                c.drawString(x, y, line);  y -= line_height
                if y < 80:
                    c.showPage();  y = page_height - 50;  c.setFont(font_name, font_size)
            line = word
    if line:
        c.drawString(x, y, line);  y -= line_height + 2
    return y


# ─────────────────────────────────────────────
# Label formatter & risk colour
# ─────────────────────────────────────────────
def format_risk_label(raw_label: str) -> str:
    s     = str(raw_label).strip()
    lower = s.lower()
    if lower in ["0", "low"]:      return "Low Risk"
    if lower in ["1", "high"]:     return "High Risk"
    if lower in ["2", "medium", "moderate"]: return "Moderate Risk"
    return s.capitalize()


def risk_color(label: str):
    label = label.lower()
    if "low"  in label: return "#0f9b6e"
    if "high" in label: return "#c0392b"
    return "#e6a817"
