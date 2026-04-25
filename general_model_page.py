# general_model_page.py
import streamlit as st
import numpy as np

from utils import (
    load_model,
    load_train_data,
    FEATURES_DS3,
    DICE_ACTIONABLE_DS3,
    FEATURE_ADVICE_DS3,
    NORMAL_RANGES_DS3,
    get_shap_values,
    plot_shap_bar,
    plot_shap_waterfall,
    render_dice_tab,
    get_flagged_features,
    create_pdf_report,
    format_risk_label,
    risk_color,
    render_advice_section,
)
import matplotlib.pyplot as plt


def render_general_model():
    model = load_model()

    # ── Aggressive widget-label CSS fix ──────────────────────────────────────
    # Streamlit renders widget labels in shadow-like internal divs.
    # We must target every selector it uses so labels show in dark mode.
    st.markdown("""
    <style>
    /* ═══════════════════════════════════════
       NUMBER INPUT — label + input box
    ═══════════════════════════════════════ */
    div[data-testid="stNumberInput"] label,
    div[data-testid="stNumberInput"] label p,
    div[data-testid="stNumberInput"] label span,
    div[data-testid="stNumberInput"] > div > label,
    div[data-testid="stNumberInput"] > div > label > div > p,
    div[data-testid="stNumberInput"] [data-testid="stWidgetLabel"] p {
        color: var(--text-primary) !important;
        opacity: 1 !important;
    }
    div[data-testid="stNumberInput"] input {
        color: var(--text-primary) !important;
        background-color: var(--bg-card-solid) !important;
        border-color: var(--border-card) !important;
    }

    /* ═══════════════════════════════════════
       RADIO — question label + option text
    ═══════════════════════════════════════ */
    div[data-testid="stRadio"] label,
    div[data-testid="stRadio"] label p,
    div[data-testid="stRadio"] label span,
    div[data-testid="stRadio"] > div > label,
    div[data-testid="stRadio"] > div > label > div > p,
    div[data-testid="stRadio"] [data-testid="stWidgetLabel"] p,
    div[data-testid="stRadio"] div[data-baseweb="radio"] p,
    div[data-testid="stRadio"] div[data-baseweb="radio"] span,
    div[data-testid="stRadio"] div[role="radiogroup"] label,
    div[data-testid="stRadio"] div[role="radiogroup"] p,
    div[data-testid="stRadio"] div[role="radiogroup"] span {
        color: var(--text-primary) !important;
        opacity: 1 !important;
    }

    /* ═══════════════════════════════════════
       GENERIC WIDGET LABEL CATCH-ALL
    ═══════════════════════════════════════ */
    div[data-testid="stWidgetLabel"] p,
    div[data-testid="stWidgetLabel"] span,
    div[data-testid="stWidgetLabel"] label {
        color: var(--text-primary) !important;
        opacity: 1 !important;
    }

    /* ═══════════════════════════════════════
       MARKDOWN CONTAINERS
    ═══════════════════════════════════════ */
    div[data-testid="stMarkdownContainer"] p,
    div[data-testid="stMarkdownContainer"] span,
    div[data-testid="stMarkdownContainer"] li {
        color: var(--text-primary) !important;
    }
    div[data-testid="stCaptionContainer"] p {
        color: var(--text-caption) !important;
    }

    /* ═══════════════════════════════════════
       CARD DARK MODE OVERRIDE
    ═══════════════════════════════════════ */
    html[data-theme="dark"] .card {
        background: rgba(36,24,54,0.97) !important;
        border-color: rgba(181,70,122,0.35) !important;
    }
    html[data-theme="dark"] .card .card-header,
    html[data-theme="dark"] .card .card-header span:not(.icon) {
        color: #ddb8f5 !important;
    }
    html[data-theme="dark"] .card p,
    html[data-theme="dark"] .card b,
    html[data-theme="dark"] .card strong {
        color: #ecddf8 !important;
    }
    html[data-theme="dark"] .result-card {
        background: linear-gradient(135deg,#1c1228 0%,#271335 50%,#191028 100%) !important;
        border-color: rgba(181,70,122,0.40) !important;
    }
    html[data-theme="dark"] .result-title {
        color: #ddb8f5 !important;
    }
    html[data-theme="dark"] .risk-subtext {
        color: #c8b8d8 !important;
    }
    </style>
    """, unsafe_allow_html=True)

    # ── Page header ──────────────────────────────────────────────────────────
    st.markdown(
        """
        <div style='margin-bottom:0.2rem;'>
            <span style='font-size:30px;'>🤱</span>
            <span style='font-family:"Playfair Display",serif; font-size:28px; font-weight:700;
                         background:linear-gradient(135deg,#b5467a,#7c3aed);
                         -webkit-background-clip:text; -webkit-text-fill-color:transparent;
                         background-clip:text; vertical-align:middle; margin-left:0.4rem;'>
                MaternalCare — Risk Assessment
            </span>
        </div>
        <p class='section-caption'>
            Enter patient vitals and clinical history to estimate maternal risk level
            with full explainability and personalised guidance.
        </p>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        "<h3 style='margin-bottom:0.4rem; margin-top:0.8rem;'>📝 Patient Information</h3>",
        unsafe_allow_html=True,
    )

    # ── Input columns ────────────────────────────────────────────────────────
    # IMPORTANT: Streamlit widgets must NOT be placed inside open HTML div tags.
    # The card header HTML is self-contained (closed). Widgets come after.
    col_left, col_right = st.columns([1.2, 1.0], gap="large")

    with col_left:
        st.markdown(
            "<div class='card'>"
            "<div class='card-header'><span class='icon'>💊</span><span>Vitals &amp; Measurements</span></div>"
            "</div>",
            unsafe_allow_html=True,
        )
        age        = st.number_input("Age (years)",               min_value=10,   max_value=60,   value=25,   step=1)
        systolic   = st.number_input("Systolic BP (mmHg)",        min_value=70,   max_value=220,  value=110,  step=1)
        diastolic  = st.number_input("Diastolic BP (mmHg)",       min_value=40,   max_value=130,  value=80,   step=1)
        bs         = st.number_input("Blood Sugar — BS (mmol/L)", min_value=1.0,  max_value=25.0, value=6.9,  step=0.1)
        body_temp  = st.number_input("Body Temperature (°F)",     min_value=95.0, max_value=105.0,value=98.0, step=0.1)
        bmi        = st.number_input("BMI",                       min_value=10.0, max_value=60.0, value=24.0, step=0.1)
        heart_rate = st.number_input("Heart Rate (bpm)",          min_value=40,   max_value=200,  value=80,   step=1)

    with col_right:
        st.markdown(
            "<div class='card'>"
            "<div class='card-header'><span class='icon'>📋</span><span>Clinical History</span></div>"
            "<p class='card-body' style='font-size:12px; margin-bottom:0.5rem;'>"
            "Select <b>Yes</b> if the condition applies to this patient.</p>"
            "</div>",
            unsafe_allow_html=True,
        )
        prev_comp     = st.radio("Previous Complications", ["No", "Yes"], horizontal=True)
        pre_diab      = st.radio("Preexisting Diabetes",   ["No", "Yes"], horizontal=True)
        gest_diab     = st.radio("Gestational Diabetes",   ["No", "Yes"], horizontal=True)
        mental_health = st.radio("Mental Health Issue",    ["No", "Yes"], horizontal=True)
        st.markdown(
            "<p class='section-caption' style='font-size:11px; font-weight:600; margin-top:1rem; margin-bottom:0.2rem;'>"
            "📐 Normal Ranges (reference)</p>"
            "<p class='section-caption' style='font-size:11px; line-height:1.7; margin:0;'>"
            "Systolic 90–140 mmHg · Diastolic 60–90 mmHg · "
            "BS 3.9–7.8 mmol/L · Temp 97–99 °F · BMI 18.5–29.9 · HR 60–90 bpm"
            "</p>",
            unsafe_allow_html=True,
        )

    # ── Action buttons ────────────────────────────────────────────────────────
    col_btn1, col_btn2 = st.columns([3, 1], gap="medium")
    with col_btn1:
        predict_clicked = st.button("🔮  Predict & Explain", key="predict_general", use_container_width=True)
    with col_btn2:
        home_clicked = st.button("🏠 Home", key="home_general", use_container_width=True)

    if home_clicked:
        st.session_state["page"] = "Home"
        return
    if not predict_clicked:
        return

    # ── Build input vector ───────────────────────────────────────────────────
    yn = {"No": 0.0, "Yes": 1.0}
    input_data = {
        "Age":                    float(age),
        "Systolic BP":            float(systolic),
        "Diastolic":              float(diastolic),
        "BS":                     float(bs),
        "Body Temp":              float(body_temp),
        "BMI":                    float(bmi),
        "Previous Complications": yn[prev_comp],
        "Preexisting Diabetes":   yn[pre_diab],
        "Gestational Diabetes":   yn[gest_diab],
        "Mental Health":          yn[mental_health],
        "Heart Rate":             float(heart_rate),
    }
    x = np.array([[input_data[f] for f in FEATURES_DS3]], dtype=float)

    # ── Prediction ───────────────────────────────────────────────────────────
    pred    = model.predict(x)[0]
    classes = getattr(model, "classes_", None)
    proba   = model.predict_proba(x)[0] if hasattr(model, "predict_proba") else None

    raw_label  = classes[int(pred)] if classes is not None else pred
    nice_label = format_risk_label(raw_label)
    color      = risk_color(nice_label)

    badge_class = "risk-moderate"
    if "low"  in nice_label.lower(): badge_class = "risk-low"
    elif "high" in nice_label.lower(): badge_class = "risk-high"

    # ── Result card ──────────────────────────────────────────────────────────
    st.markdown(
        "<h3 style='margin-top:1.4rem; margin-bottom:0.3rem;'>🧾 Assessment Result</h3>",
        unsafe_allow_html=True,
    )
    st.markdown("<div class='result-card'>", unsafe_allow_html=True)
    col_r1, col_r2, col_r3 = st.columns([1.4, 1.0, 1.3])

    with col_r1:
        st.markdown("<div class='result-title'>Maternal Risk Estimation</div>", unsafe_allow_html=True)
        st.markdown(
            f"<span class='risk-badge {badge_class}'>{nice_label}</span>"
            f"<div class='risk-main-value' style='color:{color}; margin-top:0.7rem;'>{nice_label}</div>"
            f"<div class='risk-subtext' style='margin-top:0.4rem;'>"
            f"Based on vital signs, blood metrics, and clinical history provided.</div>",
            unsafe_allow_html=True,
        )

    with col_r2:
        if proba is not None:
            conf = float(proba[int(pred)])
            st.markdown(
                "<p class='card-body' style='font-weight:700; margin-bottom:0.3rem;'>Model Confidence</p>",
                unsafe_allow_html=True,
            )
            st.markdown(
                f"<span style='font-size:26px; font-weight:700; color:{color};'>{conf*100:.1f}%</span>",
                unsafe_allow_html=True,
            )
            st.progress(conf)
            if classes is not None:
                st.markdown("<div style='margin-top:0.6rem;'>", unsafe_allow_html=True)
                for c_lbl, p in zip(classes, proba):
                    lbl   = format_risk_label(c_lbl)
                    c_col = risk_color(lbl)
                    st.markdown(
                        f"<div class='proba-row'>"
                        f"<div class='proba-label'>{lbl.replace(' Risk','')}</div>"
                        f"<div class='proba-track'>"
                        f"<div style='width:{p*100:.0f}%; background:{c_col}; height:8px; border-radius:999px;'></div>"
                        f"</div>"
                        f"<div style='font-size:11px; font-weight:600; color:{c_col}; width:36px;'>{p*100:.0f}%</div>"
                        f"</div>",
                        unsafe_allow_html=True,
                    )
                st.markdown("</div>", unsafe_allow_html=True)
        else:
            st.markdown("<p class='card-body' style='font-weight:700;'>Model Confidence</p>", unsafe_allow_html=True)
            st.write("_Not available_")

    with col_r3:
        st.markdown(
            "<p class='card-body' style='font-weight:700; margin-bottom:0.4rem;'>Clinical Interpretation</p>",
            unsafe_allow_html=True,
        )
        if "low" in nice_label.lower():
            st.success("✅ **Low maternal risk** detected.\n\nContinue routine monitoring and healthy lifestyle measures.")
        elif "high" in nice_label.lower():
            st.error("🚨 **High maternal risk** detected.\n\nCloser clinical evaluation and **prompt follow-up** are strongly recommended.")
        else:
            st.warning("⚠️ **Moderate maternal risk** detected.\n\nMonitor closely and address modifiable risk factors.")

    st.markdown("</div>", unsafe_allow_html=True)

    # ── SHAP ─────────────────────────────────────────────────────────────────
    st.markdown(
        "<h3 style='margin-top:1.4rem; margin-bottom:0.2rem;'>🧠 Why Did the Model Predict This?</h3>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<p class='section-caption'>"
        "SHAP explains what drove the prediction. "
        "The Counterfactual tab goes further — it shows what would need to change to get a different outcome, "
        "paired with personalised clinical advice.</p>",
        unsafe_allow_html=True,
    )

    shap_values, base_value = get_shap_values(
        model, x,
        predicted_class_index=int(pred) if classes is not None else None,
    )

    tab_cf, tab_bar, tab_waterfall = st.tabs([
        "💡 What-If & Advice (DiCE)",
        "📊 Feature Impact (Bar)",
        "🌊 Step-by-Step (Waterfall)",
    ])

    with tab_cf:
        X_train, y_train = load_train_data()
        if X_train is None:
            st.warning("Training data not found (`train_df-d3.pkl`). DiCE needs the original training set.")
            st.markdown("### 🩹 Personalised Clinical Advice")
            render_advice_section(
                input_data=input_data, normal_ranges=NORMAL_RANGES_DS3,
                feature_advice=FEATURE_ADVICE_DS3, shap_values=shap_values,
                feature_names=FEATURES_DS3, risk_label=nice_label, dice_deltas=None,
            )
        else:
            render_dice_tab(
                model=model, x_row=x[0], feature_cols=FEATURES_DS3,
                actionable_features=DICE_ACTIONABLE_DS3, X_train=X_train, y_train=y_train,
                predicted_label=nice_label, input_data=input_data, normal_ranges=NORMAL_RANGES_DS3,
                shap_values=shap_values, feature_names=FEATURES_DS3,
            )

    with tab_bar:
        st.markdown("<div class='shap-card'>", unsafe_allow_html=True)
        idx_sorted = np.argsort(np.abs(shap_values))[::-1]
        summary_parts = []
        for i in idx_sorted[:3]:
            direction = "🔺 raised" if shap_values[i] > 0 else "🔻 lowered"
            summary_parts.append(f"**{FEATURES_DS3[i]}** {direction} the risk")
        st.markdown(
            "<p style='font-size:13px; color:var(--text-shap-driver); font-weight:600; margin-bottom:0.5rem;'>"
            "📌 Top 3 drivers for this patient: " + " · ".join(summary_parts) + "</p>",
            unsafe_allow_html=True,
        )
        st.markdown(
            "<p style='font-size:12px; color:var(--text-shap-hint); margin-bottom:0.7rem;'>"
            "Bars → right increase risk · Bars ← left reduce it</p>",
            unsafe_allow_html=True,
        )
        fig = plot_shap_bar(shap_values, FEATURES_DS3, "Feature Impact on Maternal Risk Prediction")
        st.pyplot(fig)
        plt.close(fig)
        st.markdown("</div>", unsafe_allow_html=True)

    with tab_waterfall:
        st.markdown("<div class='shap-card'>", unsafe_allow_html=True)
        st.markdown(
            "<p style='font-size:12.5px; color:var(--text-shap-hint); margin-bottom:0.5rem;'>"
            "Each bar shows how a single feature shifts the risk score from the baseline (E[f(x)]) "
            "to the final prediction (f(x)).</p>",
            unsafe_allow_html=True,
        )
        fig2 = plot_shap_waterfall(shap_values, base_value, x[0], FEATURES_DS3,
                                   "How Each Feature Shifts the Risk Score")
        st.pyplot(fig2)
        plt.close(fig2)
        st.markdown("</div>", unsafe_allow_html=True)

    # ── PDF download ──────────────────────────────────────────────────────────
    st.markdown(
        "<h3 style='margin-top:1.4rem; margin-bottom:0.3rem;'>📄 Download Full Report</h3>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<p class='section-caption'>"
        "The PDF includes: risk summary, class probabilities, input values, "
        "top SHAP contributors, clinical advice for all flagged and high-influence parameters, "
        "and DiCE counterfactual targets.</p>",
        unsafe_allow_html=True,
    )

    idx_sorted_all = np.argsort(np.abs(shap_values))[::-1]
    top_contribs   = [(FEATURES_DS3[i], float(shap_values[i])) for i in idx_sorted_all[:5]]
    proba_dict     = ({str(c_): float(p) for c_, p in zip(classes, proba)}
                     if (proba is not None and classes is not None) else None)
    flagged        = get_flagged_features(input_data, NORMAL_RANGES_DS3)

    pdf_buffer = create_pdf_report(
        input_dict=input_data, pred_label=nice_label, proba_dict=proba_dict,
        shap_contribs=top_contribs, flagged_features=flagged,
        feature_advice=FEATURE_ADVICE_DS3, dice_deltas=None,
        shap_values=shap_values, feature_names=FEATURES_DS3,
    )

    col_dl1, col_dl2, col_dl3 = st.columns([1, 2, 1])
    with col_dl2:
        st.download_button(
            label="⬇️  Download PDF Report",
            data=pdf_buffer,
            file_name="maternalcare_risk_report.pdf",
            mime="application/pdf",
            use_container_width=True,
        )

    st.markdown(
        "<p class='disclaimer-text'>"
        "PDF includes risk summary, SHAP analysis, clinical advice, and DiCE counterfactual targets.</p>",
        unsafe_allow_html=True,
    )
