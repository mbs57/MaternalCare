# home_page.py
import streamlit as st


def render_home():
    # ── Hero Strip ────────────────────────────────────────────────────────────
    st.markdown(
        """
        <div class="hero-strip">
            <div style="display:flex; align-items:center; gap:1rem; margin-bottom:0.6rem;">
                <span style="font-size:48px; line-height:1;">🤱</span>
                <div>
                    <div class="hero-title">MaternalCare</div>
                    <div class="hero-sub">AI-powered maternal risk assessment with explainable predictions &amp; personalised clinical guidance</div>
                </div>
            </div>
            <div style="margin-top:1rem; display:flex; flex-wrap:wrap; gap:0.4rem;">
                <span class="feature-pill" style="color:white !important; background:rgba(255,255,255,0.22) !important;">
                    🧠 SHAP Explainability
                </span>
                <span class="feature-pill" style="color:white !important; background:rgba(255,255,255,0.22) !important;">
                    💡 DiCE Counterfactuals
                </span>
                <span class="feature-pill" style="color:white !important; background:rgba(255,255,255,0.22) !important;">
                    🩹 Personalised Advice
                </span>
                <span class="feature-pill" style="color:white !important; background:rgba(255,255,255,0.22) !important;">
                    📄 PDF Report
                </span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── What This Tool Does ───────────────────────────────────────────────────
    st.markdown(
        "<h3 style='margin-bottom:0.3rem;'>About This Tool</h3>"
        "<p class='section-caption'>A clinical decision-support system combining machine learning, "
        "explainability, and evidence-based guidance for maternal health professionals.</p>",
        unsafe_allow_html=True,
    )

    col1, col2, col3 = st.columns(3, gap="medium")

    with col1:
        st.markdown(
            """
            <div class="card">
                <div class="card-header"><span class="icon">🎯</span><span>What It Assesses</span></div>
                <p class="card-body">
                    Estimates maternal risk level — <strong>Low, Moderate, or High</strong> —
                    based on vital signs, blood metrics, and clinical history.
                    Designed for routine screening and antenatal visits.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col2:
        st.markdown(
            """
            <div class="card">
                <div class="card-header"><span class="icon">📋</span><span>Input Parameters</span></div>
                <p class="card-body">
                    Accepts <strong>11 clinical variables</strong> including blood pressure, blood sugar,
                    body temperature, BMI, heart rate, age, and obstetric history flags
                    such as diabetes and previous complications.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col3:
        st.markdown(
            """
            <div class="card">
                <div class="card-header"><span class="icon">📤</span><span>What You Get</span></div>
                <p class="card-body">
                    A risk classification with confidence score, SHAP feature explanations,
                    DiCE counterfactual action scenarios, tailored clinical advice,
                    and a downloadable <strong>PDF report</strong>.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # ── CTA Button ────────────────────────────────────────────────────────────
    st.markdown("<div style='margin: 1.4rem 0 0.4rem 0;'>", unsafe_allow_html=True)
    col_cta1, col_cta2, col_cta3 = st.columns([1, 2, 1])
    with col_cta2:
        if st.button("🔮  Start Risk Assessment →", use_container_width=True, key="home_cta"):
            st.session_state["page"] = "General"
    st.markdown("</div>", unsafe_allow_html=True)

    # ── XAI Explainer ─────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown(
        "<h3 style='margin-bottom:0.3rem;'>🔍 How Are Predictions Explained?</h3>"
        "<p class='section-caption'>Every prediction comes with three layers of transparency "
        "so clinicians can understand, verify, and act on the model's output.</p>",
        unsafe_allow_html=True,
    )

    col_a, col_b, col_c = st.columns(3, gap="medium")

    with col_a:
        st.markdown(
            """
            <div class="card">
                <div class="card-header"><span class="icon">📊</span><span>SHAP Bar Plot</span></div>
                <p class="card-body">
                    Shows <strong>which features had the largest impact</strong> on the prediction
                    and whether each one pushed risk <strong>up ↑ or down ↓</strong>.
                    Ideal for a quick at-a-glance overview of what drove the outcome.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col_b:
        st.markdown(
            """
            <div class="card">
                <div class="card-header"><span class="icon">🌊</span><span>Waterfall Plot</span></div>
                <p class="card-body">
                    Walks through the prediction <strong>step by step</strong>, showing how each feature
                    cumulatively shifts the score from the model baseline to the final output.
                    Best for deep inspection of individual cases.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col_c:
        st.markdown(
            """
            <div class="card">
                <div class="card-header"><span class="icon">💡</span><span>Counterfactual + Advice</span></div>
                <p class="card-body">
                    Powered by <strong>DiCE</strong>: answers <em>'What is the minimum change needed
                    to get a different prediction?'</em> Each scenario pairs quantitative targets
                    with evidence-based clinical guidance for modifiable risk factors.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # ── Advice Section explainer ──────────────────────────────────────────────
    st.markdown("---")
    st.markdown(
        "<h3 style='margin-bottom:0.3rem;'>🩹 Personalised Clinical Advice</h3>"
        "<p class='section-caption'>"
        "Beyond prediction — advice is generated based on flagged parameters, "
        "model-identified risk drivers (SHAP), and DiCE action targets.</p>",
        unsafe_allow_html=True,
    )

    col_d, col_e = st.columns(2, gap="medium")
    with col_d:
        st.markdown(
            """
            <div class="card">
                <div class="card-header"><span class="icon">⚠️</span><span>Flagged Parameter Advice</span></div>
                <p class="card-body">
                    When a feature falls outside the expected clinical range, a tailored advice card
                    is shown covering dietary changes, medication reminders, lifestyle interventions,
                    and when to seek urgent care — specific to that parameter.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col_e:
        st.markdown(
            """
            <div class="card">
                <div class="card-header"><span class="icon">🎯</span><span>DiCE-Guided Targets</span></div>
                <p class="card-body">
                    Even when all parameters are within range, if the model predicts <strong>High Risk</strong>,
                    the top SHAP-driven features are surfaced with DiCE quantitative targets — e.g.,
                    <em>'Reducing Blood Sugar by 1.8 mmol/L could shift the risk category.'</em>
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # ── Workflow steps ────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown(
        "<h3 style='margin-bottom:0.3rem;'>⚡ How To Use</h3>",
        unsafe_allow_html=True,
    )

    steps = [
        ("1️⃣", "Enter patient vitals",      "Fill in the 11 clinical parameters in the assessment form."),
        ("2️⃣", "Click Predict & Explain",   "The model returns a risk level with confidence score instantly."),
        ("3️⃣", "Review SHAP explanations",  "Understand which factors most influenced the prediction."),
        ("4️⃣", "Explore What-If scenarios", "See DiCE counterfactuals and integrated clinical advice."),
        ("5️⃣", "Download PDF Report",       "Share a structured report with the full assessment, SHAP, and advice."),
    ]

    step_cols = st.columns(5, gap="small")
    for col, (num, title, desc) in zip(step_cols, steps):
        with col:
            st.markdown(
                f"""
                <div class="card step-card">
                    <div style="font-size:26px; margin-bottom:0.4rem; text-align:center;">{num}</div>
                    <div class="step-title">{title}</div>
                    <p class="card-body step-desc">{desc}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

    # ── Disclaimer ────────────────────────────────────────────────────────────
    st.markdown(
        "<p class='disclaimer-text'>"
        "⚠️ <b>MaternalCare</b> is a <b>clinical decision-support aid</b> only. "
        "All predictions, explanations, and advice must be reviewed and validated by a "
        "qualified healthcare professional before any clinical action is taken."
        "</p>",
        unsafe_allow_html=True,
    )
