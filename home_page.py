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
                    <div class="hero-sub">AI-powered maternal risk assessment with explainable predictions & personalised clinical guidance</div>
                </div>
            </div>
            <div style="margin-top:1rem; display:flex; flex-wrap:wrap; gap:0.4rem;">
                <span style="background:rgba(255,255,255,0.2); color:white; border-radius:999px;
                             padding:0.25rem 0.9rem; font-size:12px; font-weight:600;">
                    🧠 SHAP Explainability
                </span>
                <span style="background:rgba(255,255,255,0.2); color:white; border-radius:999px;
                             padding:0.25rem 0.9rem; font-size:12px; font-weight:600;">
                    💡 DiCE Counterfactuals
                </span>
                <span style="background:rgba(255,255,255,0.2); color:white; border-radius:999px;
                             padding:0.25rem 0.9rem; font-size:12px; font-weight:600;">
                    🩹 Personalised Advice
                </span>
                <span style="background:rgba(255,255,255,0.2); color:white; border-radius:999px;
                             padding:0.25rem 0.9rem; font-size:12px; font-weight:600;">
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
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown(
            """<div class="card-header"><span class="icon">🎯</span><span>What It Assesses</span></div>""",
            unsafe_allow_html=True,
        )
        st.write(
            "Estimates maternal risk level — **Low, Moderate, or High** — "
            "based on vital signs, blood metrics, and clinical history. "
            "Designed for routine screening and antenatal visits."
        )
        st.markdown("</div>", unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown(
            """<div class="card-header"><span class="icon">📋</span><span>Input Parameters</span></div>""",
            unsafe_allow_html=True,
        )
        st.write(
            "Accepts **11 clinical variables** including blood pressure, blood sugar, "
            "body temperature, BMI, heart rate, age, and obstetric history flags "
            "such as diabetes and previous complications."
        )
        st.markdown("</div>", unsafe_allow_html=True)

    with col3:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown(
            """<div class="card-header"><span class="icon">📤</span><span>What You Get</span></div>""",
            unsafe_allow_html=True,
        )
        st.write(
            "A risk classification with confidence score, SHAP feature explanations, "
            "DiCE counterfactual action scenarios, tailored clinical advice, "
            "and a downloadable **PDF report**."
        )
        st.markdown("</div>", unsafe_allow_html=True)

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
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown(
            """<div class="card-header"><span class="icon">📊</span><span>SHAP Bar Plot</span></div>""",
            unsafe_allow_html=True,
        )
        st.write(
            "Shows **which features had the largest impact** on the prediction "
            "and whether each one pushed risk **up ↑ or down ↓**. "
            "Ideal for a quick at-a-glance overview of what drove the outcome."
        )
        st.markdown("</div>", unsafe_allow_html=True)

    with col_b:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown(
            """<div class="card-header"><span class="icon">🌊</span><span>Waterfall Plot</span></div>""",
            unsafe_allow_html=True,
        )
        st.write(
            "Walks through the prediction **step by step**, showing how each feature "
            "cumulatively shifts the score from the model baseline to the final output. "
            "Best for deep inspection of individual cases."
        )
        st.markdown("</div>", unsafe_allow_html=True)

    with col_c:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown(
            """<div class="card-header"><span class="icon">💡</span><span>Counterfactual + Advice</span></div>""",
            unsafe_allow_html=True,
        )
        st.write(
            "Powered by **DiCE**: answers *'What is the minimum change needed to get a different prediction?'* "
            "Each scenario pairs quantitative targets with evidence-based clinical guidance "
            "for modifiable risk factors."
        )
        st.markdown("</div>", unsafe_allow_html=True)

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
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown(
            """<div class="card-header"><span class="icon">⚠️</span><span>Flagged Parameter Advice</span></div>""",
            unsafe_allow_html=True,
        )
        st.write(
            "When a feature falls outside the expected clinical range, a tailored advice card "
            "is shown covering dietary changes, medication reminders, lifestyle interventions, "
            "and when to seek urgent care — specific to that parameter."
        )
        st.markdown("</div>", unsafe_allow_html=True)

    with col_e:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown(
            """<div class="card-header"><span class="icon">🎯</span><span>DiCE-Guided Targets</span></div>""",
            unsafe_allow_html=True,
        )
        st.write(
            "Even when all parameters are within range, if the model predicts **High Risk**, "
            "the top SHAP-driven features are surfaced with DiCE quantitative targets — e.g., "
            "*'Reducing Blood Sugar by 1.8 mmol/L could shift the risk category.'*"
        )
        st.markdown("</div>", unsafe_allow_html=True)

    # ── Workflow steps ────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown(
        "<h3 style='margin-bottom:0.3rem;'>⚡ How To Use</h3>",
        unsafe_allow_html=True,
    )

    steps = [
        ("1️⃣", "Enter patient vitals", "Fill in the 11 clinical parameters in the assessment form."),
        ("2️⃣", "Click Predict & Explain", "The model returns a risk level with confidence score instantly."),
        ("3️⃣", "Review SHAP explanations", "Understand which factors most influenced the prediction."),
        ("4️⃣", "Explore What-If scenarios", "See DiCE counterfactuals and integrated clinical advice."),
        ("5️⃣", "Download PDF Report", "Share a structured report with the full assessment, SHAP, and advice."),
    ]

    step_cols = st.columns(5, gap="small")
    for col, (num, title, desc) in zip(step_cols, steps):
        with col:
            st.markdown(
                f"<div class='card' style='text-align:center; padding:1rem;'>"
                f"<div style='font-size:26px; margin-bottom:0.4rem;'>{num}</div>"
                f"<div style='font-family:Playfair Display,serif; font-weight:600; font-size:13px; "
                f"color:#5b2d7a; margin-bottom:0.4rem;'>{title}</div>"
                f"<div style='font-size:11.5px; color:#8b7b8e; line-height:1.5;'>{desc}</div>"
                f"</div>",
                unsafe_allow_html=True,
            )

    # ── Disclaimer ────────────────────────────────────────────────────────────
    st.markdown(
        "<p style='font-size:11.5px; color:#aaa; text-align:center; margin-top:1.5rem; "
        "border-top: 1px solid #ede0ee; padding-top:1.2rem;'>"
        "⚠️ <b>MaternalCare</b> is a <b>clinical decision-support aid</b> only. "
        "All predictions, explanations, and advice must be reviewed and validated by a "
        "qualified healthcare professional before any clinical action is taken."
        "</p>",
        unsafe_allow_html=True,
    )