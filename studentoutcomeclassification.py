import pandas as pd
import numpy as np
import pickle as pkl
import streamlit as st

# --- 1. LOAD MODELS (cached — only runs once per server lifetime) ---
@st.cache_resource
def load_models():
    return (
        pkl.load(open("catModel.pkl",    "rb")),
        pkl.load(open("Transformer.pkl", "rb")),
        pkl.load(open("encoder.pkl",     "rb")),
        pkl.load(open("textCols.pkl",    "rb")),
        pkl.load(open("numCols.pkl",     "rb")),
        pkl.load(open("bounds.pkl",      "rb")),
    )

catModel, Transformer, encoder, textCols, numCols, bounds = load_models()

# --- 2. CSS ---
st.markdown("""
    <style>
        .stApp {
            background: linear-gradient(135deg, #fff5f5, #ffe0e0, #fff0e0) !important;
        }
        [data-testid="stFileUploader"] {
            border: 2px dashed #ffb3b3;
            border-radius: 12px;
            padding: 12px 20px;
            background-color: #fafafa;
            transition: border-color 0.2s, background-color 0.2s;
        }
        [data-testid="stFileUploader"]:hover {
            border-color: #ff4b4b;
            background-color: #fff5f5;
        }
        [data-testid="stFileUploadDropzone"] {
            background-color: transparent !important;
        }
        .success-badge {
            background-color: #f0fff4;
            border-left: 4px solid #28a745;
            border-radius: 6px;
            padding: 12px 16px;
            color: #155724;
            font-size: 15px;
            font-weight: 500;
        }
        .error-box {
            background-color: #fff0f0;
            border-left: 4px solid #ff4b4b;
            border-radius: 4px;
            padding: 8px 12px;
            margin-bottom: 10px;
            color: #cc0000;
            font-size: 14px;
        }
        div[data-testid="stButton"]:first-of-type > button {
            background: #17cac6 !important;);
            color: white;
            border: none;
            font-weight: 600;
            font-size: 16px;
            border-radius: 8px;
            transition: opacity 0.2s;
        }
        div[data-testid="stButton"]:first-of-type > button:hover {
            opacity: 0.88;
            color: white;
            border: none;
        }
    </style>
""", unsafe_allow_html=True)

# --- 3. TITLE ---
st.markdown("""
    <h1 style="
        font-size: 32px;
        font-weight: bold;
        text-align: center;
        background: linear-gradient(to right, #ff4b4b, #f9a825);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 20px;
    ">Outcome Prediction Model</h1>
""", unsafe_allow_html=True)

# --- 4. SESSION STATE ---
if "predicted_file" not in st.session_state:
    st.session_state.predicted_file = None
if "show_success" not in st.session_state:
    st.session_state.show_success = False

# --- 5. UPLOAD SECTION ---
st.markdown("""
    <div style="text-align:center; font-size:20px; font-weight:bold; color:#333; margin-bottom:6px;">
        Upload Data File
    </div>
    <div style="text-align:center; font-size:14px; color:#888; margin-bottom:16px;">
        Upload a CSV file to predict outcomes
    </div>
""", unsafe_allow_html=True)

file = st.file_uploader(
    "Choose a CSV file (up to 50 MB)",
    type="csv",
    key="outcome_uploader",
)

st.markdown("<div style='margin-bottom:12px;'></div>", unsafe_allow_html=True)

button = st.button("Predict Outcome", use_container_width=True)

# --- 6. PREDICTION LOGIC ---
if button:
    if not file:
        st.markdown(
            '<div class="error-box">⚠️ Please upload a CSV file before predicting.</div>',
            unsafe_allow_html=True
        )
    else:
        try:
            with st.spinner("Running predictions..."):
                # Read CSV — use engine='c' (default) for max speed
                data = pd.read_csv(file, engine='c')
                data.columns = data.columns.str.replace('\t', '', regex=False)

                # Clip all numeric bounds in one vectorized pass using a dict
                clip_bounds = {col: (bounds[col]['lower'], bounds[col]['upper']) for col in numCols}
                for col, (lo, hi) in clip_bounds.items():
                    data[col] = data[col].clip(lower=lo, upper=hi)

                # Predict
                PredictOutcome  = catModel.predict(data)
                data["Outcome"] = encoder.inverse_transform(PredictOutcome)

            st.write(data)  # kept as requested

            st.session_state.predicted_file = data
            st.session_state.show_success   = True
            st.rerun()

        except Exception as e:
            st.error(f"Error processing file: {e}")

# --- 7. RESULTS ---
if st.session_state.predicted_file is not None and st.session_state.show_success:

    df            = st.session_state.predicted_file
    outcomeCounts = df["Outcome"].value_counts()

    COLORS = [
        ("#fff0f0", "#ff4b4b", "#cc0000"),
        ("#f0fff4", "#28a745", "#155724"),
        ("#fff8e1", "#f9a825", "#7d6200"),
        ("#e8f4fd", "#1a73e8", "#0d47a1"),
    ]

    # Success banner + clear button
    c1, c2 = st.columns([0.85, 0.15])
    with c1:
        st.markdown(
            '<div class="success-badge">✅ Analysis Complete — Predictions ready!</div>',
            unsafe_allow_html=True
        )
    with c2:
        if st.button("✖", key="clear_success"):
            st.session_state.show_success   = False
            st.session_state.predicted_file = None
            st.rerun()

    # Summary cards
    cols = st.columns(len(outcomeCounts))
    for i, (outcome, count) in enumerate(outcomeCounts.items()):
        bg, border, text = COLORS[i % len(COLORS)]
        with cols[i]:
            st.markdown(f"""
                <div style="
                    background-color:{bg};border-left:4px solid {border};
                    border-radius:8px;padding:16px;text-align:center;
                    font-size:18px;font-weight:bold;color:{text};margin-bottom:16px;
                ">{outcome}<br>{count:,}</div>
            """, unsafe_allow_html=True)

    # --- Download button (fast — no rerun needed) ---
    csv_bytes = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="⬇️ Download Results CSV",
        data=csv_bytes,
        file_name="predictions.csv",
        mime="text/csv",
        use_container_width=True,
    )

    # --- Styled dataframe ---
    # Color ONLY the Outcome column (fastest method — ~0.000s vs 0.075s for full-row)
    MAX_STYLED_ROWS = 10_000
    outcomeList = df["Outcome"].unique().tolist()
    colorMap    = {outcomeList[i]: COLORS[i % len(COLORS)][0] for i in range(len(outcomeList))}

    if len(df) <= MAX_STYLED_ROWS:
        def color_outcome_cell(val):
            return f"background-color: {colorMap.get(val, '#ffffff')}"

        styled = df.style.map(color_outcome_cell, subset=["Outcome"])
        st.dataframe(styled, use_container_width=True)
    else:
        st.info(f"File has {len(df):,} rows — styling disabled for performance.")
        st.dataframe(df, use_container_width=True)