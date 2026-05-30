import pandas as pd
import numpy as np
import pickle as pkl
import streamlit as st

# --- 1. LOAD MODELS ---
catModel    = pkl.load(open("catModel.pkl",    "rb"))
Transformer = pkl.load(open("Transformer.pkl", "rb"))
encoder     = pkl.load(open("encoder.pkl",     "rb"))

# --- 2. CSS ---
st.markdown("""
    <style>
        .stApp {
            background: linear-gradient(135deg, #fff5f5, #ffe0e0, #fff0e0) !important;
        }
        .upload-card {
            border: 2px dashed #ccc;
            border-radius: 12px;
            padding: 40px 20px;
            text-align: center;
            background-color: #fafafa;
            transition: border-color 0.2s, background-color 0.2s;
            cursor: pointer;
            margin-bottom: 0px;
        }
        .upload-card:hover, .upload-card.active {
            border-color: #ff4b4b;
            background-color: #fff5f5;
        }
        .upload-icon  { font-size: 40px; margin-bottom: 10px; }
        .upload-title { font-size: 16px; font-weight: 600; color: #333; margin-bottom: 6px; }
        .upload-sub   { font-size: 13px; color: #888; }
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
if "predicted_file" not in st.session_state: st.session_state.predicted_file = None
if "show_success"   not in st.session_state: st.session_state.show_success   = False

# ── Upload section header ─────────────────────
st.markdown("""
    <div style="text-align:center;font-size:20px;font-weight:bold;color:#333;margin-bottom:8px;">
        Upload Data File
    </div>
    <div style="text-align:center;font-size:14px;color:#888;margin-bottom:16px;">
        Upload a CSV file to predict outcomes
    </div>
""", unsafe_allow_html=True)

# ── Dynamic upload card ───────────────────────
if "outcome_uploader" in st.session_state and st.session_state.outcome_uploader is not None:
    u_file    = st.session_state.outcome_uploader
    kb        = round(u_file.size / 1024, 1)
    size_info = f"{kb} KB" if kb < 1024 else f"{round(kb/1024,2)} MB"
    card_html = f"""
        <div class="upload-card" id="ucard">
            <div class="upload-icon">📄</div>
            <div class="upload-title">{u_file.name}</div>
            <div class="upload-sub">{size_info} • Ready to Predict</div>
        </div>
    """
else:
    card_html = """
        <div class="upload-card" id="ucard">
            <div class="upload-icon">☁️</div>
            <div class="upload-title">Choose a file or drag &amp; drop it here</div>
            <div class="upload-sub">CSV format · up to 50 MB</div>
        </div>
    """

st.markdown(card_html, unsafe_allow_html=True)

file = st.file_uploader(
    "Upload CSV",
    type="csv",
    key="outcome_uploader",
    label_visibility="collapsed"
)

# Drag & drop highlight
st.markdown("""
    <script>
    (function() {
        function init() {
            const card = document.getElementById('ucard');
            const zone = document.querySelector('[data-testid="stFileUploadDropzone"]');
            if (!card || !zone) { setTimeout(init, 150); return; }
            zone.addEventListener('dragenter', () => card.classList.add('active'));
            zone.addEventListener('dragover',  (e) => { e.preventDefault(); card.classList.add('active'); });
            zone.addEventListener('dragleave', () => card.classList.remove('active'));
            zone.addEventListener('drop',      () => setTimeout(() => card.classList.remove('active'), 250));
        }
        init();
    })();
    </script>
    <div style="margin-bottom: 16px;"></div>
""", unsafe_allow_html=True)

button = st.button("Predict Outcome", use_container_width=True)

if button:
    if not file:
        st.markdown('<div class="error-box">⚠️ Please upload a CSV file before predicting</div>', unsafe_allow_html=True)
    else:
        try:
            data         = pd.read_csv(file)
            data.columns = data.columns.str.replace('\t', '', regex=False)

            TransformData  = Transformer.transform(data)
            PredictOutcome = catModel.predict(TransformData)
            data["Outcome"] = encoder.inverse_transform(PredictOutcome)

            st.session_state.predicted_file = data
            st.session_state.show_success   = True
            st.rerun()
        except Exception as e:
            st.error(f"Error processing file: {e}")

# ── Results ───────────────────────────────────
if st.session_state.predicted_file is not None:
    if st.session_state.show_success:

        # ── Summary counts ────────────────────
        outcomeCounts = st.session_state.predicted_file["Outcome"].value_counts()

        c1, c2 = st.columns([0.85, 0.15])
        with c1:
            st.markdown('<div class="success-badge">✅ Analysis Complete — Predictions ready!</div>', unsafe_allow_html=True)
        with c2:
            if st.button("✖", key="clear_success"):
                st.session_state.show_success   = False
                st.session_state.predicted_file = None
                st.rerun()

        # ── Summary cards — one per outcome ───
        cols = st.columns(len(outcomeCounts))
        colors = [
            ("#fff0f0", "#ff4b4b", "#cc0000"),
            ("#f0fff4", "#28a745", "#155724"),
            ("#fff8e1", "#f9a825", "#7d6200"),
            ("#e8f4fd", "#1a73e8", "#0d47a1"),
        ]
        for i, (outcome, count) in enumerate(outcomeCounts.items()):
            bg, border, text = colors[i % len(colors)]
            with cols[i]:
                st.markdown(f"""
                    <div style="
                        background-color: {bg};
                        border-left: 4px solid {border};
                        border-radius: 8px;
                        padding: 16px;
                        text-align: center;
                        font-size: 18px;
                        font-weight: bold;
                        color: {text};
                        margin-bottom: 16px;
                    ">{outcome}<br>{count:,}</div>
                """, unsafe_allow_html=True)

        # ── Table ─────────────────────────────
        MAX_STYLED_ROWS = 10000

        outcomeList   = st.session_state.predicted_file["Outcome"].unique().tolist()
        colorMap      = {
            outcomeList[i]: colors[i % len(colors)][0]
            for i in range(len(outcomeList))
        }

        def highlightResult(row):
            bg = colorMap.get(row['Outcome'], '#ffffff')
            return [f'background-color: {bg}'] * len(row)

        if len(st.session_state.predicted_file) <= MAX_STYLED_ROWS:
            st.dataframe(
                st.session_state.predicted_file.style.apply(highlightResult, axis=1),
                use_container_width=True
            )
        else:
            st.info(f"File has {len(st.session_state.predicted_file):,} rows — row highlighting disabled for performance.")
            st.dataframe(
                st.session_state.predicted_file,
                use_container_width=True
            )