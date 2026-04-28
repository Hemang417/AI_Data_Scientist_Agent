"""
AI Data Scientist Agent — Streamlit Web UI
==========================================
Upload any CSV and watch 4 AI agents autonomously
clean, analyse, model, and report on your data.

Run with:  streamlit run streamlit_app.py
"""
import os
os.environ["OPENAI_API_KEY"] = "sk-no-key-required"
os.environ["CREWAI_MEMORY_ENABLED"] = "false"
os.environ["CREWAI_TRACING_ENABLED"] = "false"
os.environ["CREWAI_TELEMETRY_OPT_OUT"] = "true"
os.environ["OTEL_SDK_DISABLED"] = "true"

import os
import sys
import time
import threading
import queue
import tempfile
import base64
import warnings
from io import StringIO
from pathlib import Path

import streamlit as st
import chardet
import pandas as pd

from dotenv import load_dotenv
load_dotenv()
warnings.filterwarnings("ignore")

# ── Page config (must be first Streamlit call) ──
st.set_page_config(
    page_title="AI Data Scientist",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ──
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500;600&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

.main-title {
    font-family: 'Space Mono', monospace;
    font-size: 2.2rem;
    font-weight: 700;
    background: linear-gradient(135deg, #6c8aff 0%, #38bdf8 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0.2rem;
}

.sub-title {
    color: #8b8fa3;
    font-size: 1rem;
    margin-bottom: 2rem;
}

.agent-card {
    background: #1a1d27;
    border: 1px solid #2e3347;
    border-radius: 12px;
    padding: 1rem 1.2rem;
    margin-bottom: 0.6rem;
    transition: border-color 0.3s;
}

.agent-card.active {
    border-color: #6c8aff;
    box-shadow: 0 0 12px rgba(108,138,255,0.15);
}

.agent-card.done {
    border-color: #34d399;
}

.agent-card.failed {
    border-color: #f87171;
}

.agent-name {
    font-family: 'Space Mono', monospace;
    font-size: 0.85rem;
    font-weight: 700;
    margin-bottom: 0.2rem;
}

.agent-status {
    font-size: 0.8rem;
    color: #8b8fa3;
}

.stat-box {
    background: #1a1d27;
    border: 1px solid #2e3347;
    border-radius: 10px;
    padding: 1rem;
    text-align: center;
}

.stat-number {
    font-family: 'Space Mono', monospace;
    font-size: 1.6rem;
    font-weight: 700;
    color: #6c8aff;
}

.stat-label {
    font-size: 0.78rem;
    color: #8b8fa3;
    margin-top: 0.2rem;
}

.log-box {
    background: #0f1117;
    border: 1px solid #2e3347;
    border-radius: 8px;
    padding: 1rem;
    font-family: 'Space Mono', monospace;
    font-size: 0.75rem;
    color: #8b8fa3;
    height: 300px;
    overflow-y: auto;
    line-height: 1.6;
}

.log-line-info  { color: #38bdf8; }
.log-line-ok    { color: #34d399; }
.log-line-warn  { color: #fbbf24; }
.log-line-error { color: #f87171; }

div[data-testid="stFileUploader"] {
    border: 2px dashed #2e3347;
    border-radius: 12px;
    padding: 1rem;
    background: #1a1d27;
}

div[data-testid="stFileUploader"]:hover {
    border-color: #6c8aff;
}

.report-container {
    background: #1a1d27;
    border: 1px solid #2e3347;
    border-radius: 12px;
    padding: 1.5rem;
    font-size: 0.95rem;
    line-height: 1.8;
}

.stButton > button {
    background: linear-gradient(135deg, #6c8aff, #38bdf8);
    color: white;
    border: none;
    border-radius: 8px;
    padding: 0.6rem 2rem;
    font-family: 'Space Mono', monospace;
    font-size: 0.85rem;
    font-weight: 700;
    cursor: pointer;
    transition: opacity 0.2s;
    width: 100%;
}

.stButton > button:hover { opacity: 0.85; }
.stButton > button:disabled { opacity: 0.4; }

.step-badge {
    display: inline-block;
    background: #2e3347;
    color: #8b8fa3;
    font-family: 'Space Mono', monospace;
    font-size: 0.7rem;
    padding: 0.15rem 0.5rem;
    border-radius: 4px;
    margin-right: 0.4rem;
}

hr { border-color: #2e3347; }
</style>
""", unsafe_allow_html=True)


# ═════════════════════════════════════════════
#  SIDEBAR — Configuration
# ═════════════════════════════════════════════

with st.sidebar:
    st.markdown('<div class="main-title">⚗️ Config</div>', unsafe_allow_html=True)
    st.markdown("---")

    st.markdown("**🤖 LLM Provider**")
    provider = st.selectbox(
        "Provider",
        ["Cerebras", "Groq", "Google Gemini", "OpenAI"],
        label_visibility="collapsed"
    )

    provider_models = {
        "Cerebras": ["cerebras/llama3.1-8b"],
        "Groq": ["groq/llama-3.1-70b-versatile", "groq/llama-3.1-8b-instant", "groq/mixtral-8x7b-32768"],
        "Google Gemini": ["gemini/gemini-2.5-flash-lite", "gemini/gemini-2.5-flash"],
        "OpenAI": ["gpt-4o-mini", "gpt-4o"],
    }

    provider_key_names = {
        "Cerebras": "CEREBRAS_API_KEY",
        "Groq": "GROQ_API_KEY",
        "Google Gemini": "GOOGLE_API_KEY",
        "OpenAI": "OPENAI_API_KEY",
    }

    selected_model = st.selectbox(
        "Model",
        provider_models[provider],
        label_visibility="collapsed"
    )

    st.markdown(f"**🔑 {provider_key_names[provider]}**")
    api_key_input = st.text_input(
        "API Key",
        type="password",
        placeholder="Paste your API key here…",
        label_visibility="collapsed",
        help=f"Your {provider} API key. Stored only for this session."
    )

    st.markdown("---")
    st.markdown("**⚙️ Pipeline Options**")

    run_ml = st.checkbox("Run ML models", value=True,
                         help="Train and compare Logistic Regression, Random Forest, and Gradient Boosting")
    create_charts = st.checkbox("Generate charts", value=True,
                                help="Create correlation, distribution, and feature importance charts")
    verbose_logs = st.checkbox("Show agent logs", value=True,
                               help="Stream agent activity in real time")

    st.markdown("---")
    st.markdown(
        "<div style='font-size:0.75rem;color:#8b8fa3;line-height:1.6'>"
        "API keys are not stored or logged.<br>"
        "Each run creates a fresh <code>output/</code> folder."
        "</div>",
        unsafe_allow_html=True
    )


# ═════════════════════════════════════════════
#  MAIN LAYOUT
# ═════════════════════════════════════════════

st.markdown('<div class="main-title">🤖 AI Data Scientist</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-title">Upload any CSV — 4 autonomous agents will clean, analyse, model, and report on your data.</div>',
    unsafe_allow_html=True
)

# ── File Upload ──
uploaded_file = st.file_uploader(
    "Drop your CSV here",
    type=["csv"],
    help="Any CSV file. The agents will autonomously figure out what to do with it.",
    label_visibility="collapsed"
)

# ── Dataset Preview ──
if uploaded_file:
    raw_bytes = uploaded_file.read()
    enc = chardet.detect(raw_bytes[:50_000])["encoding"] or "utf-8"
    df_preview = pd.read_csv(StringIO(raw_bytes.decode(enc, errors="replace")))

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f'<div class="stat-box"><div class="stat-number">{df_preview.shape[0]:,}</div><div class="stat-label">Rows</div></div>', unsafe_allow_html=True)
    with col2:
        st.markdown(f'<div class="stat-box"><div class="stat-number">{df_preview.shape[1]}</div><div class="stat-label">Columns</div></div>', unsafe_allow_html=True)
    with col3:
        missing = df_preview.isnull().sum().sum()
        st.markdown(f'<div class="stat-box"><div class="stat-number">{missing:,}</div><div class="stat-label">Missing values</div></div>', unsafe_allow_html=True)
    with col4:
        num_cols = df_preview.select_dtypes(include="number").shape[1]
        st.markdown(f'<div class="stat-box"><div class="stat-number">{num_cols}</div><div class="stat-label">Numeric columns</div></div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    with st.expander("👀 Preview dataset (first 10 rows)", expanded=False):
        st.dataframe(df_preview.head(10), use_container_width=True)

    st.markdown("<br>", unsafe_allow_html=True)

# ── Run Button ──
can_run = uploaded_file is not None and (api_key_input or os.getenv(provider_key_names[provider]))
run_btn = st.button(
    "🚀 Run AI Agents",
    disabled=not can_run,
    help="Upload a CSV and provide an API key to run" if not can_run else "Start the agentic pipeline"
)

if not can_run and not uploaded_file:
    st.info("⬆️ Upload a CSV file to get started.")
elif not can_run and not api_key_input:
    st.warning(f"🔑 Add your {provider_key_names[provider]} in the sidebar to continue.")


# ═════════════════════════════════════════════
#  PIPELINE EXECUTION
# ═════════════════════════════════════════════

if run_btn and can_run:

    # Set API key for this session
    key_name = provider_key_names[provider]
    if api_key_input:
        os.environ[key_name] = api_key_input
        # For OpenAI specifically, only set if actually using OpenAI
        if provider != "OpenAI":
            os.environ["OPENAI_API_KEY"] = "sk-no-key-required"

    # Re-import with correct env
    import importlib
    import matplotlib
    matplotlib.use("Agg")

    # Save uploaded file to temp location
    with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp:
        tmp.write(raw_bytes)
        tmp_path = tmp.name

    OUTPUT_DIR = "output"
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ── Agent status placeholders ──
    st.markdown("---")
    st.markdown("### 🤖 Agent Activity")

    agent_cols = st.columns(4)
    agent_names = ["Data Engineer", "BI Analyst", "Data Scientist", "Strategist"]
    agent_icons = ["🔧", "📊", "🧠", "📋"]
    agent_placeholders = []
    for i, (col, name, icon) in enumerate(zip(agent_cols, agent_names, agent_icons)):
        with col:
            ph = st.empty()
            ph.markdown(
                f'<div class="agent-card">'
                f'<div class="agent-name">{icon} {name}</div>'
                f'<div class="agent-status">⏳ Waiting…</div>'
                f'</div>',
                unsafe_allow_html=True
            )
            agent_placeholders.append(ph)

    # ── Log area ──
    if verbose_logs:
        st.markdown("### 📡 Live Agent Logs")
        log_placeholder = st.empty()
        log_lines = []

        def update_log(line: str, level: str = "info"):
            css_class = {
                "info": "log-line-info",
                "ok": "log-line-ok",
                "warn": "log-line-warn",
                "error": "log-line-error",
            }.get(level, "")
            log_lines.append(f'<div class="{css_class}">{line}</div>')
            # Keep last 80 lines
            display = log_lines[-80:]
            log_placeholder.markdown(
                f'<div class="log-box">{"".join(display)}</div>',
                unsafe_allow_html=True
            )
    else:
        def update_log(line, level="info"):
            pass

    # ── Progress bar ──
    progress = st.progress(0, text="Initialising agents…")

    # ── Import app_v3 components ──
    update_log("Loading agentic pipeline…", "info")

    try:
        # Import the core pipeline from app_v3
        # We do a targeted import to reuse all the tools and crew logic
        import importlib.util, types

        # Load app_v3 as a module
        spec = importlib.util.spec_from_file_location(
            "app_v3",
            os.path.join(os.path.dirname(__file__), "app_v3.py")
        )
        app_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(app_module)

        # Override the model
        app_module.LLM_MODEL = selected_model
        app_module.OUTPUT_DIR = OUTPUT_DIR

        update_log(f"Model: {selected_model}", "info")
        update_log(f"Dataset: {uploaded_file.name} ({df_preview.shape[0]:,} rows × {df_preview.shape[1]} cols)", "info")

        # ── Load data into global df ──
        progress.progress(10, text="Loading dataset…")
        app_module._GLOBAL_DF = df_preview.copy()
        app_module._ORIGINAL_DF = df_preview.copy()

        update_log(f"✓ Dataset loaded — {df_preview.shape[0]:,} rows × {df_preview.shape[1]} cols", "ok")
        agent_placeholders[0].markdown(
            '<div class="agent-card active">'
            '<div class="agent-name">🔧 Data Engineer</div>'
            '<div class="agent-status">🔄 Running…</div>'
            '</div>', unsafe_allow_html=True
        )

        progress.progress(20, text="Building agent crew…")
        update_log("Building agentic crew…", "info")

        crew = app_module.build_agentic_crew()
        update_log("✓ 4 agents initialised with tools", "ok")
        progress.progress(30, text="Agents initialised — starting pipeline…")

        # ── Kick off crew ──
        update_log("🚀 Crew kickoff — agents are now working autonomously…", "info")

        max_retries = 3
        crew_output = None
        pipeline_error = None

        for attempt in range(1, max_retries + 1):
            try:
                # Update agent statuses as pipeline runs
                agent_placeholders[0].markdown(
                    '<div class="agent-card active">'
                    '<div class="agent-name">🔧 Data Engineer</div>'
                    '<div class="agent-status">🔄 Inspecting & cleaning…</div>'
                    '</div>', unsafe_allow_html=True
                )
                progress.progress(35, text="Data Engineer: inspecting dataset…")
                update_log(f"Attempt {attempt}/{max_retries} — running crew…", "info")

                crew_output = crew.kickoff()

                # Mark all agents done
                statuses = ["Cleaned dataset", "Analysed insights", "Trained ML models", "Wrote report"]
                for i, (ph, status) in enumerate(zip(agent_placeholders, statuses)):
                    ph.markdown(
                        f'<div class="agent-card done">'
                        f'<div class="agent-name">{agent_icons[i]} {agent_names[i]}</div>'
                        f'<div class="agent-status">✅ {status}</div>'
                        f'</div>', unsafe_allow_html=True
                    )

                progress.progress(100, text="✅ Pipeline complete!")
                update_log("✓ All agents completed successfully!", "ok")
                break

            except Exception as e:
                err_str = str(e)
                if ("503" in err_str or "rate_limit" in err_str.lower()) and attempt < max_retries:
                    wait = 30 * attempt
                    update_log(f"⚠️ API issue (attempt {attempt}). Retrying in {wait}s…", "warn")
                    progress.progress(30 + attempt * 5, text=f"Retrying in {wait}s…")
                    time.sleep(wait)
                else:
                    pipeline_error = err_str
                    update_log(f"❌ Pipeline error: {err_str[:200]}", "error")
                    for ph, name, icon in zip(agent_placeholders, agent_names, agent_icons):
                        ph.markdown(
                            f'<div class="agent-card failed">'
                            f'<div class="agent-name">{icon} {name}</div>'
                            f'<div class="agent-status">❌ Failed</div>'
                            f'</div>', unsafe_allow_html=True
                        )
                    break

        # Clean up temp file
        try:
            os.unlink(tmp_path)
        except Exception:
            pass

        # ══════════════════════════════════════
        #  RESULTS SECTION
        # ══════════════════════════════════════
        st.markdown("---")

        if pipeline_error and not crew_output:
            st.error(f"Pipeline failed: {pipeline_error}")
        else:
            st.markdown("## 📊 Results")

            # ── Charts tab ──
            chart_files = sorted([
                f for f in os.listdir(OUTPUT_DIR)
                if f.endswith(".png")
            ])

            if chart_files:
                st.markdown("### 🖼️ Agent-Generated Charts")
                chart_cols = st.columns(2)
                for i, cf in enumerate(chart_files):
                    with chart_cols[i % 2]:
                        img_path = os.path.join(OUTPUT_DIR, cf)
                        chart_name = os.path.splitext(cf)[0].replace("_", " ").title()
                        st.markdown(f"**{chart_name}**")
                        st.image(img_path, use_container_width=True)

            # ── Executive Report ──
            report_txt_path = os.path.join(OUTPUT_DIR, "executive_report.txt")
            report_html_path = os.path.join(OUTPUT_DIR, "report.html")

            if crew_output or os.path.exists(report_txt_path):
                st.markdown("### 📋 Executive Report")

                report_text = ""
                if os.path.exists(report_txt_path):
                    with open(report_txt_path, "r", encoding="utf-8") as f:
                        report_text = f.read()
                elif crew_output:
                    report_text = str(crew_output)

                if report_text:
                    st.markdown(
                        f'<div class="report-container">{report_text.replace(chr(10), "<br>")}</div>',
                        unsafe_allow_html=True
                    )

            # ── Download buttons ──
            st.markdown("### 💾 Download Results")
            dl_col1, dl_col2, dl_col3 = st.columns(3)

            with dl_col1:
                if os.path.exists(report_html_path):
                    with open(report_html_path, "rb") as f:
                        st.download_button(
                            "⬇️ Download HTML Report",
                            data=f.read(),
                            file_name="ai_data_scientist_report.html",
                            mime="text/html",
                        )

            with dl_col2:
                if os.path.exists(report_txt_path):
                    with open(report_txt_path, "rb") as f:
                        st.download_button(
                            "⬇️ Download Text Report",
                            data=f.read(),
                            file_name="executive_report.txt",
                            mime="text/plain",
                        )

            with dl_col3:
                # Zip all charts
                if chart_files:
                    import zipfile
                    import io
                    zip_buf = io.BytesIO()
                    with zipfile.ZipFile(zip_buf, "w") as zf:
                        for cf in chart_files:
                            zf.write(os.path.join(OUTPUT_DIR, cf), cf)
                    st.download_button(
                        "⬇️ Download All Charts",
                        data=zip_buf.getvalue(),
                        file_name="charts.zip",
                        mime="application/zip",
                    )

            st.success("✅ Analysis complete! Download your reports above.")

    except Exception as e:
        st.error(f"❌ Failed to run pipeline: {str(e)}")
        update_log(f"Fatal error: {str(e)}", "error")
        try:
            os.unlink(tmp_path)
        except Exception:
            pass


# ── Footer ──
st.markdown("---")
st.markdown(
    "<div style='text-align:center;color:#8b8fa3;font-size:0.8rem;font-family:Space Mono,monospace'>"
    "AI Data Scientist Agent v3.0 — Agentic Pipeline"
    "</div>",
    unsafe_allow_html=True
)