# AI Data Scientist Agent — Agentic Multi-LLM Pipeline

> An end-to-end agentic AI system that automates data science workflows using a 4-agent sequential pipeline, multi-provider LLM support, and McKinsey-structured HTML report generation.

---

## Overview

This project builds a **production-style agentic data science system** that takes a raw dataset as input and autonomously executes the full analytical workflow: ingestion, exploratory analysis, machine learning modelling, and final report generation — with no manual intervention between steps.

The system is built on **CrewAI** for agent orchestration, **LiteLLM** for unified multi-provider LLM access, and **Streamlit** for an interactive front-end interface. It achieved **93%+ accuracy** on a 50,000-row real-world dataset.

---

## Architecture

```
Input Dataset (CSV)
        │
        ▼
┌─────────────────────┐
│   Agent 1           │  Data Ingestion & Validation
│   (Data Engineer)   │  → Loads, cleans, and profiles raw data
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│   Agent 2           │  Exploratory Data Analysis
│   (EDA Analyst)     │  → Statistical summaries, distributions, correlations
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│   Agent 3           │  ML Modelling & Evaluation
│   (ML Engineer)     │  → Feature engineering, model training, metric scoring
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│   Agent 4           │  Report Generation
│   (Analyst)         │  → McKinsey-structured HTML report with charts
└─────────┬───────────┘
          │
          ▼
   HTML Report Output
```

Each agent is **specialised** with its own system prompt, toolset, and output schema. Agents pass structured context forward through the pipeline — no agent re-does upstream work.

---

## Key Features

- **4-agent sequential pipeline** with clear separation of responsibilities
- **6 callable tools** distributed across agents (e.g. data profiler, correlation analyser, model trainer, chart generator)
- **Multi-provider LLM support** via LiteLLM — switch between Cerebras, Groq, Google Gemini, and OpenAI from a single config
- **93%+ accuracy** on a 50,000-row dataset
- **McKinsey-structured HTML report** auto-generated at pipeline end: Executive Summary → Key Findings → Model Results → Recommendations
- **Streamlit UI** for dataset upload, provider selection, and pipeline monitoring
- Fully modular — swap any agent, tool, or LLM provider independently

---

## Tech Stack

| Layer | Tools |
|---|---|
| Agent Orchestration | CrewAI |
| LLM Interface | LiteLLM |
| LLM Providers | Cerebras, Groq, Google Gemini, OpenAI |
| ML & Data | Scikit-learn, Pandas, Matplotlib |
| Frontend | Streamlit |
| Language | Python 3.10+ |

---

## Getting Started

### Prerequisites

```bash
python >= 3.10
```

### Installation

```bash
git clone https://github.com/hemangranjan/ai-data-scientist-agent.git
cd ai-data-scientist-agent
pip install -r requirements.txt
```

### Configuration

Create a `.env` file in the root directory:

```env
# Add the key(s) for whichever provider(s) you want to use
OPENAI_API_KEY=your_key_here
GOOGLE_API_KEY=your_key_here
GROQ_API_KEY=your_key_here
CEREBRAS_API_KEY=your_key_here
```

### Run

```bash
streamlit run app.py
```

Then upload a CSV dataset via the UI, select your LLM provider, and run the pipeline. The HTML report will be generated automatically on completion.

---

## Results

| Metric | Value |
|---|---|
| Dataset Size | 50,000 rows |
| Model Accuracy | 93%+ |
| Pipeline Agents | 4 |
| Callable Tools | 6 |
| LLM Providers Supported | 4 (Cerebras, Groq, Gemini, OpenAI) |
| Output Format | McKinsey-structured HTML Report |

---

## Project Structure

```
ai-data-scientist-agent/
│
├── agents/
│   ├── data_engineer.py       # Agent 1: ingestion & validation
│   ├── eda_analyst.py         # Agent 2: exploratory analysis
│   ├── ml_engineer.py         # Agent 3: modelling & evaluation
│   └── report_analyst.py      # Agent 4: report generation
│
├── tools/
│   ├── data_profiler.py
│   ├── correlation_analyser.py
│   ├── model_trainer.py
│   ├── chart_generator.py
│   └── ...
│
├── pipeline/
│   └── crew.py                # CrewAI pipeline orchestration
│
├── output/
│   └── report_template.html   # McKinsey-structured report template
│
├── app.py                     # Streamlit frontend
├── config.py                  # LLM provider configuration
├── requirements.txt
└── README.md
```

---

## Why This Project

Most ML projects stop at a Jupyter notebook. This project was built to explore what a **genuinely autonomous, production-oriented** data science pipeline looks like — where agents reason about data, make modelling decisions, and generate board-ready output without a human in the loop for each step.

The multi-LLM design via LiteLLM was intentional: different providers have different cost/speed/quality tradeoffs, and a real system should be provider-agnostic rather than locked to a single API.

---

## Author

**Hemang Ranjan**
MSc Business Analytics, Queen Mary University of London
[linkedin.com/in/hemangranjan](https://linkedin.com/in/hemangranjan)

---

## License

MIT License — free to use, adapt, and build on with attribution.
