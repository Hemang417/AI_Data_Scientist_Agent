"""
AI Data Scientist Agent — v3.0 Final
======================================
Truly agentic pipeline: agents autonomously inspect, clean,
analyse, model, and produce a McKinsey-grade strategic report.

Run:  python app_v3_final.py yourdata.csv
  or: streamlit run streamlit_app.py
"""

# ── Block OpenAI memory/telemetry calls BEFORE any other imports ──
import os
os.environ["OPENAI_API_KEY"]          = "sk-no-key-required"
os.environ["CREWAI_MEMORY_ENABLED"]   = "false"
os.environ["CREWAI_TRACING_ENABLED"]  = "false"
os.environ["CREWAI_TELEMETRY_OPT_OUT"]= "true"
os.environ["OTEL_SDK_DISABLED"]       = "true"

import sys
import re
import warnings
import chardet
import pandas as pd
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    roc_auc_score,
)

from crewai import Agent, Task, Crew, Process
from crewai.tools import tool
from dotenv import load_dotenv

load_dotenv()
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# CONFIG  — change LLM_MODEL to your provider
# ─────────────────────────────────────────────
LLM_MODEL  = "cerebras/gpt-oss-120b"   # cerebras / groq / gemini / openai
OUTPUT_DIR = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Global dataframe — agents operate on this via tools
_GLOBAL_DF:   pd.DataFrame | None = None
_ORIGINAL_DF: pd.DataFrame | None = None


# ═══════════════════════════════════════════════════════════
#  HELPER — trim long tool outputs to stay inside context
# ═══════════════════════════════════════════════════════════
def _trim(text: str, limit: int = 1500) -> str:
    text = str(text)
    if len(text) <= limit:
        return text
    return text[:limit] + f"\n… [truncated — {len(text) - limit} chars omitted]"


# ═══════════════════════════════════════════════════════════
#  AGENTIC TOOLS
#  Each @tool is a real capability agents call autonomously.
# ═══════════════════════════════════════════════════════════

@tool("Inspect Dataset")
def inspect_dataset(query: str) -> str:
    """
    Inspect the loaded dataset. Always call this before taking any action.

    Commands:
    - "shape"           → row and column count
    - "columns"         → all columns with dtype, unique count, null count
    - "head"            → first 5 rows
    - "describe"        → statistical summary (numeric columns)
    - "info"            → missing values per column
    - "unique <col>"    → top value counts for a column
    - "sample <n>"      → random n rows
    - "dtypes"          → data types
    - "correlations"    → correlation matrix (top 10 numeric columns)
    """
    global _GLOBAL_DF
    df = _GLOBAL_DF
    if df is None:
        return "Error: No dataset loaded."
    q = query.strip().lower()
    try:
        if q == "shape":
            return f"Dataset: {df.shape[0]:,} rows × {df.shape[1]} columns."

        elif q == "columns":
            lines = ["Columns:"]
            for col in df.columns:
                lines.append(
                    f"  • {col}: dtype={df[col].dtype}, "
                    f"unique={df[col].nunique()}, nulls={df[col].isna().sum()}"
                )
            return _trim("\n".join(lines))

        elif q == "head":
            return _trim(df.head().to_string())

        elif q == "describe":
            return _trim(df.describe(include="all").to_string())

        elif q == "info":
            null_info = df.isnull().sum()
            pct = (null_info / len(df) * 100).round(1)
            result = pd.DataFrame({"nulls": null_info, "pct_missing": pct})
            non_zero = result[result["nulls"] > 0]
            return _trim(non_zero.to_string() if len(non_zero) else "No missing values found.")

        elif q == "dtypes":
            return _trim(df.dtypes.to_string())

        elif q == "correlations":
            numeric = df.select_dtypes(include=["int64", "float64"])
            if numeric.shape[1] < 2:
                return "Not enough numeric columns for correlation."
            top = numeric.columns[:10]
            return _trim(numeric[top].corr().round(3).to_string())

        elif q.startswith("unique "):
            col = query.strip()[7:].strip()
            if col not in df.columns:
                return f"Column '{col}' not found."
            return _trim(f"Top values in '{col}':\n{df[col].value_counts().head(20).to_string()}")

        elif q.startswith("sample"):
            parts = q.split()
            n = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else 5
            return _trim(df.sample(min(n, len(df))).to_string())

        else:
            return "Unknown query. Use: shape, columns, head, describe, info, unique <col>, sample <n>, dtypes, correlations"

    except Exception as e:
        return f"Error: {e}"


@tool("Clean Data")
def clean_data(action: str) -> str:
    """
    Perform a cleaning action on the dataset. Call multiple times for multiple fixes.

    Actions:
    - "remove_duplicates"
    - "fill_median <col>"
    - "fill_mode <col>"
    - "fill_zero <col>"
    - "drop_column <col>"
    - "drop_high_null <threshold>"   e.g. "drop_high_null 60"
    - "strip_whitespace"
    - "convert_datetime <col>"
    - "cap_outliers <col>"
    - "auto_clean"                   → run all standard steps automatically
    """
    global _GLOBAL_DF
    df = _GLOBAL_DF
    if df is None:
        return "Error: No dataset loaded."
    a = action.strip().lower()
    try:
        if a == "remove_duplicates":
            before = len(df)
            _GLOBAL_DF = df.drop_duplicates()
            return f"Removed {before - len(_GLOBAL_DF)} duplicates. Shape: {_GLOBAL_DF.shape}"

        elif a.startswith("fill_median "):
            col = action.strip()[12:].strip()
            if col not in df.columns:
                return f"Column '{col}' not found."
            if df[col].dtype not in ("float64", "int64"):
                return f"'{col}' is not numeric — use fill_mode."
            n, med = df[col].isna().sum(), df[col].median()
            df[col].fillna(med, inplace=True)
            _GLOBAL_DF = df
            return f"Filled {n} nulls in '{col}' with median ({med:.2f})."

        elif a.startswith("fill_mode "):
            col = action.strip()[10:].strip()
            if col not in df.columns:
                return f"Column '{col}' not found."
            n = df[col].isna().sum()
            mode_val = df[col].mode()[0] if not df[col].mode().empty else "Unknown"
            df[col].fillna(mode_val, inplace=True)
            _GLOBAL_DF = df
            return f"Filled {n} nulls in '{col}' with mode ('{mode_val}')."

        elif a.startswith("fill_zero "):
            col = action.strip()[10:].strip()
            if col not in df.columns:
                return f"Column '{col}' not found."
            n = df[col].isna().sum()
            df[col].fillna(0, inplace=True)
            _GLOBAL_DF = df
            return f"Filled {n} nulls in '{col}' with 0."

        elif a.startswith("drop_column "):
            col = action.strip()[12:].strip()
            if col not in df.columns:
                return f"Column '{col}' not found."
            df.drop(columns=[col], inplace=True)
            _GLOBAL_DF = df
            return f"Dropped '{col}'. Remaining: {len(df.columns)} columns."

        elif a.startswith("drop_high_null"):
            parts = a.split()
            threshold = float(parts[1]) if len(parts) > 1 else 60
            dropped = []
            for col in list(df.columns):
                pct = df[col].isna().sum() / len(df) * 100
                if pct > threshold:
                    df.drop(columns=[col], inplace=True)
                    dropped.append(f"{col} ({pct:.0f}%)")
            _GLOBAL_DF = df
            return f"Dropped {len(dropped)} columns: {', '.join(dropped)}" if dropped \
                   else f"No columns exceed {threshold}% missing."

        elif a == "strip_whitespace":
            count = 0
            for col in df.select_dtypes(include="object").columns:
                df[col] = df[col].str.strip()
                count += 1
            _GLOBAL_DF = df
            return f"Stripped whitespace from {count} string columns."

        elif a.startswith("convert_datetime "):
            col = action.strip()[17:].strip()
            if col not in df.columns:
                return f"Column '{col}' not found."
            df[col] = pd.to_datetime(df[col], errors="coerce")
            _GLOBAL_DF = df
            return f"Converted '{col}' to datetime."

        elif a.startswith("cap_outliers "):
            col = action.strip()[13:].strip()
            if col not in df.columns:
                return f"Column '{col}' not found."
            if df[col].dtype not in ("float64", "int64"):
                return f"'{col}' is not numeric."
            p1, p99 = df[col].quantile(0.01), df[col].quantile(0.99)
            n = ((df[col] < p1) | (df[col] > p99)).sum()
            df[col] = df[col].clip(p1, p99)
            _GLOBAL_DF = df
            return f"Capped {n} outliers in '{col}' to [{p1:.2f}, {p99:.2f}]."

        elif a == "auto_clean":
            changes = []
            n_dup = df.duplicated().sum()
            if n_dup:
                df.drop_duplicates(inplace=True)
                changes.append(f"Removed {n_dup} duplicates")
            for col in list(df.columns):
                n_miss = df[col].isna().sum()
                if n_miss == 0:
                    continue
                pct = n_miss / len(df) * 100
                if pct > 60:
                    df.drop(columns=[col], inplace=True)
                    changes.append(f"Dropped '{col}' ({pct:.0f}% missing)")
                elif df[col].dtype in ("float64", "int64"):
                    med = df[col].median()
                    df[col].fillna(med, inplace=True)
                    changes.append(f"Filled '{col}' with median ({med:.2f})")
                else:
                    mode_val = df[col].mode()[0] if not df[col].mode().empty else "Unknown"
                    df[col].fillna(mode_val, inplace=True)
                    changes.append(f"Filled '{col}' with mode ('{mode_val}')")
            for col in df.select_dtypes(include="object").columns:
                df[col] = df[col].str.strip()
            changes.append("Stripped whitespace")
            _GLOBAL_DF = df
            return "Auto-clean complete:\n" + "\n".join(f"  • {c}" for c in changes) \
                   + f"\nFinal shape: {df.shape}"

        else:
            return f"Unknown action: '{action}'. See tool description."

    except Exception as e:
        return f"Error: {e}"


@tool("Analyse Column")
def analyse_column(column_name: str) -> str:
    """
    Deep-dive analysis of a single column: stats, distribution,
    and relationship with the target variable if one exists.
    Provide the exact column name.
    """
    global _GLOBAL_DF
    df = _GLOBAL_DF
    if df is None:
        return "Error: No dataset loaded."
    col = column_name.strip()
    if col not in df.columns:
        return f"Column '{col}' not found. Available: {df.columns.tolist()}"
    try:
        lines = [f"=== Analysis: '{col}' ===",
                 f"Type: {df[col].dtype}",
                 f"Non-null: {df[col].notna().sum():,} / {len(df):,}",
                 f"Unique: {df[col].nunique()}"]
        if df[col].dtype in ("float64", "int64"):
            lines += [
                f"Mean: {df[col].mean():.4f}",
                f"Median: {df[col].median():.4f}",
                f"Std: {df[col].std():.4f}",
                f"Min: {df[col].min():.4f}  Max: {df[col].max():.4f}",
                f"Skew: {df[col].skew():.4f}  Kurtosis: {df[col].kurtosis():.4f}",
            ]
            for tc in ["Churned","Churn","churn","Target","target","Exited","Attrition","Default","Fraud"]:
                if tc in df.columns and tc != col:
                    grouped = df.groupby(tc)[col].mean()
                    lines.append(f"\nMean '{col}' by '{tc}':")
                    for idx, val in grouped.items():
                        lines.append(f"  {tc}={idx}: {val:.4f}")
                    break
        else:
            lines.append(f"\nTop values:\n{df[col].value_counts().head(10).to_string()}")
        return _trim("\n".join(lines))
    except Exception as e:
        return f"Error: {e}"


@tool("Create Chart")
def create_chart(chart_spec: str) -> str:
    """
    Create and save a chart. Agent decides which chart best tells the story.

    Format: "<chart_type> <col1> [col2]"

    Types:
    - "histogram <col>"
    - "bar <col>"
    - "scatter <col1> <col2>"
    - "boxplot <col>"
    - "correlation"
    - "pie <col>"
    - "target_bars <col>"
    """
    global _GLOBAL_DF
    df = _GLOBAL_DF
    if df is None:
        return "Error: No dataset loaded."
    parts = chart_spec.strip().split()
    chart_type = parts[0].lower() if parts else ""
    sns.set_theme(style="whitegrid", palette="muted")
    try:
        if chart_type == "histogram" and len(parts) >= 2:
            col = " ".join(parts[1:])
            if col not in df.columns: return f"Column '{col}' not found."
            fig, ax = plt.subplots(figsize=(8, 5))
            sns.histplot(df[col], kde=True, ax=ax, color="#4C72B0")
            ax.set_title(f"Distribution of {col}", fontsize=13, fontweight="bold")
            plt.tight_layout()
            path = os.path.join(OUTPUT_DIR, f"hist_{col}.png")
            fig.savefig(path, dpi=150); plt.close(fig)
            return f"Histogram saved: {path}"

        elif chart_type == "bar" and len(parts) >= 2:
            col = " ".join(parts[1:])
            if col not in df.columns: return f"Column '{col}' not found."
            fig, ax = plt.subplots(figsize=(8, 5))
            df[col].value_counts().head(15).plot.barh(ax=ax, color="#4C72B0")
            ax.set_title(f"{col} — Value Counts", fontsize=13, fontweight="bold")
            ax.invert_yaxis(); plt.tight_layout()
            path = os.path.join(OUTPUT_DIR, f"bar_{col}.png")
            fig.savefig(path, dpi=150); plt.close(fig)
            return f"Bar chart saved: {path}"

        elif chart_type == "scatter" and len(parts) >= 3:
            col1, col2 = parts[1], parts[2]
            if col1 not in df.columns or col2 not in df.columns:
                return "Column not found."
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.scatter(df[col1], df[col2], alpha=0.3, s=10, color="#4C72B0")
            ax.set_xlabel(col1); ax.set_ylabel(col2)
            ax.set_title(f"{col1} vs {col2}", fontsize=13, fontweight="bold")
            plt.tight_layout()
            path = os.path.join(OUTPUT_DIR, f"scatter_{col1}_{col2}.png")
            fig.savefig(path, dpi=150); plt.close(fig)
            return f"Scatter plot saved: {path}"

        elif chart_type == "boxplot" and len(parts) >= 2:
            col = " ".join(parts[1:])
            if col not in df.columns: return f"Column '{col}' not found."
            fig, ax = plt.subplots(figsize=(8, 5))
            sns.boxplot(x=df[col], ax=ax, color="#4C72B0")
            ax.set_title(f"Box Plot: {col}", fontsize=13, fontweight="bold")
            plt.tight_layout()
            path = os.path.join(OUTPUT_DIR, f"box_{col}.png")
            fig.savefig(path, dpi=150); plt.close(fig)
            return f"Box plot saved: {path}"

        elif chart_type == "correlation":
            numeric = df.select_dtypes(include=["int64", "float64"])
            if numeric.shape[1] < 2: return "Not enough numeric columns."
            fig, ax = plt.subplots(figsize=(12, 10))
            sns.heatmap(numeric.corr(), annot=True, fmt=".2f", cmap="RdBu_r", center=0, ax=ax)
            ax.set_title("Correlation Matrix", fontsize=14, fontweight="bold")
            plt.tight_layout()
            path = os.path.join(OUTPUT_DIR, "correlation_matrix.png")
            fig.savefig(path, dpi=150); plt.close(fig)
            return f"Correlation heatmap saved: {path}"

        elif chart_type == "pie" and len(parts) >= 2:
            col = " ".join(parts[1:])
            if col not in df.columns: return f"Column '{col}' not found."
            fig, ax = plt.subplots(figsize=(7, 6))
            df[col].value_counts().head(8).plot.pie(
                autopct="%1.1f%%", ax=ax,
                colors=["#4C72B0","#DD8452","#55A868","#C44E52","#8172B3","#937860"]
            )
            ax.set_ylabel("")
            ax.set_title(f"{col} Distribution", fontsize=13, fontweight="bold")
            plt.tight_layout()
            path = os.path.join(OUTPUT_DIR, f"pie_{col}.png")
            fig.savefig(path, dpi=150); plt.close(fig)
            return f"Pie chart saved: {path}"

        elif chart_type == "target_bars" and len(parts) >= 2:
            col = " ".join(parts[1:])
            if col not in df.columns: return f"Column '{col}' not found."
            target = next(
                (tc for tc in ["Churned","Churn","churn","Target","Exited","Attrition"] if tc in df.columns),
                None
            )
            if not target: return "No target column found."
            fig, ax = plt.subplots(figsize=(8, 5))
            df.groupby(target)[col].mean().plot.bar(ax=ax, color=["#55A868","#C44E52"])
            ax.set_title(f"Mean {col} by {target}", fontsize=13, fontweight="bold")
            ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
            plt.tight_layout()
            path = os.path.join(OUTPUT_DIR, f"target_{col}.png")
            fig.savefig(path, dpi=150); plt.close(fig)
            return f"Target bar chart saved: {path}"

        else:
            return "Unknown chart type. Use: histogram, bar, scatter, boxplot, correlation, pie, target_bars"

    except Exception as e:
        return f"Error creating chart: {e}"


@tool("Train ML Model")
def train_ml_model(config: str) -> str:
    """
    Train ML models. Format: "<target_column> <model_type>"
    model_type: logistic | randomforest | gradientboosting | all
    Example: "Churned all"
    """
    global _GLOBAL_DF
    df = _GLOBAL_DF
    if df is None:
        return "Error: No dataset loaded."
    parts = config.strip().split()
    if len(parts) < 2:
        return "Format: '<target_column> <model_type>'. Example: 'Churned all'"
    target_col, model_type = parts[0], parts[1].lower()
    if target_col not in df.columns:
        return f"Target '{target_col}' not found. Columns: {df.columns.tolist()}"
    try:
        df_ml = df.copy()
        for col in df_ml.select_dtypes(include=["object","category"]).columns:
            if col == target_col:
                continue
            le = LabelEncoder()
            df_ml[col] = le.fit_transform(df_ml[col].astype(str))
        if df_ml[target_col].dtype == "object":
            df_ml[target_col] = LabelEncoder().fit_transform(df_ml[target_col])
        df_ml.dropna(inplace=True)
        X = df_ml.drop(columns=[target_col]).select_dtypes(include=["int64","float64","int32","float32"])
        y = df_ml[target_col].astype(int)
        if X.shape[1] == 0:
            return "No numeric features after encoding."
        X_scaled = pd.DataFrame(StandardScaler().fit_transform(X), columns=X.columns)
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42, stratify=y
        )
        model_map = {
            "logistic":        {"Logistic Regression": LogisticRegression(max_iter=1000)},
            "randomforest":    {"Random Forest": RandomForestClassifier(n_estimators=100, random_state=42)},
            "gradientboosting":{"Gradient Boosting": GradientBoostingClassifier(n_estimators=100, random_state=42)},
            "all": {
                "Logistic Regression": LogisticRegression(max_iter=1000),
                "Random Forest":       RandomForestClassifier(n_estimators=100, random_state=42),
                "Gradient Boosting":   GradientBoostingClassifier(n_estimators=100, random_state=42),
            }
        }
        if model_type not in model_map:
            return f"Unknown model type '{model_type}'. Use: logistic, randomforest, gradientboosting, all"
        models_to_train = model_map[model_type]
        lines = [f"=== ML Results (target: {target_col}) ==="]
        best_name, best_acc, best_obj = None, 0, None
        for name, model in models_to_train.items():
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            acc   = accuracy_score(y_test, preds)
            cv    = cross_val_score(model, X_scaled, y, cv=5, scoring="accuracy")
            try:
                auc = roc_auc_score(y_test, model.predict_proba(X_test)[:,1]) \
                      if hasattr(model, "predict_proba") else None
            except Exception:
                auc = None
            lines += [
                f"\n{name}:",
                f"  Accuracy: {acc:.4f} ({acc:.1%})",
                f"  CV: {cv.mean():.4f} ± {cv.std():.4f}",
                f"  AUC: {auc:.4f}" if auc else "  AUC: N/A",
                classification_report(y_test, preds),
            ]
            if acc > best_acc:
                best_acc, best_name, best_obj = acc, name, model
        lines.append(f"\n🏆 Best: {best_name} — Accuracy {best_acc:.1%}")
        # Feature importance chart
        if best_obj and hasattr(best_obj, "feature_importances_"):
            feat_imp = sorted(zip(X.columns, best_obj.feature_importances_), key=lambda x: -x[1])[:10]
            lines.append("\nTop 10 Feature Importances:")
            for fn, fv in feat_imp:
                lines.append(f"  {fn}: {fv:.4f}")
            fig, ax = plt.subplots(figsize=(8, 5))
            names_r = [f[0] for f in feat_imp][::-1]
            vals_r  = [f[1] for f in feat_imp][::-1]
            ax.barh(names_r, vals_r, color="#4C72B0")
            ax.set_title("Top Feature Importances", fontsize=13, fontweight="bold")
            ax.set_xlabel("Importance")
            plt.tight_layout()
            fig.savefig(os.path.join(OUTPUT_DIR, "feature_importance.png"), dpi=150)
            plt.close(fig)
        # Confusion matrix
        if best_obj:
            cm = confusion_matrix(y_test, best_obj.predict(X_test))
            fig, ax = plt.subplots(figsize=(5, 4))
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
            ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
            ax.set_title(f"Confusion Matrix — {best_name}", fontsize=12, fontweight="bold")
            plt.tight_layout()
            fig.savefig(os.path.join(OUTPUT_DIR, "confusion_matrix.png"), dpi=150)
            plt.close(fig)
        return _trim("\n".join(lines))
    except Exception as e:
        return f"Error: {e}"


@tool("Statistical Test")
def statistical_test(test_spec: str) -> str:
    """
    Run a statistical test. Format: "<test_type> <args>"

    Tests:
    - "ttest <numeric_col> <group_col>"    → independent t-test
    - "chisq <col1> <col2>"                → chi-squared test
    - "value_counts <col>"                 → counts + percentages
    - "group_stats <numeric_col> <group>"  → grouped descriptive stats
    """
    global _GLOBAL_DF
    df = _GLOBAL_DF
    if df is None:
        return "Error: No dataset loaded."
    parts = test_spec.strip().split()
    test_type = parts[0].lower() if parts else ""
    try:
        if test_type == "ttest" and len(parts) >= 3:
            num_col, grp_col = parts[1], parts[2]
            if num_col not in df.columns or grp_col not in df.columns:
                return "Column not found."
            groups = df[grp_col].unique()
            if len(groups) != 2:
                return f"T-test needs binary group. '{grp_col}' has {len(groups)} unique values."
            from scipy import stats
            g1 = df[df[grp_col] == groups[0]][num_col].dropna()
            g2 = df[df[grp_col] == groups[1]][num_col].dropna()
            t_stat, p_val = stats.ttest_ind(g1, g2)
            return (
                f"T-test: {num_col} by {grp_col}\n"
                f"  {groups[0]}: mean={g1.mean():.4f} (n={len(g1)})\n"
                f"  {groups[1]}: mean={g2.mean():.4f} (n={len(g2)})\n"
                f"  t={t_stat:.4f}, p={p_val:.6f}\n"
                f"  Significant (p<0.05): {'Yes' if p_val < 0.05 else 'No'}"
            )

        elif test_type == "chisq" and len(parts) >= 3:
            from scipy import stats
            col1, col2 = parts[1], parts[2]
            ct = pd.crosstab(df[col1], df[col2])
            chi2, p, dof, _ = stats.chi2_contingency(ct)
            return (
                f"Chi-squared: {col1} vs {col2}\n"
                f"  chi2={chi2:.4f}, p={p:.6f}, dof={dof}\n"
                f"  Significant: {'Yes' if p < 0.05 else 'No'}"
            )

        elif test_type == "value_counts" and len(parts) >= 2:
            col = " ".join(parts[1:])
            vc  = df[col].value_counts()
            pct = (vc / len(df) * 100).round(2)
            return _trim(pd.DataFrame({"count": vc, "percent": pct}).head(20).to_string())

        elif test_type == "group_stats" and len(parts) >= 3:
            num_col, grp_col = parts[1], parts[2]
            grouped = df.groupby(grp_col)[num_col].agg(["mean","median","std","min","max","count"])
            return _trim(f"Stats of '{num_col}' by '{grp_col}':\n{grouped.to_string()}")

        else:
            return "Unknown test. Use: ttest, chisq, value_counts, group_stats"

    except Exception as e:
        return f"Error: {e}"


@tool("Save Report")
def save_report(report_content: str) -> str:
    """
    Save the final McKinsey-grade strategic report.
    Pass the FULL report text — HTML is generated automatically with
    charts embedded alongside their business interpretations.
    """
    try:
        import base64

        # ── Save plain text ──
        txt_path = os.path.join(OUTPUT_DIR, "executive_report.txt")
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(report_content)

        # ── Load charts ──
        chart_files = sorted([f for f in os.listdir(OUTPUT_DIR) if f.endswith(".png")])

        def img_b64(path):
            with open(path, "rb") as f:
                return base64.b64encode(f.read()).decode()

        # ── Mini markdown → HTML converter ──
        def md_to_html(text: str) -> str:
            lines = text.split("\n")
            out, in_ul = [], False
            for line in lines:
                line = line.strip()
                if not line:
                    if in_ul:
                        out.append("</ul>"); in_ul = False
                    out.append("<br>")
                    continue
                if line.startswith("### "):
                    if in_ul: out.append("</ul>"); in_ul = False
                    out.append(f'<h3 class="sub-heading">{line[4:]}</h3>')
                elif re.match(r"^[A-Z][A-Z &]+:", line):
                    key, _, val = line.partition(":")
                    if in_ul: out.append("</ul>"); in_ul = False
                    out.append(f'<p class="kv-line"><span class="kv-key">{key}:</span>{val}</p>')
                elif line.startswith("- ") or line.startswith("* "):
                    if not in_ul:
                        out.append("<ul>"); in_ul = True
                    out.append(f"<li>{line[2:]}</li>")
                else:
                    if in_ul: out.append("</ul>"); in_ul = False
                    out.append(f"<p>{line}</p>")
            if in_ul:
                out.append("</ul>")
            return "\n".join(out)

        # ── Extract sections from report ──
        def section(heading: str) -> str:
            m = re.search(
                rf"##\s+{re.escape(heading)}\s*\n(.*?)(?=\n##\s|\Z)",
                report_content, re.DOTALL | re.IGNORECASE
            )
            return m.group(1).strip() if m else ""

        situation    = md_to_html(section("SITUATION"))
        complication = md_to_html(section("COMPLICATION"))
        metrics      = md_to_html(section("KEY METRICS DASHBOARD"))
        chart_interp = section("CHART INTERPRETATIONS")
        findings     = md_to_html(section("CRITICAL FINDINGS"))
        ml_sec       = md_to_html(section("PREDICTIVE INTELLIGENCE"))
        recs_raw     = section("STRATEGIC RECOMMENDATIONS")
        risks        = md_to_html(section("RISKS & WATCHPOINTS"))
        fallback     = md_to_html(report_content[:600])

        # ── Chart cards with matched interpretations ──
        chart_cards_html = ""
        for cf in chart_files:
            chart_name = os.path.splitext(cf)[0].replace("_", " ").title()
            b64 = img_b64(os.path.join(OUTPUT_DIR, cf))
            interp_html = ""
            # Try exact match first
            m = re.search(
                rf"###\s+{re.escape(chart_name)}.*?\n(.*?)(?=\n###|\Z)",
                chart_interp, re.DOTALL | re.IGNORECASE
            )
            if not m:
                # Fuzzy: first word of chart name
                first = chart_name.split()[0]
                if first:
                    m = re.search(
                        rf"###.*{re.escape(first)}.*?\n(.*?)(?=\n###|\Z)",
                        chart_interp, re.DOTALL | re.IGNORECASE
                    )
            if m:
                interp_html = md_to_html(m.group(1).strip())

            chart_cards_html += f"""
            <div class="chart-card">
                <div class="chart-img-wrap">
                    <img src="data:image/png;base64,{b64}" alt="{chart_name}">
                </div>
                <div class="chart-meta">
                    <h3 class="chart-title">{chart_name}</h3>
                    {interp_html if interp_html else
                     '<p class="chart-no-interp">Chart generated by BI Analyst agent.</p>'}
                </div>
            </div>"""

        # ── Recommendations with priority colour coding ──
        recs_html = ""
        pri_colors = {"HIGH": "#f87171", "MEDIUM": "#fbbf24", "LOW": "#34d399"}
        rec_blocks = re.split(r"###\s+Recommendation\s+\d+:", recs_raw, flags=re.IGNORECASE)
        for i, block in enumerate(rec_blocks[1:], 1):
            lines     = block.strip().split("\n")
            title     = lines[0].strip() if lines else f"Recommendation {i}"
            body      = md_to_html("\n".join(lines[1:]).strip())
            pri_m     = re.search(r"PRIORITY:\s*(HIGH|MEDIUM|LOW)", block, re.IGNORECASE)
            priority  = pri_m.group(1).upper() if pri_m else "MEDIUM"
            color     = pri_colors.get(priority, "#fbbf24")
            recs_html += f"""
            <div class="rec-card">
                <div class="rec-header">
                    <span class="rec-num">#{i}</span>
                    <span class="rec-title">{title}</span>
                    <span class="rec-badge" style="background:{color}20;color:{color};border-color:{color}40">{priority}</span>
                </div>
                <div class="rec-body">{body}</div>
            </div>"""

        # ── Full HTML ──
        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1.0">
<title>Strategic Analysis Report — AI Data Scientist</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=DM+Serif+Display&family=DM+Sans:wght@300;400;500;600&family=Space+Mono:wght@400;700&display=swap" rel="stylesheet">
<style>
:root{{
    --bg:#09090f;--surface:#111118;--card:#16161f;--card2:#1c1c28;
    --border:#232330;--border2:#2d2d3f;--text:#e8e8f0;--muted:#7878a0;
    --accent:#7c6fff;--accent2:#38c8f0;--success:#34d399;--warn:#fbbf24;--danger:#f87171;
    --serif:'DM Serif Display',Georgia,serif;
    --sans:'DM Sans',system-ui,sans-serif;
    --mono:'Space Mono',monospace;
}}
*{{margin:0;padding:0;box-sizing:border-box}}
body{{font-family:var(--sans);background:var(--bg);color:var(--text);line-height:1.75;font-size:15px}}
.page{{max-width:1200px;margin:0 auto;padding:0 2rem 4rem}}

.cover{{padding:5rem 0 4rem;border-bottom:1px solid var(--border);margin-bottom:3rem}}
.cover-label{{font-family:var(--mono);font-size:.72rem;letter-spacing:.15em;text-transform:uppercase;color:var(--accent);margin-bottom:1rem}}
.cover-title{{font-family:var(--serif);font-size:clamp(2.2rem,5vw,3.8rem);line-height:1.15;margin-bottom:1.5rem;max-width:800px}}
.cover-subtitle{{color:var(--muted);font-size:1.05rem;max-width:640px;line-height:1.7}}
.cover-meta{{display:flex;gap:2rem;margin-top:2.5rem;flex-wrap:wrap}}
.meta-pill{{font-family:var(--mono);font-size:.72rem;background:var(--card);border:1px solid var(--border2);padding:.3rem .9rem;border-radius:20px;color:var(--muted)}}

.section{{margin-bottom:3.5rem}}
.section-label{{font-family:var(--mono);font-size:.68rem;letter-spacing:.18em;text-transform:uppercase;color:var(--accent);margin-bottom:.5rem}}
.section-title{{font-family:var(--serif);font-size:1.7rem;margin-bottom:1.5rem;padding-bottom:.8rem;border-bottom:1px solid var(--border)}}

.narrative-block{{background:var(--card);border:1px solid var(--border2);border-left:3px solid var(--accent);border-radius:0 10px 10px 0;padding:1.5rem 2rem;margin-bottom:1rem;font-size:1rem;line-height:1.8}}
.narrative-block.complication{{border-left-color:var(--warn)}}
.narrative-block p{{margin-bottom:.5rem}}

.info-card{{background:var(--card);border:1px solid var(--border2);border-radius:12px;padding:1.4rem 1.8rem}}
.info-card p{{font-size:.9rem;margin-bottom:.4rem}}

.charts-grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(520px,1fr));gap:2rem}}
.chart-card{{background:var(--card);border:1px solid var(--border2);border-radius:14px;overflow:hidden}}
.chart-img-wrap{{background:#fff;padding:.5rem}}
.chart-img-wrap img{{width:100%;height:auto;display:block;border-radius:4px}}
.chart-meta{{padding:1.4rem 1.6rem}}
.chart-title{{font-family:var(--serif);font-size:1.1rem;color:var(--accent2);margin-bottom:.8rem}}
.chart-meta p{{font-size:.88rem;margin-bottom:.4rem;line-height:1.65}}
.chart-no-interp{{color:var(--muted);font-size:.82rem;font-style:italic}}

.finding-card{{background:var(--card);border:1px solid var(--border2);border-radius:12px;padding:1.4rem 1.6rem;margin-bottom:1rem}}
.finding-card p{{font-size:.9rem;margin-bottom:.4rem}}

.ml-block{{background:var(--card);border:1px solid var(--border2);border-radius:12px;padding:1.6rem 2rem}}
.ml-block p{{font-size:.9rem;margin-bottom:.4rem}}

.rec-card{{background:var(--card);border:1px solid var(--border2);border-radius:14px;margin-bottom:1.2rem;overflow:hidden}}
.rec-header{{display:flex;align-items:center;gap:1rem;padding:1rem 1.6rem;background:var(--card2);border-bottom:1px solid var(--border)}}
.rec-num{{font-family:var(--mono);font-size:.8rem;color:var(--muted);min-width:24px}}
.rec-title{{font-family:var(--serif);font-size:1.05rem;flex:1}}
.rec-badge{{font-family:var(--mono);font-size:.68rem;font-weight:700;letter-spacing:.1em;padding:.2rem .6rem;border-radius:4px;border:1px solid}}
.rec-body{{padding:1.2rem 1.6rem}}
.rec-body p{{font-size:.88rem;margin-bottom:.3rem}}
.rec-body ul{{margin-left:1.2rem;margin-bottom:.5rem}}
.rec-body li{{font-size:.88rem;margin-bottom:.25rem}}

.risks-block{{background:var(--card);border:1px solid #f8717120;border-left:3px solid var(--danger);border-radius:0 12px 12px 0;padding:1.5rem 2rem}}
.risks-block p{{font-size:.9rem;margin-bottom:.4rem}}
.risks-block ul{{margin-left:1.2rem}}
.risks-block li{{font-size:.88rem;margin-bottom:.3rem}}

.kv-key{{color:var(--accent);font-weight:600;font-size:.78rem;text-transform:uppercase;letter-spacing:.05em}}
.kv-line{{margin-bottom:.5rem;font-size:.9rem}}
.sub-heading{{font-family:var(--serif);font-size:1.05rem;color:var(--accent2);margin:1rem 0 .5rem}}
p{{margin-bottom:.4rem}}
ul{{margin-left:1.4rem;margin-bottom:.5rem}}
li{{margin-bottom:.25rem}}

footer{{text-align:center;padding:3rem 0 2rem;border-top:1px solid var(--border);color:var(--muted);font-family:var(--mono);font-size:.72rem;letter-spacing:.1em}}
</style>
</head>
<body>
<div class="page">

<div class="cover">
  <div class="cover-label">Strategic Intelligence Report</div>
  <div class="cover-title">AI Data Scientist — McKinsey-Grade Analysis</div>
  <div class="cover-subtitle">Autonomous multi-agent pipeline: data engineering, business intelligence, predictive modelling, and strategic recommendations.</div>
  <div class="cover-meta">
    <span class="meta-pill">AI Data Scientist v3.0 Final</span>
    <span class="meta-pill">Agentic Pipeline</span>
    <span class="meta-pill">{len(chart_files)} Charts Generated</span>
  </div>
</div>

<div class="section">
  <div class="section-label">01 — Context</div>
  <div class="section-title">Situation</div>
  <div class="narrative-block">{situation or fallback}</div>
</div>

<div class="section">
  <div class="section-label">02 — The Problem</div>
  <div class="section-title">Complication</div>
  <div class="narrative-block complication">{complication or "<p>See full report.</p>"}</div>
</div>

<div class="section">
  <div class="section-label">03 — Key Numbers</div>
  <div class="section-title">Metrics Dashboard</div>
  <div class="info-card">{metrics or "<p>See full report for metrics.</p>"}</div>
</div>

<div class="section">
  <div class="section-label">04 — Visual Evidence</div>
  <div class="section-title">Charts &amp; Business Interpretations</div>
  <div class="charts-grid">
    {chart_cards_html or "<p style='color:var(--muted)'>No charts in this run.</p>"}
  </div>
</div>

<div class="section">
  <div class="section-label">05 — What the Data Says</div>
  <div class="section-title">Critical Findings</div>
  <div class="finding-card">{findings or "<p>See full report.</p>"}</div>
</div>

<div class="section">
  <div class="section-label">06 — Predictive Intelligence</div>
  <div class="section-title">Machine Learning Results</div>
  <div class="ml-block">{ml_sec or "<p>See full report.</p>"}</div>
</div>

<div class="section">
  <div class="section-label">07 — What to Do Next</div>
  <div class="section-title">Strategic Recommendations</div>
  {recs_html or f'<div class="rec-card"><div class="rec-body">{md_to_html(recs_raw)}</div></div>'}
</div>

<div class="section">
  <div class="section-label">08 — Risk Register</div>
  <div class="section-title">Risks &amp; Watchpoints</div>
  <div class="risks-block">{risks or "<p>See full report.</p>"}</div>
</div>

</div>
<footer>GENERATED BY AI DATA SCIENTIST AGENT v3.0 FINAL &nbsp;·&nbsp; MCKINSEY-GRADE ANALYSIS</footer>
</body>
</html>"""

        html_path = os.path.join(OUTPUT_DIR, "report.html")
        with open(html_path, "w", encoding="utf-8") as f:
            f.write(html)

        return (
            f"McKinsey-grade report saved:\n"
            f"  • Text : {os.path.abspath(txt_path)}\n"
            f"  • HTML : {os.path.abspath(html_path)}\n"
            f"  • Charts embedded: {len(chart_files)}"
        )

    except Exception as e:
        return f"Error saving report: {e}"


# ═══════════════════════════════════════════════════════════
#  AGENT + CREW DEFINITIONS
# ═══════════════════════════════════════════════════════════

data_tools   = [inspect_dataset, clean_data, analyse_column, create_chart, statistical_test]
ml_tools     = [inspect_dataset, train_ml_model, analyse_column, create_chart, statistical_test]
report_tools = [inspect_dataset, create_chart, save_report]


def build_agentic_crew() -> Crew:

    data_engineer = Agent(
        role="Senior Data Engineer",
        goal="Inspect and clean the dataset using your tools. Fix nulls, duplicates, and type issues.",
        backstory="Expert data engineer. Always inspect first, then fix. Be concise.",
        tools=data_tools,
        llm=LLM_MODEL,
        verbose=True,
        allow_delegation=False,
        memory=False,
    )

    business_analyst = Agent(
        role="Lead Business Intelligence Analyst",
        goal=(
            "Find key business insights. Use correlations, column analysis, statistical tests, "
            "and charts. Every claim must be backed by numbers from your tool calls."
        ),
        backstory="Fortune 500 BI analyst who backs every insight with data. Creates clear charts.",
        tools=data_tools,
        llm=LLM_MODEL,
        verbose=True,
        allow_delegation=False,
        memory=False,
    )

    data_scientist = Agent(
        role="Principal Data Scientist",
        goal=(
            "Train ML models and interpret results in business terms. "
            "Explain what each feature importance means for the business."
        ),
        backstory="ML expert who translates model metrics into language executives act on.",
        tools=ml_tools,
        llm=LLM_MODEL,
        verbose=True,
        allow_delegation=False,
        memory=False,
    )

    strategist = Agent(
        role="McKinsey Senior Partner & Chief Strategy Consultant",
        goal=(
            "Write a McKinsey-grade executive brief using ALL findings. "
            "For EACH chart, write: what it shows, the business implication, "
            "and what decision a leader should make because of it. "
            "Every recommendation must cite specific numbers. "
            "Use the Save Report tool to save the final report."
        ),
        backstory=(
            "Senior Partner at McKinsey, 20 years advising Fortune 500 CEOs. "
            "Famous for brutal clarity and data-backed recommendations. "
            "Never writes vague suggestions — always specific actions with quantified impact. "
            "Structures reports as: Situation → Complication → Resolution. "
            "Explains every chart in terms of the business decision it drives."
        ),
        tools=report_tools,
        llm=LLM_MODEL,
        verbose=True,
        allow_delegation=False,
        memory=False,
    )

    # ── Tasks ──

    clean_task = Task(
        description=(
            "Inspect and clean the dataset. "
            "Use 'Inspect Dataset' with 'shape', 'columns', then 'info'. "
            "Fix all issues with 'Clean Data'. Confirm final shape."
        ),
        expected_output="Cleaning summary: every action taken and the final dataset shape.",
        agent=data_engineer,
    )

    analysis_task = Task(
        description=(
            "Analyse the dataset for business insights. "
            "Use 'correlations', analyse 2-3 key columns, run 1 statistical test, "
            "create at least 2 charts. Report top 3 KPIs with exact numbers."
        ),
        expected_output="Key business insights with specific numbers and chart filenames.",
        agent=business_analyst,
    )

    ml_task = Task(
        description=(
            "Find the target column using 'Inspect Dataset columns'. "
            "Train models using 'Train ML Model' with target and 'all'. "
            "Report best model accuracy, AUC, and top 5 feature importances "
            "with a plain-English explanation of what each feature means for the business."
        ),
        expected_output="Best model name, accuracy, AUC, and business interpretation of features.",
        agent=data_scientist,
    )

    report_task = Task(
        description=(
            "You are a McKinsey Senior Partner. Write a boardroom-grade strategic brief "
            "using ALL findings from previous agents.\n\n"
            "MANDATORY STRUCTURE — follow exactly:\n\n"
            "## SITUATION\n"
            "2-3 sentences: what this dataset represents and its business context.\n\n"
            "## COMPLICATION\n"
            "2-3 sentences: the core problem or opportunity revealed. Cite the most critical number.\n\n"
            "## KEY METRICS DASHBOARD\n"
            "Exactly 5 metrics:\n"
            "- METRIC NAME: [value] — [why this number matters]\n\n"
            "## CHART INTERPRETATIONS\n"
            "For EVERY chart generated, write:\n"
            "### [Chart Name]\n"
            "WHAT IT SHOWS: [1 sentence]\n"
            "BUSINESS IMPLICATION: [1-2 sentences on what this means]\n"
            "DECISION TRIGGER: [1 sentence — what specific action does this chart justify?]\n\n"
            "## CRITICAL FINDINGS\n"
            "Exactly 3 findings:\n"
            "- FINDING: [one sentence with a specific number]\n"
            "- EVIDENCE: [which tool result or chart proved this]\n"
            "- BUSINESS IMPACT: [revenue, cost, risk, or customer impact]\n\n"
            "## PREDICTIVE INTELLIGENCE\n"
            "- Best model, accuracy, AUC\n"
            "- Top 3 features in plain English\n"
            "- How should the business use this model operationally?\n\n"
            "## STRATEGIC RECOMMENDATIONS\n"
            "Exactly 5 recommendations in this format:\n"
            "### Recommendation [N]: [Short title]\n"
            "PRIORITY: High / Medium / Low\n"
            "THE ACTION: [Specific — not vague]\n"
            "THE DATA BEHIND IT: [Which finding or metric justifies this]\n"
            "EXPECTED IMPACT: [Quantified — %, $, customers, or time]\n"
            "TIMELINE: [immediate / 30 days / 90 days / 6 months]\n\n"
            "## RISKS & WATCHPOINTS\n"
            "3 risks, each with a mitigation.\n\n"
            "IMPORTANT: Use the Save Report tool to save the complete report."
        ),
        expected_output=(
            "A complete McKinsey-style strategic brief saved via Save Report, "
            "with chart interpretations, data-backed recommendations, and quantified impact."
        ),
        agent=strategist,
    )

    return Crew(
        agents=[data_engineer, business_analyst, data_scientist, strategist],
        tasks=[clean_task, analysis_task, ml_task, report_task],
        process=Process.sequential,
        verbose=True,
        memory=False,
        embedder=None,
    )


# ═══════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════

def main():
    global _GLOBAL_DF, _ORIGINAL_DF

    print("\n" + "=" * 60)
    print("  AI DATA SCIENTIST AGENT v3.0 FINAL — AGENTIC")
    print("  McKinsey-grade reports with chart interpretations")
    print("=" * 60 + "\n")

    file_path = sys.argv[1] if len(sys.argv) > 1 \
                else input("📂 CSV file path: ").strip().strip('"').strip("'")

    if not os.path.isfile(file_path):
        print(f"❌ File not found: {file_path}")
        sys.exit(1)

    with open(file_path, "rb") as f:
        enc = chardet.detect(f.read(50_000))["encoding"]

    _GLOBAL_DF    = pd.read_csv(file_path, encoding=enc)
    _ORIGINAL_DF  = _GLOBAL_DF.copy()
    print(f"✅ Loaded — {_GLOBAL_DF.shape[0]:,} rows × {_GLOBAL_DF.shape[1]} cols")
    print(f"   Columns: {_GLOBAL_DF.columns.tolist()}\n")

    crew = build_agentic_crew()

    import time
    crew_output = None
    for attempt in range(1, 4):
        try:
            crew_output = crew.kickoff()
            print("\n✅ Pipeline complete!")
            break
        except Exception as e:
            err = str(e)
            if ("503" in err or "rate_limit" in err.lower()) and attempt < 3:
                wait = 30 * attempt
                print(f"\n⚠️  API issue (attempt {attempt}/3). Retrying in {wait}s…")
                time.sleep(wait)
            else:
                print(f"\n❌ Pipeline failed: {err}")
                break

    if crew_output:
        print(f"\n{'='*60}\n  FINAL OUTPUT\n{'='*60}\n{crew_output}")

    print(f"\n📁 Outputs: {os.path.abspath(OUTPUT_DIR)}/")
    print("   Open output/report.html in your browser.\n")


if __name__ == "__main__":
    main()