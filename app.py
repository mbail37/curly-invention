import io
import itertools
import math
from datetime import datetime
from difflib import SequenceMatcher

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from matplotlib.backends.backend_pdf import PdfPages


# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(
    page_title="CSV Insight Pro",
    page_icon="📊",
    layout="wide"
)


# -----------------------------
# STYLING
# -----------------------------
st.markdown(
    """
    <style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        margin-bottom: 0.2rem;
    }
    .subtle {
        color: #9aa0a6;
        font-size: 0.95rem;
    }
    .locked-card {
        border: 1px solid #444;
        border-radius: 14px;
        padding: 1.2rem;
        background: linear-gradient(180deg, #1f1f1f, #161616);
        color: #c9c9c9;
        margin-top: 1rem;
    }
    .section-card {
        border: 1px solid #333;
        border-radius: 12px;
        padding: 1rem;
        background: #111;
        margin-bottom: 1rem;
    }
    .metric-card {
        border: 1px solid #333;
        border-radius: 10px;
        padding: 0.8rem;
        background: #151515;
        text-align: center;
    }
    .small-note {
        color: #aaaaaa;
        font-size: 0.85rem;
    }
    </style>
    """,
    unsafe_allow_html=True
)


# -----------------------------
# HELPERS
# -----------------------------
def safe_ratio(a, b):
    if b in [0, None] or pd.isna(b):
        return np.nan
    return a / b


def format_pct(x):
    if pd.isna(x):
        return "N/A"
    return f"{x:.1%}"


def format_num(x):
    if pd.isna(x):
        return "N/A"
    if isinstance(x, (int, np.integer)):
        return f"{x:,}"
    if isinstance(x, (float, np.floating)):
        if abs(x) >= 1000:
            return f"{x:,.2f}"
        return f"{x:.2f}"
    return str(x)


def similarity(a, b):
    return SequenceMatcher(None, str(a).lower(), str(b).lower()).ratio()


def make_columns_unique(df: pd.DataFrame):
    """
    Clean whitespace and force unique column names.
    Important for cases where:
      "DuplicateCol" and "DuplicateCol "
    both become "DuplicateCol" after stripping.
    """
    original_cols = list(df.columns)
    cleaned_cols = []
    seen = {}
    rename_log = []

    for col in original_cols:
        base = str(col).strip()

        if base == "":
            base = "unnamed_column"

        if base in seen:
            seen[base] += 1
            new_name = f"{base}_{seen[base]}"
        else:
            seen[base] = 0
            new_name = base

        cleaned_cols.append(new_name)

        if str(col) != new_name:
            rename_log.append((str(col), new_name))
        elif seen[base] > 0:
            rename_log.append((str(col), new_name))

    out = df.copy()
    out.columns = cleaned_cols
    return out, rename_log


def profile_original_columns(raw_df: pd.DataFrame):
    """
    Inspect original columns before cleaning so we can explain what happened.
    """
    original = [str(c) for c in raw_df.columns]
    stripped = [c.strip() for c in original]

    original_duplicates = pd.Series(original).duplicated(keep=False)
    stripped_duplicates = pd.Series(stripped).duplicated(keep=False)

    report = pd.DataFrame({
        "original_name": original,
        "stripped_name": stripped,
        "duplicate_in_original": original_duplicates,
        "duplicate_after_strip": stripped_duplicates
    })
    return report


def read_file(uploaded_file):
    """
    Safe file reader for CSV / Excel / Parquet.
    Returns:
      raw_df, cleaned_df, metadata dict
    """
    file_name = uploaded_file.name.lower()

    try:
        if file_name.endswith(".csv"):
            raw_df = pd.read_csv(uploaded_file)
        elif file_name.endswith(".xlsx") or file_name.endswith(".xls"):
            raw_df = pd.read_excel(uploaded_file)
        elif file_name.endswith(".parquet"):
            raw_df = pd.read_parquet(uploaded_file)
        else:
            raise ValueError("Unsupported file type. Please upload CSV, Excel, or Parquet.")

        original_col_report = profile_original_columns(raw_df)
        cleaned_df, rename_log = make_columns_unique(raw_df)

        metadata = {
            "rename_log": rename_log,
            "original_column_report": original_col_report
        }
        return raw_df, cleaned_df, metadata

    except Exception as e:
        raise ValueError(f"Could not read file: {e}")


def detect_datetime_columns(df: pd.DataFrame):
    datetime_cols = []

    for col in df.columns:
        series = df[col]
        if isinstance(series, pd.DataFrame):
            series = series.iloc[:, 0]

        if pd.api.types.is_datetime64_any_dtype(series):
            datetime_cols.append(col)
            continue

        if series.dtype == "object":
            sample = series.dropna().astype(str).head(50)
            if len(sample) == 0:
                continue
            parsed = pd.to_datetime(sample, errors="coerce")
            if parsed.notna().mean() >= 0.7:
                datetime_cols.append(col)

    return datetime_cols


def coerce_datetime_columns(df: pd.DataFrame, datetime_cols):
    out = df.copy()
    for col in datetime_cols:
        try:
            out[col] = pd.to_datetime(out[col], errors="coerce")
        except Exception:
            pass
    return out


def get_column_types(df: pd.DataFrame):
    numeric_cols = list(df.select_dtypes(include=["number"]).columns)
    bool_cols = list(df.select_dtypes(include=["bool"]).columns)
    datetime_cols = detect_datetime_columns(df)

    excluded = set(numeric_cols + bool_cols + datetime_cols)
    categorical_cols = [c for c in df.columns if c not in excluded]

    return {
        "numeric": numeric_cols,
        "datetime": datetime_cols,
        "categorical": categorical_cols,
        "boolean": bool_cols
    }


def summarize_schema_issues(metadata):
    report = metadata["original_column_report"]
    rename_log = metadata["rename_log"]

    issues = []
    dup_after_strip = report[report["duplicate_after_strip"] == True]

    if len(dup_after_strip) > 0:
        issues.append(
            f"{dup_after_strip['stripped_name'].nunique()} column name collision(s) appeared after trimming whitespace."
        )

    if len(rename_log) > 0:
        issues.append(f"{len(rename_log)} column name(s) were normalized or auto-renamed.")

    blanks = (report["stripped_name"] == "").sum()
    if blanks > 0:
        issues.append(f"{blanks} blank column name(s) detected and renamed.")

    return issues


def compute_file_summary(df: pd.DataFrame):
    rows, cols = df.shape
    missing_cells = int(df.isna().sum().sum())
    duplicate_rows = int(df.duplicated().sum())

    dtypes = get_column_types(df)

    return {
        "rows": rows,
        "columns": cols,
        "missing_cells": missing_cells,
        "duplicate_rows": duplicate_rows,
        "numeric_cols": len(dtypes["numeric"]),
        "categorical_cols": len(dtypes["categorical"]),
        "datetime_cols": len(dtypes["datetime"]),
    }


def generate_single_file_insights(df: pd.DataFrame, metadata):
    insights = []
    summary = compute_file_summary(df)
    types = get_column_types(df)

    insights.append(f"The dataset contains {summary['rows']:,} rows and {summary['columns']:,} columns.")
    insights.append(
        f"It has {summary['missing_cells']:,} missing cells and {summary['duplicate_rows']:,} duplicate rows."
    )

    schema_issues = summarize_schema_issues(metadata)
    for issue in schema_issues:
        insights.append(f"Schema issue detected: {issue}")

    if types["numeric"]:
        missing_pct = df[types["numeric"]].isna().mean().sort_values(ascending=False)
        highest_missing = missing_pct.head(1)
        if len(highest_missing) > 0 and highest_missing.iloc[0] > 0:
            insights.append(
                f"The numeric column with the most missing values is '{highest_missing.index[0]}' at {highest_missing.iloc[0]:.1%} missing."
            )

        numeric_std = df[types["numeric"]].std(numeric_only=True).sort_values(ascending=False)
        if len(numeric_std) > 0:
            insights.append(
                f"'{numeric_std.index[0]}' appears to be the most volatile numeric field by standard deviation."
            )

        outlier_notes = []
        for col in types["numeric"][:10]:
            s = df[col].dropna()
            if len(s) < 8:
                continue
            q1 = s.quantile(0.25)
            q3 = s.quantile(0.75)
            iqr = q3 - q1
            if iqr == 0 or pd.isna(iqr):
                continue
            lower = q1 - 1.5 * iqr
            upper = q3 + 1.5 * iqr
            outlier_count = ((s < lower) | (s > upper)).sum()
            if outlier_count > 0:
                outlier_notes.append((col, outlier_count))

        if outlier_notes:
            outlier_notes.sort(key=lambda x: x[1], reverse=True)
            col, count = outlier_notes[0]
            insights.append(f"Potential outliers were detected in '{col}' ({count} row(s) outside IQR bounds).")

    if types["categorical"]:
        high_card = []
        for col in types["categorical"][:10]:
            nunique = df[col].nunique(dropna=True)
            high_card.append((col, nunique))
        high_card.sort(key=lambda x: x[1], reverse=True)

        if high_card:
            insights.append(
                f"'{high_card[0][0]}' is the highest-cardinality categorical field with {high_card[0][1]:,} unique values."
            )

    if types["datetime"]:
        dt_col = types["datetime"][0]
        dt_series = pd.to_datetime(df[dt_col], errors="coerce").dropna()
        if len(dt_series) > 1:
            insights.append(
                f"Date coverage runs from {dt_series.min().date()} to {dt_series.max().date()} based on '{dt_col}'."
            )

    return insights


def build_correlation_table(df: pd.DataFrame):
    numeric_cols = list(df.select_dtypes(include=["number"]).columns)
    if len(numeric_cols) < 2:
        return pd.DataFrame()

    corr = df[numeric_cols].corr(numeric_only=True)
    rows = []

    for i, c1 in enumerate(corr.columns):
        for c2 in corr.columns[i + 1:]:
            val = corr.loc[c1, c2]
            if pd.notna(val):
                rows.append({
                    "Column A": c1,
                    "Column B": c2,
                    "Correlation": val,
                    "Abs Correlation": abs(val)
                })

    out = pd.DataFrame(rows).sort_values("Abs Correlation", ascending=False)
    return out


def compare_two_dataframes(df1: pd.DataFrame, df2: pd.DataFrame):
    cols1 = list(df1.columns)
    cols2 = list(df2.columns)

    exact_matches = set(cols1).intersection(set(cols2))
    unmatched_1 = [c for c in cols1 if c not in exact_matches]
    unmatched_2 = [c for c in cols2 if c not in exact_matches]

    fuzzy_matches = []
    used_right = set()

    for left in unmatched_1:
        best_match = None
        best_score = 0.0

        for right in unmatched_2:
            if right in used_right:
                continue
            score = similarity(left, right)
            if score > best_score:
                best_score = score
                best_match = right

        if best_match is not None and best_score >= 0.65:
            fuzzy_matches.append({
                "left_column": left,
                "right_column": best_match,
                "similarity_score": round(best_score, 3)
            })
            used_right.add(best_match)

    fuzzy_left = {m["left_column"] for m in fuzzy_matches}
    fuzzy_right = {m["right_column"] for m in fuzzy_matches}

    left_only = [c for c in cols1 if c not in exact_matches and c not in fuzzy_left]
    right_only = [c for c in cols2 if c not in exact_matches and c not in fuzzy_right]

    numeric_drift_rows = []

    pairs = [(c, c, 1.0) for c in exact_matches]
    pairs += [(m["left_column"], m["right_column"], m["similarity_score"]) for m in fuzzy_matches]

    for left_col, right_col, score in pairs:
        s1 = df1[left_col]
        s2 = df2[right_col]

        if isinstance(s1, pd.DataFrame):
            s1 = s1.iloc[:, 0]
        if isinstance(s2, pd.DataFrame):
            s2 = s2.iloc[:, 0]

        if pd.api.types.is_numeric_dtype(s1) and pd.api.types.is_numeric_dtype(s2):
            mean1 = pd.to_numeric(s1, errors="coerce").mean()
            mean2 = pd.to_numeric(s2, errors="coerce").mean()
            std1 = pd.to_numeric(s1, errors="coerce").std()
            std2 = pd.to_numeric(s2, errors="coerce").std()

            pct_change = np.nan
            if pd.notna(mean1) and mean1 != 0:
                pct_change = (mean2 - mean1) / abs(mean1)

            numeric_drift_rows.append({
                "left_column": left_col,
                "right_column": right_col,
                "match_score": score,
                "left_mean": mean1,
                "right_mean": mean2,
                "mean_pct_change": pct_change,
                "left_std": std1,
                "right_std": std2
            })

    schema_overlap = safe_ratio(len(exact_matches) + len(fuzzy_matches), max(len(cols1), len(cols2)))

    result = {
        "schema_overlap": schema_overlap,
        "exact_matches": sorted(list(exact_matches)),
        "fuzzy_matches": pd.DataFrame(fuzzy_matches),
        "left_only": left_only,
        "right_only": right_only,
        "numeric_drift": pd.DataFrame(numeric_drift_rows).sort_values(
            "match_score", ascending=False
        ) if numeric_drift_rows else pd.DataFrame()
    }
    return result


def generate_pdf_report(df: pd.DataFrame, metadata, insights, title="CSV Insight Report"):
    pdf_buffer = io.BytesIO()

    with PdfPages(pdf_buffer) as pdf:
        # Page 1: Executive summary
        fig, ax = plt.subplots(figsize=(11, 8.5))
        ax.axis("off")

        summary = compute_file_summary(df)
        lines = [
            title,
            "",
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            f"Rows: {summary['rows']:,}",
            f"Columns: {summary['columns']:,}",
            f"Missing Cells: {summary['missing_cells']:,}",
            f"Duplicate Rows: {summary['duplicate_rows']:,}",
            f"Numeric Columns: {summary['numeric_cols']}",
            f"Categorical Columns: {summary['categorical_cols']}",
            f"Datetime Columns: {summary['datetime_cols']}",
            "",
            "Key Insights:",
        ]
        lines += [f"- {x}" for x in insights[:10]]

        y = 0.95
        for line in lines:
            ax.text(0.05, y, line, fontsize=12, va="top")
            y -= 0.045

        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        # Page 2: Schema issues / rename log
        fig, ax = plt.subplots(figsize=(11, 8.5))
        ax.axis("off")
        ax.text(0.05, 0.95, "Schema Normalization", fontsize=18, weight="bold", va="top")

        rename_log = metadata["rename_log"]
        issues = summarize_schema_issues(metadata)

        y = 0.88
        if issues:
            ax.text(0.05, y, "Detected Issues:", fontsize=13, weight="bold", va="top")
            y -= 0.05
            for issue in issues:
                ax.text(0.07, y, f"- {issue}", fontsize=11, va="top")
                y -= 0.04
        else:
            ax.text(0.05, y, "No major schema issues detected.", fontsize=12, va="top")
            y -= 0.05

        y -= 0.03
        ax.text(0.05, y, "Column Rename Log:", fontsize=13, weight="bold", va="top")
        y -= 0.05

        if rename_log:
            for old, new in rename_log[:20]:
                ax.text(0.07, y, f"{old}  →  {new}", fontsize=10, va="top")
                y -= 0.032
        else:
            ax.text(0.07, y, "No columns required renaming.", fontsize=11, va="top")

        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        # Page 3: Numeric distributions
        numeric_cols = list(df.select_dtypes(include=["number"]).columns)
        for col in numeric_cols[:4]:
            fig, ax = plt.subplots(figsize=(11, 8.5))
            s = pd.to_numeric(df[col], errors="coerce").dropna()

            if len(s) == 0:
                plt.close(fig)
                continue

            ax.hist(s, bins=30)
            ax.set_title(f"Distribution: {col}")
            ax.set_xlabel(col)
            ax.set_ylabel("Frequency")
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)

        # Page 4: Correlation table summary
        corr_table = build_correlation_table(df)
        if not corr_table.empty:
            fig, ax = plt.subplots(figsize=(11, 8.5))
            ax.axis("off")
            ax.text(0.05, 0.95, "Top Correlations", fontsize=18, weight="bold", va="top")

            top_corr = corr_table.head(15).copy()
            y = 0.88
            for _, row in top_corr.iterrows():
                line = f"{row['Column A']} ↔ {row['Column B']}: {row['Correlation']:.3f}"
                ax.text(0.05, y, line, fontsize=11, va="top")
                y -= 0.045

            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)

    pdf_buffer.seek(0)
    return pdf_buffer


def premium_locked_message(title, description, bullets):
    st.markdown(
        f"""
        <div class="locked-card">
            <h3 style="margin-top:0;">🔒 {title}</h3>
            <p>{description}</p>
        </div>
        """,
        unsafe_allow_html=True
    )
    for bullet in bullets:
        st.write(f"- {bullet}")


def premium_gate_enabled():
    return st.session_state.get("premium_unlocked", False)


# -----------------------------
# SIDEBAR
# -----------------------------
with st.sidebar:
    st.header("Controls")
    testing_unlock = st.toggle("Testing mode: unlock premium tabs", value=False)
    st.session_state["premium_unlocked"] = testing_unlock

    st.markdown("---")
    st.subheader("Premium Download")
    st.write("For testing, keep this behind a fake paywall and unlock it manually.")
    st.text_input("Placeholder payment link", value="https://buy.stripe.com/test_placeholder")

    st.markdown("---")
    st.caption("CSV Insight Pro — testing build")


# -----------------------------
# HEADER
# -----------------------------
st.markdown('<div class="main-title">CSV Insight Pro</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtle">Messy-data analytics, file comparison, and premium-report workflow testing.</div>',
    unsafe_allow_html=True
)
st.write("")


# -----------------------------
# TOP NAV
# -----------------------------
tab1, tab2, tab3 = st.tabs(["Single File Lab", "Compare Files", "Surprise Lab"])


# -----------------------------
# TAB 1 - SINGLE FILE
# -----------------------------
with tab1:
    st.subheader("Single File Lab")
    st.write("Upload one file and generate analytics, schema diagnostics, and a premium-style downloadable report.")

    uploaded = st.file_uploader(
        "Upload a CSV, Excel, or Parquet file",
        type=["csv", "xlsx", "xls", "parquet"],
        key="single_file_upload"
    )

    if uploaded is not None:
        try:
            raw_df, df, metadata = read_file(uploaded)
            df = coerce_datetime_columns(df, detect_datetime_columns(df))

            summary = compute_file_summary(df)
            insights = generate_single_file_insights(df, metadata)
            types = get_column_types(df)

            st.success("File loaded successfully.")

            # Core metrics
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Rows", f"{summary['rows']:,}")
            c2.metric("Columns", f"{summary['columns']:,}")
            c3.metric("Missing Cells", f"{summary['missing_cells']:,}")
            c4.metric("Duplicate Rows", f"{summary['duplicate_rows']:,}")

            # Schema issues
            with st.expander("Schema Issues & Column Normalization", expanded=True):
                issues = summarize_schema_issues(metadata)
                if issues:
                    for issue in issues:
                        st.warning(issue)
                else:
                    st.info("No major schema issues detected.")

                rename_log = metadata["rename_log"]
                if rename_log:
                    rename_df = pd.DataFrame(rename_log, columns=["Original", "Final"])
                    st.dataframe(rename_df, use_container_width=True)
                else:
                    st.write("No columns were renamed.")

                st.caption("Original-column inspection")
                st.dataframe(metadata["original_column_report"], use_container_width=True)

            # Preview
            with st.expander("Cleaned Data Preview", expanded=True):
                st.dataframe(df.head(50), use_container_width=True)

            # Type overview
            with st.expander("Column Type Summary", expanded=False):
                st.write({
                    "numeric": types["numeric"],
                    "datetime": types["datetime"],
                    "categorical": types["categorical"],
                    "boolean": types["boolean"]
                })

            # Insights
            st.markdown("### Key Insights")
            for insight in insights:
                st.write(f"- {insight}")

            # Numeric analysis
            if types["numeric"]:
                st.markdown("### Numeric Analysis")
                selected_numeric = st.selectbox(
                    "Select a numeric column to chart",
                    options=types["numeric"],
                    key="numeric_chart_select"
                )

                if selected_numeric:
                    s = pd.to_numeric(df[selected_numeric], errors="coerce").dropna()

                    if len(s) > 0:
                        fig, ax = plt.subplots(figsize=(8, 4))
                        ax.hist(s, bins=30)
                        ax.set_title(f"Distribution of {selected_numeric}")
                        ax.set_xlabel(selected_numeric)
                        ax.set_ylabel("Frequency")
                        st.pyplot(fig)

                        stats_df = pd.DataFrame({
                            "Statistic": ["Mean", "Median", "Std Dev", "Min", "Max"],
                            "Value": [
                                s.mean(),
                                s.median(),
                                s.std(),
                                s.min(),
                                s.max()
                            ]
                        })
                        st.dataframe(stats_df, use_container_width=True)

            # Correlations
            corr_table = build_correlation_table(df)
            if not corr_table.empty:
                st.markdown("### Top Correlations")
                st.dataframe(corr_table.head(20), use_container_width=True)

            # Time analysis
            if types["datetime"] and types["numeric"]:
                st.markdown("### Time-Series Snapshot")
                dt_col = st.selectbox("Datetime column", options=types["datetime"], key="dt_col_select")
                val_col = st.selectbox("Numeric metric", options=types["numeric"], key="ts_val_col_select")

                temp = df[[dt_col, val_col]].copy()
                temp[dt_col] = pd.to_datetime(temp[dt_col], errors="coerce")
                temp[val_col] = pd.to_numeric(temp[val_col], errors="coerce")
                temp = temp.dropna()

                if len(temp) > 1:
                    temp = temp.sort_values(dt_col)

                    fig, ax = plt.subplots(figsize=(9, 4))
                    ax.plot(temp[dt_col], temp[val_col])
                    ax.set_title(f"{val_col} over time")
                    ax.set_xlabel(dt_col)
                    ax.set_ylabel(val_col)
                    st.pyplot(fig)

            # Premium download gate
            st.markdown("### Premium Report")
            if premium_gate_enabled():
                pdf_data = generate_pdf_report(df, metadata, insights)
                st.download_button(
                    "Download Premium PDF Report",
                    data=pdf_data,
                    file_name="csv_insight_report.pdf",
                    mime="application/pdf"
                )
            else:
                premium_locked_message(
                    "Premium PDF Download",
                    "This is the paid deliverable. The free lab shows the analytics; the paid tier unlocks the polished export.",
                    [
                        "Executive summary report",
                        "Schema issue page",
                        "Distribution analysis pages",
                        "Correlation summary",
                        "Ready-to-send PDF artifact"
                    ]
                )

        except Exception as e:
            st.error(str(e))
    else:
        st.info("Upload a file to begin testing the single-file analytics flow.")


# -----------------------------
# TAB 2 - COMPARE FILES
# -----------------------------
with tab2:
    st.subheader("Compare Files")

    if not premium_gate_enabled():
        premium_locked_message(
            "Compare Two Files",
            "Designed for comparing similar files, but built not to fall apart when the files are only loosely related.",
            [
                "Exact and fuzzy schema matching",
                "Schema overlap scoring",
                "Left-only and right-only column detection",
                "Numeric drift comparison for matched fields",
                "Safer comparison workflow for messy real-world files"
            ]
        )
    else:
        st.write("Upload two files to compare structure and numeric drift.")

        left_file = st.file_uploader(
            "Upload first file",
            type=["csv", "xlsx", "xls", "parquet"],
            key="compare_left"
        )
        right_file = st.file_uploader(
            "Upload second file",
            type=["csv", "xlsx", "xls", "parquet"],
            key="compare_right"
        )

        if left_file is not None and right_file is not None:
            try:
                _, left_df, left_meta = read_file(left_file)
                _, right_df, right_meta = read_file(right_file)

                left_df = coerce_datetime_columns(left_df, detect_datetime_columns(left_df))
                right_df = coerce_datetime_columns(right_df, detect_datetime_columns(right_df))

                result = compare_two_dataframes(left_df, right_df)

                st.success("Both files loaded successfully.")

                c1, c2, c3 = st.columns(3)
                c1.metric("Schema Overlap", format_pct(result["schema_overlap"]))
                c2.metric("Exact Matches", len(result["exact_matches"]))
                c3.metric("Fuzzy Matches", 0 if result["fuzzy_matches"].empty else len(result["fuzzy_matches"]))

                with st.expander("Schema Normalization Logs", expanded=False):
                    st.markdown("**Left file rename log**")
                    if left_meta["rename_log"]:
                        st.dataframe(pd.DataFrame(left_meta["rename_log"], columns=["Original", "Final"]),
                                     use_container_width=True)
                    else:
                        st.write("No renames on left file.")

                    st.markdown("**Right file rename log**")
                    if right_meta["rename_log"]:
                        st.dataframe(pd.DataFrame(right_meta["rename_log"], columns=["Original", "Final"]),
                                     use_container_width=True)
                    else:
                        st.write("No renames on right file.")

                st.markdown("### Exact Matches")
                if result["exact_matches"]:
                    st.write(result["exact_matches"])
                else:
                    st.write("No exact column matches found.")

                st.markdown("### Fuzzy Matches")
                if not result["fuzzy_matches"].empty:
                    st.dataframe(result["fuzzy_matches"], use_container_width=True)
                else:
                    st.write("No fuzzy matches met the similarity threshold.")

                col1, col2 = st.columns(2)

                with col1:
                    st.markdown("### Left-Only Columns")
                    if result["left_only"]:
                        st.write(result["left_only"])
                    else:
                        st.write("None")

                with col2:
                    st.markdown("### Right-Only Columns")
                    if result["right_only"]:
                        st.write(result["right_only"])
                    else:
                        st.write("None")

                st.markdown("### Numeric Drift")
                if not result["numeric_drift"].empty:
                    drift_df = result["numeric_drift"].copy()
                    drift_df["mean_pct_change"] = drift_df["mean_pct_change"].apply(format_pct)
                    st.dataframe(drift_df, use_container_width=True)
                else:
                    st.write("No matched numeric columns were available for drift comparison.")

            except Exception as e:
                st.error(str(e))
        else:
            st.info("Upload both files to run the comparison.")


# -----------------------------
# TAB 3 - SURPRISE LAB
# -----------------------------
with tab3:
    st.subheader("Surprise Lab")

    if not premium_gate_enabled():
        premium_locked_message(
            "Surprise Lab",
            "This will become the 'wow' feature once the first two workspaces are stable.",
            [
                "A more opinionated, product-like analysis mode",
                "Likely focused on automatic issue detection",
                "Could surface 'what's wrong with your data' instantly",
                "Potential for strongest paid conversion hook"
            ]
        )
    else:
        st.info("Surprise Lab is intentionally held for the next phase once this testing build is stable.")
        st.write("For now, use this unlocked state to validate the premium-flow behavior.")
