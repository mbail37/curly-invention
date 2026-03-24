import io
import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

st.set_page_config(page_title='CSV Insight Pro', layout='wide')

MAX_PREVIEW_ROWS = 50000


@dataclass
class ColumnProfile:
    name: str
    dtype: str
    non_null_pct: float
    unique_pct: float
    missing_count: int
    inferred_role: str


def safe_read_file(uploaded_file, sample_rows: Optional[int] = None) -> pd.DataFrame:
    name = uploaded_file.name.lower()
    if name.endswith('.csv'):
        return pd.read_csv(uploaded_file, nrows=sample_rows)
    if name.endswith('.xlsx') or name.endswith('.xls'):
        return pd.read_excel(uploaded_file, nrows=sample_rows)
    if name.endswith('.parquet'):
        return pd.read_parquet(uploaded_file)
    raise ValueError('Unsupported file type. Upload CSV, Excel, or Parquet.')


def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [
        str(c).strip().lower().replace(' ', '_').replace('-', '_').replace('/', '_')
        for c in out.columns
    ]
    return out


def coerce_datetime_columns(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    out = df.copy()
    converted = []
    for col in out.columns:
        if out[col].dtype == 'object':
            sample = out[col].dropna().astype(str).head(25)
            if sample.empty:
                continue
            parsed = pd.to_datetime(sample, errors='coerce', utc=False)
            if parsed.notna().mean() >= 0.7:
                out[col] = pd.to_datetime(out[col], errors='coerce', utc=False)
                converted.append(col)
    return out, converted


def infer_role(series: pd.Series) -> str:
    if pd.api.types.is_datetime64_any_dtype(series):
        return 'datetime'
    if pd.api.types.is_numeric_dtype(series):
        unique_ratio = series.nunique(dropna=True) / max(len(series), 1)
        if unique_ratio < 0.03:
            return 'discrete_numeric'
        return 'numeric'
    unique_ratio = series.nunique(dropna=True) / max(len(series), 1)
    avg_len = series.dropna().astype(str).str.len().mean() if series.dropna().size else 0
    if unique_ratio < 0.08:
        return 'categorical'
    if avg_len > 25:
        return 'text'
    return 'id_or_text'


def build_profile(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    n = max(len(df), 1)
    for col in df.columns:
        s = df[col]
        rows.append(
            ColumnProfile(
                name=col,
                dtype=str(s.dtype),
                non_null_pct=round(100 * s.notna().mean(), 2),
                unique_pct=round(100 * s.nunique(dropna=True) / n, 2),
                missing_count=int(s.isna().sum()),
                inferred_role=infer_role(s),
            ).__dict__
        )
    return pd.DataFrame(rows)


def detect_target_candidates(df: pd.DataFrame) -> List[str]:
    numeric = df.select_dtypes(include=np.number).columns.tolist()
    ranked = []
    for col in numeric:
        s = df[col].dropna()
        if s.empty:
            continue
        score = s.std() * (1 + np.log1p(abs(s.mean())))
        ranked.append((col, score))
    ranked.sort(key=lambda x: x[1], reverse=True)
    return [c for c, _ in ranked[:12]]


def anomaly_summary(df: pd.DataFrame, numeric_cols: List[str]) -> pd.DataFrame:
    rows = []
    for col in numeric_cols:
        s = pd.to_numeric(df[col], errors='coerce').dropna()
        if len(s) < 8:
            continue
        q1, q3 = s.quantile(0.25), s.quantile(0.75)
        iqr = q3 - q1
        if iqr == 0:
            outliers = 0
        else:
            lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
            outliers = ((s < lower) | (s > upper)).sum()
        rows.append({
            'column': col,
            'mean': round(float(s.mean()), 4),
            'std': round(float(s.std()), 4),
            'min': round(float(s.min()), 4),
            'max': round(float(s.max()), 4),
            'outlier_count': int(outliers),
            'outlier_pct': round(100 * outliers / len(s), 2),
        })
    return pd.DataFrame(rows).sort_values(['outlier_pct', 'std'], ascending=False) if rows else pd.DataFrame()


def correlation_table(df: pd.DataFrame) -> pd.DataFrame:
    numeric_df = df.select_dtypes(include=np.number)
    if numeric_df.shape[1] < 2:
        return pd.DataFrame()
    corr = numeric_df.corr(numeric_only=True)
    rows = []
    cols = corr.columns.tolist()
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            rows.append({
                'col_a': cols[i],
                'col_b': cols[j],
                'correlation': round(float(corr.iloc[i, j]), 4)
            })
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out['abs_corr'] = out['correlation'].abs()
    return out.sort_values('abs_corr', ascending=False)


def segment_performance(df: pd.DataFrame, target: str, cat_cols: List[str]) -> pd.DataFrame:
    rows = []
    for col in cat_cols[:8]:
        subset = df[[col, target]].dropna()
        if subset.empty or subset[col].nunique() < 2 or subset[col].nunique() > 25:
            continue
        agg = subset.groupby(col)[target].agg(['count', 'mean', 'median', 'sum']).reset_index()
        agg.insert(0, 'segment_column', col)
        rows.append(agg)
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


def detect_time_series_view(df: pd.DataFrame, dt_cols: List[str], target: Optional[str]) -> Optional[pd.DataFrame]:
    if not dt_cols or not target:
        return None
    dt_col = dt_cols[0]
    temp = df[[dt_col, target]].dropna().sort_values(dt_col)
    if temp.empty:
        return None
    return temp.groupby(pd.Grouper(key=dt_col, freq='D'))[target].mean().reset_index()


def generate_insights(df: pd.DataFrame, profile: pd.DataFrame, target: Optional[str], corr_df: pd.DataFrame, anomaly_df: pd.DataFrame, dt_cols: List[str]) -> List[str]:
    insights = []
    rows, cols = df.shape
    insights.append(f'This dataset has {rows:,} rows and {cols:,} columns.')

    missing_cols = profile.sort_values('missing_count', ascending=False)
    if not missing_cols.empty and missing_cols.iloc[0]['missing_count'] > 0:
        top = missing_cols.iloc[0]
        insights.append(f"{top['name']} has the most missing values at {top['missing_count']:,} rows ({100-top['non_null_pct']:.2f}% missing).")

    role_counts = profile['inferred_role'].value_counts().to_dict()
    insights.append('Detected column roles: ' + ', '.join([f'{k}={v}' for k, v in role_counts.items()]))

    if target and target in df.columns:
        s = pd.to_numeric(df[target], errors='coerce').dropna()
        if not s.empty:
            insights.append(f'{target} ranges from {s.min():,.2f} to {s.max():,.2f} with an average of {s.mean():,.2f}.')
            if s.std() > 0:
                insights.append(f'{target} has coefficient of variation {s.std()/max(abs(s.mean()),1e-9):.2f}, useful for understanding volatility.')

    if not corr_df.empty:
        top_corr = corr_df.iloc[0]
        insights.append(f"Strongest numeric relationship: {top_corr['col_a']} vs {top_corr['col_b']} with correlation {top_corr['correlation']:.2f}.")

    if not anomaly_df.empty:
        top_anom = anomaly_df.iloc[0]
        if top_anom['outlier_count'] > 0:
            insights.append(f"{top_anom['column']} shows the most outliers: {int(top_anom['outlier_count'])} values ({top_anom['outlier_pct']:.2f}%).")

    if dt_cols and target:
        ts = detect_time_series_view(df, dt_cols, target)
        if ts is not None and len(ts) >= 3:
            y = ts[target].dropna()
            if len(y) >= 2:
                delta = y.iloc[-1] - y.iloc[0]
                direction = 'up' if delta > 0 else 'down' if delta < 0 else 'flat'
                insights.append(f'Time trend for {target} appears {direction} over the available date range.')

    duplicates = int(df.duplicated().sum())
    if duplicates:
        insights.append(f'There are {duplicates:,} fully duplicated rows worth checking before downstream analysis.')
    return insights


def plot_histogram(df: pd.DataFrame, col: str):
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(pd.to_numeric(df[col], errors='coerce').dropna(), bins=30)
    ax.set_title(f'Distribution of {col}')
    ax.set_xlabel(col)
    ax.set_ylabel('Frequency')
    return fig


def plot_time_series(ts_df: pd.DataFrame, dt_col: str, target: str):
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(ts_df[dt_col], ts_df[target])
    ax.set_title(f'{target} over time')
    ax.set_xlabel(dt_col)
    ax.set_ylabel(target)
    return fig


def plot_segments(seg_df: pd.DataFrame, segment_col: str):
    fig, ax = plt.subplots(figsize=(8, 4))
    subset = seg_df[seg_df['segment_column'] == segment_col].sort_values('mean', ascending=False).head(10)
    ax.bar(subset.iloc[:, 1].astype(str), subset['mean'])
    ax.set_title(f'Average target by {segment_col}')
    ax.set_xlabel(segment_col)
    ax.set_ylabel('Mean')
    plt.xticks(rotation=35, ha='right')
    return fig


def build_pdf_report(df: pd.DataFrame, profile: pd.DataFrame, insights: List[str], target: Optional[str], corr_df: pd.DataFrame, anomaly_df: pd.DataFrame, ts_df: Optional[pd.DataFrame], dt_col: Optional[str], segment_df: pd.DataFrame) -> bytes:
    buf = io.BytesIO()
    with PdfPages(buf) as pdf:
        fig = plt.figure(figsize=(8.27, 11.69))
        fig.clf()
        ax = fig.add_axes([0.08, 0.05, 0.84, 0.9])
        ax.axis('off')
        y = 0.98
        ax.text(0, y, 'CSV Insight Pro Report', fontsize=18, weight='bold', va='top')
        y -= 0.05
        ax.text(0, y, f'Rows: {len(df):,} | Columns: {len(df.columns):,}', fontsize=11, va='top')
        y -= 0.04
        ax.text(0, y, 'Executive summary', fontsize=14, weight='bold', va='top')
        y -= 0.03
        for item in insights[:10]:
            ax.text(0.02, y, f'• {item}', fontsize=10, va='top', wrap=True)
            y -= 0.05
            if y < 0.18:
                break
        y -= 0.01
        ax.text(0, y, 'Top columns by missing values', fontsize=14, weight='bold', va='top')
        y -= 0.03
        top_missing = profile.sort_values('missing_count', ascending=False).head(8)[['name', 'dtype', 'missing_count', 'inferred_role']]
        ax.table(cellText=top_missing.values, colLabels=top_missing.columns, bbox=[0, max(0.02, y-0.28), 0.95, 0.25])
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)

        if target and pd.api.types.is_numeric_dtype(df[target]):
            fig = plot_histogram(df, target)
            pdf.savefig(fig, bbox_inches='tight')
            plt.close(fig)

        if ts_df is not None and dt_col and target:
            fig = plot_time_series(ts_df, dt_col, target)
            pdf.savefig(fig, bbox_inches='tight')
            plt.close(fig)

        if not corr_df.empty:
            fig = plt.figure(figsize=(8.27, 11.69))
            ax = fig.add_axes([0.05, 0.05, 0.9, 0.9])
            ax.axis('off')
            ax.text(0, 0.98, 'Top correlations', fontsize=16, weight='bold', va='top')
            table_df = corr_df.head(20)[['col_a', 'col_b', 'correlation']]
            ax.table(cellText=table_df.values, colLabels=table_df.columns, bbox=[0, 0.05, 1, 0.88])
            pdf.savefig(fig, bbox_inches='tight')
            plt.close(fig)

        if not anomaly_df.empty:
            fig = plt.figure(figsize=(8.27, 11.69))
            ax = fig.add_axes([0.05, 0.05, 0.9, 0.9])
            ax.axis('off')
            ax.text(0, 0.98, 'Outlier summary', fontsize=16, weight='bold', va='top')
            table_df = anomaly_df.head(20)
            ax.table(cellText=table_df.values, colLabels=table_df.columns, bbox=[0, 0.05, 1, 0.88])
            pdf.savefig(fig, bbox_inches='tight')
            plt.close(fig)

        if not segment_df.empty:
            for seg_col in segment_df['segment_column'].dropna().unique()[:3]:
                fig = plot_segments(segment_df, seg_col)
                pdf.savefig(fig, bbox_inches='tight')
                plt.close(fig)
    buf.seek(0)
    return buf.read()


def maybe_filter_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    with st.expander('Optional filters', expanded=False):
        filtered = df.copy()
        cat_cols = [c for c in filtered.columns if infer_role(filtered[c]) in ('categorical', 'discrete_numeric') and filtered[c].nunique(dropna=True) <= 25]
        chosen_filters = {}
        for col in cat_cols[:6]:
            values = sorted(filtered[col].dropna().astype(str).unique().tolist())
            sel = st.multiselect(f'Filter {col}', values, default=values)
            chosen_filters[col] = sel
        for col, sel in chosen_filters.items():
            if sel and len(sel) != filtered[col].dropna().astype(str).nunique():
                filtered = filtered[filtered[col].astype(str).isin(sel)]
        return filtered
    return df


st.title('CSV Insight Pro')
st.caption('Turn messy datasets into report-ready analytics someone would actually pay for.')

with st.sidebar:
    st.header('Settings')
    sample_mode = st.toggle('Sample very large files for speed', value=True)
    sample_rows = st.number_input('Rows to load when sampling', min_value=5000, max_value=200000, value=50000, step=5000)
    standardize = st.toggle('Standardize column names', value=True)
    st.markdown('**Monetization stub**')
    stripe_link = st.text_input('Stripe payment link', value='https://buy.stripe.com/test_placeholder')
    st.markdown(f"[Premium download link]({stripe_link})")

uploaded = st.file_uploader('Upload CSV, Excel, or Parquet', type=['csv', 'xlsx', 'xls', 'parquet'])

if uploaded:
    try:
        df = safe_read_file(uploaded, sample_rows if sample_mode else None)
        if standardize:
            df = standardize_columns(df)
        df, auto_dt = coerce_datetime_columns(df)
    except Exception as e:
        st.error(f'Could not read file: {e}')
        st.stop()

    st.success(f'Loaded {len(df):,} rows and {len(df.columns):,} columns from {uploaded.name}')
    if sample_mode:
        st.info('Sampling mode is on. That makes large files usable in-browser, but some analytics are based on the sampled rows only.')

    filtered_df = maybe_filter_dataframe(df)
    profile = build_profile(filtered_df)

    numeric_cols = filtered_df.select_dtypes(include=np.number).columns.tolist()
    dt_cols = [c for c in filtered_df.columns if pd.api.types.is_datetime64_any_dtype(filtered_df[c])]
    cat_cols = [c for c in filtered_df.columns if infer_role(filtered_df[c]) in ('categorical', 'discrete_numeric')]

    target_candidates = detect_target_candidates(filtered_df)
    target = st.selectbox('Primary metric to analyze', options=[''] + target_candidates, index=1 if target_candidates else 0)
    target = target or None

    corr_df = correlation_table(filtered_df)
    anomaly_df = anomaly_summary(filtered_df, numeric_cols)
    segment_df = segment_performance(filtered_df, target, cat_cols) if target else pd.DataFrame()
    ts_df = detect_time_series_view(filtered_df, dt_cols, target) if target else None
    insights = generate_insights(filtered_df, profile, target, corr_df, anomaly_df, dt_cols)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric('Rows', f'{len(filtered_df):,}')
    c2.metric('Columns', f'{len(filtered_df.columns):,}')
    c3.metric('Missing cells', f'{int(filtered_df.isna().sum().sum()):,}')
    c4.metric('Duplicate rows', f'{int(filtered_df.duplicated().sum()):,}')

    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        'Executive Summary', 'Schema', 'Correlations', 'Anomalies', 'Segmentation', 'Data Preview'
    ])

    with tab1:
        st.subheader('Executive summary')
        for item in insights:
            st.write(f'• {item}')
        if target and target in numeric_cols:
            st.pyplot(plot_histogram(filtered_df, target))
        if ts_df is not None and dt_cols and target:
            st.pyplot(plot_time_series(ts_df, dt_cols[0], target))

    with tab2:
        st.subheader('Schema profile')
        st.dataframe(profile, use_container_width=True)

    with tab3:
        st.subheader('Top correlations')
        if corr_df.empty:
            st.info('Need at least two numeric columns for correlation analysis.')
        else:
            st.dataframe(corr_df.head(50), use_container_width=True)

    with tab4:
        st.subheader('Outlier and volatility scan')
        if anomaly_df.empty:
            st.info('No numeric columns with enough data to score anomalies.')
        else:
            st.dataframe(anomaly_df.head(50), use_container_width=True)

    with tab5:
        st.subheader('Segment performance')
        if segment_df.empty:
            st.info('Pick a primary metric and include at least one low-cardinality categorical column.')
        else:
            st.dataframe(segment_df, use_container_width=True)
            for seg_col in segment_df['segment_column'].dropna().unique()[:3]:
                st.pyplot(plot_segments(segment_df, seg_col))

    with tab6:
        st.subheader('Preview')
        st.dataframe(filtered_df.head(250), use_container_width=True)

    pdf_bytes = build_pdf_report(filtered_df, profile, insights, target, corr_df, anomaly_df, ts_df, dt_cols[0] if dt_cols else None, segment_df)
    st.download_button(
        'Download premium PDF report',
        data=pdf_bytes,
        file_name='csv_insight_pro_report.pdf',
        mime='application/pdf'
    )

    csv_profile = profile.to_csv(index=False).encode('utf-8')
    st.download_button(
        'Download schema profile CSV',
        data=csv_profile,
        file_name='schema_profile.csv',
        mime='text/csv'
    )
else:
    st.info('Upload a file to begin. This app works best when your dataset has a mix of numeric, categorical, and possibly date columns.')

