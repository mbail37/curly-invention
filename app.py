import io
import re
from datetime import datetime

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer


st.set_page_config(page_title="CSV Insight MVP", page_icon="📊", layout="wide")


@st.cache_data

def load_csv(uploaded_file: bytes, filename: str) -> pd.DataFrame:
    name = filename.lower()
    if name.endswith('.csv'):
        return pd.read_csv(io.BytesIO(uploaded_file))
    if name.endswith('.xlsx') or name.endswith('.xls'):
        return pd.read_excel(io.BytesIO(uploaded_file))
    raise ValueError('Unsupported file type. Upload a CSV or Excel file.')


def clean_column_name(col: str) -> str:
    col = str(col).strip().lower()
    col = re.sub(r'[^a-z0-9]+', '_', col)
    return col.strip('_') or 'unnamed_column'


def summarize_dataframe(df: pd.DataFrame) -> dict:
    numeric_cols = df.select_dtypes(include='number').columns.tolist()
    text_cols = df.select_dtypes(exclude='number').columns.tolist()

    summary = {
        'rows': int(len(df)),
        'columns': int(len(df.columns)),
        'numeric_columns': numeric_cols,
        'text_columns': text_cols,
        'missing_cells': int(df.isna().sum().sum()),
        'duplicate_rows': int(df.duplicated().sum()),
    }

    insights = []

    if numeric_cols:
        missing_by_col = df[numeric_cols].isna().sum().sort_values(ascending=False)
        if missing_by_col.iloc[0] > 0:
            insights.append(
                f"The numeric column with the most missing values is '{missing_by_col.index[0]}' ({int(missing_by_col.iloc[0])} missing cells)."
            )

        stds = df[numeric_cols].std(numeric_only=True).sort_values(ascending=False)
        if len(stds) > 0:
            insights.append(
                f"'{stds.index[0]}' shows the highest variation among numeric columns, which usually makes it one of the most informative fields to inspect first."
            )

        means = df[numeric_cols].mean(numeric_only=True).sort_values(ascending=False)
        top_mean = means.index[0]
        insights.append(
            f"'{top_mean}' has the highest average value among numeric columns at {means.iloc[0]:,.2f}."
        )
    else:
        insights.append('No numeric columns were found, so this file may need cleaning or categorization before deeper analysis.')

    if summary['duplicate_rows'] > 0:
        insights.append(f"There are {summary['duplicate_rows']} duplicate rows, which may be inflating totals or summaries.")

    if summary['missing_cells'] == 0:
        insights.append('No missing cells were detected, which is a strong sign the file is already relatively clean.')

    summary['insights'] = insights
    return summary


def top_bottom_table(df: pd.DataFrame, col: str, n: int = 5) -> tuple[pd.DataFrame, pd.DataFrame]:
    ordered = df[[col]].dropna().sort_values(by=col, ascending=False)
    return ordered.head(n), ordered.tail(n).sort_values(by=col, ascending=True)


def build_pdf_report(summary: dict, df: pd.DataFrame) -> bytes:
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    styles = getSampleStyleSheet()
    story = []

    title = f"CSV Insight Report - {datetime.now().strftime('%Y-%m-%d %H:%M')}"
    story.append(Paragraph(title, styles['Title']))
    story.append(Spacer(1, 12))

    story.append(Paragraph(f"Rows: {summary['rows']}", styles['BodyText']))
    story.append(Paragraph(f"Columns: {summary['columns']}", styles['BodyText']))
    story.append(Paragraph(f"Missing cells: {summary['missing_cells']}", styles['BodyText']))
    story.append(Paragraph(f"Duplicate rows: {summary['duplicate_rows']}", styles['BodyText']))
    story.append(Spacer(1, 12))

    story.append(Paragraph('Key insights', styles['Heading2']))
    for insight in summary['insights']:
        story.append(Paragraph(f"• {insight}", styles['BodyText']))
        story.append(Spacer(1, 6))

    numeric_cols = summary['numeric_columns']
    if numeric_cols:
        story.append(Spacer(1, 12))
        story.append(Paragraph('Numeric column summary', styles['Heading2']))
        desc = df[numeric_cols].describe().round(2)
        for col in desc.columns[:8]:
            vals = desc[col]
            line = (
                f"{col}: mean={vals.get('mean', 0):,.2f}, std={vals.get('std', 0):,.2f}, "
                f"min={vals.get('min', 0):,.2f}, max={vals.get('max', 0):,.2f}"
            )
            story.append(Paragraph(line, styles['BodyText']))
            story.append(Spacer(1, 4))

    doc.build(story)
    buffer.seek(0)
    return buffer.read()


def main() -> None:
    st.title('📊 CSV Insight MVP')
    st.write('Upload a CSV or Excel file and get a quick summary, a few insights, and a downloadable report.')

    st.markdown(
        """
        **Monetization stub:**
        - Keep this page free for preview.
        - Gate the PDF download behind Stripe later.
        - For now, the button is enabled so you can test the full flow locally.
        """
    )

    uploaded = st.file_uploader('Upload a CSV or Excel file', type=['csv', 'xlsx', 'xls'])

    if not uploaded:
        st.info('Upload a file to begin.')
        return

    try:
        raw_bytes = uploaded.read()
        df = load_csv(raw_bytes, uploaded.name)
    except Exception as exc:
        st.error(f'Could not read file: {exc}')
        return

    st.subheader('Preview')
    st.dataframe(df.head(20), use_container_width=True)

    with st.expander('Optional: clean column names'):
        if st.button('Clean column names'):
            df.columns = [clean_column_name(c) for c in df.columns]
            st.success('Column names cleaned for this session.')
            st.dataframe(df.head(20), use_container_width=True)

    summary = summarize_dataframe(df)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric('Rows', f"{summary['rows']:,}")
    c2.metric('Columns', summary['columns'])
    c3.metric('Missing cells', f"{summary['missing_cells']:,}")
    c4.metric('Duplicate rows', f"{summary['duplicate_rows']:,}")

    st.subheader('Generated insights')
    for insight in summary['insights']:
        st.write(f"- {insight}")

    numeric_cols = summary['numeric_columns']
    if numeric_cols:
        st.subheader('Numeric overview')
        st.dataframe(df[numeric_cols].describe().T.round(2), use_container_width=True)

        chart_col = st.selectbox('Choose a numeric column to chart', numeric_cols)

        fig, ax = plt.subplots(figsize=(8, 4.5))
        df[chart_col].dropna().hist(ax=ax, bins=30)
        ax.set_title(f'Distribution of {chart_col}')
        ax.set_xlabel(chart_col)
        ax.set_ylabel('Count')
        st.pyplot(fig)
        plt.close(fig)

        top_df, bottom_df = top_bottom_table(df, chart_col)
        left, right = st.columns(2)
        with left:
            st.markdown(f'**Top 5 values in {chart_col}**')
            st.dataframe(top_df, use_container_width=True)
        with right:
            st.markdown(f'**Bottom 5 values in {chart_col}**')
            st.dataframe(bottom_df, use_container_width=True)
    else:
        st.warning('No numeric columns detected, so charts are skipped.')

    pdf_bytes = build_pdf_report(summary, df)
    st.download_button(
        label='Download PDF report',
        data=pdf_bytes,
        file_name='csv_insight_report.pdf',
        mime='application/pdf',
    )

    st.subheader('Add Stripe in version 2')
    st.code(
        """# Easiest first pass:
# 1. Create a payment link in Stripe.
# 2. Put the link behind a button.
# 3. After payment, send users to a simple success page.
# 4. In the next version, use Stripe Checkout + a backend endpoint.

# Example:
st.link_button('Unlock premium report - $5', 'https://buy.stripe.com/your_link_here')
""",
        language='python',
    )


if __name__ == '__main__':
    main()
