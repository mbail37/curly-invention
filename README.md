# CSV Insight MVP

A tiny Streamlit app that lets a user upload a CSV or Excel file and immediately get:
- a preview
- basic summary metrics
- generated plain-English insights
- a histogram for one numeric column
- a downloadable PDF report

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Fast deployment options

### Streamlit Community Cloud
1. Push these files to a GitHub repo.
2. Go to Streamlit Community Cloud.
3. Create a new app and point it to `app.py`.
4. Add the same `requirements.txt`.

### Render
1. Create a new Web Service.
2. Connect the repo.
3. Build command:
   `pip install -r requirements.txt`
4. Start command:
   `streamlit run app.py --server.port=$PORT --server.address=0.0.0.0`

## Fast monetization path
- Keep upload + preview free.
- Put the PDF report behind a Stripe payment link.
- Once people use it, upgrade to Stripe Checkout and a small backend.

## First places to post it
- r/excel
- r/datasets
- r/dataisbeautiful (only if you show nice output)
- X / LinkedIn with a short demo gif
