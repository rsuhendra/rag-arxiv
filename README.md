# Live AI Research Assistant

A tiny but polished end-to-end AI research monitoring app.

It:
- Ingests recent arXiv abstracts.
- Tags each paper by topic using lightweight keyword rules.
- Computes embeddings in Python using `sentence-transformers`.
- Stores papers + vectors in Snowflake.
- Tracks topic trends over time.
- Provides a Streamlit dashboard and RAG-style semantic search.

## Architecture

```text
GitHub Actions / local script
        ↓
arXiv API
        ↓
Python topic tagging + embeddings
        ↓
Snowflake ARXIV_PAPERS
        ↓
Streamlit Community Cloud
        ↓
Dashboard + semantic retrieval
```

## Setup

Run `sql/01_setup.sql` in Snowflake, then add your Snowflake credentials as GitHub Actions secrets and Streamlit Cloud secrets.

Local run:

```bash
pip install -r requirements.txt
python -m src.ingest --max-papers 100
streamlit run app.py
```
