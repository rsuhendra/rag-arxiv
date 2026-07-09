# arXiv Snowflake RAG Explorer

End-to-end tiny RAG prototype:

1. Pull recent arXiv abstracts from the official arXiv API.
2. Store paper metadata in Snowflake.
3. Generate embeddings with Snowflake Cortex.
4. Retrieve semantically similar abstracts with Snowflake vector search.
5. Generate grounded answers with Snowflake Cortex LLMs.
6. Serve everything through a Streamlit app.

## Setup

Run `sql/01_setup.sql` in Snowflake first.

Then locally:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Create `.env`:

```bash
SNOWFLAKE_ACCOUNT=your_account_identifier
SNOWFLAKE_USER=your_user
SNOWFLAKE_PASSWORD=your_password
SNOWFLAKE_ROLE=E2E_SNOW_MLOPS_ROLE
SNOWFLAKE_WAREHOUSE=E2E_SNOW_MLOPS_WH
SNOWFLAKE_DATABASE=E2E_SNOW_MLOPS_DB
SNOWFLAKE_SCHEMA=MLOPS_SCHEMA
```

Run:

```bash
python -m src.ingest --max-papers 100
python -m src.update_embeddings
cp .streamlit/secrets.toml.example .streamlit/secrets.toml
streamlit run app.py
```

Add GitHub repository secrets with the same Snowflake variables to use the scheduled workflows.
