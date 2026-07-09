from __future__ import annotations
import pandas as pd
from snowflake.snowpark import Session
from src.config import SnowflakeConfig

def create_session() -> Session:
    return Session.builder.configs(SnowflakeConfig.from_env().to_snowpark_dict()).create()

def upload_papers(session: Session, df: pd.DataFrame) -> None:
    if df.empty:
        print("No papers to upload.")
        return
    session.create_dataframe(df).write.mode("overwrite").save_as_table("ARXIV_PAPERS_STAGING")
    session.sql("""
    MERGE INTO ARXIV_PAPERS t
    USING ARXIV_PAPERS_STAGING s
    ON t.ARXIV_ID = s.ARXIV_ID
    WHEN NOT MATCHED THEN INSERT (
        ARXIV_ID, TITLE, ABSTRACT, AUTHORS, CATEGORIES, PRIMARY_CATEGORY,
        PUBLISHED, UPDATED, ABS_URL, PDF_URL
    ) VALUES (
        s.ARXIV_ID, s.TITLE, s.ABSTRACT, s.AUTHORS, s.CATEGORIES, s.PRIMARY_CATEGORY,
        s.PUBLISHED, s.UPDATED, s.ABS_URL, s.PDF_URL
    )
    """).collect()
    print("Merge into ARXIV_PAPERS complete.")

def run_embedding_update(session: Session) -> None:
    session.sql("""
    INSERT INTO ARXIV_EMBEDDINGS (ARXIV_ID, TEXT, EMBEDDING)
    SELECT
        p.ARXIV_ID,
        p.TITLE || '\\n\\n' || p.ABSTRACT AS TEXT,
        SNOWFLAKE.CORTEX.EMBED_TEXT_768('snowflake-arctic-embed-m', p.TITLE || '\\n\\n' || p.ABSTRACT) AS EMBEDDING
    FROM ARXIV_PAPERS p
    LEFT JOIN ARXIV_EMBEDDINGS e ON p.ARXIV_ID = e.ARXIV_ID
    WHERE e.ARXIV_ID IS NULL
    """).collect()
    print("Embedding update complete.")

def print_counts(session: Session) -> None:
    row = session.sql("""
    SELECT
      (SELECT COUNT(*) FROM ARXIV_PAPERS) AS NUM_PAPERS,
      (SELECT COUNT(*) FROM ARXIV_EMBEDDINGS) AS NUM_EMBEDDINGS
    """).collect()[0]
    print(f"ARXIV_PAPERS: {row['NUM_PAPERS']}")
    print(f"ARXIV_EMBEDDINGS: {row['NUM_EMBEDDINGS']}")
