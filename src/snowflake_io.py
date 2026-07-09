from __future__ import annotations

import pandas as pd
from snowflake.snowpark import Session
from src.config import SnowflakeConfig

def create_session() -> Session:
    config = SnowflakeConfig.from_env()
    return Session.builder.configs(config.to_snowpark_dict()).create()

def upload_papers(session: Session, df: pd.DataFrame) -> None:
    if df.empty:
        print("No papers to upload.")
        return
    session.create_dataframe(df).write.mode("overwrite").save_as_table("ARXIV_PAPERS_STAGING")
    merge_sql = """
    MERGE INTO ARXIV_PAPERS t
    USING (
        SELECT ARXIV_ID, TITLE, ABSTRACT, AUTHORS, CATEGORIES, PRIMARY_CATEGORY, TOPIC,
               PUBLISHED, UPDATED, ABS_URL, PDF_URL, TEXT,
               EMBEDDING::VECTOR(FLOAT, 384) AS EMBEDDING
        FROM ARXIV_PAPERS_STAGING
    ) s
    ON t.ARXIV_ID = s.ARXIV_ID
    WHEN MATCHED THEN UPDATE SET
        t.TITLE=s.TITLE, t.ABSTRACT=s.ABSTRACT, t.AUTHORS=s.AUTHORS,
        t.CATEGORIES=s.CATEGORIES, t.PRIMARY_CATEGORY=s.PRIMARY_CATEGORY, t.TOPIC=s.TOPIC,
        t.PUBLISHED=s.PUBLISHED, t.UPDATED=s.UPDATED, t.ABS_URL=s.ABS_URL,
        t.PDF_URL=s.PDF_URL, t.TEXT=s.TEXT, t.EMBEDDING=s.EMBEDDING
    WHEN NOT MATCHED THEN INSERT (
        ARXIV_ID, TITLE, ABSTRACT, AUTHORS, CATEGORIES, PRIMARY_CATEGORY, TOPIC,
        PUBLISHED, UPDATED, ABS_URL, PDF_URL, TEXT, EMBEDDING
    ) VALUES (
        s.ARXIV_ID, s.TITLE, s.ABSTRACT, s.AUTHORS, s.CATEGORIES, s.PRIMARY_CATEGORY, s.TOPIC,
        s.PUBLISHED, s.UPDATED, s.ABS_URL, s.PDF_URL, s.TEXT, s.EMBEDDING
    )
    """
    session.sql(merge_sql).collect()
    print("Merge into ARXIV_PAPERS complete.")

def print_counts(session: Session) -> None:
    row = session.sql("SELECT COUNT(*) AS NUM_PAPERS, COUNT_IF(EMBEDDING IS NOT NULL) AS NUM_EMBEDDED FROM ARXIV_PAPERS").collect()[0]
    print(f"ARXIV_PAPERS: {row['NUM_PAPERS']}")
    print(f"EMBEDDED: {row['NUM_EMBEDDED']}")
