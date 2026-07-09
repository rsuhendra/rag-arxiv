from __future__ import annotations
import pandas as pd
import streamlit as st

st.set_page_config(page_title="arXiv RAG Explorer", page_icon="📚", layout="wide")
st.title("📚 arXiv RAG Explorer")
st.caption("A tiny Snowflake + Cortex + Streamlit RAG prototype over recent arXiv abstracts.")

@st.cache_resource
def get_connection():
    return st.connection("snowflake")

@st.cache_data(ttl=300)
def get_stats():
    conn = get_connection()
    return conn.query("""
        SELECT COUNT(*) AS NUM_PAPERS, MIN(PUBLISHED) AS EARLIEST_PAPER,
               MAX(PUBLISHED) AS LATEST_PAPER,
               (SELECT COUNT(*) FROM ARXIV_EMBEDDINGS) AS NUM_EMBEDDINGS
        FROM ARXIV_PAPERS
    """)

@st.cache_data(ttl=300)
def get_category_counts():
    conn = get_connection()
    return conn.query("""
        SELECT COALESCE(PRIMARY_CATEGORY, 'unknown') AS CATEGORY, COUNT(*) AS N
        FROM ARXIV_PAPERS
        GROUP BY CATEGORY
        ORDER BY N DESC
        LIMIT 15
    """)

def retrieve_papers(user_query: str, k: int) -> pd.DataFrame:
    conn = get_connection()
    sql = f"""
    WITH query_embedding AS (
        SELECT SNOWFLAKE.CORTEX.EMBED_TEXT_768('snowflake-arctic-embed-m', %(user_query)s) AS q_embedding
    )
    SELECT p.ARXIV_ID, p.TITLE, p.ABSTRACT, p.AUTHORS, p.CATEGORIES, p.PRIMARY_CATEGORY,
           p.PUBLISHED, p.ABS_URL,
           VECTOR_COSINE_SIMILARITY(e.EMBEDDING, q.q_embedding) AS SIMILARITY
    FROM ARXIV_EMBEDDINGS e
    JOIN ARXIV_PAPERS p ON e.ARXIV_ID = p.ARXIV_ID
    CROSS JOIN query_embedding q
    ORDER BY SIMILARITY DESC
    LIMIT {int(k)}
    """
    return conn.query(sql, params={"user_query": user_query})

def build_context(retrieved: pd.DataFrame) -> str:
    parts = []
    for idx, row in retrieved.reset_index(drop=True).iterrows():
        parts.append(f"""Source {idx + 1}
Title: {row["TITLE"]}
arXiv ID: {row["ARXIV_ID"]}
Categories: {row["CATEGORIES"]}
Published: {row["PUBLISHED"]}
Abstract: {row["ABSTRACT"]}""")
    return "\n\n".join(parts)

def generate_answer(user_query: str, retrieved: pd.DataFrame, model: str) -> str:
    conn = get_connection()
    prompt = f"""You are a careful research assistant.

Answer the user question using only the provided arXiv abstract context.
Do not claim you read the full papers.
If the abstracts do not contain enough information, say so.
Cite sources using bracketed source numbers like [Source 1].

User question:
{user_query}

Retrieved abstract context:
{build_context(retrieved)}"""
    result = conn.query(
        "SELECT SNOWFLAKE.CORTEX.COMPLETE(%(model)s, %(prompt)s) AS ANSWER",
        params={"model": model, "prompt": prompt},
    )
    return result.iloc[0]["ANSWER"]

def log_query(user_query: str, answer: str, retrieved: pd.DataFrame) -> None:
    conn = get_connection()
    ids = ",".join(retrieved["ARXIV_ID"].astype(str).tolist())
    conn.query("""
        INSERT INTO ARXIV_QUERIES (USER_QUERY, ANSWER, RETRIEVED_ARXIV_IDS, NUM_RETRIEVED)
        SELECT %(user_query)s, %(answer)s, %(ids)s, %(num_retrieved)s
    """, params={"user_query": user_query, "answer": answer, "ids": ids, "num_retrieved": int(len(retrieved))})

with st.sidebar:
    st.header("Settings")
    k = st.slider("Retrieved abstracts", 3, 10, 5)
    model = st.selectbox("Cortex LLM", ["mistral-large2", "llama3.1-70b", "mixtral-8x7b"], index=0)

try:
    stats = get_stats()
    s = stats.iloc[0]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Papers", int(s["NUM_PAPERS"]))
    c2.metric("Embeddings", int(s["NUM_EMBEDDINGS"]))
    c3.metric("Earliest", str(s["EARLIEST_PAPER"])[:10])
    c4.metric("Latest", str(s["LATEST_PAPER"])[:10])
    with st.expander("Category distribution"):
        cat_df = get_category_counts()
        st.bar_chart(cat_df.set_index("CATEGORY"))
except Exception as exc:
    st.error(f"Could not query Snowflake. Check secrets and Snowflake objects. Error: {exc}")
    st.stop()

st.divider()
examples = [
    "What are recent ideas for efficient transformer training?",
    "What methods are being used for diffusion models?",
    "Are there recent papers related to causal discovery?",
    "Summarize papers about reinforcement learning.",
]
user_query = st.text_input("Ask a question about the ingested arXiv abstracts", placeholder=examples[0])
cols = st.columns(len(examples))
for col, ex in zip(cols, examples):
    if col.button(ex, use_container_width=True):
        user_query = ex

if st.button("Run RAG search", type="primary") and user_query:
    with st.spinner("Retrieving relevant abstracts..."):
        retrieved = retrieve_papers(user_query, k)
    if retrieved.empty:
        st.warning("No results found. Did you run the embedding update?")
        st.stop()
    with st.spinner("Generating grounded answer..."):
        answer = generate_answer(user_query, retrieved, model)
    st.subheader("Answer")
    st.write(answer)
    try:
        log_query(user_query, answer, retrieved)
    except Exception as exc:
        st.warning(f"Answer generated, but query logging failed: {exc}")
    st.subheader("Retrieved sources")
    for idx, row in retrieved.reset_index(drop=True).iterrows():
        with st.expander(f"Source {idx + 1}: {row['TITLE']} · similarity {row['SIMILARITY']:.3f}"):
            st.write(f"**arXiv ID:** {row['ARXIV_ID']}")
            st.write(f"**Primary category:** {row['PRIMARY_CATEGORY']}")
            st.write(f"**Categories:** {row['CATEGORIES']}")
            st.write(f"**Published:** {row['PUBLISHED']}")
            st.write(f"**Authors:** {row['AUTHORS']}")
            st.write(f"**URL:** {row['ABS_URL']}")
            st.write(row["ABSTRACT"])
