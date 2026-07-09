from __future__ import annotations

import pandas as pd
import streamlit as st
from sentence_transformers import SentenceTransformer

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

st.set_page_config(page_title="Live AI Research Assistant", page_icon="📚", layout="wide")
st.title("📚 Live AI Research Assistant")
st.caption("Hourly arXiv ingestion + topic trends + Snowflake vector search.")

@st.cache_resource
def get_connection():
    return st.connection("snowflake")

@st.cache_resource
def get_embedding_model() -> SentenceTransformer:
    return SentenceTransformer(MODEL_NAME)

def embed_query(query: str) -> list[float]:
    emb = get_embedding_model().encode(query, normalize_embeddings=True, show_progress_bar=False)
    return emb.astype(float).tolist()

@st.cache_data(ttl=300)
def get_stats() -> pd.DataFrame:
    return get_connection().query("""
        SELECT COUNT(*) AS NUM_PAPERS,
               COUNT_IF(EMBEDDING IS NOT NULL) AS NUM_EMBEDDED,
               MIN(PUBLISHED) AS EARLIEST_PAPER,
               MAX(PUBLISHED) AS LATEST_PAPER
        FROM ARXIV_PAPERS
    """)

@st.cache_data(ttl=300)
def get_topics() -> list[str]:
    df = get_connection().query("SELECT DISTINCT TOPIC FROM ARXIV_PAPERS WHERE TOPIC IS NOT NULL ORDER BY TOPIC")
    return ["All"] + df["TOPIC"].dropna().tolist()

@st.cache_data(ttl=300)
def get_topic_counts() -> pd.DataFrame:
    return get_connection().query("SELECT TOPIC, COUNT(*) AS N FROM ARXIV_PAPERS GROUP BY TOPIC ORDER BY N DESC")

@st.cache_data(ttl=300)
def get_topic_trends() -> pd.DataFrame:
    return get_connection().query("""
        SELECT DATE_TRUNC('DAY', PUBLISHED) AS PUBLISHED_DAY, TOPIC, COUNT(*) AS N
        FROM ARXIV_PAPERS
        WHERE PUBLISHED >= DATEADD('DAY', -14, CURRENT_TIMESTAMP())
        GROUP BY PUBLISHED_DAY, TOPIC
        ORDER BY PUBLISHED_DAY
    """)

@st.cache_data(ttl=300)
def get_latest_papers(topic: str, limit: int = 20) -> pd.DataFrame:
    where = "" if topic == "All" else "WHERE TOPIC = %(topic)s"
    return get_connection().query(f"""
        SELECT ARXIV_ID, TITLE, TOPIC, PRIMARY_CATEGORY, PUBLISHED, ABS_URL, ABSTRACT
        FROM ARXIV_PAPERS
        {where}
        ORDER BY PUBLISHED DESC
        LIMIT {int(limit)}
    """, params={"topic": topic})

def retrieve_papers(user_query: str, topic: str, k: int) -> pd.DataFrame:
    query_embedding = embed_query(user_query)
    topic_filter = "" if topic == "All" else "AND p.TOPIC = %(topic)s"
    sql = f"""
    SELECT p.ARXIV_ID, p.TITLE, p.ABSTRACT, p.AUTHORS, p.CATEGORIES, p.PRIMARY_CATEGORY,
           p.TOPIC, p.PUBLISHED, p.ABS_URL,
           VECTOR_COSINE_SIMILARITY(p.EMBEDDING, %(query_embedding)s::VECTOR(FLOAT, 384)) AS SIMILARITY
    FROM ARXIV_PAPERS p
    WHERE p.EMBEDDING IS NOT NULL
    {topic_filter}
    ORDER BY SIMILARITY DESC
    LIMIT {int(k)}
    """
    return get_connection().query(sql, params={"query_embedding": query_embedding, "topic": topic})

def log_query(user_query: str, topic: str, retrieved: pd.DataFrame) -> None:
    ids = ",".join(retrieved["ARXIV_ID"].astype(str).tolist())
    get_connection().query("""
        INSERT INTO ARXIV_QUERIES (USER_QUERY, TOPIC_FILTER, RETRIEVED_ARXIV_IDS, NUM_RETRIEVED)
        SELECT %(user_query)s, %(topic)s, %(ids)s, %(num_retrieved)s
    """, params={"user_query": user_query, "topic": topic, "ids": ids, "num_retrieved": int(len(retrieved))})

def make_summary(user_query: str, retrieved: pd.DataFrame) -> str:
    if retrieved.empty:
        return "No relevant papers were retrieved."
    lines = [f"Top relevant papers for: **{user_query}**", ""]
    for i, row in retrieved.head(3).reset_index(drop=True).iterrows():
        preview = row["ABSTRACT"][:350] + ("..." if len(row["ABSTRACT"]) > 350 else "")
        lines.append(f"**[{i+1}] {row['TITLE']}** ({row['TOPIC']}, similarity {row['SIMILARITY']:.3f})

{preview}
")
    lines.append("This is an extractive retrieval summary. Add an external LLM later for fully generative RAG answers.")
    return "
".join(lines)

try:
    s = get_stats().iloc[0]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Papers", int(s["NUM_PAPERS"]))
    c2.metric("Embedded", int(s["NUM_EMBEDDED"]))
    c3.metric("Earliest", str(s["EARLIEST_PAPER"])[:10])
    c4.metric("Latest", str(s["LATEST_PAPER"])[:10])
except Exception as exc:
    st.error(f"Could not query Snowflake. Check Streamlit secrets and table setup. Error: {exc}")
    st.stop()

topics = get_topics()
with st.sidebar:
    st.header("Controls")
    selected_topic = st.selectbox("Topic", topics, index=0)
    k = st.slider("Retrieved papers", 3, 10, 5)

tab_dashboard, tab_search, tab_latest = st.tabs(["📈 Trends", "🔎 Research Search", "🆕 Latest Papers"])

with tab_dashboard:
    st.subheader("Topic distribution")
    topic_counts = get_topic_counts()
    if not topic_counts.empty:
        st.bar_chart(topic_counts.set_index("TOPIC"))
    st.subheader("Topic trends over the last 14 days")
    trends = get_topic_trends()
    if trends.empty:
        st.info("Not enough recent papers to show trends yet.")
    else:
        pivot = trends.pivot_table(index="PUBLISHED_DAY", columns="TOPIC", values="N", aggfunc="sum", fill_value=0)
        st.line_chart(pivot)

with tab_search:
    st.subheader("Semantic search over arXiv abstracts")
    user_query = st.text_input("Ask a research question", placeholder="causal discovery with neural networks")
    if st.button("Search", type="primary") and user_query:
        with st.spinner("Embedding query and searching Snowflake..."):
            retrieved = retrieve_papers(user_query, selected_topic, k)
        if retrieved.empty:
            st.warning("No matching papers found.")
        else:
            try:
                log_query(user_query, selected_topic, retrieved)
            except Exception as exc:
                st.warning(f"Search worked, but query logging failed: {exc}")
            st.subheader("Summary")
            st.write(make_summary(user_query, retrieved))
            st.subheader("Retrieved papers")
            for i, row in retrieved.reset_index(drop=True).iterrows():
                with st.expander(f"{i+1}. {row['TITLE']} · {row['TOPIC']} · similarity {row['SIMILARITY']:.3f}"):
                    st.write(f"**arXiv ID:** {row['ARXIV_ID']}")
                    st.write(f"**Published:** {row['PUBLISHED']}")
                    st.write(f"**Primary category:** {row['PRIMARY_CATEGORY']}")
                    st.write(f"**Categories:** {row['CATEGORIES']}")
                    st.write(f"**Authors:** {row['AUTHORS']}")
                    st.write(f"**URL:** {row['ABS_URL']}")
                    st.write(row["ABSTRACT"])

with tab_latest:
    st.subheader(f"Latest papers: {selected_topic}")
    latest = get_latest_papers(selected_topic, limit=20)
    if latest.empty:
        st.info("No papers available.")
    else:
        for _, row in latest.iterrows():
            with st.expander(f"{row['TITLE']} · {row['TOPIC']} · {str(row['PUBLISHED'])[:10]}"):
                st.write(f"**arXiv ID:** {row['ARXIV_ID']}")
                st.write(f"**Primary category:** {row['PRIMARY_CATEGORY']}")
                st.write(f"**URL:** {row['ABS_URL']}")
                st.write(row["ABSTRACT"])
