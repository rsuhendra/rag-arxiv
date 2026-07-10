from __future__ import annotations

import json

import pandas as pd
import streamlit as st
from sentence_transformers import SentenceTransformer


EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIM = 384


# -------------------------------------------------------------------
# Page configuration
# -------------------------------------------------------------------

st.set_page_config(
    page_title="Live AI Research Assistant",
    page_icon="📚",
    layout="wide",
)

st.title("📚 Live AI Research Assistant")
st.caption(
    "Explore recent arXiv papers using topic trends and semantic "
    "vector search."
)


# -------------------------------------------------------------------
# Connections and models
# -------------------------------------------------------------------

@st.cache_resource
def get_connection():
    """
    Create and cache the Streamlit Snowflake connection.

    Connection details are read from Streamlit secrets under:

    [connections.snowflake]
    """
    return st.connection("snowflake")


@st.cache_resource
def get_embedding_model() -> SentenceTransformer:
    """
    Load the sentence-transformer model once per app process.
    """
    return SentenceTransformer(EMBEDDING_MODEL)


def embed_query(query: str) -> list[float]:
    """
    Convert a query into a normalized 384-dimensional embedding.
    """
    model = get_embedding_model()

    embedding = model.encode(
        query,
        normalize_embeddings=True,
        show_progress_bar=False,
    )

    return embedding.astype(float).tolist()


# -------------------------------------------------------------------
# Snowflake read queries
# -------------------------------------------------------------------

@st.cache_data(ttl=300)
def get_stats() -> pd.DataFrame:
    conn = get_connection()

    return conn.query(
        """
        SELECT
            COUNT(*) AS NUM_PAPERS,
            COUNT_IF(EMBEDDING IS NOT NULL) AS NUM_EMBEDDED,
            MIN(PUBLISHED) AS EARLIEST_PAPER,
            MAX(PUBLISHED) AS LATEST_PAPER
        FROM ARXIV_PAPERS
        """
    )


@st.cache_data(ttl=300)
def get_topics() -> list[str]:
    conn = get_connection()

    df = conn.query(
        """
        SELECT DISTINCT TOPIC
        FROM ARXIV_PAPERS
        WHERE TOPIC IS NOT NULL
        ORDER BY TOPIC
        """
    )

    topics = df["TOPIC"].dropna().astype(str).tolist()

    return ["All"] + topics


@st.cache_data(ttl=300)
def get_topic_counts() -> pd.DataFrame:
    conn = get_connection()

    return conn.query(
        """
        SELECT
            COALESCE(TOPIC, 'Other') AS TOPIC,
            COUNT(*) AS N
        FROM ARXIV_PAPERS
        GROUP BY COALESCE(TOPIC, 'Other')
        ORDER BY N DESC
        """
    )


@st.cache_data(ttl=300)
def get_topic_trends() -> pd.DataFrame:
    conn = get_connection()

    return conn.query(
        """
        SELECT
            DATE_TRUNC('DAY', PUBLISHED) AS PUBLISHED_DAY,
            COALESCE(TOPIC, 'Other') AS TOPIC,
            COUNT(*) AS N
        FROM ARXIV_PAPERS
        WHERE PUBLISHED >= DATEADD(
            'DAY',
            -14,
            CURRENT_TIMESTAMP()
        )
        GROUP BY
            DATE_TRUNC('DAY', PUBLISHED),
            COALESCE(TOPIC, 'Other')
        ORDER BY PUBLISHED_DAY
        """
    )


@st.cache_data(ttl=300)
def get_latest_papers(
    topic: str,
    limit: int = 20,
) -> pd.DataFrame:
    """
    Return the newest papers, optionally restricted to one topic.
    """
    conn = get_connection()

    safe_limit = max(1, min(int(limit), 100))

    if topic == "All":
        return conn.query(
            f"""
            SELECT
                ARXIV_ID,
                TITLE,
                ABSTRACT,
                TOPIC,
                PRIMARY_CATEGORY,
                PUBLISHED,
                ABS_URL
            FROM ARXIV_PAPERS
            ORDER BY PUBLISHED DESC
            LIMIT {safe_limit}
            """
        )

    return conn.query(
        f"""
        SELECT
            ARXIV_ID,
            TITLE,
            ABSTRACT,
            TOPIC,
            PRIMARY_CATEGORY,
            PUBLISHED,
            ABS_URL
        FROM ARXIV_PAPERS
        WHERE TOPIC = ?
        ORDER BY PUBLISHED DESC
        LIMIT {safe_limit}
        """,
        params=[topic],
    )


def retrieve_papers(
    user_query: str,
    topic: str,
    k: int,
) -> pd.DataFrame:
    """
    Embed the user query in Python and compare it against vectors
    stored in Snowflake using cosine similarity.
    """
    conn = get_connection()

    safe_k = max(1, min(int(k), 20))

    query_embedding = embed_query(user_query)
    query_embedding_json = json.dumps(query_embedding)

    if topic == "All":
        sql = f"""
        SELECT
            p.ARXIV_ID,
            p.TITLE,
            p.ABSTRACT,
            p.AUTHORS,
            p.CATEGORIES,
            p.PRIMARY_CATEGORY,
            p.TOPIC,
            p.PUBLISHED,
            p.ABS_URL,
            VECTOR_COSINE_SIMILARITY(
                p.EMBEDDING,
                PARSE_JSON(?)::VECTOR(FLOAT, {EMBEDDING_DIM})
            ) AS SIMILARITY
        FROM ARXIV_PAPERS p
        WHERE p.EMBEDDING IS NOT NULL
        ORDER BY SIMILARITY DESC
        LIMIT {safe_k}
        """

        params = [query_embedding_json]

    else:
        sql = f"""
        SELECT
            p.ARXIV_ID,
            p.TITLE,
            p.ABSTRACT,
            p.AUTHORS,
            p.CATEGORIES,
            p.PRIMARY_CATEGORY,
            p.TOPIC,
            p.PUBLISHED,
            p.ABS_URL,
            VECTOR_COSINE_SIMILARITY(
                p.EMBEDDING,
                PARSE_JSON(?)::VECTOR(FLOAT, {EMBEDDING_DIM})
            ) AS SIMILARITY
        FROM ARXIV_PAPERS p
        WHERE p.EMBEDDING IS NOT NULL
          AND p.TOPIC = ?
        ORDER BY SIMILARITY DESC
        LIMIT {safe_k}
        """

        params = [
            query_embedding_json,
            topic,
        ]

    return conn.query(
        sql,
        params=params,
        ttl=0,
    )


# -------------------------------------------------------------------
# Snowflake write query
# -------------------------------------------------------------------

def log_query(
    user_query: str,
    topic: str,
    retrieved: pd.DataFrame,
) -> None:
    """
    Save basic search information in ARXIV_QUERIES.

    We use the underlying Snowflake connector because conn.query()
    is primarily intended for read queries.
    """
    conn = get_connection()

    retrieved_ids = ",".join(
        retrieved["ARXIV_ID"].astype(str).tolist()
    )

    raw_connection = conn.raw_connection
    cursor = raw_connection.cursor()

    try:
        cursor.execute(
            """
            INSERT INTO ARXIV_QUERIES (
                USER_QUERY,
                TOPIC_FILTER,
                RETRIEVED_ARXIV_IDS,
                NUM_RETRIEVED
            )
            VALUES (?, ?, ?, ?)
            """,
            [
                user_query,
                topic,
                retrieved_ids,
                int(len(retrieved)),
            ],
        )

        raw_connection.commit()

    finally:
        cursor.close()


# -------------------------------------------------------------------
# Display helpers
# -------------------------------------------------------------------

def make_retrieval_summary(
    user_query: str,
    retrieved: pd.DataFrame,
) -> str:
    """
    Create a lightweight extractive summary.

    This currently displays relevant abstracts rather than sending
    the retrieved context to a generative LLM.
    """
    if retrieved.empty:
        return "No relevant papers were retrieved."

    lines = [
        f"Top matches for **{user_query}**:",
        "",
    ]

    top_results = retrieved.head(3).reset_index(drop=True)

    for index, row in top_results.iterrows():
        abstract = str(row["ABSTRACT"])
        title = str(row["TITLE"])
        topic = str(row.get("TOPIC", "Other"))
        similarity = float(row["SIMILARITY"])

        preview = abstract[:400]

        if len(abstract) > 400:
            preview += "..."

        lines.append(
            (
                f"**[{index + 1}] {title}**  \n"
                f"Topic: {topic} · "
                f"Similarity: {similarity:.3f}  \n"
                f"{preview}\n"
            )
        )

    lines.append(
        "_This is an extractive retrieval summary. "
        "A generative LLM can be added later to synthesize "
        "a full answer from the retrieved abstracts._"
    )

    return "\n".join(lines)


def display_paper(
    row: pd.Series,
    index: int | None = None,
) -> None:
    """
    Display one paper in a Streamlit expander.
    """
    title = str(row.get("TITLE", "Untitled paper"))
    topic = str(row.get("TOPIC", "Other"))

    if index is not None and "SIMILARITY" in row:
        similarity = float(row["SIMILARITY"])

        label = (
            f"{index}. {title} · {topic} · "
            f"similarity {similarity:.3f}"
        )

    else:
        published = str(row.get("PUBLISHED", ""))[:10]
        label = f"{title} · {topic} · {published}"

    with st.expander(label):
        st.write(
            f"**arXiv ID:** {row.get('ARXIV_ID', '')}"
        )

        st.write(
            "**Primary category:** "
            f"{row.get('PRIMARY_CATEGORY', '')}"
        )

        if "CATEGORIES" in row:
            st.write(
                f"**Categories:** {row.get('CATEGORIES', '')}"
            )

        if "AUTHORS" in row:
            st.write(
                f"**Authors:** {row.get('AUTHORS', '')}"
            )

        st.write(
            f"**Published:** {row.get('PUBLISHED', '')}"
        )

        url = row.get("ABS_URL")

        if url:
            st.markdown(
                f"[Open on arXiv]({url})"
            )

        st.write(
            row.get("ABSTRACT", "")
        )


# -------------------------------------------------------------------
# Session-state callbacks
# -------------------------------------------------------------------

def set_example_query(example: str) -> None:
    """
    Update the text-input value before Streamlit recreates the widget.
    """
    st.session_state["research_query"] = example


if "research_query" not in st.session_state:
    st.session_state["research_query"] = ""


# -------------------------------------------------------------------
# Load app-level data
# -------------------------------------------------------------------

try:
    stats = get_stats()
    stat_row = stats.iloc[0]

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Papers",
        int(stat_row["NUM_PAPERS"]),
    )

    col2.metric(
        "Embedded",
        int(stat_row["NUM_EMBEDDED"]),
    )

    col3.metric(
        "Earliest",
        str(stat_row["EARLIEST_PAPER"])[:10],
    )

    col4.metric(
        "Latest",
        str(stat_row["LATEST_PAPER"])[:10],
    )

except Exception as exc:
    st.error(
        "Could not query Snowflake. Check your Streamlit secrets, "
        "database objects, warehouse, role, and permissions.\n\n"
        f"Error: {exc}"
    )

    st.stop()


try:
    topics = get_topics()

except Exception as exc:
    st.error(
        f"Could not load topics from Snowflake: {exc}"
    )

    st.stop()


# -------------------------------------------------------------------
# Sidebar
# -------------------------------------------------------------------

with st.sidebar:
    st.header("Search settings")

    selected_topic = st.selectbox(
        "Topic",
        topics,
        index=0,
    )

    num_results = st.slider(
        "Number of retrieved papers",
        min_value=3,
        max_value=10,
        value=5,
    )

    st.divider()

    st.caption(
        "Papers are embedded using all-MiniLM-L6-v2 and "
        "searched using cosine similarity in Snowflake."
    )


# -------------------------------------------------------------------
# Main tabs
# -------------------------------------------------------------------

tab_trends, tab_search, tab_latest = st.tabs(
    [
        "📈 Research Trends",
        "🔎 Semantic Search",
        "🆕 Latest Papers",
    ]
)


# -------------------------------------------------------------------
# Trends tab
# -------------------------------------------------------------------

with tab_trends:
    st.subheader("Paper distribution by topic")

    try:
        topic_counts = get_topic_counts()

        if topic_counts.empty:
            st.info(
                "No topic data is available yet."
            )

        else:
            st.bar_chart(
                topic_counts.set_index("TOPIC")["N"]
            )

    except Exception as exc:
        st.error(
            f"Could not load topic counts: {exc}"
        )

    st.subheader(
        "Topic trends over the last 14 days"
    )

    try:
        trends = get_topic_trends()

        if trends.empty:
            st.info(
                "There are not enough recent papers "
                "to show a trend chart yet."
            )

        else:
            trend_pivot = trends.pivot_table(
                index="PUBLISHED_DAY",
                columns="TOPIC",
                values="N",
                aggfunc="sum",
                fill_value=0,
            )

            st.line_chart(trend_pivot)

    except Exception as exc:
        st.error(
            f"Could not load topic trends: {exc}"
        )


# -------------------------------------------------------------------
# Semantic-search tab
# -------------------------------------------------------------------

with tab_search:
    st.subheader("Search recent AI research")

    examples = [
        "efficient transformer training",
        "causal discovery with neural networks",
        "reinforcement learning from human feedback",
        "diffusion models for image generation",
    ]

    st.text_input(
        "Research question",
        key="research_query",
        placeholder=examples[0],
    )

    st.caption("Example searches")

    example_columns = st.columns(2)

    for index, example in enumerate(examples):
        column = example_columns[index % 2]

        column.button(
            example,
            key=f"example_query_{index}",
            use_container_width=True,
            on_click=set_example_query,
            args=(example,),
        )

    user_query = (
        st.session_state["research_query"].strip()
    )

    search_clicked = st.button(
        "Search papers",
        type="primary",
        disabled=not user_query,
    )

    if search_clicked:
        try:
            with st.spinner(
                "Embedding your question and "
                "searching Snowflake..."
            ):
                retrieved = retrieve_papers(
                    user_query=user_query,
                    topic=selected_topic,
                    k=num_results,
                )

            if retrieved.empty:
                st.warning(
                    "No matching papers were found."
                )

            else:
                try:
                    log_query(
                        user_query=user_query,
                        topic=selected_topic,
                        retrieved=retrieved,
                    )

                except Exception as exc:
                    st.warning(
                        "The search worked, but query "
                        f"logging failed: {exc}"
                    )

                st.subheader("Retrieval summary")

                st.markdown(
                    make_retrieval_summary(
                        user_query=user_query,
                        retrieved=retrieved,
                    )
                )

                st.subheader("Retrieved papers")

                retrieved_reset = retrieved.reset_index(
                    drop=True
                )

                for index, row in retrieved_reset.iterrows():
                    display_paper(
                        row,
                        index=index + 1,
                    )

        except Exception as exc:
            st.error(
                f"Semantic search failed: {exc}"
            )


# -------------------------------------------------------------------
# Latest-papers tab
# -------------------------------------------------------------------

with tab_latest:
    st.subheader(
        f"Latest papers: {selected_topic}"
    )

    try:
        latest_papers = get_latest_papers(
            selected_topic,
            limit=20,
        )

        if latest_papers.empty:
            st.info(
                "No papers are available for this topic."
            )

        else:
            for _, row in latest_papers.iterrows():
                display_paper(row)

    except Exception as exc:
        st.error(
            f"Could not load the latest papers: {exc}"
        )