from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone

from src.arxiv_client import fetch_arxiv_abstracts
from src.embedding import embed_texts
from src.snowflake_io import create_session, upload_papers, print_counts


CATEGORY_QUERY = (
    "cat:cs.LG OR "
    "cat:stat.ML OR "
    "cat:cs.AI OR "
    "cat:cs.CL OR "
    "cat:cs.CV OR "
    "cat:cs.RO"
)


def make_date_query(days_back: int) -> string:
    """
    Build an arXiv submitted-date filter covering the previous N days.

    arXiv expects timestamps formatted as YYYYMMDDHHMM.
    """
    end_time = datetime.now(timezone.utc)
    start_time = end_time - timedelta(days=days_back)

    start_string = start_time.strftime("%Y%m%d%H%M")
    end_string = end_time.strftime("%Y%m%d%H%M")

    return (
        f"({CATEGORY_QUERY}) AND "
        f"submittedDate:[{start_string} TO {end_string}]"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Ingest recent arXiv papers into Snowflake."
    )

    parser.add_argument(
        "--days-back",
        type=int,
        default=1,
        help="Fetch papers submitted during the previous N days.",
    )

    parser.add_argument(
        "--max-papers",
        type=int,
        default=2000,
        help="Maximum number of papers to fetch as a safety limit.",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=100,
        help="Number of papers requested per arXiv API call.",
    )

    parser.add_argument(
        "--sleep-seconds",
        type=float,
        default=3.0,
        help="Delay between API requests.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    query = make_date_query(args.days_back)

    print(f"Fetching papers from the previous {args.days_back} day(s).")
    print(f"arXiv query: {query}")

    df = fetch_arxiv_abstracts(
        query=query,
        max_papers=args.max_papers,
        batch_size=args.batch_size,
        sleep_seconds=args.sleep_seconds,
    )

    print(f"Fetched dataframe shape: {df.shape}")

    if df.empty:
        print("No papers were returned. Nothing to upload.")
        return

    print("Computing Python embeddings...")

    df["EMBEDDING"] = embed_texts(
        df["TEXT"].astype(str).tolist()
    )

    session = create_session()

    try:
        upload_papers(session, df)
        print_counts(session)

    finally:
        session.close()


if __name__ == "__main__":
    main()