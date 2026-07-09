from __future__ import annotations
import argparse
from src.arxiv_client import fetch_arxiv_abstracts
from src.snowflake_io import create_session, upload_papers, print_counts

DEFAULT_QUERY = "cat:cs.LG OR cat:stat.ML OR cat:cs.AI"

def parse_args():
    parser = argparse.ArgumentParser(description="Ingest recent arXiv papers into Snowflake.")
    parser.add_argument("--query", default=DEFAULT_QUERY)
    parser.add_argument("--max-papers", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=50)
    parser.add_argument("--sleep-seconds", type=float, default=3.0)
    return parser.parse_args()

def main():
    args = parse_args()
    df = fetch_arxiv_abstracts(args.query, args.max_papers, args.batch_size, args.sleep_seconds)
    print(f"Fetched dataframe shape: {df.shape}")
    session = create_session()
    try:
        upload_papers(session, df)
        print_counts(session)
    finally:
        session.close()

if __name__ == "__main__":
    main()
