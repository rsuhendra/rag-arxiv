from __future__ import annotations
from src.snowflake_io import create_session, run_embedding_update, print_counts

def main():
    session = create_session()
    try:
        run_embedding_update(session)
        print_counts(session)
    finally:
        session.close()

if __name__ == "__main__":
    main()
