from __future__ import annotations
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from urllib.parse import urlencode
import pandas as pd
import requests

ARXIV_API_URL = "https://export.arxiv.org/api/query"
ATOM_NS = {"atom": "http://www.w3.org/2005/Atom", "arxiv": "http://arxiv.org/schemas/atom"}

@dataclass
class ArxivPaper:
    arxiv_id: str
    title: str
    abstract: str
    authors: str
    categories: str
    primary_category: str | None
    published: str
    updated: str
    abs_url: str
    pdf_url: str | None

    def as_record(self) -> dict:
        return {
            "ARXIV_ID": self.arxiv_id,
            "TITLE": self.title,
            "ABSTRACT": self.abstract,
            "AUTHORS": self.authors,
            "CATEGORIES": self.categories,
            "PRIMARY_CATEGORY": self.primary_category,
            "PUBLISHED": self.published,
            "UPDATED": self.updated,
            "ABS_URL": self.abs_url,
            "PDF_URL": self.pdf_url,
        }

def _text(entry: ET.Element, path: str) -> str:
    value = entry.find(path, ATOM_NS)
    return "" if value is None or value.text is None else value.text.strip()

def _clean_text(value: str) -> str:
    return " ".join(value.replace("\n", " ").split())

def parse_entry(entry: ET.Element) -> ArxivPaper:
    abs_url = _text(entry, "atom:id")
    arxiv_id = abs_url.split("/")[-1]
    authors = [_clean_text(_text(a, "atom:name")) for a in entry.findall("atom:author", ATOM_NS)]
    categories = [c.attrib.get("term", "") for c in entry.findall("atom:category", ATOM_NS) if c.attrib.get("term")]
    primary_node = entry.find("arxiv:primary_category", ATOM_NS)
    primary_category = primary_node.attrib.get("term") if primary_node is not None else (categories[0] if categories else None)
    pdf_url = None
    for link in entry.findall("atom:link", ATOM_NS):
        if link.attrib.get("title") == "pdf":
            pdf_url = link.attrib.get("href")
    return ArxivPaper(
        arxiv_id=arxiv_id,
        title=_clean_text(_text(entry, "atom:title")),
        abstract=_clean_text(_text(entry, "atom:summary")),
        authors=", ".join(authors),
        categories=", ".join(categories),
        primary_category=primary_category,
        published=_text(entry, "atom:published"),
        updated=_text(entry, "atom:updated"),
        abs_url=abs_url,
        pdf_url=pdf_url,
    )

def fetch_arxiv_abstracts(query: str, max_papers: int = 100, batch_size: int = 50, sleep_seconds: float = 3.0) -> pd.DataFrame:
    if max_papers <= 0:
        return pd.DataFrame()
    batch_size = max(1, min(batch_size, max_papers))
    papers = []
    for start in range(0, max_papers, batch_size):
        params = {"search_query": query, "start": start, "max_results": min(batch_size, max_papers - start), "sortBy": "submittedDate", "sortOrder": "descending"}
        url = f"{ARXIV_API_URL}?{urlencode(params)}"
        response = requests.get(url, headers={"User-Agent": "arxiv-snowflake-rag/0.1"}, timeout=30)
        response.raise_for_status()
        root = ET.fromstring(response.content)
        entries = root.findall("atom:entry", ATOM_NS)
        if not entries:
            break
        for entry in entries:
            papers.append(parse_entry(entry))
        print(f"Fetched {len(papers)} papers so far...")
        if start + batch_size < max_papers:
            time.sleep(sleep_seconds)
    return pd.DataFrame([p.as_record() for p in papers]).drop_duplicates(subset=["ARXIV_ID"])
