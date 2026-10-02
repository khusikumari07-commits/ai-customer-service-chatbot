import json
import re
from pathlib import Path

DATA_FILE = Path(__file__).parent / "arxiv_cs_subset.json"


def load_papers():
    papers = []

    with open(DATA_FILE, "r", encoding="utf-8") as file:
        for line in file:
            paper = json.loads(line)
            papers.append(paper)

    return papers


def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return set(text.split())


def search_papers(query, papers, top_k=5):
    query_words = clean_text(query)

    results = []

    for paper in papers:
        title_words = clean_text(paper.get("title", ""))
        abstract_words = clean_text(paper.get("abstract", ""))

        title_score = len(query_words & title_words) * 3
        abstract_score = len(query_words & abstract_words)

        score = title_score + abstract_score

        if score > 0:
            results.append((score, paper))

    results.sort(key=lambda item: item[0], reverse=True)

    return [paper for score, paper in results[:top_k]]


if __name__ == "__main__":
    papers = load_papers()

    print("Computer Science papers loaded:", len(papers))

    query = input("\nEnter a topic to search: ")

    results = search_papers(query, papers)

    print("\nSearch results:\n")

    if not results:
        print("No matching papers found.")
    else:
        for number, paper in enumerate(results, start=1):
            print(f"{number}. {paper['title'].strip()}")
            print(f"   arXiv ID: {paper['id']}")
            print(f"   Categories: {paper['categories']}")
            print()