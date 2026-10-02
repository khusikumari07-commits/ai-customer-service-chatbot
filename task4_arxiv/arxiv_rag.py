import json
import re
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# Location of the 2,000 Computer Science papers
DATA_FILE = Path(__file__).parent / "arxiv_cs_subset.json"


def load_papers():
    """Load the Computer Science paper subset."""
    papers = []

    with open(DATA_FILE, "r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()

            if line:
                papers.append(json.loads(line))

    return papers


def clean_text(text):
    """Clean text for NLP search."""
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def build_search_index(papers):
    """Create a TF-IDF search index from paper titles and abstracts."""
    documents = []

    for paper in papers:
        title = paper.get("title", "")
        abstract = paper.get("abstract", "")

        documents.append(
            clean_text(title + " " + abstract)
        )

    vectorizer = TfidfVectorizer(
        stop_words="english",
        max_features=20000
    )

    matrix = vectorizer.fit_transform(documents)

    return vectorizer, matrix


def search_papers(query, papers, vectorizer, matrix, top_k=5):
    """Search the most relevant papers."""
    query_vector = vectorizer.transform([clean_text(query)])

    similarities = cosine_similarity(
        query_vector,
        matrix
    ).flatten()

    ranked_indexes = similarities.argsort()[::-1]

    results = []

    for index in ranked_indexes:
        score = similarities[index]

        if score <= 0:
            continue

        paper = papers[index].copy()
        paper["similarity"] = float(score)

        results.append(paper)

        if len(results) >= top_k:
            break

    return results


def extract_information(paper):
    """Extract useful information from a research paper."""
    title = paper.get("title", "").strip()
    abstract = paper.get("abstract", "").strip()
    categories = paper.get("categories", "").strip()

    concepts = []

    keywords = [
        "machine learning",
        "deep learning",
        "neural network",
        "artificial intelligence",
        "computer vision",
        "natural language processing",
        "reinforcement learning",
        "robotics",
        "data mining",
        "optimization",
        "classification",
        "clustering",
        "transformer",
        "attention",
        "generative ai",
        "large language model",
        "graph neural network",
    ]

    abstract_lower = abstract.lower()

    for keyword in keywords:
        if keyword in abstract_lower:
            concepts.append(keyword)

    return {
        "title": title,
        "abstract": abstract,
        "categories": categories,
        "concepts": concepts
    }


if __name__ == "__main__":

    print("Loading Computer Science papers...")

    papers = load_papers()

    print(f"Loaded {len(papers)} papers.")

    print("Building NLP search index...")

    vectorizer, matrix = build_search_index(papers)

    print("Search system ready.")

    query = input(
        "\nEnter a Computer Science research topic: "
    ).strip()

    results = search_papers(
        query,
        papers,
        vectorizer,
        matrix,
        top_k=5
    )

    print("\nTop research papers:\n")

    if not results:
        print("No matching papers found.")

    else:
        for number, paper in enumerate(results, start=1):

            information = extract_information(paper)

            print(
                f"{number}. "
                f"{information['title']}"
            )

            print(
                f"   arXiv ID: "
                f"{paper.get('id', '')}"
            )

            print(
                f"   Categories: "
                f"{information['categories']}"
            )

            print(
                f"   Similarity: "
                f"{paper['similarity']:.3f}"
            )

            print(
                f"   Concepts: "
                f"{', '.join(information['concepts']) if information['concepts'] else 'General CS'}"
            )

            print(
                f"   Abstract: "
                f"{information['abstract'][:400]}..."
            )

            print()