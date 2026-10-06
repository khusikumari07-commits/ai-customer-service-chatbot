import json
import os
import requests
import matplotlib.pyplot as plt

import streamlit as st

from dotenv import load_dotenv
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


load_dotenv()


# ============================================================
# CONFIGURATION
# ============================================================

DATA_FILE = os.path.join(
    os.path.dirname(__file__),
    "task4_arxiv",
    "arxiv_cs_subset.json"
)


# ============================================================
# OPENROUTER API KEY
# ============================================================

def get_openrouter_key():

    try:
        if "OPENROUTER_API_KEY" in st.secrets:
            return st.secrets["OPENROUTER_API_KEY"]
    except Exception:
        pass

    return os.getenv("OPENROUTER_API_KEY")


# ============================================================
# LOAD ARXIV DATASET
# ============================================================

@st.cache_data
def load_papers():

    papers = []

    with open(
        DATA_FILE,
        "r",
        encoding="utf-8"
    ) as file:

        for line in file:

            line = line.strip()

            if line:
                papers.append(
                    json.loads(line)
                )

    return papers


# ============================================================
# PREPARE SEARCH DATA
# ============================================================

@st.cache_resource
def prepare_search():

    papers = load_papers()

    documents = []

    for paper in papers:

        title = paper.get(
            "title",
            ""
        )

        abstract = paper.get(
            "abstract",
            ""
        )

        categories = paper.get(
            "categories",
            ""
        )

        documents.append(
            f"{title} {abstract} {categories}"
        )

    vectorizer = TfidfVectorizer(
        stop_words="english",
        max_features=20000
    )

    tfidf_matrix = vectorizer.fit_transform(
        documents
    )

    return papers, vectorizer, tfidf_matrix


# ============================================================
# PAPER SEARCH
# ============================================================

def search_papers(
    query,
    top_k=5
):

    papers, vectorizer, tfidf_matrix = (
        prepare_search()
    )

    query_vector = vectorizer.transform(
        [query]
    )

    scores = cosine_similarity(
        query_vector,
        tfidf_matrix
    )[0]

    top_indices = scores.argsort()[
        -top_k:
    ][::-1]

    results = []

    for index in top_indices:

        if scores[index] > 0:

            results.append(
                {
                    "paper": papers[index],
                    "score": float(
                        scores[index]
                    )
                }
            )

    return results


# ============================================================
# CONCEPT EXTRACTION
# ============================================================

def extract_concepts(paper):

    categories = paper.get(
        "categories",
        ""
    )

    title = paper.get(
        "title",
        ""
    )

    concepts = []

    if categories:

        concepts.extend(
            categories.split()
        )

    important_words = [

        "machine learning",
        "deep learning",
        "artificial intelligence",
        "neural network",
        "computer vision",
        "natural language processing",
        "reinforcement learning",
        "robotics",
        "data mining",
        "optimization",
        "classification",
        "regression",
        "transformer",
        "large language model",
        "graph neural network",
        "computer security",
        "language model",
        "knowledge graph",
        "information retrieval",
        "question answering"

    ]

    text = (
        title
        + " "
        + paper.get(
            "abstract",
            ""
        )
    ).lower()

    for concept in important_words:

        if concept in text:

            concepts.append(
                concept
            )

    return list(
        dict.fromkeys(concepts)
    )


# ============================================================
# CONCEPT GRAPH
# ============================================================

def create_concept_graph(
    paper,
    concepts
):

    title = paper.get(
        "title",
        "Selected Paper"
    ).strip()

    concepts = concepts[:8]

    fig, ax = plt.subplots(
        figsize=(12, 6)
    )

    ax.set_xlim(
        0,
        10
    )

    ax.set_ylim(
        0,
        max(
            10,
            len(concepts) + 2
        )
    )

    ax.axis("off")

    paper_x = 5

    paper_y = (
        len(concepts) / 2
        + 1
    )

    ax.text(
        paper_x,
        paper_y,
        "SELECTED PAPER\n\n"
        + title[:70],
        ha="center",
        va="center",
        fontsize=11,
        fontweight="bold",
        bbox=dict(
            boxstyle="round,pad=0.8"
        )
    )

    for index, concept in enumerate(
        concepts
    ):

        concept_y = (
            len(concepts)
            - index
        )

        concept_x = 1.5

        ax.text(
            concept_x,
            concept_y,
            concept,
            ha="center",
            va="center",
            fontsize=10,
            bbox=dict(
                boxstyle="round,pad=0.5"
            )
        )

        ax.plot(
            [
                concept_x + 1.0,
                paper_x - 1.2
            ],
            [
                concept_y,
                paper_y
            ],
            linewidth=1
        )

    ax.set_title(
        "Research Concept Relationship",
        fontsize=14,
        fontweight="bold"
    )

    plt.tight_layout()

    return fig


# ============================================================
# OPENROUTER AI
# ============================================================

def ask_openrouter(prompt):

    api_key = get_openrouter_key()

    if not api_key:

        return (
            "OpenRouter API key was not found."
        )

    url = (
        "https://openrouter.ai/api/v1/"
        "chat/completions"
    )

    headers = {

        "Authorization":
            f"Bearer {api_key}",

        "Content-Type":
            "application/json"

    }

    data = {

        "model":
            "openrouter/free",

        "messages": [

            {
                "role": "system",

                "content": (
                    "You are a Computer Science "
                    "research assistant. "
                    "Use only the supplied arXiv "
                    "paper information. "
                    "Do not invent information. "
                    "Explain technical concepts "
                    "clearly for a student."
                )
            },

            {
                "role": "user",
                "content": prompt
            }

        ]

    }

    try:

        response = requests.post(
            url,
            headers=headers,
            json=data,
            timeout=60
        )

        if response.ok:

            result = response.json()

            return (
                result["choices"][0]
                ["message"]["content"]
            )

        return (
            "OpenRouter error: "
            + response.text
        )

    except Exception as error:

        return (
            "Connection error: "
            + str(error)
        )