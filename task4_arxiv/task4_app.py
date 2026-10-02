import streamlit as st
import json
import os
import requests
import matplotlib.pyplot as plt
from dotenv import load_dotenv
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# ============================================================
# CONFIGURATION
# ============================================================

load_dotenv()

DATA_FILE = os.path.join(
    os.path.dirname(__file__),
    "arxiv_cs_subset.json"
)

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")


# ============================================================
# LOAD ARXIV DATASET
# ============================================================

@st.cache_data
def load_papers():
    papers = []

    with open(DATA_FILE, "r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()

            if line:
                papers.append(json.loads(line))

    return papers


papers = load_papers()


# ============================================================
# PREPARE SEARCH DATA
# ============================================================

documents = []

for paper in papers:

    title = paper.get("title", "")
    abstract = paper.get("abstract", "")
    categories = paper.get("categories", "")

    documents.append(
        f"{title} {abstract} {categories}"
    )


vectorizer = TfidfVectorizer(
    stop_words="english",
    max_features=20000
)

tfidf_matrix = vectorizer.fit_transform(documents)


# ============================================================
# PAPER SEARCH
# ============================================================

def search_papers(query, top_k=5):

    query_vector = vectorizer.transform([query])

    scores = cosine_similarity(
        query_vector,
        tfidf_matrix
    )[0]

    top_indices = scores.argsort()[-top_k:][::-1]

    results = []

    for index in top_indices:

        if scores[index] > 0:

            results.append(
                {
                    "paper": papers[index],
                    "score": float(scores[index])
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

    # Add arXiv categories
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
        + paper.get("abstract", "")
    ).lower()

    for concept in important_words:

        if concept in text:

            concepts.append(
                concept
            )

    # Remove duplicates
    concepts = list(
        dict.fromkeys(concepts)
    )

    return concepts


# ============================================================
# CONCEPT GRAPH
# ============================================================

def create_concept_graph(paper, concepts):

    title = paper.get(
        "title",
        "Selected Paper"
    ).strip()

    # Limit graph size
    concepts = concepts[:8]

    fig, ax = plt.subplots(
        figsize=(12, 6)
    )

    ax.set_xlim(0, 10)

    ax.set_ylim(
        0,
        max(10, len(concepts) + 2)
    )

    ax.axis("off")

    # Paper node
    paper_x = 5
    paper_y = len(concepts) / 2 + 1

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

    # Concept nodes
    for index, concept in enumerate(concepts):

        concept_y = (
            len(concepts) - index
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

        # Connection line
        ax.plot(
            [concept_x + 1.0, paper_x - 1.2],
            [concept_y, paper_y],
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

    if not OPENROUTER_API_KEY:

        return (
            "OpenRouter API key was not found. "
            "Please check the .env file."
        )

    url = (
        "https://openrouter.ai/api/v1/"
        "chat/completions"
    )

    headers = {

        "Authorization":
            f"Bearer {OPENROUTER_API_KEY}",

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


# ============================================================
# STREAMLIT PAGE
# ============================================================

st.set_page_config(

    page_title=
        "ArXiv CS Expert Chatbot",

    page_icon=
        "📚",

    layout=
        "wide"
)


# ============================================================
# HEADER
# ============================================================

st.title(
    "📚 ArXiv Computer Science Expert Chatbot"
)

st.write(
    "Search Computer Science research papers, "
    "explore concepts, and ask an AI assistant "
    "for explanations."
)


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.header(
    "Dataset Information"
)

st.sidebar.write(
    f"📄 Papers loaded: {len(papers)}"
)

st.sidebar.write(
    "📚 Source: Cornell University ArXiv Dataset"
)

st.sidebar.write(
    "🧠 Domain: Computer Science"
)


# ============================================================
# PAPER SEARCH
# ============================================================

st.header(
    "🔎 Search Research Papers"
)

query = st.text_input(

    "Enter a Computer Science topic or research question:",

    placeholder=
        "Example: transformer models for natural language processing"
)


if query:

    results = search_papers(
        query,
        top_k=5
    )

    if not results:

        st.warning(
            "No matching papers found."
        )

    else:

        st.subheader(
            "Search Results"
        )

        for number, result in enumerate(
            results,
            start=1
        ):

            paper = result["paper"]

            title = paper.get(
                "title",
                "Untitled"
            ).strip()

            abstract = paper.get(
                "abstract",
                "No abstract available."
            ).strip()

            paper_id = paper.get(
                "id",
                ""
            )

            categories = paper.get(
                "categories",
                ""
            )

            with st.expander(
                f"{number}. {title}"
            ):

                st.write(
                    f"**arXiv ID:** {paper_id}"
                )

                st.write(
                    f"**Categories:** {categories}"
                )

                st.write(
                    f"**Relevance Score:** "
                    f"{result['score']:.3f}"
                )

                st.write(
                    "### Abstract"
                )

                st.write(
                    abstract
                )

                concepts = extract_concepts(
                    paper
                )

                st.write(
                    "### 🧠 Extracted Concepts"
                )

                if concepts:

                    st.write(
                        ", ".join(concepts)
                    )

                else:

                    st.write(
                        "No major concepts detected."
                    )


                # --------------------------------------------
                # AI PAPER EXPLANATION
                # --------------------------------------------

                if st.button(

                    "🤖 Explain this paper",

                    key=
                        f"explain_{number}"

                ):

                    prompt = f"""

Explain this Computer Science research paper
in simple but technically accurate language.

Title:
{title}

Abstract:
{abstract}

Extracted concepts:
{", ".join(concepts)}

Please provide:

1. Research problem
2. Main idea
3. Method or approach
4. Important concepts
5. Possible applications
6. Simple explanation for a student

Do not invent details that are not supported
by the supplied paper information.
"""

                    with st.spinner(
                        "Generating explanation..."
                    ):

                        answer = ask_openrouter(
                            prompt
                        )

                    st.markdown(
                        "### 🤖 AI Explanation"
                    )

                    st.write(
                        answer
                    )

                    # Save selected paper
                    st.session_state[
                        "selected_paper"
                    ] = paper


# ============================================================
# FOLLOW-UP QUESTIONS
# ============================================================

if "selected_paper" in st.session_state:

    selected = st.session_state[
        "selected_paper"
    ]

    st.header(
        "💬 Ask Follow-up Questions"
    )

    followup = st.text_input(

        "Ask something about the selected paper:",

        placeholder=
            "Example: What problem does this research solve?"

    )

    if followup:

        title = selected.get(
            "title",
            ""
        ).strip()

        abstract = selected.get(
            "abstract",
            ""
        ).strip()

        concepts = extract_concepts(
            selected
        )

        prompt = f"""

Answer the user's follow-up question
using the selected research paper
as context.

Paper title:
{title}

Paper abstract:
{abstract}

Research concepts:
{", ".join(concepts)}

User question:
{followup}

Answer clearly and technically accurately.

Do not invent information that is not
supported by the paper.
"""

        with st.spinner(
            "Thinking..."
        ):

            answer = ask_openrouter(
                prompt
            )

        st.markdown(
            "### 🤖 Answer"
        )

        st.write(
            answer
        )


# ============================================================
# CONCEPT VISUALIZATION
# ============================================================

st.header(
    "🧠 Concept Visualization"
)

st.write(
    "Select a paper to view its extracted "
    "research concepts and their relationship "
    "to the selected paper."
)


paper_titles = [

    paper.get(
        "title",
        ""
    ).strip()

    for paper in papers[:200]

]


selected_title = st.selectbox(

    "Choose a paper:",

    paper_titles

)


selected_paper_for_graph = next(

    (

        paper

        for paper in papers

        if paper.get(
            "title",
            ""
        ).strip()
        == selected_title

    ),

    None

)


if selected_paper_for_graph:

    concepts = extract_concepts(
        selected_paper_for_graph
    )

    st.write(
        "**Detected concepts:**"
    )

    if concepts:

        st.write(
            " • ".join(concepts)
        )

        st.info(
            "The concepts are extracted from "
            "the paper's title, abstract and "
            "arXiv categories."
        )

        # --------------------------------------------
        # VISUAL GRAPH
        # --------------------------------------------

        st.write(
            "### 📊 Concept Relationship Graph"
        )

        fig = create_concept_graph(

            selected_paper_for_graph,

            concepts

        )

        st.pyplot(
            fig
        )

        plt.close(fig)

    else:

        st.write(
            "No major concepts detected."
        )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "ArXiv CS Expert Chatbot | "
    "Built using Streamlit, TF-IDF, NLP retrieval, "
    "Matplotlib and OpenRouter"
)