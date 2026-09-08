import hashlib
import re
import xml.etree.ElementTree as ET

import chromadb
import requests
from sentence_transformers import SentenceTransformer


# ---------------------------------
# Task 3: Dynamic Knowledge Base
# ---------------------------------

DB_PATH = "./knowledge_db"
COLLECTION_NAME = "dynamic_knowledge"


# Information sources
SOURCES = {
    "Python Insider": "https://blog.python.org/rss.xml",
    "Hacker News": "https://news.ycombinator.com/rss",
}


# ---------------------------------
# Load embedding model
# ---------------------------------

print("Loading embedding model...")

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

print("Embedding model loaded successfully.")


# ---------------------------------
# Create persistent ChromaDB
# ---------------------------------

client = chromadb.PersistentClient(
    path=DB_PATH
)

collection = client.get_or_create_collection(
    name=COLLECTION_NAME
)


# ---------------------------------
# Clean downloaded text
# ---------------------------------

def clean_text(text):
    if not text:
        return ""

    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text)

    return text.strip()


# ---------------------------------
# Create unique ID for each item
# ---------------------------------

def create_id(source, title, link):

    unique_text = f"{source}-{title}-{link}"

    return hashlib.md5(
        unique_text.encode("utf-8")
    ).hexdigest()


# ---------------------------------
# Download information from source
# ---------------------------------

def fetch_source(source_name, url):

    print(f"\nFetching information from: {source_name}")

    try:

        response = requests.get(
            url,
            timeout=20,
            headers={
                "User-Agent": "AI Customer Service Chatbot"
            }
        )

        response.raise_for_status()

        root = ET.fromstring(
            response.content
        )

        articles = []

        # Handle RSS feeds
        for item in root.findall(".//item"):

            title = clean_text(
                item.findtext("title", "")
            )

            description = clean_text(
                item.findtext("description", "")
            )

            link = clean_text(
                item.findtext("link", "")
            )

            if title and description:

                articles.append({
                    "title": title,
                    "text": description,
                    "link": link
                })

        # Handle Atom feeds
        namespace = {
            "atom": "http://www.w3.org/2005/Atom"
        }

        for entry in root.findall(
            ".//atom:entry",
            namespace
        ):

            title = clean_text(
                entry.findtext(
                    "atom:title",
                    "",
                    namespace
                )
            )

            summary = clean_text(
                entry.findtext(
                    "atom:summary",
                    "",
                    namespace
                )
            )

            link_element = entry.find(
                "atom:link",
                namespace
            )

            link = ""

            if link_element is not None:

                link = link_element.get(
                    "href",
                    ""
                )

            if title and summary:

                articles.append({
                    "title": title,
                    "text": summary,
                    "link": link
                })

        print(
            f"Found {len(articles)} items."
        )

        return articles

    except Exception as error:

        print(
            f"Error while fetching "
            f"{source_name}: {error}"
        )

        return []


# ---------------------------------
# Update the vector database
# ---------------------------------

def update_knowledge_base():

    total_processed = 0

    for source_name, url in SOURCES.items():

        articles = fetch_source(
            source_name,
            url
        )

        if not articles:
            continue

        documents = []
        ids = []
        metadatas = []

        for article in articles:

            document = (
                f"Title: {article['title']}\n"
                f"Information: {article['text']}"
            )

            article_id = create_id(
                source_name,
                article["title"],
                article["link"]
            )

            documents.append(document)

            ids.append(article_id)

            metadatas.append({
                "source": source_name,
                "title": article["title"],
                "link": article["link"]
            })

        # Generate embeddings
        embeddings = embedding_model.encode(
            documents
        ).tolist()

        # Add new information or update
        # information that already exists
        collection.upsert(
            ids=ids,
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas
        )

        total_processed += len(documents)

        print(
            f"{source_name}: "
            f"{len(documents)} items "
            f"added/updated."
        )

    print("\n==============================")
    print("Knowledge base update complete")
    print("==============================")

    print(
        f"Items processed this update: "
        f"{total_processed}"
    )

    print(
        f"Total items in vector database: "
        f"{collection.count()}"
    )


# ---------------------------------
# Start the update process
# ---------------------------------

if __name__ == "__main__":

    update_knowledge_base()