import chromadb
from sentence_transformers import SentenceTransformer


DB_PATH = "./knowledge_db"
COLLECTION_NAME = "dynamic_knowledge"


print("Loading embedding model...")
model = SentenceTransformer("all-MiniLM-L6-v2")

client = chromadb.PersistentClient(path=DB_PATH)

collection = client.get_or_create_collection(
    name=COLLECTION_NAME
)


def search_knowledge(query, results=3):
    query_embedding = model.encode([query]).tolist()

    search_results = collection.query(
        query_embeddings=query_embedding,
        n_results=results
    )

    documents = search_results.get("documents", [[]])[0]
    metadatas = search_results.get("metadatas", [[]])[0]

    return documents, metadatas


if __name__ == "__main__":

    question = input("Ask something about the knowledge base: ")

    documents, metadatas = search_knowledge(question)

    print("\nSearch results:")
    print("-----------------------------")

    if not documents:
        print("No information found.")
    else:
        for i, document in enumerate(documents, start=1):
            print(f"\nResult {i}:")
            print(document)

            if i <= len(metadatas):
                print(f"Source: {metadatas[i - 1].get('source', 'Unknown')}")