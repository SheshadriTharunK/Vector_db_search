import os
from pathlib import Path
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

load_dotenv(override=True)

embeddings_model = HuggingFaceEmbeddings(
    model_name=os.getenv("HF_EMBEDDINGS_MODEL"),
    encode_kwargs={"normalize_embeddings": True},
    model_kwargs={"token": os.getenv("HUGGING_FACE_TOKEN")},
)
BASE_DIR = Path(__file__).resolve().parent
print(BASE_DIR)
vector_db_dir = BASE_DIR / "data" / "semantic-search" / "index" / "faiss1"
print(vector_db_dir)
vector_db = FAISS.load_local(
    folder_path=vector_db_dir,
    embeddings=embeddings_model,
    allow_dangerous_deserialization=True,
)

# Query sentences:
query = input("Enter a name to search for similar names: ")

# Find the closest 5 chunks for each query based on similarity measure
hits = vector_db.similarity_search_with_score(query, k=5)
print("\nQuery:", query)
print("Best Match (Most similar name): ",hits[0][0].page_content, f"Score: {hits[0][1]:.4f}")
print("Top 4 most similar chunks:")
for hit in hits[1:]:
    print(hit[0].page_content, f"Score: {hit[1]:.4f}")
    print("-----")
