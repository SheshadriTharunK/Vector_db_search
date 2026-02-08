import os
from pathlib import Path
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

load_dotenv(override=True)

print("model name", os.getenv("HF_EMBEDDINGS_MODEL"))
embeddings_model = HuggingFaceEmbeddings(
    model_name = os.getenv("HF_EMBEDDINGS_MODEL"),
    encode_kwargs = {"normalize_embeddings": True},
    model_kwargs = {"token": os.getenv("HUGGING_FACE_TOKEN")},
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

def run_search(query: str, k: int = 5):
    hits = vector_db.similarity_search_with_score(query, k=k)

    if not hits:
        return {
            "best_match": None,
            "matches": []
        }

    # Best match
    best_doc, best_score = hits[0]
    best_match = {
        "name": best_doc.page_content,
        "score": float(round(float(best_score), 4))
    }

    # Other matches
    matches = []
    for doc, score in hits[1:]:
        matches.append({
            "name": doc.page_content,
            "score": float(round(float(score), 4))
        })

    return {
        "best_match": best_match,
        "matches": matches
    }
