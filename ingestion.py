import os
from pathlib import Path
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.faiss import DistanceStrategy

load_dotenv(override=True)

embeddings_model = HuggingFaceEmbeddings(
    model_name=os.getenv("HF_EMBEDDINGS_MODEL"),
    encode_kwargs={"normalize_embeddings": True},
    model_kwargs={"token": os.getenv("HUGGING_FACE_TOKEN")},
)

chunks = [
  "Gita", "Geeta", "Gitu", "Githa",
    "Seetha", "Sita", "Seeta", "Sitha",
    "Ramesh", "Rameshwar", "Ram", "Ramu",
    "Suresh", "Suresh Kumar", "Sureshwar",
    "Mahesh", "Maheshwari",
    "Rajesh", "Raj", "Raju",
    "Ganesh", "Ganesha",
    "Kavitha", "Kavita", "Kavya",
    "Anita", "Anitha", "Anu"
]

# build vector database
vector_db = FAISS.from_texts(
    texts=chunks,
    embedding=embeddings_model,
    distance_strategy=DistanceStrategy.EUCLIDEAN_DISTANCE,
)
print(vector_db.index)
print(vector_db.docstore)
print(vector_db.index_to_docstore_id)
BASE_DIR = Path(__file__).resolve().parent
# parent.parent → move up to project root
print(BASE_DIR)
vector_db_dir = BASE_DIR /  "data" / "semantic-search" / "index" / "faiss1"

print(vector_db_dir)
vector_db.save_local(folder_path=vector_db_dir)
