# Semantic Name Matching Application 🔍

## Project Overview
This project implements a **semantic name-matching system** that finds the most similar person names from a dataset when a user enters a name.  
It uses **vector embeddings and FAISS vector database** to perform similarity search and displays results using a **Streamlit web interface**.

The application returns:
- ✅ **Best Match** (closest name with similarity score)
- 📋 **List of other similar names** ranked by similarity

---

## Features
- Semantic similarity search using embeddings
- FAISS vector database for fast retrieval
- Streamlit-based interactive UI
- OS-independent dynamic file path handling
- Clean separation between backend logic and UI

---

## Tech Stack
- **Python 3.9+**
- **LangChain**
- **HuggingFace Embeddings**
- **FAISS Vector Database**
- **Streamlit**
- **dotenv**
- **Pydantic_ai**
- **fastapi**

---

## Project Structure
semantic-search-app/

│
├── app.py # Streamlit application
├── requirements.txt
├── .env # Environment variables (not committed)
│
├── data/
│ └── semantic-search/
│    └── index/
│    └── faiss1/ # FAISS index files
|___ingestion.py 
|__retrieval.py


---

## Setup Instructions

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/<your-username>/semantic-search-app.git
cd semantic-search-app
2️⃣ Create Virtual Environment (Optional but Recommended)
python -m venv venv
source venv/bin/activate      # Linux/Mac
venv\Scripts\activate         # Windows
3️⃣ Install Dependencies
pip install -r requirements.txt
4️⃣ Environment Variables
Create a .env file in the root directory:

HF_EMBEDDINGS_MODEL=sentence-transformers/all-MiniLM-L6-v2
HUGGING_FACE_TOKEN=your_huggingface_token
How to Run the Application
Start the Streamlit app:

streamlit run app.py
The app will open automatically in your browser.

Sample Input & Output
Input
Geet
Output
Best Match:
Geeta | Score: 0.7467

Other Similar Names:
Gita  | Score: 0.81
Gitu  | Score: 0.92
Geetha | Score: 1.01
Note: Lower score indicates higher similarity.

Notes & Assumptions
FAISS similarity scores represent vector distance (lower = better).

The FAISS index is loaded once at startup for performance.

.env file should not be pushed to GitHub.

This project is fully runnable on Windows/Linux systems.
