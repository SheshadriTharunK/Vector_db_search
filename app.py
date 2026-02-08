# app.py
import streamlit as st
from retrieval import run_search

st.set_page_config(page_title="Semantic Name Search", layout="centered")

st.title("🔍 Semantic Search App")
st.write("Find the most similar names using vector similarity.")

query = st.text_input("Enter a name to search")
if query:
    results = run_search(query)
    print("results",results)
    st.subheader("✅ Best Match")
    st.success(f"{results['best_match']['name']}  |  Score: {results['best_match']['score']}")

    st.subheader("📊 Other Similar Names")
    for r in results['matches']:
        st.write(f"• {r['name']} — score: {r['score']}")
