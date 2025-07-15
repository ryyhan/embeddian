import streamlit as st
import requests

BACKEND_URL = "http://localhost:8000"

st.title("Text Tools Suite")

# Tabs for each tool
tab1, tab2 = st.tabs(["Token Calculator", "Cosine Similarity"])

with tab1:
    st.header("Token Calculator")
    text = st.text_area("Enter text to count tokens:")
    if st.button("Calculate Tokens"):
        if text.strip():
            response = requests.post(f"{BACKEND_URL}/tokenize", json={"text": text})
            if response.ok:
                st.success(f"Token count: {response.json()['token_count']}")
            else:
                st.error("Error: " + response.text)
        else:
            st.warning("Please enter some text.")

with tab2:
    st.header("Cosine Similarity")
    text1 = st.text_area("Text 1:", key="text1")
    text2 = st.text_area("Text 2:", key="text2")
    if st.button("Compare Texts"):
        if text1.strip() and text2.strip():
            response = requests.post(f"{BACKEND_URL}/cosine-similarity", json={"text1": text1, "text2": text2})
            if response.ok:
                st.success(f"Cosine similarity: {response.json()['cosine_similarity']:.4f}")
            else:
                st.error("Error: " + response.text)
        else:
            st.warning("Please enter both texts.")
