import streamlit as st
import requests

BACKEND_URL = "http://localhost:8000"

st.title("Embeddiangt")

# Sidebar tool selection
selected_tool = st.sidebar.radio("Select a tool:", ["Token Calculator", "Cosine Similarity"])

# OpenAI model options
openai_models = [
    "gpt-3.5-turbo",
    "gpt-4",
    "text-davinci-003",
    "text-curie-001",
    "text-babbage-001",
    "text-ada-001"
]

if selected_tool == "Token Calculator":
    st.header("Token Calculator")
    model = st.sidebar.selectbox("OpenAI Model", openai_models)
    text = st.text_area("Enter text to count tokens:")
    if st.button("Calculate Tokens"):
        if text.strip():
            response = requests.post(f"{BACKEND_URL}/tokenize", json={"text": text, "model": model})
            if response.ok:
                st.success(f"Token count: {response.json()['token_count']}")
            else:
                st.error("Error: " + response.text)
        else:
            st.warning("Please enter some text.")

elif selected_tool == "Cosine Similarity":
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
