import streamlit as st
import requests

BACKEND_URL = "http://localhost:8000"

st.title("Embeddian!")

# Sidebar tool selection
selected_tool = st.sidebar.radio(
    "Select a tool:",
    [
        "Token Calculator",
        "Cosine Similarity",
        "Readability Analyzer",
        "Keyword/Entity Extractor",
        "Embedding Visualizer"
    ]
)

# OpenAI model categories (as per official token calculator)
openai_model_categories = {
    "GPT-4o & GPT-4o mini": "gpt-4o",  # Use gpt-4o as representative
    "GPT-3.5 & GPT-4": "gpt-3.5-turbo",  # Use gpt-3.5-turbo as representative
    "GPT-3 (Legacy)": "text-davinci-003"  # Use text-davinci-003 as representative
}

# Expanded Hugging Face models (including DeepSeek and Qwen, with Custom as second option)
hf_models = [
    # Mistral
    "mistralai/Mistral-7B-v0.1",
    # Custom
    "Custom (enter below)",
    "OpenPipe/mistral-ft-optimized-1218",
    "mistralai/Mixtral-8x7B-v0.1",
    # Falcon
    "tiiuae/falcon-7b",
    "tiiuae/falcon-40b",
    # GPT-2 family
    "gpt2",
    "gpt2-medium",
    "gpt2-large",
    "gpt2-xl",
    # Bloom
    "bigscience/bloom-560m",
    "bigscience/bloom",
    # OPT
    "facebook/opt-1.3b",
    "facebook/opt-6.7b",
    # GPT-Neo/J
    "EleutherAI/gpt-neo-1.3B",
    "EleutherAI/gpt-neo-2.7B",
    "EleutherAI/gpt-j-6B",
    # DeepSeek
    "deepseek-ai/deepseek-llm-7b-base",
    "deepseek-ai/deepseek-llm-7b-chat",
    "deepseek-ai/deepseek-llm-67b-base",
    "deepseek-ai/deepseek-llm-67b-chat",
    # Qwen
    "Qwen/Qwen1.5-0.5B",
    "Qwen/Qwen1.5-1.8B",
    "Qwen/Qwen1.5-4B",
    "Qwen/Qwen1.5-7B",
    "Qwen/Qwen1.5-14B",
    "Qwen/Qwen1.5-32B",
    "Qwen/Qwen1.5-72B",
    "Qwen/Qwen1.5-72B-Chat",
    "Qwen/Qwen1.5-7B-Chat",
    "Qwen/Qwen1.5-4B-Chat",
    "Qwen/Qwen1.5-1.8B-Chat"
]

if selected_tool == "Token Calculator":
    st.header("Token Calculator")
    # Model provider selection (only for Token Calculator)
    provider = st.sidebar.selectbox("Model Provider", ["OpenAI", "Hugging Face"])
    if provider == "OpenAI":
        category = st.sidebar.selectbox("OpenAI Model", list(openai_model_categories.keys()))
        model = openai_model_categories[category]
        provider_key = "openai"
    else:
        hf_model_choice = st.sidebar.selectbox("Hugging Face Model", hf_models)
        if hf_model_choice == "Custom (enter below)":
            model = st.sidebar.text_input("Enter Hugging Face Model ID", "gpt2")
        else:
            model = hf_model_choice
        provider_key = "hf"
    text = st.text_area("Enter text to count tokens:")
    if st.button("Calculate Tokens"):
        if text.strip():
            response = requests.post(
                f"{BACKEND_URL}/tokenize",
                json={"text": text, "model": model, "provider": provider_key}
            )
            if response.ok:
                data = response.json()
                st.success(f"Token count: {data['token_count']}\nCharacter count: {data['character_count']}")
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

elif selected_tool == "Readability Analyzer":
    st.header("Readability Analyzer (Text Complexity)")
    st.info("This tool will analyze the readability and complexity of your text. (Coming soon)")

elif selected_tool == "Keyword/Entity Extractor":
    st.header("Keyword/Entity Extractor")
    st.info("This tool will extract keywords and named entities from your text. (Coming soon)")

elif selected_tool == "Embedding Visualizer":
    st.header("Embedding Visualizer")
    st.info("This tool will visualize text embeddings in 2D/3D space. (Coming soon)")
