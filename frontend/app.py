import streamlit as st
import requests

BACKEND_URL = "http://localhost:8000"

st.set_page_config(page_title="Embeddian! - Text Tools Suite", page_icon=None, layout="wide")

# Tool descriptions
tool_descriptions = {
    "Token Calculator": "Token Calculator\n\nCount tokens and characters for your prompt using OpenAI or Hugging Face models. Useful for checking LLM context limits.",
    "Cosine Similarity": "Cosine Similarity\n\nCompare two texts and see how similar they are using cosine similarity.",
    "Readability Analyzer": "Readability Analyzer\n\nAnalyze the readability and complexity of your text using standard metrics.",
    "Keyword/Entity Extractor": "Keyword/Entity Extractor\n\nExtract keywords and named entities from your text for quick content analysis.",
    "Embedding Visualizer": "Embedding Visualizer\n\nVisualize text embeddings in 2D/3D space to explore semantic relationships.",
    "Text Summarization": "Text Summarization\n\nGenerate concise summaries of long texts using LLMs.",
    "Paraphrasing": "Paraphrasing\n\nRephrase or rewrite text in different words using LLMs.",
    "Grammar/Spelling Correction": "Grammar/Spelling Correction\n\nAutomatically correct grammar and spelling mistakes using LLMs.",
    "Prompt Enhancer": "Prompt Enhancer\n\nRewrite or expand prompts for better LLM results.",
    "Prompt Generator": "Prompt Generator\n\nGenerate reusable prompt templates for common LLM tasks.",
    "Few-shot Example Generator": "Few-shot Example Generator\n\nGenerate more high-quality few-shot examples for prompt engineering.",
    "LLM Output Analyzer": "LLM Output Analyzer\n\nAnalyze LLM outputs for bias, toxicity, or hallucination.",
    "Prompt Cost Estimator": "Prompt Cost Estimator\n\nEstimate the token/cost usage of a prompt for different LLMs."
}

# Sidebar layout
st.sidebar.title("Embeddian! Tools")
st.sidebar.markdown("""
Welcome to Embeddian!

A suite of text tools for LLM and NLP workflows.
""")

# Sidebar tool selection
selected_tool = st.sidebar.radio(
    "Select a tool:",
    list(tool_descriptions.keys()),
    format_func=lambda x: x
)

# Show tool description in sidebar
st.sidebar.markdown(tool_descriptions[selected_tool])
st.sidebar.divider()

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

# Main area layout
st.title(f"{selected_tool}")
st.markdown(tool_descriptions[selected_tool])
st.divider()

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
    text = st.text_area("Enter text to analyze readability:")
    if st.button("Analyze Readability"):
        if text.strip():
            response = requests.post(f"{BACKEND_URL}/readability", json={"text": text})
            if response.ok:
                data = response.json()
                st.subheader("Readability Metrics:")
                st.write(f"Flesch Reading Ease: **{data['flesch_reading_ease']:.2f}**")
                st.write(f"Flesch-Kincaid Grade: **{data['flesch_kincaid_grade']:.2f}**")
                st.write(f"SMOG Index: **{data['smog_index']:.2f}**")
                st.write(f"Coleman-Liau Index: **{data['coleman_liau_index']:.2f}**")
                st.write(f"Automated Readability Index: **{data['automated_readability_index']:.2f}**")
            else:
                st.error("Error: " + response.text)
        else:
            st.warning("Please enter some text.")

elif selected_tool == "Keyword/Entity Extractor":
    st.header("Keyword/Entity Extractor")
    text = st.text_area("Enter text to extract keywords and entities:")
    if st.button("Extract Keywords/Entities"):
        if text.strip():
            response = requests.post(f"{BACKEND_URL}/extract", json={"text": text})
            if response.ok:
                data = response.json()
                st.subheader("Keywords:")
                if data["keywords"]:
                    st.write(", ".join(data["keywords"]))
                else:
                    st.write("No keywords found.")
                st.subheader("Named Entities:")
                if data["entities"]:
                    for ent in data["entities"]:
                        st.write(f"{ent['text']} ({ent['label']})")
                else:
                    st.write("No named entities found.")
            else:
                st.error("Error: " + response.text)
        else:
            st.warning("Please enter some text.")

elif selected_tool == "Embedding Visualizer":
    st.header("Embedding Visualizer")
    st.markdown("Enter multiple texts (one per line) to visualize their embeddings in 2D space.")
    text_input = st.text_area("Enter texts (one per line):")
    if st.button("Visualize Embeddings"):
        texts = [t.strip() for t in text_input.splitlines() if t.strip()]
        if len(texts) < 2:
            st.warning("Please enter at least two texts.")
        else:
            response = requests.post(f"{BACKEND_URL}/embed", json={"texts": texts})
            if response.ok:
                data = response.json()
                import numpy as np
                from sklearn.decomposition import PCA
                embeddings = np.array(data["embeddings"])
                pca = PCA(n_components=2)
                reduced = pca.fit_transform(embeddings)
                import pandas as pd
                df = pd.DataFrame(reduced, columns=["x", "y"])
                df["text"] = texts
                st.subheader("2D Embedding Visualization:")
                st.scatter_chart(df, x="x", y="y")
                for i, row in df.iterrows():
                    st.write(f"{row['text']}: ({row['x']:.2f}, {row['y']:.2f})")
            else:
                st.error("Error: " + response.text)

elif selected_tool == "Text Summarization":
    st.header("Text Summarization")
    text = st.text_area("Enter text to summarize:")
    max_length = st.slider("Maximum summary length (words):", min_value=50, max_value=300, value=150, step=10)
    if st.button("Generate Summary"):
        if text.strip():
            response = requests.post(f"{BACKEND_URL}/summarize", json={"text": text, "max_length": max_length})
            if response.ok:
                data = response.json()
                st.subheader("Generated Summary:")
                st.write(data["summary"])
            else:
                st.error("Error: " + response.text)
        else:
            st.warning("Please enter some text.")
elif selected_tool == "Paraphrasing":
    st.header("Paraphrasing")
    text = st.text_area("Enter text to paraphrase:")
    if st.button("Paraphrase Text"):
        if text.strip():
            response = requests.post(f"{BACKEND_URL}/paraphrase", json={"text": text})
            if response.ok:
                data = response.json()
                st.subheader("Paraphrased Text:")
                st.write(data["paraphrased_text"])
            else:
                st.error("Error: " + response.text)
        else:
            st.warning("Please enter some text.")
elif selected_tool == "Grammar/Spelling Correction":
    st.header("Grammar/Spelling Correction")
    text = st.text_area("Enter text to correct grammar and spelling:")
    if st.button("Correct Text"):
        if text.strip():
            response = requests.post(f"{BACKEND_URL}/grammar-correction", json={"text": text})
            if response.ok:
                data = response.json()
                st.subheader("Corrected Text:")
                st.write(data["corrected_text"])
            else:
                st.error("Error: " + response.text)
        else:
            st.warning("Please enter some text.")
elif selected_tool == "Prompt Enhancer":
    st.header("Prompt Enhancer")
    st.info("This tool will rewrite or expand prompts for better LLM results. (Coming soon)")
elif selected_tool == "Prompt Generator":
    st.header("Prompt Generator")
    st.info("This tool will generate reusable prompt templates for common LLM tasks. (Coming soon)")
elif selected_tool == "Few-shot Example Generator":
    st.header("Few-shot Example Generator")
    st.info("This tool will generate more high-quality few-shot examples for prompt engineering. (Coming soon)")
elif selected_tool == "LLM Output Analyzer":
    st.header("LLM Output Analyzer")
    st.info("This tool will analyze LLM outputs for bias, toxicity, or hallucination. (Coming soon)")
elif selected_tool == "Prompt Cost Estimator":
    st.header("Prompt Cost Estimator")
    st.info("This tool will estimate the token/cost usage of a prompt for different LLMs. (Coming soon)")
