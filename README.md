# Embeddian! - Text Tools Suite

A comprehensive suite of text processing tools for LLM and NLP workflows. This application provides a web-based interface for various text analysis and processing tasks, including token counting, text summarization, grammar correction, and more.

## Features

- **Token Calculator**: Count tokens and characters using OpenAI or Hugging Face models
- **Cosine Similarity**: Compare texts for semantic similarity
- **Readability Analyzer**: Assess text complexity using standard readability metrics
- **Keyword/Entity Extractor**: Extract keywords and named entities from text
- **Embedding Visualizer**: Visualize text embeddings in 2D space
- **Text Summarization**: Generate concise summaries using LLMs
- **Paraphrasing**: Rephrase text using LLMs
- **Grammar/Spelling Correction**: Automatically correct grammar and spelling
- **Prompt Enhancer**: Improve prompts for better LLM results
- **Prompt Generator**: Create reusable prompt templates
- **Few-shot Example Generator**: Generate high-quality examples for prompt engineering
- **LLM Output Analyzer**: Analyze LLM outputs for quality metrics
- **Prompt Cost Estimator**: Estimate token/cost usage (coming soon)

## Prerequisites

- Python 3.8 or higher
- pip (Python package installer)

## Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd tools
   ```

2. Install backend dependencies:
   ```bash
   cd backend
   pip install -r requirements.txt
   cd ..
   ```

3. Install frontend dependencies:
   ```bash
   cd frontend
   pip install -r requirements.txt
   cd ..
   ```

## Usage

### Running the Backend Server

1. Navigate to the backend directory:
   ```bash
   cd backend
   ```

2. Start the FastAPI server:
   ```bash
   uvicorn main:app --reload
   ```

   The backend API will be available at `http://localhost:8000`.

### Running the Frontend Application

1. Navigate to the frontend directory:
   ```bash
   cd frontend
   ```

2. Start the Streamlit application:
   ```bash
   streamlit run app.py
   ```

   The frontend will be available at `http://localhost:8501`.

### Using the Tools

1. Open your browser and navigate to `http://localhost:8501`
2. Select a tool from the sidebar
3. Enter the required text or parameters
4. Click the appropriate button to process your request

## API Endpoints

The backend provides the following REST API endpoints:

- `POST /tokenize` - Count tokens and characters in text
- `POST /cosine-similarity` - Calculate cosine similarity between two texts
- `POST /summarize` - Generate a summary of provided text
- `POST /paraphrase` - Rephrase provided text
- `POST /grammar-correction` - Correct grammar and spelling in text
- `POST /prompt-enhancer` - Enhance a given prompt
- `POST /prompt-generator` - Generate a prompt based on a task description
- `POST /few-shot-example-generator` - Generate few-shot examples
- `POST /llm-output-analyzer` - Analyze LLM output for quality metrics

## Supported Models

### OpenAI Models
- GPT-4o & GPT-4o mini
- GPT-3.5 & GPT-4
- GPT-3 (Legacy)

### Hugging Face Models
- Mistral models
- Falcon models
- GPT-2 family
- BLOOM models
- OPT models
- GPT-Neo/J models
- DeepSeek models
- Qwen models

## Project Structure

```
.
├── backend/                 # FastAPI backend service
│   ├── main.py             # API endpoints
│   └── requirements.txt    # Backend dependencies
├── frontend/               # Streamlit frontend application
│   ├── app.py              # Main Streamlit application
│   └── requirements.txt    # Frontend dependencies
└── README.md              # This file
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.