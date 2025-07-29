from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import tiktoken
from transformers import AutoTokenizer
from sentence_transformers import SentenceTransformer
import numpy as np
from textstat import flesch_reading_ease, flesch_kincaid_grade, smog_index, coleman_liau_index, automated_readability_index
import spacy

app = FastAPI()

class TokenizeRequest(BaseModel):
    text: str
    model: str
    provider: str = "openai"  # 'openai' or 'hf'

class CosineSimilarityRequest(BaseModel):
    text1: str
    text2: str

class ReadabilityRequest(BaseModel):
    text: str

class KeywordEntityRequest(BaseModel):
    text: str

@app.post("/tokenize")
def tokenize(request: TokenizeRequest) -> Dict[str, int]:
    char_count = len(request.text)
    if request.provider == "openai":
        try:
            encoding = tiktoken.encoding_for_model(request.model)
            tokens = encoding.encode(request.text)
        except Exception:
            raise HTTPException(status_code=400, detail=f"Unsupported or unknown OpenAI model: {request.model}")
    elif request.provider == "hf":
        try:
            tokenizer = AutoTokenizer.from_pretrained(request.model)
            tokens = tokenizer.encode(request.text)
        except Exception:
            raise HTTPException(status_code=400, detail=f"Unsupported or unknown Hugging Face model: {request.model}")
    else:
        raise HTTPException(status_code=400, detail=f"Unsupported provider: {request.provider}")
    return {"token_count": len(tokens), "character_count": char_count}

@app.post("/cosine-similarity")
def cosine_similarity_endpoint(request: CosineSimilarityRequest) -> Dict[str, float]:
    model = SentenceTransformer('all-MiniLM-L6-v2')
    texts = [request.text1, request.text2]
    embeddings = model.encode(texts)
    # Compute cosine similarity
    if embeddings.shape[0] < 2:
        raise HTTPException(status_code=400, detail="Both texts must be non-empty.")
    cos_sim = float(np.dot(embeddings[0], embeddings[1]) / (np.linalg.norm(embeddings[0]) * np.linalg.norm(embeddings[1])))
    return {"cosine_similarity": cos_sim}

@app.post("/readability")
def readability_endpoint(request: ReadabilityRequest) -> Dict[str, float]:
    text = request.text
    if not text or not text.strip():
        raise HTTPException(status_code=400, detail="Text must not be empty.")
    return {
        "flesch_reading_ease": flesch_reading_ease(text),
        "flesch_kincaid_grade": flesch_kincaid_grade(text),
        "smog_index": smog_index(text),
        "coleman_liau_index": coleman_liau_index(text),
        "automated_readability_index": automated_readability_index(text)
    }

@app.post("/extract")
def extract_keywords_entities(request: KeywordEntityRequest):
    text = request.text
    if not text or not text.strip():
        raise HTTPException(status_code=400, detail="Text must not be empty.")
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(text)
    # Simple keyword extraction: noun chunks
    keywords = list(set(chunk.text.strip() for chunk in doc.noun_chunks if chunk.text.strip()))
    # Named entities
    entities = [{"text": ent.text, "label": ent.label_} for ent in doc.ents]
    return {"keywords": keywords, "entities": entities}
