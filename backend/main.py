from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import tiktoken
from transformers import AutoTokenizer

app = FastAPI()

class TokenizeRequest(BaseModel):
    text: str
    model: str
    provider: str = "openai"  # 'openai' or 'hf'

class CosineSimilarityRequest(BaseModel):
    text1: str
    text2: str

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
    texts = [request.text1, request.text2]
    vectorizer = CountVectorizer().fit_transform(texts)
    vectors = vectorizer.toarray()
    if vectors.shape[0] < 2:
        raise HTTPException(status_code=400, detail="Both texts must be non-empty.")
    cos_sim = cosine_similarity([vectors[0]], [vectors[1]])[0][0]
    return {"cosine_similarity": float(cos_sim)}
