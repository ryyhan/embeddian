from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import tiktoken

app = FastAPI()

class TokenizeRequest(BaseModel):
    text: str

class CosineSimilarityRequest(BaseModel):
    text1: str
    text2: str

@app.post("/tokenize")
def tokenize(request: TokenizeRequest) -> Dict[str, int]:
    # Use tiktoken for OpenAI GPT-3.5/4 tokenization
    encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
    tokens = encoding.encode(request.text)
    return {"token_count": len(tokens)}

@app.post("/cosine-similarity")
def cosine_similarity_endpoint(request: CosineSimilarityRequest) -> Dict[str, float]:
    texts = [request.text1, request.text2]
    vectorizer = CountVectorizer().fit_transform(texts)
    vectors = vectorizer.toarray()
    if vectors.shape[0] < 2:
        raise HTTPException(status_code=400, detail="Both texts must be non-empty.")
    cos_sim = cosine_similarity([vectors[0]], [vectors[1]])[0][0]
    return {"cosine_similarity": float(cos_sim)}
