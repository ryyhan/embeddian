from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import tiktoken
from transformers import AutoTokenizer
import requests

app = FastAPI()

class TokenizeRequest(BaseModel):
    text: str
    model: str
    provider: str = "openai"  # 'openai' or 'hf'

class CosineSimilarityRequest(BaseModel):
    text1: str
    text2: str

class SummarizeRequest(BaseModel):
    text: str
    max_length: int = 150

class ParaphraseRequest(BaseModel):
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
    texts = [request.text1, request.text2]
    vectorizer = CountVectorizer().fit_transform(texts)
    vectors = vectorizer.toarray()
    if vectors.shape[0] < 2:
        raise HTTPException(status_code=400, detail="Both texts must be non-empty.")
    cos_sim = cosine_similarity([vectors[0]], [vectors[1]])[0][0]
    return {"cosine_similarity": float(cos_sim)}

@app.post("/summarize")
def summarize_text(request: SummarizeRequest) -> Dict[str, str]:
    OPENROUTER_API_KEY = "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"
    OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
    
    try:
        response = requests.post(
            OPENROUTER_URL,
            headers={
                "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                "Content-Type": "application/json"
            },
            json={
                "model": "deepseek/deepseek-r1:free",
                "messages": [
                    {
                        "role": "system",
                        "content": f"You are a helpful assistant that summarizes text. Provide a concise summary in {request.max_length} words or less."
                    },
                    {
                        "role": "user",
                        "content": f"Please summarize the following text:\n\n{request.text}"
                    }
                ],
                "max_tokens": 500,
                "temperature": 0.3
            },
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            summary = result["choices"][0]["message"]["content"]
            return {"summary": summary}
        else:
            raise HTTPException(status_code=response.status_code, detail=f"OpenRouter API error: {response.text}")
            
    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=500, detail=f"Error calling OpenRouter API: {str(e)}")


@app.post("/paraphrase")
def paraphrase_text(request: ParaphraseRequest) -> Dict[str, str]:
    OPENROUTER_API_KEY = "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"
    OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
    
    try:
        response = requests.post(
            OPENROUTER_URL,
            headers={
                "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                "Content-Type": "application/json"
            },
            json={
                "model": "deepseek/deepseek-r1:free",
                "messages": [
                    {
                        "role": "system",
                        "content": "You are a helpful assistant that paraphrases text. Rephrase the given text in a different way while retaining its original meaning."
                    },
                    {
                        "role": "user",
                        "content": f"Please paraphrase the following text:\n\n{request.text}"
                    }
                ],
                "max_tokens": 500,
                "temperature": 0.7
            },
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            paraphrased_text = result["choices"][0]["message"]["content"]
            return {"paraphrased_text": paraphrased_text}
        else:
            raise HTTPException(status_code=response.status_code, detail=f"OpenRouter API error: {response.text}")
            
    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=500, detail=f"Error calling OpenRouter API: {str(e)}")
