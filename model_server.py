# model_server.py
import os
from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
import uvicorn

LLM_BACKEND = os.getenv("LLM_BACKEND", "llama_cpp")
LLM_MODEL_PATH = os.getenv("LLM_MODEL_PATH", "./models/llama-2-7b-chat.Q4_K_M.gguf")
EMBED_MODEL = os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

print("Loading embedding model:", EMBED_MODEL)
embed_model = SentenceTransformer(EMBED_MODEL)

llm = None
if LLM_BACKEND == "llama_cpp":
    from llama_cpp import Llama
    print("Loading llama.cpp model:", LLM_MODEL_PATH)
    llm = Llama(
        model_path=LLM_MODEL_PATH,
        n_ctx=1024,
        n_threads=os.cpu_count(),
        n_batch=512,
        logits_all=False,
        n_gpu_layers=-1 
    )

app = FastAPI()

class TextReq(BaseModel):
    text: str
    max_tokens: int = 128
    temperature: float = 0.2

@app.post("/embed")
def embed(req: TextReq):
    vec = embed_model.encode(req.text).tolist()
    return {"embedding": vec}

@app.post("/generate")
def generate(req: TextReq):
    if llm is None:
        return {"error": "No LLM available"}
    out = llm(
        req.text,
        max_tokens=req.max_tokens,
        temperature=req.temperature,
        stop=["User:", "You:"]
    )
    text = out["choices"][0]["text"]
    return {"text": text.strip()}

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8001)
