from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import pipeline
import os

app = FastAPI(title="Qwen AI API")

# Load model ONCE at startup
generator = pipeline(
    "text-generation",
    model="Qwen/Qwen2.5-0.5B",
    device=-1,
    torch_dtype="float32",
    low_cpu_mem_usage=True
)

class Prompt(BaseModel):
    prompt: str

@app.get("/")
def health():
    return {"status": "running"}

@app.post("/generate")
def generate(data: Prompt):
    if not data.prompt.strip():
        raise HTTPException(status_code=400, detail="Empty prompt")

    result = generator(
        data.prompt,
        max_new_tokens=80,
        do_sample=True,
        temperature=0.7
    )

    return {
        "response": result[0]["generated_text"]
    }
