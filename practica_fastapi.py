from fastapi import FastAPI, Query
from transformers import pipeline

app = FastAPI()

sentiment_pipeline = pipeline("sentiment-analysis")
generation_pipeline = pipeline("text-generation", model="gpt2")

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/info")
def info():
    return {"message": "API con 5 endpoints y 2 pipelines HF"}

@app.get("/sum")
def sum_numbers(a: float, b: float):
    return {"a": a, "b": b, "sum": a + b}

@app.get("/sentiment")
def sentiment(text: str = Query(...)):
    result = sentiment_pipeline(text)
    return {"input": text, "result": result[0]}

@app.get("/generate")
def generate(prompt: str = Query(...), max_length: int = 50):
    output = generation_pipeline(prompt, max_length=max_length)[0]["generated_text"]
    return {"prompt": prompt, "generated_text": output}
