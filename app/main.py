from fastapi import FastAPI, Request
from pydantic import BaseModel
from app.rag_chain import answer_question

app = FastAPI()

# Request format
class Question(BaseModel):
    query: str

# Test route
@app.get("/")
def read_root():
    return {"message": "TechDoc AI RAG API is running 🚀"}

# /ask endpoint
@app.post("/ask")
def ask_question(question: Question):
    try:
        result = answer_question(question.query)
        return {"query": question.query, "answer": result}
    except Exception as e:
        return {"error": str(e)}
