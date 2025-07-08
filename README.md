# TechDoc AI – RAG-based Document Question Answering App

TechDoc AI is a Retrieval-Augmented Generation (RAG) system built using LangChain, Hugging Face, FAISS, and FastAPI. It processes enterprise documents (PDF, DOCX, Markdown) and allows users to ask questions and receive context-aware answers.

---

## Features

- Load and process enterprise documents
- Embed and store chunks using Hugging Face + FAISS
- RAG pipeline with custom prompt templating
- FastAPI-powered REST API for querying
- Optional tracing with LangSmith
- Evaluation-ready for Promptfoo
- Dockerized setup for easy deployment

---

## Tech Stack

- Python 3.11
- LangChain
- Hugging Face Transformers
- FAISS (vector store)
- FastAPI
- Docker (optional)
- LangSmith (optional)
- Promptfoo (optional)

---

## Installation (Local)

```bash
# Clone the repository
git clone https://github.com/your-username/techdoc_ai.git
cd techdoc_ai

# Create a virtual environment
python -m venv venv
venv\Scripts\activate  # or source venv/bin/activate

# Install required dependencies
pip install -r requirements.txt

# Add your Hugging Face and LangSmith API keys in .env

techdoc_ai/
├── app/
│   ├── config.py
│   ├── doc_loader.py
│   ├── vectorstore.py
│   ├── rag_chain.py
│   └── main.py
├── data/sample_docs/
├── .env
├── requirements.txt
├── Dockerfile
├── README.md


Author
Shanu Khamaru
AI Engineer | LangChain | Hugging Face | FastAPI | MLOps
GitHub: github.com/shanukhamaru