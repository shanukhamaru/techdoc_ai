import os
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

VECTORSTORE_PATH = "data/faiss_store"

# Initialize Hugging Face embeddings model
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

def save_vectorstore(chunks):
    """
    Convert document chunks into embeddings and save in FAISS.
    """
    vectordb = FAISS.from_documents(chunks, embeddings)
    vectordb.save_local(VECTORSTORE_PATH)
    print(f"[INFO] Vector store saved successfully at: {VECTORSTORE_PATH}")

def load_vectorstore():
    """
    Load the FAISS index from disk and return it.
    """
    if not os.path.exists(VECTORSTORE_PATH):
        raise FileNotFoundError("Vector store not found. Run save_vectorstore first.")
    # We created this index ourselves locally, so it’s safe to allow pickle deserialization
    vectordb = FAISS.load_local(
        VECTORSTORE_PATH,
        embeddings,
        allow_dangerous_deserialization=True
    )
    print(f"[INFO] Vector store loaded successfully from: {VECTORSTORE_PATH}")
    return vectordb
