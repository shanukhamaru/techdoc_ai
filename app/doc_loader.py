import os
import glob
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, UnstructuredMarkdownLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from app.vectorstore import save_vectorstore

def load_documents_from_folder(folder_path: str):
    loaders = []

    # Load by file type
    for file_path in glob.glob(os.path.join(folder_path, "*.pdf")):
        loaders.append(PyPDFLoader(file_path))
    for file_path in glob.glob(os.path.join(folder_path, "*.docx")):
        loaders.append(Docx2txtLoader(file_path))
    for file_path in glob.glob(os.path.join(folder_path, "*.md")):
        loaders.append(UnstructuredMarkdownLoader(file_path))

    all_docs = []
    for loader in loaders:
        docs = loader.load()
        all_docs.extend(docs)

    print(f"[INFO] Loaded {len(all_docs)} raw documents.")
    return all_docs


def chunk_documents(docs, chunk_size=500, chunk_overlap=50):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    chunks = splitter.split_documents(docs)
    print(f"[INFO] Split into {len(chunks)} chunks.")
    return chunks


if __name__ == "__main__":
    folder = "data/sample_docs"
    docs = load_documents_from_folder(folder)
    chunks = chunk_documents(docs)
    save_vectorstore(chunks)
