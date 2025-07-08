import os
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.llms import HuggingFacePipeline
from transformers import pipeline
from app.vectorstore import load_vectorstore

# 1. Load FAISS index + retriever
def get_retriever(k: int = 4):
    vectordb = load_vectorstore()
    return vectordb.as_retriever(search_kwargs={"k": k})

# 2. Prompt template
prompt_template = """Use the following context to answer the question concisely.

Context:
{context}

Question:
{question}
"""

# 3. Build the QA chain
def get_qa_chain():
    retriever = get_retriever()

    # — replace HFHub with a local transformers pipeline
    hf_pipe = pipeline(
        task="text2text-generation",
        model="google/flan-t5-base",
        max_length=512,
        temperature=0.3,
        device_map="auto"  # runs on GPU if available, or CPU otherwise
    )
    llm = HuggingFacePipeline(pipeline=hf_pipe)

    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"]
    )
    return RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        return_source_documents=False,
        chain_type="stuff",
        chain_type_kwargs={"prompt": prompt}
    )

# 4. Helper to answer
def answer_question(question: str) -> str:
    qa_chain = get_qa_chain()
    return qa_chain.run(question)

# 5. CLI test
if __name__ == "__main__":
    test_q = "What is the AWS Well-Architected Framework?"
    print("Q:", test_q)
    print("A:", answer_question(test_q))
