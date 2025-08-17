# import os
# import pathlib
# from typing import List

# from dotenv import load_dotenv

# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain_community.document_loaders import PyMuPDFLoader
# from langchain_huggingface import HuggingFaceEndpointEmbeddings, HuggingFaceEmbeddings
# from langchain_community.vectorstores import Chroma

# from langchain_huggingface import HuggingFaceEndpoint

# from langchain.prompts import PromptTemplate
# from langchain.chains import RetrievalQA


# load_dotenv()

# HF_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")
# if not HF_TOKEN:
#     raise RuntimeError("HUGGINGFACEHUB_API_TOKEN is not set in the environment variables.")

# EMBED_MODEL = os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
# LLM_MODEL = os.getenv("LLM_MODEL", "mistralai/Mistral-7B-Instruct-v0.2")

# PERSIST_DIR = "chroma_db"

# def ensure_pdf_exists(pdf_path: str):
#     if not pathlib.Path(pdf_path).exists():
#         raise FileNotFoundError(f"The PDF file {pdf_path} does not exist. Please provide a valid path.")
    
# def load_and_split(pdf_path: str, chunk_size: int = 1000, chunk_overlap: int = 150):
#     loader = PyMuPDFLoader(pdf_path)
#     docs = loader.load()
    
#     splitter = RecursiveCharacterTextSplitter(
#         chunk_size=chunk_size,
#         chunk_overlap=chunk_overlap,
#         separators=["\n\n", "\n", " ", ""],
#     )
#     return splitter.split_documents(docs)

# # def build_embeddings():
# #     return HuggingFaceEndpointEmbeddings(
# #         model=EMBED_MODEL,   
# #     )

# def build_embeddings():
#     return HuggingFaceEmbeddings(
#         model_name=EMBED_MODEL   
#     )
    
# def build_llm():
#     return HuggingFaceEndpoint(
#         repo_id=LLM_MODEL,
#         temperature=0.2,
#         max_new_tokens=384,
#         top_p=0.95,
#         huggingfacehub_api_token=HF_TOKEN,
#     )
    
# def build_or_load_vector_store(chunks, embeddings, persist_dir=PERSIST_DIR):
#     if chunks is None:
#         return Chroma(
#             persist_directory=persist_dir,
#             embedding_function=embeddings,
#         )
    
#     vectordb = Chroma.from_documents(
#         documents=chunks,
#         embedding=embeddings,
#         persist_directory=persist_dir,
#     )
#     vectordb.persist()
#     return vectordb

# def make_qa_chain(vectordb, llm, k:int = 4):
#     # RetrievalQA chain with a custom, concise prompt that also returns sources.
#     retriever = vectordb.as_retriever(search_kwargs={"k": k})
    
#     template = (
#         "You are a helpful assistant. Use ONLY the provided context to answer.\n"
#         "If the answer is not in the context, say you don't know.\n\n"
#         "Question:\n{question}\n\n"
#         "Context:\n{context}\n\n"
#         "Answer concisely and cite sources with [S#] markers."
#     )
    
#     prompt = PromptTemplate(
#         input_variables=["question", "context"],
#         template=template,
#     )
    
#     chain = RetrievalQA.from_chain_type(
#         llm=llm,
#         chain_type="stuff",
#         retriever=retriever,
#         return_source_documents=True,
#         chain_type_kwargs={"prompt": prompt},
#     )
#     return chain

# def pretty_sources(source_docs: List):
#     lines = []
#     for i, d in enumerate(source_docs, start=1):
#         meta = d.metadata or {}
#         source = meta.get("source", "Unknown source")
#         page = meta.get("page", None)
#         if page is not None:
#             lines.append(f"[S{i}] {pathlib.Path(source).name} (page {page + 1})")
#         else:
#             lines.append(f"[S{i}] {pathlib.Path(source).name}")
#     return "\n".join(lines)

# def ingest(pdf_path:str):
#     ensure_pdf_exists(pdf_path)
    
#     print(f"Loading & splitting: {pdf_path}")
#     chunks = load_and_split(pdf_path)
    
#     print(f"Total chunks: {len(chunks)}")
#     embeddings = build_embeddings()
#     vector_db = build_or_load_vector_store(chunks, embeddings, persist_dir=PERSIST_DIR)
#     print(f"Vector store persisted to: {PERSIST_DIR}")
    
# def interactive_qa():
#     embeddings = build_embeddings()
#     llm = build_llm()
    
#     vectordb = build_or_load_vector_store(None, embeddings, persist_dir=PERSIST_DIR)
#     qa = make_qa_chain(vectordb, llm, k=4)
    
#     print("\nRAG ready. Ask questions about your PDFs. Type `exit` to quit.")
#     while True:
#         q = input("\n> ")
#         if q.strip().lower() in {"exit", "quit", "q"}:
#             break
        
#         result= qa({"query": q})
#         answer = result["result"]
#         sources = result.get("source_documents", [])
        
#         print("\n" + answer.strip())
#         if sources:
#             print("\nSources:")
#             print(pretty_sources(sources))
            
# if __name__ == "__main__":
#     import argparse
    
#     parser = argparse.ArgumentParser(description="RAG PDF Q&A (HF Inference API + Chroma)")
#     parser.add_argument(
#         "--pdf",
#         type=str,
#         help="Path to a PDF to ingest (one-time).",
#     )
#     args = parser.parse_args()
    
#     if args.pdf:
#         ingest(args.pdf)
        
#     interactive_qa()

import os
import argparse
import fitz  # PyMuPDF
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEndpointEmbeddings, HuggingFaceEndpoint
from langchain.docstore.document import Document
from langchain.chains import RetrievalQA

from dotenv import load_dotenv
load_dotenv()


# Load Hugging Face token from env
HF_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")
if not HF_TOKEN:
    raise ValueError("⚠️ Please set HUGGINGFACEHUB_API_TOKEN in your environment!")

# Models
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL = "HuggingFaceH4/zephyr-7b-beta"

# ===============================
# PDF Loader
# ===============================
def load_pdf(pdf_path):
    print(f"Loading & splitting: {pdf_path}")
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text("text")
    doc.close()
    return text

# ===============================
# Text Splitter
# ===============================
def split_text(text):
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_text(text)
    print(f"Total chunks: {len(chunks)}")
    return [Document(page_content=chunk) for chunk in chunks]

# ===============================
# Build Embeddings
# ===============================
def build_embeddings():
    return HuggingFaceEndpointEmbeddings(
        model=EMBED_MODEL,
        huggingfacehub_api_token=HF_TOKEN,
    )

# ===============================
# Build LLM
# ===============================
def build_llm():
    return HuggingFaceEndpoint(
        repo_id=LLM_MODEL,
        huggingfacehub_api_token=HF_TOKEN,
    )

# ===============================
# Ingest PDF into Vector DB
# ===============================
def ingest(pdf_path):
    text = load_pdf(pdf_path)
    docs = split_text(text)
    embeddings = build_embeddings()
    vectordb = Chroma.from_documents(docs, embedding=embeddings, persist_directory="./chroma_db")
    vectordb.persist()
    print("✅ Ingestion complete & DB saved at ./chroma_db")

# ===============================
# Query PDF
# ===============================
def query_pdf(question):
    embeddings = build_embeddings()
    vectordb = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)
    retriever = vectordb.as_retriever()

    llm = build_llm()
    qa = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

    print(f"\n❓ Q: {question}")
    answer = qa.run(question)
    print(f"💡 A: {answer}")

# ===============================
# Main CLI
# ===============================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdf", type=str, help="Path to PDF file")
    parser.add_argument("--query", type=str, help="Ask a question from the ingested PDF")
    args = parser.parse_args()

    if args.pdf:
        ingest(args.pdf)
    elif args.query:
        query_pdf(args.query)
    else:
        print("⚠️ Use --pdf <file.pdf> to ingest OR --query 'your question' to ask.")
