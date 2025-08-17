import os
import fitz  # PyMuPDF
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from transformers import pipeline

# Load PDF 
pdf_path = "./data/The_Metamorphosis.pdf"
loader = PyMuPDFLoader(pdf_path)
documents = loader.load()

# Split into chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
docs = text_splitter.split_documents(documents)

embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectordb = Chroma.from_documents(docs, embedding, persist_directory="./chroma_db")

# Local LLM (FLAN-T5)
llm_pipeline = pipeline(
    "text2text-generation",
    model="google/flan-t5-base",
    tokenizer="google/flan-t5-base",
    max_new_tokens=256,
    device=-1  # use -1 for CPU, 0 for GPU if available
)

def answer_question(query: str):
    retriever = vectordb.as_retriever(search_kwargs={"k": 3})
    docs = retriever.get_relevant_documents(query)
    context = "\n\n".join([d.page_content for d in docs])

    prompt = f"Answer the question using the context below:\n\nContext:\n{context}\n\nQuestion: {query}\nAnswer:"
    result = llm_pipeline(prompt)[0]['generated_text']
    return result

if __name__ == "__main__":
    print("✅ RAG PDF Q&A Ready! Type 'exit' to quit.\n")
    while True:
        query = input("Ask a question: ")
        if query.lower() == "exit":
            break
        answer = answer_question(query)
        print(f"\n🤖 Answer: {answer}\n")
