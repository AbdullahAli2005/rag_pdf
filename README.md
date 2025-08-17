# 📘 RAG PDF Q\&A

A simple **Retrieval-Augmented Generation (RAG)** pipeline that allows you to **query answers from any PDF document** using embeddings, ChromaDB, and Hugging Face LLMs.

---

## 🚀 Features

* Load any PDF file and split it into chunks.
* Create embeddings with **`sentence-transformers/all-MiniLM-L6-v2`**.
* Store and retrieve chunks using **ChromaDB**.
* Query answers using **`google/flan-t5-base`** from Hugging Face.
* End-to-end pipeline using **LangChain**.

---

## 📦 Installation

Clone this repo and install dependencies:

```bash
# Clone repo
git clone https://github.com/your-username/rag_pdf.git
cd rag_pdf

# Create virtual environment
python -m venv .venv
.venv\Scripts\activate  # On Windows

# Install requirements
pip install -r requirements.txt
```

**requirements.txt**

```
langchain
langchain-community
chromadb
pymupdf
sentence-transformers
huggingface_hub
```

---

## 🔑 Hugging Face Setup

1. Go to [Hugging Face](https://huggingface.co/).
2. Create an account and generate an **Access Token** from: [https://huggingface.co/settings/tokens](https://huggingface.co/settings/tokens).
3. Create a `.env` file in the project root:

```
HUGGINGFACEHUB_API_TOKEN=your_hf_token_here
EMBED_MODEL=sentence-transformers/all-MiniLM-L6-v2
LLM_MODEL=google/flan-t5-base
```

---

## ▶️ Usage

Run the script with a PDF:

```bash
python rag_pdf_qa.py --pdf ./data/The_Metamorphosis.pdf
```

Then ask questions interactively:

```text
>> What happens to Gregor in the story?
Answer: Gregor Samsa wakes up one morning transformed into a giant insect...
```

---

## 📂 Project Structure

```
rag_pdf/
│-- rag_pdf_qa.py       # Main script
│-- requirements.txt    # Dependencies
│-- .env                # Hugging Face token + model config
│-- data/               # Folder to store PDFs
│-- README.md           # Documentation
```

---

## 🛠️ Tech Stack

* **LangChain** – Retrieval & pipeline orchestration
* **PyMuPDF** – PDF text extraction
* **ChromaDB** – Vector store for document chunks
* **Sentence-Transformers** – Embeddings model
* **Hugging Face** – LLM provider (Flan-T5)

---

## 📌 Example

```text
Question: Who is the main character?
Answer: The main character is Gregor Samsa.

Question: What is the theme of the story?
Answer: Alienation, isolation, and transformation.
```

---

## 📖 References

* [LangChain Documentation](https://python.langchain.com/)
* [ChromaDB](https://docs.trychroma.com/)
* [Hugging Face Hub](https://huggingface.co/)

---

✅ With this setup, you can load any PDF and start asking it questions!
