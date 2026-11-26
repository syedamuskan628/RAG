# Document RAG Chatbot (Streamlit + LangChain + OpenRouter + ChromaDB)

A complete Retrieval-Augmented Generation (RAG) chatbot built with **Streamlit**, **LangChain** (`langchain-openrouter`, `langchain-openai`, `langchain-chroma`), **OpenRouter**, and **ChromaDB**.

## Features

- **Left Sidebar Configuration**:
  - Takes your **OpenRouter API Key** directly from the UI (`type="password"`).
  - Select from popular OpenRouter chat models (`openai/gpt-4o-mini`, `anthropic/claude-3.5-sonnet`, `google/gemini-2.5-flash`, `deepseek/deepseek-chat`, `meta-llama/llama-3.3-70b-instruct`) or enter any custom OpenRouter model slug.
  - Select OpenRouter embedding models (`openai/text-embedding-3-small`, `openai/text-embedding-3-large`, `qwen/qwen3-embedding-8b`) or enter a custom embedding model.
  - Adjust temperature, chunk size, chunk overlap, and Top-K retrieval count.
  - Inspect knowledge base stats, clear chat history, or reset ChromaDB.
- **1. Document Upload & Processing**:
  - Upload multiple `.pdf`, `.txt`, `.md`, `.docx`, or `.csv` files.
  - Automatically extracts text, splits documents into overlapping chunks (`RecursiveCharacterTextSplitter`), generates embeddings via OpenRouter, and persists vectors to `./chroma_db`.
  - Displays sample chunk previews and a table of indexed documents.
- **2. Ask Questions (RAG Chat)**:
  - History-aware query rewriting for multi-turn follow-up questions.
  - Top-K similarity search against ChromaDB.
  - Live streaming answers using LangChain's `ChatOpenRouter` and `st.write_stream`.
  - Expandable **Sources Used** citations with filenames, page numbers, distance scores, and chunk excerpts.

## Prerequisites

- **Python 3.11+** (required by `langchain-openrouter`)
- An **OpenRouter API Key** from [openrouter.ai/keys](https://openrouter.ai/keys)

## Setup & Run

```powershell
# 1. Create and activate a virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# 2. Install dependencies
pip install -r requirements.txt

# 3. Start the Streamlit app
streamlit run app.py
```

Once the app opens in your browser:
1. Paste your **OpenRouter API Key** in the left sidebar.
2. Upload your documents in **Section 1** and click **Process & Index into ChromaDB**.
3. Ask questions about your documents in **Section 2**.
