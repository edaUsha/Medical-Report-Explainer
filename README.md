# Medical Report Explainer

A FastAPI-based medical report question-answering app that lets users upload a PDF medical report, indexes it into Pinecone with HuggingFace embeddings, and answers natural-language questions with a Groq-hosted LLM through a retrieval-augmented generation (RAG) pipeline.[1]

## Overview

This project combines PDF ingestion, document chunking, vector search, and LLM-based answer generation into a simple web app for explaining medical reports.[1] The repository contains a FastAPI backend, a single-page HTML frontend, a CLI entry point, and separate ingestion/query modules inside `src/`.[1]

## Features

- Upload a PDF report from the browser and index it for question answering.
- Split report content into chunks using `RecursiveCharacterTextSplitter`.
- Generate embeddings with `sentence-transformers/all-MiniLM-L6-v2` through `HuggingFaceEmbeddings`.
- Store and retrieve vectors from Pinecone.
- Generate answers with Groq's `llama-3.3-70b-versatile` model.
- Serve a lightweight frontend directly from FastAPI static files.
- Support both API usage and a local CLI workflow.

## Repository Structure

```text
Medical-Report-Explainer/
├── app.py                  # FastAPI app and API routes
├── main.py                 # CLI entry point for ingest/query/chat
├── langchain_trials.py     # Small prompt-chain experiment with ChatGroq
├── requirements.txt        # Python dependencies
├── static/
│   └── index.html          # Frontend UI
└── src/
    ├── ingest.py           # PDF loading, chunking, embeddings, Pinecone upsert
    └── query.py            # Vectorstore loading, LLM setup, QA chain, ask logic
```

## How It Works

1. The `/upload` endpoint accepts a PDF and saves it into a local `data/` folder.
2. `src.ingest.load_pdf()` loads the document with `PyPDFLoader`.
3. `chunk_documents()` splits the pages into overlapping chunks for retrieval.
4. `create_embeddings()` loads the HuggingFace MiniLM embedding model.
5. `upsert_to_pinecone()` clears or creates the Pinecone index and writes the chunks.
6. `src.query.build_qa_chain()` creates a retrieval QA chain using the loaded vector store and Groq LLM.
7. The `/ask` endpoint sends a user question to the chain and returns the generated answer.

## Tech Stack

- Backend: FastAPI, Uvicorn
- LLM orchestration: LangChain
- PDF parsing: PyPDF / `PyPDFLoader`
- Embeddings: HuggingFace sentence-transformers
- Vector database: Pinecone
- LLM inference: Groq (`llama-3.3-70b-versatile`)
- Frontend: HTML, CSS, vanilla JavaScript

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/edaUsha/Medical-Report-Explainer.git
cd Medical-Report-Explainer
```

### 2. Create and activate a virtual environment

```bash
python -m venv venv
```

- Windows:

```bash
venv\Scripts\activate
```

- macOS/Linux:

```bash
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Create a `.env` file

Add the following environment variables:

```env
GROQ_API_KEY=your_groq_api_key
PINECONE_API_KEY=your_pinecone_api_key
PINECONE_INDEX_NAME=medical-rag
```

## Run the App

Start the FastAPI server with Uvicorn:

```bash
uvicorn app:app --reload
```

Then open your browser at:

```text
http://127.0.0.1:8000/
```

## API Endpoints

### `GET /`
Serves the frontend from `static/index.html`.

### `POST /upload`
Uploads a PDF, runs ingestion, creates embeddings, and initializes the QA chain.

### `POST /ask`
Accepts a JSON payload and returns an answer.

Example request:

```json
{
  "question": "What does the report say about cholesterol levels?"
}
```

## CLI Usage

The repository also includes a CLI workflow in `main.py`.

### Ingest a PDF

```bash
python main.py ingest "data/sample medical rag report.pdf"
```

### Ask one question

```bash
python main.py query "Summarize the diagnosis"
```

### Start interactive chat

```bash
python main.py chat
```

## Frontend

The frontend is a simple single-page interface that allows users to:

- Upload a PDF medical report.
- Ask follow-up questions after indexing completes.
- View answers returned by the backend.

The HTML file is located at `static/index.html` and is served directly by FastAPI.

## Notes and Current Limitations

- `app.py` currently defines the `/upload` route twice; keeping only one version would avoid ambiguity.
- `main.py` accepts a PDF path for `run_ingest`, but the function currently loads a hardcoded sample file instead of the provided argument.
- The project clears all vectors in the configured Pinecone index before or during fresh ingestion, so the same index should not be shared across unrelated datasets.
- This tool is for report explanation and educational assistance; it should not be treated as medical diagnosis or professional clinical advice.

## Possible Improvements

- Add source citations or retrieved context snippets in answers.
- Add file size validation and better error handling for uploads.
- Add support for multiple reports and namespaces in Pinecone.
- Add tests for ingestion, retrieval, and API routes.
- Replace the static frontend with Streamlit or a richer UI if needed.
- Add Docker support for easier deployment.

## License

This repository is licensed under the Apache 2.0 License. See the `LICENSE` file for details.[1]
