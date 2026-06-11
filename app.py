from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from dotenv import load_dotenv
import shutil
import os

load_dotenv()

app = FastAPI(title="Medical RAG Q&A")

# serve static files (your HTML)
app.mount("/static", StaticFiles(directory="static"), name="static")

# global chain — loaded once after ingestion
qa_chain = None
vectorstore = None

# ── request models ─────────────────────────────────────────
class QuestionRequest(BaseModel):
    question: str

# ── serve frontend ─────────────────────────────────────────
@app.get("/")
def root():
    return FileResponse("static/index.html")


@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    global qa_chain, vectorstore

    # ← ADD THIS: reset old chain before building new one
    qa_chain = None
    vectorstore = None

    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files allowed")

    pdf_path = f"data/{file.filename}"
    os.makedirs("data", exist_ok=True)
    with open(pdf_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    try:
        from src.ingest import load_pdf, chunk_documents, create_embeddings, upsert_to_pinecone
        pages = load_pdf(pdf_path)
        chunks = chunk_documents(pages)
        embeddings = create_embeddings()
        vectorstore = upsert_to_pinecone(chunks, embeddings)  # clears old + upserts new

        from src.query import load_llm, build_qa_chain
        llm = load_llm()
        qa_chain = build_qa_chain(vectorstore, llm)

        return {"status": "success", "message": f"Indexed {len(chunks)} chunks from {file.filename}"}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
    
# ── upload + ingest PDF ────────────────────────────────────
@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    global qa_chain, vectorstore

    # validate file type
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files allowed")

    # save uploaded file to data/
    pdf_path = f"data/{file.filename}"
    os.makedirs("data", exist_ok=True)
    with open(pdf_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    # run ingestion pipeline
    try:
        from src.ingest import load_pdf, chunk_documents, create_embeddings, upsert_to_pinecone
        pages = load_pdf(pdf_path)
        chunks = chunk_documents(pages)
        embeddings = create_embeddings()
        vectorstore = upsert_to_pinecone(chunks, embeddings)

        # build QA chain immediately after ingestion
        from src.query import load_llm, build_qa_chain
        llm = load_llm()
        qa_chain = build_qa_chain(vectorstore, llm)

        return {"status": "success", "message": f"Indexed {len(chunks)} chunks from {file.filename}"}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ── ask a question ─────────────────────────────────────────
@app.post("/ask")
async def ask_question(body: QuestionRequest):
    global qa_chain

    if qa_chain is None:
        raise HTTPException(status_code=400, detail="Please upload a PDF first")

    if not body.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    try:
        from src.query import ask
        result = ask(qa_chain, body.question)
        return {"answer": result["result"]}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))