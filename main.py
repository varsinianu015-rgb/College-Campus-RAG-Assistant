from pathlib import Path
from uuid import uuid4

from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel, Field

from app.config import settings
from app.rag_service import rag_service

app = FastAPI(
    title="AI College Campus Assistant - RAG Document Assistant",
    version="1.0.0",
    description="Upload college PDFs, index them in ChromaDB, and ask questions with source-page citations.",
)

UPLOAD_DIR = Path(settings.upload_directory)
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=2000)


class SourceItem(BaseModel):
    source_document: str
    page_number: int
    relevance_score: float


class AskResponse(BaseModel):
    question: str
    answer: str
    found: bool
    sources: list[SourceItem]


@app.get("/")
async def root():
    return {"message": "RAG Document Assistant is running", "docs": "/docs"}


@app.get("/health")
async def health():
    return {"status": "ok", "service": "college-rag-assistant"}


@app.post("/documents/upload")
async def upload_documents(files: list[UploadFile] = File(...)):
    results = []
    for file in files:
        filename = Path(file.filename or "").name
        if not filename or Path(filename).suffix.lower() != ".pdf":
            raise HTTPException(status_code=400, detail="Only PDF files are accepted.")

        content = await file.read()
        if len(content) > settings.max_upload_mb * 1024 * 1024:
            raise HTTPException(status_code=413, detail=f"{filename} exceeds upload limit.")
        if not content.startswith(b"%PDF"):
            raise HTTPException(status_code=400, detail=f"{filename} is not a valid PDF.")

        stored_path = UPLOAD_DIR / f"{uuid4().hex}_{filename}"
        stored_path.write_bytes(content)
        try:
            results.append(rag_service.ingest_pdf(str(stored_path), filename))
        except Exception as exc:
            stored_path.unlink(missing_ok=True)
            raise HTTPException(status_code=422, detail=f"Could not process {filename}: {exc}") from exc

    return {"message": "PDF upload and indexing completed.", "uploaded_count": len(results), "documents": results}


@app.post("/ask", response_model=AskResponse)
async def ask_question(request: AskRequest):
    try:
        result = rag_service.ask(request.question)
        return AskResponse(
            question=request.question,
            answer=result["answer"],
            found=result["found"],
            sources=result["sources"],
        )
    except Exception as exc:
        raise HTTPException(
            status_code=503,
            detail=f"OpenAI RAG request failed: {type(exc).__name__}: {exc}"
        ) from exc
