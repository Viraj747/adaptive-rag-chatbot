from io import BytesIO

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from pydantic import BaseModel, Field
from pypdf import PdfReader

from src.rag import answer_query, index_document

app = FastAPI(
    title="Adaptive RAG Chatbot API",
    description="Portfolio MVP: adaptive routing, document RAG, and optional web search.",
    version="0.1.0",
)


class QueryRequest(BaseModel):
    query: str = Field(min_length=1, max_length=8000)
    session_id: str = Field(min_length=1, max_length=200)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/rag/query")
def query_rag(request: QueryRequest):
    try:
        return {"result": answer_query(request.query, request.session_id)}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/rag/documents/upload")
async def upload_document(file: UploadFile = File(...), description: str = Form(...)):
    filename = file.filename or "uploaded_file"
    suffix = filename.lower().rsplit(".", 1)[-1] if "." in filename else ""
    if suffix not in {"pdf", "txt"}:
        raise HTTPException(status_code=400, detail="Only PDF and TXT files are supported.")
    raw = await file.read()
    if len(raw) > 15 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="Maximum upload size is 15 MB.")
    try:
        if suffix == "pdf":
            reader = PdfReader(BytesIO(raw))
            text = "\n".join(page.extract_text() or "" for page in reader.pages)
        else:
            text = raw.decode("utf-8", errors="replace")
        count = index_document(text, filename, description)
        return {"status": True, "filename": filename, "chunks_indexed": count}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
