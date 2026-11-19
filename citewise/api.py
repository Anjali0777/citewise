"""REST API.

    uvicorn citewise.api:app --reload
    POST /documents  {"id": "travel-policy", "title": "Travel Policy", "text": "..."}
    POST /ask        {"question": "What is the hotel limit?", "k": 4}
"""

from __future__ import annotations

from fastapi import FastAPI
from pydantic import BaseModel, Field

from .index import Index

app = FastAPI(title="citewise", version="0.1.0")
index = Index()


class DocIn(BaseModel):
    id: str = Field(min_length=1, max_length=200)
    title: str = Field(min_length=1, max_length=500)
    text: str = Field(min_length=1)


class AskIn(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    k: int = Field(4, ge=1, le=20)


class Source(BaseModel):
    n: int
    doc_id: str
    title: str
    start: int
    end: int
    text: str
    score: float


class AskOut(BaseModel):
    answer: str
    mode: str
    cited: list[int]
    sources: list[Source]


@app.get("/health")
def health():
    return {"status": "ok", "documents": len(index.texts), "chunks": len(index.chunks)}


@app.post("/documents", status_code=201)
def add_document(doc: DocIn):
    return {"id": doc.id, "chunks": index.add(doc.id, doc.title, doc.text)}


@app.post("/ask", response_model=AskOut)
def ask(q: AskIn):
    ans, hits = index.ask(q.question, q.k)
    sources = [
        Source(n=i, doc_id=h.chunk.doc_id, title=h.title, start=h.chunk.start, end=h.chunk.end, text=h.chunk.text, score=round(h.score, 3))
        for i, h in enumerate(hits, 1)
    ]
    return AskOut(answer=ans.text, mode=ans.mode, cited=ans.citations, sources=sources)
