from fastapi.testclient import TestClient

from citewise import api
from citewise.answer import NOT_FOUND
from citewise.index import Index

POLICY = (
    "Hotels are reimbursed up to $175 per night in most cities. New York and San Francisco allow $300.\n\n"
    "Meals are reimbursed at actual cost. Alcohol is never reimbursed."
)
IT = "Laptops are refreshed every four years. Lost devices must be reported to the service desk within 24 hours."


def client():
    api.index = Index()
    return TestClient(api.app)


def test_ask_returns_cited_answer_with_exact_source_spans(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    c = client()
    assert c.post("/documents", json={"id": "travel", "title": "Travel Policy", "text": POLICY}).status_code == 201
    c.post("/documents", json={"id": "it", "title": "IT Equipment Policy", "text": IT})
    r = c.post("/ask", json={"question": "How often are laptops refreshed?"}).json()
    assert "four years" in r["answer"] and r["cited"]
    src = r["sources"][r["cited"][0] - 1]
    assert src["doc_id"] == "it"
    assert IT[src["start"] : src["end"]] == src["text"]


def test_title_header_finds_chunk_that_never_names_its_topic(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    idx = Index()
    idx.add("travel", "Travel Policy", POLICY)
    idx.add("it", "IT Equipment Policy", IT)
    hits = idx.search("travel policy alcohol", 1)
    assert hits[0].chunk.doc_id == "travel"


def test_unrelated_question_says_not_found(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    c = client()
    c.post("/documents", json={"id": "travel", "title": "Travel Policy", "text": POLICY})
    r = c.post("/ask", json={"question": "quarterly revenue guidance"}).json()
    assert r["answer"] == NOT_FOUND and r["sources"] == []


def test_reindexing_a_document_replaces_its_chunks():
    idx = Index()
    idx.add("travel", "Travel Policy", POLICY)
    idx.add("travel", "Travel Policy", "Hotels are reimbursed up to $200 per night.")
    assert all("$175" not in c.text for c in idx.chunks)


def test_validation_and_health():
    c = client()
    assert c.post("/ask", json={"question": ""}).status_code == 422
    assert c.get("/health").json() == {"status": "ok", "documents": 0, "chunks": 0}
