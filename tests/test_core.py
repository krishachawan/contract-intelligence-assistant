from src.chunker import chunk_text
from src.data_loader import assign_split
from src.retriever import Retriever

def test_chunk_offsets_reconstruct_text():
    text = "A paragraph.\\n\\nSecond paragraph.\\nThird."
    chunks = chunk_text("doc.txt", text, chunk_chars=20, overlap_chars=5)
    assert chunks
    for c in chunks:
        assert text[c.start:c.end] == c.text

def test_split_is_deterministic():
    assert assign_split("  Is this a contract? ") == assign_split("Is this a contract?")

def test_retriever_returns_source_and_offsets():
    docs = {"contract.txt": "The licensee may sublicense the licensed content to an approved platform."}
    engine = Retriever(docs, chunk_chars=100, overlap_chars=10)
    result = engine.search("may sublicense content", top_k=1)
    assert result
    assert result[0]["doc_id"] == "contract.txt"
    assert result[0]["start"] >= 0
