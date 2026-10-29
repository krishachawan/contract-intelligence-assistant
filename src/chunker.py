from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path

@dataclass
class Chunk:
    doc_id: str
    start: int
    end: int
    text: str

def chunk_text(doc_id: str, text: str, chunk_chars: int = 1200, overlap_chars: int = 200) -> list[Chunk]:
    if chunk_chars <= 0 or overlap_chars < 0 or overlap_chars >= chunk_chars:
        raise ValueError("Require chunk_chars > overlap_chars >= 0")
    chunks = []
    start = 0
    while start < len(text):
        end = min(len(text), start + chunk_chars)
        # Prefer a nearby paragraph/line boundary, but never create an empty chunk.
        if end < len(text):
            boundary = max(text.rfind("\n\n", start, end), text.rfind("\n", start, end))
            if boundary > start + chunk_chars // 2:
                end = boundary
        if end <= start:
            end = min(len(text), start + chunk_chars)
        chunks.append(Chunk(doc_id, start, end, text[start:end]))
        if end >= len(text):
            break
        start = max(start + 1, end - overlap_chars)
    return chunks

def read_corpus(corpus_dir: str | Path) -> dict[str, str]:
    root = Path(corpus_dir)
    docs = {}

    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in {".txt", ".md"}:
            # Match the path format used by the benchmark annotations.
            doc_id = "cuad/" + p.relative_to(root).as_posix().removeprefix("cuad/")
            docs[doc_id] = p.read_text(encoding="utf-8")

    return docs
