
from __future__ import annotations

from rank_bm25 import BM25Okapi
from .chunker import chunk_text


def tokenize(text: str) -> list[str]:
    return (
        text.lower()
        .replace("\n", " ")
        .replace("\t", " ")
        .split()
    )


class Retriever:
    def __init__(
        self,
        docs: dict[str, str],
        chunk_chars=1200,
        overlap_chars=200,
        method="bm25",
    ):
        self.docs = docs
        self.chunks = []

        for doc_id, text in docs.items():
            self.chunks.extend(
                chunk_text(doc_id, text, chunk_chars, overlap_chars)
            )

        self.method = method
        self.bm25 = (
            BM25Okapi([tokenize(c.text) for c in self.chunks])
            if self.chunks
            else None
        )

    
    def search(
        self,
        query: str,
        top_k=5,
        min_score=0.0,
    ) -> list[dict]:
        if not self.chunks or self.bm25 is None or top_k <= 0:
            return []

        query_tokens = tokenize(query)
        scores = self.bm25.get_scores(query_tokens)

        query_words = (
            query.lower()
            .replace("-", " ")
            .replace(".", " ")
            .replace(":", " ")
            .split()
        )

        requested_amendment = None
        for i in range(len(query_words) - 2):
            if (
                query_words[i] == "amendment"
                and query_words[i + 1] == "no"
                and query_words[i + 2].isdigit()
            ):
                requested_amendment = query_words[i + 2]
                break

        ranked = []

        for idx, score in enumerate(scores):
            chunk = self.chunks[idx]
            adjusted_score = float(score)

            if requested_amendment is not None:
                text_words = (
                    chunk.text.lower()
                    .replace("-", " ")
                    .replace(".", " ")
                    .replace(":", " ")
                    .split()
                )

                has_exact_amendment = any(
                    text_words[i] == "amendment"
                    and i + 2 < len(text_words)
                    and text_words[i + 1] == "no"
                    and text_words[i + 2] == requested_amendment
                    for i in range(len(text_words))
                )

                # Require some contract-specific overlap before boosting.
                context_terms = [
                    word for word in query_tokens
                    if word not in {
                        "consider", "the", "to", "between", "and",
                        "what", "is", "expiration", "date", "of",
                        "this", "contract", "no", "amendment",
                        requested_amendment,
                    }
                ]
                chunk_token_set = set(tokenize(chunk.text))
                context_matches = sum(
                    1 for word in context_terms if word in chunk_token_set
                )

                if has_exact_amendment and context_matches >= 3:
                    adjusted_score += max(
                        max((float(s) for s in scores), default=0.0) * 0.5,
                        1.0,
                    )

            ranked.append((idx, adjusted_score, float(score)))

        ranked.sort(key=lambda item: item[1], reverse=True)

        results = []
        for idx, adjusted_score, original_score in ranked:
            if len(results) >= top_k:
                break
            if min_score is not None and min_score != 0.0 and original_score < min_score:
                continue

            chunk = self.chunks[idx]
            results.append({
                "doc_id": chunk.doc_id,
                "start": chunk.start,
                "end": chunk.end,
                "text": chunk.text,
                "score": adjusted_score,
            })

        return results

