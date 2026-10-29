import yaml, streamlit as st
from pathlib import Path
from src.chunker import read_corpus
from src.retriever import Retriever

ROOT = Path(__file__).resolve().parent
st.set_page_config(page_title="Contract Intelligence Assistant", page_icon="📄", layout="wide")
st.title("Contract Intelligence Assistant")
st.caption("Contract passage retrieval with source paths, character offsets, and conservative abstention.")
@st.cache_resource
def load_engine():
    cfg = yaml.safe_load((ROOT / "config" / "final.yaml").read_text())["retrieval"]
    docs = read_corpus(ROOT / "data" / "raw" / "corpus")
    if not docs: return None, cfg
    return Retriever(docs, cfg["chunk_chars"], cfg["overlap_chars"], cfg["method"]), cfg
engine, cfg = load_engine()
if engine is None:
    st.warning("Dataset not found. Run `python scripts/download_data.py` first.")
else:
    query = st.text_input("Ask a question about a contract", placeholder="When does the exclusivity period end?")
    if st.button("Search", type="primary", disabled=not query.strip()):
        results = engine.search(query, cfg["top_k"], cfg["min_score"])
        if not results:
            st.info("No sufficiently relevant passage found. This is not a legal conclusion.")
        else:
            for i, r in enumerate(results, 1):
                with st.expander(f"{i}. {r['doc_id']} · score {r['score']:.3f}", expanded=(i == 1)):
                    st.caption(f"Character offsets: [{r['start']}:{r['end']}]")
                    st.code(r["text"], language="text")
                    st.caption("Retrieved passage only; verify the full contract and cross-references before relying on it.")
