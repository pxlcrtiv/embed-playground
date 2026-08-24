"""embed-playground — Streamlit UI.

Run:  pip install "embed-playground[app]"  &&  embed-playground app
or:   streamlit run app.py

The dense tier (sentence-transformers) is optional: the lexical tier and
RRF hybrid work offline with zero dependencies.
"""

from __future__ import annotations

from functools import lru_cache

import streamlit as st

from embed_playground.backends import DenseBackend, evaluate, rank_query
from embed_playground.core import TfidfIndex
from embed_playground.report import render_snippet

st.set_page_config(page_title="embed-playground", page_icon="🔍", layout="wide")
st.title("🔍 embed-playground")
st.caption("Lexical (TF-IDF) vs dense (sentence-transformers) vs hybrid (RRF) on a bundled DeFi/Web3 corpus — nothing leaves your machine.")


@lru_cache(maxsize=1)
def get_index() -> TfidfIndex:
    return TfidfIndex(_docs())


@lru_cache(maxsize=1)
def _docs():
    from embed_playground.backends import load_corpus

    return load_corpus()


@st.cache_resource
def get_dense() -> DenseBackend | None:
    try:
        return DenseBackend()
    except Exception:  # noqa: BLE001 - graceful degradation
        return None


docs = _docs()
index = get_index()
dense = get_dense()
tiers = ["lexical", "hybrid"] + (["dense"] if dense else [])

with st.sidebar:
    st.header("Settings")
    tier = st.radio("Tier", tiers, index=0)
    top_k = st.slider("Top-k", 1, 10, 5)
    show_snippet = st.checkbox("Show snippets", value=True)
    if dense is None:
        st.warning("Dense tier unavailable — `pip install 'embed-playground[dense]'`.")
    st.divider()
    st.markdown("**Sample queries** (click to fill):")
    samples = [
        "flash loan mechanics",
        "how do bots steal money from my trades",
        "prove a computation without revealing the inputs",
        "startup treasury wallet with multiple signers",
        "why do bridge withdrawals take a week to settle",
    ]
    for s in samples:
        if st.button(s, key=s, use_container_width=True):
            st.session_state["query"] = s

query = st.text_input("Search the corpus", value=st.session_state.get("query", ""), placeholder="e.g. flash loan mechanics")

if query.strip():
    try:
        results = rank_query(tier, index, dense, query.strip(), top_k=top_k)
        st.subheader(f"Results — {tier}")
        if not results:
            st.info("No results (lexical tier found no in-vocabulary terms).")
        for doc, score in results:
            with st.container(border=True):
                col1, col2, col3 = st.columns([1, 2, 7])
                col1.markdown(f"**{doc.id}**")
                col2.markdown(f"`{doc.category}` · score {score:.4f}")
                col3.markdown(f"**{doc.title}**")
                if show_snippet:
                    st.markdown(render_snippet(doc))
        st.divider()
        with st.expander("Benchmark this tier"):
            cola, colb = st.columns(2)
            if cola.button(f"Run eval ({tier})"):
                m = evaluate(tier, index, dense)
                cola.write(f"**recall@5** {m.recall_at_5:.1%} · **MRR** {m.mrr:.3f}")
    except Exception as exc:  # noqa: BLE001 - UI boundary
        st.error(str(exc))
else:
    st.info("Type a query — try a paraphrase like 'how do bots steal money from my trades' and compare tiers.")
    st.markdown(
        "The bundled corpus: 36 original docs (DeFi, NFTs, security, wallets, rollups, MEV) + a 16-query "
        "retrieval benchmark. Lexical retrieval shines on keyword queries; dense embeddings shine on "
        "paraphrase queries; RRF hybrid usually tops both."
    )