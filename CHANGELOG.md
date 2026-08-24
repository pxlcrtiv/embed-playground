# Changelog

All notable changes to embed-playground are documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

## [0.1.0] — 2026-08-24

### Added
- Initial release: `embed-playground` CLI with three retrieval tiers —
  deterministic TF-IDF lexical search (stdlib-only), optional dense search
  (sentence-transformers `all-MiniLM-L6-v2`, keyless, CPU), and RRF hybrid
  fusion with a lexical-only fallback.
- `search` (rank the bundled corpus), `compare` (side-by-side tiers),
  `eval` (16-query retrieval benchmark: recall@5 + MRR, text/CSV/JSON),
  `corpus` (stats), and `app` (Streamlit playground, optional).
- 36 original docs across DeFi, NFTs, security, wallets, rollups, and MEV
  (`data/corpus.json`), plus a 16-query benchmark with definitional,
  paraphrase, and cross-context intents (`data/bench/queries.json`).
- 27-offline-test suite with golden rankings and metrics; Daily Green
  automation (23-retrieval-tip pool); CI + daily workflows.