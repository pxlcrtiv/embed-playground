# embed-playground

[![CI](https://img.shields.io/github/actions/workflow/status/pxlcrtiv/embed-playground/ci.yml?branch=main&label=CI)](https://github.com/pxlcrtiv/embed-playground/actions)
[![License](https://img.shields.io/github/license/pxlcrtiv/embed-playground)](LICENSE)
[![Stars](https://img.shields.io/github/stars/pxlcrtiv/embed-playground)](https://github.com/pxlcrtiv/embed-playground/stargazers)
[![Forks](https://img.shields.io/github/forks/pxlcrtiv/embed-playground)](https://github.com/pxlcrtiv/embed-playground/forks)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)
[![Ruff](https://img.shields.io/badge/code%20style-ruff-000000)](https://github.com/astral-sh/ruff)

**Compare lexical and semantic search on a bundled DeFi/Web3 corpus —
on-device, keyless, and honest.** Deterministic TF-IDF retrieval runs
anywhere Python runs; an optional sentence-transformers dense tier
(`all-MiniLM-L6-v2`) and an RRF hybrid show exactly what embeddings buy
you — and what they don't. Ships a 16-query retrieval benchmark
(recall@5 + MRR) and a Streamlit playground.

## Problem

- Embedding demos are impressive until you ask *"but is it better than
  TF-IDF?"* — nobody runs the comparison, because nobody ships the baseline
  with the demo.
- Semantics are easy to oversell: on keyword-shaped queries, lexical
  retrieval wins; on paraphrase queries, dense wins. Tools that only show
  the flattering tier teach the wrong lesson.
- Benchmarks full of scraped text are pollution: retrieval corpora must be
  original and leakage-free to mean anything.

## Solution

`embed-playground` ships three retrieval tiers over one bundled corpus:

| Tier | What it is | Requires |
| --- | --- | --- |
| `lexical` | L2-normalized TF-IDF + cosine (pure stdlib, deterministic) | nothing |
| `dense` | sentence-transformers embeddings, CPU, keyless | `pip install "embed-playground[dense]"` |
| `hybrid` | Reciprocal Rank Fusion of lexical + dense | dense optional (lexical-only fallback) |

The bundled assets are **original**: 36 docs across DeFi, NFTs, security,
wallets, rollups, and MEV, plus 16 benchmark queries labeled by intent —
`definitional` (keywords match), `paraphrase` (meaning, not words), and
`cross-context` (two concepts at once).

## Quickstart

```bash
git clone https://github.com/pxlcrtiv/embed-playground
cd embed-playground
python3 -m venv .venv && source .venv/bin/activate
pip install -e .
```

```bash
embed-playground search "flash loan mechanics"
embed-playground compare "prove a computation without revealing the inputs"
embed-playground eval --tier lexical --detail
embed-playground app          # Streamlit playground (extra: pip install .[app])
```

## Demo (live transcript, 2026-08-24)

Lexical versus dense on a paraphrase query — "prove a computation without
revealing the inputs" (the zk-rollup doc, d27):

```text
$ embed-playground compare "prove a computation without revealing the inputs"
query: 'prove a computation without revealing the inputs'

  [lexical]
    1. d25    0.1648   layer2     Rollups explained
    2. d18    0.0850   security   Invariants and audits
    3. d04    0.0557   defi       Stablecoins
    4. d16    0.0526   security   Upgradeable contracts and timelocks
    5. d02    0.0483   defi       Flash loans

  [dense]
    1. d27    0.2654   layer2     Zero-knowledge rollups
    2. d18    0.2036   security   Invariants and audits
    3. d15    0.1942   security   Oracle manipulation
    4. d14    0.1867   security   Access control
    5. d24    0.1762   wallets    Allowance hygiene

  [hybrid]
    1. d18    0.0323   security   Invariants and audits
    2. d05    0.0299   defi       Governance tokens and DAOs
    3. d16    0.0297   security   Upgradeable contracts and timelocks
    4. d15    0.0287   security   Oracle manipulation
    5. d03    0.0286   defi       Yield vaults
```

Lexical never sees `prove` → `proof` or `revealing` → `validity`; the dense
embedding does, and puts d27 on top. That is the whole lesson in one table.

Search, plain:

```text
$ embed-playground search "flash loan mechanics"
query: 'flash loan mechanics'
tier: lexical  top-k: 5

rank doc   score     category  title
1    d02   0.3360    defi      Flash loans
2    d15   0.0907    security  Oracle manipulation
3    d01   0.0000    defi      Automated market makers
4    d03   0.0000    defi      Yield vaults
5    d04   0.0000    defi      Stablecoins
```

## Benchmark results (real runs, keyless, CPU-only)

```text
$ embed-playground eval --tier lexical
tier: lexical
recall@5: 75.0% (12/16)
mrr:      0.625

$ embed-playground eval --tier dense
tier: dense
recall@5: 93.8% (15/16)
mrr:      0.716

$ embed-playground eval --tier hybrid
tier: hybrid
recall@5: 81.2% (13/16)
mrr:      0.630
```

Per intent, the pattern is exactly what retrieval theory predicts:

| Intent | Lexical | Dense | What happened |
| --- | --- | --- | --- |
| definitional (6) | 5/6 | 6/6 | both tiers nail keyword queries |
| paraphrase (7) | 5/7 | 6/7 | dense recovers meaning without shared words |
| cross-context (3) | 3/3 | 3/3 | both handle multi-concept queries here |

Dense wins recall; the hybrid (RRF, k=60) smooths rankings but cannot
exceed its best tier's recall here — an honest, tunable trade-off, not a
marketing line.

## Commands

```bash
# search with any tier
embed-playground search "sandwich attack" --tier lexical --top-k 10
embed-playground search "how do bots steal money from my trades" --tier dense --snippet

# side-by-side tiers
embed-playground compare "pay transaction fees using a credit card stablecoin"

# retrieval benchmark (text / CSV / JSON)
embed-playground eval --tier lexical --detail
embed-playground eval --tier dense --format json

# corpus stats
embed-playground corpus

# Streamlit playground
pip install "embed-playground[app]"
embed-playground app --port 8501
```

## How it works

1. **Lexical** — smooth IDF (`ln((1+N)/(1+df)) + 1`), sublinear query
   term weighting, L2-normalized vectors, cosine similarity. Pure stdlib;
   bit-identical across machines and runs.
2. **Dense** — optional `sentence-transformers/all-MiniLM-L6-v2`
   (keyless public download, cached locally; ~90 MB once, CPU-fast).
   Graceful error if the extra isn't installed.
3. **Hybrid** — Reciprocal Rank Fusion (`score = Σ 1/(k + rank)`, k=60)
   over the two tier rankings; a single-list fusion is its own ranking,
   which is the deterministic fallback when dense is unavailable.
4. **Benchmark** — recall@5 (a relevant doc in the top 5) and MRR
   (1/rank of the first relevant hit) over 16 labeled queries; goldens are
   locked in the test suite so ranking drift fails CI, not users.

## Testing

```bash
pip install -e ".[dev]"
pytest tests/ -q        # 27 offline, deterministic tests (0.05s)
ruff check embed_playground tests scripts
```

## Related portfolio repos

Built alongside my AI/ML × blockchain tooling: [**pocket-eval**](
https://github.com/pxlcrtiv/pocket-eval) (keyless CPU LLM eval harness),
[**chain-chat**](https://github.com/pxlcrtiv/chain-chat) (ask on-chain
history in plain English), [**inject-scout**](
https://github.com/pxlcrtiv/inject-scout) (prompt-injection scanner), and
[**model-ledger**](https://github.com/pxlcrtiv/model-ledger) (on-chain
model provenance). See the full portfolio on my
[profile](https://github.com/pxlcrtiv).

## Daily Green automation

This repo participates in the portfolio-wide daily-commit automation
(launchd on macOS 12:07 + 18:07 local, GitHub Actions
[`daily.yml`](.github/workflows/daily.yml) 12:00 UTC as cloud fallback).
Every day `scripts/daily_update.py` appends one curated retrieval/embeddings
tip from `scripts/tips_pool.json` (23 entries) to `docs/daily-tips.md` and
pushes a dated, non-empty commit — idempotent, backfills missed days
(max 14), and never duplicates.

- Customize content: edit `scripts/tips_pool.json`.
- Pause this repo: `touch .daily-pause`.
- Pause globally: `launchctl bootout gui/$(id -u)/com.pxlcrtiv.daily-green`.

## License

MIT — see [LICENSE](LICENSE).