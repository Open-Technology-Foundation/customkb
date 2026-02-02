# Vector DB Evaluation: CustomKB Refactor

## Overview

Evaluation of three vector database candidates to replace the current FAISS+SQLite architecture in CustomKB. Benchmarks run against the `okusiassociates2` test knowledgebase (7,894 vectors, 1024 dimensions, bge-m3 embeddings).

## Current Architecture

- **FAISS IndexIDMap(IndexIVFFlat)** — 535 IVF clusters, L2 distance metric
- **SQLite** — stores document text, metadata, BM25 tokens (separate from vector storage)
- **BM25 index** — pre-computed NPZ file for hybrid search
- **Index file**: 33 MB FAISS, 103 MB SQLite, 19 MB BM25

The FAISS index handles only similarity search. SQLite stores all text/metadata and is used for result retrieval after FAISS returns matching IDs. BM25 is independent of both.

## Candidates

| Candidate | Version | Mode | Notes |
|-----------|---------|------|-------|
| **FAISS** (baseline) | 1.13.2 (GPU) | IndexFlatL2 in benchmark | Production uses IndexIVFFlat |
| **ChromaDB** | 1.4.1 | Persistent (disk) | HNSW index, built-in metadata |
| **sqlite-vec** | 0.1.6 | SQLite extension | Exact (brute-force) search |
| **Qdrant** | 1.16.2 | Local (embedded) | HNSW index, rich filtering |

## Benchmark Results

Test parameters: 7,894 vectors at 1024 dimensions, 50 queries, top-k=20.

### Performance

| Metric | FAISS | ChromaDB | sqlite-vec | Qdrant |
|--------|------:|--------:|-----------:|-------:|
| Insert time (s) | 0.011 | 1.46 | 1.78 | 53.96 |
| Insert rate (vec/s) | 706,522 | 5,406 | 4,436 | 146 |
| Query avg (ms) | 1.24 | 1.58 | 15.15 | 47.60 |
| Query p50 (ms) | 1.20 | 1.47 | 15.13 | 47.54 |
| Query p99 (ms) | 2.16 | 3.18 | 17.07 | 50.80 |

### Storage and Memory

| Metric | FAISS | ChromaDB | sqlite-vec | Qdrant |
|--------|------:|--------:|-----------:|-------:|
| Index size (MB) | 30.9 | 54.9 | 32.2 | 72.2 |
| Memory overhead (MB) | ~0 | 412.6 | ~0 | ~0 |

### Accuracy (Recall vs FAISS Exact Search)

| Candidate | Recall@20 |
|-----------|-----------|
| FAISS (baseline) | 1.000 |
| ChromaDB | 0.630 |
| sqlite-vec | 1.000 |
| Qdrant | 1.000 |

ChromaDB's lower recall is expected — its HNSW index trades accuracy for speed. The recall can be tuned via `ef_search` parameter but at the cost of higher latency.

### Production FAISS (IVFFlat) Performance

The actual production index (IndexIVFFlat with 535 clusters) benchmarks differently from the flat index used above:

| nprobe | Query avg (ms) | Query p50 (ms) |
|--------|---------------:|---------------:|
| 1 (default) | 1.52 | 0.10 |
| 32 (configured) | 0.66 | 0.26 |

The IVF index with nprobe=32 is faster than all candidates by a wide margin.

## Analysis

### FAISS (Current)

**Strengths:**
- Fastest query performance by far (0.26ms p50 with IVFFlat)
- Fastest insert (706k vec/s)
- Smallest index size (31 MB)
- GPU acceleration available (installed but not currently used)
- Mature, battle-tested library

**Weaknesses:**
- No built-in metadata filtering (post-search filtering via SQLite)
- No incremental updates (full reindex required for IVF)
- Separate storage for vectors and metadata (FAISS + SQLite)
- No built-in persistence management

### ChromaDB

**Strengths:**
- Near-FAISS query speed (1.5ms vs 1.2ms for flat search)
- Built-in metadata storage and filtering
- Built-in persistence
- Simple API
- Active community

**Weaknesses:**
- 412 MB memory overhead — unacceptable for production servers
- 0.63 recall@20 — significant accuracy loss with default settings
- 1.8x larger index (55 MB)
- Heavy dependency tree (onnxruntime, opentelemetry, etc.)
- Would require maintaining ChromaDB server process or accepting embedded overhead

### sqlite-vec

**Strengths:**
- Perfect recall (exact brute-force search)
- Minimal index size (32 MB, close to FAISS)
- Zero memory overhead (SQLite extension)
- Could unify vector storage with existing SQLite document DB
- Minimal dependency (single extension)
- No separate process needed

**Weaknesses:**
- 12x slower queries than FAISS flat (15ms vs 1.2ms)
- ~60x slower than production IVFFlat (15ms vs 0.26ms)
- Brute-force only — no approximate search for larger KBs
- Relatively new project (v0.1.6)

### Qdrant

**Strengths:**
- Perfect recall (at default settings)
- Rich filtering and payload support
- Good scaling characteristics for larger datasets
- Mature project with enterprise features

**Weaknesses:**
- Extremely slow insert (54s for 7,894 vectors — 370x slower than FAISS)
- Extremely slow queries (48ms — 38x slower than FAISS flat)
- Largest index size (72 MB)
- Heavy dependency tree
- Local embedded mode appears poorly optimized for this scale

## Recommendation

**Keep FAISS.** None of the candidates offer a compelling reason to migrate:

1. **Performance**: FAISS is 10-180x faster at queries than all candidates. At the current scale (~8k vectors), even the IVFFlat index responds in <1ms. The 20% latency budget from the requirements would be immediately blown by any alternative.

2. **Accuracy**: ChromaDB's 0.63 recall is disqualifying for a knowledge retrieval system. sqlite-vec and Qdrant achieve perfect recall but at massive speed penalties.

3. **Architecture fit**: The current separation of concerns (FAISS for vectors, SQLite for text/metadata) works well. The "problem" of separate storage is actually a clean architecture — each component does what it's good at.

4. **Migration risk**: Replacing FAISS would require migration tooling for all production KBs, changes to the embedding pipeline, and extensive regression testing — for no measurable benefit.

5. **Dependency weight**: ChromaDB adds ~413 MB of memory overhead and a large dependency tree. Qdrant similarly adds significant complexity. sqlite-vec is lightweight but too slow.

### When to Reconsider

- If KBs grow to 1M+ vectors where IVF training becomes a bottleneck
- If metadata filtering at the index level becomes a critical requirement
- If sqlite-vec adds ANN indexing (planned in their roadmap)
- If the document DB needs to be unified with vector storage for transactional consistency

## Benchmark Reproduction

```bash
cd /ai/scripts/customkb2
source .venv/bin/activate
pip install chromadb qdrant-client sqlite-vec
python docs/benchmark_vectordb.py
```

Raw results: `docs/benchmark_results.json`

---
*Evaluated 2026-01-31*
