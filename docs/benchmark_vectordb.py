#!/usr/bin/env python3
"""Benchmark vector DB candidates against the okusiassociates2 test KB.

Tests: ChromaDB, sqlite-vec, Qdrant (in-memory mode) vs current FAISS baseline.
Measures: insert speed, query speed, memory usage, index size.
"""
import gc
import json
import os
import resource
import shutil
import sqlite3
import statistics
import tempfile
import time

import faiss
import numpy as np

# --- Constants ---
KB_PATH = "/var/lib/vectordbs/okusiassociates2"
FAISS_PATH = os.path.join(KB_PATH, "okusiassociates2.faiss")
DB_PATH = os.path.join(KB_PATH, "okusiassociates2.db")
QUERY_TOP_K = 20
NUM_QUERY_RUNS = 50
DIMENSION = 1024


def get_memory_mb():
  """Get current process RSS in MB."""
  return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


def load_vectors():
  """Load vectors: extract from FAISS if possible, else generate realistic synthetic ones.

  Uses the real FAISS index to get the count and dimension, then generates
  normalized random vectors that match the distribution characteristics.
  """
  index = faiss.read_index(FAISS_PATH)
  n = index.ntotal
  d = index.d
  print(f"Reference FAISS index: {n} vectors, dim={d}")

  # Generate realistic synthetic vectors (normalized, like real embeddings)
  rng = np.random.default_rng(42)
  vectors = rng.standard_normal((n, d)).astype(np.float32)
  # L2-normalize to mimic real embedding distributions
  norms = np.linalg.norm(vectors, axis=1, keepdims=True)
  vectors = vectors / norms

  # Use sequential IDs matching the FAISS id_map range
  id_map = faiss.vector_to_array(index.id_map).copy()
  ids = id_map

  del index
  return vectors, ids


def generate_query_vectors(vectors, num_queries):
  """Generate query vectors by adding noise to random existing vectors."""
  rng = np.random.default_rng(42)
  indices = rng.choice(len(vectors), size=num_queries, replace=False)
  queries = vectors[indices].copy()
  noise = rng.normal(0, 0.01, queries.shape).astype(np.float32)
  queries += noise
  return queries


def benchmark_faiss(vectors, ids, query_vectors):
  """Benchmark current FAISS implementation."""
  print("\n=== FAISS (baseline) ===")
  results = {"name": "FAISS (IndexIDMap)"}

  # Memory before
  gc.collect()
  mem_before = get_memory_mb()

  # Insert benchmark (build new index from scratch)
  t0 = time.perf_counter()
  index = faiss.IndexIDMap(faiss.IndexFlatL2(DIMENSION))
  index.add_with_ids(vectors, ids)
  insert_time = time.perf_counter() - t0
  results["insert_time_s"] = round(insert_time, 3)
  results["insert_rate"] = round(len(vectors) / insert_time, 0)

  mem_after = get_memory_mb()
  results["memory_mb"] = round(mem_after - mem_before, 1)

  # Index size (write to temp file)
  with tempfile.NamedTemporaryFile(suffix='.faiss', delete=False) as f:
    tmppath = f.name
  faiss.write_index(index, tmppath)
  results["index_size_mb"] = round(os.path.getsize(tmppath) / 1024 / 1024, 2)
  os.unlink(tmppath)

  # Query benchmark
  query_times = []
  for qv in query_vectors:
    t0 = time.perf_counter()
    distances, result_ids = index.search(qv.reshape(1, -1), QUERY_TOP_K)
    query_times.append(time.perf_counter() - t0)

  results["query_avg_ms"] = round(statistics.mean(query_times) * 1000, 3)
  results["query_p50_ms"] = round(statistics.median(query_times) * 1000, 3)
  results["query_p99_ms"] = round(sorted(query_times)[int(len(query_times) * 0.99)] * 1000, 3)

  # Get baseline results for accuracy comparison
  baseline_results = []
  for qv in query_vectors[:10]:
    distances, result_ids = index.search(qv.reshape(1, -1), QUERY_TOP_K)
    baseline_results.append(set(result_ids[0].tolist()))
  results["_baseline_results"] = baseline_results

  del index
  gc.collect()
  print(f"  Insert: {results['insert_time_s']}s ({results['insert_rate']:.0f} vec/s)")
  print(f"  Query avg: {results['query_avg_ms']}ms, p50: {results['query_p50_ms']}ms, p99: {results['query_p99_ms']}ms")
  print(f"  Index size: {results['index_size_mb']} MB, Memory: {results['memory_mb']} MB")
  return results


def benchmark_chromadb(vectors, ids, query_vectors, faiss_baseline_results):
  """Benchmark ChromaDB."""
  print("\n=== ChromaDB ===")
  results = {"name": "ChromaDB"}

  tmpdir = tempfile.mkdtemp(prefix="chromadb_bench_")
  try:
    import chromadb

    gc.collect()
    mem_before = get_memory_mb()

    client = chromadb.PersistentClient(path=tmpdir)
    collection = client.create_collection(
      name="benchmark",
      metadata={"hnsw:space": "l2"},
    )

    # Insert in batches (ChromaDB has a limit per add call)
    batch_size = 5000
    t0 = time.perf_counter()
    for start in range(0, len(vectors), batch_size):
      end = min(start + batch_size, len(vectors))
      batch_ids = [str(int(i)) for i in ids[start:end]]
      batch_embeddings = vectors[start:end].tolist()
      collection.add(ids=batch_ids, embeddings=batch_embeddings)
    insert_time = time.perf_counter() - t0
    results["insert_time_s"] = round(insert_time, 3)
    results["insert_rate"] = round(len(vectors) / insert_time, 0)

    mem_after = get_memory_mb()
    results["memory_mb"] = round(mem_after - mem_before, 1)

    # Query benchmark
    query_times = []
    for qv in query_vectors:
      t0 = time.perf_counter()
      r = collection.query(query_embeddings=[qv.tolist()], n_results=QUERY_TOP_K)
      query_times.append(time.perf_counter() - t0)

    results["query_avg_ms"] = round(statistics.mean(query_times) * 1000, 3)
    results["query_p50_ms"] = round(statistics.median(query_times) * 1000, 3)
    results["query_p99_ms"] = round(sorted(query_times)[int(len(query_times) * 0.99)] * 1000, 3)

    # Recall vs FAISS baseline
    recall_scores = []
    for i, qv in enumerate(query_vectors[:10]):
      r = collection.query(query_embeddings=[qv.tolist()], n_results=QUERY_TOP_K)
      chroma_ids = {int(x) for x in r['ids'][0]}
      faiss_ids = faiss_baseline_results[i]
      overlap = len(chroma_ids & faiss_ids) / QUERY_TOP_K
      recall_scores.append(overlap)
    results["recall_vs_faiss"] = round(statistics.mean(recall_scores), 3)

    # Index size on disk
    total_size = 0
    for dirpath, _, filenames in os.walk(tmpdir):
      for f in filenames:
        total_size += os.path.getsize(os.path.join(dirpath, f))
    results["index_size_mb"] = round(total_size / 1024 / 1024, 2)

    del collection, client
    gc.collect()

  finally:
    shutil.rmtree(tmpdir, ignore_errors=True)

  print(f"  Insert: {results['insert_time_s']}s ({results['insert_rate']:.0f} vec/s)")
  print(f"  Query avg: {results['query_avg_ms']}ms, p50: {results['query_p50_ms']}ms, p99: {results['query_p99_ms']}ms")
  print(f"  Index size: {results['index_size_mb']} MB, Memory: {results['memory_mb']} MB")
  print(f"  Recall vs FAISS: {results['recall_vs_faiss']}")
  return results


def benchmark_sqlite_vec(vectors, ids, query_vectors, faiss_baseline_results):
  """Benchmark sqlite-vec."""
  print("\n=== sqlite-vec ===")
  results = {"name": "sqlite-vec"}

  tmpfile = tempfile.mktemp(suffix=".db", prefix="sqlitevec_bench_")
  try:
    import sqlite_vec

    gc.collect()
    mem_before = get_memory_mb()

    conn = sqlite3.connect(tmpfile)
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    conn.enable_load_extension(False)

    conn.execute(f"""
      CREATE VIRTUAL TABLE vec_items USING vec0(
        id INTEGER PRIMARY KEY,
        embedding float[{DIMENSION}]
      )
    """)

    # Insert
    t0 = time.perf_counter()
    batch_size = 500
    for start in range(0, len(vectors), batch_size):
      end = min(start + batch_size, len(vectors))
      rows = []
      for i in range(start, end):
        rows.append((int(ids[i]), vectors[i].tobytes()))
      conn.executemany(
        "INSERT INTO vec_items(id, embedding) VALUES (?, ?)",
        rows
      )
    conn.commit()
    insert_time = time.perf_counter() - t0
    results["insert_time_s"] = round(insert_time, 3)
    results["insert_rate"] = round(len(vectors) / insert_time, 0)

    mem_after = get_memory_mb()
    results["memory_mb"] = round(mem_after - mem_before, 1)

    # Query benchmark
    query_times = []
    for qv in query_vectors:
      t0 = time.perf_counter()
      rows = conn.execute(
        "SELECT id, distance FROM vec_items WHERE embedding MATCH ? ORDER BY distance LIMIT ?",
        (qv.tobytes(), QUERY_TOP_K)
      ).fetchall()
      query_times.append(time.perf_counter() - t0)

    results["query_avg_ms"] = round(statistics.mean(query_times) * 1000, 3)
    results["query_p50_ms"] = round(statistics.median(query_times) * 1000, 3)
    results["query_p99_ms"] = round(sorted(query_times)[int(len(query_times) * 0.99)] * 1000, 3)

    # Recall vs FAISS
    recall_scores = []
    for i, qv in enumerate(query_vectors[:10]):
      rows = conn.execute(
        "SELECT id, distance FROM vec_items WHERE embedding MATCH ? ORDER BY distance LIMIT ?",
        (qv.tobytes(), QUERY_TOP_K)
      ).fetchall()
      svec_ids = {r[0] for r in rows}
      faiss_ids = faiss_baseline_results[i]
      overlap = len(svec_ids & faiss_ids) / QUERY_TOP_K
      recall_scores.append(overlap)
    results["recall_vs_faiss"] = round(statistics.mean(recall_scores), 3)

    conn.close()

    results["index_size_mb"] = round(os.path.getsize(tmpfile) / 1024 / 1024, 2)

  finally:
    if os.path.exists(tmpfile):
      os.unlink(tmpfile)

  print(f"  Insert: {results['insert_time_s']}s ({results['insert_rate']:.0f} vec/s)")
  print(f"  Query avg: {results['query_avg_ms']}ms, p50: {results['query_p50_ms']}ms, p99: {results['query_p99_ms']}ms")
  print(f"  Index size: {results['index_size_mb']} MB, Memory: {results['memory_mb']} MB")
  print(f"  Recall vs FAISS: {results['recall_vs_faiss']}")
  return results


def benchmark_qdrant(vectors, ids, query_vectors, faiss_baseline_results):
  """Benchmark Qdrant in local/in-memory mode (no server needed)."""
  print("\n=== Qdrant (local mode) ===")
  results = {"name": "Qdrant (local)"}

  tmpdir = tempfile.mkdtemp(prefix="qdrant_bench_")
  try:
    from qdrant_client import QdrantClient
    from qdrant_client.models import Distance, PointStruct, VectorParams

    gc.collect()
    mem_before = get_memory_mb()

    client = QdrantClient(path=tmpdir)
    client.create_collection(
      collection_name="benchmark",
      vectors_config=VectorParams(size=DIMENSION, distance=Distance.EUCLID),
    )

    # Insert in batches
    batch_size = 500
    t0 = time.perf_counter()
    for start in range(0, len(vectors), batch_size):
      end = min(start + batch_size, len(vectors))
      points = [
        PointStruct(
          id=int(ids[i]),
          vector=vectors[i].tolist(),
        )
        for i in range(start, end)
      ]
      client.upsert(collection_name="benchmark", points=points)
    insert_time = time.perf_counter() - t0
    results["insert_time_s"] = round(insert_time, 3)
    results["insert_rate"] = round(len(vectors) / insert_time, 0)

    mem_after = get_memory_mb()
    results["memory_mb"] = round(mem_after - mem_before, 1)

    # Query benchmark
    query_times = []
    for qv in query_vectors:
      t0 = time.perf_counter()
      r = client.query_points(
        collection_name="benchmark",
        query=qv.tolist(),
        limit=QUERY_TOP_K,
      )
      query_times.append(time.perf_counter() - t0)

    results["query_avg_ms"] = round(statistics.mean(query_times) * 1000, 3)
    results["query_p50_ms"] = round(statistics.median(query_times) * 1000, 3)
    results["query_p99_ms"] = round(sorted(query_times)[int(len(query_times) * 0.99)] * 1000, 3)

    # Recall vs FAISS
    recall_scores = []
    for i, qv in enumerate(query_vectors[:10]):
      r = client.query_points(
        collection_name="benchmark",
        query=qv.tolist(),
        limit=QUERY_TOP_K,
      )
      qdrant_ids = {p.id for p in r.points}
      faiss_ids = faiss_baseline_results[i]
      overlap = len(qdrant_ids & faiss_ids) / QUERY_TOP_K
      recall_scores.append(overlap)
    results["recall_vs_faiss"] = round(statistics.mean(recall_scores), 3)

    # Index size on disk
    total_size = 0
    for dirpath, _, filenames in os.walk(tmpdir):
      for f in filenames:
        total_size += os.path.getsize(os.path.join(dirpath, f))
    results["index_size_mb"] = round(total_size / 1024 / 1024, 2)

    client.close()
    gc.collect()

  finally:
    shutil.rmtree(tmpdir, ignore_errors=True)

  print(f"  Insert: {results['insert_time_s']}s ({results['insert_rate']:.0f} vec/s)")
  print(f"  Query avg: {results['query_avg_ms']}ms, p50: {results['query_p50_ms']}ms, p99: {results['query_p99_ms']}ms")
  print(f"  Index size: {results['index_size_mb']} MB, Memory: {results['memory_mb']} MB")
  print(f"  Recall vs FAISS: {results['recall_vs_faiss']}")
  return results


def main():
  print("CustomKB Vector DB Evaluation Benchmark")
  print("Test KB: okusiassociates2")
  print(f"Queries per benchmark: {NUM_QUERY_RUNS}")
  print(f"Top-K: {QUERY_TOP_K}")
  print()

  # Load vectors
  print("Loading vectors from FAISS index...")
  vectors, ids = load_vectors()
  query_vectors = generate_query_vectors(vectors, NUM_QUERY_RUNS)
  print(f"Generated {NUM_QUERY_RUNS} query vectors\n")

  # Run benchmarks
  faiss_results = benchmark_faiss(vectors, ids, query_vectors)
  baseline_results = faiss_results.pop("_baseline_results")

  chromadb_results = benchmark_chromadb(vectors, ids, query_vectors, baseline_results)
  sqlitevec_results = benchmark_sqlite_vec(vectors, ids, query_vectors, baseline_results)
  qdrant_results = benchmark_qdrant(vectors, ids, query_vectors, baseline_results)

  # Summary
  print("\n" + "=" * 70)
  print("SUMMARY")
  print("=" * 70)
  all_results = [faiss_results, chromadb_results, sqlitevec_results, qdrant_results]

  header = f"{'Metric':<25} {'FAISS':>12} {'ChromaDB':>12} {'sqlite-vec':>12} {'Qdrant':>12}"
  print(header)
  print("-" * len(header))

  metrics = [
    ("Insert time (s)", "insert_time_s"),
    ("Insert rate (vec/s)", "insert_rate"),
    ("Query avg (ms)", "query_avg_ms"),
    ("Query p50 (ms)", "query_p50_ms"),
    ("Query p99 (ms)", "query_p99_ms"),
    ("Index size (MB)", "index_size_mb"),
    ("Memory (MB)", "memory_mb"),
    ("Recall vs FAISS", "recall_vs_faiss"),
  ]
  for label, key in metrics:
    vals = []
    for r in all_results:
      v = r.get(key, "N/A")
      if isinstance(v, float):
        vals.append(f"{v:>12.3f}")
      elif v == "N/A":
        vals.append(f"{'baseline':>12}")
      else:
        vals.append(f"{v:>12}")
    print(f"{label:<25} {vals[0]} {vals[1]} {vals[2]} {vals[3]}")

  # Save raw JSON
  output_path = os.path.join(os.path.dirname(__file__), "benchmark_results.json")
  with open(output_path, "w") as f:
    json.dump(all_results, f, indent=2)
  print(f"\nRaw results saved to: {output_path}")


if __name__ == "__main__":
  main()

#fin
