import os
import csv
from collections import defaultdict
import numpy as np
import torch

# Import ColBERTv2 code-base (adjust path if installed differently)
from colbert.infra import Run, RunConfig, ColBERTConfig
from colbert import Indexer, Searcher
from colbert.data import Queries

def load_qrels(qrels_path):
    """Load qrels into dict: qid → set of relevant pids."""
    qrels = defaultdict(set)
    with open(qrels_path, 'r', newline='') as f:
        reader = csv.reader(f, delimiter='\t')
        for qid, _, pid, rel in reader:
            if int(rel) > 0:
                # Convert to integers to match ColBERT's format
                qrels[int(qid)].add(int(pid))
    return qrels


def compute_recall_at_k(ranked_lists, qrels, K):
    total = 0.0
    num_q = 0
    for qid, retrieved in ranked_lists.items():
        rel = qrels.get(qid, set())
        if len(rel) == 0:
            continue  # Skip queries with no relevant docs
        num_q += 1
        topk = set(retrieved[:K])
        total += (len(topk & rel) / len(rel))
    return total / num_q if num_q > 0 else 0.0

def compute_mrr_at_k(ranked_lists, qrels, K):
    rr_total = 0.0
    num_q = 0
    for qid, retrieved in ranked_lists.items():
        rel = qrels.get(qid, set())
        if len(rel) == 0:
            continue  # Skip queries with no relevant docs
        num_q += 1
        for rank, pid in enumerate(retrieved[:K], start=1):
            if pid in rel:
                rr_total += 1.0 / rank
                break
    return rr_total / num_q if num_q > 0 else 0.0

def evaluate_all(ranked_lists, qrels):
    print("\n" + "="*60)
    print("EVALUATION RESULTS")
    print("="*60)
    for K in [1, 5, 10]:
        recall = compute_recall_at_k(ranked_lists, qrels, K)
        mrr    = compute_mrr_at_k(ranked_lists, qrels, K)
        print(f"Recall@{K:2d} = {recall:.4f}")
        print(f"MRR@{K:2d}    = {mrr:.4f}")
        print("-"*60)

def main():
    # Paths — adjust to your environment
    # You can also pass these as command line arguments
    import sys
    
    base_dir = r"./"
    
    # Default checkpoint: use the HF repo id (will download if not present).
    # If you already downloaded a local checkpoint tar.gz, set `colbert_ckpt` to its path.
    # Notes:
    # - Public repo id: 'colbert-ir/colbertv2.0' (recommended).
    # - If the repo is private or gated, authenticate with `huggingface-cli login`
    #   or set the env var HUGGINGFACE_HUB_TOKEN.
    colbert_ckpt = r"colbert-ir/colbertv2.0"  
    collection_path = os.path.join(base_dir, "collection.tsv")
    queries_path    = os.path.join(base_dir, "queries.dev.tsv")
    qrels_path      = os.path.join(base_dir, "qrels.dev.tsv")
    index_name      = "test_colbertv2_index"
    root            = os.path.join(base_dir, "experiments")
    
    # Check if required files exist
    print("Checking required files...")
    for name, path in [("Collection", collection_path), 
                       ("Queries", queries_path), 
                       ("Qrels", qrels_path)]:
        if not os.path.exists(path):
            print(f"❌ Error: {name} file not found: {path}")
            print("Please run split_test_data.py first to create the required files.")
            return
        else:
            print(f"✓ {name}: {path}")
    
    # Check checkpoint
    # - If the value looks like a local path (contains path separator or endswith .tar.gz), verify it exists.
    # - Otherwise treat it as a Hugging Face repo id (will be downloaded by transformers/huggingface_hub).
    # looks_like_local = (os.path.sep in colbert_ckpt) or colbert_ckpt.endswith('.tar.gz')
    # if looks_like_local:
    #     if not os.path.exists(colbert_ckpt):
    #         print(f"\n❌ Error: ColBERT checkpoint not found: {colbert_ckpt}")
    #         print("If you intended a local file, please check the path and re-run.")
    #         print("You can download the checkpoint from: https://huggingface.co/colbert-ir/colbertv2.0")
    #         return
    #     else:
    #         print(f"✓ Checkpoint (local): {colbert_ckpt}\n")
    # else:
    #     # Treat as HF repo id
    #     print(f"✓ Checkpoint (HF repo id): {colbert_ckpt} — will download if not present.")
    #     print("If this repo is private/gated, authenticate with `huggingface-cli login` or set HUGGINGFACE_HUB_TOKEN.")
    #     print("")

    # 1. Indexing (if not already done)
    config = ColBERTConfig(
        root = root,
        # you may set other params, e.g., nbits compression etc
    )

    try:
        indexer = Indexer(checkpoint=colbert_ckpt, config=config)
        # Check if index already exists
        index_path = os.path.join(root, "default", "indexes", index_name)
        if os.path.exists(index_path):
            print(f"\n✓ Index already exists at: {index_path}")
            print("Reusing existing index...\n")
        else:
            print(f"\nIndexing collection into: {index_path}")
            indexer.index(name=index_name, collection=collection_path)
    except OSError as e:
        print("\n❌ Failed to load the ColBERT checkpoint.")
        print("Reason:", str(e))
        print("\nPossible causes and fixes:")
        print(" - You provided a local path that doesn't exist. Check the path and re-run.")
        print(" - You passed a Hugging Face repo id but it's private/gated. Authenticate by:")
        print("     1) running: huggingface-cli login")
        print("     2) or setting env var: set HUGGINGFACE_HUB_TOKEN=<your_token> in PowerShell")
        print(" - Alternatively, download the checkpoint tar.gz manually and set `colbert_ckpt` to its path.")
        print("\nAfter fixing authentication or using a local checkpoint, re-run this script.")
        return
    
    # 2. Retrieval
    searcher = Searcher(index=index_name, config=config)
    queries   = Queries(queries_path)  # expects format: qid \t query_text
    k = 10  # or larger if you want Recall@100 etc
    
    print(f"\nSearching {len(queries)} queries...")
    ranking = searcher.search_all(queries, k=k)
    
    # ===== DEBUG: Inspect ranking format and content =====
    print("\n" + "="*60)
    print("DEBUG: Ranking inspection")
    print("="*60)
    print(f"Ranking type: {type(ranking)}")
    
    # Convert Ranking object to dict - try different methods
    try:
        ranking_dict = ranking.todict()
    except:
        try:
            ranking_dict = dict(ranking.data)
        except:
            ranking_dict = {qid: ranking[qid] for qid in range(len(queries))}
    
    print(f"Total queries in ranking: {len(ranking_dict)}")
    
    # Show first 5 query results
    print("\nFirst 5 queries - raw ranking format:")
    for i, (qid, hits) in enumerate(list(ranking_dict.items())[:5]):
        print(f"\n  Query ID: {qid} (type: {type(qid).__name__})")
        print(f"  Number of hits: {len(hits)}")
        if len(hits) > 0:
            print(f"  First hit raw: {hits[0]} (type: {type(hits[0]).__name__})")
            if len(hits) > 1:
                print(f"  Second hit raw: {hits[1]}")
    
    # Count unique returned pids
    all_returned_pids = set()
    for qid, hits in ranking_dict.items():
        for h in hits:
            if isinstance(h, (list, tuple)) and len(h) >= 3:
                pid = h[0]  # (pid, rank, score)
            elif isinstance(h, (list, tuple)) and len(h) >= 1:
                pid = h[0]  # (pid, ...)
            else:
                pid = h
            all_returned_pids.add(pid)
    
    print(f"\n  Total unique PIDs returned across all queries: {len(all_returned_pids)}")
    print(f"  Sample PIDs (first 10): {list(all_returned_pids)[:10]}")
    print("="*60 + "\n")
    
    # ranking will contain for each qid a list of (pid, rank, score)
    ranked_lists = {
        int(qid): [int(pid) for (pid, rank, score) in hits]
        for qid, hits in ranking_dict.items()
    }
    print(f"Retrieved results for {len(ranked_lists)} queries")

    # 3. Load qrels
    qrels = load_qrels(qrels_path)
    
    # ===== DEBUG: Inspect qrels =====
    print("\n" + "="*60)
    print("DEBUG: Qrels inspection")
    print("="*60)
    print(f"Total queries with relevance judgments: {len(qrels)}")
    total_relevant = sum(len(pids) for pids in qrels.values())
    print(f"Total relevant documents: {total_relevant}")
    if len(qrels) > 0:
        avg_relevant = total_relevant / len(qrels)
        print(f"Average relevant docs per query: {avg_relevant:.2f}")
    
    # Show first 5 qrels
    print("\nFirst 5 queries in qrels:")
    for i, (qid, rel_pids) in enumerate(list(qrels.items())[:5]):
        print(f"  QID {qid}: {len(rel_pids)} relevant docs -> {sorted(list(rel_pids))[:10]}")
    
    # Check overlap between retrieved and qrels
    qids_in_ranking = set(ranked_lists.keys())
    qids_in_qrels = set(qrels.keys())
    overlap = qids_in_ranking & qids_in_qrels
    print(f"\nQuery ID overlap:")
    print(f"  QIDs in ranking: {len(qids_in_ranking)}")
    print(f"  QIDs in qrels: {len(qids_in_qrels)}")
    print(f"  Overlap (will be evaluated): {len(overlap)}")
    
    # Check a sample query's retrieval vs relevance
    if len(overlap) > 0:
        sample_qid = list(overlap)[0]
        retrieved = ranked_lists[sample_qid]
        relevant = qrels[sample_qid]
        hits = set(retrieved) & relevant
        print(f"\nSample evaluation (QID {sample_qid}):")
        print(f"  Retrieved (top {len(retrieved)}): {retrieved}")
        print(f"  Relevant: {sorted(list(relevant))}")
        print(f"  Hits: {sorted(list(hits))} ({len(hits)}/{len(relevant)} relevant found)")
    print("="*60 + "\n")

    # 4. Evaluate
    evaluate_all(ranked_lists, qrels)

if __name__ == "__main__":
    main()