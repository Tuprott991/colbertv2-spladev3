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
    
    base_dir = r"d:\Github Repos\ColBERT-and-SPLADE"
    
    # TODO: Download ColBERTv2 checkpoint from: https://huggingface.co/colbert-ir/colbertv2.0
    colbert_ckpt = r"colbertv2.0.tar.gz"  
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
    if not os.path.exists(colbert_ckpt):
        print(f"\n❌ Error: ColBERT checkpoint not found: {colbert_ckpt}")
        print("Please download the checkpoint from: https://huggingface.co/colbert-ir/colbertv2.0")
        print("Or use another ColBERT checkpoint and update the path.")
        return
    print(f"✓ Checkpoint: {colbert_ckpt}\n")

    # 1. Indexing (if not already done)
    config = ColBERTConfig(
        root = root,
        # you may set other params, e.g., nbits compression etc
    )
    indexer = Indexer(checkpoint=colbert_ckpt, config=config)
    indexer.index(name=index_name, collection=collection_path)
    
    # 2. Retrieval
    searcher = Searcher(index=index_name, config=config)
    queries   = Queries(queries_path)  # expects format: qid \t query_text
    k = 10  # or larger if you want Recall@100 etc
    
    print(f"\nSearching {len(queries)} queries...")
    ranking = searcher.search_all(queries, k=k)
    
    # ranking will contain for each qid a list of (pid, rank, score)
    ranked_lists = {
        int(qid): [int(pid) for (pid, rank, score) in hits]
        for qid, hits in ranking.items()
    }
    print(f"Retrieved results for {len(ranked_lists)} queries")

    # 3. Load qrels
    qrels = load_qrels(qrels_path)

    # 4. Evaluate
    evaluate_all(ranked_lists, qrels)

if __name__ == "__main__":
    main()