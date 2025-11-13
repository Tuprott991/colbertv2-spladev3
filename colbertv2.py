import os
import csv
from collections import defaultdict
import numpy as np
import torch
import argparse
import requests
import gzip
import shutil
from tqdm import tqdm

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


def download_file(url, output_path, description="Downloading"):
    """Download file with progress bar."""
    print(f"{description}: {url}")
    response = requests.get(url, stream=True)
    response.raise_for_status()
    
    total_size = int(response.headers.get('content-length', 0))
    
    with open(output_path, 'wb') as f:
        if total_size == 0:
            f.write(response.content)
        else:
            with tqdm(total=total_size, unit='B', unit_scale=True, desc=description) as pbar:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
                    pbar.update(len(chunk))
    
    print(f"✓ Saved to: {output_path}\n")


def decompress_gz(gz_path, output_path):
    """Decompress .gz file."""
    print(f"Decompressing: {gz_path}")
    with gzip.open(gz_path, 'rb') as f_in:
        with open(output_path, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)
    print(f"✓ Decompressed to: {output_path}\n")


def download_msmarco_dev(base_dir):
    """
    Download MS MARCO passage ranking dev set (small version for testing).
    
    Files:
    - collection.tsv: ~8.8M passages
    - queries.dev.small.tsv: 6,980 queries
    - qrels.dev.small.tsv: relevance judgments
    
    This is a much harder dataset than the toy dataset.
    """
    print("\n" + "="*60)
    print("DOWNLOADING MS MARCO DEV DATASET (SMALL)")
    print("="*60)
    
    msmarco_dir = os.path.join(base_dir, "msmarco_data")
    os.makedirs(msmarco_dir, exist_ok=True)
    
    # MS MARCO URLs
    base_url = "https://msmarco.blob.core.windows.net/msmarcoranking"
    
    files_to_download = [
        {
            "url": f"{base_url}/collection.tar.gz",
            "compressed": os.path.join(msmarco_dir, "collection.tar.gz"),
            "final": os.path.join(msmarco_dir, "collection.tsv"),
            "is_tar": True
        },
        {
            "url": f"{base_url}/queries.dev.small.tar.gz",
            "compressed": os.path.join(msmarco_dir, "queries.dev.small.tar.gz"),
            "final": os.path.join(msmarco_dir, "queries.dev.small.tsv"),
            "is_tar": True
        },
        {
            "url": f"{base_url}/qrels.dev.small.tsv",
            "final": os.path.join(msmarco_dir, "qrels.dev.small.tsv"),
            "is_tar": False
        }
    ]
    
    for file_info in files_to_download:
        final_path = file_info["final"]
        
        # Check if already exists
        if os.path.exists(final_path):
            print(f"✓ Already exists: {final_path}")
            continue
        
        # Download
        if file_info["is_tar"]:
            compressed_path = file_info["compressed"]
            if not os.path.exists(compressed_path):
                download_file(file_info["url"], compressed_path, 
                            description=f"Downloading {os.path.basename(compressed_path)}")
            
            # Extract tar.gz
            print(f"Extracting: {compressed_path}")
            import tarfile
            with tarfile.open(compressed_path, 'r:gz') as tar:
                tar.extractall(path=msmarco_dir)
            print(f"✓ Extracted to: {msmarco_dir}\n")
        else:
            download_file(file_info["url"], final_path,
                        description=f"Downloading {os.path.basename(final_path)}")
    
    print("="*60)
    print("MS MARCO DATASET READY")
    print("="*60)
    print(f"Collection: {os.path.join(msmarco_dir, 'collection.tsv')}")
    print(f"Queries: {os.path.join(msmarco_dir, 'queries.dev.small.tsv')}")
    print(f"Qrels: {os.path.join(msmarco_dir, 'qrels.dev.small.tsv')}")
    print("="*60 + "\n")
    
    return {
        "collection": os.path.join(msmarco_dir, "collection.tsv"),
        "queries": os.path.join(msmarco_dir, "queries.dev.small.tsv"),
        "qrels": os.path.join(msmarco_dir, "qrels.dev.small.tsv"),
        "index_name": "msmarco_colbertv2_index"
    }


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='ColBERTv2 Evaluation Script')
    parser.add_argument('--msmarco', action='store_true', 
                       help='Download and use MS MARCO dev dataset instead of local test data')
    parser.add_argument('--base-dir', type=str, default='./',
                       help='Base directory for data and experiments')
    parser.add_argument('--checkpoint', type=str, default='colbert-ir/colbertv2.0',
                       help='ColBERT checkpoint path or HuggingFace repo id')
    args = parser.parse_args()
    
    base_dir = args.base_dir
    colbert_ckpt = args.checkpoint
    
    # Determine which dataset to use
    if args.msmarco:
        print("\n" + "="*60)
        print("MS MARCO MODE SELECTED")
        print("="*60)
        dataset_info = download_msmarco_dev(base_dir)
        collection_path = dataset_info["collection"]
        queries_path = dataset_info["queries"]
        qrels_path = dataset_info["qrels"]
        index_name = dataset_info["index_name"]
    else:
        print("\n" + "="*60)
        print("LOCAL TEST DATA MODE")
        print("="*60)
        collection_path = os.path.join(base_dir, "collection.tsv")
        queries_path = os.path.join(base_dir, "queries.dev.tsv")
        qrels_path = os.path.join(base_dir, "qrels.dev.tsv")
        index_name = "test_colbertv2_index"
    
    root = os.path.join(base_dir, "experiments")
    
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