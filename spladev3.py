"""
SPLADE-v3 Evaluation Script
Evaluate SPLADE-v3 model on test dataset with same metrics as ColBERTv2
"""

from collections import defaultdict
import os
import csv
from tqdm import tqdm
import torch
from sentence_transformers import SparseEncoder
import argparse
import time


def load_collection(collection_path):
    """Load collection (passages) from TSV file."""
    print(f"\nLoading collection from: {collection_path}")
    collection = {}
    with open(collection_path, 'r', encoding='utf-8', newline='') as f:
        reader = csv.reader(f, delimiter='\t')
        for row in reader:
            if len(row) >= 2:
                pid = int(row[0])
                passage = row[1]
                collection[pid] = passage
    print(f"✓ Loaded {len(collection)} passages")
    return collection


def load_queries(queries_path):
    """Load queries from TSV file."""
    print(f"\nLoading queries from: {queries_path}")
    queries = {}
    with open(queries_path, 'r', encoding='utf-8', newline='') as f:
        reader = csv.reader(f, delimiter='\t')
        for row in reader:
            if len(row) >= 2:
                qid = int(row[0])
                query = row[1]
                queries[qid] = query
    print(f"✓ Loaded {len(queries)} queries")
    return queries


def load_qrels(qrels_path):
    """Load qrels into dict: qid → set of relevant pids."""
    print(f"\nLoading qrels from: {qrels_path}")
    qrels = defaultdict(set)
    with open(qrels_path, 'r', encoding='utf-8', newline='') as f:
        reader = csv.reader(f, delimiter='\t')
        for row in reader:
            if len(row) >= 4:
                qid = int(row[0])
                pid = int(row[2])
                relevance = int(row[3])
                if relevance > 0:
                    qrels[qid].add(pid)
    print(f"✓ Loaded {len(qrels)} queries with relevance judgments")
    total_relevant = sum(len(pids) for pids in qrels.values())
    print(f"✓ Total relevant documents: {total_relevant}")
    return qrels


def compute_recall_at_k(ranked_lists, qrels, K):
    """Compute Recall@K."""
    total = 0.0
    num_q = 0
    for qid, retrieved in ranked_lists.items():
        if qid not in qrels or len(qrels[qid]) == 0:
            continue
        relevant = qrels[qid]
        retrieved_at_k = set(retrieved[:K])
        hits = len(relevant & retrieved_at_k)
        total += hits / len(relevant)
        num_q += 1
    return total / num_q if num_q > 0 else 0.0


def compute_mrr_at_k(ranked_lists, qrels, K):
    """Compute MRR@K (Mean Reciprocal Rank)."""
    rr_total = 0.0
    num_q = 0
    for qid, retrieved in ranked_lists.items():
        if qid not in qrels or len(qrels[qid]) == 0:
            continue
        relevant = qrels[qid]
        for rank, pid in enumerate(retrieved[:K], 1):
            if pid in relevant:
                rr_total += 1.0 / rank
                break
        num_q += 1
    return rr_total / num_q if num_q > 0 else 0.0


def evaluate_all(ranked_lists, qrels):
    """Evaluate and print all metrics."""
    print("\n" + "="*60)
    print("EVALUATION RESULTS")
    print("="*60)
    for K in [1, 5, 10]:
        recall = compute_recall_at_k(ranked_lists, qrels, K)
        mrr = compute_mrr_at_k(ranked_lists, qrels, K)
        print(f"Recall@{K:2d} = {recall:.4f}")
        print(f"MRR@{K:2d}    = {mrr:.4f}")
        print("-" * 60)


def search_with_splade(model, queries, collection, k=10, batch_size=32, device='cuda'):
    """
    Search using SPLADE-v3 model.
    
    Args:
        model: SparseEncoder model
        queries: dict {qid: query_text}
        collection: dict {pid: passage_text}
        k: number of top documents to retrieve
        batch_size: batch size for encoding
        device: 'cuda' or 'cpu'
    
    Returns:
        ranked_lists: dict {qid: [pid1, pid2, ...]}
    """
    print("\n" + "="*60)
    print("SPLADE-v3 RETRIEVAL")
    print("="*60)
    
    # Move model to device
    model = model.to(device)
    
    # Encode all documents first
    print(f"\nEncoding {len(collection)} documents...")
    doc_ids = list(collection.keys())
    doc_texts = [collection[pid] for pid in doc_ids]
    
    all_doc_embeddings = []
    doc_encoding_time = 0.0
    
    for i in tqdm(range(0, len(doc_texts), batch_size), desc="Encoding documents"):
        batch = doc_texts[i:i+batch_size]
        
        # Measure encoding time
        start_time = time.time()
        with torch.no_grad():
            embeddings = model.encode_document(batch)
            if device == 'cuda':
                torch.cuda.synchronize()  # Wait for GPU operations to complete
        doc_encoding_time += time.time() - start_time
        
        all_doc_embeddings.append(embeddings)
    
    # Concatenate all document embeddings
    doc_embeddings = torch.cat(all_doc_embeddings, dim=0)  # [num_docs, vocab_size]
    print(f"✓ Document embeddings shape: {doc_embeddings.shape}")
    print(f"✓ Document encoding time: {doc_encoding_time:.2f}s")
    print(f"✓ Avg time per document: {doc_encoding_time / len(collection) * 1000:.2f}ms")
    
    # Calculate approximate FLOPs for document encoding
    # Approximate: embedding_dim * num_tokens * num_docs
    vocab_size = doc_embeddings.shape[1]
    avg_tokens_per_doc = 100  # Approximate average
    approx_flops_per_doc = vocab_size * avg_tokens_per_doc * 2  # 2 for multiply-add
    total_doc_flops = approx_flops_per_doc * len(collection)
    flops_per_sec = total_doc_flops / doc_encoding_time if doc_encoding_time > 0 else 0
    print(f"✓ Approx document encoding: {total_doc_flops / 1e9:.2f} GFLOPs")
    print(f"✓ Throughput: {flops_per_sec / 1e9:.2f} GFLOPs/s")
    
    # Encode queries and search
    print(f"\nSearching {len(queries)} queries...")
    ranked_lists = {}
    
    query_ids = list(queries.keys())
    query_texts = [queries[qid] for qid in query_ids]
    
    query_encoding_time = 0.0
    search_time = 0.0
    
    for i in tqdm(range(0, len(query_texts), batch_size), desc="Searching"):
        batch_qids = query_ids[i:i+batch_size]
        batch_queries = query_texts[i:i+batch_size]
        
        with torch.no_grad():
            # Encode queries
            start_time = time.time()
            query_embeddings = model.encode_query(batch_queries)  # [batch_size, vocab_size]
            if device == 'cuda':
                torch.cuda.synchronize()
            query_encoding_time += time.time() - start_time
            
            # Compute similarities with all documents
            start_time = time.time()
            similarities = model.similarity(query_embeddings, doc_embeddings)  # [batch_size, num_docs]
            
            # Get top-k for each query
            top_k_scores, top_k_indices = torch.topk(similarities, k=min(k, len(doc_ids)), dim=1)
            if device == 'cuda':
                torch.cuda.synchronize()
            search_time += time.time() - start_time
            
            # Convert to ranked lists
            for j, qid in enumerate(batch_qids):
                indices = top_k_indices[j].cpu().tolist()
                ranked_pids = [doc_ids[idx] for idx in indices]
                ranked_lists[qid] = ranked_pids
    
    print(f"✓ Retrieved results for {len(ranked_lists)} queries")
    print(f"✓ Query encoding time: {query_encoding_time:.2f}s")
    print(f"✓ Avg time per query: {query_encoding_time / len(queries) * 1000:.2f}ms")
    
    # Calculate approximate FLOPs for query encoding
    vocab_size = doc_embeddings.shape[1]
    avg_tokens_per_query = 10  # Approximate average
    approx_flops_per_query = vocab_size * avg_tokens_per_query * 2
    total_query_flops = approx_flops_per_query * len(queries)
    query_flops_per_sec = total_query_flops / query_encoding_time if query_encoding_time > 0 else 0
    print(f"✓ Approx query encoding: {total_query_flops / 1e9:.2f} GFLOPs")
    print(f"✓ Throughput: {query_flops_per_sec / 1e9:.2f} GFLOPs/s")
    
    print(f"\n✓ Search/ranking time: {search_time:.2f}s")
    print(f"✓ Avg search time per query: {search_time / len(queries) * 1000:.2f}ms")
    print(f"✓ Total latency per query: {(query_encoding_time + search_time) / len(queries) * 1000:.2f}ms")
    
    return ranked_lists


def main():
    parser = argparse.ArgumentParser(description='Evaluate SPLADE-v3 on test dataset')
    parser.add_argument('--collection', type=str, default='collection.tsv',
                        help='Path to collection.tsv file')
    parser.add_argument('--queries', type=str, default='queries.dev.tsv',
                        help='Path to queries.dev.tsv file')
    parser.add_argument('--qrels', type=str, default='qrels.dev.tsv',
                        help='Path to qrels.dev.tsv file')
    parser.add_argument('--k', type=int, default=10,
                        help='Number of top documents to retrieve (default: 10)')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for encoding (default: 32)')
    parser.add_argument('--device', type=str, default='cuda',
                        choices=['cuda', 'cpu'],
                        help='Device to use: cuda or cpu (default: cuda)')
    parser.add_argument('--model', type=str, default='naver/splade-v3',
                        help='SPLADE model name (default: naver/splade-v3)')
    
    args = parser.parse_args()
    
    # Check if CUDA is available
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("⚠️  CUDA not available, falling back to CPU")
        args.device = 'cpu'
    
    print("="*60)
    print("SPLADE-v3 EVALUATION")
    print("="*60)
    print(f"Model: {args.model}")
    print(f"Device: {args.device}")
    print(f"Batch size: {args.batch_size}")
    print(f"Top-K: {args.k}")
    print("="*60)
    
    # Load data
    collection = load_collection(args.collection)
    queries = load_queries(args.queries)
    qrels = load_qrels(args.qrels)
    
    # Load SPLADE model
    print("\n" + "="*60)
    print("LOADING SPLADE-v3 MODEL")
    print("="*60)
    print(f"Model: {args.model}")
    model = SparseEncoder(args.model)
    print("✓ Model loaded successfully")
    
    # Perform retrieval
    ranked_lists = search_with_splade(
        model=model,
        queries=queries,
        collection=collection,
        k=args.k,
        batch_size=args.batch_size,
        device=args.device
    )
    
    # Evaluate
    evaluate_all(ranked_lists, qrels)
    
    # Print sample results
    print("\n" + "="*60)
    print("SAMPLE RESULTS (first 3 queries)")
    print("="*60)
    for i, (qid, pids) in enumerate(list(ranked_lists.items())[:3]):
        print(f"\nQuery {qid}: {queries[qid]}")
        print(f"Relevant: {qrels[qid]}")
        print(f"Retrieved (top 5):")
        for rank, pid in enumerate(pids[:5], 1):
            marker = "✓" if pid in qrels[qid] else "✗"
            print(f"  {rank}. [{marker}] Doc {pid}: {collection[pid][:100]}...")
    
    print("\n" + "="*60)
    print("EVALUATION COMPLETE")
    print("="*60)


if __name__ == "__main__":
    main()
