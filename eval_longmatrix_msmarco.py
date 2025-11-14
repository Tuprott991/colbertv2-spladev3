"""
LongMatrix Model Evaluation on MS MARCO
Evaluate trained LongMatrix model on MS MARCO dev/test set from Hugging Face

Metrics:
- Recall@1, Recall@5, Recall@10, Recall@50, Recall@1000
- MRR@1, MRR@5, MRR@10

Usage:
    python eval_longmatrix_msmarco.py --checkpoint runs/longmatrix/epoch1.pt --dataset msmarco --split dev
"""

import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from tqdm import tqdm
from collections import defaultdict
import math
from typing import List, Dict, Tuple
import numpy as np

# Try to import datasets from Hugging Face
try:
    from datasets import load_dataset
    _HAS_DATASETS = True
except (ImportError, AttributeError) as e:
    _HAS_DATASETS = False
    print(f"⚠️  datasets not available: {e}")
    print("    Solution 1: Use local TSV files with --queries_file, --collection_file, --qrels_file")
    print("    Solution 2: Fix httpx: pip install --upgrade httpx datasets")

# Try to import optimized BM25 libraries
try:
    from rank_bm25 import BM25Okapi
    _HAS_RANK_BM25 = True
except ImportError:
    _HAS_RANK_BM25 = False

try:
    from pyserini.search.lucene import LuceneSearcher
    from pyserini.index.lucene import IndexReader
    _HAS_PYSERINI = True
except ImportError:
    _HAS_PYSERINI = False


# ========================= Model Definition =========================
# (Copy from train_longmatrix.py)

class LexicalEncoder(nn.Module):
    """Multi-head + fused QKV attention with SDPA"""
    def __init__(self, vocab_size: int, d_lex_emb: int = 128, d_lex: int = 128,
                 attn_backend: str = 'sdpa', use_ckpt: bool = False, heads: int = 4):
        super().__init__()
        assert d_lex_emb % heads == 0, "d_lex_emb must be divisible by heads"
        self.emb = nn.Embedding(vocab_size, d_lex_emb)
        self.ln = nn.LayerNorm(d_lex_emb)
        self.heads = heads
        self.head_dim = d_lex_emb // heads

        # Fused QKV + output projection
        self.qkv = nn.Linear(d_lex_emb, 3 * d_lex_emb)
        self.o_proj = nn.Linear(d_lex_emb, d_lex_emb)

        self.attn_dropout = nn.Dropout(0.1)
        self.pool_vec = nn.Linear(d_lex_emb, 1)
        self.proj_h = nn.Linear(d_lex_emb, d_lex)

        self.attn_backend = attn_backend
        self.use_ckpt = use_ckpt
        self.last_attention = None
        self._sdpa_broken_warned = False

    def _shape_qkv(self, x):
        B, T, D = x.shape
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)
        def reshape(t):
            return t.view(B, T, self.heads, self.head_dim).transpose(1, 2).contiguous()
        return reshape(q), reshape(k), reshape(v)

    def _attend_sdpa(self, Q, K, V, attention_mask):
        attn_mask = attention_mask.bool().unsqueeze(1).unsqueeze(2)
        return F.scaled_dot_product_attention(
            Q, K, V,
            attn_mask=attn_mask,
            dropout_p=self.attn_dropout.p if self.training else 0.0,
            is_causal=False
        )

    def _attend_matmul(self, Q, K, V, attention_mask):
        B, H, T, dh = Q.shape
        scores = torch.matmul(Q, K.transpose(-1, -2)).float() / math.sqrt(dh)
        mask = attention_mask.unsqueeze(1).unsqueeze(2).to(dtype=torch.bool)
        scores = scores.masked_fill(~mask, float('-inf'))
        attn = torch.softmax(scores, dim=-1).to(Q.dtype)
        attn = self.attn_dropout(attn)
        ctx = torch.matmul(attn, V)
        return ctx, attn

    def _attend(self, Q, K, V, attention_mask):
        if self.attn_backend == 'sdpa':
            try:
                x_ctx = self._attend_sdpa(Q, K, V, attention_mask)
                return x_ctx, None
            except Exception as e:
                if not self._sdpa_broken_warned:
                    print(f"⚠️  SDPA failed: {e}, falling back to matmul")
                    self._sdpa_broken_warned = True
                return self._attend_matmul(Q, K, V, attention_mask)
        else:
            return self._attend_matmul(Q, K, V, attention_mask)

    def forward_block(self, input_ids, attention_mask, topk_tokens: int = 1):
        x = self.emb(input_ids)
        x = self.ln(x)

        Q, K, V = self._shape_qkv(x)

        # Safe mask
        mask_safe = attention_mask
        lens = mask_safe.sum(dim=-1)
        need_fix = (lens == 0)
        if need_fix.any():
            mask_safe = mask_safe.clone()
            mask_safe[need_fix, 0] = 1

        x_ctx, _ = self._attend(Q, K, V, mask_safe)

        # Merge heads
        B, H, T, dh = x_ctx.shape
        x_ctx = x_ctx.transpose(1, 2).contiguous().view(B, T, H * dh)
        x_ctx = self.o_proj(x_ctx)

        # Attention pooling
        alpha_logits = self.pool_vec(x_ctx).squeeze(-1).float()
        alpha_logits = alpha_logits.masked_fill(mask_safe == 0, -1e9)
        a = torch.softmax(alpha_logits, dim=-1).to(x_ctx.dtype)
        
        if torch.isnan(a).any():
            bad = torch.isnan(a).any(dim=1)
            if bad.any():
                a[bad] = torch.zeros_like(a[bad])
                a[bad, 0] = 1.0

        self.last_attention = a

        if topk_tokens is None or topk_tokens <= 1:
            h = torch.sum(a.unsqueeze(-1) * x_ctx, dim=1)
            h = self.proj_h(h)
            return h, a
        else:
            K_sel = min(int(topk_tokens), x_ctx.size(1))
            topk_idx = torch.topk(a, k=K_sel, dim=-1).indices
            B = x_ctx.size(0)
            topk_vec = x_ctx.gather(1, topk_idx.unsqueeze(-1).expand(B, K_sel, x_ctx.size(-1)))
            topk_vec = self.proj_h(topk_vec)
            return topk_vec, a

    def forward(self, input_ids, attention_mask, topk_tokens: int = 1):
        if self.use_ckpt and self.training:
            def _fn(ids, mask, k):
                return self.forward_block(ids, mask, k)
            h = torch.utils.checkpoint.checkpoint(_fn, input_ids, attention_mask, topk_tokens, use_reentrant=False)
            with torch.no_grad():
                _, a = self.forward_block(input_ids, attention_mask, topk_tokens)
            return h, a
        else:
            return self.forward_block(input_ids, attention_mask, topk_tokens)


class LowRankProjection(nn.Module):
    def __init__(self, d_lex: int, m_teacher: int, rank: int = 64):
        super().__init__()
        self.U = nn.Linear(d_lex, rank, bias=False)
        self.V = nn.Linear(m_teacher, rank, bias=False)
        nn.init.xavier_uniform_(self.U.weight)
        nn.init.xavier_uniform_(self.V.weight)
        self.m_teacher = m_teacher
        self.rank = rank

    def forward(self, h):
        if h.dim() == 3:
            B, K, D = h.shape
            r = self.U(h.view(B*K, D))
            z_hat = torch.matmul(r, self.V.weight)
            z_hat = F.normalize(z_hat, p=2, dim=-1)
            return z_hat.view(B, K, -1)
        else:
            r = self.U(h)
            z_hat = torch.matmul(r, self.V.weight)
            return F.normalize(z_hat, p=2, dim=-1)

    def ortho_reg(self):
        U = self.U.weight
        Vt = self.V.weight
        Iu = torch.eye(U.size(0), device=U.device)
        Iv = torch.eye(Vt.size(0), device=Vt.device)
        Mu = U @ U.t() - Iu
        Mv = Vt @ Vt.t() - Iv
        reg_u = (Mu * Mu).mean()
        reg_v = (Mv * Mv).mean()
        return reg_u + reg_v


class LongMatrixModel(nn.Module):
    def __init__(self, tokenizer: AutoTokenizer, d_lex_emb=128, d_lex=128, m_teacher=1024, rank=64,
                 attn_backend='sdpa', use_ckpt=False, heads=4):
        super().__init__()
        self.tokenizer = tokenizer
        self.lexical = LexicalEncoder(
            vocab_size=tokenizer.vocab_size,
            d_lex_emb=d_lex_emb, d_lex=d_lex,
            attn_backend=attn_backend, use_ckpt=use_ckpt, heads=heads
        )
        self.proj = LowRankProjection(d_lex=d_lex, m_teacher=m_teacher, rank=rank)

    def encode(self, batch_tok, topk_tokens: int = 1) -> Dict[str, torch.Tensor]:
        h, att = self.lexical(batch_tok['input_ids'], batch_tok['attention_mask'], topk_tokens=topk_tokens)
        z = self.proj(h)
        return {'z': z, 'h': h, 'att': self.lexical.last_attention}


# ========================= Late Interaction Scoring =========================

def _batch_late_scores(zq: torch.Tensor, zd: torch.Tensor) -> torch.Tensor:
    """
    zq: (Bq, Kq, m), zd: (Bd, Kd, m)
    Returns: sims (Bq, Bd)
    """
    Bq, Kq, M = zq.shape
    Bd, Kd, _ = zd.shape
    
    if Kd == 1:
        zq_mean = zq.mean(dim=1)
        zd_flat = zd.squeeze(1)
        return zq_mean @ zd_flat.t()
    
    # Kd > 1: compute in chunks
    chunk = max(1, 2048 // max(1, Kd))
    sims = []
    for start in range(0, Bd, chunk):
        end = min(Bd, start + chunk)
        zd_blk = zd[start:end]
        zq_e = zq.unsqueeze(1)
        zd_e = zd_blk.unsqueeze(0)
        sim = torch.matmul(zq_e, zd_e.transpose(-1, -2))
        sim = sim.max(dim=-1).values.mean(dim=-1)
        sims.append(sim)
    return torch.cat(sims, dim=1)


# ========================= Data Loading =========================

def load_msmarco_dataset(split='dev', version='v2.1'):
    """Load MS MARCO dataset from Hugging Face
    
    Args:
        split: 'train', 'dev', or 'validation'
        version: 'v1.1' or 'v2.1'
    """
    if not _HAS_DATASETS:
        raise RuntimeError(
            "Hugging Face datasets not available.\n"
            "Solution 1: Fix dependency - Run: pip install --upgrade httpx datasets\n"
            "Solution 2: Use local files - Add: --queries_file queries.tsv --collection_file collection.tsv --qrels_file qrels.tsv"
        )
    
    print(f"\nLoading MS MARCO {version} {split} set from Hugging Face...")
    
    if version == 'v1.1':
        return _load_msmarco_v1_1(split)
    elif version == 'v2.1':
        return _load_msmarco_v2_1(split)
    else:
        raise ValueError(f"Unsupported version: {version}. Use 'v1.1' or 'v2.1'")


def _load_msmarco_v1_1(split='dev'):
    """Load MS MARCO v1.1 - has passages embedded in each query example"""
    dataset = load_dataset("microsoft/ms_marco", "v1.1", split=split)
    
    queries = {}
    passages = {}
    qrels = defaultdict(set)
    
    print(f"Processing {len(dataset)} examples...")
    
    for example in tqdm(dataset, desc=f"Loading v1.1 {split}"):
        qid = example['query_id']
        query = example['query']
        
        queries[qid] = query
        
        # Get passages and relevance judgments
        for i, (passage_text, is_selected) in enumerate(zip(example['passages']['passage_text'], 
                                                              example['passages']['is_selected'])):
            # Create unique passage ID
            pid = f"{qid}_{i}"
            passages[pid] = passage_text
            
            if is_selected == 1:
                qrels[qid].add(pid)
    
    print(f"✓ Loaded {len(queries)} queries")
    print(f"✓ Loaded {len(passages)} passages")
    print(f"✓ Loaded {len(qrels)} queries with relevance judgments")
    total_relevant = sum(len(pids) for pids in qrels.values())
    print(f"✓ Total relevant documents: {total_relevant}")
    
    return queries, passages, qrels


def _load_msmarco_v2_1(split='test'):
    """Load MS MARCO v2.1 - has separate corpus, queries, and qrels"""
    
    # MS MARCO v2.1 structure:
    # - queries: separate dataset with query_id and query text
    # - corpus: separate dataset with passage_id and passage text  
    # - qrels: relevance judgments linking queries to passages
    
    print("Loading queries...")
    if split == 'dev':
        queries_split = 'dev1'  # MS MARCO v2.1 uses 'dev1' for dev queries
    else:
        queries_split = split
    
    try:
        queries_dataset = load_dataset("microsoft/ms_marco", "v2.1", split=f"queries.{queries_split}")
    except:
        # Try alternative split names
        try:
            queries_dataset = load_dataset("microsoft/ms_marco", "v2.1", split="queries.dev")
        except:
            queries_dataset = load_dataset("microsoft/ms_marco", "v2.1", split="queries")
    
    queries = {}
    for example in tqdm(queries_dataset, desc="Loading queries"):
        qid = str(example['query_id'])
        queries[qid] = example['query']
    
    print(f"✓ Loaded {len(queries)} queries")
    
    # For v2.1, we need to load the full corpus
    # This is HUGE (millions of passages), so we'll only load passages that are relevant
    print("Loading corpus (this may take a while for v2.1)...")
    
    try:
        corpus_dataset = load_dataset("microsoft/ms_marco", "v2.1", split="corpus")
    except:
        print("⚠️  Could not load full corpus. This is expected for v2.1.")
        print("    MS MARCO v2.1 corpus is too large to load directly.")
        print("    Please use local files with --queries_file, --collection_file, --qrels_file")
        raise RuntimeError("MS MARCO v2.1 corpus is too large. Use local TSV files instead.")
    
    passages = {}
    for example in tqdm(corpus_dataset, desc="Loading corpus"):
        pid = str(example['passage_id'])
        passages[pid] = example['passage']
    
    print(f"✓ Loaded {len(passages)} passages")
    
    # Load qrels (relevance judgments)
    print("Loading qrels...")
    try:
        qrels_dataset = load_dataset("microsoft/ms_marco", "v2.1", split=f"qrels.{queries_split}")
    except:
        try:
            qrels_dataset = load_dataset("microsoft/ms_marco", "v2.1", split="qrels.dev")
        except:
            qrels_dataset = load_dataset("microsoft/ms_marco", "v2.1", split="qrels")
    
    qrels = defaultdict(set)
    for example in tqdm(qrels_dataset, desc="Loading qrels"):
        qid = str(example['query_id'])
        pid = str(example['passage_id'])
        qrels[qid].add(pid)
    
    print(f"✓ Loaded {len(qrels)} queries with relevance judgments")
    total_relevant = sum(len(pids) for pids in qrels.values())
    print(f"✓ Total relevant documents: {total_relevant}")
    
    return queries, passages, qrels


def load_msmarco_from_files(queries_file, collection_file, qrels_file):
    """Load MS MARCO from local TSV files (alternative method)"""
    import csv
    
    print("\nLoading MS MARCO from local files...")
    
    # Load queries
    queries = {}
    with open(queries_file, 'r', encoding='utf-8') as f:
        reader = csv.reader(f, delimiter='\t')
        for row in reader:
            if len(row) >= 2:
                queries[row[0]] = row[1]
    print(f"✓ Loaded {len(queries)} queries")
    
    # Load collection
    passages = {}
    with open(collection_file, 'r', encoding='utf-8') as f:
        reader = csv.reader(f, delimiter='\t')
        for row in reader:
            if len(row) >= 2:
                passages[row[0]] = row[1]
    print(f"✓ Loaded {len(passages)} passages")
    
    # Load qrels
    qrels = defaultdict(set)
    with open(qrels_file, 'r', encoding='utf-8') as f:
        reader = csv.reader(f, delimiter='\t')
        for row in reader:
            if len(row) >= 4:
                qid = row[0]
                pid = row[2]
                relevance = int(row[3])
                if relevance > 0:
                    qrels[qid].add(pid)
    print(f"✓ Loaded {len(qrels)} queries with relevance judgments")
    
    return queries, passages, qrels


# ========================= Evaluation Metrics =========================

def compute_recall_at_k(ranked_lists: Dict, qrels: Dict, k: int) -> float:
    """Compute Recall@K"""
    total = 0.0
    num_q = 0
    for qid, retrieved in ranked_lists.items():
        if qid not in qrels or len(qrels[qid]) == 0:
            continue
        relevant = qrels[qid]
        retrieved_at_k = set(retrieved[:k])
        hits = len(relevant & retrieved_at_k)
        total += hits / len(relevant)
        num_q += 1
    return total / num_q if num_q > 0 else 0.0


def compute_mrr_at_k(ranked_lists: Dict, qrels: Dict, k: int) -> float:
    """Compute MRR@K (Mean Reciprocal Rank)"""
    rr_total = 0.0
    num_q = 0
    for qid, retrieved in ranked_lists.items():
        if qid not in qrels or len(qrels[qid]) == 0:
            continue
        relevant = qrels[qid]
        for rank, pid in enumerate(retrieved[:k], 1):
            if pid in relevant:
                rr_total += 1.0 / rank
                break
        num_q += 1
    return rr_total / num_q if num_q > 0 else 0.0


def evaluate_all(ranked_lists: Dict, qrels: Dict):
    """Evaluate and print all metrics"""
    print("\n" + "="*70)
    print("EVALUATION RESULTS")
    print("="*70)
    
    # Recall metrics
    print("\nRecall Metrics:")
    print("-" * 70)
    for k in [1, 5, 10, 50, 1000]:
        recall = compute_recall_at_k(ranked_lists, qrels, k)
        print(f"Recall@{k:4d} = {recall:.4f}")
    
    # MRR metrics
    print("\nMRR Metrics:")
    print("-" * 70)
    for k in [1, 5, 10]:
        mrr = compute_mrr_at_k(ranked_lists, qrels, k)
        print(f"MRR@{k:2d}    = {mrr:.4f}")
    
    print("="*70)


# ========================= BM25 Implementation =========================

class BM25:
    """Simple BM25 implementation for baseline comparison"""
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.idf = {}
        self.doc_lens = {}
        self.avg_doc_len = 0
        self.N = 0
        
    def tokenize(self, text: str) -> List[str]:
        """Simple whitespace tokenization and lowercasing"""
        return text.lower().split()
    
    def fit(self, passages: Dict[str, str]):
        """Build IDF scores from passage collection"""
        print("\nBuilding BM25 index...")
        self.N = len(passages)
        
        # Calculate document frequencies
        df = defaultdict(int)
        total_len = 0
        
        for pid, text in tqdm(passages.items(), desc="Indexing passages"):
            tokens = self.tokenize(text)
            self.doc_lens[pid] = len(tokens)
            total_len += len(tokens)
            
            # Count unique terms in document
            unique_tokens = set(tokens)
            for token in unique_tokens:
                df[token] += 1
        
        # Calculate average document length
        self.avg_doc_len = total_len / self.N if self.N > 0 else 0
        
        # Calculate IDF scores
        for term, freq in df.items():
            self.idf[term] = math.log((self.N - freq + 0.5) / (freq + 0.5) + 1)
        
        print(f"✓ Built BM25 index: {self.N} docs, avg_len={self.avg_doc_len:.1f}, vocab={len(self.idf)}")
    
    def score(self, query: str, doc: str) -> float:
        """Calculate BM25 score for a query-document pair"""
        query_tokens = self.tokenize(query)
        doc_tokens = self.tokenize(doc)
        
        # Count term frequencies in document
        doc_tf = defaultdict(int)
        for token in doc_tokens:
            doc_tf[token] += 1
        
        doc_len = len(doc_tokens)
        score = 0.0
        
        for token in query_tokens:
            if token not in doc_tf:
                continue
                
            tf = doc_tf[token]
            idf = self.idf.get(token, 0)
            
            # BM25 formula
            numerator = tf * (self.k1 + 1)
            denominator = tf + self.k1 * (1 - self.b + self.b * doc_len / self.avg_doc_len)
            score += idf * (numerator / denominator)
        
        return score
    
    def retrieve(self, queries: Dict[str, str], passages: Dict[str, str], k: int = 1000) -> Dict[str, List[str]]:
        """Retrieve top-k passages for each query"""
        print(f"\nRetrieving with BM25 (k1={self.k1}, b={self.b})...")
        
        ranked_lists = {}
        passage_ids = list(passages.keys())
        
        for qid, query in tqdm(queries.items(), desc="Searching with BM25"):
            scores = []
            for pid in passage_ids:
                score = self.score(query, passages[pid])
                scores.append((score, pid))
            
            # Sort by score descending
            scores.sort(reverse=True, key=lambda x: x[0])
            
            # Get top-k
            top_k = min(k, len(scores))
            ranked_lists[qid] = [pid for _, pid in scores[:top_k]]
        
        print(f"✓ Retrieved results for {len(ranked_lists)} queries")
        return ranked_lists


class BM25Optimized:
    """Memory-efficient BM25 using rank-bm25 library (fast, no GPU needed)"""
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        if not _HAS_RANK_BM25:
            raise ImportError(
                "rank-bm25 not installed. Install with: pip install rank-bm25"
            )
        self.k1 = k1
        self.b = b
        self.bm25 = None
        self.passage_ids = []
        
    def tokenize(self, text: str) -> List[str]:
        """Simple whitespace tokenization and lowercasing"""
        return text.lower().split()
    
    def fit(self, passages: Dict[str, str]):
        """Build BM25 index using rank-bm25"""
        print("\nBuilding BM25 index (rank-bm25)...")
        self.passage_ids = list(passages.keys())
        
        # Tokenize all documents
        print("Tokenizing documents...")
        tokenized_corpus = [
            self.tokenize(passages[pid]) 
            for pid in tqdm(self.passage_ids, desc="Tokenizing")
        ]
        
        # Build BM25 index
        print("Building BM25 index...")
        self.bm25 = BM25Okapi(tokenized_corpus, k1=self.k1, b=self.b)
        
        print(f"✓ Built BM25 index:")
        print(f"  - {len(self.passage_ids)} documents")
        print(f"  - Average doc length: {self.bm25.avgdl:.1f}")
        print(f"  - Memory-efficient inverted index")
    
    def retrieve(self, queries: Dict[str, str], passages: Dict[str, str], k: int = 1000) -> Dict[str, List[str]]:
        """Retrieve top-k passages for each query"""
        print(f"\nRetrieving with BM25 Optimized (rank-bm25, k1={self.k1}, b={self.b})...")
        
        ranked_lists = {}
        
        for qid, query in tqdm(queries.items(), desc="Searching with BM25 Optimized"):
            # Tokenize query
            query_tokens = self.tokenize(query)
            
            # Get scores for all documents
            scores = self.bm25.get_scores(query_tokens)
            
            # Get top-k indices
            top_k = min(k, len(self.passage_ids))
            top_k_indices = np.argsort(scores)[::-1][:top_k]
            
            # Convert to passage IDs
            ranked_pids = [self.passage_ids[idx] for idx in top_k_indices]
            ranked_lists[qid] = ranked_pids
        
        print(f"✓ Retrieved results for {len(ranked_lists)} queries")
        return ranked_lists


class BM25CUDA:
    """GPU-accelerated BM25 using PyTorch for vectorized operations
    
    WARNING: This uses dense matrices and can consume 50-100+ GB of VRAM for large collections!
    Use BM25Optimized (rank-bm25) instead for memory efficiency.
    """
    def __init__(self, k1: float = 1.5, b: float = 0.75, device: str = 'cuda'):
        self.k1 = k1
        self.b = b
        self.device = device
        self.vocab = {}  # token -> index
        self.idf = None  # tensor of IDF scores
        self.doc_term_freqs = None  # sparse matrix of term frequencies
        self.doc_lens = None  # tensor of document lengths
        self.avg_doc_len = 0
        self.N = 0
        self.passage_ids = []
        
        print("⚠️  WARNING: BM25CUDA uses dense matrices and may consume 50-100+ GB VRAM!")
        print("    Consider using --bm25_backend rank_bm25 for memory-efficient alternative.")
        
    def tokenize(self, text: str) -> List[str]:
        """Simple whitespace tokenization and lowercasing"""
        return text.lower().split()
    
    def fit(self, passages: Dict[str, str]):
        """Build BM25 index with GPU-friendly structures"""
        print(f"\nBuilding BM25 CUDA index on {self.device}...")
        self.N = len(passages)
        self.passage_ids = list(passages.keys())
        
        # Build vocabulary and document representations
        print("Building vocabulary...")
        all_tokens = []
        doc_tokens_list = []
        
        for pid in tqdm(self.passage_ids, desc="Tokenizing"):
            tokens = self.tokenize(passages[pid])
            doc_tokens_list.append(tokens)
            all_tokens.extend(set(tokens))
        
        # Create vocab: token -> index
        unique_tokens = sorted(set(all_tokens))
        self.vocab = {token: idx for idx, token in enumerate(unique_tokens)}
        vocab_size = len(self.vocab)
        
        print(f"✓ Vocabulary size: {vocab_size}")
        
        # Build term frequency matrix and calculate IDF
        print("Building term frequency matrix...")
        doc_lens = []
        df = torch.zeros(vocab_size, dtype=torch.float32)
        
        # Store as list of dicts for now (will convert to tensor later)
        doc_tf_dicts = []
        
        for tokens in tqdm(doc_tokens_list, desc="Processing documents"):
            doc_lens.append(len(tokens))
            
            # Count term frequencies
            tf_dict = defaultdict(int)
            unique_in_doc = set()
            
            for token in tokens:
                if token in self.vocab:
                    idx = self.vocab[token]
                    tf_dict[idx] += 1
                    unique_in_doc.add(idx)
            
            # Update document frequency
            for idx in unique_in_doc:
                df[idx] += 1
            
            doc_tf_dicts.append(tf_dict)
        
        # Calculate IDF scores
        print("Calculating IDF scores...")
        idf = torch.log((self.N - df + 0.5) / (df + 0.5) + 1)
        self.idf = idf.to(self.device)
        
        # Calculate average document length
        self.doc_lens = torch.tensor(doc_lens, dtype=torch.float32).to(self.device)
        self.avg_doc_len = self.doc_lens.mean().item()
        
        # Create sparse term frequency matrix
        print("Creating sparse TF matrix...")
        # We'll store as dense for simplicity (can optimize with sparse tensors if needed)
        tf_matrix = torch.zeros(self.N, vocab_size, dtype=torch.float32)
        
        for doc_idx, tf_dict in enumerate(tqdm(doc_tf_dicts, desc="Building TF matrix")):
            for term_idx, freq in tf_dict.items():
                tf_matrix[doc_idx, term_idx] = freq
        
        self.doc_term_freqs = tf_matrix.to(self.device)
        
        print(f"✓ Built BM25 CUDA index:")
        print(f"  - {self.N} documents")
        print(f"  - Vocab size: {vocab_size}")
        print(f"  - Avg doc length: {self.avg_doc_len:.1f}")
        print(f"  - Device: {self.device}")
        print(f"  - TF matrix shape: {self.doc_term_freqs.shape}")
        print(f"  - Memory: {self.doc_term_freqs.element_size() * self.doc_term_freqs.nelement() / 1e9:.2f} GB")
    
    @torch.no_grad()
    def score_batch(self, query_term_indices: List[int]) -> torch.Tensor:
        """Calculate BM25 scores for all documents given a query"""
        # Get term frequencies for query terms across all documents
        # query_term_indices: list of vocab indices
        
        if len(query_term_indices) == 0:
            return torch.zeros(self.N, dtype=torch.float32, device=self.device)
        
        # Get TF for query terms: (N, len(query_terms))
        query_tfs = self.doc_term_freqs[:, query_term_indices]
        
        # Get IDF for query terms: (len(query_terms),)
        query_idfs = self.idf[query_term_indices]
        
        # Calculate BM25 score
        # BM25 = sum over terms: idf * (tf * (k1+1)) / (tf + k1 * (1-b + b*dl/avgdl))
        
        # Denominator: tf + k1 * (1 - b + b * dl / avgdl)
        # dl is doc_lens (N,), need to broadcast
        doc_lens_expanded = self.doc_lens.unsqueeze(1)  # (N, 1)
        
        # k1 * (1 - b + b * dl / avgdl)
        length_norm = self.k1 * (1 - self.b + self.b * doc_lens_expanded / self.avg_doc_len)
        
        # Denominator for each term
        denominators = query_tfs + length_norm  # (N, len(query_terms))
        
        # Numerator
        numerators = query_tfs * (self.k1 + 1)  # (N, len(query_terms))
        
        # BM25 component for each term
        term_scores = (numerators / denominators) * query_idfs.unsqueeze(0)  # (N, len(query_terms))
        
        # Sum over terms
        scores = term_scores.sum(dim=1)  # (N,)
        
        return scores
    
    def retrieve(self, queries: Dict[str, str], passages: Dict[str, str], k: int = 1000) -> Dict[str, List[str]]:
        """Retrieve top-k passages for each query using GPU acceleration"""
        print(f"\nRetrieving with BM25 CUDA (k1={self.k1}, b={self.b}, device={self.device})...")
        
        ranked_lists = {}
        
        for qid, query in tqdm(queries.items(), desc="Searching with BM25 CUDA"):
            # Tokenize query and convert to vocab indices
            query_tokens = self.tokenize(query)
            query_term_indices = [self.vocab[token] for token in query_tokens if token in self.vocab]
            
            if len(query_term_indices) == 0:
                # No terms in vocabulary, return empty
                ranked_lists[qid] = []
                continue
            
            # Calculate scores for all documents
            scores = self.score_batch(query_term_indices)
            
            # Get top-k
            top_k = min(k, len(self.passage_ids))
            top_k_scores, top_k_indices = torch.topk(scores, k=top_k)
            
            # Convert to passage IDs
            top_k_indices = top_k_indices.cpu().tolist()
            ranked_pids = [self.passage_ids[idx] for idx in top_k_indices]
            ranked_lists[qid] = ranked_pids
        
        print(f"✓ Retrieved results for {len(ranked_lists)} queries")
        return ranked_lists


# ========================= Retrieval =========================

@torch.no_grad()
def retrieve_with_longmatrix(model, queries: Dict, passages: Dict, 
                              tokenizer, device: str, 
                              max_len: int = 128, 
                              batch_size: int = 32,
                              k: int = 1000,
                              late_interaction: bool = False,
                              topk_q: int = 4,
                              topk_d: int = 1):
    """
    Retrieve documents using LongMatrix model
    
    Args:
        model: LongMatrixModel
        queries: dict {qid: query_text}
        passages: dict {pid: passage_text}
        tokenizer: tokenizer
        device: 'cuda' or 'cpu'
        max_len: max sequence length
        batch_size: batch size for encoding
        k: number of top documents to retrieve
        late_interaction: use late interaction (ColBERT-like)
        topk_q: number of tokens for query (if late_interaction)
        topk_d: number of tokens for doc (if late_interaction)
    
    Returns:
        ranked_lists: dict {qid: [pid1, pid2, ...]}
    """
    model.eval()
    
    print("\n" + "="*70)
    print("LONGMATRIX RETRIEVAL")
    print("="*70)
    print(f"Late Interaction: {late_interaction}")
    if late_interaction:
        print(f"Query tokens (Kq): {topk_q}")
        print(f"Doc tokens (Kd): {topk_d}")
    
    KQ = topk_q if late_interaction else 1
    KD = topk_d if late_interaction else 1
    
    # Warning for very large topk values
    if KD > 50:
        print(f"⚠️  WARNING: topk_d={KD} is very large!")
        print(f"   This will be VERY slow and memory-intensive.")
        print(f"   Consider using topk_d=1-4 for practical retrieval.")
    if KQ > 50:
        print(f"⚠️  WARNING: topk_q={KQ} is very large!")
        print(f"   Consider using topk_q=4-8 for good balance.")
    
    # Encode all passages
    print(f"\nEncoding {len(passages)} passages...")
    passage_ids = list(passages.keys())
    passage_texts = [passages[pid] for pid in passage_ids]
    
    all_passage_embeddings = []
    max_kd_actual = 0  # Track the actual max number of tokens
    
    for i in tqdm(range(0, len(passage_texts), batch_size), desc="Encoding passages"):
        batch = passage_texts[i:i+batch_size]
        tok = tokenizer(batch, padding=True, truncation=True, 
                       max_length=max_len, return_tensors='pt')
        tok = {k: v.to(device) for k, v in tok.items()}
        
        out = model.encode(tok, topk_tokens=KD)
        z = out['z']  # (B, Kd, m) or (B, m)
        
        if z.dim() == 2:
            z = z.unsqueeze(1)  # (B, 1, m)
        
        # Track actual max tokens (may be less than KD for short docs)
        max_kd_actual = max(max_kd_actual, z.size(1))
        
        all_passage_embeddings.append(z.cpu())
    
    # Pad all embeddings to the same size (max_kd_actual)
    # This handles cases where documents have fewer than KD tokens
    padded_embeddings = []
    for z_batch in all_passage_embeddings:
        B, K_actual, M = z_batch.shape
        if K_actual < max_kd_actual:
            # Pad with zeros to match max_kd_actual
            padding = torch.zeros(B, max_kd_actual - K_actual, M, dtype=z_batch.dtype)
            z_batch = torch.cat([z_batch, padding], dim=1)
        padded_embeddings.append(z_batch)
    
    passage_embeddings = torch.cat(padded_embeddings, dim=0)  # (num_passages, max_kd_actual, m)
    passage_embeddings = passage_embeddings.to(device)
    
    # Update KD to actual value for consistency
    KD_actual = max_kd_actual
    if KD_actual < KD:
        print(f"⚠️  Note: Requested topk_d={KD}, but max actual tokens in docs is {KD_actual}")
    
    print(f"✓ Passage embeddings shape: {passage_embeddings.shape}")
    
    # Encode queries and retrieve
    print(f"\nRetrieving for {len(queries)} queries...")
    ranked_lists = {}
    
    query_ids = list(queries.keys())
    query_texts = [queries[qid] for qid in query_ids]
    
    for i in tqdm(range(0, len(query_texts), batch_size), desc="Searching"):
        batch_qids = query_ids[i:i+batch_size]
        batch_queries = query_texts[i:i+batch_size]
        
        tok = tokenizer(batch_queries, padding=True, truncation=True,
                       max_length=max_len, return_tensors='pt')
        tok = {k: v.to(device) for k, v in tok.items()}
        
        out = model.encode(tok, topk_tokens=KQ)
        zq = out['z']  # (B, Kq, m) or (B, m)
        
        if zq.dim() == 2:
            zq = zq.unsqueeze(1)  # (B, 1, m)
        
        # Pad queries to match KD_actual if needed (for late interaction consistency)
        # This is only needed if we have very short queries
        B_q, K_q_actual, M_q = zq.shape
        # No padding needed for queries - they can have different K from docs
        
        # Compute similarities
        if late_interaction:
            sims = _batch_late_scores(zq, passage_embeddings)  # (B, num_passages)
        else:
            zq_flat = zq.squeeze(1)  # (B, m)
            zd_flat = passage_embeddings.squeeze(1)  # (num_passages, m)
            sims = zq_flat @ zd_flat.t()  # (B, num_passages)
        
        # Get top-k for each query
        top_k = min(k, len(passage_ids))
        top_k_scores, top_k_indices = torch.topk(sims, k=top_k, dim=1)
        
        # Convert to ranked lists
        for j, qid in enumerate(batch_qids):
            indices = top_k_indices[j].cpu().tolist()
            ranked_pids = [passage_ids[idx] for idx in indices]
            ranked_lists[qid] = ranked_pids
    
    print(f"✓ Retrieved results for {len(ranked_lists)} queries")
    
    return ranked_lists


# ========================= Main =========================

def main():
    parser = argparse.ArgumentParser(description='Evaluate LongMatrix on MS MARCO')
    parser.add_argument('--checkpoint', type=str, required=False, default=None,
                        help='Path to model checkpoint (.pt file) - not required if --use_bm25 is set')
    parser.add_argument('--dataset', type=str, default='msmarco',
                        choices=['msmarco'],
                        help='Dataset to evaluate on')
    parser.add_argument('--split', type=str, default='dev',
                        choices=['dev', 'test'],
                        help='Dataset split to use')
    parser.add_argument('--version', type=str, default='v2.1',
                        choices=['v1.1', 'v2.1'],
                        help='MS MARCO version (v1.1 or v2.1)')
    
    # Alternative: load from local files
    parser.add_argument('--queries_file', type=str, default=None,
                        help='Path to queries TSV file (alternative to HF dataset)')
    parser.add_argument('--collection_file', type=str, default=None,
                        help='Path to collection TSV file (alternative to HF dataset)')
    parser.add_argument('--qrels_file', type=str, default=None,
                        help='Path to qrels TSV file (alternative to HF dataset)')
    
    parser.add_argument('--max_len', type=int, default=128,
                        help='Max sequence length')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for encoding')
    parser.add_argument('--k', type=int, default=1000,
                        help='Number of top documents to retrieve')
    parser.add_argument('--device', type=str, default='cuda',
                        choices=['cuda', 'cpu'],
                        help='Device to use')
    
    # Late interaction options
    parser.add_argument('--late_interaction', action='store_true',
                        help='Use late interaction (ColBERT-like)')
    parser.add_argument('--topk_q', type=int, default=4,
                        help='Number of query tokens for late interaction')
    parser.add_argument('--topk_d', type=int, default=1,
                        help='Number of doc tokens for late interaction')
    
    # BM25 baseline options
    parser.add_argument('--use_bm25', action='store_true',
                        help='Use BM25 for retrieval instead of LongMatrix')
    parser.add_argument('--bm25_k1', type=float, default=1.5,
                        help='BM25 k1 parameter (term saturation)')
    parser.add_argument('--bm25_b', type=float, default=0.75,
                        help='BM25 b parameter (length normalization)')
    parser.add_argument('--bm25_backend', type=str, default='auto',
                        choices=['auto', 'rank_bm25', 'cuda', 'cpu'],
                        help='BM25 backend: auto (rank_bm25 if available), rank_bm25 (memory-efficient), cuda (GPU, high VRAM), cpu (slow)')
    parser.add_argument('--compare_with_bm25', action='store_true',
                        help='Compare LongMatrix results with BM25 baseline')
    
    args = parser.parse_args()
    
    # Validate arguments
    if not args.use_bm25 and not args.checkpoint:
        parser.error("--checkpoint is required when not using --use_bm25")
    
    # Check device
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("⚠️  CUDA not available, falling back to CPU")
        args.device = 'cpu'
    
    print("="*70)
    print("LONGMATRIX MODEL EVALUATION ON MS MARCO")
    print("="*70)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Dataset: {args.dataset}")
    print(f"Split: {args.split}")
    print(f"Device: {args.device}")
    print(f"Batch size: {args.batch_size}")
    print(f"Top-K: {args.k}")
    if args.use_bm25 or args.compare_with_bm25:
        print(f"BM25 Backend: {args.bm25_backend}")
        print(f"BM25 Parameters: k1={args.bm25_k1}, b={args.bm25_b}")
    print("="*70)
    
    # Print available BM25 backends
    if args.use_bm25 or args.compare_with_bm25:
        print("\n📊 BM25 Backend Availability:")
        print(f"  - rank_bm25: {'✓ Available (RECOMMENDED)' if _HAS_RANK_BM25 else '✗ Not installed (pip install rank-bm25)'}")
        print(f"  - CUDA: {'✓ Available (High VRAM!)' if torch.cuda.is_available() else '✗ CUDA not available'}")
        print(f"  - CPU: ✓ Available (Slow for large collections)")
        print()
    
    # Load model (skip if using BM25 only)
    model = None
    tokenizer = None
    late_interaction = args.late_interaction
    topk_q = args.topk_q
    topk_d = args.topk_d
    
    if not args.use_bm25:
        # Load checkpoint
        print("\nLoading model checkpoint...")
        if not os.path.exists(args.checkpoint):
            raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")
        
        ckpt = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
        
        # Get model args from checkpoint
        model_args = ckpt.get('args', {})
        if isinstance(model_args, dict):
            d_lex_emb = model_args.get('d_lex_emb', 128)
            d_lex = model_args.get('d_lex', 128)
            m_teacher = model_args.get('m_teacher', 1024)
            rank = model_args.get('rank', 64)
            heads = model_args.get('heads', 4)
            tokenizer_name = model_args.get('tokenizer', 'BAAI/bge-m3')
            
            # Late interaction settings from checkpoint
            late_interaction = model_args.get('late_interaction', False)
            topk_q = model_args.get('topk_q', 4)
            topk_d = model_args.get('topk_d', 1)
            
            # Override with command line args if provided
            if args.late_interaction:
                late_interaction = True
                topk_q = args.topk_q
                topk_d = args.topk_d
        else:
            # Default values
            d_lex_emb = 128
            d_lex = 128
            m_teacher = 1024
            rank = 64
            heads = 4
            tokenizer_name = 'BAAI/bge-m3'
            late_interaction = args.late_interaction
            topk_q = args.topk_q
            topk_d = args.topk_d
        
        print(f"Model config: d_lex_emb={d_lex_emb}, d_lex={d_lex}, m={m_teacher}, rank={rank}, heads={heads}")
        print(f"Tokenizer: {tokenizer_name}")
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)
        
        # Create model
        model = LongMatrixModel(
            tokenizer=tokenizer,
            d_lex_emb=d_lex_emb,
            d_lex=d_lex,
            m_teacher=m_teacher,
            rank=rank,
            attn_backend='sdpa',
            use_ckpt=False,
            heads=heads
        ).to(args.device)
        
        # Load weights
        model.load_state_dict(ckpt['model'])
        print("✓ Model loaded successfully")
    
    # Load dataset
    if args.queries_file and args.collection_file and args.qrels_file:
        # Load from local files
        queries, passages, qrels = load_msmarco_from_files(
            args.queries_file, args.collection_file, args.qrels_file
        )
    else:
        # Load from Hugging Face
        queries, passages, qrels = load_msmarco_dataset(args.split, version=args.version)
    
    # Helper function to create BM25 instance
    def create_bm25(backend='auto'):
        """Create BM25 instance based on backend choice"""
        if backend == 'auto':
            # Auto-select: prefer rank_bm25 if available, otherwise CPU
            if _HAS_RANK_BM25:
                print("Using rank-bm25 (memory-efficient, recommended)")
                return BM25Optimized(k1=args.bm25_k1, b=args.bm25_b)
            else:
                print("Using CPU BM25 (may be slow)")
                print("💡 TIP: Install rank-bm25 for better performance: pip install rank-bm25")
                return BM25(k1=args.bm25_k1, b=args.bm25_b)
        
        elif backend == 'rank_bm25':
            if not _HAS_RANK_BM25:
                raise ImportError("rank-bm25 not installed. Install with: pip install rank-bm25")
            print("Using rank-bm25 (memory-efficient)")
            return BM25Optimized(k1=args.bm25_k1, b=args.bm25_b)
        
        elif backend == 'cuda':
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA not available")
            print("⚠️  Using BM25 CUDA (WARNING: High VRAM usage!)")
            return BM25CUDA(k1=args.bm25_k1, b=args.bm25_b, device=args.device)
        
        elif backend == 'cpu':
            print("Using CPU BM25 (may be slow)")
            return BM25(k1=args.bm25_k1, b=args.bm25_b)
        
        else:
            raise ValueError(f"Unknown backend: {backend}")
    
    # Perform retrieval
    if args.use_bm25:
        # Use BM25 only
        print("\n" + "="*70)
        print("USING BM25 FOR RETRIEVAL")
        print("="*70)
        
        bm25 = create_bm25(args.bm25_backend)
        
        bm25.fit(passages)
        ranked_lists = bm25.retrieve(queries, passages, k=args.k)
        
        # Evaluate BM25
        print("\n" + "="*70)
        print("BM25 RESULTS")
        print("="*70)
        evaluate_all(ranked_lists, qrels)
        
    elif args.compare_with_bm25:
        # Compare LongMatrix with BM25
        print("\n" + "="*70)
        print("COMPARING LONGMATRIX WITH BM25")
        print("="*70)
        
        # Get BM25 results
        print("\n[1/2] Running BM25 baseline...")
        bm25 = create_bm25(args.bm25_backend)
        bm25.fit(passages)
        bm25_ranked_lists = bm25.retrieve(queries, passages, k=args.k)
        
        # Get LongMatrix results
        print("\n[2/2] Running LongMatrix...")
        ranked_lists = retrieve_with_longmatrix(
            model=model,
            queries=queries,
            passages=passages,
            tokenizer=tokenizer,
            device=args.device,
            max_len=args.max_len,
            batch_size=args.batch_size,
            k=args.k,
            late_interaction=late_interaction,
            topk_q=topk_q,
            topk_d=topk_d
        )
        
        # Evaluate both
        print("\n" + "="*70)
        print("BM25 BASELINE RESULTS")
        print("="*70)
        evaluate_all(bm25_ranked_lists, qrels)
        
        print("\n" + "="*70)
        print("LONGMATRIX RESULTS")
        print("="*70)
        evaluate_all(ranked_lists, qrels)
        
    else:
        # Use LongMatrix only
        ranked_lists = retrieve_with_longmatrix(
            model=model,
            queries=queries,
            passages=passages,
            tokenizer=tokenizer,
            device=args.device,
            max_len=args.max_len,
            batch_size=args.batch_size,
            k=args.k,
            late_interaction=late_interaction,
            topk_q=topk_q,
            topk_d=topk_d
        )
        
        # Evaluate
        evaluate_all(ranked_lists, qrels)
    
    # Print sample results
    print("\n" + "="*70)
    print("SAMPLE RESULTS (first 3 queries)")
    print("="*70)
    for i, (qid, pids) in enumerate(list(ranked_lists.items())[:3]):
        print(f"\nQuery {qid}: {queries[qid]}")
        print(f"Relevant: {qrels[qid]}")
        print(f"Retrieved (top 5):")
        for rank, pid in enumerate(pids[:5], 1):
            marker = "✓" if pid in qrels[qid] else "✗"
            passage_text = passages[pid][:100] if len(passages[pid]) > 100 else passages[pid]
            print(f"  {rank}. [{marker}] Doc {pid}: {passage_text}...")
    
    print("\n" + "="*70)
    print("EVALUATION COMPLETE")
    print("="*70)


if __name__ == "__main__":
    main()
