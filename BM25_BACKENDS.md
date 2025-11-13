# BM25 Backend Options for eval_longmatrix_msmarco.py

## TL;DR - Recommended Option ⭐

```bash
pip install rank-bm25
python eval_longmatrix_msmarco.py --use_bm25 --bm25_backend rank_bm25
```

## Backend Comparison

| Backend | Speed | Memory | Installation | Use Case |
|---------|-------|--------|--------------|----------|
| **rank_bm25** ⭐ | Fast | Low (~1-2 GB) | `pip install rank-bm25` | **RECOMMENDED for most users** |
| **cpu** | Slow | Low | Built-in | Fallback, no dependencies |
| **cuda** | Very Fast | **VERY HIGH (50-100+ GB VRAM!)** | Built-in | ❌ Not recommended - VRAM hungry |

## Installation

### Option 1: rank-bm25 (Recommended)
```bash
pip install rank-bm25
```

### Option 2: Use built-in implementations
No installation needed, but slower performance.

## Usage Examples

### 1. Evaluate BM25 only (auto-select best backend)
```bash
python eval_longmatrix_msmarco.py --use_bm25 --dataset msmarco --split dev
```

### 2. Explicitly use rank-bm25 (recommended)
```bash
python eval_longmatrix_msmarco.py --use_bm25 --bm25_backend rank_bm25
```

### 3. Compare LongMatrix with BM25
```bash
python eval_longmatrix_msmarco.py \
    --checkpoint runs/longmatrix/epoch1.pt \
    --compare_with_bm25 \
    --bm25_backend rank_bm25
```

### 4. Use CPU backend (no extra dependencies, but slow)
```bash
python eval_longmatrix_msmarco.py --use_bm25 --bm25_backend cpu
```

### 5. Use CUDA backend (⚠️ WARNING: High VRAM!)
```bash
# Only use if you have 100+ GB VRAM!
python eval_longmatrix_msmarco.py --use_bm25 --bm25_backend cuda --device cuda
```

### 6. Tune BM25 parameters
```bash
python eval_longmatrix_msmarco.py \
    --use_bm25 \
    --bm25_backend rank_bm25 \
    --bm25_k1 1.2 \
    --bm25_b 0.8
```

## Performance Comparison (MS MARCO Dev, ~6,980 queries)

| Backend | Indexing Time | Search Time | Memory Usage | Total Time |
|---------|---------------|-------------|--------------|------------|
| rank_bm25 | ~30s | ~2-3 min | 1-2 GB RAM | **~3 min** ⭐ |
| cpu | ~30s | ~20-30 min | 1-2 GB RAM | ~25 min |
| cuda | ~5 min | ~10-20s | **70-100 GB VRAM** | ~6 min ⚠️ |

## Backend Details

### 1. rank_bm25 (Recommended ⭐)

**Pros:**
- ✅ Fast inverted index
- ✅ Memory-efficient (~1-2 GB for MS MARCO)
- ✅ No GPU needed
- ✅ Production-ready implementation
- ✅ Easy to install

**Cons:**
- ❌ Requires pip install

**Best for:**
- Most users
- Production deployments
- Limited GPU memory
- Quick experiments

### 2. CPU (Built-in)

**Pros:**
- ✅ No dependencies
- ✅ Low memory usage
- ✅ Works everywhere

**Cons:**
- ❌ Very slow for large collections
- ❌ O(n) per query (brute force)

**Best for:**
- Small collections (<10k docs)
- When you can't install dependencies
- Fallback option

### 3. CUDA (⚠️ Not Recommended)

**Pros:**
- ✅ Very fast queries (~10ms per query)
- ✅ Fully vectorized on GPU

**Cons:**
- ❌ **EXTREMELY high VRAM usage** (50-100+ GB)
- ❌ Dense matrix storage
- ❌ Slow indexing
- ❌ Not practical for most setups

**Best for:**
- ❌ Generally **NOT recommended**
- Research with high-end GPUs (A100 80GB, H100)
- When you need absolute fastest query speed AND have massive VRAM

## BM25 Parameters

- **k1** (default: 1.5): Controls term frequency saturation
  - Higher k1 → more weight to repeated terms
  - Typical range: 1.2 - 2.0
  
- **b** (default: 0.75): Controls length normalization
  - Higher b → more penalty for long documents
  - Typical range: 0.5 - 0.9

## Expected Results (MS MARCO Dev)

Typical BM25 baseline performance:
- **Recall@1**: ~0.08-0.12
- **Recall@10**: ~0.30-0.40
- **Recall@100**: ~0.60-0.70
- **Recall@1000**: ~0.85-0.95
- **MRR@10**: ~0.18-0.22

## Troubleshooting

### "rank-bm25 not installed"
```bash
pip install rank-bm25
```

### "CUDA out of memory" with BM25 CUDA
Switch to rank_bm25:
```bash
python eval_longmatrix_msmarco.py --use_bm25 --bm25_backend rank_bm25
```

### Slow performance with CPU backend
Install rank-bm25 for 10-20x speedup:
```bash
pip install rank-bm25
```

## Recommendations by Use Case

### Quick Experiment / Testing
```bash
pip install rank-bm25
python eval_longmatrix_msmarco.py --use_bm25 --bm25_backend rank_bm25
```

### Comparing with LongMatrix
```bash
python eval_longmatrix_msmarco.py \
    --checkpoint runs/longmatrix/epoch1.pt \
    --compare_with_bm25 \
    --bm25_backend rank_bm25
```

### No Dependencies Available
```bash
python eval_longmatrix_msmarco.py --use_bm25 --bm25_backend cpu
# Be patient - this will take ~20-30 minutes for MS MARCO dev
```

### Production Deployment
Use Elasticsearch, Lucene, or Pyserini for production-grade BM25.
