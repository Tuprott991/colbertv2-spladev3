# MS MARCO Dataset Testing Guide

## Overview

Script `colbertv2.py` hiện hỗ trợ 2 chế độ:
1. **Local Test Data** (mặc định): Dataset nhỏ cục bộ (9,853 queries)
2. **MS MARCO Dev** (`--msmarco`): Dataset benchmark chuẩn (~8.8M documents, 6,980 queries)

## Usage

### 1. Chạy với Local Test Data (mặc định)
```bash
python colbertv2.py
```

### 2. Chạy với MS MARCO Dataset
```bash
python colbertv2.py --msmarco
```

### 3. Tùy chọn nâng cao
```bash
# Chỉ định base directory
python colbertv2.py --msmarco --base-dir "D:/data"

# Sử dụng checkpoint cục bộ
python colbertv2.py --msmarco --checkpoint "path/to/colbert.checkpoint"

# Hiển thị help
python colbertv2.py --help
```

## MS MARCO Dataset Details

Khi sử dụng `--msmarco`, script sẽ tự động:

1. **Download** (nếu chưa có):
   - `collection.tar.gz` (~2.9GB) → 8.8M passages
   - `queries.dev.small.tar.gz` (~300KB) → 6,980 queries  
   - `qrels.dev.small.tsv` (~140KB) → relevance judgments

2. **Extract** vào thư mục `msmarco_data/`

3. **Tạo index** với tên `msmarco_colbertv2_index`

4. **Evaluate** trên dev set

## Expected Performance on MS MARCO

MS MARCO là dataset **khó hơn nhiều** so với local test:
- **Recall@10**: ~80-85% (vs 99% trên toy data)
- **MRR@10**: ~35-40% (vs 97% trên toy data)

Lý do:
- 8.8M documents (vs 9,853)
- Hard negatives (các documents gần giống nhau)
- Natural language queries phức tạp hơn

## Storage Requirements

- **Downloaded files**: ~3.2GB
- **Extracted files**: ~4.5GB
- **Index**: ~30-40GB (tùy thuộc vào compression settings)
- **Total**: ~50GB

## Time Estimates

Trên GPU (V100/A100):
- **Download**: 5-15 phút (tùy internet)
- **Indexing**: 20-30 phút
- **Search (6,980 queries)**: 2-5 phút

Trên CPU: chậm hơn 10-20x

## Troubleshooting

### Download fails
```bash
# Thử download thủ công
wget https://msmarco.blob.core.windows.net/msmarcoranking/collection.tar.gz
```

### Index too large
```python
# Trong colbertv2.py, thêm compression config:
config = ColBERTConfig(
    root=root,
    nbits=2,  # Giảm từ 4 xuống 2 để tiết kiệm ~50% dung lượng
)
```

### Out of memory
- Giảm batch size trong ColBERTConfig
- Sử dụng GPU có nhiều RAM hơn
- Index theo chunks (advanced)

## File Locations

```
msmarco_data/
├── collection.tar.gz          (downloaded)
├── collection.tsv              (extracted, ~20GB)
├── queries.dev.small.tar.gz   (downloaded)  
├── queries.dev.small.tsv       (extracted)
└── qrels.dev.small.tsv         (downloaded/extracted)

experiments/
└── default/
    └── indexes/
        └── msmarco_colbertv2_index/  (~30-40GB)
```

## Next Steps

Sau khi test xong MS MARCO, bạn có thể:
1. Test trên TREC DL (dataset khó hơn nữa)
2. Fine-tune ColBERT trên domain-specific data
3. Thử các compression settings khác nhau
4. So sánh với BM25 baseline
