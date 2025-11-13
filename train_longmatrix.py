import os, sys, math, json, random, argparse, time, gc
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from transformers import AutoTokenizer, get_cosine_schedule_with_warmup, get_linear_schedule_with_warmup
import numpy as np
import yaml  # <-- để lưu YAML config local

# ---- Optional experiment tracker ----
try:
    import wandb
    _HAS_WANDB = True
except Exception:
    _HAS_WANDB = False

# ---- Optional FAISS for indexing ----
try:
    import faiss
    _HAS_FAISS = True
except Exception:
    _HAS_FAISS = False

try:
    from sentence_transformers import SentenceTransformer
    _HAS_ST = True
except Exception:
    _HAS_ST = False

# ------------------------- Data utils -------------------------

def read_tsv(path: str, max_rows: Optional[int] = None) -> List[Tuple[str, str, List[str]]]:
    rows: List[Tuple[str, str, List[str]]] = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = [p.strip() for p in line.rstrip('\n').split('\t')]
            if len(parts) >= 2:
                q, p = parts[0], parts[1]
                negs = parts[2:] if len(parts) > 2 else []
                rows.append((q, p, negs))
                if max_rows and len(rows) >= max_rows:
                    break
    return rows

class TripletDataset(Dataset):
    def __init__(self, triples: List[Tuple[str, str, List[str]]], neg_per_sample: int = 7):
        self.data = triples
        self.k = neg_per_sample
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        q, p, negs = self.data[idx]
        if len(negs) == 0:
            cand = []
        elif len(negs) >= self.k:
            cand = random.sample(negs, self.k)
        else:
            cand = list(negs) + random.choices(negs, k=self.k - len(negs))
        return {'q': q, 'p': p, 'negs': cand}

@dataclass
class Collator:
    tokenizer: AutoTokenizer
    max_len: int
    def __call__(self, batch):
        qs = [x['q'] for x in batch]
        ps = [x['p'] for x in batch]
        neg_lists = [x['negs'] for x in batch]

        docs, doc_spans = [], []
        for p, negs in zip(ps, neg_lists):
            start = len(docs)
            docs.append(p)
            docs.extend(negs)
            doc_spans.append((start, 1 + len(negs)))

        tok_q = self.tokenizer(qs, padding=True, truncation=True,
                               max_length=self.max_len, return_tensors='pt')
        tok_d = self.tokenizer(docs, padding=True, truncation=True,
                               max_length=self.max_len, return_tensors='pt')
        # trả luôn raw texts để distill không phải batch_decode
        return tok_q, tok_d, doc_spans, qs, docs

# ------------------------- Model -------------------------

class LexicalEncoder(nn.Module):
    """
    Multi-head + fused QKV attention with SDPA (FlashAttention/Triton) route.
    """
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

        self._sdpa_broken_warned = False  # warn once then fallback to matmul

    def _shape_qkv(self, x):
        # x: (B,T,D) -> q,k,v each (B,H,T,dh)
        B, T, D = x.shape
        qkv = self.qkv(x)                        # (B,T,3D)
        q, k, v = qkv.chunk(3, dim=-1)           # (B,T,D) each
        def reshape(t):
            return t.view(B, T, self.heads, self.head_dim).transpose(1, 2).contiguous()
        return reshape(q), reshape(k), reshape(v)

    def _attend_sdpa(self, Q, K, V, attention_mask):
        # Q,K,V: (B,H,T,dh) ; mask: (B,1,1,T) bool (True = keep)
        attn_mask = attention_mask.bool().unsqueeze(1).unsqueeze(2)
        return F.scaled_dot_product_attention(
            Q, K, V,
            attn_mask=attn_mask,
            dropout_p=self.attn_dropout.p if self.training else 0.0,
            is_causal=False
        )  # (B,H,T,dh)

    def _attend_matmul(self, Q, K, V, attention_mask):
        # Q,K,V: (B,H,T,dh)
        B,H,T,dh = Q.shape
        scores = torch.matmul(Q, K.transpose(-1, -2)).float() / math.sqrt(dh)  # (B,H,T,T)
        mask = attention_mask.unsqueeze(1).unsqueeze(2).to(dtype=torch.bool)   # (B,1,1,T)
        scores = scores.masked_fill(~mask, float('-inf'))
        attn = torch.softmax(scores, dim=-1).to(Q.dtype)
        attn = self.attn_dropout(attn)
        ctx = torch.matmul(attn, V)                                            # (B,H,T,dh)
        return ctx, attn

    def _attend(self, Q, K, V, attention_mask):
        if self.attn_backend == 'sdpa':
            try:
                x_ctx = self._attend_sdpa(Q, K, V, attention_mask)
                return x_ctx, None
            except Exception as e:
                if not self._sdpa_broken_warned:
                    print(f'[warn] SDPA failed ("{e}"). Falling back to matmul for stability.', file=sys.stderr)
                    self._sdpa_broken_warned = True
                x_ctx, attn = self._attend_matmul(Q, K, V, attention_mask)
                return x_ctx, attn
        else:
            x_ctx, attn = self._attend_matmul(Q, K, V, attention_mask)
            return x_ctx, attn

    def forward_block(self, input_ids, attention_mask, topk_tokens: int = 1):
        """
        Nếu topk_tokens<=1: trả pooling 1 vector.
        Nếu topk_tokens>1: trả ra (B, K, d_lex) gồm K token vector nổi bật (ColBERT-lite).
        """
        x = self.emb(input_ids)
        x = self.ln(x)

        Q, K, V = self._shape_qkv(x)                      # (B,H,T,dh)

        # safe mask: nếu hàng toàn 0 → bật vị trí đầu
        mask_safe = attention_mask
        lens = mask_safe.sum(dim=-1)
        need_fix = (lens == 0)
        if need_fix.any():
            mask_safe = mask_safe.clone()
            mask_safe[need_fix, 0] = 1

        x_ctx, _ = self._attend(Q, K, V, mask_safe)       # (B,H,T,dh)

        # merge heads
        B,H,T,dh = x_ctx.shape
        x_ctx = x_ctx.transpose(1, 2).contiguous().view(B, T, H * dh)  # (B,T,D)
        x_ctx = self.o_proj(x_ctx)                         # (B,T,D)

        # attention pooling ổn định số
        alpha_logits = self.pool_vec(x_ctx).squeeze(-1).float()           # (B,T)
        alpha_logits = alpha_logits.masked_fill(mask_safe == 0, -1e9)     # tránh -inf
        a = torch.softmax(alpha_logits, dim=-1).to(x_ctx.dtype)
        if torch.isnan(a).any():
            bad = torch.isnan(a).any(dim=1)
            if bad.any():
                denom = mask_safe[bad].sum(dim=1, keepdim=True).clamp_min(1)
                a[bad] = (mask_safe[bad] / denom).to(a.dtype)
        self.last_attention = a

        if topk_tokens is None or topk_tokens <= 1:
            h = torch.sum(a.unsqueeze(-1) * x_ctx, dim=1)                 # (B, D)
            h = self.proj_h(h)                                            # (B, d_lex)
            return h, a
        else:
            K_sel = min(int(topk_tokens), x_ctx.size(1))
            topk_idx = torch.topk(a, k=K_sel, dim=-1).indices             # (B, K_sel)
            # gather token vectors
            topk_vec = x_ctx.gather(1, topk_idx.unsqueeze(-1).expand(B, K_sel, x_ctx.size(-1)))  # (B, K_sel, D)
            topk_vec = self.proj_h(topk_vec)                               # (B, K_sel, d_lex)
            return topk_vec, a

    def forward(self, input_ids, attention_mask, topk_tokens: int = 1):
        if self.use_ckpt and self.training:
            def _fn(ids, mask, k):
                return self.forward_block(ids, mask, k)[0]
            h = torch.utils.checkpoint.checkpoint(_fn, input_ids, attention_mask, topk_tokens, use_reentrant=False)
            with torch.no_grad():
                _, a = self.forward_block(input_ids, attention_mask, topk_tokens)
        else:
            h, a = self.forward_block(input_ids, attention_mask, topk_tokens)
        return h, a

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
        """
        h: (B, d_lex) hoặc (B, K, d_lex)
        Trả: (B, m) hoặc (B, K, m)
        """
        if h.dim() == 3:
            B, K, D = h.shape
            r = self.U(h.view(B*K, D))                         # (B*K, r)
            z_hat = torch.matmul(r, self.V.weight)             # (B*K, m)
            z_hat = F.normalize(z_hat, p=2, dim=-1)
            return z_hat.view(B, K, -1)
        else:
            r = self.U(h)                                      # (B, r)
            z_hat = torch.matmul(r, self.V.weight)             # (B, m)
            return F.normalize(z_hat, p=2, dim=-1)

    def ortho_reg(self):
        # Stabilized mean-square formulation (nhỏ gọn, scale-invariant hơn)
        U = self.U.weight  # (rank, d)
        Vt = self.V.weight # (rank, m)
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
        """
        topk_tokens=1 → single-vector; >1 → multi-vector (K token).
        Trả về 'z' cùng shape (B,m) hoặc (B,K,m).
        """
        h, att = self.lexical(batch_tok['input_ids'], batch_tok['attention_mask'], topk_tokens=topk_tokens)
        z = self.proj(h)
        return {'z': z, 'h': h, 'att': self.lexical.last_attention}

# ------------------------- Teacher wrapper -------------------------

class TeacherEncoder:
    def __init__(self, name: str, device: str):
        if not _HAS_ST:
            raise RuntimeError('sentence-transformers not installed. pip install sentence-transformers')
        self.model = SentenceTransformer(name, device=device)
    @torch.no_grad()
    def encode(self, texts: List[str], batch_size=64) -> torch.Tensor:
        arr = self.model.encode(texts, batch_size=batch_size, normalize_embeddings=True,
                                convert_to_numpy=True, show_progress_bar=False)
        return torch.from_numpy(arr)
    @property
    def dim(self) -> int:
        return self.model.get_sentence_embedding_dimension()

# ------------------------- EMA -------------------------

class EMA:
    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = {n: p.detach().clone() for n, p in model.named_parameters() if p.requires_grad}
        self.backup = {}

    @torch.no_grad()
    def update(self, model):
        for n, p in model.named_parameters():
            if p.requires_grad:
                self.shadow[n].mul_(self.decay).add_(p.detach(), alpha=1 - self.decay)

    def apply_to(self, model):
        self.backup = {}
        for n, p in model.named_parameters():
            if p.requires_grad:
                self.backup[n] = p.data.detach().clone()
                p.data.copy_(self.shadow[n])

    def restore(self, model):
        for n, p in model.named_parameters():
            if p.requires_grad and n in self.backup:
                p.data.copy_(self.backup[n])
        self.backup = {}

# ------------------------- Helpers (late interaction) -------------------------

def _late_score(zq_one: torch.Tensor, zd_one: torch.Tensor) -> torch.Tensor:
    """
    zq_one: (Kq, m)
    zd_one: (Kd, m)
    Trả: scalar sim = mean_i max_j <zq_i, zd_j>
    """
    sim = zq_one @ zd_one.t()           # (Kq, Kd)
    return sim.max(dim=1).values.mean()

def _batch_late_scores(zq: torch.Tensor, zd: torch.Tensor) -> torch.Tensor:
    """
    zq: (Bq, Kq, m), zd: (Bd, Kd, m)
    Trả: sims (Bq, Bd) bằng vòng lặp doc theo khối (ổn khi Bd vừa).
    Tối ưu: nếu Kd==1 → sims = mean_Kq(zq) @ zd.squeeze(1).T
    """
    Bq, Kq, M = zq.shape
    Bd, Kd, _ = zd.shape
    if Kd == 1:
        zq_mean = zq.mean(dim=1)              # (Bq, m)
        zd_flat = zd.squeeze(1)               # (Bd, m)
        return zq_mean @ zd_flat.t()          # (Bq, Bd)
    # Kd > 1: compute in chunks to save memory
    chunk = max(1, 2048 // max(1, Kd))        # heuristic
    sims = []
    for start in range(0, Bd, chunk):
        end = min(Bd, start + chunk)
        zd_blk = zd[start:end]                # (b, Kd, m)
        # expand & compute
        # zq: (Bq,Kq,m) → (Bq,1,Kq,m)
        # zd_blk: (b,Kd,m) → (1,b,Kd,m)
        zq_e = zq.unsqueeze(1)                # (Bq,1,Kq,m)
        zd_e = zd_blk.unsqueeze(0)            # (1,b,Kd,m)
        sim = torch.matmul(zq_e, zd_e.transpose(-1, -2))  # (Bq,b,Kq,Kd)
        sim = sim.max(dim=-1).values.mean(dim=-1)         # (Bq,b)
        sims.append(sim)
    return torch.cat(sims, dim=1)             # (Bq,Bd)

# ------------------------- Training -------------------------

def train_one_epoch(model, teacher, data_loader, tokenizer, device, args, scaler, optimizer, scheduler):
    model.train()
    total_loss = 0.0
    optimizer.zero_grad(set_to_none=True)

    use_amp = (args.dtype != 'fp32')
    amp_dtype = torch.bfloat16 if args.dtype == 'bf16' else torch.float16

    # K cho query và doc
    KQ = args.topk_q if args.late_interaction else 1
    KD = args.topk_d if args.late_interaction else 1

    for step, batch in enumerate(tqdm(data_loader, desc='train')):
        tok_q, tok_d, spans, qs_texts, docs_texts = batch
        tok_q = {k: v.to(device, non_blocking=True) for k, v in tok_q.items()}
        tok_d = {k: v.to(device, non_blocking=True) for k, v in tok_d.items()}

        comp_ret = torch.tensor(0.0, device=device)
        comp_lex = torch.tensor(0.0, device=device)
        comp_ent = torch.tensor(0.0, device=device)
        comp_dist = torch.tensor(0.0, device=device)
        comp_ortho = torch.tensor(0.0, device=device)

        with torch.amp.autocast(device_type='cuda', dtype=amp_dtype, enabled=use_amp):
            out_q = model.encode(tok_q, topk_tokens=KQ)
            out_d = model.encode(tok_d, topk_tokens=KD)
            zq = out_q['z']        # (B,Kq,m) hoặc (B,m)
            hq = out_q['h']        # (B,Kq,d) hoặc (B,d)
            zd_all = out_d['z']    # (sum_docs,Kd,m) hoặc (sum_docs,m)
            hd_all = out_d['h']    # (sum_docs,Kd,d) hoặc (sum_docs,d)

            if zq.dim() == 2:  # unify shapes
                zq = zq.unsqueeze(1)      # (B,1,m)
                hq = hq.unsqueeze(1)
            if zd_all.dim() == 2:
                zd_all = zd_all.unsqueeze(1)  # (Ndoc,1,m)
                hd_all = hd_all.unsqueeze(1)

            # --- build positives for in-batch negatives ---
            if len(spans) > 0:
                z_pos = torch.stack([zd_all[s] for s, _ in spans], dim=0)  # (B,Kd,m)
                h_pos = torch.stack([hd_all[s] for s, _ in spans], dim=0)  # (B,Kd,d)
            else:
                z_pos = zq.new_zeros((0, zd_all.size(1), zd_all.size(-1)))
                h_pos = hq.new_zeros((0, hd_all.size(1), hd_all.size(-1)))

            # --- retrieval + optional lexical margin ---
            loss_list = []
            lex_list = []
            for i, (start, count) in enumerate(spans):
                cand_z = zd_all[start:start+count]  # (Ncand, Kd, m)
                cand_h = hd_all[start:start+count]  # (Ncand, Kd, d)

                if args.late_interaction:
                    # tính điểm từng ứng viên qua late-score
                    scores = []
                    for j in range(cand_z.size(0)):
                        scores.append(_late_score(zq[i], cand_z[j]))
                    sim = torch.stack(scores)  # (Ncand,)
                else:
                    sim = torch.matmul(zq[i:i+1].mean(dim=1), cand_z.squeeze(1).t()).squeeze(0)  # (Ncand,)

                # append in-batch negatives (positives của query khác)
                if args.inbatch_negs and z_pos.size(0) > 1:
                    others = [j for j in range(len(spans)) if j != i]
                    if len(others) > 0:
                        if args.late_interaction:
                            sim_inb = []
                            for j in others:
                                sim_inb.append(_late_score(zq[i], z_pos[j]))
                            sim_inb = torch.stack(sim_inb)  # (B-1,)
                        else:
                            sim_inb = torch.matmul(zq[i:i+1].mean(dim=1), z_pos[others].squeeze(1).mean(dim=1).t()).squeeze(0)
                        sim = torch.cat([sim, sim_inb], dim=0)

                target = torch.tensor(0, device=device)
                loss_ret_i = F.cross_entropy(sim / args.temp, target)
                loss_i = loss_ret_i

                # --- optional lexical margin trên không gian h ---
                loss_lex_i = torch.tensor(0.0, device=device)
                if args.lambda_lex > 0.0 and cand_h.size(0) > 1:
                    hq_i = F.normalize(hq[i].mean(dim=0, keepdim=True), p=2, dim=-1)  # (1,d)
                    h_pos_i = F.normalize(cand_h[0].mean(dim=0, keepdim=True), p=2, dim=-1)
                    s_pos = torch.matmul(hq_i, h_pos_i.t()).squeeze(0)                # ()
                    s_negs = torch.matmul(
                        hq_i, F.normalize(cand_h[1:].mean(dim=1), p=2, dim=-1).t()
                    )                                                                 # (K,)
                    margins = F.relu(args.margin_lex - s_pos + s_negs)
                    loss_lex_i = margins.mean()
                    loss_i = loss_i + args.lambda_lex * loss_lex_i

                loss_list.append(loss_i)
                lex_list.append(loss_lex_i)

            comp_ret = torch.stack(loss_list).mean() if loss_list else torch.tensor(0.0, device=device)
            comp_lex = torch.stack(lex_list).mean() if lex_list else torch.tensor(0.0, device=device)
            loss_mix_all = comp_ret  # đã gồm lexical*lambda_lex ở trên

            # --- entropy reg: comp_ent = sum p log p (âm). Đổi dấu để khuyến khích entropy cao ---
            a_q = out_q['att']; a_d = out_d['att']             # (B,T), (sum_docs,T)
            safe_a_q = a_q.clamp_min(1e-6); safe_a_d = a_d.clamp_min(1e-6)
            ent_q_raw = (safe_a_q * safe_a_q.log()).sum(dim=-1).mean()
            ent_d_raw = (safe_a_d * safe_a_d.log()).sum(dim=-1).mean()
            comp_ent = ent_q_raw + ent_d_raw                   # âm
            loss_ent = -comp_ent                               # dương (positive entropy)

            # --- distillation (stream) ---
            if args.lambda_distill > 0.0:
                with torch.no_grad():
                    zt_q = teacher.encode(qs_texts, batch_size=args.teacher_bs)
                zt_q = zt_q.to(device, non_blocking=True)
                # dùng mean Kq khi late_interaction
                zq_mean = zq.mean(dim=1)
                loss_dist_q = (1.0 - F.cosine_similarity(zq_mean, zt_q, dim=-1)).mean()
                del zt_q

                loss_d_sum = 0.0
                for (start, count) in spans:
                    texts_chunk = docs_texts[start:start+count]
                    with torch.no_grad():
                        zt_chunk = teacher.encode(texts_chunk, batch_size=args.teacher_bs).to(device, non_blocking=True)
                    zd_chunk = zd_all[start:start+count].mean(dim=1)
                    loss_d_sum = loss_d_sum + (1.0 - F.cosine_similarity(zd_chunk, zt_chunk, dim=-1)).mean()
                    del zt_chunk, zd_chunk
                loss_dist_d = loss_d_sum / max(1, len(spans))
                comp_dist = 0.5 * (loss_dist_q + loss_dist_d)
            else:
                comp_dist = torch.tensor(0.0, device=device)

            comp_ortho = model.proj.ortho_reg()

            loss = (
                args.lambda_ret * loss_mix_all +
                args.lambda_ent * loss_ent +        # dùng entropy dương
                args.lambda_distill * comp_dist +
                args.lambda_ortho * comp_ortho
            )

        # --- backward / step ---
        if scaler is not None:
            scaler.scale(loss / args.accum_steps).backward()
        else:
            (loss / args.accum_steps).backward()

        just_stepped = False
        if (step + 1) % args.accum_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            if scaler is not None:
                scaler.step(optimizer); scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            if scheduler is not None:
                scheduler.step()
            # EMA update sau mỗi optimizer.step
            if hasattr(args, '_ema_obj') and args._ema_obj is not None:
                args._ema_obj.update(model)
            just_stepped = True

        total_loss += loss.item()

        # --- logging huấn luyện per-step ---
        if (step + 1) % args.log_every == 0:
            avg = total_loss / (step + 1)
            ent_pos_for_log = float(loss_ent.item())  # log bản dương, dễ đọc
            print(f"step {step+1}: loss={avg:.4f} (ret+lex={loss_mix_all.item():.4f}, lex={comp_lex.item():.4f}, ent_pos={ent_pos_for_log:.4f}, dist={comp_dist.item():.4f}, ortho={comp_ortho.item():.4f})")
            if args.wandb:
                try:
                    cur_lr = optimizer.param_groups[0].get('lr', None)
                except Exception:
                    cur_lr = None
                payload = {
                    'train/loss_step': float(avg),
                    'train/retlex_step': float(loss_mix_all.item()),
                    'train/lex_only_step': float(comp_lex.item()),
                    'train/ent_pos_step': ent_pos_for_log,
                    'train/dist_step': float(comp_dist.item()),
                    'train/ortho_step': float(comp_ortho.item()),
                    'train/accum_steps': int(args.accum_steps),
                    'batch_step_in_epoch': int(step + 1),
                }
                if cur_lr is not None:
                    payload['train/lr'] = float(cur_lr)
                wandb.log(payload)

            # dọn bộ nhớ định kỳ
            del out_q, out_d, zq, hq, zd_all, hd_all, loss_list, lex_list
            gc.collect()
            if device == 'cuda':
                torch.cuda.empty_cache()

        # --- tăng optimizer-step counter & eval giữa epoch theo "Step" ---
        if just_stepped:
            args._global_opt_step += 1
            # mỗi _eval_every_opt_steps thì eval dev + log theo Step
            if (args._dev_rows is not None) and (args._eval_every_opt_steps > 0) and (args._global_opt_step % args._eval_every_opt_steps == 0):
                if hasattr(args, '_ema_obj') and args._ema_obj is not None:
                    args._ema_obj.apply_to(model)
                metrics_mid = evaluate_dev(
                    model, args._dev_rows, tokenizer, device, args,
                    max_len=args.max_len, topk=10,
                    sample=(None if args.dev_eval_sample in (-1, None) else args.dev_eval_sample)
                )
                if hasattr(args, '_ema_obj') and args._ema_obj is not None:
                    args._ema_obj.restore(model)

                print(f"[dev@step {args._global_opt_step}] {metrics_mid}")
                if args.wandb:
                    wandb.log({
                        'dev/Recall@10': metrics_mid['Recall@10'],
                        'dev/MRR@10':    metrics_mid['MRR@10'],
                        'dev/nDCG@10':   metrics_mid['nDCG@10'],
                        'step': int(args._global_opt_step),
                    })

    # flush leftover nếu không chia hết accum_steps
    if (step + 1) % args.accum_steps != 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
        if scaler is not None:
            scaler.step(optimizer); scaler.update()
        else:
            optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        if scheduler is not None:
            scheduler.step()
        if hasattr(args, '_ema_obj') and args._ema_obj is not None:
            args._ema_obj.update(model)
        args._global_opt_step += 1

    return total_loss / max(1, (step + 1))

def _sanitize_args_for_save(args):
    d = dict(vars(args))
    d.pop('_ema_obj', None)
    return d

@torch.no_grad()
def evaluate_dev(model, dev_rows, tokenizer, device, args, max_len=128, topk=10, sample=2000):
    model.eval()
    rows = dev_rows[:sample] if sample and (len(dev_rows) > sample) else dev_rows
    qs = [q for q, p, _ in rows]
    ps = [p for q, p, _ in rows]

    KQ = args.topk_q if args.late_interaction else 1
    KD = args.topk_d if args.late_interaction else 1

    tok_p = tokenizer(ps, padding=True, truncation=True, max_length=max_len, return_tensors='pt')
    tok_p = {k: v.to(device) for k, v in tok_p.items()}
    z_p = model.encode(tok_p, topk_tokens=KD)['z']  # (Nd,Kd,m) or (Nd,m)
    if z_p.dim() == 2: z_p = z_p.unsqueeze(1)

    tok_q = tokenizer(qs, padding=True, truncation=True, max_length=max_len, return_tensors='pt')
    tok_q = {k: v.to(device) for k, v in tok_q.items()}
    z_q = model.encode(tok_q, topk_tokens=KQ)['z']  # (Nq,Kq,m) or (Nq,m)
    if z_q.dim() == 2: z_q = z_q.unsqueeze(1)

    if args.late_interaction:
        sims = _batch_late_scores(z_q, z_p)  # (Nq,Nd)
    else:
        sims = z_q.mean(dim=1) @ z_p.squeeze(1).t()

    mrr = 0.0
    ndcg = 0.0
    recall_hits = []
    for i in range(sims.size(0)):
        scores = sims[i]
        sorted_idx = torch.argsort(scores, descending=True)
        rank = (sorted_idx == i).nonzero(as_tuple=False).item() + 1
        recall_hits.append(1 if rank <= topk else 0)
        if rank <= topk:
            mrr += 1.0 / rank
            ndcg += 1.0 / math.log2(rank + 1)
    N = max(1, sims.size(0))
    return {f'Recall@{topk}': sum(recall_hits) / N, f'MRR@{topk}': mrr / N, f'nDCG@{topk}': ndcg / N}

@torch.no_grad()
def wandb_log_table_examples(model, tokenizer, device, rows, args, max_len=128, topk=5, nsamples=8, truncate=160):
    if not _HAS_WANDB:
        return

    sample = rows[:nsamples] if len(rows) > nsamples else rows
    qs = [q for q, p, _ in sample]
    ps = [p for q, p, _ in sample]

    KQ = args.topk_q if args.late_interaction else 1
    KD = args.topk_d if args.late_interaction else 1

    tok_p = tokenizer(ps, padding=True, truncation=True, max_length=max_len, return_tensors='pt')
    tok_p = {k: v.to(device) for k, v in tok_p.items()}
    z_p = model.encode(tok_p, topk_tokens=KD)['z']
    if z_p.dim() == 2: z_p = z_p.unsqueeze(1)

    tok_q = tokenizer(qs, padding=True, truncation=True, max_length=max_len, return_tensors='pt')
    tok_q = {k: v.to(device) for k, v in tok_q.items()}
    z_q = model.encode(tok_q, topk_tokens=KQ)['z']
    if z_q.dim() == 2: z_q = z_q.unsqueeze(1)

    if args.late_interaction:
        sims = _batch_late_scores(z_q, z_p)
    else:
        sims = z_q.mean(dim=1) @ z_p.squeeze(1).t()

    K = int(min(topk, sims.size(1)))

    columns = ['query', 'gold']
    for j in range(1, K + 1):
        columns += [f'rank{j}_score', f'rank{j}_text']
    table = wandb.Table(columns=columns)

    def _trim(t, n):
        return (t[:n] + '…') if isinstance(t, str) and len(t) > n else t

    for i, q in enumerate(qs):
        topv, topi = torch.topk(sims[i], k=K)
        row = [_trim(q, truncate), _trim(ps[i], truncate)]
        for j in range(K):
            row.append(float(topv[j]))
            row.append(_trim(ps[int(topi[j])], truncate))
        table.add_data(*row)

    wandb.log({'examples/topk_v2': table})

# ------------------------- Export & FAISS -------------------------

def _extract_passages(rows: List[Tuple[str,str,List[str]]], include_negs: bool=False) -> List[str]:
    docs = []
    for q, p, negs in rows:
        docs.append(p)
        if include_negs:
            docs.extend(negs)
    seen, uniq = set(), []
    for t in docs:
        if t not in seen:
            seen.add(t); uniq.append(t)
    return uniq

@torch.no_grad()
def export_index(model, tokenizer, device, texts: List[str], out_dir: str, max_len: int, hnsw_m: int, efc: int, efs: int, demo_query: str=None, topk: int=5):
    """
    Xuất index FAISS: luôn encode **doc K=1** để ra đúng 1 vector/doc.
    """
    os.makedirs(out_dir, exist_ok=True)
    bs = 128
    vecs = []
    for i in tqdm(range(0, len(texts), bs), desc='encode export'):
        batch = texts[i:i+bs]
        tok = tokenizer(batch, padding=True, truncation=True, max_length=max_len, return_tensors='pt')
        tok = {k: v.to(device) for k, v in tok.items()}
        z = model.encode(tok, topk_tokens=1)['z'].cpu().numpy()   # (B,m)
        if z.ndim == 3:  # phòng trường hợp
            z = z.mean(axis=1)
        vecs.append(z)
    X = np.vstack(vecs).astype('float32') if len(vecs) else np.zeros((0, model.proj.V.weight.shape[1]), dtype='float32')
    np.save(os.path.join(out_dir, 'embeddings.npy'), X)
    with open(os.path.join(out_dir, 'texts.json'), 'w', encoding='utf-8') as f:
        json.dump(texts, f, ensure_ascii=False)

    if not _HAS_FAISS:
        print('[export] faiss not installed; skip index build')
        return

    dim = X.shape[1]
    index = faiss.IndexHNSWFlat(dim, hnsw_m)
    index.hnsw.efConstruction = efc
    index.add(X)
    faiss.write_index(index, os.path.join(out_dir, 'faiss_hnsw.index'))
    print('[export] wrote', os.path.join(out_dir, 'faiss_hnsw.index'))

    if demo_query:
        tokq = tokenizer([demo_query], padding=True, truncation=True, max_length=max_len, return_tensors='pt')
        tokq = {k: v.to(device) for k, v in tokq.items()}
        zq = model.encode(tokq, topk_tokens=1)['z'].cpu().numpy()
        index.hnsw.efSearch = efs
        D, I = index.search(zq, topk)
        print('[export demo] query =', demo_query)
        for rank, (d,i) in enumerate(zip(D[0], I[0]), start=1):
            print(f'  #{rank} (score={float(d):.4f}) -> {texts[i][:200]}')

# ------------------------- Main -------------------------

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument('--train_tsv', type=str, required=True)
    ap.add_argument('--dev_tsv', type=str, default=None)
    ap.add_argument('--val_ratio', type=float, default=0.02)
    ap.add_argument('--shuffle_before_split', action='store_true')
    ap.add_argument('--max_train_rows', type=int, default=None)
    ap.add_argument('--max_len', type=int, default=128)

    ap.add_argument('--teacher', type=str, default='BAAI/bge-m3')
    ap.add_argument('--tokenizer', type=str, default='BAAI/bge-m3')
    ap.add_argument('--teacher_bs', type=int, default=64)

    ap.add_argument('--d_lex_emb', type=int, default=128)
    ap.add_argument('--d_lex', type=int, default=128)
    ap.add_argument('--rank', type=int, default=64)
    ap.add_argument('--m_teacher', type=int, default=1024)

    ap.add_argument('--neg_per_sample', type=int, default=7)
    ap.add_argument('--batch_size', type=int, default=64)
    ap.add_argument('--epochs', type=int, default=1)
    ap.add_argument('--lr', type=float, default=2e-4)
    ap.add_argument('--weight_decay', type=float, default=0.01)
    ap.add_argument('--temp', type=float, default=0.07)
    ap.add_argument('--max_grad_norm', type=float, default=1.0)

    ap.add_argument('--warmup_steps', type=int, default=1000)
    ap.add_argument('--lr_schedule', type=str, default='cosine', choices=['cosine','linear','constant'])

    ap.add_argument('--lambda_ret', type=float, default=1.0)
    ap.add_argument('--lambda_ent', type=float, default=1e-3)
    ap.add_argument('--lambda_distill', type=float, default=0.0)
    ap.add_argument('--lambda_ortho', type=float, default=1e-3)
    ap.add_argument('--lambda_lex', type=float, default=0.0)
    ap.add_argument('--margin_lex', type=float, default=0.2)

    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--fp16', action='store_true')  # giữ để không break CLI cũ (không còn dùng, ưu tiên --dtype)
    ap.add_argument('--dtype', type=str, default='bf16', choices=['fp32','fp16','bf16'])
    ap.add_argument('--num_workers', type=int, default=2)
    ap.add_argument('--log_every', type=int, default=100)

    ap.add_argument('--early_stop_patience', type=int, default=3)
    ap.add_argument('--early_stop_min_delta', type=float, default=1e-4)

    ap.add_argument('--output_dir', type=str, default='runs/longmatrix')
    ap.add_argument('--resume', type=str, default=None)

    ap.add_argument('--wandb', action='store_true')
    ap.add_argument('--wandb_project', type=str, default='longmatrix')
    ap.add_argument('--wandb_run', type=str, default=None)
    ap.add_argument('--wandb_mode', type=str, default='online', choices=['online','offline','disabled'])
    ap.add_argument('--wandb_table_samples', type=int, default=8)

    ap.add_argument('--export_after_train', action='store_true')
    ap.add_argument('--export_from', type=str, default='train', choices=['train','dev','file'])
    ap.add_argument('--export_file', type=str, default=None)
    ap.add_argument('--export_include_negs', action='store_true')
    ap.add_argument('--export_out_dir', type=str, default=None)
    ap.add_argument('--faiss_hnsw_m', type=int, default=64)
    ap.add_argument('--faiss_hnsw_efc', type=int, default=200)
    ap.add_argument('--faiss_search_efs', type=int, default=256)
    ap.add_argument('--export_demo_query', type=str, default=None)
    ap.add_argument('--export_demo_topk', type=int, default=5)

    ap.add_argument('--attn_backend', type=str, default='sdpa', choices=['sdpa','matmul'])
    ap.add_argument('--grad_ckpt', action='store_true')
    ap.add_argument('--accum_steps', type=int, default=1)
    ap.add_argument('--allow_tf32', action='store_true')
    ap.add_argument('--heads', type=int, default=4)
    ap.add_argument('--torch_compile', action='store_true')

    ap.add_argument('--inbatch_negs', action='store_true')
    ap.add_argument('--ema', type=float, default=0.0)   # 0 = off; gợi ý 0.999

    ap.add_argument('--dev_eval_sample', type=int, default=2000)

    # NEW: eval giữa epoch
    ap.add_argument('--dev_eval_every_epochs', type=float, default=1.0,
                    help='Đánh giá dev sau mỗi X epoch (float, ví dụ 0.5) theo đơn vị optimizer steps')

    # NEW: Late interaction flags
    ap.add_argument('--late_interaction', action='store_true',
                    help='Bật ColBERT-lite: top-K token vectors cho query/doc.')
    ap.add_argument('--topk_q', type=int, default=4, help='Số token K cho query khi late_interaction.')
    ap.add_argument('--topk_d', type=int, default=1, help='Số token K cho doc khi late_interaction (1 để nhanh).')

    return ap.parse_args()

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # Lưu YAML config local (tái hiện run sau này)
    yaml_path = os.path.join(args.output_dir, "config_used.yaml")
    safe_args = _sanitize_args_for_save(args)
    with open(yaml_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(safe_args, f, allow_unicode=True)

    print(f"[save] wrote config to {yaml_path}")

    # Ưu tiên --dtype; nếu user dùng --fp16 cũ thì ép dtype=fp16 để tương thích
    if args.fp16:
        args.dtype = 'fp16'

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('Device:', device, '| dtype:', args.dtype)

    if device == 'cuda':
        try:
            import torch.backends.cuda as cuda_backends
            cuda_backends.sdp_kernel(enable_flash=True, enable_mem_efficient=True, enable_math=False)
        except Exception as e:
            print('[warn] cannot set sdp_kernel flags:', e)
        torch.backends.cuda.matmul.allow_tf32 = bool(args.allow_tf32)
        torch.set_float32_matmul_precision('high' if args.allow_tf32 else 'medium')
        os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')

    if args.wandb:
        if not _HAS_WANDB:
            raise RuntimeError('wandb not installed. pip install wandb')
        if args.wandb_mode == 'disabled':
            os.environ['WANDB_MODE'] = 'disabled'
        elif args.wandb_mode == 'offline':
            os.environ['WANDB_MODE'] = 'offline'
        wandb.init(project=args.wandb_project, name=args.wandb_run, config=vars(args))

    all_rows = read_tsv(args.train_tsv, max_rows=args.max_train_rows)
    print(f'Loaded rows: {len(all_rows):,} from {args.train_tsv}')

    if args.dev_tsv is None:
        n = len(all_rows)
        idx = list(range(n))
        if args.shuffle_before_split:
            random.shuffle(idx)
        cut = int(max(1, round(n * (1.0 - args.val_ratio))))
        train_idx, dev_idx = idx[:cut], idx[cut:]
        train_rows = [all_rows[i] for i in train_idx]
        dev_rows = [all_rows[i] for i in dev_idx]
        print(f'Split train/dev = {len(train_rows):,} / {len(dev_rows):,} (val_ratio={args.val_ratio})')
    else:
        train_rows = all_rows
        dev_rows = read_tsv(args.dev_tsv)

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, use_fast=True)
    ds = TripletDataset(train_rows, neg_per_sample=args.neg_per_sample)
    collate = Collator(tokenizer, args.max_len)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True,
                    num_workers=args.num_workers, pin_memory=(device=='cuda'),
                    persistent_workers=(args.num_workers>0),
                    prefetch_factor=4,
                    collate_fn=collate)

    teacher = None
    if args.lambda_distill > 0.0:
        if not _HAS_ST:
            raise RuntimeError('Install sentence-transformers for distillation')
        teacher = TeacherEncoder(args.teacher, device=device)
        args.m_teacher = teacher.dim
        print('Teacher dim:', args.m_teacher)

    model = LongMatrixModel(
        tokenizer,
        d_lex_emb=args.d_lex_emb, d_lex=args.d_lex,
        m_teacher=args.m_teacher, rank=args.rank,
        attn_backend=args.attn_backend,
        use_ckpt=args.grad_ckpt,
        heads=args.heads
    ).to(device)

    # torch.compile (PyTorch >= 2.3)
    if args.torch_compile and device == 'cuda':
        try:
            model = torch.compile(model, mode='max-autotune')
            print('[compile] torch.compile enabled (mode=max-autotune)')
        except Exception as e:
            print('[warn] torch.compile failed, continue without compile:', e)

    if args.resume and os.path.exists(args.resume):
        ckpt = torch.load(args.resume, map_location='cpu', weights_only=False)
        model.load_state_dict(ckpt['model'])
        print('[resume] loaded weights from', args.resume)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # --- Thiết lập eval giữa epoch theo optimizer steps ---
    steps_per_epoch_batches = len(dl)
    steps_per_epoch_opt = max(1, math.ceil(steps_per_epoch_batches / max(1, args.accum_steps)))
    eval_every_epochs = max(0.0, float(args.dev_eval_every_epochs))
    eval_every_opt_steps = int(round(steps_per_epoch_opt * eval_every_epochs)) if eval_every_epochs > 0 else 0
    args._global_opt_step = 0
    args._eval_every_opt_steps = eval_every_opt_steps
    args._dev_rows = dev_rows if (dev_rows and len(dev_rows) > 0) else None

    if eval_every_opt_steps > 0:
        print(f"[dev-eval] steps_per_epoch_opt={steps_per_epoch_opt}, dev_eval_every_epochs={eval_every_epochs} → eval mỗi {eval_every_opt_steps} optimizer steps")

    # LRscheduler tính theo optimizer steps (chuẩn khi có accum_steps)
    total_opt_steps = steps_per_epoch_opt * max(1, args.epochs)
    if args.lr_schedule == 'cosine':
        scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=total_opt_steps)
    elif args.lr_schedule == 'linear':
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=total_opt_steps)
    else:
        scheduler = None

    # GradScaler: chỉ dùng cho fp16; bf16 không cần scaler
    scaler = torch.amp.GradScaler('cuda', enabled=(device=='cuda' and args.dtype=='fp16'))

    # EMA
    ema = EMA(model, decay=args.ema) if args.ema and args.ema > 0.0 else None
    args._ema_obj = ema  # pass vào train_one_epoch qua args

    best_recall = -1.0
    patience_left = args.early_stop_patience

    for epoch in range(1, args.epochs + 1):
        print(f'=== Epoch {epoch}/{args.epochs} ===')
        avg_loss = train_one_epoch(model, teacher, dl, tokenizer, device, args, scaler, optimizer, scheduler)
        print(f'[epoch {epoch}] train loss: {avg_loss:.4f}')
        if args.wandb:
            wandb.log({'train/loss': avg_loss, 'epoch': epoch})

        metrics = None
        if dev_rows and len(dev_rows) > 0:
            sample_n = None if args.dev_eval_sample in (-1, None) else args.dev_eval_sample
            if ema is not None:
                ema.apply_to(model)
            metrics = evaluate_dev(model, dev_rows, tokenizer, device, args, max_len=args.max_len, topk=10, sample=sample_n)
            if ema is not None:
                ema.restore(model)

            print('[dev]', metrics)
            if args.wandb:
                wandb.log({
                    'dev/Recall@10': metrics['Recall@10'],
                    'dev/MRR@10': metrics['MRR@10'],
                    'dev/nDCG@10': metrics['nDCG@10'],
                    'epoch': epoch
                })
                wandb_log_table_examples(model, tokenizer, device, dev_rows, args, max_len=args.max_len, topk=5, nsamples=args.wandb_table_samples)
            recall = metrics['Recall@10']
            improved = (recall - best_recall) > args.early_stop_min_delta
            if improved:
                best_recall = recall
                patience_left = args.early_stop_patience
                ckpt = os.path.join(args.output_dir, f'best.pt')
                safe_args = _sanitize_args_for_save(args)
                torch.save({'model': model.state_dict(), 'args': safe_args, 'metrics': metrics}, ckpt)

                print('Saved', ckpt)
                if args.wandb:
                    wandb.save(ckpt)
                    art = wandb.Artifact('longmatrix-best', type='model')
                    art.add_file(ckpt)
                    wandb.log_artifact(art)
            else:
                patience_left -= 1
                print(f'[early-stop] no improvement. patience_left={patience_left}')
                if patience_left <= 0:
                    print('[early-stop] stopping training')
                    break

        ckpt = os.path.join(args.output_dir, f'epoch{epoch}.pt')
        safe_args = _sanitize_args_for_save(args)
        torch.save({'model': model.state_dict(), 'args': safe_args}, ckpt)

        print('Saved', ckpt)
        if args.wandb:
            wandb.save(ckpt)

    if args.export_after_train:
        src = args.export_from
        export_dir = args.export_out_dir or os.path.join(args.output_dir, 'export')
        if src == 'train':
            texts = _extract_passages(train_rows, include_negs=args.export_include_negs)
        elif src == 'dev':
            texts = _extract_passages(dev_rows, include_negs=False) if dev_rows else []
        else:
            texts = []
            if args.export_file is None or not os.path.exists(args.export_file):
                print('[export] export_file not found; skip')
            else:
                with open(args.export_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        parts = line.rstrip('\n').split('\t')
                        if parts and parts[0].strip():
                            texts.append(parts[0].strip())
        print(f'[export] texts={len(texts):,}')
        export_index(model, tokenizer, device, texts, export_dir, args.max_len, args.faiss_hnsw_m, args.faiss_hnsw_efc, args.faiss_search_efs, args.export_demo_query, args.export_demo_topk)

if __name__ == '__main__':
    main()