"""
flops_estimator.py

Estimate model size (params), total FLOPs and average FLOPs for encoding the first N
queries from a `queries.dev.tsv` file.

Supported model types:
 - splade  : loads sentence_transformers.SparseEncoder(model_name) and uses empirical timing
 - hf      : loads a HuggingFace transformer (AutoModel) and runs ptflops if available
 - pt      : loads a local torch .pt file that contains an nn.Module

Behavior:
 - If `ptflops` is installed and the model is a torch.nn.Module that accepts tensor input,
   the script will try to compute MACs via ptflops -> FLOPs = 2 * MACs.
 - Otherwise it will fall back to empirical timing and a simple FLOPs estimate for
   sparse-style encoders (vocab_size * avg_tokens * 2).

Example:
  python flops_estimator.py --model "naver/splade-v3" --type splade --num_queries 50

"""

import argparse
import csv
import os
import sys
import time
import math
from collections import OrderedDict

import torch


def load_queries(queries_path, n=50):
    queries = OrderedDict()
    with open(queries_path, 'r', encoding='utf-8', newline='') as f:
        reader = csv.reader(f, delimiter='\t')
        for row in reader:
            if len(row) >= 2:
                try:
                    qid = int(row[0])
                except Exception:
                    # fallback if qid not int
                    qid = len(queries)
                queries[qid] = row[1]
                if len(queries) >= n:
                    break
    return queries


def count_params(model):
    try:
        return sum(p.numel() for p in model.parameters())
    except Exception:
        return None


def try_ptflops(model, input_res, device, batch_size=1):
    """Try to compute MACs using ptflops.get_model_complexity_info.
    Returns (macs, params) or (None,None) on failure.
    
    Args:
        model: The model to analyze
        input_res: Input resolution (e.g., (seq_len,))
        device: Device to run on
        batch_size: Batch size for FLOPs calculation (default=1 for per-query cost)
    """
    try:
        from ptflops import get_model_complexity_info
    except Exception:
        print("ptflops not installed (pip install ptflops) - skipping precise flops computation.")
        return None, None

    try:
        # model should be on device
        model = model.to(device)
        # get_model_complexity_info expects a function signature that accepts the tensor size
        # For embedding models, we need to use input_constructor to generate integer tensors
        def input_constructor(input_shape):
            # input_shape is a tuple like (seq_len,)
            # Return tensor as positional argument so ptflops can detect batch size
            seq_len = input_shape[0]
            # Generate random integer tensor for input_ids
            input_ids = torch.randint(0, 30522, (batch_size, seq_len), dtype=torch.long, device=device)
            return input_ids
        
        macs, params = get_model_complexity_info(
            model, 
            input_res, 
            as_strings=False, 
            print_per_layer_stat=False, 
            verbose=False,
            input_constructor=input_constructor
        )
        return macs, params
    except Exception as e:
        print(f"ptflops failed: {e}")
        return None, None


def estimate_flops_from_sparse_embedding(embeddings, tokens_counts, encode_time):
    """Estimate FLOPs for sparse-style embeddings.
    embeddings: torch.Tensor shape [N, vocab_size]
    tokens_counts: list of token counts per input (length N)
    encode_time: total time to encode those N inputs
    Returns: total_flops_estimate (float), gflops_per_sec empirical (float)
    """
    N, vocab_size = embeddings.shape
    avg_tokens = float(sum(tokens_counts)) / max(1, len(tokens_counts))
    # naive estimate: vocab_size * avg_tokens * 2 FLOPs per input
    approx_flops_per_input = vocab_size * avg_tokens * 2
    total_flops = approx_flops_per_input * N
    gflops_per_sec = (total_flops / encode_time) / 1e9 if encode_time > 0 else 0.0
    return total_flops, gflops_per_sec, avg_tokens


class HFWrapper(torch.nn.Module):
    """Wrap a HuggingFace model so ptflops can call forward with input_ids tensor."""
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_ids):
        # input_ids: [B, L] - tensor passed as positional argument from input_constructor
        outputs = self.model(input_ids=input_ids)
        # return logits or last hidden state to let ptflops count operations
        # Most models return BaseModelOutputWithPoolingAndCrossAttentions or tuple
        if hasattr(outputs, 'last_hidden_state'):
            return outputs.last_hidden_state
        elif isinstance(outputs, (tuple, list)) and len(outputs) > 0:
            return outputs[0]
        else:
            # fallback
            return outputs


class LongMatrixWrapper(torch.nn.Module):
    """Wrap a LongMatrix model so ptflops can call forward with input_ids tensor."""
    def __init__(self, model, topk_tokens=1):
        super().__init__()
        self.model = model
        self.topk_tokens = topk_tokens

    def forward(self, input_ids):
        # input_ids: [B, L] - tensor passed as positional argument from input_constructor
        # Create attention_mask (all 1s for simplicity in FLOPs estimation)
        attention_mask = torch.ones_like(input_ids)
        batch_tok = {'input_ids': input_ids, 'attention_mask': attention_mask}
        
        # Call encode method
        out = self.model.encode(batch_tok, topk_tokens=self.topk_tokens)
        # Return the dense embedding 'z'
        return out['z']


def human_readable_params(n):
    if n is None:
        return 'unknown'
    for unit in ['','K','M','B']:
        if abs(n) < 1000.0:
            return f"{n:.2f}{unit}"
        n /= 1000.0
    return f"{n:.2f}T"


def main():
    parser = argparse.ArgumentParser(description='Estimate FLOPs and params for SPLADE/ColBERT/HF/local models')
    parser.add_argument('--model', type=str, required=True, help='Model name (HF/splade) or path to .pt')
    parser.add_argument('--type', choices=['splade','hf','pt'], default='splade', help='Model type')
    parser.add_argument('--device', choices=['cpu','cuda'], default='cuda', help='Device')
    parser.add_argument('--queries', type=str, default='queries.dev.tsv', help='Path to queries.dev.tsv')
    parser.add_argument('--num_queries', type=int, default=50, help='Number of first queries to encode')
    parser.add_argument('--seq_len', type=int, default=32, help='Sequence length to use for HF ptflops input')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size for encoding')
    parser.add_argument('--flops_batch_size', type=int, default=1, help='Batch size for FLOPs calculation (default=1 for per-query cost)')

    args = parser.parse_args()
    device = args.device
    if device == 'cuda' and not torch.cuda.is_available():
        print('CUDA not available, falling back to CPU')
        device = 'cpu'

    queries = load_queries(args.queries, n=args.num_queries)
    print(f"Loaded {len(queries)} queries from {args.queries}")

    if args.type == 'splade':
        try:
            from sentence_transformers import SparseEncoder
        except Exception as e:
            print('sentence_transformers not installed. pip install sentence-transformers')
            return

        print(f"Loading SPLADE model: {args.model}")
        model = SparseEncoder(args.model)
        # try to get underlying pytorch module if available for param counting
        underlying = None
        if hasattr(model, 'model'):
            underlying = model.model
        elif hasattr(model, 'encoder'):
            underlying = model.encoder
        # Also try the transformer model inside SentenceTransformer
        if underlying is None and hasattr(model, '_first_module'):
            underlying = model._first_module()
        if underlying is None and hasattr(model, 'modules') and callable(model.modules):
            try:
                mods = list(model.modules())
                if len(mods) > 1:
                    underlying = mods[1]  # First real module after wrapper
            except Exception:
                pass

        params = count_params(underlying) if underlying is not None else None
        print(f"Model params: {human_readable_params(params)}")

        # encode queries empirically
        texts = list(queries.values())
        tokens_counts = []
        t0 = time.time()
        with torch.no_grad():
            embeddings = model.encode_query(texts)  # tensor [N, vocab]
            if device == 'cuda':
                try:
                    torch.cuda.synchronize()
                except Exception:
                    pass
        t1 = time.time()
        encode_time = t1 - t0
        if isinstance(embeddings, (list, tuple)):
            # some implementations return list; convert to tensor if possible
            embeddings = torch.stack([e for e in embeddings])
        if hasattr(model, 'tokenizer'):
            # count tokens using tokenizer
            for q in texts:
                try:
                    tokens = model.tokenizer.tokenize(q)
                    tokens_counts.append(len(tokens))
                except Exception:
                    tokens_counts.append(10)
        else:
            tokens_counts = [10] * len(texts)

        # embeddings may be numpy
        if hasattr(embeddings, 'shape'):
            emb_shape = embeddings.shape
        else:
            emb_shape = (len(texts), 0)
        if isinstance(embeddings, torch.Tensor):
            total_flops, gflops_per_sec, avg_tokens = estimate_flops_from_sparse_embedding(embeddings, tokens_counts, encode_time)
            print(f"\n--- Performance Metrics ---")
            print(f"Embedding shape: {emb_shape}")
            print(f"Estimated avg tokens/query: {avg_tokens:.2f}")
            print(f"FLOPs per query (estimated): {total_flops/len(texts):,.0f} ({total_flops/len(texts)/1e9:.4f} GFLOPs)")
            print(f"Total FLOPs for {len(texts)} queries: {total_flops/1e9:.4f} GFLOPs")
            print(f"\n--- Encoding Speed ---")
            print(f"Encoded {len(texts)} queries in {encode_time:.4f}s")
            print(f"Throughput: {len(texts)/encode_time:.2f} queries/sec")
            print(f"Latency: {encode_time/len(texts)*1000:.4f} ms/query")
            print(f"Empirical GFLOPs/s: {gflops_per_sec:.4f}")
        else:
            print("Could not interpret embeddings shape; skipping sparse FLOPs estimate.")

    elif args.type == 'hf':
        try:
            from transformers import AutoModel, AutoTokenizer
        except Exception as e:
            print('transformers not installed. pip install transformers')
            return

        print(f"Loading HF model: {args.model}")
        tokenizer = AutoTokenizer.from_pretrained(args.model)
        model = AutoModel.from_pretrained(args.model)
        params = count_params(model)
        print(f"Model params: {human_readable_params(params)}")

        # try ptflops
        wrapper = HFWrapper(model)
        input_res = (args.seq_len,)  # sequence length
        macs, params_pt = try_ptflops(wrapper, input_res, device, batch_size=args.flops_batch_size)
        
        print(f"\n--- Performance Metrics ---")
        if macs is not None:
            flops = 2 * macs
            flops_per_query = flops / args.flops_batch_size if args.flops_batch_size > 1 else flops
            print(f"FLOPs per query (ptflops): {flops_per_query:,.0f} ({flops_per_query/1e9:.4f} GFLOPs)")
            if args.flops_batch_size > 1:
                print(f"Total FLOPs for batch_size={args.flops_batch_size}: {flops:,} ({flops/1e9:.4f} GFLOPs)")
        else:
            print("FLOPs estimation not available (ptflops failed)")

        # empirical encoding of first N queries
        texts = list(queries.values())
        enc_inputs = tokenizer(texts, padding=True, truncation=True, max_length=args.seq_len, return_tensors='pt')
        input_ids = enc_inputs['input_ids'].to(device)
        model = model.to(device)
        model.eval()
        
        print(f"\n--- Encoding Speed ---")
        t0 = time.time()
        with torch.no_grad():
            out = model(input_ids=input_ids)
            if device == 'cuda':
                torch.cuda.synchronize()
        t1 = time.time()
        enc_time = t1 - t0
        print(f"Encoded {len(texts)} queries in {enc_time:.4f}s")
        print(f"Throughput: {len(texts)/enc_time:.2f} queries/sec")
        print(f"Latency: {enc_time/len(texts)*1000:.4f} ms/query")
        if macs is not None:
            empirical_gflops_per_sec = (flops_per_query * len(texts) / enc_time) / 1e9
            print(f"Empirical GFLOPs/s: {empirical_gflops_per_sec:.4f}")

    elif args.type == 'pt':
        # load a local .pt which should contain an nn.Module
        path = args.model
        if not os.path.exists(path):
            print(f"File not found: {path}")
            return
        print(f"Loading local .pt module: {path}")
        obj = torch.load(path, map_location=device)
        
        # Debug: print what we loaded
        print(f"Loaded object type: {type(obj)}")
        if isinstance(obj, dict):
            print(f"Dict keys: {list(obj.keys())}")
        
        if isinstance(obj, torch.nn.Module):
            model = obj
        elif isinstance(obj, dict):
            # Try common checkpoint formats
            if 'model' in obj:
                if isinstance(obj['model'], torch.nn.Module):
                    model = obj['model']
                    print("Loaded model from checkpoint['model']")
                elif isinstance(obj['model'], dict):
                    # It's a state dict - try to reconstruct the model
                    print("checkpoint['model'] contains a state_dict")
                    
                    # Check if this is a LongMatrix checkpoint
                    if 'args' in obj:
                        args_obj = obj['args']
                        print(f"Checkpoint has args of type: {type(args_obj)}")
                        
                        # Try LongMatrix model reconstruction
                        is_longmatrix = False
                        if isinstance(args_obj, dict):
                            # Check for LongMatrix-specific args
                            if 'd_lex' in args_obj or 'd_lex_emb' in args_obj or 'tokenizer' in args_obj:
                                is_longmatrix = True
                                print("Detected LongMatrix checkpoint")
                        
                        if is_longmatrix:
                            try:
                                print("Attempting to reconstruct LongMatrix model from checkpoint args...")
                                # Import local LongMatrix if available
                                sys.path.insert(0, os.path.dirname(path))
                                try:
                                    from train_longmatrix import LongMatrixModel
                                    print("Imported LongMatrixModel from train_longmatrix.py")
                                except ImportError:
                                    print("Could not import from train_longmatrix.py")
                                    print("Please ensure train_longmatrix.py is in the same directory as the checkpoint")
                                    print("Or use --type hf to load the underlying HuggingFace model")
                                    return
                                
                                # Reconstruct model with args
                                tokenizer_name = args_obj.get('tokenizer', 'BAAI/bge-m3')
                                from transformers import AutoTokenizer
                                tok = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)
                                
                                model = LongMatrixModel(
                                    tokenizer=tok,
                                    d_lex_emb=args_obj.get('d_lex_emb', 128),
                                    d_lex=args_obj.get('d_lex', 128),
                                    m_teacher=args_obj.get('m_teacher', 1024),
                                    rank=args_obj.get('rank', 64),
                                    attn_backend=args_obj.get('attn_backend', 'sdpa'),
                                    use_ckpt=False,  # disable checkpoint for inference
                                    heads=args_obj.get('heads', 4)
                                )
                                model.load_state_dict(obj['model'])
                                model = model.to(device)
                                print("Successfully loaded LongMatrix model")
                            except Exception as e:
                                print(f"Could not reconstruct LongMatrix model: {e}")
                                import traceback
                                traceback.print_exc()
                                print('\nPlease ensure train_longmatrix.py is available in the same directory.')
                                return
                        else:
                            # Try to load using ColBERT's checkpoint loader
                            try:
                                from colbert.modeling.checkpoint import Checkpoint
                                print(f"Attempting to load ColBERT checkpoint from {path}")
                                model = Checkpoint(path, colbert_config=None)
                                print("Successfully loaded ColBERT model")
                            except Exception as e:
                                print(f"Could not load as ColBERT checkpoint: {e}")
                                print('Please provide the model architecture or use --type hf for HuggingFace models.')
                                return
                    else:
                        print('The loaded file contains a state_dict. Please provide the model architecture.')
                        return
                else:
                    print(f"checkpoint['model'] has unexpected type: {type(obj['model'])}")
                    return
            elif 'state_dict' in obj:
                print('The loaded file contains a state_dict. Please provide the model architecture or use --type hf/splade.')
                print('Available keys in checkpoint:', list(obj.keys()))
                return
            elif 'model_state_dict' in obj:
                print('The loaded file contains a model_state_dict. Please provide the model architecture or use --type hf/splade.')
                print('Available keys in checkpoint:', list(obj.keys()))
                return
            else:
                # Maybe it's a plain state dict without wrapper
                print('The loaded dict does not contain a model or recognizable state_dict key.')
                print('Available keys:', list(obj.keys()))
                return
        else:
            print(f'Unsupported .pt file content type: {type(obj)}')
            print('Expect a saved nn.Module or checkpoint dict with model/state_dict.')
            return

        params = count_params(model)
        print(f"Model params: {human_readable_params(params)}")
        
        # Check if this is a LongMatrix model (has 'encode' method but not standard forward)
        is_longmatrix = hasattr(model, 'encode') and hasattr(model, 'lexical') and hasattr(model, 'proj')
        
        # try ptflops
        if is_longmatrix:
            print("Detected LongMatrix model - using LongMatrixWrapper for FLOPs estimation")
            wrapper = LongMatrixWrapper(model, topk_tokens=1)
        else:
            wrapper = model
        
        # try input_res approximate
        input_res = (args.seq_len,)
        macs, params_pt = try_ptflops(wrapper, input_res, device, batch_size=args.flops_batch_size)
        
        print(f"\n--- Performance Metrics ---")
        if macs is not None:
            flops = 2 * macs
            flops_per_query = flops / args.flops_batch_size if args.flops_batch_size > 1 else flops
            print(f"FLOPs per query (ptflops): {flops_per_query:,.0f} ({flops_per_query/1e9:.4f} GFLOPs)")
            if args.flops_batch_size > 1:
                print(f"Total FLOPs for batch_size={args.flops_batch_size}: {flops:,} ({flops/1e9:.4f} GFLOPs)")
        else:
            print("FLOPs estimation not available (ptflops failed)")

        # Empirical encoding if LongMatrix
        if is_longmatrix:
            texts = list(queries.values())
            tok = model.tokenizer if hasattr(model, 'tokenizer') else None
            if tok is None:
                print("\nWarning: LongMatrix model has no tokenizer, skipping empirical encoding")
            else:
                enc_inputs = tok(texts, padding=True, truncation=True, max_length=args.seq_len, return_tensors='pt')
                enc_inputs = {k: v.to(device) for k, v in enc_inputs.items()}
                model.eval()
                
                print(f"\n--- Encoding Speed ---")
                t0 = time.time()
                with torch.no_grad():
                    out = model.encode(enc_inputs, topk_tokens=1)
                    if device == 'cuda':
                        torch.cuda.synchronize()
                t1 = time.time()
                enc_time = t1 - t0
                print(f"Encoded {len(texts)} queries in {enc_time:.4f}s")
                print(f"Throughput: {len(texts)/enc_time:.2f} queries/sec")
                print(f"Latency: {enc_time/len(texts)*1000:.4f} ms/query")
                if macs is not None:
                    empirical_gflops_per_sec = (flops_per_query * len(texts) / enc_time) / 1e9
                    print(f"Empirical GFLOPs/s: {empirical_gflops_per_sec:.4f}")

    else:
        print('Unknown model type')

    print('\nDone.')


if __name__ == '__main__':
    main()
