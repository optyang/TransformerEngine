'''
In transformer_engine/tests/pytorch, run:
python benchmark_fused_attn.py --batch_size=16 --num_attention_heads=32 --seq_len=512 --head_dim=64 --attn_mask_type="no_mask" --repetition=100
'''

import argparse
import time
import torch
from torch.utils import benchmark

from transformer_engine.pytorch.utils import (
    init_method_normal,
    scaled_init_method_normal,
)
from transformer_engine.pytorch import TransformerLayer
from transformer_engine.pytorch.attention import DotProductAttention
import os

parser = argparse.ArgumentParser(description='benchmarking fMHA in Transformer Engine')
parser.add_argument('--attn_mask_type', default='causal',
                    choices=['causal', 'no_mask'])
parser.add_argument('--batch_size', default=16, type=int)
parser.add_argument('--num_attention_heads', default=16, type=int)
parser.add_argument('--head_dim', default=64, type=int)
parser.add_argument('--seq_len', default=128, type=int)
parser.add_argument('--dropout_p', default=0.0, type=float)
parser.add_argument('--repetition', default=1, type=int)


class ModelConfig:
    def __init__(
        self, num_attention_heads, head_dim, seq_len,
        dropout_p, attn_mask_type,
    ):
        self.num_attention_heads = num_attention_heads
        self.head_dim = head_dim
        self.seq_len = seq_len
        self.dropout_p = dropout_p
        self.attn_mask_type  = attn_mask_type

def benchmark_dot_product_attention(dtype, bs, config, repetition):
    """Test DotProductAttention module with three backends,
    FlashAttention, FusedAttention and UnfusedDotProductAttention"""

    benchmark_forward, benchmark_backward = _run_dot_product_attention(dtype, bs, config, "FlashAttention", repetition)
    time_forward = benchmark_forward.timeit(repetition)
    time_backward = benchmark_backward.timeit(repetition)
    print(f"FlashAttention: forward/backward {time_forward.mean*1e6:.2f}/{time_backward.mean*1e6:.2f} us")

    benchmark_forward, benchmark_backward = _run_dot_product_attention(dtype, bs, config, "FusedAttention", repetition)
    time_forward = benchmark_forward.timeit(repetition)
    time_backward = benchmark_backward.timeit(repetition)
    print(f"FusedAttention: forward/backward {time_forward.mean*1e6:.2f}/{time_backward.mean*1e6:.2f} us")

    benchmark_forward, benchmark_backward = _run_dot_product_attention(dtype, bs, config, "UnfusedDotProductAttention", repetition)
    time_forward = benchmark_forward.timeit(repetition)
    time_backward = benchmark_backward.timeit(repetition)
    print(f"UnfusedDotProductAttention: forward/backward {time_forward.mean*1e6:.2f}/{time_backward.mean*1e6:.2f} us")

def _run_dot_product_attention(dtype, bs, config, backend, repetition):
    torch.manual_seed(1234)
    torch.cuda.manual_seed(1234)
    os.environ["NVTE_FLASH_ATTN"] = "0"
    os.environ["NVTE_FUSED_ATTN"] = "0"
    if backend == "FlashAttention":
        os.environ["NVTE_FLASH_ATTN"] = "1"
    if backend == "FusedAttention":
        os.environ["NVTE_FUSED_ATTN"] = "1"

    q = 0.1 * torch.randn(
            config.seq_len, bs, config.num_attention_heads, config.head_dim,
            dtype=dtype).cuda()
    q.requires_grad=True

    k = 0.1 * torch.randn(
            config.seq_len, bs, config.num_attention_heads, config.head_dim,
            dtype=dtype).cuda()
    k.requires_grad=True

    v = 0.1 * torch.randn(
            config.seq_len, bs, config.num_attention_heads, config.head_dim,
            dtype=dtype).cuda()
    v.requires_grad=True

    op_grad = torch.ones((
            config.seq_len, bs, config.num_attention_heads * config.head_dim
            ), dtype = dtype).cuda()

    block = (
         DotProductAttention(
                config.num_attention_heads,
                config.head_dim,
                attention_dropout = config.dropout_p,
                attn_mask_type = config.attn_mask_type,
                sequence_parallel = False,
                tp_size = 1,
                get_rng_state_tracker = None,
                tp_group = None,
                layer_number = 1,
                attention_type = "self"
        ).to(dtype = dtype).cuda()
    )

    s_forward = torch.cuda.Stream()
    s_forward.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s_forward):
        for _ in range(3):
            op = block(q, k, v)
    torch.cuda.current_stream().wait_stream(s_forward)

    graph_forward = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph_forward):
        op = block(q, k, v)

    benchmark_forward = benchmark.Timer(
        stmt="graph.replay()",
        globals={
            "graph": graph_forward,
        },
        label=backend,
        description="forward pass"
    )

    s_backward = torch.cuda.Stream()
    s_backward.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s_backward):
        for _ in range(3):
            op.backward(op_grad, retain_graph=True)
    torch.cuda.current_stream().wait_stream(s_backward)

    graph_backward = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph_backward):
        op.backward(op_grad, retain_graph=True)

    benchmark_backward = benchmark.Timer(
        stmt="graph.replay()",
        globals={
            "graph": graph_backward,
        },
        label=backend,
        description="backward pass"
    )

    return benchmark_forward, benchmark_backward

if __name__ == "__main__":
    args = parser.parse_args()
    model_config = ModelConfig(args.num_attention_heads, args.head_dim, args.seq_len, args.dropout_p, args.attn_mask_type)

    param_types = [torch.float16]
    if torch.cuda.is_bf16_supported():
        param_types.append(torch.bfloat16)

    assert args.repetition > 0, f"The number of repetitions in benchmark must be larger than 0 (but got {repetition})."
    for dtype in param_types:
        print(f"\n====={dtype}, {args.batch_size}-{args.seq_len}-{args.num_attention_heads}-{args.head_dim}, p={args.dropout_p}, {args.attn_mask_type}, repetition={args.repetition}=========================")
        benchmark_dot_product_attention(dtype, args.batch_size, model_config, args.repetition)
    print("\n")
