# Copyright (c) 2022-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

import argparse
import time
import torch
import pytest

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
parser.add_argument('--repetition_warmup', default=0, type=int)
parser.add_argument('--repetition_benchmark', default=1, type=int)


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

def benchmark_dot_product_attention(dtype, bs, config, repetition_warmup, repetition_benchmark):
    """Test DotProductAttention module with three backends,
    FlashAttention, FusedAttention and UnfusedDotProductAttention"""

    time_forward, time_backward = _run_dot_product_attention(dtype, bs, config, "FlashAttention", repetition_warmup, repetition_benchmark)
    print(f"FlashAttention,  forward/backward time: {time_forward*1000:.2f}/{time_backward*1000:.2f} ms")

    time_forward, time_backward = _run_dot_product_attention(dtype, bs, config, "FusedAttention", repetition_warmup, repetition_benchmark)
    print(f"FusedAttention,  forward/backward time: {time_forward*1000:.2f}/{time_backward*1000:.2f} ms")

    time_forward, time_backward = _run_dot_product_attention(dtype, bs, config, "UnfusedDotProductAttention", repetition_warmup, repetition_benchmark)
    print(f"UnfusedDotProductAttention,  forward/backward time: {time_forward*1000:.2f}/{time_backward*1000:.2f} ms")

def _run_dot_product_attention(dtype, bs, config, backend, repetition_warmup, repetition_benchmark):
    torch.manual_seed(1234)
    torch.cuda.manual_seed(1234)
    os.environ["NVTE_FLASH_ATTN"] = "0"
    os.environ["NVTE_FUSED_ATTN"] = "0"
    if backend == "FlashAttention":
        os.environ["NVTE_FLASH_ATTN"] = "1"
    if backend == "FusedAttention":
        os.environ["NVTE_FUSED_ATTN"] = "1"

    inp = 0.1 * torch.randn(
            config.seq_len, bs, 3, config.num_attention_heads, config.head_dim,
            dtype = dtype).cuda()
    inp.requires_grad=True
    seqlens = torch.empty(bs, dtype = torch.int32).cuda()
    seqlens.fill_(config.seq_len)
    cu_seqlens = torch.zeros(bs + 1, device = inp.device, dtype = torch.int32)
    cu_seqlens[1:] = torch.cumsum(seqlens, dim = 0)
    op_grad = 0.001 * torch.randint(0, 200, (
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

    q = inp[:, :,0,:,:]
    k = inp[:, :,1,:,:]
    v = inp[:, :,2,:,:]

    # warm-up
    for _ in range(repetition_warmup):
        op = block(q, k, v)
        op.backward(op_grad)

    # benchmarking: repeat the same operation for repetition_benchmark times
    time_forward = 0.0
    time_backward = 0.0
    for i in range(repetition_benchmark):
        # forward pass
        torch.cuda.synchronize()
        time_start = time.time()
        op = block(q, k, v)
        torch.cuda.synchronize()
        time_end = time.time()
        time_forward += time_end - time_start

        # backward pass
        torch.cuda.synchronize()
        time_start = time.time()
        op.backward(op_grad)
        torch.cuda.synchronize()
        time_end = time.time()
        time_backward += time_end - time_start

    # return the average forward and backward pass time
    return time_forward/repetition_benchmark, time_backward/repetition_benchmark

if __name__ == "__main__":
    args = parser.parse_args()
    model_config = ModelConfig(args.num_attention_heads, args.head_dim, args.seq_len, args.dropout_p, args.attn_mask_type)

    param_types = [torch.float16]
    if torch.cuda.is_bf16_supported():
        param_types.append(torch.bfloat16)

    assert args.repetition_benchmark > 0, f"The number of repetitions in benchmark must be larger than 0 (but got {repetition_benchmark})."
    for dtype in param_types:
        print(f"\n====={dtype}, {args.batch_size}-{args.seq_len}-{args.num_attention_heads}-{args.head_dim}, p={args.dropout_p}, {args.attn_mask_type}, repetition_warmup={args.repetition_warmup}, repetition_benchmark={args.repetition_benchmark}=========================")
        benchmark_dot_product_attention(dtype, args.batch_size, model_config, args.repetition_warmup, args.repetition_benchmark)
    print("\n")
