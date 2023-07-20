# Copyright (c) 2022-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

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

class ModelConfig:
    def __init__(
        self, num_layers, hidden_size, num_attention_heads, head_dim, seq_len,
        dropout_p, attn_mask_type,
    ):
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.head_dim = head_dim
        assert (hidden_size == num_attention_heads * head_dim
                ), """hidden_size must be = num_heads x head_dim."""
        self.seq_len = seq_len
        self.dropout_p = dropout_p
        self.attn_mask_type  = attn_mask_type

def benchmark_dot_product_attention(dtype, bs, config):
    """Test DotProductAttention module with three backends,
    FlashAttention, FusedAttention and UnfusedDotProductAttention"""

    warm_up, repetitions = 10, 100

    for _ in range(warm_up):
        _, _ = _run_dot_product_attention(dtype, bs, config, "FlashAttention")

    time_forward, time_backward = [0.0] * repetitions, [0.0] * repetitions
    for i in range(repetitions):
        time_forward[i], time_backward[i] = _run_dot_product_attention(dtype, bs, config, "FlashAttention")
    print(f"FlashAttention,  forward/backward time: {sum(time_forward)/repetitions*1000:.2f}/{sum(time_backward)/repetitions*1000:.2f} ms")

    for _ in range(warm_up):
        _, _ = _run_dot_product_attention(dtype, bs, config, "FusedAttention")

    time_forward, time_backward = [0.0] * repetitions, [0.0] * repetitions
    for i in range(repetitions):
        time_forward[i], time_backward[i] = _run_dot_product_attention(dtype, bs, config, "FusedAttention")
    print(f"FusedAttention,  forward/backward time: {sum(time_forward)/repetitions*1000:.2f}/{sum(time_backward)/repetitions*1000:.2f} ms")

    for _ in range(warm_up):
        _, _ = _run_dot_product_attention(dtype, bs, config, "UnfusedDotProductAttention")

    time_forward, time_backward = [0.0] * repetitions, [0.0] * repetitions
    for i in range(repetitions):
        time_forward[i], time_backward[i] = _run_dot_product_attention(dtype, bs, config, "UnfusedDotProductAttention")
    print(f"UnfusedDotProductAttention,  forward/backward time: {sum(time_forward)/repetitions*1000:.2f}/{sum(time_backward)/repetitions*1000:.2f} ms")

def _run_dot_product_attention(dtype, bs, config, backend):

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

    time_forward_begin = time.time()
    op = block(q, k, v)
    time_forward_end = time.time()

    time_backward_begin = time.time()
    op.backward(op_grad)
    time_backward_end = time.time()

    return time_forward_end-time_forward_begin, time_backward_end-time_backward_begin

if __name__ == "__main__":
    batch_size = 16
    num_layers = 1
    hidden_size = 1024
    num_attention_heads = 16
    head_dim = 64
    seq_len = 128
    dropout_p = 0.0
    attn_mask_type = "causal"
    model_config = ModelConfig(num_layers, hidden_size, num_attention_heads, head_dim, seq_len, dropout_p, attn_mask_type)

    param_types = [torch.float16]
    if torch.cuda.is_bf16_supported():
        param_types.append(torch.bfloat16)

    for dtype in param_types:
        print(f"=========================dtype: {dtype}=========================")
        benchmark_dot_product_attention(dtype, batch_size, model_config)
        print("\n")
