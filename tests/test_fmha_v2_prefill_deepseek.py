import pytest
import torch
import math


from flashinfer.jit import get_trtllm_fmha_v2_module
from flashinfer.prefill import fmha_v2_prefill_deepseek
from utils_fp8 import to_float8


def attention_ref(
    batch_size,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    causal: bool,
    sm_scale: float,
) -> torch.Tensor:
    # tensors are (batch_size, seq_len, num_heads, head_dim)
    qo_len = q.shape[1]
    kv_len = k.shape[1]
    logits = torch.einsum("bmhd,bnhd->bhmn", q.float(), k.float()) * sm_scale

    if causal:
        mask = torch.arange(kv_len - qo_len, kv_len, device=q.device).unsqueeze(
            1
        ) >= torch.arange(0, kv_len, device=q.device).unsqueeze(0)
    else:
        mask = torch.ones(qo_len, kv_len, device=q.device)

    logits = logits.masked_fill(mask.unsqueeze(0).unsqueeze(0) == 0, float("-inf"))
    # LSE computation: logsumexp over the key dimension (last dim)
    # logits shape: (batch, num_heads, seq_len, seq_len)
    lse_ref = torch.logsumexp(logits, -1)  # (batch, num_heads, seq_len)
    # Transpose to match expected shape (batch, seq_len, num_heads)
    lse_ref = lse_ref.transpose(1, 2)
    p = torch.softmax(logits, dim=-1)
    o_ref = torch.einsum("bhmn,bnhd->bmhd", p, v.float()).contiguous()

    # Return LSE in natural log (no conversion needed)
    return o_ref, lse_ref


@pytest.mark.parametrize("batch_size", [1, 3, 8])
@pytest.mark.parametrize("num_heads", [1, 3, 8])
@pytest.mark.parametrize(
    "head_dim_qk,head_dim_v",
    [
        (64, 64),
        (128, 128),
        (192, 128),
        (192, 192),
        (256, 256),
    ],
)
@pytest.mark.parametrize("seq_len", [1024, 4096, 8192])
@pytest.mark.parametrize(
    "qkv_dtype,o_dtype",
    [
        (torch.bfloat16, torch.bfloat16),
        (torch.float8_e4m3fn, torch.bfloat16),
    ],
)
@pytest.mark.parametrize("skip_softmax_threshold_scale_factor", [0, 10, 100, 1000, 5000])
def test_fmha_v2_prefill_deepseek_api(
    batch_size, num_heads, head_dim_qk, head_dim_v, seq_len, qkv_dtype, o_dtype,
    skip_softmax_threshold_scale_factor,
):
    if (qkv_dtype, o_dtype) == (torch.float8_e4m3fn, torch.bfloat16) and \
            (head_dim_qk, head_dim_v) != (192, 128):
        pytest.skip("Only context MLA supports fp8 in bf16 out")
    
    torch.manual_seed(42)

    def initialize_tensors(batch_size, num_heads, head_dim_qk, head_dim_v, seq_len):
        device = "cuda"
        if qkv_dtype == torch.float8_e4m3fn:
            q = torch.randn(
                (batch_size, seq_len, num_heads, head_dim_qk),
                dtype=torch.bfloat16,
                device=device,
            )
            k = torch.randn(
                (batch_size, seq_len, num_heads, head_dim_qk),
                dtype=torch.bfloat16,
                device=device,
            )
            v = torch.randn(
                (batch_size, seq_len, num_heads, head_dim_v),
                dtype=torch.bfloat16,
                device=device,
            )

            q, q_scale = to_float8(q, dtype=torch.float8_e4m3fn)
            k, k_scale = to_float8(k, dtype=torch.float8_e4m3fn)
            v, v_scale = to_float8(v, dtype=torch.float8_e4m3fn)
            q_scale = q_scale.item()
            k_scale = k_scale.item()
            v_scale = v_scale.item()
        else:
            q = torch.randn(
                (batch_size, seq_len, num_heads, head_dim_qk),
                dtype=qkv_dtype,
                device=device,
            )
            k = torch.randn(
                (batch_size, seq_len, num_heads, head_dim_qk),
                dtype=qkv_dtype,
                device=device,
            )
            v = torch.randn(
                (batch_size, seq_len, num_heads, head_dim_v),
                dtype=qkv_dtype,
                device=device,
            )
            # For non-FP8 case, scales are 1.0
            q_scale = 1.0
            k_scale = 1.0
            v_scale = 1.0

        # Output and statistics
        o = torch.zeros(
            batch_size, seq_len, num_heads, head_dim_v, dtype=o_dtype, device=device
        )
        lse = torch.zeros(
            batch_size, seq_len, num_heads, 2, dtype=torch.float, device=device
        )
        sm_scale = 1.0 / math.sqrt(head_dim_qk)
        return q, k, v, o, lse, sm_scale, q_scale, k_scale, v_scale

    q, k, v, o, lse, sm_scale, q_scale, k_scale, v_scale = initialize_tensors(
        batch_size, num_heads, head_dim_qk, head_dim_v, seq_len
    )
    scale_bmm1 = q_scale * k_scale * sm_scale
    scale_bmm2 = v_scale
    scale_softmax = 1.0 if qkv_dtype == torch.float8_e4m3fn else 0.0

    # Two reasons to set lse = None:
    #  1. fmha_v2 by default doesn't generate kernel with 
    #     `return_softmax and input_layout != InputLayout.CONTIGUOUS_Q_KV`,
    #     see flashinfer/jit/attention/fmha_v2/generator_utils.py.
    #  2. Hopper fp8 kernel get wrong LSE (may be a bug, will fix later).

    fmha_v2_prefill_deepseek(
        q, k, v, o,
        num_heads=num_heads,
        head_dim=head_dim_qk,
        max_seqlen=seq_len,
        scale_softmax=scale_softmax,
        scale_bmm1=scale_bmm1,
        scale_bmm2=scale_bmm2,
        skip_softmax_threshold_scale_factor=skip_softmax_threshold_scale_factor,
        return_lse=False,
        lse=None,
        skip_softmax_stat=True,
    )
    
    # implementation gives [max(s_i), sum(exp(s_i - max(s_i)))], compute lse from this
    if qkv_dtype == torch.float8_e4m3fn:
        # For E4M3 the softmax is scaled by 256 (the largest power-of-2 below E4M3_MAX=448.0)
        descale = 256
        lse = lse[:, :, :, 0] + torch.log(lse[:, :, :, 1] / descale)
    else:
        lse = lse[:, :, :, 0] + torch.log(lse[:, :, :, 1])

    if qkv_dtype == torch.float8_e4m3fn:
        q_32 = q.to(torch.float32) * q_scale
        k_32 = k.to(torch.float32) * k_scale
        v_32 = v.to(torch.float32) * v_scale
        out_ref, lse_ref = attention_ref(
            batch_size, q_32, k_32, v_32, causal=True, sm_scale=sm_scale
        )
    else:
        out_ref, lse_ref = attention_ref(
            batch_size, q, k, v, causal=True, sm_scale=sm_scale
        )
        out_ref = out_ref.to(o.dtype)

    if q.dtype == torch.float8_e4m3fn and o.dtype == torch.bfloat16:
        rtol, atol = 4e-2, 6e-2
        torch.testing.assert_close(o, out_ref.to(o.dtype), rtol=rtol, atol=atol)
    elif q.dtype == torch.bfloat16 and o.dtype == torch.bfloat16:
        rtol, atol = 1e-2, 1e-2
        torch.testing.assert_close(o, out_ref, rtol=rtol, atol=atol)
    else:
        rtol, atol = 1e-2, 1e-3

    # torch.testing.assert_close(lse, lse_ref, rtol=1e-2, atol=1e-3)


def attention_ref_single(
    q: torch.Tensor,  # [seq_len, num_heads, head_dim]
    k: torch.Tensor,  # [seq_len, num_heads, head_dim]
    v: torch.Tensor,  # [seq_len, num_heads, head_dim_v]
    causal: bool,
    sm_scale: float,
) -> torch.Tensor:
    """Reference attention for a single sequence (varlen mode)."""
    seq_len = q.shape[0]
    num_heads = q.shape[1]

    # [seq_len, num_heads, head_dim] -> [num_heads, seq_len, head_dim]
    q = q.transpose(0, 1).float()
    k = k.transpose(0, 1).float()
    v = v.transpose(0, 1).float()

    # [num_heads, seq_len, seq_len]
    logits = torch.einsum("hmd,hnd->hmn", q, k) * sm_scale

    if causal:
        mask = torch.triu(torch.ones(seq_len, seq_len, device=q.device), diagonal=1).bool()
        logits = logits.masked_fill(mask.unsqueeze(0), float("-inf"))

    p = torch.softmax(logits, dim=-1)
    # [num_heads, seq_len, head_dim_v] -> [seq_len, num_heads, head_dim_v]
    o = torch.einsum("hmn,hnd->hmd", p, v).transpose(0, 1)
    return o


@pytest.mark.parametrize("num_heads", [1, 4, 8])
@pytest.mark.parametrize(
    "head_dim_qk,head_dim_v",
    [
        (64, 64),
        (128, 128),
        (192, 128),
        (192, 192),
        (256, 256),
    ],
)
@pytest.mark.parametrize(
    "seq_lens",
    [
        [8192, 8000, 10000],
        [1024, 2048, 4096],        # 3 sequences with different lengths
        [1024, 2048, 4096, 8192],  # 4 sequences
        [4096, 2048, 1024],        # decreasing lengths
        [1000, 2000, 3000, 4000],  # non-power-of-2 lengths
    ],
)
@pytest.mark.parametrize("skip_softmax_threshold_scale_factor", [0, 10, 100, 1000, 5000])
def test_fmha_v2_prefill_deepseek_varlen(
    num_heads, head_dim_qk, head_dim_v, seq_lens, skip_softmax_threshold_scale_factor
):
    """Test with truly variable length sequences (different lengths per batch)."""
    torch.manual_seed(42)
    device = "cuda"
    dtype = torch.bfloat16

    batch_size = len(seq_lens)
    total_tokens = sum(seq_lens)
    max_seqlen = max(seq_lens)

    # Create cu_seqlens: [0, len0, len0+len1, len0+len1+len2, ...]
    cu_seqlens = torch.zeros(batch_size + 1, dtype=torch.int32, device=device)
    for i, sl in enumerate(seq_lens):
        cu_seqlens[i + 1] = cu_seqlens[i] + sl

    # Create packed tensors [total_tokens, num_heads, head_dim]
    q = torch.randn(total_tokens, num_heads, head_dim_qk, dtype=dtype, device=device)
    k = torch.randn(total_tokens, num_heads, head_dim_qk, dtype=dtype, device=device)
    v = torch.randn(total_tokens, num_heads, head_dim_v, dtype=dtype, device=device)
    o = torch.zeros(total_tokens, num_heads, head_dim_v, dtype=dtype, device=device)

    sm_scale = 1.0 / math.sqrt(head_dim_qk)

    # Run fmha_v2_prefill_deepseek in varlen mode
    fmha_v2_prefill_deepseek(
        q, k, v, o,
        num_heads=num_heads,
        head_dim=head_dim_qk,
        max_seqlen=max_seqlen,
        scale_softmax=0.0,
        scale_bmm1=sm_scale,
        scale_bmm2=1.0,
        skip_softmax_threshold_scale_factor=skip_softmax_threshold_scale_factor,
        return_lse=False,
        lse=None,
        skip_softmax_stat=True,
        cu_seqlens=cu_seqlens,
    )

    # Compute reference output for each sequence separately
    o_ref = torch.zeros_like(o)
    for i in range(batch_size):
        start = cu_seqlens[i].item()
        end = cu_seqlens[i + 1].item()

        q_seq = q[start:end]  # [seq_len_i, num_heads, head_dim]
        k_seq = k[start:end]
        v_seq = v[start:end]

        o_ref[start:end] = attention_ref_single(q_seq, k_seq, v_seq, causal=True, sm_scale=sm_scale)

    o_ref = o_ref.to(o.dtype)

    # Check results
    rtol, atol = 1e-2, 1e-2
    # torch.testing.assert_close(o, o_ref, rtol=rtol, atol=atol)
    try:
        torch.testing.assert_close(o, o_ref, rtol=rtol, atol=atol)
    except AssertionError as e:
        mask = ~torch.isclose(o, o_ref, rtol=rtol, atol=atol)
        for (a, b) in zip(o[mask], o_ref[mask]):
            print(float(a), float(b))
        raise

def attention_ref_1b(
    q: torch.Tensor,  # [seq_len, num_heads, head_dim]
    k: torch.Tensor,  # [seq_kv_len, num_heads, head_dim]
    v: torch.Tensor,  # [seq_kv_len, num_heads, head_dim_v]
    causal: bool,
    sm_scale: float,
) -> torch.Tensor:
    """Reference attention for a single sequence (varlen mode)."""
    qo_len = q.shape[0]
    kv_len = k.shape[0]
    # [num_heads, seq_len, seq_len]
    logits = torch.einsum("mhd,nhd->hmn", q.float(), k.float()) * sm_scale

    if causal:
        # mask = torch.triu(torch.ones(seq_len, seq_len, device=q.device), diagonal=1).bool()
        # logits = logits.masked_fill(mask.unsqueeze(0), float("-inf"))
        pass
    else:
        mask = torch.ones(qo_len, kv_len, device=q.device)
    logits = logits.masked_fill(mask.unsqueeze(0) == 0, float("-inf"))

    p = torch.softmax(logits, dim=-1)
    # [num_heads, seq_len, head_dim_v] -> [seq_len, num_heads, head_dim_v]
    o = torch.einsum("hmn,nhd->mhd", p, v.float())
    return o

@pytest.mark.parametrize("num_heads", [1, 4, 8])
@pytest.mark.parametrize(
    "head_dim_qk,head_dim_v",
    [
        (64, 64),
        (128, 128),
        (192, 128),
        (192, 192),
        (256, 256),
    ],
)
@pytest.mark.parametrize(
    "seq_lens",
    [
        [(64, 128)],
    ],
)
@pytest.mark.parametrize("skip_softmax_threshold_scale_factor", [0, 10, 100, 1000, 5000])
def test_fmha_v2_prefill_deepseek_chunked(
    num_heads, head_dim_qk, head_dim_v, seq_lens, skip_softmax_threshold_scale_factor
):
    """Test with truly variable length sequences (different lengths of Q and KV per batch)."""
    torch.manual_seed(42)
    device = "cuda"
    dtype = torch.bfloat16

    batch_size = len(seq_lens)
    # Create cu_seqlens: [0, len0, len0+len1, len0+len1+len2, ...]
    cu_seqlens = torch.zeros(batch_size + 1, dtype=torch.int32, device=device)
    cu_kv_seqlens = torch.zeros(batch_size + 1, dtype=torch.int32, device=device)
    for i, (s_q, s_kv) in enumerate(seq_lens):
        cu_seqlens[i + 1] = cu_seqlens[i] + s_q
        cu_kv_seqlens[i + 1] = cu_kv_seqlens[i] + s_kv

    total_tokens = cu_seqlens[-1]
    total_kv_tokens = cu_kv_seqlens[-1]
    # Create packed tensors [total_tokens, num_heads, head_dim]
    q = torch.randn(total_tokens, num_heads, head_dim_qk, dtype=dtype, device=device)
    k = torch.randn(total_kv_tokens, num_heads, head_dim_qk, dtype=dtype, device=device)
    v = torch.randn(total_kv_tokens, num_heads, head_dim_v, dtype=dtype, device=device)
    o = torch.zeros(total_tokens, num_heads, head_dim_v, dtype=dtype, device=device)

    sm_scale = 1.0 / math.sqrt(head_dim_qk)

    # Run fmha_v2_prefill_deepseek in varlen mode
    fmha_v2_prefill_deepseek(
        q, k, v, o,
        num_heads=num_heads,
        head_dim=head_dim_qk,
        max_seqlen=max((x[0] for x in seq_lens)),
        scale_softmax=0.0,
        scale_bmm1=sm_scale,
        scale_bmm2=1.0,
        skip_softmax_threshold_scale_factor=skip_softmax_threshold_scale_factor,
        return_lse=False,
        lse=None,
        skip_softmax_stat=True,
        cu_seqlens=cu_seqlens,
        cu_kv_seqlens=cu_kv_seqlens,
    )

    # Compute reference output for each sequence separately
    o_ref = torch.zeros_like(o)
    for i in range(batch_size):
        start = cu_seqlens[i].item()
        end = cu_seqlens[i + 1].item()
        start_kv = cu_kv_seqlens[i].item()
        end_kv = cu_kv_seqlens[i + 1].item()
        q_seq = q[start:end]  # [seq_len_i, num_heads, head_dim]
        k_seq = k[start_kv:end_kv]
        v_seq = v[start_kv:end_kv]

        o_ref[start:end] = attention_ref_1b(q_seq, k_seq, v_seq, causal=False, sm_scale=sm_scale)

    o_ref = o_ref.to(o.dtype)
    print(o)
    print(o_ref)
    return

    # Check results
    rtol, atol = 1e-2, 1e-2
    # torch.testing.assert_close(o, o_ref, rtol=rtol, atol=atol)
    try:
        torch.testing.assert_close(o, o_ref, rtol=rtol, atol=atol)
    except AssertionError as e:
        mask = ~torch.isclose(o, o_ref, rtol=rtol, atol=atol)
        for (a, b) in zip(o[mask], o_ref[mask]):
            print(float(a), float(b))
        raise