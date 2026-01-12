/*
 * Copyright (c) 2023-2025 by FlashInfer team.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <float.h>
#include <fused_multihead_attention.h>
#include <fused_multihead_attention_utils.h>
#include <fused_multihead_attention_api.h>
#include <math.h>

#include <algorithm>
#include <cassert>
#include <cstring>
#include <numeric>

#include "pytorch_extension_utils.h"

using Attention_mask_type = fmha::Attention_mask_type;
using Attention_input_layout = fmha::Attention_input_layout;
using Kv_block_array = fmha::Kv_block_array;

////////////////////////////////////////////////////////////////////////////////////////////////////
namespace flashinfer {

static inline void set_params(bert::Fused_multihead_attention_params_v2& params,
                              const Launch_params launch_params,
                              // types
                              Data_type data_type, Data_type acc_type, Data_type output_dtype,
                              // attention input layout
                              Attention_input_layout input_layout,
                              // sizes
                              const size_t b, const size_t s_q, const size_t s_kv, const size_t h,
                              const size_t h_kv, const size_t d, const size_t dv,
                              const size_t total, const size_t num_grouped_heads,
                              const size_t sliding_window_size, const size_t chunked_attention_size,
                              // paged kv cache block size.
                              const size_t tokens_per_block,
                              // device pointers
                              void* qkv_packed_d,
                              // contiguous q.
                              void* q_d,
                              // separate k.
                              void* k_d,
                              // separate v.
                              void* v_d,
                              // contiguous kv.
                              void* kv_d,
                              // start address of the paged kv pool.
                              void* paged_kv_pool_ptr,
                              // offsets for different blocks in terms of the start address.
                              int32_t* paged_block_offsets,
                              // mask input.
                              void* packed_mask_d, void* cu_mask_rows_d,
                              // attention sinks.
                              void* attention_sinks_d, void* cu_kv_seqlens_d, void* cu_q_seqlens_d,
                              void* o_packed_d, void* p_d, void* s_d, void* softmax_stats_d,
                              void* scale_bmm2_d,
                              // scale factors
                              float const scale_bmm1, float const scale_softmax,
                              float const scale_bmm2, float const softcapping_scale_bmm1,
                              // flags
                              bool const use_int8_scale_max, bool const interleaved,
                              bool const is_s_padded, bool const has_alibi) {
  memset(&params, 0, sizeof(params));

  params.o_ptr = o_packed_d;
  params.o_stride_in_bytes = get_size_in_bytes(h * dv, output_dtype);

  if (interleaved) {
    params.q_stride_in_bytes = total;
    params.o_stride_in_bytes = total;
  }

  if (input_layout == Attention_input_layout::PACKED_QKV) {
    // For grouped- or multi-query attention (h denotes num_q_heads; h' denotes h_kv):
    //   qkv_layout = [b, s, [q_hd, k_h'd, v_h'd]]
    //   qkv_stride = (h+2*h')d * bytes_per_elt
    // Otherwise:
    //   qkv_layout = [b, s, 3, h, d] or [b, s, h, 3, d]
    //   qkv_stride = 3hd * bytes_per_elt
    params.qkv_ptr = qkv_packed_d;
    params.q_stride_in_bytes = params.k_stride_in_bytes = params.v_stride_in_bytes =
        get_size_in_bytes(h * d + h_kv * d + h_kv * dv, data_type);
  } else {
    // Layout [B, S, H, D].
    params.q_ptr = q_d;
    params.q_stride_in_bytes = get_size_in_bytes(h * d, data_type);

    if (input_layout == Attention_input_layout::CONTIGUOUS_Q_KV) {
      // Layout [B, S, 2, H, D].
      params.kv_ptr = kv_d;
      params.k_stride_in_bytes = params.v_stride_in_bytes =
          get_size_in_bytes(h_kv * (d + dv), data_type);
    } else if (input_layout == Attention_input_layout::Q_PAGED_KV) {
      int max_blocks_per_sequence = (s_kv + tokens_per_block - 1) / tokens_per_block;
      params.paged_kv_cache =
          Kv_block_array(b, max_blocks_per_sequence, tokens_per_block,
                         get_size_in_bytes(tokens_per_block * h_kv * std::gcd(d, dv), data_type),
                         paged_kv_pool_ptr);
      params.paged_kv_cache.mBlockOffsets = paged_block_offsets;
      params.k_stride_in_bytes = get_size_in_bytes(tokens_per_block * d, data_type);
      params.v_stride_in_bytes = get_size_in_bytes(tokens_per_block * dv, data_type);
    } else if (input_layout == Attention_input_layout::SEPARATE_Q_K_V) {
      // Layout [B, S, H_kv, D].
      params.k_ptr = k_d;
      // Layout [B, S, H_kv, Dv].
      params.v_ptr = v_d;
      params.k_stride_in_bytes = get_size_in_bytes(h_kv * d, data_type);
      params.v_stride_in_bytes = get_size_in_bytes(h_kv * dv, data_type);
    }
  }

  // Packed mask.
  params.packed_mask_ptr = packed_mask_d;
  // The N dimension has to be aligned.
  params.packed_mask_stride_in_bytes =
      (align_to(int64_t(s_kv), int64_t(fmha::FLASH_ATTEN_MASK_N_ALIGNMENT))) / 8;

  // Attention sinks.
  params.attention_sinks = reinterpret_cast<float*>(attention_sinks_d);

#if defined(STORE_P)
  params.p_ptr = p_d;
  params.p_stride_in_bytes = get_size_in_bytes(b * h * s_kv, acc_type);
#endif  // defined(STORE_P)

#if defined(STORE_S)
  params.s_ptr = s_d;
  params.s_stride_in_bytes = get_size_in_bytes(b * h * s_kv, data_type);
#endif  // defined(STORE_S)

  params.softmax_stats_ptr = softmax_stats_d;
  params.softmax_stats_stride_in_bytes = get_size_in_bytes(h * 2, DATA_TYPE_FP32);

  // Set the dimensions.
  params.b = b;
  params.h = h;
  params.s = s_q;
  params.d = d;
  params.dv = dv;
  params.num_grouped_heads = num_grouped_heads;
  params.sliding_window_size = sliding_window_size;
  assert((chunked_attention_size == 0 ||
          (chunked_attention_size & (chunked_attention_size - 1)) == 0) &&
         "chunked_attention_size has to be a power of 2");
  params.log2_chunked_attention_size =
      chunked_attention_size > 0 ? std::log2(chunked_attention_size) : 0;

  // cumulative q or kv sequence lengths.
  params.cu_q_seqlens = static_cast<int*>(cu_q_seqlens_d);
  params.cu_kv_seqlens = static_cast<int*>(cu_kv_seqlens_d);
  // cumulative mask sequence lengths.
  params.cu_mask_rows = static_cast<int*>(cu_mask_rows_d);

  // Set the different scale values.
  Data_type scale_type1 =
      (data_type == DATA_TYPE_FP16) || (data_type == DATA_TYPE_BF16) ? acc_type : DATA_TYPE_FP32;
  Data_type scale_softmax_type = scale_type1;
  Data_type scale_type2 =
      (data_type == DATA_TYPE_FP16) || (data_type == DATA_TYPE_BF16) ? data_type : DATA_TYPE_FP32;
  if (data_type == DATA_TYPE_E4M3) {
    scale_type1 = acc_type;
    scale_type2 = acc_type;
  }

  // Fuse 1.0f / softcapping_scale into scale_bmm1.
  bool const enable_attn_logit_softcapping = softcapping_scale_bmm1 != 0.f;
  float fused_scale_bmm1 =
      enable_attn_logit_softcapping ? scale_bmm1 / softcapping_scale_bmm1 : scale_bmm1;

  // use specialized hopper kernels without alibi support.
  // alibi or softcapping_scale cannot utilize the exp2f with fused_scale optimization.
  if (launch_params.warp_specialization && !has_alibi && !enable_attn_logit_softcapping) {
    set_alpha(params.scale_bmm1, fused_scale_bmm1 * float(M_LOG2E), DATA_TYPE_FP32);
  } else {
    set_alpha(params.scale_bmm1, fused_scale_bmm1, scale_type1);
  }
  set_alpha(params.scale_softmax, scale_softmax, scale_softmax_type);
  set_alpha(params.scale_bmm2, scale_bmm2, scale_type2);
  params.scale_bmm2_d = reinterpret_cast<uint32_t*>(scale_bmm2_d);
  params.softcapping_scale_bmm1 = softcapping_scale_bmm1;

  // Cache scale_bmm2 to avoid repeated cudaMemcpy when value unchanged
  static uint32_t cached_scale_bmm2 = 0;
  static bool scale_bmm2_initialized = false;
  if (!scale_bmm2_initialized || cached_scale_bmm2 != params.scale_bmm2) {
    FMHA_CHECK_CUDA(cudaMemcpy(params.scale_bmm2_d, &params.scale_bmm2, sizeof(uint32_t),
                               cudaMemcpyHostToDevice));
    cached_scale_bmm2 = params.scale_bmm2;
    scale_bmm2_initialized = true;
  }

  // attention type, h_kv < h if MQA or GQA
  params.h_kv = h_kv;
  assert(h % h_kv == 0 && "MQA/GQA needs h to be divisible by h_kv!");
  params.h_q_per_kv = h / h_kv;
  params.has_alibi = has_alibi;
  params.alibi_params = fmha::AlibiParams(h);

  // Set flags
  params.is_s_padded = is_s_padded;
  params.use_int8_scale_max = use_int8_scale_max;

  // Do we enable the trick to replace I2F with FP math in the 2nd GEMM?
  if (data_type == DATA_TYPE_INT8) {
    params.enable_i2f_trick = -double(1 << 22) * double(scale_bmm2) <= -128.f &&
                              double(1 << 22) * double(scale_bmm2) >= 127.f;
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

static inline void determine_launch_params(
    Launch_params& launch_params, Data_type data_type, int sm, const size_t s, const size_t d,
    const Attention_mask_type attention_mask_type, const Attention_input_layout input_layout,
    bool const interleaved, bool const ignore_b1opt, bool const force_unroll, bool const use_tma,
    bool const force_non_flash_attention, bool const force_non_warp_specialization,
    bool const force_non_granular_tiling, bool const force_fp32_acc,
    // device props
    const cudaDeviceProp props) {
  // Set launch params to choose kernels
  launch_params.ignore_b1opt = ignore_b1opt;
  launch_params.force_unroll = force_unroll;
  launch_params.force_fp32_acc = force_fp32_acc;
  launch_params.interleaved = interleaved;
  launch_params.attention_mask_type = attention_mask_type;
  launch_params.attention_input_layout = input_layout;

  // Set SM count and L2 cache size (used to determine launch blocks/grids to maximum performance)
  launch_params.multi_processor_count = props.multiProcessorCount;
  launch_params.device_l2_cache_size = props.l2CacheSize;

  // threshold for adopting flash attention or warp_specialized kernels.
  launch_params.flash_attention =
      (data_type == DATA_TYPE_FP16 || data_type == DATA_TYPE_BF16 || data_type == DATA_TYPE_E4M3) &&
      (s >= 16 && d >= 16) && !force_non_flash_attention;

  // enable warp_speialized kernels when s >= 512 on hopper
  // note that warp_speialized kernels need flash attention + tma
  launch_params.warp_specialization =
      (data_type == DATA_TYPE_FP16 || data_type == DATA_TYPE_BF16 || data_type == DATA_TYPE_E4M3) &&
      sm == 90 && launch_params.flash_attention && !force_non_warp_specialization;
  // warp specialization kernels on hopper need tma
  launch_params.use_tma = use_tma || launch_params.warp_specialization;

  // use granular tiling on Ampere-style flash attention
  launch_params.use_granular_tiling = !force_non_granular_tiling && launch_params.flash_attention &&
                                      !launch_params.warp_specialization && sm >= 80;

  if (launch_params.use_granular_tiling && (data_type == DATA_TYPE_E4M3 && sm == 80)) {
    printf(
        "Fallback to non-granular-tiling kernels as tiled e4m3 kernels"
        "are not supported on Ada currently.\n");
    launch_params.use_granular_tiling = false;
  }
}

/**
 * @brief TVM FFI binding for TRTLLM FMHA V2 kernel for MLA attention
 *
 * This function calls a specific TRTLLM kernel variant with:
 * - Input type: E4M3 (8-bit floating point)
 * - Accumulator type: FP32
 * - Warp configuration: 64x64
 * - Layout: Separate Q, K, V tensors
 * - Q/K dimension: 192, V dimension: 128 (MLA-specific)
 * - Output type: BF16
 * - Target architecture: SM120 (Blackwell)
 *
 * @param q Query tensor [batch, q_seqlen, num_heads, 192] in E4M3
 * @param k Key tensor [batch, kv_seqlen, num_kv_heads, 192] in E4M3
 * @param v Value tensor [batch, kv_seqlen, num_kv_heads, 128] in E4M3
 * @param o Output tensor [batch, q_seqlen, num_heads, 128] in BF16
 * @param maybe_lse Optional log-sum-exp tensor for softmax statistics
 * @param num_heads Number of query heads
 * @param head_dim Head dimension (must be 192 for this kernel)
 * @param seq_len Sequence length (not used, extracted from tensor shapes)
 * @param scale_softmax Softmax scale factor
 * @param scale_bmm1 Scale factor for the first GEMM
 * @param scale_bmm2 Scale factor for the second GEMM
 * @param is_e4m3 Whether the input is E4M3
 * @param is_bf16_output Whether the output is BF16
 * @param maybe_cu_seqlens Optional cumulative sequence lengths for varlen mode [batch_size + 1]
 *
 * Supports two modes:
 * - Padded mode (maybe_cu_seqlens is None): q/k/v shape [batch, seq_len, heads, dim]
 * - Varlen mode (maybe_cu_seqlens provided): q/k/v shape [total_tokens, heads, dim]
 */
void TRTLLMFMHAv2Run(at::Tensor q, at::Tensor k, at::Tensor v, at::Tensor o,
                     std::optional<at::Tensor> maybe_lse, int64_t num_heads, int64_t head_dim,
                     int64_t seq_len, const double scale_softmax, const double scale_bmm1,
                     const double scale_bmm2, bool is_e4m3, bool is_bf16_output,
                     const double skip_softmax_threshold_scale_factor,
                     std::optional<at::Tensor> maybe_cu_seqlens = std::nullopt,
                     std::optional<at::Tensor> maybe_cu_kv_seqlens = std::nullopt) {
  // Determine if we're in varlen mode
  const bool is_varlen = maybe_cu_seqlens.has_value();

  int batch_size, q_seqlen, kv_seqlen, total_tokens, total_kv_tokens, num_kv_heads, head_dim_v;

  if (is_varlen) {
    // Varlen mode: q/k/v shape [total_tokens, num_heads, head_dim]
    batch_size = maybe_cu_seqlens.value().size(0) - 1;
    total_tokens = q.size(0);
    total_kv_tokens = k.size(0);
    q_seqlen = seq_len;  // max_seqlen passed as seq_len parameter
    kv_seqlen = 0;       // no need
    assert(num_heads == q.size(1) &&
           "num_heads must be equal to the number of heads in the query tensor");
    num_kv_heads = k.size(1);
    assert(head_dim == q.size(2) &&
           "head_dim must be equal to the head dimension in the query tensor");
    head_dim_v = v.size(2);
  } else {
    // Padded mode: q/k/v shape [batch_size, seq_len, num_heads, head_dim]
    batch_size = q.size(0);
    q_seqlen = q.size(1);
    kv_seqlen = k.size(1);
    total_tokens = q_seqlen * batch_size;
    total_kv_tokens = kv_seqlen * batch_size;;
    assert(num_heads == q.size(2) &&
           "num_heads must be equal to the number of heads in the query tensor");
    num_kv_heads = k.size(2);
    assert(head_dim == q.size(3) &&
           "head_dim must be equal to the head dimension in the query tensor");
    head_dim_v = v.size(3);
  }

  Data_type data_type = is_e4m3 ? DATA_TYPE_E4M3 : DATA_TYPE_BF16;
  Data_type acc_type = DATA_TYPE_FP32;
  Data_type output_dtype = is_bf16_output ? DATA_TYPE_BF16 : DATA_TYPE_FP16;
  // Attention_mask_type attention_mask_type = Attention_mask_type::CAUSAL;
  Attention_mask_type attention_mask_type = Attention_mask_type::PADDING;
  Attention_input_layout input_layout = Attention_input_layout::SEPARATE_Q_K_V;

  CudaDevice device;
  int sm = device.sm;
  cudaDeviceProp props = device.props;

  // cudaStream_t stream = static_cast<cudaStream_t>(get_stream(q.device()));
  const cudaStream_t stream = c10::cuda::getCurrentCUDAStream();

  Launch_params launch_params;
  determine_launch_params(launch_params, data_type, sm, q_seqlen, head_dim, attention_mask_type,
                          input_layout,
                          false,  // interleaved
                          false,  // ignore_b1opt
                          false,  // force_unroll
                          false,  // use_tma (let determine_launch_params decide)
                          false,  // force_non_flash_attention
                          false,   // force_non_warp_specialization (for non-SM90)
                          false,  // force_non_granular_tiling
                          true,   // force_fp32_acc
                          props);

  launch_params.total_q_seqlen = total_tokens;
  launch_params.total_kv_seqlen = total_kv_tokens;
  launch_params.enable_attn_logit_softcapping = false;

  // Static buffers for small fixed-size allocations to avoid repeated cudaMalloc/cudaFree.
  // These are allocated once and reused across all calls.
  static void* tile_id_counter_d = nullptr;
  static void* scale_bmm2_d = nullptr;
  static bool static_buffers_initialized = false;

  if (!static_buffers_initialized) {
    FMHA_CHECK_CUDA(cudaMalloc(&tile_id_counter_d, sizeof(uint32_t)));
    FMHA_CHECK_CUDA(cudaMalloc(&scale_bmm2_d, sizeof(uint32_t)));
    static_buffers_initialized = true;
  }

  // Reset tile_id_counter to 0 for each call (kernel increments it)
  FMHA_CHECK_CUDA(cudaMemsetAsync(tile_id_counter_d, 0, sizeof(uint32_t), stream));

  // Handle cumulative sequence lengths
  void *cu_seqlens_d, *cu_kv_seqlens_d;
  if (is_varlen) {
    // Varlen mode: use provided cu_seqlens directly
    cu_seqlens_d = maybe_cu_seqlens.value().data_ptr();
    cu_kv_seqlens_d = maybe_cu_kv_seqlens.has_value() ?
        maybe_cu_kv_seqlens.value().data_ptr() : cu_seqlens_d;
  } else {
    // Padded mode: cache cu_seqlens_d to avoid repeated cudaMalloc/cudaMemcpy
    // Only reallocate when batch_size or seq_len changes
    static void* cached_cu_seqlens_d = nullptr;
    static int cached_batch_size = -1;
    static int cached_seqlen = -1;

    if (cached_batch_size != batch_size || cached_seqlen != q_seqlen) {
      // Need to reallocate or update
      if (cached_cu_seqlens_d != nullptr) {
        FMHA_CHECK_CUDA(cudaFree(cached_cu_seqlens_d));
      }

      std::vector<uint32_t> cu_seqlens(batch_size + 1);
      for (int i = 0; i <= batch_size; i++) {
        cu_seqlens[i] = i * q_seqlen;
      }
      FMHA_CHECK_CUDA(cudaMalloc(&cached_cu_seqlens_d, sizeof(uint32_t) * cu_seqlens.size()));
      FMHA_CHECK_CUDA(cudaMemcpy(cached_cu_seqlens_d, cu_seqlens.data(),
                                 sizeof(uint32_t) * cu_seqlens.size(), cudaMemcpyHostToDevice));

      cached_batch_size = batch_size;
      cached_seqlen = q_seqlen;
    }
    cu_seqlens_d = cached_cu_seqlens_d;
  }
  // LSE buffer: [total_tokens, num_heads, 2] for varlen, [batch, seq_len, num_heads, 2] for padded
  if (maybe_lse.has_value()) {
    FMHA_CHECK_CUDA(cudaMemset(maybe_lse.value().data_ptr(), 0,
                               sizeof(float) * total_tokens * num_heads * 2));
  }

  bert::Fused_multihead_attention_params_v2 params;

  set_params(params, launch_params, data_type, acc_type, output_dtype, input_layout,
             batch_size,         // b
             q_seqlen,           // s_q
             kv_seqlen,          // s_kv
             num_heads,          // h
             num_kv_heads,       // h_kv
             head_dim,           // d
             head_dim_v,         // dv
             total_tokens,       // total tokens
             1,                  // num_grouped_heads (not used for regular attention)
             INT_MAX,            // sliding_window_size (disabled)
             0,                  // chunked_attention_size (disabled)
             64,                 // tokens_per_block (not used with SEPARATE_Q_K_V)
             nullptr,            // qkv_packed_d (not used)
             q.data_ptr(),       // q_d
             k.data_ptr(),       // k_d
             v.data_ptr(),       // v_d
             nullptr,            // kv_d (not used)
             nullptr,            // paged_kv_pool_ptr (not used)
             nullptr,            // paged_block_offsets (not used)
             nullptr,            // packed_mask_d (not used for causal)
             nullptr,            // cu_mask_rows_d (not used)
             nullptr,            // attention_sinks_d (not used)
             cu_kv_seqlens_d,    // cu_kv_seqlens_d
             cu_seqlens_d,       // cu_q_seqlens_d
             o.data_ptr(),       // o_packed_d
             nullptr,            // p_d (not storing)
             nullptr,            // s_d (not storing)
             maybe_lse.has_value() ? maybe_lse.value().data_ptr() : nullptr,
             scale_bmm2_d,
             scale_bmm1,     // scale_bmm1
             scale_softmax,  // scale_softmax
             scale_bmm2,     // scale_bmm2
             0.0f,           // softcapping_scale_bmm1 (disabled)
             false,          // use_int8_scale_max
             false,          // interleaved
             !is_varlen,     // is_s_padded (true for padded mode, false for varlen)
             false);         // has_alibi

  params.tile_id_counter_ptr = (uint32_t *)tile_id_counter_d;
  params.skip_softmax_threshold_scale_factor = skip_softmax_threshold_scale_factor;
#ifdef SKIP_SOFTMAX_STAT
  // Static buffer for skip_softmax statistics
  static uint32_t* skip_softmax_stat_d = nullptr;
  static bool skip_softmax_stat_initialized = false;
  if (!skip_softmax_stat_initialized) {
    FMHA_CHECK_CUDA(cudaMalloc(&skip_softmax_stat_d, sizeof(uint32_t) * 2));
    skip_softmax_stat_initialized = true;
  }
  // Reset statistics for each call
  FMHA_CHECK_CUDA(cudaMemsetAsync(skip_softmax_stat_d, 0, sizeof(uint32_t) * 2, stream));
  params.skip_softmax_total_blocks = skip_softmax_stat_d;
  params.skip_softmax_skipped_blocks = skip_softmax_stat_d + 1;
#endif

  run_fmha_v2(params, launch_params, data_type, output_dtype, sm, stream);

#ifdef SKIP_SOFTMAX_STAT
  uint32_t skip_softmax_stat[2];
  FMHA_CHECK_CUDA(cudaMemcpy(skip_softmax_stat, skip_softmax_stat_d,
    sizeof(uint32_t) * 2, cudaMemcpyDeviceToHost));
  printf("Skip-Softmax: Sparsity(@scale = %.2f) = %u/%u = %.2f%%\n",
          skip_softmax_threshold_scale_factor,
          skip_softmax_stat[1], skip_softmax_stat[0],
          100.f * skip_softmax_stat[1] / skip_softmax_stat[0]);
#endif

}

// TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, flashinfer::TRTLLMFMHAv2Run);

}  // namespace flashinfer

TORCH_LIBRARY_FRAGMENT(TORCH_EXTENSION_NAME, m) {
  m.def("run", &flashinfer::TRTLLMFMHAv2Run);
}
