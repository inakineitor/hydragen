def _fwd_kernel_splitK(
    Q,
    K,
    V,
    sm_scale,
    Out_splitK,
    Metadata,
    Seq_len,
    stride_qz,
    stride_qm,
    stride_qg,
    stride_qh,
    stride_qk,
    stride_kz,
    stride_kn,
    stride_kg,
    stride_kh,
    stride_kk,
    stride_vz,
    stride_vn,
    stride_vg,
    stride_vh,
    stride_vk,
    stride_osk_zhg,
    stride_osk_s,
    stride_osk_m,
    stride_osk_k,
    stride_mzhg,
    stride_m2,
    stride_ms,
    stride_mm,
    Z,
    N_CTX_Q,
    N_CTX_K,
    BLOCK_N_PER_SPLIT,
    H: tl.constexpr,
    G: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BOUNDS_CHECKS_N: tl.constexpr,
    USE_SEQ_LEN: tl.constexpr,
    PACKED_PER_VAL: tl.constexpr = 1,
    N_GROUPS: tl.constexpr = 1,
):
    """This kernel can accept non-quantized or int4-quantized keys/values.
    PACKED_PER_VAL determines the quantization type:
        - PACKED_PER_VAL == 1 means no quantization
        - PACKED_PER_VAL == 8 means 4-bit quantization (8 packed quantized values inside one int32)
    For the quantized case K/V should be int32 tensors.
    Quantization can be row-wise (when N_GROUPS = 1) or group-wise with N_GROUPS = 2, 4, or 8.
    Quantization coefficients are stored at the beginning of the row along the last dimension of K/V
    So K[B, H, M, :] has a form
    [   quant_coef0, quant_coef1, ...|
        group0_quant_value0, group0_quant_value1,... |
        group1_quant_value0, group1_quant_value1,...]
    where each quant_coef is an int32 which should be interpreted as 2 packed float16: scale and offset.

    Note: this kernel needs to be processed by xformers.triton.vararg_kernel.unroll_varargs
    before compilation. That will unroll variables marked with "VAR_ARGS_ARRAY" into lists.
    See how FwOp.apply does it below.
    """
    tl.static_assert(
        PACKED_PER_VAL == 1
        and tl.constexpr(K.dtype.element_ty != tl.int32)
        or (PACKED_PER_VAL == 8 and tl.constexpr(K.dtype.element_ty == tl.int32)),
        f"Only 4-bit quantization is supported, K/V should have dtype int32 in the quantized case: PACKED_PER_VAL={PACKED_PER_VAL!r} tl.constexpr(K.dtype)={tl.constexpr(K.dtype)!r} tl.constexpr(K.dtype.element_ty)={tl.constexpr(K.dtype.element_ty)!r}",
    )
    tl.static_assert(
        ((N_GROUPS == 1 or N_GROUPS == 2) or N_GROUPS == 4) or N_GROUPS == 8,
        "Number of quantization groups can be 1 (row-wise quantization), 2, 4, or 8.",
    )
    QUANTIZED: tl.constexpr = PACKED_PER_VAL > 1
    PACKED_D_PER_GROUP: tl.constexpr = BLOCK_DMODEL // PACKED_PER_VAL // N_GROUPS
    D_PER_GROUP: tl.constexpr = BLOCK_DMODEL // N_GROUPS
    start_m = tl.program_id(0)
    off_zhg = tl.program_id(1)
    off_z = off_zhg // (H * G)
    off_h = off_zhg // G % H
    off_g = off_zhg % G
    splitk_idx = tl.program_id(2)
    lo = splitk_idx * BLOCK_N_PER_SPLIT
    if USE_SEQ_LEN:
        kv_len = tl.load(Seq_len + off_z)
    else:
        kv_len = N_CTX_K
    hi = tl.minimum((splitk_idx + 1) * BLOCK_N_PER_SPLIT, kv_len)
    Q_block_ptr = tl.make_block_ptr(
        base=Q + off_h * stride_qh + off_z * stride_qz + off_g * stride_qg,
        shape=(N_CTX_Q, D_PER_GROUP),
        strides=(stride_qm, stride_qk),
        offsets=(start_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, D_PER_GROUP),
        order=(1, 0),
    )
    k_base = K + off_h * stride_kh + off_z * stride_kz + off_g * stride_kg
    K_block_ptr = tl.make_block_ptr(
        base=k_base + stride_kk * QUANTIZED * N_GROUPS,
        shape=(PACKED_D_PER_GROUP, hi),
        strides=(stride_kk, stride_kn),
        offsets=(0, lo),
        block_shape=(PACKED_D_PER_GROUP, BLOCK_N),
        order=(0, 1),
    )
    v_base = V + off_h * stride_vh + off_z * stride_vz + off_g * stride_vg
    V_block_ptr = tl.make_block_ptr(
        base=v_base + stride_vk * QUANTIZED * N_GROUPS,
        shape=(hi, PACKED_D_PER_GROUP),
        strides=(stride_vn, stride_vk),
        offsets=(lo, 0),
        block_shape=(BLOCK_N, PACKED_D_PER_GROUP),
        order=(1, 0),
    )
    if QUANTIZED:
        K_scale_shift_block_ptr = tl.make_block_ptr(
            base=k_base,
            shape=(1, hi),
            strides=(stride_kk, stride_kn),
            offsets=(0, lo),
            block_shape=(1, BLOCK_N),
            order=(0, 1),
        )
        V_scale_shift_block_ptr = tl.make_block_ptr(
            base=v_base,
            shape=(hi, 1),
            strides=(stride_vn, stride_vk),
            offsets=(lo, 0),
            block_shape=(BLOCK_N, 1),
            order=(1, 0),
        )
    else:
        K_scale_shift_block_ptr = None
        V_scale_shift_block_ptr = None
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc0 = tl.zeros([BLOCK_M, D_PER_GROUP], dtype=tl.float32)
    qk_scale = sm_scale * 1.44269504
    q0 = tl.load(tl.advance(Q_block_ptr, (0, 0 * D_PER_GROUP)), boundary_check=(0,))
    for start_n in range(lo, hi, BLOCK_N):
        k0, v0 = load_dequantize_k_v_group(
            K_block_ptr,
            V_block_ptr,
            K_scale_shift_block_ptr,
            V_scale_shift_block_ptr,
            BOUNDS_CHECKS_N,
            PACKED_PER_VAL,
            PACKED_D_PER_GROUP,
            Q.dtype.element_ty,
            0,
        )
        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        qk += tl.dot(q0, k0)
        qk *= qk_scale
        if BOUNDS_CHECKS_N:
            qk = tl.where(tl.arange(0, BLOCK_N) < hi - start_n, qk, float("-inf"))
        m_i_new = tl.maximum(m_i, tl.max(qk, 1))
        alpha = tl.math.exp2(m_i - m_i_new)
        p = tl.math.exp2(qk - m_i_new[:, None])
        l_i = l_i * alpha + tl.sum(p, 1)
        m_i = m_i_new
        p = p.to(Q.dtype.element_ty)
        acc0 *= alpha[:, None]
        acc0 += tl.dot(p, v0)
        K_block_ptr = tl.advance(K_block_ptr, (0, BLOCK_N))
        V_block_ptr = tl.advance(V_block_ptr, (BLOCK_N, 0))
        if PACKED_PER_VAL > 1:
            K_scale_shift_block_ptr = tl.advance(K_scale_shift_block_ptr, (0, BLOCK_N))
            V_scale_shift_block_ptr = tl.advance(V_scale_shift_block_ptr, (BLOCK_N, 0))
    O_block_ptr = tl.make_block_ptr(
        base=Out_splitK + off_zhg * stride_osk_zhg + splitk_idx * stride_osk_s,
        shape=(N_CTX_Q, D_PER_GROUP),
        strides=(stride_osk_m, 1),
        offsets=(start_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, D_PER_GROUP),
        order=(1, 0),
    )
    tl.store(tl.advance(O_block_ptr, (0, 0 * D_PER_GROUP)), acc0, boundary_check=(0,))
    Metadata_ptr = (
        Metadata
        + off_zhg * stride_mzhg
        + splitk_idx * stride_ms
        + start_m * BLOCK_M
        + tl.arange(0, BLOCK_M)
    )
    tl.store(Metadata_ptr, m_i)
    tl.store(Metadata_ptr + stride_m2, l_i)
