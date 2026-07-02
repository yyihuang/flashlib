typedef unsigned char      uint8_t;
typedef unsigned short     uint16_t;
typedef unsigned int       uint32_t;
typedef unsigned long long uint64_t;
typedef signed int         int32_t;
typedef short int          int16_t;

#include <cuda_bf16.h>

#define TMEM_NCOLS 128
#define TMEM_ACC_OFFSET 0
#define NUM_MAIN_STAGES 1
#define SMEM_SMEM_A_OFF 1024
#define SMEM_SMEM_A_STAGE_BYTES 16384
#define SMEM_SMEM_A_STRIDE 16384
#define SMEM_SMEM_B_OFF 17408
#define SMEM_SMEM_B_STAGE_BYTES 32768
#define SMEM_SMEM_B_STRIDE 32768
#define SMEM_SMEM_Q_NORM_PART_OFF 50176
#define SMEM_SMEM_Q_NORM_PART_STAGE_BYTES 65536
#define SMEM_SMEM_Q_NORM_PART_STRIDE 65536
#define SMEM_SMEM_DB_NORM_PART_OFF 115712
#define SMEM_SMEM_DB_NORM_PART_STAGE_BYTES 2048
#define SMEM_SMEM_DB_NORM_PART_STRIDE 2048
#define SMEM_SMEM_DB_NORM_OFF 117760
#define SMEM_SMEM_DB_NORM_STAGE_BYTES 512
#define SMEM_SMEM_DB_NORM_STRIDE 512
#define SMEM_SMEM_LOCAL_D_OFF 118272
#define SMEM_SMEM_LOCAL_D_STAGE_BYTES 20480
#define SMEM_SMEM_LOCAL_D_STRIDE 20480
#define SMEM_SMEM_LOCAL_I_OFF 138752
#define SMEM_SMEM_LOCAL_I_STAGE_BYTES 20480
#define SMEM_SMEM_LOCAL_I_STRIDE 20480
#define SMEM_TOTAL 159488
#define THREADS 512
#define K_MAX_ 10

#include <math_constants.h>
#define LOOM_INF CUDART_INF_F

__device__ __forceinline__ uint32_t elect_sync() {
    uint32_t pred = 0;
    asm volatile(
        "{\n\t"
        ".reg .pred %%px;\n\t"
        "elect.sync _|%%px, %1;\n\t"
        "@%%px mov.s32 %0, 1;\n\t"
        "}\n"
        : "+r"(pred)
        : "r"(0xFFFFFFFF));
    return pred;
}


__device__ __forceinline__ void mbarrier_init(int mbar_addr, int count) {
    asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;"
        :: "r"(mbar_addr), "r"(count));
}


__device__ __forceinline__ uint32_t mbarrier_try_wait(int mbar_addr, int phase) {
    uint32_t token;
    asm volatile(
        "{\n\t"
        ".reg .pred P1;\n\t"
        "mbarrier.try_wait.parity.shared::cta.b64"
        " P1, [%1], %2;\n\t"
        "selp.u32 %0, 1, 0, P1;\n\t"
        "}\n"
        : "=r"(token)
        : "r"(mbar_addr), "r"(phase) : "memory");
    return token;
}

__device__ __forceinline__ void mbarrier_wait(int mbar_addr, int phase) {
    uint32_t ticks = 0x989680;
    asm volatile(
        "{\n\t"
        ".reg .pred P1;\n\t"
        "LAB_WAIT:\n\t"
        "mbarrier.try_wait.parity.acquire.cta.shared::cta.b64"
        " P1, [%0], %1, %2;\n\t"
        "@P1 bra.uni DONE;\n\t"
        "bra.uni LAB_WAIT;\n\t"
        "DONE:\n\t"
        "}\n"
        :: "r"(mbar_addr), "r"(phase), "r"(ticks) : "memory");
}

__device__ __forceinline__ void mbarrier_wait_token(int mbar_addr, int phase, uint32_t token) {
    if (token == 0) {
        mbarrier_wait(mbar_addr, phase);
    }
}


__device__ __forceinline__ void tcgen05_mma_f16(
    int taddr, uint64_t a_desc, uint64_t b_desc,
    uint32_t i_desc, int enable_input_d) {
    asm volatile(
        "{\n\t"
        ".reg .pred p;\n\t"
        "setp.ne.b32 p, %4, 0;\n\t"
        "tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, p;\n\t"
        "}\n"
        :: "r"(taddr), "l"(a_desc), "l"(b_desc),
           "r"(i_desc), "r"(enable_input_d));
}


__device__ __forceinline__ uint64_t desc_encode(uint64_t x) {
    return (x & 0x3FFFFULL) >> 4ULL;
}


__device__ __forceinline__ void mma_ss_step(
    int a_lo, int b_lo, int taddr, uint32_t i_desc, int enable_d) {
    asm volatile(
        "{\n\t"
        ".reg .pred leader, p;\n\t"
        ".reg .b32 dhi;\n\t"
        ".reg .b64 da, db;\n\t"
        "elect.sync _|leader, 0xFFFFFFFF;\n\t"
        "setp.ne.b32 p, %4, 0;\n\t"
        "mov.b32 dhi, 0x40004040;\n\t"
        "mov.b64 da, {%0, dhi};\n\t"
        "mov.b64 db, {%1, dhi};\n\t"
        "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, %3, p;\n\t"
        "}\n"
        :: "r"(a_lo), "r"(b_lo), "r"(taddr), "r"(i_desc), "r"(enable_d));
}


__device__ __forceinline__ void elect_commit(int mbar_addr) {
    asm volatile(
        "{\n\t"
        ".reg .pred leader;\n\t"
        "elect.sync _|leader, 0xFFFFFFFF;\n\t"
        "@leader tcgen05.commit.cta_group::1.mbarrier::arrive::one"
        ".shared::cluster.b64 [%0];\n\t"
        "}\n"
        :: "r"(mbar_addr));
}


__device__ __forceinline__ void mbarrier_arrive(int mbar_addr) {
    asm volatile(
        "mbarrier.arrive.release.cta.shared::cta.b64 _, [%0];"
        :: "r"(mbar_addr) : "memory");
}


__device__ __forceinline__ void mbarrier_arrive_expect_tx(int mbar_addr, uint32_t bytes) {
    asm volatile(
        "mbarrier.arrive.expect_tx.release.cta.shared::cta.b64 _, [%0], %1;"
        :: "r"(mbar_addr), "r"(bytes) : "memory");
}


__device__ __forceinline__ void tmem_ld_x32(float* dst, int tmem_addr) {
    asm volatile(
        "tcgen05.ld.sync.aligned.32x32b.x32.b32"
        " {%0, %1, %2, %3, %4, %5, %6, %7,"
        "  %8, %9, %10, %11, %12, %13, %14, %15,"
        "  %16, %17, %18, %19, %20, %21, %22, %23,"
        "  %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
        : "=f"(dst[0]),  "=f"(dst[1]),  "=f"(dst[2]),  "=f"(dst[3]),
          "=f"(dst[4]),  "=f"(dst[5]),  "=f"(dst[6]),  "=f"(dst[7]),
          "=f"(dst[8]),  "=f"(dst[9]),  "=f"(dst[10]), "=f"(dst[11]),
          "=f"(dst[12]), "=f"(dst[13]), "=f"(dst[14]), "=f"(dst[15]),
          "=f"(dst[16]), "=f"(dst[17]), "=f"(dst[18]), "=f"(dst[19]),
          "=f"(dst[20]), "=f"(dst[21]), "=f"(dst[22]), "=f"(dst[23]),
          "=f"(dst[24]), "=f"(dst[25]), "=f"(dst[26]), "=f"(dst[27]),
          "=f"(dst[28]), "=f"(dst[29]), "=f"(dst[30]), "=f"(dst[31])
        : "r"(tmem_addr));
}


__device__ __forceinline__ void mbarrier_init_pred(int mbar_addr, uint32_t count, uint32_t pred) {
    asm volatile(
        "{\n\t"
        ".reg .pred p;\n\t"
        "setp.ne.b32 p, %2, 0;\n\t"
        "@p mbarrier.init.shared::cta.b64 [%0], %1;\n\t"
        "}\n" :: "r"(mbar_addr), "r"(count), "r"(pred));
}


__device__ __forceinline__ float max_noftz(float a, float b) {
    float c;
    asm("max.f32 %0, %1, %2;" : "=f"(c) : "f"(a), "f"(b));
    return c;
}


__device__ __forceinline__ void fence_async_shared() {
    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
}


__device__ __forceinline__ void tcgen05_commit(int mbar_addr) {
    asm volatile(
        "tcgen05.commit.cta_group::1.mbarrier::arrive::one"
        ".shared::cluster.b64 [%0];"
        :: "r"(mbar_addr) : "memory");
}


__device__ __forceinline__ uint32_t make_warp_uniform(uint32_t val) {
    uint32_t result;
    asm volatile("shfl.sync.idx.b32 %0, %1, 0, 0x1f, 0xffffffff;"
        : "=r"(result) : "r"(val));
    return result;
}

extern "C" {

__global__ __launch_bounds__(512) void
kernel_knn_search_d4096_q4q8_m8192m16384_k10_partial_0623_5ff7_v1(__nv_bfloat16* __restrict__ queries, __nv_bfloat16* __restrict__ database, float* __restrict__ partial_distances, int32_t* __restrict__ partial_indices, int B, int Q, int M, int split_m, int num_q_tiles, int total_m_tiles)
{
    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;

    extern __shared__ __align__(1024) char smem_raw[];
    int smem;
    smem = (int)(unsigned long long)__cvta_generic_to_shared(smem_raw);
    const int smem_smem_a = smem + 1024;
    const int smem_smem_b = smem + 17408;
    const int smem_smem_q_norm_part = smem + 50176;
    const int smem_smem_db_norm_part = smem + 115712;
    const int smem_smem_db_norm = smem + 117760;
    const int smem_smem_local_d = smem + 118272;
    const int smem_smem_local_i = smem + 138752;

    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    // Mbarrier init (1 groups, 1 barriers)
    // Mbarriers at smem_raw[0..8)

    if (warp == 0) {
        uint32_t leader = elect_sync();
        // mma_done: 1 barriers, init_count=1
        mbarrier_init_pred(smem + 0, 1, leader);
        asm volatile("fence.mbarrier_init.release.cluster;");
    }

    __syncthreads();

    // TMEM alloc (128 columns, 128 used)
    volatile int* tmem_addr_storage = (volatile int*)(smem_raw + 8);
    if (warp == 0) {
        int _tmem_hold = smem + 8;
        asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;" :: "r"(_tmem_hold), "r"(128) : "memory");
    }

    __syncthreads();
    asm volatile("tcgen05.fence::after_thread_sync;");

    const int warp_id = warp;
    const int lane_id = lane;
    __nv_bfloat16* smem_a = (__nv_bfloat16*)(smem_raw + 1024);
    #define smem_a_addr (smem + 1024)
    __nv_bfloat16* smem_b = (__nv_bfloat16*)(smem_raw + 17408);
    #define smem_b_addr (smem + 17408)
    float* smem_q_norm_part = (float*)(smem_raw + 50176);
    #define smem_q_norm_part_addr (smem + 50176)
    float* smem_db_norm_part = (float*)(smem_raw + 115712);
    #define smem_db_norm_part_addr (smem + 115712)
    float* smem_db_norm = (float*)(smem_raw + 117760);
    #define smem_db_norm_addr (smem + 117760)
    float* smem_local_d = (float*)(smem_raw + 118272);
    #define smem_local_d_addr (smem + 118272)
    int* smem_local_i = (int*)(smem_raw + 138752);
    #define smem_local_i_addr (smem + 138752)
    const int mbar_base = smem;
    #define mma_done_addr (mbar_base + 0)
    const int taddr = tmem_addr_storage[0];

    const int tmem_row_base = (warp % 16) * 32;
    const int my_row = tmem_row_base + (lane / 4);
    const int tmem_acc = taddr + TMEM_ACC_OFFSET;
    // === Task calls (dependency order) ===
    int _desc_lo_0 = make_warp_uniform((smem_a_addr >> 4) & 0x3FFF);
    int _desc_lo_1 = make_warp_uniform((smem_b_addr >> 4) & 0x3FFF);
    int work_id = bid;
    int split_id = work_id % split_m;
    int q_tile_linear = work_id / split_m;
    int batch_id = q_tile_linear / num_q_tiles;
    int q_tile = q_tile_linear - batch_id * num_q_tiles;
    int q_start = q_tile * 64;
    int row_top = 0;
    int row_bot = 0;
    int slot = 0;
    int col_origin = 0;
    float q_norm_top = 0.0f;
    float q_norm_bot = 0.0f;
    float best_top_d[10];
    float best_bot_d[10];
    int best_top_i[10];
    int best_bot_i[10];
    #pragma unroll 1
    for (int e_vec = tid; e_vec < 16384; e_vec += 512) {
        int q_row = e_vec / 256;
        int q_part = e_vec - q_row * 256;
        int d_col = q_part * 16;
        int q_abs = q_start + q_row;
        float q_vals[16];
        #pragma unroll
        for (int vi = 0; vi < 16; vi++) {
            q_vals[vi] = 0.0f;
        }
        if (batch_id < B) {
            if (q_abs < Q) {
                {
                    const uint4* _vptr_0 = reinterpret_cast<const uint4*>(queries + (unsigned long long)((batch_id * Q + q_abs) * 4096 + d_col));
                    uint4 _vld_0[2];
                    #pragma unroll
                    for (int _blk = 0; _blk < 2; _blk++) {
                        _vld_0[_blk] = _vptr_0[_blk];
                        __nv_bfloat16* _velems_0 = reinterpret_cast<__nv_bfloat16*>(&_vld_0[_blk]);
                        #pragma unroll
                        for (int _j = 0; _j < 8; _j++)
                            q_vals[0 + _blk * 8 + _j] = __bfloat162float(_velems_0[_j]);
                    }
                }
            }
        }
        float q_norm_part = 0.0f;
        #pragma unroll
        for (int vi = 0; vi < 16; vi++) {
            q_norm_part += q_vals[vi] * q_vals[vi];
        }
        smem_q_norm_part[q_row * 256 + q_part] = q_norm_part;
    }
    __syncthreads();
    if (warp < 8) {
        const int row_group = warp % 4;
        const int col_block = warp / 4;
        const int logical_row_origin = row_group * 16;
        row_top = logical_row_origin + lane / 4;
        row_bot = row_top + 8;
        const int lane_col = lane % 4;
        slot = col_block * 4 + lane_col;
        col_origin = col_block * 64;
        #pragma unroll
        for (int part = 0; part < 256; part++) {
            q_norm_top += smem_q_norm_part[row_top * 256 + part];
            q_norm_bot += smem_q_norm_part[row_bot * 256 + part];
        }
    }
    #pragma unroll
    for (int kk = 0; kk < K_MAX_; kk++) {
        best_top_d[kk] = LOOM_INF;
        best_bot_d[kk] = LOOM_INF;
        best_top_i[kk] = -1;
        best_bot_i[kk] = -1;
    }
    int tile_begin = split_id * total_m_tiles / split_m;
    int next_split = split_id + 1;
    int tile_end = next_split * total_m_tiles / split_m;
    uint32_t _phase_mma_done_0 = 0;
    #pragma unroll 1
    for (int m_tile = tile_begin; m_tile < tile_end; m_tile++) {
        int m_start = m_tile * 128;
        #pragma unroll 1
        for (int e_vec = tid; e_vec < 512; e_vec += 512) {
            int q_elem = e_vec * 16;
            int q_row = q_elem / 128;
            int d_col = q_elem - q_row * 128;
            int global_d = d_col + 0;
            int q_abs = q_start + q_row;
            float q_vals[16];
            unsigned int q_pack[8];
            #pragma unroll
            for (int vi = 0; vi < 16; vi++) {
                q_vals[vi] = 0.0f;
            }
            if (batch_id < B) {
                if (q_abs < Q) {
                    {
                        const uint4* _vptr_1 = reinterpret_cast<const uint4*>(queries + (unsigned long long)((batch_id * Q + q_abs) * 4096 + global_d));
                        uint4 _vld_1[2];
                        #pragma unroll
                        for (int _blk = 0; _blk < 2; _blk++) {
                            _vld_1[_blk] = _vptr_1[_blk];
                            __nv_bfloat16* _velems_1 = reinterpret_cast<__nv_bfloat16*>(&_vld_1[_blk]);
                            #pragma unroll
                            for (int _j = 0; _j < 8; _j++)
                                q_vals[0 + _blk * 8 + _j] = __bfloat162float(_velems_1[_j]);
                        }
                    }
                }
            }
            #pragma unroll
            for (int _lp = 0; _lp < 8; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(q_vals[_lp*2 + 0], q_vals[_lp*2+1 + 0]));
                q_pack[_lp] = *(uint32_t*)&_bf2;
            }
            int q_store_addr = (smem_a_addr + (d_col / 64 * 8192 + q_row * 128 + d_col % 64 * 2 ^ (d_col / 64 * 8192 + q_row * 128 + d_col % 64 * 2 >> 7 & 7) << 4));
            asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(q_store_addr), "r"(q_pack[0]), "r"(q_pack[1]), "r"(q_pack[2]), "r"(q_pack[3]) : "memory");
            int q_store_addr_hi = (smem_a_addr + ((d_col + 8) / 64 * 8192 + q_row * 128 + (d_col + 8) % 64 * 2 ^ ((d_col + 8) / 64 * 8192 + q_row * 128 + (d_col + 8) % 64 * 2 >> 7 & 7) << 4));
            asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(q_store_addr_hi), "r"(q_pack[4]), "r"(q_pack[5]), "r"(q_pack[6]), "r"(q_pack[7]) : "memory");
        }
        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
        __syncthreads();
        int norm_row = tid % 128;
        int norm_part = tid / 128;
        int d_base = norm_part * 32;
        int m_abs_part = m_start + norm_row;
        float acc_part = 0.0f;
        if (tid < 512) {
            #pragma unroll 1
            for (int vv = 0; vv < 2; vv++) {
                int d_col = d_base + vv * 16;
                int global_d = d_col + 0;
                float db_vals[16];
                unsigned int db_pack[8];
                #pragma unroll
                for (int vi = 0; vi < 16; vi++) {
                    db_vals[vi] = 0.0f;
                }
                if (batch_id < B) {
                    if (m_abs_part < M) {
                        {
                            const uint4* _vptr_2 = reinterpret_cast<const uint4*>(database + (unsigned long long)((batch_id * M + m_abs_part) * 4096 + global_d));
                            uint4 _vld_2[2];
                            #pragma unroll
                            for (int _blk = 0; _blk < 2; _blk++) {
                                _vld_2[_blk] = _vptr_2[_blk];
                                __nv_bfloat16* _velems_2 = reinterpret_cast<__nv_bfloat16*>(&_vld_2[_blk]);
                                #pragma unroll
                                for (int _j = 0; _j < 8; _j++)
                                    db_vals[0 + _blk * 8 + _j] = __bfloat162float(_velems_2[_j]);
                            }
                        }
                    }
                }
                #pragma unroll
                for (int _lp = 0; _lp < 8; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(db_vals[_lp*2 + 0], db_vals[_lp*2+1 + 0]));
                    db_pack[_lp] = *(uint32_t*)&_bf2;
                }
                int b_store_addr = (smem_b_addr + (d_col / 64 * 16384 + norm_row * 128 + d_col % 64 * 2 ^ (d_col / 64 * 16384 + norm_row * 128 + d_col % 64 * 2 >> 7 & 7) << 4));
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(b_store_addr), "r"(db_pack[0]), "r"(db_pack[1]), "r"(db_pack[2]), "r"(db_pack[3]) : "memory");
                int b_store_addr_hi = (smem_b_addr + ((d_col + 8) / 64 * 16384 + norm_row * 128 + (d_col + 8) % 64 * 2 ^ ((d_col + 8) / 64 * 16384 + norm_row * 128 + (d_col + 8) % 64 * 2 >> 7 & 7) << 4));
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(b_store_addr_hi), "r"(db_pack[4]), "r"(db_pack[5]), "r"(db_pack[6]), "r"(db_pack[7]) : "memory");
                #pragma unroll
                for (int vi = 0; vi < 16; vi++) {
                    acc_part += db_vals[vi] * db_vals[vi];
                }
            }
            smem_db_norm_part[tid] = acc_part;
        }
        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
        __syncthreads();
        if (tid < 128) {
            int m_abs = m_start + tid;
            float pass_norm = LOOM_INF;
            if (m_abs < M) {
                pass_norm = 0.0f;
                #pragma unroll
                for (int part = 0; part < 4; part++) {
                    pass_norm += smem_db_norm_part[tid + part * 128];
                }
            }
            {
                smem_db_norm[tid] = pass_norm;
            }
        }
        __syncthreads();
        if (warp == 0) {
            asm volatile(
            "{\n\t"
            ".reg .pred leader, p0, p1;\n\t"
            ".reg .b32 adhi, bdhi, alo, blo, id;\n\t"
            ".reg .b64 da, db;\n\t"
            "elect.sync _|leader, 0xFFFFFFFF;\n\t"
            "setp.ne.b32 p0, %3, 0;\n\t"
            "setp.ne.b32 p1, 1, 0;\n\t"
            ""
            "mov.b32 adhi, 0x40004040;\n\t"
            "mov.b32 bdhi, 0x40004040;\n\t"
            "mov.b32 id, 69207184;\n\t"
            "mov.b32 alo, %0;\n\t"
            "mov.b32 blo, %1;\n\t"
            "mov.b64 da, {alo, adhi};\n\t"
            "mov.b64 db, {blo, bdhi};\n\t"
            "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p0;\n\t"
            "add.u32 alo, alo, 2;\n\t"
            "add.u32 blo, blo, 2;\n\t"
            "mov.b64 da, {alo, adhi};\n\t"
            "mov.b64 db, {blo, bdhi};\n\t"
            "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
            "add.u32 alo, alo, 2;\n\t"
            "add.u32 blo, blo, 2;\n\t"
            "mov.b64 da, {alo, adhi};\n\t"
            "mov.b64 db, {blo, bdhi};\n\t"
            "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
            "add.u32 alo, alo, 2;\n\t"
            "add.u32 blo, blo, 2;\n\t"
            "mov.b64 da, {alo, adhi};\n\t"
            "mov.b64 db, {blo, bdhi};\n\t"
            "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
            "add.u32 alo, alo, 506;\n\t"
            "add.u32 blo, blo, 1018;\n\t"
            "mov.b64 da, {alo, adhi};\n\t"
            "mov.b64 db, {blo, bdhi};\n\t"
            "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
            "add.u32 alo, alo, 2;\n\t"
            "add.u32 blo, blo, 2;\n\t"
            "mov.b64 da, {alo, adhi};\n\t"
            "mov.b64 db, {blo, bdhi};\n\t"
            "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
            "add.u32 alo, alo, 2;\n\t"
            "add.u32 blo, blo, 2;\n\t"
            "mov.b64 da, {alo, adhi};\n\t"
            "mov.b64 db, {blo, bdhi};\n\t"
            "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
            "add.u32 alo, alo, 2;\n\t"
            "add.u32 blo, blo, 2;\n\t"
            "mov.b64 da, {alo, adhi};\n\t"
            "mov.b64 db, {blo, bdhi};\n\t"
            "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
            "}\n"
            :: "r"(_desc_lo_0), "r"(_desc_lo_1), "r"(taddr), "r"(0));
            elect_commit(mma_done_addr);
        }
        mbarrier_wait(mma_done_addr, _phase_mma_done_0);
        _phase_mma_done_0 ^= 1;
        #pragma unroll
        for (int d_pass = 1; d_pass < 32; d_pass++) {
            const int d_offset = d_pass * 128;
            #pragma unroll 1
            for (int e_vec = tid; e_vec < 512; e_vec += 512) {
                int q_elem = e_vec * 16;
                int q_row = q_elem / 128;
                int d_col = q_elem - q_row * 128;
                int global_d = d_col + d_offset;
                int q_abs = q_start + q_row;
                float q_vals[16];
                unsigned int q_pack[8];
                #pragma unroll
                for (int vi = 0; vi < 16; vi++) {
                    q_vals[vi] = 0.0f;
                }
                if (batch_id < B) {
                    if (q_abs < Q) {
                        {
                            const uint4* _vptr_3 = reinterpret_cast<const uint4*>(queries + (unsigned long long)((batch_id * Q + q_abs) * 4096 + global_d));
                            uint4 _vld_3[2];
                            #pragma unroll
                            for (int _blk = 0; _blk < 2; _blk++) {
                                _vld_3[_blk] = _vptr_3[_blk];
                                __nv_bfloat16* _velems_3 = reinterpret_cast<__nv_bfloat16*>(&_vld_3[_blk]);
                                #pragma unroll
                                for (int _j = 0; _j < 8; _j++)
                                    q_vals[0 + _blk * 8 + _j] = __bfloat162float(_velems_3[_j]);
                            }
                        }
                    }
                }
                #pragma unroll
                for (int _lp = 0; _lp < 8; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(q_vals[_lp*2 + 0], q_vals[_lp*2+1 + 0]));
                    q_pack[_lp] = *(uint32_t*)&_bf2;
                }
                int q_store_addr = (smem_a_addr + (d_col / 64 * 8192 + q_row * 128 + d_col % 64 * 2 ^ (d_col / 64 * 8192 + q_row * 128 + d_col % 64 * 2 >> 7 & 7) << 4));
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(q_store_addr), "r"(q_pack[0]), "r"(q_pack[1]), "r"(q_pack[2]), "r"(q_pack[3]) : "memory");
                int q_store_addr_hi = (smem_a_addr + ((d_col + 8) / 64 * 8192 + q_row * 128 + (d_col + 8) % 64 * 2 ^ ((d_col + 8) / 64 * 8192 + q_row * 128 + (d_col + 8) % 64 * 2 >> 7 & 7) << 4));
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(q_store_addr_hi), "r"(q_pack[4]), "r"(q_pack[5]), "r"(q_pack[6]), "r"(q_pack[7]) : "memory");
            }
            asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
            __syncthreads();
            int norm_row_0 = tid % 128;
            int norm_part_1 = tid / 128;
            int d_base_2 = norm_part_1 * 32;
            int m_abs_part_3 = m_start + norm_row_0;
            float acc_part_4 = 0.0f;
            if (tid < 512) {
                #pragma unroll 1
                for (int vv = 0; vv < 2; vv++) {
                    int d_col = d_base_2 + vv * 16;
                    int global_d = d_col + d_offset;
                    float db_vals[16];
                    unsigned int db_pack[8];
                    #pragma unroll
                    for (int vi = 0; vi < 16; vi++) {
                        db_vals[vi] = 0.0f;
                    }
                    if (batch_id < B) {
                        if (m_abs_part_3 < M) {
                            {
                                const uint4* _vptr_4 = reinterpret_cast<const uint4*>(database + (unsigned long long)((batch_id * M + m_abs_part_3) * 4096 + global_d));
                                uint4 _vld_4[2];
                                #pragma unroll
                                for (int _blk = 0; _blk < 2; _blk++) {
                                    _vld_4[_blk] = _vptr_4[_blk];
                                    __nv_bfloat16* _velems_4 = reinterpret_cast<__nv_bfloat16*>(&_vld_4[_blk]);
                                    #pragma unroll
                                    for (int _j = 0; _j < 8; _j++)
                                        db_vals[0 + _blk * 8 + _j] = __bfloat162float(_velems_4[_j]);
                                }
                            }
                        }
                    }
                    #pragma unroll
                    for (int _lp = 0; _lp < 8; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(db_vals[_lp*2 + 0], db_vals[_lp*2+1 + 0]));
                        db_pack[_lp] = *(uint32_t*)&_bf2;
                    }
                    int b_store_addr = (smem_b_addr + (d_col / 64 * 16384 + norm_row_0 * 128 + d_col % 64 * 2 ^ (d_col / 64 * 16384 + norm_row_0 * 128 + d_col % 64 * 2 >> 7 & 7) << 4));
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(b_store_addr), "r"(db_pack[0]), "r"(db_pack[1]), "r"(db_pack[2]), "r"(db_pack[3]) : "memory");
                    int b_store_addr_hi = (smem_b_addr + ((d_col + 8) / 64 * 16384 + norm_row_0 * 128 + (d_col + 8) % 64 * 2 ^ ((d_col + 8) / 64 * 16384 + norm_row_0 * 128 + (d_col + 8) % 64 * 2 >> 7 & 7) << 4));
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(b_store_addr_hi), "r"(db_pack[4]), "r"(db_pack[5]), "r"(db_pack[6]), "r"(db_pack[7]) : "memory");
                    #pragma unroll
                    for (int vi = 0; vi < 16; vi++) {
                        acc_part_4 += db_vals[vi] * db_vals[vi];
                    }
                }
                smem_db_norm_part[tid] = acc_part_4;
            }
            asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
            __syncthreads();
            if (tid < 128) {
                int m_abs = m_start + tid;
                float pass_norm = LOOM_INF;
                if (m_abs < M) {
                    pass_norm = 0.0f;
                    #pragma unroll
                    for (int part = 0; part < 4; part++) {
                        pass_norm += smem_db_norm_part[tid + part * 128];
                    }
                }
                {
                    if (m_abs < M) {
                        smem_db_norm[tid] = smem_db_norm[tid] + pass_norm;
                    } else {
                        smem_db_norm[tid] = LOOM_INF;
                    }
                }
            }
            __syncthreads();
            if (warp == 0) {
                asm volatile(
            "{\n\t"
            ".reg .pred leader, p0, p1;\n\t"
            ".reg .b32 adhi, bdhi, alo, blo, id;\n\t"
            ".reg .b64 da, db;\n\t"
            "elect.sync _|leader, 0xFFFFFFFF;\n\t"
            "setp.ne.b32 p0, %3, 0;\n\t"
            "setp.ne.b32 p1, 1, 0;\n\t"
            ""
            "mov.b32 adhi, 0x40004040;\n\t"
            "mov.b32 bdhi, 0x40004040;\n\t"
            "mov.b32 id, 69207184;\n\t"
            "mov.b32 alo, %0;\n\t"
            "mov.b32 blo, %1;\n\t"
            "mov.b64 da, {alo, adhi};\n\t"
            "mov.b64 db, {blo, bdhi};\n\t"
            "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p0;\n\t"
            "add.u32 alo, alo, 2;\n\t"
            "add.u32 blo, blo, 2;\n\t"
            "mov.b64 da, {alo, adhi};\n\t"
            "mov.b64 db, {blo, bdhi};\n\t"
            "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
            "add.u32 alo, alo, 2;\n\t"
            "add.u32 blo, blo, 2;\n\t"
            "mov.b64 da, {alo, adhi};\n\t"
            "mov.b64 db, {blo, bdhi};\n\t"
            "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
            "add.u32 alo, alo, 2;\n\t"
            "add.u32 blo, blo, 2;\n\t"
            "mov.b64 da, {alo, adhi};\n\t"
            "mov.b64 db, {blo, bdhi};\n\t"
            "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
            "add.u32 alo, alo, 506;\n\t"
            "add.u32 blo, blo, 1018;\n\t"
            "mov.b64 da, {alo, adhi};\n\t"
            "mov.b64 db, {blo, bdhi};\n\t"
            "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
            "add.u32 alo, alo, 2;\n\t"
            "add.u32 blo, blo, 2;\n\t"
            "mov.b64 da, {alo, adhi};\n\t"
            "mov.b64 db, {blo, bdhi};\n\t"
            "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
            "add.u32 alo, alo, 2;\n\t"
            "add.u32 blo, blo, 2;\n\t"
            "mov.b64 da, {alo, adhi};\n\t"
            "mov.b64 db, {blo, bdhi};\n\t"
            "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
            "add.u32 alo, alo, 2;\n\t"
            "add.u32 blo, blo, 2;\n\t"
            "mov.b64 da, {alo, adhi};\n\t"
            "mov.b64 db, {blo, bdhi};\n\t"
            "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
            "}\n"
            :: "r"(_desc_lo_0), "r"(_desc_lo_1), "r"(taddr), "r"(1));
                elect_commit(mma_done_addr);
            }
            mbarrier_wait(mma_done_addr, _phase_mma_done_0);
            _phase_mma_done_0 ^= 1;
        }
        if (warp < 8) {
            const int row_group2 = warp % 4;
            const int tmem_row_origin = row_group2 * 32;
            float dots[32];
            asm volatile(
                "tcgen05.ld.sync.aligned.16x256b.x8.b32"
                " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                : "=r"(*reinterpret_cast<uint32_t*>(&dots[0])), "=r"(*reinterpret_cast<uint32_t*>(&dots[1])), "=r"(*reinterpret_cast<uint32_t*>(&dots[2])), "=r"(*reinterpret_cast<uint32_t*>(&dots[3])), "=r"(*reinterpret_cast<uint32_t*>(&dots[4])), "=r"(*reinterpret_cast<uint32_t*>(&dots[5])), "=r"(*reinterpret_cast<uint32_t*>(&dots[6])), "=r"(*reinterpret_cast<uint32_t*>(&dots[7])), "=r"(*reinterpret_cast<uint32_t*>(&dots[8])), "=r"(*reinterpret_cast<uint32_t*>(&dots[9])), "=r"(*reinterpret_cast<uint32_t*>(&dots[10])), "=r"(*reinterpret_cast<uint32_t*>(&dots[11])), "=r"(*reinterpret_cast<uint32_t*>(&dots[12])), "=r"(*reinterpret_cast<uint32_t*>(&dots[13])), "=r"(*reinterpret_cast<uint32_t*>(&dots[14])), "=r"(*reinterpret_cast<uint32_t*>(&dots[15])), "=r"(*reinterpret_cast<uint32_t*>(&dots[16])), "=r"(*reinterpret_cast<uint32_t*>(&dots[17])), "=r"(*reinterpret_cast<uint32_t*>(&dots[18])), "=r"(*reinterpret_cast<uint32_t*>(&dots[19])), "=r"(*reinterpret_cast<uint32_t*>(&dots[20])), "=r"(*reinterpret_cast<uint32_t*>(&dots[21])), "=r"(*reinterpret_cast<uint32_t*>(&dots[22])), "=r"(*reinterpret_cast<uint32_t*>(&dots[23])), "=r"(*reinterpret_cast<uint32_t*>(&dots[24])), "=r"(*reinterpret_cast<uint32_t*>(&dots[25])), "=r"(*reinterpret_cast<uint32_t*>(&dots[26])), "=r"(*reinterpret_cast<uint32_t*>(&dots[27])), "=r"(*reinterpret_cast<uint32_t*>(&dots[28])), "=r"(*reinterpret_cast<uint32_t*>(&dots[29])), "=r"(*reinterpret_cast<uint32_t*>(&dots[30])), "=r"(*reinterpret_cast<uint32_t*>(&dots[31]))
                : "r"(taddr + (tmem_row_origin << 16) + col_origin)
                : "memory");
            asm volatile("tcgen05.wait::ld.sync.aligned;");
            #pragma unroll
            for (int repeat = 0; repeat < 8; repeat++) {
                const int reg_base = repeat * 4;
                int col_base = col_origin + repeat * 8 + lane % 4 * 2;
                int m_abs0 = m_start + col_base;
                int m_abs1 = m_abs0 + 1;
                float top_d0 = q_norm_top + smem_db_norm[col_base] - 2.0f * dots[reg_base];
                float top_d1 = q_norm_top + smem_db_norm[col_base + 1] - 2.0f * dots[reg_base + 1];
                top_d0 = max_noftz(top_d0, 0.0f);
                top_d1 = max_noftz(top_d1, 0.0f);
                int top_take1 = ((top_d1 < top_d0) ? 1 : 0);
                if (((top_take1 != 0) ? top_d1 : top_d0) < best_top_d[9]) {
                    best_top_d[9] = ((top_take1 != 0) ? top_d1 : top_d0);
                    best_top_i[9] = ((top_take1 != 0) ? m_abs1 : m_abs0);
                    #pragma unroll
                    for (int kk = 8; kk >= 0; kk--) {
                        float lower0_d = best_top_d[kk + 1];
                        int lower0_i = best_top_i[kk + 1];
                        float upper0_d = best_top_d[kk];
                        int upper0_i = best_top_i[kk];
                        int swap0_up = ((lower0_d < upper0_d) ? 1 : 0);
                        best_top_d[kk] = ((swap0_up != 0) ? lower0_d : upper0_d);
                        best_top_i[kk] = ((swap0_up != 0) ? lower0_i : upper0_i);
                        best_top_d[kk + 1] = ((swap0_up != 0) ? upper0_d : lower0_d);
                        best_top_i[kk + 1] = ((swap0_up != 0) ? upper0_i : lower0_i);
                    }
                    if (((top_take1 != 0) ? top_d0 : top_d1) < best_top_d[9]) {
                        best_top_d[9] = ((top_take1 != 0) ? top_d0 : top_d1);
                        best_top_i[9] = ((top_take1 != 0) ? m_abs0 : m_abs1);
                        #pragma unroll
                        for (int kk = 8; kk >= 0; kk--) {
                            float lower1_d = best_top_d[kk + 1];
                            int lower1_i = best_top_i[kk + 1];
                            float upper1_d = best_top_d[kk];
                            int upper1_i = best_top_i[kk];
                            int swap1_up = ((lower1_d < upper1_d) ? 1 : 0);
                            best_top_d[kk] = ((swap1_up != 0) ? lower1_d : upper1_d);
                            best_top_i[kk] = ((swap1_up != 0) ? lower1_i : upper1_i);
                            best_top_d[kk + 1] = ((swap1_up != 0) ? upper1_d : lower1_d);
                            best_top_i[kk + 1] = ((swap1_up != 0) ? upper1_i : lower1_i);
                        }
                    }
                }
                float bot_d0 = q_norm_bot + smem_db_norm[col_base] - 2.0f * dots[reg_base + 2];
                float bot_d1 = q_norm_bot + smem_db_norm[col_base + 1] - 2.0f * dots[reg_base + 3];
                bot_d0 = max_noftz(bot_d0, 0.0f);
                bot_d1 = max_noftz(bot_d1, 0.0f);
                int bot_take1 = ((bot_d1 < bot_d0) ? 1 : 0);
                if (((bot_take1 != 0) ? bot_d1 : bot_d0) < best_bot_d[9]) {
                    best_bot_d[9] = ((bot_take1 != 0) ? bot_d1 : bot_d0);
                    best_bot_i[9] = ((bot_take1 != 0) ? m_abs1 : m_abs0);
                    #pragma unroll
                    for (int kk = 8; kk >= 0; kk--) {
                        float lower0_d = best_bot_d[kk + 1];
                        int lower0_i = best_bot_i[kk + 1];
                        float upper0_d = best_bot_d[kk];
                        int upper0_i = best_bot_i[kk];
                        int swap0_up = ((lower0_d < upper0_d) ? 1 : 0);
                        best_bot_d[kk] = ((swap0_up != 0) ? lower0_d : upper0_d);
                        best_bot_i[kk] = ((swap0_up != 0) ? lower0_i : upper0_i);
                        best_bot_d[kk + 1] = ((swap0_up != 0) ? upper0_d : lower0_d);
                        best_bot_i[kk + 1] = ((swap0_up != 0) ? upper0_i : lower0_i);
                    }
                    if (((bot_take1 != 0) ? bot_d0 : bot_d1) < best_bot_d[9]) {
                        best_bot_d[9] = ((bot_take1 != 0) ? bot_d0 : bot_d1);
                        best_bot_i[9] = ((bot_take1 != 0) ? m_abs0 : m_abs1);
                        #pragma unroll
                        for (int kk = 8; kk >= 0; kk--) {
                            float lower1_d = best_bot_d[kk + 1];
                            int lower1_i = best_bot_i[kk + 1];
                            float upper1_d = best_bot_d[kk];
                            int upper1_i = best_bot_i[kk];
                            int swap1_up = ((lower1_d < upper1_d) ? 1 : 0);
                            best_bot_d[kk] = ((swap1_up != 0) ? lower1_d : upper1_d);
                            best_bot_i[kk] = ((swap1_up != 0) ? lower1_i : upper1_i);
                            best_bot_d[kk + 1] = ((swap1_up != 0) ? upper1_d : lower1_d);
                            best_bot_i[kk + 1] = ((swap1_up != 0) ? upper1_i : lower1_i);
                        }
                    }
                }
            }
        }
    }
    if (warp < 8) {
        int top_slot_base = (row_top * 8 + slot) * K_MAX_;
        int bot_slot_base = (row_bot * 8 + slot) * K_MAX_;
        #pragma unroll
        for (int kk = 0; kk < K_MAX_; kk++) {
            smem_local_d[top_slot_base + kk] = best_top_d[kk];
            smem_local_i[top_slot_base + kk] = best_top_i[kk];
            smem_local_d[bot_slot_base + kk] = best_bot_d[kk];
            smem_local_i[bot_slot_base + kk] = best_bot_i[kk];
        }
    }
    __syncthreads();
    if (tid < 64) {
        int row = tid;
        int q_global = q_start + row;
        if (q_global < Q) {
            float head_d[8];
            int head_i[8];
            int head_k[8];
            #pragma unroll
            for (int slot_idx = 0; slot_idx < 8; slot_idx++) {
                int local_base = (row * 8 + slot_idx) * K_MAX_;
                head_k[slot_idx] = 0;
                head_d[slot_idx] = smem_local_d[local_base];
                head_i[slot_idx] = smem_local_i[local_base];
            }
            unsigned long long partial_base = (unsigned long long)((((batch_id * num_q_tiles + q_tile) * split_m + split_id) * 128 + row) * K_MAX_);
            #pragma unroll
            for (int out_k = 0; out_k < K_MAX_; out_k++) {
                float winner_d = head_d[0];
                int winner_i = head_i[0];
                int winner_slot = 0;
                #pragma unroll
                for (int slot_idx = 1; slot_idx < 8; slot_idx++) {
                    float cand_d = head_d[slot_idx];
                    int take = ((cand_d < winner_d) ? 1 : 0);
                    winner_d = ((take != 0) ? cand_d : winner_d);
                    winner_i = ((take != 0) ? head_i[slot_idx] : winner_i);
                    winner_slot = ((take != 0) ? slot_idx : winner_slot);
                }
                partial_distances[partial_base + out_k] = winner_d;
                partial_indices[partial_base + out_k] = winner_i;
                #pragma unroll
                for (int slot_idx = 0; slot_idx < 8; slot_idx++) {
                    if (winner_slot == slot_idx) {
                        int next_head = head_k[slot_idx] + 1;
                        head_k[slot_idx] = next_head;
                        head_d[slot_idx] = LOOM_INF;
                        head_i[slot_idx] = -1;
                        if (next_head < K_MAX_) {
                            int local_base = (row * 8 + slot_idx) * K_MAX_;
                            head_d[slot_idx] = smem_local_d[local_base + next_head];
                            head_i[slot_idx] = smem_local_i[local_base + next_head];
                        }
                    }
                }
            }
        } else {
            unsigned long long partial_base_invalid = (unsigned long long)((((batch_id * num_q_tiles + q_tile) * split_m + split_id) * 128 + row) * K_MAX_);
            #pragma unroll
            for (int kk = 0; kk < K_MAX_; kk++) {
                partial_distances[partial_base_invalid + kk] = LOOM_INF;
                partial_indices[partial_base_invalid + kk] = -1;
            }
        }
    }

    // Cleanup
    __syncthreads();

    if (warp == 0) {
        asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;" :: "r"(tmem_addr_storage[0]), "r"(128));
        asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;");
    }
}

} // extern "C"

