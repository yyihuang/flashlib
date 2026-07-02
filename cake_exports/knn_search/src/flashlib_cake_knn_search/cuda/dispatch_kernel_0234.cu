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
#define SMEM_SMEM_A_STAGE_BYTES 32768
#define SMEM_SMEM_A_STRIDE 32768
#define SMEM_SMEM_B_OFF 33792
#define SMEM_SMEM_B_STAGE_BYTES 32768
#define SMEM_SMEM_B_STRIDE 32768
#define SMEM_SMEM_B_NEXT_OFF 66560
#define SMEM_SMEM_B_NEXT_STAGE_BYTES 32768
#define SMEM_SMEM_B_NEXT_STRIDE 32768
#define SMEM_SMEM_DB_NORM_PART_OFF 103424
#define SMEM_SMEM_DB_NORM_PART_STAGE_BYTES 2048
#define SMEM_SMEM_DB_NORM_PART_STRIDE 2048
#define SMEM_SMEM_DB_NORM_PART_NEXT_OFF 105472
#define SMEM_SMEM_DB_NORM_PART_NEXT_STAGE_BYTES 2048
#define SMEM_SMEM_DB_NORM_PART_NEXT_STRIDE 2048
#define SMEM_SMEM_DB_NORM_OFF 107520
#define SMEM_SMEM_DB_NORM_STAGE_BYTES 512
#define SMEM_SMEM_DB_NORM_STRIDE 512
#define SMEM_SMEM_DB_NORM_NEXT_OFF 108032
#define SMEM_SMEM_DB_NORM_NEXT_STAGE_BYTES 512
#define SMEM_SMEM_DB_NORM_NEXT_STRIDE 512
#define SMEM_SMEM_Q_NORM_PART_OFF 99328
#define SMEM_SMEM_Q_NORM_PART_STAGE_BYTES 4096
#define SMEM_SMEM_Q_NORM_PART_STRIDE 4096
#define SMEM_TOTAL 165120
#define THREADS 512
#define K_MAX_ 64

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


__device__ __forceinline__ void fma_f32x2_inplace(float2* a, float2 b, float2 c) {
    unsigned long long r;
    asm("fma.rn.ftz.f32x2 %0, %1, %2, %3;"
        : "=l"(r)
        : "l"(*(unsigned long long*)a), "l"(*(unsigned long long*)&b),
          "l"(*(unsigned long long*)&c));
    *(unsigned long long*)a = r;
}

__device__ __forceinline__ void mul_f32x2_inplace(float2* a, float2 b) {
    asm("mul.rn.ftz.f32x2 %0, %0, %1;"
        : "+l"(*(unsigned long long*)a) : "l"(*(unsigned long long*)&b));
}

__device__ __forceinline__ void add_f32x2_inplace(float2* a, float2 b) {
    asm("add.rn.ftz.f32x2 %0, %0, %1;"
        : "+l"(*(unsigned long long*)a) : "l"(*(unsigned long long*)&b));
}

__device__ __forceinline__ void sub_f32x2_inplace(float2* a, float2 b) {
    asm("sub.rn.ftz.f32x2 %0, %0, %1;"
        : "+l"(*(unsigned long long*)a) : "l"(*(unsigned long long*)&b));
}

__device__ __forceinline__ float2 add_f32x2(float2 a, float2 b) {
    float2 r;
    asm("add.rn.ftz.f32x2 %0, %1, %2;"
        : "=l"(*(unsigned long long*)&r)
        : "l"(*(unsigned long long*)&a), "l"(*(unsigned long long*)&b));
    return r;
}

__device__ __forceinline__ float2 sub_f32x2(float2 a, float2 b) {
    float2 r;
    asm("sub.rn.ftz.f32x2 %0, %1, %2;"
        : "=l"(*(unsigned long long*)&r)
        : "l"(*(unsigned long long*)&a), "l"(*(unsigned long long*)&b));
    return r;
}

__device__ __forceinline__ void fma_scale_x32(
    float* sv, const float2* scale2, const float2* neg_max2)
{
    float2* sv_2 = reinterpret_cast<float2*>(sv);
    #pragma unroll
    for (int j = 0; j < 16; j++)
        fma_f32x2_inplace(&sv_2[j], *scale2, *neg_max2);
}

__device__ __forceinline__ float2 fma_f32x2(float2 a, float2 b, float2 c) {
    float2 r;
    asm("fma.rn.ftz.f32x2 %0, %1, %2, %3;"
        : "=l"(*(unsigned long long*)&r)
        : "l"(*(unsigned long long*)&a), "l"(*(unsigned long long*)&b),
          "l"(*(unsigned long long*)&c));
    return r;
}

__device__ __forceinline__ float2 mul_f32x2(float2 a, float2 b) {
    float2 r;
    asm("mul.rn.ftz.f32x2 %0, %1, %2;"
        : "=l"(*(unsigned long long*)&r)
        : "l"(*(unsigned long long*)&a), "l"(*(unsigned long long*)&b));
    return r;
}

// ex2_emulation_f32x2 defined in softmax_frag_exp2_cast helper (or standalone)


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
kernel_knn_search_k64_q4096split80_twotile_distanceonly_stagingunrolled_partial_0612_r31_11c1_v1(__nv_bfloat16* __restrict__ queries, __nv_bfloat16* __restrict__ database, float* __restrict__ partial_distances, int32_t* __restrict__ partial_indices, int B, int Q, int M, int split_m, int num_q_tiles, int total_m_tiles, int tiles_per_split)
{
    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;

    extern __shared__ __align__(1024) char smem_raw[];
    int smem;
    smem = (int)(unsigned long long)__cvta_generic_to_shared(smem_raw);
    const int smem_smem_a = smem + 1024;
    const int smem_smem_b = smem + 33792;
    const int smem_smem_b_next = smem + 66560;
    const int smem_smem_db_norm_part = smem + 103424;
    const int smem_smem_db_norm_part_next = smem + 105472;
    const int smem_smem_db_norm = smem + 107520;
    const int smem_smem_db_norm_next = smem + 108032;
    const int smem_smem_q_norm_part = smem + 99328;

    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    // Mbarrier init (2 groups, 2 barriers)
    // Mbarriers at smem_raw[0..16)

    if (warp == 0) {
        uint32_t leader = elect_sync();
        // mma_done_first: 1 barriers, init_count=1
        mbarrier_init_pred(smem + 0, 1, leader);
        // mma_done_second: 1 barriers, init_count=1
        mbarrier_init_pred(smem + 8, 1, leader);
        asm volatile("fence.mbarrier_init.release.cluster;");
    }

    __syncthreads();

    // TMEM alloc (128 columns, 128 used)
    volatile int* tmem_addr_storage = (volatile int*)(smem_raw + 16);
    if (warp == 0) {
        int _tmem_hold = smem + 16;
        asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;" :: "r"(_tmem_hold), "r"(128) : "memory");
    }

    __syncthreads();
    asm volatile("tcgen05.fence::after_thread_sync;");

    const int warp_id = warp;
    const int lane_id = lane;
    __nv_bfloat16* smem_a = (__nv_bfloat16*)(smem_raw + 1024);
    #define smem_a_addr (smem + 1024)
    __nv_bfloat16* smem_b = (__nv_bfloat16*)(smem_raw + 33792);
    #define smem_b_addr (smem + 33792)
    __nv_bfloat16* smem_b_next = (__nv_bfloat16*)(smem_raw + 66560);
    #define smem_b_next_addr (smem + 66560)
    float* smem_db_norm_part = (float*)(smem_raw + 103424);
    #define smem_db_norm_part_addr (smem + 103424)
    float* smem_db_norm_part_next = (float*)(smem_raw + 105472);
    #define smem_db_norm_part_next_addr (smem + 105472)
    float* smem_db_norm = (float*)(smem_raw + 107520);
    #define smem_db_norm_addr (smem + 107520)
    float* smem_db_norm_next = (float*)(smem_raw + 108032);
    #define smem_db_norm_next_addr (smem + 108032)
    float* smem_q_norm_part = (float*)(smem_raw + 99328);
    #define smem_q_norm_part_addr (smem + 99328)
    const int mbar_base = smem;
    #define mma_done_first_addr (mbar_base + 0)
    #define mma_done_second_addr (mbar_base + 8)
    const int taddr = tmem_addr_storage[0];

    const int tmem_row_base = (warp % 16) * 32;
    const int my_row = tmem_row_base + (lane / 4);
    const int tmem_acc = taddr + TMEM_ACC_OFFSET;
    // === Task calls (dependency order) ===
    int _desc_lo_0 = make_warp_uniform((smem_a_addr >> 4) & 0x3FFF);
    int _desc_lo_1 = make_warp_uniform((smem_b_addr >> 4) & 0x3FFF);
    int _desc_lo_2 = make_warp_uniform((smem_b_next_addr >> 4) & 0x3FFF);
    int work_id = bid;
    int split_id = work_id % 80;
    int q_tile = work_id / 80;
    int q_start = q_tile * 128;
    const int col_chunk = warp / 4;
    const int row_base_tmem = warp % 4 * 32;
    int q_local = row_base_tmem + lane;
    float q_norm = 0.0f;
    float best_d[K_MAX_];
    int best_i[K_MAX_];
    #pragma unroll
    for (int kk = 0; kk < K_MAX_; kk++) {
        best_d[kk] = LOOM_INF;
        best_i[kk] = -1;
    }
    int first_q_vec = tid;
    int second_q_vec = tid + 512;
    int q_elem = first_q_vec * 16;
    int q_row = q_elem / 128;
    int d_col = q_elem - q_row * 128;
    int q_abs = q_start + q_row;
    float q_vals[16];
    unsigned int q_pack[8];
    {
        const uint4* _vptr_0 = reinterpret_cast<const uint4*>(queries + (unsigned long long)(q_abs * 128 + d_col));
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
    float q_norm_part = 0.0f;
    #pragma unroll
    for (int vi = 0; vi < 16; vi++) {
        q_norm_part += q_vals[vi] * q_vals[vi];
    }
    int q_norm_part_col = d_col / 16;
    smem_q_norm_part[q_row * 8 + q_norm_part_col] = q_norm_part;
    #pragma unroll
    for (int _lp = 0; _lp < 8; _lp++) {
        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(q_vals[_lp*2 + 0], q_vals[_lp*2+1 + 0]));
        q_pack[_lp] = *(uint32_t*)&_bf2;
    }
    int q_store_addr = (smem_a_addr + (d_col / 64 * 16384 + q_row * 128 + d_col % 64 * 2 ^ (d_col / 64 * 16384 + q_row * 128 + d_col % 64 * 2 >> 7 & 7) << 4));
    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(q_store_addr), "r"(q_pack[0]), "r"(q_pack[1]), "r"(q_pack[2]), "r"(q_pack[3]) : "memory");
    int q_store_addr_hi = (smem_a_addr + ((d_col + 8) / 64 * 16384 + q_row * 128 + (d_col + 8) % 64 * 2 ^ ((d_col + 8) / 64 * 16384 + q_row * 128 + (d_col + 8) % 64 * 2 >> 7 & 7) << 4));
    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(q_store_addr_hi), "r"(q_pack[4]), "r"(q_pack[5]), "r"(q_pack[6]), "r"(q_pack[7]) : "memory");
    int q_elem_0 = second_q_vec * 16;
    int q_row_1 = q_elem_0 / 128;
    int d_col_2 = q_elem_0 - q_row_1 * 128;
    int q_abs_3 = q_start + q_row_1;
    float q_vals_4[16];
    unsigned int q_pack_5[8];
    {
        const uint4* _vptr_1 = reinterpret_cast<const uint4*>(queries + (unsigned long long)(q_abs_3 * 128 + d_col_2));
        uint4 _vld_1[2];
        #pragma unroll
        for (int _blk = 0; _blk < 2; _blk++) {
            _vld_1[_blk] = _vptr_1[_blk];
            __nv_bfloat16* _velems_1 = reinterpret_cast<__nv_bfloat16*>(&_vld_1[_blk]);
            #pragma unroll
            for (int _j = 0; _j < 8; _j++)
                q_vals_4[0 + _blk * 8 + _j] = __bfloat162float(_velems_1[_j]);
        }
    }
    float q_norm_part_6 = 0.0f;
    #pragma unroll
    for (int vi = 0; vi < 16; vi++) {
        q_norm_part_6 += q_vals_4[vi] * q_vals_4[vi];
    }
    int q_norm_part_col_7 = d_col_2 / 16;
    smem_q_norm_part[q_row_1 * 8 + q_norm_part_col_7] = q_norm_part_6;
    #pragma unroll
    for (int _lp = 0; _lp < 8; _lp++) {
        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(q_vals_4[_lp*2 + 0], q_vals_4[_lp*2+1 + 0]));
        q_pack_5[_lp] = *(uint32_t*)&_bf2;
    }
    int q_store_addr_8 = (smem_a_addr + (d_col_2 / 64 * 16384 + q_row_1 * 128 + d_col_2 % 64 * 2 ^ (d_col_2 / 64 * 16384 + q_row_1 * 128 + d_col_2 % 64 * 2 >> 7 & 7) << 4));
    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(q_store_addr_8), "r"(q_pack_5[0]), "r"(q_pack_5[1]), "r"(q_pack_5[2]), "r"(q_pack_5[3]) : "memory");
    int q_store_addr_hi_9 = (smem_a_addr + ((d_col_2 + 8) / 64 * 16384 + q_row_1 * 128 + (d_col_2 + 8) % 64 * 2 ^ ((d_col_2 + 8) / 64 * 16384 + q_row_1 * 128 + (d_col_2 + 8) % 64 * 2 >> 7 & 7) << 4));
    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(q_store_addr_hi_9), "r"(q_pack_5[4]), "r"(q_pack_5[5]), "r"(q_pack_5[6]), "r"(q_pack_5[7]) : "memory");
    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
    __syncthreads();
    #pragma unroll
    for (int part = 0; part < 8; part++) {
        q_norm += smem_q_norm_part[q_local * 8 + part];
    }
    int tile_begin = split_id * 157 / 80;
    int next_split = split_id + 1;
    int tile_end = next_split * 157 / 80;
    int first_m_start = tile_begin * 128;
    int norm_row = tid % 128;
    int norm_part = tid / 128;
    int d_base = norm_part * 32;
    int m_abs_part = first_m_start + norm_row;
    float acc_part = 0.0f;
    int d_col0 = d_base;
    float db_vals0[16];
    unsigned int db_pack0[8];
    #pragma unroll
    for (int vi = 0; vi < 16; vi++) {
        db_vals0[vi] = 0.0f;
    }
    if (m_abs_part < 20000) {
        {
            const uint4* _vptr_2 = reinterpret_cast<const uint4*>(database + (unsigned long long)(m_abs_part * 128 + d_col0));
            uint4 _vld_2[2];
            #pragma unroll
            for (int _blk = 0; _blk < 2; _blk++) {
                _vld_2[_blk] = _vptr_2[_blk];
                __nv_bfloat16* _velems_2 = reinterpret_cast<__nv_bfloat16*>(&_vld_2[_blk]);
                #pragma unroll
                for (int _j = 0; _j < 8; _j++)
                    db_vals0[0 + _blk * 8 + _j] = __bfloat162float(_velems_2[_j]);
            }
        }
    }
    #pragma unroll
    for (int _lp = 0; _lp < 8; _lp++) {
        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(db_vals0[_lp*2 + 0], db_vals0[_lp*2+1 + 0]));
        db_pack0[_lp] = *(uint32_t*)&_bf2;
    }
    int b_store_addr0 = (smem_b_addr + (d_col0 / 64 * 16384 + norm_row * 128 + d_col0 % 64 * 2 ^ (d_col0 / 64 * 16384 + norm_row * 128 + d_col0 % 64 * 2 >> 7 & 7) << 4));
    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(b_store_addr0), "r"(db_pack0[0]), "r"(db_pack0[1]), "r"(db_pack0[2]), "r"(db_pack0[3]) : "memory");
    int b_store_addr0_hi = (smem_b_addr + ((d_col0 + 8) / 64 * 16384 + norm_row * 128 + (d_col0 + 8) % 64 * 2 ^ ((d_col0 + 8) / 64 * 16384 + norm_row * 128 + (d_col0 + 8) % 64 * 2 >> 7 & 7) << 4));
    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(b_store_addr0_hi), "r"(db_pack0[4]), "r"(db_pack0[5]), "r"(db_pack0[6]), "r"(db_pack0[7]) : "memory");
    #pragma unroll
    for (int vi = 0; vi < 16; vi++) {
        acc_part += db_vals0[vi] * db_vals0[vi];
    }
    int d_col1 = d_base + 16;
    float db_vals1[16];
    unsigned int db_pack1[8];
    #pragma unroll
    for (int vi = 0; vi < 16; vi++) {
        db_vals1[vi] = 0.0f;
    }
    if (m_abs_part < 20000) {
        {
            const uint4* _vptr_3 = reinterpret_cast<const uint4*>(database + (unsigned long long)(m_abs_part * 128 + d_col1));
            uint4 _vld_3[2];
            #pragma unroll
            for (int _blk = 0; _blk < 2; _blk++) {
                _vld_3[_blk] = _vptr_3[_blk];
                __nv_bfloat16* _velems_3 = reinterpret_cast<__nv_bfloat16*>(&_vld_3[_blk]);
                #pragma unroll
                for (int _j = 0; _j < 8; _j++)
                    db_vals1[0 + _blk * 8 + _j] = __bfloat162float(_velems_3[_j]);
            }
        }
    }
    #pragma unroll
    for (int _lp = 0; _lp < 8; _lp++) {
        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(db_vals1[_lp*2 + 0], db_vals1[_lp*2+1 + 0]));
        db_pack1[_lp] = *(uint32_t*)&_bf2;
    }
    int b_store_addr1 = (smem_b_addr + (d_col1 / 64 * 16384 + norm_row * 128 + d_col1 % 64 * 2 ^ (d_col1 / 64 * 16384 + norm_row * 128 + d_col1 % 64 * 2 >> 7 & 7) << 4));
    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(b_store_addr1), "r"(db_pack1[0]), "r"(db_pack1[1]), "r"(db_pack1[2]), "r"(db_pack1[3]) : "memory");
    int b_store_addr1_hi = (smem_b_addr + ((d_col1 + 8) / 64 * 16384 + norm_row * 128 + (d_col1 + 8) % 64 * 2 ^ ((d_col1 + 8) / 64 * 16384 + norm_row * 128 + (d_col1 + 8) % 64 * 2 >> 7 & 7) << 4));
    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(b_store_addr1_hi), "r"(db_pack1[4]), "r"(db_pack1[5]), "r"(db_pack1[6]), "r"(db_pack1[7]) : "memory");
    #pragma unroll
    for (int vi = 0; vi < 16; vi++) {
        acc_part += db_vals1[vi] * db_vals1[vi];
    }
    smem_db_norm_part[tid] = acc_part;
    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
    __syncthreads();
    if (tid < 128) {
        int m_abs = first_m_start + tid;
        float db_norm = LOOM_INF;
        if (m_abs < 20000) {
            db_norm = 0.0f;
            #pragma unroll
            for (int part = 0; part < 4; part++) {
                db_norm += smem_db_norm_part[tid + part * 128];
            }
        }
        smem_db_norm[tid] = db_norm;
    }
    int second_m_tile = tile_begin + 1;
    int second_m_start = second_m_tile * 128;
    int has_second_tile = ((second_m_tile < tile_end) ? 1 : 0);
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
            "mov.b32 id, 136316048;\n\t"
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
            "add.u32 alo, alo, 1018;\n\t"
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
        elect_commit(mma_done_first_addr);
    }
    if (has_second_tile != 0) {
        int norm_row_0 = tid % 128;
        int norm_part_1 = tid / 128;
        int d_base_2 = norm_part_1 * 32;
        int m_abs_part_3 = second_m_start + norm_row_0;
        float acc_part_4 = 0.0f;
        int d_col0_5 = d_base_2;
        float db_vals0_6[16];
        unsigned int db_pack0_7[8];
        #pragma unroll
        for (int vi = 0; vi < 16; vi++) {
            db_vals0_6[vi] = 0.0f;
        }
        if (m_abs_part_3 < 20000) {
            {
                const uint4* _vptr_4 = reinterpret_cast<const uint4*>(database + (unsigned long long)(m_abs_part_3 * 128 + d_col0_5));
                uint4 _vld_4[2];
                #pragma unroll
                for (int _blk = 0; _blk < 2; _blk++) {
                    _vld_4[_blk] = _vptr_4[_blk];
                    __nv_bfloat16* _velems_4 = reinterpret_cast<__nv_bfloat16*>(&_vld_4[_blk]);
                    #pragma unroll
                    for (int _j = 0; _j < 8; _j++)
                        db_vals0_6[0 + _blk * 8 + _j] = __bfloat162float(_velems_4[_j]);
                }
            }
        }
        #pragma unroll
        for (int _lp = 0; _lp < 8; _lp++) {
            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(db_vals0_6[_lp*2 + 0], db_vals0_6[_lp*2+1 + 0]));
            db_pack0_7[_lp] = *(uint32_t*)&_bf2;
        }
        int b_store_addr0_8 = (smem_b_next_addr + (d_col0_5 / 64 * 16384 + norm_row_0 * 128 + d_col0_5 % 64 * 2 ^ (d_col0_5 / 64 * 16384 + norm_row_0 * 128 + d_col0_5 % 64 * 2 >> 7 & 7) << 4));
        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(b_store_addr0_8), "r"(db_pack0_7[0]), "r"(db_pack0_7[1]), "r"(db_pack0_7[2]), "r"(db_pack0_7[3]) : "memory");
        int b_store_addr0_hi_9 = (smem_b_next_addr + ((d_col0_5 + 8) / 64 * 16384 + norm_row_0 * 128 + (d_col0_5 + 8) % 64 * 2 ^ ((d_col0_5 + 8) / 64 * 16384 + norm_row_0 * 128 + (d_col0_5 + 8) % 64 * 2 >> 7 & 7) << 4));
        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(b_store_addr0_hi_9), "r"(db_pack0_7[4]), "r"(db_pack0_7[5]), "r"(db_pack0_7[6]), "r"(db_pack0_7[7]) : "memory");
        #pragma unroll
        for (int vi = 0; vi < 16; vi++) {
            acc_part_4 += db_vals0_6[vi] * db_vals0_6[vi];
        }
        int d_col1_10 = d_base_2 + 16;
        float db_vals1_11[16];
        unsigned int db_pack1_12[8];
        #pragma unroll
        for (int vi = 0; vi < 16; vi++) {
            db_vals1_11[vi] = 0.0f;
        }
        if (m_abs_part_3 < 20000) {
            {
                const uint4* _vptr_5 = reinterpret_cast<const uint4*>(database + (unsigned long long)(m_abs_part_3 * 128 + d_col1_10));
                uint4 _vld_5[2];
                #pragma unroll
                for (int _blk = 0; _blk < 2; _blk++) {
                    _vld_5[_blk] = _vptr_5[_blk];
                    __nv_bfloat16* _velems_5 = reinterpret_cast<__nv_bfloat16*>(&_vld_5[_blk]);
                    #pragma unroll
                    for (int _j = 0; _j < 8; _j++)
                        db_vals1_11[0 + _blk * 8 + _j] = __bfloat162float(_velems_5[_j]);
                }
            }
        }
        #pragma unroll
        for (int _lp = 0; _lp < 8; _lp++) {
            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(db_vals1_11[_lp*2 + 0], db_vals1_11[_lp*2+1 + 0]));
            db_pack1_12[_lp] = *(uint32_t*)&_bf2;
        }
        int b_store_addr1_13 = (smem_b_next_addr + (d_col1_10 / 64 * 16384 + norm_row_0 * 128 + d_col1_10 % 64 * 2 ^ (d_col1_10 / 64 * 16384 + norm_row_0 * 128 + d_col1_10 % 64 * 2 >> 7 & 7) << 4));
        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(b_store_addr1_13), "r"(db_pack1_12[0]), "r"(db_pack1_12[1]), "r"(db_pack1_12[2]), "r"(db_pack1_12[3]) : "memory");
        int b_store_addr1_hi_14 = (smem_b_next_addr + ((d_col1_10 + 8) / 64 * 16384 + norm_row_0 * 128 + (d_col1_10 + 8) % 64 * 2 ^ ((d_col1_10 + 8) / 64 * 16384 + norm_row_0 * 128 + (d_col1_10 + 8) % 64 * 2 >> 7 & 7) << 4));
        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(b_store_addr1_hi_14), "r"(db_pack1_12[4]), "r"(db_pack1_12[5]), "r"(db_pack1_12[6]), "r"(db_pack1_12[7]) : "memory");
        #pragma unroll
        for (int vi = 0; vi < 16; vi++) {
            acc_part_4 += db_vals1_11[vi] * db_vals1_11[vi];
        }
        smem_db_norm_part_next[tid] = acc_part_4;
        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
        __syncthreads();
        if (tid < 128) {
            int m_abs = second_m_start + tid;
            float db_norm = LOOM_INF;
            if (m_abs < 20000) {
                db_norm = 0.0f;
                #pragma unroll
                for (int part = 0; part < 4; part++) {
                    db_norm += smem_db_norm_part_next[tid + part * 128];
                }
            }
            smem_db_norm_next[tid] = db_norm;
        }
    }
    uint32_t _phase_mma_done_first_0 = 0;
    mbarrier_wait(mma_done_first_addr, _phase_mma_done_first_0);
    _phase_mma_done_first_0 ^= 1;
    const int col_base0 = col_chunk * 32;
    float dots0[32];
    tmem_ld_x32(&dots0[0], taddr + (row_base_tmem << 16) + col_base0);
    asm volatile("tcgen05.wait::ld.sync.aligned;");
    const int slot_base0 = 0;
    #pragma unroll 4
    for (int j_rel = 0; j_rel < 32; j_rel += 8) {
        int j_base0 = col_base0 + j_rel;
        float dist_pair0[2];
        float norm_pair0[2];
        dist_pair0[0] = dots0[j_rel];
        dist_pair0[1] = dots0[j_rel + 1];
        const float2 _fma_b2_6 = {-2.0f, -2.0f};
        const float2 _fma_c2_7 = {q_norm, q_norm};
        #pragma unroll
        for (int _lf = 0; _lf < 1; _lf++)
            fma_f32x2_inplace(&reinterpret_cast<float2*>(dist_pair0)[_lf], _fma_b2_6, _fma_c2_7);
        norm_pair0[0] = smem_db_norm[j_base0];
        norm_pair0[1] = smem_db_norm[j_base0 + 1];
        float _t0[2];
        #pragma unroll
        for (int _la = 0; _la < 1; _la++)
            reinterpret_cast<float2*>(_t0)[_la] = add_f32x2(reinterpret_cast<float2*>(dist_pair0)[_la], reinterpret_cast<const float2*>(norm_pair0)[_la]);
        int m_abs00 = first_m_start + j_base0;
        int m_abs01 = m_abs00 + 1;
        best_d[slot_base0 + j_rel] = _t0[0];
        best_i[slot_base0 + j_rel] = m_abs00;
        best_d[slot_base0 + j_rel + 1] = _t0[1];
        best_i[slot_base0 + j_rel + 1] = m_abs01;
        int j_base1 = j_base0 + 2;
        float dist_pair1[2];
        float norm_pair1[2];
        dist_pair1[0] = dots0[j_rel + 2];
        dist_pair1[1] = dots0[j_rel + 3];
        const float2 _fma_b2_8 = {-2.0f, -2.0f};
        const float2 _fma_c2_9 = {q_norm, q_norm};
        #pragma unroll
        for (int _lf = 0; _lf < 1; _lf++)
            fma_f32x2_inplace(&reinterpret_cast<float2*>(dist_pair1)[_lf], _fma_b2_8, _fma_c2_9);
        norm_pair1[0] = smem_db_norm[j_base1];
        norm_pair1[1] = smem_db_norm[j_base1 + 1];
        float _t1[2];
        #pragma unroll
        for (int _la = 0; _la < 1; _la++)
            reinterpret_cast<float2*>(_t1)[_la] = add_f32x2(reinterpret_cast<float2*>(dist_pair1)[_la], reinterpret_cast<const float2*>(norm_pair1)[_la]);
        int m_abs10 = first_m_start + j_base1;
        int m_abs11 = m_abs10 + 1;
        best_d[slot_base0 + j_rel + 2] = _t1[0];
        best_i[slot_base0 + j_rel + 2] = m_abs10;
        best_d[slot_base0 + j_rel + 3] = _t1[1];
        best_i[slot_base0 + j_rel + 3] = m_abs11;
        int j_base2 = j_base0 + 4;
        float dist_pair2[2];
        float norm_pair2[2];
        dist_pair2[0] = dots0[j_rel + 4];
        dist_pair2[1] = dots0[j_rel + 5];
        const float2 _fma_b2_10 = {-2.0f, -2.0f};
        const float2 _fma_c2_11 = {q_norm, q_norm};
        #pragma unroll
        for (int _lf = 0; _lf < 1; _lf++)
            fma_f32x2_inplace(&reinterpret_cast<float2*>(dist_pair2)[_lf], _fma_b2_10, _fma_c2_11);
        norm_pair2[0] = smem_db_norm[j_base2];
        norm_pair2[1] = smem_db_norm[j_base2 + 1];
        float _t2[2];
        #pragma unroll
        for (int _la = 0; _la < 1; _la++)
            reinterpret_cast<float2*>(_t2)[_la] = add_f32x2(reinterpret_cast<float2*>(dist_pair2)[_la], reinterpret_cast<const float2*>(norm_pair2)[_la]);
        int m_abs20 = first_m_start + j_base2;
        int m_abs21 = m_abs20 + 1;
        best_d[slot_base0 + j_rel + 4] = _t2[0];
        best_i[slot_base0 + j_rel + 4] = m_abs20;
        best_d[slot_base0 + j_rel + 5] = _t2[1];
        best_i[slot_base0 + j_rel + 5] = m_abs21;
        int j_base3 = j_base0 + 6;
        float dist_pair3[2];
        float norm_pair3[2];
        dist_pair3[0] = dots0[j_rel + 6];
        dist_pair3[1] = dots0[j_rel + 7];
        const float2 _fma_b2_12 = {-2.0f, -2.0f};
        const float2 _fma_c2_13 = {q_norm, q_norm};
        #pragma unroll
        for (int _lf = 0; _lf < 1; _lf++)
            fma_f32x2_inplace(&reinterpret_cast<float2*>(dist_pair3)[_lf], _fma_b2_12, _fma_c2_13);
        norm_pair3[0] = smem_db_norm[j_base3];
        norm_pair3[1] = smem_db_norm[j_base3 + 1];
        float _t3[2];
        #pragma unroll
        for (int _la = 0; _la < 1; _la++)
            reinterpret_cast<float2*>(_t3)[_la] = add_f32x2(reinterpret_cast<float2*>(dist_pair3)[_la], reinterpret_cast<const float2*>(norm_pair3)[_la]);
        int m_abs30 = first_m_start + j_base3;
        int m_abs31 = m_abs30 + 1;
        best_d[slot_base0 + j_rel + 6] = _t3[0];
        best_i[slot_base0 + j_rel + 6] = m_abs30;
        best_d[slot_base0 + j_rel + 7] = _t3[1];
        best_i[slot_base0 + j_rel + 7] = m_abs31;
    }
    uint32_t _phase_mma_done_second_0 = 0;
    if (has_second_tile != 0) {
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
            "mov.b32 id, 136316048;\n\t"
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
            "add.u32 alo, alo, 1018;\n\t"
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
            :: "r"(_desc_lo_0), "r"(_desc_lo_2), "r"(taddr), "r"(0));
            elect_commit(mma_done_second_addr);
        }
        mbarrier_wait(mma_done_second_addr, _phase_mma_done_second_0);
        _phase_mma_done_second_0 ^= 1;
        const int col_base1 = col_chunk * 32;
        float dots1[32];
        tmem_ld_x32(&dots1[0], taddr + (row_base_tmem << 16) + col_base1);
        asm volatile("tcgen05.wait::ld.sync.aligned;");
        const int slot_base1 = 32;
        #pragma unroll 4
        for (int j_rel = 0; j_rel < 32; j_rel += 8) {
            int j_base0 = col_base1 + j_rel;
            float dist_pair0[2];
            float norm_pair0[2];
            dist_pair0[0] = dots1[j_rel];
            dist_pair0[1] = dots1[j_rel + 1];
            const float2 _fma_b2_14 = {-2.0f, -2.0f};
            const float2 _fma_c2_15 = {q_norm, q_norm};
            #pragma unroll
            for (int _lf = 0; _lf < 1; _lf++)
                fma_f32x2_inplace(&reinterpret_cast<float2*>(dist_pair0)[_lf], _fma_b2_14, _fma_c2_15);
            norm_pair0[0] = smem_db_norm_next[j_base0];
            norm_pair0[1] = smem_db_norm_next[j_base0 + 1];
            float _t4[2];
            #pragma unroll
            for (int _la = 0; _la < 1; _la++)
                reinterpret_cast<float2*>(_t4)[_la] = add_f32x2(reinterpret_cast<float2*>(dist_pair0)[_la], reinterpret_cast<const float2*>(norm_pair0)[_la]);
            int m_abs00 = second_m_start + j_base0;
            int m_abs01 = m_abs00 + 1;
            best_d[slot_base1 + j_rel] = _t4[0];
            best_i[slot_base1 + j_rel] = m_abs00;
            best_d[slot_base1 + j_rel + 1] = _t4[1];
            best_i[slot_base1 + j_rel + 1] = m_abs01;
            int j_base1 = j_base0 + 2;
            float dist_pair1[2];
            float norm_pair1[2];
            dist_pair1[0] = dots1[j_rel + 2];
            dist_pair1[1] = dots1[j_rel + 3];
            const float2 _fma_b2_16 = {-2.0f, -2.0f};
            const float2 _fma_c2_17 = {q_norm, q_norm};
            #pragma unroll
            for (int _lf = 0; _lf < 1; _lf++)
                fma_f32x2_inplace(&reinterpret_cast<float2*>(dist_pair1)[_lf], _fma_b2_16, _fma_c2_17);
            norm_pair1[0] = smem_db_norm_next[j_base1];
            norm_pair1[1] = smem_db_norm_next[j_base1 + 1];
            float _t5[2];
            #pragma unroll
            for (int _la = 0; _la < 1; _la++)
                reinterpret_cast<float2*>(_t5)[_la] = add_f32x2(reinterpret_cast<float2*>(dist_pair1)[_la], reinterpret_cast<const float2*>(norm_pair1)[_la]);
            int m_abs10 = second_m_start + j_base1;
            int m_abs11 = m_abs10 + 1;
            best_d[slot_base1 + j_rel + 2] = _t5[0];
            best_i[slot_base1 + j_rel + 2] = m_abs10;
            best_d[slot_base1 + j_rel + 3] = _t5[1];
            best_i[slot_base1 + j_rel + 3] = m_abs11;
            int j_base2 = j_base0 + 4;
            float dist_pair2[2];
            float norm_pair2[2];
            dist_pair2[0] = dots1[j_rel + 4];
            dist_pair2[1] = dots1[j_rel + 5];
            const float2 _fma_b2_18 = {-2.0f, -2.0f};
            const float2 _fma_c2_19 = {q_norm, q_norm};
            #pragma unroll
            for (int _lf = 0; _lf < 1; _lf++)
                fma_f32x2_inplace(&reinterpret_cast<float2*>(dist_pair2)[_lf], _fma_b2_18, _fma_c2_19);
            norm_pair2[0] = smem_db_norm_next[j_base2];
            norm_pair2[1] = smem_db_norm_next[j_base2 + 1];
            float _t6[2];
            #pragma unroll
            for (int _la = 0; _la < 1; _la++)
                reinterpret_cast<float2*>(_t6)[_la] = add_f32x2(reinterpret_cast<float2*>(dist_pair2)[_la], reinterpret_cast<const float2*>(norm_pair2)[_la]);
            int m_abs20 = second_m_start + j_base2;
            int m_abs21 = m_abs20 + 1;
            best_d[slot_base1 + j_rel + 4] = _t6[0];
            best_i[slot_base1 + j_rel + 4] = m_abs20;
            best_d[slot_base1 + j_rel + 5] = _t6[1];
            best_i[slot_base1 + j_rel + 5] = m_abs21;
            int j_base3 = j_base0 + 6;
            float dist_pair3[2];
            float norm_pair3[2];
            dist_pair3[0] = dots1[j_rel + 6];
            dist_pair3[1] = dots1[j_rel + 7];
            const float2 _fma_b2_20 = {-2.0f, -2.0f};
            const float2 _fma_c2_21 = {q_norm, q_norm};
            #pragma unroll
            for (int _lf = 0; _lf < 1; _lf++)
                fma_f32x2_inplace(&reinterpret_cast<float2*>(dist_pair3)[_lf], _fma_b2_20, _fma_c2_21);
            norm_pair3[0] = smem_db_norm_next[j_base3];
            norm_pair3[1] = smem_db_norm_next[j_base3 + 1];
            float _t7[2];
            #pragma unroll
            for (int _la = 0; _la < 1; _la++)
                reinterpret_cast<float2*>(_t7)[_la] = add_f32x2(reinterpret_cast<float2*>(dist_pair3)[_la], reinterpret_cast<const float2*>(norm_pair3)[_la]);
            int m_abs30 = second_m_start + j_base3;
            int m_abs31 = m_abs30 + 1;
            best_d[slot_base1 + j_rel + 6] = _t7[0];
            best_i[slot_base1 + j_rel + 6] = m_abs30;
            best_d[slot_base1 + j_rel + 7] = _t7[1];
            best_i[slot_base1 + j_rel + 7] = m_abs31;
        }
    }
    int partial_split_id = split_id * 4 + col_chunk;
    unsigned long long partial_col_base = (unsigned long long)(((q_tile * 320 + partial_split_id) * 128 + q_local) * K_MAX_);
    {
        {
            float left_d = best_d[0];
            float right_d = best_d[1];
            int left_i = best_i[0];
            int right_i = best_i[1];
            int swap = ((right_d < left_d) ? 1 : 0);
            best_d[0] = ((swap != 0) ? right_d : left_d);
            best_i[0] = ((swap != 0) ? right_i : left_i);
            best_d[1] = ((swap != 0) ? left_d : right_d);
            best_i[1] = ((swap != 0) ? left_i : right_i);
        }
    }
    {
        {
            float left_d = best_d[2];
            float right_d = best_d[3];
            int left_i = best_i[2];
            int right_i = best_i[3];
            int swap = ((left_d < right_d) ? 1 : 0);
            best_d[2] = ((swap != 0) ? right_d : left_d);
            best_i[2] = ((swap != 0) ? right_i : left_i);
            best_d[3] = ((swap != 0) ? left_d : right_d);
            best_i[3] = ((swap != 0) ? left_i : right_i);
        }
    }
    {
        {
            float left_d = best_d[4];
            float right_d = best_d[5];
            int left_i = best_i[4];
            int right_i = best_i[5];
            int swap = ((right_d < left_d) ? 1 : 0);
            best_d[4] = ((swap != 0) ? right_d : left_d);
            best_i[4] = ((swap != 0) ? right_i : left_i);
            best_d[5] = ((swap != 0) ? left_d : right_d);
            best_i[5] = ((swap != 0) ? left_i : right_i);
        }
    }
    {
        {
            float left_d = best_d[6];
            float right_d = best_d[7];
            int left_i = best_i[6];
            int right_i = best_i[7];
            int swap = ((left_d < right_d) ? 1 : 0);
            best_d[6] = ((swap != 0) ? right_d : left_d);
            best_i[6] = ((swap != 0) ? right_i : left_i);
            best_d[7] = ((swap != 0) ? left_d : right_d);
            best_i[7] = ((swap != 0) ? left_i : right_i);
        }
    }
    {
        {
            float left_d = best_d[8];
            float right_d = best_d[9];
            int left_i = best_i[8];
            int right_i = best_i[9];
            int swap = ((right_d < left_d) ? 1 : 0);
            best_d[8] = ((swap != 0) ? right_d : left_d);
            best_i[8] = ((swap != 0) ? right_i : left_i);
            best_d[9] = ((swap != 0) ? left_d : right_d);
            best_i[9] = ((swap != 0) ? left_i : right_i);
        }
    }
    {
        {
            float left_d = best_d[10];
            float right_d = best_d[11];
            int left_i = best_i[10];
            int right_i = best_i[11];
            int swap = ((left_d < right_d) ? 1 : 0);
            best_d[10] = ((swap != 0) ? right_d : left_d);
            best_i[10] = ((swap != 0) ? right_i : left_i);
            best_d[11] = ((swap != 0) ? left_d : right_d);
            best_i[11] = ((swap != 0) ? left_i : right_i);
        }
    }
    {
        {
            float left_d = best_d[12];
            float right_d = best_d[13];
            int left_i = best_i[12];
            int right_i = best_i[13];
            int swap = ((right_d < left_d) ? 1 : 0);
            best_d[12] = ((swap != 0) ? right_d : left_d);
            best_i[12] = ((swap != 0) ? right_i : left_i);
            best_d[13] = ((swap != 0) ? left_d : right_d);
            best_i[13] = ((swap != 0) ? left_i : right_i);
        }
    }
    {
        {
            float left_d = best_d[14];
            float right_d = best_d[15];
            int left_i = best_i[14];
            int right_i = best_i[15];
            int swap = ((left_d < right_d) ? 1 : 0);
            best_d[14] = ((swap != 0) ? right_d : left_d);
            best_i[14] = ((swap != 0) ? right_i : left_i);
            best_d[15] = ((swap != 0) ? left_d : right_d);
            best_i[15] = ((swap != 0) ? left_i : right_i);
        }
    }
    {
        {
            float left_d = best_d[16];
            float right_d = best_d[17];
            int left_i = best_i[16];
            int right_i = best_i[17];
            int swap = ((right_d < left_d) ? 1 : 0);
            best_d[16] = ((swap != 0) ? right_d : left_d);
            best_i[16] = ((swap != 0) ? right_i : left_i);
            best_d[17] = ((swap != 0) ? left_d : right_d);
            best_i[17] = ((swap != 0) ? left_i : right_i);
        }
    }
    {
        {
            float left_d = best_d[18];
            float right_d = best_d[19];
            int left_i = best_i[18];
            int right_i = best_i[19];
            int swap = ((left_d < right_d) ? 1 : 0);
            best_d[18] = ((swap != 0) ? right_d : left_d);
            best_i[18] = ((swap != 0) ? right_i : left_i);
            best_d[19] = ((swap != 0) ? left_d : right_d);
            best_i[19] = ((swap != 0) ? left_i : right_i);
        }
    }
    {
        {
            float left_d = best_d[20];
            float right_d = best_d[21];
            int left_i = best_i[20];
            int right_i = best_i[21];
            int swap = ((right_d < left_d) ? 1 : 0);
            best_d[20] = ((swap != 0) ? right_d : left_d);
            best_i[20] = ((swap != 0) ? right_i : left_i);
            best_d[21] = ((swap != 0) ? left_d : right_d);
            best_i[21] = ((swap != 0) ? left_i : right_i);
        }
    }
    {
        {
            float left_d = best_d[22];
            float right_d = best_d[23];
            int left_i = best_i[22];
            int right_i = best_i[23];
            int swap = ((left_d < right_d) ? 1 : 0);
            best_d[22] = ((swap != 0) ? right_d : left_d);
            best_i[22] = ((swap != 0) ? right_i : left_i);
            best_d[23] = ((swap != 0) ? left_d : right_d);
            best_i[23] = ((swap != 0) ? left_i : right_i);
        }
    }
    {
        {
            float left_d = best_d[24];
            float right_d = best_d[25];
            int left_i = best_i[24];
            int right_i = best_i[25];
            int swap = ((right_d < left_d) ? 1 : 0);
            best_d[24] = ((swap != 0) ? right_d : left_d);
            best_i[24] = ((swap != 0) ? right_i : left_i);
            best_d[25] = ((swap != 0) ? left_d : right_d);
            best_i[25] = ((swap != 0) ? left_i : right_i);
        }
    }
    {
        {
            float left_d = best_d[26];
            float right_d = best_d[27];
            int left_i = best_i[26];
            int right_i = best_i[27];
            int swap = ((left_d < right_d) ? 1 : 0);
            best_d[26] = ((swap != 0) ? right_d : left_d);
            best_i[26] = ((swap != 0) ? right_i : left_i);
            best_d[27] = ((swap != 0) ? left_d : right_d);
            best_i[27] = ((swap != 0) ? left_i : right_i);
        }
    }
    {
        {
            float left_d = best_d[28];
            float right_d = best_d[29];
            int left_i = best_i[28];
            int right_i = best_i[29];
            int swap = ((right_d < left_d) ? 1 : 0);
            best_d[28] = ((swap != 0) ? right_d : left_d);
            best_i[28] = ((swap != 0) ? right_i : left_i);
            best_d[29] = ((swap != 0) ? left_d : right_d);
            best_i[29] = ((swap != 0) ? left_i : right_i);
        }
    }
    {
        {
            float left_d = best_d[30];
            float right_d = best_d[31];
            int left_i = best_i[30];
            int right_i = best_i[31];
            int swap = ((left_d < right_d) ? 1 : 0);
            best_d[30] = ((swap != 0) ? right_d : left_d);
            best_i[30] = ((swap != 0) ? right_i : left_i);
            best_d[31] = ((swap != 0) ? left_d : right_d);
            best_i[31] = ((swap != 0) ? left_i : right_i);
        }
    }
    {
        {
            float left_d = best_d[32];
            float right_d = best_d[33];
            int left_i = best_i[32];
            int right_i = best_i[33];
            int swap = ((right_d < left_d) ? 1 : 0);
            best_d[32] = ((swap != 0) ? right_d : left_d);
            best_i[32] = ((swap != 0) ? right_i : left_i);
            best_d[33] = ((swap != 0) ? left_d : right_d);
            best_i[33] = ((swap != 0) ? left_i : right_i);
        }
    }
    {
        {
            float left_d = best_d[34];
            float right_d = best_d[35];
            int left_i = best_i[34];
            int right_i = best_i[35];
            int swap = ((left_d < right_d) ? 1 : 0);
            best_d[34] = ((swap != 0) ? right_d : left_d);
            best_i[34] = ((swap != 0) ? right_i : left_i);
            best_d[35] = ((swap != 0) ? left_d : right_d);
            best_i[35] = ((swap != 0) ? left_i : right_i);
        }
    }
    {
        {
            float left_d = best_d[36];
            float right_d = best_d[37];
            int left_i = best_i[36];
            int right_i = best_i[37];
            int swap = ((right_d < left_d) ? 1 : 0);
            best_d[36] = ((swap != 0) ? right_d : left_d);
            best_i[36] = ((swap != 0) ? right_i : left_i);
            best_d[37] = ((swap != 0) ? left_d : right_d);
            best_i[37] = ((swap != 0) ? left_i : right_i);
        }
    }
    {
        {
            float left_d = best_d[38];
            float right_d = best_d[39];
            int left_i = best_i[38];
            int right_i = best_i[39];
            int swap = ((left_d < right_d) ? 1 : 0);
            best_d[38] = ((swap != 0) ? right_d : left_d);
            best_i[38] = ((swap != 0) ? right_i : left_i);
            best_d[39] = ((swap != 0) ? left_d : right_d);
            best_i[39] = ((swap != 0) ? left_i : right_i);
        }
    }
    {
        {
            float left_d = best_d[40];
            float right_d = best_d[41];
            int left_i = best_i[40];
            int right_i = best_i[41];
            int swap = ((right_d < left_d) ? 1 : 0);
            best_d[40] = ((swap != 0) ? right_d : left_d);
            best_i[40] = ((swap != 0) ? right_i : left_i);
            best_d[41] = ((swap != 0) ? left_d : right_d);
            best_i[41] = ((swap != 0) ? left_i : right_i);
        }
    }
    {
        {
            float left_d = best_d[42];
            float right_d = best_d[43];
            int left_i = best_i[42];
            int right_i = best_i[43];
            int swap = ((left_d < right_d) ? 1 : 0);
            best_d[42] = ((swap != 0) ? right_d : left_d);
            best_i[42] = ((swap != 0) ? right_i : left_i);
            best_d[43] = ((swap != 0) ? left_d : right_d);
            best_i[43] = ((swap != 0) ? left_i : right_i);
        }
    }
    {
        {
            float left_d = best_d[44];
            float right_d = best_d[45];
            int left_i = best_i[44];
            int right_i = best_i[45];
            int swap = ((right_d < left_d) ? 1 : 0);
            best_d[44] = ((swap != 0) ? right_d : left_d);
            best_i[44] = ((swap != 0) ? right_i : left_i);
            best_d[45] = ((swap != 0) ? left_d : right_d);
            best_i[45] = ((swap != 0) ? left_i : right_i);
        }
    }
    {
        {
            float left_d = best_d[46];
            float right_d = best_d[47];
            int left_i = best_i[46];
            int right_i = best_i[47];
            int swap = ((left_d < right_d) ? 1 : 0);
            best_d[46] = ((swap != 0) ? right_d : left_d);
            best_i[46] = ((swap != 0) ? right_i : left_i);
            best_d[47] = ((swap != 0) ? left_d : right_d);
            best_i[47] = ((swap != 0) ? left_i : right_i);
        }
    }
    {
        {
            float left_d = best_d[48];
            float right_d = best_d[49];
            int left_i = best_i[48];
            int right_i = best_i[49];
            int swap = ((right_d < left_d) ? 1 : 0);
            best_d[48] = ((swap != 0) ? right_d : left_d);
            best_i[48] = ((swap != 0) ? right_i : left_i);
            best_d[49] = ((swap != 0) ? left_d : right_d);
            best_i[49] = ((swap != 0) ? left_i : right_i);
        }
    }
    {
        {
            float left_d = best_d[50];
            float right_d = best_d[51];
            int left_i = best_i[50];
            int right_i = best_i[51];
            int swap = ((left_d < right_d) ? 1 : 0);
            best_d[50] = ((swap != 0) ? right_d : left_d);
            best_i[50] = ((swap != 0) ? right_i : left_i);
            best_d[51] = ((swap != 0) ? left_d : right_d);
            best_i[51] = ((swap != 0) ? left_i : right_i);
        }
    }
    {
        {
            float left_d = best_d[52];
            float right_d = best_d[53];
            int left_i = best_i[52];
            int right_i = best_i[53];
            int swap = ((right_d < left_d) ? 1 : 0);
            best_d[52] = ((swap != 0) ? right_d : left_d);
            best_i[52] = ((swap != 0) ? right_i : left_i);
            best_d[53] = ((swap != 0) ? left_d : right_d);
            best_i[53] = ((swap != 0) ? left_i : right_i);
        }
    }
    {
        {
            float left_d = best_d[54];
            float right_d = best_d[55];
            int left_i = best_i[54];
            int right_i = best_i[55];
            int swap = ((left_d < right_d) ? 1 : 0);
            best_d[54] = ((swap != 0) ? right_d : left_d);
            best_i[54] = ((swap != 0) ? right_i : left_i);
            best_d[55] = ((swap != 0) ? left_d : right_d);
            best_i[55] = ((swap != 0) ? left_i : right_i);
        }
    }
    {
        {
            float left_d = best_d[56];
            float right_d = best_d[57];
            int left_i = best_i[56];
            int right_i = best_i[57];
            int swap = ((right_d < left_d) ? 1 : 0);
            best_d[56] = ((swap != 0) ? right_d : left_d);
            best_i[56] = ((swap != 0) ? right_i : left_i);
            best_d[57] = ((swap != 0) ? left_d : right_d);
            best_i[57] = ((swap != 0) ? left_i : right_i);
        }
    }
    {
        {
            float left_d = best_d[58];
            float right_d = best_d[59];
            int left_i = best_i[58];
            int right_i = best_i[59];
            int swap = ((left_d < right_d) ? 1 : 0);
            best_d[58] = ((swap != 0) ? right_d : left_d);
            best_i[58] = ((swap != 0) ? right_i : left_i);
            best_d[59] = ((swap != 0) ? left_d : right_d);
            best_i[59] = ((swap != 0) ? left_i : right_i);
        }
    }
    {
        {
            float left_d = best_d[60];
            float right_d = best_d[61];
            int left_i = best_i[60];
            int right_i = best_i[61];
            int swap = ((right_d < left_d) ? 1 : 0);
            best_d[60] = ((swap != 0) ? right_d : left_d);
            best_i[60] = ((swap != 0) ? right_i : left_i);
            best_d[61] = ((swap != 0) ? left_d : right_d);
            best_i[61] = ((swap != 0) ? left_i : right_i);
        }
    }
    {
        {
            float left_d = best_d[62];
            float right_d = best_d[63];
            int left_i = best_i[62];
            int right_i = best_i[63];
            int swap = ((left_d < right_d) ? 1 : 0);
            best_d[62] = ((swap != 0) ? right_d : left_d);
            best_i[62] = ((swap != 0) ? right_i : left_i);
            best_d[63] = ((swap != 0) ? left_d : right_d);
            best_i[63] = ((swap != 0) ? left_i : right_i);
        }
    }
    {
        {
            float left_d = best_d[0];
            float right_d = best_d[2];
            int left_i = best_i[0];
            int right_i = best_i[2];
            int swap = ((right_d < left_d) ? 1 : 0);
            best_d[0] = ((swap != 0) ? right_d : left_d);
            best_i[0] = ((swap != 0) ? right_i : left_i);
            best_d[2] = ((swap != 0) ? left_d : right_d);
            best_i[2] = ((swap != 0) ? left_i : right_i);
        }
    }
    {
        {
            float left_d = best_d[1];
            float right_d = best_d[3];
            int left_i = best_i[1];
            int right_i = best_i[3];
            int swap = ((right_d < left_d) ? 1 : 0);
            best_d[1] = ((swap != 0) ? right_d : left_d);
            best_i[1] = ((swap != 0) ? right_i : left_i);
            best_d[3] = ((swap != 0) ? left_d : right_d);
            best_i[3] = ((swap != 0) ? left_i : right_i);
        }
    }
    {
        {
            float left_d = best_d[4];
            float right_d = best_d[6];
            int left_i = best_i[4];
            int right_i = best_i[6];
            int swap = ((left_d < right_d) ? 1 : 0);
            best_d[4] = ((swap != 0) ? right_d : left_d);
            best_i[4] = ((swap != 0) ? right_i : left_i);
            best_d[6] = ((swap != 0) ? left_d : right_d);
            best_i[6] = ((swap != 0) ? left_i : right_i);
        }
    }
    {
        {
            float left_d = best_d[5];
            float right_d = best_d[7];
            int left_i = best_i[5];
            int right_i = best_i[7];
            int swap = ((left_d < right_d) ? 1 : 0);
            best_d[5] = ((swap != 0) ? right_d : left_d);
            best_i[5] = ((swap != 0) ? right_i : left_i);
            best_d[7] = ((swap != 0) ? left_d : right_d);
            best_i[7] = ((swap != 0) ? left_i : right_i);
        }
    }
    {
        {
            float left_d = best_d[8];
            float right_d = best_d[10];
            int left_i = best_i[8];
            int right_i = best_i[10];
            int swap = ((right_d < left_d) ? 1 : 0);
            best_d[8] = ((swap != 0) ? right_d : left_d);
            best_i[8] = ((swap != 0) ? right_i : left_i);
            best_d[10] = ((swap != 0) ? left_d : right_d);
            best_i[10] = ((swap != 0) ? left_i : right_i);
        }
    }
    {
        {
            float left_d = best_d[9];
            float right_d = best_d[11];
            int left_i = best_i[9];
            int right_i = best_i[11];
            int swap = ((right_d < left_d) ? 1 : 0);
            best_d[9] = ((swap != 0) ? right_d : left_d);
            best_i[9] = ((swap != 0) ? right_i : left_i);
            best_d[11] = ((swap != 0) ? left_d : right_d);
            best_i[11] = ((swap != 0) ? left_i : right_i);
        }
    }
    {
        {
            float left_d = best_d[12];
            float right_d = best_d[14];
            int left_i = best_i[12];
            int right_i = best_i[14];
            int swap = ((left_d < right_d) ? 1 : 0);
            best_d[12] = ((swap != 0) ? right_d : left_d);
            best_i[12] = ((swap != 0) ? right_i : left_i);
            best_d[14] = ((swap != 0) ? left_d : right_d);
            best_i[14] = ((swap != 0) ? left_i : right_i);
        }
    }
    {
        {
            float left_d = best_d[13];
            float right_d = best_d[15];
            int left_i = best_i[13];
            int right_i = best_i[15];
            int swap = ((left_d < right_d) ? 1 : 0);
            best_d[13] = ((swap != 0) ? right_d : left_d);
            best_i[13] = ((swap != 0) ? right_i : left_i);
            best_d[15] = ((swap != 0) ? left_d : right_d);
            best_i[15] = ((swap != 0) ? left_i : right_i);
        }
    }
    {
        {
            float left_d = best_d[16];
            float right_d = best_d[18];
            int left_i = best_i[16];
            int right_i = best_i[18];
            int swap = ((right_d < left_d) ? 1 : 0);
            best_d[16] = ((swap != 0) ? right_d : left_d);
            best_i[16] = ((swap != 0) ? right_i : left_i);
            best_d[18] = ((swap != 0) ? left_d : right_d);
            best_i[18] = ((swap != 0) ? left_i : right_i);
        }
    }
    {
        {
            float left_d = best_d[17];
            float right_d = best_d[19];
            int left_i = best_i[17];
            int right_i = best_i[19];
            int swap = ((right_d < left_d) ? 1 : 0);
            best_d[17] = ((swap != 0) ? right_d : left_d);
            best_i[17] = ((swap != 0) ? right_i : left_i);
            best_d[19] = ((swap != 0) ? left_d : right_d);
            best_i[19] = ((swap != 0) ? left_i : right_i);
        }
    }
    {
        {
            float left_d = best_d[20];
            float right_d = best_d[22];
            int left_i = best_i[20];
            int right_i = best_i[22];
            int swap = ((left_d < right_d) ? 1 : 0);
            best_d[20] = ((swap != 0) ? right_d : left_d);
            best_i[20] = ((swap != 0) ? right_i : left_i);
            best_d[22] = ((swap != 0) ? left_d : right_d);
            best_i[22] = ((swap != 0) ? left_i : right_i);
        }
    }
    {
        {
            float left_d = best_d[21];
            float right_d = best_d[23];
            int left_i = best_i[21];
            int right_i = best_i[23];
            int swap = ((left_d < right_d) ? 1 : 0);
            best_d[21] = ((swap != 0) ? right_d : left_d);
            best_i[21] = ((swap != 0) ? right_i : left_i);
            best_d[23] = ((swap != 0) ? left_d : right_d);
            best_i[23] = ((swap != 0) ? left_i : right_i);
        }
    }
    {
        {
            float left_d = best_d[24];
            float right_d = best_d[26];
            int left_i = best_i[24];
            int right_i = best_i[26];
            int swap = ((right_d < left_d) ? 1 : 0);
            best_d[24] = ((swap != 0) ? right_d : left_d);
            best_i[24] = ((swap != 0) ? right_i : left_i);
            best_d[26] = ((swap != 0) ? left_d : right_d);
            best_i[26] = ((swap != 0) ? left_i : right_i);
        }
    }
    {
        {
            float left_d = best_d[25];
            float right_d = best_d[27];
            int left_i = best_i[25];
            int right_i = best_i[27];
            int swap = ((right_d < left_d) ? 1 : 0);
            best_d[25] = ((swap != 0) ? right_d : left_d);
            best_i[25] = ((swap != 0) ? right_i : left_i);
            best_d[27] = ((swap != 0) ? left_d : right_d);
            best_i[27] = ((swap != 0) ? left_i : right_i);
        }
    }
    {
        {
            float left_d = best_d[28];
            float right_d = best_d[30];
            int left_i = best_i[28];
            int right_i = best_i[30];
            int swap = ((left_d < right_d) ? 1 : 0);
            best_d[28] = ((swap != 0) ? right_d : left_d);
            best_i[28] = ((swap != 0) ? right_i : left_i);
            best_d[30] = ((swap != 0) ? left_d : right_d);
            best_i[30] = ((swap != 0) ? left_i : right_i);
        }
    }
    {
        {
            float left_d = best_d[29];
            float right_d = best_d[31];
            int left_i = best_i[29];
            int right_i = best_i[31];
            int swap = ((left_d < right_d) ? 1 : 0);
            best_d[29] = ((swap != 0) ? right_d : left_d);
            best_i[29] = ((swap != 0) ? right_i : left_i);
            best_d[31] = ((swap != 0) ? left_d : right_d);
            best_i[31] = ((swap != 0) ? left_i : right_i);
        }
    }
    {
        {
            float left_d = best_d[32];
            float right_d = best_d[34];
            int left_i = best_i[32];
            int right_i = best_i[34];
            int swap = ((right_d < left_d) ? 1 : 0);
            best_d[32] = ((swap != 0) ? right_d : left_d);
            best_i[32] = ((swap != 0) ? right_i : left_i);
            best_d[34] = ((swap != 0) ? left_d : right_d);
            best_i[34] = ((swap != 0) ? left_i : right_i);
        }
    }
    {
        {
            float left_d = best_d[33];
            float right_d = best_d[35];
            int left_i = best_i[33];
            int right_i = best_i[35];
            int swap = ((right_d < left_d) ? 1 : 0);
            best_d[33] = ((swap != 0) ? right_d : left_d);
            best_i[33] = ((swap != 0) ? right_i : left_i);
            best_d[35] = ((swap != 0) ? left_d : right_d);
            best_i[35] = ((swap != 0) ? left_i : right_i);
        }
    }
    {
        {
            float left_d = best_d[36];
            float right_d = best_d[38];
            int left_i = best_i[36];
            int right_i = best_i[38];
            int swap = ((left_d < right_d) ? 1 : 0);
            best_d[36] = ((swap != 0) ? right_d : left_d);
            best_i[36] = ((swap != 0) ? right_i : left_i);
            best_d[38] = ((swap != 0) ? left_d : right_d);
            best_i[38] = ((swap != 0) ? left_i : right_i);
        }
    }
    {
        {
            float left_d = best_d[37];
            float right_d = best_d[39];
            int left_i = best_i[37];
            int right_i = best_i[39];
            int swap = ((left_d < right_d) ? 1 : 0);
            best_d[37] = ((swap != 0) ? right_d : left_d);
            best_i[37] = ((swap != 0) ? right_i : left_i);
            best_d[39] = ((swap != 0) ? left_d : right_d);
            best_i[39] = ((swap != 0) ? left_i : right_i);
        }
    }
    {
        {
            float left_d = best_d[40];
            float right_d = best_d[42];
            int left_i = best_i[40];
            int right_i = best_i[42];
            int swap = ((right_d < left_d) ? 1 : 0);
            best_d[40] = ((swap != 0) ? right_d : left_d);
            best_i[40] = ((swap != 0) ? right_i : left_i);
            best_d[42] = ((swap != 0) ? left_d : right_d);
            best_i[42] = ((swap != 0) ? left_i : right_i);
        }
    }
    {
        {
            float left_d = best_d[41];
            float right_d = best_d[43];
            int left_i = best_i[41];
            int right_i = best_i[43];
            int swap = ((right_d < left_d) ? 1 : 0);
            best_d[41] = ((swap != 0) ? right_d : left_d);
            best_i[41] = ((swap != 0) ? right_i : left_i);
            best_d[43] = ((swap != 0) ? left_d : right_d);
            best_i[43] = ((swap != 0) ? left_i : right_i);
        }
    }
    {
        {
            float left_d = best_d[44];
            float right_d = best_d[46];
            int left_i = best_i[44];
            int right_i = best_i[46];
            int swap = ((left_d < right_d) ? 1 : 0);
            best_d[44] = ((swap != 0) ? right_d : left_d);
            best_i[44] = ((swap != 0) ? right_i : left_i);
            best_d[46] = ((swap != 0) ? left_d : right_d);
            best_i[46] = ((swap != 0) ? left_i : right_i);
        }
    }
    {
        {
            float left_d = best_d[45];
            float right_d = best_d[47];
            int left_i = best_i[45];
            int right_i = best_i[47];
            int swap = ((left_d < right_d) ? 1 : 0);
            best_d[45] = ((swap != 0) ? right_d : left_d);
            best_i[45] = ((swap != 0) ? right_i : left_i);
            best_d[47] = ((swap != 0) ? left_d : right_d);
            best_i[47] = ((swap != 0) ? left_i : right_i);
        }
    }
    {
        {
            float left_d = best_d[48];
            float right_d = best_d[50];
            int left_i = best_i[48];
            int right_i = best_i[50];
            int swap = ((right_d < left_d) ? 1 : 0);
            best_d[48] = ((swap != 0) ? right_d : left_d);
            best_i[48] = ((swap != 0) ? right_i : left_i);
            best_d[50] = ((swap != 0) ? left_d : right_d);
            best_i[50] = ((swap != 0) ? left_i : right_i);
        }
    }
    {
        {
            float left_d = best_d[49];
            float right_d = best_d[51];
            int left_i = best_i[49];
            int right_i = best_i[51];
            int swap = ((right_d < left_d) ? 1 : 0);
            best_d[49] = ((swap != 0) ? right_d : left_d);
            best_i[49] = ((swap != 0) ? right_i : left_i);
            best_d[51] = ((swap != 0) ? left_d : right_d);
            best_i[51] = ((swap != 0) ? left_i : right_i);
        }
    }
    {
        {
            float left_d = best_d[52];
            float right_d = best_d[54];
            int left_i = best_i[52];
            int right_i = best_i[54];
            int swap = ((left_d < right_d) ? 1 : 0);
            best_d[52] = ((swap != 0) ? right_d : left_d);
            best_i[52] = ((swap != 0) ? right_i : left_i);
            best_d[54] = ((swap != 0) ? left_d : right_d);
            best_i[54] = ((swap != 0) ? left_i : right_i);
        }
    }
    {
        {
            float left_d = best_d[53];
            float right_d = best_d[55];
            int left_i = best_i[53];
            int right_i = best_i[55];
            int swap = ((left_d < right_d) ? 1 : 0);
            best_d[53] = ((swap != 0) ? right_d : left_d);
            best_i[53] = ((swap != 0) ? right_i : left_i);
            best_d[55] = ((swap != 0) ? left_d : right_d);
            best_i[55] = ((swap != 0) ? left_i : right_i);
        }
    }
    {
        {
            float left_d = best_d[56];
            float right_d = best_d[58];
            int left_i = best_i[56];
            int right_i = best_i[58];
            int swap = ((right_d < left_d) ? 1 : 0);
            best_d[56] = ((swap != 0) ? right_d : left_d);
            best_i[56] = ((swap != 0) ? right_i : left_i);
            best_d[58] = ((swap != 0) ? left_d : right_d);
            best_i[58] = ((swap != 0) ? left_i : right_i);
        }
    }
    {
        {
            float left_d = best_d[57];
            float right_d = best_d[59];
            int left_i = best_i[57];
            int right_i = best_i[59];
            int swap = ((right_d < left_d) ? 1 : 0);
            best_d[57] = ((swap != 0) ? right_d : left_d);
            best_i[57] = ((swap != 0) ? right_i : left_i);
            best_d[59] = ((swap != 0) ? left_d : right_d);
            best_i[59] = ((swap != 0) ? left_i : right_i);
        }
    }
    {
        {
            float left_d = best_d[60];
            float right_d = best_d[62];
            int left_i = best_i[60];
            int right_i = best_i[62];
            int swap = ((left_d < right_d) ? 1 : 0);
            best_d[60] = ((swap != 0) ? right_d : left_d);
            best_i[60] = ((swap != 0) ? right_i : left_i);
            best_d[62] = ((swap != 0) ? left_d : right_d);
            best_i[62] = ((swap != 0) ? left_i : right_i);
        }
    }
    {
        {
            float left_d = best_d[61];
            float right_d = best_d[63];
            int left_i = best_i[61];
            int right_i = best_i[63];
            int swap = ((left_d < right_d) ? 1 : 0);
            best_d[61] = ((swap != 0) ? right_d : left_d);
            best_i[61] = ((swap != 0) ? right_i : left_i);
            best_d[63] = ((swap != 0) ? left_d : right_d);
            best_i[63] = ((swap != 0) ? left_i : right_i);
        }
    }
    {
        {
            float left_d = best_d[0];
            float right_d = best_d[1];
            int left_i = best_i[0];
            int right_i = best_i[1];
            int swap = ((right_d < left_d) ? 1 : 0);
            best_d[0] = ((swap != 0) ? right_d : left_d);
            best_i[0] = ((swap != 0) ? right_i : left_i);
            best_d[1] = ((swap != 0) ? left_d : right_d);
            best_i[1] = ((swap != 0) ? left_i : right_i);
        }
    }
    {
        {
            float left_d = best_d[2];
            float right_d = best_d[3];
            int left_i = best_i[2];
            int right_i = best_i[3];
            int swap = ((right_d < left_d) ? 1 : 0);
            best_d[2] = ((swap != 0) ? right_d : left_d);
            best_i[2] = ((swap != 0) ? right_i : left_i);
            best_d[3] = ((swap != 0) ? left_d : right_d);
            best_i[3] = ((swap != 0) ? left_i : right_i);
        }
    }
    {
        {
            float left_d = best_d[4];
            float right_d = best_d[5];
            int left_i = best_i[4];
            int right_i = best_i[5];
            int swap = ((left_d < right_d) ? 1 : 0);
            best_d[4] = ((swap != 0) ? right_d : left_d);
            best_i[4] = ((swap != 0) ? right_i : left_i);
            best_d[5] = ((swap != 0) ? left_d : right_d);
            best_i[5] = ((swap != 0) ? left_i : right_i);
        }
    }
    {
        {
            float left_d = best_d[6];
            float right_d = best_d[7];
            int left_i = best_i[6];
            int right_i = best_i[7];
            int swap = ((left_d < right_d) ? 1 : 0);
            best_d[6] = ((swap != 0) ? right_d : left_d);
            best_i[6] = ((swap != 0) ? right_i : left_i);
            best_d[7] = ((swap != 0) ? left_d : right_d);
            best_i[7] = ((swap != 0) ? left_i : right_i);
        }
    }
    {
        {
            float left_d = best_d[8];
            float right_d = best_d[9];
            int left_i = best_i[8];
            int right_i = best_i[9];
            int swap = ((right_d < left_d) ? 1 : 0);
            best_d[8] = ((swap != 0) ? right_d : left_d);
            best_i[8] = ((swap != 0) ? right_i : left_i);
            best_d[9] = ((swap != 0) ? left_d : right_d);
            best_i[9] = ((swap != 0) ? left_i : right_i);
        }
    }
    {
        {
            float left_d = best_d[10];
            float right_d = best_d[11];
            int left_i = best_i[10];
            int right_i = best_i[11];
            int swap = ((right_d < left_d) ? 1 : 0);
            best_d[10] = ((swap != 0) ? right_d : left_d);
            best_i[10] = ((swap != 0) ? right_i : left_i);
            best_d[11] = ((swap != 0) ? left_d : right_d);
            best_i[11] = ((swap != 0) ? left_i : right_i);
        }
    }
    {
        {
            float left_d = best_d[12];
            float right_d = best_d[13];
            int left_i = best_i[12];
            int right_i = best_i[13];
            int swap = ((left_d < right_d) ? 1 : 0);
            best_d[12] = ((swap != 0) ? right_d : left_d);
            best_i[12] = ((swap != 0) ? right_i : left_i);
            best_d[13] = ((swap != 0) ? left_d : right_d);
            best_i[13] = ((swap != 0) ? left_i : right_i);
        }
    }
    {
        {
            float left_d = best_d[14];
            float right_d = best_d[15];
            int left_i = best_i[14];
            int right_i = best_i[15];
            int swap = ((left_d < right_d) ? 1 : 0);
            best_d[14] = ((swap != 0) ? right_d : left_d);
            best_i[14] = ((swap != 0) ? right_i : left_i);
            best_d[15] = ((swap != 0) ? left_d : right_d);
            best_i[15] = ((swap != 0) ? left_i : right_i);
        }
    }
    {
        {
            float left_d = best_d[16];
            float right_d = best_d[17];
            int left_i = best_i[16];
            int right_i = best_i[17];
            int swap = ((right_d < left_d) ? 1 : 0);
            best_d[16] = ((swap != 0) ? right_d : left_d);
            best_i[16] = ((swap != 0) ? right_i : left_i);
            best_d[17] = ((swap != 0) ? left_d : right_d);
            best_i[17] = ((swap != 0) ? left_i : right_i);
        }
    }
    {
        {
            float left_d = best_d[18];
            float right_d = best_d[19];
            int left_i = best_i[18];
            int right_i = best_i[19];
            int swap = ((right_d < left_d) ? 1 : 0);
            best_d[18] = ((swap != 0) ? right_d : left_d);
            best_i[18] = ((swap != 0) ? right_i : left_i);
            best_d[19] = ((swap != 0) ? left_d : right_d);
            best_i[19] = ((swap != 0) ? left_i : right_i);
        }
    }
    {
        {
            float left_d = best_d[20];
            float right_d = best_d[21];
            int left_i = best_i[20];
            int right_i = best_i[21];
            int swap = ((left_d < right_d) ? 1 : 0);
            best_d[20] = ((swap != 0) ? right_d : left_d);
            best_i[20] = ((swap != 0) ? right_i : left_i);
            best_d[21] = ((swap != 0) ? left_d : right_d);
            best_i[21] = ((swap != 0) ? left_i : right_i);
        }
    }
    {
        {
            float left_d = best_d[22];
            float right_d = best_d[23];
            int left_i = best_i[22];
            int right_i = best_i[23];
            int swap = ((left_d < right_d) ? 1 : 0);
            best_d[22] = ((swap != 0) ? right_d : left_d);
            best_i[22] = ((swap != 0) ? right_i : left_i);
            best_d[23] = ((swap != 0) ? left_d : right_d);
            best_i[23] = ((swap != 0) ? left_i : right_i);
        }
    }
    {
        {
            float left_d = best_d[24];
            float right_d = best_d[25];
            int left_i = best_i[24];
            int right_i = best_i[25];
            int swap = ((right_d < left_d) ? 1 : 0);
            best_d[24] = ((swap != 0) ? right_d : left_d);
            best_i[24] = ((swap != 0) ? right_i : left_i);
            best_d[25] = ((swap != 0) ? left_d : right_d);
            best_i[25] = ((swap != 0) ? left_i : right_i);
        }
    }
    {
        {
            float left_d = best_d[26];
            float right_d = best_d[27];
            int left_i = best_i[26];
            int right_i = best_i[27];
            int swap = ((right_d < left_d) ? 1 : 0);
            best_d[26] = ((swap != 0) ? right_d : left_d);
            best_i[26] = ((swap != 0) ? right_i : left_i);
            best_d[27] = ((swap != 0) ? left_d : right_d);
            best_i[27] = ((swap != 0) ? left_i : right_i);
        }
    }
    {
        {
            float left_d = best_d[28];
            float right_d = best_d[29];
            int left_i = best_i[28];
            int right_i = best_i[29];
            int swap = ((left_d < right_d) ? 1 : 0);
            best_d[28] = ((swap != 0) ? right_d : left_d);
            best_i[28] = ((swap != 0) ? right_i : left_i);
            best_d[29] = ((swap != 0) ? left_d : right_d);
            best_i[29] = ((swap != 0) ? left_i : right_i);
        }
    }
    {
        {
            float left_d = best_d[30];
            float right_d = best_d[31];
            int left_i = best_i[30];
            int right_i = best_i[31];
            int swap = ((left_d < right_d) ? 1 : 0);
            best_d[30] = ((swap != 0) ? right_d : left_d);
            best_i[30] = ((swap != 0) ? right_i : left_i);
            best_d[31] = ((swap != 0) ? left_d : right_d);
            best_i[31] = ((swap != 0) ? left_i : right_i);
        }
    }
    {
        {
            float left_d = best_d[32];
            float right_d = best_d[33];
            int left_i = best_i[32];
            int right_i = best_i[33];
            int swap = ((right_d < left_d) ? 1 : 0);
            best_d[32] = ((swap != 0) ? right_d : left_d);
            best_i[32] = ((swap != 0) ? right_i : left_i);
            best_d[33] = ((swap != 0) ? left_d : right_d);
            best_i[33] = ((swap != 0) ? left_i : right_i);
        }
    }
    {
        {
            float left_d = best_d[34];
            float right_d = best_d[35];
            int left_i = best_i[34];
            int right_i = best_i[35];
            int swap = ((right_d < left_d) ? 1 : 0);
            best_d[34] = ((swap != 0) ? right_d : left_d);
            best_i[34] = ((swap != 0) ? right_i : left_i);
            best_d[35] = ((swap != 0) ? left_d : right_d);
            best_i[35] = ((swap != 0) ? left_i : right_i);
        }
    }
    {
        {
            float left_d = best_d[36];
            float right_d = best_d[37];
            int left_i = best_i[36];
            int right_i = best_i[37];
            int swap = ((left_d < right_d) ? 1 : 0);
            best_d[36] = ((swap != 0) ? right_d : left_d);
            best_i[36] = ((swap != 0) ? right_i : left_i);
            best_d[37] = ((swap != 0) ? left_d : right_d);
            best_i[37] = ((swap != 0) ? left_i : right_i);
        }
    }
    {
        {
            float left_d = best_d[38];
            float right_d = best_d[39];
            int left_i = best_i[38];
            int right_i = best_i[39];
            int swap = ((left_d < right_d) ? 1 : 0);
            best_d[38] = ((swap != 0) ? right_d : left_d);
            best_i[38] = ((swap != 0) ? right_i : left_i);
            best_d[39] = ((swap != 0) ? left_d : right_d);
            best_i[39] = ((swap != 0) ? left_i : right_i);
        }
    }
    {
        {
            float left_d = best_d[40];
            float right_d = best_d[41];
            int left_i = best_i[40];
            int right_i = best_i[41];
            int swap = ((right_d < left_d) ? 1 : 0);
            best_d[40] = ((swap != 0) ? right_d : left_d);
            best_i[40] = ((swap != 0) ? right_i : left_i);
            best_d[41] = ((swap != 0) ? left_d : right_d);
            best_i[41] = ((swap != 0) ? left_i : right_i);
        }
    }
    {
        {
            float left_d = best_d[42];
            float right_d = best_d[43];
            int left_i = best_i[42];
            int right_i = best_i[43];
            int swap = ((right_d < left_d) ? 1 : 0);
            best_d[42] = ((swap != 0) ? right_d : left_d);
            best_i[42] = ((swap != 0) ? right_i : left_i);
            best_d[43] = ((swap != 0) ? left_d : right_d);
            best_i[43] = ((swap != 0) ? left_i : right_i);
        }
    }
    {
        {
            float left_d = best_d[44];
            float right_d = best_d[45];
            int left_i = best_i[44];
            int right_i = best_i[45];
            int swap = ((left_d < right_d) ? 1 : 0);
            best_d[44] = ((swap != 0) ? right_d : left_d);
            best_i[44] = ((swap != 0) ? right_i : left_i);
            best_d[45] = ((swap != 0) ? left_d : right_d);
            best_i[45] = ((swap != 0) ? left_i : right_i);
        }
    }
    {
        {
            float left_d = best_d[46];
            float right_d = best_d[47];
            int left_i = best_i[46];
            int right_i = best_i[47];
            int swap = ((left_d < right_d) ? 1 : 0);
            best_d[46] = ((swap != 0) ? right_d : left_d);
            best_i[46] = ((swap != 0) ? right_i : left_i);
            best_d[47] = ((swap != 0) ? left_d : right_d);
            best_i[47] = ((swap != 0) ? left_i : right_i);
        }
    }
    {
        {
            float left_d = best_d[48];
            float right_d = best_d[49];
            int left_i = best_i[48];
            int right_i = best_i[49];
            int swap = ((right_d < left_d) ? 1 : 0);
            best_d[48] = ((swap != 0) ? right_d : left_d);
            best_i[48] = ((swap != 0) ? right_i : left_i);
            best_d[49] = ((swap != 0) ? left_d : right_d);
            best_i[49] = ((swap != 0) ? left_i : right_i);
        }
    }
    {
        {
            float left_d = best_d[50];
            float right_d = best_d[51];
            int left_i = best_i[50];
            int right_i = best_i[51];
            int swap = ((right_d < left_d) ? 1 : 0);
            best_d[50] = ((swap != 0) ? right_d : left_d);
            best_i[50] = ((swap != 0) ? right_i : left_i);
            best_d[51] = ((swap != 0) ? left_d : right_d);
            best_i[51] = ((swap != 0) ? left_i : right_i);
        }
    }
    {
        {
            float left_d = best_d[52];
            float right_d = best_d[53];
            int left_i = best_i[52];
            int right_i = best_i[53];
            int swap = ((left_d < right_d) ? 1 : 0);
            best_d[52] = ((swap != 0) ? right_d : left_d);
            best_i[52] = ((swap != 0) ? right_i : left_i);
            best_d[53] = ((swap != 0) ? left_d : right_d);
            best_i[53] = ((swap != 0) ? left_i : right_i);
        }
    }
    {
        {
            float left_d = best_d[54];
            float right_d = best_d[55];
            int left_i = best_i[54];
            int right_i = best_i[55];
            int swap = ((left_d < right_d) ? 1 : 0);
            best_d[54] = ((swap != 0) ? right_d : left_d);
            best_i[54] = ((swap != 0) ? right_i : left_i);
            best_d[55] = ((swap != 0) ? left_d : right_d);
            best_i[55] = ((swap != 0) ? left_i : right_i);
        }
    }
    {
        {
            float left_d = best_d[56];
            float right_d = best_d[57];
            int left_i = best_i[56];
            int right_i = best_i[57];
            int swap = ((right_d < left_d) ? 1 : 0);
            best_d[56] = ((swap != 0) ? right_d : left_d);
            best_i[56] = ((swap != 0) ? right_i : left_i);
            best_d[57] = ((swap != 0) ? left_d : right_d);
            best_i[57] = ((swap != 0) ? left_i : right_i);
        }
    }
    {
        {
            float left_d = best_d[58];
            float right_d = best_d[59];
            int left_i = best_i[58];
            int right_i = best_i[59];
            int swap = ((right_d < left_d) ? 1 : 0);
            best_d[58] = ((swap != 0) ? right_d : left_d);
            best_i[58] = ((swap != 0) ? right_i : left_i);
            best_d[59] = ((swap != 0) ? left_d : right_d);
            best_i[59] = ((swap != 0) ? left_i : right_i);
        }
    }
    {
        {
            float left_d = best_d[60];
            float right_d = best_d[61];
            int left_i = best_i[60];
            int right_i = best_i[61];
            int swap = ((left_d < right_d) ? 1 : 0);
            best_d[60] = ((swap != 0) ? right_d : left_d);
            best_i[60] = ((swap != 0) ? right_i : left_i);
            best_d[61] = ((swap != 0) ? left_d : right_d);
            best_i[61] = ((swap != 0) ? left_i : right_i);
        }
    }
    {
        {
            float left_d = best_d[62];
            float right_d = best_d[63];
            int left_i = best_i[62];
            int right_i = best_i[63];
            int swap = ((left_d < right_d) ? 1 : 0);
            best_d[62] = ((swap != 0) ? right_d : left_d);
            best_i[62] = ((swap != 0) ? right_i : left_i);
            best_d[63] = ((swap != 0) ? left_d : right_d);
            best_i[63] = ((swap != 0) ? left_i : right_i);
        }
    }
    {
        {
            float left_d = best_d[0];
            float right_d = best_d[4];
            int left_i = best_i[0];
            int right_i = best_i[4];
            int swap = ((right_d < left_d) ? 1 : 0);
            best_d[0] = ((swap != 0) ? right_d : left_d);
            best_i[0] = ((swap != 0) ? right_i : left_i);
            best_d[4] = ((swap != 0) ? left_d : right_d);
            best_i[4] = ((swap != 0) ? left_i : right_i);
        }
    }
    {
        {
            float left_d = best_d[1];
            float right_d = best_d[5];
            int left_i = best_i[1];
            int right_i = best_i[5];
            int swap = ((right_d < left_d) ? 1 : 0);
            best_d[1] = ((swap != 0) ? right_d : left_d);
            best_i[1] = ((swap != 0) ? right_i : left_i);
            best_d[5] = ((swap != 0) ? left_d : right_d);
            best_i[5] = ((swap != 0) ? left_i : right_i);
        }
    }
    {
        {
            float left_d = best_d[2];
            float right_d = best_d[6];
            int left_i = best_i[2];
            int right_i = best_i[6];
            int swap = ((right_d < left_d) ? 1 : 0);
            best_d[2] = ((swap != 0) ? right_d : left_d);
            best_i[2] = ((swap != 0) ? right_i : left_i);
            best_d[6] = ((swap != 0) ? left_d : right_d);
            best_i[6] = ((swap != 0) ? left_i : right_i);
        }
    }
    {
        {
            float left_d = best_d[3];
            float right_d = best_d[7];
            int left_i = best_i[3];
            int right_i = best_i[7];
            int swap = ((right_d < left_d) ? 1 : 0);
            best_d[3] = ((swap != 0) ? right_d : left_d);
            best_i[3] = ((swap != 0) ? right_i : left_i);
            best_d[7] = ((swap != 0) ? left_d : right_d);
            best_i[7] = ((swap != 0) ? left_i : right_i);
        }
    }
    {
        {
            float left_d = best_d[8];
            float right_d = best_d[12];
            int left_i = best_i[8];
            int right_i = best_i[12];
            int swap = ((left_d < right_d) ? 1 : 0);
            best_d[8] = ((swap != 0) ? right_d : left_d);
            best_i[8] = ((swap != 0) ? right_i : left_i);
            best_d[12] = ((swap != 0) ? left_d : right_d);
            best_i[12] = ((swap != 0) ? left_i : right_i);
        }
    }
    {
        {
            float left_d = best_d[9];
            float right_d = best_d[13];
            int left_i = best_i[9];
            int right_i = best_i[13];
            int swap = ((left_d < right_d) ? 1 : 0);
            best_d[9] = ((swap != 0) ? right_d : left_d);
            best_i[9] = ((swap != 0) ? right_i : left_i);
            best_d[13] = ((swap != 0) ? left_d : right_d);
            best_i[13] = ((swap != 0) ? left_i : right_i);
        }
    }
    {
        {
            float left_d = best_d[10];
            float right_d = best_d[14];
            int left_i = best_i[10];
            int right_i = best_i[14];
            int swap = ((left_d < right_d) ? 1 : 0);
            best_d[10] = ((swap != 0) ? right_d : left_d);
            best_i[10] = ((swap != 0) ? right_i : left_i);
            best_d[14] = ((swap != 0) ? left_d : right_d);
            best_i[14] = ((swap != 0) ? left_i : right_i);
        }
    }
    {
        {
            float left_d = best_d[11];
            float right_d = best_d[15];
            int left_i = best_i[11];
            int right_i = best_i[15];
            int swap = ((left_d < right_d) ? 1 : 0);
            best_d[11] = ((swap != 0) ? right_d : left_d);
            best_i[11] = ((swap != 0) ? right_i : left_i);
            best_d[15] = ((swap != 0) ? left_d : right_d);
            best_i[15] = ((swap != 0) ? left_i : right_i);
        }
    }
    {
        {
            float left_d = best_d[16];
            float right_d = best_d[20];
            int left_i = best_i[16];
            int right_i = best_i[20];
            int swap = ((right_d < left_d) ? 1 : 0);
            best_d[16] = ((swap != 0) ? right_d : left_d);
            best_i[16] = ((swap != 0) ? right_i : left_i);
            best_d[20] = ((swap != 0) ? left_d : right_d);
            best_i[20] = ((swap != 0) ? left_i : right_i);
        }
    }
    {
        {
            float left_d = best_d[17];
            float right_d = best_d[21];
            int left_i = best_i[17];
            int right_i = best_i[21];
            int swap = ((right_d < left_d) ? 1 : 0);
            best_d[17] = ((swap != 0) ? right_d : left_d);
            best_i[17] = ((swap != 0) ? right_i : left_i);
            best_d[21] = ((swap != 0) ? left_d : right_d);
            best_i[21] = ((swap != 0) ? left_i : right_i);
        }
    }
    {
        {
            float left_d = best_d[18];
            float right_d = best_d[22];
            int left_i = best_i[18];
            int right_i = best_i[22];
            int swap = ((right_d < left_d) ? 1 : 0);
            best_d[18] = ((swap != 0) ? right_d : left_d);
            best_i[18] = ((swap != 0) ? right_i : left_i);
            best_d[22] = ((swap != 0) ? left_d : right_d);
            best_i[22] = ((swap != 0) ? left_i : right_i);
        }
    }
    {
        {
            float left_d = best_d[19];
            float right_d = best_d[23];
            int left_i = best_i[19];
            int right_i = best_i[23];
            int swap = ((right_d < left_d) ? 1 : 0);
            best_d[19] = ((swap != 0) ? right_d : left_d);
            best_i[19] = ((swap != 0) ? right_i : left_i);
            best_d[23] = ((swap != 0) ? left_d : right_d);
            best_i[23] = ((swap != 0) ? left_i : right_i);
        }
    }
    {
        {
            float left_d = best_d[24];
            float right_d = best_d[28];
            int left_i = best_i[24];
            int right_i = best_i[28];
            int swap = ((left_d < right_d) ? 1 : 0);
            best_d[24] = ((swap != 0) ? right_d : left_d);
            best_i[24] = ((swap != 0) ? right_i : left_i);
            best_d[28] = ((swap != 0) ? left_d : right_d);
            best_i[28] = ((swap != 0) ? left_i : right_i);
        }
    }
    {
        {
            float left_d = best_d[25];
            float right_d = best_d[29];
            int left_i = best_i[25];
            int right_i = best_i[29];
            int swap = ((left_d < right_d) ? 1 : 0);
            best_d[25] = ((swap != 0) ? right_d : left_d);
            best_i[25] = ((swap != 0) ? right_i : left_i);
            best_d[29] = ((swap != 0) ? left_d : right_d);
            best_i[29] = ((swap != 0) ? left_i : right_i);
        }
    }
    {
        {
            float left_d = best_d[26];
            float right_d = best_d[30];
            int left_i = best_i[26];
            int right_i = best_i[30];
            int swap = ((left_d < right_d) ? 1 : 0);
            best_d[26] = ((swap != 0) ? right_d : left_d);
            best_i[26] = ((swap != 0) ? right_i : left_i);
            best_d[30] = ((swap != 0) ? left_d : right_d);
            best_i[30] = ((swap != 0) ? left_i : right_i);
        }
    }
    {
        {
            float left_d = best_d[27];
            float right_d = best_d[31];
            int left_i = best_i[27];
            int right_i = best_i[31];
            int swap = ((left_d < right_d) ? 1 : 0);
            best_d[27] = ((swap != 0) ? right_d : left_d);
            best_i[27] = ((swap != 0) ? right_i : left_i);
            best_d[31] = ((swap != 0) ? left_d : right_d);
            best_i[31] = ((swap != 0) ? left_i : right_i);
        }
    }
    {
        {
            float left_d = best_d[32];
            float right_d = best_d[36];
            int left_i = best_i[32];
            int right_i = best_i[36];
            int swap = ((right_d < left_d) ? 1 : 0);
            best_d[32] = ((swap != 0) ? right_d : left_d);
            best_i[32] = ((swap != 0) ? right_i : left_i);
            best_d[36] = ((swap != 0) ? left_d : right_d);
            best_i[36] = ((swap != 0) ? left_i : right_i);
        }
    }
    {
        {
            float left_d = best_d[33];
            float right_d = best_d[37];
            int left_i = best_i[33];
            int right_i = best_i[37];
            int swap = ((right_d < left_d) ? 1 : 0);
            best_d[33] = ((swap != 0) ? right_d : left_d);
            best_i[33] = ((swap != 0) ? right_i : left_i);
            best_d[37] = ((swap != 0) ? left_d : right_d);
            best_i[37] = ((swap != 0) ? left_i : right_i);
        }
    }
    {
        {
            float left_d = best_d[34];
            float right_d = best_d[38];
            int left_i = best_i[34];
            int right_i = best_i[38];
            int swap = ((right_d < left_d) ? 1 : 0);
            best_d[34] = ((swap != 0) ? right_d : left_d);
            best_i[34] = ((swap != 0) ? right_i : left_i);
            best_d[38] = ((swap != 0) ? left_d : right_d);
            best_i[38] = ((swap != 0) ? left_i : right_i);
        }
    }
    {
        {
            float left_d = best_d[35];
            float right_d = best_d[39];
            int left_i = best_i[35];
            int right_i = best_i[39];
            int swap = ((right_d < left_d) ? 1 : 0);
            best_d[35] = ((swap != 0) ? right_d : left_d);
            best_i[35] = ((swap != 0) ? right_i : left_i);
            best_d[39] = ((swap != 0) ? left_d : right_d);
            best_i[39] = ((swap != 0) ? left_i : right_i);
        }
    }
    {
        {
            float left_d = best_d[40];
            float right_d = best_d[44];
            int left_i = best_i[40];
            int right_i = best_i[44];
            int swap = ((left_d < right_d) ? 1 : 0);
            best_d[40] = ((swap != 0) ? right_d : left_d);
            best_i[40] = ((swap != 0) ? right_i : left_i);
            best_d[44] = ((swap != 0) ? left_d : right_d);
            best_i[44] = ((swap != 0) ? left_i : right_i);
        }
    }
    {
        {
            float left_d = best_d[41];
            float right_d = best_d[45];
            int left_i = best_i[41];
            int right_i = best_i[45];
            int swap = ((left_d < right_d) ? 1 : 0);
            best_d[41] = ((swap != 0) ? right_d : left_d);
            best_i[41] = ((swap != 0) ? right_i : left_i);
            best_d[45] = ((swap != 0) ? left_d : right_d);
            best_i[45] = ((swap != 0) ? left_i : right_i);
        }
    }
    {
        {
            float left_d = best_d[42];
            float right_d = best_d[46];
            int left_i = best_i[42];
            int right_i = best_i[46];
            int swap = ((left_d < right_d) ? 1 : 0);
            best_d[42] = ((swap != 0) ? right_d : left_d);
            best_i[42] = ((swap != 0) ? right_i : left_i);
            best_d[46] = ((swap != 0) ? left_d : right_d);
            best_i[46] = ((swap != 0) ? left_i : right_i);
        }
    }
    {
        {
            float left_d = best_d[43];
            float right_d = best_d[47];
            int left_i = best_i[43];
            int right_i = best_i[47];
            int swap = ((left_d < right_d) ? 1 : 0);
            best_d[43] = ((swap != 0) ? right_d : left_d);
            best_i[43] = ((swap != 0) ? right_i : left_i);
            best_d[47] = ((swap != 0) ? left_d : right_d);
            best_i[47] = ((swap != 0) ? left_i : right_i);
        }
    }
    {
        {
            float left_d = best_d[48];
            float right_d = best_d[52];
            int left_i = best_i[48];
            int right_i = best_i[52];
            int swap = ((right_d < left_d) ? 1 : 0);
            best_d[48] = ((swap != 0) ? right_d : left_d);
            best_i[48] = ((swap != 0) ? right_i : left_i);
            best_d[52] = ((swap != 0) ? left_d : right_d);
            best_i[52] = ((swap != 0) ? left_i : right_i);
        }
    }
    {
        {
            float left_d = best_d[49];
            float right_d = best_d[53];
            int left_i = best_i[49];
            int right_i = best_i[53];
            int swap = ((right_d < left_d) ? 1 : 0);
            best_d[49] = ((swap != 0) ? right_d : left_d);
            best_i[49] = ((swap != 0) ? right_i : left_i);
            best_d[53] = ((swap != 0) ? left_d : right_d);
            best_i[53] = ((swap != 0) ? left_i : right_i);
        }
    }
    {
        {
            float left_d = best_d[50];
            float right_d = best_d[54];
            int left_i = best_i[50];
            int right_i = best_i[54];
            int swap = ((right_d < left_d) ? 1 : 0);
            best_d[50] = ((swap != 0) ? right_d : left_d);
            best_i[50] = ((swap != 0) ? right_i : left_i);
            best_d[54] = ((swap != 0) ? left_d : right_d);
            best_i[54] = ((swap != 0) ? left_i : right_i);
        }
    }
    {
        {
            float left_d = best_d[51];
            float right_d = best_d[55];
            int left_i = best_i[51];
            int right_i = best_i[55];
            int swap = ((right_d < left_d) ? 1 : 0);
            best_d[51] = ((swap != 0) ? right_d : left_d);
            best_i[51] = ((swap != 0) ? right_i : left_i);
            best_d[55] = ((swap != 0) ? left_d : right_d);
            best_i[55] = ((swap != 0) ? left_i : right_i);
        }
    }
    {
        {
            float left_d = best_d[56];
            float right_d = best_d[60];
            int left_i = best_i[56];
            int right_i = best_i[60];
            int swap = ((left_d < right_d) ? 1 : 0);
            best_d[56] = ((swap != 0) ? right_d : left_d);
            best_i[56] = ((swap != 0) ? right_i : left_i);
            best_d[60] = ((swap != 0) ? left_d : right_d);
            best_i[60] = ((swap != 0) ? left_i : right_i);
        }
    }
    {
        {
            float left_d = best_d[57];
            float right_d = best_d[61];
            int left_i = best_i[57];
            int right_i = best_i[61];
            int swap = ((left_d < right_d) ? 1 : 0);
            best_d[57] = ((swap != 0) ? right_d : left_d);
            best_i[57] = ((swap != 0) ? right_i : left_i);
            best_d[61] = ((swap != 0) ? left_d : right_d);
            best_i[61] = ((swap != 0) ? left_i : right_i);
        }
    }
    {
        {
            float left_d = best_d[58];
            float right_d = best_d[62];
            int left_i = best_i[58];
            int right_i = best_i[62];
            int swap = ((left_d < right_d) ? 1 : 0);
            best_d[58] = ((swap != 0) ? right_d : left_d);
            best_i[58] = ((swap != 0) ? right_i : left_i);
            best_d[62] = ((swap != 0) ? left_d : right_d);
            best_i[62] = ((swap != 0) ? left_i : right_i);
        }
    }
    {
        {
            float left_d = best_d[59];
            float right_d = best_d[63];
            int left_i = best_i[59];
            int right_i = best_i[63];
            int swap = ((left_d < right_d) ? 1 : 0);
            best_d[59] = ((swap != 0) ? right_d : left_d);
            best_i[59] = ((swap != 0) ? right_i : left_i);
            best_d[63] = ((swap != 0) ? left_d : right_d);
            best_i[63] = ((swap != 0) ? left_i : right_i);
        }
    }
    {
        {
            float left_d = best_d[0];
            float right_d = best_d[2];
            int left_i = best_i[0];
            int right_i = best_i[2];
            int swap = ((right_d < left_d) ? 1 : 0);
            best_d[0] = ((swap != 0) ? right_d : left_d);
            best_i[0] = ((swap != 0) ? right_i : left_i);
            best_d[2] = ((swap != 0) ? left_d : right_d);
            best_i[2] = ((swap != 0) ? left_i : right_i);
        }
    }
    {
        {
            float left_d = best_d[1];
            float right_d = best_d[3];
            int left_i = best_i[1];
            int right_i = best_i[3];
            int swap = ((right_d < left_d) ? 1 : 0);
            best_d[1] = ((swap != 0) ? right_d : left_d);
            best_i[1] = ((swap != 0) ? right_i : left_i);
            best_d[3] = ((swap != 0) ? left_d : right_d);
            best_i[3] = ((swap != 0) ? left_i : right_i);
        }
    }
    {
        {
            float left_d = best_d[4];
            float right_d = best_d[6];
            int left_i = best_i[4];
            int right_i = best_i[6];
            int swap = ((right_d < left_d) ? 1 : 0);
            best_d[4] = ((swap != 0) ? right_d : left_d);
            best_i[4] = ((swap != 0) ? right_i : left_i);
            best_d[6] = ((swap != 0) ? left_d : right_d);
            best_i[6] = ((swap != 0) ? left_i : right_i);
        }
    }
    {
        {
            float left_d = best_d[5];
            float right_d = best_d[7];
            int left_i = best_i[5];
            int right_i = best_i[7];
            int swap = ((right_d < left_d) ? 1 : 0);
            best_d[5] = ((swap != 0) ? right_d : left_d);
            best_i[5] = ((swap != 0) ? right_i : left_i);
            best_d[7] = ((swap != 0) ? left_d : right_d);
            best_i[7] = ((swap != 0) ? left_i : right_i);
        }
    }
    {
        {
            float left_d = best_d[8];
            float right_d = best_d[10];
            int left_i = best_i[8];
            int right_i = best_i[10];
            int swap = ((left_d < right_d) ? 1 : 0);
            best_d[8] = ((swap != 0) ? right_d : left_d);
            best_i[8] = ((swap != 0) ? right_i : left_i);
            best_d[10] = ((swap != 0) ? left_d : right_d);
            best_i[10] = ((swap != 0) ? left_i : right_i);
        }
    }
    {
        {
            float left_d = best_d[9];
            float right_d = best_d[11];
            int left_i = best_i[9];
            int right_i = best_i[11];
            int swap = ((left_d < right_d) ? 1 : 0);
            best_d[9] = ((swap != 0) ? right_d : left_d);
            best_i[9] = ((swap != 0) ? right_i : left_i);
            best_d[11] = ((swap != 0) ? left_d : right_d);
            best_i[11] = ((swap != 0) ? left_i : right_i);
        }
    }
    {
        {
            float left_d = best_d[12];
            float right_d = best_d[14];
            int left_i = best_i[12];
            int right_i = best_i[14];
            int swap = ((left_d < right_d) ? 1 : 0);
            best_d[12] = ((swap != 0) ? right_d : left_d);
            best_i[12] = ((swap != 0) ? right_i : left_i);
            best_d[14] = ((swap != 0) ? left_d : right_d);
            best_i[14] = ((swap != 0) ? left_i : right_i);
        }
    }
    {
        {
            float left_d = best_d[13];
            float right_d = best_d[15];
            int left_i = best_i[13];
            int right_i = best_i[15];
            int swap = ((left_d < right_d) ? 1 : 0);
            best_d[13] = ((swap != 0) ? right_d : left_d);
            best_i[13] = ((swap != 0) ? right_i : left_i);
            best_d[15] = ((swap != 0) ? left_d : right_d);
            best_i[15] = ((swap != 0) ? left_i : right_i);
        }
    }
    {
        {
            float left_d = best_d[16];
            float right_d = best_d[18];
            int left_i = best_i[16];
            int right_i = best_i[18];
            int swap = ((right_d < left_d) ? 1 : 0);
            best_d[16] = ((swap != 0) ? right_d : left_d);
            best_i[16] = ((swap != 0) ? right_i : left_i);
            best_d[18] = ((swap != 0) ? left_d : right_d);
            best_i[18] = ((swap != 0) ? left_i : right_i);
        }
    }
    {
        {
            float left_d = best_d[17];
            float right_d = best_d[19];
            int left_i = best_i[17];
            int right_i = best_i[19];
            int swap = ((right_d < left_d) ? 1 : 0);
            best_d[17] = ((swap != 0) ? right_d : left_d);
            best_i[17] = ((swap != 0) ? right_i : left_i);
            best_d[19] = ((swap != 0) ? left_d : right_d);
            best_i[19] = ((swap != 0) ? left_i : right_i);
        }
    }
    {
        {
            float left_d = best_d[20];
            float right_d = best_d[22];
            int left_i = best_i[20];
            int right_i = best_i[22];
            int swap = ((right_d < left_d) ? 1 : 0);
            best_d[20] = ((swap != 0) ? right_d : left_d);
            best_i[20] = ((swap != 0) ? right_i : left_i);
            best_d[22] = ((swap != 0) ? left_d : right_d);
            best_i[22] = ((swap != 0) ? left_i : right_i);
        }
    }
    {
        {
            float left_d = best_d[21];
            float right_d = best_d[23];
            int left_i = best_i[21];
            int right_i = best_i[23];
            int swap = ((right_d < left_d) ? 1 : 0);
            best_d[21] = ((swap != 0) ? right_d : left_d);
            best_i[21] = ((swap != 0) ? right_i : left_i);
            best_d[23] = ((swap != 0) ? left_d : right_d);
            best_i[23] = ((swap != 0) ? left_i : right_i);
        }
    }
    {
        {
            float left_d = best_d[24];
            float right_d = best_d[26];
            int left_i = best_i[24];
            int right_i = best_i[26];
            int swap = ((left_d < right_d) ? 1 : 0);
            best_d[24] = ((swap != 0) ? right_d : left_d);
            best_i[24] = ((swap != 0) ? right_i : left_i);
            best_d[26] = ((swap != 0) ? left_d : right_d);
            best_i[26] = ((swap != 0) ? left_i : right_i);
        }
    }
    {
        {
            float left_d = best_d[25];
            float right_d = best_d[27];
            int left_i = best_i[25];
            int right_i = best_i[27];
            int swap = ((left_d < right_d) ? 1 : 0);
            best_d[25] = ((swap != 0) ? right_d : left_d);
            best_i[25] = ((swap != 0) ? right_i : left_i);
            best_d[27] = ((swap != 0) ? left_d : right_d);
            best_i[27] = ((swap != 0) ? left_i : right_i);
        }
    }
    {
        {
            float left_d = best_d[28];
            float right_d = best_d[30];
            int left_i = best_i[28];
            int right_i = best_i[30];
            int swap = ((left_d < right_d) ? 1 : 0);
            best_d[28] = ((swap != 0) ? right_d : left_d);
            best_i[28] = ((swap != 0) ? right_i : left_i);
            best_d[30] = ((swap != 0) ? left_d : right_d);
            best_i[30] = ((swap != 0) ? left_i : right_i);
        }
    }
    {
        {
            float left_d = best_d[29];
            float right_d = best_d[31];
            int left_i = best_i[29];
            int right_i = best_i[31];
            int swap = ((left_d < right_d) ? 1 : 0);
            best_d[29] = ((swap != 0) ? right_d : left_d);
            best_i[29] = ((swap != 0) ? right_i : left_i);
            best_d[31] = ((swap != 0) ? left_d : right_d);
            best_i[31] = ((swap != 0) ? left_i : right_i);
        }
    }
    {
        {
            float left_d = best_d[32];
            float right_d = best_d[34];
            int left_i = best_i[32];
            int right_i = best_i[34];
            int swap = ((right_d < left_d) ? 1 : 0);
            best_d[32] = ((swap != 0) ? right_d : left_d);
            best_i[32] = ((swap != 0) ? right_i : left_i);
            best_d[34] = ((swap != 0) ? left_d : right_d);
            best_i[34] = ((swap != 0) ? left_i : right_i);
        }
    }
    {
        {
            float left_d = best_d[33];
            float right_d = best_d[35];
            int left_i = best_i[33];
            int right_i = best_i[35];
            int swap = ((right_d < left_d) ? 1 : 0);
            best_d[33] = ((swap != 0) ? right_d : left_d);
            best_i[33] = ((swap != 0) ? right_i : left_i);
            best_d[35] = ((swap != 0) ? left_d : right_d);
            best_i[35] = ((swap != 0) ? left_i : right_i);
        }
    }
    {
        {
            float left_d = best_d[36];
            float right_d = best_d[38];
            int left_i = best_i[36];
            int right_i = best_i[38];
            int swap = ((right_d < left_d) ? 1 : 0);
            best_d[36] = ((swap != 0) ? right_d : left_d);
            best_i[36] = ((swap != 0) ? right_i : left_i);
            best_d[38] = ((swap != 0) ? left_d : right_d);
            best_i[38] = ((swap != 0) ? left_i : right_i);
        }
    }
    {
        {
            float left_d = best_d[37];
            float right_d = best_d[39];
            int left_i = best_i[37];
            int right_i = best_i[39];
            int swap = ((right_d < left_d) ? 1 : 0);
            best_d[37] = ((swap != 0) ? right_d : left_d);
            best_i[37] = ((swap != 0) ? right_i : left_i);
            best_d[39] = ((swap != 0) ? left_d : right_d);
            best_i[39] = ((swap != 0) ? left_i : right_i);
        }
    }
    {
        {
            float left_d = best_d[40];
            float right_d = best_d[42];
            int left_i = best_i[40];
            int right_i = best_i[42];
            int swap = ((left_d < right_d) ? 1 : 0);
            best_d[40] = ((swap != 0) ? right_d : left_d);
            best_i[40] = ((swap != 0) ? right_i : left_i);
            best_d[42] = ((swap != 0) ? left_d : right_d);
            best_i[42] = ((swap != 0) ? left_i : right_i);
        }
    }
    {
        {
            float left_d = best_d[41];
            float right_d = best_d[43];
            int left_i = best_i[41];
            int right_i = best_i[43];
            int swap = ((left_d < right_d) ? 1 : 0);
            best_d[41] = ((swap != 0) ? right_d : left_d);
            best_i[41] = ((swap != 0) ? right_i : left_i);
            best_d[43] = ((swap != 0) ? left_d : right_d);
            best_i[43] = ((swap != 0) ? left_i : right_i);
        }
    }
    {
        {
            float left_d = best_d[44];
            float right_d = best_d[46];
            int left_i = best_i[44];
            int right_i = best_i[46];
            int swap = ((left_d < right_d) ? 1 : 0);
            best_d[44] = ((swap != 0) ? right_d : left_d);
            best_i[44] = ((swap != 0) ? right_i : left_i);
            best_d[46] = ((swap != 0) ? left_d : right_d);
            best_i[46] = ((swap != 0) ? left_i : right_i);
        }
    }
    {
        {
            float left_d = best_d[45];
            float right_d = best_d[47];
            int left_i = best_i[45];
            int right_i = best_i[47];
            int swap = ((left_d < right_d) ? 1 : 0);
            best_d[45] = ((swap != 0) ? right_d : left_d);
            best_i[45] = ((swap != 0) ? right_i : left_i);
            best_d[47] = ((swap != 0) ? left_d : right_d);
            best_i[47] = ((swap != 0) ? left_i : right_i);
        }
    }
    {
        {
            float left_d = best_d[48];
            float right_d = best_d[50];
            int left_i = best_i[48];
            int right_i = best_i[50];
            int swap = ((right_d < left_d) ? 1 : 0);
            best_d[48] = ((swap != 0) ? right_d : left_d);
            best_i[48] = ((swap != 0) ? right_i : left_i);
            best_d[50] = ((swap != 0) ? left_d : right_d);
            best_i[50] = ((swap != 0) ? left_i : right_i);
        }
    }
    {
        {
            float left_d = best_d[49];
            float right_d = best_d[51];
            int left_i = best_i[49];
            int right_i = best_i[51];
            int swap = ((right_d < left_d) ? 1 : 0);
            best_d[49] = ((swap != 0) ? right_d : left_d);
            best_i[49] = ((swap != 0) ? right_i : left_i);
            best_d[51] = ((swap != 0) ? left_d : right_d);
            best_i[51] = ((swap != 0) ? left_i : right_i);
        }
    }
    {
        {
            float left_d = best_d[52];
            float right_d = best_d[54];
            int left_i = best_i[52];
            int right_i = best_i[54];
            int swap = ((right_d < left_d) ? 1 : 0);
            best_d[52] = ((swap != 0) ? right_d : left_d);
            best_i[52] = ((swap != 0) ? right_i : left_i);
            best_d[54] = ((swap != 0) ? left_d : right_d);
            best_i[54] = ((swap != 0) ? left_i : right_i);
        }
    }
    {
        {
            float left_d = best_d[53];
            float right_d = best_d[55];
            int left_i = best_i[53];
            int right_i = best_i[55];
            int swap = ((right_d < left_d) ? 1 : 0);
            best_d[53] = ((swap != 0) ? right_d : left_d);
            best_i[53] = ((swap != 0) ? right_i : left_i);
            best_d[55] = ((swap != 0) ? left_d : right_d);
            best_i[55] = ((swap != 0) ? left_i : right_i);
        }
    }
    {
        {
            float left_d = best_d[56];
            float right_d = best_d[58];
            int left_i = best_i[56];
            int right_i = best_i[58];
            int swap = ((left_d < right_d) ? 1 : 0);
            best_d[56] = ((swap != 0) ? right_d : left_d);
            best_i[56] = ((swap != 0) ? right_i : left_i);
            best_d[58] = ((swap != 0) ? left_d : right_d);
            best_i[58] = ((swap != 0) ? left_i : right_i);
        }
    }
    {
        {
            float left_d = best_d[57];
            float right_d = best_d[59];
            int left_i = best_i[57];
            int right_i = best_i[59];
            int swap = ((left_d < right_d) ? 1 : 0);
            best_d[57] = ((swap != 0) ? right_d : left_d);
            best_i[57] = ((swap != 0) ? right_i : left_i);
            best_d[59] = ((swap != 0) ? left_d : right_d);
            best_i[59] = ((swap != 0) ? left_i : right_i);
        }
    }
    {
        {
            float left_d = best_d[60];
            float right_d = best_d[62];
            int left_i = best_i[60];
            int right_i = best_i[62];
            int swap = ((left_d < right_d) ? 1 : 0);
            best_d[60] = ((swap != 0) ? right_d : left_d);
            best_i[60] = ((swap != 0) ? right_i : left_i);
            best_d[62] = ((swap != 0) ? left_d : right_d);
            best_i[62] = ((swap != 0) ? left_i : right_i);
        }
    }
    {
        {
            float left_d = best_d[61];
            float right_d = best_d[63];
            int left_i = best_i[61];
            int right_i = best_i[63];
            int swap = ((left_d < right_d) ? 1 : 0);
            best_d[61] = ((swap != 0) ? right_d : left_d);
            best_i[61] = ((swap != 0) ? right_i : left_i);
            best_d[63] = ((swap != 0) ? left_d : right_d);
            best_i[63] = ((swap != 0) ? left_i : right_i);
        }
    }
    {
        {
            float left_d = best_d[0];
            float right_d = best_d[1];
            int left_i = best_i[0];
            int right_i = best_i[1];
            int swap = ((right_d < left_d) ? 1 : 0);
            best_d[0] = ((swap != 0) ? right_d : left_d);
            best_i[0] = ((swap != 0) ? right_i : left_i);
            best_d[1] = ((swap != 0) ? left_d : right_d);
            best_i[1] = ((swap != 0) ? left_i : right_i);
        }
    }
    {
        {
            float left_d = best_d[2];
            float right_d = best_d[3];
            int left_i = best_i[2];
            int right_i = best_i[3];
            int swap = ((right_d < left_d) ? 1 : 0);
            best_d[2] = ((swap != 0) ? right_d : left_d);
            best_i[2] = ((swap != 0) ? right_i : left_i);
            best_d[3] = ((swap != 0) ? left_d : right_d);
            best_i[3] = ((swap != 0) ? left_i : right_i);
        }
    }
    {
        {
            float left_d = best_d[4];
            float right_d = best_d[5];
            int left_i = best_i[4];
            int right_i = best_i[5];
            int swap = ((right_d < left_d) ? 1 : 0);
            best_d[4] = ((swap != 0) ? right_d : left_d);
            best_i[4] = ((swap != 0) ? right_i : left_i);
            best_d[5] = ((swap != 0) ? left_d : right_d);
            best_i[5] = ((swap != 0) ? left_i : right_i);
        }
    }
    {
        {
            float left_d = best_d[6];
            float right_d = best_d[7];
            int left_i = best_i[6];
            int right_i = best_i[7];
            int swap = ((right_d < left_d) ? 1 : 0);
            best_d[6] = ((swap != 0) ? right_d : left_d);
            best_i[6] = ((swap != 0) ? right_i : left_i);
            best_d[7] = ((swap != 0) ? left_d : right_d);
            best_i[7] = ((swap != 0) ? left_i : right_i);
        }
    }
    {
        {
            float left_d = best_d[8];
            float right_d = best_d[9];
            int left_i = best_i[8];
            int right_i = best_i[9];
            int swap = ((left_d < right_d) ? 1 : 0);
            best_d[8] = ((swap != 0) ? right_d : left_d);
            best_i[8] = ((swap != 0) ? right_i : left_i);
            best_d[9] = ((swap != 0) ? left_d : right_d);
            best_i[9] = ((swap != 0) ? left_i : right_i);
        }
    }
    {
        {
            float left_d = best_d[10];
            float right_d = best_d[11];
            int left_i = best_i[10];
            int right_i = best_i[11];
            int swap = ((left_d < right_d) ? 1 : 0);
            best_d[10] = ((swap != 0) ? right_d : left_d);
            best_i[10] = ((swap != 0) ? right_i : left_i);
            best_d[11] = ((swap != 0) ? left_d : right_d);
            best_i[11] = ((swap != 0) ? left_i : right_i);
        }
    }
    {
        {
            float left_d = best_d[12];
            float right_d = best_d[13];
            int left_i = best_i[12];
            int right_i = best_i[13];
            int swap = ((left_d < right_d) ? 1 : 0);
            best_d[12] = ((swap != 0) ? right_d : left_d);
            best_i[12] = ((swap != 0) ? right_i : left_i);
            best_d[13] = ((swap != 0) ? left_d : right_d);
            best_i[13] = ((swap != 0) ? left_i : right_i);
        }
    }
    {
        {
            float left_d = best_d[14];
            float right_d = best_d[15];
            int left_i = best_i[14];
            int right_i = best_i[15];
            int swap = ((left_d < right_d) ? 1 : 0);
            best_d[14] = ((swap != 0) ? right_d : left_d);
            best_i[14] = ((swap != 0) ? right_i : left_i);
            best_d[15] = ((swap != 0) ? left_d : right_d);
            best_i[15] = ((swap != 0) ? left_i : right_i);
        }
    }
    {
        {
            float left_d = best_d[16];
            float right_d = best_d[17];
            int left_i = best_i[16];
            int right_i = best_i[17];
            int swap = ((right_d < left_d) ? 1 : 0);
            best_d[16] = ((swap != 0) ? right_d : left_d);
            best_i[16] = ((swap != 0) ? right_i : left_i);
            best_d[17] = ((swap != 0) ? left_d : right_d);
            best_i[17] = ((swap != 0) ? left_i : right_i);
        }
    }
    {
        {
            float left_d = best_d[18];
            float right_d = best_d[19];
            int left_i = best_i[18];
            int right_i = best_i[19];
            int swap = ((right_d < left_d) ? 1 : 0);
            best_d[18] = ((swap != 0) ? right_d : left_d);
            best_i[18] = ((swap != 0) ? right_i : left_i);
            best_d[19] = ((swap != 0) ? left_d : right_d);
            best_i[19] = ((swap != 0) ? left_i : right_i);
        }
    }
    {
        {
            float left_d = best_d[20];
            float right_d = best_d[21];
            int left_i = best_i[20];
            int right_i = best_i[21];
            int swap = ((right_d < left_d) ? 1 : 0);
            best_d[20] = ((swap != 0) ? right_d : left_d);
            best_i[20] = ((swap != 0) ? right_i : left_i);
            best_d[21] = ((swap != 0) ? left_d : right_d);
            best_i[21] = ((swap != 0) ? left_i : right_i);
        }
    }
    {
        {
            float left_d = best_d[22];
            float right_d = best_d[23];
            int left_i = best_i[22];
            int right_i = best_i[23];
            int swap = ((right_d < left_d) ? 1 : 0);
            best_d[22] = ((swap != 0) ? right_d : left_d);
            best_i[22] = ((swap != 0) ? right_i : left_i);
            best_d[23] = ((swap != 0) ? left_d : right_d);
            best_i[23] = ((swap != 0) ? left_i : right_i);
        }
    }
    {
        {
            float left_d = best_d[24];
            float right_d = best_d[25];
            int left_i = best_i[24];
            int right_i = best_i[25];
            int swap = ((left_d < right_d) ? 1 : 0);
            best_d[24] = ((swap != 0) ? right_d : left_d);
            best_i[24] = ((swap != 0) ? right_i : left_i);
            best_d[25] = ((swap != 0) ? left_d : right_d);
            best_i[25] = ((swap != 0) ? left_i : right_i);
        }
    }
    {
        {
            float left_d = best_d[26];
            float right_d = best_d[27];
            int left_i = best_i[26];
            int right_i = best_i[27];
            int swap = ((left_d < right_d) ? 1 : 0);
            best_d[26] = ((swap != 0) ? right_d : left_d);
            best_i[26] = ((swap != 0) ? right_i : left_i);
            best_d[27] = ((swap != 0) ? left_d : right_d);
            best_i[27] = ((swap != 0) ? left_i : right_i);
        }
    }
    {
        {
            float left_d = best_d[28];
            float right_d = best_d[29];
            int left_i = best_i[28];
            int right_i = best_i[29];
            int swap = ((left_d < right_d) ? 1 : 0);
            best_d[28] = ((swap != 0) ? right_d : left_d);
            best_i[28] = ((swap != 0) ? right_i : left_i);
            best_d[29] = ((swap != 0) ? left_d : right_d);
            best_i[29] = ((swap != 0) ? left_i : right_i);
        }
    }
    {
        {
            float left_d = best_d[30];
            float right_d = best_d[31];
            int left_i = best_i[30];
            int right_i = best_i[31];
            int swap = ((left_d < right_d) ? 1 : 0);
            best_d[30] = ((swap != 0) ? right_d : left_d);
            best_i[30] = ((swap != 0) ? right_i : left_i);
            best_d[31] = ((swap != 0) ? left_d : right_d);
            best_i[31] = ((swap != 0) ? left_i : right_i);
        }
    }
    {
        {
            float left_d = best_d[32];
            float right_d = best_d[33];
            int left_i = best_i[32];
            int right_i = best_i[33];
            int swap = ((right_d < left_d) ? 1 : 0);
            best_d[32] = ((swap != 0) ? right_d : left_d);
            best_i[32] = ((swap != 0) ? right_i : left_i);
            best_d[33] = ((swap != 0) ? left_d : right_d);
            best_i[33] = ((swap != 0) ? left_i : right_i);
        }
    }
    {
        {
            float left_d = best_d[34];
            float right_d = best_d[35];
            int left_i = best_i[34];
            int right_i = best_i[35];
            int swap = ((right_d < left_d) ? 1 : 0);
            best_d[34] = ((swap != 0) ? right_d : left_d);
            best_i[34] = ((swap != 0) ? right_i : left_i);
            best_d[35] = ((swap != 0) ? left_d : right_d);
            best_i[35] = ((swap != 0) ? left_i : right_i);
        }
    }
    {
        {
            float left_d = best_d[36];
            float right_d = best_d[37];
            int left_i = best_i[36];
            int right_i = best_i[37];
            int swap = ((right_d < left_d) ? 1 : 0);
            best_d[36] = ((swap != 0) ? right_d : left_d);
            best_i[36] = ((swap != 0) ? right_i : left_i);
            best_d[37] = ((swap != 0) ? left_d : right_d);
            best_i[37] = ((swap != 0) ? left_i : right_i);
        }
    }
    {
        {
            float left_d = best_d[38];
            float right_d = best_d[39];
            int left_i = best_i[38];
            int right_i = best_i[39];
            int swap = ((right_d < left_d) ? 1 : 0);
            best_d[38] = ((swap != 0) ? right_d : left_d);
            best_i[38] = ((swap != 0) ? right_i : left_i);
            best_d[39] = ((swap != 0) ? left_d : right_d);
            best_i[39] = ((swap != 0) ? left_i : right_i);
        }
    }
    {
        {
            float left_d = best_d[40];
            float right_d = best_d[41];
            int left_i = best_i[40];
            int right_i = best_i[41];
            int swap = ((left_d < right_d) ? 1 : 0);
            best_d[40] = ((swap != 0) ? right_d : left_d);
            best_i[40] = ((swap != 0) ? right_i : left_i);
            best_d[41] = ((swap != 0) ? left_d : right_d);
            best_i[41] = ((swap != 0) ? left_i : right_i);
        }
    }
    {
        {
            float left_d = best_d[42];
            float right_d = best_d[43];
            int left_i = best_i[42];
            int right_i = best_i[43];
            int swap = ((left_d < right_d) ? 1 : 0);
            best_d[42] = ((swap != 0) ? right_d : left_d);
            best_i[42] = ((swap != 0) ? right_i : left_i);
            best_d[43] = ((swap != 0) ? left_d : right_d);
            best_i[43] = ((swap != 0) ? left_i : right_i);
        }
    }
    {
        {
            float left_d = best_d[44];
            float right_d = best_d[45];
            int left_i = best_i[44];
            int right_i = best_i[45];
            int swap = ((left_d < right_d) ? 1 : 0);
            best_d[44] = ((swap != 0) ? right_d : left_d);
            best_i[44] = ((swap != 0) ? right_i : left_i);
            best_d[45] = ((swap != 0) ? left_d : right_d);
            best_i[45] = ((swap != 0) ? left_i : right_i);
        }
    }
    {
        {
            float left_d = best_d[46];
            float right_d = best_d[47];
            int left_i = best_i[46];
            int right_i = best_i[47];
            int swap = ((left_d < right_d) ? 1 : 0);
            best_d[46] = ((swap != 0) ? right_d : left_d);
            best_i[46] = ((swap != 0) ? right_i : left_i);
            best_d[47] = ((swap != 0) ? left_d : right_d);
            best_i[47] = ((swap != 0) ? left_i : right_i);
        }
    }
    {
        {
            float left_d = best_d[48];
            float right_d = best_d[49];
            int left_i = best_i[48];
            int right_i = best_i[49];
            int swap = ((right_d < left_d) ? 1 : 0);
            best_d[48] = ((swap != 0) ? right_d : left_d);
            best_i[48] = ((swap != 0) ? right_i : left_i);
            best_d[49] = ((swap != 0) ? left_d : right_d);
            best_i[49] = ((swap != 0) ? left_i : right_i);
        }
    }
    {
        {
            float left_d = best_d[50];
            float right_d = best_d[51];
            int left_i = best_i[50];
            int right_i = best_i[51];
            int swap = ((right_d < left_d) ? 1 : 0);
            best_d[50] = ((swap != 0) ? right_d : left_d);
            best_i[50] = ((swap != 0) ? right_i : left_i);
            best_d[51] = ((swap != 0) ? left_d : right_d);
            best_i[51] = ((swap != 0) ? left_i : right_i);
        }
    }
    {
        {
            float left_d = best_d[52];
            float right_d = best_d[53];
            int left_i = best_i[52];
            int right_i = best_i[53];
            int swap = ((right_d < left_d) ? 1 : 0);
            best_d[52] = ((swap != 0) ? right_d : left_d);
            best_i[52] = ((swap != 0) ? right_i : left_i);
            best_d[53] = ((swap != 0) ? left_d : right_d);
            best_i[53] = ((swap != 0) ? left_i : right_i);
        }
    }
    {
        {
            float left_d = best_d[54];
            float right_d = best_d[55];
            int left_i = best_i[54];
            int right_i = best_i[55];
            int swap = ((right_d < left_d) ? 1 : 0);
            best_d[54] = ((swap != 0) ? right_d : left_d);
            best_i[54] = ((swap != 0) ? right_i : left_i);
            best_d[55] = ((swap != 0) ? left_d : right_d);
            best_i[55] = ((swap != 0) ? left_i : right_i);
        }
    }
    {
        {
            float left_d = best_d[56];
            float right_d = best_d[57];
            int left_i = best_i[56];
            int right_i = best_i[57];
            int swap = ((left_d < right_d) ? 1 : 0);
            best_d[56] = ((swap != 0) ? right_d : left_d);
            best_i[56] = ((swap != 0) ? right_i : left_i);
            best_d[57] = ((swap != 0) ? left_d : right_d);
            best_i[57] = ((swap != 0) ? left_i : right_i);
        }
    }
    {
        {
            float left_d = best_d[58];
            float right_d = best_d[59];
            int left_i = best_i[58];
            int right_i = best_i[59];
            int swap = ((left_d < right_d) ? 1 : 0);
            best_d[58] = ((swap != 0) ? right_d : left_d);
            best_i[58] = ((swap != 0) ? right_i : left_i);
            best_d[59] = ((swap != 0) ? left_d : right_d);
            best_i[59] = ((swap != 0) ? left_i : right_i);
        }
    }
    {
        {
            float left_d = best_d[60];
            float right_d = best_d[61];
            int left_i = best_i[60];
            int right_i = best_i[61];
            int swap = ((left_d < right_d) ? 1 : 0);
            best_d[60] = ((swap != 0) ? right_d : left_d);
            best_i[60] = ((swap != 0) ? right_i : left_i);
            best_d[61] = ((swap != 0) ? left_d : right_d);
            best_i[61] = ((swap != 0) ? left_i : right_i);
        }
    }
    {
        {
            float left_d = best_d[62];
            float right_d = best_d[63];
            int left_i = best_i[62];
            int right_i = best_i[63];
            int swap = ((left_d < right_d) ? 1 : 0);
            best_d[62] = ((swap != 0) ? right_d : left_d);
            best_i[62] = ((swap != 0) ? right_i : left_i);
            best_d[63] = ((swap != 0) ? left_d : right_d);
            best_i[63] = ((swap != 0) ? left_i : right_i);
        }
    }
    {
        {
            float left_d = best_d[0];
            float right_d = best_d[8];
            int left_i = best_i[0];
            int right_i = best_i[8];
            int swap = ((right_d < left_d) ? 1 : 0);
            best_d[0] = ((swap != 0) ? right_d : left_d);
            best_i[0] = ((swap != 0) ? right_i : left_i);
            best_d[8] = ((swap != 0) ? left_d : right_d);
            best_i[8] = ((swap != 0) ? left_i : right_i);
        }
    }
    {
        {
            float left_d = best_d[1];
            float right_d = best_d[9];
            int left_i = best_i[1];
            int right_i = best_i[9];
            int swap = ((right_d < left_d) ? 1 : 0);
            best_d[1] = ((swap != 0) ? right_d : left_d);
            best_i[1] = ((swap != 0) ? right_i : left_i);
            best_d[9] = ((swap != 0) ? left_d : right_d);
            best_i[9] = ((swap != 0) ? left_i : right_i);
        }
    }
    {
        {
            float left_d = best_d[2];
            float right_d = best_d[10];
            int left_i = best_i[2];
            int right_i = best_i[10];
            int swap = ((right_d < left_d) ? 1 : 0);
            best_d[2] = ((swap != 0) ? right_d : left_d);
            best_i[2] = ((swap != 0) ? right_i : left_i);
            best_d[10] = ((swap != 0) ? left_d : right_d);
            best_i[10] = ((swap != 0) ? left_i : right_i);
        }
    }
    {
        {
            float left_d = best_d[3];
            float right_d = best_d[11];
            int left_i = best_i[3];
            int right_i = best_i[11];
            int swap = ((right_d < left_d) ? 1 : 0);
            best_d[3] = ((swap != 0) ? right_d : left_d);
            best_i[3] = ((swap != 0) ? right_i : left_i);
            best_d[11] = ((swap != 0) ? left_d : right_d);
            best_i[11] = ((swap != 0) ? left_i : right_i);
        }
    }
    {
        {
            float left_d = best_d[4];
            float right_d = best_d[12];
            int left_i = best_i[4];
            int right_i = best_i[12];
            int swap = ((right_d < left_d) ? 1 : 0);
            best_d[4] = ((swap != 0) ? right_d : left_d);
            best_i[4] = ((swap != 0) ? right_i : left_i);
            best_d[12] = ((swap != 0) ? left_d : right_d);
            best_i[12] = ((swap != 0) ? left_i : right_i);
        }
    }
    {
        {
            float left_d = best_d[5];
            float right_d = best_d[13];
            int left_i = best_i[5];
            int right_i = best_i[13];
            int swap = ((right_d < left_d) ? 1 : 0);
            best_d[5] = ((swap != 0) ? right_d : left_d);
            best_i[5] = ((swap != 0) ? right_i : left_i);
            best_d[13] = ((swap != 0) ? left_d : right_d);
            best_i[13] = ((swap != 0) ? left_i : right_i);
        }
    }
    {
        {
            float left_d = best_d[6];
            float right_d = best_d[14];
            int left_i = best_i[6];
            int right_i = best_i[14];
            int swap = ((right_d < left_d) ? 1 : 0);
            best_d[6] = ((swap != 0) ? right_d : left_d);
            best_i[6] = ((swap != 0) ? right_i : left_i);
            best_d[14] = ((swap != 0) ? left_d : right_d);
            best_i[14] = ((swap != 0) ? left_i : right_i);
        }
    }
    {
        {
            float left_d = best_d[7];
            float right_d = best_d[15];
            int left_i = best_i[7];
            int right_i = best_i[15];
            int swap = ((right_d < left_d) ? 1 : 0);
            best_d[7] = ((swap != 0) ? right_d : left_d);
            best_i[7] = ((swap != 0) ? right_i : left_i);
            best_d[15] = ((swap != 0) ? left_d : right_d);
            best_i[15] = ((swap != 0) ? left_i : right_i);
        }
    }
    {
        {
            float left_d = best_d[16];
            float right_d = best_d[24];
            int left_i = best_i[16];
            int right_i = best_i[24];
            int swap = ((left_d < right_d) ? 1 : 0);
            best_d[16] = ((swap != 0) ? right_d : left_d);
            best_i[16] = ((swap != 0) ? right_i : left_i);
            best_d[24] = ((swap != 0) ? left_d : right_d);
            best_i[24] = ((swap != 0) ? left_i : right_i);
        }
    }
    {
        {
            float left_d = best_d[17];
            float right_d = best_d[25];
            int left_i = best_i[17];
            int right_i = best_i[25];
            int swap = ((left_d < right_d) ? 1 : 0);
            best_d[17] = ((swap != 0) ? right_d : left_d);
            best_i[17] = ((swap != 0) ? right_i : left_i);
            best_d[25] = ((swap != 0) ? left_d : right_d);
            best_i[25] = ((swap != 0) ? left_i : right_i);
        }
    }
    {
        {
            float left_d = best_d[18];
            float right_d = best_d[26];
            int left_i = best_i[18];
            int right_i = best_i[26];
            int swap = ((left_d < right_d) ? 1 : 0);
            best_d[18] = ((swap != 0) ? right_d : left_d);
            best_i[18] = ((swap != 0) ? right_i : left_i);
            best_d[26] = ((swap != 0) ? left_d : right_d);
            best_i[26] = ((swap != 0) ? left_i : right_i);
        }
    }
    {
        {
            float left_d = best_d[19];
            float right_d = best_d[27];
            int left_i = best_i[19];
            int right_i = best_i[27];
            int swap = ((left_d < right_d) ? 1 : 0);
            best_d[19] = ((swap != 0) ? right_d : left_d);
            best_i[19] = ((swap != 0) ? right_i : left_i);
            best_d[27] = ((swap != 0) ? left_d : right_d);
            best_i[27] = ((swap != 0) ? left_i : right_i);
        }
    }
    {
        {
            float left_d = best_d[20];
            float right_d = best_d[28];
            int left_i = best_i[20];
            int right_i = best_i[28];
            int swap = ((left_d < right_d) ? 1 : 0);
            best_d[20] = ((swap != 0) ? right_d : left_d);
            best_i[20] = ((swap != 0) ? right_i : left_i);
            best_d[28] = ((swap != 0) ? left_d : right_d);
            best_i[28] = ((swap != 0) ? left_i : right_i);
        }
    }
    {
        {
            float left_d = best_d[21];
            float right_d = best_d[29];
            int left_i = best_i[21];
            int right_i = best_i[29];
            int swap = ((left_d < right_d) ? 1 : 0);
            best_d[21] = ((swap != 0) ? right_d : left_d);
            best_i[21] = ((swap != 0) ? right_i : left_i);
            best_d[29] = ((swap != 0) ? left_d : right_d);
            best_i[29] = ((swap != 0) ? left_i : right_i);
        }
    }
    {
        {
            float left_d = best_d[22];
            float right_d = best_d[30];
            int left_i = best_i[22];
            int right_i = best_i[30];
            int swap = ((left_d < right_d) ? 1 : 0);
            best_d[22] = ((swap != 0) ? right_d : left_d);
            best_i[22] = ((swap != 0) ? right_i : left_i);
            best_d[30] = ((swap != 0) ? left_d : right_d);
            best_i[30] = ((swap != 0) ? left_i : right_i);
        }
    }
    {
        {
            float left_d = best_d[23];
            float right_d = best_d[31];
            int left_i = best_i[23];
            int right_i = best_i[31];
            int swap = ((left_d < right_d) ? 1 : 0);
            best_d[23] = ((swap != 0) ? right_d : left_d);
            best_i[23] = ((swap != 0) ? right_i : left_i);
            best_d[31] = ((swap != 0) ? left_d : right_d);
            best_i[31] = ((swap != 0) ? left_i : right_i);
        }
    }
    {
        {
            float left_d = best_d[32];
            float right_d = best_d[40];
            int left_i = best_i[32];
            int right_i = best_i[40];
            int swap = ((right_d < left_d) ? 1 : 0);
            best_d[32] = ((swap != 0) ? right_d : left_d);
            best_i[32] = ((swap != 0) ? right_i : left_i);
            best_d[40] = ((swap != 0) ? left_d : right_d);
            best_i[40] = ((swap != 0) ? left_i : right_i);
        }
    }
    {
        {
            float left_d = best_d[33];
            float right_d = best_d[41];
            int left_i = best_i[33];
            int right_i = best_i[41];
            int swap = ((right_d < left_d) ? 1 : 0);
            best_d[33] = ((swap != 0) ? right_d : left_d);
            best_i[33] = ((swap != 0) ? right_i : left_i);
            best_d[41] = ((swap != 0) ? left_d : right_d);
            best_i[41] = ((swap != 0) ? left_i : right_i);
        }
    }
    {
        {
            float left_d = best_d[34];
            float right_d = best_d[42];
            int left_i = best_i[34];
            int right_i = best_i[42];
            int swap = ((right_d < left_d) ? 1 : 0);
            best_d[34] = ((swap != 0) ? right_d : left_d);
            best_i[34] = ((swap != 0) ? right_i : left_i);
            best_d[42] = ((swap != 0) ? left_d : right_d);
            best_i[42] = ((swap != 0) ? left_i : right_i);
        }
    }
    {
        {
            float left_d = best_d[35];
            float right_d = best_d[43];
            int left_i = best_i[35];
            int right_i = best_i[43];
            int swap = ((right_d < left_d) ? 1 : 0);
            best_d[35] = ((swap != 0) ? right_d : left_d);
            best_i[35] = ((swap != 0) ? right_i : left_i);
            best_d[43] = ((swap != 0) ? left_d : right_d);
            best_i[43] = ((swap != 0) ? left_i : right_i);
        }
    }
    {
        {
            float left_d = best_d[36];
            float right_d = best_d[44];
            int left_i = best_i[36];
            int right_i = best_i[44];
            int swap = ((right_d < left_d) ? 1 : 0);
            best_d[36] = ((swap != 0) ? right_d : left_d);
            best_i[36] = ((swap != 0) ? right_i : left_i);
            best_d[44] = ((swap != 0) ? left_d : right_d);
            best_i[44] = ((swap != 0) ? left_i : right_i);
        }
    }
    {
        {
            float left_d = best_d[37];
            float right_d = best_d[45];
            int left_i = best_i[37];
            int right_i = best_i[45];
            int swap = ((right_d < left_d) ? 1 : 0);
            best_d[37] = ((swap != 0) ? right_d : left_d);
            best_i[37] = ((swap != 0) ? right_i : left_i);
            best_d[45] = ((swap != 0) ? left_d : right_d);
            best_i[45] = ((swap != 0) ? left_i : right_i);
        }
    }
    {
        {
            float left_d = best_d[38];
            float right_d = best_d[46];
            int left_i = best_i[38];
            int right_i = best_i[46];
            int swap = ((right_d < left_d) ? 1 : 0);
            best_d[38] = ((swap != 0) ? right_d : left_d);
            best_i[38] = ((swap != 0) ? right_i : left_i);
            best_d[46] = ((swap != 0) ? left_d : right_d);
            best_i[46] = ((swap != 0) ? left_i : right_i);
        }
    }
    {
        {
            float left_d = best_d[39];
            float right_d = best_d[47];
            int left_i = best_i[39];
            int right_i = best_i[47];
            int swap = ((right_d < left_d) ? 1 : 0);
            best_d[39] = ((swap != 0) ? right_d : left_d);
            best_i[39] = ((swap != 0) ? right_i : left_i);
            best_d[47] = ((swap != 0) ? left_d : right_d);
            best_i[47] = ((swap != 0) ? left_i : right_i);
        }
    }
    {
        {
            float left_d = best_d[48];
            float right_d = best_d[56];
            int left_i = best_i[48];
            int right_i = best_i[56];
            int swap = ((left_d < right_d) ? 1 : 0);
            best_d[48] = ((swap != 0) ? right_d : left_d);
            best_i[48] = ((swap != 0) ? right_i : left_i);
            best_d[56] = ((swap != 0) ? left_d : right_d);
            best_i[56] = ((swap != 0) ? left_i : right_i);
        }
    }
    {
        {
            float left_d = best_d[49];
            float right_d = best_d[57];
            int left_i = best_i[49];
            int right_i = best_i[57];
            int swap = ((left_d < right_d) ? 1 : 0);
            best_d[49] = ((swap != 0) ? right_d : left_d);
            best_i[49] = ((swap != 0) ? right_i : left_i);
            best_d[57] = ((swap != 0) ? left_d : right_d);
            best_i[57] = ((swap != 0) ? left_i : right_i);
        }
    }
    {
        {
            float left_d = best_d[50];
            float right_d = best_d[58];
            int left_i = best_i[50];
            int right_i = best_i[58];
            int swap = ((left_d < right_d) ? 1 : 0);
            best_d[50] = ((swap != 0) ? right_d : left_d);
            best_i[50] = ((swap != 0) ? right_i : left_i);
            best_d[58] = ((swap != 0) ? left_d : right_d);
            best_i[58] = ((swap != 0) ? left_i : right_i);
        }
    }
    {
        {
            float left_d = best_d[51];
            float right_d = best_d[59];
            int left_i = best_i[51];
            int right_i = best_i[59];
            int swap = ((left_d < right_d) ? 1 : 0);
            best_d[51] = ((swap != 0) ? right_d : left_d);
            best_i[51] = ((swap != 0) ? right_i : left_i);
            best_d[59] = ((swap != 0) ? left_d : right_d);
            best_i[59] = ((swap != 0) ? left_i : right_i);
        }
    }
    {
        {
            float left_d = best_d[52];
            float right_d = best_d[60];
            int left_i = best_i[52];
            int right_i = best_i[60];
            int swap = ((left_d < right_d) ? 1 : 0);
            best_d[52] = ((swap != 0) ? right_d : left_d);
            best_i[52] = ((swap != 0) ? right_i : left_i);
            best_d[60] = ((swap != 0) ? left_d : right_d);
            best_i[60] = ((swap != 0) ? left_i : right_i);
        }
    }
    {
        {
            float left_d = best_d[53];
            float right_d = best_d[61];
            int left_i = best_i[53];
            int right_i = best_i[61];
            int swap = ((left_d < right_d) ? 1 : 0);
            best_d[53] = ((swap != 0) ? right_d : left_d);
            best_i[53] = ((swap != 0) ? right_i : left_i);
            best_d[61] = ((swap != 0) ? left_d : right_d);
            best_i[61] = ((swap != 0) ? left_i : right_i);
        }
    }
    {
        {
            float left_d = best_d[54];
            float right_d = best_d[62];
            int left_i = best_i[54];
            int right_i = best_i[62];
            int swap = ((left_d < right_d) ? 1 : 0);
            best_d[54] = ((swap != 0) ? right_d : left_d);
            best_i[54] = ((swap != 0) ? right_i : left_i);
            best_d[62] = ((swap != 0) ? left_d : right_d);
            best_i[62] = ((swap != 0) ? left_i : right_i);
        }
    }
    {
        {
            float left_d = best_d[55];
            float right_d = best_d[63];
            int left_i = best_i[55];
            int right_i = best_i[63];
            int swap = ((left_d < right_d) ? 1 : 0);
            best_d[55] = ((swap != 0) ? right_d : left_d);
            best_i[55] = ((swap != 0) ? right_i : left_i);
            best_d[63] = ((swap != 0) ? left_d : right_d);
            best_i[63] = ((swap != 0) ? left_i : right_i);
        }
    }
    {
        {
            float left_d = best_d[0];
            float right_d = best_d[4];
            int left_i = best_i[0];
            int right_i = best_i[4];
            int swap = ((right_d < left_d) ? 1 : 0);
            best_d[0] = ((swap != 0) ? right_d : left_d);
            best_i[0] = ((swap != 0) ? right_i : left_i);
            best_d[4] = ((swap != 0) ? left_d : right_d);
            best_i[4] = ((swap != 0) ? left_i : right_i);
        }
    }
    {
        {
            float left_d = best_d[1];
            float right_d = best_d[5];
            int left_i = best_i[1];
            int right_i = best_i[5];
            int swap = ((right_d < left_d) ? 1 : 0);
            best_d[1] = ((swap != 0) ? right_d : left_d);
            best_i[1] = ((swap != 0) ? right_i : left_i);
            best_d[5] = ((swap != 0) ? left_d : right_d);
            best_i[5] = ((swap != 0) ? left_i : right_i);
        }
    }
    {
        {
            float left_d = best_d[2];
            float right_d = best_d[6];
            int left_i = best_i[2];
            int right_i = best_i[6];
            int swap = ((right_d < left_d) ? 1 : 0);
            best_d[2] = ((swap != 0) ? right_d : left_d);
            best_i[2] = ((swap != 0) ? right_i : left_i);
            best_d[6] = ((swap != 0) ? left_d : right_d);
            best_i[6] = ((swap != 0) ? left_i : right_i);
        }
    }
    {
        {
            float left_d = best_d[3];
            float right_d = best_d[7];
            int left_i = best_i[3];
            int right_i = best_i[7];
            int swap = ((right_d < left_d) ? 1 : 0);
            best_d[3] = ((swap != 0) ? right_d : left_d);
            best_i[3] = ((swap != 0) ? right_i : left_i);
            best_d[7] = ((swap != 0) ? left_d : right_d);
            best_i[7] = ((swap != 0) ? left_i : right_i);
        }
    }
    {
        {
            float left_d = best_d[8];
            float right_d = best_d[12];
            int left_i = best_i[8];
            int right_i = best_i[12];
            int swap = ((right_d < left_d) ? 1 : 0);
            best_d[8] = ((swap != 0) ? right_d : left_d);
            best_i[8] = ((swap != 0) ? right_i : left_i);
            best_d[12] = ((swap != 0) ? left_d : right_d);
            best_i[12] = ((swap != 0) ? left_i : right_i);
        }
    }
    {
        {
            float left_d = best_d[9];
            float right_d = best_d[13];
            int left_i = best_i[9];
            int right_i = best_i[13];
            int swap = ((right_d < left_d) ? 1 : 0);
            best_d[9] = ((swap != 0) ? right_d : left_d);
            best_i[9] = ((swap != 0) ? right_i : left_i);
            best_d[13] = ((swap != 0) ? left_d : right_d);
            best_i[13] = ((swap != 0) ? left_i : right_i);
        }
    }
    {
        {
            float left_d = best_d[10];
            float right_d = best_d[14];
            int left_i = best_i[10];
            int right_i = best_i[14];
            int swap = ((right_d < left_d) ? 1 : 0);
            best_d[10] = ((swap != 0) ? right_d : left_d);
            best_i[10] = ((swap != 0) ? right_i : left_i);
            best_d[14] = ((swap != 0) ? left_d : right_d);
            best_i[14] = ((swap != 0) ? left_i : right_i);
        }
    }
    {
        {
            float left_d = best_d[11];
            float right_d = best_d[15];
            int left_i = best_i[11];
            int right_i = best_i[15];
            int swap = ((right_d < left_d) ? 1 : 0);
            best_d[11] = ((swap != 0) ? right_d : left_d);
            best_i[11] = ((swap != 0) ? right_i : left_i);
            best_d[15] = ((swap != 0) ? left_d : right_d);
            best_i[15] = ((swap != 0) ? left_i : right_i);
        }
    }
    {
        {
            float left_d = best_d[16];
            float right_d = best_d[20];
            int left_i = best_i[16];
            int right_i = best_i[20];
            int swap = ((left_d < right_d) ? 1 : 0);
            best_d[16] = ((swap != 0) ? right_d : left_d);
            best_i[16] = ((swap != 0) ? right_i : left_i);
            best_d[20] = ((swap != 0) ? left_d : right_d);
            best_i[20] = ((swap != 0) ? left_i : right_i);
        }
    }
    {
        {
            float left_d = best_d[17];
            float right_d = best_d[21];
            int left_i = best_i[17];
            int right_i = best_i[21];
            int swap = ((left_d < right_d) ? 1 : 0);
            best_d[17] = ((swap != 0) ? right_d : left_d);
            best_i[17] = ((swap != 0) ? right_i : left_i);
            best_d[21] = ((swap != 0) ? left_d : right_d);
            best_i[21] = ((swap != 0) ? left_i : right_i);
        }
    }
    {
        {
            float left_d = best_d[18];
            float right_d = best_d[22];
            int left_i = best_i[18];
            int right_i = best_i[22];
            int swap = ((left_d < right_d) ? 1 : 0);
            best_d[18] = ((swap != 0) ? right_d : left_d);
            best_i[18] = ((swap != 0) ? right_i : left_i);
            best_d[22] = ((swap != 0) ? left_d : right_d);
            best_i[22] = ((swap != 0) ? left_i : right_i);
        }
    }
    {
        {
            float left_d = best_d[19];
            float right_d = best_d[23];
            int left_i = best_i[19];
            int right_i = best_i[23];
            int swap = ((left_d < right_d) ? 1 : 0);
            best_d[19] = ((swap != 0) ? right_d : left_d);
            best_i[19] = ((swap != 0) ? right_i : left_i);
            best_d[23] = ((swap != 0) ? left_d : right_d);
            best_i[23] = ((swap != 0) ? left_i : right_i);
        }
    }
    {
        {
            float left_d = best_d[24];
            float right_d = best_d[28];
            int left_i = best_i[24];
            int right_i = best_i[28];
            int swap = ((left_d < right_d) ? 1 : 0);
            best_d[24] = ((swap != 0) ? right_d : left_d);
            best_i[24] = ((swap != 0) ? right_i : left_i);
            best_d[28] = ((swap != 0) ? left_d : right_d);
            best_i[28] = ((swap != 0) ? left_i : right_i);
        }
    }
    {
        {
            float left_d = best_d[25];
            float right_d = best_d[29];
            int left_i = best_i[25];
            int right_i = best_i[29];
            int swap = ((left_d < right_d) ? 1 : 0);
            best_d[25] = ((swap != 0) ? right_d : left_d);
            best_i[25] = ((swap != 0) ? right_i : left_i);
            best_d[29] = ((swap != 0) ? left_d : right_d);
            best_i[29] = ((swap != 0) ? left_i : right_i);
        }
    }
    {
        {
            float left_d = best_d[26];
            float right_d = best_d[30];
            int left_i = best_i[26];
            int right_i = best_i[30];
            int swap = ((left_d < right_d) ? 1 : 0);
            best_d[26] = ((swap != 0) ? right_d : left_d);
            best_i[26] = ((swap != 0) ? right_i : left_i);
            best_d[30] = ((swap != 0) ? left_d : right_d);
            best_i[30] = ((swap != 0) ? left_i : right_i);
        }
    }
    {
        {
            float left_d = best_d[27];
            float right_d = best_d[31];
            int left_i = best_i[27];
            int right_i = best_i[31];
            int swap = ((left_d < right_d) ? 1 : 0);
            best_d[27] = ((swap != 0) ? right_d : left_d);
            best_i[27] = ((swap != 0) ? right_i : left_i);
            best_d[31] = ((swap != 0) ? left_d : right_d);
            best_i[31] = ((swap != 0) ? left_i : right_i);
        }
    }
    {
        {
            float left_d = best_d[32];
            float right_d = best_d[36];
            int left_i = best_i[32];
            int right_i = best_i[36];
            int swap = ((right_d < left_d) ? 1 : 0);
            best_d[32] = ((swap != 0) ? right_d : left_d);
            best_i[32] = ((swap != 0) ? right_i : left_i);
            best_d[36] = ((swap != 0) ? left_d : right_d);
            best_i[36] = ((swap != 0) ? left_i : right_i);
        }
    }
    {
        {
            float left_d = best_d[33];
            float right_d = best_d[37];
            int left_i = best_i[33];
            int right_i = best_i[37];
            int swap = ((right_d < left_d) ? 1 : 0);
            best_d[33] = ((swap != 0) ? right_d : left_d);
            best_i[33] = ((swap != 0) ? right_i : left_i);
            best_d[37] = ((swap != 0) ? left_d : right_d);
            best_i[37] = ((swap != 0) ? left_i : right_i);
        }
    }
    {
        {
            float left_d = best_d[34];
            float right_d = best_d[38];
            int left_i = best_i[34];
            int right_i = best_i[38];
            int swap = ((right_d < left_d) ? 1 : 0);
            best_d[34] = ((swap != 0) ? right_d : left_d);
            best_i[34] = ((swap != 0) ? right_i : left_i);
            best_d[38] = ((swap != 0) ? left_d : right_d);
            best_i[38] = ((swap != 0) ? left_i : right_i);
        }
    }
    {
        {
            float left_d = best_d[35];
            float right_d = best_d[39];
            int left_i = best_i[35];
            int right_i = best_i[39];
            int swap = ((right_d < left_d) ? 1 : 0);
            best_d[35] = ((swap != 0) ? right_d : left_d);
            best_i[35] = ((swap != 0) ? right_i : left_i);
            best_d[39] = ((swap != 0) ? left_d : right_d);
            best_i[39] = ((swap != 0) ? left_i : right_i);
        }
    }
    {
        {
            float left_d = best_d[40];
            float right_d = best_d[44];
            int left_i = best_i[40];
            int right_i = best_i[44];
            int swap = ((right_d < left_d) ? 1 : 0);
            best_d[40] = ((swap != 0) ? right_d : left_d);
            best_i[40] = ((swap != 0) ? right_i : left_i);
            best_d[44] = ((swap != 0) ? left_d : right_d);
            best_i[44] = ((swap != 0) ? left_i : right_i);
        }
    }
    {
        {
            float left_d = best_d[41];
            float right_d = best_d[45];
            int left_i = best_i[41];
            int right_i = best_i[45];
            int swap = ((right_d < left_d) ? 1 : 0);
            best_d[41] = ((swap != 0) ? right_d : left_d);
            best_i[41] = ((swap != 0) ? right_i : left_i);
            best_d[45] = ((swap != 0) ? left_d : right_d);
            best_i[45] = ((swap != 0) ? left_i : right_i);
        }
    }
    {
        {
            float left_d = best_d[42];
            float right_d = best_d[46];
            int left_i = best_i[42];
            int right_i = best_i[46];
            int swap = ((right_d < left_d) ? 1 : 0);
            best_d[42] = ((swap != 0) ? right_d : left_d);
            best_i[42] = ((swap != 0) ? right_i : left_i);
            best_d[46] = ((swap != 0) ? left_d : right_d);
            best_i[46] = ((swap != 0) ? left_i : right_i);
        }
    }
    {
        {
            float left_d = best_d[43];
            float right_d = best_d[47];
            int left_i = best_i[43];
            int right_i = best_i[47];
            int swap = ((right_d < left_d) ? 1 : 0);
            best_d[43] = ((swap != 0) ? right_d : left_d);
            best_i[43] = ((swap != 0) ? right_i : left_i);
            best_d[47] = ((swap != 0) ? left_d : right_d);
            best_i[47] = ((swap != 0) ? left_i : right_i);
        }
    }
    {
        {
            float left_d = best_d[48];
            float right_d = best_d[52];
            int left_i = best_i[48];
            int right_i = best_i[52];
            int swap = ((left_d < right_d) ? 1 : 0);
            best_d[48] = ((swap != 0) ? right_d : left_d);
            best_i[48] = ((swap != 0) ? right_i : left_i);
            best_d[52] = ((swap != 0) ? left_d : right_d);
            best_i[52] = ((swap != 0) ? left_i : right_i);
        }
    }
    {
        {
            float left_d = best_d[49];
            float right_d = best_d[53];
            int left_i = best_i[49];
            int right_i = best_i[53];
            int swap = ((left_d < right_d) ? 1 : 0);
            best_d[49] = ((swap != 0) ? right_d : left_d);
            best_i[49] = ((swap != 0) ? right_i : left_i);
            best_d[53] = ((swap != 0) ? left_d : right_d);
            best_i[53] = ((swap != 0) ? left_i : right_i);
        }
    }
    {
        {
            float left_d = best_d[50];
            float right_d = best_d[54];
            int left_i = best_i[50];
            int right_i = best_i[54];
            int swap = ((left_d < right_d) ? 1 : 0);
            best_d[50] = ((swap != 0) ? right_d : left_d);
            best_i[50] = ((swap != 0) ? right_i : left_i);
            best_d[54] = ((swap != 0) ? left_d : right_d);
            best_i[54] = ((swap != 0) ? left_i : right_i);
        }
    }
    {
        {
            float left_d = best_d[51];
            float right_d = best_d[55];
            int left_i = best_i[51];
            int right_i = best_i[55];
            int swap = ((left_d < right_d) ? 1 : 0);
            best_d[51] = ((swap != 0) ? right_d : left_d);
            best_i[51] = ((swap != 0) ? right_i : left_i);
            best_d[55] = ((swap != 0) ? left_d : right_d);
            best_i[55] = ((swap != 0) ? left_i : right_i);
        }
    }
    {
        {
            float left_d = best_d[56];
            float right_d = best_d[60];
            int left_i = best_i[56];
            int right_i = best_i[60];
            int swap = ((left_d < right_d) ? 1 : 0);
            best_d[56] = ((swap != 0) ? right_d : left_d);
            best_i[56] = ((swap != 0) ? right_i : left_i);
            best_d[60] = ((swap != 0) ? left_d : right_d);
            best_i[60] = ((swap != 0) ? left_i : right_i);
        }
    }
    {
        {
            float left_d = best_d[57];
            float right_d = best_d[61];
            int left_i = best_i[57];
            int right_i = best_i[61];
            int swap = ((left_d < right_d) ? 1 : 0);
            best_d[57] = ((swap != 0) ? right_d : left_d);
            best_i[57] = ((swap != 0) ? right_i : left_i);
            best_d[61] = ((swap != 0) ? left_d : right_d);
            best_i[61] = ((swap != 0) ? left_i : right_i);
        }
    }
    {
        {
            float left_d = best_d[58];
            float right_d = best_d[62];
            int left_i = best_i[58];
            int right_i = best_i[62];
            int swap = ((left_d < right_d) ? 1 : 0);
            best_d[58] = ((swap != 0) ? right_d : left_d);
            best_i[58] = ((swap != 0) ? right_i : left_i);
            best_d[62] = ((swap != 0) ? left_d : right_d);
            best_i[62] = ((swap != 0) ? left_i : right_i);
        }
    }
    {
        {
            float left_d = best_d[59];
            float right_d = best_d[63];
            int left_i = best_i[59];
            int right_i = best_i[63];
            int swap = ((left_d < right_d) ? 1 : 0);
            best_d[59] = ((swap != 0) ? right_d : left_d);
            best_i[59] = ((swap != 0) ? right_i : left_i);
            best_d[63] = ((swap != 0) ? left_d : right_d);
            best_i[63] = ((swap != 0) ? left_i : right_i);
        }
    }
    {
        {
            float left_d = best_d[0];
            float right_d = best_d[2];
            int left_i = best_i[0];
            int right_i = best_i[2];
            int swap = ((right_d < left_d) ? 1 : 0);
            best_d[0] = ((swap != 0) ? right_d : left_d);
            best_i[0] = ((swap != 0) ? right_i : left_i);
            best_d[2] = ((swap != 0) ? left_d : right_d);
            best_i[2] = ((swap != 0) ? left_i : right_i);
        }
    }
    {
        {
            float left_d = best_d[1];
            float right_d = best_d[3];
            int left_i = best_i[1];
            int right_i = best_i[3];
            int swap = ((right_d < left_d) ? 1 : 0);
            best_d[1] = ((swap != 0) ? right_d : left_d);
            best_i[1] = ((swap != 0) ? right_i : left_i);
            best_d[3] = ((swap != 0) ? left_d : right_d);
            best_i[3] = ((swap != 0) ? left_i : right_i);
        }
    }
    {
        {
            float left_d = best_d[4];
            float right_d = best_d[6];
            int left_i = best_i[4];
            int right_i = best_i[6];
            int swap = ((right_d < left_d) ? 1 : 0);
            best_d[4] = ((swap != 0) ? right_d : left_d);
            best_i[4] = ((swap != 0) ? right_i : left_i);
            best_d[6] = ((swap != 0) ? left_d : right_d);
            best_i[6] = ((swap != 0) ? left_i : right_i);
        }
    }
    {
        {
            float left_d = best_d[5];
            float right_d = best_d[7];
            int left_i = best_i[5];
            int right_i = best_i[7];
            int swap = ((right_d < left_d) ? 1 : 0);
            best_d[5] = ((swap != 0) ? right_d : left_d);
            best_i[5] = ((swap != 0) ? right_i : left_i);
            best_d[7] = ((swap != 0) ? left_d : right_d);
            best_i[7] = ((swap != 0) ? left_i : right_i);
        }
    }
    {
        {
            float left_d = best_d[8];
            float right_d = best_d[10];
            int left_i = best_i[8];
            int right_i = best_i[10];
            int swap = ((right_d < left_d) ? 1 : 0);
            best_d[8] = ((swap != 0) ? right_d : left_d);
            best_i[8] = ((swap != 0) ? right_i : left_i);
            best_d[10] = ((swap != 0) ? left_d : right_d);
            best_i[10] = ((swap != 0) ? left_i : right_i);
        }
    }
    {
        {
            float left_d = best_d[9];
            float right_d = best_d[11];
            int left_i = best_i[9];
            int right_i = best_i[11];
            int swap = ((right_d < left_d) ? 1 : 0);
            best_d[9] = ((swap != 0) ? right_d : left_d);
            best_i[9] = ((swap != 0) ? right_i : left_i);
            best_d[11] = ((swap != 0) ? left_d : right_d);
            best_i[11] = ((swap != 0) ? left_i : right_i);
        }
    }
    {
        {
            float left_d = best_d[12];
            float right_d = best_d[14];
            int left_i = best_i[12];
            int right_i = best_i[14];
            int swap = ((right_d < left_d) ? 1 : 0);
            best_d[12] = ((swap != 0) ? right_d : left_d);
            best_i[12] = ((swap != 0) ? right_i : left_i);
            best_d[14] = ((swap != 0) ? left_d : right_d);
            best_i[14] = ((swap != 0) ? left_i : right_i);
        }
    }
    {
        {
            float left_d = best_d[13];
            float right_d = best_d[15];
            int left_i = best_i[13];
            int right_i = best_i[15];
            int swap = ((right_d < left_d) ? 1 : 0);
            best_d[13] = ((swap != 0) ? right_d : left_d);
            best_i[13] = ((swap != 0) ? right_i : left_i);
            best_d[15] = ((swap != 0) ? left_d : right_d);
            best_i[15] = ((swap != 0) ? left_i : right_i);
        }
    }
    {
        {
            float left_d = best_d[16];
            float right_d = best_d[18];
            int left_i = best_i[16];
            int right_i = best_i[18];
            int swap = ((left_d < right_d) ? 1 : 0);
            best_d[16] = ((swap != 0) ? right_d : left_d);
            best_i[16] = ((swap != 0) ? right_i : left_i);
            best_d[18] = ((swap != 0) ? left_d : right_d);
            best_i[18] = ((swap != 0) ? left_i : right_i);
        }
    }
    {
        {
            float left_d = best_d[17];
            float right_d = best_d[19];
            int left_i = best_i[17];
            int right_i = best_i[19];
            int swap = ((left_d < right_d) ? 1 : 0);
            best_d[17] = ((swap != 0) ? right_d : left_d);
            best_i[17] = ((swap != 0) ? right_i : left_i);
            best_d[19] = ((swap != 0) ? left_d : right_d);
            best_i[19] = ((swap != 0) ? left_i : right_i);
        }
    }
    {
        {
            float left_d = best_d[20];
            float right_d = best_d[22];
            int left_i = best_i[20];
            int right_i = best_i[22];
            int swap = ((left_d < right_d) ? 1 : 0);
            best_d[20] = ((swap != 0) ? right_d : left_d);
            best_i[20] = ((swap != 0) ? right_i : left_i);
            best_d[22] = ((swap != 0) ? left_d : right_d);
            best_i[22] = ((swap != 0) ? left_i : right_i);
        }
    }
    {
        {
            float left_d = best_d[21];
            float right_d = best_d[23];
            int left_i = best_i[21];
            int right_i = best_i[23];
            int swap = ((left_d < right_d) ? 1 : 0);
            best_d[21] = ((swap != 0) ? right_d : left_d);
            best_i[21] = ((swap != 0) ? right_i : left_i);
            best_d[23] = ((swap != 0) ? left_d : right_d);
            best_i[23] = ((swap != 0) ? left_i : right_i);
        }
    }
    {
        {
            float left_d = best_d[24];
            float right_d = best_d[26];
            int left_i = best_i[24];
            int right_i = best_i[26];
            int swap = ((left_d < right_d) ? 1 : 0);
            best_d[24] = ((swap != 0) ? right_d : left_d);
            best_i[24] = ((swap != 0) ? right_i : left_i);
            best_d[26] = ((swap != 0) ? left_d : right_d);
            best_i[26] = ((swap != 0) ? left_i : right_i);
        }
    }
    {
        {
            float left_d = best_d[25];
            float right_d = best_d[27];
            int left_i = best_i[25];
            int right_i = best_i[27];
            int swap = ((left_d < right_d) ? 1 : 0);
            best_d[25] = ((swap != 0) ? right_d : left_d);
            best_i[25] = ((swap != 0) ? right_i : left_i);
            best_d[27] = ((swap != 0) ? left_d : right_d);
            best_i[27] = ((swap != 0) ? left_i : right_i);
        }
    }
    {
        {
            float left_d = best_d[28];
            float right_d = best_d[30];
            int left_i = best_i[28];
            int right_i = best_i[30];
            int swap = ((left_d < right_d) ? 1 : 0);
            best_d[28] = ((swap != 0) ? right_d : left_d);
            best_i[28] = ((swap != 0) ? right_i : left_i);
            best_d[30] = ((swap != 0) ? left_d : right_d);
            best_i[30] = ((swap != 0) ? left_i : right_i);
        }
    }
    {
        {
            float left_d = best_d[29];
            float right_d = best_d[31];
            int left_i = best_i[29];
            int right_i = best_i[31];
            int swap = ((left_d < right_d) ? 1 : 0);
            best_d[29] = ((swap != 0) ? right_d : left_d);
            best_i[29] = ((swap != 0) ? right_i : left_i);
            best_d[31] = ((swap != 0) ? left_d : right_d);
            best_i[31] = ((swap != 0) ? left_i : right_i);
        }
    }
    {
        {
            float left_d = best_d[32];
            float right_d = best_d[34];
            int left_i = best_i[32];
            int right_i = best_i[34];
            int swap = ((right_d < left_d) ? 1 : 0);
            best_d[32] = ((swap != 0) ? right_d : left_d);
            best_i[32] = ((swap != 0) ? right_i : left_i);
            best_d[34] = ((swap != 0) ? left_d : right_d);
            best_i[34] = ((swap != 0) ? left_i : right_i);
        }
    }
    {
        {
            float left_d = best_d[33];
            float right_d = best_d[35];
            int left_i = best_i[33];
            int right_i = best_i[35];
            int swap = ((right_d < left_d) ? 1 : 0);
            best_d[33] = ((swap != 0) ? right_d : left_d);
            best_i[33] = ((swap != 0) ? right_i : left_i);
            best_d[35] = ((swap != 0) ? left_d : right_d);
            best_i[35] = ((swap != 0) ? left_i : right_i);
        }
    }
    {
        {
            float left_d = best_d[36];
            float right_d = best_d[38];
            int left_i = best_i[36];
            int right_i = best_i[38];
            int swap = ((right_d < left_d) ? 1 : 0);
            best_d[36] = ((swap != 0) ? right_d : left_d);
            best_i[36] = ((swap != 0) ? right_i : left_i);
            best_d[38] = ((swap != 0) ? left_d : right_d);
            best_i[38] = ((swap != 0) ? left_i : right_i);
        }
    }
    {
        {
            float left_d = best_d[37];
            float right_d = best_d[39];
            int left_i = best_i[37];
            int right_i = best_i[39];
            int swap = ((right_d < left_d) ? 1 : 0);
            best_d[37] = ((swap != 0) ? right_d : left_d);
            best_i[37] = ((swap != 0) ? right_i : left_i);
            best_d[39] = ((swap != 0) ? left_d : right_d);
            best_i[39] = ((swap != 0) ? left_i : right_i);
        }
    }
    {
        {
            float left_d = best_d[40];
            float right_d = best_d[42];
            int left_i = best_i[40];
            int right_i = best_i[42];
            int swap = ((right_d < left_d) ? 1 : 0);
            best_d[40] = ((swap != 0) ? right_d : left_d);
            best_i[40] = ((swap != 0) ? right_i : left_i);
            best_d[42] = ((swap != 0) ? left_d : right_d);
            best_i[42] = ((swap != 0) ? left_i : right_i);
        }
    }
    {
        {
            float left_d = best_d[41];
            float right_d = best_d[43];
            int left_i = best_i[41];
            int right_i = best_i[43];
            int swap = ((right_d < left_d) ? 1 : 0);
            best_d[41] = ((swap != 0) ? right_d : left_d);
            best_i[41] = ((swap != 0) ? right_i : left_i);
            best_d[43] = ((swap != 0) ? left_d : right_d);
            best_i[43] = ((swap != 0) ? left_i : right_i);
        }
    }
    {
        {
            float left_d = best_d[44];
            float right_d = best_d[46];
            int left_i = best_i[44];
            int right_i = best_i[46];
            int swap = ((right_d < left_d) ? 1 : 0);
            best_d[44] = ((swap != 0) ? right_d : left_d);
            best_i[44] = ((swap != 0) ? right_i : left_i);
            best_d[46] = ((swap != 0) ? left_d : right_d);
            best_i[46] = ((swap != 0) ? left_i : right_i);
        }
    }
    {
        {
            float left_d = best_d[45];
            float right_d = best_d[47];
            int left_i = best_i[45];
            int right_i = best_i[47];
            int swap = ((right_d < left_d) ? 1 : 0);
            best_d[45] = ((swap != 0) ? right_d : left_d);
            best_i[45] = ((swap != 0) ? right_i : left_i);
            best_d[47] = ((swap != 0) ? left_d : right_d);
            best_i[47] = ((swap != 0) ? left_i : right_i);
        }
    }
    {
        {
            float left_d = best_d[48];
            float right_d = best_d[50];
            int left_i = best_i[48];
            int right_i = best_i[50];
            int swap = ((left_d < right_d) ? 1 : 0);
            best_d[48] = ((swap != 0) ? right_d : left_d);
            best_i[48] = ((swap != 0) ? right_i : left_i);
            best_d[50] = ((swap != 0) ? left_d : right_d);
            best_i[50] = ((swap != 0) ? left_i : right_i);
        }
    }
    {
        {
            float left_d = best_d[49];
            float right_d = best_d[51];
            int left_i = best_i[49];
            int right_i = best_i[51];
            int swap = ((left_d < right_d) ? 1 : 0);
            best_d[49] = ((swap != 0) ? right_d : left_d);
            best_i[49] = ((swap != 0) ? right_i : left_i);
            best_d[51] = ((swap != 0) ? left_d : right_d);
            best_i[51] = ((swap != 0) ? left_i : right_i);
        }
    }
    {
        {
            float left_d = best_d[52];
            float right_d = best_d[54];
            int left_i = best_i[52];
            int right_i = best_i[54];
            int swap = ((left_d < right_d) ? 1 : 0);
            best_d[52] = ((swap != 0) ? right_d : left_d);
            best_i[52] = ((swap != 0) ? right_i : left_i);
            best_d[54] = ((swap != 0) ? left_d : right_d);
            best_i[54] = ((swap != 0) ? left_i : right_i);
        }
    }
    {
        {
            float left_d = best_d[53];
            float right_d = best_d[55];
            int left_i = best_i[53];
            int right_i = best_i[55];
            int swap = ((left_d < right_d) ? 1 : 0);
            best_d[53] = ((swap != 0) ? right_d : left_d);
            best_i[53] = ((swap != 0) ? right_i : left_i);
            best_d[55] = ((swap != 0) ? left_d : right_d);
            best_i[55] = ((swap != 0) ? left_i : right_i);
        }
    }
    {
        {
            float left_d = best_d[56];
            float right_d = best_d[58];
            int left_i = best_i[56];
            int right_i = best_i[58];
            int swap = ((left_d < right_d) ? 1 : 0);
            best_d[56] = ((swap != 0) ? right_d : left_d);
            best_i[56] = ((swap != 0) ? right_i : left_i);
            best_d[58] = ((swap != 0) ? left_d : right_d);
            best_i[58] = ((swap != 0) ? left_i : right_i);
        }
    }
    {
        {
            float left_d = best_d[57];
            float right_d = best_d[59];
            int left_i = best_i[57];
            int right_i = best_i[59];
            int swap = ((left_d < right_d) ? 1 : 0);
            best_d[57] = ((swap != 0) ? right_d : left_d);
            best_i[57] = ((swap != 0) ? right_i : left_i);
            best_d[59] = ((swap != 0) ? left_d : right_d);
            best_i[59] = ((swap != 0) ? left_i : right_i);
        }
    }
    {
        {
            float left_d = best_d[60];
            float right_d = best_d[62];
            int left_i = best_i[60];
            int right_i = best_i[62];
            int swap = ((left_d < right_d) ? 1 : 0);
            best_d[60] = ((swap != 0) ? right_d : left_d);
            best_i[60] = ((swap != 0) ? right_i : left_i);
            best_d[62] = ((swap != 0) ? left_d : right_d);
            best_i[62] = ((swap != 0) ? left_i : right_i);
        }
    }
    {
        {
            float left_d = best_d[61];
            float right_d = best_d[63];
            int left_i = best_i[61];
            int right_i = best_i[63];
            int swap = ((left_d < right_d) ? 1 : 0);
            best_d[61] = ((swap != 0) ? right_d : left_d);
            best_i[61] = ((swap != 0) ? right_i : left_i);
            best_d[63] = ((swap != 0) ? left_d : right_d);
            best_i[63] = ((swap != 0) ? left_i : right_i);
        }
    }
    {
        {
            float left_d = best_d[0];
            float right_d = best_d[1];
            int left_i = best_i[0];
            int right_i = best_i[1];
            int swap = ((right_d < left_d) ? 1 : 0);
            best_d[0] = ((swap != 0) ? right_d : left_d);
            best_i[0] = ((swap != 0) ? right_i : left_i);
            best_d[1] = ((swap != 0) ? left_d : right_d);
            best_i[1] = ((swap != 0) ? left_i : right_i);
        }
    }
    {
        {
            float left_d = best_d[2];
            float right_d = best_d[3];
            int left_i = best_i[2];
            int right_i = best_i[3];
            int swap = ((right_d < left_d) ? 1 : 0);
            best_d[2] = ((swap != 0) ? right_d : left_d);
            best_i[2] = ((swap != 0) ? right_i : left_i);
            best_d[3] = ((swap != 0) ? left_d : right_d);
            best_i[3] = ((swap != 0) ? left_i : right_i);
        }
    }
    {
        {
            float left_d = best_d[4];
            float right_d = best_d[5];
            int left_i = best_i[4];
            int right_i = best_i[5];
            int swap = ((right_d < left_d) ? 1 : 0);
            best_d[4] = ((swap != 0) ? right_d : left_d);
            best_i[4] = ((swap != 0) ? right_i : left_i);
            best_d[5] = ((swap != 0) ? left_d : right_d);
            best_i[5] = ((swap != 0) ? left_i : right_i);
        }
    }
    {
        {
            float left_d = best_d[6];
            float right_d = best_d[7];
            int left_i = best_i[6];
            int right_i = best_i[7];
            int swap = ((right_d < left_d) ? 1 : 0);
            best_d[6] = ((swap != 0) ? right_d : left_d);
            best_i[6] = ((swap != 0) ? right_i : left_i);
            best_d[7] = ((swap != 0) ? left_d : right_d);
            best_i[7] = ((swap != 0) ? left_i : right_i);
        }
    }
    {
        {
            float left_d = best_d[8];
            float right_d = best_d[9];
            int left_i = best_i[8];
            int right_i = best_i[9];
            int swap = ((right_d < left_d) ? 1 : 0);
            best_d[8] = ((swap != 0) ? right_d : left_d);
            best_i[8] = ((swap != 0) ? right_i : left_i);
            best_d[9] = ((swap != 0) ? left_d : right_d);
            best_i[9] = ((swap != 0) ? left_i : right_i);
        }
    }
    {
        {
            float left_d = best_d[10];
            float right_d = best_d[11];
            int left_i = best_i[10];
            int right_i = best_i[11];
            int swap = ((right_d < left_d) ? 1 : 0);
            best_d[10] = ((swap != 0) ? right_d : left_d);
            best_i[10] = ((swap != 0) ? right_i : left_i);
            best_d[11] = ((swap != 0) ? left_d : right_d);
            best_i[11] = ((swap != 0) ? left_i : right_i);
        }
    }
    {
        {
            float left_d = best_d[12];
            float right_d = best_d[13];
            int left_i = best_i[12];
            int right_i = best_i[13];
            int swap = ((right_d < left_d) ? 1 : 0);
            best_d[12] = ((swap != 0) ? right_d : left_d);
            best_i[12] = ((swap != 0) ? right_i : left_i);
            best_d[13] = ((swap != 0) ? left_d : right_d);
            best_i[13] = ((swap != 0) ? left_i : right_i);
        }
    }
    {
        {
            float left_d = best_d[14];
            float right_d = best_d[15];
            int left_i = best_i[14];
            int right_i = best_i[15];
            int swap = ((right_d < left_d) ? 1 : 0);
            best_d[14] = ((swap != 0) ? right_d : left_d);
            best_i[14] = ((swap != 0) ? right_i : left_i);
            best_d[15] = ((swap != 0) ? left_d : right_d);
            best_i[15] = ((swap != 0) ? left_i : right_i);
        }
    }
    {
        {
            float left_d = best_d[16];
            float right_d = best_d[17];
            int left_i = best_i[16];
            int right_i = best_i[17];
            int swap = ((left_d < right_d) ? 1 : 0);
            best_d[16] = ((swap != 0) ? right_d : left_d);
            best_i[16] = ((swap != 0) ? right_i : left_i);
            best_d[17] = ((swap != 0) ? left_d : right_d);
            best_i[17] = ((swap != 0) ? left_i : right_i);
        }
    }
    {
        {
            float left_d = best_d[18];
            float right_d = best_d[19];
            int left_i = best_i[18];
            int right_i = best_i[19];
            int swap = ((left_d < right_d) ? 1 : 0);
            best_d[18] = ((swap != 0) ? right_d : left_d);
            best_i[18] = ((swap != 0) ? right_i : left_i);
            best_d[19] = ((swap != 0) ? left_d : right_d);
            best_i[19] = ((swap != 0) ? left_i : right_i);
        }
    }
    {
        {
            float left_d = best_d[20];
            float right_d = best_d[21];
            int left_i = best_i[20];
            int right_i = best_i[21];
            int swap = ((left_d < right_d) ? 1 : 0);
            best_d[20] = ((swap != 0) ? right_d : left_d);
            best_i[20] = ((swap != 0) ? right_i : left_i);
            best_d[21] = ((swap != 0) ? left_d : right_d);
            best_i[21] = ((swap != 0) ? left_i : right_i);
        }
    }
    {
        {
            float left_d = best_d[22];
            float right_d = best_d[23];
            int left_i = best_i[22];
            int right_i = best_i[23];
            int swap = ((left_d < right_d) ? 1 : 0);
            best_d[22] = ((swap != 0) ? right_d : left_d);
            best_i[22] = ((swap != 0) ? right_i : left_i);
            best_d[23] = ((swap != 0) ? left_d : right_d);
            best_i[23] = ((swap != 0) ? left_i : right_i);
        }
    }
    {
        {
            float left_d = best_d[24];
            float right_d = best_d[25];
            int left_i = best_i[24];
            int right_i = best_i[25];
            int swap = ((left_d < right_d) ? 1 : 0);
            best_d[24] = ((swap != 0) ? right_d : left_d);
            best_i[24] = ((swap != 0) ? right_i : left_i);
            best_d[25] = ((swap != 0) ? left_d : right_d);
            best_i[25] = ((swap != 0) ? left_i : right_i);
        }
    }
    {
        {
            float left_d = best_d[26];
            float right_d = best_d[27];
            int left_i = best_i[26];
            int right_i = best_i[27];
            int swap = ((left_d < right_d) ? 1 : 0);
            best_d[26] = ((swap != 0) ? right_d : left_d);
            best_i[26] = ((swap != 0) ? right_i : left_i);
            best_d[27] = ((swap != 0) ? left_d : right_d);
            best_i[27] = ((swap != 0) ? left_i : right_i);
        }
    }
    {
        {
            float left_d = best_d[28];
            float right_d = best_d[29];
            int left_i = best_i[28];
            int right_i = best_i[29];
            int swap = ((left_d < right_d) ? 1 : 0);
            best_d[28] = ((swap != 0) ? right_d : left_d);
            best_i[28] = ((swap != 0) ? right_i : left_i);
            best_d[29] = ((swap != 0) ? left_d : right_d);
            best_i[29] = ((swap != 0) ? left_i : right_i);
        }
    }
    {
        {
            float left_d = best_d[30];
            float right_d = best_d[31];
            int left_i = best_i[30];
            int right_i = best_i[31];
            int swap = ((left_d < right_d) ? 1 : 0);
            best_d[30] = ((swap != 0) ? right_d : left_d);
            best_i[30] = ((swap != 0) ? right_i : left_i);
            best_d[31] = ((swap != 0) ? left_d : right_d);
            best_i[31] = ((swap != 0) ? left_i : right_i);
        }
    }
    {
        {
            float left_d = best_d[32];
            float right_d = best_d[33];
            int left_i = best_i[32];
            int right_i = best_i[33];
            int swap = ((right_d < left_d) ? 1 : 0);
            best_d[32] = ((swap != 0) ? right_d : left_d);
            best_i[32] = ((swap != 0) ? right_i : left_i);
            best_d[33] = ((swap != 0) ? left_d : right_d);
            best_i[33] = ((swap != 0) ? left_i : right_i);
        }
    }
    {
        {
            float left_d = best_d[34];
            float right_d = best_d[35];
            int left_i = best_i[34];
            int right_i = best_i[35];
            int swap = ((right_d < left_d) ? 1 : 0);
            best_d[34] = ((swap != 0) ? right_d : left_d);
            best_i[34] = ((swap != 0) ? right_i : left_i);
            best_d[35] = ((swap != 0) ? left_d : right_d);
            best_i[35] = ((swap != 0) ? left_i : right_i);
        }
    }
    {
        {
            float left_d = best_d[36];
            float right_d = best_d[37];
            int left_i = best_i[36];
            int right_i = best_i[37];
            int swap = ((right_d < left_d) ? 1 : 0);
            best_d[36] = ((swap != 0) ? right_d : left_d);
            best_i[36] = ((swap != 0) ? right_i : left_i);
            best_d[37] = ((swap != 0) ? left_d : right_d);
            best_i[37] = ((swap != 0) ? left_i : right_i);
        }
    }
    {
        {
            float left_d = best_d[38];
            float right_d = best_d[39];
            int left_i = best_i[38];
            int right_i = best_i[39];
            int swap = ((right_d < left_d) ? 1 : 0);
            best_d[38] = ((swap != 0) ? right_d : left_d);
            best_i[38] = ((swap != 0) ? right_i : left_i);
            best_d[39] = ((swap != 0) ? left_d : right_d);
            best_i[39] = ((swap != 0) ? left_i : right_i);
        }
    }
    {
        {
            float left_d = best_d[40];
            float right_d = best_d[41];
            int left_i = best_i[40];
            int right_i = best_i[41];
            int swap = ((right_d < left_d) ? 1 : 0);
            best_d[40] = ((swap != 0) ? right_d : left_d);
            best_i[40] = ((swap != 0) ? right_i : left_i);
            best_d[41] = ((swap != 0) ? left_d : right_d);
            best_i[41] = ((swap != 0) ? left_i : right_i);
        }
    }
    {
        {
            float left_d = best_d[42];
            float right_d = best_d[43];
            int left_i = best_i[42];
            int right_i = best_i[43];
            int swap = ((right_d < left_d) ? 1 : 0);
            best_d[42] = ((swap != 0) ? right_d : left_d);
            best_i[42] = ((swap != 0) ? right_i : left_i);
            best_d[43] = ((swap != 0) ? left_d : right_d);
            best_i[43] = ((swap != 0) ? left_i : right_i);
        }
    }
    {
        {
            float left_d = best_d[44];
            float right_d = best_d[45];
            int left_i = best_i[44];
            int right_i = best_i[45];
            int swap = ((right_d < left_d) ? 1 : 0);
            best_d[44] = ((swap != 0) ? right_d : left_d);
            best_i[44] = ((swap != 0) ? right_i : left_i);
            best_d[45] = ((swap != 0) ? left_d : right_d);
            best_i[45] = ((swap != 0) ? left_i : right_i);
        }
    }
    {
        {
            float left_d = best_d[46];
            float right_d = best_d[47];
            int left_i = best_i[46];
            int right_i = best_i[47];
            int swap = ((right_d < left_d) ? 1 : 0);
            best_d[46] = ((swap != 0) ? right_d : left_d);
            best_i[46] = ((swap != 0) ? right_i : left_i);
            best_d[47] = ((swap != 0) ? left_d : right_d);
            best_i[47] = ((swap != 0) ? left_i : right_i);
        }
    }
    {
        {
            float left_d = best_d[48];
            float right_d = best_d[49];
            int left_i = best_i[48];
            int right_i = best_i[49];
            int swap = ((left_d < right_d) ? 1 : 0);
            best_d[48] = ((swap != 0) ? right_d : left_d);
            best_i[48] = ((swap != 0) ? right_i : left_i);
            best_d[49] = ((swap != 0) ? left_d : right_d);
            best_i[49] = ((swap != 0) ? left_i : right_i);
        }
    }
    {
        {
            float left_d = best_d[50];
            float right_d = best_d[51];
            int left_i = best_i[50];
            int right_i = best_i[51];
            int swap = ((left_d < right_d) ? 1 : 0);
            best_d[50] = ((swap != 0) ? right_d : left_d);
            best_i[50] = ((swap != 0) ? right_i : left_i);
            best_d[51] = ((swap != 0) ? left_d : right_d);
            best_i[51] = ((swap != 0) ? left_i : right_i);
        }
    }
    {
        {
            float left_d = best_d[52];
            float right_d = best_d[53];
            int left_i = best_i[52];
            int right_i = best_i[53];
            int swap = ((left_d < right_d) ? 1 : 0);
            best_d[52] = ((swap != 0) ? right_d : left_d);
            best_i[52] = ((swap != 0) ? right_i : left_i);
            best_d[53] = ((swap != 0) ? left_d : right_d);
            best_i[53] = ((swap != 0) ? left_i : right_i);
        }
    }
    {
        {
            float left_d = best_d[54];
            float right_d = best_d[55];
            int left_i = best_i[54];
            int right_i = best_i[55];
            int swap = ((left_d < right_d) ? 1 : 0);
            best_d[54] = ((swap != 0) ? right_d : left_d);
            best_i[54] = ((swap != 0) ? right_i : left_i);
            best_d[55] = ((swap != 0) ? left_d : right_d);
            best_i[55] = ((swap != 0) ? left_i : right_i);
        }
    }
    {
        {
            float left_d = best_d[56];
            float right_d = best_d[57];
            int left_i = best_i[56];
            int right_i = best_i[57];
            int swap = ((left_d < right_d) ? 1 : 0);
            best_d[56] = ((swap != 0) ? right_d : left_d);
            best_i[56] = ((swap != 0) ? right_i : left_i);
            best_d[57] = ((swap != 0) ? left_d : right_d);
            best_i[57] = ((swap != 0) ? left_i : right_i);
        }
    }
    {
        {
            float left_d = best_d[58];
            float right_d = best_d[59];
            int left_i = best_i[58];
            int right_i = best_i[59];
            int swap = ((left_d < right_d) ? 1 : 0);
            best_d[58] = ((swap != 0) ? right_d : left_d);
            best_i[58] = ((swap != 0) ? right_i : left_i);
            best_d[59] = ((swap != 0) ? left_d : right_d);
            best_i[59] = ((swap != 0) ? left_i : right_i);
        }
    }
    {
        {
            float left_d = best_d[60];
            float right_d = best_d[61];
            int left_i = best_i[60];
            int right_i = best_i[61];
            int swap = ((left_d < right_d) ? 1 : 0);
            best_d[60] = ((swap != 0) ? right_d : left_d);
            best_i[60] = ((swap != 0) ? right_i : left_i);
            best_d[61] = ((swap != 0) ? left_d : right_d);
            best_i[61] = ((swap != 0) ? left_i : right_i);
        }
    }
    {
        {
            float left_d = best_d[62];
            float right_d = best_d[63];
            int left_i = best_i[62];
            int right_i = best_i[63];
            int swap = ((left_d < right_d) ? 1 : 0);
            best_d[62] = ((swap != 0) ? right_d : left_d);
            best_i[62] = ((swap != 0) ? right_i : left_i);
            best_d[63] = ((swap != 0) ? left_d : right_d);
            best_i[63] = ((swap != 0) ? left_i : right_i);
        }
    }
    {
        {
            float left_d = best_d[0];
            float right_d = best_d[16];
            int left_i = best_i[0];
            int right_i = best_i[16];
            int swap = ((right_d < left_d) ? 1 : 0);
            best_d[0] = ((swap != 0) ? right_d : left_d);
            best_i[0] = ((swap != 0) ? right_i : left_i);
            best_d[16] = ((swap != 0) ? left_d : right_d);
            best_i[16] = ((swap != 0) ? left_i : right_i);
        }
    }
    {
        {
            float left_d = best_d[1];
            float right_d = best_d[17];
            int left_i = best_i[1];
            int right_i = best_i[17];
            int swap = ((right_d < left_d) ? 1 : 0);
            best_d[1] = ((swap != 0) ? right_d : left_d);
            best_i[1] = ((swap != 0) ? right_i : left_i);
            best_d[17] = ((swap != 0) ? left_d : right_d);
            best_i[17] = ((swap != 0) ? left_i : right_i);
        }
    }
    {
        {
            float left_d = best_d[2];
            float right_d = best_d[18];
            int left_i = best_i[2];
            int right_i = best_i[18];
            int swap = ((right_d < left_d) ? 1 : 0);
            best_d[2] = ((swap != 0) ? right_d : left_d);
            best_i[2] = ((swap != 0) ? right_i : left_i);
            best_d[18] = ((swap != 0) ? left_d : right_d);
            best_i[18] = ((swap != 0) ? left_i : right_i);
        }
    }
    {
        {
            float left_d = best_d[3];
            float right_d = best_d[19];
            int left_i = best_i[3];
            int right_i = best_i[19];
            int swap = ((right_d < left_d) ? 1 : 0);
            best_d[3] = ((swap != 0) ? right_d : left_d);
            best_i[3] = ((swap != 0) ? right_i : left_i);
            best_d[19] = ((swap != 0) ? left_d : right_d);
            best_i[19] = ((swap != 0) ? left_i : right_i);
        }
    }
    {
        {
            float left_d = best_d[4];
            float right_d = best_d[20];
            int left_i = best_i[4];
            int right_i = best_i[20];
            int swap = ((right_d < left_d) ? 1 : 0);
            best_d[4] = ((swap != 0) ? right_d : left_d);
            best_i[4] = ((swap != 0) ? right_i : left_i);
            best_d[20] = ((swap != 0) ? left_d : right_d);
            best_i[20] = ((swap != 0) ? left_i : right_i);
        }
    }
    {
        {
            float left_d = best_d[5];
            float right_d = best_d[21];
            int left_i = best_i[5];
            int right_i = best_i[21];
            int swap = ((right_d < left_d) ? 1 : 0);
            best_d[5] = ((swap != 0) ? right_d : left_d);
            best_i[5] = ((swap != 0) ? right_i : left_i);
            best_d[21] = ((swap != 0) ? left_d : right_d);
            best_i[21] = ((swap != 0) ? left_i : right_i);
        }
    }
    {
        {
            float left_d = best_d[6];
            float right_d = best_d[22];
            int left_i = best_i[6];
            int right_i = best_i[22];
            int swap = ((right_d < left_d) ? 1 : 0);
            best_d[6] = ((swap != 0) ? right_d : left_d);
            best_i[6] = ((swap != 0) ? right_i : left_i);
            best_d[22] = ((swap != 0) ? left_d : right_d);
            best_i[22] = ((swap != 0) ? left_i : right_i);
        }
    }
    {
        {
            float left_d = best_d[7];
            float right_d = best_d[23];
            int left_i = best_i[7];
            int right_i = best_i[23];
            int swap = ((right_d < left_d) ? 1 : 0);
            best_d[7] = ((swap != 0) ? right_d : left_d);
            best_i[7] = ((swap != 0) ? right_i : left_i);
            best_d[23] = ((swap != 0) ? left_d : right_d);
            best_i[23] = ((swap != 0) ? left_i : right_i);
        }
    }
    {
        {
            float left_d = best_d[8];
            float right_d = best_d[24];
            int left_i = best_i[8];
            int right_i = best_i[24];
            int swap = ((right_d < left_d) ? 1 : 0);
            best_d[8] = ((swap != 0) ? right_d : left_d);
            best_i[8] = ((swap != 0) ? right_i : left_i);
            best_d[24] = ((swap != 0) ? left_d : right_d);
            best_i[24] = ((swap != 0) ? left_i : right_i);
        }
    }
    {
        {
            float left_d = best_d[9];
            float right_d = best_d[25];
            int left_i = best_i[9];
            int right_i = best_i[25];
            int swap = ((right_d < left_d) ? 1 : 0);
            best_d[9] = ((swap != 0) ? right_d : left_d);
            best_i[9] = ((swap != 0) ? right_i : left_i);
            best_d[25] = ((swap != 0) ? left_d : right_d);
            best_i[25] = ((swap != 0) ? left_i : right_i);
        }
    }
    {
        {
            float left_d = best_d[10];
            float right_d = best_d[26];
            int left_i = best_i[10];
            int right_i = best_i[26];
            int swap = ((right_d < left_d) ? 1 : 0);
            best_d[10] = ((swap != 0) ? right_d : left_d);
            best_i[10] = ((swap != 0) ? right_i : left_i);
            best_d[26] = ((swap != 0) ? left_d : right_d);
            best_i[26] = ((swap != 0) ? left_i : right_i);
        }
    }
    {
        {
            float left_d = best_d[11];
            float right_d = best_d[27];
            int left_i = best_i[11];
            int right_i = best_i[27];
            int swap = ((right_d < left_d) ? 1 : 0);
            best_d[11] = ((swap != 0) ? right_d : left_d);
            best_i[11] = ((swap != 0) ? right_i : left_i);
            best_d[27] = ((swap != 0) ? left_d : right_d);
            best_i[27] = ((swap != 0) ? left_i : right_i);
        }
    }
    {
        {
            float left_d = best_d[12];
            float right_d = best_d[28];
            int left_i = best_i[12];
            int right_i = best_i[28];
            int swap = ((right_d < left_d) ? 1 : 0);
            best_d[12] = ((swap != 0) ? right_d : left_d);
            best_i[12] = ((swap != 0) ? right_i : left_i);
            best_d[28] = ((swap != 0) ? left_d : right_d);
            best_i[28] = ((swap != 0) ? left_i : right_i);
        }
    }
    {
        {
            float left_d = best_d[13];
            float right_d = best_d[29];
            int left_i = best_i[13];
            int right_i = best_i[29];
            int swap = ((right_d < left_d) ? 1 : 0);
            best_d[13] = ((swap != 0) ? right_d : left_d);
            best_i[13] = ((swap != 0) ? right_i : left_i);
            best_d[29] = ((swap != 0) ? left_d : right_d);
            best_i[29] = ((swap != 0) ? left_i : right_i);
        }
    }
    {
        {
            float left_d = best_d[14];
            float right_d = best_d[30];
            int left_i = best_i[14];
            int right_i = best_i[30];
            int swap = ((right_d < left_d) ? 1 : 0);
            best_d[14] = ((swap != 0) ? right_d : left_d);
            best_i[14] = ((swap != 0) ? right_i : left_i);
            best_d[30] = ((swap != 0) ? left_d : right_d);
            best_i[30] = ((swap != 0) ? left_i : right_i);
        }
    }
    {
        {
            float left_d = best_d[15];
            float right_d = best_d[31];
            int left_i = best_i[15];
            int right_i = best_i[31];
            int swap = ((right_d < left_d) ? 1 : 0);
            best_d[15] = ((swap != 0) ? right_d : left_d);
            best_i[15] = ((swap != 0) ? right_i : left_i);
            best_d[31] = ((swap != 0) ? left_d : right_d);
            best_i[31] = ((swap != 0) ? left_i : right_i);
        }
    }
    {
        {
            float left_d = best_d[32];
            float right_d = best_d[48];
            int left_i = best_i[32];
            int right_i = best_i[48];
            int swap = ((left_d < right_d) ? 1 : 0);
            best_d[32] = ((swap != 0) ? right_d : left_d);
            best_i[32] = ((swap != 0) ? right_i : left_i);
            best_d[48] = ((swap != 0) ? left_d : right_d);
            best_i[48] = ((swap != 0) ? left_i : right_i);
        }
    }
    {
        {
            float left_d = best_d[33];
            float right_d = best_d[49];
            int left_i = best_i[33];
            int right_i = best_i[49];
            int swap = ((left_d < right_d) ? 1 : 0);
            best_d[33] = ((swap != 0) ? right_d : left_d);
            best_i[33] = ((swap != 0) ? right_i : left_i);
            best_d[49] = ((swap != 0) ? left_d : right_d);
            best_i[49] = ((swap != 0) ? left_i : right_i);
        }
    }
    {
        {
            float left_d = best_d[34];
            float right_d = best_d[50];
            int left_i = best_i[34];
            int right_i = best_i[50];
            int swap = ((left_d < right_d) ? 1 : 0);
            best_d[34] = ((swap != 0) ? right_d : left_d);
            best_i[34] = ((swap != 0) ? right_i : left_i);
            best_d[50] = ((swap != 0) ? left_d : right_d);
            best_i[50] = ((swap != 0) ? left_i : right_i);
        }
    }
    {
        {
            float left_d = best_d[35];
            float right_d = best_d[51];
            int left_i = best_i[35];
            int right_i = best_i[51];
            int swap = ((left_d < right_d) ? 1 : 0);
            best_d[35] = ((swap != 0) ? right_d : left_d);
            best_i[35] = ((swap != 0) ? right_i : left_i);
            best_d[51] = ((swap != 0) ? left_d : right_d);
            best_i[51] = ((swap != 0) ? left_i : right_i);
        }
    }
    {
        {
            float left_d = best_d[36];
            float right_d = best_d[52];
            int left_i = best_i[36];
            int right_i = best_i[52];
            int swap = ((left_d < right_d) ? 1 : 0);
            best_d[36] = ((swap != 0) ? right_d : left_d);
            best_i[36] = ((swap != 0) ? right_i : left_i);
            best_d[52] = ((swap != 0) ? left_d : right_d);
            best_i[52] = ((swap != 0) ? left_i : right_i);
        }
    }
    {
        {
            float left_d = best_d[37];
            float right_d = best_d[53];
            int left_i = best_i[37];
            int right_i = best_i[53];
            int swap = ((left_d < right_d) ? 1 : 0);
            best_d[37] = ((swap != 0) ? right_d : left_d);
            best_i[37] = ((swap != 0) ? right_i : left_i);
            best_d[53] = ((swap != 0) ? left_d : right_d);
            best_i[53] = ((swap != 0) ? left_i : right_i);
        }
    }
    {
        {
            float left_d = best_d[38];
            float right_d = best_d[54];
            int left_i = best_i[38];
            int right_i = best_i[54];
            int swap = ((left_d < right_d) ? 1 : 0);
            best_d[38] = ((swap != 0) ? right_d : left_d);
            best_i[38] = ((swap != 0) ? right_i : left_i);
            best_d[54] = ((swap != 0) ? left_d : right_d);
            best_i[54] = ((swap != 0) ? left_i : right_i);
        }
    }
    {
        {
            float left_d = best_d[39];
            float right_d = best_d[55];
            int left_i = best_i[39];
            int right_i = best_i[55];
            int swap = ((left_d < right_d) ? 1 : 0);
            best_d[39] = ((swap != 0) ? right_d : left_d);
            best_i[39] = ((swap != 0) ? right_i : left_i);
            best_d[55] = ((swap != 0) ? left_d : right_d);
            best_i[55] = ((swap != 0) ? left_i : right_i);
        }
    }
    {
        {
            float left_d = best_d[40];
            float right_d = best_d[56];
            int left_i = best_i[40];
            int right_i = best_i[56];
            int swap = ((left_d < right_d) ? 1 : 0);
            best_d[40] = ((swap != 0) ? right_d : left_d);
            best_i[40] = ((swap != 0) ? right_i : left_i);
            best_d[56] = ((swap != 0) ? left_d : right_d);
            best_i[56] = ((swap != 0) ? left_i : right_i);
        }
    }
    {
        {
            float left_d = best_d[41];
            float right_d = best_d[57];
            int left_i = best_i[41];
            int right_i = best_i[57];
            int swap = ((left_d < right_d) ? 1 : 0);
            best_d[41] = ((swap != 0) ? right_d : left_d);
            best_i[41] = ((swap != 0) ? right_i : left_i);
            best_d[57] = ((swap != 0) ? left_d : right_d);
            best_i[57] = ((swap != 0) ? left_i : right_i);
        }
    }
    {
        {
            float left_d = best_d[42];
            float right_d = best_d[58];
            int left_i = best_i[42];
            int right_i = best_i[58];
            int swap = ((left_d < right_d) ? 1 : 0);
            best_d[42] = ((swap != 0) ? right_d : left_d);
            best_i[42] = ((swap != 0) ? right_i : left_i);
            best_d[58] = ((swap != 0) ? left_d : right_d);
            best_i[58] = ((swap != 0) ? left_i : right_i);
        }
    }
    {
        {
            float left_d = best_d[43];
            float right_d = best_d[59];
            int left_i = best_i[43];
            int right_i = best_i[59];
            int swap = ((left_d < right_d) ? 1 : 0);
            best_d[43] = ((swap != 0) ? right_d : left_d);
            best_i[43] = ((swap != 0) ? right_i : left_i);
            best_d[59] = ((swap != 0) ? left_d : right_d);
            best_i[59] = ((swap != 0) ? left_i : right_i);
        }
    }
    {
        {
            float left_d = best_d[44];
            float right_d = best_d[60];
            int left_i = best_i[44];
            int right_i = best_i[60];
            int swap = ((left_d < right_d) ? 1 : 0);
            best_d[44] = ((swap != 0) ? right_d : left_d);
            best_i[44] = ((swap != 0) ? right_i : left_i);
            best_d[60] = ((swap != 0) ? left_d : right_d);
            best_i[60] = ((swap != 0) ? left_i : right_i);
        }
    }
    {
        {
            float left_d = best_d[45];
            float right_d = best_d[61];
            int left_i = best_i[45];
            int right_i = best_i[61];
            int swap = ((left_d < right_d) ? 1 : 0);
            best_d[45] = ((swap != 0) ? right_d : left_d);
            best_i[45] = ((swap != 0) ? right_i : left_i);
            best_d[61] = ((swap != 0) ? left_d : right_d);
            best_i[61] = ((swap != 0) ? left_i : right_i);
        }
    }
    {
        {
            float left_d = best_d[46];
            float right_d = best_d[62];
            int left_i = best_i[46];
            int right_i = best_i[62];
            int swap = ((left_d < right_d) ? 1 : 0);
            best_d[46] = ((swap != 0) ? right_d : left_d);
            best_i[46] = ((swap != 0) ? right_i : left_i);
            best_d[62] = ((swap != 0) ? left_d : right_d);
            best_i[62] = ((swap != 0) ? left_i : right_i);
        }
    }
    {
        {
            float left_d = best_d[47];
            float right_d = best_d[63];
            int left_i = best_i[47];
            int right_i = best_i[63];
            int swap = ((left_d < right_d) ? 1 : 0);
            best_d[47] = ((swap != 0) ? right_d : left_d);
            best_i[47] = ((swap != 0) ? right_i : left_i);
            best_d[63] = ((swap != 0) ? left_d : right_d);
            best_i[63] = ((swap != 0) ? left_i : right_i);
        }
    }
    {
        {
            float left_d = best_d[0];
            float right_d = best_d[8];
            int left_i = best_i[0];
            int right_i = best_i[8];
            int swap = ((right_d < left_d) ? 1 : 0);
            best_d[0] = ((swap != 0) ? right_d : left_d);
            best_i[0] = ((swap != 0) ? right_i : left_i);
            best_d[8] = ((swap != 0) ? left_d : right_d);
            best_i[8] = ((swap != 0) ? left_i : right_i);
        }
    }
    {
        {
            float left_d = best_d[1];
            float right_d = best_d[9];
            int left_i = best_i[1];
            int right_i = best_i[9];
            int swap = ((right_d < left_d) ? 1 : 0);
            best_d[1] = ((swap != 0) ? right_d : left_d);
            best_i[1] = ((swap != 0) ? right_i : left_i);
            best_d[9] = ((swap != 0) ? left_d : right_d);
            best_i[9] = ((swap != 0) ? left_i : right_i);
        }
    }
    {
        {
            float left_d = best_d[2];
            float right_d = best_d[10];
            int left_i = best_i[2];
            int right_i = best_i[10];
            int swap = ((right_d < left_d) ? 1 : 0);
            best_d[2] = ((swap != 0) ? right_d : left_d);
            best_i[2] = ((swap != 0) ? right_i : left_i);
            best_d[10] = ((swap != 0) ? left_d : right_d);
            best_i[10] = ((swap != 0) ? left_i : right_i);
        }
    }
    {
        {
            float left_d = best_d[3];
            float right_d = best_d[11];
            int left_i = best_i[3];
            int right_i = best_i[11];
            int swap = ((right_d < left_d) ? 1 : 0);
            best_d[3] = ((swap != 0) ? right_d : left_d);
            best_i[3] = ((swap != 0) ? right_i : left_i);
            best_d[11] = ((swap != 0) ? left_d : right_d);
            best_i[11] = ((swap != 0) ? left_i : right_i);
        }
    }
    {
        {
            float left_d = best_d[4];
            float right_d = best_d[12];
            int left_i = best_i[4];
            int right_i = best_i[12];
            int swap = ((right_d < left_d) ? 1 : 0);
            best_d[4] = ((swap != 0) ? right_d : left_d);
            best_i[4] = ((swap != 0) ? right_i : left_i);
            best_d[12] = ((swap != 0) ? left_d : right_d);
            best_i[12] = ((swap != 0) ? left_i : right_i);
        }
    }
    {
        {
            float left_d = best_d[5];
            float right_d = best_d[13];
            int left_i = best_i[5];
            int right_i = best_i[13];
            int swap = ((right_d < left_d) ? 1 : 0);
            best_d[5] = ((swap != 0) ? right_d : left_d);
            best_i[5] = ((swap != 0) ? right_i : left_i);
            best_d[13] = ((swap != 0) ? left_d : right_d);
            best_i[13] = ((swap != 0) ? left_i : right_i);
        }
    }
    {
        {
            float left_d = best_d[6];
            float right_d = best_d[14];
            int left_i = best_i[6];
            int right_i = best_i[14];
            int swap = ((right_d < left_d) ? 1 : 0);
            best_d[6] = ((swap != 0) ? right_d : left_d);
            best_i[6] = ((swap != 0) ? right_i : left_i);
            best_d[14] = ((swap != 0) ? left_d : right_d);
            best_i[14] = ((swap != 0) ? left_i : right_i);
        }
    }
    {
        {
            float left_d = best_d[7];
            float right_d = best_d[15];
            int left_i = best_i[7];
            int right_i = best_i[15];
            int swap = ((right_d < left_d) ? 1 : 0);
            best_d[7] = ((swap != 0) ? right_d : left_d);
            best_i[7] = ((swap != 0) ? right_i : left_i);
            best_d[15] = ((swap != 0) ? left_d : right_d);
            best_i[15] = ((swap != 0) ? left_i : right_i);
        }
    }
    {
        {
            float left_d = best_d[16];
            float right_d = best_d[24];
            int left_i = best_i[16];
            int right_i = best_i[24];
            int swap = ((right_d < left_d) ? 1 : 0);
            best_d[16] = ((swap != 0) ? right_d : left_d);
            best_i[16] = ((swap != 0) ? right_i : left_i);
            best_d[24] = ((swap != 0) ? left_d : right_d);
            best_i[24] = ((swap != 0) ? left_i : right_i);
        }
    }
    {
        {
            float left_d = best_d[17];
            float right_d = best_d[25];
            int left_i = best_i[17];
            int right_i = best_i[25];
            int swap = ((right_d < left_d) ? 1 : 0);
            best_d[17] = ((swap != 0) ? right_d : left_d);
            best_i[17] = ((swap != 0) ? right_i : left_i);
            best_d[25] = ((swap != 0) ? left_d : right_d);
            best_i[25] = ((swap != 0) ? left_i : right_i);
        }
    }
    {
        {
            float left_d = best_d[18];
            float right_d = best_d[26];
            int left_i = best_i[18];
            int right_i = best_i[26];
            int swap = ((right_d < left_d) ? 1 : 0);
            best_d[18] = ((swap != 0) ? right_d : left_d);
            best_i[18] = ((swap != 0) ? right_i : left_i);
            best_d[26] = ((swap != 0) ? left_d : right_d);
            best_i[26] = ((swap != 0) ? left_i : right_i);
        }
    }
    {
        {
            float left_d = best_d[19];
            float right_d = best_d[27];
            int left_i = best_i[19];
            int right_i = best_i[27];
            int swap = ((right_d < left_d) ? 1 : 0);
            best_d[19] = ((swap != 0) ? right_d : left_d);
            best_i[19] = ((swap != 0) ? right_i : left_i);
            best_d[27] = ((swap != 0) ? left_d : right_d);
            best_i[27] = ((swap != 0) ? left_i : right_i);
        }
    }
    {
        {
            float left_d = best_d[20];
            float right_d = best_d[28];
            int left_i = best_i[20];
            int right_i = best_i[28];
            int swap = ((right_d < left_d) ? 1 : 0);
            best_d[20] = ((swap != 0) ? right_d : left_d);
            best_i[20] = ((swap != 0) ? right_i : left_i);
            best_d[28] = ((swap != 0) ? left_d : right_d);
            best_i[28] = ((swap != 0) ? left_i : right_i);
        }
    }
    {
        {
            float left_d = best_d[21];
            float right_d = best_d[29];
            int left_i = best_i[21];
            int right_i = best_i[29];
            int swap = ((right_d < left_d) ? 1 : 0);
            best_d[21] = ((swap != 0) ? right_d : left_d);
            best_i[21] = ((swap != 0) ? right_i : left_i);
            best_d[29] = ((swap != 0) ? left_d : right_d);
            best_i[29] = ((swap != 0) ? left_i : right_i);
        }
    }
    {
        {
            float left_d = best_d[22];
            float right_d = best_d[30];
            int left_i = best_i[22];
            int right_i = best_i[30];
            int swap = ((right_d < left_d) ? 1 : 0);
            best_d[22] = ((swap != 0) ? right_d : left_d);
            best_i[22] = ((swap != 0) ? right_i : left_i);
            best_d[30] = ((swap != 0) ? left_d : right_d);
            best_i[30] = ((swap != 0) ? left_i : right_i);
        }
    }
    {
        {
            float left_d = best_d[23];
            float right_d = best_d[31];
            int left_i = best_i[23];
            int right_i = best_i[31];
            int swap = ((right_d < left_d) ? 1 : 0);
            best_d[23] = ((swap != 0) ? right_d : left_d);
            best_i[23] = ((swap != 0) ? right_i : left_i);
            best_d[31] = ((swap != 0) ? left_d : right_d);
            best_i[31] = ((swap != 0) ? left_i : right_i);
        }
    }
    {
        {
            float left_d = best_d[32];
            float right_d = best_d[40];
            int left_i = best_i[32];
            int right_i = best_i[40];
            int swap = ((left_d < right_d) ? 1 : 0);
            best_d[32] = ((swap != 0) ? right_d : left_d);
            best_i[32] = ((swap != 0) ? right_i : left_i);
            best_d[40] = ((swap != 0) ? left_d : right_d);
            best_i[40] = ((swap != 0) ? left_i : right_i);
        }
    }
    {
        {
            float left_d = best_d[33];
            float right_d = best_d[41];
            int left_i = best_i[33];
            int right_i = best_i[41];
            int swap = ((left_d < right_d) ? 1 : 0);
            best_d[33] = ((swap != 0) ? right_d : left_d);
            best_i[33] = ((swap != 0) ? right_i : left_i);
            best_d[41] = ((swap != 0) ? left_d : right_d);
            best_i[41] = ((swap != 0) ? left_i : right_i);
        }
    }
    {
        {
            float left_d = best_d[34];
            float right_d = best_d[42];
            int left_i = best_i[34];
            int right_i = best_i[42];
            int swap = ((left_d < right_d) ? 1 : 0);
            best_d[34] = ((swap != 0) ? right_d : left_d);
            best_i[34] = ((swap != 0) ? right_i : left_i);
            best_d[42] = ((swap != 0) ? left_d : right_d);
            best_i[42] = ((swap != 0) ? left_i : right_i);
        }
    }
    {
        {
            float left_d = best_d[35];
            float right_d = best_d[43];
            int left_i = best_i[35];
            int right_i = best_i[43];
            int swap = ((left_d < right_d) ? 1 : 0);
            best_d[35] = ((swap != 0) ? right_d : left_d);
            best_i[35] = ((swap != 0) ? right_i : left_i);
            best_d[43] = ((swap != 0) ? left_d : right_d);
            best_i[43] = ((swap != 0) ? left_i : right_i);
        }
    }
    {
        {
            float left_d = best_d[36];
            float right_d = best_d[44];
            int left_i = best_i[36];
            int right_i = best_i[44];
            int swap = ((left_d < right_d) ? 1 : 0);
            best_d[36] = ((swap != 0) ? right_d : left_d);
            best_i[36] = ((swap != 0) ? right_i : left_i);
            best_d[44] = ((swap != 0) ? left_d : right_d);
            best_i[44] = ((swap != 0) ? left_i : right_i);
        }
    }
    {
        {
            float left_d = best_d[37];
            float right_d = best_d[45];
            int left_i = best_i[37];
            int right_i = best_i[45];
            int swap = ((left_d < right_d) ? 1 : 0);
            best_d[37] = ((swap != 0) ? right_d : left_d);
            best_i[37] = ((swap != 0) ? right_i : left_i);
            best_d[45] = ((swap != 0) ? left_d : right_d);
            best_i[45] = ((swap != 0) ? left_i : right_i);
        }
    }
    {
        {
            float left_d = best_d[38];
            float right_d = best_d[46];
            int left_i = best_i[38];
            int right_i = best_i[46];
            int swap = ((left_d < right_d) ? 1 : 0);
            best_d[38] = ((swap != 0) ? right_d : left_d);
            best_i[38] = ((swap != 0) ? right_i : left_i);
            best_d[46] = ((swap != 0) ? left_d : right_d);
            best_i[46] = ((swap != 0) ? left_i : right_i);
        }
    }
    {
        {
            float left_d = best_d[39];
            float right_d = best_d[47];
            int left_i = best_i[39];
            int right_i = best_i[47];
            int swap = ((left_d < right_d) ? 1 : 0);
            best_d[39] = ((swap != 0) ? right_d : left_d);
            best_i[39] = ((swap != 0) ? right_i : left_i);
            best_d[47] = ((swap != 0) ? left_d : right_d);
            best_i[47] = ((swap != 0) ? left_i : right_i);
        }
    }
    {
        {
            float left_d = best_d[48];
            float right_d = best_d[56];
            int left_i = best_i[48];
            int right_i = best_i[56];
            int swap = ((left_d < right_d) ? 1 : 0);
            best_d[48] = ((swap != 0) ? right_d : left_d);
            best_i[48] = ((swap != 0) ? right_i : left_i);
            best_d[56] = ((swap != 0) ? left_d : right_d);
            best_i[56] = ((swap != 0) ? left_i : right_i);
        }
    }
    {
        {
            float left_d = best_d[49];
            float right_d = best_d[57];
            int left_i = best_i[49];
            int right_i = best_i[57];
            int swap = ((left_d < right_d) ? 1 : 0);
            best_d[49] = ((swap != 0) ? right_d : left_d);
            best_i[49] = ((swap != 0) ? right_i : left_i);
            best_d[57] = ((swap != 0) ? left_d : right_d);
            best_i[57] = ((swap != 0) ? left_i : right_i);
        }
    }
    {
        {
            float left_d = best_d[50];
            float right_d = best_d[58];
            int left_i = best_i[50];
            int right_i = best_i[58];
            int swap = ((left_d < right_d) ? 1 : 0);
            best_d[50] = ((swap != 0) ? right_d : left_d);
            best_i[50] = ((swap != 0) ? right_i : left_i);
            best_d[58] = ((swap != 0) ? left_d : right_d);
            best_i[58] = ((swap != 0) ? left_i : right_i);
        }
    }
    {
        {
            float left_d = best_d[51];
            float right_d = best_d[59];
            int left_i = best_i[51];
            int right_i = best_i[59];
            int swap = ((left_d < right_d) ? 1 : 0);
            best_d[51] = ((swap != 0) ? right_d : left_d);
            best_i[51] = ((swap != 0) ? right_i : left_i);
            best_d[59] = ((swap != 0) ? left_d : right_d);
            best_i[59] = ((swap != 0) ? left_i : right_i);
        }
    }
    {
        {
            float left_d = best_d[52];
            float right_d = best_d[60];
            int left_i = best_i[52];
            int right_i = best_i[60];
            int swap = ((left_d < right_d) ? 1 : 0);
            best_d[52] = ((swap != 0) ? right_d : left_d);
            best_i[52] = ((swap != 0) ? right_i : left_i);
            best_d[60] = ((swap != 0) ? left_d : right_d);
            best_i[60] = ((swap != 0) ? left_i : right_i);
        }
    }
    {
        {
            float left_d = best_d[53];
            float right_d = best_d[61];
            int left_i = best_i[53];
            int right_i = best_i[61];
            int swap = ((left_d < right_d) ? 1 : 0);
            best_d[53] = ((swap != 0) ? right_d : left_d);
            best_i[53] = ((swap != 0) ? right_i : left_i);
            best_d[61] = ((swap != 0) ? left_d : right_d);
            best_i[61] = ((swap != 0) ? left_i : right_i);
        }
    }
    {
        {
            float left_d = best_d[54];
            float right_d = best_d[62];
            int left_i = best_i[54];
            int right_i = best_i[62];
            int swap = ((left_d < right_d) ? 1 : 0);
            best_d[54] = ((swap != 0) ? right_d : left_d);
            best_i[54] = ((swap != 0) ? right_i : left_i);
            best_d[62] = ((swap != 0) ? left_d : right_d);
            best_i[62] = ((swap != 0) ? left_i : right_i);
        }
    }
    {
        {
            float left_d = best_d[55];
            float right_d = best_d[63];
            int left_i = best_i[55];
            int right_i = best_i[63];
            int swap = ((left_d < right_d) ? 1 : 0);
            best_d[55] = ((swap != 0) ? right_d : left_d);
            best_i[55] = ((swap != 0) ? right_i : left_i);
            best_d[63] = ((swap != 0) ? left_d : right_d);
            best_i[63] = ((swap != 0) ? left_i : right_i);
        }
    }
    {
        {
            float left_d = best_d[0];
            float right_d = best_d[4];
            int left_i = best_i[0];
            int right_i = best_i[4];
            int swap = ((right_d < left_d) ? 1 : 0);
            best_d[0] = ((swap != 0) ? right_d : left_d);
            best_i[0] = ((swap != 0) ? right_i : left_i);
            best_d[4] = ((swap != 0) ? left_d : right_d);
            best_i[4] = ((swap != 0) ? left_i : right_i);
        }
    }
    {
        {
            float left_d = best_d[1];
            float right_d = best_d[5];
            int left_i = best_i[1];
            int right_i = best_i[5];
            int swap = ((right_d < left_d) ? 1 : 0);
            best_d[1] = ((swap != 0) ? right_d : left_d);
            best_i[1] = ((swap != 0) ? right_i : left_i);
            best_d[5] = ((swap != 0) ? left_d : right_d);
            best_i[5] = ((swap != 0) ? left_i : right_i);
        }
    }
    {
        {
            float left_d = best_d[2];
            float right_d = best_d[6];
            int left_i = best_i[2];
            int right_i = best_i[6];
            int swap = ((right_d < left_d) ? 1 : 0);
            best_d[2] = ((swap != 0) ? right_d : left_d);
            best_i[2] = ((swap != 0) ? right_i : left_i);
            best_d[6] = ((swap != 0) ? left_d : right_d);
            best_i[6] = ((swap != 0) ? left_i : right_i);
        }
    }
    {
        {
            float left_d = best_d[3];
            float right_d = best_d[7];
            int left_i = best_i[3];
            int right_i = best_i[7];
            int swap = ((right_d < left_d) ? 1 : 0);
            best_d[3] = ((swap != 0) ? right_d : left_d);
            best_i[3] = ((swap != 0) ? right_i : left_i);
            best_d[7] = ((swap != 0) ? left_d : right_d);
            best_i[7] = ((swap != 0) ? left_i : right_i);
        }
    }
    {
        {
            float left_d = best_d[8];
            float right_d = best_d[12];
            int left_i = best_i[8];
            int right_i = best_i[12];
            int swap = ((right_d < left_d) ? 1 : 0);
            best_d[8] = ((swap != 0) ? right_d : left_d);
            best_i[8] = ((swap != 0) ? right_i : left_i);
            best_d[12] = ((swap != 0) ? left_d : right_d);
            best_i[12] = ((swap != 0) ? left_i : right_i);
        }
    }
    {
        {
            float left_d = best_d[9];
            float right_d = best_d[13];
            int left_i = best_i[9];
            int right_i = best_i[13];
            int swap = ((right_d < left_d) ? 1 : 0);
            best_d[9] = ((swap != 0) ? right_d : left_d);
            best_i[9] = ((swap != 0) ? right_i : left_i);
            best_d[13] = ((swap != 0) ? left_d : right_d);
            best_i[13] = ((swap != 0) ? left_i : right_i);
        }
    }
    {
        {
            float left_d = best_d[10];
            float right_d = best_d[14];
            int left_i = best_i[10];
            int right_i = best_i[14];
            int swap = ((right_d < left_d) ? 1 : 0);
            best_d[10] = ((swap != 0) ? right_d : left_d);
            best_i[10] = ((swap != 0) ? right_i : left_i);
            best_d[14] = ((swap != 0) ? left_d : right_d);
            best_i[14] = ((swap != 0) ? left_i : right_i);
        }
    }
    {
        {
            float left_d = best_d[11];
            float right_d = best_d[15];
            int left_i = best_i[11];
            int right_i = best_i[15];
            int swap = ((right_d < left_d) ? 1 : 0);
            best_d[11] = ((swap != 0) ? right_d : left_d);
            best_i[11] = ((swap != 0) ? right_i : left_i);
            best_d[15] = ((swap != 0) ? left_d : right_d);
            best_i[15] = ((swap != 0) ? left_i : right_i);
        }
    }
    {
        {
            float left_d = best_d[16];
            float right_d = best_d[20];
            int left_i = best_i[16];
            int right_i = best_i[20];
            int swap = ((right_d < left_d) ? 1 : 0);
            best_d[16] = ((swap != 0) ? right_d : left_d);
            best_i[16] = ((swap != 0) ? right_i : left_i);
            best_d[20] = ((swap != 0) ? left_d : right_d);
            best_i[20] = ((swap != 0) ? left_i : right_i);
        }
    }
    {
        {
            float left_d = best_d[17];
            float right_d = best_d[21];
            int left_i = best_i[17];
            int right_i = best_i[21];
            int swap = ((right_d < left_d) ? 1 : 0);
            best_d[17] = ((swap != 0) ? right_d : left_d);
            best_i[17] = ((swap != 0) ? right_i : left_i);
            best_d[21] = ((swap != 0) ? left_d : right_d);
            best_i[21] = ((swap != 0) ? left_i : right_i);
        }
    }
    {
        {
            float left_d = best_d[18];
            float right_d = best_d[22];
            int left_i = best_i[18];
            int right_i = best_i[22];
            int swap = ((right_d < left_d) ? 1 : 0);
            best_d[18] = ((swap != 0) ? right_d : left_d);
            best_i[18] = ((swap != 0) ? right_i : left_i);
            best_d[22] = ((swap != 0) ? left_d : right_d);
            best_i[22] = ((swap != 0) ? left_i : right_i);
        }
    }
    {
        {
            float left_d = best_d[19];
            float right_d = best_d[23];
            int left_i = best_i[19];
            int right_i = best_i[23];
            int swap = ((right_d < left_d) ? 1 : 0);
            best_d[19] = ((swap != 0) ? right_d : left_d);
            best_i[19] = ((swap != 0) ? right_i : left_i);
            best_d[23] = ((swap != 0) ? left_d : right_d);
            best_i[23] = ((swap != 0) ? left_i : right_i);
        }
    }
    {
        {
            float left_d = best_d[24];
            float right_d = best_d[28];
            int left_i = best_i[24];
            int right_i = best_i[28];
            int swap = ((right_d < left_d) ? 1 : 0);
            best_d[24] = ((swap != 0) ? right_d : left_d);
            best_i[24] = ((swap != 0) ? right_i : left_i);
            best_d[28] = ((swap != 0) ? left_d : right_d);
            best_i[28] = ((swap != 0) ? left_i : right_i);
        }
    }
    {
        {
            float left_d = best_d[25];
            float right_d = best_d[29];
            int left_i = best_i[25];
            int right_i = best_i[29];
            int swap = ((right_d < left_d) ? 1 : 0);
            best_d[25] = ((swap != 0) ? right_d : left_d);
            best_i[25] = ((swap != 0) ? right_i : left_i);
            best_d[29] = ((swap != 0) ? left_d : right_d);
            best_i[29] = ((swap != 0) ? left_i : right_i);
        }
    }
    {
        {
            float left_d = best_d[26];
            float right_d = best_d[30];
            int left_i = best_i[26];
            int right_i = best_i[30];
            int swap = ((right_d < left_d) ? 1 : 0);
            best_d[26] = ((swap != 0) ? right_d : left_d);
            best_i[26] = ((swap != 0) ? right_i : left_i);
            best_d[30] = ((swap != 0) ? left_d : right_d);
            best_i[30] = ((swap != 0) ? left_i : right_i);
        }
    }
    {
        {
            float left_d = best_d[27];
            float right_d = best_d[31];
            int left_i = best_i[27];
            int right_i = best_i[31];
            int swap = ((right_d < left_d) ? 1 : 0);
            best_d[27] = ((swap != 0) ? right_d : left_d);
            best_i[27] = ((swap != 0) ? right_i : left_i);
            best_d[31] = ((swap != 0) ? left_d : right_d);
            best_i[31] = ((swap != 0) ? left_i : right_i);
        }
    }
    {
        {
            float left_d = best_d[32];
            float right_d = best_d[36];
            int left_i = best_i[32];
            int right_i = best_i[36];
            int swap = ((left_d < right_d) ? 1 : 0);
            best_d[32] = ((swap != 0) ? right_d : left_d);
            best_i[32] = ((swap != 0) ? right_i : left_i);
            best_d[36] = ((swap != 0) ? left_d : right_d);
            best_i[36] = ((swap != 0) ? left_i : right_i);
        }
    }
    {
        {
            float left_d = best_d[33];
            float right_d = best_d[37];
            int left_i = best_i[33];
            int right_i = best_i[37];
            int swap = ((left_d < right_d) ? 1 : 0);
            best_d[33] = ((swap != 0) ? right_d : left_d);
            best_i[33] = ((swap != 0) ? right_i : left_i);
            best_d[37] = ((swap != 0) ? left_d : right_d);
            best_i[37] = ((swap != 0) ? left_i : right_i);
        }
    }
    {
        {
            float left_d = best_d[34];
            float right_d = best_d[38];
            int left_i = best_i[34];
            int right_i = best_i[38];
            int swap = ((left_d < right_d) ? 1 : 0);
            best_d[34] = ((swap != 0) ? right_d : left_d);
            best_i[34] = ((swap != 0) ? right_i : left_i);
            best_d[38] = ((swap != 0) ? left_d : right_d);
            best_i[38] = ((swap != 0) ? left_i : right_i);
        }
    }
    {
        {
            float left_d = best_d[35];
            float right_d = best_d[39];
            int left_i = best_i[35];
            int right_i = best_i[39];
            int swap = ((left_d < right_d) ? 1 : 0);
            best_d[35] = ((swap != 0) ? right_d : left_d);
            best_i[35] = ((swap != 0) ? right_i : left_i);
            best_d[39] = ((swap != 0) ? left_d : right_d);
            best_i[39] = ((swap != 0) ? left_i : right_i);
        }
    }
    {
        {
            float left_d = best_d[40];
            float right_d = best_d[44];
            int left_i = best_i[40];
            int right_i = best_i[44];
            int swap = ((left_d < right_d) ? 1 : 0);
            best_d[40] = ((swap != 0) ? right_d : left_d);
            best_i[40] = ((swap != 0) ? right_i : left_i);
            best_d[44] = ((swap != 0) ? left_d : right_d);
            best_i[44] = ((swap != 0) ? left_i : right_i);
        }
    }
    {
        {
            float left_d = best_d[41];
            float right_d = best_d[45];
            int left_i = best_i[41];
            int right_i = best_i[45];
            int swap = ((left_d < right_d) ? 1 : 0);
            best_d[41] = ((swap != 0) ? right_d : left_d);
            best_i[41] = ((swap != 0) ? right_i : left_i);
            best_d[45] = ((swap != 0) ? left_d : right_d);
            best_i[45] = ((swap != 0) ? left_i : right_i);
        }
    }
    {
        {
            float left_d = best_d[42];
            float right_d = best_d[46];
            int left_i = best_i[42];
            int right_i = best_i[46];
            int swap = ((left_d < right_d) ? 1 : 0);
            best_d[42] = ((swap != 0) ? right_d : left_d);
            best_i[42] = ((swap != 0) ? right_i : left_i);
            best_d[46] = ((swap != 0) ? left_d : right_d);
            best_i[46] = ((swap != 0) ? left_i : right_i);
        }
    }
    {
        {
            float left_d = best_d[43];
            float right_d = best_d[47];
            int left_i = best_i[43];
            int right_i = best_i[47];
            int swap = ((left_d < right_d) ? 1 : 0);
            best_d[43] = ((swap != 0) ? right_d : left_d);
            best_i[43] = ((swap != 0) ? right_i : left_i);
            best_d[47] = ((swap != 0) ? left_d : right_d);
            best_i[47] = ((swap != 0) ? left_i : right_i);
        }
    }
    {
        {
            float left_d = best_d[48];
            float right_d = best_d[52];
            int left_i = best_i[48];
            int right_i = best_i[52];
            int swap = ((left_d < right_d) ? 1 : 0);
            best_d[48] = ((swap != 0) ? right_d : left_d);
            best_i[48] = ((swap != 0) ? right_i : left_i);
            best_d[52] = ((swap != 0) ? left_d : right_d);
            best_i[52] = ((swap != 0) ? left_i : right_i);
        }
    }
    {
        {
            float left_d = best_d[49];
            float right_d = best_d[53];
            int left_i = best_i[49];
            int right_i = best_i[53];
            int swap = ((left_d < right_d) ? 1 : 0);
            best_d[49] = ((swap != 0) ? right_d : left_d);
            best_i[49] = ((swap != 0) ? right_i : left_i);
            best_d[53] = ((swap != 0) ? left_d : right_d);
            best_i[53] = ((swap != 0) ? left_i : right_i);
        }
    }
    {
        {
            float left_d = best_d[50];
            float right_d = best_d[54];
            int left_i = best_i[50];
            int right_i = best_i[54];
            int swap = ((left_d < right_d) ? 1 : 0);
            best_d[50] = ((swap != 0) ? right_d : left_d);
            best_i[50] = ((swap != 0) ? right_i : left_i);
            best_d[54] = ((swap != 0) ? left_d : right_d);
            best_i[54] = ((swap != 0) ? left_i : right_i);
        }
    }
    {
        {
            float left_d = best_d[51];
            float right_d = best_d[55];
            int left_i = best_i[51];
            int right_i = best_i[55];
            int swap = ((left_d < right_d) ? 1 : 0);
            best_d[51] = ((swap != 0) ? right_d : left_d);
            best_i[51] = ((swap != 0) ? right_i : left_i);
            best_d[55] = ((swap != 0) ? left_d : right_d);
            best_i[55] = ((swap != 0) ? left_i : right_i);
        }
    }
    {
        {
            float left_d = best_d[56];
            float right_d = best_d[60];
            int left_i = best_i[56];
            int right_i = best_i[60];
            int swap = ((left_d < right_d) ? 1 : 0);
            best_d[56] = ((swap != 0) ? right_d : left_d);
            best_i[56] = ((swap != 0) ? right_i : left_i);
            best_d[60] = ((swap != 0) ? left_d : right_d);
            best_i[60] = ((swap != 0) ? left_i : right_i);
        }
    }
    {
        {
            float left_d = best_d[57];
            float right_d = best_d[61];
            int left_i = best_i[57];
            int right_i = best_i[61];
            int swap = ((left_d < right_d) ? 1 : 0);
            best_d[57] = ((swap != 0) ? right_d : left_d);
            best_i[57] = ((swap != 0) ? right_i : left_i);
            best_d[61] = ((swap != 0) ? left_d : right_d);
            best_i[61] = ((swap != 0) ? left_i : right_i);
        }
    }
    {
        {
            float left_d = best_d[58];
            float right_d = best_d[62];
            int left_i = best_i[58];
            int right_i = best_i[62];
            int swap = ((left_d < right_d) ? 1 : 0);
            best_d[58] = ((swap != 0) ? right_d : left_d);
            best_i[58] = ((swap != 0) ? right_i : left_i);
            best_d[62] = ((swap != 0) ? left_d : right_d);
            best_i[62] = ((swap != 0) ? left_i : right_i);
        }
    }
    {
        {
            float left_d = best_d[59];
            float right_d = best_d[63];
            int left_i = best_i[59];
            int right_i = best_i[63];
            int swap = ((left_d < right_d) ? 1 : 0);
            best_d[59] = ((swap != 0) ? right_d : left_d);
            best_i[59] = ((swap != 0) ? right_i : left_i);
            best_d[63] = ((swap != 0) ? left_d : right_d);
            best_i[63] = ((swap != 0) ? left_i : right_i);
        }
    }
    {
        {
            float left_d = best_d[0];
            float right_d = best_d[2];
            int left_i = best_i[0];
            int right_i = best_i[2];
            int swap = ((right_d < left_d) ? 1 : 0);
            best_d[0] = ((swap != 0) ? right_d : left_d);
            best_i[0] = ((swap != 0) ? right_i : left_i);
            best_d[2] = ((swap != 0) ? left_d : right_d);
            best_i[2] = ((swap != 0) ? left_i : right_i);
        }
    }
    {
        {
            float left_d = best_d[1];
            float right_d = best_d[3];
            int left_i = best_i[1];
            int right_i = best_i[3];
            int swap = ((right_d < left_d) ? 1 : 0);
            best_d[1] = ((swap != 0) ? right_d : left_d);
            best_i[1] = ((swap != 0) ? right_i : left_i);
            best_d[3] = ((swap != 0) ? left_d : right_d);
            best_i[3] = ((swap != 0) ? left_i : right_i);
        }
    }
    {
        {
            float left_d = best_d[4];
            float right_d = best_d[6];
            int left_i = best_i[4];
            int right_i = best_i[6];
            int swap = ((right_d < left_d) ? 1 : 0);
            best_d[4] = ((swap != 0) ? right_d : left_d);
            best_i[4] = ((swap != 0) ? right_i : left_i);
            best_d[6] = ((swap != 0) ? left_d : right_d);
            best_i[6] = ((swap != 0) ? left_i : right_i);
        }
    }
    {
        {
            float left_d = best_d[5];
            float right_d = best_d[7];
            int left_i = best_i[5];
            int right_i = best_i[7];
            int swap = ((right_d < left_d) ? 1 : 0);
            best_d[5] = ((swap != 0) ? right_d : left_d);
            best_i[5] = ((swap != 0) ? right_i : left_i);
            best_d[7] = ((swap != 0) ? left_d : right_d);
            best_i[7] = ((swap != 0) ? left_i : right_i);
        }
    }
    {
        {
            float left_d = best_d[8];
            float right_d = best_d[10];
            int left_i = best_i[8];
            int right_i = best_i[10];
            int swap = ((right_d < left_d) ? 1 : 0);
            best_d[8] = ((swap != 0) ? right_d : left_d);
            best_i[8] = ((swap != 0) ? right_i : left_i);
            best_d[10] = ((swap != 0) ? left_d : right_d);
            best_i[10] = ((swap != 0) ? left_i : right_i);
        }
    }
    {
        {
            float left_d = best_d[9];
            float right_d = best_d[11];
            int left_i = best_i[9];
            int right_i = best_i[11];
            int swap = ((right_d < left_d) ? 1 : 0);
            best_d[9] = ((swap != 0) ? right_d : left_d);
            best_i[9] = ((swap != 0) ? right_i : left_i);
            best_d[11] = ((swap != 0) ? left_d : right_d);
            best_i[11] = ((swap != 0) ? left_i : right_i);
        }
    }
    {
        {
            float left_d = best_d[12];
            float right_d = best_d[14];
            int left_i = best_i[12];
            int right_i = best_i[14];
            int swap = ((right_d < left_d) ? 1 : 0);
            best_d[12] = ((swap != 0) ? right_d : left_d);
            best_i[12] = ((swap != 0) ? right_i : left_i);
            best_d[14] = ((swap != 0) ? left_d : right_d);
            best_i[14] = ((swap != 0) ? left_i : right_i);
        }
    }
    {
        {
            float left_d = best_d[13];
            float right_d = best_d[15];
            int left_i = best_i[13];
            int right_i = best_i[15];
            int swap = ((right_d < left_d) ? 1 : 0);
            best_d[13] = ((swap != 0) ? right_d : left_d);
            best_i[13] = ((swap != 0) ? right_i : left_i);
            best_d[15] = ((swap != 0) ? left_d : right_d);
            best_i[15] = ((swap != 0) ? left_i : right_i);
        }
    }
    {
        {
            float left_d = best_d[16];
            float right_d = best_d[18];
            int left_i = best_i[16];
            int right_i = best_i[18];
            int swap = ((right_d < left_d) ? 1 : 0);
            best_d[16] = ((swap != 0) ? right_d : left_d);
            best_i[16] = ((swap != 0) ? right_i : left_i);
            best_d[18] = ((swap != 0) ? left_d : right_d);
            best_i[18] = ((swap != 0) ? left_i : right_i);
        }
    }
    {
        {
            float left_d = best_d[17];
            float right_d = best_d[19];
            int left_i = best_i[17];
            int right_i = best_i[19];
            int swap = ((right_d < left_d) ? 1 : 0);
            best_d[17] = ((swap != 0) ? right_d : left_d);
            best_i[17] = ((swap != 0) ? right_i : left_i);
            best_d[19] = ((swap != 0) ? left_d : right_d);
            best_i[19] = ((swap != 0) ? left_i : right_i);
        }
    }
    {
        {
            float left_d = best_d[20];
            float right_d = best_d[22];
            int left_i = best_i[20];
            int right_i = best_i[22];
            int swap = ((right_d < left_d) ? 1 : 0);
            best_d[20] = ((swap != 0) ? right_d : left_d);
            best_i[20] = ((swap != 0) ? right_i : left_i);
            best_d[22] = ((swap != 0) ? left_d : right_d);
            best_i[22] = ((swap != 0) ? left_i : right_i);
        }
    }
    {
        {
            float left_d = best_d[21];
            float right_d = best_d[23];
            int left_i = best_i[21];
            int right_i = best_i[23];
            int swap = ((right_d < left_d) ? 1 : 0);
            best_d[21] = ((swap != 0) ? right_d : left_d);
            best_i[21] = ((swap != 0) ? right_i : left_i);
            best_d[23] = ((swap != 0) ? left_d : right_d);
            best_i[23] = ((swap != 0) ? left_i : right_i);
        }
    }
    {
        {
            float left_d = best_d[24];
            float right_d = best_d[26];
            int left_i = best_i[24];
            int right_i = best_i[26];
            int swap = ((right_d < left_d) ? 1 : 0);
            best_d[24] = ((swap != 0) ? right_d : left_d);
            best_i[24] = ((swap != 0) ? right_i : left_i);
            best_d[26] = ((swap != 0) ? left_d : right_d);
            best_i[26] = ((swap != 0) ? left_i : right_i);
        }
    }
    {
        {
            float left_d = best_d[25];
            float right_d = best_d[27];
            int left_i = best_i[25];
            int right_i = best_i[27];
            int swap = ((right_d < left_d) ? 1 : 0);
            best_d[25] = ((swap != 0) ? right_d : left_d);
            best_i[25] = ((swap != 0) ? right_i : left_i);
            best_d[27] = ((swap != 0) ? left_d : right_d);
            best_i[27] = ((swap != 0) ? left_i : right_i);
        }
    }
    {
        {
            float left_d = best_d[28];
            float right_d = best_d[30];
            int left_i = best_i[28];
            int right_i = best_i[30];
            int swap = ((right_d < left_d) ? 1 : 0);
            best_d[28] = ((swap != 0) ? right_d : left_d);
            best_i[28] = ((swap != 0) ? right_i : left_i);
            best_d[30] = ((swap != 0) ? left_d : right_d);
            best_i[30] = ((swap != 0) ? left_i : right_i);
        }
    }
    {
        {
            float left_d = best_d[29];
            float right_d = best_d[31];
            int left_i = best_i[29];
            int right_i = best_i[31];
            int swap = ((right_d < left_d) ? 1 : 0);
            best_d[29] = ((swap != 0) ? right_d : left_d);
            best_i[29] = ((swap != 0) ? right_i : left_i);
            best_d[31] = ((swap != 0) ? left_d : right_d);
            best_i[31] = ((swap != 0) ? left_i : right_i);
        }
    }
    {
        {
            float left_d = best_d[32];
            float right_d = best_d[34];
            int left_i = best_i[32];
            int right_i = best_i[34];
            int swap = ((left_d < right_d) ? 1 : 0);
            best_d[32] = ((swap != 0) ? right_d : left_d);
            best_i[32] = ((swap != 0) ? right_i : left_i);
            best_d[34] = ((swap != 0) ? left_d : right_d);
            best_i[34] = ((swap != 0) ? left_i : right_i);
        }
    }
    {
        {
            float left_d = best_d[33];
            float right_d = best_d[35];
            int left_i = best_i[33];
            int right_i = best_i[35];
            int swap = ((left_d < right_d) ? 1 : 0);
            best_d[33] = ((swap != 0) ? right_d : left_d);
            best_i[33] = ((swap != 0) ? right_i : left_i);
            best_d[35] = ((swap != 0) ? left_d : right_d);
            best_i[35] = ((swap != 0) ? left_i : right_i);
        }
    }
    {
        {
            float left_d = best_d[36];
            float right_d = best_d[38];
            int left_i = best_i[36];
            int right_i = best_i[38];
            int swap = ((left_d < right_d) ? 1 : 0);
            best_d[36] = ((swap != 0) ? right_d : left_d);
            best_i[36] = ((swap != 0) ? right_i : left_i);
            best_d[38] = ((swap != 0) ? left_d : right_d);
            best_i[38] = ((swap != 0) ? left_i : right_i);
        }
    }
    {
        {
            float left_d = best_d[37];
            float right_d = best_d[39];
            int left_i = best_i[37];
            int right_i = best_i[39];
            int swap = ((left_d < right_d) ? 1 : 0);
            best_d[37] = ((swap != 0) ? right_d : left_d);
            best_i[37] = ((swap != 0) ? right_i : left_i);
            best_d[39] = ((swap != 0) ? left_d : right_d);
            best_i[39] = ((swap != 0) ? left_i : right_i);
        }
    }
    {
        {
            float left_d = best_d[40];
            float right_d = best_d[42];
            int left_i = best_i[40];
            int right_i = best_i[42];
            int swap = ((left_d < right_d) ? 1 : 0);
            best_d[40] = ((swap != 0) ? right_d : left_d);
            best_i[40] = ((swap != 0) ? right_i : left_i);
            best_d[42] = ((swap != 0) ? left_d : right_d);
            best_i[42] = ((swap != 0) ? left_i : right_i);
        }
    }
    {
        {
            float left_d = best_d[41];
            float right_d = best_d[43];
            int left_i = best_i[41];
            int right_i = best_i[43];
            int swap = ((left_d < right_d) ? 1 : 0);
            best_d[41] = ((swap != 0) ? right_d : left_d);
            best_i[41] = ((swap != 0) ? right_i : left_i);
            best_d[43] = ((swap != 0) ? left_d : right_d);
            best_i[43] = ((swap != 0) ? left_i : right_i);
        }
    }
    {
        {
            float left_d = best_d[44];
            float right_d = best_d[46];
            int left_i = best_i[44];
            int right_i = best_i[46];
            int swap = ((left_d < right_d) ? 1 : 0);
            best_d[44] = ((swap != 0) ? right_d : left_d);
            best_i[44] = ((swap != 0) ? right_i : left_i);
            best_d[46] = ((swap != 0) ? left_d : right_d);
            best_i[46] = ((swap != 0) ? left_i : right_i);
        }
    }
    {
        {
            float left_d = best_d[45];
            float right_d = best_d[47];
            int left_i = best_i[45];
            int right_i = best_i[47];
            int swap = ((left_d < right_d) ? 1 : 0);
            best_d[45] = ((swap != 0) ? right_d : left_d);
            best_i[45] = ((swap != 0) ? right_i : left_i);
            best_d[47] = ((swap != 0) ? left_d : right_d);
            best_i[47] = ((swap != 0) ? left_i : right_i);
        }
    }
    {
        {
            float left_d = best_d[48];
            float right_d = best_d[50];
            int left_i = best_i[48];
            int right_i = best_i[50];
            int swap = ((left_d < right_d) ? 1 : 0);
            best_d[48] = ((swap != 0) ? right_d : left_d);
            best_i[48] = ((swap != 0) ? right_i : left_i);
            best_d[50] = ((swap != 0) ? left_d : right_d);
            best_i[50] = ((swap != 0) ? left_i : right_i);
        }
    }
    {
        {
            float left_d = best_d[49];
            float right_d = best_d[51];
            int left_i = best_i[49];
            int right_i = best_i[51];
            int swap = ((left_d < right_d) ? 1 : 0);
            best_d[49] = ((swap != 0) ? right_d : left_d);
            best_i[49] = ((swap != 0) ? right_i : left_i);
            best_d[51] = ((swap != 0) ? left_d : right_d);
            best_i[51] = ((swap != 0) ? left_i : right_i);
        }
    }
    {
        {
            float left_d = best_d[52];
            float right_d = best_d[54];
            int left_i = best_i[52];
            int right_i = best_i[54];
            int swap = ((left_d < right_d) ? 1 : 0);
            best_d[52] = ((swap != 0) ? right_d : left_d);
            best_i[52] = ((swap != 0) ? right_i : left_i);
            best_d[54] = ((swap != 0) ? left_d : right_d);
            best_i[54] = ((swap != 0) ? left_i : right_i);
        }
    }
    {
        {
            float left_d = best_d[53];
            float right_d = best_d[55];
            int left_i = best_i[53];
            int right_i = best_i[55];
            int swap = ((left_d < right_d) ? 1 : 0);
            best_d[53] = ((swap != 0) ? right_d : left_d);
            best_i[53] = ((swap != 0) ? right_i : left_i);
            best_d[55] = ((swap != 0) ? left_d : right_d);
            best_i[55] = ((swap != 0) ? left_i : right_i);
        }
    }
    {
        {
            float left_d = best_d[56];
            float right_d = best_d[58];
            int left_i = best_i[56];
            int right_i = best_i[58];
            int swap = ((left_d < right_d) ? 1 : 0);
            best_d[56] = ((swap != 0) ? right_d : left_d);
            best_i[56] = ((swap != 0) ? right_i : left_i);
            best_d[58] = ((swap != 0) ? left_d : right_d);
            best_i[58] = ((swap != 0) ? left_i : right_i);
        }
    }
    {
        {
            float left_d = best_d[57];
            float right_d = best_d[59];
            int left_i = best_i[57];
            int right_i = best_i[59];
            int swap = ((left_d < right_d) ? 1 : 0);
            best_d[57] = ((swap != 0) ? right_d : left_d);
            best_i[57] = ((swap != 0) ? right_i : left_i);
            best_d[59] = ((swap != 0) ? left_d : right_d);
            best_i[59] = ((swap != 0) ? left_i : right_i);
        }
    }
    {
        {
            float left_d = best_d[60];
            float right_d = best_d[62];
            int left_i = best_i[60];
            int right_i = best_i[62];
            int swap = ((left_d < right_d) ? 1 : 0);
            best_d[60] = ((swap != 0) ? right_d : left_d);
            best_i[60] = ((swap != 0) ? right_i : left_i);
            best_d[62] = ((swap != 0) ? left_d : right_d);
            best_i[62] = ((swap != 0) ? left_i : right_i);
        }
    }
    {
        {
            float left_d = best_d[61];
            float right_d = best_d[63];
            int left_i = best_i[61];
            int right_i = best_i[63];
            int swap = ((left_d < right_d) ? 1 : 0);
            best_d[61] = ((swap != 0) ? right_d : left_d);
            best_i[61] = ((swap != 0) ? right_i : left_i);
            best_d[63] = ((swap != 0) ? left_d : right_d);
            best_i[63] = ((swap != 0) ? left_i : right_i);
        }
    }
    {
        {
            float left_d = best_d[0];
            float right_d = best_d[1];
            int left_i = best_i[0];
            int right_i = best_i[1];
            int swap = ((right_d < left_d) ? 1 : 0);
            best_d[0] = ((swap != 0) ? right_d : left_d);
            best_i[0] = ((swap != 0) ? right_i : left_i);
            best_d[1] = ((swap != 0) ? left_d : right_d);
            best_i[1] = ((swap != 0) ? left_i : right_i);
        }
    }
    {
        {
            float left_d = best_d[2];
            float right_d = best_d[3];
            int left_i = best_i[2];
            int right_i = best_i[3];
            int swap = ((right_d < left_d) ? 1 : 0);
            best_d[2] = ((swap != 0) ? right_d : left_d);
            best_i[2] = ((swap != 0) ? right_i : left_i);
            best_d[3] = ((swap != 0) ? left_d : right_d);
            best_i[3] = ((swap != 0) ? left_i : right_i);
        }
    }
    {
        {
            float left_d = best_d[4];
            float right_d = best_d[5];
            int left_i = best_i[4];
            int right_i = best_i[5];
            int swap = ((right_d < left_d) ? 1 : 0);
            best_d[4] = ((swap != 0) ? right_d : left_d);
            best_i[4] = ((swap != 0) ? right_i : left_i);
            best_d[5] = ((swap != 0) ? left_d : right_d);
            best_i[5] = ((swap != 0) ? left_i : right_i);
        }
    }
    {
        {
            float left_d = best_d[6];
            float right_d = best_d[7];
            int left_i = best_i[6];
            int right_i = best_i[7];
            int swap = ((right_d < left_d) ? 1 : 0);
            best_d[6] = ((swap != 0) ? right_d : left_d);
            best_i[6] = ((swap != 0) ? right_i : left_i);
            best_d[7] = ((swap != 0) ? left_d : right_d);
            best_i[7] = ((swap != 0) ? left_i : right_i);
        }
    }
    {
        {
            float left_d = best_d[8];
            float right_d = best_d[9];
            int left_i = best_i[8];
            int right_i = best_i[9];
            int swap = ((right_d < left_d) ? 1 : 0);
            best_d[8] = ((swap != 0) ? right_d : left_d);
            best_i[8] = ((swap != 0) ? right_i : left_i);
            best_d[9] = ((swap != 0) ? left_d : right_d);
            best_i[9] = ((swap != 0) ? left_i : right_i);
        }
    }
    {
        {
            float left_d = best_d[10];
            float right_d = best_d[11];
            int left_i = best_i[10];
            int right_i = best_i[11];
            int swap = ((right_d < left_d) ? 1 : 0);
            best_d[10] = ((swap != 0) ? right_d : left_d);
            best_i[10] = ((swap != 0) ? right_i : left_i);
            best_d[11] = ((swap != 0) ? left_d : right_d);
            best_i[11] = ((swap != 0) ? left_i : right_i);
        }
    }
    {
        {
            float left_d = best_d[12];
            float right_d = best_d[13];
            int left_i = best_i[12];
            int right_i = best_i[13];
            int swap = ((right_d < left_d) ? 1 : 0);
            best_d[12] = ((swap != 0) ? right_d : left_d);
            best_i[12] = ((swap != 0) ? right_i : left_i);
            best_d[13] = ((swap != 0) ? left_d : right_d);
            best_i[13] = ((swap != 0) ? left_i : right_i);
        }
    }
    {
        {
            float left_d = best_d[14];
            float right_d = best_d[15];
            int left_i = best_i[14];
            int right_i = best_i[15];
            int swap = ((right_d < left_d) ? 1 : 0);
            best_d[14] = ((swap != 0) ? right_d : left_d);
            best_i[14] = ((swap != 0) ? right_i : left_i);
            best_d[15] = ((swap != 0) ? left_d : right_d);
            best_i[15] = ((swap != 0) ? left_i : right_i);
        }
    }
    {
        {
            float left_d = best_d[16];
            float right_d = best_d[17];
            int left_i = best_i[16];
            int right_i = best_i[17];
            int swap = ((right_d < left_d) ? 1 : 0);
            best_d[16] = ((swap != 0) ? right_d : left_d);
            best_i[16] = ((swap != 0) ? right_i : left_i);
            best_d[17] = ((swap != 0) ? left_d : right_d);
            best_i[17] = ((swap != 0) ? left_i : right_i);
        }
    }
    {
        {
            float left_d = best_d[18];
            float right_d = best_d[19];
            int left_i = best_i[18];
            int right_i = best_i[19];
            int swap = ((right_d < left_d) ? 1 : 0);
            best_d[18] = ((swap != 0) ? right_d : left_d);
            best_i[18] = ((swap != 0) ? right_i : left_i);
            best_d[19] = ((swap != 0) ? left_d : right_d);
            best_i[19] = ((swap != 0) ? left_i : right_i);
        }
    }
    {
        {
            float left_d = best_d[20];
            float right_d = best_d[21];
            int left_i = best_i[20];
            int right_i = best_i[21];
            int swap = ((right_d < left_d) ? 1 : 0);
            best_d[20] = ((swap != 0) ? right_d : left_d);
            best_i[20] = ((swap != 0) ? right_i : left_i);
            best_d[21] = ((swap != 0) ? left_d : right_d);
            best_i[21] = ((swap != 0) ? left_i : right_i);
        }
    }
    {
        {
            float left_d = best_d[22];
            float right_d = best_d[23];
            int left_i = best_i[22];
            int right_i = best_i[23];
            int swap = ((right_d < left_d) ? 1 : 0);
            best_d[22] = ((swap != 0) ? right_d : left_d);
            best_i[22] = ((swap != 0) ? right_i : left_i);
            best_d[23] = ((swap != 0) ? left_d : right_d);
            best_i[23] = ((swap != 0) ? left_i : right_i);
        }
    }
    {
        {
            float left_d = best_d[24];
            float right_d = best_d[25];
            int left_i = best_i[24];
            int right_i = best_i[25];
            int swap = ((right_d < left_d) ? 1 : 0);
            best_d[24] = ((swap != 0) ? right_d : left_d);
            best_i[24] = ((swap != 0) ? right_i : left_i);
            best_d[25] = ((swap != 0) ? left_d : right_d);
            best_i[25] = ((swap != 0) ? left_i : right_i);
        }
    }
    {
        {
            float left_d = best_d[26];
            float right_d = best_d[27];
            int left_i = best_i[26];
            int right_i = best_i[27];
            int swap = ((right_d < left_d) ? 1 : 0);
            best_d[26] = ((swap != 0) ? right_d : left_d);
            best_i[26] = ((swap != 0) ? right_i : left_i);
            best_d[27] = ((swap != 0) ? left_d : right_d);
            best_i[27] = ((swap != 0) ? left_i : right_i);
        }
    }
    {
        {
            float left_d = best_d[28];
            float right_d = best_d[29];
            int left_i = best_i[28];
            int right_i = best_i[29];
            int swap = ((right_d < left_d) ? 1 : 0);
            best_d[28] = ((swap != 0) ? right_d : left_d);
            best_i[28] = ((swap != 0) ? right_i : left_i);
            best_d[29] = ((swap != 0) ? left_d : right_d);
            best_i[29] = ((swap != 0) ? left_i : right_i);
        }
    }
    {
        {
            float left_d = best_d[30];
            float right_d = best_d[31];
            int left_i = best_i[30];
            int right_i = best_i[31];
            int swap = ((right_d < left_d) ? 1 : 0);
            best_d[30] = ((swap != 0) ? right_d : left_d);
            best_i[30] = ((swap != 0) ? right_i : left_i);
            best_d[31] = ((swap != 0) ? left_d : right_d);
            best_i[31] = ((swap != 0) ? left_i : right_i);
        }
    }
    {
        {
            float left_d = best_d[32];
            float right_d = best_d[33];
            int left_i = best_i[32];
            int right_i = best_i[33];
            int swap = ((left_d < right_d) ? 1 : 0);
            best_d[32] = ((swap != 0) ? right_d : left_d);
            best_i[32] = ((swap != 0) ? right_i : left_i);
            best_d[33] = ((swap != 0) ? left_d : right_d);
            best_i[33] = ((swap != 0) ? left_i : right_i);
        }
    }
    {
        {
            float left_d = best_d[34];
            float right_d = best_d[35];
            int left_i = best_i[34];
            int right_i = best_i[35];
            int swap = ((left_d < right_d) ? 1 : 0);
            best_d[34] = ((swap != 0) ? right_d : left_d);
            best_i[34] = ((swap != 0) ? right_i : left_i);
            best_d[35] = ((swap != 0) ? left_d : right_d);
            best_i[35] = ((swap != 0) ? left_i : right_i);
        }
    }
    {
        {
            float left_d = best_d[36];
            float right_d = best_d[37];
            int left_i = best_i[36];
            int right_i = best_i[37];
            int swap = ((left_d < right_d) ? 1 : 0);
            best_d[36] = ((swap != 0) ? right_d : left_d);
            best_i[36] = ((swap != 0) ? right_i : left_i);
            best_d[37] = ((swap != 0) ? left_d : right_d);
            best_i[37] = ((swap != 0) ? left_i : right_i);
        }
    }
    {
        {
            float left_d = best_d[38];
            float right_d = best_d[39];
            int left_i = best_i[38];
            int right_i = best_i[39];
            int swap = ((left_d < right_d) ? 1 : 0);
            best_d[38] = ((swap != 0) ? right_d : left_d);
            best_i[38] = ((swap != 0) ? right_i : left_i);
            best_d[39] = ((swap != 0) ? left_d : right_d);
            best_i[39] = ((swap != 0) ? left_i : right_i);
        }
    }
    {
        {
            float left_d = best_d[40];
            float right_d = best_d[41];
            int left_i = best_i[40];
            int right_i = best_i[41];
            int swap = ((left_d < right_d) ? 1 : 0);
            best_d[40] = ((swap != 0) ? right_d : left_d);
            best_i[40] = ((swap != 0) ? right_i : left_i);
            best_d[41] = ((swap != 0) ? left_d : right_d);
            best_i[41] = ((swap != 0) ? left_i : right_i);
        }
    }
    {
        {
            float left_d = best_d[42];
            float right_d = best_d[43];
            int left_i = best_i[42];
            int right_i = best_i[43];
            int swap = ((left_d < right_d) ? 1 : 0);
            best_d[42] = ((swap != 0) ? right_d : left_d);
            best_i[42] = ((swap != 0) ? right_i : left_i);
            best_d[43] = ((swap != 0) ? left_d : right_d);
            best_i[43] = ((swap != 0) ? left_i : right_i);
        }
    }
    {
        {
            float left_d = best_d[44];
            float right_d = best_d[45];
            int left_i = best_i[44];
            int right_i = best_i[45];
            int swap = ((left_d < right_d) ? 1 : 0);
            best_d[44] = ((swap != 0) ? right_d : left_d);
            best_i[44] = ((swap != 0) ? right_i : left_i);
            best_d[45] = ((swap != 0) ? left_d : right_d);
            best_i[45] = ((swap != 0) ? left_i : right_i);
        }
    }
    {
        {
            float left_d = best_d[46];
            float right_d = best_d[47];
            int left_i = best_i[46];
            int right_i = best_i[47];
            int swap = ((left_d < right_d) ? 1 : 0);
            best_d[46] = ((swap != 0) ? right_d : left_d);
            best_i[46] = ((swap != 0) ? right_i : left_i);
            best_d[47] = ((swap != 0) ? left_d : right_d);
            best_i[47] = ((swap != 0) ? left_i : right_i);
        }
    }
    {
        {
            float left_d = best_d[48];
            float right_d = best_d[49];
            int left_i = best_i[48];
            int right_i = best_i[49];
            int swap = ((left_d < right_d) ? 1 : 0);
            best_d[48] = ((swap != 0) ? right_d : left_d);
            best_i[48] = ((swap != 0) ? right_i : left_i);
            best_d[49] = ((swap != 0) ? left_d : right_d);
            best_i[49] = ((swap != 0) ? left_i : right_i);
        }
    }
    {
        {
            float left_d = best_d[50];
            float right_d = best_d[51];
            int left_i = best_i[50];
            int right_i = best_i[51];
            int swap = ((left_d < right_d) ? 1 : 0);
            best_d[50] = ((swap != 0) ? right_d : left_d);
            best_i[50] = ((swap != 0) ? right_i : left_i);
            best_d[51] = ((swap != 0) ? left_d : right_d);
            best_i[51] = ((swap != 0) ? left_i : right_i);
        }
    }
    {
        {
            float left_d = best_d[52];
            float right_d = best_d[53];
            int left_i = best_i[52];
            int right_i = best_i[53];
            int swap = ((left_d < right_d) ? 1 : 0);
            best_d[52] = ((swap != 0) ? right_d : left_d);
            best_i[52] = ((swap != 0) ? right_i : left_i);
            best_d[53] = ((swap != 0) ? left_d : right_d);
            best_i[53] = ((swap != 0) ? left_i : right_i);
        }
    }
    {
        {
            float left_d = best_d[54];
            float right_d = best_d[55];
            int left_i = best_i[54];
            int right_i = best_i[55];
            int swap = ((left_d < right_d) ? 1 : 0);
            best_d[54] = ((swap != 0) ? right_d : left_d);
            best_i[54] = ((swap != 0) ? right_i : left_i);
            best_d[55] = ((swap != 0) ? left_d : right_d);
            best_i[55] = ((swap != 0) ? left_i : right_i);
        }
    }
    {
        {
            float left_d = best_d[56];
            float right_d = best_d[57];
            int left_i = best_i[56];
            int right_i = best_i[57];
            int swap = ((left_d < right_d) ? 1 : 0);
            best_d[56] = ((swap != 0) ? right_d : left_d);
            best_i[56] = ((swap != 0) ? right_i : left_i);
            best_d[57] = ((swap != 0) ? left_d : right_d);
            best_i[57] = ((swap != 0) ? left_i : right_i);
        }
    }
    {
        {
            float left_d = best_d[58];
            float right_d = best_d[59];
            int left_i = best_i[58];
            int right_i = best_i[59];
            int swap = ((left_d < right_d) ? 1 : 0);
            best_d[58] = ((swap != 0) ? right_d : left_d);
            best_i[58] = ((swap != 0) ? right_i : left_i);
            best_d[59] = ((swap != 0) ? left_d : right_d);
            best_i[59] = ((swap != 0) ? left_i : right_i);
        }
    }
    {
        {
            float left_d = best_d[60];
            float right_d = best_d[61];
            int left_i = best_i[60];
            int right_i = best_i[61];
            int swap = ((left_d < right_d) ? 1 : 0);
            best_d[60] = ((swap != 0) ? right_d : left_d);
            best_i[60] = ((swap != 0) ? right_i : left_i);
            best_d[61] = ((swap != 0) ? left_d : right_d);
            best_i[61] = ((swap != 0) ? left_i : right_i);
        }
    }
    {
        {
            float left_d = best_d[62];
            float right_d = best_d[63];
            int left_i = best_i[62];
            int right_i = best_i[63];
            int swap = ((left_d < right_d) ? 1 : 0);
            best_d[62] = ((swap != 0) ? right_d : left_d);
            best_i[62] = ((swap != 0) ? right_i : left_i);
            best_d[63] = ((swap != 0) ? left_d : right_d);
            best_i[63] = ((swap != 0) ? left_i : right_i);
        }
    }
    {
        {
            float left_d = best_d[0];
            float right_d = best_d[32];
            int left_i = best_i[0];
            int right_i = best_i[32];
            int swap = ((right_d < left_d) ? 1 : 0);
            best_d[0] = ((swap != 0) ? right_d : left_d);
            best_i[0] = ((swap != 0) ? right_i : left_i);
            best_d[32] = ((swap != 0) ? left_d : right_d);
            best_i[32] = ((swap != 0) ? left_i : right_i);
        }
    }
    {
        {
            float left_d = best_d[1];
            float right_d = best_d[33];
            int left_i = best_i[1];
            int right_i = best_i[33];
            int swap = ((right_d < left_d) ? 1 : 0);
            best_d[1] = ((swap != 0) ? right_d : left_d);
            best_i[1] = ((swap != 0) ? right_i : left_i);
            best_d[33] = ((swap != 0) ? left_d : right_d);
            best_i[33] = ((swap != 0) ? left_i : right_i);
        }
    }
    {
        {
            float left_d = best_d[2];
            float right_d = best_d[34];
            int left_i = best_i[2];
            int right_i = best_i[34];
            int swap = ((right_d < left_d) ? 1 : 0);
            best_d[2] = ((swap != 0) ? right_d : left_d);
            best_i[2] = ((swap != 0) ? right_i : left_i);
            best_d[34] = ((swap != 0) ? left_d : right_d);
            best_i[34] = ((swap != 0) ? left_i : right_i);
        }
    }
    {
        {
            float left_d = best_d[3];
            float right_d = best_d[35];
            int left_i = best_i[3];
            int right_i = best_i[35];
            int swap = ((right_d < left_d) ? 1 : 0);
            best_d[3] = ((swap != 0) ? right_d : left_d);
            best_i[3] = ((swap != 0) ? right_i : left_i);
            best_d[35] = ((swap != 0) ? left_d : right_d);
            best_i[35] = ((swap != 0) ? left_i : right_i);
        }
    }
    {
        {
            float left_d = best_d[4];
            float right_d = best_d[36];
            int left_i = best_i[4];
            int right_i = best_i[36];
            int swap = ((right_d < left_d) ? 1 : 0);
            best_d[4] = ((swap != 0) ? right_d : left_d);
            best_i[4] = ((swap != 0) ? right_i : left_i);
            best_d[36] = ((swap != 0) ? left_d : right_d);
            best_i[36] = ((swap != 0) ? left_i : right_i);
        }
    }
    {
        {
            float left_d = best_d[5];
            float right_d = best_d[37];
            int left_i = best_i[5];
            int right_i = best_i[37];
            int swap = ((right_d < left_d) ? 1 : 0);
            best_d[5] = ((swap != 0) ? right_d : left_d);
            best_i[5] = ((swap != 0) ? right_i : left_i);
            best_d[37] = ((swap != 0) ? left_d : right_d);
            best_i[37] = ((swap != 0) ? left_i : right_i);
        }
    }
    {
        {
            float left_d = best_d[6];
            float right_d = best_d[38];
            int left_i = best_i[6];
            int right_i = best_i[38];
            int swap = ((right_d < left_d) ? 1 : 0);
            best_d[6] = ((swap != 0) ? right_d : left_d);
            best_i[6] = ((swap != 0) ? right_i : left_i);
            best_d[38] = ((swap != 0) ? left_d : right_d);
            best_i[38] = ((swap != 0) ? left_i : right_i);
        }
    }
    {
        {
            float left_d = best_d[7];
            float right_d = best_d[39];
            int left_i = best_i[7];
            int right_i = best_i[39];
            int swap = ((right_d < left_d) ? 1 : 0);
            best_d[7] = ((swap != 0) ? right_d : left_d);
            best_i[7] = ((swap != 0) ? right_i : left_i);
            best_d[39] = ((swap != 0) ? left_d : right_d);
            best_i[39] = ((swap != 0) ? left_i : right_i);
        }
    }
    {
        {
            float left_d = best_d[8];
            float right_d = best_d[40];
            int left_i = best_i[8];
            int right_i = best_i[40];
            int swap = ((right_d < left_d) ? 1 : 0);
            best_d[8] = ((swap != 0) ? right_d : left_d);
            best_i[8] = ((swap != 0) ? right_i : left_i);
            best_d[40] = ((swap != 0) ? left_d : right_d);
            best_i[40] = ((swap != 0) ? left_i : right_i);
        }
    }
    {
        {
            float left_d = best_d[9];
            float right_d = best_d[41];
            int left_i = best_i[9];
            int right_i = best_i[41];
            int swap = ((right_d < left_d) ? 1 : 0);
            best_d[9] = ((swap != 0) ? right_d : left_d);
            best_i[9] = ((swap != 0) ? right_i : left_i);
            best_d[41] = ((swap != 0) ? left_d : right_d);
            best_i[41] = ((swap != 0) ? left_i : right_i);
        }
    }
    {
        {
            float left_d = best_d[10];
            float right_d = best_d[42];
            int left_i = best_i[10];
            int right_i = best_i[42];
            int swap = ((right_d < left_d) ? 1 : 0);
            best_d[10] = ((swap != 0) ? right_d : left_d);
            best_i[10] = ((swap != 0) ? right_i : left_i);
            best_d[42] = ((swap != 0) ? left_d : right_d);
            best_i[42] = ((swap != 0) ? left_i : right_i);
        }
    }
    {
        {
            float left_d = best_d[11];
            float right_d = best_d[43];
            int left_i = best_i[11];
            int right_i = best_i[43];
            int swap = ((right_d < left_d) ? 1 : 0);
            best_d[11] = ((swap != 0) ? right_d : left_d);
            best_i[11] = ((swap != 0) ? right_i : left_i);
            best_d[43] = ((swap != 0) ? left_d : right_d);
            best_i[43] = ((swap != 0) ? left_i : right_i);
        }
    }
    {
        {
            float left_d = best_d[12];
            float right_d = best_d[44];
            int left_i = best_i[12];
            int right_i = best_i[44];
            int swap = ((right_d < left_d) ? 1 : 0);
            best_d[12] = ((swap != 0) ? right_d : left_d);
            best_i[12] = ((swap != 0) ? right_i : left_i);
            best_d[44] = ((swap != 0) ? left_d : right_d);
            best_i[44] = ((swap != 0) ? left_i : right_i);
        }
    }
    {
        {
            float left_d = best_d[13];
            float right_d = best_d[45];
            int left_i = best_i[13];
            int right_i = best_i[45];
            int swap = ((right_d < left_d) ? 1 : 0);
            best_d[13] = ((swap != 0) ? right_d : left_d);
            best_i[13] = ((swap != 0) ? right_i : left_i);
            best_d[45] = ((swap != 0) ? left_d : right_d);
            best_i[45] = ((swap != 0) ? left_i : right_i);
        }
    }
    {
        {
            float left_d = best_d[14];
            float right_d = best_d[46];
            int left_i = best_i[14];
            int right_i = best_i[46];
            int swap = ((right_d < left_d) ? 1 : 0);
            best_d[14] = ((swap != 0) ? right_d : left_d);
            best_i[14] = ((swap != 0) ? right_i : left_i);
            best_d[46] = ((swap != 0) ? left_d : right_d);
            best_i[46] = ((swap != 0) ? left_i : right_i);
        }
    }
    {
        {
            float left_d = best_d[15];
            float right_d = best_d[47];
            int left_i = best_i[15];
            int right_i = best_i[47];
            int swap = ((right_d < left_d) ? 1 : 0);
            best_d[15] = ((swap != 0) ? right_d : left_d);
            best_i[15] = ((swap != 0) ? right_i : left_i);
            best_d[47] = ((swap != 0) ? left_d : right_d);
            best_i[47] = ((swap != 0) ? left_i : right_i);
        }
    }
    {
        {
            float left_d = best_d[16];
            float right_d = best_d[48];
            int left_i = best_i[16];
            int right_i = best_i[48];
            int swap = ((right_d < left_d) ? 1 : 0);
            best_d[16] = ((swap != 0) ? right_d : left_d);
            best_i[16] = ((swap != 0) ? right_i : left_i);
            best_d[48] = ((swap != 0) ? left_d : right_d);
            best_i[48] = ((swap != 0) ? left_i : right_i);
        }
    }
    {
        {
            float left_d = best_d[17];
            float right_d = best_d[49];
            int left_i = best_i[17];
            int right_i = best_i[49];
            int swap = ((right_d < left_d) ? 1 : 0);
            best_d[17] = ((swap != 0) ? right_d : left_d);
            best_i[17] = ((swap != 0) ? right_i : left_i);
            best_d[49] = ((swap != 0) ? left_d : right_d);
            best_i[49] = ((swap != 0) ? left_i : right_i);
        }
    }
    {
        {
            float left_d = best_d[18];
            float right_d = best_d[50];
            int left_i = best_i[18];
            int right_i = best_i[50];
            int swap = ((right_d < left_d) ? 1 : 0);
            best_d[18] = ((swap != 0) ? right_d : left_d);
            best_i[18] = ((swap != 0) ? right_i : left_i);
            best_d[50] = ((swap != 0) ? left_d : right_d);
            best_i[50] = ((swap != 0) ? left_i : right_i);
        }
    }
    {
        {
            float left_d = best_d[19];
            float right_d = best_d[51];
            int left_i = best_i[19];
            int right_i = best_i[51];
            int swap = ((right_d < left_d) ? 1 : 0);
            best_d[19] = ((swap != 0) ? right_d : left_d);
            best_i[19] = ((swap != 0) ? right_i : left_i);
            best_d[51] = ((swap != 0) ? left_d : right_d);
            best_i[51] = ((swap != 0) ? left_i : right_i);
        }
    }
    {
        {
            float left_d = best_d[20];
            float right_d = best_d[52];
            int left_i = best_i[20];
            int right_i = best_i[52];
            int swap = ((right_d < left_d) ? 1 : 0);
            best_d[20] = ((swap != 0) ? right_d : left_d);
            best_i[20] = ((swap != 0) ? right_i : left_i);
            best_d[52] = ((swap != 0) ? left_d : right_d);
            best_i[52] = ((swap != 0) ? left_i : right_i);
        }
    }
    {
        {
            float left_d = best_d[21];
            float right_d = best_d[53];
            int left_i = best_i[21];
            int right_i = best_i[53];
            int swap = ((right_d < left_d) ? 1 : 0);
            best_d[21] = ((swap != 0) ? right_d : left_d);
            best_i[21] = ((swap != 0) ? right_i : left_i);
            best_d[53] = ((swap != 0) ? left_d : right_d);
            best_i[53] = ((swap != 0) ? left_i : right_i);
        }
    }
    {
        {
            float left_d = best_d[22];
            float right_d = best_d[54];
            int left_i = best_i[22];
            int right_i = best_i[54];
            int swap = ((right_d < left_d) ? 1 : 0);
            best_d[22] = ((swap != 0) ? right_d : left_d);
            best_i[22] = ((swap != 0) ? right_i : left_i);
            best_d[54] = ((swap != 0) ? left_d : right_d);
            best_i[54] = ((swap != 0) ? left_i : right_i);
        }
    }
    {
        {
            float left_d = best_d[23];
            float right_d = best_d[55];
            int left_i = best_i[23];
            int right_i = best_i[55];
            int swap = ((right_d < left_d) ? 1 : 0);
            best_d[23] = ((swap != 0) ? right_d : left_d);
            best_i[23] = ((swap != 0) ? right_i : left_i);
            best_d[55] = ((swap != 0) ? left_d : right_d);
            best_i[55] = ((swap != 0) ? left_i : right_i);
        }
    }
    {
        {
            float left_d = best_d[24];
            float right_d = best_d[56];
            int left_i = best_i[24];
            int right_i = best_i[56];
            int swap = ((right_d < left_d) ? 1 : 0);
            best_d[24] = ((swap != 0) ? right_d : left_d);
            best_i[24] = ((swap != 0) ? right_i : left_i);
            best_d[56] = ((swap != 0) ? left_d : right_d);
            best_i[56] = ((swap != 0) ? left_i : right_i);
        }
    }
    {
        {
            float left_d = best_d[25];
            float right_d = best_d[57];
            int left_i = best_i[25];
            int right_i = best_i[57];
            int swap = ((right_d < left_d) ? 1 : 0);
            best_d[25] = ((swap != 0) ? right_d : left_d);
            best_i[25] = ((swap != 0) ? right_i : left_i);
            best_d[57] = ((swap != 0) ? left_d : right_d);
            best_i[57] = ((swap != 0) ? left_i : right_i);
        }
    }
    {
        {
            float left_d = best_d[26];
            float right_d = best_d[58];
            int left_i = best_i[26];
            int right_i = best_i[58];
            int swap = ((right_d < left_d) ? 1 : 0);
            best_d[26] = ((swap != 0) ? right_d : left_d);
            best_i[26] = ((swap != 0) ? right_i : left_i);
            best_d[58] = ((swap != 0) ? left_d : right_d);
            best_i[58] = ((swap != 0) ? left_i : right_i);
        }
    }
    {
        {
            float left_d = best_d[27];
            float right_d = best_d[59];
            int left_i = best_i[27];
            int right_i = best_i[59];
            int swap = ((right_d < left_d) ? 1 : 0);
            best_d[27] = ((swap != 0) ? right_d : left_d);
            best_i[27] = ((swap != 0) ? right_i : left_i);
            best_d[59] = ((swap != 0) ? left_d : right_d);
            best_i[59] = ((swap != 0) ? left_i : right_i);
        }
    }
    {
        {
            float left_d = best_d[28];
            float right_d = best_d[60];
            int left_i = best_i[28];
            int right_i = best_i[60];
            int swap = ((right_d < left_d) ? 1 : 0);
            best_d[28] = ((swap != 0) ? right_d : left_d);
            best_i[28] = ((swap != 0) ? right_i : left_i);
            best_d[60] = ((swap != 0) ? left_d : right_d);
            best_i[60] = ((swap != 0) ? left_i : right_i);
        }
    }
    {
        {
            float left_d = best_d[29];
            float right_d = best_d[61];
            int left_i = best_i[29];
            int right_i = best_i[61];
            int swap = ((right_d < left_d) ? 1 : 0);
            best_d[29] = ((swap != 0) ? right_d : left_d);
            best_i[29] = ((swap != 0) ? right_i : left_i);
            best_d[61] = ((swap != 0) ? left_d : right_d);
            best_i[61] = ((swap != 0) ? left_i : right_i);
        }
    }
    {
        {
            float left_d = best_d[30];
            float right_d = best_d[62];
            int left_i = best_i[30];
            int right_i = best_i[62];
            int swap = ((right_d < left_d) ? 1 : 0);
            best_d[30] = ((swap != 0) ? right_d : left_d);
            best_i[30] = ((swap != 0) ? right_i : left_i);
            best_d[62] = ((swap != 0) ? left_d : right_d);
            best_i[62] = ((swap != 0) ? left_i : right_i);
        }
    }
    {
        {
            float left_d = best_d[31];
            float right_d = best_d[63];
            int left_i = best_i[31];
            int right_i = best_i[63];
            int swap = ((right_d < left_d) ? 1 : 0);
            best_d[31] = ((swap != 0) ? right_d : left_d);
            best_i[31] = ((swap != 0) ? right_i : left_i);
            best_d[63] = ((swap != 0) ? left_d : right_d);
            best_i[63] = ((swap != 0) ? left_i : right_i);
        }
    }
    {
        {
            float left_d = best_d[0];
            float right_d = best_d[16];
            int left_i = best_i[0];
            int right_i = best_i[16];
            int swap = ((right_d < left_d) ? 1 : 0);
            best_d[0] = ((swap != 0) ? right_d : left_d);
            best_i[0] = ((swap != 0) ? right_i : left_i);
            best_d[16] = ((swap != 0) ? left_d : right_d);
            best_i[16] = ((swap != 0) ? left_i : right_i);
        }
    }
    {
        {
            float left_d = best_d[1];
            float right_d = best_d[17];
            int left_i = best_i[1];
            int right_i = best_i[17];
            int swap = ((right_d < left_d) ? 1 : 0);
            best_d[1] = ((swap != 0) ? right_d : left_d);
            best_i[1] = ((swap != 0) ? right_i : left_i);
            best_d[17] = ((swap != 0) ? left_d : right_d);
            best_i[17] = ((swap != 0) ? left_i : right_i);
        }
    }
    {
        {
            float left_d = best_d[2];
            float right_d = best_d[18];
            int left_i = best_i[2];
            int right_i = best_i[18];
            int swap = ((right_d < left_d) ? 1 : 0);
            best_d[2] = ((swap != 0) ? right_d : left_d);
            best_i[2] = ((swap != 0) ? right_i : left_i);
            best_d[18] = ((swap != 0) ? left_d : right_d);
            best_i[18] = ((swap != 0) ? left_i : right_i);
        }
    }
    {
        {
            float left_d = best_d[3];
            float right_d = best_d[19];
            int left_i = best_i[3];
            int right_i = best_i[19];
            int swap = ((right_d < left_d) ? 1 : 0);
            best_d[3] = ((swap != 0) ? right_d : left_d);
            best_i[3] = ((swap != 0) ? right_i : left_i);
            best_d[19] = ((swap != 0) ? left_d : right_d);
            best_i[19] = ((swap != 0) ? left_i : right_i);
        }
    }
    {
        {
            float left_d = best_d[4];
            float right_d = best_d[20];
            int left_i = best_i[4];
            int right_i = best_i[20];
            int swap = ((right_d < left_d) ? 1 : 0);
            best_d[4] = ((swap != 0) ? right_d : left_d);
            best_i[4] = ((swap != 0) ? right_i : left_i);
            best_d[20] = ((swap != 0) ? left_d : right_d);
            best_i[20] = ((swap != 0) ? left_i : right_i);
        }
    }
    {
        {
            float left_d = best_d[5];
            float right_d = best_d[21];
            int left_i = best_i[5];
            int right_i = best_i[21];
            int swap = ((right_d < left_d) ? 1 : 0);
            best_d[5] = ((swap != 0) ? right_d : left_d);
            best_i[5] = ((swap != 0) ? right_i : left_i);
            best_d[21] = ((swap != 0) ? left_d : right_d);
            best_i[21] = ((swap != 0) ? left_i : right_i);
        }
    }
    {
        {
            float left_d = best_d[6];
            float right_d = best_d[22];
            int left_i = best_i[6];
            int right_i = best_i[22];
            int swap = ((right_d < left_d) ? 1 : 0);
            best_d[6] = ((swap != 0) ? right_d : left_d);
            best_i[6] = ((swap != 0) ? right_i : left_i);
            best_d[22] = ((swap != 0) ? left_d : right_d);
            best_i[22] = ((swap != 0) ? left_i : right_i);
        }
    }
    {
        {
            float left_d = best_d[7];
            float right_d = best_d[23];
            int left_i = best_i[7];
            int right_i = best_i[23];
            int swap = ((right_d < left_d) ? 1 : 0);
            best_d[7] = ((swap != 0) ? right_d : left_d);
            best_i[7] = ((swap != 0) ? right_i : left_i);
            best_d[23] = ((swap != 0) ? left_d : right_d);
            best_i[23] = ((swap != 0) ? left_i : right_i);
        }
    }
    {
        {
            float left_d = best_d[8];
            float right_d = best_d[24];
            int left_i = best_i[8];
            int right_i = best_i[24];
            int swap = ((right_d < left_d) ? 1 : 0);
            best_d[8] = ((swap != 0) ? right_d : left_d);
            best_i[8] = ((swap != 0) ? right_i : left_i);
            best_d[24] = ((swap != 0) ? left_d : right_d);
            best_i[24] = ((swap != 0) ? left_i : right_i);
        }
    }
    {
        {
            float left_d = best_d[9];
            float right_d = best_d[25];
            int left_i = best_i[9];
            int right_i = best_i[25];
            int swap = ((right_d < left_d) ? 1 : 0);
            best_d[9] = ((swap != 0) ? right_d : left_d);
            best_i[9] = ((swap != 0) ? right_i : left_i);
            best_d[25] = ((swap != 0) ? left_d : right_d);
            best_i[25] = ((swap != 0) ? left_i : right_i);
        }
    }
    {
        {
            float left_d = best_d[10];
            float right_d = best_d[26];
            int left_i = best_i[10];
            int right_i = best_i[26];
            int swap = ((right_d < left_d) ? 1 : 0);
            best_d[10] = ((swap != 0) ? right_d : left_d);
            best_i[10] = ((swap != 0) ? right_i : left_i);
            best_d[26] = ((swap != 0) ? left_d : right_d);
            best_i[26] = ((swap != 0) ? left_i : right_i);
        }
    }
    {
        {
            float left_d = best_d[11];
            float right_d = best_d[27];
            int left_i = best_i[11];
            int right_i = best_i[27];
            int swap = ((right_d < left_d) ? 1 : 0);
            best_d[11] = ((swap != 0) ? right_d : left_d);
            best_i[11] = ((swap != 0) ? right_i : left_i);
            best_d[27] = ((swap != 0) ? left_d : right_d);
            best_i[27] = ((swap != 0) ? left_i : right_i);
        }
    }
    {
        {
            float left_d = best_d[12];
            float right_d = best_d[28];
            int left_i = best_i[12];
            int right_i = best_i[28];
            int swap = ((right_d < left_d) ? 1 : 0);
            best_d[12] = ((swap != 0) ? right_d : left_d);
            best_i[12] = ((swap != 0) ? right_i : left_i);
            best_d[28] = ((swap != 0) ? left_d : right_d);
            best_i[28] = ((swap != 0) ? left_i : right_i);
        }
    }
    {
        {
            float left_d = best_d[13];
            float right_d = best_d[29];
            int left_i = best_i[13];
            int right_i = best_i[29];
            int swap = ((right_d < left_d) ? 1 : 0);
            best_d[13] = ((swap != 0) ? right_d : left_d);
            best_i[13] = ((swap != 0) ? right_i : left_i);
            best_d[29] = ((swap != 0) ? left_d : right_d);
            best_i[29] = ((swap != 0) ? left_i : right_i);
        }
    }
    {
        {
            float left_d = best_d[14];
            float right_d = best_d[30];
            int left_i = best_i[14];
            int right_i = best_i[30];
            int swap = ((right_d < left_d) ? 1 : 0);
            best_d[14] = ((swap != 0) ? right_d : left_d);
            best_i[14] = ((swap != 0) ? right_i : left_i);
            best_d[30] = ((swap != 0) ? left_d : right_d);
            best_i[30] = ((swap != 0) ? left_i : right_i);
        }
    }
    {
        {
            float left_d = best_d[15];
            float right_d = best_d[31];
            int left_i = best_i[15];
            int right_i = best_i[31];
            int swap = ((right_d < left_d) ? 1 : 0);
            best_d[15] = ((swap != 0) ? right_d : left_d);
            best_i[15] = ((swap != 0) ? right_i : left_i);
            best_d[31] = ((swap != 0) ? left_d : right_d);
            best_i[31] = ((swap != 0) ? left_i : right_i);
        }
    }
    {
        {
            float left_d = best_d[32];
            float right_d = best_d[48];
            int left_i = best_i[32];
            int right_i = best_i[48];
            int swap = ((right_d < left_d) ? 1 : 0);
            best_d[32] = ((swap != 0) ? right_d : left_d);
            best_i[32] = ((swap != 0) ? right_i : left_i);
            best_d[48] = ((swap != 0) ? left_d : right_d);
            best_i[48] = ((swap != 0) ? left_i : right_i);
        }
    }
    {
        {
            float left_d = best_d[33];
            float right_d = best_d[49];
            int left_i = best_i[33];
            int right_i = best_i[49];
            int swap = ((right_d < left_d) ? 1 : 0);
            best_d[33] = ((swap != 0) ? right_d : left_d);
            best_i[33] = ((swap != 0) ? right_i : left_i);
            best_d[49] = ((swap != 0) ? left_d : right_d);
            best_i[49] = ((swap != 0) ? left_i : right_i);
        }
    }
    {
        {
            float left_d = best_d[34];
            float right_d = best_d[50];
            int left_i = best_i[34];
            int right_i = best_i[50];
            int swap = ((right_d < left_d) ? 1 : 0);
            best_d[34] = ((swap != 0) ? right_d : left_d);
            best_i[34] = ((swap != 0) ? right_i : left_i);
            best_d[50] = ((swap != 0) ? left_d : right_d);
            best_i[50] = ((swap != 0) ? left_i : right_i);
        }
    }
    {
        {
            float left_d = best_d[35];
            float right_d = best_d[51];
            int left_i = best_i[35];
            int right_i = best_i[51];
            int swap = ((right_d < left_d) ? 1 : 0);
            best_d[35] = ((swap != 0) ? right_d : left_d);
            best_i[35] = ((swap != 0) ? right_i : left_i);
            best_d[51] = ((swap != 0) ? left_d : right_d);
            best_i[51] = ((swap != 0) ? left_i : right_i);
        }
    }
    {
        {
            float left_d = best_d[36];
            float right_d = best_d[52];
            int left_i = best_i[36];
            int right_i = best_i[52];
            int swap = ((right_d < left_d) ? 1 : 0);
            best_d[36] = ((swap != 0) ? right_d : left_d);
            best_i[36] = ((swap != 0) ? right_i : left_i);
            best_d[52] = ((swap != 0) ? left_d : right_d);
            best_i[52] = ((swap != 0) ? left_i : right_i);
        }
    }
    {
        {
            float left_d = best_d[37];
            float right_d = best_d[53];
            int left_i = best_i[37];
            int right_i = best_i[53];
            int swap = ((right_d < left_d) ? 1 : 0);
            best_d[37] = ((swap != 0) ? right_d : left_d);
            best_i[37] = ((swap != 0) ? right_i : left_i);
            best_d[53] = ((swap != 0) ? left_d : right_d);
            best_i[53] = ((swap != 0) ? left_i : right_i);
        }
    }
    {
        {
            float left_d = best_d[38];
            float right_d = best_d[54];
            int left_i = best_i[38];
            int right_i = best_i[54];
            int swap = ((right_d < left_d) ? 1 : 0);
            best_d[38] = ((swap != 0) ? right_d : left_d);
            best_i[38] = ((swap != 0) ? right_i : left_i);
            best_d[54] = ((swap != 0) ? left_d : right_d);
            best_i[54] = ((swap != 0) ? left_i : right_i);
        }
    }
    {
        {
            float left_d = best_d[39];
            float right_d = best_d[55];
            int left_i = best_i[39];
            int right_i = best_i[55];
            int swap = ((right_d < left_d) ? 1 : 0);
            best_d[39] = ((swap != 0) ? right_d : left_d);
            best_i[39] = ((swap != 0) ? right_i : left_i);
            best_d[55] = ((swap != 0) ? left_d : right_d);
            best_i[55] = ((swap != 0) ? left_i : right_i);
        }
    }
    {
        {
            float left_d = best_d[40];
            float right_d = best_d[56];
            int left_i = best_i[40];
            int right_i = best_i[56];
            int swap = ((right_d < left_d) ? 1 : 0);
            best_d[40] = ((swap != 0) ? right_d : left_d);
            best_i[40] = ((swap != 0) ? right_i : left_i);
            best_d[56] = ((swap != 0) ? left_d : right_d);
            best_i[56] = ((swap != 0) ? left_i : right_i);
        }
    }
    {
        {
            float left_d = best_d[41];
            float right_d = best_d[57];
            int left_i = best_i[41];
            int right_i = best_i[57];
            int swap = ((right_d < left_d) ? 1 : 0);
            best_d[41] = ((swap != 0) ? right_d : left_d);
            best_i[41] = ((swap != 0) ? right_i : left_i);
            best_d[57] = ((swap != 0) ? left_d : right_d);
            best_i[57] = ((swap != 0) ? left_i : right_i);
        }
    }
    {
        {
            float left_d = best_d[42];
            float right_d = best_d[58];
            int left_i = best_i[42];
            int right_i = best_i[58];
            int swap = ((right_d < left_d) ? 1 : 0);
            best_d[42] = ((swap != 0) ? right_d : left_d);
            best_i[42] = ((swap != 0) ? right_i : left_i);
            best_d[58] = ((swap != 0) ? left_d : right_d);
            best_i[58] = ((swap != 0) ? left_i : right_i);
        }
    }
    {
        {
            float left_d = best_d[43];
            float right_d = best_d[59];
            int left_i = best_i[43];
            int right_i = best_i[59];
            int swap = ((right_d < left_d) ? 1 : 0);
            best_d[43] = ((swap != 0) ? right_d : left_d);
            best_i[43] = ((swap != 0) ? right_i : left_i);
            best_d[59] = ((swap != 0) ? left_d : right_d);
            best_i[59] = ((swap != 0) ? left_i : right_i);
        }
    }
    {
        {
            float left_d = best_d[44];
            float right_d = best_d[60];
            int left_i = best_i[44];
            int right_i = best_i[60];
            int swap = ((right_d < left_d) ? 1 : 0);
            best_d[44] = ((swap != 0) ? right_d : left_d);
            best_i[44] = ((swap != 0) ? right_i : left_i);
            best_d[60] = ((swap != 0) ? left_d : right_d);
            best_i[60] = ((swap != 0) ? left_i : right_i);
        }
    }
    {
        {
            float left_d = best_d[45];
            float right_d = best_d[61];
            int left_i = best_i[45];
            int right_i = best_i[61];
            int swap = ((right_d < left_d) ? 1 : 0);
            best_d[45] = ((swap != 0) ? right_d : left_d);
            best_i[45] = ((swap != 0) ? right_i : left_i);
            best_d[61] = ((swap != 0) ? left_d : right_d);
            best_i[61] = ((swap != 0) ? left_i : right_i);
        }
    }
    {
        {
            float left_d = best_d[46];
            float right_d = best_d[62];
            int left_i = best_i[46];
            int right_i = best_i[62];
            int swap = ((right_d < left_d) ? 1 : 0);
            best_d[46] = ((swap != 0) ? right_d : left_d);
            best_i[46] = ((swap != 0) ? right_i : left_i);
            best_d[62] = ((swap != 0) ? left_d : right_d);
            best_i[62] = ((swap != 0) ? left_i : right_i);
        }
    }
    {
        {
            float left_d = best_d[47];
            float right_d = best_d[63];
            int left_i = best_i[47];
            int right_i = best_i[63];
            int swap = ((right_d < left_d) ? 1 : 0);
            best_d[47] = ((swap != 0) ? right_d : left_d);
            best_i[47] = ((swap != 0) ? right_i : left_i);
            best_d[63] = ((swap != 0) ? left_d : right_d);
            best_i[63] = ((swap != 0) ? left_i : right_i);
        }
    }
    {
        {
            float left_d = best_d[0];
            float right_d = best_d[8];
            int left_i = best_i[0];
            int right_i = best_i[8];
            int swap = ((right_d < left_d) ? 1 : 0);
            best_d[0] = ((swap != 0) ? right_d : left_d);
            best_i[0] = ((swap != 0) ? right_i : left_i);
            best_d[8] = ((swap != 0) ? left_d : right_d);
            best_i[8] = ((swap != 0) ? left_i : right_i);
        }
    }
    {
        {
            float left_d = best_d[1];
            float right_d = best_d[9];
            int left_i = best_i[1];
            int right_i = best_i[9];
            int swap = ((right_d < left_d) ? 1 : 0);
            best_d[1] = ((swap != 0) ? right_d : left_d);
            best_i[1] = ((swap != 0) ? right_i : left_i);
            best_d[9] = ((swap != 0) ? left_d : right_d);
            best_i[9] = ((swap != 0) ? left_i : right_i);
        }
    }
    {
        {
            float left_d = best_d[2];
            float right_d = best_d[10];
            int left_i = best_i[2];
            int right_i = best_i[10];
            int swap = ((right_d < left_d) ? 1 : 0);
            best_d[2] = ((swap != 0) ? right_d : left_d);
            best_i[2] = ((swap != 0) ? right_i : left_i);
            best_d[10] = ((swap != 0) ? left_d : right_d);
            best_i[10] = ((swap != 0) ? left_i : right_i);
        }
    }
    {
        {
            float left_d = best_d[3];
            float right_d = best_d[11];
            int left_i = best_i[3];
            int right_i = best_i[11];
            int swap = ((right_d < left_d) ? 1 : 0);
            best_d[3] = ((swap != 0) ? right_d : left_d);
            best_i[3] = ((swap != 0) ? right_i : left_i);
            best_d[11] = ((swap != 0) ? left_d : right_d);
            best_i[11] = ((swap != 0) ? left_i : right_i);
        }
    }
    {
        {
            float left_d = best_d[4];
            float right_d = best_d[12];
            int left_i = best_i[4];
            int right_i = best_i[12];
            int swap = ((right_d < left_d) ? 1 : 0);
            best_d[4] = ((swap != 0) ? right_d : left_d);
            best_i[4] = ((swap != 0) ? right_i : left_i);
            best_d[12] = ((swap != 0) ? left_d : right_d);
            best_i[12] = ((swap != 0) ? left_i : right_i);
        }
    }
    {
        {
            float left_d = best_d[5];
            float right_d = best_d[13];
            int left_i = best_i[5];
            int right_i = best_i[13];
            int swap = ((right_d < left_d) ? 1 : 0);
            best_d[5] = ((swap != 0) ? right_d : left_d);
            best_i[5] = ((swap != 0) ? right_i : left_i);
            best_d[13] = ((swap != 0) ? left_d : right_d);
            best_i[13] = ((swap != 0) ? left_i : right_i);
        }
    }
    {
        {
            float left_d = best_d[6];
            float right_d = best_d[14];
            int left_i = best_i[6];
            int right_i = best_i[14];
            int swap = ((right_d < left_d) ? 1 : 0);
            best_d[6] = ((swap != 0) ? right_d : left_d);
            best_i[6] = ((swap != 0) ? right_i : left_i);
            best_d[14] = ((swap != 0) ? left_d : right_d);
            best_i[14] = ((swap != 0) ? left_i : right_i);
        }
    }
    {
        {
            float left_d = best_d[7];
            float right_d = best_d[15];
            int left_i = best_i[7];
            int right_i = best_i[15];
            int swap = ((right_d < left_d) ? 1 : 0);
            best_d[7] = ((swap != 0) ? right_d : left_d);
            best_i[7] = ((swap != 0) ? right_i : left_i);
            best_d[15] = ((swap != 0) ? left_d : right_d);
            best_i[15] = ((swap != 0) ? left_i : right_i);
        }
    }
    {
        {
            float left_d = best_d[16];
            float right_d = best_d[24];
            int left_i = best_i[16];
            int right_i = best_i[24];
            int swap = ((right_d < left_d) ? 1 : 0);
            best_d[16] = ((swap != 0) ? right_d : left_d);
            best_i[16] = ((swap != 0) ? right_i : left_i);
            best_d[24] = ((swap != 0) ? left_d : right_d);
            best_i[24] = ((swap != 0) ? left_i : right_i);
        }
    }
    {
        {
            float left_d = best_d[17];
            float right_d = best_d[25];
            int left_i = best_i[17];
            int right_i = best_i[25];
            int swap = ((right_d < left_d) ? 1 : 0);
            best_d[17] = ((swap != 0) ? right_d : left_d);
            best_i[17] = ((swap != 0) ? right_i : left_i);
            best_d[25] = ((swap != 0) ? left_d : right_d);
            best_i[25] = ((swap != 0) ? left_i : right_i);
        }
    }
    {
        {
            float left_d = best_d[18];
            float right_d = best_d[26];
            int left_i = best_i[18];
            int right_i = best_i[26];
            int swap = ((right_d < left_d) ? 1 : 0);
            best_d[18] = ((swap != 0) ? right_d : left_d);
            best_i[18] = ((swap != 0) ? right_i : left_i);
            best_d[26] = ((swap != 0) ? left_d : right_d);
            best_i[26] = ((swap != 0) ? left_i : right_i);
        }
    }
    {
        {
            float left_d = best_d[19];
            float right_d = best_d[27];
            int left_i = best_i[19];
            int right_i = best_i[27];
            int swap = ((right_d < left_d) ? 1 : 0);
            best_d[19] = ((swap != 0) ? right_d : left_d);
            best_i[19] = ((swap != 0) ? right_i : left_i);
            best_d[27] = ((swap != 0) ? left_d : right_d);
            best_i[27] = ((swap != 0) ? left_i : right_i);
        }
    }
    {
        {
            float left_d = best_d[20];
            float right_d = best_d[28];
            int left_i = best_i[20];
            int right_i = best_i[28];
            int swap = ((right_d < left_d) ? 1 : 0);
            best_d[20] = ((swap != 0) ? right_d : left_d);
            best_i[20] = ((swap != 0) ? right_i : left_i);
            best_d[28] = ((swap != 0) ? left_d : right_d);
            best_i[28] = ((swap != 0) ? left_i : right_i);
        }
    }
    {
        {
            float left_d = best_d[21];
            float right_d = best_d[29];
            int left_i = best_i[21];
            int right_i = best_i[29];
            int swap = ((right_d < left_d) ? 1 : 0);
            best_d[21] = ((swap != 0) ? right_d : left_d);
            best_i[21] = ((swap != 0) ? right_i : left_i);
            best_d[29] = ((swap != 0) ? left_d : right_d);
            best_i[29] = ((swap != 0) ? left_i : right_i);
        }
    }
    {
        {
            float left_d = best_d[22];
            float right_d = best_d[30];
            int left_i = best_i[22];
            int right_i = best_i[30];
            int swap = ((right_d < left_d) ? 1 : 0);
            best_d[22] = ((swap != 0) ? right_d : left_d);
            best_i[22] = ((swap != 0) ? right_i : left_i);
            best_d[30] = ((swap != 0) ? left_d : right_d);
            best_i[30] = ((swap != 0) ? left_i : right_i);
        }
    }
    {
        {
            float left_d = best_d[23];
            float right_d = best_d[31];
            int left_i = best_i[23];
            int right_i = best_i[31];
            int swap = ((right_d < left_d) ? 1 : 0);
            best_d[23] = ((swap != 0) ? right_d : left_d);
            best_i[23] = ((swap != 0) ? right_i : left_i);
            best_d[31] = ((swap != 0) ? left_d : right_d);
            best_i[31] = ((swap != 0) ? left_i : right_i);
        }
    }
    {
        {
            float left_d = best_d[32];
            float right_d = best_d[40];
            int left_i = best_i[32];
            int right_i = best_i[40];
            int swap = ((right_d < left_d) ? 1 : 0);
            best_d[32] = ((swap != 0) ? right_d : left_d);
            best_i[32] = ((swap != 0) ? right_i : left_i);
            best_d[40] = ((swap != 0) ? left_d : right_d);
            best_i[40] = ((swap != 0) ? left_i : right_i);
        }
    }
    {
        {
            float left_d = best_d[33];
            float right_d = best_d[41];
            int left_i = best_i[33];
            int right_i = best_i[41];
            int swap = ((right_d < left_d) ? 1 : 0);
            best_d[33] = ((swap != 0) ? right_d : left_d);
            best_i[33] = ((swap != 0) ? right_i : left_i);
            best_d[41] = ((swap != 0) ? left_d : right_d);
            best_i[41] = ((swap != 0) ? left_i : right_i);
        }
    }
    {
        {
            float left_d = best_d[34];
            float right_d = best_d[42];
            int left_i = best_i[34];
            int right_i = best_i[42];
            int swap = ((right_d < left_d) ? 1 : 0);
            best_d[34] = ((swap != 0) ? right_d : left_d);
            best_i[34] = ((swap != 0) ? right_i : left_i);
            best_d[42] = ((swap != 0) ? left_d : right_d);
            best_i[42] = ((swap != 0) ? left_i : right_i);
        }
    }
    {
        {
            float left_d = best_d[35];
            float right_d = best_d[43];
            int left_i = best_i[35];
            int right_i = best_i[43];
            int swap = ((right_d < left_d) ? 1 : 0);
            best_d[35] = ((swap != 0) ? right_d : left_d);
            best_i[35] = ((swap != 0) ? right_i : left_i);
            best_d[43] = ((swap != 0) ? left_d : right_d);
            best_i[43] = ((swap != 0) ? left_i : right_i);
        }
    }
    {
        {
            float left_d = best_d[36];
            float right_d = best_d[44];
            int left_i = best_i[36];
            int right_i = best_i[44];
            int swap = ((right_d < left_d) ? 1 : 0);
            best_d[36] = ((swap != 0) ? right_d : left_d);
            best_i[36] = ((swap != 0) ? right_i : left_i);
            best_d[44] = ((swap != 0) ? left_d : right_d);
            best_i[44] = ((swap != 0) ? left_i : right_i);
        }
    }
    {
        {
            float left_d = best_d[37];
            float right_d = best_d[45];
            int left_i = best_i[37];
            int right_i = best_i[45];
            int swap = ((right_d < left_d) ? 1 : 0);
            best_d[37] = ((swap != 0) ? right_d : left_d);
            best_i[37] = ((swap != 0) ? right_i : left_i);
            best_d[45] = ((swap != 0) ? left_d : right_d);
            best_i[45] = ((swap != 0) ? left_i : right_i);
        }
    }
    {
        {
            float left_d = best_d[38];
            float right_d = best_d[46];
            int left_i = best_i[38];
            int right_i = best_i[46];
            int swap = ((right_d < left_d) ? 1 : 0);
            best_d[38] = ((swap != 0) ? right_d : left_d);
            best_i[38] = ((swap != 0) ? right_i : left_i);
            best_d[46] = ((swap != 0) ? left_d : right_d);
            best_i[46] = ((swap != 0) ? left_i : right_i);
        }
    }
    {
        {
            float left_d = best_d[39];
            float right_d = best_d[47];
            int left_i = best_i[39];
            int right_i = best_i[47];
            int swap = ((right_d < left_d) ? 1 : 0);
            best_d[39] = ((swap != 0) ? right_d : left_d);
            best_i[39] = ((swap != 0) ? right_i : left_i);
            best_d[47] = ((swap != 0) ? left_d : right_d);
            best_i[47] = ((swap != 0) ? left_i : right_i);
        }
    }
    {
        {
            float left_d = best_d[48];
            float right_d = best_d[56];
            int left_i = best_i[48];
            int right_i = best_i[56];
            int swap = ((right_d < left_d) ? 1 : 0);
            best_d[48] = ((swap != 0) ? right_d : left_d);
            best_i[48] = ((swap != 0) ? right_i : left_i);
            best_d[56] = ((swap != 0) ? left_d : right_d);
            best_i[56] = ((swap != 0) ? left_i : right_i);
        }
    }
    {
        {
            float left_d = best_d[49];
            float right_d = best_d[57];
            int left_i = best_i[49];
            int right_i = best_i[57];
            int swap = ((right_d < left_d) ? 1 : 0);
            best_d[49] = ((swap != 0) ? right_d : left_d);
            best_i[49] = ((swap != 0) ? right_i : left_i);
            best_d[57] = ((swap != 0) ? left_d : right_d);
            best_i[57] = ((swap != 0) ? left_i : right_i);
        }
    }
    {
        {
            float left_d = best_d[50];
            float right_d = best_d[58];
            int left_i = best_i[50];
            int right_i = best_i[58];
            int swap = ((right_d < left_d) ? 1 : 0);
            best_d[50] = ((swap != 0) ? right_d : left_d);
            best_i[50] = ((swap != 0) ? right_i : left_i);
            best_d[58] = ((swap != 0) ? left_d : right_d);
            best_i[58] = ((swap != 0) ? left_i : right_i);
        }
    }
    {
        {
            float left_d = best_d[51];
            float right_d = best_d[59];
            int left_i = best_i[51];
            int right_i = best_i[59];
            int swap = ((right_d < left_d) ? 1 : 0);
            best_d[51] = ((swap != 0) ? right_d : left_d);
            best_i[51] = ((swap != 0) ? right_i : left_i);
            best_d[59] = ((swap != 0) ? left_d : right_d);
            best_i[59] = ((swap != 0) ? left_i : right_i);
        }
    }
    {
        {
            float left_d = best_d[52];
            float right_d = best_d[60];
            int left_i = best_i[52];
            int right_i = best_i[60];
            int swap = ((right_d < left_d) ? 1 : 0);
            best_d[52] = ((swap != 0) ? right_d : left_d);
            best_i[52] = ((swap != 0) ? right_i : left_i);
            best_d[60] = ((swap != 0) ? left_d : right_d);
            best_i[60] = ((swap != 0) ? left_i : right_i);
        }
    }
    {
        {
            float left_d = best_d[53];
            float right_d = best_d[61];
            int left_i = best_i[53];
            int right_i = best_i[61];
            int swap = ((right_d < left_d) ? 1 : 0);
            best_d[53] = ((swap != 0) ? right_d : left_d);
            best_i[53] = ((swap != 0) ? right_i : left_i);
            best_d[61] = ((swap != 0) ? left_d : right_d);
            best_i[61] = ((swap != 0) ? left_i : right_i);
        }
    }
    {
        {
            float left_d = best_d[54];
            float right_d = best_d[62];
            int left_i = best_i[54];
            int right_i = best_i[62];
            int swap = ((right_d < left_d) ? 1 : 0);
            best_d[54] = ((swap != 0) ? right_d : left_d);
            best_i[54] = ((swap != 0) ? right_i : left_i);
            best_d[62] = ((swap != 0) ? left_d : right_d);
            best_i[62] = ((swap != 0) ? left_i : right_i);
        }
    }
    {
        {
            float left_d = best_d[55];
            float right_d = best_d[63];
            int left_i = best_i[55];
            int right_i = best_i[63];
            int swap = ((right_d < left_d) ? 1 : 0);
            best_d[55] = ((swap != 0) ? right_d : left_d);
            best_i[55] = ((swap != 0) ? right_i : left_i);
            best_d[63] = ((swap != 0) ? left_d : right_d);
            best_i[63] = ((swap != 0) ? left_i : right_i);
        }
    }
    {
        {
            float left_d = best_d[0];
            float right_d = best_d[4];
            int left_i = best_i[0];
            int right_i = best_i[4];
            int swap = ((right_d < left_d) ? 1 : 0);
            best_d[0] = ((swap != 0) ? right_d : left_d);
            best_i[0] = ((swap != 0) ? right_i : left_i);
            best_d[4] = ((swap != 0) ? left_d : right_d);
            best_i[4] = ((swap != 0) ? left_i : right_i);
        }
    }
    {
        {
            float left_d = best_d[1];
            float right_d = best_d[5];
            int left_i = best_i[1];
            int right_i = best_i[5];
            int swap = ((right_d < left_d) ? 1 : 0);
            best_d[1] = ((swap != 0) ? right_d : left_d);
            best_i[1] = ((swap != 0) ? right_i : left_i);
            best_d[5] = ((swap != 0) ? left_d : right_d);
            best_i[5] = ((swap != 0) ? left_i : right_i);
        }
    }
    {
        {
            float left_d = best_d[2];
            float right_d = best_d[6];
            int left_i = best_i[2];
            int right_i = best_i[6];
            int swap = ((right_d < left_d) ? 1 : 0);
            best_d[2] = ((swap != 0) ? right_d : left_d);
            best_i[2] = ((swap != 0) ? right_i : left_i);
            best_d[6] = ((swap != 0) ? left_d : right_d);
            best_i[6] = ((swap != 0) ? left_i : right_i);
        }
    }
    {
        {
            float left_d = best_d[3];
            float right_d = best_d[7];
            int left_i = best_i[3];
            int right_i = best_i[7];
            int swap = ((right_d < left_d) ? 1 : 0);
            best_d[3] = ((swap != 0) ? right_d : left_d);
            best_i[3] = ((swap != 0) ? right_i : left_i);
            best_d[7] = ((swap != 0) ? left_d : right_d);
            best_i[7] = ((swap != 0) ? left_i : right_i);
        }
    }
    {
        {
            float left_d = best_d[8];
            float right_d = best_d[12];
            int left_i = best_i[8];
            int right_i = best_i[12];
            int swap = ((right_d < left_d) ? 1 : 0);
            best_d[8] = ((swap != 0) ? right_d : left_d);
            best_i[8] = ((swap != 0) ? right_i : left_i);
            best_d[12] = ((swap != 0) ? left_d : right_d);
            best_i[12] = ((swap != 0) ? left_i : right_i);
        }
    }
    {
        {
            float left_d = best_d[9];
            float right_d = best_d[13];
            int left_i = best_i[9];
            int right_i = best_i[13];
            int swap = ((right_d < left_d) ? 1 : 0);
            best_d[9] = ((swap != 0) ? right_d : left_d);
            best_i[9] = ((swap != 0) ? right_i : left_i);
            best_d[13] = ((swap != 0) ? left_d : right_d);
            best_i[13] = ((swap != 0) ? left_i : right_i);
        }
    }
    {
        {
            float left_d = best_d[10];
            float right_d = best_d[14];
            int left_i = best_i[10];
            int right_i = best_i[14];
            int swap = ((right_d < left_d) ? 1 : 0);
            best_d[10] = ((swap != 0) ? right_d : left_d);
            best_i[10] = ((swap != 0) ? right_i : left_i);
            best_d[14] = ((swap != 0) ? left_d : right_d);
            best_i[14] = ((swap != 0) ? left_i : right_i);
        }
    }
    {
        {
            float left_d = best_d[11];
            float right_d = best_d[15];
            int left_i = best_i[11];
            int right_i = best_i[15];
            int swap = ((right_d < left_d) ? 1 : 0);
            best_d[11] = ((swap != 0) ? right_d : left_d);
            best_i[11] = ((swap != 0) ? right_i : left_i);
            best_d[15] = ((swap != 0) ? left_d : right_d);
            best_i[15] = ((swap != 0) ? left_i : right_i);
        }
    }
    {
        {
            float left_d = best_d[16];
            float right_d = best_d[20];
            int left_i = best_i[16];
            int right_i = best_i[20];
            int swap = ((right_d < left_d) ? 1 : 0);
            best_d[16] = ((swap != 0) ? right_d : left_d);
            best_i[16] = ((swap != 0) ? right_i : left_i);
            best_d[20] = ((swap != 0) ? left_d : right_d);
            best_i[20] = ((swap != 0) ? left_i : right_i);
        }
    }
    {
        {
            float left_d = best_d[17];
            float right_d = best_d[21];
            int left_i = best_i[17];
            int right_i = best_i[21];
            int swap = ((right_d < left_d) ? 1 : 0);
            best_d[17] = ((swap != 0) ? right_d : left_d);
            best_i[17] = ((swap != 0) ? right_i : left_i);
            best_d[21] = ((swap != 0) ? left_d : right_d);
            best_i[21] = ((swap != 0) ? left_i : right_i);
        }
    }
    {
        {
            float left_d = best_d[18];
            float right_d = best_d[22];
            int left_i = best_i[18];
            int right_i = best_i[22];
            int swap = ((right_d < left_d) ? 1 : 0);
            best_d[18] = ((swap != 0) ? right_d : left_d);
            best_i[18] = ((swap != 0) ? right_i : left_i);
            best_d[22] = ((swap != 0) ? left_d : right_d);
            best_i[22] = ((swap != 0) ? left_i : right_i);
        }
    }
    {
        {
            float left_d = best_d[19];
            float right_d = best_d[23];
            int left_i = best_i[19];
            int right_i = best_i[23];
            int swap = ((right_d < left_d) ? 1 : 0);
            best_d[19] = ((swap != 0) ? right_d : left_d);
            best_i[19] = ((swap != 0) ? right_i : left_i);
            best_d[23] = ((swap != 0) ? left_d : right_d);
            best_i[23] = ((swap != 0) ? left_i : right_i);
        }
    }
    {
        {
            float left_d = best_d[24];
            float right_d = best_d[28];
            int left_i = best_i[24];
            int right_i = best_i[28];
            int swap = ((right_d < left_d) ? 1 : 0);
            best_d[24] = ((swap != 0) ? right_d : left_d);
            best_i[24] = ((swap != 0) ? right_i : left_i);
            best_d[28] = ((swap != 0) ? left_d : right_d);
            best_i[28] = ((swap != 0) ? left_i : right_i);
        }
    }
    {
        {
            float left_d = best_d[25];
            float right_d = best_d[29];
            int left_i = best_i[25];
            int right_i = best_i[29];
            int swap = ((right_d < left_d) ? 1 : 0);
            best_d[25] = ((swap != 0) ? right_d : left_d);
            best_i[25] = ((swap != 0) ? right_i : left_i);
            best_d[29] = ((swap != 0) ? left_d : right_d);
            best_i[29] = ((swap != 0) ? left_i : right_i);
        }
    }
    {
        {
            float left_d = best_d[26];
            float right_d = best_d[30];
            int left_i = best_i[26];
            int right_i = best_i[30];
            int swap = ((right_d < left_d) ? 1 : 0);
            best_d[26] = ((swap != 0) ? right_d : left_d);
            best_i[26] = ((swap != 0) ? right_i : left_i);
            best_d[30] = ((swap != 0) ? left_d : right_d);
            best_i[30] = ((swap != 0) ? left_i : right_i);
        }
    }
    {
        {
            float left_d = best_d[27];
            float right_d = best_d[31];
            int left_i = best_i[27];
            int right_i = best_i[31];
            int swap = ((right_d < left_d) ? 1 : 0);
            best_d[27] = ((swap != 0) ? right_d : left_d);
            best_i[27] = ((swap != 0) ? right_i : left_i);
            best_d[31] = ((swap != 0) ? left_d : right_d);
            best_i[31] = ((swap != 0) ? left_i : right_i);
        }
    }
    {
        {
            float left_d = best_d[32];
            float right_d = best_d[36];
            int left_i = best_i[32];
            int right_i = best_i[36];
            int swap = ((right_d < left_d) ? 1 : 0);
            best_d[32] = ((swap != 0) ? right_d : left_d);
            best_i[32] = ((swap != 0) ? right_i : left_i);
            best_d[36] = ((swap != 0) ? left_d : right_d);
            best_i[36] = ((swap != 0) ? left_i : right_i);
        }
    }
    {
        {
            float left_d = best_d[33];
            float right_d = best_d[37];
            int left_i = best_i[33];
            int right_i = best_i[37];
            int swap = ((right_d < left_d) ? 1 : 0);
            best_d[33] = ((swap != 0) ? right_d : left_d);
            best_i[33] = ((swap != 0) ? right_i : left_i);
            best_d[37] = ((swap != 0) ? left_d : right_d);
            best_i[37] = ((swap != 0) ? left_i : right_i);
        }
    }
    {
        {
            float left_d = best_d[34];
            float right_d = best_d[38];
            int left_i = best_i[34];
            int right_i = best_i[38];
            int swap = ((right_d < left_d) ? 1 : 0);
            best_d[34] = ((swap != 0) ? right_d : left_d);
            best_i[34] = ((swap != 0) ? right_i : left_i);
            best_d[38] = ((swap != 0) ? left_d : right_d);
            best_i[38] = ((swap != 0) ? left_i : right_i);
        }
    }
    {
        {
            float left_d = best_d[35];
            float right_d = best_d[39];
            int left_i = best_i[35];
            int right_i = best_i[39];
            int swap = ((right_d < left_d) ? 1 : 0);
            best_d[35] = ((swap != 0) ? right_d : left_d);
            best_i[35] = ((swap != 0) ? right_i : left_i);
            best_d[39] = ((swap != 0) ? left_d : right_d);
            best_i[39] = ((swap != 0) ? left_i : right_i);
        }
    }
    {
        {
            float left_d = best_d[40];
            float right_d = best_d[44];
            int left_i = best_i[40];
            int right_i = best_i[44];
            int swap = ((right_d < left_d) ? 1 : 0);
            best_d[40] = ((swap != 0) ? right_d : left_d);
            best_i[40] = ((swap != 0) ? right_i : left_i);
            best_d[44] = ((swap != 0) ? left_d : right_d);
            best_i[44] = ((swap != 0) ? left_i : right_i);
        }
    }
    {
        {
            float left_d = best_d[41];
            float right_d = best_d[45];
            int left_i = best_i[41];
            int right_i = best_i[45];
            int swap = ((right_d < left_d) ? 1 : 0);
            best_d[41] = ((swap != 0) ? right_d : left_d);
            best_i[41] = ((swap != 0) ? right_i : left_i);
            best_d[45] = ((swap != 0) ? left_d : right_d);
            best_i[45] = ((swap != 0) ? left_i : right_i);
        }
    }
    {
        {
            float left_d = best_d[42];
            float right_d = best_d[46];
            int left_i = best_i[42];
            int right_i = best_i[46];
            int swap = ((right_d < left_d) ? 1 : 0);
            best_d[42] = ((swap != 0) ? right_d : left_d);
            best_i[42] = ((swap != 0) ? right_i : left_i);
            best_d[46] = ((swap != 0) ? left_d : right_d);
            best_i[46] = ((swap != 0) ? left_i : right_i);
        }
    }
    {
        {
            float left_d = best_d[43];
            float right_d = best_d[47];
            int left_i = best_i[43];
            int right_i = best_i[47];
            int swap = ((right_d < left_d) ? 1 : 0);
            best_d[43] = ((swap != 0) ? right_d : left_d);
            best_i[43] = ((swap != 0) ? right_i : left_i);
            best_d[47] = ((swap != 0) ? left_d : right_d);
            best_i[47] = ((swap != 0) ? left_i : right_i);
        }
    }
    {
        {
            float left_d = best_d[48];
            float right_d = best_d[52];
            int left_i = best_i[48];
            int right_i = best_i[52];
            int swap = ((right_d < left_d) ? 1 : 0);
            best_d[48] = ((swap != 0) ? right_d : left_d);
            best_i[48] = ((swap != 0) ? right_i : left_i);
            best_d[52] = ((swap != 0) ? left_d : right_d);
            best_i[52] = ((swap != 0) ? left_i : right_i);
        }
    }
    {
        {
            float left_d = best_d[49];
            float right_d = best_d[53];
            int left_i = best_i[49];
            int right_i = best_i[53];
            int swap = ((right_d < left_d) ? 1 : 0);
            best_d[49] = ((swap != 0) ? right_d : left_d);
            best_i[49] = ((swap != 0) ? right_i : left_i);
            best_d[53] = ((swap != 0) ? left_d : right_d);
            best_i[53] = ((swap != 0) ? left_i : right_i);
        }
    }
    {
        {
            float left_d = best_d[50];
            float right_d = best_d[54];
            int left_i = best_i[50];
            int right_i = best_i[54];
            int swap = ((right_d < left_d) ? 1 : 0);
            best_d[50] = ((swap != 0) ? right_d : left_d);
            best_i[50] = ((swap != 0) ? right_i : left_i);
            best_d[54] = ((swap != 0) ? left_d : right_d);
            best_i[54] = ((swap != 0) ? left_i : right_i);
        }
    }
    {
        {
            float left_d = best_d[51];
            float right_d = best_d[55];
            int left_i = best_i[51];
            int right_i = best_i[55];
            int swap = ((right_d < left_d) ? 1 : 0);
            best_d[51] = ((swap != 0) ? right_d : left_d);
            best_i[51] = ((swap != 0) ? right_i : left_i);
            best_d[55] = ((swap != 0) ? left_d : right_d);
            best_i[55] = ((swap != 0) ? left_i : right_i);
        }
    }
    {
        {
            float left_d = best_d[56];
            float right_d = best_d[60];
            int left_i = best_i[56];
            int right_i = best_i[60];
            int swap = ((right_d < left_d) ? 1 : 0);
            best_d[56] = ((swap != 0) ? right_d : left_d);
            best_i[56] = ((swap != 0) ? right_i : left_i);
            best_d[60] = ((swap != 0) ? left_d : right_d);
            best_i[60] = ((swap != 0) ? left_i : right_i);
        }
    }
    {
        {
            float left_d = best_d[57];
            float right_d = best_d[61];
            int left_i = best_i[57];
            int right_i = best_i[61];
            int swap = ((right_d < left_d) ? 1 : 0);
            best_d[57] = ((swap != 0) ? right_d : left_d);
            best_i[57] = ((swap != 0) ? right_i : left_i);
            best_d[61] = ((swap != 0) ? left_d : right_d);
            best_i[61] = ((swap != 0) ? left_i : right_i);
        }
    }
    {
        {
            float left_d = best_d[58];
            float right_d = best_d[62];
            int left_i = best_i[58];
            int right_i = best_i[62];
            int swap = ((right_d < left_d) ? 1 : 0);
            best_d[58] = ((swap != 0) ? right_d : left_d);
            best_i[58] = ((swap != 0) ? right_i : left_i);
            best_d[62] = ((swap != 0) ? left_d : right_d);
            best_i[62] = ((swap != 0) ? left_i : right_i);
        }
    }
    {
        {
            float left_d = best_d[59];
            float right_d = best_d[63];
            int left_i = best_i[59];
            int right_i = best_i[63];
            int swap = ((right_d < left_d) ? 1 : 0);
            best_d[59] = ((swap != 0) ? right_d : left_d);
            best_i[59] = ((swap != 0) ? right_i : left_i);
            best_d[63] = ((swap != 0) ? left_d : right_d);
            best_i[63] = ((swap != 0) ? left_i : right_i);
        }
    }
    {
        {
            float left_d = best_d[0];
            float right_d = best_d[2];
            int left_i = best_i[0];
            int right_i = best_i[2];
            int swap = ((right_d < left_d) ? 1 : 0);
            best_d[0] = ((swap != 0) ? right_d : left_d);
            best_i[0] = ((swap != 0) ? right_i : left_i);
            best_d[2] = ((swap != 0) ? left_d : right_d);
            best_i[2] = ((swap != 0) ? left_i : right_i);
        }
    }
    {
        {
            float left_d = best_d[1];
            float right_d = best_d[3];
            int left_i = best_i[1];
            int right_i = best_i[3];
            int swap = ((right_d < left_d) ? 1 : 0);
            best_d[1] = ((swap != 0) ? right_d : left_d);
            best_i[1] = ((swap != 0) ? right_i : left_i);
            best_d[3] = ((swap != 0) ? left_d : right_d);
            best_i[3] = ((swap != 0) ? left_i : right_i);
        }
    }
    {
        {
            float left_d = best_d[4];
            float right_d = best_d[6];
            int left_i = best_i[4];
            int right_i = best_i[6];
            int swap = ((right_d < left_d) ? 1 : 0);
            best_d[4] = ((swap != 0) ? right_d : left_d);
            best_i[4] = ((swap != 0) ? right_i : left_i);
            best_d[6] = ((swap != 0) ? left_d : right_d);
            best_i[6] = ((swap != 0) ? left_i : right_i);
        }
    }
    {
        {
            float left_d = best_d[5];
            float right_d = best_d[7];
            int left_i = best_i[5];
            int right_i = best_i[7];
            int swap = ((right_d < left_d) ? 1 : 0);
            best_d[5] = ((swap != 0) ? right_d : left_d);
            best_i[5] = ((swap != 0) ? right_i : left_i);
            best_d[7] = ((swap != 0) ? left_d : right_d);
            best_i[7] = ((swap != 0) ? left_i : right_i);
        }
    }
    {
        {
            float left_d = best_d[8];
            float right_d = best_d[10];
            int left_i = best_i[8];
            int right_i = best_i[10];
            int swap = ((right_d < left_d) ? 1 : 0);
            best_d[8] = ((swap != 0) ? right_d : left_d);
            best_i[8] = ((swap != 0) ? right_i : left_i);
            best_d[10] = ((swap != 0) ? left_d : right_d);
            best_i[10] = ((swap != 0) ? left_i : right_i);
        }
    }
    {
        {
            float left_d = best_d[9];
            float right_d = best_d[11];
            int left_i = best_i[9];
            int right_i = best_i[11];
            int swap = ((right_d < left_d) ? 1 : 0);
            best_d[9] = ((swap != 0) ? right_d : left_d);
            best_i[9] = ((swap != 0) ? right_i : left_i);
            best_d[11] = ((swap != 0) ? left_d : right_d);
            best_i[11] = ((swap != 0) ? left_i : right_i);
        }
    }
    {
        {
            float left_d = best_d[12];
            float right_d = best_d[14];
            int left_i = best_i[12];
            int right_i = best_i[14];
            int swap = ((right_d < left_d) ? 1 : 0);
            best_d[12] = ((swap != 0) ? right_d : left_d);
            best_i[12] = ((swap != 0) ? right_i : left_i);
            best_d[14] = ((swap != 0) ? left_d : right_d);
            best_i[14] = ((swap != 0) ? left_i : right_i);
        }
    }
    {
        {
            float left_d = best_d[13];
            float right_d = best_d[15];
            int left_i = best_i[13];
            int right_i = best_i[15];
            int swap = ((right_d < left_d) ? 1 : 0);
            best_d[13] = ((swap != 0) ? right_d : left_d);
            best_i[13] = ((swap != 0) ? right_i : left_i);
            best_d[15] = ((swap != 0) ? left_d : right_d);
            best_i[15] = ((swap != 0) ? left_i : right_i);
        }
    }
    {
        {
            float left_d = best_d[16];
            float right_d = best_d[18];
            int left_i = best_i[16];
            int right_i = best_i[18];
            int swap = ((right_d < left_d) ? 1 : 0);
            best_d[16] = ((swap != 0) ? right_d : left_d);
            best_i[16] = ((swap != 0) ? right_i : left_i);
            best_d[18] = ((swap != 0) ? left_d : right_d);
            best_i[18] = ((swap != 0) ? left_i : right_i);
        }
    }
    {
        {
            float left_d = best_d[17];
            float right_d = best_d[19];
            int left_i = best_i[17];
            int right_i = best_i[19];
            int swap = ((right_d < left_d) ? 1 : 0);
            best_d[17] = ((swap != 0) ? right_d : left_d);
            best_i[17] = ((swap != 0) ? right_i : left_i);
            best_d[19] = ((swap != 0) ? left_d : right_d);
            best_i[19] = ((swap != 0) ? left_i : right_i);
        }
    }
    {
        {
            float left_d = best_d[20];
            float right_d = best_d[22];
            int left_i = best_i[20];
            int right_i = best_i[22];
            int swap = ((right_d < left_d) ? 1 : 0);
            best_d[20] = ((swap != 0) ? right_d : left_d);
            best_i[20] = ((swap != 0) ? right_i : left_i);
            best_d[22] = ((swap != 0) ? left_d : right_d);
            best_i[22] = ((swap != 0) ? left_i : right_i);
        }
    }
    {
        {
            float left_d = best_d[21];
            float right_d = best_d[23];
            int left_i = best_i[21];
            int right_i = best_i[23];
            int swap = ((right_d < left_d) ? 1 : 0);
            best_d[21] = ((swap != 0) ? right_d : left_d);
            best_i[21] = ((swap != 0) ? right_i : left_i);
            best_d[23] = ((swap != 0) ? left_d : right_d);
            best_i[23] = ((swap != 0) ? left_i : right_i);
        }
    }
    {
        {
            float left_d = best_d[24];
            float right_d = best_d[26];
            int left_i = best_i[24];
            int right_i = best_i[26];
            int swap = ((right_d < left_d) ? 1 : 0);
            best_d[24] = ((swap != 0) ? right_d : left_d);
            best_i[24] = ((swap != 0) ? right_i : left_i);
            best_d[26] = ((swap != 0) ? left_d : right_d);
            best_i[26] = ((swap != 0) ? left_i : right_i);
        }
    }
    {
        {
            float left_d = best_d[25];
            float right_d = best_d[27];
            int left_i = best_i[25];
            int right_i = best_i[27];
            int swap = ((right_d < left_d) ? 1 : 0);
            best_d[25] = ((swap != 0) ? right_d : left_d);
            best_i[25] = ((swap != 0) ? right_i : left_i);
            best_d[27] = ((swap != 0) ? left_d : right_d);
            best_i[27] = ((swap != 0) ? left_i : right_i);
        }
    }
    {
        {
            float left_d = best_d[28];
            float right_d = best_d[30];
            int left_i = best_i[28];
            int right_i = best_i[30];
            int swap = ((right_d < left_d) ? 1 : 0);
            best_d[28] = ((swap != 0) ? right_d : left_d);
            best_i[28] = ((swap != 0) ? right_i : left_i);
            best_d[30] = ((swap != 0) ? left_d : right_d);
            best_i[30] = ((swap != 0) ? left_i : right_i);
        }
    }
    {
        {
            float left_d = best_d[29];
            float right_d = best_d[31];
            int left_i = best_i[29];
            int right_i = best_i[31];
            int swap = ((right_d < left_d) ? 1 : 0);
            best_d[29] = ((swap != 0) ? right_d : left_d);
            best_i[29] = ((swap != 0) ? right_i : left_i);
            best_d[31] = ((swap != 0) ? left_d : right_d);
            best_i[31] = ((swap != 0) ? left_i : right_i);
        }
    }
    {
        {
            float left_d = best_d[32];
            float right_d = best_d[34];
            int left_i = best_i[32];
            int right_i = best_i[34];
            int swap = ((right_d < left_d) ? 1 : 0);
            best_d[32] = ((swap != 0) ? right_d : left_d);
            best_i[32] = ((swap != 0) ? right_i : left_i);
            best_d[34] = ((swap != 0) ? left_d : right_d);
            best_i[34] = ((swap != 0) ? left_i : right_i);
        }
    }
    {
        {
            float left_d = best_d[33];
            float right_d = best_d[35];
            int left_i = best_i[33];
            int right_i = best_i[35];
            int swap = ((right_d < left_d) ? 1 : 0);
            best_d[33] = ((swap != 0) ? right_d : left_d);
            best_i[33] = ((swap != 0) ? right_i : left_i);
            best_d[35] = ((swap != 0) ? left_d : right_d);
            best_i[35] = ((swap != 0) ? left_i : right_i);
        }
    }
    {
        {
            float left_d = best_d[36];
            float right_d = best_d[38];
            int left_i = best_i[36];
            int right_i = best_i[38];
            int swap = ((right_d < left_d) ? 1 : 0);
            best_d[36] = ((swap != 0) ? right_d : left_d);
            best_i[36] = ((swap != 0) ? right_i : left_i);
            best_d[38] = ((swap != 0) ? left_d : right_d);
            best_i[38] = ((swap != 0) ? left_i : right_i);
        }
    }
    {
        {
            float left_d = best_d[37];
            float right_d = best_d[39];
            int left_i = best_i[37];
            int right_i = best_i[39];
            int swap = ((right_d < left_d) ? 1 : 0);
            best_d[37] = ((swap != 0) ? right_d : left_d);
            best_i[37] = ((swap != 0) ? right_i : left_i);
            best_d[39] = ((swap != 0) ? left_d : right_d);
            best_i[39] = ((swap != 0) ? left_i : right_i);
        }
    }
    {
        {
            float left_d = best_d[40];
            float right_d = best_d[42];
            int left_i = best_i[40];
            int right_i = best_i[42];
            int swap = ((right_d < left_d) ? 1 : 0);
            best_d[40] = ((swap != 0) ? right_d : left_d);
            best_i[40] = ((swap != 0) ? right_i : left_i);
            best_d[42] = ((swap != 0) ? left_d : right_d);
            best_i[42] = ((swap != 0) ? left_i : right_i);
        }
    }
    {
        {
            float left_d = best_d[41];
            float right_d = best_d[43];
            int left_i = best_i[41];
            int right_i = best_i[43];
            int swap = ((right_d < left_d) ? 1 : 0);
            best_d[41] = ((swap != 0) ? right_d : left_d);
            best_i[41] = ((swap != 0) ? right_i : left_i);
            best_d[43] = ((swap != 0) ? left_d : right_d);
            best_i[43] = ((swap != 0) ? left_i : right_i);
        }
    }
    {
        {
            float left_d = best_d[44];
            float right_d = best_d[46];
            int left_i = best_i[44];
            int right_i = best_i[46];
            int swap = ((right_d < left_d) ? 1 : 0);
            best_d[44] = ((swap != 0) ? right_d : left_d);
            best_i[44] = ((swap != 0) ? right_i : left_i);
            best_d[46] = ((swap != 0) ? left_d : right_d);
            best_i[46] = ((swap != 0) ? left_i : right_i);
        }
    }
    {
        {
            float left_d = best_d[45];
            float right_d = best_d[47];
            int left_i = best_i[45];
            int right_i = best_i[47];
            int swap = ((right_d < left_d) ? 1 : 0);
            best_d[45] = ((swap != 0) ? right_d : left_d);
            best_i[45] = ((swap != 0) ? right_i : left_i);
            best_d[47] = ((swap != 0) ? left_d : right_d);
            best_i[47] = ((swap != 0) ? left_i : right_i);
        }
    }
    {
        {
            float left_d = best_d[48];
            float right_d = best_d[50];
            int left_i = best_i[48];
            int right_i = best_i[50];
            int swap = ((right_d < left_d) ? 1 : 0);
            best_d[48] = ((swap != 0) ? right_d : left_d);
            best_i[48] = ((swap != 0) ? right_i : left_i);
            best_d[50] = ((swap != 0) ? left_d : right_d);
            best_i[50] = ((swap != 0) ? left_i : right_i);
        }
    }
    {
        {
            float left_d = best_d[49];
            float right_d = best_d[51];
            int left_i = best_i[49];
            int right_i = best_i[51];
            int swap = ((right_d < left_d) ? 1 : 0);
            best_d[49] = ((swap != 0) ? right_d : left_d);
            best_i[49] = ((swap != 0) ? right_i : left_i);
            best_d[51] = ((swap != 0) ? left_d : right_d);
            best_i[51] = ((swap != 0) ? left_i : right_i);
        }
    }
    {
        {
            float left_d = best_d[52];
            float right_d = best_d[54];
            int left_i = best_i[52];
            int right_i = best_i[54];
            int swap = ((right_d < left_d) ? 1 : 0);
            best_d[52] = ((swap != 0) ? right_d : left_d);
            best_i[52] = ((swap != 0) ? right_i : left_i);
            best_d[54] = ((swap != 0) ? left_d : right_d);
            best_i[54] = ((swap != 0) ? left_i : right_i);
        }
    }
    {
        {
            float left_d = best_d[53];
            float right_d = best_d[55];
            int left_i = best_i[53];
            int right_i = best_i[55];
            int swap = ((right_d < left_d) ? 1 : 0);
            best_d[53] = ((swap != 0) ? right_d : left_d);
            best_i[53] = ((swap != 0) ? right_i : left_i);
            best_d[55] = ((swap != 0) ? left_d : right_d);
            best_i[55] = ((swap != 0) ? left_i : right_i);
        }
    }
    {
        {
            float left_d = best_d[56];
            float right_d = best_d[58];
            int left_i = best_i[56];
            int right_i = best_i[58];
            int swap = ((right_d < left_d) ? 1 : 0);
            best_d[56] = ((swap != 0) ? right_d : left_d);
            best_i[56] = ((swap != 0) ? right_i : left_i);
            best_d[58] = ((swap != 0) ? left_d : right_d);
            best_i[58] = ((swap != 0) ? left_i : right_i);
        }
    }
    {
        {
            float left_d = best_d[57];
            float right_d = best_d[59];
            int left_i = best_i[57];
            int right_i = best_i[59];
            int swap = ((right_d < left_d) ? 1 : 0);
            best_d[57] = ((swap != 0) ? right_d : left_d);
            best_i[57] = ((swap != 0) ? right_i : left_i);
            best_d[59] = ((swap != 0) ? left_d : right_d);
            best_i[59] = ((swap != 0) ? left_i : right_i);
        }
    }
    {
        {
            float left_d = best_d[60];
            float right_d = best_d[62];
            int left_i = best_i[60];
            int right_i = best_i[62];
            int swap = ((right_d < left_d) ? 1 : 0);
            best_d[60] = ((swap != 0) ? right_d : left_d);
            best_i[60] = ((swap != 0) ? right_i : left_i);
            best_d[62] = ((swap != 0) ? left_d : right_d);
            best_i[62] = ((swap != 0) ? left_i : right_i);
        }
    }
    {
        {
            float left_d = best_d[61];
            float right_d = best_d[63];
            int left_i = best_i[61];
            int right_i = best_i[63];
            int swap = ((right_d < left_d) ? 1 : 0);
            best_d[61] = ((swap != 0) ? right_d : left_d);
            best_i[61] = ((swap != 0) ? right_i : left_i);
            best_d[63] = ((swap != 0) ? left_d : right_d);
            best_i[63] = ((swap != 0) ? left_i : right_i);
        }
    }
    {
        {
            float left_d = best_d[0];
            float right_d = best_d[1];
            int left_i = best_i[0];
            int right_i = best_i[1];
            int swap = ((right_d < left_d) ? 1 : 0);
            best_d[0] = ((swap != 0) ? right_d : left_d);
            best_i[0] = ((swap != 0) ? right_i : left_i);
            best_d[1] = ((swap != 0) ? left_d : right_d);
            best_i[1] = ((swap != 0) ? left_i : right_i);
        }
    }
    {
        {
            float left_d = best_d[2];
            float right_d = best_d[3];
            int left_i = best_i[2];
            int right_i = best_i[3];
            int swap = ((right_d < left_d) ? 1 : 0);
            best_d[2] = ((swap != 0) ? right_d : left_d);
            best_i[2] = ((swap != 0) ? right_i : left_i);
            best_d[3] = ((swap != 0) ? left_d : right_d);
            best_i[3] = ((swap != 0) ? left_i : right_i);
        }
    }
    {
        {
            float left_d = best_d[4];
            float right_d = best_d[5];
            int left_i = best_i[4];
            int right_i = best_i[5];
            int swap = ((right_d < left_d) ? 1 : 0);
            best_d[4] = ((swap != 0) ? right_d : left_d);
            best_i[4] = ((swap != 0) ? right_i : left_i);
            best_d[5] = ((swap != 0) ? left_d : right_d);
            best_i[5] = ((swap != 0) ? left_i : right_i);
        }
    }
    {
        {
            float left_d = best_d[6];
            float right_d = best_d[7];
            int left_i = best_i[6];
            int right_i = best_i[7];
            int swap = ((right_d < left_d) ? 1 : 0);
            best_d[6] = ((swap != 0) ? right_d : left_d);
            best_i[6] = ((swap != 0) ? right_i : left_i);
            best_d[7] = ((swap != 0) ? left_d : right_d);
            best_i[7] = ((swap != 0) ? left_i : right_i);
        }
    }
    {
        {
            float left_d = best_d[8];
            float right_d = best_d[9];
            int left_i = best_i[8];
            int right_i = best_i[9];
            int swap = ((right_d < left_d) ? 1 : 0);
            best_d[8] = ((swap != 0) ? right_d : left_d);
            best_i[8] = ((swap != 0) ? right_i : left_i);
            best_d[9] = ((swap != 0) ? left_d : right_d);
            best_i[9] = ((swap != 0) ? left_i : right_i);
        }
    }
    {
        {
            float left_d = best_d[10];
            float right_d = best_d[11];
            int left_i = best_i[10];
            int right_i = best_i[11];
            int swap = ((right_d < left_d) ? 1 : 0);
            best_d[10] = ((swap != 0) ? right_d : left_d);
            best_i[10] = ((swap != 0) ? right_i : left_i);
            best_d[11] = ((swap != 0) ? left_d : right_d);
            best_i[11] = ((swap != 0) ? left_i : right_i);
        }
    }
    {
        {
            float left_d = best_d[12];
            float right_d = best_d[13];
            int left_i = best_i[12];
            int right_i = best_i[13];
            int swap = ((right_d < left_d) ? 1 : 0);
            best_d[12] = ((swap != 0) ? right_d : left_d);
            best_i[12] = ((swap != 0) ? right_i : left_i);
            best_d[13] = ((swap != 0) ? left_d : right_d);
            best_i[13] = ((swap != 0) ? left_i : right_i);
        }
    }
    {
        {
            float left_d = best_d[14];
            float right_d = best_d[15];
            int left_i = best_i[14];
            int right_i = best_i[15];
            int swap = ((right_d < left_d) ? 1 : 0);
            best_d[14] = ((swap != 0) ? right_d : left_d);
            best_i[14] = ((swap != 0) ? right_i : left_i);
            best_d[15] = ((swap != 0) ? left_d : right_d);
            best_i[15] = ((swap != 0) ? left_i : right_i);
        }
    }
    {
        {
            float left_d = best_d[16];
            float right_d = best_d[17];
            int left_i = best_i[16];
            int right_i = best_i[17];
            int swap = ((right_d < left_d) ? 1 : 0);
            best_d[16] = ((swap != 0) ? right_d : left_d);
            best_i[16] = ((swap != 0) ? right_i : left_i);
            best_d[17] = ((swap != 0) ? left_d : right_d);
            best_i[17] = ((swap != 0) ? left_i : right_i);
        }
    }
    {
        {
            float left_d = best_d[18];
            float right_d = best_d[19];
            int left_i = best_i[18];
            int right_i = best_i[19];
            int swap = ((right_d < left_d) ? 1 : 0);
            best_d[18] = ((swap != 0) ? right_d : left_d);
            best_i[18] = ((swap != 0) ? right_i : left_i);
            best_d[19] = ((swap != 0) ? left_d : right_d);
            best_i[19] = ((swap != 0) ? left_i : right_i);
        }
    }
    {
        {
            float left_d = best_d[20];
            float right_d = best_d[21];
            int left_i = best_i[20];
            int right_i = best_i[21];
            int swap = ((right_d < left_d) ? 1 : 0);
            best_d[20] = ((swap != 0) ? right_d : left_d);
            best_i[20] = ((swap != 0) ? right_i : left_i);
            best_d[21] = ((swap != 0) ? left_d : right_d);
            best_i[21] = ((swap != 0) ? left_i : right_i);
        }
    }
    {
        {
            float left_d = best_d[22];
            float right_d = best_d[23];
            int left_i = best_i[22];
            int right_i = best_i[23];
            int swap = ((right_d < left_d) ? 1 : 0);
            best_d[22] = ((swap != 0) ? right_d : left_d);
            best_i[22] = ((swap != 0) ? right_i : left_i);
            best_d[23] = ((swap != 0) ? left_d : right_d);
            best_i[23] = ((swap != 0) ? left_i : right_i);
        }
    }
    {
        {
            float left_d = best_d[24];
            float right_d = best_d[25];
            int left_i = best_i[24];
            int right_i = best_i[25];
            int swap = ((right_d < left_d) ? 1 : 0);
            best_d[24] = ((swap != 0) ? right_d : left_d);
            best_i[24] = ((swap != 0) ? right_i : left_i);
            best_d[25] = ((swap != 0) ? left_d : right_d);
            best_i[25] = ((swap != 0) ? left_i : right_i);
        }
    }
    {
        {
            float left_d = best_d[26];
            float right_d = best_d[27];
            int left_i = best_i[26];
            int right_i = best_i[27];
            int swap = ((right_d < left_d) ? 1 : 0);
            best_d[26] = ((swap != 0) ? right_d : left_d);
            best_i[26] = ((swap != 0) ? right_i : left_i);
            best_d[27] = ((swap != 0) ? left_d : right_d);
            best_i[27] = ((swap != 0) ? left_i : right_i);
        }
    }
    {
        {
            float left_d = best_d[28];
            float right_d = best_d[29];
            int left_i = best_i[28];
            int right_i = best_i[29];
            int swap = ((right_d < left_d) ? 1 : 0);
            best_d[28] = ((swap != 0) ? right_d : left_d);
            best_i[28] = ((swap != 0) ? right_i : left_i);
            best_d[29] = ((swap != 0) ? left_d : right_d);
            best_i[29] = ((swap != 0) ? left_i : right_i);
        }
    }
    {
        {
            float left_d = best_d[30];
            float right_d = best_d[31];
            int left_i = best_i[30];
            int right_i = best_i[31];
            int swap = ((right_d < left_d) ? 1 : 0);
            best_d[30] = ((swap != 0) ? right_d : left_d);
            best_i[30] = ((swap != 0) ? right_i : left_i);
            best_d[31] = ((swap != 0) ? left_d : right_d);
            best_i[31] = ((swap != 0) ? left_i : right_i);
        }
    }
    {
        {
            float left_d = best_d[32];
            float right_d = best_d[33];
            int left_i = best_i[32];
            int right_i = best_i[33];
            int swap = ((right_d < left_d) ? 1 : 0);
            best_d[32] = ((swap != 0) ? right_d : left_d);
            best_i[32] = ((swap != 0) ? right_i : left_i);
            best_d[33] = ((swap != 0) ? left_d : right_d);
            best_i[33] = ((swap != 0) ? left_i : right_i);
        }
    }
    {
        {
            float left_d = best_d[34];
            float right_d = best_d[35];
            int left_i = best_i[34];
            int right_i = best_i[35];
            int swap = ((right_d < left_d) ? 1 : 0);
            best_d[34] = ((swap != 0) ? right_d : left_d);
            best_i[34] = ((swap != 0) ? right_i : left_i);
            best_d[35] = ((swap != 0) ? left_d : right_d);
            best_i[35] = ((swap != 0) ? left_i : right_i);
        }
    }
    {
        {
            float left_d = best_d[36];
            float right_d = best_d[37];
            int left_i = best_i[36];
            int right_i = best_i[37];
            int swap = ((right_d < left_d) ? 1 : 0);
            best_d[36] = ((swap != 0) ? right_d : left_d);
            best_i[36] = ((swap != 0) ? right_i : left_i);
            best_d[37] = ((swap != 0) ? left_d : right_d);
            best_i[37] = ((swap != 0) ? left_i : right_i);
        }
    }
    {
        {
            float left_d = best_d[38];
            float right_d = best_d[39];
            int left_i = best_i[38];
            int right_i = best_i[39];
            int swap = ((right_d < left_d) ? 1 : 0);
            best_d[38] = ((swap != 0) ? right_d : left_d);
            best_i[38] = ((swap != 0) ? right_i : left_i);
            best_d[39] = ((swap != 0) ? left_d : right_d);
            best_i[39] = ((swap != 0) ? left_i : right_i);
        }
    }
    {
        {
            float left_d = best_d[40];
            float right_d = best_d[41];
            int left_i = best_i[40];
            int right_i = best_i[41];
            int swap = ((right_d < left_d) ? 1 : 0);
            best_d[40] = ((swap != 0) ? right_d : left_d);
            best_i[40] = ((swap != 0) ? right_i : left_i);
            best_d[41] = ((swap != 0) ? left_d : right_d);
            best_i[41] = ((swap != 0) ? left_i : right_i);
        }
    }
    {
        {
            float left_d = best_d[42];
            float right_d = best_d[43];
            int left_i = best_i[42];
            int right_i = best_i[43];
            int swap = ((right_d < left_d) ? 1 : 0);
            best_d[42] = ((swap != 0) ? right_d : left_d);
            best_i[42] = ((swap != 0) ? right_i : left_i);
            best_d[43] = ((swap != 0) ? left_d : right_d);
            best_i[43] = ((swap != 0) ? left_i : right_i);
        }
    }
    {
        {
            float left_d = best_d[44];
            float right_d = best_d[45];
            int left_i = best_i[44];
            int right_i = best_i[45];
            int swap = ((right_d < left_d) ? 1 : 0);
            best_d[44] = ((swap != 0) ? right_d : left_d);
            best_i[44] = ((swap != 0) ? right_i : left_i);
            best_d[45] = ((swap != 0) ? left_d : right_d);
            best_i[45] = ((swap != 0) ? left_i : right_i);
        }
    }
    {
        {
            float left_d = best_d[46];
            float right_d = best_d[47];
            int left_i = best_i[46];
            int right_i = best_i[47];
            int swap = ((right_d < left_d) ? 1 : 0);
            best_d[46] = ((swap != 0) ? right_d : left_d);
            best_i[46] = ((swap != 0) ? right_i : left_i);
            best_d[47] = ((swap != 0) ? left_d : right_d);
            best_i[47] = ((swap != 0) ? left_i : right_i);
        }
    }
    {
        {
            float left_d = best_d[48];
            float right_d = best_d[49];
            int left_i = best_i[48];
            int right_i = best_i[49];
            int swap = ((right_d < left_d) ? 1 : 0);
            best_d[48] = ((swap != 0) ? right_d : left_d);
            best_i[48] = ((swap != 0) ? right_i : left_i);
            best_d[49] = ((swap != 0) ? left_d : right_d);
            best_i[49] = ((swap != 0) ? left_i : right_i);
        }
    }
    {
        {
            float left_d = best_d[50];
            float right_d = best_d[51];
            int left_i = best_i[50];
            int right_i = best_i[51];
            int swap = ((right_d < left_d) ? 1 : 0);
            best_d[50] = ((swap != 0) ? right_d : left_d);
            best_i[50] = ((swap != 0) ? right_i : left_i);
            best_d[51] = ((swap != 0) ? left_d : right_d);
            best_i[51] = ((swap != 0) ? left_i : right_i);
        }
    }
    {
        {
            float left_d = best_d[52];
            float right_d = best_d[53];
            int left_i = best_i[52];
            int right_i = best_i[53];
            int swap = ((right_d < left_d) ? 1 : 0);
            best_d[52] = ((swap != 0) ? right_d : left_d);
            best_i[52] = ((swap != 0) ? right_i : left_i);
            best_d[53] = ((swap != 0) ? left_d : right_d);
            best_i[53] = ((swap != 0) ? left_i : right_i);
        }
    }
    {
        {
            float left_d = best_d[54];
            float right_d = best_d[55];
            int left_i = best_i[54];
            int right_i = best_i[55];
            int swap = ((right_d < left_d) ? 1 : 0);
            best_d[54] = ((swap != 0) ? right_d : left_d);
            best_i[54] = ((swap != 0) ? right_i : left_i);
            best_d[55] = ((swap != 0) ? left_d : right_d);
            best_i[55] = ((swap != 0) ? left_i : right_i);
        }
    }
    {
        {
            float left_d = best_d[56];
            float right_d = best_d[57];
            int left_i = best_i[56];
            int right_i = best_i[57];
            int swap = ((right_d < left_d) ? 1 : 0);
            best_d[56] = ((swap != 0) ? right_d : left_d);
            best_i[56] = ((swap != 0) ? right_i : left_i);
            best_d[57] = ((swap != 0) ? left_d : right_d);
            best_i[57] = ((swap != 0) ? left_i : right_i);
        }
    }
    {
        {
            float left_d = best_d[58];
            float right_d = best_d[59];
            int left_i = best_i[58];
            int right_i = best_i[59];
            int swap = ((right_d < left_d) ? 1 : 0);
            best_d[58] = ((swap != 0) ? right_d : left_d);
            best_i[58] = ((swap != 0) ? right_i : left_i);
            best_d[59] = ((swap != 0) ? left_d : right_d);
            best_i[59] = ((swap != 0) ? left_i : right_i);
        }
    }
    {
        {
            float left_d = best_d[60];
            float right_d = best_d[61];
            int left_i = best_i[60];
            int right_i = best_i[61];
            int swap = ((right_d < left_d) ? 1 : 0);
            best_d[60] = ((swap != 0) ? right_d : left_d);
            best_i[60] = ((swap != 0) ? right_i : left_i);
            best_d[61] = ((swap != 0) ? left_d : right_d);
            best_i[61] = ((swap != 0) ? left_i : right_i);
        }
    }
    {
        {
            float left_d = best_d[62];
            float right_d = best_d[63];
            int left_i = best_i[62];
            int right_i = best_i[63];
            int swap = ((right_d < left_d) ? 1 : 0);
            best_d[62] = ((swap != 0) ? right_d : left_d);
            best_i[62] = ((swap != 0) ? right_i : left_i);
            best_d[63] = ((swap != 0) ? left_d : right_d);
            best_i[63] = ((swap != 0) ? left_i : right_i);
        }
    }
    {
        #pragma unroll
        for (int kk = 0; kk < K_MAX_; kk += 2) {
            {
                float2 _v2 = make_float2(best_d[kk + 0], best_d[kk + 1]);
                *reinterpret_cast<float2*>(partial_distances + partial_col_base + kk) = _v2;
            }
        }
        #pragma unroll
        for (int kk = 0; kk < K_MAX_; kk += 2) {
            {
                int2 _iv2 = make_int2(best_i[kk + 0], best_i[kk + 1]);
                *reinterpret_cast<int2*>(partial_indices + partial_col_base + kk) = _iv2;
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

