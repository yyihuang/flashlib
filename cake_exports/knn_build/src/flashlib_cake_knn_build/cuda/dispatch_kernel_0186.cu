typedef unsigned char      uint8_t;
typedef unsigned short     uint16_t;
typedef unsigned int       uint32_t;
typedef unsigned long long uint64_t;
typedef signed int         int32_t;
typedef short int          int16_t;

#include <cuda_bf16.h>

#define TMEM_NCOLS 64
#define TMEM_CROSS_OFFSET 0
#define NUM_MAIN_STAGES 1
#define SMEM_SMEM_QUERY_OFF 1024
#define SMEM_SMEM_QUERY_STAGE_BYTES 16384
#define SMEM_SMEM_QUERY_STRIDE 16384
#define SMEM_SMEM_DATABASE_OFF 17408
#define SMEM_SMEM_DATABASE_STAGE_BYTES 16384
#define SMEM_SMEM_DATABASE_STRIDE 16384
#define SMEM_SMEM_UPPER_DOTS_OFF 34048
#define SMEM_SMEM_UPPER_DOTS_STAGE_BYTES 2048
#define SMEM_SMEM_UPPER_DOTS_STRIDE 2048
#define SMEM_SMEM_LOCAL_D_OFF 36096
#define SMEM_SMEM_LOCAL_D_STAGE_BYTES 12288
#define SMEM_SMEM_LOCAL_D_STRIDE 12288
#define SMEM_SMEM_LOCAL_I_OFF 48384
#define SMEM_SMEM_LOCAL_I_STAGE_BYTES 12288
#define SMEM_SMEM_LOCAL_I_STRIDE 12288
#define SMEM_TOTAL 60672
#define THREADS 128
#define BLOCK_Q_CONST 64
#define BLOCK_M_CONST 64
#define FEAT_D_CONST 128
#define TOP_K_MAX 48
#define ROWS_COVERED_CONST 16

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


__device__ __forceinline__ uint64_t make_smem_desc(int addr) {
    const int SBO = 1024;
    return desc_encode(addr)
         | (desc_encode(SBO) << 32ULL)
         | (1ULL << 46ULL)
         | (2ULL << 61ULL);
}


__device__ __forceinline__ void tma_3d_gmem2smem(
    int dst, const void *tmap_ptr, int x, int y, int z, int mbar_addr) {
    asm volatile(
        "cp.async.bulk.tensor.3d.shared::cta.global"
        ".mbarrier::complete_tx::bytes"
        " [%0], [%1, {%2, %3, %4}], [%5];"
        :: "r"(dst), "l"(tmap_ptr), "r"(x), "r"(y), "r"(z),
           "r"(mbar_addr) : "memory");
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

__global__ __launch_bounds__(128, 1) void
kernel_knn_build_v12_d128_q16_k48_dd2b_v1_stage1(float* __restrict__ query_sq, float* __restrict__ database_sq, float* __restrict__ partial_dists, int32_t* __restrict__ partial_indices, const void* tmap_query, const void* tmap_database, int B, int Q, int M, int K, int num_q_tiles, int db_tiles_per_split, int split_count, int total_work)
{
    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;

    extern __shared__ __align__(1024) char smem_raw[];
    int smem;
    smem = (int)(unsigned long long)__cvta_generic_to_shared(smem_raw);
    const int smem_smem_query = smem + 1024;
    const int smem_smem_database = smem + 17408;
    const int smem_smem_upper_dots = smem + 34048;
    const int smem_smem_local_d = smem + 36096;
    const int smem_smem_local_i = smem + 48384;

    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    // Mbarrier init (6 groups, 6 barriers)
    // Mbarriers at smem_raw[0..48)

    if (warp == 0) {
        uint32_t leader = elect_sync();
        // query_full: 1 barriers, init_count=1
        mbarrier_init_pred(smem + 0, 1, leader);
        // query_empty: 1 barriers, init_count=1
        mbarrier_init_pred(smem + 8, 1, leader);
        // database_full: 1 barriers, init_count=1
        mbarrier_init_pred(smem + 16, 1, leader);
        // database_empty: 1 barriers, init_count=1
        mbarrier_init_pred(smem + 24, 1, leader);
        // score_full: 1 barriers, init_count=1
        mbarrier_init_pred(smem + 32, 1, leader);
        // score_empty: 1 barriers, init_count=2
        mbarrier_init_pred(smem + 40, 2, leader);
        asm volatile("fence.mbarrier_init.release.cluster;");
    }

    __syncthreads();

    // TMEM alloc (64 columns, 64 used)
    volatile int* tmem_addr_storage = (volatile int*)(smem_raw + 48);
    if (warp == 3) {
        int _tmem_hold = smem + 48;
        asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;" :: "r"(_tmem_hold), "r"(64) : "memory");
    }

    __syncthreads();
    asm volatile("tcgen05.fence::after_thread_sync;");

    const int warp_id = warp;
    const int lane_id = lane;
    __nv_bfloat16* smem_query = (__nv_bfloat16*)(smem_raw + 1024);
    #define smem_query_addr (smem + 1024)
    __nv_bfloat16* smem_database = (__nv_bfloat16*)(smem_raw + 17408);
    #define smem_database_addr (smem + 17408)
    float* smem_upper_dots = (float*)(smem_raw + 34048);
    #define smem_upper_dots_addr (smem + 34048)
    float* smem_local_d = (float*)(smem_raw + 36096);
    #define smem_local_d_addr (smem + 36096)
    int* smem_local_i = (int*)(smem_raw + 48384);
    #define smem_local_i_addr (smem + 48384)
    const int mbar_base = smem;
    #define query_full_addr (mbar_base + 0)
    #define query_empty_addr (mbar_base + 8)
    #define database_full_addr (mbar_base + 16)
    #define database_empty_addr (mbar_base + 24)
    #define score_full_addr (mbar_base + 32)
    #define score_empty_addr (mbar_base + 40)
    const int taddr = tmem_addr_storage[0];

    // ---- Role: compute ----
    if (warp <= 1) {
        const int tmem_row_base = (warp % 2) * 32;
        const int my_row = tmem_row_base + (lane / 4);
        { // compute_main
            uint32_t _phase_score_full_0 = 0;
            #pragma unroll 1
            for (unsigned int work_idx = bid; work_idx < total_work; work_idx += num_bids) {
                int split_idx = work_idx % split_count;
                int query_work = work_idx / split_count;
                int batch_idx = query_work / num_q_tiles;
                int q_tile = query_work % num_q_tiles;
                int off_q = q_tile * BLOCK_Q_CONST;
                int row = warp * 8 + lane / 4;
                int lane_col = lane % 4;
                int slot = lane_col;
                int q_idx = off_q + row;
                int valid_row = ((q_idx < Q) ? 1 : 0);
                float q_sq_val = 0.0f;
                if (valid_row != 0) {
                    q_sq_val = (float)query_sq[batch_idx * Q + q_idx];
                }
                float best_d[TOP_K_MAX];
                int best_i[TOP_K_MAX];
                #pragma unroll
                for (int kk = 0; kk < TOP_K_MAX; kk++) {
                    best_d[kk] = 3.4e+38f;
                    best_i[kk] = -1;
                }
                int db_tile_start = split_idx * db_tiles_per_split;
                #pragma unroll 1
                for (int local_db_tile = 0; local_db_tile < db_tiles_per_split; local_db_tile++) {
                    int db_tile = db_tile_start + local_db_tile;
                    int db_start = db_tile * BLOCK_M_CONST;
                    mbarrier_wait(score_full_addr, _phase_score_full_0);
                    _phase_score_full_0 ^= 1;
                    float dots[32];
                    asm volatile(
                        "tcgen05.ld.sync.aligned.16x256b.x8.b32"
                        " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&dots[0])), "=r"(*reinterpret_cast<uint32_t*>(&dots[1])), "=r"(*reinterpret_cast<uint32_t*>(&dots[2])), "=r"(*reinterpret_cast<uint32_t*>(&dots[3])), "=r"(*reinterpret_cast<uint32_t*>(&dots[4])), "=r"(*reinterpret_cast<uint32_t*>(&dots[5])), "=r"(*reinterpret_cast<uint32_t*>(&dots[6])), "=r"(*reinterpret_cast<uint32_t*>(&dots[7])), "=r"(*reinterpret_cast<uint32_t*>(&dots[8])), "=r"(*reinterpret_cast<uint32_t*>(&dots[9])), "=r"(*reinterpret_cast<uint32_t*>(&dots[10])), "=r"(*reinterpret_cast<uint32_t*>(&dots[11])), "=r"(*reinterpret_cast<uint32_t*>(&dots[12])), "=r"(*reinterpret_cast<uint32_t*>(&dots[13])), "=r"(*reinterpret_cast<uint32_t*>(&dots[14])), "=r"(*reinterpret_cast<uint32_t*>(&dots[15])), "=r"(*reinterpret_cast<uint32_t*>(&dots[16])), "=r"(*reinterpret_cast<uint32_t*>(&dots[17])), "=r"(*reinterpret_cast<uint32_t*>(&dots[18])), "=r"(*reinterpret_cast<uint32_t*>(&dots[19])), "=r"(*reinterpret_cast<uint32_t*>(&dots[20])), "=r"(*reinterpret_cast<uint32_t*>(&dots[21])), "=r"(*reinterpret_cast<uint32_t*>(&dots[22])), "=r"(*reinterpret_cast<uint32_t*>(&dots[23])), "=r"(*reinterpret_cast<uint32_t*>(&dots[24])), "=r"(*reinterpret_cast<uint32_t*>(&dots[25])), "=r"(*reinterpret_cast<uint32_t*>(&dots[26])), "=r"(*reinterpret_cast<uint32_t*>(&dots[27])), "=r"(*reinterpret_cast<uint32_t*>(&dots[28])), "=r"(*reinterpret_cast<uint32_t*>(&dots[29])), "=r"(*reinterpret_cast<uint32_t*>(&dots[30])), "=r"(*reinterpret_cast<uint32_t*>(&dots[31]))
                        : "r"(taddr)
                        : "memory");
                    asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                    if (warp == 0) {
                        int scratch_row = lane / 4;
                        #pragma unroll
                        for (int repeat = 0; repeat < 8; repeat++) {
                            const int reg_base = repeat * 4;
                            int scratch_col = repeat * 8 + lane_col * 2;
                            int scratch_base = scratch_row * 64 + scratch_col;
                            smem_upper_dots[scratch_base] = dots[reg_base + 2];
                            smem_upper_dots[scratch_base + 1] = dots[reg_base + 3];
                        }
                    }
                    asm volatile("barrier.sync 8, 64;");
                    if (elect_sync()) {
                        mbarrier_arrive(score_empty_addr);
                    }
                    #pragma unroll
                    for (int repeat = 0; repeat < 8; repeat++) {
                        const int reg_base = repeat * 4;
                        int col_base = repeat * 8 + lane_col * 2;
                        int db_idx0 = db_start + col_base;
                        int db_idx1 = db_idx0 + 1;
                        float dot0 = dots[reg_base];
                        float dot1 = dots[reg_base + 1];
                        if (warp != 0) {
                            int scratch_row = lane / 4;
                            int scratch_base = scratch_row * 64 + col_base;
                            dot0 = smem_upper_dots[scratch_base];
                            dot1 = smem_upper_dots[scratch_base + 1];
                        }
                        float cand0_d = 3.4e+38f;
                        float cand1_d = 3.4e+38f;
                        if (valid_row != 0 & db_idx0 < M) {
                            cand0_d = max_noftz(q_sq_val + (float)database_sq[batch_idx * M + db_idx0] - 2.0f * dot0, 0.0f);
                        }
                        if (valid_row != 0 & db_idx1 < M) {
                            cand1_d = max_noftz(q_sq_val + (float)database_sq[batch_idx * M + db_idx1] - 2.0f * dot1, 0.0f);
                        }
                        int take1 = ((cand1_d < cand0_d) ? 1 : 0);
                        if (((take1 != 0) ? cand1_d : cand0_d) < best_d[47]) {
                            best_d[47] = ((take1 != 0) ? cand1_d : cand0_d);
                            best_i[47] = ((take1 != 0) ? db_idx1 : db_idx0);
                            #pragma unroll
                            for (int kk = 46; kk >= 0; kk--) {
                                float lower0_d = best_d[kk + 1];
                                int lower0_i = best_i[kk + 1];
                                float upper0_d = best_d[kk];
                                int upper0_i = best_i[kk];
                                int swap0_up = ((lower0_d < upper0_d) ? 1 : 0);
                                best_d[kk] = ((swap0_up != 0) ? lower0_d : upper0_d);
                                best_i[kk] = ((swap0_up != 0) ? lower0_i : upper0_i);
                                best_d[kk + 1] = ((swap0_up != 0) ? upper0_d : lower0_d);
                                best_i[kk + 1] = ((swap0_up != 0) ? upper0_i : lower0_i);
                            }
                            if (((take1 != 0) ? cand0_d : cand1_d) < best_d[47]) {
                                best_d[47] = ((take1 != 0) ? cand0_d : cand1_d);
                                best_i[47] = ((take1 != 0) ? db_idx0 : db_idx1);
                                #pragma unroll
                                for (int kk = 46; kk >= 0; kk--) {
                                    float lower1_d = best_d[kk + 1];
                                    int lower1_i = best_i[kk + 1];
                                    float upper1_d = best_d[kk];
                                    int upper1_i = best_i[kk];
                                    int swap1_up = ((lower1_d < upper1_d) ? 1 : 0);
                                    best_d[kk] = ((swap1_up != 0) ? lower1_d : upper1_d);
                                    best_i[kk] = ((swap1_up != 0) ? lower1_i : upper1_i);
                                    best_d[kk + 1] = ((swap1_up != 0) ? upper1_d : lower1_d);
                                    best_i[kk + 1] = ((swap1_up != 0) ? upper1_i : lower1_i);
                                }
                            }
                        }
                    }
                    asm volatile("barrier.sync 8, 64;");
                }
                int slot_base = (row * 4 + slot) * TOP_K_MAX;
                #pragma unroll
                for (int kk = 0; kk < TOP_K_MAX; kk++) {
                    smem_local_d[slot_base + kk] = best_d[kk];
                    smem_local_i[slot_base + kk] = best_i[kk];
                }
                asm volatile("barrier.sync 8, 64;");
                if (tid < ROWS_COVERED_CONST) {
                    int out_row = tid;
                    int out_q_idx = off_q + out_row;
                    float head_d[4];
                    int head_i[4];
                    int head_k[4];
                    #pragma unroll
                    for (int slot_idx = 0; slot_idx < 4; slot_idx++) {
                        int local_base = (out_row * 4 + slot_idx) * TOP_K_MAX;
                        head_k[slot_idx] = 0;
                        head_d[slot_idx] = smem_local_d[local_base];
                        head_i[slot_idx] = smem_local_i[local_base];
                    }
                    int out_base = ((split_idx * B + batch_idx) * Q + out_q_idx) * K;
                    #pragma unroll
                    for (int out_k = 0; out_k < TOP_K_MAX; out_k++) {
                        float winner_d = head_d[0];
                        int winner_i = head_i[0];
                        int winner_slot = 0;
                        #pragma unroll
                        for (int slot_idx = 1; slot_idx < 4; slot_idx++) {
                            float cand_d = head_d[slot_idx];
                            int take = ((cand_d < winner_d) ? 1 : 0);
                            winner_d = ((take != 0) ? cand_d : winner_d);
                            winner_i = ((take != 0) ? head_i[slot_idx] : winner_i);
                            winner_slot = ((take != 0) ? slot_idx : winner_slot);
                        }
                        if (out_q_idx < Q & out_k < K) {
                            *((float*)(partial_dists + out_base + out_k)) = winner_d;
                            *((int*)(partial_indices + out_base + out_k)) = winner_i;
                        }
                        #pragma unroll
                        for (int slot_idx = 0; slot_idx < 4; slot_idx++) {
                            if (winner_slot == slot_idx) {
                                int next_head = head_k[slot_idx] + 1;
                                head_k[slot_idx] = next_head;
                                head_d[slot_idx] = 3.4e+38f;
                                head_i[slot_idx] = -1;
                                if (next_head < TOP_K_MAX) {
                                    int local_base = (out_row * 4 + slot_idx) * TOP_K_MAX;
                                    head_d[slot_idx] = smem_local_d[local_base + next_head];
                                    head_i[slot_idx] = smem_local_i[local_base + next_head];
                                }
                            }
                        }
                    }
                }
                asm volatile("barrier.sync 8, 64;");
            }
        }
    // ---- Role: load ----
    } else if (warp == 2) {
        { // load_main
            uint32_t _phase_query_empty_0 = 1;
            uint32_t _phase_database_empty_0 = 1;
            if (warp_id == 2) {
                if (elect_sync()) {
                    #pragma unroll 1
                    for (unsigned int work_idx = bid; work_idx < total_work; work_idx += num_bids) {
                        int split_idx = work_idx % split_count;
                        int query_work = work_idx / split_count;
                        int batch_idx = query_work / num_q_tiles;
                        int q_tile = query_work % num_q_tiles;
                        int off_q = q_tile * BLOCK_Q_CONST;
                        int global_q = batch_idx * Q + off_q;
                        int db_tile_start = split_idx * db_tiles_per_split;
                        mbarrier_wait(query_empty_addr, _phase_query_empty_0);
                        _phase_query_empty_0 ^= 1;
                        mbarrier_arrive_expect_tx(query_full_addr, 16384);
                        tma_3d_gmem2smem(smem_query_addr, tmap_query, 0, global_q, 0, query_full_addr);
                        #pragma unroll 1
                        for (int local_db_tile = 0; local_db_tile < db_tiles_per_split; local_db_tile++) {
                            int db_tile = db_tile_start + local_db_tile;
                            int off_m = db_tile * BLOCK_M_CONST;
                            int global_m = batch_idx * M + off_m;
                            mbarrier_wait(database_empty_addr, _phase_database_empty_0);
                            _phase_database_empty_0 ^= 1;
                            mbarrier_arrive_expect_tx(database_full_addr, 16384);
                            tma_3d_gmem2smem(smem_database_addr, tmap_database, 0, global_m, 0, database_full_addr);
                        }
                    }
                }
            }
        }
    // ---- Role: mma ----
    } else if (warp == 3) {
        { // mma_main
            uint32_t _phase_query_full_0 = 0;
            uint32_t _phase_score_empty_0 = 1;
            uint32_t _phase_database_full_0 = 0;
            #pragma unroll 1
            for (unsigned int _work_idx = bid; _work_idx < total_work; _work_idx += num_bids) {
                mbarrier_wait(query_full_addr, _phase_query_full_0);
                _phase_query_full_0 ^= 1;
                #pragma unroll 1
                for (int _local_db_tile = 0; _local_db_tile < db_tiles_per_split; _local_db_tile++) {
                    mbarrier_wait(score_empty_addr, _phase_score_empty_0);
                    _phase_score_empty_0 ^= 1;
                    mbarrier_wait(database_full_addr, _phase_database_full_0);
                    _phase_database_full_0 ^= 1;
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    int _mma_ss_a_lo_0 = make_warp_uniform((smem_query_addr >> 4) & 0x3FFF);
                    int _mma_ss_b_lo_0 = make_warp_uniform((smem_database_addr >> 4) & 0x3FFF);
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
                    "mov.b32 id, 68158608;\n\t"
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
                    "add.u32 blo, blo, 506;\n\t"
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
                    :: "r"(_mma_ss_a_lo_0), "r"(_mma_ss_b_lo_0), "r"(taddr), "r"(0));
                    elect_commit(score_full_addr);
                    elect_commit(database_empty_addr);
                }
                elect_commit(query_empty_addr);
            }
        }
    }

    // Cleanup
    __syncthreads(); // barrier before TMEM dealloc

    if (warp == 3) {
        asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;" :: "r"(tmem_addr_storage[0]), "r"(64));
        asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;");
    }
}

} // extern "C"

