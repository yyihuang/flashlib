typedef unsigned char      uint8_t;
typedef unsigned short     uint16_t;
typedef unsigned int       uint32_t;
typedef unsigned long long uint64_t;
typedef signed int         int32_t;
typedef short int          int16_t;

#include <cuda_bf16.h>

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

__device__ __forceinline__ void tcgen05_mma_f16_cta2(
    int taddr, uint64_t a_desc, uint64_t b_desc,
    uint32_t i_desc, int enable_input_d) {
    asm volatile(
        "{\n\t"
        ".reg .pred p;\n\t"
        ".reg .b32 m0, m1, m2, m3, m4, m5, m6, m7;\n\t"
        "setp.ne.b32 p, %4, 0;\n\t"
        "mov.b32 m0, 0; mov.b32 m1, 0; mov.b32 m2, 0; mov.b32 m3, 0;\n\t"
        "mov.b32 m4, 0; mov.b32 m5, 0; mov.b32 m6, 0; mov.b32 m7, 0;\n\t"
        "tcgen05.mma.cta_group::2.kind::f16 [%0], %1, %2, %3, {m0, m1, m2, m3, m4, m5, m6, m7}, p;\n\t"
        "}\n"
        :: "r"(taddr), "l"(a_desc), "l"(b_desc),
           "r"(i_desc), "r"(enable_input_d));
}

__device__ __forceinline__ uint64_t desc_encode(uint64_t x) {
    return (x & 0x3FFFFULL) >> 4ULL;
}

__device__ __forceinline__ void mma_ss_step_cg2(
    int a_lo, int b_lo, int taddr, uint32_t i_desc, int enable_d) {
    asm volatile(
        "{\n\t"
        ".reg .pred leader, p;\n\t"
        ".reg .b32 dhi, m0, m1, m2, m3, m4, m5, m6, m7;\n\t"
        ".reg .b64 da, db;\n\t"
        "elect.sync _|leader, 0xFFFFFFFF;\n\t"
        "setp.ne.b32 p, %4, 0;\n\t"
        "mov.b32 m0, 0; mov.b32 m1, 0; mov.b32 m2, 0; mov.b32 m3, 0;\n\t"
        "mov.b32 m4, 0; mov.b32 m5, 0; mov.b32 m6, 0; mov.b32 m7, 0;\n\t"
        "mov.b32 dhi, 0x40004040;\n\t"
        "mov.b64 da, {%0, dhi};\n\t"
        "mov.b64 db, {%1, dhi};\n\t"
        "@leader tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, %3, "
        "{m0, m1, m2, m3, m4, m5, m6, m7}, p;\n\t"
        "}\n"
        :: "r"(a_lo), "r"(b_lo), "r"(taddr), "r"(i_desc), "r"(enable_d));
}

__device__ __forceinline__ void elect_commit_cg2_multicast(int mbar_addr, uint16_t cta_mask) {
    asm volatile(
        "{\n\t"
        ".reg .pred leader;\n\t"
        "elect.sync _|leader, 0xFFFFFFFF;\n\t"
        "@leader tcgen05.commit.cta_group::2.mbarrier::arrive::one"
        ".shared::cluster.multicast::cluster.b64 [%0], %1;\n\t"
        "}\n"
        :: "r"(mbar_addr), "h"(cta_mask) : "memory");
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

__device__ __forceinline__ void tmem_ld_x16(float* dst, int tmem_addr) {
    asm volatile(
        "tcgen05.ld.sync.aligned.32x32b.x16.b32"
        " {%0, %1, %2, %3, %4, %5, %6, %7,"
        "  %8, %9, %10, %11, %12, %13, %14, %15}, [%16];"
        : "=f"(dst[0]),  "=f"(dst[1]),  "=f"(dst[2]),  "=f"(dst[3]),
          "=f"(dst[4]),  "=f"(dst[5]),  "=f"(dst[6]),  "=f"(dst[7]),
          "=f"(dst[8]),  "=f"(dst[9]),  "=f"(dst[10]), "=f"(dst[11]),
          "=f"(dst[12]), "=f"(dst[13]), "=f"(dst[14]), "=f"(dst[15])
        : "r"(tmem_addr));
}

__device__ __forceinline__ uint32_t smem_addr(const void* ptr) {
    uint32_t addr;
    asm("{\n\t"
        ".reg .u64 u64addr;\n\t"
        "cvta.to.shared.u64 u64addr, %1;\n\t"
        "cvt.u32.u64 %0, u64addr;\n\t"
        "}\n" : "=r"(addr) : "l"(ptr));
    return addr;
}

__device__ __forceinline__ uint32_t mapa_to_rank(uint32_t local_addr, uint32_t rank) {
    uint32_t remote;
    asm volatile("mapa.shared::cluster.u32 %0, %1, %2;"
        : "=r"(remote) : "r"(local_addr), "r"(rank));
    return remote;
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

__device__ __forceinline__ uint64_t make_smem_desc(int addr) {
    const int SBO = 1024;
    return desc_encode(addr)
         | (desc_encode(SBO) << 32ULL)
         | (1ULL << 46ULL)
         | (2ULL << 61ULL);
}

__device__ __forceinline__ void tma_3d_gmem2smem_cta2(
    int dst, const void *tmap_ptr, int x, int y, int z, int mbar_addr) {
    asm volatile(
        "cp.async.bulk.tensor.3d.shared::cluster.global"
        ".mbarrier::complete_tx::bytes.cta_group::2"
        " [%0], [%1, {%2, %3, %4}], [%5];"
        :: "r"(dst), "l"(tmap_ptr), "r"(x), "r"(y), "r"(z),
           "r"(mbar_addr) : "memory");
}

__device__ __forceinline__ void tcgen05_commit_cg2_multicast(int mbar_addr, uint16_t cta_mask) {
    asm volatile(
        "{\n\t"
        ".reg .b16 lo, hi;\n\t"
        "mov.b32 {lo, hi}, %1;\n\t"
        "tcgen05.commit.cta_group::2.mbarrier::arrive::one"
        ".shared::cluster.multicast::cluster.b64 [%0], lo;\n\t"
        "}\n"
        :: "r"(mbar_addr), "r"((uint32_t)cta_mask) : "memory");
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

__device__ __forceinline__ float max_noftz(float a, float b) {
    float c;
    asm("max.f32 %0, %1, %2;" : "=f"(c) : "f"(a), "f"(b));
    return c;
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

#define TMEM_NCOLS 64
#define TMEM_CROSS_OFFSET 0
#define NUM_MAIN_STAGES 1
#define SMEM_SMEM_QUERY_OFF 1024
#define SMEM_SMEM_QUERY_STAGE_BYTES 32768
#define SMEM_SMEM_QUERY_STRIDE 32768
#define SMEM_SMEM_DATABASE_OFF 33792
#define SMEM_SMEM_DATABASE_STAGE_BYTES 16384
#define SMEM_SMEM_DATABASE_STRIDE 16384
#define SMEM_SMEM_DATABASE_SQ_OFF 50176
#define SMEM_SMEM_DATABASE_SQ_STAGE_BYTES 256
#define SMEM_SMEM_DATABASE_SQ_STRIDE 256
#define SMEM_TOTAL 50432
#define THREADS 192
#define BLOCK_Q 128
#define BLOCK_M 64
#define FEAT_D 128
#define TOP_K_MAX 32

extern "C" {

__global__ __launch_bounds__(192, 1) void
kernel_knn_build_evolve_7bfc_split_cg2_u2_stage1_k32_unordered(float* __restrict__ query_sq, float* __restrict__ database_sq, float* __restrict__ partial_dists, int32_t* __restrict__ partial_indices, const void* tmap_query, const void* tmap_database, int B, int Q, int M, int K, int num_q_tile_pairs, int db_tiles_per_split, int split_count, int total_work)
{
    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;

    extern __shared__ __align__(1024) char smem_raw[];
    int smem;
    smem = (int)(unsigned long long)__cvta_generic_to_shared(smem_raw);
    const int smem_smem_query = smem + 1024;
    const int smem_smem_database = smem + 33792;
    const int smem_smem_database_sq = smem + 50176;

    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;
    const unsigned int clusters_x = gridDim.x / 2;
    const unsigned int cluster_id = ((blockIdx.z * gridDim.y + blockIdx.y) * clusters_x) + blockIdx.x / 2;
    const unsigned int num_clusters = clusters_x * gridDim.y * gridDim.z;

    int cta_rank;
    asm volatile("mov.b32 %0, %%cluster_ctarank;" : "=r"(cta_rank));

    // Mbarrier init (6 groups, 6 barriers)
    // Mbarriers at smem_raw[0..48)

    if (warp == 0) {
        uint32_t leader = elect_sync();
        // query_full: 1 barriers, init_count=2
        mbarrier_init_pred(smem + 0, 2, leader);
        // query_empty: 1 barriers, init_count=1
        mbarrier_init_pred(smem + 8, 1, leader);
        // database_full: 1 barriers, init_count=2
        mbarrier_init_pred(smem + 16, 2, leader);
        // database_empty: 1 barriers, init_count=1
        mbarrier_init_pred(smem + 24, 1, leader);
        // score_full: 1 barriers, init_count=1
        mbarrier_init_pred(smem + 32, 1, leader);
        // score_empty: 1 barriers, init_count=8
        mbarrier_init_pred(smem + 40, 8, leader);
        asm volatile("fence.mbarrier_init.release.cluster;");
    }

    __syncthreads();

    // TMEM alloc (64 columns, 64 used)
    volatile int* tmem_addr_storage = (volatile int*)(smem_raw + 48);
    if (warp == 1) {
        int _tmem_hold = smem + 48;
        asm volatile("tcgen05.alloc.cta_group::2.sync.aligned.shared::cta.b32 [%0], %1;" :: "r"(_tmem_hold), "r"(64) : "memory");
    }

    asm volatile("barrier.cluster.arrive.release.aligned;");
    asm volatile("barrier.cluster.wait.acquire.aligned;");
    asm volatile("tcgen05.fence::after_thread_sync;");

    const int warp_id = warp;
    const int lane_id = lane;
    __nv_bfloat16* smem_query = (__nv_bfloat16*)(smem_raw + 1024);
    #define smem_query_addr (smem + 1024)
    __nv_bfloat16* smem_database = (__nv_bfloat16*)(smem_raw + 33792);
    #define smem_database_addr (smem + 33792)
    float* smem_database_sq = (float*)(smem_raw + 50176);
    #define smem_database_sq_addr (smem + 50176)
    const int mbar_base = smem;
    #define query_full_addr (mbar_base + 0)
    #define query_empty_addr (mbar_base + 8)
    #define database_full_addr (mbar_base + 16)
    #define database_empty_addr (mbar_base + 24)
    #define score_full_addr (mbar_base + 32)
    #define score_empty_addr (mbar_base + 40)
    const int taddr = tmem_addr_storage[0];

    // ---- Role: load ----
    if (warp == 0) {
        { // load_main
            uint32_t _phase_query_empty_0 = 1;
            uint32_t _phase_database_empty_0 = 1;
            if (warp_id == 0) {
                if (elect_sync()) {
                    #pragma unroll 1
                    for (unsigned int work_idx = cluster_id; work_idx < total_work; work_idx += num_clusters) {
                        int split_idx = work_idx % split_count;
                        int query_work = work_idx / split_count;
                        int batch_idx = query_work / num_q_tile_pairs;
                        int q_tile_pair = query_work % num_q_tile_pairs;
                        int q_tile = q_tile_pair * 2 + cta_rank;
                        int off_q = q_tile * BLOCK_Q;
                        int global_q = batch_idx * Q + off_q;
                        int db_tile_start = split_idx * db_tiles_per_split;
                        mbarrier_wait(query_empty_addr, _phase_query_empty_0);
                        _phase_query_empty_0 ^= 1;
                        asm volatile(
                            "mbarrier.arrive.expect_tx.release.cta.shared::cluster.b64 _, [%0], %1;"
                            :: "r"((query_full_addr) & 0xFEFFFFFF), "r"((uint32_t)(32768)) : "memory");
                        asm volatile(
                            "cp.async.bulk.tensor.3d.shared::cluster.global.mbarrier::complete_tx::bytes.cta_group::2"
                            " [%0], [%1, {%2, %3, %4}], [%5];"
                            :: "r"(smem_query_addr), "l"(tmap_query), "r"(0), "r"(global_q), "r"(0),
                               "r"(((query_full_addr) & 0xFEFFFFFF)) : "memory");
                        #pragma unroll 1
                        for (int local_db_tile = 0; local_db_tile < db_tiles_per_split; local_db_tile++) {
                            int db_tile = db_tile_start + local_db_tile;
                            int off_m = db_tile * BLOCK_M;
                            int global_m = batch_idx * M + off_m;
                            mbarrier_wait(database_empty_addr, _phase_database_empty_0);
                            _phase_database_empty_0 ^= 1;
                            asm volatile(
                                "mbarrier.arrive.expect_tx.release.cta.shared::cluster.b64 _, [%0], %1;"
                                :: "r"((database_full_addr) & 0xFEFFFFFF), "r"((uint32_t)(16384)) : "memory");
                            asm volatile(
                                "cp.async.bulk.tensor.3d.shared::cluster.global.mbarrier::complete_tx::bytes.cta_group::2"
                                " [%0], [%1, {%2, %3, %4}], [%5];"
                                :: "r"(smem_database_addr), "l"(tmap_database), "r"(0), "r"(global_m), "r"(0),
                                   "r"(((database_full_addr) & 0xFEFFFFFF)) : "memory");
                        }
                    }
                }
            }
        }
    // ---- Role: mma ----
    } else if (warp == 1) {
        { // mma_main
            uint32_t _phase_query_full_0 = 0;
            uint32_t _phase_score_empty_0 = 1;
            uint32_t _phase_database_full_0 = 0;
            if (cta_rank == 0) {
                #pragma unroll 1
                for (unsigned int work_idx = cluster_id; work_idx < total_work; work_idx += num_clusters) {
                    mbarrier_wait(query_full_addr, _phase_query_full_0);
                    _phase_query_full_0 ^= 1;
                    #pragma unroll 1
                    for (int _local_db_tile = 0; _local_db_tile < db_tiles_per_split; _local_db_tile++) {
                        mbarrier_wait(score_empty_addr, _phase_score_empty_0);
                        _phase_score_empty_0 ^= 1;
                        mbarrier_wait(database_full_addr, _phase_database_full_0);
                        _phase_database_full_0 ^= 1;
                        asm volatile("tcgen05.fence::after_thread_sync;");
                        int _mma_ss_a_lo_0 = (smem_query_addr >> 4) & 0x3FFF;
                        int _mma_ss_b_lo_0 = (smem_database_addr >> 4) & 0x3FFF;
                        asm volatile(
                    "{\n\t"
                    ".reg .pred leader, p0, p1;\n\t"
                    ".reg .b32 adhi, bdhi, alo, blo, id, m0, m1, m2, m3, m4, m5, m6, m7;\n\t"
                    ".reg .b64 da, db;\n\t"
                    "elect.sync _|leader, 0xFFFFFFFF;\n\t"
                    "setp.ne.b32 p0, %3, 0;\n\t"
                    "setp.ne.b32 p1, 1, 0;\n\t"
                    "mov.b32 m0, 0; mov.b32 m1, 0; mov.b32 m2, 0; mov.b32 m3, 0;\n\tmov.b32 m4, 0; mov.b32 m5, 0; mov.b32 m6, 0; mov.b32 m7, 0;\n\t"
                    "mov.b32 adhi, 0x40004040;\n\t"
                    "mov.b32 bdhi, 0x40004040;\n\t"
                    "mov.b32 id, 270533776;\n\t"
                    "mov.b32 alo, %0;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p0;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "add.u32 alo, alo, 1018;\n\t"
                    "add.u32 blo, blo, 506;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "}\n"
                    :: "r"(_mma_ss_a_lo_0), "r"(_mma_ss_b_lo_0), "r"(taddr), "r"(0));
                        elect_commit_cg2_multicast(score_full_addr, (uint16_t)(3));
                        elect_commit_cg2_multicast(database_empty_addr, (uint16_t)(3));
                    }
                    elect_commit_cg2_multicast(query_empty_addr, (uint16_t)(3));
                }
            }
        }
    // ---- Role: compute ----
    } else if (warp >= 2 && warp <= 5) {
        const int warp_id_in_wg = warp % 4;
        const int my_row = warp_id_in_wg * 32 + lane;
        const int tmem_row_base_v = (warp_id_in_wg * 32) << 16;
        const int tmem_cross = taddr + TMEM_CROSS_OFFSET;
        { // compute_main
            uint32_t _phase_score_full_0 = 0;
            #pragma unroll 1
            for (unsigned int work_idx = cluster_id; work_idx < total_work; work_idx += num_clusters) {
                int split_idx = work_idx % split_count;
                int query_work = work_idx / split_count;
                int batch_idx = query_work / num_q_tile_pairs;
                int q_tile_pair = query_work % num_q_tile_pairs;
                int q_tile = q_tile_pair * 2 + cta_rank;
                int off_q = q_tile * BLOCK_Q;
                int q_idx = off_q + my_row;
                int valid_q = ((q_idx < Q) ? 1 : 0);
                float q_sq_val = 0.0f;
                if (valid_q != 0) {
                    q_sq_val = (float)query_sq[batch_idx * Q + q_idx];
                }
                float best_d[TOP_K_MAX];
                int best_i[TOP_K_MAX];
                #pragma unroll
                for (int kk = 0; kk < TOP_K_MAX; kk++) {
                    best_d[kk] = 3.4e+38f;
                    best_i[kk] = -1;
                }
                float worst_d = 3.4e+38f;
                int worst_pos = 0;
                int db_tile_start = split_idx * db_tiles_per_split;
                #pragma unroll 1
                for (int local_db_tile = 0; local_db_tile < db_tiles_per_split; local_db_tile++) {
                    int db_tile = db_tile_start + local_db_tile;
                    int db_start = db_tile * BLOCK_M;
                    int db_sq_idx = db_start + my_row;
                    if (my_row < BLOCK_M) {
                        if (db_sq_idx < M) {
                            smem_database_sq[my_row] = (float)database_sq[batch_idx * M + db_sq_idx];
                        } else {
                            smem_database_sq[my_row] = 0.0f;
                        }
                    }
                    asm volatile("barrier.sync 8, 128;");
                    mbarrier_wait(score_full_addr, _phase_score_full_0);
                    _phase_score_full_0 ^= 1;
                    int cross_addr = taddr + (cta_rank * BLOCK_Q + tmem_row_base_v << 16);
                    float dots[64];
                    asm volatile(
                        "tcgen05.ld.sync.aligned.32x32b.x64.b32"
                        " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63}, [%64];"
                        : "=f"(dots[0]), "=f"(dots[1]), "=f"(dots[2]), "=f"(dots[3]), "=f"(dots[4]), "=f"(dots[5]), "=f"(dots[6]), "=f"(dots[7]), "=f"(dots[8]), "=f"(dots[9]), "=f"(dots[10]), "=f"(dots[11]), "=f"(dots[12]), "=f"(dots[13]), "=f"(dots[14]), "=f"(dots[15]), "=f"(dots[16]), "=f"(dots[17]), "=f"(dots[18]), "=f"(dots[19]), "=f"(dots[20]), "=f"(dots[21]), "=f"(dots[22]), "=f"(dots[23]), "=f"(dots[24]), "=f"(dots[25]), "=f"(dots[26]), "=f"(dots[27]), "=f"(dots[28]), "=f"(dots[29]), "=f"(dots[30]), "=f"(dots[31]), "=f"(dots[32]), "=f"(dots[33]), "=f"(dots[34]), "=f"(dots[35]), "=f"(dots[36]), "=f"(dots[37]), "=f"(dots[38]), "=f"(dots[39]), "=f"(dots[40]), "=f"(dots[41]), "=f"(dots[42]), "=f"(dots[43]), "=f"(dots[44]), "=f"(dots[45]), "=f"(dots[46]), "=f"(dots[47]), "=f"(dots[48]), "=f"(dots[49]), "=f"(dots[50]), "=f"(dots[51]), "=f"(dots[52]), "=f"(dots[53]), "=f"(dots[54]), "=f"(dots[55]), "=f"(dots[56]), "=f"(dots[57]), "=f"(dots[58]), "=f"(dots[59]), "=f"(dots[60]), "=f"(dots[61]), "=f"(dots[62]), "=f"(dots[63])
                        : "r"(cross_addr)
                        : "memory");
                    asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                    asm volatile("barrier.sync 8, 128;");
                    if (elect_sync()) {
                        asm volatile(
                            "mbarrier.arrive.release.cta.shared::cluster.b64 _, [%0];"
                            :: "r"((score_empty_addr) & 0xFEFFFFFF) : "memory");
                    }
                    if (valid_q != 0) {
                        #pragma unroll 2
                        for (int col_base = 0; col_base < 64; col_base += 4) {
                            float dist_vec[4];
                            dist_vec[0] = dots[col_base];
                            dist_vec[1] = dots[col_base + 1];
                            dist_vec[2] = dots[col_base + 2];
                            dist_vec[3] = dots[col_base + 3];
                            const float2 _fma_b2_0 = {-2.0f, -2.0f};
                            const float2 _fma_c2_1 = {q_sq_val, q_sq_val};
                            #pragma unroll
                            for (int _lf = 0; _lf < 2; _lf++)
                                fma_f32x2_inplace(&reinterpret_cast<float2*>(dist_vec)[_lf], _fma_b2_0, _fma_c2_1);
                            float db_sq_vec[4];
                            db_sq_vec[0] = smem_database_sq[col_base];
                            db_sq_vec[1] = smem_database_sq[col_base + 1];
                            db_sq_vec[2] = smem_database_sq[col_base + 2];
                            db_sq_vec[3] = smem_database_sq[col_base + 3];
                            float _t0[4];
                            #pragma unroll
                            for (int _la = 0; _la < 2; _la++)
                                reinterpret_cast<float2*>(_t0)[_la] = add_f32x2(reinterpret_cast<float2*>(dist_vec)[_la], reinterpret_cast<const float2*>(db_sq_vec)[_la]);
                            float group_min = _t0[0];
                            if (_t0[1] < group_min) {
                                group_min = _t0[1];
                            }
                            if (_t0[2] < group_min) {
                                group_min = _t0[2];
                            }
                            if (_t0[3] < group_min) {
                                group_min = _t0[3];
                            }
                            if (group_min < worst_d) {
                                #pragma unroll
                                for (int vec_col = 0; vec_col < 4; vec_col++) {
                                    int db_idx = db_start + col_base + vec_col;
                                    if (db_idx < M) {
                                        float dist = _t0[vec_col];
                                        if (dist < worst_d) {
                                            best_d[worst_pos] = dist;
                                            best_i[worst_pos] = db_idx;
                                            worst_d = best_d[0];
                                            worst_pos = 0;
                                            #pragma unroll
                                            for (int scan_pos = 1; scan_pos < TOP_K_MAX; scan_pos++) {
                                                if (best_d[scan_pos] > worst_d) {
                                                    worst_d = best_d[scan_pos];
                                                    worst_pos = scan_pos;
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                    asm volatile("barrier.sync 8, 128;");
                }
                if (valid_q != 0) {
                    int out_base = ((split_idx * B + batch_idx) * Q + q_idx) * K;
                    #pragma unroll
                    for (int out_k = 0; out_k < TOP_K_MAX; out_k++) {
                        *((float*)(partial_dists + out_base + out_k)) = best_d[out_k];
                        *((int*)(partial_indices + out_base + out_k)) = best_i[out_k];
                    }
                }
            }
        }
    }

    // Cleanup
    asm volatile("barrier.cluster.arrive.release.aligned;");
    asm volatile("barrier.cluster.wait.acquire.aligned;");

    if (warp == 1) {
        asm volatile("tcgen05.dealloc.cta_group::2.sync.aligned.b32 %0, %1;" :: "r"(tmem_addr_storage[0]), "r"(64));
        asm volatile("tcgen05.relinquish_alloc_permit.cta_group::2.sync.aligned;");
    }
}

} // extern "C"

#undef BLOCK_M
#undef BLOCK_Q
#undef FEAT_D
#undef NUM_MAIN_STAGES
#undef SMEM_SMEM_DATABASE_OFF
#undef SMEM_SMEM_DATABASE_SQ_OFF
#undef SMEM_SMEM_DATABASE_SQ_STAGE_BYTES
#undef SMEM_SMEM_DATABASE_SQ_STRIDE
#undef SMEM_SMEM_DATABASE_STAGE_BYTES
#undef SMEM_SMEM_DATABASE_STRIDE
#undef SMEM_SMEM_QUERY_OFF
#undef SMEM_SMEM_QUERY_STAGE_BYTES
#undef SMEM_SMEM_QUERY_STRIDE
#undef SMEM_TOTAL
#undef THREADS
#undef TMEM_CROSS_OFFSET
#undef TMEM_NCOLS
#undef TOP_K_MAX
#undef database_empty_addr
#undef database_full_addr
#undef query_empty_addr
#undef query_full_addr
#undef score_empty_addr
#undef score_full_addr
#undef smem_database_addr
#undef smem_database_sq_addr
#undef smem_query_addr

#define TMEM_NCOLS 64
#define TMEM_CROSS_OFFSET 0
#define NUM_MAIN_STAGES 1
#define SMEM_SMEM_QUERY_OFF 1024
#define SMEM_SMEM_QUERY_STAGE_BYTES 32768
#define SMEM_SMEM_QUERY_STRIDE 32768
#define SMEM_SMEM_DATABASE_OFF 33792
#define SMEM_SMEM_DATABASE_STAGE_BYTES 16384
#define SMEM_SMEM_DATABASE_STRIDE 16384
#define SMEM_SMEM_DATABASE_SQ_OFF 50176
#define SMEM_SMEM_DATABASE_SQ_STAGE_BYTES 256
#define SMEM_SMEM_DATABASE_SQ_STRIDE 256
#define SMEM_TOTAL 50432
#define THREADS 192
#define BLOCK_Q 128
#define BLOCK_M 64
#define FEAT_D 128
#define TOP_K_MAX 12

extern "C" {

__global__ __launch_bounds__(192, 1) void
kernel_knn_build_evolve_7bfc_split_cg2_u2_stage1_k12split(float* __restrict__ query_sq, float* __restrict__ database_sq, float* __restrict__ partial_dists, int32_t* __restrict__ partial_indices, const void* tmap_query, const void* tmap_database, int B, int Q, int M, int K, int num_q_tile_pairs, int db_tiles_per_split, int split_count, int total_work)
{
    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;

    extern __shared__ __align__(1024) char smem_raw[];
    int smem;
    smem = (int)(unsigned long long)__cvta_generic_to_shared(smem_raw);
    const int smem_smem_query = smem + 1024;
    const int smem_smem_database = smem + 33792;
    const int smem_smem_database_sq = smem + 50176;

    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;
    const unsigned int clusters_x = gridDim.x / 2;
    const unsigned int cluster_id = ((blockIdx.z * gridDim.y + blockIdx.y) * clusters_x) + blockIdx.x / 2;
    const unsigned int num_clusters = clusters_x * gridDim.y * gridDim.z;

    int cta_rank;
    asm volatile("mov.b32 %0, %%cluster_ctarank;" : "=r"(cta_rank));

    // Mbarrier init (6 groups, 6 barriers)
    // Mbarriers at smem_raw[0..48)

    if (warp == 0) {
        uint32_t leader = elect_sync();
        // query_full: 1 barriers, init_count=2
        mbarrier_init_pred(smem + 0, 2, leader);
        // query_empty: 1 barriers, init_count=1
        mbarrier_init_pred(smem + 8, 1, leader);
        // database_full: 1 barriers, init_count=2
        mbarrier_init_pred(smem + 16, 2, leader);
        // database_empty: 1 barriers, init_count=1
        mbarrier_init_pred(smem + 24, 1, leader);
        // score_full: 1 barriers, init_count=1
        mbarrier_init_pred(smem + 32, 1, leader);
        // score_empty: 1 barriers, init_count=8
        mbarrier_init_pred(smem + 40, 8, leader);
        asm volatile("fence.mbarrier_init.release.cluster;");
    }

    __syncthreads();

    // TMEM alloc (64 columns, 64 used)
    volatile int* tmem_addr_storage = (volatile int*)(smem_raw + 48);
    if (warp == 1) {
        int _tmem_hold = smem + 48;
        asm volatile("tcgen05.alloc.cta_group::2.sync.aligned.shared::cta.b32 [%0], %1;" :: "r"(_tmem_hold), "r"(64) : "memory");
    }

    asm volatile("barrier.cluster.arrive.release.aligned;");
    asm volatile("barrier.cluster.wait.acquire.aligned;");
    asm volatile("tcgen05.fence::after_thread_sync;");

    const int warp_id = warp;
    const int lane_id = lane;
    __nv_bfloat16* smem_query = (__nv_bfloat16*)(smem_raw + 1024);
    #define smem_query_addr (smem + 1024)
    __nv_bfloat16* smem_database = (__nv_bfloat16*)(smem_raw + 33792);
    #define smem_database_addr (smem + 33792)
    float* smem_database_sq = (float*)(smem_raw + 50176);
    #define smem_database_sq_addr (smem + 50176)
    const int mbar_base = smem;
    #define query_full_addr (mbar_base + 0)
    #define query_empty_addr (mbar_base + 8)
    #define database_full_addr (mbar_base + 16)
    #define database_empty_addr (mbar_base + 24)
    #define score_full_addr (mbar_base + 32)
    #define score_empty_addr (mbar_base + 40)
    const int taddr = tmem_addr_storage[0];

    // ---- Role: load ----
    if (warp == 0) {
        { // load_main
            uint32_t _phase_query_empty_0 = 1;
            uint32_t _phase_database_empty_0 = 1;
            if (warp_id == 0) {
                if (elect_sync()) {
                    #pragma unroll 1
                    for (unsigned int work_idx = cluster_id; work_idx < total_work; work_idx += num_clusters) {
                        int split_idx = work_idx % split_count;
                        int query_work = work_idx / split_count;
                        int batch_idx = query_work / num_q_tile_pairs;
                        int q_tile_pair = query_work % num_q_tile_pairs;
                        int q_tile = q_tile_pair * 2 + cta_rank;
                        int off_q = q_tile * BLOCK_Q;
                        int global_q = batch_idx * Q + off_q;
                        int db_tile_start = split_idx * db_tiles_per_split;
                        mbarrier_wait(query_empty_addr, _phase_query_empty_0);
                        _phase_query_empty_0 ^= 1;
                        asm volatile(
                            "mbarrier.arrive.expect_tx.release.cta.shared::cluster.b64 _, [%0], %1;"
                            :: "r"((query_full_addr) & 0xFEFFFFFF), "r"((uint32_t)(32768)) : "memory");
                        asm volatile(
                            "cp.async.bulk.tensor.3d.shared::cluster.global.mbarrier::complete_tx::bytes.cta_group::2"
                            " [%0], [%1, {%2, %3, %4}], [%5];"
                            :: "r"(smem_query_addr), "l"(tmap_query), "r"(0), "r"(global_q), "r"(0),
                               "r"(((query_full_addr) & 0xFEFFFFFF)) : "memory");
                        #pragma unroll 1
                        for (int local_db_tile = 0; local_db_tile < db_tiles_per_split; local_db_tile++) {
                            int db_tile = db_tile_start + local_db_tile;
                            int off_m = db_tile * BLOCK_M;
                            int global_m = batch_idx * M + off_m;
                            mbarrier_wait(database_empty_addr, _phase_database_empty_0);
                            _phase_database_empty_0 ^= 1;
                            asm volatile(
                                "mbarrier.arrive.expect_tx.release.cta.shared::cluster.b64 _, [%0], %1;"
                                :: "r"((database_full_addr) & 0xFEFFFFFF), "r"((uint32_t)(16384)) : "memory");
                            asm volatile(
                                "cp.async.bulk.tensor.3d.shared::cluster.global.mbarrier::complete_tx::bytes.cta_group::2"
                                " [%0], [%1, {%2, %3, %4}], [%5];"
                                :: "r"(smem_database_addr), "l"(tmap_database), "r"(0), "r"(global_m), "r"(0),
                                   "r"(((database_full_addr) & 0xFEFFFFFF)) : "memory");
                        }
                    }
                }
            }
        }
    // ---- Role: mma ----
    } else if (warp == 1) {
        { // mma_main
            uint32_t _phase_query_full_0 = 0;
            uint32_t _phase_score_empty_0 = 1;
            uint32_t _phase_database_full_0 = 0;
            if (cta_rank == 0) {
                #pragma unroll 1
                for (unsigned int work_idx = cluster_id; work_idx < total_work; work_idx += num_clusters) {
                    mbarrier_wait(query_full_addr, _phase_query_full_0);
                    _phase_query_full_0 ^= 1;
                    #pragma unroll 1
                    for (int _local_db_tile = 0; _local_db_tile < db_tiles_per_split; _local_db_tile++) {
                        mbarrier_wait(score_empty_addr, _phase_score_empty_0);
                        _phase_score_empty_0 ^= 1;
                        mbarrier_wait(database_full_addr, _phase_database_full_0);
                        _phase_database_full_0 ^= 1;
                        asm volatile("tcgen05.fence::after_thread_sync;");
                        int _mma_ss_a_lo_0 = (smem_query_addr >> 4) & 0x3FFF;
                        int _mma_ss_b_lo_0 = (smem_database_addr >> 4) & 0x3FFF;
                        asm volatile(
                    "{\n\t"
                    ".reg .pred leader, p0, p1;\n\t"
                    ".reg .b32 adhi, bdhi, alo, blo, id, m0, m1, m2, m3, m4, m5, m6, m7;\n\t"
                    ".reg .b64 da, db;\n\t"
                    "elect.sync _|leader, 0xFFFFFFFF;\n\t"
                    "setp.ne.b32 p0, %3, 0;\n\t"
                    "setp.ne.b32 p1, 1, 0;\n\t"
                    "mov.b32 m0, 0; mov.b32 m1, 0; mov.b32 m2, 0; mov.b32 m3, 0;\n\tmov.b32 m4, 0; mov.b32 m5, 0; mov.b32 m6, 0; mov.b32 m7, 0;\n\t"
                    "mov.b32 adhi, 0x40004040;\n\t"
                    "mov.b32 bdhi, 0x40004040;\n\t"
                    "mov.b32 id, 270533776;\n\t"
                    "mov.b32 alo, %0;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p0;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "add.u32 alo, alo, 1018;\n\t"
                    "add.u32 blo, blo, 506;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "}\n"
                    :: "r"(_mma_ss_a_lo_0), "r"(_mma_ss_b_lo_0), "r"(taddr), "r"(0));
                        elect_commit_cg2_multicast(score_full_addr, (uint16_t)(3));
                        elect_commit_cg2_multicast(database_empty_addr, (uint16_t)(3));
                    }
                    elect_commit_cg2_multicast(query_empty_addr, (uint16_t)(3));
                }
            }
        }
    // ---- Role: compute ----
    } else if (warp >= 2 && warp <= 5) {
        const int warp_id_in_wg = warp % 4;
        const int my_row = warp_id_in_wg * 32 + lane;
        const int tmem_row_base_v = (warp_id_in_wg * 32) << 16;
        const int tmem_cross = taddr + TMEM_CROSS_OFFSET;
        { // compute_main
            uint32_t _phase_score_full_0 = 0;
            #pragma unroll 1
            for (unsigned int work_idx = cluster_id; work_idx < total_work; work_idx += num_clusters) {
                int split_idx = work_idx % split_count;
                int query_work = work_idx / split_count;
                int batch_idx = query_work / num_q_tile_pairs;
                int q_tile_pair = query_work % num_q_tile_pairs;
                int q_tile = q_tile_pair * 2 + cta_rank;
                int off_q = q_tile * BLOCK_Q;
                int q_idx = off_q + my_row;
                int valid_q = ((q_idx < Q) ? 1 : 0);
                float q_sq_val = 0.0f;
                if (valid_q != 0) {
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
                    int db_start = db_tile * BLOCK_M;
                    int db_sq_idx = db_start + my_row;
                    if (my_row < BLOCK_M) {
                        if (db_sq_idx < M) {
                            smem_database_sq[my_row] = (float)database_sq[batch_idx * M + db_sq_idx];
                        } else {
                            smem_database_sq[my_row] = 0.0f;
                        }
                    }
                    asm volatile("barrier.sync 8, 128;");
                    mbarrier_wait(score_full_addr, _phase_score_full_0);
                    _phase_score_full_0 ^= 1;
                    int cross_addr = taddr + (cta_rank * BLOCK_Q + tmem_row_base_v << 16);
                    float dots[64];
                    asm volatile(
                        "tcgen05.ld.sync.aligned.32x32b.x64.b32"
                        " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63}, [%64];"
                        : "=f"(dots[0]), "=f"(dots[1]), "=f"(dots[2]), "=f"(dots[3]), "=f"(dots[4]), "=f"(dots[5]), "=f"(dots[6]), "=f"(dots[7]), "=f"(dots[8]), "=f"(dots[9]), "=f"(dots[10]), "=f"(dots[11]), "=f"(dots[12]), "=f"(dots[13]), "=f"(dots[14]), "=f"(dots[15]), "=f"(dots[16]), "=f"(dots[17]), "=f"(dots[18]), "=f"(dots[19]), "=f"(dots[20]), "=f"(dots[21]), "=f"(dots[22]), "=f"(dots[23]), "=f"(dots[24]), "=f"(dots[25]), "=f"(dots[26]), "=f"(dots[27]), "=f"(dots[28]), "=f"(dots[29]), "=f"(dots[30]), "=f"(dots[31]), "=f"(dots[32]), "=f"(dots[33]), "=f"(dots[34]), "=f"(dots[35]), "=f"(dots[36]), "=f"(dots[37]), "=f"(dots[38]), "=f"(dots[39]), "=f"(dots[40]), "=f"(dots[41]), "=f"(dots[42]), "=f"(dots[43]), "=f"(dots[44]), "=f"(dots[45]), "=f"(dots[46]), "=f"(dots[47]), "=f"(dots[48]), "=f"(dots[49]), "=f"(dots[50]), "=f"(dots[51]), "=f"(dots[52]), "=f"(dots[53]), "=f"(dots[54]), "=f"(dots[55]), "=f"(dots[56]), "=f"(dots[57]), "=f"(dots[58]), "=f"(dots[59]), "=f"(dots[60]), "=f"(dots[61]), "=f"(dots[62]), "=f"(dots[63])
                        : "r"(cross_addr)
                        : "memory");
                    asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                    asm volatile("barrier.sync 8, 128;");
                    if (elect_sync()) {
                        asm volatile(
                            "mbarrier.arrive.release.cta.shared::cluster.b64 _, [%0];"
                            :: "r"((score_empty_addr) & 0xFEFFFFFF) : "memory");
                    }
                    if (valid_q != 0) {
                        #pragma unroll 2
                        for (int col_base = 0; col_base < 64; col_base += 4) {
                            float dist_vec[4];
                            dist_vec[0] = dots[col_base];
                            dist_vec[1] = dots[col_base + 1];
                            dist_vec[2] = dots[col_base + 2];
                            dist_vec[3] = dots[col_base + 3];
                            const float2 _fma_b2_0 = {-2.0f, -2.0f};
                            const float2 _fma_c2_1 = {q_sq_val, q_sq_val};
                            #pragma unroll
                            for (int _lf = 0; _lf < 2; _lf++)
                                fma_f32x2_inplace(&reinterpret_cast<float2*>(dist_vec)[_lf], _fma_b2_0, _fma_c2_1);
                            float db_sq_vec[4];
                            db_sq_vec[0] = smem_database_sq[col_base];
                            db_sq_vec[1] = smem_database_sq[col_base + 1];
                            db_sq_vec[2] = smem_database_sq[col_base + 2];
                            db_sq_vec[3] = smem_database_sq[col_base + 3];
                            float _t0[4];
                            #pragma unroll
                            for (int _la = 0; _la < 2; _la++)
                                reinterpret_cast<float2*>(_t0)[_la] = add_f32x2(reinterpret_cast<float2*>(dist_vec)[_la], reinterpret_cast<const float2*>(db_sq_vec)[_la]);
                            float group_min = _t0[0];
                            if (_t0[1] < group_min) {
                                group_min = _t0[1];
                            }
                            if (_t0[2] < group_min) {
                                group_min = _t0[2];
                            }
                            if (_t0[3] < group_min) {
                                group_min = _t0[3];
                            }
                            if (group_min < best_d[TOP_K_MAX - 1]) {
                                #pragma unroll
                                for (int vec_col = 0; vec_col < 4; vec_col++) {
                                    int db_idx = db_start + col_base + vec_col;
                                    if (db_idx < M) {
                                        float dist = _t0[vec_col];
                                        if (dist < best_d[TOP_K_MAX - 1]) {
                                            best_d[TOP_K_MAX - 1] = dist;
                                            best_i[TOP_K_MAX - 1] = db_idx;
                                            #pragma unroll
                                            for (int pos = TOP_K_MAX - 1; pos >= 1; pos--) {
                                                if (best_d[pos] < best_d[pos - 1]) {
                                                    float tmp_d = best_d[pos - 1];
                                                    int tmp_i = best_i[pos - 1];
                                                    best_d[pos - 1] = best_d[pos];
                                                    best_i[pos - 1] = best_i[pos];
                                                    best_d[pos] = tmp_d;
                                                    best_i[pos] = tmp_i;
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                    asm volatile("barrier.sync 8, 128;");
                }
                if (valid_q != 0) {
                    int out_base = ((split_idx * B + batch_idx) * Q + q_idx) * K;
                    #pragma unroll
                    for (int out_k = 0; out_k < TOP_K_MAX; out_k++) {
                        if (out_k < K) {
                            *((float*)(partial_dists + out_base + out_k)) = best_d[out_k];
                            *((int*)(partial_indices + out_base + out_k)) = best_i[out_k];
                        }
                    }
                }
            }
        }
    }

    // Cleanup
    asm volatile("barrier.cluster.arrive.release.aligned;");
    asm volatile("barrier.cluster.wait.acquire.aligned;");

    if (warp == 1) {
        asm volatile("tcgen05.dealloc.cta_group::2.sync.aligned.b32 %0, %1;" :: "r"(tmem_addr_storage[0]), "r"(64));
        asm volatile("tcgen05.relinquish_alloc_permit.cta_group::2.sync.aligned;");
    }
}

} // extern "C"

#undef BLOCK_M
#undef BLOCK_Q
#undef FEAT_D
#undef NUM_MAIN_STAGES
#undef SMEM_SMEM_DATABASE_OFF
#undef SMEM_SMEM_DATABASE_SQ_OFF
#undef SMEM_SMEM_DATABASE_SQ_STAGE_BYTES
#undef SMEM_SMEM_DATABASE_SQ_STRIDE
#undef SMEM_SMEM_DATABASE_STAGE_BYTES
#undef SMEM_SMEM_DATABASE_STRIDE
#undef SMEM_SMEM_QUERY_OFF
#undef SMEM_SMEM_QUERY_STAGE_BYTES
#undef SMEM_SMEM_QUERY_STRIDE
#undef SMEM_TOTAL
#undef THREADS
#undef TMEM_CROSS_OFFSET
#undef TMEM_NCOLS
#undef TOP_K_MAX
#undef database_empty_addr
#undef database_full_addr
#undef query_empty_addr
#undef query_full_addr
#undef score_empty_addr
#undef score_full_addr
#undef smem_database_addr
#undef smem_database_sq_addr
#undef smem_query_addr

#define TMEM_NCOLS 64
#define TMEM_CROSS_OFFSET 0
#define NUM_MAIN_STAGES 1
#define SMEM_SMEM_QUERY_OFF 1024
#define SMEM_SMEM_QUERY_STAGE_BYTES 32768
#define SMEM_SMEM_QUERY_STRIDE 32768
#define SMEM_SMEM_DATABASE_OFF 33792
#define SMEM_SMEM_DATABASE_STAGE_BYTES 16384
#define SMEM_SMEM_DATABASE_STRIDE 16384
#define SMEM_SMEM_DATABASE_SQ_OFF 50176
#define SMEM_SMEM_DATABASE_SQ_STAGE_BYTES 256
#define SMEM_SMEM_DATABASE_SQ_STRIDE 256
#define SMEM_TOTAL 50432
#define THREADS 192
#define BLOCK_Q 128
#define BLOCK_M 64
#define FEAT_D 128
#define TOP_K_MAX 16

extern "C" {

__global__ __launch_bounds__(192, 1) void
kernel_knn_build_evolve_7bfc_split_cg2_u2_stage1_k16split(float* __restrict__ query_sq, float* __restrict__ database_sq, float* __restrict__ partial_dists, int32_t* __restrict__ partial_indices, const void* tmap_query, const void* tmap_database, int B, int Q, int M, int K, int num_q_tile_pairs, int db_tiles_per_split, int split_count, int total_work)
{
    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;

    extern __shared__ __align__(1024) char smem_raw[];
    int smem;
    smem = (int)(unsigned long long)__cvta_generic_to_shared(smem_raw);
    const int smem_smem_query = smem + 1024;
    const int smem_smem_database = smem + 33792;
    const int smem_smem_database_sq = smem + 50176;

    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;
    const unsigned int clusters_x = gridDim.x / 2;
    const unsigned int cluster_id = ((blockIdx.z * gridDim.y + blockIdx.y) * clusters_x) + blockIdx.x / 2;
    const unsigned int num_clusters = clusters_x * gridDim.y * gridDim.z;

    int cta_rank;
    asm volatile("mov.b32 %0, %%cluster_ctarank;" : "=r"(cta_rank));

    // Mbarrier init (6 groups, 6 barriers)
    // Mbarriers at smem_raw[0..48)

    if (warp == 0) {
        uint32_t leader = elect_sync();
        // query_full: 1 barriers, init_count=2
        mbarrier_init_pred(smem + 0, 2, leader);
        // query_empty: 1 barriers, init_count=1
        mbarrier_init_pred(smem + 8, 1, leader);
        // database_full: 1 barriers, init_count=2
        mbarrier_init_pred(smem + 16, 2, leader);
        // database_empty: 1 barriers, init_count=1
        mbarrier_init_pred(smem + 24, 1, leader);
        // score_full: 1 barriers, init_count=1
        mbarrier_init_pred(smem + 32, 1, leader);
        // score_empty: 1 barriers, init_count=8
        mbarrier_init_pred(smem + 40, 8, leader);
        asm volatile("fence.mbarrier_init.release.cluster;");
    }

    __syncthreads();

    // TMEM alloc (64 columns, 64 used)
    volatile int* tmem_addr_storage = (volatile int*)(smem_raw + 48);
    if (warp == 1) {
        int _tmem_hold = smem + 48;
        asm volatile("tcgen05.alloc.cta_group::2.sync.aligned.shared::cta.b32 [%0], %1;" :: "r"(_tmem_hold), "r"(64) : "memory");
    }

    asm volatile("barrier.cluster.arrive.release.aligned;");
    asm volatile("barrier.cluster.wait.acquire.aligned;");
    asm volatile("tcgen05.fence::after_thread_sync;");

    const int warp_id = warp;
    const int lane_id = lane;
    __nv_bfloat16* smem_query = (__nv_bfloat16*)(smem_raw + 1024);
    #define smem_query_addr (smem + 1024)
    __nv_bfloat16* smem_database = (__nv_bfloat16*)(smem_raw + 33792);
    #define smem_database_addr (smem + 33792)
    float* smem_database_sq = (float*)(smem_raw + 50176);
    #define smem_database_sq_addr (smem + 50176)
    const int mbar_base = smem;
    #define query_full_addr (mbar_base + 0)
    #define query_empty_addr (mbar_base + 8)
    #define database_full_addr (mbar_base + 16)
    #define database_empty_addr (mbar_base + 24)
    #define score_full_addr (mbar_base + 32)
    #define score_empty_addr (mbar_base + 40)
    const int taddr = tmem_addr_storage[0];

    // ---- Role: load ----
    if (warp == 0) {
        { // load_main
            uint32_t _phase_query_empty_0 = 1;
            uint32_t _phase_database_empty_0 = 1;
            if (warp_id == 0) {
                if (elect_sync()) {
                    #pragma unroll 1
                    for (unsigned int work_idx = cluster_id; work_idx < total_work; work_idx += num_clusters) {
                        int split_idx = work_idx % split_count;
                        int query_work = work_idx / split_count;
                        int batch_idx = query_work / num_q_tile_pairs;
                        int q_tile_pair = query_work % num_q_tile_pairs;
                        int q_tile = q_tile_pair * 2 + cta_rank;
                        int off_q = q_tile * BLOCK_Q;
                        int global_q = batch_idx * Q + off_q;
                        int db_tile_start = split_idx * db_tiles_per_split;
                        mbarrier_wait(query_empty_addr, _phase_query_empty_0);
                        _phase_query_empty_0 ^= 1;
                        asm volatile(
                            "mbarrier.arrive.expect_tx.release.cta.shared::cluster.b64 _, [%0], %1;"
                            :: "r"((query_full_addr) & 0xFEFFFFFF), "r"((uint32_t)(32768)) : "memory");
                        asm volatile(
                            "cp.async.bulk.tensor.3d.shared::cluster.global.mbarrier::complete_tx::bytes.cta_group::2"
                            " [%0], [%1, {%2, %3, %4}], [%5];"
                            :: "r"(smem_query_addr), "l"(tmap_query), "r"(0), "r"(global_q), "r"(0),
                               "r"(((query_full_addr) & 0xFEFFFFFF)) : "memory");
                        #pragma unroll 1
                        for (int local_db_tile = 0; local_db_tile < db_tiles_per_split; local_db_tile++) {
                            int db_tile = db_tile_start + local_db_tile;
                            int off_m = db_tile * BLOCK_M;
                            int global_m = batch_idx * M + off_m;
                            mbarrier_wait(database_empty_addr, _phase_database_empty_0);
                            _phase_database_empty_0 ^= 1;
                            asm volatile(
                                "mbarrier.arrive.expect_tx.release.cta.shared::cluster.b64 _, [%0], %1;"
                                :: "r"((database_full_addr) & 0xFEFFFFFF), "r"((uint32_t)(16384)) : "memory");
                            asm volatile(
                                "cp.async.bulk.tensor.3d.shared::cluster.global.mbarrier::complete_tx::bytes.cta_group::2"
                                " [%0], [%1, {%2, %3, %4}], [%5];"
                                :: "r"(smem_database_addr), "l"(tmap_database), "r"(0), "r"(global_m), "r"(0),
                                   "r"(((database_full_addr) & 0xFEFFFFFF)) : "memory");
                        }
                    }
                }
            }
        }
    // ---- Role: mma ----
    } else if (warp == 1) {
        { // mma_main
            uint32_t _phase_query_full_0 = 0;
            uint32_t _phase_score_empty_0 = 1;
            uint32_t _phase_database_full_0 = 0;
            if (cta_rank == 0) {
                #pragma unroll 1
                for (unsigned int work_idx = cluster_id; work_idx < total_work; work_idx += num_clusters) {
                    mbarrier_wait(query_full_addr, _phase_query_full_0);
                    _phase_query_full_0 ^= 1;
                    #pragma unroll 1
                    for (int _local_db_tile = 0; _local_db_tile < db_tiles_per_split; _local_db_tile++) {
                        mbarrier_wait(score_empty_addr, _phase_score_empty_0);
                        _phase_score_empty_0 ^= 1;
                        mbarrier_wait(database_full_addr, _phase_database_full_0);
                        _phase_database_full_0 ^= 1;
                        asm volatile("tcgen05.fence::after_thread_sync;");
                        int _mma_ss_a_lo_0 = (smem_query_addr >> 4) & 0x3FFF;
                        int _mma_ss_b_lo_0 = (smem_database_addr >> 4) & 0x3FFF;
                        asm volatile(
                    "{\n\t"
                    ".reg .pred leader, p0, p1;\n\t"
                    ".reg .b32 adhi, bdhi, alo, blo, id, m0, m1, m2, m3, m4, m5, m6, m7;\n\t"
                    ".reg .b64 da, db;\n\t"
                    "elect.sync _|leader, 0xFFFFFFFF;\n\t"
                    "setp.ne.b32 p0, %3, 0;\n\t"
                    "setp.ne.b32 p1, 1, 0;\n\t"
                    "mov.b32 m0, 0; mov.b32 m1, 0; mov.b32 m2, 0; mov.b32 m3, 0;\n\tmov.b32 m4, 0; mov.b32 m5, 0; mov.b32 m6, 0; mov.b32 m7, 0;\n\t"
                    "mov.b32 adhi, 0x40004040;\n\t"
                    "mov.b32 bdhi, 0x40004040;\n\t"
                    "mov.b32 id, 270533776;\n\t"
                    "mov.b32 alo, %0;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p0;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "add.u32 alo, alo, 1018;\n\t"
                    "add.u32 blo, blo, 506;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "}\n"
                    :: "r"(_mma_ss_a_lo_0), "r"(_mma_ss_b_lo_0), "r"(taddr), "r"(0));
                        elect_commit_cg2_multicast(score_full_addr, (uint16_t)(3));
                        elect_commit_cg2_multicast(database_empty_addr, (uint16_t)(3));
                    }
                    elect_commit_cg2_multicast(query_empty_addr, (uint16_t)(3));
                }
            }
        }
    // ---- Role: compute ----
    } else if (warp >= 2 && warp <= 5) {
        const int warp_id_in_wg = warp % 4;
        const int my_row = warp_id_in_wg * 32 + lane;
        const int tmem_row_base_v = (warp_id_in_wg * 32) << 16;
        const int tmem_cross = taddr + TMEM_CROSS_OFFSET;
        { // compute_main
            uint32_t _phase_score_full_0 = 0;
            #pragma unroll 1
            for (unsigned int work_idx = cluster_id; work_idx < total_work; work_idx += num_clusters) {
                int split_idx = work_idx % split_count;
                int query_work = work_idx / split_count;
                int batch_idx = query_work / num_q_tile_pairs;
                int q_tile_pair = query_work % num_q_tile_pairs;
                int q_tile = q_tile_pair * 2 + cta_rank;
                int off_q = q_tile * BLOCK_Q;
                int q_idx = off_q + my_row;
                int valid_q = ((q_idx < Q) ? 1 : 0);
                float q_sq_val = 0.0f;
                if (valid_q != 0) {
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
                    int db_start = db_tile * BLOCK_M;
                    int db_sq_idx = db_start + my_row;
                    if (my_row < BLOCK_M) {
                        if (db_sq_idx < M) {
                            smem_database_sq[my_row] = (float)database_sq[batch_idx * M + db_sq_idx];
                        } else {
                            smem_database_sq[my_row] = 0.0f;
                        }
                    }
                    asm volatile("barrier.sync 8, 128;");
                    mbarrier_wait(score_full_addr, _phase_score_full_0);
                    _phase_score_full_0 ^= 1;
                    int cross_addr = taddr + (cta_rank * BLOCK_Q + tmem_row_base_v << 16);
                    float dots[64];
                    asm volatile(
                        "tcgen05.ld.sync.aligned.32x32b.x64.b32"
                        " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63}, [%64];"
                        : "=f"(dots[0]), "=f"(dots[1]), "=f"(dots[2]), "=f"(dots[3]), "=f"(dots[4]), "=f"(dots[5]), "=f"(dots[6]), "=f"(dots[7]), "=f"(dots[8]), "=f"(dots[9]), "=f"(dots[10]), "=f"(dots[11]), "=f"(dots[12]), "=f"(dots[13]), "=f"(dots[14]), "=f"(dots[15]), "=f"(dots[16]), "=f"(dots[17]), "=f"(dots[18]), "=f"(dots[19]), "=f"(dots[20]), "=f"(dots[21]), "=f"(dots[22]), "=f"(dots[23]), "=f"(dots[24]), "=f"(dots[25]), "=f"(dots[26]), "=f"(dots[27]), "=f"(dots[28]), "=f"(dots[29]), "=f"(dots[30]), "=f"(dots[31]), "=f"(dots[32]), "=f"(dots[33]), "=f"(dots[34]), "=f"(dots[35]), "=f"(dots[36]), "=f"(dots[37]), "=f"(dots[38]), "=f"(dots[39]), "=f"(dots[40]), "=f"(dots[41]), "=f"(dots[42]), "=f"(dots[43]), "=f"(dots[44]), "=f"(dots[45]), "=f"(dots[46]), "=f"(dots[47]), "=f"(dots[48]), "=f"(dots[49]), "=f"(dots[50]), "=f"(dots[51]), "=f"(dots[52]), "=f"(dots[53]), "=f"(dots[54]), "=f"(dots[55]), "=f"(dots[56]), "=f"(dots[57]), "=f"(dots[58]), "=f"(dots[59]), "=f"(dots[60]), "=f"(dots[61]), "=f"(dots[62]), "=f"(dots[63])
                        : "r"(cross_addr)
                        : "memory");
                    asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                    asm volatile("barrier.sync 8, 128;");
                    if (elect_sync()) {
                        asm volatile(
                            "mbarrier.arrive.release.cta.shared::cluster.b64 _, [%0];"
                            :: "r"((score_empty_addr) & 0xFEFFFFFF) : "memory");
                    }
                    if (valid_q != 0) {
                        #pragma unroll 2
                        for (int col_base = 0; col_base < 64; col_base += 4) {
                            float dist_vec[4];
                            dist_vec[0] = dots[col_base];
                            dist_vec[1] = dots[col_base + 1];
                            dist_vec[2] = dots[col_base + 2];
                            dist_vec[3] = dots[col_base + 3];
                            const float2 _fma_b2_0 = {-2.0f, -2.0f};
                            const float2 _fma_c2_1 = {q_sq_val, q_sq_val};
                            #pragma unroll
                            for (int _lf = 0; _lf < 2; _lf++)
                                fma_f32x2_inplace(&reinterpret_cast<float2*>(dist_vec)[_lf], _fma_b2_0, _fma_c2_1);
                            float db_sq_vec[4];
                            db_sq_vec[0] = smem_database_sq[col_base];
                            db_sq_vec[1] = smem_database_sq[col_base + 1];
                            db_sq_vec[2] = smem_database_sq[col_base + 2];
                            db_sq_vec[3] = smem_database_sq[col_base + 3];
                            float _t0[4];
                            #pragma unroll
                            for (int _la = 0; _la < 2; _la++)
                                reinterpret_cast<float2*>(_t0)[_la] = add_f32x2(reinterpret_cast<float2*>(dist_vec)[_la], reinterpret_cast<const float2*>(db_sq_vec)[_la]);
                            float group_min = _t0[0];
                            if (_t0[1] < group_min) {
                                group_min = _t0[1];
                            }
                            if (_t0[2] < group_min) {
                                group_min = _t0[2];
                            }
                            if (_t0[3] < group_min) {
                                group_min = _t0[3];
                            }
                            if (group_min < best_d[TOP_K_MAX - 1]) {
                                #pragma unroll
                                for (int vec_col = 0; vec_col < 4; vec_col++) {
                                    int db_idx = db_start + col_base + vec_col;
                                    if (db_idx < M) {
                                        float dist = _t0[vec_col];
                                        if (dist < best_d[TOP_K_MAX - 1]) {
                                            best_d[TOP_K_MAX - 1] = dist;
                                            best_i[TOP_K_MAX - 1] = db_idx;
                                            #pragma unroll
                                            for (int pos = TOP_K_MAX - 1; pos >= 1; pos--) {
                                                if (best_d[pos] < best_d[pos - 1]) {
                                                    float tmp_d = best_d[pos - 1];
                                                    int tmp_i = best_i[pos - 1];
                                                    best_d[pos - 1] = best_d[pos];
                                                    best_i[pos - 1] = best_i[pos];
                                                    best_d[pos] = tmp_d;
                                                    best_i[pos] = tmp_i;
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                    asm volatile("barrier.sync 8, 128;");
                }
                if (valid_q != 0) {
                    int out_base = ((split_idx * B + batch_idx) * Q + q_idx) * K;
                    #pragma unroll
                    for (int out_k = 0; out_k < TOP_K_MAX; out_k++) {
                        if (out_k < K) {
                            *((float*)(partial_dists + out_base + out_k)) = best_d[out_k];
                            *((int*)(partial_indices + out_base + out_k)) = best_i[out_k];
                        }
                    }
                }
            }
        }
    }

    // Cleanup
    asm volatile("barrier.cluster.arrive.release.aligned;");
    asm volatile("barrier.cluster.wait.acquire.aligned;");

    if (warp == 1) {
        asm volatile("tcgen05.dealloc.cta_group::2.sync.aligned.b32 %0, %1;" :: "r"(tmem_addr_storage[0]), "r"(64));
        asm volatile("tcgen05.relinquish_alloc_permit.cta_group::2.sync.aligned;");
    }
}

} // extern "C"

#undef BLOCK_M
#undef BLOCK_Q
#undef FEAT_D
#undef NUM_MAIN_STAGES
#undef SMEM_SMEM_DATABASE_OFF
#undef SMEM_SMEM_DATABASE_SQ_OFF
#undef SMEM_SMEM_DATABASE_SQ_STAGE_BYTES
#undef SMEM_SMEM_DATABASE_SQ_STRIDE
#undef SMEM_SMEM_DATABASE_STAGE_BYTES
#undef SMEM_SMEM_DATABASE_STRIDE
#undef SMEM_SMEM_QUERY_OFF
#undef SMEM_SMEM_QUERY_STAGE_BYTES
#undef SMEM_SMEM_QUERY_STRIDE
#undef SMEM_TOTAL
#undef THREADS
#undef TMEM_CROSS_OFFSET
#undef TMEM_NCOLS
#undef TOP_K_MAX
#undef database_empty_addr
#undef database_full_addr
#undef query_empty_addr
#undef query_full_addr
#undef score_empty_addr
#undef score_full_addr
#undef smem_database_addr
#undef smem_database_sq_addr
#undef smem_query_addr

#define TMEM_NCOLS 64
#define TMEM_CROSS_OFFSET 0
#define NUM_MAIN_STAGES 1
#define SMEM_SMEM_QUERY_OFF 1024
#define SMEM_SMEM_QUERY_STAGE_BYTES 32768
#define SMEM_SMEM_QUERY_STRIDE 32768
#define SMEM_SMEM_DATABASE_OFF 33792
#define SMEM_SMEM_DATABASE_STAGE_BYTES 16384
#define SMEM_SMEM_DATABASE_STRIDE 16384
#define SMEM_SMEM_DATABASE_SQ_OFF 50176
#define SMEM_SMEM_DATABASE_SQ_STAGE_BYTES 256
#define SMEM_SMEM_DATABASE_SQ_STRIDE 256
#define SMEM_TOTAL 50432
#define THREADS 192
#define BLOCK_Q 128
#define BLOCK_M 64
#define FEAT_D 128
#define TOP_K_MAX 20

extern "C" {

__global__ __launch_bounds__(192, 1) void
kernel_knn_build_evolve_7bfc_split_cg2_u2_stage1_k20split(float* __restrict__ query_sq, float* __restrict__ database_sq, float* __restrict__ partial_dists, int32_t* __restrict__ partial_indices, const void* tmap_query, const void* tmap_database, int B, int Q, int M, int K, int num_q_tile_pairs, int db_tiles_per_split, int split_count, int total_work)
{
    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;

    extern __shared__ __align__(1024) char smem_raw[];
    int smem;
    smem = (int)(unsigned long long)__cvta_generic_to_shared(smem_raw);
    const int smem_smem_query = smem + 1024;
    const int smem_smem_database = smem + 33792;
    const int smem_smem_database_sq = smem + 50176;

    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;
    const unsigned int clusters_x = gridDim.x / 2;
    const unsigned int cluster_id = ((blockIdx.z * gridDim.y + blockIdx.y) * clusters_x) + blockIdx.x / 2;
    const unsigned int num_clusters = clusters_x * gridDim.y * gridDim.z;

    int cta_rank;
    asm volatile("mov.b32 %0, %%cluster_ctarank;" : "=r"(cta_rank));

    // Mbarrier init (6 groups, 6 barriers)
    // Mbarriers at smem_raw[0..48)

    if (warp == 0) {
        uint32_t leader = elect_sync();
        // query_full: 1 barriers, init_count=2
        mbarrier_init_pred(smem + 0, 2, leader);
        // query_empty: 1 barriers, init_count=1
        mbarrier_init_pred(smem + 8, 1, leader);
        // database_full: 1 barriers, init_count=2
        mbarrier_init_pred(smem + 16, 2, leader);
        // database_empty: 1 barriers, init_count=1
        mbarrier_init_pred(smem + 24, 1, leader);
        // score_full: 1 barriers, init_count=1
        mbarrier_init_pred(smem + 32, 1, leader);
        // score_empty: 1 barriers, init_count=8
        mbarrier_init_pred(smem + 40, 8, leader);
        asm volatile("fence.mbarrier_init.release.cluster;");
    }

    __syncthreads();

    // TMEM alloc (64 columns, 64 used)
    volatile int* tmem_addr_storage = (volatile int*)(smem_raw + 48);
    if (warp == 1) {
        int _tmem_hold = smem + 48;
        asm volatile("tcgen05.alloc.cta_group::2.sync.aligned.shared::cta.b32 [%0], %1;" :: "r"(_tmem_hold), "r"(64) : "memory");
    }

    asm volatile("barrier.cluster.arrive.release.aligned;");
    asm volatile("barrier.cluster.wait.acquire.aligned;");
    asm volatile("tcgen05.fence::after_thread_sync;");

    const int warp_id = warp;
    const int lane_id = lane;
    __nv_bfloat16* smem_query = (__nv_bfloat16*)(smem_raw + 1024);
    #define smem_query_addr (smem + 1024)
    __nv_bfloat16* smem_database = (__nv_bfloat16*)(smem_raw + 33792);
    #define smem_database_addr (smem + 33792)
    float* smem_database_sq = (float*)(smem_raw + 50176);
    #define smem_database_sq_addr (smem + 50176)
    const int mbar_base = smem;
    #define query_full_addr (mbar_base + 0)
    #define query_empty_addr (mbar_base + 8)
    #define database_full_addr (mbar_base + 16)
    #define database_empty_addr (mbar_base + 24)
    #define score_full_addr (mbar_base + 32)
    #define score_empty_addr (mbar_base + 40)
    const int taddr = tmem_addr_storage[0];

    // ---- Role: load ----
    if (warp == 0) {
        { // load_main
            uint32_t _phase_query_empty_0 = 1;
            uint32_t _phase_database_empty_0 = 1;
            if (warp_id == 0) {
                if (elect_sync()) {
                    #pragma unroll 1
                    for (unsigned int work_idx = cluster_id; work_idx < total_work; work_idx += num_clusters) {
                        int split_idx = work_idx % split_count;
                        int query_work = work_idx / split_count;
                        int batch_idx = query_work / num_q_tile_pairs;
                        int q_tile_pair = query_work % num_q_tile_pairs;
                        int q_tile = q_tile_pair * 2 + cta_rank;
                        int off_q = q_tile * BLOCK_Q;
                        int global_q = batch_idx * Q + off_q;
                        int db_tile_start = split_idx * db_tiles_per_split;
                        mbarrier_wait(query_empty_addr, _phase_query_empty_0);
                        _phase_query_empty_0 ^= 1;
                        asm volatile(
                            "mbarrier.arrive.expect_tx.release.cta.shared::cluster.b64 _, [%0], %1;"
                            :: "r"((query_full_addr) & 0xFEFFFFFF), "r"((uint32_t)(32768)) : "memory");
                        asm volatile(
                            "cp.async.bulk.tensor.3d.shared::cluster.global.mbarrier::complete_tx::bytes.cta_group::2"
                            " [%0], [%1, {%2, %3, %4}], [%5];"
                            :: "r"(smem_query_addr), "l"(tmap_query), "r"(0), "r"(global_q), "r"(0),
                               "r"(((query_full_addr) & 0xFEFFFFFF)) : "memory");
                        #pragma unroll 1
                        for (int local_db_tile = 0; local_db_tile < db_tiles_per_split; local_db_tile++) {
                            int db_tile = db_tile_start + local_db_tile;
                            int off_m = db_tile * BLOCK_M;
                            int global_m = batch_idx * M + off_m;
                            mbarrier_wait(database_empty_addr, _phase_database_empty_0);
                            _phase_database_empty_0 ^= 1;
                            asm volatile(
                                "mbarrier.arrive.expect_tx.release.cta.shared::cluster.b64 _, [%0], %1;"
                                :: "r"((database_full_addr) & 0xFEFFFFFF), "r"((uint32_t)(16384)) : "memory");
                            asm volatile(
                                "cp.async.bulk.tensor.3d.shared::cluster.global.mbarrier::complete_tx::bytes.cta_group::2"
                                " [%0], [%1, {%2, %3, %4}], [%5];"
                                :: "r"(smem_database_addr), "l"(tmap_database), "r"(0), "r"(global_m), "r"(0),
                                   "r"(((database_full_addr) & 0xFEFFFFFF)) : "memory");
                        }
                    }
                }
            }
        }
    // ---- Role: mma ----
    } else if (warp == 1) {
        { // mma_main
            uint32_t _phase_query_full_0 = 0;
            uint32_t _phase_score_empty_0 = 1;
            uint32_t _phase_database_full_0 = 0;
            if (cta_rank == 0) {
                #pragma unroll 1
                for (unsigned int work_idx = cluster_id; work_idx < total_work; work_idx += num_clusters) {
                    mbarrier_wait(query_full_addr, _phase_query_full_0);
                    _phase_query_full_0 ^= 1;
                    #pragma unroll 1
                    for (int _local_db_tile = 0; _local_db_tile < db_tiles_per_split; _local_db_tile++) {
                        mbarrier_wait(score_empty_addr, _phase_score_empty_0);
                        _phase_score_empty_0 ^= 1;
                        mbarrier_wait(database_full_addr, _phase_database_full_0);
                        _phase_database_full_0 ^= 1;
                        asm volatile("tcgen05.fence::after_thread_sync;");
                        int _mma_ss_a_lo_0 = (smem_query_addr >> 4) & 0x3FFF;
                        int _mma_ss_b_lo_0 = (smem_database_addr >> 4) & 0x3FFF;
                        asm volatile(
                    "{\n\t"
                    ".reg .pred leader, p0, p1;\n\t"
                    ".reg .b32 adhi, bdhi, alo, blo, id, m0, m1, m2, m3, m4, m5, m6, m7;\n\t"
                    ".reg .b64 da, db;\n\t"
                    "elect.sync _|leader, 0xFFFFFFFF;\n\t"
                    "setp.ne.b32 p0, %3, 0;\n\t"
                    "setp.ne.b32 p1, 1, 0;\n\t"
                    "mov.b32 m0, 0; mov.b32 m1, 0; mov.b32 m2, 0; mov.b32 m3, 0;\n\tmov.b32 m4, 0; mov.b32 m5, 0; mov.b32 m6, 0; mov.b32 m7, 0;\n\t"
                    "mov.b32 adhi, 0x40004040;\n\t"
                    "mov.b32 bdhi, 0x40004040;\n\t"
                    "mov.b32 id, 270533776;\n\t"
                    "mov.b32 alo, %0;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p0;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "add.u32 alo, alo, 1018;\n\t"
                    "add.u32 blo, blo, 506;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "}\n"
                    :: "r"(_mma_ss_a_lo_0), "r"(_mma_ss_b_lo_0), "r"(taddr), "r"(0));
                        elect_commit_cg2_multicast(score_full_addr, (uint16_t)(3));
                        elect_commit_cg2_multicast(database_empty_addr, (uint16_t)(3));
                    }
                    elect_commit_cg2_multicast(query_empty_addr, (uint16_t)(3));
                }
            }
        }
    // ---- Role: compute ----
    } else if (warp >= 2 && warp <= 5) {
        const int warp_id_in_wg = warp % 4;
        const int my_row = warp_id_in_wg * 32 + lane;
        const int tmem_row_base_v = (warp_id_in_wg * 32) << 16;
        const int tmem_cross = taddr + TMEM_CROSS_OFFSET;
        { // compute_main
            uint32_t _phase_score_full_0 = 0;
            #pragma unroll 1
            for (unsigned int work_idx = cluster_id; work_idx < total_work; work_idx += num_clusters) {
                int split_idx = work_idx % split_count;
                int query_work = work_idx / split_count;
                int batch_idx = query_work / num_q_tile_pairs;
                int q_tile_pair = query_work % num_q_tile_pairs;
                int q_tile = q_tile_pair * 2 + cta_rank;
                int off_q = q_tile * BLOCK_Q;
                int q_idx = off_q + my_row;
                int valid_q = ((q_idx < Q) ? 1 : 0);
                float q_sq_val = 0.0f;
                if (valid_q != 0) {
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
                    int db_start = db_tile * BLOCK_M;
                    int db_sq_idx = db_start + my_row;
                    if (my_row < BLOCK_M) {
                        if (db_sq_idx < M) {
                            smem_database_sq[my_row] = (float)database_sq[batch_idx * M + db_sq_idx];
                        } else {
                            smem_database_sq[my_row] = 0.0f;
                        }
                    }
                    asm volatile("barrier.sync 8, 128;");
                    mbarrier_wait(score_full_addr, _phase_score_full_0);
                    _phase_score_full_0 ^= 1;
                    int cross_addr = taddr + (cta_rank * BLOCK_Q + tmem_row_base_v << 16);
                    float dots[64];
                    asm volatile(
                        "tcgen05.ld.sync.aligned.32x32b.x64.b32"
                        " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63}, [%64];"
                        : "=f"(dots[0]), "=f"(dots[1]), "=f"(dots[2]), "=f"(dots[3]), "=f"(dots[4]), "=f"(dots[5]), "=f"(dots[6]), "=f"(dots[7]), "=f"(dots[8]), "=f"(dots[9]), "=f"(dots[10]), "=f"(dots[11]), "=f"(dots[12]), "=f"(dots[13]), "=f"(dots[14]), "=f"(dots[15]), "=f"(dots[16]), "=f"(dots[17]), "=f"(dots[18]), "=f"(dots[19]), "=f"(dots[20]), "=f"(dots[21]), "=f"(dots[22]), "=f"(dots[23]), "=f"(dots[24]), "=f"(dots[25]), "=f"(dots[26]), "=f"(dots[27]), "=f"(dots[28]), "=f"(dots[29]), "=f"(dots[30]), "=f"(dots[31]), "=f"(dots[32]), "=f"(dots[33]), "=f"(dots[34]), "=f"(dots[35]), "=f"(dots[36]), "=f"(dots[37]), "=f"(dots[38]), "=f"(dots[39]), "=f"(dots[40]), "=f"(dots[41]), "=f"(dots[42]), "=f"(dots[43]), "=f"(dots[44]), "=f"(dots[45]), "=f"(dots[46]), "=f"(dots[47]), "=f"(dots[48]), "=f"(dots[49]), "=f"(dots[50]), "=f"(dots[51]), "=f"(dots[52]), "=f"(dots[53]), "=f"(dots[54]), "=f"(dots[55]), "=f"(dots[56]), "=f"(dots[57]), "=f"(dots[58]), "=f"(dots[59]), "=f"(dots[60]), "=f"(dots[61]), "=f"(dots[62]), "=f"(dots[63])
                        : "r"(cross_addr)
                        : "memory");
                    asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                    asm volatile("barrier.sync 8, 128;");
                    if (elect_sync()) {
                        asm volatile(
                            "mbarrier.arrive.release.cta.shared::cluster.b64 _, [%0];"
                            :: "r"((score_empty_addr) & 0xFEFFFFFF) : "memory");
                    }
                    if (valid_q != 0) {
                        #pragma unroll 2
                        for (int col_base = 0; col_base < 64; col_base += 4) {
                            float dist_vec[4];
                            dist_vec[0] = dots[col_base];
                            dist_vec[1] = dots[col_base + 1];
                            dist_vec[2] = dots[col_base + 2];
                            dist_vec[3] = dots[col_base + 3];
                            const float2 _fma_b2_0 = {-2.0f, -2.0f};
                            const float2 _fma_c2_1 = {q_sq_val, q_sq_val};
                            #pragma unroll
                            for (int _lf = 0; _lf < 2; _lf++)
                                fma_f32x2_inplace(&reinterpret_cast<float2*>(dist_vec)[_lf], _fma_b2_0, _fma_c2_1);
                            float db_sq_vec[4];
                            db_sq_vec[0] = smem_database_sq[col_base];
                            db_sq_vec[1] = smem_database_sq[col_base + 1];
                            db_sq_vec[2] = smem_database_sq[col_base + 2];
                            db_sq_vec[3] = smem_database_sq[col_base + 3];
                            float _t0[4];
                            #pragma unroll
                            for (int _la = 0; _la < 2; _la++)
                                reinterpret_cast<float2*>(_t0)[_la] = add_f32x2(reinterpret_cast<float2*>(dist_vec)[_la], reinterpret_cast<const float2*>(db_sq_vec)[_la]);
                            float group_min = _t0[0];
                            if (_t0[1] < group_min) {
                                group_min = _t0[1];
                            }
                            if (_t0[2] < group_min) {
                                group_min = _t0[2];
                            }
                            if (_t0[3] < group_min) {
                                group_min = _t0[3];
                            }
                            if (group_min < best_d[TOP_K_MAX - 1]) {
                                #pragma unroll
                                for (int vec_col = 0; vec_col < 4; vec_col++) {
                                    int db_idx = db_start + col_base + vec_col;
                                    if (db_idx < M) {
                                        float dist = _t0[vec_col];
                                        if (dist < best_d[TOP_K_MAX - 1]) {
                                            best_d[TOP_K_MAX - 1] = dist;
                                            best_i[TOP_K_MAX - 1] = db_idx;
                                            #pragma unroll
                                            for (int pos = TOP_K_MAX - 1; pos >= 1; pos--) {
                                                if (best_d[pos] < best_d[pos - 1]) {
                                                    float tmp_d = best_d[pos - 1];
                                                    int tmp_i = best_i[pos - 1];
                                                    best_d[pos - 1] = best_d[pos];
                                                    best_i[pos - 1] = best_i[pos];
                                                    best_d[pos] = tmp_d;
                                                    best_i[pos] = tmp_i;
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                    asm volatile("barrier.sync 8, 128;");
                }
                if (valid_q != 0) {
                    int out_base = ((split_idx * B + batch_idx) * Q + q_idx) * K;
                    #pragma unroll
                    for (int out_k = 0; out_k < TOP_K_MAX; out_k++) {
                        if (out_k < K) {
                            *((float*)(partial_dists + out_base + out_k)) = best_d[out_k];
                            *((int*)(partial_indices + out_base + out_k)) = best_i[out_k];
                        }
                    }
                }
            }
        }
    }

    // Cleanup
    asm volatile("barrier.cluster.arrive.release.aligned;");
    asm volatile("barrier.cluster.wait.acquire.aligned;");

    if (warp == 1) {
        asm volatile("tcgen05.dealloc.cta_group::2.sync.aligned.b32 %0, %1;" :: "r"(tmem_addr_storage[0]), "r"(64));
        asm volatile("tcgen05.relinquish_alloc_permit.cta_group::2.sync.aligned;");
    }
}

} // extern "C"

#undef BLOCK_M
#undef BLOCK_Q
#undef FEAT_D
#undef NUM_MAIN_STAGES
#undef SMEM_SMEM_DATABASE_OFF
#undef SMEM_SMEM_DATABASE_SQ_OFF
#undef SMEM_SMEM_DATABASE_SQ_STAGE_BYTES
#undef SMEM_SMEM_DATABASE_SQ_STRIDE
#undef SMEM_SMEM_DATABASE_STAGE_BYTES
#undef SMEM_SMEM_DATABASE_STRIDE
#undef SMEM_SMEM_QUERY_OFF
#undef SMEM_SMEM_QUERY_STAGE_BYTES
#undef SMEM_SMEM_QUERY_STRIDE
#undef SMEM_TOTAL
#undef THREADS
#undef TMEM_CROSS_OFFSET
#undef TMEM_NCOLS
#undef TOP_K_MAX
#undef database_empty_addr
#undef database_full_addr
#undef query_empty_addr
#undef query_full_addr
#undef score_empty_addr
#undef score_full_addr
#undef smem_database_addr
#undef smem_database_sq_addr
#undef smem_query_addr

#define TMEM_NCOLS 64
#define TMEM_CROSS_OFFSET 0
#define NUM_MAIN_STAGES 1
#define SMEM_SMEM_QUERY_OFF 1024
#define SMEM_SMEM_QUERY_STAGE_BYTES 32768
#define SMEM_SMEM_QUERY_STRIDE 32768
#define SMEM_SMEM_DATABASE_OFF 33792
#define SMEM_SMEM_DATABASE_STAGE_BYTES 16384
#define SMEM_SMEM_DATABASE_STRIDE 16384
#define SMEM_SMEM_DATABASE_SQ_OFF 50176
#define SMEM_SMEM_DATABASE_SQ_STAGE_BYTES 256
#define SMEM_SMEM_DATABASE_SQ_STRIDE 256
#define SMEM_TOTAL 50432
#define THREADS 192
#define BLOCK_Q 128
#define BLOCK_M 64
#define FEAT_D 128
#define TOP_K_MAX 25

extern "C" {

__global__ __launch_bounds__(192, 1) void
kernel_knn_build_evolve_7bfc_split_cg2_u2_stage1_k25split(float* __restrict__ query_sq, float* __restrict__ database_sq, float* __restrict__ partial_dists, int32_t* __restrict__ partial_indices, const void* tmap_query, const void* tmap_database, int B, int Q, int M, int K, int num_q_tile_pairs, int db_tiles_per_split, int split_count, int total_work)
{
    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;

    extern __shared__ __align__(1024) char smem_raw[];
    int smem;
    smem = (int)(unsigned long long)__cvta_generic_to_shared(smem_raw);
    const int smem_smem_query = smem + 1024;
    const int smem_smem_database = smem + 33792;
    const int smem_smem_database_sq = smem + 50176;

    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;
    const unsigned int clusters_x = gridDim.x / 2;
    const unsigned int cluster_id = ((blockIdx.z * gridDim.y + blockIdx.y) * clusters_x) + blockIdx.x / 2;
    const unsigned int num_clusters = clusters_x * gridDim.y * gridDim.z;

    int cta_rank;
    asm volatile("mov.b32 %0, %%cluster_ctarank;" : "=r"(cta_rank));

    // Mbarrier init (6 groups, 6 barriers)
    // Mbarriers at smem_raw[0..48)

    if (warp == 0) {
        uint32_t leader = elect_sync();
        // query_full: 1 barriers, init_count=2
        mbarrier_init_pred(smem + 0, 2, leader);
        // query_empty: 1 barriers, init_count=1
        mbarrier_init_pred(smem + 8, 1, leader);
        // database_full: 1 barriers, init_count=2
        mbarrier_init_pred(smem + 16, 2, leader);
        // database_empty: 1 barriers, init_count=1
        mbarrier_init_pred(smem + 24, 1, leader);
        // score_full: 1 barriers, init_count=1
        mbarrier_init_pred(smem + 32, 1, leader);
        // score_empty: 1 barriers, init_count=8
        mbarrier_init_pred(smem + 40, 8, leader);
        asm volatile("fence.mbarrier_init.release.cluster;");
    }

    __syncthreads();

    // TMEM alloc (64 columns, 64 used)
    volatile int* tmem_addr_storage = (volatile int*)(smem_raw + 48);
    if (warp == 1) {
        int _tmem_hold = smem + 48;
        asm volatile("tcgen05.alloc.cta_group::2.sync.aligned.shared::cta.b32 [%0], %1;" :: "r"(_tmem_hold), "r"(64) : "memory");
    }

    asm volatile("barrier.cluster.arrive.release.aligned;");
    asm volatile("barrier.cluster.wait.acquire.aligned;");
    asm volatile("tcgen05.fence::after_thread_sync;");

    const int warp_id = warp;
    const int lane_id = lane;
    __nv_bfloat16* smem_query = (__nv_bfloat16*)(smem_raw + 1024);
    #define smem_query_addr (smem + 1024)
    __nv_bfloat16* smem_database = (__nv_bfloat16*)(smem_raw + 33792);
    #define smem_database_addr (smem + 33792)
    float* smem_database_sq = (float*)(smem_raw + 50176);
    #define smem_database_sq_addr (smem + 50176)
    const int mbar_base = smem;
    #define query_full_addr (mbar_base + 0)
    #define query_empty_addr (mbar_base + 8)
    #define database_full_addr (mbar_base + 16)
    #define database_empty_addr (mbar_base + 24)
    #define score_full_addr (mbar_base + 32)
    #define score_empty_addr (mbar_base + 40)
    const int taddr = tmem_addr_storage[0];

    // ---- Role: load ----
    if (warp == 0) {
        { // load_main
            uint32_t _phase_query_empty_0 = 1;
            uint32_t _phase_database_empty_0 = 1;
            if (warp_id == 0) {
                if (elect_sync()) {
                    #pragma unroll 1
                    for (unsigned int work_idx = cluster_id; work_idx < total_work; work_idx += num_clusters) {
                        int split_idx = work_idx % split_count;
                        int query_work = work_idx / split_count;
                        int batch_idx = query_work / num_q_tile_pairs;
                        int q_tile_pair = query_work % num_q_tile_pairs;
                        int q_tile = q_tile_pair * 2 + cta_rank;
                        int off_q = q_tile * BLOCK_Q;
                        int global_q = batch_idx * Q + off_q;
                        int db_tile_start = split_idx * db_tiles_per_split;
                        mbarrier_wait(query_empty_addr, _phase_query_empty_0);
                        _phase_query_empty_0 ^= 1;
                        asm volatile(
                            "mbarrier.arrive.expect_tx.release.cta.shared::cluster.b64 _, [%0], %1;"
                            :: "r"((query_full_addr) & 0xFEFFFFFF), "r"((uint32_t)(32768)) : "memory");
                        asm volatile(
                            "cp.async.bulk.tensor.3d.shared::cluster.global.mbarrier::complete_tx::bytes.cta_group::2"
                            " [%0], [%1, {%2, %3, %4}], [%5];"
                            :: "r"(smem_query_addr), "l"(tmap_query), "r"(0), "r"(global_q), "r"(0),
                               "r"(((query_full_addr) & 0xFEFFFFFF)) : "memory");
                        #pragma unroll 1
                        for (int local_db_tile = 0; local_db_tile < db_tiles_per_split; local_db_tile++) {
                            int db_tile = db_tile_start + local_db_tile;
                            int off_m = db_tile * BLOCK_M;
                            int global_m = batch_idx * M + off_m;
                            mbarrier_wait(database_empty_addr, _phase_database_empty_0);
                            _phase_database_empty_0 ^= 1;
                            asm volatile(
                                "mbarrier.arrive.expect_tx.release.cta.shared::cluster.b64 _, [%0], %1;"
                                :: "r"((database_full_addr) & 0xFEFFFFFF), "r"((uint32_t)(16384)) : "memory");
                            asm volatile(
                                "cp.async.bulk.tensor.3d.shared::cluster.global.mbarrier::complete_tx::bytes.cta_group::2"
                                " [%0], [%1, {%2, %3, %4}], [%5];"
                                :: "r"(smem_database_addr), "l"(tmap_database), "r"(0), "r"(global_m), "r"(0),
                                   "r"(((database_full_addr) & 0xFEFFFFFF)) : "memory");
                        }
                    }
                }
            }
        }
    // ---- Role: mma ----
    } else if (warp == 1) {
        { // mma_main
            uint32_t _phase_query_full_0 = 0;
            uint32_t _phase_score_empty_0 = 1;
            uint32_t _phase_database_full_0 = 0;
            if (cta_rank == 0) {
                #pragma unroll 1
                for (unsigned int work_idx = cluster_id; work_idx < total_work; work_idx += num_clusters) {
                    mbarrier_wait(query_full_addr, _phase_query_full_0);
                    _phase_query_full_0 ^= 1;
                    #pragma unroll 1
                    for (int _local_db_tile = 0; _local_db_tile < db_tiles_per_split; _local_db_tile++) {
                        mbarrier_wait(score_empty_addr, _phase_score_empty_0);
                        _phase_score_empty_0 ^= 1;
                        mbarrier_wait(database_full_addr, _phase_database_full_0);
                        _phase_database_full_0 ^= 1;
                        asm volatile("tcgen05.fence::after_thread_sync;");
                        int _mma_ss_a_lo_0 = (smem_query_addr >> 4) & 0x3FFF;
                        int _mma_ss_b_lo_0 = (smem_database_addr >> 4) & 0x3FFF;
                        asm volatile(
                    "{\n\t"
                    ".reg .pred leader, p0, p1;\n\t"
                    ".reg .b32 adhi, bdhi, alo, blo, id, m0, m1, m2, m3, m4, m5, m6, m7;\n\t"
                    ".reg .b64 da, db;\n\t"
                    "elect.sync _|leader, 0xFFFFFFFF;\n\t"
                    "setp.ne.b32 p0, %3, 0;\n\t"
                    "setp.ne.b32 p1, 1, 0;\n\t"
                    "mov.b32 m0, 0; mov.b32 m1, 0; mov.b32 m2, 0; mov.b32 m3, 0;\n\tmov.b32 m4, 0; mov.b32 m5, 0; mov.b32 m6, 0; mov.b32 m7, 0;\n\t"
                    "mov.b32 adhi, 0x40004040;\n\t"
                    "mov.b32 bdhi, 0x40004040;\n\t"
                    "mov.b32 id, 270533776;\n\t"
                    "mov.b32 alo, %0;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p0;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "add.u32 alo, alo, 1018;\n\t"
                    "add.u32 blo, blo, 506;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "}\n"
                    :: "r"(_mma_ss_a_lo_0), "r"(_mma_ss_b_lo_0), "r"(taddr), "r"(0));
                        elect_commit_cg2_multicast(score_full_addr, (uint16_t)(3));
                        elect_commit_cg2_multicast(database_empty_addr, (uint16_t)(3));
                    }
                    elect_commit_cg2_multicast(query_empty_addr, (uint16_t)(3));
                }
            }
        }
    // ---- Role: compute ----
    } else if (warp >= 2 && warp <= 5) {
        const int warp_id_in_wg = warp % 4;
        const int my_row = warp_id_in_wg * 32 + lane;
        const int tmem_row_base_v = (warp_id_in_wg * 32) << 16;
        const int tmem_cross = taddr + TMEM_CROSS_OFFSET;
        { // compute_main
            uint32_t _phase_score_full_0 = 0;
            #pragma unroll 1
            for (unsigned int work_idx = cluster_id; work_idx < total_work; work_idx += num_clusters) {
                int split_idx = work_idx % split_count;
                int query_work = work_idx / split_count;
                int batch_idx = query_work / num_q_tile_pairs;
                int q_tile_pair = query_work % num_q_tile_pairs;
                int q_tile = q_tile_pair * 2 + cta_rank;
                int off_q = q_tile * BLOCK_Q;
                int q_idx = off_q + my_row;
                int valid_q = ((q_idx < Q) ? 1 : 0);
                float q_sq_val = 0.0f;
                if (valid_q != 0) {
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
                    int db_start = db_tile * BLOCK_M;
                    int db_sq_idx = db_start + my_row;
                    if (my_row < BLOCK_M) {
                        if (db_sq_idx < M) {
                            smem_database_sq[my_row] = (float)database_sq[batch_idx * M + db_sq_idx];
                        } else {
                            smem_database_sq[my_row] = 0.0f;
                        }
                    }
                    asm volatile("barrier.sync 8, 128;");
                    mbarrier_wait(score_full_addr, _phase_score_full_0);
                    _phase_score_full_0 ^= 1;
                    int cross_addr = taddr + (cta_rank * BLOCK_Q + tmem_row_base_v << 16);
                    float dots[64];
                    asm volatile(
                        "tcgen05.ld.sync.aligned.32x32b.x64.b32"
                        " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63}, [%64];"
                        : "=f"(dots[0]), "=f"(dots[1]), "=f"(dots[2]), "=f"(dots[3]), "=f"(dots[4]), "=f"(dots[5]), "=f"(dots[6]), "=f"(dots[7]), "=f"(dots[8]), "=f"(dots[9]), "=f"(dots[10]), "=f"(dots[11]), "=f"(dots[12]), "=f"(dots[13]), "=f"(dots[14]), "=f"(dots[15]), "=f"(dots[16]), "=f"(dots[17]), "=f"(dots[18]), "=f"(dots[19]), "=f"(dots[20]), "=f"(dots[21]), "=f"(dots[22]), "=f"(dots[23]), "=f"(dots[24]), "=f"(dots[25]), "=f"(dots[26]), "=f"(dots[27]), "=f"(dots[28]), "=f"(dots[29]), "=f"(dots[30]), "=f"(dots[31]), "=f"(dots[32]), "=f"(dots[33]), "=f"(dots[34]), "=f"(dots[35]), "=f"(dots[36]), "=f"(dots[37]), "=f"(dots[38]), "=f"(dots[39]), "=f"(dots[40]), "=f"(dots[41]), "=f"(dots[42]), "=f"(dots[43]), "=f"(dots[44]), "=f"(dots[45]), "=f"(dots[46]), "=f"(dots[47]), "=f"(dots[48]), "=f"(dots[49]), "=f"(dots[50]), "=f"(dots[51]), "=f"(dots[52]), "=f"(dots[53]), "=f"(dots[54]), "=f"(dots[55]), "=f"(dots[56]), "=f"(dots[57]), "=f"(dots[58]), "=f"(dots[59]), "=f"(dots[60]), "=f"(dots[61]), "=f"(dots[62]), "=f"(dots[63])
                        : "r"(cross_addr)
                        : "memory");
                    asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                    asm volatile("barrier.sync 8, 128;");
                    if (elect_sync()) {
                        asm volatile(
                            "mbarrier.arrive.release.cta.shared::cluster.b64 _, [%0];"
                            :: "r"((score_empty_addr) & 0xFEFFFFFF) : "memory");
                    }
                    if (valid_q != 0) {
                        #pragma unroll 2
                        for (int col_base = 0; col_base < 64; col_base += 4) {
                            float dist_vec[4];
                            dist_vec[0] = dots[col_base];
                            dist_vec[1] = dots[col_base + 1];
                            dist_vec[2] = dots[col_base + 2];
                            dist_vec[3] = dots[col_base + 3];
                            const float2 _fma_b2_0 = {-2.0f, -2.0f};
                            const float2 _fma_c2_1 = {q_sq_val, q_sq_val};
                            #pragma unroll
                            for (int _lf = 0; _lf < 2; _lf++)
                                fma_f32x2_inplace(&reinterpret_cast<float2*>(dist_vec)[_lf], _fma_b2_0, _fma_c2_1);
                            float db_sq_vec[4];
                            db_sq_vec[0] = smem_database_sq[col_base];
                            db_sq_vec[1] = smem_database_sq[col_base + 1];
                            db_sq_vec[2] = smem_database_sq[col_base + 2];
                            db_sq_vec[3] = smem_database_sq[col_base + 3];
                            float _t0[4];
                            #pragma unroll
                            for (int _la = 0; _la < 2; _la++)
                                reinterpret_cast<float2*>(_t0)[_la] = add_f32x2(reinterpret_cast<float2*>(dist_vec)[_la], reinterpret_cast<const float2*>(db_sq_vec)[_la]);
                            float group_min = _t0[0];
                            if (_t0[1] < group_min) {
                                group_min = _t0[1];
                            }
                            if (_t0[2] < group_min) {
                                group_min = _t0[2];
                            }
                            if (_t0[3] < group_min) {
                                group_min = _t0[3];
                            }
                            if (group_min < best_d[TOP_K_MAX - 1]) {
                                #pragma unroll
                                for (int vec_col = 0; vec_col < 4; vec_col++) {
                                    int db_idx = db_start + col_base + vec_col;
                                    if (db_idx < M) {
                                        float dist = _t0[vec_col];
                                        if (dist < best_d[TOP_K_MAX - 1]) {
                                            best_d[TOP_K_MAX - 1] = dist;
                                            best_i[TOP_K_MAX - 1] = db_idx;
                                            #pragma unroll
                                            for (int pos = TOP_K_MAX - 1; pos >= 1; pos--) {
                                                if (best_d[pos] < best_d[pos - 1]) {
                                                    float tmp_d = best_d[pos - 1];
                                                    int tmp_i = best_i[pos - 1];
                                                    best_d[pos - 1] = best_d[pos];
                                                    best_i[pos - 1] = best_i[pos];
                                                    best_d[pos] = tmp_d;
                                                    best_i[pos] = tmp_i;
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                    asm volatile("barrier.sync 8, 128;");
                }
                if (valid_q != 0) {
                    int out_base = ((split_idx * B + batch_idx) * Q + q_idx) * K;
                    #pragma unroll
                    for (int out_k = 0; out_k < TOP_K_MAX; out_k++) {
                        if (out_k < K) {
                            *((float*)(partial_dists + out_base + out_k)) = best_d[out_k];
                            *((int*)(partial_indices + out_base + out_k)) = best_i[out_k];
                        }
                    }
                }
            }
        }
    }

    // Cleanup
    asm volatile("barrier.cluster.arrive.release.aligned;");
    asm volatile("barrier.cluster.wait.acquire.aligned;");

    if (warp == 1) {
        asm volatile("tcgen05.dealloc.cta_group::2.sync.aligned.b32 %0, %1;" :: "r"(tmem_addr_storage[0]), "r"(64));
        asm volatile("tcgen05.relinquish_alloc_permit.cta_group::2.sync.aligned;");
    }
}

} // extern "C"

#undef BLOCK_M
#undef BLOCK_Q
#undef FEAT_D
#undef NUM_MAIN_STAGES
#undef SMEM_SMEM_DATABASE_OFF
#undef SMEM_SMEM_DATABASE_SQ_OFF
#undef SMEM_SMEM_DATABASE_SQ_STAGE_BYTES
#undef SMEM_SMEM_DATABASE_SQ_STRIDE
#undef SMEM_SMEM_DATABASE_STAGE_BYTES
#undef SMEM_SMEM_DATABASE_STRIDE
#undef SMEM_SMEM_QUERY_OFF
#undef SMEM_SMEM_QUERY_STAGE_BYTES
#undef SMEM_SMEM_QUERY_STRIDE
#undef SMEM_TOTAL
#undef THREADS
#undef TMEM_CROSS_OFFSET
#undef TMEM_NCOLS
#undef TOP_K_MAX
#undef database_empty_addr
#undef database_full_addr
#undef query_empty_addr
#undef query_full_addr
#undef score_empty_addr
#undef score_full_addr
#undef smem_database_addr
#undef smem_database_sq_addr
#undef smem_query_addr

#define TMEM_NCOLS 64
#define TMEM_CROSS_OFFSET 0
#define NUM_MAIN_STAGES 1
#define SMEM_SMEM_QUERY_OFF 1024
#define SMEM_SMEM_QUERY_STAGE_BYTES 32768
#define SMEM_SMEM_QUERY_STRIDE 32768
#define SMEM_SMEM_DATABASE_OFF 33792
#define SMEM_SMEM_DATABASE_STAGE_BYTES 16384
#define SMEM_SMEM_DATABASE_STRIDE 16384
#define SMEM_SMEM_DATABASE_SQ_OFF 50176
#define SMEM_SMEM_DATABASE_SQ_STAGE_BYTES 256
#define SMEM_SMEM_DATABASE_SQ_STRIDE 256
#define SMEM_TOTAL 50432
#define THREADS 192
#define BLOCK_Q 128
#define BLOCK_M 64
#define FEAT_D 128
#define TOP_K_MAX 30

extern "C" {

__global__ __launch_bounds__(192, 1) void
kernel_knn_build_evolve_7bfc_split_cg2_u2_stage1_k30split(float* __restrict__ query_sq, float* __restrict__ database_sq, float* __restrict__ partial_dists, int32_t* __restrict__ partial_indices, const void* tmap_query, const void* tmap_database, int B, int Q, int M, int K, int num_q_tile_pairs, int db_tiles_per_split, int split_count, int total_work)
{
    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;

    extern __shared__ __align__(1024) char smem_raw[];
    int smem;
    smem = (int)(unsigned long long)__cvta_generic_to_shared(smem_raw);
    const int smem_smem_query = smem + 1024;
    const int smem_smem_database = smem + 33792;
    const int smem_smem_database_sq = smem + 50176;

    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;
    const unsigned int clusters_x = gridDim.x / 2;
    const unsigned int cluster_id = ((blockIdx.z * gridDim.y + blockIdx.y) * clusters_x) + blockIdx.x / 2;
    const unsigned int num_clusters = clusters_x * gridDim.y * gridDim.z;

    int cta_rank;
    asm volatile("mov.b32 %0, %%cluster_ctarank;" : "=r"(cta_rank));

    // Mbarrier init (6 groups, 6 barriers)
    // Mbarriers at smem_raw[0..48)

    if (warp == 0) {
        uint32_t leader = elect_sync();
        // query_full: 1 barriers, init_count=2
        mbarrier_init_pred(smem + 0, 2, leader);
        // query_empty: 1 barriers, init_count=1
        mbarrier_init_pred(smem + 8, 1, leader);
        // database_full: 1 barriers, init_count=2
        mbarrier_init_pred(smem + 16, 2, leader);
        // database_empty: 1 barriers, init_count=1
        mbarrier_init_pred(smem + 24, 1, leader);
        // score_full: 1 barriers, init_count=1
        mbarrier_init_pred(smem + 32, 1, leader);
        // score_empty: 1 barriers, init_count=8
        mbarrier_init_pred(smem + 40, 8, leader);
        asm volatile("fence.mbarrier_init.release.cluster;");
    }

    __syncthreads();

    // TMEM alloc (64 columns, 64 used)
    volatile int* tmem_addr_storage = (volatile int*)(smem_raw + 48);
    if (warp == 1) {
        int _tmem_hold = smem + 48;
        asm volatile("tcgen05.alloc.cta_group::2.sync.aligned.shared::cta.b32 [%0], %1;" :: "r"(_tmem_hold), "r"(64) : "memory");
    }

    asm volatile("barrier.cluster.arrive.release.aligned;");
    asm volatile("barrier.cluster.wait.acquire.aligned;");
    asm volatile("tcgen05.fence::after_thread_sync;");

    const int warp_id = warp;
    const int lane_id = lane;
    __nv_bfloat16* smem_query = (__nv_bfloat16*)(smem_raw + 1024);
    #define smem_query_addr (smem + 1024)
    __nv_bfloat16* smem_database = (__nv_bfloat16*)(smem_raw + 33792);
    #define smem_database_addr (smem + 33792)
    float* smem_database_sq = (float*)(smem_raw + 50176);
    #define smem_database_sq_addr (smem + 50176)
    const int mbar_base = smem;
    #define query_full_addr (mbar_base + 0)
    #define query_empty_addr (mbar_base + 8)
    #define database_full_addr (mbar_base + 16)
    #define database_empty_addr (mbar_base + 24)
    #define score_full_addr (mbar_base + 32)
    #define score_empty_addr (mbar_base + 40)
    const int taddr = tmem_addr_storage[0];

    // ---- Role: load ----
    if (warp == 0) {
        { // load_main
            uint32_t _phase_query_empty_0 = 1;
            uint32_t _phase_database_empty_0 = 1;
            if (warp_id == 0) {
                if (elect_sync()) {
                    #pragma unroll 1
                    for (unsigned int work_idx = cluster_id; work_idx < total_work; work_idx += num_clusters) {
                        int split_idx = work_idx % split_count;
                        int query_work = work_idx / split_count;
                        int batch_idx = query_work / num_q_tile_pairs;
                        int q_tile_pair = query_work % num_q_tile_pairs;
                        int q_tile = q_tile_pair * 2 + cta_rank;
                        int off_q = q_tile * BLOCK_Q;
                        int global_q = batch_idx * Q + off_q;
                        int db_tile_start = split_idx * db_tiles_per_split;
                        mbarrier_wait(query_empty_addr, _phase_query_empty_0);
                        _phase_query_empty_0 ^= 1;
                        asm volatile(
                            "mbarrier.arrive.expect_tx.release.cta.shared::cluster.b64 _, [%0], %1;"
                            :: "r"((query_full_addr) & 0xFEFFFFFF), "r"((uint32_t)(32768)) : "memory");
                        asm volatile(
                            "cp.async.bulk.tensor.3d.shared::cluster.global.mbarrier::complete_tx::bytes.cta_group::2"
                            " [%0], [%1, {%2, %3, %4}], [%5];"
                            :: "r"(smem_query_addr), "l"(tmap_query), "r"(0), "r"(global_q), "r"(0),
                               "r"(((query_full_addr) & 0xFEFFFFFF)) : "memory");
                        #pragma unroll 1
                        for (int local_db_tile = 0; local_db_tile < db_tiles_per_split; local_db_tile++) {
                            int db_tile = db_tile_start + local_db_tile;
                            int off_m = db_tile * BLOCK_M;
                            int global_m = batch_idx * M + off_m;
                            mbarrier_wait(database_empty_addr, _phase_database_empty_0);
                            _phase_database_empty_0 ^= 1;
                            asm volatile(
                                "mbarrier.arrive.expect_tx.release.cta.shared::cluster.b64 _, [%0], %1;"
                                :: "r"((database_full_addr) & 0xFEFFFFFF), "r"((uint32_t)(16384)) : "memory");
                            asm volatile(
                                "cp.async.bulk.tensor.3d.shared::cluster.global.mbarrier::complete_tx::bytes.cta_group::2"
                                " [%0], [%1, {%2, %3, %4}], [%5];"
                                :: "r"(smem_database_addr), "l"(tmap_database), "r"(0), "r"(global_m), "r"(0),
                                   "r"(((database_full_addr) & 0xFEFFFFFF)) : "memory");
                        }
                    }
                }
            }
        }
    // ---- Role: mma ----
    } else if (warp == 1) {
        { // mma_main
            uint32_t _phase_query_full_0 = 0;
            uint32_t _phase_score_empty_0 = 1;
            uint32_t _phase_database_full_0 = 0;
            if (cta_rank == 0) {
                #pragma unroll 1
                for (unsigned int work_idx = cluster_id; work_idx < total_work; work_idx += num_clusters) {
                    mbarrier_wait(query_full_addr, _phase_query_full_0);
                    _phase_query_full_0 ^= 1;
                    #pragma unroll 1
                    for (int _local_db_tile = 0; _local_db_tile < db_tiles_per_split; _local_db_tile++) {
                        mbarrier_wait(score_empty_addr, _phase_score_empty_0);
                        _phase_score_empty_0 ^= 1;
                        mbarrier_wait(database_full_addr, _phase_database_full_0);
                        _phase_database_full_0 ^= 1;
                        asm volatile("tcgen05.fence::after_thread_sync;");
                        int _mma_ss_a_lo_0 = (smem_query_addr >> 4) & 0x3FFF;
                        int _mma_ss_b_lo_0 = (smem_database_addr >> 4) & 0x3FFF;
                        asm volatile(
                    "{\n\t"
                    ".reg .pred leader, p0, p1;\n\t"
                    ".reg .b32 adhi, bdhi, alo, blo, id, m0, m1, m2, m3, m4, m5, m6, m7;\n\t"
                    ".reg .b64 da, db;\n\t"
                    "elect.sync _|leader, 0xFFFFFFFF;\n\t"
                    "setp.ne.b32 p0, %3, 0;\n\t"
                    "setp.ne.b32 p1, 1, 0;\n\t"
                    "mov.b32 m0, 0; mov.b32 m1, 0; mov.b32 m2, 0; mov.b32 m3, 0;\n\tmov.b32 m4, 0; mov.b32 m5, 0; mov.b32 m6, 0; mov.b32 m7, 0;\n\t"
                    "mov.b32 adhi, 0x40004040;\n\t"
                    "mov.b32 bdhi, 0x40004040;\n\t"
                    "mov.b32 id, 270533776;\n\t"
                    "mov.b32 alo, %0;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p0;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "add.u32 alo, alo, 1018;\n\t"
                    "add.u32 blo, blo, 506;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "}\n"
                    :: "r"(_mma_ss_a_lo_0), "r"(_mma_ss_b_lo_0), "r"(taddr), "r"(0));
                        elect_commit_cg2_multicast(score_full_addr, (uint16_t)(3));
                        elect_commit_cg2_multicast(database_empty_addr, (uint16_t)(3));
                    }
                    elect_commit_cg2_multicast(query_empty_addr, (uint16_t)(3));
                }
            }
        }
    // ---- Role: compute ----
    } else if (warp >= 2 && warp <= 5) {
        const int warp_id_in_wg = warp % 4;
        const int my_row = warp_id_in_wg * 32 + lane;
        const int tmem_row_base_v = (warp_id_in_wg * 32) << 16;
        const int tmem_cross = taddr + TMEM_CROSS_OFFSET;
        { // compute_main
            uint32_t _phase_score_full_0 = 0;
            #pragma unroll 1
            for (unsigned int work_idx = cluster_id; work_idx < total_work; work_idx += num_clusters) {
                int split_idx = work_idx % split_count;
                int query_work = work_idx / split_count;
                int batch_idx = query_work / num_q_tile_pairs;
                int q_tile_pair = query_work % num_q_tile_pairs;
                int q_tile = q_tile_pair * 2 + cta_rank;
                int off_q = q_tile * BLOCK_Q;
                int q_idx = off_q + my_row;
                int valid_q = ((q_idx < Q) ? 1 : 0);
                float q_sq_val = 0.0f;
                if (valid_q != 0) {
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
                    int db_start = db_tile * BLOCK_M;
                    int db_sq_idx = db_start + my_row;
                    if (my_row < BLOCK_M) {
                        if (db_sq_idx < M) {
                            smem_database_sq[my_row] = (float)database_sq[batch_idx * M + db_sq_idx];
                        } else {
                            smem_database_sq[my_row] = 0.0f;
                        }
                    }
                    asm volatile("barrier.sync 8, 128;");
                    mbarrier_wait(score_full_addr, _phase_score_full_0);
                    _phase_score_full_0 ^= 1;
                    int cross_addr = taddr + (cta_rank * BLOCK_Q + tmem_row_base_v << 16);
                    float dots[64];
                    asm volatile(
                        "tcgen05.ld.sync.aligned.32x32b.x64.b32"
                        " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63}, [%64];"
                        : "=f"(dots[0]), "=f"(dots[1]), "=f"(dots[2]), "=f"(dots[3]), "=f"(dots[4]), "=f"(dots[5]), "=f"(dots[6]), "=f"(dots[7]), "=f"(dots[8]), "=f"(dots[9]), "=f"(dots[10]), "=f"(dots[11]), "=f"(dots[12]), "=f"(dots[13]), "=f"(dots[14]), "=f"(dots[15]), "=f"(dots[16]), "=f"(dots[17]), "=f"(dots[18]), "=f"(dots[19]), "=f"(dots[20]), "=f"(dots[21]), "=f"(dots[22]), "=f"(dots[23]), "=f"(dots[24]), "=f"(dots[25]), "=f"(dots[26]), "=f"(dots[27]), "=f"(dots[28]), "=f"(dots[29]), "=f"(dots[30]), "=f"(dots[31]), "=f"(dots[32]), "=f"(dots[33]), "=f"(dots[34]), "=f"(dots[35]), "=f"(dots[36]), "=f"(dots[37]), "=f"(dots[38]), "=f"(dots[39]), "=f"(dots[40]), "=f"(dots[41]), "=f"(dots[42]), "=f"(dots[43]), "=f"(dots[44]), "=f"(dots[45]), "=f"(dots[46]), "=f"(dots[47]), "=f"(dots[48]), "=f"(dots[49]), "=f"(dots[50]), "=f"(dots[51]), "=f"(dots[52]), "=f"(dots[53]), "=f"(dots[54]), "=f"(dots[55]), "=f"(dots[56]), "=f"(dots[57]), "=f"(dots[58]), "=f"(dots[59]), "=f"(dots[60]), "=f"(dots[61]), "=f"(dots[62]), "=f"(dots[63])
                        : "r"(cross_addr)
                        : "memory");
                    asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                    asm volatile("barrier.sync 8, 128;");
                    if (elect_sync()) {
                        asm volatile(
                            "mbarrier.arrive.release.cta.shared::cluster.b64 _, [%0];"
                            :: "r"((score_empty_addr) & 0xFEFFFFFF) : "memory");
                    }
                    if (valid_q != 0) {
                        #pragma unroll 2
                        for (int col_base = 0; col_base < 64; col_base += 4) {
                            float dist_vec[4];
                            dist_vec[0] = dots[col_base];
                            dist_vec[1] = dots[col_base + 1];
                            dist_vec[2] = dots[col_base + 2];
                            dist_vec[3] = dots[col_base + 3];
                            const float2 _fma_b2_0 = {-2.0f, -2.0f};
                            const float2 _fma_c2_1 = {q_sq_val, q_sq_val};
                            #pragma unroll
                            for (int _lf = 0; _lf < 2; _lf++)
                                fma_f32x2_inplace(&reinterpret_cast<float2*>(dist_vec)[_lf], _fma_b2_0, _fma_c2_1);
                            float db_sq_vec[4];
                            db_sq_vec[0] = smem_database_sq[col_base];
                            db_sq_vec[1] = smem_database_sq[col_base + 1];
                            db_sq_vec[2] = smem_database_sq[col_base + 2];
                            db_sq_vec[3] = smem_database_sq[col_base + 3];
                            float _t0[4];
                            #pragma unroll
                            for (int _la = 0; _la < 2; _la++)
                                reinterpret_cast<float2*>(_t0)[_la] = add_f32x2(reinterpret_cast<float2*>(dist_vec)[_la], reinterpret_cast<const float2*>(db_sq_vec)[_la]);
                            float group_min = _t0[0];
                            if (_t0[1] < group_min) {
                                group_min = _t0[1];
                            }
                            if (_t0[2] < group_min) {
                                group_min = _t0[2];
                            }
                            if (_t0[3] < group_min) {
                                group_min = _t0[3];
                            }
                            if (group_min < best_d[TOP_K_MAX - 1]) {
                                #pragma unroll
                                for (int vec_col = 0; vec_col < 4; vec_col++) {
                                    int db_idx = db_start + col_base + vec_col;
                                    if (db_idx < M) {
                                        float dist = _t0[vec_col];
                                        if (dist < best_d[TOP_K_MAX - 1]) {
                                            best_d[TOP_K_MAX - 1] = dist;
                                            best_i[TOP_K_MAX - 1] = db_idx;
                                            #pragma unroll
                                            for (int pos = TOP_K_MAX - 1; pos >= 1; pos--) {
                                                if (best_d[pos] < best_d[pos - 1]) {
                                                    float tmp_d = best_d[pos - 1];
                                                    int tmp_i = best_i[pos - 1];
                                                    best_d[pos - 1] = best_d[pos];
                                                    best_i[pos - 1] = best_i[pos];
                                                    best_d[pos] = tmp_d;
                                                    best_i[pos] = tmp_i;
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                    asm volatile("barrier.sync 8, 128;");
                }
                if (valid_q != 0) {
                    int out_base = ((split_idx * B + batch_idx) * Q + q_idx) * K;
                    #pragma unroll
                    for (int out_k = 0; out_k < TOP_K_MAX; out_k++) {
                        if (out_k < K) {
                            *((float*)(partial_dists + out_base + out_k)) = best_d[out_k];
                            *((int*)(partial_indices + out_base + out_k)) = best_i[out_k];
                        }
                    }
                }
            }
        }
    }

    // Cleanup
    asm volatile("barrier.cluster.arrive.release.aligned;");
    asm volatile("barrier.cluster.wait.acquire.aligned;");

    if (warp == 1) {
        asm volatile("tcgen05.dealloc.cta_group::2.sync.aligned.b32 %0, %1;" :: "r"(tmem_addr_storage[0]), "r"(64));
        asm volatile("tcgen05.relinquish_alloc_permit.cta_group::2.sync.aligned;");
    }
}

} // extern "C"

#undef BLOCK_M
#undef BLOCK_Q
#undef FEAT_D
#undef NUM_MAIN_STAGES
#undef SMEM_SMEM_DATABASE_OFF
#undef SMEM_SMEM_DATABASE_SQ_OFF
#undef SMEM_SMEM_DATABASE_SQ_STAGE_BYTES
#undef SMEM_SMEM_DATABASE_SQ_STRIDE
#undef SMEM_SMEM_DATABASE_STAGE_BYTES
#undef SMEM_SMEM_DATABASE_STRIDE
#undef SMEM_SMEM_QUERY_OFF
#undef SMEM_SMEM_QUERY_STAGE_BYTES
#undef SMEM_SMEM_QUERY_STRIDE
#undef SMEM_TOTAL
#undef THREADS
#undef TMEM_CROSS_OFFSET
#undef TMEM_NCOLS
#undef TOP_K_MAX
#undef database_empty_addr
#undef database_full_addr
#undef query_empty_addr
#undef query_full_addr
#undef score_empty_addr
#undef score_full_addr
#undef smem_database_addr
#undef smem_database_sq_addr
#undef smem_query_addr

#define TMEM_NCOLS 64
#define TMEM_CROSS_OFFSET 0
#define NUM_MAIN_STAGES 1
#define SMEM_SMEM_QUERY_OFF 1024
#define SMEM_SMEM_QUERY_STAGE_BYTES 32768
#define SMEM_SMEM_QUERY_STRIDE 32768
#define SMEM_SMEM_DATABASE_OFF 33792
#define SMEM_SMEM_DATABASE_STAGE_BYTES 16384
#define SMEM_SMEM_DATABASE_STRIDE 16384
#define SMEM_SMEM_DATABASE_SQ_OFF 50176
#define SMEM_SMEM_DATABASE_SQ_STAGE_BYTES 256
#define SMEM_SMEM_DATABASE_SQ_STRIDE 256
#define SMEM_TOTAL 50432
#define THREADS 192
#define BLOCK_Q 128
#define BLOCK_M 64
#define FEAT_D 128
#define TOP_K_MAX 32

extern "C" {

__global__ __launch_bounds__(192, 1) void
kernel_knn_build_evolve_7bfc_split_cg2_u2_stage1_k32split(float* __restrict__ query_sq, float* __restrict__ database_sq, float* __restrict__ partial_dists, int32_t* __restrict__ partial_indices, const void* tmap_query, const void* tmap_database, int B, int Q, int M, int K, int num_q_tile_pairs, int db_tiles_per_split, int split_count, int total_work)
{
    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;

    extern __shared__ __align__(1024) char smem_raw[];
    int smem;
    smem = (int)(unsigned long long)__cvta_generic_to_shared(smem_raw);
    const int smem_smem_query = smem + 1024;
    const int smem_smem_database = smem + 33792;
    const int smem_smem_database_sq = smem + 50176;

    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;
    const unsigned int clusters_x = gridDim.x / 2;
    const unsigned int cluster_id = ((blockIdx.z * gridDim.y + blockIdx.y) * clusters_x) + blockIdx.x / 2;
    const unsigned int num_clusters = clusters_x * gridDim.y * gridDim.z;

    int cta_rank;
    asm volatile("mov.b32 %0, %%cluster_ctarank;" : "=r"(cta_rank));

    // Mbarrier init (6 groups, 6 barriers)
    // Mbarriers at smem_raw[0..48)

    if (warp == 0) {
        uint32_t leader = elect_sync();
        // query_full: 1 barriers, init_count=2
        mbarrier_init_pred(smem + 0, 2, leader);
        // query_empty: 1 barriers, init_count=1
        mbarrier_init_pred(smem + 8, 1, leader);
        // database_full: 1 barriers, init_count=2
        mbarrier_init_pred(smem + 16, 2, leader);
        // database_empty: 1 barriers, init_count=1
        mbarrier_init_pred(smem + 24, 1, leader);
        // score_full: 1 barriers, init_count=1
        mbarrier_init_pred(smem + 32, 1, leader);
        // score_empty: 1 barriers, init_count=8
        mbarrier_init_pred(smem + 40, 8, leader);
        asm volatile("fence.mbarrier_init.release.cluster;");
    }

    __syncthreads();

    // TMEM alloc (64 columns, 64 used)
    volatile int* tmem_addr_storage = (volatile int*)(smem_raw + 48);
    if (warp == 1) {
        int _tmem_hold = smem + 48;
        asm volatile("tcgen05.alloc.cta_group::2.sync.aligned.shared::cta.b32 [%0], %1;" :: "r"(_tmem_hold), "r"(64) : "memory");
    }

    asm volatile("barrier.cluster.arrive.release.aligned;");
    asm volatile("barrier.cluster.wait.acquire.aligned;");
    asm volatile("tcgen05.fence::after_thread_sync;");

    const int warp_id = warp;
    const int lane_id = lane;
    __nv_bfloat16* smem_query = (__nv_bfloat16*)(smem_raw + 1024);
    #define smem_query_addr (smem + 1024)
    __nv_bfloat16* smem_database = (__nv_bfloat16*)(smem_raw + 33792);
    #define smem_database_addr (smem + 33792)
    float* smem_database_sq = (float*)(smem_raw + 50176);
    #define smem_database_sq_addr (smem + 50176)
    const int mbar_base = smem;
    #define query_full_addr (mbar_base + 0)
    #define query_empty_addr (mbar_base + 8)
    #define database_full_addr (mbar_base + 16)
    #define database_empty_addr (mbar_base + 24)
    #define score_full_addr (mbar_base + 32)
    #define score_empty_addr (mbar_base + 40)
    const int taddr = tmem_addr_storage[0];

    // ---- Role: load ----
    if (warp == 0) {
        { // load_main
            uint32_t _phase_query_empty_0 = 1;
            uint32_t _phase_database_empty_0 = 1;
            if (warp_id == 0) {
                if (elect_sync()) {
                    #pragma unroll 1
                    for (unsigned int work_idx = cluster_id; work_idx < total_work; work_idx += num_clusters) {
                        int split_idx = work_idx % split_count;
                        int query_work = work_idx / split_count;
                        int batch_idx = query_work / num_q_tile_pairs;
                        int q_tile_pair = query_work % num_q_tile_pairs;
                        int q_tile = q_tile_pair * 2 + cta_rank;
                        int off_q = q_tile * BLOCK_Q;
                        int global_q = batch_idx * Q + off_q;
                        int db_tile_start = split_idx * db_tiles_per_split;
                        mbarrier_wait(query_empty_addr, _phase_query_empty_0);
                        _phase_query_empty_0 ^= 1;
                        asm volatile(
                            "mbarrier.arrive.expect_tx.release.cta.shared::cluster.b64 _, [%0], %1;"
                            :: "r"((query_full_addr) & 0xFEFFFFFF), "r"((uint32_t)(32768)) : "memory");
                        asm volatile(
                            "cp.async.bulk.tensor.3d.shared::cluster.global.mbarrier::complete_tx::bytes.cta_group::2"
                            " [%0], [%1, {%2, %3, %4}], [%5];"
                            :: "r"(smem_query_addr), "l"(tmap_query), "r"(0), "r"(global_q), "r"(0),
                               "r"(((query_full_addr) & 0xFEFFFFFF)) : "memory");
                        #pragma unroll 1
                        for (int local_db_tile = 0; local_db_tile < db_tiles_per_split; local_db_tile++) {
                            int db_tile = db_tile_start + local_db_tile;
                            int off_m = db_tile * BLOCK_M;
                            int global_m = batch_idx * M + off_m;
                            mbarrier_wait(database_empty_addr, _phase_database_empty_0);
                            _phase_database_empty_0 ^= 1;
                            asm volatile(
                                "mbarrier.arrive.expect_tx.release.cta.shared::cluster.b64 _, [%0], %1;"
                                :: "r"((database_full_addr) & 0xFEFFFFFF), "r"((uint32_t)(16384)) : "memory");
                            asm volatile(
                                "cp.async.bulk.tensor.3d.shared::cluster.global.mbarrier::complete_tx::bytes.cta_group::2"
                                " [%0], [%1, {%2, %3, %4}], [%5];"
                                :: "r"(smem_database_addr), "l"(tmap_database), "r"(0), "r"(global_m), "r"(0),
                                   "r"(((database_full_addr) & 0xFEFFFFFF)) : "memory");
                        }
                    }
                }
            }
        }
    // ---- Role: mma ----
    } else if (warp == 1) {
        { // mma_main
            uint32_t _phase_query_full_0 = 0;
            uint32_t _phase_score_empty_0 = 1;
            uint32_t _phase_database_full_0 = 0;
            if (cta_rank == 0) {
                #pragma unroll 1
                for (unsigned int work_idx = cluster_id; work_idx < total_work; work_idx += num_clusters) {
                    mbarrier_wait(query_full_addr, _phase_query_full_0);
                    _phase_query_full_0 ^= 1;
                    #pragma unroll 1
                    for (int _local_db_tile = 0; _local_db_tile < db_tiles_per_split; _local_db_tile++) {
                        mbarrier_wait(score_empty_addr, _phase_score_empty_0);
                        _phase_score_empty_0 ^= 1;
                        mbarrier_wait(database_full_addr, _phase_database_full_0);
                        _phase_database_full_0 ^= 1;
                        asm volatile("tcgen05.fence::after_thread_sync;");
                        int _mma_ss_a_lo_0 = (smem_query_addr >> 4) & 0x3FFF;
                        int _mma_ss_b_lo_0 = (smem_database_addr >> 4) & 0x3FFF;
                        asm volatile(
                    "{\n\t"
                    ".reg .pred leader, p0, p1;\n\t"
                    ".reg .b32 adhi, bdhi, alo, blo, id, m0, m1, m2, m3, m4, m5, m6, m7;\n\t"
                    ".reg .b64 da, db;\n\t"
                    "elect.sync _|leader, 0xFFFFFFFF;\n\t"
                    "setp.ne.b32 p0, %3, 0;\n\t"
                    "setp.ne.b32 p1, 1, 0;\n\t"
                    "mov.b32 m0, 0; mov.b32 m1, 0; mov.b32 m2, 0; mov.b32 m3, 0;\n\tmov.b32 m4, 0; mov.b32 m5, 0; mov.b32 m6, 0; mov.b32 m7, 0;\n\t"
                    "mov.b32 adhi, 0x40004040;\n\t"
                    "mov.b32 bdhi, 0x40004040;\n\t"
                    "mov.b32 id, 270533776;\n\t"
                    "mov.b32 alo, %0;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p0;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "add.u32 alo, alo, 1018;\n\t"
                    "add.u32 blo, blo, 506;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "}\n"
                    :: "r"(_mma_ss_a_lo_0), "r"(_mma_ss_b_lo_0), "r"(taddr), "r"(0));
                        elect_commit_cg2_multicast(score_full_addr, (uint16_t)(3));
                        elect_commit_cg2_multicast(database_empty_addr, (uint16_t)(3));
                    }
                    elect_commit_cg2_multicast(query_empty_addr, (uint16_t)(3));
                }
            }
        }
    // ---- Role: compute ----
    } else if (warp >= 2 && warp <= 5) {
        const int warp_id_in_wg = warp % 4;
        const int my_row = warp_id_in_wg * 32 + lane;
        const int tmem_row_base_v = (warp_id_in_wg * 32) << 16;
        const int tmem_cross = taddr + TMEM_CROSS_OFFSET;
        { // compute_main
            uint32_t _phase_score_full_0 = 0;
            #pragma unroll 1
            for (unsigned int work_idx = cluster_id; work_idx < total_work; work_idx += num_clusters) {
                int split_idx = work_idx % split_count;
                int query_work = work_idx / split_count;
                int batch_idx = query_work / num_q_tile_pairs;
                int q_tile_pair = query_work % num_q_tile_pairs;
                int q_tile = q_tile_pair * 2 + cta_rank;
                int off_q = q_tile * BLOCK_Q;
                int q_idx = off_q + my_row;
                int valid_q = ((q_idx < Q) ? 1 : 0);
                float q_sq_val = 0.0f;
                if (valid_q != 0) {
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
                    int db_start = db_tile * BLOCK_M;
                    int db_sq_idx = db_start + my_row;
                    if (my_row < BLOCK_M) {
                        if (db_sq_idx < M) {
                            smem_database_sq[my_row] = (float)database_sq[batch_idx * M + db_sq_idx];
                        } else {
                            smem_database_sq[my_row] = 0.0f;
                        }
                    }
                    asm volatile("barrier.sync 8, 128;");
                    mbarrier_wait(score_full_addr, _phase_score_full_0);
                    _phase_score_full_0 ^= 1;
                    int cross_addr = taddr + (cta_rank * BLOCK_Q + tmem_row_base_v << 16);
                    float dots[64];
                    asm volatile(
                        "tcgen05.ld.sync.aligned.32x32b.x64.b32"
                        " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63}, [%64];"
                        : "=f"(dots[0]), "=f"(dots[1]), "=f"(dots[2]), "=f"(dots[3]), "=f"(dots[4]), "=f"(dots[5]), "=f"(dots[6]), "=f"(dots[7]), "=f"(dots[8]), "=f"(dots[9]), "=f"(dots[10]), "=f"(dots[11]), "=f"(dots[12]), "=f"(dots[13]), "=f"(dots[14]), "=f"(dots[15]), "=f"(dots[16]), "=f"(dots[17]), "=f"(dots[18]), "=f"(dots[19]), "=f"(dots[20]), "=f"(dots[21]), "=f"(dots[22]), "=f"(dots[23]), "=f"(dots[24]), "=f"(dots[25]), "=f"(dots[26]), "=f"(dots[27]), "=f"(dots[28]), "=f"(dots[29]), "=f"(dots[30]), "=f"(dots[31]), "=f"(dots[32]), "=f"(dots[33]), "=f"(dots[34]), "=f"(dots[35]), "=f"(dots[36]), "=f"(dots[37]), "=f"(dots[38]), "=f"(dots[39]), "=f"(dots[40]), "=f"(dots[41]), "=f"(dots[42]), "=f"(dots[43]), "=f"(dots[44]), "=f"(dots[45]), "=f"(dots[46]), "=f"(dots[47]), "=f"(dots[48]), "=f"(dots[49]), "=f"(dots[50]), "=f"(dots[51]), "=f"(dots[52]), "=f"(dots[53]), "=f"(dots[54]), "=f"(dots[55]), "=f"(dots[56]), "=f"(dots[57]), "=f"(dots[58]), "=f"(dots[59]), "=f"(dots[60]), "=f"(dots[61]), "=f"(dots[62]), "=f"(dots[63])
                        : "r"(cross_addr)
                        : "memory");
                    asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                    asm volatile("barrier.sync 8, 128;");
                    if (elect_sync()) {
                        asm volatile(
                            "mbarrier.arrive.release.cta.shared::cluster.b64 _, [%0];"
                            :: "r"((score_empty_addr) & 0xFEFFFFFF) : "memory");
                    }
                    if (valid_q != 0) {
                        #pragma unroll 2
                        for (int col_base = 0; col_base < 64; col_base += 4) {
                            float dist_vec[4];
                            dist_vec[0] = dots[col_base];
                            dist_vec[1] = dots[col_base + 1];
                            dist_vec[2] = dots[col_base + 2];
                            dist_vec[3] = dots[col_base + 3];
                            const float2 _fma_b2_0 = {-2.0f, -2.0f};
                            const float2 _fma_c2_1 = {q_sq_val, q_sq_val};
                            #pragma unroll
                            for (int _lf = 0; _lf < 2; _lf++)
                                fma_f32x2_inplace(&reinterpret_cast<float2*>(dist_vec)[_lf], _fma_b2_0, _fma_c2_1);
                            float db_sq_vec[4];
                            db_sq_vec[0] = smem_database_sq[col_base];
                            db_sq_vec[1] = smem_database_sq[col_base + 1];
                            db_sq_vec[2] = smem_database_sq[col_base + 2];
                            db_sq_vec[3] = smem_database_sq[col_base + 3];
                            float _t0[4];
                            #pragma unroll
                            for (int _la = 0; _la < 2; _la++)
                                reinterpret_cast<float2*>(_t0)[_la] = add_f32x2(reinterpret_cast<float2*>(dist_vec)[_la], reinterpret_cast<const float2*>(db_sq_vec)[_la]);
                            float group_min = _t0[0];
                            if (_t0[1] < group_min) {
                                group_min = _t0[1];
                            }
                            if (_t0[2] < group_min) {
                                group_min = _t0[2];
                            }
                            if (_t0[3] < group_min) {
                                group_min = _t0[3];
                            }
                            if (group_min < best_d[TOP_K_MAX - 1]) {
                                #pragma unroll
                                for (int vec_col = 0; vec_col < 4; vec_col++) {
                                    int db_idx = db_start + col_base + vec_col;
                                    if (db_idx < M) {
                                        float dist = _t0[vec_col];
                                        if (dist < best_d[TOP_K_MAX - 1]) {
                                            best_d[TOP_K_MAX - 1] = dist;
                                            best_i[TOP_K_MAX - 1] = db_idx;
                                            #pragma unroll
                                            for (int pos = TOP_K_MAX - 1; pos >= 1; pos--) {
                                                if (best_d[pos] < best_d[pos - 1]) {
                                                    float tmp_d = best_d[pos - 1];
                                                    int tmp_i = best_i[pos - 1];
                                                    best_d[pos - 1] = best_d[pos];
                                                    best_i[pos - 1] = best_i[pos];
                                                    best_d[pos] = tmp_d;
                                                    best_i[pos] = tmp_i;
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                    asm volatile("barrier.sync 8, 128;");
                }
                if (valid_q != 0) {
                    int out_base = ((split_idx * B + batch_idx) * Q + q_idx) * K;
                    #pragma unroll
                    for (int out_k = 0; out_k < TOP_K_MAX; out_k++) {
                        if (out_k < K) {
                            *((float*)(partial_dists + out_base + out_k)) = best_d[out_k];
                            *((int*)(partial_indices + out_base + out_k)) = best_i[out_k];
                        }
                    }
                }
            }
        }
    }

    // Cleanup
    asm volatile("barrier.cluster.arrive.release.aligned;");
    asm volatile("barrier.cluster.wait.acquire.aligned;");

    if (warp == 1) {
        asm volatile("tcgen05.dealloc.cta_group::2.sync.aligned.b32 %0, %1;" :: "r"(tmem_addr_storage[0]), "r"(64));
        asm volatile("tcgen05.relinquish_alloc_permit.cta_group::2.sync.aligned;");
    }
}

} // extern "C"

#undef BLOCK_M
#undef BLOCK_Q
#undef FEAT_D
#undef NUM_MAIN_STAGES
#undef SMEM_SMEM_DATABASE_OFF
#undef SMEM_SMEM_DATABASE_SQ_OFF
#undef SMEM_SMEM_DATABASE_SQ_STAGE_BYTES
#undef SMEM_SMEM_DATABASE_SQ_STRIDE
#undef SMEM_SMEM_DATABASE_STAGE_BYTES
#undef SMEM_SMEM_DATABASE_STRIDE
#undef SMEM_SMEM_QUERY_OFF
#undef SMEM_SMEM_QUERY_STAGE_BYTES
#undef SMEM_SMEM_QUERY_STRIDE
#undef SMEM_TOTAL
#undef THREADS
#undef TMEM_CROSS_OFFSET
#undef TMEM_NCOLS
#undef TOP_K_MAX
#undef database_empty_addr
#undef database_full_addr
#undef query_empty_addr
#undef query_full_addr
#undef score_empty_addr
#undef score_full_addr
#undef smem_database_addr
#undef smem_database_sq_addr
#undef smem_query_addr

#define TMEM_NCOLS 64
#define TMEM_CROSS_OFFSET 0
#define NUM_MAIN_STAGES 1
#define SMEM_SMEM_QUERY_OFF 1024
#define SMEM_SMEM_QUERY_STAGE_BYTES 32768
#define SMEM_SMEM_QUERY_STRIDE 32768
#define SMEM_SMEM_DATABASE_OFF 33792
#define SMEM_SMEM_DATABASE_STAGE_BYTES 16384
#define SMEM_SMEM_DATABASE_STRIDE 16384
#define SMEM_SMEM_DATABASE_SQ_OFF 50176
#define SMEM_SMEM_DATABASE_SQ_STAGE_BYTES 256
#define SMEM_SMEM_DATABASE_SQ_STRIDE 256
#define SMEM_TOTAL 50432
#define THREADS 192
#define BLOCK_Q 128
#define BLOCK_M 64
#define FEAT_D 128
#define TOP_K_MAX 20

extern "C" {

__global__ __launch_bounds__(192, 1) void
kernel_knn_build_evolve_7bfc_split_cg2_u2_stage1_k32_unordered_k20unordered(float* __restrict__ query_sq, float* __restrict__ database_sq, float* __restrict__ partial_dists, int32_t* __restrict__ partial_indices, const void* tmap_query, const void* tmap_database, int B, int Q, int M, int K, int num_q_tile_pairs, int db_tiles_per_split, int split_count, int total_work)
{
    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;

    extern __shared__ __align__(1024) char smem_raw[];
    int smem;
    smem = (int)(unsigned long long)__cvta_generic_to_shared(smem_raw);
    const int smem_smem_query = smem + 1024;
    const int smem_smem_database = smem + 33792;
    const int smem_smem_database_sq = smem + 50176;

    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;
    const unsigned int clusters_x = gridDim.x / 2;
    const unsigned int cluster_id = ((blockIdx.z * gridDim.y + blockIdx.y) * clusters_x) + blockIdx.x / 2;
    const unsigned int num_clusters = clusters_x * gridDim.y * gridDim.z;

    int cta_rank;
    asm volatile("mov.b32 %0, %%cluster_ctarank;" : "=r"(cta_rank));

    // Mbarrier init (6 groups, 6 barriers)
    // Mbarriers at smem_raw[0..48)

    if (warp == 0) {
        uint32_t leader = elect_sync();
        // query_full: 1 barriers, init_count=2
        mbarrier_init_pred(smem + 0, 2, leader);
        // query_empty: 1 barriers, init_count=1
        mbarrier_init_pred(smem + 8, 1, leader);
        // database_full: 1 barriers, init_count=2
        mbarrier_init_pred(smem + 16, 2, leader);
        // database_empty: 1 barriers, init_count=1
        mbarrier_init_pred(smem + 24, 1, leader);
        // score_full: 1 barriers, init_count=1
        mbarrier_init_pred(smem + 32, 1, leader);
        // score_empty: 1 barriers, init_count=8
        mbarrier_init_pred(smem + 40, 8, leader);
        asm volatile("fence.mbarrier_init.release.cluster;");
    }

    __syncthreads();

    // TMEM alloc (64 columns, 64 used)
    volatile int* tmem_addr_storage = (volatile int*)(smem_raw + 48);
    if (warp == 1) {
        int _tmem_hold = smem + 48;
        asm volatile("tcgen05.alloc.cta_group::2.sync.aligned.shared::cta.b32 [%0], %1;" :: "r"(_tmem_hold), "r"(64) : "memory");
    }

    asm volatile("barrier.cluster.arrive.release.aligned;");
    asm volatile("barrier.cluster.wait.acquire.aligned;");
    asm volatile("tcgen05.fence::after_thread_sync;");

    const int warp_id = warp;
    const int lane_id = lane;
    __nv_bfloat16* smem_query = (__nv_bfloat16*)(smem_raw + 1024);
    #define smem_query_addr (smem + 1024)
    __nv_bfloat16* smem_database = (__nv_bfloat16*)(smem_raw + 33792);
    #define smem_database_addr (smem + 33792)
    float* smem_database_sq = (float*)(smem_raw + 50176);
    #define smem_database_sq_addr (smem + 50176)
    const int mbar_base = smem;
    #define query_full_addr (mbar_base + 0)
    #define query_empty_addr (mbar_base + 8)
    #define database_full_addr (mbar_base + 16)
    #define database_empty_addr (mbar_base + 24)
    #define score_full_addr (mbar_base + 32)
    #define score_empty_addr (mbar_base + 40)
    const int taddr = tmem_addr_storage[0];

    // ---- Role: load ----
    if (warp == 0) {
        { // load_main
            uint32_t _phase_query_empty_0 = 1;
            uint32_t _phase_database_empty_0 = 1;
            if (warp_id == 0) {
                if (elect_sync()) {
                    #pragma unroll 1
                    for (unsigned int work_idx = cluster_id; work_idx < total_work; work_idx += num_clusters) {
                        int split_idx = work_idx % split_count;
                        int query_work = work_idx / split_count;
                        int batch_idx = query_work / num_q_tile_pairs;
                        int q_tile_pair = query_work % num_q_tile_pairs;
                        int q_tile = q_tile_pair * 2 + cta_rank;
                        int off_q = q_tile * BLOCK_Q;
                        int global_q = batch_idx * Q + off_q;
                        int db_tile_start = split_idx * db_tiles_per_split;
                        mbarrier_wait(query_empty_addr, _phase_query_empty_0);
                        _phase_query_empty_0 ^= 1;
                        asm volatile(
                            "mbarrier.arrive.expect_tx.release.cta.shared::cluster.b64 _, [%0], %1;"
                            :: "r"((query_full_addr) & 0xFEFFFFFF), "r"((uint32_t)(32768)) : "memory");
                        asm volatile(
                            "cp.async.bulk.tensor.3d.shared::cluster.global.mbarrier::complete_tx::bytes.cta_group::2"
                            " [%0], [%1, {%2, %3, %4}], [%5];"
                            :: "r"(smem_query_addr), "l"(tmap_query), "r"(0), "r"(global_q), "r"(0),
                               "r"(((query_full_addr) & 0xFEFFFFFF)) : "memory");
                        #pragma unroll 1
                        for (int local_db_tile = 0; local_db_tile < db_tiles_per_split; local_db_tile++) {
                            int db_tile = db_tile_start + local_db_tile;
                            int off_m = db_tile * BLOCK_M;
                            int global_m = batch_idx * M + off_m;
                            mbarrier_wait(database_empty_addr, _phase_database_empty_0);
                            _phase_database_empty_0 ^= 1;
                            asm volatile(
                                "mbarrier.arrive.expect_tx.release.cta.shared::cluster.b64 _, [%0], %1;"
                                :: "r"((database_full_addr) & 0xFEFFFFFF), "r"((uint32_t)(16384)) : "memory");
                            asm volatile(
                                "cp.async.bulk.tensor.3d.shared::cluster.global.mbarrier::complete_tx::bytes.cta_group::2"
                                " [%0], [%1, {%2, %3, %4}], [%5];"
                                :: "r"(smem_database_addr), "l"(tmap_database), "r"(0), "r"(global_m), "r"(0),
                                   "r"(((database_full_addr) & 0xFEFFFFFF)) : "memory");
                        }
                    }
                }
            }
        }
    // ---- Role: mma ----
    } else if (warp == 1) {
        { // mma_main
            uint32_t _phase_query_full_0 = 0;
            uint32_t _phase_score_empty_0 = 1;
            uint32_t _phase_database_full_0 = 0;
            if (cta_rank == 0) {
                #pragma unroll 1
                for (unsigned int work_idx = cluster_id; work_idx < total_work; work_idx += num_clusters) {
                    mbarrier_wait(query_full_addr, _phase_query_full_0);
                    _phase_query_full_0 ^= 1;
                    #pragma unroll 1
                    for (int _local_db_tile = 0; _local_db_tile < db_tiles_per_split; _local_db_tile++) {
                        mbarrier_wait(score_empty_addr, _phase_score_empty_0);
                        _phase_score_empty_0 ^= 1;
                        mbarrier_wait(database_full_addr, _phase_database_full_0);
                        _phase_database_full_0 ^= 1;
                        asm volatile("tcgen05.fence::after_thread_sync;");
                        int _mma_ss_a_lo_0 = (smem_query_addr >> 4) & 0x3FFF;
                        int _mma_ss_b_lo_0 = (smem_database_addr >> 4) & 0x3FFF;
                        asm volatile(
                    "{\n\t"
                    ".reg .pred leader, p0, p1;\n\t"
                    ".reg .b32 adhi, bdhi, alo, blo, id, m0, m1, m2, m3, m4, m5, m6, m7;\n\t"
                    ".reg .b64 da, db;\n\t"
                    "elect.sync _|leader, 0xFFFFFFFF;\n\t"
                    "setp.ne.b32 p0, %3, 0;\n\t"
                    "setp.ne.b32 p1, 1, 0;\n\t"
                    "mov.b32 m0, 0; mov.b32 m1, 0; mov.b32 m2, 0; mov.b32 m3, 0;\n\tmov.b32 m4, 0; mov.b32 m5, 0; mov.b32 m6, 0; mov.b32 m7, 0;\n\t"
                    "mov.b32 adhi, 0x40004040;\n\t"
                    "mov.b32 bdhi, 0x40004040;\n\t"
                    "mov.b32 id, 270533776;\n\t"
                    "mov.b32 alo, %0;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p0;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "add.u32 alo, alo, 1018;\n\t"
                    "add.u32 blo, blo, 506;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "}\n"
                    :: "r"(_mma_ss_a_lo_0), "r"(_mma_ss_b_lo_0), "r"(taddr), "r"(0));
                        elect_commit_cg2_multicast(score_full_addr, (uint16_t)(3));
                        elect_commit_cg2_multicast(database_empty_addr, (uint16_t)(3));
                    }
                    elect_commit_cg2_multicast(query_empty_addr, (uint16_t)(3));
                }
            }
        }
    // ---- Role: compute ----
    } else if (warp >= 2 && warp <= 5) {
        const int warp_id_in_wg = warp % 4;
        const int my_row = warp_id_in_wg * 32 + lane;
        const int tmem_row_base_v = (warp_id_in_wg * 32) << 16;
        const int tmem_cross = taddr + TMEM_CROSS_OFFSET;
        { // compute_main
            uint32_t _phase_score_full_0 = 0;
            #pragma unroll 1
            for (unsigned int work_idx = cluster_id; work_idx < total_work; work_idx += num_clusters) {
                int split_idx = work_idx % split_count;
                int query_work = work_idx / split_count;
                int batch_idx = query_work / num_q_tile_pairs;
                int q_tile_pair = query_work % num_q_tile_pairs;
                int q_tile = q_tile_pair * 2 + cta_rank;
                int off_q = q_tile * BLOCK_Q;
                int q_idx = off_q + my_row;
                int valid_q = ((q_idx < Q) ? 1 : 0);
                float q_sq_val = 0.0f;
                if (valid_q != 0) {
                    q_sq_val = (float)query_sq[batch_idx * Q + q_idx];
                }
                float best_d[TOP_K_MAX];
                int best_i[TOP_K_MAX];
                #pragma unroll
                for (int kk = 0; kk < TOP_K_MAX; kk++) {
                    best_d[kk] = 3.4e+38f;
                    best_i[kk] = -1;
                }
                float worst_d = 3.4e+38f;
                int worst_pos = 0;
                int db_tile_start = split_idx * db_tiles_per_split;
                #pragma unroll 1
                for (int local_db_tile = 0; local_db_tile < db_tiles_per_split; local_db_tile++) {
                    int db_tile = db_tile_start + local_db_tile;
                    int db_start = db_tile * BLOCK_M;
                    int db_sq_idx = db_start + my_row;
                    if (my_row < BLOCK_M) {
                        if (db_sq_idx < M) {
                            smem_database_sq[my_row] = (float)database_sq[batch_idx * M + db_sq_idx];
                        } else {
                            smem_database_sq[my_row] = 0.0f;
                        }
                    }
                    asm volatile("barrier.sync 8, 128;");
                    mbarrier_wait(score_full_addr, _phase_score_full_0);
                    _phase_score_full_0 ^= 1;
                    int cross_addr = taddr + (cta_rank * BLOCK_Q + tmem_row_base_v << 16);
                    float dots[64];
                    asm volatile(
                        "tcgen05.ld.sync.aligned.32x32b.x64.b32"
                        " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63}, [%64];"
                        : "=f"(dots[0]), "=f"(dots[1]), "=f"(dots[2]), "=f"(dots[3]), "=f"(dots[4]), "=f"(dots[5]), "=f"(dots[6]), "=f"(dots[7]), "=f"(dots[8]), "=f"(dots[9]), "=f"(dots[10]), "=f"(dots[11]), "=f"(dots[12]), "=f"(dots[13]), "=f"(dots[14]), "=f"(dots[15]), "=f"(dots[16]), "=f"(dots[17]), "=f"(dots[18]), "=f"(dots[19]), "=f"(dots[20]), "=f"(dots[21]), "=f"(dots[22]), "=f"(dots[23]), "=f"(dots[24]), "=f"(dots[25]), "=f"(dots[26]), "=f"(dots[27]), "=f"(dots[28]), "=f"(dots[29]), "=f"(dots[30]), "=f"(dots[31]), "=f"(dots[32]), "=f"(dots[33]), "=f"(dots[34]), "=f"(dots[35]), "=f"(dots[36]), "=f"(dots[37]), "=f"(dots[38]), "=f"(dots[39]), "=f"(dots[40]), "=f"(dots[41]), "=f"(dots[42]), "=f"(dots[43]), "=f"(dots[44]), "=f"(dots[45]), "=f"(dots[46]), "=f"(dots[47]), "=f"(dots[48]), "=f"(dots[49]), "=f"(dots[50]), "=f"(dots[51]), "=f"(dots[52]), "=f"(dots[53]), "=f"(dots[54]), "=f"(dots[55]), "=f"(dots[56]), "=f"(dots[57]), "=f"(dots[58]), "=f"(dots[59]), "=f"(dots[60]), "=f"(dots[61]), "=f"(dots[62]), "=f"(dots[63])
                        : "r"(cross_addr)
                        : "memory");
                    asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                    asm volatile("barrier.sync 8, 128;");
                    if (elect_sync()) {
                        asm volatile(
                            "mbarrier.arrive.release.cta.shared::cluster.b64 _, [%0];"
                            :: "r"((score_empty_addr) & 0xFEFFFFFF) : "memory");
                    }
                    if (valid_q != 0) {
                        #pragma unroll 2
                        for (int col_base = 0; col_base < 64; col_base += 4) {
                            float dist_vec[4];
                            dist_vec[0] = dots[col_base];
                            dist_vec[1] = dots[col_base + 1];
                            dist_vec[2] = dots[col_base + 2];
                            dist_vec[3] = dots[col_base + 3];
                            const float2 _fma_b2_0 = {-2.0f, -2.0f};
                            const float2 _fma_c2_1 = {q_sq_val, q_sq_val};
                            #pragma unroll
                            for (int _lf = 0; _lf < 2; _lf++)
                                fma_f32x2_inplace(&reinterpret_cast<float2*>(dist_vec)[_lf], _fma_b2_0, _fma_c2_1);
                            float db_sq_vec[4];
                            db_sq_vec[0] = smem_database_sq[col_base];
                            db_sq_vec[1] = smem_database_sq[col_base + 1];
                            db_sq_vec[2] = smem_database_sq[col_base + 2];
                            db_sq_vec[3] = smem_database_sq[col_base + 3];
                            float _t0[4];
                            #pragma unroll
                            for (int _la = 0; _la < 2; _la++)
                                reinterpret_cast<float2*>(_t0)[_la] = add_f32x2(reinterpret_cast<float2*>(dist_vec)[_la], reinterpret_cast<const float2*>(db_sq_vec)[_la]);
                            float group_min = _t0[0];
                            if (_t0[1] < group_min) {
                                group_min = _t0[1];
                            }
                            if (_t0[2] < group_min) {
                                group_min = _t0[2];
                            }
                            if (_t0[3] < group_min) {
                                group_min = _t0[3];
                            }
                            if (group_min < worst_d) {
                                #pragma unroll
                                for (int vec_col = 0; vec_col < 4; vec_col++) {
                                    int db_idx = db_start + col_base + vec_col;
                                    if (db_idx < M) {
                                        float dist = _t0[vec_col];
                                        if (dist < worst_d) {
                                            best_d[worst_pos] = dist;
                                            best_i[worst_pos] = db_idx;
                                            worst_d = best_d[0];
                                            worst_pos = 0;
                                            #pragma unroll
                                            for (int scan_pos = 1; scan_pos < TOP_K_MAX; scan_pos++) {
                                                if (best_d[scan_pos] > worst_d) {
                                                    worst_d = best_d[scan_pos];
                                                    worst_pos = scan_pos;
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                    asm volatile("barrier.sync 8, 128;");
                }
                if (valid_q != 0) {
                    int out_base = ((split_idx * B + batch_idx) * Q + q_idx) * K;
                    #pragma unroll
                    for (int out_k = 0; out_k < TOP_K_MAX; out_k++) {
                        *((float*)(partial_dists + out_base + out_k)) = best_d[out_k];
                        *((int*)(partial_indices + out_base + out_k)) = best_i[out_k];
                    }
                }
            }
        }
    }

    // Cleanup
    asm volatile("barrier.cluster.arrive.release.aligned;");
    asm volatile("barrier.cluster.wait.acquire.aligned;");

    if (warp == 1) {
        asm volatile("tcgen05.dealloc.cta_group::2.sync.aligned.b32 %0, %1;" :: "r"(tmem_addr_storage[0]), "r"(64));
        asm volatile("tcgen05.relinquish_alloc_permit.cta_group::2.sync.aligned;");
    }
}

} // extern "C"

#undef BLOCK_M
#undef BLOCK_Q
#undef FEAT_D
#undef NUM_MAIN_STAGES
#undef SMEM_SMEM_DATABASE_OFF
#undef SMEM_SMEM_DATABASE_SQ_OFF
#undef SMEM_SMEM_DATABASE_SQ_STAGE_BYTES
#undef SMEM_SMEM_DATABASE_SQ_STRIDE
#undef SMEM_SMEM_DATABASE_STAGE_BYTES
#undef SMEM_SMEM_DATABASE_STRIDE
#undef SMEM_SMEM_QUERY_OFF
#undef SMEM_SMEM_QUERY_STAGE_BYTES
#undef SMEM_SMEM_QUERY_STRIDE
#undef SMEM_TOTAL
#undef THREADS
#undef TMEM_CROSS_OFFSET
#undef TMEM_NCOLS
#undef TOP_K_MAX
#undef database_empty_addr
#undef database_full_addr
#undef query_empty_addr
#undef query_full_addr
#undef score_empty_addr
#undef score_full_addr
#undef smem_database_addr
#undef smem_database_sq_addr
#undef smem_query_addr

#define TMEM_NCOLS 64
#define TMEM_CROSS_OFFSET 0
#define NUM_MAIN_STAGES 1
#define SMEM_SMEM_QUERY_OFF 1024
#define SMEM_SMEM_QUERY_STAGE_BYTES 32768
#define SMEM_SMEM_QUERY_STRIDE 32768
#define SMEM_SMEM_DATABASE_OFF 33792
#define SMEM_SMEM_DATABASE_STAGE_BYTES 16384
#define SMEM_SMEM_DATABASE_STRIDE 16384
#define SMEM_SMEM_DATABASE_SQ_OFF 50176
#define SMEM_SMEM_DATABASE_SQ_STAGE_BYTES 256
#define SMEM_SMEM_DATABASE_SQ_STRIDE 256
#define SMEM_TOTAL 50432
#define THREADS 192
#define BLOCK_Q 128
#define BLOCK_M 64
#define FEAT_D 128
#define TOP_K_MAX 30

extern "C" {

__global__ __launch_bounds__(192, 1) void
kernel_knn_build_evolve_7bfc_split_cg2_u2_stage1_k32_unordered_k30unordered(float* __restrict__ query_sq, float* __restrict__ database_sq, float* __restrict__ partial_dists, int32_t* __restrict__ partial_indices, const void* tmap_query, const void* tmap_database, int B, int Q, int M, int K, int num_q_tile_pairs, int db_tiles_per_split, int split_count, int total_work)
{
    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;

    extern __shared__ __align__(1024) char smem_raw[];
    int smem;
    smem = (int)(unsigned long long)__cvta_generic_to_shared(smem_raw);
    const int smem_smem_query = smem + 1024;
    const int smem_smem_database = smem + 33792;
    const int smem_smem_database_sq = smem + 50176;

    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;
    const unsigned int clusters_x = gridDim.x / 2;
    const unsigned int cluster_id = ((blockIdx.z * gridDim.y + blockIdx.y) * clusters_x) + blockIdx.x / 2;
    const unsigned int num_clusters = clusters_x * gridDim.y * gridDim.z;

    int cta_rank;
    asm volatile("mov.b32 %0, %%cluster_ctarank;" : "=r"(cta_rank));

    // Mbarrier init (6 groups, 6 barriers)
    // Mbarriers at smem_raw[0..48)

    if (warp == 0) {
        uint32_t leader = elect_sync();
        // query_full: 1 barriers, init_count=2
        mbarrier_init_pred(smem + 0, 2, leader);
        // query_empty: 1 barriers, init_count=1
        mbarrier_init_pred(smem + 8, 1, leader);
        // database_full: 1 barriers, init_count=2
        mbarrier_init_pred(smem + 16, 2, leader);
        // database_empty: 1 barriers, init_count=1
        mbarrier_init_pred(smem + 24, 1, leader);
        // score_full: 1 barriers, init_count=1
        mbarrier_init_pred(smem + 32, 1, leader);
        // score_empty: 1 barriers, init_count=8
        mbarrier_init_pred(smem + 40, 8, leader);
        asm volatile("fence.mbarrier_init.release.cluster;");
    }

    __syncthreads();

    // TMEM alloc (64 columns, 64 used)
    volatile int* tmem_addr_storage = (volatile int*)(smem_raw + 48);
    if (warp == 1) {
        int _tmem_hold = smem + 48;
        asm volatile("tcgen05.alloc.cta_group::2.sync.aligned.shared::cta.b32 [%0], %1;" :: "r"(_tmem_hold), "r"(64) : "memory");
    }

    asm volatile("barrier.cluster.arrive.release.aligned;");
    asm volatile("barrier.cluster.wait.acquire.aligned;");
    asm volatile("tcgen05.fence::after_thread_sync;");

    const int warp_id = warp;
    const int lane_id = lane;
    __nv_bfloat16* smem_query = (__nv_bfloat16*)(smem_raw + 1024);
    #define smem_query_addr (smem + 1024)
    __nv_bfloat16* smem_database = (__nv_bfloat16*)(smem_raw + 33792);
    #define smem_database_addr (smem + 33792)
    float* smem_database_sq = (float*)(smem_raw + 50176);
    #define smem_database_sq_addr (smem + 50176)
    const int mbar_base = smem;
    #define query_full_addr (mbar_base + 0)
    #define query_empty_addr (mbar_base + 8)
    #define database_full_addr (mbar_base + 16)
    #define database_empty_addr (mbar_base + 24)
    #define score_full_addr (mbar_base + 32)
    #define score_empty_addr (mbar_base + 40)
    const int taddr = tmem_addr_storage[0];

    // ---- Role: load ----
    if (warp == 0) {
        { // load_main
            uint32_t _phase_query_empty_0 = 1;
            uint32_t _phase_database_empty_0 = 1;
            if (warp_id == 0) {
                if (elect_sync()) {
                    #pragma unroll 1
                    for (unsigned int work_idx = cluster_id; work_idx < total_work; work_idx += num_clusters) {
                        int split_idx = work_idx % split_count;
                        int query_work = work_idx / split_count;
                        int batch_idx = query_work / num_q_tile_pairs;
                        int q_tile_pair = query_work % num_q_tile_pairs;
                        int q_tile = q_tile_pair * 2 + cta_rank;
                        int off_q = q_tile * BLOCK_Q;
                        int global_q = batch_idx * Q + off_q;
                        int db_tile_start = split_idx * db_tiles_per_split;
                        mbarrier_wait(query_empty_addr, _phase_query_empty_0);
                        _phase_query_empty_0 ^= 1;
                        asm volatile(
                            "mbarrier.arrive.expect_tx.release.cta.shared::cluster.b64 _, [%0], %1;"
                            :: "r"((query_full_addr) & 0xFEFFFFFF), "r"((uint32_t)(32768)) : "memory");
                        asm volatile(
                            "cp.async.bulk.tensor.3d.shared::cluster.global.mbarrier::complete_tx::bytes.cta_group::2"
                            " [%0], [%1, {%2, %3, %4}], [%5];"
                            :: "r"(smem_query_addr), "l"(tmap_query), "r"(0), "r"(global_q), "r"(0),
                               "r"(((query_full_addr) & 0xFEFFFFFF)) : "memory");
                        #pragma unroll 1
                        for (int local_db_tile = 0; local_db_tile < db_tiles_per_split; local_db_tile++) {
                            int db_tile = db_tile_start + local_db_tile;
                            int off_m = db_tile * BLOCK_M;
                            int global_m = batch_idx * M + off_m;
                            mbarrier_wait(database_empty_addr, _phase_database_empty_0);
                            _phase_database_empty_0 ^= 1;
                            asm volatile(
                                "mbarrier.arrive.expect_tx.release.cta.shared::cluster.b64 _, [%0], %1;"
                                :: "r"((database_full_addr) & 0xFEFFFFFF), "r"((uint32_t)(16384)) : "memory");
                            asm volatile(
                                "cp.async.bulk.tensor.3d.shared::cluster.global.mbarrier::complete_tx::bytes.cta_group::2"
                                " [%0], [%1, {%2, %3, %4}], [%5];"
                                :: "r"(smem_database_addr), "l"(tmap_database), "r"(0), "r"(global_m), "r"(0),
                                   "r"(((database_full_addr) & 0xFEFFFFFF)) : "memory");
                        }
                    }
                }
            }
        }
    // ---- Role: mma ----
    } else if (warp == 1) {
        { // mma_main
            uint32_t _phase_query_full_0 = 0;
            uint32_t _phase_score_empty_0 = 1;
            uint32_t _phase_database_full_0 = 0;
            if (cta_rank == 0) {
                #pragma unroll 1
                for (unsigned int work_idx = cluster_id; work_idx < total_work; work_idx += num_clusters) {
                    mbarrier_wait(query_full_addr, _phase_query_full_0);
                    _phase_query_full_0 ^= 1;
                    #pragma unroll 1
                    for (int _local_db_tile = 0; _local_db_tile < db_tiles_per_split; _local_db_tile++) {
                        mbarrier_wait(score_empty_addr, _phase_score_empty_0);
                        _phase_score_empty_0 ^= 1;
                        mbarrier_wait(database_full_addr, _phase_database_full_0);
                        _phase_database_full_0 ^= 1;
                        asm volatile("tcgen05.fence::after_thread_sync;");
                        int _mma_ss_a_lo_0 = (smem_query_addr >> 4) & 0x3FFF;
                        int _mma_ss_b_lo_0 = (smem_database_addr >> 4) & 0x3FFF;
                        asm volatile(
                    "{\n\t"
                    ".reg .pred leader, p0, p1;\n\t"
                    ".reg .b32 adhi, bdhi, alo, blo, id, m0, m1, m2, m3, m4, m5, m6, m7;\n\t"
                    ".reg .b64 da, db;\n\t"
                    "elect.sync _|leader, 0xFFFFFFFF;\n\t"
                    "setp.ne.b32 p0, %3, 0;\n\t"
                    "setp.ne.b32 p1, 1, 0;\n\t"
                    "mov.b32 m0, 0; mov.b32 m1, 0; mov.b32 m2, 0; mov.b32 m3, 0;\n\tmov.b32 m4, 0; mov.b32 m5, 0; mov.b32 m6, 0; mov.b32 m7, 0;\n\t"
                    "mov.b32 adhi, 0x40004040;\n\t"
                    "mov.b32 bdhi, 0x40004040;\n\t"
                    "mov.b32 id, 270533776;\n\t"
                    "mov.b32 alo, %0;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p0;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "add.u32 alo, alo, 1018;\n\t"
                    "add.u32 blo, blo, 506;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "}\n"
                    :: "r"(_mma_ss_a_lo_0), "r"(_mma_ss_b_lo_0), "r"(taddr), "r"(0));
                        elect_commit_cg2_multicast(score_full_addr, (uint16_t)(3));
                        elect_commit_cg2_multicast(database_empty_addr, (uint16_t)(3));
                    }
                    elect_commit_cg2_multicast(query_empty_addr, (uint16_t)(3));
                }
            }
        }
    // ---- Role: compute ----
    } else if (warp >= 2 && warp <= 5) {
        const int warp_id_in_wg = warp % 4;
        const int my_row = warp_id_in_wg * 32 + lane;
        const int tmem_row_base_v = (warp_id_in_wg * 32) << 16;
        const int tmem_cross = taddr + TMEM_CROSS_OFFSET;
        { // compute_main
            uint32_t _phase_score_full_0 = 0;
            #pragma unroll 1
            for (unsigned int work_idx = cluster_id; work_idx < total_work; work_idx += num_clusters) {
                int split_idx = work_idx % split_count;
                int query_work = work_idx / split_count;
                int batch_idx = query_work / num_q_tile_pairs;
                int q_tile_pair = query_work % num_q_tile_pairs;
                int q_tile = q_tile_pair * 2 + cta_rank;
                int off_q = q_tile * BLOCK_Q;
                int q_idx = off_q + my_row;
                int valid_q = ((q_idx < Q) ? 1 : 0);
                float q_sq_val = 0.0f;
                if (valid_q != 0) {
                    q_sq_val = (float)query_sq[batch_idx * Q + q_idx];
                }
                float best_d[TOP_K_MAX];
                int best_i[TOP_K_MAX];
                #pragma unroll
                for (int kk = 0; kk < TOP_K_MAX; kk++) {
                    best_d[kk] = 3.4e+38f;
                    best_i[kk] = -1;
                }
                float worst_d = 3.4e+38f;
                int worst_pos = 0;
                int db_tile_start = split_idx * db_tiles_per_split;
                #pragma unroll 1
                for (int local_db_tile = 0; local_db_tile < db_tiles_per_split; local_db_tile++) {
                    int db_tile = db_tile_start + local_db_tile;
                    int db_start = db_tile * BLOCK_M;
                    int db_sq_idx = db_start + my_row;
                    if (my_row < BLOCK_M) {
                        if (db_sq_idx < M) {
                            smem_database_sq[my_row] = (float)database_sq[batch_idx * M + db_sq_idx];
                        } else {
                            smem_database_sq[my_row] = 0.0f;
                        }
                    }
                    asm volatile("barrier.sync 8, 128;");
                    mbarrier_wait(score_full_addr, _phase_score_full_0);
                    _phase_score_full_0 ^= 1;
                    int cross_addr = taddr + (cta_rank * BLOCK_Q + tmem_row_base_v << 16);
                    float dots[64];
                    asm volatile(
                        "tcgen05.ld.sync.aligned.32x32b.x64.b32"
                        " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63}, [%64];"
                        : "=f"(dots[0]), "=f"(dots[1]), "=f"(dots[2]), "=f"(dots[3]), "=f"(dots[4]), "=f"(dots[5]), "=f"(dots[6]), "=f"(dots[7]), "=f"(dots[8]), "=f"(dots[9]), "=f"(dots[10]), "=f"(dots[11]), "=f"(dots[12]), "=f"(dots[13]), "=f"(dots[14]), "=f"(dots[15]), "=f"(dots[16]), "=f"(dots[17]), "=f"(dots[18]), "=f"(dots[19]), "=f"(dots[20]), "=f"(dots[21]), "=f"(dots[22]), "=f"(dots[23]), "=f"(dots[24]), "=f"(dots[25]), "=f"(dots[26]), "=f"(dots[27]), "=f"(dots[28]), "=f"(dots[29]), "=f"(dots[30]), "=f"(dots[31]), "=f"(dots[32]), "=f"(dots[33]), "=f"(dots[34]), "=f"(dots[35]), "=f"(dots[36]), "=f"(dots[37]), "=f"(dots[38]), "=f"(dots[39]), "=f"(dots[40]), "=f"(dots[41]), "=f"(dots[42]), "=f"(dots[43]), "=f"(dots[44]), "=f"(dots[45]), "=f"(dots[46]), "=f"(dots[47]), "=f"(dots[48]), "=f"(dots[49]), "=f"(dots[50]), "=f"(dots[51]), "=f"(dots[52]), "=f"(dots[53]), "=f"(dots[54]), "=f"(dots[55]), "=f"(dots[56]), "=f"(dots[57]), "=f"(dots[58]), "=f"(dots[59]), "=f"(dots[60]), "=f"(dots[61]), "=f"(dots[62]), "=f"(dots[63])
                        : "r"(cross_addr)
                        : "memory");
                    asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                    asm volatile("barrier.sync 8, 128;");
                    if (elect_sync()) {
                        asm volatile(
                            "mbarrier.arrive.release.cta.shared::cluster.b64 _, [%0];"
                            :: "r"((score_empty_addr) & 0xFEFFFFFF) : "memory");
                    }
                    if (valid_q != 0) {
                        #pragma unroll 2
                        for (int col_base = 0; col_base < 64; col_base += 4) {
                            float dist_vec[4];
                            dist_vec[0] = dots[col_base];
                            dist_vec[1] = dots[col_base + 1];
                            dist_vec[2] = dots[col_base + 2];
                            dist_vec[3] = dots[col_base + 3];
                            const float2 _fma_b2_0 = {-2.0f, -2.0f};
                            const float2 _fma_c2_1 = {q_sq_val, q_sq_val};
                            #pragma unroll
                            for (int _lf = 0; _lf < 2; _lf++)
                                fma_f32x2_inplace(&reinterpret_cast<float2*>(dist_vec)[_lf], _fma_b2_0, _fma_c2_1);
                            float db_sq_vec[4];
                            db_sq_vec[0] = smem_database_sq[col_base];
                            db_sq_vec[1] = smem_database_sq[col_base + 1];
                            db_sq_vec[2] = smem_database_sq[col_base + 2];
                            db_sq_vec[3] = smem_database_sq[col_base + 3];
                            float _t0[4];
                            #pragma unroll
                            for (int _la = 0; _la < 2; _la++)
                                reinterpret_cast<float2*>(_t0)[_la] = add_f32x2(reinterpret_cast<float2*>(dist_vec)[_la], reinterpret_cast<const float2*>(db_sq_vec)[_la]);
                            float group_min = _t0[0];
                            if (_t0[1] < group_min) {
                                group_min = _t0[1];
                            }
                            if (_t0[2] < group_min) {
                                group_min = _t0[2];
                            }
                            if (_t0[3] < group_min) {
                                group_min = _t0[3];
                            }
                            if (group_min < worst_d) {
                                #pragma unroll
                                for (int vec_col = 0; vec_col < 4; vec_col++) {
                                    int db_idx = db_start + col_base + vec_col;
                                    if (db_idx < M) {
                                        float dist = _t0[vec_col];
                                        if (dist < worst_d) {
                                            best_d[worst_pos] = dist;
                                            best_i[worst_pos] = db_idx;
                                            worst_d = best_d[0];
                                            worst_pos = 0;
                                            #pragma unroll
                                            for (int scan_pos = 1; scan_pos < TOP_K_MAX; scan_pos++) {
                                                if (best_d[scan_pos] > worst_d) {
                                                    worst_d = best_d[scan_pos];
                                                    worst_pos = scan_pos;
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                    asm volatile("barrier.sync 8, 128;");
                }
                if (valid_q != 0) {
                    int out_base = ((split_idx * B + batch_idx) * Q + q_idx) * K;
                    #pragma unroll
                    for (int out_k = 0; out_k < TOP_K_MAX; out_k++) {
                        *((float*)(partial_dists + out_base + out_k)) = best_d[out_k];
                        *((int*)(partial_indices + out_base + out_k)) = best_i[out_k];
                    }
                }
            }
        }
    }

    // Cleanup
    asm volatile("barrier.cluster.arrive.release.aligned;");
    asm volatile("barrier.cluster.wait.acquire.aligned;");

    if (warp == 1) {
        asm volatile("tcgen05.dealloc.cta_group::2.sync.aligned.b32 %0, %1;" :: "r"(tmem_addr_storage[0]), "r"(64));
        asm volatile("tcgen05.relinquish_alloc_permit.cta_group::2.sync.aligned;");
    }
}

} // extern "C"

#undef BLOCK_M
#undef BLOCK_Q
#undef FEAT_D
#undef NUM_MAIN_STAGES
#undef SMEM_SMEM_DATABASE_OFF
#undef SMEM_SMEM_DATABASE_SQ_OFF
#undef SMEM_SMEM_DATABASE_SQ_STAGE_BYTES
#undef SMEM_SMEM_DATABASE_SQ_STRIDE
#undef SMEM_SMEM_DATABASE_STAGE_BYTES
#undef SMEM_SMEM_DATABASE_STRIDE
#undef SMEM_SMEM_QUERY_OFF
#undef SMEM_SMEM_QUERY_STAGE_BYTES
#undef SMEM_SMEM_QUERY_STRIDE
#undef SMEM_TOTAL
#undef THREADS
#undef TMEM_CROSS_OFFSET
#undef TMEM_NCOLS
#undef TOP_K_MAX
#undef database_empty_addr
#undef database_full_addr
#undef query_empty_addr
#undef query_full_addr
#undef score_empty_addr
#undef score_full_addr
#undef smem_database_addr
#undef smem_database_sq_addr
#undef smem_query_addr

#define NUM_MAIN_STAGES 1
#define THREADS 32
#define TOP_K_MAX 32

extern "C" {

__global__ __launch_bounds__(32, 1) void
kernel_knn_build_evolve_7bfc_k32_merge_s4_rowbase_cache(float* __restrict__ partial_dists, int32_t* __restrict__ partial_indices, float* __restrict__ out_dists, int32_t* __restrict__ out_indices, int K, int total_queries)
{
    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;


    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    const int warp_id = warp;
    const int lane_id = lane;

    // === Task calls (dependency order) ===
    int start_row = bid * 32 + tid;
    int stride = num_bids * 32;
    #pragma unroll 1
    for (int row = start_row; row < total_queries; row += stride) {
        int base_row = row * K;
        int split_stride = total_queries * K;
        int out_base = base_row;
        int split_base0 = base_row;
        int split_base1 = base_row + split_stride;
        int split_base2 = split_base1 + split_stride;
        int split_base3 = split_base2 + split_stride;
        int pos0 = 0;
        int pos1 = 0;
        int pos2 = 0;
        int pos3 = 0;
        float cand_d0 = (float)partial_dists[split_base0];
        int cand_i0 = partial_indices[split_base0];
        float cand_d1 = (float)partial_dists[split_base1];
        int cand_i1 = partial_indices[split_base1];
        float cand_d2 = (float)partial_dists[split_base2];
        int cand_i2 = partial_indices[split_base2];
        float cand_d3 = (float)partial_dists[split_base3];
        int cand_i3 = partial_indices[split_base3];
        #pragma unroll
        for (int out_k = 0; out_k < TOP_K_MAX; out_k++) {
            if (out_k < K) {
                int cand01_cmp = ((cand_d1 < cand_d0) ? 1 : 0);
                float best01_d = ((cand01_cmp != 0) ? cand_d1 : cand_d0);
                int best01_i = ((cand01_cmp != 0) ? cand_i1 : cand_i0);
                int best01_split = ((cand01_cmp != 0) ? 1 : 0);
                int cand23_cmp = ((cand_d3 < cand_d2) ? 1 : 0);
                float best23_d = ((cand23_cmp != 0) ? cand_d3 : cand_d2);
                int best23_i = ((cand23_cmp != 0) ? cand_i3 : cand_i2);
                int best23_split = ((cand23_cmp != 0) ? 3 : 2);
                int best_cmp = ((best23_d < best01_d) ? 1 : 0);
                float best_d = ((best_cmp != 0) ? best23_d : best01_d);
                int best_i = ((best_cmp != 0) ? best23_i : best01_i);
                int best_split = ((best_cmp != 0) ? best23_split : best01_split);
                *((float*)(out_dists + out_base + out_k)) = best_d;
                *((int*)(out_indices + out_base + out_k)) = best_i;
                if (out_k + 1 < K) {
                    if (best_split == 0) {
                        pos0 = pos0 + 1;
                        int next_addr0 = split_base0 + pos0;
                        cand_d0 = (float)partial_dists[next_addr0];
                        cand_i0 = partial_indices[next_addr0];
                    } else if (best_split == 1) {
                        pos1 = pos1 + 1;
                        int next_addr1 = split_base1 + pos1;
                        cand_d1 = (float)partial_dists[next_addr1];
                        cand_i1 = partial_indices[next_addr1];
                    } else {
                        if (best_split == 2) {
                            pos2 = pos2 + 1;
                            int next_addr2 = split_base2 + pos2;
                            cand_d2 = (float)partial_dists[next_addr2];
                            cand_i2 = partial_indices[next_addr2];
                        } else {
                            pos3 = pos3 + 1;
                            int next_addr3 = split_base3 + pos3;
                            cand_d3 = (float)partial_dists[next_addr3];
                            cand_i3 = partial_indices[next_addr3];
                        }
                    }
                }
            }
        }
    }
}

} // extern "C"

#undef NUM_MAIN_STAGES
#undef THREADS
#undef TOP_K_MAX

#define NUM_MAIN_STAGES 1
#define THREADS 32
#define TOP_K_MAX 12

extern "C" {

__global__ __launch_bounds__(32, 1) void
kernel_knn_build_evolve_7bfc_k32_merge_s4_rowbase_cache_k12split(float* __restrict__ partial_dists, int32_t* __restrict__ partial_indices, float* __restrict__ out_dists, int32_t* __restrict__ out_indices, int K, int total_queries)
{
    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;


    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    const int warp_id = warp;
    const int lane_id = lane;

    // === Task calls (dependency order) ===
    int start_row = bid * 32 + tid;
    int stride = num_bids * 32;
    #pragma unroll 1
    for (int row = start_row; row < total_queries; row += stride) {
        int base_row = row * K;
        int split_stride = total_queries * K;
        int out_base = base_row;
        int split_base0 = base_row;
        int split_base1 = base_row + split_stride;
        int split_base2 = split_base1 + split_stride;
        int split_base3 = split_base2 + split_stride;
        int pos0 = 0;
        int pos1 = 0;
        int pos2 = 0;
        int pos3 = 0;
        float cand_d0 = (float)partial_dists[split_base0];
        int cand_i0 = partial_indices[split_base0];
        float cand_d1 = (float)partial_dists[split_base1];
        int cand_i1 = partial_indices[split_base1];
        float cand_d2 = (float)partial_dists[split_base2];
        int cand_i2 = partial_indices[split_base2];
        float cand_d3 = (float)partial_dists[split_base3];
        int cand_i3 = partial_indices[split_base3];
        #pragma unroll
        for (int out_k = 0; out_k < TOP_K_MAX; out_k++) {
            if (out_k < K) {
                int cand01_cmp = ((cand_d1 < cand_d0) ? 1 : 0);
                float best01_d = ((cand01_cmp != 0) ? cand_d1 : cand_d0);
                int best01_i = ((cand01_cmp != 0) ? cand_i1 : cand_i0);
                int best01_split = ((cand01_cmp != 0) ? 1 : 0);
                int cand23_cmp = ((cand_d3 < cand_d2) ? 1 : 0);
                float best23_d = ((cand23_cmp != 0) ? cand_d3 : cand_d2);
                int best23_i = ((cand23_cmp != 0) ? cand_i3 : cand_i2);
                int best23_split = ((cand23_cmp != 0) ? 3 : 2);
                int best_cmp = ((best23_d < best01_d) ? 1 : 0);
                float best_d = ((best_cmp != 0) ? best23_d : best01_d);
                int best_i = ((best_cmp != 0) ? best23_i : best01_i);
                int best_split = ((best_cmp != 0) ? best23_split : best01_split);
                *((float*)(out_dists + out_base + out_k)) = best_d;
                *((int*)(out_indices + out_base + out_k)) = best_i;
                if (out_k + 1 < K) {
                    if (best_split == 0) {
                        pos0 = pos0 + 1;
                        int next_addr0 = split_base0 + pos0;
                        cand_d0 = (float)partial_dists[next_addr0];
                        cand_i0 = partial_indices[next_addr0];
                    } else if (best_split == 1) {
                        pos1 = pos1 + 1;
                        int next_addr1 = split_base1 + pos1;
                        cand_d1 = (float)partial_dists[next_addr1];
                        cand_i1 = partial_indices[next_addr1];
                    } else {
                        if (best_split == 2) {
                            pos2 = pos2 + 1;
                            int next_addr2 = split_base2 + pos2;
                            cand_d2 = (float)partial_dists[next_addr2];
                            cand_i2 = partial_indices[next_addr2];
                        } else {
                            pos3 = pos3 + 1;
                            int next_addr3 = split_base3 + pos3;
                            cand_d3 = (float)partial_dists[next_addr3];
                            cand_i3 = partial_indices[next_addr3];
                        }
                    }
                }
            }
        }
    }
}

} // extern "C"

#undef NUM_MAIN_STAGES
#undef THREADS
#undef TOP_K_MAX

#define NUM_MAIN_STAGES 1
#define THREADS 32
#define TOP_K_MAX 16

extern "C" {

__global__ __launch_bounds__(32, 1) void
kernel_knn_build_evolve_7bfc_k32_merge_s4_rowbase_cache_k16split(float* __restrict__ partial_dists, int32_t* __restrict__ partial_indices, float* __restrict__ out_dists, int32_t* __restrict__ out_indices, int K, int total_queries)
{
    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;


    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    const int warp_id = warp;
    const int lane_id = lane;

    // === Task calls (dependency order) ===
    int start_row = bid * 32 + tid;
    int stride = num_bids * 32;
    #pragma unroll 1
    for (int row = start_row; row < total_queries; row += stride) {
        int base_row = row * K;
        int split_stride = total_queries * K;
        int out_base = base_row;
        int split_base0 = base_row;
        int split_base1 = base_row + split_stride;
        int split_base2 = split_base1 + split_stride;
        int split_base3 = split_base2 + split_stride;
        int pos0 = 0;
        int pos1 = 0;
        int pos2 = 0;
        int pos3 = 0;
        float cand_d0 = (float)partial_dists[split_base0];
        int cand_i0 = partial_indices[split_base0];
        float cand_d1 = (float)partial_dists[split_base1];
        int cand_i1 = partial_indices[split_base1];
        float cand_d2 = (float)partial_dists[split_base2];
        int cand_i2 = partial_indices[split_base2];
        float cand_d3 = (float)partial_dists[split_base3];
        int cand_i3 = partial_indices[split_base3];
        #pragma unroll
        for (int out_k = 0; out_k < TOP_K_MAX; out_k++) {
            if (out_k < K) {
                int cand01_cmp = ((cand_d1 < cand_d0) ? 1 : 0);
                float best01_d = ((cand01_cmp != 0) ? cand_d1 : cand_d0);
                int best01_i = ((cand01_cmp != 0) ? cand_i1 : cand_i0);
                int best01_split = ((cand01_cmp != 0) ? 1 : 0);
                int cand23_cmp = ((cand_d3 < cand_d2) ? 1 : 0);
                float best23_d = ((cand23_cmp != 0) ? cand_d3 : cand_d2);
                int best23_i = ((cand23_cmp != 0) ? cand_i3 : cand_i2);
                int best23_split = ((cand23_cmp != 0) ? 3 : 2);
                int best_cmp = ((best23_d < best01_d) ? 1 : 0);
                float best_d = ((best_cmp != 0) ? best23_d : best01_d);
                int best_i = ((best_cmp != 0) ? best23_i : best01_i);
                int best_split = ((best_cmp != 0) ? best23_split : best01_split);
                *((float*)(out_dists + out_base + out_k)) = best_d;
                *((int*)(out_indices + out_base + out_k)) = best_i;
                if (out_k + 1 < K) {
                    if (best_split == 0) {
                        pos0 = pos0 + 1;
                        int next_addr0 = split_base0 + pos0;
                        cand_d0 = (float)partial_dists[next_addr0];
                        cand_i0 = partial_indices[next_addr0];
                    } else if (best_split == 1) {
                        pos1 = pos1 + 1;
                        int next_addr1 = split_base1 + pos1;
                        cand_d1 = (float)partial_dists[next_addr1];
                        cand_i1 = partial_indices[next_addr1];
                    } else {
                        if (best_split == 2) {
                            pos2 = pos2 + 1;
                            int next_addr2 = split_base2 + pos2;
                            cand_d2 = (float)partial_dists[next_addr2];
                            cand_i2 = partial_indices[next_addr2];
                        } else {
                            pos3 = pos3 + 1;
                            int next_addr3 = split_base3 + pos3;
                            cand_d3 = (float)partial_dists[next_addr3];
                            cand_i3 = partial_indices[next_addr3];
                        }
                    }
                }
            }
        }
    }
}

} // extern "C"

#undef NUM_MAIN_STAGES
#undef THREADS
#undef TOP_K_MAX

#define NUM_MAIN_STAGES 1
#define THREADS 32
#define TOP_K_MAX 20

extern "C" {

__global__ __launch_bounds__(32, 1) void
kernel_knn_build_evolve_7bfc_k32_merge_s4_rowbase_cache_k20split(float* __restrict__ partial_dists, int32_t* __restrict__ partial_indices, float* __restrict__ out_dists, int32_t* __restrict__ out_indices, int K, int total_queries)
{
    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;


    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    const int warp_id = warp;
    const int lane_id = lane;

    // === Task calls (dependency order) ===
    int start_row = bid * 32 + tid;
    int stride = num_bids * 32;
    #pragma unroll 1
    for (int row = start_row; row < total_queries; row += stride) {
        int base_row = row * K;
        int split_stride = total_queries * K;
        int out_base = base_row;
        int split_base0 = base_row;
        int split_base1 = base_row + split_stride;
        int split_base2 = split_base1 + split_stride;
        int split_base3 = split_base2 + split_stride;
        int pos0 = 0;
        int pos1 = 0;
        int pos2 = 0;
        int pos3 = 0;
        float cand_d0 = (float)partial_dists[split_base0];
        int cand_i0 = partial_indices[split_base0];
        float cand_d1 = (float)partial_dists[split_base1];
        int cand_i1 = partial_indices[split_base1];
        float cand_d2 = (float)partial_dists[split_base2];
        int cand_i2 = partial_indices[split_base2];
        float cand_d3 = (float)partial_dists[split_base3];
        int cand_i3 = partial_indices[split_base3];
        #pragma unroll
        for (int out_k = 0; out_k < TOP_K_MAX; out_k++) {
            if (out_k < K) {
                int cand01_cmp = ((cand_d1 < cand_d0) ? 1 : 0);
                float best01_d = ((cand01_cmp != 0) ? cand_d1 : cand_d0);
                int best01_i = ((cand01_cmp != 0) ? cand_i1 : cand_i0);
                int best01_split = ((cand01_cmp != 0) ? 1 : 0);
                int cand23_cmp = ((cand_d3 < cand_d2) ? 1 : 0);
                float best23_d = ((cand23_cmp != 0) ? cand_d3 : cand_d2);
                int best23_i = ((cand23_cmp != 0) ? cand_i3 : cand_i2);
                int best23_split = ((cand23_cmp != 0) ? 3 : 2);
                int best_cmp = ((best23_d < best01_d) ? 1 : 0);
                float best_d = ((best_cmp != 0) ? best23_d : best01_d);
                int best_i = ((best_cmp != 0) ? best23_i : best01_i);
                int best_split = ((best_cmp != 0) ? best23_split : best01_split);
                *((float*)(out_dists + out_base + out_k)) = best_d;
                *((int*)(out_indices + out_base + out_k)) = best_i;
                if (out_k + 1 < K) {
                    if (best_split == 0) {
                        pos0 = pos0 + 1;
                        int next_addr0 = split_base0 + pos0;
                        cand_d0 = (float)partial_dists[next_addr0];
                        cand_i0 = partial_indices[next_addr0];
                    } else if (best_split == 1) {
                        pos1 = pos1 + 1;
                        int next_addr1 = split_base1 + pos1;
                        cand_d1 = (float)partial_dists[next_addr1];
                        cand_i1 = partial_indices[next_addr1];
                    } else {
                        if (best_split == 2) {
                            pos2 = pos2 + 1;
                            int next_addr2 = split_base2 + pos2;
                            cand_d2 = (float)partial_dists[next_addr2];
                            cand_i2 = partial_indices[next_addr2];
                        } else {
                            pos3 = pos3 + 1;
                            int next_addr3 = split_base3 + pos3;
                            cand_d3 = (float)partial_dists[next_addr3];
                            cand_i3 = partial_indices[next_addr3];
                        }
                    }
                }
            }
        }
    }
}

} // extern "C"

#undef NUM_MAIN_STAGES
#undef THREADS
#undef TOP_K_MAX

#define NUM_MAIN_STAGES 1
#define THREADS 32
#define TOP_K_MAX 25

extern "C" {

__global__ __launch_bounds__(32, 1) void
kernel_knn_build_evolve_7bfc_k32_merge_s4_rowbase_cache_k25split(float* __restrict__ partial_dists, int32_t* __restrict__ partial_indices, float* __restrict__ out_dists, int32_t* __restrict__ out_indices, int K, int total_queries)
{
    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;


    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    const int warp_id = warp;
    const int lane_id = lane;

    // === Task calls (dependency order) ===
    int start_row = bid * 32 + tid;
    int stride = num_bids * 32;
    #pragma unroll 1
    for (int row = start_row; row < total_queries; row += stride) {
        int base_row = row * K;
        int split_stride = total_queries * K;
        int out_base = base_row;
        int split_base0 = base_row;
        int split_base1 = base_row + split_stride;
        int split_base2 = split_base1 + split_stride;
        int split_base3 = split_base2 + split_stride;
        int pos0 = 0;
        int pos1 = 0;
        int pos2 = 0;
        int pos3 = 0;
        float cand_d0 = (float)partial_dists[split_base0];
        int cand_i0 = partial_indices[split_base0];
        float cand_d1 = (float)partial_dists[split_base1];
        int cand_i1 = partial_indices[split_base1];
        float cand_d2 = (float)partial_dists[split_base2];
        int cand_i2 = partial_indices[split_base2];
        float cand_d3 = (float)partial_dists[split_base3];
        int cand_i3 = partial_indices[split_base3];
        #pragma unroll
        for (int out_k = 0; out_k < TOP_K_MAX; out_k++) {
            if (out_k < K) {
                int cand01_cmp = ((cand_d1 < cand_d0) ? 1 : 0);
                float best01_d = ((cand01_cmp != 0) ? cand_d1 : cand_d0);
                int best01_i = ((cand01_cmp != 0) ? cand_i1 : cand_i0);
                int best01_split = ((cand01_cmp != 0) ? 1 : 0);
                int cand23_cmp = ((cand_d3 < cand_d2) ? 1 : 0);
                float best23_d = ((cand23_cmp != 0) ? cand_d3 : cand_d2);
                int best23_i = ((cand23_cmp != 0) ? cand_i3 : cand_i2);
                int best23_split = ((cand23_cmp != 0) ? 3 : 2);
                int best_cmp = ((best23_d < best01_d) ? 1 : 0);
                float best_d = ((best_cmp != 0) ? best23_d : best01_d);
                int best_i = ((best_cmp != 0) ? best23_i : best01_i);
                int best_split = ((best_cmp != 0) ? best23_split : best01_split);
                *((float*)(out_dists + out_base + out_k)) = best_d;
                *((int*)(out_indices + out_base + out_k)) = best_i;
                if (out_k + 1 < K) {
                    if (best_split == 0) {
                        pos0 = pos0 + 1;
                        int next_addr0 = split_base0 + pos0;
                        cand_d0 = (float)partial_dists[next_addr0];
                        cand_i0 = partial_indices[next_addr0];
                    } else if (best_split == 1) {
                        pos1 = pos1 + 1;
                        int next_addr1 = split_base1 + pos1;
                        cand_d1 = (float)partial_dists[next_addr1];
                        cand_i1 = partial_indices[next_addr1];
                    } else {
                        if (best_split == 2) {
                            pos2 = pos2 + 1;
                            int next_addr2 = split_base2 + pos2;
                            cand_d2 = (float)partial_dists[next_addr2];
                            cand_i2 = partial_indices[next_addr2];
                        } else {
                            pos3 = pos3 + 1;
                            int next_addr3 = split_base3 + pos3;
                            cand_d3 = (float)partial_dists[next_addr3];
                            cand_i3 = partial_indices[next_addr3];
                        }
                    }
                }
            }
        }
    }
}

} // extern "C"

#undef NUM_MAIN_STAGES
#undef THREADS
#undef TOP_K_MAX

#define NUM_MAIN_STAGES 1
#define THREADS 32
#define TOP_K_MAX 30

extern "C" {

__global__ __launch_bounds__(32, 1) void
kernel_knn_build_evolve_7bfc_k32_merge_s4_rowbase_cache_k30split(float* __restrict__ partial_dists, int32_t* __restrict__ partial_indices, float* __restrict__ out_dists, int32_t* __restrict__ out_indices, int K, int total_queries)
{
    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;


    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    const int warp_id = warp;
    const int lane_id = lane;

    // === Task calls (dependency order) ===
    int start_row = bid * 32 + tid;
    int stride = num_bids * 32;
    #pragma unroll 1
    for (int row = start_row; row < total_queries; row += stride) {
        int base_row = row * K;
        int split_stride = total_queries * K;
        int out_base = base_row;
        int split_base0 = base_row;
        int split_base1 = base_row + split_stride;
        int split_base2 = split_base1 + split_stride;
        int split_base3 = split_base2 + split_stride;
        int pos0 = 0;
        int pos1 = 0;
        int pos2 = 0;
        int pos3 = 0;
        float cand_d0 = (float)partial_dists[split_base0];
        int cand_i0 = partial_indices[split_base0];
        float cand_d1 = (float)partial_dists[split_base1];
        int cand_i1 = partial_indices[split_base1];
        float cand_d2 = (float)partial_dists[split_base2];
        int cand_i2 = partial_indices[split_base2];
        float cand_d3 = (float)partial_dists[split_base3];
        int cand_i3 = partial_indices[split_base3];
        #pragma unroll
        for (int out_k = 0; out_k < TOP_K_MAX; out_k++) {
            if (out_k < K) {
                int cand01_cmp = ((cand_d1 < cand_d0) ? 1 : 0);
                float best01_d = ((cand01_cmp != 0) ? cand_d1 : cand_d0);
                int best01_i = ((cand01_cmp != 0) ? cand_i1 : cand_i0);
                int best01_split = ((cand01_cmp != 0) ? 1 : 0);
                int cand23_cmp = ((cand_d3 < cand_d2) ? 1 : 0);
                float best23_d = ((cand23_cmp != 0) ? cand_d3 : cand_d2);
                int best23_i = ((cand23_cmp != 0) ? cand_i3 : cand_i2);
                int best23_split = ((cand23_cmp != 0) ? 3 : 2);
                int best_cmp = ((best23_d < best01_d) ? 1 : 0);
                float best_d = ((best_cmp != 0) ? best23_d : best01_d);
                int best_i = ((best_cmp != 0) ? best23_i : best01_i);
                int best_split = ((best_cmp != 0) ? best23_split : best01_split);
                *((float*)(out_dists + out_base + out_k)) = best_d;
                *((int*)(out_indices + out_base + out_k)) = best_i;
                if (out_k + 1 < K) {
                    if (best_split == 0) {
                        pos0 = pos0 + 1;
                        int next_addr0 = split_base0 + pos0;
                        cand_d0 = (float)partial_dists[next_addr0];
                        cand_i0 = partial_indices[next_addr0];
                    } else if (best_split == 1) {
                        pos1 = pos1 + 1;
                        int next_addr1 = split_base1 + pos1;
                        cand_d1 = (float)partial_dists[next_addr1];
                        cand_i1 = partial_indices[next_addr1];
                    } else {
                        if (best_split == 2) {
                            pos2 = pos2 + 1;
                            int next_addr2 = split_base2 + pos2;
                            cand_d2 = (float)partial_dists[next_addr2];
                            cand_i2 = partial_indices[next_addr2];
                        } else {
                            pos3 = pos3 + 1;
                            int next_addr3 = split_base3 + pos3;
                            cand_d3 = (float)partial_dists[next_addr3];
                            cand_i3 = partial_indices[next_addr3];
                        }
                    }
                }
            }
        }
    }
}

} // extern "C"

#undef NUM_MAIN_STAGES
#undef THREADS
#undef TOP_K_MAX

#define NUM_MAIN_STAGES 1
#define THREADS 32
#define TOP_K_MAX 32

extern "C" {

__global__ __launch_bounds__(32, 1) void
kernel_knn_build_evolve_7bfc_k32_merge_s4_rowbase_cache_k32split(float* __restrict__ partial_dists, int32_t* __restrict__ partial_indices, float* __restrict__ out_dists, int32_t* __restrict__ out_indices, int K, int total_queries)
{
    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;


    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    const int warp_id = warp;
    const int lane_id = lane;

    // === Task calls (dependency order) ===
    int start_row = bid * 32 + tid;
    int stride = num_bids * 32;
    #pragma unroll 1
    for (int row = start_row; row < total_queries; row += stride) {
        int base_row = row * K;
        int split_stride = total_queries * K;
        int out_base = base_row;
        int split_base0 = base_row;
        int split_base1 = base_row + split_stride;
        int split_base2 = split_base1 + split_stride;
        int split_base3 = split_base2 + split_stride;
        int pos0 = 0;
        int pos1 = 0;
        int pos2 = 0;
        int pos3 = 0;
        float cand_d0 = (float)partial_dists[split_base0];
        int cand_i0 = partial_indices[split_base0];
        float cand_d1 = (float)partial_dists[split_base1];
        int cand_i1 = partial_indices[split_base1];
        float cand_d2 = (float)partial_dists[split_base2];
        int cand_i2 = partial_indices[split_base2];
        float cand_d3 = (float)partial_dists[split_base3];
        int cand_i3 = partial_indices[split_base3];
        #pragma unroll
        for (int out_k = 0; out_k < TOP_K_MAX; out_k++) {
            if (out_k < K) {
                int cand01_cmp = ((cand_d1 < cand_d0) ? 1 : 0);
                float best01_d = ((cand01_cmp != 0) ? cand_d1 : cand_d0);
                int best01_i = ((cand01_cmp != 0) ? cand_i1 : cand_i0);
                int best01_split = ((cand01_cmp != 0) ? 1 : 0);
                int cand23_cmp = ((cand_d3 < cand_d2) ? 1 : 0);
                float best23_d = ((cand23_cmp != 0) ? cand_d3 : cand_d2);
                int best23_i = ((cand23_cmp != 0) ? cand_i3 : cand_i2);
                int best23_split = ((cand23_cmp != 0) ? 3 : 2);
                int best_cmp = ((best23_d < best01_d) ? 1 : 0);
                float best_d = ((best_cmp != 0) ? best23_d : best01_d);
                int best_i = ((best_cmp != 0) ? best23_i : best01_i);
                int best_split = ((best_cmp != 0) ? best23_split : best01_split);
                *((float*)(out_dists + out_base + out_k)) = best_d;
                *((int*)(out_indices + out_base + out_k)) = best_i;
                if (out_k + 1 < K) {
                    if (best_split == 0) {
                        pos0 = pos0 + 1;
                        int next_addr0 = split_base0 + pos0;
                        cand_d0 = (float)partial_dists[next_addr0];
                        cand_i0 = partial_indices[next_addr0];
                    } else if (best_split == 1) {
                        pos1 = pos1 + 1;
                        int next_addr1 = split_base1 + pos1;
                        cand_d1 = (float)partial_dists[next_addr1];
                        cand_i1 = partial_indices[next_addr1];
                    } else {
                        if (best_split == 2) {
                            pos2 = pos2 + 1;
                            int next_addr2 = split_base2 + pos2;
                            cand_d2 = (float)partial_dists[next_addr2];
                            cand_i2 = partial_indices[next_addr2];
                        } else {
                            pos3 = pos3 + 1;
                            int next_addr3 = split_base3 + pos3;
                            cand_d3 = (float)partial_dists[next_addr3];
                            cand_i3 = partial_indices[next_addr3];
                        }
                    }
                }
            }
        }
    }
}

} // extern "C"

#undef NUM_MAIN_STAGES
#undef THREADS
#undef TOP_K_MAX

#define NUM_MAIN_STAGES 1
#define THREADS 32
#define TOP_K_MAX 32
#define SPLIT_COUNT 4

extern "C" {

__global__ __launch_bounds__(32, 1) void
kernel_knn_build_evolve_7bfc_k32_merge_s4_unordered(float* __restrict__ partial_dists, int32_t* __restrict__ partial_indices, float* __restrict__ out_dists, int32_t* __restrict__ out_indices, int total_queries)
{
    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;


    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    const int warp_id = warp;
    const int lane_id = lane;

    // === Task calls (dependency order) ===
    int start_row = bid * 32 + tid;
    int stride = num_bids * 32;
    #pragma unroll 1
    for (int row = start_row; row < total_queries; row += stride) {
        int base_row = row * TOP_K_MAX;
        int split_stride = total_queries * TOP_K_MAX;
        float best_d[TOP_K_MAX];
        int best_i[TOP_K_MAX];
        #pragma unroll
        for (int kk = 0; kk < TOP_K_MAX; kk++) {
            best_d[kk] = 3.4e+38f;
            best_i[kk] = -1;
        }
        float worst_d = 3.4e+38f;
        int worst_pos = 0;
        #pragma unroll
        for (int split_idx = 0; split_idx < SPLIT_COUNT; split_idx++) {
            int partial_base = base_row + split_idx * split_stride;
            #pragma unroll
            for (int cand_k = 0; cand_k < TOP_K_MAX; cand_k++) {
                float cand_d = (float)partial_dists[partial_base + cand_k];
                int cand_i = partial_indices[partial_base + cand_k];
                if (cand_d < worst_d) {
                    best_d[worst_pos] = cand_d;
                    best_i[worst_pos] = cand_i;
                    worst_d = best_d[0];
                    worst_pos = 0;
                    #pragma unroll
                    for (int scan_pos = 1; scan_pos < TOP_K_MAX; scan_pos++) {
                        if (best_d[scan_pos] > worst_d) {
                            worst_d = best_d[scan_pos];
                            worst_pos = scan_pos;
                        }
                    }
                }
            }
        }
        #pragma unroll
        for (int out_k = 0; out_k < TOP_K_MAX; out_k++) {
            *((float*)(out_dists + base_row + out_k)) = best_d[out_k];
            *((int*)(out_indices + base_row + out_k)) = best_i[out_k];
        }
    }
}

} // extern "C"

#undef NUM_MAIN_STAGES
#undef SPLIT_COUNT
#undef THREADS
#undef TOP_K_MAX

#define NUM_MAIN_STAGES 1
#define THREADS 32
#define TOP_K_MAX 20
#define SPLIT_COUNT 4

extern "C" {

__global__ __launch_bounds__(32, 1) void
kernel_knn_build_evolve_7bfc_k32_merge_s4_unordered_k20unordered(float* __restrict__ partial_dists, int32_t* __restrict__ partial_indices, float* __restrict__ out_dists, int32_t* __restrict__ out_indices, int total_queries)
{
    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;


    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    const int warp_id = warp;
    const int lane_id = lane;

    // === Task calls (dependency order) ===
    int start_row = bid * 32 + tid;
    int stride = num_bids * 32;
    #pragma unroll 1
    for (int row = start_row; row < total_queries; row += stride) {
        int base_row = row * TOP_K_MAX;
        int split_stride = total_queries * TOP_K_MAX;
        float best_d[TOP_K_MAX];
        int best_i[TOP_K_MAX];
        #pragma unroll
        for (int kk = 0; kk < TOP_K_MAX; kk++) {
            best_d[kk] = 3.4e+38f;
            best_i[kk] = -1;
        }
        float worst_d = 3.4e+38f;
        int worst_pos = 0;
        #pragma unroll
        for (int split_idx = 0; split_idx < SPLIT_COUNT; split_idx++) {
            int partial_base = base_row + split_idx * split_stride;
            #pragma unroll
            for (int cand_k = 0; cand_k < TOP_K_MAX; cand_k++) {
                float cand_d = (float)partial_dists[partial_base + cand_k];
                int cand_i = partial_indices[partial_base + cand_k];
                if (cand_d < worst_d) {
                    best_d[worst_pos] = cand_d;
                    best_i[worst_pos] = cand_i;
                    worst_d = best_d[0];
                    worst_pos = 0;
                    #pragma unroll
                    for (int scan_pos = 1; scan_pos < TOP_K_MAX; scan_pos++) {
                        if (best_d[scan_pos] > worst_d) {
                            worst_d = best_d[scan_pos];
                            worst_pos = scan_pos;
                        }
                    }
                }
            }
        }
        #pragma unroll
        for (int out_k = 0; out_k < TOP_K_MAX; out_k++) {
            *((float*)(out_dists + base_row + out_k)) = best_d[out_k];
            *((int*)(out_indices + base_row + out_k)) = best_i[out_k];
        }
    }
}

} // extern "C"

#undef NUM_MAIN_STAGES
#undef SPLIT_COUNT
#undef THREADS
#undef TOP_K_MAX

#define NUM_MAIN_STAGES 1
#define THREADS 32
#define TOP_K_MAX 30
#define SPLIT_COUNT 4

extern "C" {

__global__ __launch_bounds__(32, 1) void
kernel_knn_build_evolve_7bfc_k32_merge_s4_unordered_k30unordered(float* __restrict__ partial_dists, int32_t* __restrict__ partial_indices, float* __restrict__ out_dists, int32_t* __restrict__ out_indices, int total_queries)
{
    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;


    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    const int warp_id = warp;
    const int lane_id = lane;

    // === Task calls (dependency order) ===
    int start_row = bid * 32 + tid;
    int stride = num_bids * 32;
    #pragma unroll 1
    for (int row = start_row; row < total_queries; row += stride) {
        int base_row = row * TOP_K_MAX;
        int split_stride = total_queries * TOP_K_MAX;
        float best_d[TOP_K_MAX];
        int best_i[TOP_K_MAX];
        #pragma unroll
        for (int kk = 0; kk < TOP_K_MAX; kk++) {
            best_d[kk] = 3.4e+38f;
            best_i[kk] = -1;
        }
        float worst_d = 3.4e+38f;
        int worst_pos = 0;
        #pragma unroll
        for (int split_idx = 0; split_idx < SPLIT_COUNT; split_idx++) {
            int partial_base = base_row + split_idx * split_stride;
            #pragma unroll
            for (int cand_k = 0; cand_k < TOP_K_MAX; cand_k++) {
                float cand_d = (float)partial_dists[partial_base + cand_k];
                int cand_i = partial_indices[partial_base + cand_k];
                if (cand_d < worst_d) {
                    best_d[worst_pos] = cand_d;
                    best_i[worst_pos] = cand_i;
                    worst_d = best_d[0];
                    worst_pos = 0;
                    #pragma unroll
                    for (int scan_pos = 1; scan_pos < TOP_K_MAX; scan_pos++) {
                        if (best_d[scan_pos] > worst_d) {
                            worst_d = best_d[scan_pos];
                            worst_pos = scan_pos;
                        }
                    }
                }
            }
        }
        #pragma unroll
        for (int out_k = 0; out_k < TOP_K_MAX; out_k++) {
            *((float*)(out_dists + base_row + out_k)) = best_d[out_k];
            *((int*)(out_indices + base_row + out_k)) = best_i[out_k];
        }
    }
}

} // extern "C"

#undef NUM_MAIN_STAGES
#undef SPLIT_COUNT
#undef THREADS
#undef TOP_K_MAX

#define NUM_MAIN_STAGES 1
#define THREADS 32
#define TOP_K_MAX 30
#define SPLIT_COUNT 8

extern "C" {

__global__ __launch_bounds__(32, 1) void
kernel_knn_build_evolve_7bfc_k30_merge_s8_rowbase_cache(float* __restrict__ partial_dists, int32_t* __restrict__ partial_indices, float* __restrict__ out_dists, int32_t* __restrict__ out_indices, int total_queries)
{
    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;


    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    const int warp_id = warp;
    const int lane_id = lane;

    // === Task calls (dependency order) ===
    int start_row = bid * 32 + tid;
    int stride = num_bids * 32;
    #pragma unroll 1
    for (int row = start_row; row < total_queries; row += stride) {
        int base_row = row * TOP_K_MAX;
        int split_stride = total_queries * TOP_K_MAX;
        int out_base = base_row;
        int split_pos[SPLIT_COUNT];
        int split_base[SPLIT_COUNT];
        float cand_d[SPLIT_COUNT];
        int cand_i[SPLIT_COUNT];
        #pragma unroll
        for (int split_idx = 0; split_idx < SPLIT_COUNT; split_idx++) {
            split_pos[split_idx] = 0;
            split_base[split_idx] = base_row + split_idx * split_stride;
            cand_d[split_idx] = (float)partial_dists[split_base[split_idx]];
            cand_i[split_idx] = partial_indices[split_base[split_idx]];
        }
        #pragma unroll
        for (int out_k = 0; out_k < TOP_K_MAX; out_k++) {
            float best_d = cand_d[0];
            int best_i = cand_i[0];
            int best_split = 0;
            #pragma unroll
            for (int split_idx = 1; split_idx < SPLIT_COUNT; split_idx++) {
                if (cand_d[split_idx] < best_d) {
                    best_d = cand_d[split_idx];
                    best_i = cand_i[split_idx];
                    best_split = split_idx;
                }
            }
            *((float*)(out_dists + out_base + out_k)) = best_d;
            *((int*)(out_indices + out_base + out_k)) = best_i;
            split_pos[best_split] = split_pos[best_split] + 1;
            if (out_k + 1 < TOP_K_MAX) {
                int next_pos = split_pos[best_split];
                int next_addr = split_base[best_split] + next_pos;
                cand_d[best_split] = (float)partial_dists[next_addr];
                cand_i[best_split] = partial_indices[next_addr];
            }
        }
    }
}

} // extern "C"

#undef NUM_MAIN_STAGES
#undef SPLIT_COUNT
#undef THREADS
#undef TOP_K_MAX

#define NUM_MAIN_STAGES 1
#define THREADS 32
#define TOP_K_MAX 12
#define SPLIT_COUNT 8

extern "C" {

__global__ __launch_bounds__(32, 1) void
kernel_knn_build_evolve_7bfc_k30_merge_s8_rowbase_cache_k12s8(float* __restrict__ partial_dists, int32_t* __restrict__ partial_indices, float* __restrict__ out_dists, int32_t* __restrict__ out_indices, int total_queries)
{
    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;


    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    const int warp_id = warp;
    const int lane_id = lane;

    // === Task calls (dependency order) ===
    int start_row = bid * 32 + tid;
    int stride = num_bids * 32;
    #pragma unroll 1
    for (int row = start_row; row < total_queries; row += stride) {
        int base_row = row * TOP_K_MAX;
        int split_stride = total_queries * TOP_K_MAX;
        int out_base = base_row;
        int split_pos[SPLIT_COUNT];
        int split_base[SPLIT_COUNT];
        float cand_d[SPLIT_COUNT];
        int cand_i[SPLIT_COUNT];
        #pragma unroll
        for (int split_idx = 0; split_idx < SPLIT_COUNT; split_idx++) {
            split_pos[split_idx] = 0;
            split_base[split_idx] = base_row + split_idx * split_stride;
            cand_d[split_idx] = (float)partial_dists[split_base[split_idx]];
            cand_i[split_idx] = partial_indices[split_base[split_idx]];
        }
        #pragma unroll
        for (int out_k = 0; out_k < TOP_K_MAX; out_k++) {
            float best_d = cand_d[0];
            int best_i = cand_i[0];
            int best_split = 0;
            #pragma unroll
            for (int split_idx = 1; split_idx < SPLIT_COUNT; split_idx++) {
                if (cand_d[split_idx] < best_d) {
                    best_d = cand_d[split_idx];
                    best_i = cand_i[split_idx];
                    best_split = split_idx;
                }
            }
            *((float*)(out_dists + out_base + out_k)) = best_d;
            *((int*)(out_indices + out_base + out_k)) = best_i;
            split_pos[best_split] = split_pos[best_split] + 1;
            if (out_k + 1 < TOP_K_MAX) {
                int next_pos = split_pos[best_split];
                int next_addr = split_base[best_split] + next_pos;
                cand_d[best_split] = (float)partial_dists[next_addr];
                cand_i[best_split] = partial_indices[next_addr];
            }
        }
    }
}

} // extern "C"

#undef NUM_MAIN_STAGES
#undef SPLIT_COUNT
#undef THREADS
#undef TOP_K_MAX

#define TMEM_NCOLS 64
#define TMEM_CROSS_OFFSET 0
#define NUM_MAIN_STAGES 1
#define SMEM_SMEM_QUERY_OFF 1024
#define SMEM_SMEM_QUERY_STAGE_BYTES 32768
#define SMEM_SMEM_QUERY_STRIDE 32768
#define SMEM_SMEM_DATABASE_OFF 33792
#define SMEM_SMEM_DATABASE_STAGE_BYTES 16384
#define SMEM_SMEM_DATABASE_STRIDE 16384
#define SMEM_SMEM_DATABASE_SQ_OFF 50176
#define SMEM_SMEM_DATABASE_SQ_STAGE_BYTES 256
#define SMEM_SMEM_DATABASE_SQ_STRIDE 256
#define SMEM_TOTAL 50432
#define THREADS 192
#define BLOCK_Q 128
#define BLOCK_M 64
#define FEAT_D 128
#define TOP_K_SMALL 5

extern "C" {

__global__ __launch_bounds__(192, 1) void
kernel_knn_build_evolve_7bfc_split_cg2_stage1_batch8_cond4_vmin_threshold_k5_mintree(float* __restrict__ query_sq, float* __restrict__ database_sq, float* __restrict__ partial_dists, int32_t* __restrict__ partial_indices, const void* tmap_query, const void* tmap_database, int B, int Q, int M, int K, int num_q_tile_pairs, int db_tiles_per_split, int split_count, int total_work)
{
    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;

    extern __shared__ __align__(1024) char smem_raw[];
    int smem;
    smem = (int)(unsigned long long)__cvta_generic_to_shared(smem_raw);
    const int smem_smem_query = smem + 1024;
    const int smem_smem_database = smem + 33792;
    const int smem_smem_database_sq = smem + 50176;

    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;
    const unsigned int clusters_x = gridDim.x / 2;
    const unsigned int cluster_id = ((blockIdx.z * gridDim.y + blockIdx.y) * clusters_x) + blockIdx.x / 2;
    const unsigned int num_clusters = clusters_x * gridDim.y * gridDim.z;

    int cta_rank;
    asm volatile("mov.b32 %0, %%cluster_ctarank;" : "=r"(cta_rank));

    // Mbarrier init (6 groups, 6 barriers)
    // Mbarriers at smem_raw[0..48)

    if (warp == 0) {
        uint32_t leader = elect_sync();
        // query_full: 1 barriers, init_count=2
        mbarrier_init_pred(smem + 0, 2, leader);
        // query_empty: 1 barriers, init_count=1
        mbarrier_init_pred(smem + 8, 1, leader);
        // database_full: 1 barriers, init_count=2
        mbarrier_init_pred(smem + 16, 2, leader);
        // database_empty: 1 barriers, init_count=1
        mbarrier_init_pred(smem + 24, 1, leader);
        // score_full: 1 barriers, init_count=1
        mbarrier_init_pred(smem + 32, 1, leader);
        // score_empty: 1 barriers, init_count=8
        mbarrier_init_pred(smem + 40, 8, leader);
        asm volatile("fence.mbarrier_init.release.cluster;");
    }

    __syncthreads();

    // TMEM alloc (64 columns, 64 used)
    volatile int* tmem_addr_storage = (volatile int*)(smem_raw + 48);
    if (warp == 1) {
        int _tmem_hold = smem + 48;
        asm volatile("tcgen05.alloc.cta_group::2.sync.aligned.shared::cta.b32 [%0], %1;" :: "r"(_tmem_hold), "r"(64) : "memory");
    }

    asm volatile("barrier.cluster.arrive.release.aligned;");
    asm volatile("barrier.cluster.wait.acquire.aligned;");
    asm volatile("tcgen05.fence::after_thread_sync;");

    const int warp_id = warp;
    const int lane_id = lane;
    __nv_bfloat16* smem_query = (__nv_bfloat16*)(smem_raw + 1024);
    #define smem_query_addr (smem + 1024)
    __nv_bfloat16* smem_database = (__nv_bfloat16*)(smem_raw + 33792);
    #define smem_database_addr (smem + 33792)
    float* smem_database_sq = (float*)(smem_raw + 50176);
    #define smem_database_sq_addr (smem + 50176)
    const int mbar_base = smem;
    #define query_full_addr (mbar_base + 0)
    #define query_empty_addr (mbar_base + 8)
    #define database_full_addr (mbar_base + 16)
    #define database_empty_addr (mbar_base + 24)
    #define score_full_addr (mbar_base + 32)
    #define score_empty_addr (mbar_base + 40)
    const int taddr = tmem_addr_storage[0];

    // ---- Role: load ----
    if (warp == 0) {
        { // load_main
            uint32_t _phase_query_empty_0 = 1;
            uint32_t _phase_database_empty_0 = 1;
            if (warp_id == 0) {
                if (elect_sync()) {
                    #pragma unroll 1
                    for (unsigned int work_idx = cluster_id; work_idx < total_work; work_idx += num_clusters) {
                        int split_idx = work_idx % split_count;
                        int query_work = work_idx / split_count;
                        int batch_idx = query_work / num_q_tile_pairs;
                        int q_tile_pair = query_work % num_q_tile_pairs;
                        int q_tile = q_tile_pair * 2 + cta_rank;
                        int off_q = q_tile * BLOCK_Q;
                        int global_q = batch_idx * Q + off_q;
                        int db_tile_start = split_idx * db_tiles_per_split;
                        mbarrier_wait(query_empty_addr, _phase_query_empty_0);
                        _phase_query_empty_0 ^= 1;
                        asm volatile(
                            "mbarrier.arrive.expect_tx.release.cta.shared::cluster.b64 _, [%0], %1;"
                            :: "r"((query_full_addr) & 0xFEFFFFFF), "r"((uint32_t)(32768)) : "memory");
                        asm volatile(
                            "cp.async.bulk.tensor.3d.shared::cluster.global.mbarrier::complete_tx::bytes.cta_group::2"
                            " [%0], [%1, {%2, %3, %4}], [%5];"
                            :: "r"(smem_query_addr), "l"(tmap_query), "r"(0), "r"(global_q), "r"(0),
                               "r"(((query_full_addr) & 0xFEFFFFFF)) : "memory");
                        #pragma unroll 1
                        for (int local_db_tile = 0; local_db_tile < db_tiles_per_split; local_db_tile++) {
                            int db_tile = db_tile_start + local_db_tile;
                            int off_m = db_tile * BLOCK_M;
                            int global_m = batch_idx * M + off_m;
                            mbarrier_wait(database_empty_addr, _phase_database_empty_0);
                            _phase_database_empty_0 ^= 1;
                            asm volatile(
                                "mbarrier.arrive.expect_tx.release.cta.shared::cluster.b64 _, [%0], %1;"
                                :: "r"((database_full_addr) & 0xFEFFFFFF), "r"((uint32_t)(16384)) : "memory");
                            asm volatile(
                                "cp.async.bulk.tensor.3d.shared::cluster.global.mbarrier::complete_tx::bytes.cta_group::2"
                                " [%0], [%1, {%2, %3, %4}], [%5];"
                                :: "r"(smem_database_addr), "l"(tmap_database), "r"(0), "r"(global_m), "r"(0),
                                   "r"(((database_full_addr) & 0xFEFFFFFF)) : "memory");
                        }
                    }
                }
            }
        }
    // ---- Role: mma ----
    } else if (warp == 1) {
        { // mma_main
            uint32_t _phase_query_full_0 = 0;
            uint32_t _phase_score_empty_0 = 1;
            uint32_t _phase_database_full_0 = 0;
            if (cta_rank == 0) {
                #pragma unroll 1
                for (unsigned int work_idx = cluster_id; work_idx < total_work; work_idx += num_clusters) {
                    mbarrier_wait(query_full_addr, _phase_query_full_0);
                    _phase_query_full_0 ^= 1;
                    #pragma unroll 1
                    for (int _local_db_tile = 0; _local_db_tile < db_tiles_per_split; _local_db_tile++) {
                        mbarrier_wait(score_empty_addr, _phase_score_empty_0);
                        _phase_score_empty_0 ^= 1;
                        mbarrier_wait(database_full_addr, _phase_database_full_0);
                        _phase_database_full_0 ^= 1;
                        asm volatile("tcgen05.fence::after_thread_sync;");
                        int _mma_ss_a_lo_0 = (smem_query_addr >> 4) & 0x3FFF;
                        int _mma_ss_b_lo_0 = (smem_database_addr >> 4) & 0x3FFF;
                        asm volatile(
                    "{\n\t"
                    ".reg .pred leader, p0, p1;\n\t"
                    ".reg .b32 adhi, bdhi, alo, blo, id, m0, m1, m2, m3, m4, m5, m6, m7;\n\t"
                    ".reg .b64 da, db;\n\t"
                    "elect.sync _|leader, 0xFFFFFFFF;\n\t"
                    "setp.ne.b32 p0, %3, 0;\n\t"
                    "setp.ne.b32 p1, 1, 0;\n\t"
                    "mov.b32 m0, 0; mov.b32 m1, 0; mov.b32 m2, 0; mov.b32 m3, 0;\n\tmov.b32 m4, 0; mov.b32 m5, 0; mov.b32 m6, 0; mov.b32 m7, 0;\n\t"
                    "mov.b32 adhi, 0x40004040;\n\t"
                    "mov.b32 bdhi, 0x40004040;\n\t"
                    "mov.b32 id, 270533776;\n\t"
                    "mov.b32 alo, %0;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p0;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "add.u32 alo, alo, 1018;\n\t"
                    "add.u32 blo, blo, 506;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "}\n"
                    :: "r"(_mma_ss_a_lo_0), "r"(_mma_ss_b_lo_0), "r"(taddr), "r"(0));
                        elect_commit_cg2_multicast(score_full_addr, (uint16_t)(3));
                        elect_commit_cg2_multicast(database_empty_addr, (uint16_t)(3));
                    }
                    elect_commit_cg2_multicast(query_empty_addr, (uint16_t)(3));
                }
            }
        }
    // ---- Role: compute ----
    } else if (warp >= 2 && warp <= 5) {
        const int warp_id_in_wg = warp % 4;
        const int my_row = warp_id_in_wg * 32 + lane;
        const int tmem_row_base_v = (warp_id_in_wg * 32) << 16;
        const int tmem_cross = taddr + TMEM_CROSS_OFFSET;
        { // compute_main
            uint32_t _phase_score_full_0 = 0;
            #pragma unroll 1
            for (unsigned int work_idx = cluster_id; work_idx < total_work; work_idx += num_clusters) {
                int split_idx = work_idx % split_count;
                int query_work = work_idx / split_count;
                int batch_idx = query_work / num_q_tile_pairs;
                int q_tile_pair = query_work % num_q_tile_pairs;
                int q_tile = q_tile_pair * 2 + cta_rank;
                int off_q = q_tile * BLOCK_Q;
                int q_idx = off_q + my_row;
                int valid_q = ((q_idx < Q) ? 1 : 0);
                float q_sq_val = 0.0f;
                if (valid_q != 0) {
                    q_sq_val = (float)query_sq[batch_idx * Q + q_idx];
                }
                float best_d[TOP_K_SMALL];
                int best_i[TOP_K_SMALL];
                float worst_d = 3.4e+38f;
                int worst_pos = 0;
                #pragma unroll
                for (int kk = 0; kk < TOP_K_SMALL; kk++) {
                    best_d[kk] = 3.4e+38f;
                    best_i[kk] = -1;
                }
                int db_tile_start = split_idx * db_tiles_per_split;
                #pragma unroll 1
                for (int local_db_tile = 0; local_db_tile < db_tiles_per_split; local_db_tile++) {
                    int db_tile = db_tile_start + local_db_tile;
                    int db_start = db_tile * BLOCK_M;
                    int db_sq_idx = db_start + my_row;
                    if (my_row < BLOCK_M) {
                        if (db_sq_idx < M) {
                            smem_database_sq[my_row] = (float)database_sq[batch_idx * M + db_sq_idx];
                        } else {
                            smem_database_sq[my_row] = 0.0f;
                        }
                    }
                    asm volatile("barrier.sync 8, 128;");
                    mbarrier_wait(score_full_addr, _phase_score_full_0);
                    _phase_score_full_0 ^= 1;
                    int cross_addr = taddr + (cta_rank * BLOCK_Q + tmem_row_base_v << 16);
                    float dots[64];
                    asm volatile(
                        "tcgen05.ld.sync.aligned.32x32b.x64.b32"
                        " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63}, [%64];"
                        : "=f"(dots[0]), "=f"(dots[1]), "=f"(dots[2]), "=f"(dots[3]), "=f"(dots[4]), "=f"(dots[5]), "=f"(dots[6]), "=f"(dots[7]), "=f"(dots[8]), "=f"(dots[9]), "=f"(dots[10]), "=f"(dots[11]), "=f"(dots[12]), "=f"(dots[13]), "=f"(dots[14]), "=f"(dots[15]), "=f"(dots[16]), "=f"(dots[17]), "=f"(dots[18]), "=f"(dots[19]), "=f"(dots[20]), "=f"(dots[21]), "=f"(dots[22]), "=f"(dots[23]), "=f"(dots[24]), "=f"(dots[25]), "=f"(dots[26]), "=f"(dots[27]), "=f"(dots[28]), "=f"(dots[29]), "=f"(dots[30]), "=f"(dots[31]), "=f"(dots[32]), "=f"(dots[33]), "=f"(dots[34]), "=f"(dots[35]), "=f"(dots[36]), "=f"(dots[37]), "=f"(dots[38]), "=f"(dots[39]), "=f"(dots[40]), "=f"(dots[41]), "=f"(dots[42]), "=f"(dots[43]), "=f"(dots[44]), "=f"(dots[45]), "=f"(dots[46]), "=f"(dots[47]), "=f"(dots[48]), "=f"(dots[49]), "=f"(dots[50]), "=f"(dots[51]), "=f"(dots[52]), "=f"(dots[53]), "=f"(dots[54]), "=f"(dots[55]), "=f"(dots[56]), "=f"(dots[57]), "=f"(dots[58]), "=f"(dots[59]), "=f"(dots[60]), "=f"(dots[61]), "=f"(dots[62]), "=f"(dots[63])
                        : "r"(cross_addr)
                        : "memory");
                    asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                    asm volatile("barrier.sync 8, 128;");
                    if (elect_sync()) {
                        asm volatile(
                            "mbarrier.arrive.release.cta.shared::cluster.b64 _, [%0];"
                            :: "r"((score_empty_addr) & 0xFEFFFFFF) : "memory");
                    }
                    if (valid_q != 0) {
                        #pragma unroll 1
                        for (int col_base = 0; col_base < 64; col_base += 8) {
                            float dist_vec0[4];
                            dist_vec0[0] = dots[col_base];
                            dist_vec0[1] = dots[col_base + 1];
                            dist_vec0[2] = dots[col_base + 2];
                            dist_vec0[3] = dots[col_base + 3];
                            const float2 _fma_b2_0 = {-2.0f, -2.0f};
                            const float2 _fma_c2_1 = {q_sq_val, q_sq_val};
                            #pragma unroll
                            for (int _lf = 0; _lf < 2; _lf++)
                                fma_f32x2_inplace(&reinterpret_cast<float2*>(dist_vec0)[_lf], _fma_b2_0, _fma_c2_1);
                            float db_sq_vec0[4];
                            db_sq_vec0[0] = smem_database_sq[col_base];
                            db_sq_vec0[1] = smem_database_sq[col_base + 1];
                            db_sq_vec0[2] = smem_database_sq[col_base + 2];
                            db_sq_vec0[3] = smem_database_sq[col_base + 3];
                            float _t0[4];
                            #pragma unroll
                            for (int _la = 0; _la < 2; _la++)
                                reinterpret_cast<float2*>(_t0)[_la] = add_f32x2(reinterpret_cast<float2*>(dist_vec0)[_la], reinterpret_cast<const float2*>(db_sq_vec0)[_la]);
                            float dist_vec1[4];
                            dist_vec1[0] = dots[col_base + 4];
                            dist_vec1[1] = dots[col_base + 5];
                            dist_vec1[2] = dots[col_base + 6];
                            dist_vec1[3] = dots[col_base + 7];
                            const float2 _fma_b2_2 = {-2.0f, -2.0f};
                            const float2 _fma_c2_3 = {q_sq_val, q_sq_val};
                            #pragma unroll
                            for (int _lf = 0; _lf < 2; _lf++)
                                fma_f32x2_inplace(&reinterpret_cast<float2*>(dist_vec1)[_lf], _fma_b2_2, _fma_c2_3);
                            float db_sq_vec1[4];
                            db_sq_vec1[0] = smem_database_sq[col_base + 4];
                            db_sq_vec1[1] = smem_database_sq[col_base + 5];
                            db_sq_vec1[2] = smem_database_sq[col_base + 6];
                            db_sq_vec1[3] = smem_database_sq[col_base + 7];
                            float _t1[4];
                            #pragma unroll
                            for (int _la = 0; _la < 2; _la++)
                                reinterpret_cast<float2*>(_t1)[_la] = add_f32x2(reinterpret_cast<float2*>(dist_vec1)[_la], reinterpret_cast<const float2*>(db_sq_vec1)[_la]);
                            float _t0_min = _t0[0];
                            #pragma unroll
                            for (int _lr = 1; _lr < 4; _lr++) {
                                _t0_min = fminf(_t0_min, _t0[_lr]);
                            }
                            float group_min0 = _t0_min;
                            float _t1_min = _t1[0];
                            #pragma unroll
                            for (int _lr = 1; _lr < 4; _lr++) {
                                _t1_min = fminf(_t1_min, _t1[_lr]);
                            }
                            float group_min1 = _t1_min;
                            if (group_min0 < worst_d) {
                                #pragma unroll
                                for (int vec_col = 0; vec_col < 4; vec_col++) {
                                    int db_idx = db_start + col_base + vec_col;
                                    if (db_idx < M) {
                                        float dist = _t0[vec_col];
                                        if (dist < worst_d) {
                                            best_d[worst_pos] = dist;
                                            best_i[worst_pos] = db_idx;
                                            int cmp01 = ((best_d[1] > best_d[0]) ? 1 : 0);
                                            float max01_d = ((cmp01 != 0) ? best_d[1] : best_d[0]);
                                            int max01_p = ((cmp01 != 0) ? 1 : 0);
                                            int cmp23 = ((best_d[3] > best_d[2]) ? 1 : 0);
                                            float max23_d = ((cmp23 != 0) ? best_d[3] : best_d[2]);
                                            int max23_p = ((cmp23 != 0) ? 3 : 2);
                                            int cmp0123 = ((max23_d > max01_d) ? 1 : 0);
                                            float max0123_d = ((cmp0123 != 0) ? max23_d : max01_d);
                                            int max0123_p = ((cmp0123 != 0) ? max23_p : max01_p);
                                            int cmp_all = ((best_d[4] > max0123_d) ? 1 : 0);
                                            worst_d = ((cmp_all != 0) ? best_d[4] : max0123_d);
                                            worst_pos = ((cmp_all != 0) ? 4 : max0123_p);
                                        }
                                    }
                                }
                            }
                            if (group_min1 < worst_d) {
                                #pragma unroll
                                for (int vec_col = 0; vec_col < 4; vec_col++) {
                                    int db_idx = db_start + col_base + 4 + vec_col;
                                    if (db_idx < M) {
                                        float dist = _t1[vec_col];
                                        if (dist < worst_d) {
                                            best_d[worst_pos] = dist;
                                            best_i[worst_pos] = db_idx;
                                            int cmp01 = ((best_d[1] > best_d[0]) ? 1 : 0);
                                            float max01_d = ((cmp01 != 0) ? best_d[1] : best_d[0]);
                                            int max01_p = ((cmp01 != 0) ? 1 : 0);
                                            int cmp23 = ((best_d[3] > best_d[2]) ? 1 : 0);
                                            float max23_d = ((cmp23 != 0) ? best_d[3] : best_d[2]);
                                            int max23_p = ((cmp23 != 0) ? 3 : 2);
                                            int cmp0123 = ((max23_d > max01_d) ? 1 : 0);
                                            float max0123_d = ((cmp0123 != 0) ? max23_d : max01_d);
                                            int max0123_p = ((cmp0123 != 0) ? max23_p : max01_p);
                                            int cmp_all = ((best_d[4] > max0123_d) ? 1 : 0);
                                            worst_d = ((cmp_all != 0) ? best_d[4] : max0123_d);
                                            worst_pos = ((cmp_all != 0) ? 4 : max0123_p);
                                        }
                                    }
                                }
                            }
                        }
                    }
                    asm volatile("barrier.sync 8, 128;");
                }
                if (valid_q != 0) {
                    int out_base = ((split_idx * B + batch_idx) * Q + q_idx) * K;
                    #pragma unroll
                    for (int out_k = 0; out_k < TOP_K_SMALL; out_k++) {
                        int cmp01 = ((best_d[1] < best_d[0]) ? 1 : 0);
                        float min01_d = ((cmp01 != 0) ? best_d[1] : best_d[0]);
                        int min01_i = ((cmp01 != 0) ? best_i[1] : best_i[0]);
                        int min01_p = ((cmp01 != 0) ? 1 : 0);
                        int cmp23 = ((best_d[3] < best_d[2]) ? 1 : 0);
                        float min23_d = ((cmp23 != 0) ? best_d[3] : best_d[2]);
                        int min23_i = ((cmp23 != 0) ? best_i[3] : best_i[2]);
                        int min23_p = ((cmp23 != 0) ? 3 : 2);
                        int cmp0123 = ((min23_d < min01_d) ? 1 : 0);
                        float min0123_d = ((cmp0123 != 0) ? min23_d : min01_d);
                        int min0123_i = ((cmp0123 != 0) ? min23_i : min01_i);
                        int min0123_p = ((cmp0123 != 0) ? min23_p : min01_p);
                        int cmp_all = ((best_d[4] < min0123_d) ? 1 : 0);
                        float selected_d = ((cmp_all != 0) ? best_d[4] : min0123_d);
                        int selected_i = ((cmp_all != 0) ? best_i[4] : min0123_i);
                        int selected_pos = ((cmp_all != 0) ? 4 : min0123_p);
                        if (out_k < K) {
                            *((float*)(partial_dists + out_base + out_k)) = selected_d;
                            *((int*)(partial_indices + out_base + out_k)) = selected_i;
                        }
                        best_d[selected_pos] = 3.4e+38f;
                        best_i[selected_pos] = -1;
                    }
                }
            }
        }
    }

    // Cleanup
    asm volatile("barrier.cluster.arrive.release.aligned;");
    asm volatile("barrier.cluster.wait.acquire.aligned;");

    if (warp == 1) {
        asm volatile("tcgen05.dealloc.cta_group::2.sync.aligned.b32 %0, %1;" :: "r"(tmem_addr_storage[0]), "r"(64));
        asm volatile("tcgen05.relinquish_alloc_permit.cta_group::2.sync.aligned;");
    }
}

} // extern "C"

#undef BLOCK_M
#undef BLOCK_Q
#undef FEAT_D
#undef NUM_MAIN_STAGES
#undef SMEM_SMEM_DATABASE_OFF
#undef SMEM_SMEM_DATABASE_SQ_OFF
#undef SMEM_SMEM_DATABASE_SQ_STAGE_BYTES
#undef SMEM_SMEM_DATABASE_SQ_STRIDE
#undef SMEM_SMEM_DATABASE_STAGE_BYTES
#undef SMEM_SMEM_DATABASE_STRIDE
#undef SMEM_SMEM_QUERY_OFF
#undef SMEM_SMEM_QUERY_STAGE_BYTES
#undef SMEM_SMEM_QUERY_STRIDE
#undef SMEM_TOTAL
#undef THREADS
#undef TMEM_CROSS_OFFSET
#undef TMEM_NCOLS
#undef TOP_K_SMALL
#undef database_empty_addr
#undef database_full_addr
#undef query_empty_addr
#undef query_full_addr
#undef score_empty_addr
#undef score_full_addr
#undef smem_database_addr
#undef smem_database_sq_addr
#undef smem_query_addr

#define NUM_MAIN_STAGES 1
#define THREADS 256
#define TOP_K_SMALL 5

extern "C" {

__global__ __launch_bounds__(256, 1) void
kernel_knn_build_evolve_7bfc_k5_merge_s4_tree_rowbase(float* __restrict__ partial_dists, int32_t* __restrict__ partial_indices, float* __restrict__ out_dists, int32_t* __restrict__ out_indices, int total_queries)
{
    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;


    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    const int warp_id = warp;
    const int lane_id = lane;

    // === Task calls (dependency order) ===
    int start_row = bid * 256 + tid;
    int stride = num_bids * 256;
    #pragma unroll 1
    for (int row = start_row; row < total_queries; row += stride) {
        int base_row = row * TOP_K_SMALL;
        int split_stride = total_queries * TOP_K_SMALL;
        int partial_base0 = base_row;
        int partial_base1 = base_row + split_stride;
        int partial_base2 = partial_base1 + split_stride;
        int partial_base3 = partial_base2 + split_stride;
        int pos0 = 0;
        int pos1 = 0;
        int pos2 = 0;
        int pos3 = 0;
        int out_base = row * TOP_K_SMALL;
        #pragma unroll
        for (int out_k = 0; out_k < TOP_K_SMALL; out_k++) {
            float cand_d0 = (float)partial_dists[partial_base0 + pos0];
            int cand_i0 = partial_indices[partial_base0 + pos0];
            float cand_d1 = (float)partial_dists[partial_base1 + pos1];
            int cand_i1 = partial_indices[partial_base1 + pos1];
            float cand_d2 = (float)partial_dists[partial_base2 + pos2];
            int cand_i2 = partial_indices[partial_base2 + pos2];
            float cand_d3 = (float)partial_dists[partial_base3 + pos3];
            int cand_i3 = partial_indices[partial_base3 + pos3];
            int cmp01 = ((cand_d1 < cand_d0) ? 1 : 0);
            float best01_d = ((cmp01 != 0) ? cand_d1 : cand_d0);
            int best01_i = ((cmp01 != 0) ? cand_i1 : cand_i0);
            int best01_split = ((cmp01 != 0) ? 1 : 0);
            int cmp23 = ((cand_d3 < cand_d2) ? 1 : 0);
            float best23_d = ((cmp23 != 0) ? cand_d3 : cand_d2);
            int best23_i = ((cmp23 != 0) ? cand_i3 : cand_i2);
            int best23_split = ((cmp23 != 0) ? 3 : 2);
            int cmp_all = ((best23_d < best01_d) ? 1 : 0);
            float best_d = ((cmp_all != 0) ? best23_d : best01_d);
            int best_i = ((cmp_all != 0) ? best23_i : best01_i);
            int best_split = ((cmp_all != 0) ? best23_split : best01_split);
            *((float*)(out_dists + out_base + out_k)) = best_d;
            *((int*)(out_indices + out_base + out_k)) = best_i;
            if (best_split == 0) {
                pos0 = pos0 + 1;
            } else if (best_split == 1) {
                pos1 = pos1 + 1;
            } else {
                if (best_split == 2) {
                    pos2 = pos2 + 1;
                } else {
                    pos3 = pos3 + 1;
                }
            }
        }
    }
}

} // extern "C"

#undef NUM_MAIN_STAGES
#undef THREADS
#undef TOP_K_SMALL

#define NUM_MAIN_STAGES 1
#define THREADS 64
#define TOP_K_MAX 10
#define SPLIT_COUNT 4

extern "C" {

__global__ __launch_bounds__(64, 1) void
kernel_knn_build_evolve_7bfc_k10_merge_s4_rowbase_cache(float* __restrict__ partial_dists, int32_t* __restrict__ partial_indices, float* __restrict__ out_dists, int32_t* __restrict__ out_indices, int total_queries)
{
    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;


    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    const int warp_id = warp;
    const int lane_id = lane;

    // === Task calls (dependency order) ===
    int start_row = bid * 64 + tid;
    int stride = num_bids * 64;
    #pragma unroll 1
    for (int row = start_row; row < total_queries; row += stride) {
        int base_row = row * TOP_K_MAX;
        int split_stride = total_queries * TOP_K_MAX;
        int out_base = base_row;
        int split_base0 = base_row;
        int split_base1 = base_row + split_stride;
        int split_base2 = split_base1 + split_stride;
        int split_base3 = split_base2 + split_stride;
        int pos0 = 0;
        int pos1 = 0;
        int pos2 = 0;
        int pos3 = 0;
        float cand_d0 = (float)partial_dists[split_base0];
        int cand_i0 = partial_indices[split_base0];
        float cand_d1 = (float)partial_dists[split_base1];
        int cand_i1 = partial_indices[split_base1];
        float cand_d2 = (float)partial_dists[split_base2];
        int cand_i2 = partial_indices[split_base2];
        float cand_d3 = (float)partial_dists[split_base3];
        int cand_i3 = partial_indices[split_base3];
        #pragma unroll
        for (int out_k = 0; out_k < TOP_K_MAX; out_k++) {
            int cand01_cmp = ((cand_d1 < cand_d0) ? 1 : 0);
            float best01_d = ((cand01_cmp != 0) ? cand_d1 : cand_d0);
            int best01_i = ((cand01_cmp != 0) ? cand_i1 : cand_i0);
            int best01_split = ((cand01_cmp != 0) ? 1 : 0);
            int cand23_cmp = ((cand_d3 < cand_d2) ? 1 : 0);
            float best23_d = ((cand23_cmp != 0) ? cand_d3 : cand_d2);
            int best23_i = ((cand23_cmp != 0) ? cand_i3 : cand_i2);
            int best23_split = ((cand23_cmp != 0) ? 3 : 2);
            int best_cmp = ((best23_d < best01_d) ? 1 : 0);
            float best_d = ((best_cmp != 0) ? best23_d : best01_d);
            int best_i = ((best_cmp != 0) ? best23_i : best01_i);
            int best_split = ((best_cmp != 0) ? best23_split : best01_split);
            *((float*)(out_dists + out_base + out_k)) = best_d;
            *((int*)(out_indices + out_base + out_k)) = best_i;
            if (out_k + 1 < TOP_K_MAX) {
                if (best_split == 0) {
                    pos0 = pos0 + 1;
                    int next_addr0 = split_base0 + pos0;
                    cand_d0 = (float)partial_dists[next_addr0];
                    cand_i0 = partial_indices[next_addr0];
                } else if (best_split == 1) {
                    pos1 = pos1 + 1;
                    int next_addr1 = split_base1 + pos1;
                    cand_d1 = (float)partial_dists[next_addr1];
                    cand_i1 = partial_indices[next_addr1];
                } else {
                    if (best_split == 2) {
                        pos2 = pos2 + 1;
                        int next_addr2 = split_base2 + pos2;
                        cand_d2 = (float)partial_dists[next_addr2];
                        cand_i2 = partial_indices[next_addr2];
                    } else {
                        pos3 = pos3 + 1;
                        int next_addr3 = split_base3 + pos3;
                        cand_d3 = (float)partial_dists[next_addr3];
                        cand_i3 = partial_indices[next_addr3];
                    }
                }
            }
        }
    }
}

} // extern "C"

#undef NUM_MAIN_STAGES
#undef SPLIT_COUNT
#undef THREADS
#undef TOP_K_MAX

#define NUM_MAIN_STAGES 1
#define THREADS 256
#define TOP_K_MAX 10
#define SPLIT_COUNT 7

extern "C" {

__global__ __launch_bounds__(256, 1) void
kernel_knn_build_evolve_7bfc_k10_merge_s7_rowbase_cache(float* __restrict__ partial_dists, int32_t* __restrict__ partial_indices, float* __restrict__ out_dists, int32_t* __restrict__ out_indices, int total_queries)
{
    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;


    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    const int warp_id = warp;
    const int lane_id = lane;

    // === Task calls (dependency order) ===
    int start_row = bid * 256 + tid;
    int stride = num_bids * 256;
    #pragma unroll 1
    for (int row = start_row; row < total_queries; row += stride) {
        int base_row = row * TOP_K_MAX;
        int split_stride = total_queries * TOP_K_MAX;
        int out_base = base_row;
        int split_pos[SPLIT_COUNT];
        int split_base[SPLIT_COUNT];
        float cand_d[SPLIT_COUNT];
        int cand_i[SPLIT_COUNT];
        #pragma unroll
        for (int split_idx = 0; split_idx < SPLIT_COUNT; split_idx++) {
            split_pos[split_idx] = 0;
            split_base[split_idx] = base_row + split_idx * split_stride;
            cand_d[split_idx] = (float)partial_dists[split_base[split_idx]];
            cand_i[split_idx] = partial_indices[split_base[split_idx]];
        }
        #pragma unroll
        for (int out_k = 0; out_k < TOP_K_MAX; out_k++) {
            float best_d = cand_d[0];
            int best_i = cand_i[0];
            int best_split = 0;
            #pragma unroll
            for (int split_idx = 1; split_idx < SPLIT_COUNT; split_idx++) {
                if (cand_d[split_idx] < best_d) {
                    best_d = cand_d[split_idx];
                    best_i = cand_i[split_idx];
                    best_split = split_idx;
                }
            }
            *((float*)(out_dists + out_base + out_k)) = best_d;
            *((int*)(out_indices + out_base + out_k)) = best_i;
            split_pos[best_split] = split_pos[best_split] + 1;
            if (out_k + 1 < TOP_K_MAX) {
                int next_pos = split_pos[best_split];
                int next_addr = split_base[best_split] + next_pos;
                cand_d[best_split] = (float)partial_dists[next_addr];
                cand_i[best_split] = partial_indices[next_addr];
            }
        }
    }
}

} // extern "C"

#undef NUM_MAIN_STAGES
#undef SPLIT_COUNT
#undef THREADS
#undef TOP_K_MAX

#define TMEM_NCOLS 64
#define TMEM_CROSS_OFFSET 0
#define NUM_MAIN_STAGES 1
#define SMEM_SMEM_QUERY_OFF 1024
#define SMEM_SMEM_QUERY_STAGE_BYTES 32768
#define SMEM_SMEM_QUERY_STRIDE 32768
#define SMEM_SMEM_DATABASE_OFF 33792
#define SMEM_SMEM_DATABASE_STAGE_BYTES 16384
#define SMEM_SMEM_DATABASE_STRIDE 16384
#define SMEM_SMEM_DATABASE_SQ_OFF 50176
#define SMEM_SMEM_DATABASE_SQ_STAGE_BYTES 256
#define SMEM_SMEM_DATABASE_SQ_STRIDE 256
#define SMEM_TOTAL 50432
#define THREADS 192
#define BLOCK_Q 128
#define BLOCK_M 64
#define FEAT_D 128
#define TOP_K_MAX 10

extern "C" {

__global__ __launch_bounds__(192, 1) void
kernel_knn_build_evolve_7bfc_split_cg2_stage1_batch8_cond4_vmin_maxtree(float* __restrict__ query_sq, float* __restrict__ database_sq, float* __restrict__ partial_dists, int32_t* __restrict__ partial_indices, const void* tmap_query, const void* tmap_database, int B, int Q, int M, int K, int num_q_tile_pairs, int db_tiles_per_split, int split_count, int total_work)
{
    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;

    extern __shared__ __align__(1024) char smem_raw[];
    int smem;
    smem = (int)(unsigned long long)__cvta_generic_to_shared(smem_raw);
    const int smem_smem_query = smem + 1024;
    const int smem_smem_database = smem + 33792;
    const int smem_smem_database_sq = smem + 50176;

    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;
    const unsigned int clusters_x = gridDim.x / 2;
    const unsigned int cluster_id = ((blockIdx.z * gridDim.y + blockIdx.y) * clusters_x) + blockIdx.x / 2;
    const unsigned int num_clusters = clusters_x * gridDim.y * gridDim.z;

    int cta_rank;
    asm volatile("mov.b32 %0, %%cluster_ctarank;" : "=r"(cta_rank));

    // Mbarrier init (6 groups, 6 barriers)
    // Mbarriers at smem_raw[0..48)

    if (warp == 0) {
        uint32_t leader = elect_sync();
        // query_full: 1 barriers, init_count=2
        mbarrier_init_pred(smem + 0, 2, leader);
        // query_empty: 1 barriers, init_count=1
        mbarrier_init_pred(smem + 8, 1, leader);
        // database_full: 1 barriers, init_count=2
        mbarrier_init_pred(smem + 16, 2, leader);
        // database_empty: 1 barriers, init_count=1
        mbarrier_init_pred(smem + 24, 1, leader);
        // score_full: 1 barriers, init_count=1
        mbarrier_init_pred(smem + 32, 1, leader);
        // score_empty: 1 barriers, init_count=8
        mbarrier_init_pred(smem + 40, 8, leader);
        asm volatile("fence.mbarrier_init.release.cluster;");
    }

    __syncthreads();

    // TMEM alloc (64 columns, 64 used)
    volatile int* tmem_addr_storage = (volatile int*)(smem_raw + 48);
    if (warp == 1) {
        int _tmem_hold = smem + 48;
        asm volatile("tcgen05.alloc.cta_group::2.sync.aligned.shared::cta.b32 [%0], %1;" :: "r"(_tmem_hold), "r"(64) : "memory");
    }

    asm volatile("barrier.cluster.arrive.release.aligned;");
    asm volatile("barrier.cluster.wait.acquire.aligned;");
    asm volatile("tcgen05.fence::after_thread_sync;");

    const int warp_id = warp;
    const int lane_id = lane;
    __nv_bfloat16* smem_query = (__nv_bfloat16*)(smem_raw + 1024);
    #define smem_query_addr (smem + 1024)
    __nv_bfloat16* smem_database = (__nv_bfloat16*)(smem_raw + 33792);
    #define smem_database_addr (smem + 33792)
    float* smem_database_sq = (float*)(smem_raw + 50176);
    #define smem_database_sq_addr (smem + 50176)
    const int mbar_base = smem;
    #define query_full_addr (mbar_base + 0)
    #define query_empty_addr (mbar_base + 8)
    #define database_full_addr (mbar_base + 16)
    #define database_empty_addr (mbar_base + 24)
    #define score_full_addr (mbar_base + 32)
    #define score_empty_addr (mbar_base + 40)
    const int taddr = tmem_addr_storage[0];

    // ---- Role: load ----
    if (warp == 0) {
        { // load_main
            uint32_t _phase_query_empty_0 = 1;
            uint32_t _phase_database_empty_0 = 1;
            if (warp_id == 0) {
                if (elect_sync()) {
                    #pragma unroll 1
                    for (unsigned int work_idx = cluster_id; work_idx < total_work; work_idx += num_clusters) {
                        int split_idx = work_idx % split_count;
                        int query_work = work_idx / split_count;
                        int batch_idx = query_work / num_q_tile_pairs;
                        int q_tile_pair = query_work % num_q_tile_pairs;
                        int q_tile = q_tile_pair * 2 + cta_rank;
                        int off_q = q_tile * BLOCK_Q;
                        int global_q = batch_idx * Q + off_q;
                        int db_tile_start = split_idx * db_tiles_per_split;
                        mbarrier_wait(query_empty_addr, _phase_query_empty_0);
                        _phase_query_empty_0 ^= 1;
                        asm volatile(
                            "mbarrier.arrive.expect_tx.release.cta.shared::cluster.b64 _, [%0], %1;"
                            :: "r"((query_full_addr) & 0xFEFFFFFF), "r"((uint32_t)(32768)) : "memory");
                        asm volatile(
                            "cp.async.bulk.tensor.3d.shared::cluster.global.mbarrier::complete_tx::bytes.cta_group::2"
                            " [%0], [%1, {%2, %3, %4}], [%5];"
                            :: "r"(smem_query_addr), "l"(tmap_query), "r"(0), "r"(global_q), "r"(0),
                               "r"(((query_full_addr) & 0xFEFFFFFF)) : "memory");
                        #pragma unroll 1
                        for (int local_db_tile = 0; local_db_tile < db_tiles_per_split; local_db_tile++) {
                            int db_tile = db_tile_start + local_db_tile;
                            int off_m = db_tile * BLOCK_M;
                            int global_m = batch_idx * M + off_m;
                            mbarrier_wait(database_empty_addr, _phase_database_empty_0);
                            _phase_database_empty_0 ^= 1;
                            asm volatile(
                                "mbarrier.arrive.expect_tx.release.cta.shared::cluster.b64 _, [%0], %1;"
                                :: "r"((database_full_addr) & 0xFEFFFFFF), "r"((uint32_t)(16384)) : "memory");
                            asm volatile(
                                "cp.async.bulk.tensor.3d.shared::cluster.global.mbarrier::complete_tx::bytes.cta_group::2"
                                " [%0], [%1, {%2, %3, %4}], [%5];"
                                :: "r"(smem_database_addr), "l"(tmap_database), "r"(0), "r"(global_m), "r"(0),
                                   "r"(((database_full_addr) & 0xFEFFFFFF)) : "memory");
                        }
                    }
                }
            }
        }
    // ---- Role: mma ----
    } else if (warp == 1) {
        { // mma_main
            uint32_t _phase_query_full_0 = 0;
            uint32_t _phase_score_empty_0 = 1;
            uint32_t _phase_database_full_0 = 0;
            if (cta_rank == 0) {
                #pragma unroll 1
                for (unsigned int work_idx = cluster_id; work_idx < total_work; work_idx += num_clusters) {
                    mbarrier_wait(query_full_addr, _phase_query_full_0);
                    _phase_query_full_0 ^= 1;
                    #pragma unroll 1
                    for (int _local_db_tile = 0; _local_db_tile < db_tiles_per_split; _local_db_tile++) {
                        mbarrier_wait(score_empty_addr, _phase_score_empty_0);
                        _phase_score_empty_0 ^= 1;
                        mbarrier_wait(database_full_addr, _phase_database_full_0);
                        _phase_database_full_0 ^= 1;
                        asm volatile("tcgen05.fence::after_thread_sync;");
                        int _mma_ss_a_lo_0 = (smem_query_addr >> 4) & 0x3FFF;
                        int _mma_ss_b_lo_0 = (smem_database_addr >> 4) & 0x3FFF;
                        asm volatile(
                    "{\n\t"
                    ".reg .pred leader, p0, p1;\n\t"
                    ".reg .b32 adhi, bdhi, alo, blo, id, m0, m1, m2, m3, m4, m5, m6, m7;\n\t"
                    ".reg .b64 da, db;\n\t"
                    "elect.sync _|leader, 0xFFFFFFFF;\n\t"
                    "setp.ne.b32 p0, %3, 0;\n\t"
                    "setp.ne.b32 p1, 1, 0;\n\t"
                    "mov.b32 m0, 0; mov.b32 m1, 0; mov.b32 m2, 0; mov.b32 m3, 0;\n\tmov.b32 m4, 0; mov.b32 m5, 0; mov.b32 m6, 0; mov.b32 m7, 0;\n\t"
                    "mov.b32 adhi, 0x40004040;\n\t"
                    "mov.b32 bdhi, 0x40004040;\n\t"
                    "mov.b32 id, 270533776;\n\t"
                    "mov.b32 alo, %0;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p0;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "add.u32 alo, alo, 1018;\n\t"
                    "add.u32 blo, blo, 506;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "}\n"
                    :: "r"(_mma_ss_a_lo_0), "r"(_mma_ss_b_lo_0), "r"(taddr), "r"(0));
                        elect_commit_cg2_multicast(score_full_addr, (uint16_t)(3));
                        elect_commit_cg2_multicast(database_empty_addr, (uint16_t)(3));
                    }
                    elect_commit_cg2_multicast(query_empty_addr, (uint16_t)(3));
                }
            }
        }
    // ---- Role: compute ----
    } else if (warp >= 2 && warp <= 5) {
        const int warp_id_in_wg = warp % 4;
        const int my_row = warp_id_in_wg * 32 + lane;
        const int tmem_row_base_v = (warp_id_in_wg * 32) << 16;
        const int tmem_cross = taddr + TMEM_CROSS_OFFSET;
        { // compute_main
            uint32_t _phase_score_full_0 = 0;
            #pragma unroll 1
            for (unsigned int work_idx = cluster_id; work_idx < total_work; work_idx += num_clusters) {
                int split_idx = work_idx % split_count;
                int query_work = work_idx / split_count;
                int batch_idx = query_work / num_q_tile_pairs;
                int q_tile_pair = query_work % num_q_tile_pairs;
                int q_tile = q_tile_pair * 2 + cta_rank;
                int off_q = q_tile * BLOCK_Q;
                int q_idx = off_q + my_row;
                int valid_q = ((q_idx < Q) ? 1 : 0);
                float q_sq_val = 0.0f;
                if (valid_q != 0) {
                    q_sq_val = (float)query_sq[batch_idx * Q + q_idx];
                }
                float best_d[TOP_K_MAX];
                int best_i[TOP_K_MAX];
                float worst_d = 3.4e+38f;
                int worst_pos = 0;
                #pragma unroll
                for (int kk = 0; kk < TOP_K_MAX; kk++) {
                    best_d[kk] = 3.4e+38f;
                    best_i[kk] = -1;
                }
                int db_tile_start = split_idx * db_tiles_per_split;
                #pragma unroll 1
                for (int local_db_tile = 0; local_db_tile < db_tiles_per_split; local_db_tile++) {
                    int db_tile = db_tile_start + local_db_tile;
                    int db_start = db_tile * BLOCK_M;
                    int db_sq_idx = db_start + my_row;
                    if (my_row < BLOCK_M) {
                        if (db_sq_idx < M) {
                            smem_database_sq[my_row] = (float)database_sq[batch_idx * M + db_sq_idx];
                        } else {
                            smem_database_sq[my_row] = 0.0f;
                        }
                    }
                    asm volatile("barrier.sync 8, 128;");
                    mbarrier_wait(score_full_addr, _phase_score_full_0);
                    _phase_score_full_0 ^= 1;
                    int cross_addr = taddr + (cta_rank * BLOCK_Q + tmem_row_base_v << 16);
                    float dots[64];
                    asm volatile(
                        "tcgen05.ld.sync.aligned.32x32b.x64.b32"
                        " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63}, [%64];"
                        : "=f"(dots[0]), "=f"(dots[1]), "=f"(dots[2]), "=f"(dots[3]), "=f"(dots[4]), "=f"(dots[5]), "=f"(dots[6]), "=f"(dots[7]), "=f"(dots[8]), "=f"(dots[9]), "=f"(dots[10]), "=f"(dots[11]), "=f"(dots[12]), "=f"(dots[13]), "=f"(dots[14]), "=f"(dots[15]), "=f"(dots[16]), "=f"(dots[17]), "=f"(dots[18]), "=f"(dots[19]), "=f"(dots[20]), "=f"(dots[21]), "=f"(dots[22]), "=f"(dots[23]), "=f"(dots[24]), "=f"(dots[25]), "=f"(dots[26]), "=f"(dots[27]), "=f"(dots[28]), "=f"(dots[29]), "=f"(dots[30]), "=f"(dots[31]), "=f"(dots[32]), "=f"(dots[33]), "=f"(dots[34]), "=f"(dots[35]), "=f"(dots[36]), "=f"(dots[37]), "=f"(dots[38]), "=f"(dots[39]), "=f"(dots[40]), "=f"(dots[41]), "=f"(dots[42]), "=f"(dots[43]), "=f"(dots[44]), "=f"(dots[45]), "=f"(dots[46]), "=f"(dots[47]), "=f"(dots[48]), "=f"(dots[49]), "=f"(dots[50]), "=f"(dots[51]), "=f"(dots[52]), "=f"(dots[53]), "=f"(dots[54]), "=f"(dots[55]), "=f"(dots[56]), "=f"(dots[57]), "=f"(dots[58]), "=f"(dots[59]), "=f"(dots[60]), "=f"(dots[61]), "=f"(dots[62]), "=f"(dots[63])
                        : "r"(cross_addr)
                        : "memory");
                    asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                    asm volatile("barrier.sync 8, 128;");
                    if (elect_sync()) {
                        asm volatile(
                            "mbarrier.arrive.release.cta.shared::cluster.b64 _, [%0];"
                            :: "r"((score_empty_addr) & 0xFEFFFFFF) : "memory");
                    }
                    if (valid_q != 0) {
                        #pragma unroll 1
                        for (int col_base = 0; col_base < 64; col_base += 8) {
                            float dist_vec0[4];
                            dist_vec0[0] = dots[col_base];
                            dist_vec0[1] = dots[col_base + 1];
                            dist_vec0[2] = dots[col_base + 2];
                            dist_vec0[3] = dots[col_base + 3];
                            const float2 _fma_b2_0 = {-2.0f, -2.0f};
                            const float2 _fma_c2_1 = {q_sq_val, q_sq_val};
                            #pragma unroll
                            for (int _lf = 0; _lf < 2; _lf++)
                                fma_f32x2_inplace(&reinterpret_cast<float2*>(dist_vec0)[_lf], _fma_b2_0, _fma_c2_1);
                            float db_sq_vec0[4];
                            db_sq_vec0[0] = smem_database_sq[col_base];
                            db_sq_vec0[1] = smem_database_sq[col_base + 1];
                            db_sq_vec0[2] = smem_database_sq[col_base + 2];
                            db_sq_vec0[3] = smem_database_sq[col_base + 3];
                            float _t0[4];
                            #pragma unroll
                            for (int _la = 0; _la < 2; _la++)
                                reinterpret_cast<float2*>(_t0)[_la] = add_f32x2(reinterpret_cast<float2*>(dist_vec0)[_la], reinterpret_cast<const float2*>(db_sq_vec0)[_la]);
                            float dist_vec1[4];
                            dist_vec1[0] = dots[col_base + 4];
                            dist_vec1[1] = dots[col_base + 5];
                            dist_vec1[2] = dots[col_base + 6];
                            dist_vec1[3] = dots[col_base + 7];
                            const float2 _fma_b2_2 = {-2.0f, -2.0f};
                            const float2 _fma_c2_3 = {q_sq_val, q_sq_val};
                            #pragma unroll
                            for (int _lf = 0; _lf < 2; _lf++)
                                fma_f32x2_inplace(&reinterpret_cast<float2*>(dist_vec1)[_lf], _fma_b2_2, _fma_c2_3);
                            float db_sq_vec1[4];
                            db_sq_vec1[0] = smem_database_sq[col_base + 4];
                            db_sq_vec1[1] = smem_database_sq[col_base + 5];
                            db_sq_vec1[2] = smem_database_sq[col_base + 6];
                            db_sq_vec1[3] = smem_database_sq[col_base + 7];
                            float _t1[4];
                            #pragma unroll
                            for (int _la = 0; _la < 2; _la++)
                                reinterpret_cast<float2*>(_t1)[_la] = add_f32x2(reinterpret_cast<float2*>(dist_vec1)[_la], reinterpret_cast<const float2*>(db_sq_vec1)[_la]);
                            float _t0_min = _t0[0];
                            #pragma unroll
                            for (int _lr = 1; _lr < 4; _lr++) {
                                _t0_min = fminf(_t0_min, _t0[_lr]);
                            }
                            float group_min0 = _t0_min;
                            float _t1_min = _t1[0];
                            #pragma unroll
                            for (int _lr = 1; _lr < 4; _lr++) {
                                _t1_min = fminf(_t1_min, _t1[_lr]);
                            }
                            float group_min1 = _t1_min;
                            if (group_min0 < worst_d) {
                                #pragma unroll
                                for (int vec_col = 0; vec_col < 4; vec_col++) {
                                    int db_idx = db_start + col_base + vec_col;
                                    if (db_idx < M) {
                                        float dist = _t0[vec_col];
                                        if (dist < worst_d) {
                                            best_d[worst_pos] = dist;
                                            best_i[worst_pos] = db_idx;
                                            int cmp01 = ((best_d[1] > best_d[0]) ? 1 : 0);
                                            float max01_d = ((cmp01 != 0) ? best_d[1] : best_d[0]);
                                            int max01_p = ((cmp01 != 0) ? 1 : 0);
                                            int cmp23 = ((best_d[3] > best_d[2]) ? 1 : 0);
                                            float max23_d = ((cmp23 != 0) ? best_d[3] : best_d[2]);
                                            int max23_p = ((cmp23 != 0) ? 3 : 2);
                                            int cmp45 = ((best_d[5] > best_d[4]) ? 1 : 0);
                                            float max45_d = ((cmp45 != 0) ? best_d[5] : best_d[4]);
                                            int max45_p = ((cmp45 != 0) ? 5 : 4);
                                            int cmp67 = ((best_d[7] > best_d[6]) ? 1 : 0);
                                            float max67_d = ((cmp67 != 0) ? best_d[7] : best_d[6]);
                                            int max67_p = ((cmp67 != 0) ? 7 : 6);
                                            int cmp89 = ((best_d[9] > best_d[8]) ? 1 : 0);
                                            float max89_d = ((cmp89 != 0) ? best_d[9] : best_d[8]);
                                            int max89_p = ((cmp89 != 0) ? 9 : 8);
                                            int cmp0123 = ((max23_d > max01_d) ? 1 : 0);
                                            float max0123_d = ((cmp0123 != 0) ? max23_d : max01_d);
                                            int max0123_p = ((cmp0123 != 0) ? max23_p : max01_p);
                                            int cmp4567 = ((max67_d > max45_d) ? 1 : 0);
                                            float max4567_d = ((cmp4567 != 0) ? max67_d : max45_d);
                                            int max4567_p = ((cmp4567 != 0) ? max67_p : max45_p);
                                            int cmp0_7 = ((max4567_d > max0123_d) ? 1 : 0);
                                            float max0_7_d = ((cmp0_7 != 0) ? max4567_d : max0123_d);
                                            int max0_7_p = ((cmp0_7 != 0) ? max4567_p : max0123_p);
                                            int cmp_all = ((max89_d > max0_7_d) ? 1 : 0);
                                            worst_d = ((cmp_all != 0) ? max89_d : max0_7_d);
                                            worst_pos = ((cmp_all != 0) ? max89_p : max0_7_p);
                                        }
                                    }
                                }
                            }
                            if (group_min1 < worst_d) {
                                #pragma unroll
                                for (int vec_col = 0; vec_col < 4; vec_col++) {
                                    int db_idx = db_start + col_base + 4 + vec_col;
                                    if (db_idx < M) {
                                        float dist = _t1[vec_col];
                                        if (dist < worst_d) {
                                            best_d[worst_pos] = dist;
                                            best_i[worst_pos] = db_idx;
                                            int cmp01 = ((best_d[1] > best_d[0]) ? 1 : 0);
                                            float max01_d = ((cmp01 != 0) ? best_d[1] : best_d[0]);
                                            int max01_p = ((cmp01 != 0) ? 1 : 0);
                                            int cmp23 = ((best_d[3] > best_d[2]) ? 1 : 0);
                                            float max23_d = ((cmp23 != 0) ? best_d[3] : best_d[2]);
                                            int max23_p = ((cmp23 != 0) ? 3 : 2);
                                            int cmp45 = ((best_d[5] > best_d[4]) ? 1 : 0);
                                            float max45_d = ((cmp45 != 0) ? best_d[5] : best_d[4]);
                                            int max45_p = ((cmp45 != 0) ? 5 : 4);
                                            int cmp67 = ((best_d[7] > best_d[6]) ? 1 : 0);
                                            float max67_d = ((cmp67 != 0) ? best_d[7] : best_d[6]);
                                            int max67_p = ((cmp67 != 0) ? 7 : 6);
                                            int cmp89 = ((best_d[9] > best_d[8]) ? 1 : 0);
                                            float max89_d = ((cmp89 != 0) ? best_d[9] : best_d[8]);
                                            int max89_p = ((cmp89 != 0) ? 9 : 8);
                                            int cmp0123 = ((max23_d > max01_d) ? 1 : 0);
                                            float max0123_d = ((cmp0123 != 0) ? max23_d : max01_d);
                                            int max0123_p = ((cmp0123 != 0) ? max23_p : max01_p);
                                            int cmp4567 = ((max67_d > max45_d) ? 1 : 0);
                                            float max4567_d = ((cmp4567 != 0) ? max67_d : max45_d);
                                            int max4567_p = ((cmp4567 != 0) ? max67_p : max45_p);
                                            int cmp0_7 = ((max4567_d > max0123_d) ? 1 : 0);
                                            float max0_7_d = ((cmp0_7 != 0) ? max4567_d : max0123_d);
                                            int max0_7_p = ((cmp0_7 != 0) ? max4567_p : max0123_p);
                                            int cmp_all = ((max89_d > max0_7_d) ? 1 : 0);
                                            worst_d = ((cmp_all != 0) ? max89_d : max0_7_d);
                                            worst_pos = ((cmp_all != 0) ? max89_p : max0_7_p);
                                        }
                                    }
                                }
                            }
                        }
                    }
                    asm volatile("barrier.sync 8, 128;");
                }
                if (valid_q != 0) {
                    int out_base = ((split_idx * B + batch_idx) * Q + q_idx) * K;
                    #pragma unroll
                    for (int out_k = 0; out_k < TOP_K_MAX; out_k++) {
                        float selected_d = best_d[0];
                        int selected_i = best_i[0];
                        int selected_pos = 0;
                        #pragma unroll
                        for (int scan = 1; scan < TOP_K_MAX; scan++) {
                            if (best_d[scan] < selected_d) {
                                selected_d = best_d[scan];
                                selected_i = best_i[scan];
                                selected_pos = scan;
                            }
                        }
                        if (out_k < K) {
                            *((float*)(partial_dists + out_base + out_k)) = selected_d;
                            *((int*)(partial_indices + out_base + out_k)) = selected_i;
                        }
                        best_d[selected_pos] = 3.4e+38f;
                        best_i[selected_pos] = -1;
                    }
                }
            }
        }
    }

    // Cleanup
    asm volatile("barrier.cluster.arrive.release.aligned;");
    asm volatile("barrier.cluster.wait.acquire.aligned;");

    if (warp == 1) {
        asm volatile("tcgen05.dealloc.cta_group::2.sync.aligned.b32 %0, %1;" :: "r"(tmem_addr_storage[0]), "r"(64));
        asm volatile("tcgen05.relinquish_alloc_permit.cta_group::2.sync.aligned;");
    }
}

} // extern "C"

#undef BLOCK_M
#undef BLOCK_Q
#undef FEAT_D
#undef NUM_MAIN_STAGES
#undef SMEM_SMEM_DATABASE_OFF
#undef SMEM_SMEM_DATABASE_SQ_OFF
#undef SMEM_SMEM_DATABASE_SQ_STAGE_BYTES
#undef SMEM_SMEM_DATABASE_SQ_STRIDE
#undef SMEM_SMEM_DATABASE_STAGE_BYTES
#undef SMEM_SMEM_DATABASE_STRIDE
#undef SMEM_SMEM_QUERY_OFF
#undef SMEM_SMEM_QUERY_STAGE_BYTES
#undef SMEM_SMEM_QUERY_STRIDE
#undef SMEM_TOTAL
#undef THREADS
#undef TMEM_CROSS_OFFSET
#undef TMEM_NCOLS
#undef TOP_K_MAX
#undef database_empty_addr
#undef database_full_addr
#undef query_empty_addr
#undef query_full_addr
#undef score_empty_addr
#undef score_full_addr
#undef smem_database_addr
#undef smem_database_sq_addr
#undef smem_query_addr

#define TMEM_NCOLS 64
#define TMEM_CROSS_OFFSET 0
#define NUM_MAIN_STAGES 1
#define SMEM_SMEM_QUERY_OFF 1024
#define SMEM_SMEM_QUERY_STAGE_BYTES 16384
#define SMEM_SMEM_QUERY_STRIDE 16384
#define SMEM_SMEM_DATABASE_OFF 17408
#define SMEM_SMEM_DATABASE_STAGE_BYTES 8192
#define SMEM_SMEM_DATABASE_STRIDE 8192
#define SMEM_SMEM_DATABASE_SQ_OFF 25600
#define SMEM_SMEM_DATABASE_SQ_STAGE_BYTES 256
#define SMEM_SMEM_DATABASE_SQ_STRIDE 256
#define SMEM_TOTAL 25856
#define THREADS 192
#define BLOCK_Q 128
#define BLOCK_M 64
#define TOP_K_MAX 10

extern "C" {

__global__ __launch_bounds__(192, 1) void
kernel_knn_build_dim_midk_73a9_d64_split_stage1(float* __restrict__ query_sq, float* __restrict__ database_sq, float* __restrict__ partial_dists, int32_t* __restrict__ partial_indices, const void* tmap_query, const void* tmap_database, int B, int Q, int M, int K, int num_q_tiles, int db_tiles_per_split, int split_count, int total_work)
{
    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;

    extern __shared__ __align__(1024) char smem_raw[];
    int smem;
    smem = (int)(unsigned long long)__cvta_generic_to_shared(smem_raw);
    const int smem_smem_query = smem + 1024;
    const int smem_smem_database = smem + 17408;
    const int smem_smem_database_sq = smem + 25600;

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
        // score_empty: 1 barriers, init_count=128
        mbarrier_init_pred(smem + 40, 128, leader);
        asm volatile("fence.mbarrier_init.release.cluster;");
    }

    __syncthreads();

    // TMEM alloc (64 columns, 64 used)
    volatile int* tmem_addr_storage = (volatile int*)(smem_raw + 48);
    if (warp == 1) {
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
    float* smem_database_sq = (float*)(smem_raw + 25600);
    #define smem_database_sq_addr (smem + 25600)
    const int mbar_base = smem;
    #define query_full_addr (mbar_base + 0)
    #define query_empty_addr (mbar_base + 8)
    #define database_full_addr (mbar_base + 16)
    #define database_empty_addr (mbar_base + 24)
    #define score_full_addr (mbar_base + 32)
    #define score_empty_addr (mbar_base + 40)
    const int taddr = tmem_addr_storage[0];

    // ---- Role: load ----
    if (warp == 0) {
        { // load_main
            uint32_t _phase_query_empty_0 = 1;
            uint32_t _phase_database_empty_0 = 1;
            if (warp_id == 0) {
                if (elect_sync()) {
                    #pragma unroll 1
                    for (unsigned int work_idx = bid; work_idx < total_work; work_idx += num_bids) {
                        int split_idx = work_idx % split_count;
                        int query_work = work_idx / split_count;
                        int batch_idx = query_work / num_q_tiles;
                        int q_tile = query_work % num_q_tiles;
                        int off_q = q_tile * BLOCK_Q;
                        int global_q = batch_idx * Q + off_q;
                        int db_tile_start = split_idx * db_tiles_per_split;
                        mbarrier_wait(query_empty_addr, _phase_query_empty_0);
                        _phase_query_empty_0 ^= 1;
                        mbarrier_arrive_expect_tx(query_full_addr, 16384);
                        tma_3d_gmem2smem(smem_query_addr, tmap_query, 0, global_q, 0, query_full_addr);
                        #pragma unroll 1
                        for (int local_db_tile = 0; local_db_tile < db_tiles_per_split; local_db_tile++) {
                            int db_tile = db_tile_start + local_db_tile;
                            int off_m = db_tile * BLOCK_M;
                            int global_m = batch_idx * M + off_m;
                            mbarrier_wait(database_empty_addr, _phase_database_empty_0);
                            _phase_database_empty_0 ^= 1;
                            mbarrier_arrive_expect_tx(database_full_addr, 8192);
                            tma_3d_gmem2smem(smem_database_addr, tmap_database, 0, global_m, 0, database_full_addr);
                        }
                    }
                }
            }
        }
    // ---- Role: mma ----
    } else if (warp == 1) {
        { // mma_main
            uint32_t _phase_query_full_0 = 0;
            uint32_t _phase_score_empty_0 = 1;
            uint32_t _phase_database_full_0 = 0;
            #pragma unroll 1
            for (unsigned int work_idx = bid; work_idx < total_work; work_idx += num_bids) {
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
                    "mov.b32 id, 135267472;\n\t"
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
                    "}\n"
                    :: "r"(_mma_ss_a_lo_0), "r"(_mma_ss_b_lo_0), "r"(taddr), "r"(0));
                    elect_commit(score_full_addr);
                    elect_commit(database_empty_addr);
                }
                elect_commit(query_empty_addr);
            }
        }
    // ---- Role: compute ----
    } else if (warp >= 2 && warp <= 5) {
        const int warp_id_in_wg = warp % 4;
        const int my_row = warp_id_in_wg * 32 + lane;
        const int tmem_row_base_v = (warp_id_in_wg * 32) << 16;
        const int tmem_cross = taddr + TMEM_CROSS_OFFSET;
        { // compute_main
            uint32_t _phase_score_full_0 = 0;
            #pragma unroll 1
            for (unsigned int work_idx = bid; work_idx < total_work; work_idx += num_bids) {
                int split_idx = work_idx % split_count;
                int query_work = work_idx / split_count;
                int batch_idx = query_work / num_q_tiles;
                int q_tile = query_work % num_q_tiles;
                int off_q = q_tile * BLOCK_Q;
                int q_idx = off_q + my_row;
                int valid_q = ((q_idx < Q) ? 1 : 0);
                float q_sq_val = 0.0f;
                if (valid_q != 0) {
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
                    int db_start = db_tile * BLOCK_M;
                    int db_sq_idx = db_start + my_row;
                    if (my_row < BLOCK_M) {
                        if (db_sq_idx < M) {
                            smem_database_sq[my_row] = (float)database_sq[batch_idx * M + db_sq_idx];
                        } else {
                            smem_database_sq[my_row] = 0.0f;
                        }
                    }
                    asm volatile("barrier.sync 8, 128;");
                    mbarrier_wait(score_full_addr, _phase_score_full_0);
                    _phase_score_full_0 ^= 1;
                    int cross_addr = taddr + tmem_row_base_v;
                    float dots[64];
                    tmem_ld_x32(&dots[0], cross_addr);
                    tmem_ld_x32(&dots[32], cross_addr + 32);
                    asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                    asm volatile("barrier.sync 8, 128;");
                    mbarrier_arrive(score_empty_addr);
                    if (valid_q != 0) {
                        #pragma unroll 2
                        for (int col_base = 0; col_base < 64; col_base += 4) {
                            float dist_vec[4];
                            dist_vec[0] = dots[col_base];
                            dist_vec[1] = dots[col_base + 1];
                            dist_vec[2] = dots[col_base + 2];
                            dist_vec[3] = dots[col_base + 3];
                            const float2 _fma_b2_0 = {-2.0f, -2.0f};
                            const float2 _fma_c2_1 = {q_sq_val, q_sq_val};
                            #pragma unroll
                            for (int _lf = 0; _lf < 2; _lf++)
                                fma_f32x2_inplace(&reinterpret_cast<float2*>(dist_vec)[_lf], _fma_b2_0, _fma_c2_1);
                            float db_sq_vec[4];
                            db_sq_vec[0] = smem_database_sq[col_base];
                            db_sq_vec[1] = smem_database_sq[col_base + 1];
                            db_sq_vec[2] = smem_database_sq[col_base + 2];
                            db_sq_vec[3] = smem_database_sq[col_base + 3];
                            float _t0[4];
                            #pragma unroll
                            for (int _la = 0; _la < 2; _la++)
                                reinterpret_cast<float2*>(_t0)[_la] = add_f32x2(reinterpret_cast<float2*>(dist_vec)[_la], reinterpret_cast<const float2*>(db_sq_vec)[_la]);
                            float group_min = _t0[0];
                            if (_t0[1] < group_min) {
                                group_min = _t0[1];
                            }
                            if (_t0[2] < group_min) {
                                group_min = _t0[2];
                            }
                            if (_t0[3] < group_min) {
                                group_min = _t0[3];
                            }
                            if (group_min < best_d[TOP_K_MAX - 1]) {
                                #pragma unroll
                                for (int vec_col = 0; vec_col < 4; vec_col++) {
                                    int db_idx = db_start + col_base + vec_col;
                                    if (db_idx < M) {
                                        float dist = _t0[vec_col];
                                        dist = max_noftz(dist, 0.0f);
                                        if (dist < best_d[TOP_K_MAX - 1]) {
                                            best_d[TOP_K_MAX - 1] = dist;
                                            best_i[TOP_K_MAX - 1] = db_idx;
                                            #pragma unroll
                                            for (int pos = TOP_K_MAX - 1; pos >= 1; pos--) {
                                                if (best_d[pos] < best_d[pos - 1]) {
                                                    float tmp_d = best_d[pos - 1];
                                                    int tmp_i = best_i[pos - 1];
                                                    best_d[pos - 1] = best_d[pos];
                                                    best_i[pos - 1] = best_i[pos];
                                                    best_d[pos] = tmp_d;
                                                    best_i[pos] = tmp_i;
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                    asm volatile("barrier.sync 8, 128;");
                }
                if (valid_q != 0) {
                    int out_base = ((split_idx * B + batch_idx) * Q + q_idx) * K;
                    #pragma unroll
                    for (int out_k = 0; out_k < TOP_K_MAX; out_k++) {
                        if (out_k < K) {
                            *((float*)(partial_dists + out_base + out_k)) = best_d[out_k];
                            *((int*)(partial_indices + out_base + out_k)) = best_i[out_k];
                        }
                    }
                }
            }
        }
    }

    // Cleanup
    __syncthreads(); // barrier before TMEM dealloc

    if (warp == 1) {
        asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;" :: "r"(tmem_addr_storage[0]), "r"(64));
        asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;");
    }
}

} // extern "C"

#undef BLOCK_M
#undef BLOCK_Q
#undef NUM_MAIN_STAGES
#undef SMEM_SMEM_DATABASE_OFF
#undef SMEM_SMEM_DATABASE_SQ_OFF
#undef SMEM_SMEM_DATABASE_SQ_STAGE_BYTES
#undef SMEM_SMEM_DATABASE_SQ_STRIDE
#undef SMEM_SMEM_DATABASE_STAGE_BYTES
#undef SMEM_SMEM_DATABASE_STRIDE
#undef SMEM_SMEM_QUERY_OFF
#undef SMEM_SMEM_QUERY_STAGE_BYTES
#undef SMEM_SMEM_QUERY_STRIDE
#undef SMEM_TOTAL
#undef THREADS
#undef TMEM_CROSS_OFFSET
#undef TMEM_NCOLS
#undef TOP_K_MAX
#undef database_empty_addr
#undef database_full_addr
#undef query_empty_addr
#undef query_full_addr
#undef score_empty_addr
#undef score_full_addr
#undef smem_database_addr
#undef smem_database_sq_addr
#undef smem_query_addr

#define NUM_MAIN_STAGES 1
#define THREADS 256
#define TOP_K_MAX 10

extern "C" {

__global__ __launch_bounds__(256, 1) void
kernel_knn_build_evolve_7bfc_split_merge(float* __restrict__ partial_dists, int32_t* __restrict__ partial_indices, float* __restrict__ out_dists, int32_t* __restrict__ out_indices, int B, int Q, int K, int split_count, int total_queries)
{
    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;


    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    const int warp_id = warp;
    const int lane_id = lane;

    // === Task calls (dependency order) ===
    int start_row = bid * 256 + tid;
    int stride = num_bids * 256;
    #pragma unroll 1
    for (int row = start_row; row < total_queries; row += stride) {
        int batch_idx = row / Q;
        int q_idx = row - batch_idx * Q;
        float best_d[TOP_K_MAX];
        int best_i[TOP_K_MAX];
        #pragma unroll
        for (int kk = 0; kk < TOP_K_MAX; kk++) {
            best_d[kk] = 3.4e+38f;
            best_i[kk] = -1;
        }
        #pragma unroll 1
        for (int split_idx = 0; split_idx < split_count; split_idx++) {
            int partial_base = ((split_idx * B + batch_idx) * Q + q_idx) * K;
            #pragma unroll
            for (int cand_k = 0; cand_k < TOP_K_MAX; cand_k++) {
                if (cand_k < K) {
                    float cand_d = (float)partial_dists[partial_base + cand_k];
                    int cand_i = partial_indices[partial_base + cand_k];
                    if (cand_d < best_d[TOP_K_MAX - 1]) {
                        best_d[TOP_K_MAX - 1] = cand_d;
                        best_i[TOP_K_MAX - 1] = cand_i;
                        #pragma unroll
                        for (int pos = TOP_K_MAX - 1; pos >= 1; pos--) {
                            if (best_d[pos] < best_d[pos - 1]) {
                                float tmp_d = best_d[pos - 1];
                                int tmp_i = best_i[pos - 1];
                                best_d[pos - 1] = best_d[pos];
                                best_i[pos - 1] = best_i[pos];
                                best_d[pos] = tmp_d;
                                best_i[pos] = tmp_i;
                            }
                        }
                    }
                }
            }
        }
        int out_base = row * K;
        #pragma unroll
        for (int out_k = 0; out_k < TOP_K_MAX; out_k++) {
            if (out_k < K) {
                *((float*)(out_dists + out_base + out_k)) = best_d[out_k];
                *((int*)(out_indices + out_base + out_k)) = best_i[out_k];
            }
        }
    }
}

} // extern "C"

#undef NUM_MAIN_STAGES
#undef THREADS
#undef TOP_K_MAX

#define NUM_MAIN_STAGES 1
#define THREADS 32
#define TOP_K_MAX 10
#define SPLIT_COUNT 8

extern "C" {

__global__ __launch_bounds__(32, 1) void
kernel_knn_build_d64_build_aa88_k10_merge_s8_rowbase_cache(float* __restrict__ partial_dists, int32_t* __restrict__ partial_indices, float* __restrict__ out_dists, int32_t* __restrict__ out_indices, int total_queries)
{
    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;


    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    const int warp_id = warp;
    const int lane_id = lane;

    // === Task calls (dependency order) ===
    int start_row = bid * 32 + tid;
    int stride = num_bids * 32;
    #pragma unroll 1
    for (int row = start_row; row < total_queries; row += stride) {
        int base_row = row * TOP_K_MAX;
        int split_stride = total_queries * TOP_K_MAX;
        int split_pos[SPLIT_COUNT];
        int split_base[SPLIT_COUNT];
        float cand_d[SPLIT_COUNT];
        int cand_i[SPLIT_COUNT];
        #pragma unroll
        for (int split_idx = 0; split_idx < SPLIT_COUNT; split_idx++) {
            split_pos[split_idx] = 0;
            split_base[split_idx] = base_row + split_idx * split_stride;
            cand_d[split_idx] = (float)partial_dists[split_base[split_idx]];
            cand_i[split_idx] = partial_indices[split_base[split_idx]];
        }
        #pragma unroll
        for (int out_k = 0; out_k < TOP_K_MAX; out_k++) {
            float best_d = cand_d[0];
            int best_i = cand_i[0];
            int best_split = 0;
            #pragma unroll
            for (int split_idx = 1; split_idx < SPLIT_COUNT; split_idx++) {
                if (cand_d[split_idx] < best_d) {
                    best_d = cand_d[split_idx];
                    best_i = cand_i[split_idx];
                    best_split = split_idx;
                }
            }
            *((float*)(out_dists + base_row + out_k)) = best_d;
            *((int*)(out_indices + base_row + out_k)) = best_i;
            split_pos[best_split] = split_pos[best_split] + 1;
            if (out_k + 1 < TOP_K_MAX) {
                int next_pos = split_pos[best_split];
                int next_addr = split_base[best_split] + next_pos;
                cand_d[best_split] = (float)partial_dists[next_addr];
                cand_i[best_split] = partial_indices[next_addr];
            }
        }
    }
}

} // extern "C"

#undef NUM_MAIN_STAGES
#undef SPLIT_COUNT
#undef THREADS
#undef TOP_K_MAX

#define NUM_MAIN_STAGES 1
#define THREADS 32
#define TOP_K_MAX 10
#define SPLIT_COUNT 4

extern "C" {

__global__ __launch_bounds__(32, 1) void
kernel_knn_build_d64_build_aa88_k10_merge_s8_rowbase_cache_s4(float* __restrict__ partial_dists, int32_t* __restrict__ partial_indices, float* __restrict__ out_dists, int32_t* __restrict__ out_indices, int total_queries)
{
    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;


    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    const int warp_id = warp;
    const int lane_id = lane;

    // === Task calls (dependency order) ===
    int start_row = bid * 32 + tid;
    int stride = num_bids * 32;
    #pragma unroll 1
    for (int row = start_row; row < total_queries; row += stride) {
        int base_row = row * TOP_K_MAX;
        int split_stride = total_queries * TOP_K_MAX;
        int split_pos[SPLIT_COUNT];
        int split_base[SPLIT_COUNT];
        float cand_d[SPLIT_COUNT];
        int cand_i[SPLIT_COUNT];
        #pragma unroll
        for (int split_idx = 0; split_idx < SPLIT_COUNT; split_idx++) {
            split_pos[split_idx] = 0;
            split_base[split_idx] = base_row + split_idx * split_stride;
            cand_d[split_idx] = (float)partial_dists[split_base[split_idx]];
            cand_i[split_idx] = partial_indices[split_base[split_idx]];
        }
        #pragma unroll
        for (int out_k = 0; out_k < TOP_K_MAX; out_k++) {
            float best_d = cand_d[0];
            int best_i = cand_i[0];
            int best_split = 0;
            #pragma unroll
            for (int split_idx = 1; split_idx < SPLIT_COUNT; split_idx++) {
                if (cand_d[split_idx] < best_d) {
                    best_d = cand_d[split_idx];
                    best_i = cand_i[split_idx];
                    best_split = split_idx;
                }
            }
            *((float*)(out_dists + base_row + out_k)) = best_d;
            *((int*)(out_indices + base_row + out_k)) = best_i;
            split_pos[best_split] = split_pos[best_split] + 1;
            if (out_k + 1 < TOP_K_MAX) {
                int next_pos = split_pos[best_split];
                int next_addr = split_base[best_split] + next_pos;
                cand_d[best_split] = (float)partial_dists[next_addr];
                cand_i[best_split] = partial_indices[next_addr];
            }
        }
    }
}

} // extern "C"

#undef NUM_MAIN_STAGES
#undef SPLIT_COUNT
#undef THREADS
#undef TOP_K_MAX

#define TMEM_NCOLS 64
#define TMEM_CROSS_OFFSET 0
#define NUM_MAIN_STAGES 1
#define SMEM_SMEM_QUERY_OFF 1024
#define SMEM_SMEM_QUERY_STAGE_BYTES 32768
#define SMEM_SMEM_QUERY_STRIDE 32768
#define SMEM_SMEM_DATABASE_OFF 33792
#define SMEM_SMEM_DATABASE_STAGE_BYTES 16384
#define SMEM_SMEM_DATABASE_STRIDE 16384
#define SMEM_SMEM_DATABASE_SQ_OFF 50176
#define SMEM_SMEM_DATABASE_SQ_STAGE_BYTES 256
#define SMEM_SMEM_DATABASE_SQ_STRIDE 256
#define SMEM_TOTAL 50432
#define THREADS 192
#define BLOCK_Q 128
#define BLOCK_M 64
#define FEAT_D 128
#define TOP_K_MAX 24

extern "C" {

__global__ __launch_bounds__(192, 1) void
kernel_knn_build_evolve_7bfc_split_cg2_u2_stage1_bad5k24s8(float* __restrict__ query_sq, float* __restrict__ database_sq, float* __restrict__ partial_dists, int32_t* __restrict__ partial_indices, const void* tmap_query, const void* tmap_database, int B, int Q, int M, int K, int num_q_tile_pairs, int db_tiles_per_split, int split_count, int total_work)
{
    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;

    extern __shared__ __align__(1024) char smem_raw[];
    int smem;
    smem = (int)(unsigned long long)__cvta_generic_to_shared(smem_raw);
    const int smem_smem_query = smem + 1024;
    const int smem_smem_database = smem + 33792;
    const int smem_smem_database_sq = smem + 50176;

    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;
    const unsigned int clusters_x = gridDim.x / 2;
    const unsigned int cluster_id = ((blockIdx.z * gridDim.y + blockIdx.y) * clusters_x) + blockIdx.x / 2;
    const unsigned int num_clusters = clusters_x * gridDim.y * gridDim.z;

    int cta_rank;
    asm volatile("mov.b32 %0, %%cluster_ctarank;" : "=r"(cta_rank));

    // Mbarrier init (6 groups, 6 barriers)
    // Mbarriers at smem_raw[0..48)

    if (warp == 0) {
        uint32_t leader = elect_sync();
        // query_full: 1 barriers, init_count=2
        mbarrier_init_pred(smem + 0, 2, leader);
        // query_empty: 1 barriers, init_count=1
        mbarrier_init_pred(smem + 8, 1, leader);
        // database_full: 1 barriers, init_count=2
        mbarrier_init_pred(smem + 16, 2, leader);
        // database_empty: 1 barriers, init_count=1
        mbarrier_init_pred(smem + 24, 1, leader);
        // score_full: 1 barriers, init_count=1
        mbarrier_init_pred(smem + 32, 1, leader);
        // score_empty: 1 barriers, init_count=8
        mbarrier_init_pred(smem + 40, 8, leader);
        asm volatile("fence.mbarrier_init.release.cluster;");
    }

    __syncthreads();

    // TMEM alloc (64 columns, 64 used)
    volatile int* tmem_addr_storage = (volatile int*)(smem_raw + 48);
    if (warp == 1) {
        int _tmem_hold = smem + 48;
        asm volatile("tcgen05.alloc.cta_group::2.sync.aligned.shared::cta.b32 [%0], %1;" :: "r"(_tmem_hold), "r"(64) : "memory");
    }

    asm volatile("barrier.cluster.arrive.release.aligned;");
    asm volatile("barrier.cluster.wait.acquire.aligned;");
    asm volatile("tcgen05.fence::after_thread_sync;");

    const int warp_id = warp;
    const int lane_id = lane;
    __nv_bfloat16* smem_query = (__nv_bfloat16*)(smem_raw + 1024);
    #define smem_query_addr (smem + 1024)
    __nv_bfloat16* smem_database = (__nv_bfloat16*)(smem_raw + 33792);
    #define smem_database_addr (smem + 33792)
    float* smem_database_sq = (float*)(smem_raw + 50176);
    #define smem_database_sq_addr (smem + 50176)
    const int mbar_base = smem;
    #define query_full_addr (mbar_base + 0)
    #define query_empty_addr (mbar_base + 8)
    #define database_full_addr (mbar_base + 16)
    #define database_empty_addr (mbar_base + 24)
    #define score_full_addr (mbar_base + 32)
    #define score_empty_addr (mbar_base + 40)
    const int taddr = tmem_addr_storage[0];

    // ---- Role: load ----
    if (warp == 0) {
        { // load_main
            uint32_t _phase_query_empty_0 = 1;
            uint32_t _phase_database_empty_0 = 1;
            if (warp_id == 0) {
                if (elect_sync()) {
                    #pragma unroll 1
                    for (unsigned int work_idx = cluster_id; work_idx < total_work; work_idx += num_clusters) {
                        int split_idx = work_idx % split_count;
                        int query_work = work_idx / split_count;
                        int batch_idx = query_work / num_q_tile_pairs;
                        int q_tile_pair = query_work % num_q_tile_pairs;
                        int q_tile = q_tile_pair * 2 + cta_rank;
                        int off_q = q_tile * BLOCK_Q;
                        int global_q = batch_idx * Q + off_q;
                        int db_tile_start = split_idx * db_tiles_per_split;
                        mbarrier_wait(query_empty_addr, _phase_query_empty_0);
                        _phase_query_empty_0 ^= 1;
                        asm volatile(
                            "mbarrier.arrive.expect_tx.release.cta.shared::cluster.b64 _, [%0], %1;"
                            :: "r"((query_full_addr) & 0xFEFFFFFF), "r"((uint32_t)(32768)) : "memory");
                        asm volatile(
                            "cp.async.bulk.tensor.3d.shared::cluster.global.mbarrier::complete_tx::bytes.cta_group::2"
                            " [%0], [%1, {%2, %3, %4}], [%5];"
                            :: "r"(smem_query_addr), "l"(tmap_query), "r"(0), "r"(global_q), "r"(0),
                               "r"(((query_full_addr) & 0xFEFFFFFF)) : "memory");
                        #pragma unroll 1
                        for (int local_db_tile = 0; local_db_tile < db_tiles_per_split; local_db_tile++) {
                            int db_tile = db_tile_start + local_db_tile;
                            int off_m = db_tile * BLOCK_M;
                            int global_m = batch_idx * M + off_m;
                            mbarrier_wait(database_empty_addr, _phase_database_empty_0);
                            _phase_database_empty_0 ^= 1;
                            asm volatile(
                                "mbarrier.arrive.expect_tx.release.cta.shared::cluster.b64 _, [%0], %1;"
                                :: "r"((database_full_addr) & 0xFEFFFFFF), "r"((uint32_t)(16384)) : "memory");
                            asm volatile(
                                "cp.async.bulk.tensor.3d.shared::cluster.global.mbarrier::complete_tx::bytes.cta_group::2"
                                " [%0], [%1, {%2, %3, %4}], [%5];"
                                :: "r"(smem_database_addr), "l"(tmap_database), "r"(0), "r"(global_m), "r"(0),
                                   "r"(((database_full_addr) & 0xFEFFFFFF)) : "memory");
                        }
                    }
                }
            }
        }
    // ---- Role: mma ----
    } else if (warp == 1) {
        { // mma_main
            uint32_t _phase_query_full_0 = 0;
            uint32_t _phase_score_empty_0 = 1;
            uint32_t _phase_database_full_0 = 0;
            if (cta_rank == 0) {
                #pragma unroll 1
                for (unsigned int work_idx = cluster_id; work_idx < total_work; work_idx += num_clusters) {
                    mbarrier_wait(query_full_addr, _phase_query_full_0);
                    _phase_query_full_0 ^= 1;
                    #pragma unroll 1
                    for (int _local_db_tile = 0; _local_db_tile < db_tiles_per_split; _local_db_tile++) {
                        mbarrier_wait(score_empty_addr, _phase_score_empty_0);
                        _phase_score_empty_0 ^= 1;
                        mbarrier_wait(database_full_addr, _phase_database_full_0);
                        _phase_database_full_0 ^= 1;
                        asm volatile("tcgen05.fence::after_thread_sync;");
                        int _mma_ss_a_lo_0 = (smem_query_addr >> 4) & 0x3FFF;
                        int _mma_ss_b_lo_0 = (smem_database_addr >> 4) & 0x3FFF;
                        asm volatile(
                    "{\n\t"
                    ".reg .pred leader, p0, p1;\n\t"
                    ".reg .b32 adhi, bdhi, alo, blo, id, m0, m1, m2, m3, m4, m5, m6, m7;\n\t"
                    ".reg .b64 da, db;\n\t"
                    "elect.sync _|leader, 0xFFFFFFFF;\n\t"
                    "setp.ne.b32 p0, %3, 0;\n\t"
                    "setp.ne.b32 p1, 1, 0;\n\t"
                    "mov.b32 m0, 0; mov.b32 m1, 0; mov.b32 m2, 0; mov.b32 m3, 0;\n\tmov.b32 m4, 0; mov.b32 m5, 0; mov.b32 m6, 0; mov.b32 m7, 0;\n\t"
                    "mov.b32 adhi, 0x40004040;\n\t"
                    "mov.b32 bdhi, 0x40004040;\n\t"
                    "mov.b32 id, 270533776;\n\t"
                    "mov.b32 alo, %0;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p0;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "add.u32 alo, alo, 1018;\n\t"
                    "add.u32 blo, blo, 506;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "}\n"
                    :: "r"(_mma_ss_a_lo_0), "r"(_mma_ss_b_lo_0), "r"(taddr), "r"(0));
                        elect_commit_cg2_multicast(score_full_addr, (uint16_t)(3));
                        elect_commit_cg2_multicast(database_empty_addr, (uint16_t)(3));
                    }
                    elect_commit_cg2_multicast(query_empty_addr, (uint16_t)(3));
                }
            }
        }
    // ---- Role: compute ----
    } else if (warp >= 2 && warp <= 5) {
        const int warp_id_in_wg = warp % 4;
        const int my_row = warp_id_in_wg * 32 + lane;
        const int tmem_row_base_v = (warp_id_in_wg * 32) << 16;
        const int tmem_cross = taddr + TMEM_CROSS_OFFSET;
        { // compute_main
            uint32_t _phase_score_full_0 = 0;
            #pragma unroll 1
            for (unsigned int work_idx = cluster_id; work_idx < total_work; work_idx += num_clusters) {
                int split_idx = work_idx % split_count;
                int query_work = work_idx / split_count;
                int batch_idx = query_work / num_q_tile_pairs;
                int q_tile_pair = query_work % num_q_tile_pairs;
                int q_tile = q_tile_pair * 2 + cta_rank;
                int off_q = q_tile * BLOCK_Q;
                int q_idx = off_q + my_row;
                int valid_q = ((q_idx < Q) ? 1 : 0);
                float q_sq_val = 0.0f;
                if (valid_q != 0) {
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
                    int db_start = db_tile * BLOCK_M;
                    int db_sq_idx = db_start + my_row;
                    if (my_row < BLOCK_M) {
                        if (db_sq_idx < M) {
                            smem_database_sq[my_row] = (float)database_sq[batch_idx * M + db_sq_idx];
                        } else {
                            smem_database_sq[my_row] = 0.0f;
                        }
                    }
                    asm volatile("barrier.sync 8, 128;");
                    mbarrier_wait(score_full_addr, _phase_score_full_0);
                    _phase_score_full_0 ^= 1;
                    int cross_addr = taddr + (cta_rank * BLOCK_Q + tmem_row_base_v << 16);
                    float dots[64];
                    asm volatile(
                        "tcgen05.ld.sync.aligned.32x32b.x64.b32"
                        " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63}, [%64];"
                        : "=f"(dots[0]), "=f"(dots[1]), "=f"(dots[2]), "=f"(dots[3]), "=f"(dots[4]), "=f"(dots[5]), "=f"(dots[6]), "=f"(dots[7]), "=f"(dots[8]), "=f"(dots[9]), "=f"(dots[10]), "=f"(dots[11]), "=f"(dots[12]), "=f"(dots[13]), "=f"(dots[14]), "=f"(dots[15]), "=f"(dots[16]), "=f"(dots[17]), "=f"(dots[18]), "=f"(dots[19]), "=f"(dots[20]), "=f"(dots[21]), "=f"(dots[22]), "=f"(dots[23]), "=f"(dots[24]), "=f"(dots[25]), "=f"(dots[26]), "=f"(dots[27]), "=f"(dots[28]), "=f"(dots[29]), "=f"(dots[30]), "=f"(dots[31]), "=f"(dots[32]), "=f"(dots[33]), "=f"(dots[34]), "=f"(dots[35]), "=f"(dots[36]), "=f"(dots[37]), "=f"(dots[38]), "=f"(dots[39]), "=f"(dots[40]), "=f"(dots[41]), "=f"(dots[42]), "=f"(dots[43]), "=f"(dots[44]), "=f"(dots[45]), "=f"(dots[46]), "=f"(dots[47]), "=f"(dots[48]), "=f"(dots[49]), "=f"(dots[50]), "=f"(dots[51]), "=f"(dots[52]), "=f"(dots[53]), "=f"(dots[54]), "=f"(dots[55]), "=f"(dots[56]), "=f"(dots[57]), "=f"(dots[58]), "=f"(dots[59]), "=f"(dots[60]), "=f"(dots[61]), "=f"(dots[62]), "=f"(dots[63])
                        : "r"(cross_addr)
                        : "memory");
                    asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                    asm volatile("barrier.sync 8, 128;");
                    if (elect_sync()) {
                        asm volatile(
                            "mbarrier.arrive.release.cta.shared::cluster.b64 _, [%0];"
                            :: "r"((score_empty_addr) & 0xFEFFFFFF) : "memory");
                    }
                    if (valid_q != 0) {
                        #pragma unroll 2
                        for (int col_base = 0; col_base < 64; col_base += 4) {
                            float dist_vec[4];
                            dist_vec[0] = dots[col_base];
                            dist_vec[1] = dots[col_base + 1];
                            dist_vec[2] = dots[col_base + 2];
                            dist_vec[3] = dots[col_base + 3];
                            const float2 _fma_b2_0 = {-2.0f, -2.0f};
                            const float2 _fma_c2_1 = {q_sq_val, q_sq_val};
                            #pragma unroll
                            for (int _lf = 0; _lf < 2; _lf++)
                                fma_f32x2_inplace(&reinterpret_cast<float2*>(dist_vec)[_lf], _fma_b2_0, _fma_c2_1);
                            float db_sq_vec[4];
                            db_sq_vec[0] = smem_database_sq[col_base];
                            db_sq_vec[1] = smem_database_sq[col_base + 1];
                            db_sq_vec[2] = smem_database_sq[col_base + 2];
                            db_sq_vec[3] = smem_database_sq[col_base + 3];
                            float _t0[4];
                            #pragma unroll
                            for (int _la = 0; _la < 2; _la++)
                                reinterpret_cast<float2*>(_t0)[_la] = add_f32x2(reinterpret_cast<float2*>(dist_vec)[_la], reinterpret_cast<const float2*>(db_sq_vec)[_la]);
                            float group_min = _t0[0];
                            if (_t0[1] < group_min) {
                                group_min = _t0[1];
                            }
                            if (_t0[2] < group_min) {
                                group_min = _t0[2];
                            }
                            if (_t0[3] < group_min) {
                                group_min = _t0[3];
                            }
                            if (group_min < best_d[TOP_K_MAX - 1]) {
                                #pragma unroll
                                for (int vec_col = 0; vec_col < 4; vec_col++) {
                                    int db_idx = db_start + col_base + vec_col;
                                    if (db_idx < M) {
                                        float dist = _t0[vec_col];
                                        if (dist < best_d[TOP_K_MAX - 1]) {
                                            best_d[TOP_K_MAX - 1] = dist;
                                            best_i[TOP_K_MAX - 1] = db_idx;
                                            #pragma unroll
                                            for (int pos = TOP_K_MAX - 1; pos >= 1; pos--) {
                                                if (best_d[pos] < best_d[pos - 1]) {
                                                    float tmp_d = best_d[pos - 1];
                                                    int tmp_i = best_i[pos - 1];
                                                    best_d[pos - 1] = best_d[pos];
                                                    best_i[pos - 1] = best_i[pos];
                                                    best_d[pos] = tmp_d;
                                                    best_i[pos] = tmp_i;
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                    asm volatile("barrier.sync 8, 128;");
                }
                if (valid_q != 0) {
                    int out_base = ((split_idx * B + batch_idx) * Q + q_idx) * K;
                    #pragma unroll
                    for (int out_k = 0; out_k < TOP_K_MAX; out_k++) {
                        if (out_k < K) {
                            *((float*)(partial_dists + out_base + out_k)) = best_d[out_k];
                            *((int*)(partial_indices + out_base + out_k)) = best_i[out_k];
                        }
                    }
                }
            }
        }
    }

    // Cleanup
    asm volatile("barrier.cluster.arrive.release.aligned;");
    asm volatile("barrier.cluster.wait.acquire.aligned;");

    if (warp == 1) {
        asm volatile("tcgen05.dealloc.cta_group::2.sync.aligned.b32 %0, %1;" :: "r"(tmem_addr_storage[0]), "r"(64));
        asm volatile("tcgen05.relinquish_alloc_permit.cta_group::2.sync.aligned;");
    }
}

} // extern "C"

#undef BLOCK_M
#undef BLOCK_Q
#undef FEAT_D
#undef NUM_MAIN_STAGES
#undef SMEM_SMEM_DATABASE_OFF
#undef SMEM_SMEM_DATABASE_SQ_OFF
#undef SMEM_SMEM_DATABASE_SQ_STAGE_BYTES
#undef SMEM_SMEM_DATABASE_SQ_STRIDE
#undef SMEM_SMEM_DATABASE_STAGE_BYTES
#undef SMEM_SMEM_DATABASE_STRIDE
#undef SMEM_SMEM_QUERY_OFF
#undef SMEM_SMEM_QUERY_STAGE_BYTES
#undef SMEM_SMEM_QUERY_STRIDE
#undef SMEM_TOTAL
#undef THREADS
#undef TMEM_CROSS_OFFSET
#undef TMEM_NCOLS
#undef TOP_K_MAX
#undef database_empty_addr
#undef database_full_addr
#undef query_empty_addr
#undef query_full_addr
#undef score_empty_addr
#undef score_full_addr
#undef smem_database_addr
#undef smem_database_sq_addr
#undef smem_query_addr

#define TMEM_NCOLS 64
#define TMEM_CROSS_OFFSET 0
#define NUM_MAIN_STAGES 1
#define SMEM_SMEM_QUERY_OFF 1024
#define SMEM_SMEM_QUERY_STAGE_BYTES 32768
#define SMEM_SMEM_QUERY_STRIDE 32768
#define SMEM_SMEM_DATABASE_OFF 33792
#define SMEM_SMEM_DATABASE_STAGE_BYTES 16384
#define SMEM_SMEM_DATABASE_STRIDE 16384
#define SMEM_SMEM_DATABASE_SQ_OFF 50176
#define SMEM_SMEM_DATABASE_SQ_STAGE_BYTES 256
#define SMEM_SMEM_DATABASE_SQ_STRIDE 256
#define SMEM_TOTAL 50432
#define THREADS 192
#define BLOCK_Q 128
#define BLOCK_M 64
#define FEAT_D 128
#define TOP_K_MAX 28

extern "C" {

__global__ __launch_bounds__(192, 1) void
kernel_knn_build_evolve_7bfc_split_cg2_u2_stage1_bad5k28s8(float* __restrict__ query_sq, float* __restrict__ database_sq, float* __restrict__ partial_dists, int32_t* __restrict__ partial_indices, const void* tmap_query, const void* tmap_database, int B, int Q, int M, int K, int num_q_tile_pairs, int db_tiles_per_split, int split_count, int total_work)
{
    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;

    extern __shared__ __align__(1024) char smem_raw[];
    int smem;
    smem = (int)(unsigned long long)__cvta_generic_to_shared(smem_raw);
    const int smem_smem_query = smem + 1024;
    const int smem_smem_database = smem + 33792;
    const int smem_smem_database_sq = smem + 50176;

    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;
    const unsigned int clusters_x = gridDim.x / 2;
    const unsigned int cluster_id = ((blockIdx.z * gridDim.y + blockIdx.y) * clusters_x) + blockIdx.x / 2;
    const unsigned int num_clusters = clusters_x * gridDim.y * gridDim.z;

    int cta_rank;
    asm volatile("mov.b32 %0, %%cluster_ctarank;" : "=r"(cta_rank));

    // Mbarrier init (6 groups, 6 barriers)
    // Mbarriers at smem_raw[0..48)

    if (warp == 0) {
        uint32_t leader = elect_sync();
        // query_full: 1 barriers, init_count=2
        mbarrier_init_pred(smem + 0, 2, leader);
        // query_empty: 1 barriers, init_count=1
        mbarrier_init_pred(smem + 8, 1, leader);
        // database_full: 1 barriers, init_count=2
        mbarrier_init_pred(smem + 16, 2, leader);
        // database_empty: 1 barriers, init_count=1
        mbarrier_init_pred(smem + 24, 1, leader);
        // score_full: 1 barriers, init_count=1
        mbarrier_init_pred(smem + 32, 1, leader);
        // score_empty: 1 barriers, init_count=8
        mbarrier_init_pred(smem + 40, 8, leader);
        asm volatile("fence.mbarrier_init.release.cluster;");
    }

    __syncthreads();

    // TMEM alloc (64 columns, 64 used)
    volatile int* tmem_addr_storage = (volatile int*)(smem_raw + 48);
    if (warp == 1) {
        int _tmem_hold = smem + 48;
        asm volatile("tcgen05.alloc.cta_group::2.sync.aligned.shared::cta.b32 [%0], %1;" :: "r"(_tmem_hold), "r"(64) : "memory");
    }

    asm volatile("barrier.cluster.arrive.release.aligned;");
    asm volatile("barrier.cluster.wait.acquire.aligned;");
    asm volatile("tcgen05.fence::after_thread_sync;");

    const int warp_id = warp;
    const int lane_id = lane;
    __nv_bfloat16* smem_query = (__nv_bfloat16*)(smem_raw + 1024);
    #define smem_query_addr (smem + 1024)
    __nv_bfloat16* smem_database = (__nv_bfloat16*)(smem_raw + 33792);
    #define smem_database_addr (smem + 33792)
    float* smem_database_sq = (float*)(smem_raw + 50176);
    #define smem_database_sq_addr (smem + 50176)
    const int mbar_base = smem;
    #define query_full_addr (mbar_base + 0)
    #define query_empty_addr (mbar_base + 8)
    #define database_full_addr (mbar_base + 16)
    #define database_empty_addr (mbar_base + 24)
    #define score_full_addr (mbar_base + 32)
    #define score_empty_addr (mbar_base + 40)
    const int taddr = tmem_addr_storage[0];

    // ---- Role: load ----
    if (warp == 0) {
        { // load_main
            uint32_t _phase_query_empty_0 = 1;
            uint32_t _phase_database_empty_0 = 1;
            if (warp_id == 0) {
                if (elect_sync()) {
                    #pragma unroll 1
                    for (unsigned int work_idx = cluster_id; work_idx < total_work; work_idx += num_clusters) {
                        int split_idx = work_idx % split_count;
                        int query_work = work_idx / split_count;
                        int batch_idx = query_work / num_q_tile_pairs;
                        int q_tile_pair = query_work % num_q_tile_pairs;
                        int q_tile = q_tile_pair * 2 + cta_rank;
                        int off_q = q_tile * BLOCK_Q;
                        int global_q = batch_idx * Q + off_q;
                        int db_tile_start = split_idx * db_tiles_per_split;
                        mbarrier_wait(query_empty_addr, _phase_query_empty_0);
                        _phase_query_empty_0 ^= 1;
                        asm volatile(
                            "mbarrier.arrive.expect_tx.release.cta.shared::cluster.b64 _, [%0], %1;"
                            :: "r"((query_full_addr) & 0xFEFFFFFF), "r"((uint32_t)(32768)) : "memory");
                        asm volatile(
                            "cp.async.bulk.tensor.3d.shared::cluster.global.mbarrier::complete_tx::bytes.cta_group::2"
                            " [%0], [%1, {%2, %3, %4}], [%5];"
                            :: "r"(smem_query_addr), "l"(tmap_query), "r"(0), "r"(global_q), "r"(0),
                               "r"(((query_full_addr) & 0xFEFFFFFF)) : "memory");
                        #pragma unroll 1
                        for (int local_db_tile = 0; local_db_tile < db_tiles_per_split; local_db_tile++) {
                            int db_tile = db_tile_start + local_db_tile;
                            int off_m = db_tile * BLOCK_M;
                            int global_m = batch_idx * M + off_m;
                            mbarrier_wait(database_empty_addr, _phase_database_empty_0);
                            _phase_database_empty_0 ^= 1;
                            asm volatile(
                                "mbarrier.arrive.expect_tx.release.cta.shared::cluster.b64 _, [%0], %1;"
                                :: "r"((database_full_addr) & 0xFEFFFFFF), "r"((uint32_t)(16384)) : "memory");
                            asm volatile(
                                "cp.async.bulk.tensor.3d.shared::cluster.global.mbarrier::complete_tx::bytes.cta_group::2"
                                " [%0], [%1, {%2, %3, %4}], [%5];"
                                :: "r"(smem_database_addr), "l"(tmap_database), "r"(0), "r"(global_m), "r"(0),
                                   "r"(((database_full_addr) & 0xFEFFFFFF)) : "memory");
                        }
                    }
                }
            }
        }
    // ---- Role: mma ----
    } else if (warp == 1) {
        { // mma_main
            uint32_t _phase_query_full_0 = 0;
            uint32_t _phase_score_empty_0 = 1;
            uint32_t _phase_database_full_0 = 0;
            if (cta_rank == 0) {
                #pragma unroll 1
                for (unsigned int work_idx = cluster_id; work_idx < total_work; work_idx += num_clusters) {
                    mbarrier_wait(query_full_addr, _phase_query_full_0);
                    _phase_query_full_0 ^= 1;
                    #pragma unroll 1
                    for (int _local_db_tile = 0; _local_db_tile < db_tiles_per_split; _local_db_tile++) {
                        mbarrier_wait(score_empty_addr, _phase_score_empty_0);
                        _phase_score_empty_0 ^= 1;
                        mbarrier_wait(database_full_addr, _phase_database_full_0);
                        _phase_database_full_0 ^= 1;
                        asm volatile("tcgen05.fence::after_thread_sync;");
                        int _mma_ss_a_lo_0 = (smem_query_addr >> 4) & 0x3FFF;
                        int _mma_ss_b_lo_0 = (smem_database_addr >> 4) & 0x3FFF;
                        asm volatile(
                    "{\n\t"
                    ".reg .pred leader, p0, p1;\n\t"
                    ".reg .b32 adhi, bdhi, alo, blo, id, m0, m1, m2, m3, m4, m5, m6, m7;\n\t"
                    ".reg .b64 da, db;\n\t"
                    "elect.sync _|leader, 0xFFFFFFFF;\n\t"
                    "setp.ne.b32 p0, %3, 0;\n\t"
                    "setp.ne.b32 p1, 1, 0;\n\t"
                    "mov.b32 m0, 0; mov.b32 m1, 0; mov.b32 m2, 0; mov.b32 m3, 0;\n\tmov.b32 m4, 0; mov.b32 m5, 0; mov.b32 m6, 0; mov.b32 m7, 0;\n\t"
                    "mov.b32 adhi, 0x40004040;\n\t"
                    "mov.b32 bdhi, 0x40004040;\n\t"
                    "mov.b32 id, 270533776;\n\t"
                    "mov.b32 alo, %0;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p0;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "add.u32 alo, alo, 1018;\n\t"
                    "add.u32 blo, blo, 506;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "}\n"
                    :: "r"(_mma_ss_a_lo_0), "r"(_mma_ss_b_lo_0), "r"(taddr), "r"(0));
                        elect_commit_cg2_multicast(score_full_addr, (uint16_t)(3));
                        elect_commit_cg2_multicast(database_empty_addr, (uint16_t)(3));
                    }
                    elect_commit_cg2_multicast(query_empty_addr, (uint16_t)(3));
                }
            }
        }
    // ---- Role: compute ----
    } else if (warp >= 2 && warp <= 5) {
        const int warp_id_in_wg = warp % 4;
        const int my_row = warp_id_in_wg * 32 + lane;
        const int tmem_row_base_v = (warp_id_in_wg * 32) << 16;
        const int tmem_cross = taddr + TMEM_CROSS_OFFSET;
        { // compute_main
            uint32_t _phase_score_full_0 = 0;
            #pragma unroll 1
            for (unsigned int work_idx = cluster_id; work_idx < total_work; work_idx += num_clusters) {
                int split_idx = work_idx % split_count;
                int query_work = work_idx / split_count;
                int batch_idx = query_work / num_q_tile_pairs;
                int q_tile_pair = query_work % num_q_tile_pairs;
                int q_tile = q_tile_pair * 2 + cta_rank;
                int off_q = q_tile * BLOCK_Q;
                int q_idx = off_q + my_row;
                int valid_q = ((q_idx < Q) ? 1 : 0);
                float q_sq_val = 0.0f;
                if (valid_q != 0) {
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
                    int db_start = db_tile * BLOCK_M;
                    int db_sq_idx = db_start + my_row;
                    if (my_row < BLOCK_M) {
                        if (db_sq_idx < M) {
                            smem_database_sq[my_row] = (float)database_sq[batch_idx * M + db_sq_idx];
                        } else {
                            smem_database_sq[my_row] = 0.0f;
                        }
                    }
                    asm volatile("barrier.sync 8, 128;");
                    mbarrier_wait(score_full_addr, _phase_score_full_0);
                    _phase_score_full_0 ^= 1;
                    int cross_addr = taddr + (cta_rank * BLOCK_Q + tmem_row_base_v << 16);
                    float dots[64];
                    asm volatile(
                        "tcgen05.ld.sync.aligned.32x32b.x64.b32"
                        " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63}, [%64];"
                        : "=f"(dots[0]), "=f"(dots[1]), "=f"(dots[2]), "=f"(dots[3]), "=f"(dots[4]), "=f"(dots[5]), "=f"(dots[6]), "=f"(dots[7]), "=f"(dots[8]), "=f"(dots[9]), "=f"(dots[10]), "=f"(dots[11]), "=f"(dots[12]), "=f"(dots[13]), "=f"(dots[14]), "=f"(dots[15]), "=f"(dots[16]), "=f"(dots[17]), "=f"(dots[18]), "=f"(dots[19]), "=f"(dots[20]), "=f"(dots[21]), "=f"(dots[22]), "=f"(dots[23]), "=f"(dots[24]), "=f"(dots[25]), "=f"(dots[26]), "=f"(dots[27]), "=f"(dots[28]), "=f"(dots[29]), "=f"(dots[30]), "=f"(dots[31]), "=f"(dots[32]), "=f"(dots[33]), "=f"(dots[34]), "=f"(dots[35]), "=f"(dots[36]), "=f"(dots[37]), "=f"(dots[38]), "=f"(dots[39]), "=f"(dots[40]), "=f"(dots[41]), "=f"(dots[42]), "=f"(dots[43]), "=f"(dots[44]), "=f"(dots[45]), "=f"(dots[46]), "=f"(dots[47]), "=f"(dots[48]), "=f"(dots[49]), "=f"(dots[50]), "=f"(dots[51]), "=f"(dots[52]), "=f"(dots[53]), "=f"(dots[54]), "=f"(dots[55]), "=f"(dots[56]), "=f"(dots[57]), "=f"(dots[58]), "=f"(dots[59]), "=f"(dots[60]), "=f"(dots[61]), "=f"(dots[62]), "=f"(dots[63])
                        : "r"(cross_addr)
                        : "memory");
                    asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                    asm volatile("barrier.sync 8, 128;");
                    if (elect_sync()) {
                        asm volatile(
                            "mbarrier.arrive.release.cta.shared::cluster.b64 _, [%0];"
                            :: "r"((score_empty_addr) & 0xFEFFFFFF) : "memory");
                    }
                    if (valid_q != 0) {
                        #pragma unroll 2
                        for (int col_base = 0; col_base < 64; col_base += 4) {
                            float dist_vec[4];
                            dist_vec[0] = dots[col_base];
                            dist_vec[1] = dots[col_base + 1];
                            dist_vec[2] = dots[col_base + 2];
                            dist_vec[3] = dots[col_base + 3];
                            const float2 _fma_b2_0 = {-2.0f, -2.0f};
                            const float2 _fma_c2_1 = {q_sq_val, q_sq_val};
                            #pragma unroll
                            for (int _lf = 0; _lf < 2; _lf++)
                                fma_f32x2_inplace(&reinterpret_cast<float2*>(dist_vec)[_lf], _fma_b2_0, _fma_c2_1);
                            float db_sq_vec[4];
                            db_sq_vec[0] = smem_database_sq[col_base];
                            db_sq_vec[1] = smem_database_sq[col_base + 1];
                            db_sq_vec[2] = smem_database_sq[col_base + 2];
                            db_sq_vec[3] = smem_database_sq[col_base + 3];
                            float _t0[4];
                            #pragma unroll
                            for (int _la = 0; _la < 2; _la++)
                                reinterpret_cast<float2*>(_t0)[_la] = add_f32x2(reinterpret_cast<float2*>(dist_vec)[_la], reinterpret_cast<const float2*>(db_sq_vec)[_la]);
                            float group_min = _t0[0];
                            if (_t0[1] < group_min) {
                                group_min = _t0[1];
                            }
                            if (_t0[2] < group_min) {
                                group_min = _t0[2];
                            }
                            if (_t0[3] < group_min) {
                                group_min = _t0[3];
                            }
                            if (group_min < best_d[TOP_K_MAX - 1]) {
                                #pragma unroll
                                for (int vec_col = 0; vec_col < 4; vec_col++) {
                                    int db_idx = db_start + col_base + vec_col;
                                    if (db_idx < M) {
                                        float dist = _t0[vec_col];
                                        if (dist < best_d[TOP_K_MAX - 1]) {
                                            best_d[TOP_K_MAX - 1] = dist;
                                            best_i[TOP_K_MAX - 1] = db_idx;
                                            #pragma unroll
                                            for (int pos = TOP_K_MAX - 1; pos >= 1; pos--) {
                                                if (best_d[pos] < best_d[pos - 1]) {
                                                    float tmp_d = best_d[pos - 1];
                                                    int tmp_i = best_i[pos - 1];
                                                    best_d[pos - 1] = best_d[pos];
                                                    best_i[pos - 1] = best_i[pos];
                                                    best_d[pos] = tmp_d;
                                                    best_i[pos] = tmp_i;
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                    asm volatile("barrier.sync 8, 128;");
                }
                if (valid_q != 0) {
                    int out_base = ((split_idx * B + batch_idx) * Q + q_idx) * K;
                    #pragma unroll
                    for (int out_k = 0; out_k < TOP_K_MAX; out_k++) {
                        if (out_k < K) {
                            *((float*)(partial_dists + out_base + out_k)) = best_d[out_k];
                            *((int*)(partial_indices + out_base + out_k)) = best_i[out_k];
                        }
                    }
                }
            }
        }
    }

    // Cleanup
    asm volatile("barrier.cluster.arrive.release.aligned;");
    asm volatile("barrier.cluster.wait.acquire.aligned;");

    if (warp == 1) {
        asm volatile("tcgen05.dealloc.cta_group::2.sync.aligned.b32 %0, %1;" :: "r"(tmem_addr_storage[0]), "r"(64));
        asm volatile("tcgen05.relinquish_alloc_permit.cta_group::2.sync.aligned;");
    }
}

} // extern "C"

#undef BLOCK_M
#undef BLOCK_Q
#undef FEAT_D
#undef NUM_MAIN_STAGES
#undef SMEM_SMEM_DATABASE_OFF
#undef SMEM_SMEM_DATABASE_SQ_OFF
#undef SMEM_SMEM_DATABASE_SQ_STAGE_BYTES
#undef SMEM_SMEM_DATABASE_SQ_STRIDE
#undef SMEM_SMEM_DATABASE_STAGE_BYTES
#undef SMEM_SMEM_DATABASE_STRIDE
#undef SMEM_SMEM_QUERY_OFF
#undef SMEM_SMEM_QUERY_STAGE_BYTES
#undef SMEM_SMEM_QUERY_STRIDE
#undef SMEM_TOTAL
#undef THREADS
#undef TMEM_CROSS_OFFSET
#undef TMEM_NCOLS
#undef TOP_K_MAX
#undef database_empty_addr
#undef database_full_addr
#undef query_empty_addr
#undef query_full_addr
#undef score_empty_addr
#undef score_full_addr
#undef smem_database_addr
#undef smem_database_sq_addr
#undef smem_query_addr

#define TMEM_NCOLS 64
#define TMEM_CROSS_OFFSET 0
#define NUM_MAIN_STAGES 1
#define SMEM_SMEM_QUERY_OFF 1024
#define SMEM_SMEM_QUERY_STAGE_BYTES 32768
#define SMEM_SMEM_QUERY_STRIDE 32768
#define SMEM_SMEM_DATABASE_OFF 33792
#define SMEM_SMEM_DATABASE_STAGE_BYTES 16384
#define SMEM_SMEM_DATABASE_STRIDE 16384
#define SMEM_SMEM_DATABASE_SQ_OFF 50176
#define SMEM_SMEM_DATABASE_SQ_STAGE_BYTES 256
#define SMEM_SMEM_DATABASE_SQ_STRIDE 256
#define SMEM_TOTAL 50432
#define THREADS 192
#define BLOCK_Q 128
#define BLOCK_M 64
#define FEAT_D 128
#define TOP_K_MAX 28

extern "C" {

__global__ __launch_bounds__(192, 1) void
kernel_knn_build_evolve_7bfc_split_cg2_u2_stage1_k32_unordered_k30unordered_bad5k28unordered(float* __restrict__ query_sq, float* __restrict__ database_sq, float* __restrict__ partial_dists, int32_t* __restrict__ partial_indices, const void* tmap_query, const void* tmap_database, int B, int Q, int M, int K, int num_q_tile_pairs, int db_tiles_per_split, int split_count, int total_work)
{
    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;

    extern __shared__ __align__(1024) char smem_raw[];
    int smem;
    smem = (int)(unsigned long long)__cvta_generic_to_shared(smem_raw);
    const int smem_smem_query = smem + 1024;
    const int smem_smem_database = smem + 33792;
    const int smem_smem_database_sq = smem + 50176;

    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;
    const unsigned int clusters_x = gridDim.x / 2;
    const unsigned int cluster_id = ((blockIdx.z * gridDim.y + blockIdx.y) * clusters_x) + blockIdx.x / 2;
    const unsigned int num_clusters = clusters_x * gridDim.y * gridDim.z;

    int cta_rank;
    asm volatile("mov.b32 %0, %%cluster_ctarank;" : "=r"(cta_rank));

    // Mbarrier init (6 groups, 6 barriers)
    // Mbarriers at smem_raw[0..48)

    if (warp == 0) {
        uint32_t leader = elect_sync();
        // query_full: 1 barriers, init_count=2
        mbarrier_init_pred(smem + 0, 2, leader);
        // query_empty: 1 barriers, init_count=1
        mbarrier_init_pred(smem + 8, 1, leader);
        // database_full: 1 barriers, init_count=2
        mbarrier_init_pred(smem + 16, 2, leader);
        // database_empty: 1 barriers, init_count=1
        mbarrier_init_pred(smem + 24, 1, leader);
        // score_full: 1 barriers, init_count=1
        mbarrier_init_pred(smem + 32, 1, leader);
        // score_empty: 1 barriers, init_count=8
        mbarrier_init_pred(smem + 40, 8, leader);
        asm volatile("fence.mbarrier_init.release.cluster;");
    }

    __syncthreads();

    // TMEM alloc (64 columns, 64 used)
    volatile int* tmem_addr_storage = (volatile int*)(smem_raw + 48);
    if (warp == 1) {
        int _tmem_hold = smem + 48;
        asm volatile("tcgen05.alloc.cta_group::2.sync.aligned.shared::cta.b32 [%0], %1;" :: "r"(_tmem_hold), "r"(64) : "memory");
    }

    asm volatile("barrier.cluster.arrive.release.aligned;");
    asm volatile("barrier.cluster.wait.acquire.aligned;");
    asm volatile("tcgen05.fence::after_thread_sync;");

    const int warp_id = warp;
    const int lane_id = lane;
    __nv_bfloat16* smem_query = (__nv_bfloat16*)(smem_raw + 1024);
    #define smem_query_addr (smem + 1024)
    __nv_bfloat16* smem_database = (__nv_bfloat16*)(smem_raw + 33792);
    #define smem_database_addr (smem + 33792)
    float* smem_database_sq = (float*)(smem_raw + 50176);
    #define smem_database_sq_addr (smem + 50176)
    const int mbar_base = smem;
    #define query_full_addr (mbar_base + 0)
    #define query_empty_addr (mbar_base + 8)
    #define database_full_addr (mbar_base + 16)
    #define database_empty_addr (mbar_base + 24)
    #define score_full_addr (mbar_base + 32)
    #define score_empty_addr (mbar_base + 40)
    const int taddr = tmem_addr_storage[0];

    // ---- Role: load ----
    if (warp == 0) {
        { // load_main
            uint32_t _phase_query_empty_0 = 1;
            uint32_t _phase_database_empty_0 = 1;
            if (warp_id == 0) {
                if (elect_sync()) {
                    #pragma unroll 1
                    for (unsigned int work_idx = cluster_id; work_idx < total_work; work_idx += num_clusters) {
                        int split_idx = work_idx % split_count;
                        int query_work = work_idx / split_count;
                        int batch_idx = query_work / num_q_tile_pairs;
                        int q_tile_pair = query_work % num_q_tile_pairs;
                        int q_tile = q_tile_pair * 2 + cta_rank;
                        int off_q = q_tile * BLOCK_Q;
                        int global_q = batch_idx * Q + off_q;
                        int db_tile_start = split_idx * db_tiles_per_split;
                        mbarrier_wait(query_empty_addr, _phase_query_empty_0);
                        _phase_query_empty_0 ^= 1;
                        asm volatile(
                            "mbarrier.arrive.expect_tx.release.cta.shared::cluster.b64 _, [%0], %1;"
                            :: "r"((query_full_addr) & 0xFEFFFFFF), "r"((uint32_t)(32768)) : "memory");
                        asm volatile(
                            "cp.async.bulk.tensor.3d.shared::cluster.global.mbarrier::complete_tx::bytes.cta_group::2"
                            " [%0], [%1, {%2, %3, %4}], [%5];"
                            :: "r"(smem_query_addr), "l"(tmap_query), "r"(0), "r"(global_q), "r"(0),
                               "r"(((query_full_addr) & 0xFEFFFFFF)) : "memory");
                        #pragma unroll 1
                        for (int local_db_tile = 0; local_db_tile < db_tiles_per_split; local_db_tile++) {
                            int db_tile = db_tile_start + local_db_tile;
                            int off_m = db_tile * BLOCK_M;
                            int global_m = batch_idx * M + off_m;
                            mbarrier_wait(database_empty_addr, _phase_database_empty_0);
                            _phase_database_empty_0 ^= 1;
                            asm volatile(
                                "mbarrier.arrive.expect_tx.release.cta.shared::cluster.b64 _, [%0], %1;"
                                :: "r"((database_full_addr) & 0xFEFFFFFF), "r"((uint32_t)(16384)) : "memory");
                            asm volatile(
                                "cp.async.bulk.tensor.3d.shared::cluster.global.mbarrier::complete_tx::bytes.cta_group::2"
                                " [%0], [%1, {%2, %3, %4}], [%5];"
                                :: "r"(smem_database_addr), "l"(tmap_database), "r"(0), "r"(global_m), "r"(0),
                                   "r"(((database_full_addr) & 0xFEFFFFFF)) : "memory");
                        }
                    }
                }
            }
        }
    // ---- Role: mma ----
    } else if (warp == 1) {
        { // mma_main
            uint32_t _phase_query_full_0 = 0;
            uint32_t _phase_score_empty_0 = 1;
            uint32_t _phase_database_full_0 = 0;
            if (cta_rank == 0) {
                #pragma unroll 1
                for (unsigned int work_idx = cluster_id; work_idx < total_work; work_idx += num_clusters) {
                    mbarrier_wait(query_full_addr, _phase_query_full_0);
                    _phase_query_full_0 ^= 1;
                    #pragma unroll 1
                    for (int _local_db_tile = 0; _local_db_tile < db_tiles_per_split; _local_db_tile++) {
                        mbarrier_wait(score_empty_addr, _phase_score_empty_0);
                        _phase_score_empty_0 ^= 1;
                        mbarrier_wait(database_full_addr, _phase_database_full_0);
                        _phase_database_full_0 ^= 1;
                        asm volatile("tcgen05.fence::after_thread_sync;");
                        int _mma_ss_a_lo_0 = (smem_query_addr >> 4) & 0x3FFF;
                        int _mma_ss_b_lo_0 = (smem_database_addr >> 4) & 0x3FFF;
                        asm volatile(
                    "{\n\t"
                    ".reg .pred leader, p0, p1;\n\t"
                    ".reg .b32 adhi, bdhi, alo, blo, id, m0, m1, m2, m3, m4, m5, m6, m7;\n\t"
                    ".reg .b64 da, db;\n\t"
                    "elect.sync _|leader, 0xFFFFFFFF;\n\t"
                    "setp.ne.b32 p0, %3, 0;\n\t"
                    "setp.ne.b32 p1, 1, 0;\n\t"
                    "mov.b32 m0, 0; mov.b32 m1, 0; mov.b32 m2, 0; mov.b32 m3, 0;\n\tmov.b32 m4, 0; mov.b32 m5, 0; mov.b32 m6, 0; mov.b32 m7, 0;\n\t"
                    "mov.b32 adhi, 0x40004040;\n\t"
                    "mov.b32 bdhi, 0x40004040;\n\t"
                    "mov.b32 id, 270533776;\n\t"
                    "mov.b32 alo, %0;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p0;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "add.u32 alo, alo, 1018;\n\t"
                    "add.u32 blo, blo, 506;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "}\n"
                    :: "r"(_mma_ss_a_lo_0), "r"(_mma_ss_b_lo_0), "r"(taddr), "r"(0));
                        elect_commit_cg2_multicast(score_full_addr, (uint16_t)(3));
                        elect_commit_cg2_multicast(database_empty_addr, (uint16_t)(3));
                    }
                    elect_commit_cg2_multicast(query_empty_addr, (uint16_t)(3));
                }
            }
        }
    // ---- Role: compute ----
    } else if (warp >= 2 && warp <= 5) {
        const int warp_id_in_wg = warp % 4;
        const int my_row = warp_id_in_wg * 32 + lane;
        const int tmem_row_base_v = (warp_id_in_wg * 32) << 16;
        const int tmem_cross = taddr + TMEM_CROSS_OFFSET;
        { // compute_main
            uint32_t _phase_score_full_0 = 0;
            #pragma unroll 1
            for (unsigned int work_idx = cluster_id; work_idx < total_work; work_idx += num_clusters) {
                int split_idx = work_idx % split_count;
                int query_work = work_idx / split_count;
                int batch_idx = query_work / num_q_tile_pairs;
                int q_tile_pair = query_work % num_q_tile_pairs;
                int q_tile = q_tile_pair * 2 + cta_rank;
                int off_q = q_tile * BLOCK_Q;
                int q_idx = off_q + my_row;
                int valid_q = ((q_idx < Q) ? 1 : 0);
                float q_sq_val = 0.0f;
                if (valid_q != 0) {
                    q_sq_val = (float)query_sq[batch_idx * Q + q_idx];
                }
                float best_d[TOP_K_MAX];
                int best_i[TOP_K_MAX];
                #pragma unroll
                for (int kk = 0; kk < TOP_K_MAX; kk++) {
                    best_d[kk] = 3.4e+38f;
                    best_i[kk] = -1;
                }
                float worst_d = 3.4e+38f;
                int worst_pos = 0;
                int db_tile_start = split_idx * db_tiles_per_split;
                #pragma unroll 1
                for (int local_db_tile = 0; local_db_tile < db_tiles_per_split; local_db_tile++) {
                    int db_tile = db_tile_start + local_db_tile;
                    int db_start = db_tile * BLOCK_M;
                    int db_sq_idx = db_start + my_row;
                    if (my_row < BLOCK_M) {
                        if (db_sq_idx < M) {
                            smem_database_sq[my_row] = (float)database_sq[batch_idx * M + db_sq_idx];
                        } else {
                            smem_database_sq[my_row] = 0.0f;
                        }
                    }
                    asm volatile("barrier.sync 8, 128;");
                    mbarrier_wait(score_full_addr, _phase_score_full_0);
                    _phase_score_full_0 ^= 1;
                    int cross_addr = taddr + (cta_rank * BLOCK_Q + tmem_row_base_v << 16);
                    float dots[64];
                    asm volatile(
                        "tcgen05.ld.sync.aligned.32x32b.x64.b32"
                        " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63}, [%64];"
                        : "=f"(dots[0]), "=f"(dots[1]), "=f"(dots[2]), "=f"(dots[3]), "=f"(dots[4]), "=f"(dots[5]), "=f"(dots[6]), "=f"(dots[7]), "=f"(dots[8]), "=f"(dots[9]), "=f"(dots[10]), "=f"(dots[11]), "=f"(dots[12]), "=f"(dots[13]), "=f"(dots[14]), "=f"(dots[15]), "=f"(dots[16]), "=f"(dots[17]), "=f"(dots[18]), "=f"(dots[19]), "=f"(dots[20]), "=f"(dots[21]), "=f"(dots[22]), "=f"(dots[23]), "=f"(dots[24]), "=f"(dots[25]), "=f"(dots[26]), "=f"(dots[27]), "=f"(dots[28]), "=f"(dots[29]), "=f"(dots[30]), "=f"(dots[31]), "=f"(dots[32]), "=f"(dots[33]), "=f"(dots[34]), "=f"(dots[35]), "=f"(dots[36]), "=f"(dots[37]), "=f"(dots[38]), "=f"(dots[39]), "=f"(dots[40]), "=f"(dots[41]), "=f"(dots[42]), "=f"(dots[43]), "=f"(dots[44]), "=f"(dots[45]), "=f"(dots[46]), "=f"(dots[47]), "=f"(dots[48]), "=f"(dots[49]), "=f"(dots[50]), "=f"(dots[51]), "=f"(dots[52]), "=f"(dots[53]), "=f"(dots[54]), "=f"(dots[55]), "=f"(dots[56]), "=f"(dots[57]), "=f"(dots[58]), "=f"(dots[59]), "=f"(dots[60]), "=f"(dots[61]), "=f"(dots[62]), "=f"(dots[63])
                        : "r"(cross_addr)
                        : "memory");
                    asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                    asm volatile("barrier.sync 8, 128;");
                    if (elect_sync()) {
                        asm volatile(
                            "mbarrier.arrive.release.cta.shared::cluster.b64 _, [%0];"
                            :: "r"((score_empty_addr) & 0xFEFFFFFF) : "memory");
                    }
                    if (valid_q != 0) {
                        #pragma unroll 2
                        for (int col_base = 0; col_base < 64; col_base += 4) {
                            float dist_vec[4];
                            dist_vec[0] = dots[col_base];
                            dist_vec[1] = dots[col_base + 1];
                            dist_vec[2] = dots[col_base + 2];
                            dist_vec[3] = dots[col_base + 3];
                            const float2 _fma_b2_0 = {-2.0f, -2.0f};
                            const float2 _fma_c2_1 = {q_sq_val, q_sq_val};
                            #pragma unroll
                            for (int _lf = 0; _lf < 2; _lf++)
                                fma_f32x2_inplace(&reinterpret_cast<float2*>(dist_vec)[_lf], _fma_b2_0, _fma_c2_1);
                            float db_sq_vec[4];
                            db_sq_vec[0] = smem_database_sq[col_base];
                            db_sq_vec[1] = smem_database_sq[col_base + 1];
                            db_sq_vec[2] = smem_database_sq[col_base + 2];
                            db_sq_vec[3] = smem_database_sq[col_base + 3];
                            float _t0[4];
                            #pragma unroll
                            for (int _la = 0; _la < 2; _la++)
                                reinterpret_cast<float2*>(_t0)[_la] = add_f32x2(reinterpret_cast<float2*>(dist_vec)[_la], reinterpret_cast<const float2*>(db_sq_vec)[_la]);
                            float group_min = _t0[0];
                            if (_t0[1] < group_min) {
                                group_min = _t0[1];
                            }
                            if (_t0[2] < group_min) {
                                group_min = _t0[2];
                            }
                            if (_t0[3] < group_min) {
                                group_min = _t0[3];
                            }
                            if (group_min < worst_d) {
                                #pragma unroll
                                for (int vec_col = 0; vec_col < 4; vec_col++) {
                                    int db_idx = db_start + col_base + vec_col;
                                    if (db_idx < M) {
                                        float dist = _t0[vec_col];
                                        if (dist < worst_d) {
                                            best_d[worst_pos] = dist;
                                            best_i[worst_pos] = db_idx;
                                            worst_d = best_d[0];
                                            worst_pos = 0;
                                            #pragma unroll
                                            for (int scan_pos = 1; scan_pos < TOP_K_MAX; scan_pos++) {
                                                if (best_d[scan_pos] > worst_d) {
                                                    worst_d = best_d[scan_pos];
                                                    worst_pos = scan_pos;
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                    asm volatile("barrier.sync 8, 128;");
                }
                if (valid_q != 0) {
                    int out_base = ((split_idx * B + batch_idx) * Q + q_idx) * K;
                    #pragma unroll
                    for (int out_k = 0; out_k < TOP_K_MAX; out_k++) {
                        *((float*)(partial_dists + out_base + out_k)) = best_d[out_k];
                        *((int*)(partial_indices + out_base + out_k)) = best_i[out_k];
                    }
                }
            }
        }
    }

    // Cleanup
    asm volatile("barrier.cluster.arrive.release.aligned;");
    asm volatile("barrier.cluster.wait.acquire.aligned;");

    if (warp == 1) {
        asm volatile("tcgen05.dealloc.cta_group::2.sync.aligned.b32 %0, %1;" :: "r"(tmem_addr_storage[0]), "r"(64));
        asm volatile("tcgen05.relinquish_alloc_permit.cta_group::2.sync.aligned;");
    }
}

} // extern "C"

#undef BLOCK_M
#undef BLOCK_Q
#undef FEAT_D
#undef NUM_MAIN_STAGES
#undef SMEM_SMEM_DATABASE_OFF
#undef SMEM_SMEM_DATABASE_SQ_OFF
#undef SMEM_SMEM_DATABASE_SQ_STAGE_BYTES
#undef SMEM_SMEM_DATABASE_SQ_STRIDE
#undef SMEM_SMEM_DATABASE_STAGE_BYTES
#undef SMEM_SMEM_DATABASE_STRIDE
#undef SMEM_SMEM_QUERY_OFF
#undef SMEM_SMEM_QUERY_STAGE_BYTES
#undef SMEM_SMEM_QUERY_STRIDE
#undef SMEM_TOTAL
#undef THREADS
#undef TMEM_CROSS_OFFSET
#undef TMEM_NCOLS
#undef TOP_K_MAX
#undef database_empty_addr
#undef database_full_addr
#undef query_empty_addr
#undef query_full_addr
#undef score_empty_addr
#undef score_full_addr
#undef smem_database_addr
#undef smem_database_sq_addr
#undef smem_query_addr

#define NUM_MAIN_STAGES 1
#define THREADS 32
#define TOP_K_MAX 24
#define SPLIT_COUNT 8

extern "C" {

__global__ __launch_bounds__(32, 1) void
kernel_knn_build_evolve_7bfc_k30_merge_s8_rowbase_cache_bad5k24s8(float* __restrict__ partial_dists, int32_t* __restrict__ partial_indices, float* __restrict__ out_dists, int32_t* __restrict__ out_indices, int total_queries)
{
    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;


    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    const int warp_id = warp;
    const int lane_id = lane;

    // === Task calls (dependency order) ===
    int start_row = bid * 32 + tid;
    int stride = num_bids * 32;
    #pragma unroll 1
    for (int row = start_row; row < total_queries; row += stride) {
        int base_row = row * TOP_K_MAX;
        int split_stride = total_queries * TOP_K_MAX;
        int out_base = base_row;
        int split_pos[SPLIT_COUNT];
        int split_base[SPLIT_COUNT];
        float cand_d[SPLIT_COUNT];
        int cand_i[SPLIT_COUNT];
        #pragma unroll
        for (int split_idx = 0; split_idx < SPLIT_COUNT; split_idx++) {
            split_pos[split_idx] = 0;
            split_base[split_idx] = base_row + split_idx * split_stride;
            cand_d[split_idx] = (float)partial_dists[split_base[split_idx]];
            cand_i[split_idx] = partial_indices[split_base[split_idx]];
        }
        #pragma unroll
        for (int out_k = 0; out_k < TOP_K_MAX; out_k++) {
            float best_d = cand_d[0];
            int best_i = cand_i[0];
            int best_split = 0;
            #pragma unroll
            for (int split_idx = 1; split_idx < SPLIT_COUNT; split_idx++) {
                if (cand_d[split_idx] < best_d) {
                    best_d = cand_d[split_idx];
                    best_i = cand_i[split_idx];
                    best_split = split_idx;
                }
            }
            *((float*)(out_dists + out_base + out_k)) = best_d;
            *((int*)(out_indices + out_base + out_k)) = best_i;
            split_pos[best_split] = split_pos[best_split] + 1;
            if (out_k + 1 < TOP_K_MAX) {
                int next_pos = split_pos[best_split];
                int next_addr = split_base[best_split] + next_pos;
                cand_d[best_split] = (float)partial_dists[next_addr];
                cand_i[best_split] = partial_indices[next_addr];
            }
        }
    }
}

} // extern "C"

#undef NUM_MAIN_STAGES
#undef SPLIT_COUNT
#undef THREADS
#undef TOP_K_MAX

#define NUM_MAIN_STAGES 1
#define THREADS 32
#define TOP_K_MAX 28
#define SPLIT_COUNT 8

extern "C" {

__global__ __launch_bounds__(32, 1) void
kernel_knn_build_evolve_7bfc_k30_merge_s8_rowbase_cache_bad5k28s8(float* __restrict__ partial_dists, int32_t* __restrict__ partial_indices, float* __restrict__ out_dists, int32_t* __restrict__ out_indices, int total_queries)
{
    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;


    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    const int warp_id = warp;
    const int lane_id = lane;

    // === Task calls (dependency order) ===
    int start_row = bid * 32 + tid;
    int stride = num_bids * 32;
    #pragma unroll 1
    for (int row = start_row; row < total_queries; row += stride) {
        int base_row = row * TOP_K_MAX;
        int split_stride = total_queries * TOP_K_MAX;
        int out_base = base_row;
        int split_pos[SPLIT_COUNT];
        int split_base[SPLIT_COUNT];
        float cand_d[SPLIT_COUNT];
        int cand_i[SPLIT_COUNT];
        #pragma unroll
        for (int split_idx = 0; split_idx < SPLIT_COUNT; split_idx++) {
            split_pos[split_idx] = 0;
            split_base[split_idx] = base_row + split_idx * split_stride;
            cand_d[split_idx] = (float)partial_dists[split_base[split_idx]];
            cand_i[split_idx] = partial_indices[split_base[split_idx]];
        }
        #pragma unroll
        for (int out_k = 0; out_k < TOP_K_MAX; out_k++) {
            float best_d = cand_d[0];
            int best_i = cand_i[0];
            int best_split = 0;
            #pragma unroll
            for (int split_idx = 1; split_idx < SPLIT_COUNT; split_idx++) {
                if (cand_d[split_idx] < best_d) {
                    best_d = cand_d[split_idx];
                    best_i = cand_i[split_idx];
                    best_split = split_idx;
                }
            }
            *((float*)(out_dists + out_base + out_k)) = best_d;
            *((int*)(out_indices + out_base + out_k)) = best_i;
            split_pos[best_split] = split_pos[best_split] + 1;
            if (out_k + 1 < TOP_K_MAX) {
                int next_pos = split_pos[best_split];
                int next_addr = split_base[best_split] + next_pos;
                cand_d[best_split] = (float)partial_dists[next_addr];
                cand_i[best_split] = partial_indices[next_addr];
            }
        }
    }
}

} // extern "C"

#undef NUM_MAIN_STAGES
#undef SPLIT_COUNT
#undef THREADS
#undef TOP_K_MAX

#define NUM_MAIN_STAGES 1
#define THREADS 32
#define TOP_K_MAX 28
#define SPLIT_COUNT 4

extern "C" {

__global__ __launch_bounds__(32, 1) void
kernel_knn_build_evolve_7bfc_k32_merge_s4_unordered_k30unordered_bad5k28unordered(float* __restrict__ partial_dists, int32_t* __restrict__ partial_indices, float* __restrict__ out_dists, int32_t* __restrict__ out_indices, int total_queries)
{
    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;


    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    const int warp_id = warp;
    const int lane_id = lane;

    // === Task calls (dependency order) ===
    int start_row = bid * 32 + tid;
    int stride = num_bids * 32;
    #pragma unroll 1
    for (int row = start_row; row < total_queries; row += stride) {
        int base_row = row * TOP_K_MAX;
        int split_stride = total_queries * TOP_K_MAX;
        float best_d[TOP_K_MAX];
        int best_i[TOP_K_MAX];
        #pragma unroll
        for (int kk = 0; kk < TOP_K_MAX; kk++) {
            best_d[kk] = 3.4e+38f;
            best_i[kk] = -1;
        }
        float worst_d = 3.4e+38f;
        int worst_pos = 0;
        #pragma unroll
        for (int split_idx = 0; split_idx < SPLIT_COUNT; split_idx++) {
            int partial_base = base_row + split_idx * split_stride;
            #pragma unroll
            for (int cand_k = 0; cand_k < TOP_K_MAX; cand_k++) {
                float cand_d = (float)partial_dists[partial_base + cand_k];
                int cand_i = partial_indices[partial_base + cand_k];
                if (cand_d < worst_d) {
                    best_d[worst_pos] = cand_d;
                    best_i[worst_pos] = cand_i;
                    worst_d = best_d[0];
                    worst_pos = 0;
                    #pragma unroll
                    for (int scan_pos = 1; scan_pos < TOP_K_MAX; scan_pos++) {
                        if (best_d[scan_pos] > worst_d) {
                            worst_d = best_d[scan_pos];
                            worst_pos = scan_pos;
                        }
                    }
                }
            }
        }
        #pragma unroll
        for (int out_k = 0; out_k < TOP_K_MAX; out_k++) {
            *((float*)(out_dists + base_row + out_k)) = best_d[out_k];
            *((int*)(out_indices + base_row + out_k)) = best_i[out_k];
        }
    }
}

} // extern "C"

#undef NUM_MAIN_STAGES
#undef SPLIT_COUNT
#undef THREADS
#undef TOP_K_MAX

#define TMEM_NCOLS 64
#define TMEM_CROSS_OFFSET 0
#define NUM_MAIN_STAGES 1
#define SMEM_SMEM_QUERY_OFF 1024
#define SMEM_SMEM_QUERY_STAGE_BYTES 32768
#define SMEM_SMEM_QUERY_STRIDE 32768
#define SMEM_SMEM_DATABASE_OFF 33792
#define SMEM_SMEM_DATABASE_STAGE_BYTES 16384
#define SMEM_SMEM_DATABASE_STRIDE 16384
#define SMEM_SMEM_DATABASE_SQ_OFF 50176
#define SMEM_SMEM_DATABASE_SQ_STAGE_BYTES 256
#define SMEM_SMEM_DATABASE_SQ_STRIDE 256
#define SMEM_TOTAL 50432
#define THREADS 192
#define BLOCK_Q 128
#define BLOCK_M 64
#define FEAT_D 128
#define TOP_K_MAX 64

extern "C" {

__global__ __launch_bounds__(192, 1) void
kernel_knn_build_k64_stage1_tailinf_k64over32tailinfsplitgrid(float* __restrict__ query_sq, float* __restrict__ database_sq, float* __restrict__ partial_dists, int32_t* __restrict__ partial_indices, const void* tmap_query, const void* tmap_database, int B, int Q, int M, int K, int num_q_tile_pairs, int db_tiles_per_split, int split_count, int total_work)
{
    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;

    extern __shared__ __align__(1024) char smem_raw[];
    int smem;
    smem = (int)(unsigned long long)__cvta_generic_to_shared(smem_raw);
    const int smem_smem_query = smem + 1024;
    const int smem_smem_database = smem + 33792;
    const int smem_smem_database_sq = smem + 50176;

    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;
    const unsigned int clusters_x = gridDim.x / 2;
    const unsigned int cluster_id = ((blockIdx.z * gridDim.y + blockIdx.y) * clusters_x) + blockIdx.x / 2;
    const unsigned int num_clusters = clusters_x * gridDim.y * gridDim.z;

    int cta_rank;
    asm volatile("mov.b32 %0, %%cluster_ctarank;" : "=r"(cta_rank));

    // Mbarrier init (6 groups, 6 barriers)
    // Mbarriers at smem_raw[0..48)

    if (warp == 0) {
        uint32_t leader = elect_sync();
        // query_full: 1 barriers, init_count=2
        mbarrier_init_pred(smem + 0, 2, leader);
        // query_empty: 1 barriers, init_count=1
        mbarrier_init_pred(smem + 8, 1, leader);
        // database_full: 1 barriers, init_count=2
        mbarrier_init_pred(smem + 16, 2, leader);
        // database_empty: 1 barriers, init_count=1
        mbarrier_init_pred(smem + 24, 1, leader);
        // score_full: 1 barriers, init_count=1
        mbarrier_init_pred(smem + 32, 1, leader);
        // score_empty: 1 barriers, init_count=8
        mbarrier_init_pred(smem + 40, 8, leader);
        asm volatile("fence.mbarrier_init.release.cluster;");
    }

    __syncthreads();

    // TMEM alloc (64 columns, 64 used)
    volatile int* tmem_addr_storage = (volatile int*)(smem_raw + 48);
    if (warp == 5) {
        int _tmem_hold = smem + 48;
        asm volatile("tcgen05.alloc.cta_group::2.sync.aligned.shared::cta.b32 [%0], %1;" :: "r"(_tmem_hold), "r"(64) : "memory");
    }

    asm volatile("barrier.cluster.arrive.release.aligned;");
    asm volatile("barrier.cluster.wait.acquire.aligned;");
    asm volatile("tcgen05.fence::after_thread_sync;");

    const int warp_id = warp;
    const int lane_id = lane;
    __nv_bfloat16* smem_query = (__nv_bfloat16*)(smem_raw + 1024);
    #define smem_query_addr (smem + 1024)
    __nv_bfloat16* smem_database = (__nv_bfloat16*)(smem_raw + 33792);
    #define smem_database_addr (smem + 33792)
    float* smem_database_sq = (float*)(smem_raw + 50176);
    #define smem_database_sq_addr (smem + 50176)
    const int mbar_base = smem;
    #define query_full_addr (mbar_base + 0)
    #define query_empty_addr (mbar_base + 8)
    #define database_full_addr (mbar_base + 16)
    #define database_empty_addr (mbar_base + 24)
    #define score_full_addr (mbar_base + 32)
    #define score_empty_addr (mbar_base + 40)
    const int taddr = tmem_addr_storage[0];

    // ---- Role: compute ----
    if (warp <= 3) {
        const int warp_id_in_wg = warp % 4;
        const int my_row = warp_id_in_wg * 32 + lane;
        const int tmem_row_base_v = (warp_id_in_wg * 32) << 16;
        const int tmem_cross = taddr + TMEM_CROSS_OFFSET;
        { // compute_main
            uint32_t _phase_score_full_0 = 0;
            #pragma unroll 1
            for (unsigned int work_idx = cluster_id; work_idx < total_work; work_idx += num_clusters) {
                int split_idx = work_idx % split_count;
                int query_work = work_idx / split_count;
                int batch_idx = query_work / num_q_tile_pairs;
                int q_tile_pair = query_work % num_q_tile_pairs;
                int q_tile = q_tile_pair * 2 + cta_rank;
                int off_q = q_tile * BLOCK_Q;
                int q_idx = off_q + my_row;
                int valid_q = ((q_idx < Q) ? 1 : 0);
                float q_sq_val = 0.0f;
                if (valid_q != 0) {
                    q_sq_val = (float)query_sq[batch_idx * Q + q_idx];
                }
                float best_d[64];
                int best_i[64];
                #pragma unroll
                for (int kk = 0; kk < 64; kk++) {
                    best_d[kk] = 3.4e+38f;
                    best_i[kk] = -1;
                }
                float worst_d = 3.4e+38f;
                int worst_pos = 0;
                int worst_chunk = 0;
                float c0_worst_d = 3.4e+38f;
                int c0_worst_pos = 0;
                float c1_worst_d = 3.4e+38f;
                int c1_worst_pos = 4;
                float c2_worst_d = 3.4e+38f;
                int c2_worst_pos = 8;
                float c3_worst_d = 3.4e+38f;
                int c3_worst_pos = 12;
                float c4_worst_d = 3.4e+38f;
                int c4_worst_pos = 16;
                float c5_worst_d = 3.4e+38f;
                int c5_worst_pos = 20;
                float c6_worst_d = 3.4e+38f;
                int c6_worst_pos = 24;
                float c7_worst_d = 3.4e+38f;
                int c7_worst_pos = 28;
                float c8_worst_d = 3.4e+38f;
                int c8_worst_pos = 32;
                float c9_worst_d = 3.4e+38f;
                int c9_worst_pos = 36;
                float c10_worst_d = 3.4e+38f;
                int c10_worst_pos = 40;
                float c11_worst_d = 3.4e+38f;
                int c11_worst_pos = 44;
                float c12_worst_d = 3.4e+38f;
                int c12_worst_pos = 48;
                float c13_worst_d = 3.4e+38f;
                int c13_worst_pos = 52;
                float c14_worst_d = 3.4e+38f;
                int c14_worst_pos = 56;
                float c15_worst_d = 3.4e+38f;
                int c15_worst_pos = 60;
                int db_tile_start = split_idx * db_tiles_per_split;
                #pragma unroll 1
                for (int local_db_tile = 0; local_db_tile < db_tiles_per_split; local_db_tile++) {
                    int db_tile = db_tile_start + local_db_tile;
                    int db_start = db_tile * BLOCK_M;
                    int db_sq_idx = db_start + my_row;
                    if (my_row < BLOCK_M) {
                        if (db_sq_idx < M) {
                            smem_database_sq[my_row] = (float)database_sq[batch_idx * M + db_sq_idx];
                        } else {
                            smem_database_sq[my_row] = 3.4e+38f;
                        }
                    }
                    asm volatile("barrier.sync 8, 128;");
                    mbarrier_wait(score_full_addr, _phase_score_full_0);
                    _phase_score_full_0 ^= 1;
                    int cross_addr = taddr + (cta_rank * BLOCK_Q + tmem_row_base_v << 16);
                    float dots[64];
                    asm volatile(
                        "tcgen05.ld.sync.aligned.32x32b.x64.b32"
                        " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63}, [%64];"
                        : "=f"(dots[0]), "=f"(dots[1]), "=f"(dots[2]), "=f"(dots[3]), "=f"(dots[4]), "=f"(dots[5]), "=f"(dots[6]), "=f"(dots[7]), "=f"(dots[8]), "=f"(dots[9]), "=f"(dots[10]), "=f"(dots[11]), "=f"(dots[12]), "=f"(dots[13]), "=f"(dots[14]), "=f"(dots[15]), "=f"(dots[16]), "=f"(dots[17]), "=f"(dots[18]), "=f"(dots[19]), "=f"(dots[20]), "=f"(dots[21]), "=f"(dots[22]), "=f"(dots[23]), "=f"(dots[24]), "=f"(dots[25]), "=f"(dots[26]), "=f"(dots[27]), "=f"(dots[28]), "=f"(dots[29]), "=f"(dots[30]), "=f"(dots[31]), "=f"(dots[32]), "=f"(dots[33]), "=f"(dots[34]), "=f"(dots[35]), "=f"(dots[36]), "=f"(dots[37]), "=f"(dots[38]), "=f"(dots[39]), "=f"(dots[40]), "=f"(dots[41]), "=f"(dots[42]), "=f"(dots[43]), "=f"(dots[44]), "=f"(dots[45]), "=f"(dots[46]), "=f"(dots[47]), "=f"(dots[48]), "=f"(dots[49]), "=f"(dots[50]), "=f"(dots[51]), "=f"(dots[52]), "=f"(dots[53]), "=f"(dots[54]), "=f"(dots[55]), "=f"(dots[56]), "=f"(dots[57]), "=f"(dots[58]), "=f"(dots[59]), "=f"(dots[60]), "=f"(dots[61]), "=f"(dots[62]), "=f"(dots[63])
                        : "r"(cross_addr)
                        : "memory");
                    asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                    asm volatile("barrier.sync 8, 128;");
                    if (elect_sync()) {
                        asm volatile(
                            "mbarrier.arrive.release.cta.shared::cluster.b64 _, [%0];"
                            :: "r"((score_empty_addr) & 0xFEFFFFFF) : "memory");
                    }
                    if (valid_q != 0) {
                        #pragma unroll 1
                        for (int col_base = 0; col_base < 64; col_base += 4) {
                            float dist_vec[4];
                            dist_vec[0] = dots[col_base];
                            dist_vec[1] = dots[col_base + 1];
                            dist_vec[2] = dots[col_base + 2];
                            dist_vec[3] = dots[col_base + 3];
                            const float2 _fma_b2_0 = {-2.0f, -2.0f};
                            const float2 _fma_c2_1 = {q_sq_val, q_sq_val};
                            #pragma unroll
                            for (int _lf = 0; _lf < 2; _lf++)
                                fma_f32x2_inplace(&reinterpret_cast<float2*>(dist_vec)[_lf], _fma_b2_0, _fma_c2_1);
                            float db_sq_vec[4];
                            db_sq_vec[0] = smem_database_sq[col_base];
                            db_sq_vec[1] = smem_database_sq[col_base + 1];
                            db_sq_vec[2] = smem_database_sq[col_base + 2];
                            db_sq_vec[3] = smem_database_sq[col_base + 3];
                            float _t0[4];
                            #pragma unroll
                            for (int _la = 0; _la < 2; _la++)
                                reinterpret_cast<float2*>(_t0)[_la] = add_f32x2(reinterpret_cast<float2*>(dist_vec)[_la], reinterpret_cast<const float2*>(db_sq_vec)[_la]);
                            if (local_db_tile == 0) {
                                #pragma unroll
                                for (int vec_col = 0; vec_col < 4; vec_col++) {
                                    int db_idx = db_start + col_base + vec_col;
                                    int slot = col_base + vec_col;
                                    best_d[slot] = _t0[vec_col];
                                    best_i[slot] = db_idx;
                                }
                            } else {
                                float group_min = _t0[0];
                                if (_t0[1] < group_min) {
                                    group_min = _t0[1];
                                }
                                if (_t0[2] < group_min) {
                                    group_min = _t0[2];
                                }
                                if (_t0[3] < group_min) {
                                    group_min = _t0[3];
                                }
                                if (group_min < worst_d) {
                                    float sort_d0 = _t0[0];
                                    float sort_d1 = _t0[1];
                                    float sort_d2 = _t0[2];
                                    float sort_d3 = _t0[3];
                                    int sort_col0 = 0;
                                    int sort_col1 = 1;
                                    int sort_col2 = 2;
                                    int sort_col3 = 3;
                                    float tmp_d = 0.0f;
                                    int tmp_col = 0;
                                    if (sort_d1 < sort_d0) {
                                        tmp_d = sort_d0;
                                        sort_d0 = sort_d1;
                                        sort_d1 = tmp_d;
                                        tmp_col = sort_col0;
                                        sort_col0 = sort_col1;
                                        sort_col1 = tmp_col;
                                    }
                                    if (sort_d3 < sort_d2) {
                                        tmp_d = sort_d2;
                                        sort_d2 = sort_d3;
                                        sort_d3 = tmp_d;
                                        tmp_col = sort_col2;
                                        sort_col2 = sort_col3;
                                        sort_col3 = tmp_col;
                                    }
                                    if (sort_d2 < sort_d0) {
                                        tmp_d = sort_d0;
                                        sort_d0 = sort_d2;
                                        sort_d2 = tmp_d;
                                        tmp_col = sort_col0;
                                        sort_col0 = sort_col2;
                                        sort_col2 = tmp_col;
                                    }
                                    if (sort_d3 < sort_d1) {
                                        tmp_d = sort_d1;
                                        sort_d1 = sort_d3;
                                        sort_d3 = tmp_d;
                                        tmp_col = sort_col1;
                                        sort_col1 = sort_col3;
                                        sort_col3 = tmp_col;
                                    }
                                    if (sort_d2 < sort_d1) {
                                        tmp_d = sort_d1;
                                        sort_d1 = sort_d2;
                                        sort_d2 = tmp_d;
                                        tmp_col = sort_col1;
                                        sort_col1 = sort_col2;
                                        sort_col2 = tmp_col;
                                    }
                                    #pragma unroll
                                    for (int visit = 0; visit < 4; visit++) {
                                        int vec_col = sort_col0;
                                        float dist = sort_d0;
                                        if (visit == 1) {
                                            vec_col = sort_col1;
                                            dist = sort_d1;
                                        }
                                        if (visit == 2) {
                                            vec_col = sort_col2;
                                            dist = sort_d2;
                                        }
                                        if (visit == 3) {
                                            vec_col = sort_col3;
                                            dist = sort_d3;
                                        }
                                        if (dist >= worst_d) {
                                            break;
                                        }
                                        int db_idx = db_start + col_base + vec_col;
                                        best_d[worst_pos] = dist;
                                        best_i[worst_pos] = db_idx;
                                        int refresh_base = worst_chunk * 4;
                                        float refresh_worst_d = best_d[refresh_base];
                                        int refresh_worst_pos = refresh_base;
                                        #pragma unroll
                                        for (int offset = 1; offset < 4; offset++) {
                                            int scan_pos = refresh_base + offset;
                                            if (best_d[scan_pos] > refresh_worst_d) {
                                                refresh_worst_d = best_d[scan_pos];
                                                refresh_worst_pos = scan_pos;
                                            }
                                        }
                                        if (worst_chunk == 0) {
                                            c0_worst_d = refresh_worst_d;
                                            c0_worst_pos = refresh_worst_pos;
                                        }
                                        if (worst_chunk == 1) {
                                            c1_worst_d = refresh_worst_d;
                                            c1_worst_pos = refresh_worst_pos;
                                        }
                                        if (worst_chunk == 2) {
                                            c2_worst_d = refresh_worst_d;
                                            c2_worst_pos = refresh_worst_pos;
                                        }
                                        if (worst_chunk == 3) {
                                            c3_worst_d = refresh_worst_d;
                                            c3_worst_pos = refresh_worst_pos;
                                        }
                                        if (worst_chunk == 4) {
                                            c4_worst_d = refresh_worst_d;
                                            c4_worst_pos = refresh_worst_pos;
                                        }
                                        if (worst_chunk == 5) {
                                            c5_worst_d = refresh_worst_d;
                                            c5_worst_pos = refresh_worst_pos;
                                        }
                                        if (worst_chunk == 6) {
                                            c6_worst_d = refresh_worst_d;
                                            c6_worst_pos = refresh_worst_pos;
                                        }
                                        if (worst_chunk == 7) {
                                            c7_worst_d = refresh_worst_d;
                                            c7_worst_pos = refresh_worst_pos;
                                        }
                                        if (worst_chunk == 8) {
                                            c8_worst_d = refresh_worst_d;
                                            c8_worst_pos = refresh_worst_pos;
                                        }
                                        if (worst_chunk == 9) {
                                            c9_worst_d = refresh_worst_d;
                                            c9_worst_pos = refresh_worst_pos;
                                        }
                                        if (worst_chunk == 10) {
                                            c10_worst_d = refresh_worst_d;
                                            c10_worst_pos = refresh_worst_pos;
                                        }
                                        if (worst_chunk == 11) {
                                            c11_worst_d = refresh_worst_d;
                                            c11_worst_pos = refresh_worst_pos;
                                        }
                                        if (worst_chunk == 12) {
                                            c12_worst_d = refresh_worst_d;
                                            c12_worst_pos = refresh_worst_pos;
                                        }
                                        if (worst_chunk == 13) {
                                            c13_worst_d = refresh_worst_d;
                                            c13_worst_pos = refresh_worst_pos;
                                        }
                                        if (worst_chunk == 14) {
                                            c14_worst_d = refresh_worst_d;
                                            c14_worst_pos = refresh_worst_pos;
                                        }
                                        if (worst_chunk == 15) {
                                            c15_worst_d = refresh_worst_d;
                                            c15_worst_pos = refresh_worst_pos;
                                        }
                                        worst_d = c0_worst_d;
                                        worst_pos = c0_worst_pos;
                                        worst_chunk = 0;
                                        if (c1_worst_d > worst_d) {
                                            worst_d = c1_worst_d;
                                            worst_pos = c1_worst_pos;
                                            worst_chunk = 1;
                                        }
                                        if (c2_worst_d > worst_d) {
                                            worst_d = c2_worst_d;
                                            worst_pos = c2_worst_pos;
                                            worst_chunk = 2;
                                        }
                                        if (c3_worst_d > worst_d) {
                                            worst_d = c3_worst_d;
                                            worst_pos = c3_worst_pos;
                                            worst_chunk = 3;
                                        }
                                        if (c4_worst_d > worst_d) {
                                            worst_d = c4_worst_d;
                                            worst_pos = c4_worst_pos;
                                            worst_chunk = 4;
                                        }
                                        if (c5_worst_d > worst_d) {
                                            worst_d = c5_worst_d;
                                            worst_pos = c5_worst_pos;
                                            worst_chunk = 5;
                                        }
                                        if (c6_worst_d > worst_d) {
                                            worst_d = c6_worst_d;
                                            worst_pos = c6_worst_pos;
                                            worst_chunk = 6;
                                        }
                                        if (c7_worst_d > worst_d) {
                                            worst_d = c7_worst_d;
                                            worst_pos = c7_worst_pos;
                                            worst_chunk = 7;
                                        }
                                        if (c8_worst_d > worst_d) {
                                            worst_d = c8_worst_d;
                                            worst_pos = c8_worst_pos;
                                            worst_chunk = 8;
                                        }
                                        if (c9_worst_d > worst_d) {
                                            worst_d = c9_worst_d;
                                            worst_pos = c9_worst_pos;
                                            worst_chunk = 9;
                                        }
                                        if (c10_worst_d > worst_d) {
                                            worst_d = c10_worst_d;
                                            worst_pos = c10_worst_pos;
                                            worst_chunk = 10;
                                        }
                                        if (c11_worst_d > worst_d) {
                                            worst_d = c11_worst_d;
                                            worst_pos = c11_worst_pos;
                                            worst_chunk = 11;
                                        }
                                        if (c12_worst_d > worst_d) {
                                            worst_d = c12_worst_d;
                                            worst_pos = c12_worst_pos;
                                            worst_chunk = 12;
                                        }
                                        if (c13_worst_d > worst_d) {
                                            worst_d = c13_worst_d;
                                            worst_pos = c13_worst_pos;
                                            worst_chunk = 13;
                                        }
                                        if (c14_worst_d > worst_d) {
                                            worst_d = c14_worst_d;
                                            worst_pos = c14_worst_pos;
                                            worst_chunk = 14;
                                        }
                                        if (c15_worst_d > worst_d) {
                                            worst_d = c15_worst_d;
                                            worst_pos = c15_worst_pos;
                                            worst_chunk = 15;
                                        }
                                    }
                                }
                            }
                        }
                        if (local_db_tile == 0) {
                            c0_worst_d = best_d[0];
                            c0_worst_pos = 0;
                            #pragma unroll
                            for (int offset = 1; offset < 4; offset++) {
                                if (best_d[offset] > c0_worst_d) {
                                    c0_worst_d = best_d[offset];
                                    c0_worst_pos = offset;
                                }
                            }
                            c1_worst_d = best_d[4];
                            c1_worst_pos = 4;
                            #pragma unroll
                            for (int offset = 1; offset < 4; offset++) {
                                int scan_pos = 4 + offset;
                                if (best_d[scan_pos] > c1_worst_d) {
                                    c1_worst_d = best_d[scan_pos];
                                    c1_worst_pos = scan_pos;
                                }
                            }
                            c2_worst_d = best_d[8];
                            c2_worst_pos = 8;
                            #pragma unroll
                            for (int offset = 1; offset < 4; offset++) {
                                int scan_pos = 8 + offset;
                                if (best_d[scan_pos] > c2_worst_d) {
                                    c2_worst_d = best_d[scan_pos];
                                    c2_worst_pos = scan_pos;
                                }
                            }
                            c3_worst_d = best_d[12];
                            c3_worst_pos = 12;
                            #pragma unroll
                            for (int offset = 1; offset < 4; offset++) {
                                int scan_pos = 12 + offset;
                                if (best_d[scan_pos] > c3_worst_d) {
                                    c3_worst_d = best_d[scan_pos];
                                    c3_worst_pos = scan_pos;
                                }
                            }
                            c4_worst_d = best_d[16];
                            c4_worst_pos = 16;
                            #pragma unroll
                            for (int offset = 1; offset < 4; offset++) {
                                int scan_pos = 16 + offset;
                                if (best_d[scan_pos] > c4_worst_d) {
                                    c4_worst_d = best_d[scan_pos];
                                    c4_worst_pos = scan_pos;
                                }
                            }
                            c5_worst_d = best_d[20];
                            c5_worst_pos = 20;
                            #pragma unroll
                            for (int offset = 1; offset < 4; offset++) {
                                int scan_pos = 20 + offset;
                                if (best_d[scan_pos] > c5_worst_d) {
                                    c5_worst_d = best_d[scan_pos];
                                    c5_worst_pos = scan_pos;
                                }
                            }
                            c6_worst_d = best_d[24];
                            c6_worst_pos = 24;
                            #pragma unroll
                            for (int offset = 1; offset < 4; offset++) {
                                int scan_pos = 24 + offset;
                                if (best_d[scan_pos] > c6_worst_d) {
                                    c6_worst_d = best_d[scan_pos];
                                    c6_worst_pos = scan_pos;
                                }
                            }
                            c7_worst_d = best_d[28];
                            c7_worst_pos = 28;
                            #pragma unroll
                            for (int offset = 1; offset < 4; offset++) {
                                int scan_pos = 28 + offset;
                                if (best_d[scan_pos] > c7_worst_d) {
                                    c7_worst_d = best_d[scan_pos];
                                    c7_worst_pos = scan_pos;
                                }
                            }
                            c8_worst_d = best_d[32];
                            c8_worst_pos = 32;
                            #pragma unroll
                            for (int offset = 1; offset < 4; offset++) {
                                int scan_pos = 32 + offset;
                                if (best_d[scan_pos] > c8_worst_d) {
                                    c8_worst_d = best_d[scan_pos];
                                    c8_worst_pos = scan_pos;
                                }
                            }
                            c9_worst_d = best_d[36];
                            c9_worst_pos = 36;
                            #pragma unroll
                            for (int offset = 1; offset < 4; offset++) {
                                int scan_pos = 36 + offset;
                                if (best_d[scan_pos] > c9_worst_d) {
                                    c9_worst_d = best_d[scan_pos];
                                    c9_worst_pos = scan_pos;
                                }
                            }
                            c10_worst_d = best_d[40];
                            c10_worst_pos = 40;
                            #pragma unroll
                            for (int offset = 1; offset < 4; offset++) {
                                int scan_pos = 40 + offset;
                                if (best_d[scan_pos] > c10_worst_d) {
                                    c10_worst_d = best_d[scan_pos];
                                    c10_worst_pos = scan_pos;
                                }
                            }
                            c11_worst_d = best_d[44];
                            c11_worst_pos = 44;
                            #pragma unroll
                            for (int offset = 1; offset < 4; offset++) {
                                int scan_pos = 44 + offset;
                                if (best_d[scan_pos] > c11_worst_d) {
                                    c11_worst_d = best_d[scan_pos];
                                    c11_worst_pos = scan_pos;
                                }
                            }
                            c12_worst_d = best_d[48];
                            c12_worst_pos = 48;
                            #pragma unroll
                            for (int offset = 1; offset < 4; offset++) {
                                int scan_pos = 48 + offset;
                                if (best_d[scan_pos] > c12_worst_d) {
                                    c12_worst_d = best_d[scan_pos];
                                    c12_worst_pos = scan_pos;
                                }
                            }
                            c13_worst_d = best_d[52];
                            c13_worst_pos = 52;
                            #pragma unroll
                            for (int offset = 1; offset < 4; offset++) {
                                int scan_pos = 52 + offset;
                                if (best_d[scan_pos] > c13_worst_d) {
                                    c13_worst_d = best_d[scan_pos];
                                    c13_worst_pos = scan_pos;
                                }
                            }
                            c14_worst_d = best_d[56];
                            c14_worst_pos = 56;
                            #pragma unroll
                            for (int offset = 1; offset < 4; offset++) {
                                int scan_pos = 56 + offset;
                                if (best_d[scan_pos] > c14_worst_d) {
                                    c14_worst_d = best_d[scan_pos];
                                    c14_worst_pos = scan_pos;
                                }
                            }
                            c15_worst_d = best_d[60];
                            c15_worst_pos = 60;
                            #pragma unroll
                            for (int offset = 1; offset < 4; offset++) {
                                int scan_pos = 60 + offset;
                                if (best_d[scan_pos] > c15_worst_d) {
                                    c15_worst_d = best_d[scan_pos];
                                    c15_worst_pos = scan_pos;
                                }
                            }
                            worst_d = c0_worst_d;
                            worst_pos = c0_worst_pos;
                            worst_chunk = 0;
                            if (c1_worst_d > worst_d) {
                                worst_d = c1_worst_d;
                                worst_pos = c1_worst_pos;
                                worst_chunk = 1;
                            }
                            if (c2_worst_d > worst_d) {
                                worst_d = c2_worst_d;
                                worst_pos = c2_worst_pos;
                                worst_chunk = 2;
                            }
                            if (c3_worst_d > worst_d) {
                                worst_d = c3_worst_d;
                                worst_pos = c3_worst_pos;
                                worst_chunk = 3;
                            }
                            if (c4_worst_d > worst_d) {
                                worst_d = c4_worst_d;
                                worst_pos = c4_worst_pos;
                                worst_chunk = 4;
                            }
                            if (c5_worst_d > worst_d) {
                                worst_d = c5_worst_d;
                                worst_pos = c5_worst_pos;
                                worst_chunk = 5;
                            }
                            if (c6_worst_d > worst_d) {
                                worst_d = c6_worst_d;
                                worst_pos = c6_worst_pos;
                                worst_chunk = 6;
                            }
                            if (c7_worst_d > worst_d) {
                                worst_d = c7_worst_d;
                                worst_pos = c7_worst_pos;
                                worst_chunk = 7;
                            }
                            if (c8_worst_d > worst_d) {
                                worst_d = c8_worst_d;
                                worst_pos = c8_worst_pos;
                                worst_chunk = 8;
                            }
                            if (c9_worst_d > worst_d) {
                                worst_d = c9_worst_d;
                                worst_pos = c9_worst_pos;
                                worst_chunk = 9;
                            }
                            if (c10_worst_d > worst_d) {
                                worst_d = c10_worst_d;
                                worst_pos = c10_worst_pos;
                                worst_chunk = 10;
                            }
                            if (c11_worst_d > worst_d) {
                                worst_d = c11_worst_d;
                                worst_pos = c11_worst_pos;
                                worst_chunk = 11;
                            }
                            if (c12_worst_d > worst_d) {
                                worst_d = c12_worst_d;
                                worst_pos = c12_worst_pos;
                                worst_chunk = 12;
                            }
                            if (c13_worst_d > worst_d) {
                                worst_d = c13_worst_d;
                                worst_pos = c13_worst_pos;
                                worst_chunk = 13;
                            }
                            if (c14_worst_d > worst_d) {
                                worst_d = c14_worst_d;
                                worst_pos = c14_worst_pos;
                                worst_chunk = 14;
                            }
                            if (c15_worst_d > worst_d) {
                                worst_d = c15_worst_d;
                                worst_pos = c15_worst_pos;
                                worst_chunk = 15;
                            }
                        }
                    }
                    asm volatile("barrier.sync 8, 128;");
                }
                if (valid_q != 0) {
                    int out_base = ((split_idx * B + batch_idx) * Q + q_idx) * K;
                    #pragma unroll
                    for (int out_k = 0; out_k < 64; out_k++) {
                        *((float*)(partial_dists + out_base + out_k)) = best_d[out_k];
                        *((int*)(partial_indices + out_base + out_k)) = best_i[out_k];
                    }
                }
            }
        }
    // ---- Role: load ----
    } else if (warp == 4) {
        { // load_main
            uint32_t _phase_query_empty_0 = 1;
            uint32_t _phase_database_empty_0 = 1;
            if (warp_id == 4) {
                if (elect_sync()) {
                    #pragma unroll 1
                    for (unsigned int work_idx = cluster_id; work_idx < total_work; work_idx += num_clusters) {
                        int split_idx = work_idx % split_count;
                        int query_work = work_idx / split_count;
                        int batch_idx = query_work / num_q_tile_pairs;
                        int q_tile_pair = query_work % num_q_tile_pairs;
                        int q_tile = q_tile_pair * 2 + cta_rank;
                        int off_q = q_tile * BLOCK_Q;
                        int global_q = batch_idx * Q + off_q;
                        int db_tile_start = split_idx * db_tiles_per_split;
                        mbarrier_wait(query_empty_addr, _phase_query_empty_0);
                        _phase_query_empty_0 ^= 1;
                        asm volatile(
                            "mbarrier.arrive.expect_tx.release.cta.shared::cluster.b64 _, [%0], %1;"
                            :: "r"((query_full_addr) & 0xFEFFFFFF), "r"((uint32_t)(32768)) : "memory");
                        asm volatile(
                            "cp.async.bulk.tensor.3d.shared::cluster.global.mbarrier::complete_tx::bytes.cta_group::2"
                            " [%0], [%1, {%2, %3, %4}], [%5];"
                            :: "r"(smem_query_addr), "l"(tmap_query), "r"(0), "r"(global_q), "r"(0),
                               "r"(((query_full_addr) & 0xFEFFFFFF)) : "memory");
                        #pragma unroll 1
                        for (int local_db_tile = 0; local_db_tile < db_tiles_per_split; local_db_tile++) {
                            int db_tile = db_tile_start + local_db_tile;
                            int off_m = db_tile * BLOCK_M;
                            int global_m = batch_idx * M + off_m;
                            mbarrier_wait(database_empty_addr, _phase_database_empty_0);
                            _phase_database_empty_0 ^= 1;
                            asm volatile(
                                "mbarrier.arrive.expect_tx.release.cta.shared::cluster.b64 _, [%0], %1;"
                                :: "r"((database_full_addr) & 0xFEFFFFFF), "r"((uint32_t)(16384)) : "memory");
                            asm volatile(
                                "cp.async.bulk.tensor.3d.shared::cluster.global.mbarrier::complete_tx::bytes.cta_group::2"
                                " [%0], [%1, {%2, %3, %4}], [%5];"
                                :: "r"(smem_database_addr), "l"(tmap_database), "r"(0), "r"(global_m), "r"(0),
                                   "r"(((database_full_addr) & 0xFEFFFFFF)) : "memory");
                        }
                    }
                }
            }
        }
    // ---- Role: mma ----
    } else if (warp == 5) {
        { // mma_main
            uint32_t _phase_query_full_0 = 0;
            uint32_t _phase_score_empty_0 = 1;
            uint32_t _phase_database_full_0 = 0;
            if (cta_rank == 0) {
                #pragma unroll 1
                for (unsigned int work_idx = cluster_id; work_idx < total_work; work_idx += num_clusters) {
                    mbarrier_wait(query_full_addr, _phase_query_full_0);
                    _phase_query_full_0 ^= 1;
                    #pragma unroll 1
                    for (int _local_db_tile = 0; _local_db_tile < db_tiles_per_split; _local_db_tile++) {
                        mbarrier_wait(score_empty_addr, _phase_score_empty_0);
                        _phase_score_empty_0 ^= 1;
                        mbarrier_wait(database_full_addr, _phase_database_full_0);
                        _phase_database_full_0 ^= 1;
                        asm volatile("tcgen05.fence::after_thread_sync;");
                        int _mma_ss_a_lo_0 = (smem_query_addr >> 4) & 0x3FFF;
                        int _mma_ss_b_lo_0 = (smem_database_addr >> 4) & 0x3FFF;
                        asm volatile(
                    "{\n\t"
                    ".reg .pred leader, p0, p1;\n\t"
                    ".reg .b32 adhi, bdhi, alo, blo, id, m0, m1, m2, m3, m4, m5, m6, m7;\n\t"
                    ".reg .b64 da, db;\n\t"
                    "elect.sync _|leader, 0xFFFFFFFF;\n\t"
                    "setp.ne.b32 p0, %3, 0;\n\t"
                    "setp.ne.b32 p1, 1, 0;\n\t"
                    "mov.b32 m0, 0; mov.b32 m1, 0; mov.b32 m2, 0; mov.b32 m3, 0;\n\tmov.b32 m4, 0; mov.b32 m5, 0; mov.b32 m6, 0; mov.b32 m7, 0;\n\t"
                    "mov.b32 adhi, 0x40004040;\n\t"
                    "mov.b32 bdhi, 0x40004040;\n\t"
                    "mov.b32 id, 270533776;\n\t"
                    "mov.b32 alo, %0;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p0;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "add.u32 alo, alo, 1018;\n\t"
                    "add.u32 blo, blo, 506;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "}\n"
                    :: "r"(_mma_ss_a_lo_0), "r"(_mma_ss_b_lo_0), "r"(taddr), "r"(0));
                        elect_commit_cg2_multicast(score_full_addr, (uint16_t)(3));
                        elect_commit_cg2_multicast(database_empty_addr, (uint16_t)(3));
                    }
                    elect_commit_cg2_multicast(query_empty_addr, (uint16_t)(3));
                }
            }
        }
    }

    // Cleanup
    asm volatile("barrier.cluster.arrive.release.aligned;");
    asm volatile("barrier.cluster.wait.acquire.aligned;");

    if (warp == 5) {
        asm volatile("tcgen05.dealloc.cta_group::2.sync.aligned.b32 %0, %1;" :: "r"(tmem_addr_storage[0]), "r"(64));
        asm volatile("tcgen05.relinquish_alloc_permit.cta_group::2.sync.aligned;");
    }
}

} // extern "C"

#undef BLOCK_M
#undef BLOCK_Q
#undef FEAT_D
#undef NUM_MAIN_STAGES
#undef SMEM_SMEM_DATABASE_OFF
#undef SMEM_SMEM_DATABASE_SQ_OFF
#undef SMEM_SMEM_DATABASE_SQ_STAGE_BYTES
#undef SMEM_SMEM_DATABASE_SQ_STRIDE
#undef SMEM_SMEM_DATABASE_STAGE_BYTES
#undef SMEM_SMEM_DATABASE_STRIDE
#undef SMEM_SMEM_QUERY_OFF
#undef SMEM_SMEM_QUERY_STAGE_BYTES
#undef SMEM_SMEM_QUERY_STRIDE
#undef SMEM_TOTAL
#undef THREADS
#undef TMEM_CROSS_OFFSET
#undef TMEM_NCOLS
#undef TOP_K_MAX
#undef database_empty_addr
#undef database_full_addr
#undef query_empty_addr
#undef query_full_addr
#undef score_empty_addr
#undef score_full_addr
#undef smem_database_addr
#undef smem_database_sq_addr
#undef smem_query_addr

#define NUM_MAIN_STAGES 1
#define THREADS 128
#define TOP_K_MAX 64
#define SPLIT_COUNT 8

extern "C" {

__global__ __launch_bounds__(128, 1) void
kernel_knn_build_k64_merge_s8_unordered_warp_select_k64over32s8warpselect(float* __restrict__ partial_dists, int32_t* __restrict__ partial_indices, float* __restrict__ out_dists, int32_t* __restrict__ out_indices, int total_queries)
{
    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;


    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    const int warp_id = warp;
    const int lane_id = lane;

    // === Task calls (dependency order) ===
    int row = bid * 4 + warp;
    int base_row = row * TOP_K_MAX;
    int split_stride = total_queries * TOP_K_MAX;
    int cand_lo = lane;
    int cand_hi = lane + 32;
    if (row < total_queries) {
        float cand_d[16];
        int cand_i[16];
        #pragma unroll
        for (int split_idx = 0; split_idx < 8; split_idx++) {
            int split_base = base_row + split_idx * split_stride;
            cand_d[split_idx * 2] = (float)partial_dists[split_base + cand_lo];
            cand_i[split_idx * 2] = partial_indices[split_base + cand_lo];
            cand_d[split_idx * 2 + 1] = (float)partial_dists[split_base + cand_hi];
            cand_i[split_idx * 2 + 1] = partial_indices[split_base + cand_hi];
        }
        #pragma unroll
        for (int out_k = 0; out_k < 64; out_k++) {
            float winner_d = cand_d[0];
            int winner_i = cand_i[0];
            int winner_slot = 0;
            #pragma unroll
            for (int slot = 1; slot < 16; slot++) {
                if (cand_d[slot] < winner_d) {
                    winner_d = cand_d[slot];
                    winner_i = cand_i[slot];
                    winner_slot = slot;
                }
            }
            float warp_min = winner_d;
            #pragma unroll
            for (int offset = 16; offset > 0; offset >>= 1)
                warp_min = fminf(warp_min, __shfl_xor_sync(0xFFFFFFFF, warp_min, offset));
            int _vote_0 = __ballot_sync(0xFFFFFFFF, winner_d == warp_min);
            int owner_ballot = _vote_0;
            int _ffs_0 = __ffs(owner_ballot);
            int winner_lane = _ffs_0 - 1;
            winner_i = __shfl_sync(0xFFFFFFFF, winner_i, winner_lane);
            winner_slot = __shfl_sync(0xFFFFFFFF, winner_slot, winner_lane);
            if (lane == 0) {
                *((float*)(out_dists + base_row + out_k)) = warp_min;
                *((int*)(out_indices + base_row + out_k)) = winner_i;
            }
            if (lane == winner_lane) {
                #pragma unroll
                for (int slot = 0; slot < 16; slot++) {
                    if (winner_slot == slot) {
                        cand_d[slot] = 3.4e+38f;
                    }
                }
            }
        }
    }
}

} // extern "C"

#undef NUM_MAIN_STAGES
#undef SPLIT_COUNT
#undef THREADS
#undef TOP_K_MAX

#define TMEM_NCOLS 64
#define TMEM_CROSS_OFFSET 0
#define NUM_MAIN_STAGES 1
#define SMEM_SMEM_QUERY_OFF 1024
#define SMEM_SMEM_QUERY_STAGE_BYTES 32768
#define SMEM_SMEM_QUERY_STRIDE 32768
#define SMEM_SMEM_DATABASE_OFF 33792
#define SMEM_SMEM_DATABASE_STAGE_BYTES 16384
#define SMEM_SMEM_DATABASE_STRIDE 16384
#define SMEM_SMEM_DATABASE_SQ_OFF 50176
#define SMEM_SMEM_DATABASE_SQ_STAGE_BYTES 256
#define SMEM_SMEM_DATABASE_SQ_STRIDE 256
#define SMEM_TOTAL 50432
#define THREADS 192
#define BLOCK_Q 128
#define BLOCK_M 64
#define FEAT_D 128
#define TOP_K_MAX 24

extern "C" {

__global__ __launch_bounds__(192, 1) void
kernel_knn_build_evolve_7bfc_split_cg2_u2_stage1_bad5midks8k24(float* __restrict__ query_sq, float* __restrict__ database_sq, float* __restrict__ partial_dists, int32_t* __restrict__ partial_indices, const void* tmap_query, const void* tmap_database, int B, int Q, int M, int K, int num_q_tile_pairs, int db_tiles_per_split, int split_count, int total_work)
{
    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;

    extern __shared__ __align__(1024) char smem_raw[];
    int smem;
    smem = (int)(unsigned long long)__cvta_generic_to_shared(smem_raw);
    const int smem_smem_query = smem + 1024;
    const int smem_smem_database = smem + 33792;
    const int smem_smem_database_sq = smem + 50176;

    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;
    const unsigned int clusters_x = gridDim.x / 2;
    const unsigned int cluster_id = ((blockIdx.z * gridDim.y + blockIdx.y) * clusters_x) + blockIdx.x / 2;
    const unsigned int num_clusters = clusters_x * gridDim.y * gridDim.z;

    int cta_rank;
    asm volatile("mov.b32 %0, %%cluster_ctarank;" : "=r"(cta_rank));

    // Mbarrier init (6 groups, 6 barriers)
    // Mbarriers at smem_raw[0..48)

    if (warp == 0) {
        uint32_t leader = elect_sync();
        // query_full: 1 barriers, init_count=2
        mbarrier_init_pred(smem + 0, 2, leader);
        // query_empty: 1 barriers, init_count=1
        mbarrier_init_pred(smem + 8, 1, leader);
        // database_full: 1 barriers, init_count=2
        mbarrier_init_pred(smem + 16, 2, leader);
        // database_empty: 1 barriers, init_count=1
        mbarrier_init_pred(smem + 24, 1, leader);
        // score_full: 1 barriers, init_count=1
        mbarrier_init_pred(smem + 32, 1, leader);
        // score_empty: 1 barriers, init_count=8
        mbarrier_init_pred(smem + 40, 8, leader);
        asm volatile("fence.mbarrier_init.release.cluster;");
    }

    __syncthreads();

    // TMEM alloc (64 columns, 64 used)
    volatile int* tmem_addr_storage = (volatile int*)(smem_raw + 48);
    if (warp == 1) {
        int _tmem_hold = smem + 48;
        asm volatile("tcgen05.alloc.cta_group::2.sync.aligned.shared::cta.b32 [%0], %1;" :: "r"(_tmem_hold), "r"(64) : "memory");
    }

    asm volatile("barrier.cluster.arrive.release.aligned;");
    asm volatile("barrier.cluster.wait.acquire.aligned;");
    asm volatile("tcgen05.fence::after_thread_sync;");

    const int warp_id = warp;
    const int lane_id = lane;
    __nv_bfloat16* smem_query = (__nv_bfloat16*)(smem_raw + 1024);
    #define smem_query_addr (smem + 1024)
    __nv_bfloat16* smem_database = (__nv_bfloat16*)(smem_raw + 33792);
    #define smem_database_addr (smem + 33792)
    float* smem_database_sq = (float*)(smem_raw + 50176);
    #define smem_database_sq_addr (smem + 50176)
    const int mbar_base = smem;
    #define query_full_addr (mbar_base + 0)
    #define query_empty_addr (mbar_base + 8)
    #define database_full_addr (mbar_base + 16)
    #define database_empty_addr (mbar_base + 24)
    #define score_full_addr (mbar_base + 32)
    #define score_empty_addr (mbar_base + 40)
    const int taddr = tmem_addr_storage[0];

    // ---- Role: load ----
    if (warp == 0) {
        { // load_main
            uint32_t _phase_query_empty_0 = 1;
            uint32_t _phase_database_empty_0 = 1;
            if (warp_id == 0) {
                if (elect_sync()) {
                    #pragma unroll 1
                    for (unsigned int work_idx = cluster_id; work_idx < total_work; work_idx += num_clusters) {
                        int split_idx = work_idx % split_count;
                        int query_work = work_idx / split_count;
                        int batch_idx = query_work / num_q_tile_pairs;
                        int q_tile_pair = query_work % num_q_tile_pairs;
                        int q_tile = q_tile_pair * 2 + cta_rank;
                        int off_q = q_tile * BLOCK_Q;
                        int global_q = batch_idx * Q + off_q;
                        int db_tile_start = split_idx * db_tiles_per_split;
                        mbarrier_wait(query_empty_addr, _phase_query_empty_0);
                        _phase_query_empty_0 ^= 1;
                        asm volatile(
                            "mbarrier.arrive.expect_tx.release.cta.shared::cluster.b64 _, [%0], %1;"
                            :: "r"((query_full_addr) & 0xFEFFFFFF), "r"((uint32_t)(32768)) : "memory");
                        asm volatile(
                            "cp.async.bulk.tensor.3d.shared::cluster.global.mbarrier::complete_tx::bytes.cta_group::2"
                            " [%0], [%1, {%2, %3, %4}], [%5];"
                            :: "r"(smem_query_addr), "l"(tmap_query), "r"(0), "r"(global_q), "r"(0),
                               "r"(((query_full_addr) & 0xFEFFFFFF)) : "memory");
                        #pragma unroll 1
                        for (int local_db_tile = 0; local_db_tile < db_tiles_per_split; local_db_tile++) {
                            int db_tile = db_tile_start + local_db_tile;
                            int off_m = db_tile * BLOCK_M;
                            int global_m = batch_idx * M + off_m;
                            mbarrier_wait(database_empty_addr, _phase_database_empty_0);
                            _phase_database_empty_0 ^= 1;
                            asm volatile(
                                "mbarrier.arrive.expect_tx.release.cta.shared::cluster.b64 _, [%0], %1;"
                                :: "r"((database_full_addr) & 0xFEFFFFFF), "r"((uint32_t)(16384)) : "memory");
                            asm volatile(
                                "cp.async.bulk.tensor.3d.shared::cluster.global.mbarrier::complete_tx::bytes.cta_group::2"
                                " [%0], [%1, {%2, %3, %4}], [%5];"
                                :: "r"(smem_database_addr), "l"(tmap_database), "r"(0), "r"(global_m), "r"(0),
                                   "r"(((database_full_addr) & 0xFEFFFFFF)) : "memory");
                        }
                    }
                }
            }
        }
    // ---- Role: mma ----
    } else if (warp == 1) {
        { // mma_main
            uint32_t _phase_query_full_0 = 0;
            uint32_t _phase_score_empty_0 = 1;
            uint32_t _phase_database_full_0 = 0;
            if (cta_rank == 0) {
                #pragma unroll 1
                for (unsigned int work_idx = cluster_id; work_idx < total_work; work_idx += num_clusters) {
                    mbarrier_wait(query_full_addr, _phase_query_full_0);
                    _phase_query_full_0 ^= 1;
                    #pragma unroll 1
                    for (int _local_db_tile = 0; _local_db_tile < db_tiles_per_split; _local_db_tile++) {
                        mbarrier_wait(score_empty_addr, _phase_score_empty_0);
                        _phase_score_empty_0 ^= 1;
                        mbarrier_wait(database_full_addr, _phase_database_full_0);
                        _phase_database_full_0 ^= 1;
                        asm volatile("tcgen05.fence::after_thread_sync;");
                        int _mma_ss_a_lo_0 = (smem_query_addr >> 4) & 0x3FFF;
                        int _mma_ss_b_lo_0 = (smem_database_addr >> 4) & 0x3FFF;
                        asm volatile(
                    "{\n\t"
                    ".reg .pred leader, p0, p1;\n\t"
                    ".reg .b32 adhi, bdhi, alo, blo, id, m0, m1, m2, m3, m4, m5, m6, m7;\n\t"
                    ".reg .b64 da, db;\n\t"
                    "elect.sync _|leader, 0xFFFFFFFF;\n\t"
                    "setp.ne.b32 p0, %3, 0;\n\t"
                    "setp.ne.b32 p1, 1, 0;\n\t"
                    "mov.b32 m0, 0; mov.b32 m1, 0; mov.b32 m2, 0; mov.b32 m3, 0;\n\tmov.b32 m4, 0; mov.b32 m5, 0; mov.b32 m6, 0; mov.b32 m7, 0;\n\t"
                    "mov.b32 adhi, 0x40004040;\n\t"
                    "mov.b32 bdhi, 0x40004040;\n\t"
                    "mov.b32 id, 270533776;\n\t"
                    "mov.b32 alo, %0;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p0;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "add.u32 alo, alo, 1018;\n\t"
                    "add.u32 blo, blo, 506;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "}\n"
                    :: "r"(_mma_ss_a_lo_0), "r"(_mma_ss_b_lo_0), "r"(taddr), "r"(0));
                        elect_commit_cg2_multicast(score_full_addr, (uint16_t)(3));
                        elect_commit_cg2_multicast(database_empty_addr, (uint16_t)(3));
                    }
                    elect_commit_cg2_multicast(query_empty_addr, (uint16_t)(3));
                }
            }
        }
    // ---- Role: compute ----
    } else if (warp >= 2 && warp <= 5) {
        const int warp_id_in_wg = warp % 4;
        const int my_row = warp_id_in_wg * 32 + lane;
        const int tmem_row_base_v = (warp_id_in_wg * 32) << 16;
        const int tmem_cross = taddr + TMEM_CROSS_OFFSET;
        { // compute_main
            uint32_t _phase_score_full_0 = 0;
            #pragma unroll 1
            for (unsigned int work_idx = cluster_id; work_idx < total_work; work_idx += num_clusters) {
                int split_idx = work_idx % split_count;
                int query_work = work_idx / split_count;
                int batch_idx = query_work / num_q_tile_pairs;
                int q_tile_pair = query_work % num_q_tile_pairs;
                int q_tile = q_tile_pair * 2 + cta_rank;
                int off_q = q_tile * BLOCK_Q;
                int q_idx = off_q + my_row;
                int valid_q = ((q_idx < Q) ? 1 : 0);
                float q_sq_val = 0.0f;
                if (valid_q != 0) {
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
                    int db_start = db_tile * BLOCK_M;
                    int db_sq_idx = db_start + my_row;
                    if (my_row < BLOCK_M) {
                        if (db_sq_idx < M) {
                            smem_database_sq[my_row] = (float)database_sq[batch_idx * M + db_sq_idx];
                        } else {
                            smem_database_sq[my_row] = 0.0f;
                        }
                    }
                    asm volatile("barrier.sync 8, 128;");
                    mbarrier_wait(score_full_addr, _phase_score_full_0);
                    _phase_score_full_0 ^= 1;
                    int cross_addr = taddr + (cta_rank * BLOCK_Q + tmem_row_base_v << 16);
                    float dots[64];
                    asm volatile(
                        "tcgen05.ld.sync.aligned.32x32b.x64.b32"
                        " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63}, [%64];"
                        : "=f"(dots[0]), "=f"(dots[1]), "=f"(dots[2]), "=f"(dots[3]), "=f"(dots[4]), "=f"(dots[5]), "=f"(dots[6]), "=f"(dots[7]), "=f"(dots[8]), "=f"(dots[9]), "=f"(dots[10]), "=f"(dots[11]), "=f"(dots[12]), "=f"(dots[13]), "=f"(dots[14]), "=f"(dots[15]), "=f"(dots[16]), "=f"(dots[17]), "=f"(dots[18]), "=f"(dots[19]), "=f"(dots[20]), "=f"(dots[21]), "=f"(dots[22]), "=f"(dots[23]), "=f"(dots[24]), "=f"(dots[25]), "=f"(dots[26]), "=f"(dots[27]), "=f"(dots[28]), "=f"(dots[29]), "=f"(dots[30]), "=f"(dots[31]), "=f"(dots[32]), "=f"(dots[33]), "=f"(dots[34]), "=f"(dots[35]), "=f"(dots[36]), "=f"(dots[37]), "=f"(dots[38]), "=f"(dots[39]), "=f"(dots[40]), "=f"(dots[41]), "=f"(dots[42]), "=f"(dots[43]), "=f"(dots[44]), "=f"(dots[45]), "=f"(dots[46]), "=f"(dots[47]), "=f"(dots[48]), "=f"(dots[49]), "=f"(dots[50]), "=f"(dots[51]), "=f"(dots[52]), "=f"(dots[53]), "=f"(dots[54]), "=f"(dots[55]), "=f"(dots[56]), "=f"(dots[57]), "=f"(dots[58]), "=f"(dots[59]), "=f"(dots[60]), "=f"(dots[61]), "=f"(dots[62]), "=f"(dots[63])
                        : "r"(cross_addr)
                        : "memory");
                    asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                    asm volatile("barrier.sync 8, 128;");
                    if (elect_sync()) {
                        asm volatile(
                            "mbarrier.arrive.release.cta.shared::cluster.b64 _, [%0];"
                            :: "r"((score_empty_addr) & 0xFEFFFFFF) : "memory");
                    }
                    if (valid_q != 0) {
                        #pragma unroll 2
                        for (int col_base = 0; col_base < 64; col_base += 4) {
                            float dist_vec[4];
                            dist_vec[0] = dots[col_base];
                            dist_vec[1] = dots[col_base + 1];
                            dist_vec[2] = dots[col_base + 2];
                            dist_vec[3] = dots[col_base + 3];
                            const float2 _fma_b2_0 = {-2.0f, -2.0f};
                            const float2 _fma_c2_1 = {q_sq_val, q_sq_val};
                            #pragma unroll
                            for (int _lf = 0; _lf < 2; _lf++)
                                fma_f32x2_inplace(&reinterpret_cast<float2*>(dist_vec)[_lf], _fma_b2_0, _fma_c2_1);
                            float db_sq_vec[4];
                            db_sq_vec[0] = smem_database_sq[col_base];
                            db_sq_vec[1] = smem_database_sq[col_base + 1];
                            db_sq_vec[2] = smem_database_sq[col_base + 2];
                            db_sq_vec[3] = smem_database_sq[col_base + 3];
                            float _t0[4];
                            #pragma unroll
                            for (int _la = 0; _la < 2; _la++)
                                reinterpret_cast<float2*>(_t0)[_la] = add_f32x2(reinterpret_cast<float2*>(dist_vec)[_la], reinterpret_cast<const float2*>(db_sq_vec)[_la]);
                            float group_min = _t0[0];
                            if (_t0[1] < group_min) {
                                group_min = _t0[1];
                            }
                            if (_t0[2] < group_min) {
                                group_min = _t0[2];
                            }
                            if (_t0[3] < group_min) {
                                group_min = _t0[3];
                            }
                            if (group_min < best_d[TOP_K_MAX - 1]) {
                                #pragma unroll
                                for (int vec_col = 0; vec_col < 4; vec_col++) {
                                    int db_idx = db_start + col_base + vec_col;
                                    if (db_idx < M) {
                                        float dist = _t0[vec_col];
                                        if (dist < best_d[TOP_K_MAX - 1]) {
                                            best_d[TOP_K_MAX - 1] = dist;
                                            best_i[TOP_K_MAX - 1] = db_idx;
                                            #pragma unroll
                                            for (int pos = TOP_K_MAX - 1; pos >= 1; pos--) {
                                                if (best_d[pos] < best_d[pos - 1]) {
                                                    float tmp_d = best_d[pos - 1];
                                                    int tmp_i = best_i[pos - 1];
                                                    best_d[pos - 1] = best_d[pos];
                                                    best_i[pos - 1] = best_i[pos];
                                                    best_d[pos] = tmp_d;
                                                    best_i[pos] = tmp_i;
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                    asm volatile("barrier.sync 8, 128;");
                }
                if (valid_q != 0) {
                    int out_base = ((split_idx * B + batch_idx) * Q + q_idx) * K;
                    #pragma unroll
                    for (int out_k = 0; out_k < TOP_K_MAX; out_k++) {
                        if (out_k < K) {
                            *((float*)(partial_dists + out_base + out_k)) = best_d[out_k];
                            *((int*)(partial_indices + out_base + out_k)) = best_i[out_k];
                        }
                    }
                }
            }
        }
    }

    // Cleanup
    asm volatile("barrier.cluster.arrive.release.aligned;");
    asm volatile("barrier.cluster.wait.acquire.aligned;");

    if (warp == 1) {
        asm volatile("tcgen05.dealloc.cta_group::2.sync.aligned.b32 %0, %1;" :: "r"(tmem_addr_storage[0]), "r"(64));
        asm volatile("tcgen05.relinquish_alloc_permit.cta_group::2.sync.aligned;");
    }
}

} // extern "C"

#undef BLOCK_M
#undef BLOCK_Q
#undef FEAT_D
#undef NUM_MAIN_STAGES
#undef SMEM_SMEM_DATABASE_OFF
#undef SMEM_SMEM_DATABASE_SQ_OFF
#undef SMEM_SMEM_DATABASE_SQ_STAGE_BYTES
#undef SMEM_SMEM_DATABASE_SQ_STRIDE
#undef SMEM_SMEM_DATABASE_STAGE_BYTES
#undef SMEM_SMEM_DATABASE_STRIDE
#undef SMEM_SMEM_QUERY_OFF
#undef SMEM_SMEM_QUERY_STAGE_BYTES
#undef SMEM_SMEM_QUERY_STRIDE
#undef SMEM_TOTAL
#undef THREADS
#undef TMEM_CROSS_OFFSET
#undef TMEM_NCOLS
#undef TOP_K_MAX
#undef database_empty_addr
#undef database_full_addr
#undef query_empty_addr
#undef query_full_addr
#undef score_empty_addr
#undef score_full_addr
#undef smem_database_addr
#undef smem_database_sq_addr
#undef smem_query_addr

#define TMEM_NCOLS 64
#define TMEM_CROSS_OFFSET 0
#define NUM_MAIN_STAGES 1
#define SMEM_SMEM_QUERY_OFF 1024
#define SMEM_SMEM_QUERY_STAGE_BYTES 32768
#define SMEM_SMEM_QUERY_STRIDE 32768
#define SMEM_SMEM_DATABASE_OFF 33792
#define SMEM_SMEM_DATABASE_STAGE_BYTES 16384
#define SMEM_SMEM_DATABASE_STRIDE 16384
#define SMEM_SMEM_DATABASE_SQ_OFF 50176
#define SMEM_SMEM_DATABASE_SQ_STAGE_BYTES 256
#define SMEM_SMEM_DATABASE_SQ_STRIDE 256
#define SMEM_TOTAL 50432
#define THREADS 192
#define BLOCK_Q 128
#define BLOCK_M 64
#define FEAT_D 128
#define TOP_K_MAX 28

extern "C" {

__global__ __launch_bounds__(192, 1) void
kernel_knn_build_evolve_7bfc_split_cg2_u2_stage1_bad5midks8k28(float* __restrict__ query_sq, float* __restrict__ database_sq, float* __restrict__ partial_dists, int32_t* __restrict__ partial_indices, const void* tmap_query, const void* tmap_database, int B, int Q, int M, int K, int num_q_tile_pairs, int db_tiles_per_split, int split_count, int total_work)
{
    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;

    extern __shared__ __align__(1024) char smem_raw[];
    int smem;
    smem = (int)(unsigned long long)__cvta_generic_to_shared(smem_raw);
    const int smem_smem_query = smem + 1024;
    const int smem_smem_database = smem + 33792;
    const int smem_smem_database_sq = smem + 50176;

    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;
    const unsigned int clusters_x = gridDim.x / 2;
    const unsigned int cluster_id = ((blockIdx.z * gridDim.y + blockIdx.y) * clusters_x) + blockIdx.x / 2;
    const unsigned int num_clusters = clusters_x * gridDim.y * gridDim.z;

    int cta_rank;
    asm volatile("mov.b32 %0, %%cluster_ctarank;" : "=r"(cta_rank));

    // Mbarrier init (6 groups, 6 barriers)
    // Mbarriers at smem_raw[0..48)

    if (warp == 0) {
        uint32_t leader = elect_sync();
        // query_full: 1 barriers, init_count=2
        mbarrier_init_pred(smem + 0, 2, leader);
        // query_empty: 1 barriers, init_count=1
        mbarrier_init_pred(smem + 8, 1, leader);
        // database_full: 1 barriers, init_count=2
        mbarrier_init_pred(smem + 16, 2, leader);
        // database_empty: 1 barriers, init_count=1
        mbarrier_init_pred(smem + 24, 1, leader);
        // score_full: 1 barriers, init_count=1
        mbarrier_init_pred(smem + 32, 1, leader);
        // score_empty: 1 barriers, init_count=8
        mbarrier_init_pred(smem + 40, 8, leader);
        asm volatile("fence.mbarrier_init.release.cluster;");
    }

    __syncthreads();

    // TMEM alloc (64 columns, 64 used)
    volatile int* tmem_addr_storage = (volatile int*)(smem_raw + 48);
    if (warp == 1) {
        int _tmem_hold = smem + 48;
        asm volatile("tcgen05.alloc.cta_group::2.sync.aligned.shared::cta.b32 [%0], %1;" :: "r"(_tmem_hold), "r"(64) : "memory");
    }

    asm volatile("barrier.cluster.arrive.release.aligned;");
    asm volatile("barrier.cluster.wait.acquire.aligned;");
    asm volatile("tcgen05.fence::after_thread_sync;");

    const int warp_id = warp;
    const int lane_id = lane;
    __nv_bfloat16* smem_query = (__nv_bfloat16*)(smem_raw + 1024);
    #define smem_query_addr (smem + 1024)
    __nv_bfloat16* smem_database = (__nv_bfloat16*)(smem_raw + 33792);
    #define smem_database_addr (smem + 33792)
    float* smem_database_sq = (float*)(smem_raw + 50176);
    #define smem_database_sq_addr (smem + 50176)
    const int mbar_base = smem;
    #define query_full_addr (mbar_base + 0)
    #define query_empty_addr (mbar_base + 8)
    #define database_full_addr (mbar_base + 16)
    #define database_empty_addr (mbar_base + 24)
    #define score_full_addr (mbar_base + 32)
    #define score_empty_addr (mbar_base + 40)
    const int taddr = tmem_addr_storage[0];

    // ---- Role: load ----
    if (warp == 0) {
        { // load_main
            uint32_t _phase_query_empty_0 = 1;
            uint32_t _phase_database_empty_0 = 1;
            if (warp_id == 0) {
                if (elect_sync()) {
                    #pragma unroll 1
                    for (unsigned int work_idx = cluster_id; work_idx < total_work; work_idx += num_clusters) {
                        int split_idx = work_idx % split_count;
                        int query_work = work_idx / split_count;
                        int batch_idx = query_work / num_q_tile_pairs;
                        int q_tile_pair = query_work % num_q_tile_pairs;
                        int q_tile = q_tile_pair * 2 + cta_rank;
                        int off_q = q_tile * BLOCK_Q;
                        int global_q = batch_idx * Q + off_q;
                        int db_tile_start = split_idx * db_tiles_per_split;
                        mbarrier_wait(query_empty_addr, _phase_query_empty_0);
                        _phase_query_empty_0 ^= 1;
                        asm volatile(
                            "mbarrier.arrive.expect_tx.release.cta.shared::cluster.b64 _, [%0], %1;"
                            :: "r"((query_full_addr) & 0xFEFFFFFF), "r"((uint32_t)(32768)) : "memory");
                        asm volatile(
                            "cp.async.bulk.tensor.3d.shared::cluster.global.mbarrier::complete_tx::bytes.cta_group::2"
                            " [%0], [%1, {%2, %3, %4}], [%5];"
                            :: "r"(smem_query_addr), "l"(tmap_query), "r"(0), "r"(global_q), "r"(0),
                               "r"(((query_full_addr) & 0xFEFFFFFF)) : "memory");
                        #pragma unroll 1
                        for (int local_db_tile = 0; local_db_tile < db_tiles_per_split; local_db_tile++) {
                            int db_tile = db_tile_start + local_db_tile;
                            int off_m = db_tile * BLOCK_M;
                            int global_m = batch_idx * M + off_m;
                            mbarrier_wait(database_empty_addr, _phase_database_empty_0);
                            _phase_database_empty_0 ^= 1;
                            asm volatile(
                                "mbarrier.arrive.expect_tx.release.cta.shared::cluster.b64 _, [%0], %1;"
                                :: "r"((database_full_addr) & 0xFEFFFFFF), "r"((uint32_t)(16384)) : "memory");
                            asm volatile(
                                "cp.async.bulk.tensor.3d.shared::cluster.global.mbarrier::complete_tx::bytes.cta_group::2"
                                " [%0], [%1, {%2, %3, %4}], [%5];"
                                :: "r"(smem_database_addr), "l"(tmap_database), "r"(0), "r"(global_m), "r"(0),
                                   "r"(((database_full_addr) & 0xFEFFFFFF)) : "memory");
                        }
                    }
                }
            }
        }
    // ---- Role: mma ----
    } else if (warp == 1) {
        { // mma_main
            uint32_t _phase_query_full_0 = 0;
            uint32_t _phase_score_empty_0 = 1;
            uint32_t _phase_database_full_0 = 0;
            if (cta_rank == 0) {
                #pragma unroll 1
                for (unsigned int work_idx = cluster_id; work_idx < total_work; work_idx += num_clusters) {
                    mbarrier_wait(query_full_addr, _phase_query_full_0);
                    _phase_query_full_0 ^= 1;
                    #pragma unroll 1
                    for (int _local_db_tile = 0; _local_db_tile < db_tiles_per_split; _local_db_tile++) {
                        mbarrier_wait(score_empty_addr, _phase_score_empty_0);
                        _phase_score_empty_0 ^= 1;
                        mbarrier_wait(database_full_addr, _phase_database_full_0);
                        _phase_database_full_0 ^= 1;
                        asm volatile("tcgen05.fence::after_thread_sync;");
                        int _mma_ss_a_lo_0 = (smem_query_addr >> 4) & 0x3FFF;
                        int _mma_ss_b_lo_0 = (smem_database_addr >> 4) & 0x3FFF;
                        asm volatile(
                    "{\n\t"
                    ".reg .pred leader, p0, p1;\n\t"
                    ".reg .b32 adhi, bdhi, alo, blo, id, m0, m1, m2, m3, m4, m5, m6, m7;\n\t"
                    ".reg .b64 da, db;\n\t"
                    "elect.sync _|leader, 0xFFFFFFFF;\n\t"
                    "setp.ne.b32 p0, %3, 0;\n\t"
                    "setp.ne.b32 p1, 1, 0;\n\t"
                    "mov.b32 m0, 0; mov.b32 m1, 0; mov.b32 m2, 0; mov.b32 m3, 0;\n\tmov.b32 m4, 0; mov.b32 m5, 0; mov.b32 m6, 0; mov.b32 m7, 0;\n\t"
                    "mov.b32 adhi, 0x40004040;\n\t"
                    "mov.b32 bdhi, 0x40004040;\n\t"
                    "mov.b32 id, 270533776;\n\t"
                    "mov.b32 alo, %0;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p0;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "add.u32 alo, alo, 1018;\n\t"
                    "add.u32 blo, blo, 506;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "}\n"
                    :: "r"(_mma_ss_a_lo_0), "r"(_mma_ss_b_lo_0), "r"(taddr), "r"(0));
                        elect_commit_cg2_multicast(score_full_addr, (uint16_t)(3));
                        elect_commit_cg2_multicast(database_empty_addr, (uint16_t)(3));
                    }
                    elect_commit_cg2_multicast(query_empty_addr, (uint16_t)(3));
                }
            }
        }
    // ---- Role: compute ----
    } else if (warp >= 2 && warp <= 5) {
        const int warp_id_in_wg = warp % 4;
        const int my_row = warp_id_in_wg * 32 + lane;
        const int tmem_row_base_v = (warp_id_in_wg * 32) << 16;
        const int tmem_cross = taddr + TMEM_CROSS_OFFSET;
        { // compute_main
            uint32_t _phase_score_full_0 = 0;
            #pragma unroll 1
            for (unsigned int work_idx = cluster_id; work_idx < total_work; work_idx += num_clusters) {
                int split_idx = work_idx % split_count;
                int query_work = work_idx / split_count;
                int batch_idx = query_work / num_q_tile_pairs;
                int q_tile_pair = query_work % num_q_tile_pairs;
                int q_tile = q_tile_pair * 2 + cta_rank;
                int off_q = q_tile * BLOCK_Q;
                int q_idx = off_q + my_row;
                int valid_q = ((q_idx < Q) ? 1 : 0);
                float q_sq_val = 0.0f;
                if (valid_q != 0) {
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
                    int db_start = db_tile * BLOCK_M;
                    int db_sq_idx = db_start + my_row;
                    if (my_row < BLOCK_M) {
                        if (db_sq_idx < M) {
                            smem_database_sq[my_row] = (float)database_sq[batch_idx * M + db_sq_idx];
                        } else {
                            smem_database_sq[my_row] = 0.0f;
                        }
                    }
                    asm volatile("barrier.sync 8, 128;");
                    mbarrier_wait(score_full_addr, _phase_score_full_0);
                    _phase_score_full_0 ^= 1;
                    int cross_addr = taddr + (cta_rank * BLOCK_Q + tmem_row_base_v << 16);
                    float dots[64];
                    asm volatile(
                        "tcgen05.ld.sync.aligned.32x32b.x64.b32"
                        " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63}, [%64];"
                        : "=f"(dots[0]), "=f"(dots[1]), "=f"(dots[2]), "=f"(dots[3]), "=f"(dots[4]), "=f"(dots[5]), "=f"(dots[6]), "=f"(dots[7]), "=f"(dots[8]), "=f"(dots[9]), "=f"(dots[10]), "=f"(dots[11]), "=f"(dots[12]), "=f"(dots[13]), "=f"(dots[14]), "=f"(dots[15]), "=f"(dots[16]), "=f"(dots[17]), "=f"(dots[18]), "=f"(dots[19]), "=f"(dots[20]), "=f"(dots[21]), "=f"(dots[22]), "=f"(dots[23]), "=f"(dots[24]), "=f"(dots[25]), "=f"(dots[26]), "=f"(dots[27]), "=f"(dots[28]), "=f"(dots[29]), "=f"(dots[30]), "=f"(dots[31]), "=f"(dots[32]), "=f"(dots[33]), "=f"(dots[34]), "=f"(dots[35]), "=f"(dots[36]), "=f"(dots[37]), "=f"(dots[38]), "=f"(dots[39]), "=f"(dots[40]), "=f"(dots[41]), "=f"(dots[42]), "=f"(dots[43]), "=f"(dots[44]), "=f"(dots[45]), "=f"(dots[46]), "=f"(dots[47]), "=f"(dots[48]), "=f"(dots[49]), "=f"(dots[50]), "=f"(dots[51]), "=f"(dots[52]), "=f"(dots[53]), "=f"(dots[54]), "=f"(dots[55]), "=f"(dots[56]), "=f"(dots[57]), "=f"(dots[58]), "=f"(dots[59]), "=f"(dots[60]), "=f"(dots[61]), "=f"(dots[62]), "=f"(dots[63])
                        : "r"(cross_addr)
                        : "memory");
                    asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                    asm volatile("barrier.sync 8, 128;");
                    if (elect_sync()) {
                        asm volatile(
                            "mbarrier.arrive.release.cta.shared::cluster.b64 _, [%0];"
                            :: "r"((score_empty_addr) & 0xFEFFFFFF) : "memory");
                    }
                    if (valid_q != 0) {
                        #pragma unroll 2
                        for (int col_base = 0; col_base < 64; col_base += 4) {
                            float dist_vec[4];
                            dist_vec[0] = dots[col_base];
                            dist_vec[1] = dots[col_base + 1];
                            dist_vec[2] = dots[col_base + 2];
                            dist_vec[3] = dots[col_base + 3];
                            const float2 _fma_b2_0 = {-2.0f, -2.0f};
                            const float2 _fma_c2_1 = {q_sq_val, q_sq_val};
                            #pragma unroll
                            for (int _lf = 0; _lf < 2; _lf++)
                                fma_f32x2_inplace(&reinterpret_cast<float2*>(dist_vec)[_lf], _fma_b2_0, _fma_c2_1);
                            float db_sq_vec[4];
                            db_sq_vec[0] = smem_database_sq[col_base];
                            db_sq_vec[1] = smem_database_sq[col_base + 1];
                            db_sq_vec[2] = smem_database_sq[col_base + 2];
                            db_sq_vec[3] = smem_database_sq[col_base + 3];
                            float _t0[4];
                            #pragma unroll
                            for (int _la = 0; _la < 2; _la++)
                                reinterpret_cast<float2*>(_t0)[_la] = add_f32x2(reinterpret_cast<float2*>(dist_vec)[_la], reinterpret_cast<const float2*>(db_sq_vec)[_la]);
                            float group_min = _t0[0];
                            if (_t0[1] < group_min) {
                                group_min = _t0[1];
                            }
                            if (_t0[2] < group_min) {
                                group_min = _t0[2];
                            }
                            if (_t0[3] < group_min) {
                                group_min = _t0[3];
                            }
                            if (group_min < best_d[TOP_K_MAX - 1]) {
                                #pragma unroll
                                for (int vec_col = 0; vec_col < 4; vec_col++) {
                                    int db_idx = db_start + col_base + vec_col;
                                    if (db_idx < M) {
                                        float dist = _t0[vec_col];
                                        if (dist < best_d[TOP_K_MAX - 1]) {
                                            best_d[TOP_K_MAX - 1] = dist;
                                            best_i[TOP_K_MAX - 1] = db_idx;
                                            #pragma unroll
                                            for (int pos = TOP_K_MAX - 1; pos >= 1; pos--) {
                                                if (best_d[pos] < best_d[pos - 1]) {
                                                    float tmp_d = best_d[pos - 1];
                                                    int tmp_i = best_i[pos - 1];
                                                    best_d[pos - 1] = best_d[pos];
                                                    best_i[pos - 1] = best_i[pos];
                                                    best_d[pos] = tmp_d;
                                                    best_i[pos] = tmp_i;
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                    asm volatile("barrier.sync 8, 128;");
                }
                if (valid_q != 0) {
                    int out_base = ((split_idx * B + batch_idx) * Q + q_idx) * K;
                    #pragma unroll
                    for (int out_k = 0; out_k < TOP_K_MAX; out_k++) {
                        if (out_k < K) {
                            *((float*)(partial_dists + out_base + out_k)) = best_d[out_k];
                            *((int*)(partial_indices + out_base + out_k)) = best_i[out_k];
                        }
                    }
                }
            }
        }
    }

    // Cleanup
    asm volatile("barrier.cluster.arrive.release.aligned;");
    asm volatile("barrier.cluster.wait.acquire.aligned;");

    if (warp == 1) {
        asm volatile("tcgen05.dealloc.cta_group::2.sync.aligned.b32 %0, %1;" :: "r"(tmem_addr_storage[0]), "r"(64));
        asm volatile("tcgen05.relinquish_alloc_permit.cta_group::2.sync.aligned;");
    }
}

} // extern "C"

#undef BLOCK_M
#undef BLOCK_Q
#undef FEAT_D
#undef NUM_MAIN_STAGES
#undef SMEM_SMEM_DATABASE_OFF
#undef SMEM_SMEM_DATABASE_SQ_OFF
#undef SMEM_SMEM_DATABASE_SQ_STAGE_BYTES
#undef SMEM_SMEM_DATABASE_SQ_STRIDE
#undef SMEM_SMEM_DATABASE_STAGE_BYTES
#undef SMEM_SMEM_DATABASE_STRIDE
#undef SMEM_SMEM_QUERY_OFF
#undef SMEM_SMEM_QUERY_STAGE_BYTES
#undef SMEM_SMEM_QUERY_STRIDE
#undef SMEM_TOTAL
#undef THREADS
#undef TMEM_CROSS_OFFSET
#undef TMEM_NCOLS
#undef TOP_K_MAX
#undef database_empty_addr
#undef database_full_addr
#undef query_empty_addr
#undef query_full_addr
#undef score_empty_addr
#undef score_full_addr
#undef smem_database_addr
#undef smem_database_sq_addr
#undef smem_query_addr

#define NUM_MAIN_STAGES 1
#define THREADS 32
#define TOP_K_MAX 24
#define SPLIT_COUNT 8

extern "C" {

__global__ __launch_bounds__(32, 1) void
kernel_knn_build_evolve_7bfc_k30_merge_s8_rowbase_cache_bad5midks8k24(float* __restrict__ partial_dists, int32_t* __restrict__ partial_indices, float* __restrict__ out_dists, int32_t* __restrict__ out_indices, int total_queries)
{
    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;


    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    const int warp_id = warp;
    const int lane_id = lane;

    // === Task calls (dependency order) ===
    int start_row = bid * 32 + tid;
    int stride = num_bids * 32;
    #pragma unroll 1
    for (int row = start_row; row < total_queries; row += stride) {
        int base_row = row * TOP_K_MAX;
        int split_stride = total_queries * TOP_K_MAX;
        int out_base = base_row;
        int split_pos[SPLIT_COUNT];
        int split_base[SPLIT_COUNT];
        float cand_d[SPLIT_COUNT];
        int cand_i[SPLIT_COUNT];
        #pragma unroll
        for (int split_idx = 0; split_idx < SPLIT_COUNT; split_idx++) {
            split_pos[split_idx] = 0;
            split_base[split_idx] = base_row + split_idx * split_stride;
            cand_d[split_idx] = (float)partial_dists[split_base[split_idx]];
            cand_i[split_idx] = partial_indices[split_base[split_idx]];
        }
        #pragma unroll
        for (int out_k = 0; out_k < TOP_K_MAX; out_k++) {
            float best_d = cand_d[0];
            int best_i = cand_i[0];
            int best_split = 0;
            #pragma unroll
            for (int split_idx = 1; split_idx < SPLIT_COUNT; split_idx++) {
                if (cand_d[split_idx] < best_d) {
                    best_d = cand_d[split_idx];
                    best_i = cand_i[split_idx];
                    best_split = split_idx;
                }
            }
            *((float*)(out_dists + out_base + out_k)) = best_d;
            *((int*)(out_indices + out_base + out_k)) = best_i;
            split_pos[best_split] = split_pos[best_split] + 1;
            if (out_k + 1 < TOP_K_MAX) {
                int next_pos = split_pos[best_split];
                int next_addr = split_base[best_split] + next_pos;
                cand_d[best_split] = (float)partial_dists[next_addr];
                cand_i[best_split] = partial_indices[next_addr];
            }
        }
    }
}

} // extern "C"

#undef NUM_MAIN_STAGES
#undef SPLIT_COUNT
#undef THREADS
#undef TOP_K_MAX

#define NUM_MAIN_STAGES 1
#define THREADS 32
#define TOP_K_MAX 28
#define SPLIT_COUNT 8

extern "C" {

__global__ __launch_bounds__(32, 1) void
kernel_knn_build_evolve_7bfc_k30_merge_s8_rowbase_cache_bad5midks8k28(float* __restrict__ partial_dists, int32_t* __restrict__ partial_indices, float* __restrict__ out_dists, int32_t* __restrict__ out_indices, int total_queries)
{
    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;


    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    const int warp_id = warp;
    const int lane_id = lane;

    // === Task calls (dependency order) ===
    int start_row = bid * 32 + tid;
    int stride = num_bids * 32;
    #pragma unroll 1
    for (int row = start_row; row < total_queries; row += stride) {
        int base_row = row * TOP_K_MAX;
        int split_stride = total_queries * TOP_K_MAX;
        int out_base = base_row;
        int split_pos[SPLIT_COUNT];
        int split_base[SPLIT_COUNT];
        float cand_d[SPLIT_COUNT];
        int cand_i[SPLIT_COUNT];
        #pragma unroll
        for (int split_idx = 0; split_idx < SPLIT_COUNT; split_idx++) {
            split_pos[split_idx] = 0;
            split_base[split_idx] = base_row + split_idx * split_stride;
            cand_d[split_idx] = (float)partial_dists[split_base[split_idx]];
            cand_i[split_idx] = partial_indices[split_base[split_idx]];
        }
        #pragma unroll
        for (int out_k = 0; out_k < TOP_K_MAX; out_k++) {
            float best_d = cand_d[0];
            int best_i = cand_i[0];
            int best_split = 0;
            #pragma unroll
            for (int split_idx = 1; split_idx < SPLIT_COUNT; split_idx++) {
                if (cand_d[split_idx] < best_d) {
                    best_d = cand_d[split_idx];
                    best_i = cand_i[split_idx];
                    best_split = split_idx;
                }
            }
            *((float*)(out_dists + out_base + out_k)) = best_d;
            *((int*)(out_indices + out_base + out_k)) = best_i;
            split_pos[best_split] = split_pos[best_split] + 1;
            if (out_k + 1 < TOP_K_MAX) {
                int next_pos = split_pos[best_split];
                int next_addr = split_base[best_split] + next_pos;
                cand_d[best_split] = (float)partial_dists[next_addr];
                cand_i[best_split] = partial_indices[next_addr];
            }
        }
    }
}

} // extern "C"

#undef NUM_MAIN_STAGES
#undef SPLIT_COUNT
#undef THREADS
#undef TOP_K_MAX

#define TMEM_NCOLS 64
#define TMEM_CROSS_OFFSET 0
#define NUM_MAIN_STAGES 1
#define SMEM_SMEM_QUERY_OFF 1024
#define SMEM_SMEM_QUERY_STAGE_BYTES 65536
#define SMEM_SMEM_QUERY_STRIDE 65536
#define SMEM_SMEM_DATABASE_OFF 66560
#define SMEM_SMEM_DATABASE_STAGE_BYTES 32768
#define SMEM_SMEM_DATABASE_STRIDE 32768
#define SMEM_SMEM_DATABASE_SQ_OFF 99328
#define SMEM_SMEM_DATABASE_SQ_STAGE_BYTES 256
#define SMEM_SMEM_DATABASE_SQ_STRIDE 256
#define SMEM_TOTAL 99584
#define THREADS 192
#define BLOCK_Q 128
#define BLOCK_M 64
#define TOP_K_MAX 10

extern "C" {

__global__ __launch_bounds__(192, 1) void
kernel_knn_build_dim_midk_df2f_d256_split_stage1(float* __restrict__ query_sq, float* __restrict__ database_sq, float* __restrict__ partial_dists, int32_t* __restrict__ partial_indices, const void* tmap_query, const void* tmap_database, int B, int Q, int M, int K, int num_q_tiles, int db_tiles_per_split, int split_count, int total_work)
{
    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;

    extern __shared__ __align__(1024) char smem_raw[];
    int smem;
    smem = (int)(unsigned long long)__cvta_generic_to_shared(smem_raw);
    const int smem_smem_query = smem + 1024;
    const int smem_smem_database = smem + 66560;
    const int smem_smem_database_sq = smem + 99328;

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
        // score_empty: 1 barriers, init_count=128
        mbarrier_init_pred(smem + 40, 128, leader);
        asm volatile("fence.mbarrier_init.release.cluster;");
    }

    __syncthreads();

    // TMEM alloc (64 columns, 64 used)
    volatile int* tmem_addr_storage = (volatile int*)(smem_raw + 48);
    if (warp == 1) {
        int _tmem_hold = smem + 48;
        asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;" :: "r"(_tmem_hold), "r"(64) : "memory");
    }

    __syncthreads();
    asm volatile("tcgen05.fence::after_thread_sync;");

    const int warp_id = warp;
    const int lane_id = lane;
    __nv_bfloat16* smem_query = (__nv_bfloat16*)(smem_raw + 1024);
    #define smem_query_addr (smem + 1024)
    __nv_bfloat16* smem_database = (__nv_bfloat16*)(smem_raw + 66560);
    #define smem_database_addr (smem + 66560)
    float* smem_database_sq = (float*)(smem_raw + 99328);
    #define smem_database_sq_addr (smem + 99328)
    const int mbar_base = smem;
    #define query_full_addr (mbar_base + 0)
    #define query_empty_addr (mbar_base + 8)
    #define database_full_addr (mbar_base + 16)
    #define database_empty_addr (mbar_base + 24)
    #define score_full_addr (mbar_base + 32)
    #define score_empty_addr (mbar_base + 40)
    const int taddr = tmem_addr_storage[0];

    // ---- Role: load ----
    if (warp == 0) {
        { // load_main
            uint32_t _phase_query_empty_0 = 1;
            uint32_t _phase_database_empty_0 = 1;
            if (warp_id == 0) {
                if (elect_sync()) {
                    #pragma unroll 1
                    for (unsigned int work_idx = bid; work_idx < total_work; work_idx += num_bids) {
                        int split_idx = work_idx % split_count;
                        int query_work = work_idx / split_count;
                        int batch_idx = query_work / num_q_tiles;
                        int q_tile = query_work % num_q_tiles;
                        int off_q = q_tile * BLOCK_Q;
                        int global_q = batch_idx * Q + off_q;
                        int db_tile_start = split_idx * db_tiles_per_split;
                        mbarrier_wait(query_empty_addr, _phase_query_empty_0);
                        _phase_query_empty_0 ^= 1;
                        mbarrier_arrive_expect_tx(query_full_addr, 65536);
                        tma_3d_gmem2smem(smem_query_addr, tmap_query, 0, global_q, 0, query_full_addr);
                        #pragma unroll 1
                        for (int local_db_tile = 0; local_db_tile < db_tiles_per_split; local_db_tile++) {
                            int db_tile = db_tile_start + local_db_tile;
                            int off_m = db_tile * BLOCK_M;
                            int global_m = batch_idx * M + off_m;
                            mbarrier_wait(database_empty_addr, _phase_database_empty_0);
                            _phase_database_empty_0 ^= 1;
                            mbarrier_arrive_expect_tx(database_full_addr, 32768);
                            tma_3d_gmem2smem(smem_database_addr, tmap_database, 0, global_m, 0, database_full_addr);
                        }
                    }
                }
            }
        }
    // ---- Role: mma ----
    } else if (warp == 1) {
        { // mma_main
            uint32_t _phase_query_full_0 = 0;
            uint32_t _phase_score_empty_0 = 1;
            uint32_t _phase_database_full_0 = 0;
            #pragma unroll 1
            for (unsigned int work_idx = bid; work_idx < total_work; work_idx += num_bids) {
                mbarrier_wait(query_full_addr, _phase_query_full_0);
                _phase_query_full_0 ^= 1;
                #pragma unroll 1
                for (int _local_db_tile = 0; _local_db_tile < db_tiles_per_split; _local_db_tile++) {
                    mbarrier_wait(score_empty_addr, _phase_score_empty_0);
                    _phase_score_empty_0 ^= 1;
                    mbarrier_wait(database_full_addr, _phase_database_full_0);
                    _phase_database_full_0 ^= 1;
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    int _desc_lo_0 = make_warp_uniform((smem_query_addr + 32768 >> 4) & 0x3FFF);
                    int _desc_lo_1 = make_warp_uniform((smem_database_addr + 16384 >> 4) & 0x3FFF);
                    int _mma_ss_a_lo_2 = make_warp_uniform((smem_query_addr >> 4) & 0x3FFF);
                    int _mma_ss_b_lo_2 = make_warp_uniform((smem_database_addr >> 4) & 0x3FFF);
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
                    "mov.b32 id, 135267472;\n\t"
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
                    :: "r"(_mma_ss_a_lo_2), "r"(_mma_ss_b_lo_2), "r"(taddr), "r"(0));
                    asm volatile("tcgen05.fence::after_thread_sync;");
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
                    "mov.b32 id, 135267472;\n\t"
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
                    :: "r"(_desc_lo_0), "r"(_desc_lo_1), "r"(taddr), "r"(1));
                    elect_commit(score_full_addr);
                    elect_commit(database_empty_addr);
                }
                elect_commit(query_empty_addr);
            }
        }
    // ---- Role: compute ----
    } else if (warp >= 2 && warp <= 5) {
        const int warp_id_in_wg = warp % 4;
        const int my_row = warp_id_in_wg * 32 + lane;
        const int tmem_row_base_v = (warp_id_in_wg * 32) << 16;
        const int tmem_cross = taddr + TMEM_CROSS_OFFSET;
        { // compute_main
            uint32_t _phase_score_full_0 = 0;
            #pragma unroll 1
            for (unsigned int work_idx = bid; work_idx < total_work; work_idx += num_bids) {
                int split_idx = work_idx % split_count;
                int query_work = work_idx / split_count;
                int batch_idx = query_work / num_q_tiles;
                int q_tile = query_work % num_q_tiles;
                int off_q = q_tile * BLOCK_Q;
                int q_idx = off_q + my_row;
                int valid_q = ((q_idx < Q) ? 1 : 0);
                float q_sq_val = 0.0f;
                if (valid_q != 0) {
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
                    int db_start = db_tile * BLOCK_M;
                    int db_sq_idx = db_start + my_row;
                    if (my_row < BLOCK_M) {
                        if (db_sq_idx < M) {
                            smem_database_sq[my_row] = (float)database_sq[batch_idx * M + db_sq_idx];
                        } else {
                            smem_database_sq[my_row] = 0.0f;
                        }
                    }
                    asm volatile("barrier.sync 8, 128;");
                    mbarrier_wait(score_full_addr, _phase_score_full_0);
                    _phase_score_full_0 ^= 1;
                    int cross_addr = taddr + tmem_row_base_v;
                    float dots[64];
                    tmem_ld_x32(&dots[0], cross_addr);
                    tmem_ld_x32(&dots[32], cross_addr + 32);
                    asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                    asm volatile("barrier.sync 8, 128;");
                    mbarrier_arrive(score_empty_addr);
                    if (valid_q != 0) {
                        #pragma unroll 2
                        for (int col_base = 0; col_base < 64; col_base += 4) {
                            float dist_vec[4];
                            dist_vec[0] = dots[col_base];
                            dist_vec[1] = dots[col_base + 1];
                            dist_vec[2] = dots[col_base + 2];
                            dist_vec[3] = dots[col_base + 3];
                            const float2 _fma_b2_0 = {-2.0f, -2.0f};
                            const float2 _fma_c2_1 = {q_sq_val, q_sq_val};
                            #pragma unroll
                            for (int _lf = 0; _lf < 2; _lf++)
                                fma_f32x2_inplace(&reinterpret_cast<float2*>(dist_vec)[_lf], _fma_b2_0, _fma_c2_1);
                            float db_sq_vec[4];
                            db_sq_vec[0] = smem_database_sq[col_base];
                            db_sq_vec[1] = smem_database_sq[col_base + 1];
                            db_sq_vec[2] = smem_database_sq[col_base + 2];
                            db_sq_vec[3] = smem_database_sq[col_base + 3];
                            float _t0[4];
                            #pragma unroll
                            for (int _la = 0; _la < 2; _la++)
                                reinterpret_cast<float2*>(_t0)[_la] = add_f32x2(reinterpret_cast<float2*>(dist_vec)[_la], reinterpret_cast<const float2*>(db_sq_vec)[_la]);
                            float group_min = _t0[0];
                            if (_t0[1] < group_min) {
                                group_min = _t0[1];
                            }
                            if (_t0[2] < group_min) {
                                group_min = _t0[2];
                            }
                            if (_t0[3] < group_min) {
                                group_min = _t0[3];
                            }
                            if (group_min < best_d[TOP_K_MAX - 1]) {
                                #pragma unroll
                                for (int vec_col = 0; vec_col < 4; vec_col++) {
                                    int db_idx = db_start + col_base + vec_col;
                                    if (db_idx < M) {
                                        float dist = _t0[vec_col];
                                        dist = max_noftz(dist, 0.0f);
                                        if (dist < best_d[TOP_K_MAX - 1]) {
                                            best_d[TOP_K_MAX - 1] = dist;
                                            best_i[TOP_K_MAX - 1] = db_idx;
                                            #pragma unroll
                                            for (int pos = TOP_K_MAX - 1; pos >= 1; pos--) {
                                                if (best_d[pos] < best_d[pos - 1]) {
                                                    float tmp_d = best_d[pos - 1];
                                                    int tmp_i = best_i[pos - 1];
                                                    best_d[pos - 1] = best_d[pos];
                                                    best_i[pos - 1] = best_i[pos];
                                                    best_d[pos] = tmp_d;
                                                    best_i[pos] = tmp_i;
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                    asm volatile("barrier.sync 8, 128;");
                }
                if (valid_q != 0) {
                    int out_base = ((split_idx * B + batch_idx) * Q + q_idx) * K;
                    #pragma unroll
                    for (int out_k = 0; out_k < TOP_K_MAX; out_k++) {
                        if (out_k < K) {
                            *((float*)(partial_dists + out_base + out_k)) = best_d[out_k];
                            *((int*)(partial_indices + out_base + out_k)) = best_i[out_k];
                        }
                    }
                }
            }
        }
    }

    // Cleanup
    __syncthreads(); // barrier before TMEM dealloc

    if (warp == 1) {
        asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;" :: "r"(tmem_addr_storage[0]), "r"(64));
        asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;");
    }
}

} // extern "C"

#undef BLOCK_M
#undef BLOCK_Q
#undef NUM_MAIN_STAGES
#undef SMEM_SMEM_DATABASE_OFF
#undef SMEM_SMEM_DATABASE_SQ_OFF
#undef SMEM_SMEM_DATABASE_SQ_STAGE_BYTES
#undef SMEM_SMEM_DATABASE_SQ_STRIDE
#undef SMEM_SMEM_DATABASE_STAGE_BYTES
#undef SMEM_SMEM_DATABASE_STRIDE
#undef SMEM_SMEM_QUERY_OFF
#undef SMEM_SMEM_QUERY_STAGE_BYTES
#undef SMEM_SMEM_QUERY_STRIDE
#undef SMEM_TOTAL
#undef THREADS
#undef TMEM_CROSS_OFFSET
#undef TMEM_NCOLS
#undef TOP_K_MAX
#undef database_empty_addr
#undef database_full_addr
#undef query_empty_addr
#undef query_full_addr
#undef score_empty_addr
#undef score_full_addr
#undef smem_database_addr
#undef smem_database_sq_addr
#undef smem_query_addr

#define TMEM_NCOLS 64
#define TMEM_CROSS_OFFSET 0
#define NUM_MAIN_STAGES 1
#define SMEM_SMEM_QUERY_OFF 1024
#define SMEM_SMEM_QUERY_STAGE_BYTES 32768
#define SMEM_SMEM_QUERY_STRIDE 32768
#define SMEM_SMEM_DATABASE_OFF 33792
#define SMEM_SMEM_DATABASE_STAGE_BYTES 16384
#define SMEM_SMEM_DATABASE_STRIDE 16384
#define SMEM_SMEM_DATABASE_SQ_OFF 50176
#define SMEM_SMEM_DATABASE_SQ_STAGE_BYTES 256
#define SMEM_SMEM_DATABASE_SQ_STRIDE 256
#define SMEM_TOTAL 50432
#define THREADS 192
#define BLOCK_Q 128
#define BLOCK_M 64
#define TOP_K_MAX 10

extern "C" {

__global__ __launch_bounds__(192, 1) void
kernel_knn_build_dim_midk_df2f_fp16_split_stage1(float* __restrict__ query_sq, float* __restrict__ database_sq, float* __restrict__ partial_dists, int32_t* __restrict__ partial_indices, const void* tmap_query, const void* tmap_database, int B, int Q, int M, int K, int num_q_tiles, int db_tiles_per_split, int split_count, int total_work)
{
    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;

    extern __shared__ __align__(1024) char smem_raw[];
    int smem;
    smem = (int)(unsigned long long)__cvta_generic_to_shared(smem_raw);
    const int smem_smem_query = smem + 1024;
    const int smem_smem_database = smem + 33792;
    const int smem_smem_database_sq = smem + 50176;

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
        // score_empty: 1 barriers, init_count=128
        mbarrier_init_pred(smem + 40, 128, leader);
        asm volatile("fence.mbarrier_init.release.cluster;");
    }

    __syncthreads();

    // TMEM alloc (64 columns, 64 used)
    volatile int* tmem_addr_storage = (volatile int*)(smem_raw + 48);
    if (warp == 1) {
        int _tmem_hold = smem + 48;
        asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;" :: "r"(_tmem_hold), "r"(64) : "memory");
    }

    __syncthreads();
    asm volatile("tcgen05.fence::after_thread_sync;");

    const int warp_id = warp;
    const int lane_id = lane;
    __half* smem_query = (__half*)(smem_raw + 1024);
    #define smem_query_addr (smem + 1024)
    __half* smem_database = (__half*)(smem_raw + 33792);
    #define smem_database_addr (smem + 33792)
    float* smem_database_sq = (float*)(smem_raw + 50176);
    #define smem_database_sq_addr (smem + 50176)
    const int mbar_base = smem;
    #define query_full_addr (mbar_base + 0)
    #define query_empty_addr (mbar_base + 8)
    #define database_full_addr (mbar_base + 16)
    #define database_empty_addr (mbar_base + 24)
    #define score_full_addr (mbar_base + 32)
    #define score_empty_addr (mbar_base + 40)
    const int taddr = tmem_addr_storage[0];

    // ---- Role: load ----
    if (warp == 0) {
        { // load_main
            uint32_t _phase_query_empty_0 = 1;
            uint32_t _phase_database_empty_0 = 1;
            if (warp_id == 0) {
                if (elect_sync()) {
                    #pragma unroll 1
                    for (unsigned int work_idx = bid; work_idx < total_work; work_idx += num_bids) {
                        int split_idx = work_idx % split_count;
                        int query_work = work_idx / split_count;
                        int batch_idx = query_work / num_q_tiles;
                        int q_tile = query_work % num_q_tiles;
                        int off_q = q_tile * BLOCK_Q;
                        int global_q = batch_idx * Q + off_q;
                        int db_tile_start = split_idx * db_tiles_per_split;
                        mbarrier_wait(query_empty_addr, _phase_query_empty_0);
                        _phase_query_empty_0 ^= 1;
                        mbarrier_arrive_expect_tx(query_full_addr, 32768);
                        tma_3d_gmem2smem(smem_query_addr, tmap_query, 0, global_q, 0, query_full_addr);
                        #pragma unroll 1
                        for (int local_db_tile = 0; local_db_tile < db_tiles_per_split; local_db_tile++) {
                            int db_tile = db_tile_start + local_db_tile;
                            int off_m = db_tile * BLOCK_M;
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
    } else if (warp == 1) {
        { // mma_main
            uint32_t _phase_query_full_0 = 0;
            uint32_t _phase_score_empty_0 = 1;
            uint32_t _phase_database_full_0 = 0;
            #pragma unroll 1
            for (unsigned int work_idx = bid; work_idx < total_work; work_idx += num_bids) {
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
                    "mov.b32 id, 135266320;\n\t"
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
    // ---- Role: compute ----
    } else if (warp >= 2 && warp <= 5) {
        const int warp_id_in_wg = warp % 4;
        const int my_row = warp_id_in_wg * 32 + lane;
        const int tmem_row_base_v = (warp_id_in_wg * 32) << 16;
        const int tmem_cross = taddr + TMEM_CROSS_OFFSET;
        { // compute_main
            uint32_t _phase_score_full_0 = 0;
            #pragma unroll 1
            for (unsigned int work_idx = bid; work_idx < total_work; work_idx += num_bids) {
                int split_idx = work_idx % split_count;
                int query_work = work_idx / split_count;
                int batch_idx = query_work / num_q_tiles;
                int q_tile = query_work % num_q_tiles;
                int off_q = q_tile * BLOCK_Q;
                int q_idx = off_q + my_row;
                int valid_q = ((q_idx < Q) ? 1 : 0);
                float q_sq_val = 0.0f;
                if (valid_q != 0) {
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
                    int db_start = db_tile * BLOCK_M;
                    int db_sq_idx = db_start + my_row;
                    if (my_row < BLOCK_M) {
                        if (db_sq_idx < M) {
                            smem_database_sq[my_row] = (float)database_sq[batch_idx * M + db_sq_idx];
                        } else {
                            smem_database_sq[my_row] = 0.0f;
                        }
                    }
                    asm volatile("barrier.sync 8, 128;");
                    mbarrier_wait(score_full_addr, _phase_score_full_0);
                    _phase_score_full_0 ^= 1;
                    int cross_addr = taddr + tmem_row_base_v;
                    float dots[64];
                    tmem_ld_x32(&dots[0], cross_addr);
                    tmem_ld_x32(&dots[32], cross_addr + 32);
                    asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                    asm volatile("barrier.sync 8, 128;");
                    mbarrier_arrive(score_empty_addr);
                    if (valid_q != 0) {
                        #pragma unroll 2
                        for (int col_base = 0; col_base < 64; col_base += 4) {
                            float dist_vec[4];
                            dist_vec[0] = dots[col_base];
                            dist_vec[1] = dots[col_base + 1];
                            dist_vec[2] = dots[col_base + 2];
                            dist_vec[3] = dots[col_base + 3];
                            const float2 _fma_b2_0 = {-2.0f, -2.0f};
                            const float2 _fma_c2_1 = {q_sq_val, q_sq_val};
                            #pragma unroll
                            for (int _lf = 0; _lf < 2; _lf++)
                                fma_f32x2_inplace(&reinterpret_cast<float2*>(dist_vec)[_lf], _fma_b2_0, _fma_c2_1);
                            float db_sq_vec[4];
                            db_sq_vec[0] = smem_database_sq[col_base];
                            db_sq_vec[1] = smem_database_sq[col_base + 1];
                            db_sq_vec[2] = smem_database_sq[col_base + 2];
                            db_sq_vec[3] = smem_database_sq[col_base + 3];
                            float _t0[4];
                            #pragma unroll
                            for (int _la = 0; _la < 2; _la++)
                                reinterpret_cast<float2*>(_t0)[_la] = add_f32x2(reinterpret_cast<float2*>(dist_vec)[_la], reinterpret_cast<const float2*>(db_sq_vec)[_la]);
                            float group_min = _t0[0];
                            if (_t0[1] < group_min) {
                                group_min = _t0[1];
                            }
                            if (_t0[2] < group_min) {
                                group_min = _t0[2];
                            }
                            if (_t0[3] < group_min) {
                                group_min = _t0[3];
                            }
                            if (group_min < best_d[TOP_K_MAX - 1]) {
                                #pragma unroll
                                for (int vec_col = 0; vec_col < 4; vec_col++) {
                                    int db_idx = db_start + col_base + vec_col;
                                    if (db_idx < M) {
                                        float dist = _t0[vec_col];
                                        dist = max_noftz(dist, 0.0f);
                                        if (dist < best_d[TOP_K_MAX - 1]) {
                                            best_d[TOP_K_MAX - 1] = dist;
                                            best_i[TOP_K_MAX - 1] = db_idx;
                                            #pragma unroll
                                            for (int pos = TOP_K_MAX - 1; pos >= 1; pos--) {
                                                if (best_d[pos] < best_d[pos - 1]) {
                                                    float tmp_d = best_d[pos - 1];
                                                    int tmp_i = best_i[pos - 1];
                                                    best_d[pos - 1] = best_d[pos];
                                                    best_i[pos - 1] = best_i[pos];
                                                    best_d[pos] = tmp_d;
                                                    best_i[pos] = tmp_i;
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                    asm volatile("barrier.sync 8, 128;");
                }
                if (valid_q != 0) {
                    int out_base = ((split_idx * B + batch_idx) * Q + q_idx) * K;
                    #pragma unroll
                    for (int out_k = 0; out_k < TOP_K_MAX; out_k++) {
                        if (out_k < K) {
                            *((float*)(partial_dists + out_base + out_k)) = best_d[out_k];
                            *((int*)(partial_indices + out_base + out_k)) = best_i[out_k];
                        }
                    }
                }
            }
        }
    }

    // Cleanup
    __syncthreads(); // barrier before TMEM dealloc

    if (warp == 1) {
        asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;" :: "r"(tmem_addr_storage[0]), "r"(64));
        asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;");
    }
}

} // extern "C"

#undef BLOCK_M
#undef BLOCK_Q
#undef NUM_MAIN_STAGES
#undef SMEM_SMEM_DATABASE_OFF
#undef SMEM_SMEM_DATABASE_SQ_OFF
#undef SMEM_SMEM_DATABASE_SQ_STAGE_BYTES
#undef SMEM_SMEM_DATABASE_SQ_STRIDE
#undef SMEM_SMEM_DATABASE_STAGE_BYTES
#undef SMEM_SMEM_DATABASE_STRIDE
#undef SMEM_SMEM_QUERY_OFF
#undef SMEM_SMEM_QUERY_STAGE_BYTES
#undef SMEM_SMEM_QUERY_STRIDE
#undef SMEM_TOTAL
#undef THREADS
#undef TMEM_CROSS_OFFSET
#undef TMEM_NCOLS
#undef TOP_K_MAX
#undef database_empty_addr
#undef database_full_addr
#undef query_empty_addr
#undef query_full_addr
#undef score_empty_addr
#undef score_full_addr
#undef smem_database_addr
#undef smem_database_sq_addr
#undef smem_query_addr

#define NUM_MAIN_STAGES 1
#define THREADS 32
#define TOP_K_MAX 64
#define SPLIT_COUNT 12

extern "C" {

__global__ __launch_bounds__(32, 1) void
kernel_knn_build_k64_merge_sN_unordered_chunkprefill_k64over32s12chunkprefill(float* __restrict__ partial_dists, int32_t* __restrict__ partial_indices, float* __restrict__ out_dists, int32_t* __restrict__ out_indices, int total_queries)
{
    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;


    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    const int warp_id = warp;
    const int lane_id = lane;

    // === Task calls (dependency order) ===
    int start_row = bid * 32 + tid;
    int stride = num_bids * 32;
    #pragma unroll 1
    for (int row = start_row; row < total_queries; row += stride) {
        int base_row = row * TOP_K_MAX;
        int split_stride = total_queries * TOP_K_MAX;
        float best_d[64];
        int best_i[64];
        #pragma unroll
        for (int cand_k = 0; cand_k < 64; cand_k++) {
            best_d[cand_k] = (float)partial_dists[base_row + cand_k];
            best_i[cand_k] = partial_indices[base_row + cand_k];
        }
        float chunk_worst_d[8];
        int chunk_worst_pos[8];
        #pragma unroll
        for (int chunk = 0; chunk < 8; chunk++) {
            int chunk_base = chunk * 8;
            chunk_worst_d[chunk] = best_d[chunk_base];
            chunk_worst_pos[chunk] = chunk_base;
            #pragma unroll
            for (int offset = 1; offset < 8; offset++) {
                int scan_pos = chunk_base + offset;
                if (best_d[scan_pos] > chunk_worst_d[chunk]) {
                    chunk_worst_d[chunk] = best_d[scan_pos];
                    chunk_worst_pos[chunk] = scan_pos;
                }
            }
        }
        float worst_d = chunk_worst_d[0];
        int worst_pos = chunk_worst_pos[0];
        int worst_chunk = 0;
        #pragma unroll
        for (int chunk = 1; chunk < 8; chunk++) {
            if (chunk_worst_d[chunk] > worst_d) {
                worst_d = chunk_worst_d[chunk];
                worst_pos = chunk_worst_pos[chunk];
                worst_chunk = chunk;
            }
        }
        #pragma unroll
        for (int split_idx = 1; split_idx < SPLIT_COUNT; split_idx++) {
            int partial_base = base_row + split_idx * split_stride;
            #pragma unroll
            for (int cand_k = 0; cand_k < 64; cand_k++) {
                float cand_d = (float)partial_dists[partial_base + cand_k];
                int cand_i = partial_indices[partial_base + cand_k];
                if (cand_d < worst_d) {
                    best_d[worst_pos] = cand_d;
                    best_i[worst_pos] = cand_i;
                    int refresh_base = worst_chunk * 8;
                    chunk_worst_d[worst_chunk] = best_d[refresh_base];
                    chunk_worst_pos[worst_chunk] = refresh_base;
                    #pragma unroll
                    for (int offset = 1; offset < 8; offset++) {
                        int scan_pos = refresh_base + offset;
                        if (best_d[scan_pos] > chunk_worst_d[worst_chunk]) {
                            chunk_worst_d[worst_chunk] = best_d[scan_pos];
                            chunk_worst_pos[worst_chunk] = scan_pos;
                        }
                    }
                    worst_d = chunk_worst_d[0];
                    worst_pos = chunk_worst_pos[0];
                    worst_chunk = 0;
                    #pragma unroll
                    for (int chunk = 1; chunk < 8; chunk++) {
                        if (chunk_worst_d[chunk] > worst_d) {
                            worst_d = chunk_worst_d[chunk];
                            worst_pos = chunk_worst_pos[chunk];
                            worst_chunk = chunk;
                        }
                    }
                }
            }
        }
        #pragma unroll
        for (int out_k = 0; out_k < 64; out_k++) {
            *((float*)(out_dists + base_row + out_k)) = best_d[out_k];
            *((int*)(out_indices + base_row + out_k)) = best_i[out_k];
        }
    }
}

} // extern "C"

#undef NUM_MAIN_STAGES
#undef SPLIT_COUNT
#undef THREADS
#undef TOP_K_MAX

#define NUM_MAIN_STAGES 1
#define THREADS 32
#define TOP_K_MAX 64
#define SPLIT_COUNT 16

extern "C" {

__global__ __launch_bounds__(32, 1) void
kernel_knn_build_k64_merge_sN_unordered_chunkprefill_k64over32s16chunkprefill(float* __restrict__ partial_dists, int32_t* __restrict__ partial_indices, float* __restrict__ out_dists, int32_t* __restrict__ out_indices, int total_queries)
{
    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;


    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    const int warp_id = warp;
    const int lane_id = lane;

    // === Task calls (dependency order) ===
    int start_row = bid * 32 + tid;
    int stride = num_bids * 32;
    #pragma unroll 1
    for (int row = start_row; row < total_queries; row += stride) {
        int base_row = row * TOP_K_MAX;
        int split_stride = total_queries * TOP_K_MAX;
        float best_d[64];
        int best_i[64];
        #pragma unroll
        for (int cand_k = 0; cand_k < 64; cand_k++) {
            best_d[cand_k] = (float)partial_dists[base_row + cand_k];
            best_i[cand_k] = partial_indices[base_row + cand_k];
        }
        float chunk_worst_d[8];
        int chunk_worst_pos[8];
        #pragma unroll
        for (int chunk = 0; chunk < 8; chunk++) {
            int chunk_base = chunk * 8;
            chunk_worst_d[chunk] = best_d[chunk_base];
            chunk_worst_pos[chunk] = chunk_base;
            #pragma unroll
            for (int offset = 1; offset < 8; offset++) {
                int scan_pos = chunk_base + offset;
                if (best_d[scan_pos] > chunk_worst_d[chunk]) {
                    chunk_worst_d[chunk] = best_d[scan_pos];
                    chunk_worst_pos[chunk] = scan_pos;
                }
            }
        }
        float worst_d = chunk_worst_d[0];
        int worst_pos = chunk_worst_pos[0];
        int worst_chunk = 0;
        #pragma unroll
        for (int chunk = 1; chunk < 8; chunk++) {
            if (chunk_worst_d[chunk] > worst_d) {
                worst_d = chunk_worst_d[chunk];
                worst_pos = chunk_worst_pos[chunk];
                worst_chunk = chunk;
            }
        }
        #pragma unroll
        for (int split_idx = 1; split_idx < SPLIT_COUNT; split_idx++) {
            int partial_base = base_row + split_idx * split_stride;
            #pragma unroll
            for (int cand_k = 0; cand_k < 64; cand_k++) {
                float cand_d = (float)partial_dists[partial_base + cand_k];
                int cand_i = partial_indices[partial_base + cand_k];
                if (cand_d < worst_d) {
                    best_d[worst_pos] = cand_d;
                    best_i[worst_pos] = cand_i;
                    int refresh_base = worst_chunk * 8;
                    chunk_worst_d[worst_chunk] = best_d[refresh_base];
                    chunk_worst_pos[worst_chunk] = refresh_base;
                    #pragma unroll
                    for (int offset = 1; offset < 8; offset++) {
                        int scan_pos = refresh_base + offset;
                        if (best_d[scan_pos] > chunk_worst_d[worst_chunk]) {
                            chunk_worst_d[worst_chunk] = best_d[scan_pos];
                            chunk_worst_pos[worst_chunk] = scan_pos;
                        }
                    }
                    worst_d = chunk_worst_d[0];
                    worst_pos = chunk_worst_pos[0];
                    worst_chunk = 0;
                    #pragma unroll
                    for (int chunk = 1; chunk < 8; chunk++) {
                        if (chunk_worst_d[chunk] > worst_d) {
                            worst_d = chunk_worst_d[chunk];
                            worst_pos = chunk_worst_pos[chunk];
                            worst_chunk = chunk;
                        }
                    }
                }
            }
        }
        #pragma unroll
        for (int out_k = 0; out_k < 64; out_k++) {
            *((float*)(out_dists + base_row + out_k)) = best_d[out_k];
            *((int*)(out_indices + base_row + out_k)) = best_i[out_k];
        }
    }
}

} // extern "C"

#undef NUM_MAIN_STAGES
#undef SPLIT_COUNT
#undef THREADS
#undef TOP_K_MAX

#define TMEM_NCOLS 64
#define TMEM_CROSS_OFFSET 0
#define NUM_MAIN_STAGES 1
#define SMEM_SMEM_QUERY_OFF 1024
#define SMEM_SMEM_QUERY_STAGE_BYTES 32768
#define SMEM_SMEM_QUERY_STRIDE 32768
#define SMEM_SMEM_DATABASE_OFF 33792
#define SMEM_SMEM_DATABASE_STAGE_BYTES 16384
#define SMEM_SMEM_DATABASE_STRIDE 16384
#define SMEM_SMEM_DATABASE_SQ_OFF 50176
#define SMEM_SMEM_DATABASE_SQ_STAGE_BYTES 256
#define SMEM_SMEM_DATABASE_SQ_STRIDE 256
#define SMEM_TOTAL 50432
#define THREADS 192
#define BLOCK_Q 128
#define BLOCK_M 64
#define FEAT_D 128
#define TOP_K_MAX 48

extern "C" {

__global__ __launch_bounds__(192, 1) void
kernel_knn_build_evolve_7bfc_split_cg2_u2_stage1_k32_unordered_k48over32(float* __restrict__ query_sq, float* __restrict__ database_sq, float* __restrict__ partial_dists, int32_t* __restrict__ partial_indices, const void* tmap_query, const void* tmap_database, int B, int Q, int M, int K, int num_q_tile_pairs, int db_tiles_per_split, int split_count, int total_work)
{
    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;

    extern __shared__ __align__(1024) char smem_raw[];
    int smem;
    smem = (int)(unsigned long long)__cvta_generic_to_shared(smem_raw);
    const int smem_smem_query = smem + 1024;
    const int smem_smem_database = smem + 33792;
    const int smem_smem_database_sq = smem + 50176;

    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;
    const unsigned int clusters_x = gridDim.x / 2;
    const unsigned int cluster_id = ((blockIdx.z * gridDim.y + blockIdx.y) * clusters_x) + blockIdx.x / 2;
    const unsigned int num_clusters = clusters_x * gridDim.y * gridDim.z;

    int cta_rank;
    asm volatile("mov.b32 %0, %%cluster_ctarank;" : "=r"(cta_rank));

    // Mbarrier init (6 groups, 6 barriers)
    // Mbarriers at smem_raw[0..48)

    if (warp == 0) {
        uint32_t leader = elect_sync();
        // query_full: 1 barriers, init_count=2
        mbarrier_init_pred(smem + 0, 2, leader);
        // query_empty: 1 barriers, init_count=1
        mbarrier_init_pred(smem + 8, 1, leader);
        // database_full: 1 barriers, init_count=2
        mbarrier_init_pred(smem + 16, 2, leader);
        // database_empty: 1 barriers, init_count=1
        mbarrier_init_pred(smem + 24, 1, leader);
        // score_full: 1 barriers, init_count=1
        mbarrier_init_pred(smem + 32, 1, leader);
        // score_empty: 1 barriers, init_count=8
        mbarrier_init_pred(smem + 40, 8, leader);
        asm volatile("fence.mbarrier_init.release.cluster;");
    }

    __syncthreads();

    // TMEM alloc (64 columns, 64 used)
    volatile int* tmem_addr_storage = (volatile int*)(smem_raw + 48);
    if (warp == 1) {
        int _tmem_hold = smem + 48;
        asm volatile("tcgen05.alloc.cta_group::2.sync.aligned.shared::cta.b32 [%0], %1;" :: "r"(_tmem_hold), "r"(64) : "memory");
    }

    asm volatile("barrier.cluster.arrive.release.aligned;");
    asm volatile("barrier.cluster.wait.acquire.aligned;");
    asm volatile("tcgen05.fence::after_thread_sync;");

    const int warp_id = warp;
    const int lane_id = lane;
    __nv_bfloat16* smem_query = (__nv_bfloat16*)(smem_raw + 1024);
    #define smem_query_addr (smem + 1024)
    __nv_bfloat16* smem_database = (__nv_bfloat16*)(smem_raw + 33792);
    #define smem_database_addr (smem + 33792)
    float* smem_database_sq = (float*)(smem_raw + 50176);
    #define smem_database_sq_addr (smem + 50176)
    const int mbar_base = smem;
    #define query_full_addr (mbar_base + 0)
    #define query_empty_addr (mbar_base + 8)
    #define database_full_addr (mbar_base + 16)
    #define database_empty_addr (mbar_base + 24)
    #define score_full_addr (mbar_base + 32)
    #define score_empty_addr (mbar_base + 40)
    const int taddr = tmem_addr_storage[0];

    // ---- Role: load ----
    if (warp == 0) {
        { // load_main
            uint32_t _phase_query_empty_0 = 1;
            uint32_t _phase_database_empty_0 = 1;
            if (warp_id == 0) {
                if (elect_sync()) {
                    #pragma unroll 1
                    for (unsigned int work_idx = cluster_id; work_idx < total_work; work_idx += num_clusters) {
                        int split_idx = work_idx % split_count;
                        int query_work = work_idx / split_count;
                        int batch_idx = query_work / num_q_tile_pairs;
                        int q_tile_pair = query_work % num_q_tile_pairs;
                        int q_tile = q_tile_pair * 2 + cta_rank;
                        int off_q = q_tile * BLOCK_Q;
                        int global_q = batch_idx * Q + off_q;
                        int db_tile_start = split_idx * db_tiles_per_split;
                        mbarrier_wait(query_empty_addr, _phase_query_empty_0);
                        _phase_query_empty_0 ^= 1;
                        asm volatile(
                            "mbarrier.arrive.expect_tx.release.cta.shared::cluster.b64 _, [%0], %1;"
                            :: "r"((query_full_addr) & 0xFEFFFFFF), "r"((uint32_t)(32768)) : "memory");
                        asm volatile(
                            "cp.async.bulk.tensor.3d.shared::cluster.global.mbarrier::complete_tx::bytes.cta_group::2"
                            " [%0], [%1, {%2, %3, %4}], [%5];"
                            :: "r"(smem_query_addr), "l"(tmap_query), "r"(0), "r"(global_q), "r"(0),
                               "r"(((query_full_addr) & 0xFEFFFFFF)) : "memory");
                        #pragma unroll 1
                        for (int local_db_tile = 0; local_db_tile < db_tiles_per_split; local_db_tile++) {
                            int db_tile = db_tile_start + local_db_tile;
                            int off_m = db_tile * BLOCK_M;
                            int global_m = batch_idx * M + off_m;
                            mbarrier_wait(database_empty_addr, _phase_database_empty_0);
                            _phase_database_empty_0 ^= 1;
                            asm volatile(
                                "mbarrier.arrive.expect_tx.release.cta.shared::cluster.b64 _, [%0], %1;"
                                :: "r"((database_full_addr) & 0xFEFFFFFF), "r"((uint32_t)(16384)) : "memory");
                            asm volatile(
                                "cp.async.bulk.tensor.3d.shared::cluster.global.mbarrier::complete_tx::bytes.cta_group::2"
                                " [%0], [%1, {%2, %3, %4}], [%5];"
                                :: "r"(smem_database_addr), "l"(tmap_database), "r"(0), "r"(global_m), "r"(0),
                                   "r"(((database_full_addr) & 0xFEFFFFFF)) : "memory");
                        }
                    }
                }
            }
        }
    // ---- Role: mma ----
    } else if (warp == 1) {
        { // mma_main
            uint32_t _phase_query_full_0 = 0;
            uint32_t _phase_score_empty_0 = 1;
            uint32_t _phase_database_full_0 = 0;
            if (cta_rank == 0) {
                #pragma unroll 1
                for (unsigned int work_idx = cluster_id; work_idx < total_work; work_idx += num_clusters) {
                    mbarrier_wait(query_full_addr, _phase_query_full_0);
                    _phase_query_full_0 ^= 1;
                    #pragma unroll 1
                    for (int _local_db_tile = 0; _local_db_tile < db_tiles_per_split; _local_db_tile++) {
                        mbarrier_wait(score_empty_addr, _phase_score_empty_0);
                        _phase_score_empty_0 ^= 1;
                        mbarrier_wait(database_full_addr, _phase_database_full_0);
                        _phase_database_full_0 ^= 1;
                        asm volatile("tcgen05.fence::after_thread_sync;");
                        int _mma_ss_a_lo_0 = (smem_query_addr >> 4) & 0x3FFF;
                        int _mma_ss_b_lo_0 = (smem_database_addr >> 4) & 0x3FFF;
                        asm volatile(
                    "{\n\t"
                    ".reg .pred leader, p0, p1;\n\t"
                    ".reg .b32 adhi, bdhi, alo, blo, id, m0, m1, m2, m3, m4, m5, m6, m7;\n\t"
                    ".reg .b64 da, db;\n\t"
                    "elect.sync _|leader, 0xFFFFFFFF;\n\t"
                    "setp.ne.b32 p0, %3, 0;\n\t"
                    "setp.ne.b32 p1, 1, 0;\n\t"
                    "mov.b32 m0, 0; mov.b32 m1, 0; mov.b32 m2, 0; mov.b32 m3, 0;\n\tmov.b32 m4, 0; mov.b32 m5, 0; mov.b32 m6, 0; mov.b32 m7, 0;\n\t"
                    "mov.b32 adhi, 0x40004040;\n\t"
                    "mov.b32 bdhi, 0x40004040;\n\t"
                    "mov.b32 id, 270533776;\n\t"
                    "mov.b32 alo, %0;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p0;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "add.u32 alo, alo, 1018;\n\t"
                    "add.u32 blo, blo, 506;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "}\n"
                    :: "r"(_mma_ss_a_lo_0), "r"(_mma_ss_b_lo_0), "r"(taddr), "r"(0));
                        elect_commit_cg2_multicast(score_full_addr, (uint16_t)(3));
                        elect_commit_cg2_multicast(database_empty_addr, (uint16_t)(3));
                    }
                    elect_commit_cg2_multicast(query_empty_addr, (uint16_t)(3));
                }
            }
        }
    // ---- Role: compute ----
    } else if (warp >= 2 && warp <= 5) {
        const int warp_id_in_wg = warp % 4;
        const int my_row = warp_id_in_wg * 32 + lane;
        const int tmem_row_base_v = (warp_id_in_wg * 32) << 16;
        const int tmem_cross = taddr + TMEM_CROSS_OFFSET;
        { // compute_main
            uint32_t _phase_score_full_0 = 0;
            #pragma unroll 1
            for (unsigned int work_idx = cluster_id; work_idx < total_work; work_idx += num_clusters) {
                int split_idx = work_idx % split_count;
                int query_work = work_idx / split_count;
                int batch_idx = query_work / num_q_tile_pairs;
                int q_tile_pair = query_work % num_q_tile_pairs;
                int q_tile = q_tile_pair * 2 + cta_rank;
                int off_q = q_tile * BLOCK_Q;
                int q_idx = off_q + my_row;
                int valid_q = ((q_idx < Q) ? 1 : 0);
                float q_sq_val = 0.0f;
                if (valid_q != 0) {
                    q_sq_val = (float)query_sq[batch_idx * Q + q_idx];
                }
                float best_d[TOP_K_MAX];
                int best_i[TOP_K_MAX];
                #pragma unroll
                for (int kk = 0; kk < TOP_K_MAX; kk++) {
                    best_d[kk] = 3.4e+38f;
                    best_i[kk] = -1;
                }
                float worst_d = 3.4e+38f;
                int worst_pos = 0;
                int db_tile_start = split_idx * db_tiles_per_split;
                #pragma unroll 1
                for (int local_db_tile = 0; local_db_tile < db_tiles_per_split; local_db_tile++) {
                    int db_tile = db_tile_start + local_db_tile;
                    int db_start = db_tile * BLOCK_M;
                    int db_sq_idx = db_start + my_row;
                    if (my_row < BLOCK_M) {
                        if (db_sq_idx < M) {
                            smem_database_sq[my_row] = (float)database_sq[batch_idx * M + db_sq_idx];
                        } else {
                            smem_database_sq[my_row] = 0.0f;
                        }
                    }
                    asm volatile("barrier.sync 8, 128;");
                    mbarrier_wait(score_full_addr, _phase_score_full_0);
                    _phase_score_full_0 ^= 1;
                    int cross_addr = taddr + (cta_rank * BLOCK_Q + tmem_row_base_v << 16);
                    float dots[64];
                    asm volatile(
                        "tcgen05.ld.sync.aligned.32x32b.x64.b32"
                        " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63}, [%64];"
                        : "=f"(dots[0]), "=f"(dots[1]), "=f"(dots[2]), "=f"(dots[3]), "=f"(dots[4]), "=f"(dots[5]), "=f"(dots[6]), "=f"(dots[7]), "=f"(dots[8]), "=f"(dots[9]), "=f"(dots[10]), "=f"(dots[11]), "=f"(dots[12]), "=f"(dots[13]), "=f"(dots[14]), "=f"(dots[15]), "=f"(dots[16]), "=f"(dots[17]), "=f"(dots[18]), "=f"(dots[19]), "=f"(dots[20]), "=f"(dots[21]), "=f"(dots[22]), "=f"(dots[23]), "=f"(dots[24]), "=f"(dots[25]), "=f"(dots[26]), "=f"(dots[27]), "=f"(dots[28]), "=f"(dots[29]), "=f"(dots[30]), "=f"(dots[31]), "=f"(dots[32]), "=f"(dots[33]), "=f"(dots[34]), "=f"(dots[35]), "=f"(dots[36]), "=f"(dots[37]), "=f"(dots[38]), "=f"(dots[39]), "=f"(dots[40]), "=f"(dots[41]), "=f"(dots[42]), "=f"(dots[43]), "=f"(dots[44]), "=f"(dots[45]), "=f"(dots[46]), "=f"(dots[47]), "=f"(dots[48]), "=f"(dots[49]), "=f"(dots[50]), "=f"(dots[51]), "=f"(dots[52]), "=f"(dots[53]), "=f"(dots[54]), "=f"(dots[55]), "=f"(dots[56]), "=f"(dots[57]), "=f"(dots[58]), "=f"(dots[59]), "=f"(dots[60]), "=f"(dots[61]), "=f"(dots[62]), "=f"(dots[63])
                        : "r"(cross_addr)
                        : "memory");
                    asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                    asm volatile("barrier.sync 8, 128;");
                    if (elect_sync()) {
                        asm volatile(
                            "mbarrier.arrive.release.cta.shared::cluster.b64 _, [%0];"
                            :: "r"((score_empty_addr) & 0xFEFFFFFF) : "memory");
                    }
                    if (valid_q != 0) {
                        #pragma unroll 2
                        for (int col_base = 0; col_base < 64; col_base += 4) {
                            float dist_vec[4];
                            dist_vec[0] = dots[col_base];
                            dist_vec[1] = dots[col_base + 1];
                            dist_vec[2] = dots[col_base + 2];
                            dist_vec[3] = dots[col_base + 3];
                            const float2 _fma_b2_0 = {-2.0f, -2.0f};
                            const float2 _fma_c2_1 = {q_sq_val, q_sq_val};
                            #pragma unroll
                            for (int _lf = 0; _lf < 2; _lf++)
                                fma_f32x2_inplace(&reinterpret_cast<float2*>(dist_vec)[_lf], _fma_b2_0, _fma_c2_1);
                            float db_sq_vec[4];
                            db_sq_vec[0] = smem_database_sq[col_base];
                            db_sq_vec[1] = smem_database_sq[col_base + 1];
                            db_sq_vec[2] = smem_database_sq[col_base + 2];
                            db_sq_vec[3] = smem_database_sq[col_base + 3];
                            float _t0[4];
                            #pragma unroll
                            for (int _la = 0; _la < 2; _la++)
                                reinterpret_cast<float2*>(_t0)[_la] = add_f32x2(reinterpret_cast<float2*>(dist_vec)[_la], reinterpret_cast<const float2*>(db_sq_vec)[_la]);
                            float group_min = _t0[0];
                            if (_t0[1] < group_min) {
                                group_min = _t0[1];
                            }
                            if (_t0[2] < group_min) {
                                group_min = _t0[2];
                            }
                            if (_t0[3] < group_min) {
                                group_min = _t0[3];
                            }
                            if (group_min < worst_d) {
                                #pragma unroll
                                for (int vec_col = 0; vec_col < 4; vec_col++) {
                                    int db_idx = db_start + col_base + vec_col;
                                    if (db_idx < M) {
                                        float dist = _t0[vec_col];
                                        if (dist < worst_d) {
                                            best_d[worst_pos] = dist;
                                            best_i[worst_pos] = db_idx;
                                            worst_d = best_d[0];
                                            worst_pos = 0;
                                            #pragma unroll
                                            for (int scan_pos = 1; scan_pos < TOP_K_MAX; scan_pos++) {
                                                if (best_d[scan_pos] > worst_d) {
                                                    worst_d = best_d[scan_pos];
                                                    worst_pos = scan_pos;
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                    asm volatile("barrier.sync 8, 128;");
                }
                if (valid_q != 0) {
                    int out_base = ((split_idx * B + batch_idx) * Q + q_idx) * K;
                    #pragma unroll
                    for (int out_k = 0; out_k < TOP_K_MAX; out_k++) {
                        *((float*)(partial_dists + out_base + out_k)) = best_d[out_k];
                        *((int*)(partial_indices + out_base + out_k)) = best_i[out_k];
                    }
                }
            }
        }
    }

    // Cleanup
    asm volatile("barrier.cluster.arrive.release.aligned;");
    asm volatile("barrier.cluster.wait.acquire.aligned;");

    if (warp == 1) {
        asm volatile("tcgen05.dealloc.cta_group::2.sync.aligned.b32 %0, %1;" :: "r"(tmem_addr_storage[0]), "r"(64));
        asm volatile("tcgen05.relinquish_alloc_permit.cta_group::2.sync.aligned;");
    }
}

} // extern "C"

#undef BLOCK_M
#undef BLOCK_Q
#undef FEAT_D
#undef NUM_MAIN_STAGES
#undef SMEM_SMEM_DATABASE_OFF
#undef SMEM_SMEM_DATABASE_SQ_OFF
#undef SMEM_SMEM_DATABASE_SQ_STAGE_BYTES
#undef SMEM_SMEM_DATABASE_SQ_STRIDE
#undef SMEM_SMEM_DATABASE_STAGE_BYTES
#undef SMEM_SMEM_DATABASE_STRIDE
#undef SMEM_SMEM_QUERY_OFF
#undef SMEM_SMEM_QUERY_STAGE_BYTES
#undef SMEM_SMEM_QUERY_STRIDE
#undef SMEM_TOTAL
#undef THREADS
#undef TMEM_CROSS_OFFSET
#undef TMEM_NCOLS
#undef TOP_K_MAX
#undef database_empty_addr
#undef database_full_addr
#undef query_empty_addr
#undef query_full_addr
#undef score_empty_addr
#undef score_full_addr
#undef smem_database_addr
#undef smem_database_sq_addr
#undef smem_query_addr

#define NUM_MAIN_STAGES 1
#define THREADS 128
#define TOP_K_MAX 32

extern "C" {

__global__ __launch_bounds__(128, 1) void
kernel_knn_build_large_square_k32_s2_warp_select(float* __restrict__ partial_dists, int32_t* __restrict__ partial_indices, float* __restrict__ out_dists, int32_t* __restrict__ out_indices, int total_queries)
{
    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;


    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    const int warp_id = warp;
    const int lane_id = lane;

    // === Task calls (dependency order) ===
    int row = bid * 4 + warp;
    int base_row = row * TOP_K_MAX;
    int split_stride = total_queries * TOP_K_MAX;
    int cand_k = lane;
    if (row < total_queries) {
        float d0 = (float)partial_dists[base_row + cand_k];
        int i0 = partial_indices[base_row + cand_k];
        int base1 = base_row + split_stride;
        float d1 = (float)partial_dists[base1 + cand_k];
        int i1 = partial_indices[base1 + cand_k];
        #pragma unroll
        for (int out_k = 0; out_k < TOP_K_MAX; out_k++) {
            float winner_d = d0;
            int winner_i = i0;
            int winner_src = 0;
            if (d1 < winner_d) {
                winner_d = d1;
                winner_i = i1;
                winner_src = 1;
            }
            float warp_min = winner_d;
            #pragma unroll
            for (int offset = 16; offset > 0; offset >>= 1)
                warp_min = fminf(warp_min, __shfl_xor_sync(0xFFFFFFFF, warp_min, offset));
            int _vote_0 = __ballot_sync(0xFFFFFFFF, winner_d == warp_min);
            int owner_ballot = _vote_0;
            int _ffs_0 = __ffs(owner_ballot);
            int winner_lane = _ffs_0 - 1;
            winner_i = __shfl_sync(0xFFFFFFFF, winner_i, winner_lane);
            winner_src = __shfl_sync(0xFFFFFFFF, winner_src, winner_lane);
            if (lane == 0) {
                *((float*)(out_dists + base_row + out_k)) = warp_min;
                *((int*)(out_indices + base_row + out_k)) = winner_i;
            }
            if (lane == winner_lane) {
                if (winner_src == 0) {
                    d0 = 3.4e+38f;
                } else {
                    d1 = 3.4e+38f;
                }
            }
        }
    }
}

} // extern "C"

#undef NUM_MAIN_STAGES
#undef THREADS
#undef TOP_K_MAX

#define NUM_MAIN_STAGES 1
#define THREADS 32
#define TOP_K_MAX 16
#define SPLIT_COUNT 8

extern "C" {

__global__ __launch_bounds__(32, 1) void
kernel_knn_build_evolve_7bfc_k30_merge_s8_rowbase_cache_f8c3lowk_k16s8(float* __restrict__ partial_dists, int32_t* __restrict__ partial_indices, float* __restrict__ out_dists, int32_t* __restrict__ out_indices, int total_queries)
{
    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;


    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    const int warp_id = warp;
    const int lane_id = lane;

    // === Task calls (dependency order) ===
    int start_row = bid * 32 + tid;
    int stride = num_bids * 32;
    #pragma unroll 1
    for (int row = start_row; row < total_queries; row += stride) {
        int base_row = row * TOP_K_MAX;
        int split_stride = total_queries * TOP_K_MAX;
        int out_base = base_row;
        int split_pos[SPLIT_COUNT];
        int split_base[SPLIT_COUNT];
        float cand_d[SPLIT_COUNT];
        int cand_i[SPLIT_COUNT];
        #pragma unroll
        for (int split_idx = 0; split_idx < SPLIT_COUNT; split_idx++) {
            split_pos[split_idx] = 0;
            split_base[split_idx] = base_row + split_idx * split_stride;
            cand_d[split_idx] = (float)partial_dists[split_base[split_idx]];
            cand_i[split_idx] = partial_indices[split_base[split_idx]];
        }
        #pragma unroll
        for (int out_k = 0; out_k < TOP_K_MAX; out_k++) {
            float best_d = cand_d[0];
            int best_i = cand_i[0];
            int best_split = 0;
            #pragma unroll
            for (int split_idx = 1; split_idx < SPLIT_COUNT; split_idx++) {
                if (cand_d[split_idx] < best_d) {
                    best_d = cand_d[split_idx];
                    best_i = cand_i[split_idx];
                    best_split = split_idx;
                }
            }
            *((float*)(out_dists + out_base + out_k)) = best_d;
            *((int*)(out_indices + out_base + out_k)) = best_i;
            split_pos[best_split] = split_pos[best_split] + 1;
            if (out_k + 1 < TOP_K_MAX) {
                int next_pos = split_pos[best_split];
                int next_addr = split_base[best_split] + next_pos;
                cand_d[best_split] = (float)partial_dists[next_addr];
                cand_i[best_split] = partial_indices[next_addr];
            }
        }
    }
}

} // extern "C"

#undef NUM_MAIN_STAGES
#undef SPLIT_COUNT
#undef THREADS
#undef TOP_K_MAX

#define NUM_MAIN_STAGES 1
#define THREADS 32
#define TOP_K_MAX 16
#define SPLIT_COUNT 16

extern "C" {

__global__ __launch_bounds__(32, 1) void
kernel_knn_build_evolve_7bfc_k30_merge_s8_rowbase_cache_f8c3lowk_k16s16(float* __restrict__ partial_dists, int32_t* __restrict__ partial_indices, float* __restrict__ out_dists, int32_t* __restrict__ out_indices, int total_queries)
{
    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;


    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    const int warp_id = warp;
    const int lane_id = lane;

    // === Task calls (dependency order) ===
    int start_row = bid * 32 + tid;
    int stride = num_bids * 32;
    #pragma unroll 1
    for (int row = start_row; row < total_queries; row += stride) {
        int base_row = row * TOP_K_MAX;
        int split_stride = total_queries * TOP_K_MAX;
        int out_base = base_row;
        int split_pos[SPLIT_COUNT];
        int split_base[SPLIT_COUNT];
        float cand_d[SPLIT_COUNT];
        int cand_i[SPLIT_COUNT];
        #pragma unroll
        for (int split_idx = 0; split_idx < SPLIT_COUNT; split_idx++) {
            split_pos[split_idx] = 0;
            split_base[split_idx] = base_row + split_idx * split_stride;
            cand_d[split_idx] = (float)partial_dists[split_base[split_idx]];
            cand_i[split_idx] = partial_indices[split_base[split_idx]];
        }
        #pragma unroll
        for (int out_k = 0; out_k < TOP_K_MAX; out_k++) {
            float best_d = cand_d[0];
            int best_i = cand_i[0];
            int best_split = 0;
            #pragma unroll
            for (int split_idx = 1; split_idx < SPLIT_COUNT; split_idx++) {
                if (cand_d[split_idx] < best_d) {
                    best_d = cand_d[split_idx];
                    best_i = cand_i[split_idx];
                    best_split = split_idx;
                }
            }
            *((float*)(out_dists + out_base + out_k)) = best_d;
            *((int*)(out_indices + out_base + out_k)) = best_i;
            split_pos[best_split] = split_pos[best_split] + 1;
            if (out_k + 1 < TOP_K_MAX) {
                int next_pos = split_pos[best_split];
                int next_addr = split_base[best_split] + next_pos;
                cand_d[best_split] = (float)partial_dists[next_addr];
                cand_i[best_split] = partial_indices[next_addr];
            }
        }
    }
}

} // extern "C"

#undef NUM_MAIN_STAGES
#undef SPLIT_COUNT
#undef THREADS
#undef TOP_K_MAX

#define TMEM_NCOLS 64
#define TMEM_CROSS_OFFSET 0
#define NUM_MAIN_STAGES 1
#define SMEM_SMEM_QUERY_OFF 1024
#define SMEM_SMEM_QUERY_STAGE_BYTES 32768
#define SMEM_SMEM_QUERY_STRIDE 32768
#define SMEM_SMEM_DATABASE_OFF 33792
#define SMEM_SMEM_DATABASE_STAGE_BYTES 16384
#define SMEM_SMEM_DATABASE_STRIDE 16384
#define SMEM_SMEM_DATABASE_SQ_OFF 50176
#define SMEM_SMEM_DATABASE_SQ_STAGE_BYTES 256
#define SMEM_SMEM_DATABASE_SQ_STRIDE 256
#define SMEM_TOTAL 50432
#define THREADS 192
#define BLOCK_Q 128
#define BLOCK_M 64
#define FEAT_D 128
#define TOP_K_MAX 96

extern "C" {

__global__ __launch_bounds__(192, 1) void
kernel_knn_build_evolve_7bfc_split_cg2_u2_stage1_k32_unordered_k96over64(float* __restrict__ query_sq, float* __restrict__ database_sq, float* __restrict__ partial_dists, int32_t* __restrict__ partial_indices, const void* tmap_query, const void* tmap_database, int B, int Q, int M, int K, int num_q_tile_pairs, int db_tiles_per_split, int split_count, int total_work)
{
    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;

    extern __shared__ __align__(1024) char smem_raw[];
    int smem;
    smem = (int)(unsigned long long)__cvta_generic_to_shared(smem_raw);
    const int smem_smem_query = smem + 1024;
    const int smem_smem_database = smem + 33792;
    const int smem_smem_database_sq = smem + 50176;

    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;
    const unsigned int clusters_x = gridDim.x / 2;
    const unsigned int cluster_id = ((blockIdx.z * gridDim.y + blockIdx.y) * clusters_x) + blockIdx.x / 2;
    const unsigned int num_clusters = clusters_x * gridDim.y * gridDim.z;

    int cta_rank;
    asm volatile("mov.b32 %0, %%cluster_ctarank;" : "=r"(cta_rank));

    // Mbarrier init (6 groups, 6 barriers)
    // Mbarriers at smem_raw[0..48)

    if (warp == 0) {
        uint32_t leader = elect_sync();
        // query_full: 1 barriers, init_count=2
        mbarrier_init_pred(smem + 0, 2, leader);
        // query_empty: 1 barriers, init_count=1
        mbarrier_init_pred(smem + 8, 1, leader);
        // database_full: 1 barriers, init_count=2
        mbarrier_init_pred(smem + 16, 2, leader);
        // database_empty: 1 barriers, init_count=1
        mbarrier_init_pred(smem + 24, 1, leader);
        // score_full: 1 barriers, init_count=1
        mbarrier_init_pred(smem + 32, 1, leader);
        // score_empty: 1 barriers, init_count=8
        mbarrier_init_pred(smem + 40, 8, leader);
        asm volatile("fence.mbarrier_init.release.cluster;");
    }

    __syncthreads();

    // TMEM alloc (64 columns, 64 used)
    volatile int* tmem_addr_storage = (volatile int*)(smem_raw + 48);
    if (warp == 1) {
        int _tmem_hold = smem + 48;
        asm volatile("tcgen05.alloc.cta_group::2.sync.aligned.shared::cta.b32 [%0], %1;" :: "r"(_tmem_hold), "r"(64) : "memory");
    }

    asm volatile("barrier.cluster.arrive.release.aligned;");
    asm volatile("barrier.cluster.wait.acquire.aligned;");
    asm volatile("tcgen05.fence::after_thread_sync;");

    const int warp_id = warp;
    const int lane_id = lane;
    __nv_bfloat16* smem_query = (__nv_bfloat16*)(smem_raw + 1024);
    #define smem_query_addr (smem + 1024)
    __nv_bfloat16* smem_database = (__nv_bfloat16*)(smem_raw + 33792);
    #define smem_database_addr (smem + 33792)
    float* smem_database_sq = (float*)(smem_raw + 50176);
    #define smem_database_sq_addr (smem + 50176)
    const int mbar_base = smem;
    #define query_full_addr (mbar_base + 0)
    #define query_empty_addr (mbar_base + 8)
    #define database_full_addr (mbar_base + 16)
    #define database_empty_addr (mbar_base + 24)
    #define score_full_addr (mbar_base + 32)
    #define score_empty_addr (mbar_base + 40)
    const int taddr = tmem_addr_storage[0];

    // ---- Role: load ----
    if (warp == 0) {
        { // load_main
            uint32_t _phase_query_empty_0 = 1;
            uint32_t _phase_database_empty_0 = 1;
            if (warp_id == 0) {
                if (elect_sync()) {
                    #pragma unroll 1
                    for (unsigned int work_idx = cluster_id; work_idx < total_work; work_idx += num_clusters) {
                        int split_idx = work_idx % split_count;
                        int query_work = work_idx / split_count;
                        int batch_idx = query_work / num_q_tile_pairs;
                        int q_tile_pair = query_work % num_q_tile_pairs;
                        int q_tile = q_tile_pair * 2 + cta_rank;
                        int off_q = q_tile * BLOCK_Q;
                        int global_q = batch_idx * Q + off_q;
                        int db_tile_start = split_idx * db_tiles_per_split;
                        mbarrier_wait(query_empty_addr, _phase_query_empty_0);
                        _phase_query_empty_0 ^= 1;
                        asm volatile(
                            "mbarrier.arrive.expect_tx.release.cta.shared::cluster.b64 _, [%0], %1;"
                            :: "r"((query_full_addr) & 0xFEFFFFFF), "r"((uint32_t)(32768)) : "memory");
                        asm volatile(
                            "cp.async.bulk.tensor.3d.shared::cluster.global.mbarrier::complete_tx::bytes.cta_group::2"
                            " [%0], [%1, {%2, %3, %4}], [%5];"
                            :: "r"(smem_query_addr), "l"(tmap_query), "r"(0), "r"(global_q), "r"(0),
                               "r"(((query_full_addr) & 0xFEFFFFFF)) : "memory");
                        #pragma unroll 1
                        for (int local_db_tile = 0; local_db_tile < db_tiles_per_split; local_db_tile++) {
                            int db_tile = db_tile_start + local_db_tile;
                            int off_m = db_tile * BLOCK_M;
                            int global_m = batch_idx * M + off_m;
                            mbarrier_wait(database_empty_addr, _phase_database_empty_0);
                            _phase_database_empty_0 ^= 1;
                            asm volatile(
                                "mbarrier.arrive.expect_tx.release.cta.shared::cluster.b64 _, [%0], %1;"
                                :: "r"((database_full_addr) & 0xFEFFFFFF), "r"((uint32_t)(16384)) : "memory");
                            asm volatile(
                                "cp.async.bulk.tensor.3d.shared::cluster.global.mbarrier::complete_tx::bytes.cta_group::2"
                                " [%0], [%1, {%2, %3, %4}], [%5];"
                                :: "r"(smem_database_addr), "l"(tmap_database), "r"(0), "r"(global_m), "r"(0),
                                   "r"(((database_full_addr) & 0xFEFFFFFF)) : "memory");
                        }
                    }
                }
            }
        }
    // ---- Role: mma ----
    } else if (warp == 1) {
        { // mma_main
            uint32_t _phase_query_full_0 = 0;
            uint32_t _phase_score_empty_0 = 1;
            uint32_t _phase_database_full_0 = 0;
            if (cta_rank == 0) {
                #pragma unroll 1
                for (unsigned int work_idx = cluster_id; work_idx < total_work; work_idx += num_clusters) {
                    mbarrier_wait(query_full_addr, _phase_query_full_0);
                    _phase_query_full_0 ^= 1;
                    #pragma unroll 1
                    for (int _local_db_tile = 0; _local_db_tile < db_tiles_per_split; _local_db_tile++) {
                        mbarrier_wait(score_empty_addr, _phase_score_empty_0);
                        _phase_score_empty_0 ^= 1;
                        mbarrier_wait(database_full_addr, _phase_database_full_0);
                        _phase_database_full_0 ^= 1;
                        asm volatile("tcgen05.fence::after_thread_sync;");
                        int _mma_ss_a_lo_0 = (smem_query_addr >> 4) & 0x3FFF;
                        int _mma_ss_b_lo_0 = (smem_database_addr >> 4) & 0x3FFF;
                        asm volatile(
                    "{\n\t"
                    ".reg .pred leader, p0, p1;\n\t"
                    ".reg .b32 adhi, bdhi, alo, blo, id, m0, m1, m2, m3, m4, m5, m6, m7;\n\t"
                    ".reg .b64 da, db;\n\t"
                    "elect.sync _|leader, 0xFFFFFFFF;\n\t"
                    "setp.ne.b32 p0, %3, 0;\n\t"
                    "setp.ne.b32 p1, 1, 0;\n\t"
                    "mov.b32 m0, 0; mov.b32 m1, 0; mov.b32 m2, 0; mov.b32 m3, 0;\n\tmov.b32 m4, 0; mov.b32 m5, 0; mov.b32 m6, 0; mov.b32 m7, 0;\n\t"
                    "mov.b32 adhi, 0x40004040;\n\t"
                    "mov.b32 bdhi, 0x40004040;\n\t"
                    "mov.b32 id, 270533776;\n\t"
                    "mov.b32 alo, %0;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p0;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "add.u32 alo, alo, 1018;\n\t"
                    "add.u32 blo, blo, 506;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "}\n"
                    :: "r"(_mma_ss_a_lo_0), "r"(_mma_ss_b_lo_0), "r"(taddr), "r"(0));
                        elect_commit_cg2_multicast(score_full_addr, (uint16_t)(3));
                        elect_commit_cg2_multicast(database_empty_addr, (uint16_t)(3));
                    }
                    elect_commit_cg2_multicast(query_empty_addr, (uint16_t)(3));
                }
            }
        }
    // ---- Role: compute ----
    } else if (warp >= 2 && warp <= 5) {
        const int warp_id_in_wg = warp % 4;
        const int my_row = warp_id_in_wg * 32 + lane;
        const int tmem_row_base_v = (warp_id_in_wg * 32) << 16;
        const int tmem_cross = taddr + TMEM_CROSS_OFFSET;
        { // compute_main
            uint32_t _phase_score_full_0 = 0;
            #pragma unroll 1
            for (unsigned int work_idx = cluster_id; work_idx < total_work; work_idx += num_clusters) {
                int split_idx = work_idx % split_count;
                int query_work = work_idx / split_count;
                int batch_idx = query_work / num_q_tile_pairs;
                int q_tile_pair = query_work % num_q_tile_pairs;
                int q_tile = q_tile_pair * 2 + cta_rank;
                int off_q = q_tile * BLOCK_Q;
                int q_idx = off_q + my_row;
                int valid_q = ((q_idx < Q) ? 1 : 0);
                float q_sq_val = 0.0f;
                if (valid_q != 0) {
                    q_sq_val = (float)query_sq[batch_idx * Q + q_idx];
                }
                float best_d[TOP_K_MAX];
                int best_i[TOP_K_MAX];
                #pragma unroll
                for (int kk = 0; kk < TOP_K_MAX; kk++) {
                    best_d[kk] = 3.4e+38f;
                    best_i[kk] = -1;
                }
                float worst_d = 3.4e+38f;
                int worst_pos = 0;
                int db_tile_start = split_idx * db_tiles_per_split;
                #pragma unroll 1
                for (int local_db_tile = 0; local_db_tile < db_tiles_per_split; local_db_tile++) {
                    int db_tile = db_tile_start + local_db_tile;
                    int db_start = db_tile * BLOCK_M;
                    int db_sq_idx = db_start + my_row;
                    if (my_row < BLOCK_M) {
                        if (db_sq_idx < M) {
                            smem_database_sq[my_row] = (float)database_sq[batch_idx * M + db_sq_idx];
                        } else {
                            smem_database_sq[my_row] = 0.0f;
                        }
                    }
                    asm volatile("barrier.sync 8, 128;");
                    mbarrier_wait(score_full_addr, _phase_score_full_0);
                    _phase_score_full_0 ^= 1;
                    int cross_addr = taddr + (cta_rank * BLOCK_Q + tmem_row_base_v << 16);
                    float dots[64];
                    asm volatile(
                        "tcgen05.ld.sync.aligned.32x32b.x64.b32"
                        " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63}, [%64];"
                        : "=f"(dots[0]), "=f"(dots[1]), "=f"(dots[2]), "=f"(dots[3]), "=f"(dots[4]), "=f"(dots[5]), "=f"(dots[6]), "=f"(dots[7]), "=f"(dots[8]), "=f"(dots[9]), "=f"(dots[10]), "=f"(dots[11]), "=f"(dots[12]), "=f"(dots[13]), "=f"(dots[14]), "=f"(dots[15]), "=f"(dots[16]), "=f"(dots[17]), "=f"(dots[18]), "=f"(dots[19]), "=f"(dots[20]), "=f"(dots[21]), "=f"(dots[22]), "=f"(dots[23]), "=f"(dots[24]), "=f"(dots[25]), "=f"(dots[26]), "=f"(dots[27]), "=f"(dots[28]), "=f"(dots[29]), "=f"(dots[30]), "=f"(dots[31]), "=f"(dots[32]), "=f"(dots[33]), "=f"(dots[34]), "=f"(dots[35]), "=f"(dots[36]), "=f"(dots[37]), "=f"(dots[38]), "=f"(dots[39]), "=f"(dots[40]), "=f"(dots[41]), "=f"(dots[42]), "=f"(dots[43]), "=f"(dots[44]), "=f"(dots[45]), "=f"(dots[46]), "=f"(dots[47]), "=f"(dots[48]), "=f"(dots[49]), "=f"(dots[50]), "=f"(dots[51]), "=f"(dots[52]), "=f"(dots[53]), "=f"(dots[54]), "=f"(dots[55]), "=f"(dots[56]), "=f"(dots[57]), "=f"(dots[58]), "=f"(dots[59]), "=f"(dots[60]), "=f"(dots[61]), "=f"(dots[62]), "=f"(dots[63])
                        : "r"(cross_addr)
                        : "memory");
                    asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                    asm volatile("barrier.sync 8, 128;");
                    if (elect_sync()) {
                        asm volatile(
                            "mbarrier.arrive.release.cta.shared::cluster.b64 _, [%0];"
                            :: "r"((score_empty_addr) & 0xFEFFFFFF) : "memory");
                    }
                    if (valid_q != 0) {
                        #pragma unroll 2
                        for (int col_base = 0; col_base < 64; col_base += 4) {
                            float dist_vec[4];
                            dist_vec[0] = dots[col_base];
                            dist_vec[1] = dots[col_base + 1];
                            dist_vec[2] = dots[col_base + 2];
                            dist_vec[3] = dots[col_base + 3];
                            const float2 _fma_b2_0 = {-2.0f, -2.0f};
                            const float2 _fma_c2_1 = {q_sq_val, q_sq_val};
                            #pragma unroll
                            for (int _lf = 0; _lf < 2; _lf++)
                                fma_f32x2_inplace(&reinterpret_cast<float2*>(dist_vec)[_lf], _fma_b2_0, _fma_c2_1);
                            float db_sq_vec[4];
                            db_sq_vec[0] = smem_database_sq[col_base];
                            db_sq_vec[1] = smem_database_sq[col_base + 1];
                            db_sq_vec[2] = smem_database_sq[col_base + 2];
                            db_sq_vec[3] = smem_database_sq[col_base + 3];
                            float _t0[4];
                            #pragma unroll
                            for (int _la = 0; _la < 2; _la++)
                                reinterpret_cast<float2*>(_t0)[_la] = add_f32x2(reinterpret_cast<float2*>(dist_vec)[_la], reinterpret_cast<const float2*>(db_sq_vec)[_la]);
                            float group_min = _t0[0];
                            if (_t0[1] < group_min) {
                                group_min = _t0[1];
                            }
                            if (_t0[2] < group_min) {
                                group_min = _t0[2];
                            }
                            if (_t0[3] < group_min) {
                                group_min = _t0[3];
                            }
                            if (group_min < worst_d) {
                                #pragma unroll
                                for (int vec_col = 0; vec_col < 4; vec_col++) {
                                    int db_idx = db_start + col_base + vec_col;
                                    if (db_idx < M) {
                                        float dist = _t0[vec_col];
                                        if (dist < worst_d) {
                                            best_d[worst_pos] = dist;
                                            best_i[worst_pos] = db_idx;
                                            worst_d = best_d[0];
                                            worst_pos = 0;
                                            #pragma unroll
                                            for (int scan_pos = 1; scan_pos < TOP_K_MAX; scan_pos++) {
                                                if (best_d[scan_pos] > worst_d) {
                                                    worst_d = best_d[scan_pos];
                                                    worst_pos = scan_pos;
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                    asm volatile("barrier.sync 8, 128;");
                }
                if (valid_q != 0) {
                    int out_base = ((split_idx * B + batch_idx) * Q + q_idx) * K;
                    #pragma unroll
                    for (int out_k = 0; out_k < TOP_K_MAX; out_k++) {
                        *((float*)(partial_dists + out_base + out_k)) = best_d[out_k];
                        *((int*)(partial_indices + out_base + out_k)) = best_i[out_k];
                    }
                }
            }
        }
    }

    // Cleanup
    asm volatile("barrier.cluster.arrive.release.aligned;");
    asm volatile("barrier.cluster.wait.acquire.aligned;");

    if (warp == 1) {
        asm volatile("tcgen05.dealloc.cta_group::2.sync.aligned.b32 %0, %1;" :: "r"(tmem_addr_storage[0]), "r"(64));
        asm volatile("tcgen05.relinquish_alloc_permit.cta_group::2.sync.aligned;");
    }
}

} // extern "C"

#undef BLOCK_M
#undef BLOCK_Q
#undef FEAT_D
#undef NUM_MAIN_STAGES
#undef SMEM_SMEM_DATABASE_OFF
#undef SMEM_SMEM_DATABASE_SQ_OFF
#undef SMEM_SMEM_DATABASE_SQ_STAGE_BYTES
#undef SMEM_SMEM_DATABASE_SQ_STRIDE
#undef SMEM_SMEM_DATABASE_STAGE_BYTES
#undef SMEM_SMEM_DATABASE_STRIDE
#undef SMEM_SMEM_QUERY_OFF
#undef SMEM_SMEM_QUERY_STAGE_BYTES
#undef SMEM_SMEM_QUERY_STRIDE
#undef SMEM_TOTAL
#undef THREADS
#undef TMEM_CROSS_OFFSET
#undef TMEM_NCOLS
#undef TOP_K_MAX
#undef database_empty_addr
#undef database_full_addr
#undef query_empty_addr
#undef query_full_addr
#undef score_empty_addr
#undef score_full_addr
#undef smem_database_addr
#undef smem_database_sq_addr
#undef smem_query_addr

#define NUM_MAIN_STAGES 1
#define THREADS 32
#define TOP_K_MAX 96
#define SPLIT_COUNT 8

extern "C" {

__global__ __launch_bounds__(32, 1) void
kernel_knn_build_evolve_7bfc_k32_merge_s4_unordered_k96over64(float* __restrict__ partial_dists, int32_t* __restrict__ partial_indices, float* __restrict__ out_dists, int32_t* __restrict__ out_indices, int total_queries)
{
    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;


    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    const int warp_id = warp;
    const int lane_id = lane;

    // === Task calls (dependency order) ===
    int start_row = bid * 32 + tid;
    int stride = num_bids * 32;
    #pragma unroll 1
    for (int row = start_row; row < total_queries; row += stride) {
        int base_row = row * TOP_K_MAX;
        int split_stride = total_queries * TOP_K_MAX;
        float best_d[TOP_K_MAX];
        int best_i[TOP_K_MAX];
        #pragma unroll
        for (int kk = 0; kk < TOP_K_MAX; kk++) {
            best_d[kk] = 3.4e+38f;
            best_i[kk] = -1;
        }
        float worst_d = 3.4e+38f;
        int worst_pos = 0;
        #pragma unroll
        for (int split_idx = 0; split_idx < SPLIT_COUNT; split_idx++) {
            int partial_base = base_row + split_idx * split_stride;
            #pragma unroll
            for (int cand_k = 0; cand_k < TOP_K_MAX; cand_k++) {
                float cand_d = (float)partial_dists[partial_base + cand_k];
                int cand_i = partial_indices[partial_base + cand_k];
                if (cand_d < worst_d) {
                    best_d[worst_pos] = cand_d;
                    best_i[worst_pos] = cand_i;
                    worst_d = best_d[0];
                    worst_pos = 0;
                    #pragma unroll
                    for (int scan_pos = 1; scan_pos < TOP_K_MAX; scan_pos++) {
                        if (best_d[scan_pos] > worst_d) {
                            worst_d = best_d[scan_pos];
                            worst_pos = scan_pos;
                        }
                    }
                }
            }
        }
        #pragma unroll
        for (int out_k = 0; out_k < TOP_K_MAX; out_k++) {
            *((float*)(out_dists + base_row + out_k)) = best_d[out_k];
            *((int*)(out_indices + base_row + out_k)) = best_i[out_k];
        }
    }
}

} // extern "C"

#undef NUM_MAIN_STAGES
#undef SPLIT_COUNT
#undef THREADS
#undef TOP_K_MAX

#define TMEM_NCOLS 64
#define TMEM_CROSS_OFFSET 0
#define NUM_MAIN_STAGES 1
#define SMEM_SMEM_QUERY_OFF 1024
#define SMEM_SMEM_QUERY_STAGE_BYTES 32768
#define SMEM_SMEM_QUERY_STRIDE 32768
#define SMEM_SMEM_DATABASE_OFF 33792
#define SMEM_SMEM_DATABASE_STAGE_BYTES 16384
#define SMEM_SMEM_DATABASE_STRIDE 16384
#define SMEM_SMEM_DATABASE_SQ_OFF 50176
#define SMEM_SMEM_DATABASE_SQ_STAGE_BYTES 256
#define SMEM_SMEM_DATABASE_SQ_STRIDE 256
#define SMEM_TOTAL 50432
#define THREADS 192
#define BLOCK_Q 128
#define BLOCK_M 64
#define FEAT_D 128
#define TOP_K_MAX 96

extern "C" {

__global__ __launch_bounds__(192, 1) void
kernel_knn_build_k96_stage1_sort4_chunked(float* __restrict__ query_sq, float* __restrict__ database_sq, float* __restrict__ partial_dists, int32_t* __restrict__ partial_indices, const void* tmap_query, const void* tmap_database, int B, int Q, int M, int K, int num_q_tile_pairs, int db_tiles_per_split, int split_count, int total_work)
{
    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;

    extern __shared__ __align__(1024) char smem_raw[];
    int smem;
    smem = (int)(unsigned long long)__cvta_generic_to_shared(smem_raw);
    const int smem_smem_query = smem + 1024;
    const int smem_smem_database = smem + 33792;
    const int smem_smem_database_sq = smem + 50176;

    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;
    const unsigned int clusters_x = gridDim.x / 2;
    const unsigned int cluster_id = ((blockIdx.z * gridDim.y + blockIdx.y) * clusters_x) + blockIdx.x / 2;
    const unsigned int num_clusters = clusters_x * gridDim.y * gridDim.z;

    int cta_rank;
    asm volatile("mov.b32 %0, %%cluster_ctarank;" : "=r"(cta_rank));

    // Mbarrier init (6 groups, 6 barriers)
    // Mbarriers at smem_raw[0..48)

    if (warp == 0) {
        uint32_t leader = elect_sync();
        // query_full: 1 barriers, init_count=2
        mbarrier_init_pred(smem + 0, 2, leader);
        // query_empty: 1 barriers, init_count=1
        mbarrier_init_pred(smem + 8, 1, leader);
        // database_full: 1 barriers, init_count=2
        mbarrier_init_pred(smem + 16, 2, leader);
        // database_empty: 1 barriers, init_count=1
        mbarrier_init_pred(smem + 24, 1, leader);
        // score_full: 1 barriers, init_count=1
        mbarrier_init_pred(smem + 32, 1, leader);
        // score_empty: 1 barriers, init_count=8
        mbarrier_init_pred(smem + 40, 8, leader);
        asm volatile("fence.mbarrier_init.release.cluster;");
    }

    __syncthreads();

    // TMEM alloc (64 columns, 64 used)
    volatile int* tmem_addr_storage = (volatile int*)(smem_raw + 48);
    if (warp == 5) {
        int _tmem_hold = smem + 48;
        asm volatile("tcgen05.alloc.cta_group::2.sync.aligned.shared::cta.b32 [%0], %1;" :: "r"(_tmem_hold), "r"(64) : "memory");
    }

    asm volatile("barrier.cluster.arrive.release.aligned;");
    asm volatile("barrier.cluster.wait.acquire.aligned;");
    asm volatile("tcgen05.fence::after_thread_sync;");

    const int warp_id = warp;
    const int lane_id = lane;
    __nv_bfloat16* smem_query = (__nv_bfloat16*)(smem_raw + 1024);
    #define smem_query_addr (smem + 1024)
    __nv_bfloat16* smem_database = (__nv_bfloat16*)(smem_raw + 33792);
    #define smem_database_addr (smem + 33792)
    float* smem_database_sq = (float*)(smem_raw + 50176);
    #define smem_database_sq_addr (smem + 50176)
    const int mbar_base = smem;
    #define query_full_addr (mbar_base + 0)
    #define query_empty_addr (mbar_base + 8)
    #define database_full_addr (mbar_base + 16)
    #define database_empty_addr (mbar_base + 24)
    #define score_full_addr (mbar_base + 32)
    #define score_empty_addr (mbar_base + 40)
    const int taddr = tmem_addr_storage[0];

    // ---- Role: compute ----
    if (warp <= 3) {
        const int warp_id_in_wg = warp % 4;
        const int my_row = warp_id_in_wg * 32 + lane;
        const int tmem_row_base_v = (warp_id_in_wg * 32) << 16;
        const int tmem_cross = taddr + TMEM_CROSS_OFFSET;
        { // compute_main
            uint32_t _phase_score_full_0 = 0;
            #pragma unroll 1
            for (unsigned int work_idx = cluster_id; work_idx < total_work; work_idx += num_clusters) {
                int split_idx = work_idx % split_count;
                int query_work = work_idx / split_count;
                int batch_idx = query_work / num_q_tile_pairs;
                int q_tile_pair = query_work % num_q_tile_pairs;
                int q_tile = q_tile_pair * 2 + cta_rank;
                int off_q = q_tile * BLOCK_Q;
                int q_idx = off_q + my_row;
                int valid_q = ((q_idx < Q) ? 1 : 0);
                float q_sq_val = 0.0f;
                if (valid_q != 0) {
                    q_sq_val = (float)query_sq[batch_idx * Q + q_idx];
                }
                float best_d[TOP_K_MAX];
                int best_i[TOP_K_MAX];
                #pragma unroll
                for (int kk = 0; kk < TOP_K_MAX; kk++) {
                    best_d[kk] = 3.4e+38f;
                    best_i[kk] = -1;
                }
                float chunk_worst_d[24];
                int chunk_worst_pos[24];
                #pragma unroll
                for (int chunk = 0; chunk < 24; chunk++) {
                    int chunk_base = chunk * 4;
                    chunk_worst_d[chunk] = 3.4e+38f;
                    chunk_worst_pos[chunk] = chunk_base;
                }
                float worst_d = 3.4e+38f;
                int worst_pos = 0;
                int worst_chunk = 0;
                int db_tile_start = split_idx * db_tiles_per_split;
                #pragma unroll 1
                for (int local_db_tile = 0; local_db_tile < db_tiles_per_split; local_db_tile++) {
                    int db_tile = db_tile_start + local_db_tile;
                    int db_start = db_tile * BLOCK_M;
                    int db_sq_idx = db_start + my_row;
                    if (my_row < BLOCK_M) {
                        if (db_sq_idx < M) {
                            smem_database_sq[my_row] = (float)database_sq[batch_idx * M + db_sq_idx];
                        } else {
                            smem_database_sq[my_row] = 0.0f;
                        }
                    }
                    asm volatile("barrier.sync 8, 128;");
                    mbarrier_wait(score_full_addr, _phase_score_full_0);
                    _phase_score_full_0 ^= 1;
                    int cross_addr = taddr + (cta_rank * BLOCK_Q + tmem_row_base_v << 16);
                    float dots[64];
                    asm volatile(
                        "tcgen05.ld.sync.aligned.32x32b.x64.b32"
                        " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63}, [%64];"
                        : "=f"(dots[0]), "=f"(dots[1]), "=f"(dots[2]), "=f"(dots[3]), "=f"(dots[4]), "=f"(dots[5]), "=f"(dots[6]), "=f"(dots[7]), "=f"(dots[8]), "=f"(dots[9]), "=f"(dots[10]), "=f"(dots[11]), "=f"(dots[12]), "=f"(dots[13]), "=f"(dots[14]), "=f"(dots[15]), "=f"(dots[16]), "=f"(dots[17]), "=f"(dots[18]), "=f"(dots[19]), "=f"(dots[20]), "=f"(dots[21]), "=f"(dots[22]), "=f"(dots[23]), "=f"(dots[24]), "=f"(dots[25]), "=f"(dots[26]), "=f"(dots[27]), "=f"(dots[28]), "=f"(dots[29]), "=f"(dots[30]), "=f"(dots[31]), "=f"(dots[32]), "=f"(dots[33]), "=f"(dots[34]), "=f"(dots[35]), "=f"(dots[36]), "=f"(dots[37]), "=f"(dots[38]), "=f"(dots[39]), "=f"(dots[40]), "=f"(dots[41]), "=f"(dots[42]), "=f"(dots[43]), "=f"(dots[44]), "=f"(dots[45]), "=f"(dots[46]), "=f"(dots[47]), "=f"(dots[48]), "=f"(dots[49]), "=f"(dots[50]), "=f"(dots[51]), "=f"(dots[52]), "=f"(dots[53]), "=f"(dots[54]), "=f"(dots[55]), "=f"(dots[56]), "=f"(dots[57]), "=f"(dots[58]), "=f"(dots[59]), "=f"(dots[60]), "=f"(dots[61]), "=f"(dots[62]), "=f"(dots[63])
                        : "r"(cross_addr)
                        : "memory");
                    asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                    asm volatile("barrier.sync 8, 128;");
                    if (elect_sync()) {
                        asm volatile(
                            "mbarrier.arrive.release.cta.shared::cluster.b64 _, [%0];"
                            :: "r"((score_empty_addr) & 0xFEFFFFFF) : "memory");
                    }
                    if (valid_q != 0) {
                        #pragma unroll 1
                        for (int col_base = 0; col_base < 64; col_base += 4) {
                            float dist_vec[4];
                            dist_vec[0] = dots[col_base];
                            dist_vec[1] = dots[col_base + 1];
                            dist_vec[2] = dots[col_base + 2];
                            dist_vec[3] = dots[col_base + 3];
                            const float2 _fma_b2_0 = {-2.0f, -2.0f};
                            const float2 _fma_c2_1 = {q_sq_val, q_sq_val};
                            #pragma unroll
                            for (int _lf = 0; _lf < 2; _lf++)
                                fma_f32x2_inplace(&reinterpret_cast<float2*>(dist_vec)[_lf], _fma_b2_0, _fma_c2_1);
                            float db_sq_vec[4];
                            db_sq_vec[0] = smem_database_sq[col_base];
                            db_sq_vec[1] = smem_database_sq[col_base + 1];
                            db_sq_vec[2] = smem_database_sq[col_base + 2];
                            db_sq_vec[3] = smem_database_sq[col_base + 3];
                            float _t0[4];
                            #pragma unroll
                            for (int _la = 0; _la < 2; _la++)
                                reinterpret_cast<float2*>(_t0)[_la] = add_f32x2(reinterpret_cast<float2*>(dist_vec)[_la], reinterpret_cast<const float2*>(db_sq_vec)[_la]);
                            float group_min = _t0[0];
                            if (_t0[1] < group_min) {
                                group_min = _t0[1];
                            }
                            if (_t0[2] < group_min) {
                                group_min = _t0[2];
                            }
                            if (_t0[3] < group_min) {
                                group_min = _t0[3];
                            }
                            if (group_min < worst_d) {
                                float sort_d0 = _t0[0];
                                float sort_d1 = _t0[1];
                                float sort_d2 = _t0[2];
                                float sort_d3 = _t0[3];
                                int sort_col0 = 0;
                                int sort_col1 = 1;
                                int sort_col2 = 2;
                                int sort_col3 = 3;
                                float tmp_d = 0.0f;
                                int tmp_col = 0;
                                if (sort_d1 < sort_d0) {
                                    tmp_d = sort_d0;
                                    sort_d0 = sort_d1;
                                    sort_d1 = tmp_d;
                                    tmp_col = sort_col0;
                                    sort_col0 = sort_col1;
                                    sort_col1 = tmp_col;
                                }
                                if (sort_d3 < sort_d2) {
                                    tmp_d = sort_d2;
                                    sort_d2 = sort_d3;
                                    sort_d3 = tmp_d;
                                    tmp_col = sort_col2;
                                    sort_col2 = sort_col3;
                                    sort_col3 = tmp_col;
                                }
                                if (sort_d2 < sort_d0) {
                                    tmp_d = sort_d0;
                                    sort_d0 = sort_d2;
                                    sort_d2 = tmp_d;
                                    tmp_col = sort_col0;
                                    sort_col0 = sort_col2;
                                    sort_col2 = tmp_col;
                                }
                                if (sort_d3 < sort_d1) {
                                    tmp_d = sort_d1;
                                    sort_d1 = sort_d3;
                                    sort_d3 = tmp_d;
                                    tmp_col = sort_col1;
                                    sort_col1 = sort_col3;
                                    sort_col3 = tmp_col;
                                }
                                if (sort_d2 < sort_d1) {
                                    tmp_d = sort_d1;
                                    sort_d1 = sort_d2;
                                    sort_d2 = tmp_d;
                                    tmp_col = sort_col1;
                                    sort_col1 = sort_col2;
                                    sort_col2 = tmp_col;
                                }
                                #pragma unroll
                                for (int visit = 0; visit < 4; visit++) {
                                    int vec_col = sort_col0;
                                    float dist = sort_d0;
                                    if (visit == 1) {
                                        vec_col = sort_col1;
                                        dist = sort_d1;
                                    }
                                    if (visit == 2) {
                                        vec_col = sort_col2;
                                        dist = sort_d2;
                                    }
                                    if (visit == 3) {
                                        vec_col = sort_col3;
                                        dist = sort_d3;
                                    }
                                    if (dist >= worst_d) {
                                        break;
                                    }
                                    int db_idx = db_start + col_base + vec_col;
                                    if (db_idx < M) {
                                        best_d[worst_pos] = dist;
                                        best_i[worst_pos] = db_idx;
                                        int refresh_base = worst_chunk * 4;
                                        chunk_worst_d[worst_chunk] = best_d[refresh_base];
                                        chunk_worst_pos[worst_chunk] = refresh_base;
                                        #pragma unroll
                                        for (int offset = 1; offset < 4; offset++) {
                                            int scan_pos = refresh_base + offset;
                                            if (best_d[scan_pos] > chunk_worst_d[worst_chunk]) {
                                                chunk_worst_d[worst_chunk] = best_d[scan_pos];
                                                chunk_worst_pos[worst_chunk] = scan_pos;
                                            }
                                        }
                                        worst_d = chunk_worst_d[0];
                                        worst_pos = chunk_worst_pos[0];
                                        worst_chunk = 0;
                                        #pragma unroll
                                        for (int chunk = 1; chunk < 24; chunk++) {
                                            if (chunk_worst_d[chunk] > worst_d) {
                                                worst_d = chunk_worst_d[chunk];
                                                worst_pos = chunk_worst_pos[chunk];
                                                worst_chunk = chunk;
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                    asm volatile("barrier.sync 8, 128;");
                }
                if (valid_q != 0) {
                    int out_base = ((split_idx * B + batch_idx) * Q + q_idx) * K;
                    #pragma unroll
                    for (int out_k = 0; out_k < TOP_K_MAX; out_k++) {
                        *((float*)(partial_dists + out_base + out_k)) = best_d[out_k];
                        *((int*)(partial_indices + out_base + out_k)) = best_i[out_k];
                    }
                }
            }
        }
    // ---- Role: load ----
    } else if (warp == 4) {
        { // load_main
            uint32_t _phase_query_empty_0 = 1;
            uint32_t _phase_database_empty_0 = 1;
            if (warp_id == 4) {
                if (elect_sync()) {
                    #pragma unroll 1
                    for (unsigned int work_idx = cluster_id; work_idx < total_work; work_idx += num_clusters) {
                        int split_idx = work_idx % split_count;
                        int query_work = work_idx / split_count;
                        int batch_idx = query_work / num_q_tile_pairs;
                        int q_tile_pair = query_work % num_q_tile_pairs;
                        int q_tile = q_tile_pair * 2 + cta_rank;
                        int off_q = q_tile * BLOCK_Q;
                        int global_q = batch_idx * Q + off_q;
                        int db_tile_start = split_idx * db_tiles_per_split;
                        mbarrier_wait(query_empty_addr, _phase_query_empty_0);
                        _phase_query_empty_0 ^= 1;
                        asm volatile(
                            "mbarrier.arrive.expect_tx.release.cta.shared::cluster.b64 _, [%0], %1;"
                            :: "r"((query_full_addr) & 0xFEFFFFFF), "r"((uint32_t)(32768)) : "memory");
                        asm volatile(
                            "cp.async.bulk.tensor.3d.shared::cluster.global.mbarrier::complete_tx::bytes.cta_group::2"
                            " [%0], [%1, {%2, %3, %4}], [%5];"
                            :: "r"(smem_query_addr), "l"(tmap_query), "r"(0), "r"(global_q), "r"(0),
                               "r"(((query_full_addr) & 0xFEFFFFFF)) : "memory");
                        #pragma unroll 1
                        for (int local_db_tile = 0; local_db_tile < db_tiles_per_split; local_db_tile++) {
                            int db_tile = db_tile_start + local_db_tile;
                            int off_m = db_tile * BLOCK_M;
                            int global_m = batch_idx * M + off_m;
                            mbarrier_wait(database_empty_addr, _phase_database_empty_0);
                            _phase_database_empty_0 ^= 1;
                            asm volatile(
                                "mbarrier.arrive.expect_tx.release.cta.shared::cluster.b64 _, [%0], %1;"
                                :: "r"((database_full_addr) & 0xFEFFFFFF), "r"((uint32_t)(16384)) : "memory");
                            asm volatile(
                                "cp.async.bulk.tensor.3d.shared::cluster.global.mbarrier::complete_tx::bytes.cta_group::2"
                                " [%0], [%1, {%2, %3, %4}], [%5];"
                                :: "r"(smem_database_addr), "l"(tmap_database), "r"(0), "r"(global_m), "r"(0),
                                   "r"(((database_full_addr) & 0xFEFFFFFF)) : "memory");
                        }
                    }
                }
            }
        }
    // ---- Role: mma ----
    } else if (warp == 5) {
        { // mma_main
            uint32_t _phase_query_full_0 = 0;
            uint32_t _phase_score_empty_0 = 1;
            uint32_t _phase_database_full_0 = 0;
            if (cta_rank == 0) {
                #pragma unroll 1
                for (unsigned int work_idx = cluster_id; work_idx < total_work; work_idx += num_clusters) {
                    mbarrier_wait(query_full_addr, _phase_query_full_0);
                    _phase_query_full_0 ^= 1;
                    #pragma unroll 1
                    for (int _local_db_tile = 0; _local_db_tile < db_tiles_per_split; _local_db_tile++) {
                        mbarrier_wait(score_empty_addr, _phase_score_empty_0);
                        _phase_score_empty_0 ^= 1;
                        mbarrier_wait(database_full_addr, _phase_database_full_0);
                        _phase_database_full_0 ^= 1;
                        asm volatile("tcgen05.fence::after_thread_sync;");
                        int _mma_ss_a_lo_0 = (smem_query_addr >> 4) & 0x3FFF;
                        int _mma_ss_b_lo_0 = (smem_database_addr >> 4) & 0x3FFF;
                        asm volatile(
                    "{\n\t"
                    ".reg .pred leader, p0, p1;\n\t"
                    ".reg .b32 adhi, bdhi, alo, blo, id, m0, m1, m2, m3, m4, m5, m6, m7;\n\t"
                    ".reg .b64 da, db;\n\t"
                    "elect.sync _|leader, 0xFFFFFFFF;\n\t"
                    "setp.ne.b32 p0, %3, 0;\n\t"
                    "setp.ne.b32 p1, 1, 0;\n\t"
                    "mov.b32 m0, 0; mov.b32 m1, 0; mov.b32 m2, 0; mov.b32 m3, 0;\n\tmov.b32 m4, 0; mov.b32 m5, 0; mov.b32 m6, 0; mov.b32 m7, 0;\n\t"
                    "mov.b32 adhi, 0x40004040;\n\t"
                    "mov.b32 bdhi, 0x40004040;\n\t"
                    "mov.b32 id, 270533776;\n\t"
                    "mov.b32 alo, %0;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p0;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "add.u32 alo, alo, 1018;\n\t"
                    "add.u32 blo, blo, 506;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "}\n"
                    :: "r"(_mma_ss_a_lo_0), "r"(_mma_ss_b_lo_0), "r"(taddr), "r"(0));
                        elect_commit_cg2_multicast(score_full_addr, (uint16_t)(3));
                        elect_commit_cg2_multicast(database_empty_addr, (uint16_t)(3));
                    }
                    elect_commit_cg2_multicast(query_empty_addr, (uint16_t)(3));
                }
            }
        }
    }

    // Cleanup
    asm volatile("barrier.cluster.arrive.release.aligned;");
    asm volatile("barrier.cluster.wait.acquire.aligned;");

    if (warp == 5) {
        asm volatile("tcgen05.dealloc.cta_group::2.sync.aligned.b32 %0, %1;" :: "r"(tmem_addr_storage[0]), "r"(64));
        asm volatile("tcgen05.relinquish_alloc_permit.cta_group::2.sync.aligned;");
    }
}

} // extern "C"

#undef BLOCK_M
#undef BLOCK_Q
#undef FEAT_D
#undef NUM_MAIN_STAGES
#undef SMEM_SMEM_DATABASE_OFF
#undef SMEM_SMEM_DATABASE_SQ_OFF
#undef SMEM_SMEM_DATABASE_SQ_STAGE_BYTES
#undef SMEM_SMEM_DATABASE_SQ_STRIDE
#undef SMEM_SMEM_DATABASE_STAGE_BYTES
#undef SMEM_SMEM_DATABASE_STRIDE
#undef SMEM_SMEM_QUERY_OFF
#undef SMEM_SMEM_QUERY_STAGE_BYTES
#undef SMEM_SMEM_QUERY_STRIDE
#undef SMEM_TOTAL
#undef THREADS
#undef TMEM_CROSS_OFFSET
#undef TMEM_NCOLS
#undef TOP_K_MAX
#undef database_empty_addr
#undef database_full_addr
#undef query_empty_addr
#undef query_full_addr
#undef score_empty_addr
#undef score_full_addr
#undef smem_database_addr
#undef smem_database_sq_addr
#undef smem_query_addr

#define TMEM_NCOLS 64
#define TMEM_CROSS_OFFSET 0
#define NUM_MAIN_STAGES 1
#define SMEM_SMEM_QUERY_OFF 1024
#define SMEM_SMEM_QUERY_STAGE_BYTES 32768
#define SMEM_SMEM_QUERY_STRIDE 32768
#define SMEM_SMEM_DATABASE_OFF 33792
#define SMEM_SMEM_DATABASE_STAGE_BYTES 16384
#define SMEM_SMEM_DATABASE_STRIDE 16384
#define SMEM_SMEM_DATABASE_SQ_OFF 50176
#define SMEM_SMEM_DATABASE_SQ_STAGE_BYTES 256
#define SMEM_SMEM_DATABASE_SQ_STRIDE 256
#define SMEM_TOTAL 50432
#define THREADS 192
#define BLOCK_Q 128
#define BLOCK_M 64
#define FEAT_D 128
#define TOP_K_MAX 96

extern "C" {

__global__ __launch_bounds__(192, 1) void
kernel_knn_build_k96_stage1_sort4_chunked_k96over64sort4chunked(float* __restrict__ query_sq, float* __restrict__ database_sq, float* __restrict__ partial_dists, int32_t* __restrict__ partial_indices, const void* tmap_query, const void* tmap_database, int B, int Q, int M, int K, int num_q_tile_pairs, int db_tiles_per_split, int split_count, int total_work)
{
    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;

    extern __shared__ __align__(1024) char smem_raw[];
    int smem;
    smem = (int)(unsigned long long)__cvta_generic_to_shared(smem_raw);
    const int smem_smem_query = smem + 1024;
    const int smem_smem_database = smem + 33792;
    const int smem_smem_database_sq = smem + 50176;

    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;
    const unsigned int clusters_x = gridDim.x / 2;
    const unsigned int cluster_id = ((blockIdx.z * gridDim.y + blockIdx.y) * clusters_x) + blockIdx.x / 2;
    const unsigned int num_clusters = clusters_x * gridDim.y * gridDim.z;

    int cta_rank;
    asm volatile("mov.b32 %0, %%cluster_ctarank;" : "=r"(cta_rank));

    // Mbarrier init (6 groups, 6 barriers)
    // Mbarriers at smem_raw[0..48)

    if (warp == 0) {
        uint32_t leader = elect_sync();
        // query_full: 1 barriers, init_count=2
        mbarrier_init_pred(smem + 0, 2, leader);
        // query_empty: 1 barriers, init_count=1
        mbarrier_init_pred(smem + 8, 1, leader);
        // database_full: 1 barriers, init_count=2
        mbarrier_init_pred(smem + 16, 2, leader);
        // database_empty: 1 barriers, init_count=1
        mbarrier_init_pred(smem + 24, 1, leader);
        // score_full: 1 barriers, init_count=1
        mbarrier_init_pred(smem + 32, 1, leader);
        // score_empty: 1 barriers, init_count=8
        mbarrier_init_pred(smem + 40, 8, leader);
        asm volatile("fence.mbarrier_init.release.cluster;");
    }

    __syncthreads();

    // TMEM alloc (64 columns, 64 used)
    volatile int* tmem_addr_storage = (volatile int*)(smem_raw + 48);
    if (warp == 5) {
        int _tmem_hold = smem + 48;
        asm volatile("tcgen05.alloc.cta_group::2.sync.aligned.shared::cta.b32 [%0], %1;" :: "r"(_tmem_hold), "r"(64) : "memory");
    }

    asm volatile("barrier.cluster.arrive.release.aligned;");
    asm volatile("barrier.cluster.wait.acquire.aligned;");
    asm volatile("tcgen05.fence::after_thread_sync;");

    const int warp_id = warp;
    const int lane_id = lane;
    __nv_bfloat16* smem_query = (__nv_bfloat16*)(smem_raw + 1024);
    #define smem_query_addr (smem + 1024)
    __nv_bfloat16* smem_database = (__nv_bfloat16*)(smem_raw + 33792);
    #define smem_database_addr (smem + 33792)
    float* smem_database_sq = (float*)(smem_raw + 50176);
    #define smem_database_sq_addr (smem + 50176)
    const int mbar_base = smem;
    #define query_full_addr (mbar_base + 0)
    #define query_empty_addr (mbar_base + 8)
    #define database_full_addr (mbar_base + 16)
    #define database_empty_addr (mbar_base + 24)
    #define score_full_addr (mbar_base + 32)
    #define score_empty_addr (mbar_base + 40)
    const int taddr = tmem_addr_storage[0];

    // ---- Role: compute ----
    if (warp <= 3) {
        const int warp_id_in_wg = warp % 4;
        const int my_row = warp_id_in_wg * 32 + lane;
        const int tmem_row_base_v = (warp_id_in_wg * 32) << 16;
        const int tmem_cross = taddr + TMEM_CROSS_OFFSET;
        { // compute_main
            uint32_t _phase_score_full_0 = 0;
            #pragma unroll 1
            for (unsigned int work_idx = cluster_id; work_idx < total_work; work_idx += num_clusters) {
                int split_idx = work_idx % split_count;
                int query_work = work_idx / split_count;
                int batch_idx = query_work / num_q_tile_pairs;
                int q_tile_pair = query_work % num_q_tile_pairs;
                int q_tile = q_tile_pair * 2 + cta_rank;
                int off_q = q_tile * BLOCK_Q;
                int q_idx = off_q + my_row;
                int valid_q = ((q_idx < Q) ? 1 : 0);
                float q_sq_val = 0.0f;
                if (valid_q != 0) {
                    q_sq_val = (float)query_sq[batch_idx * Q + q_idx];
                }
                float best_d[TOP_K_MAX];
                int best_i[TOP_K_MAX];
                #pragma unroll
                for (int kk = 0; kk < TOP_K_MAX; kk++) {
                    best_d[kk] = 3.4e+38f;
                    best_i[kk] = -1;
                }
                float chunk_worst_d[24];
                int chunk_worst_pos[24];
                #pragma unroll
                for (int chunk = 0; chunk < 24; chunk++) {
                    int chunk_base = chunk * 4;
                    chunk_worst_d[chunk] = 3.4e+38f;
                    chunk_worst_pos[chunk] = chunk_base;
                }
                float worst_d = 3.4e+38f;
                int worst_pos = 0;
                int worst_chunk = 0;
                int db_tile_start = split_idx * db_tiles_per_split;
                #pragma unroll 1
                for (int local_db_tile = 0; local_db_tile < db_tiles_per_split; local_db_tile++) {
                    int db_tile = db_tile_start + local_db_tile;
                    int db_start = db_tile * BLOCK_M;
                    int db_sq_idx = db_start + my_row;
                    if (my_row < BLOCK_M) {
                        if (db_sq_idx < M) {
                            smem_database_sq[my_row] = (float)database_sq[batch_idx * M + db_sq_idx];
                        } else {
                            smem_database_sq[my_row] = 0.0f;
                        }
                    }
                    asm volatile("barrier.sync 8, 128;");
                    mbarrier_wait(score_full_addr, _phase_score_full_0);
                    _phase_score_full_0 ^= 1;
                    int cross_addr = taddr + (cta_rank * BLOCK_Q + tmem_row_base_v << 16);
                    float dots[64];
                    asm volatile(
                        "tcgen05.ld.sync.aligned.32x32b.x64.b32"
                        " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63}, [%64];"
                        : "=f"(dots[0]), "=f"(dots[1]), "=f"(dots[2]), "=f"(dots[3]), "=f"(dots[4]), "=f"(dots[5]), "=f"(dots[6]), "=f"(dots[7]), "=f"(dots[8]), "=f"(dots[9]), "=f"(dots[10]), "=f"(dots[11]), "=f"(dots[12]), "=f"(dots[13]), "=f"(dots[14]), "=f"(dots[15]), "=f"(dots[16]), "=f"(dots[17]), "=f"(dots[18]), "=f"(dots[19]), "=f"(dots[20]), "=f"(dots[21]), "=f"(dots[22]), "=f"(dots[23]), "=f"(dots[24]), "=f"(dots[25]), "=f"(dots[26]), "=f"(dots[27]), "=f"(dots[28]), "=f"(dots[29]), "=f"(dots[30]), "=f"(dots[31]), "=f"(dots[32]), "=f"(dots[33]), "=f"(dots[34]), "=f"(dots[35]), "=f"(dots[36]), "=f"(dots[37]), "=f"(dots[38]), "=f"(dots[39]), "=f"(dots[40]), "=f"(dots[41]), "=f"(dots[42]), "=f"(dots[43]), "=f"(dots[44]), "=f"(dots[45]), "=f"(dots[46]), "=f"(dots[47]), "=f"(dots[48]), "=f"(dots[49]), "=f"(dots[50]), "=f"(dots[51]), "=f"(dots[52]), "=f"(dots[53]), "=f"(dots[54]), "=f"(dots[55]), "=f"(dots[56]), "=f"(dots[57]), "=f"(dots[58]), "=f"(dots[59]), "=f"(dots[60]), "=f"(dots[61]), "=f"(dots[62]), "=f"(dots[63])
                        : "r"(cross_addr)
                        : "memory");
                    asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                    asm volatile("barrier.sync 8, 128;");
                    if (elect_sync()) {
                        asm volatile(
                            "mbarrier.arrive.release.cta.shared::cluster.b64 _, [%0];"
                            :: "r"((score_empty_addr) & 0xFEFFFFFF) : "memory");
                    }
                    if (valid_q != 0) {
                        #pragma unroll 1
                        for (int col_base = 0; col_base < 64; col_base += 4) {
                            float dist_vec[4];
                            dist_vec[0] = dots[col_base];
                            dist_vec[1] = dots[col_base + 1];
                            dist_vec[2] = dots[col_base + 2];
                            dist_vec[3] = dots[col_base + 3];
                            const float2 _fma_b2_0 = {-2.0f, -2.0f};
                            const float2 _fma_c2_1 = {q_sq_val, q_sq_val};
                            #pragma unroll
                            for (int _lf = 0; _lf < 2; _lf++)
                                fma_f32x2_inplace(&reinterpret_cast<float2*>(dist_vec)[_lf], _fma_b2_0, _fma_c2_1);
                            float db_sq_vec[4];
                            db_sq_vec[0] = smem_database_sq[col_base];
                            db_sq_vec[1] = smem_database_sq[col_base + 1];
                            db_sq_vec[2] = smem_database_sq[col_base + 2];
                            db_sq_vec[3] = smem_database_sq[col_base + 3];
                            float _t0[4];
                            #pragma unroll
                            for (int _la = 0; _la < 2; _la++)
                                reinterpret_cast<float2*>(_t0)[_la] = add_f32x2(reinterpret_cast<float2*>(dist_vec)[_la], reinterpret_cast<const float2*>(db_sq_vec)[_la]);
                            float group_min = _t0[0];
                            if (_t0[1] < group_min) {
                                group_min = _t0[1];
                            }
                            if (_t0[2] < group_min) {
                                group_min = _t0[2];
                            }
                            if (_t0[3] < group_min) {
                                group_min = _t0[3];
                            }
                            if (group_min < worst_d) {
                                float sort_d0 = _t0[0];
                                float sort_d1 = _t0[1];
                                float sort_d2 = _t0[2];
                                float sort_d3 = _t0[3];
                                int sort_col0 = 0;
                                int sort_col1 = 1;
                                int sort_col2 = 2;
                                int sort_col3 = 3;
                                float tmp_d = 0.0f;
                                int tmp_col = 0;
                                if (sort_d1 < sort_d0) {
                                    tmp_d = sort_d0;
                                    sort_d0 = sort_d1;
                                    sort_d1 = tmp_d;
                                    tmp_col = sort_col0;
                                    sort_col0 = sort_col1;
                                    sort_col1 = tmp_col;
                                }
                                if (sort_d3 < sort_d2) {
                                    tmp_d = sort_d2;
                                    sort_d2 = sort_d3;
                                    sort_d3 = tmp_d;
                                    tmp_col = sort_col2;
                                    sort_col2 = sort_col3;
                                    sort_col3 = tmp_col;
                                }
                                if (sort_d2 < sort_d0) {
                                    tmp_d = sort_d0;
                                    sort_d0 = sort_d2;
                                    sort_d2 = tmp_d;
                                    tmp_col = sort_col0;
                                    sort_col0 = sort_col2;
                                    sort_col2 = tmp_col;
                                }
                                if (sort_d3 < sort_d1) {
                                    tmp_d = sort_d1;
                                    sort_d1 = sort_d3;
                                    sort_d3 = tmp_d;
                                    tmp_col = sort_col1;
                                    sort_col1 = sort_col3;
                                    sort_col3 = tmp_col;
                                }
                                if (sort_d2 < sort_d1) {
                                    tmp_d = sort_d1;
                                    sort_d1 = sort_d2;
                                    sort_d2 = tmp_d;
                                    tmp_col = sort_col1;
                                    sort_col1 = sort_col2;
                                    sort_col2 = tmp_col;
                                }
                                #pragma unroll
                                for (int visit = 0; visit < 4; visit++) {
                                    int vec_col = sort_col0;
                                    float dist = sort_d0;
                                    if (visit == 1) {
                                        vec_col = sort_col1;
                                        dist = sort_d1;
                                    }
                                    if (visit == 2) {
                                        vec_col = sort_col2;
                                        dist = sort_d2;
                                    }
                                    if (visit == 3) {
                                        vec_col = sort_col3;
                                        dist = sort_d3;
                                    }
                                    if (dist >= worst_d) {
                                        break;
                                    }
                                    int db_idx = db_start + col_base + vec_col;
                                    if (db_idx < M) {
                                        best_d[worst_pos] = dist;
                                        best_i[worst_pos] = db_idx;
                                        int refresh_base = worst_chunk * 4;
                                        chunk_worst_d[worst_chunk] = best_d[refresh_base];
                                        chunk_worst_pos[worst_chunk] = refresh_base;
                                        #pragma unroll
                                        for (int offset = 1; offset < 4; offset++) {
                                            int scan_pos = refresh_base + offset;
                                            if (best_d[scan_pos] > chunk_worst_d[worst_chunk]) {
                                                chunk_worst_d[worst_chunk] = best_d[scan_pos];
                                                chunk_worst_pos[worst_chunk] = scan_pos;
                                            }
                                        }
                                        worst_d = chunk_worst_d[0];
                                        worst_pos = chunk_worst_pos[0];
                                        worst_chunk = 0;
                                        #pragma unroll
                                        for (int chunk = 1; chunk < 24; chunk++) {
                                            if (chunk_worst_d[chunk] > worst_d) {
                                                worst_d = chunk_worst_d[chunk];
                                                worst_pos = chunk_worst_pos[chunk];
                                                worst_chunk = chunk;
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                    asm volatile("barrier.sync 8, 128;");
                }
                if (valid_q != 0) {
                    int out_base = ((split_idx * B + batch_idx) * Q + q_idx) * K;
                    #pragma unroll
                    for (int out_k = 0; out_k < TOP_K_MAX; out_k++) {
                        *((float*)(partial_dists + out_base + out_k)) = best_d[out_k];
                        *((int*)(partial_indices + out_base + out_k)) = best_i[out_k];
                    }
                }
            }
        }
    // ---- Role: load ----
    } else if (warp == 4) {
        { // load_main
            uint32_t _phase_query_empty_0 = 1;
            uint32_t _phase_database_empty_0 = 1;
            if (warp_id == 4) {
                if (elect_sync()) {
                    #pragma unroll 1
                    for (unsigned int work_idx = cluster_id; work_idx < total_work; work_idx += num_clusters) {
                        int split_idx = work_idx % split_count;
                        int query_work = work_idx / split_count;
                        int batch_idx = query_work / num_q_tile_pairs;
                        int q_tile_pair = query_work % num_q_tile_pairs;
                        int q_tile = q_tile_pair * 2 + cta_rank;
                        int off_q = q_tile * BLOCK_Q;
                        int global_q = batch_idx * Q + off_q;
                        int db_tile_start = split_idx * db_tiles_per_split;
                        mbarrier_wait(query_empty_addr, _phase_query_empty_0);
                        _phase_query_empty_0 ^= 1;
                        asm volatile(
                            "mbarrier.arrive.expect_tx.release.cta.shared::cluster.b64 _, [%0], %1;"
                            :: "r"((query_full_addr) & 0xFEFFFFFF), "r"((uint32_t)(32768)) : "memory");
                        asm volatile(
                            "cp.async.bulk.tensor.3d.shared::cluster.global.mbarrier::complete_tx::bytes.cta_group::2"
                            " [%0], [%1, {%2, %3, %4}], [%5];"
                            :: "r"(smem_query_addr), "l"(tmap_query), "r"(0), "r"(global_q), "r"(0),
                               "r"(((query_full_addr) & 0xFEFFFFFF)) : "memory");
                        #pragma unroll 1
                        for (int local_db_tile = 0; local_db_tile < db_tiles_per_split; local_db_tile++) {
                            int db_tile = db_tile_start + local_db_tile;
                            int off_m = db_tile * BLOCK_M;
                            int global_m = batch_idx * M + off_m;
                            mbarrier_wait(database_empty_addr, _phase_database_empty_0);
                            _phase_database_empty_0 ^= 1;
                            asm volatile(
                                "mbarrier.arrive.expect_tx.release.cta.shared::cluster.b64 _, [%0], %1;"
                                :: "r"((database_full_addr) & 0xFEFFFFFF), "r"((uint32_t)(16384)) : "memory");
                            asm volatile(
                                "cp.async.bulk.tensor.3d.shared::cluster.global.mbarrier::complete_tx::bytes.cta_group::2"
                                " [%0], [%1, {%2, %3, %4}], [%5];"
                                :: "r"(smem_database_addr), "l"(tmap_database), "r"(0), "r"(global_m), "r"(0),
                                   "r"(((database_full_addr) & 0xFEFFFFFF)) : "memory");
                        }
                    }
                }
            }
        }
    // ---- Role: mma ----
    } else if (warp == 5) {
        { // mma_main
            uint32_t _phase_query_full_0 = 0;
            uint32_t _phase_score_empty_0 = 1;
            uint32_t _phase_database_full_0 = 0;
            if (cta_rank == 0) {
                #pragma unroll 1
                for (unsigned int work_idx = cluster_id; work_idx < total_work; work_idx += num_clusters) {
                    mbarrier_wait(query_full_addr, _phase_query_full_0);
                    _phase_query_full_0 ^= 1;
                    #pragma unroll 1
                    for (int _local_db_tile = 0; _local_db_tile < db_tiles_per_split; _local_db_tile++) {
                        mbarrier_wait(score_empty_addr, _phase_score_empty_0);
                        _phase_score_empty_0 ^= 1;
                        mbarrier_wait(database_full_addr, _phase_database_full_0);
                        _phase_database_full_0 ^= 1;
                        asm volatile("tcgen05.fence::after_thread_sync;");
                        int _mma_ss_a_lo_0 = (smem_query_addr >> 4) & 0x3FFF;
                        int _mma_ss_b_lo_0 = (smem_database_addr >> 4) & 0x3FFF;
                        asm volatile(
                    "{\n\t"
                    ".reg .pred leader, p0, p1;\n\t"
                    ".reg .b32 adhi, bdhi, alo, blo, id, m0, m1, m2, m3, m4, m5, m6, m7;\n\t"
                    ".reg .b64 da, db;\n\t"
                    "elect.sync _|leader, 0xFFFFFFFF;\n\t"
                    "setp.ne.b32 p0, %3, 0;\n\t"
                    "setp.ne.b32 p1, 1, 0;\n\t"
                    "mov.b32 m0, 0; mov.b32 m1, 0; mov.b32 m2, 0; mov.b32 m3, 0;\n\tmov.b32 m4, 0; mov.b32 m5, 0; mov.b32 m6, 0; mov.b32 m7, 0;\n\t"
                    "mov.b32 adhi, 0x40004040;\n\t"
                    "mov.b32 bdhi, 0x40004040;\n\t"
                    "mov.b32 id, 270533776;\n\t"
                    "mov.b32 alo, %0;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p0;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "add.u32 alo, alo, 1018;\n\t"
                    "add.u32 blo, blo, 506;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "}\n"
                    :: "r"(_mma_ss_a_lo_0), "r"(_mma_ss_b_lo_0), "r"(taddr), "r"(0));
                        elect_commit_cg2_multicast(score_full_addr, (uint16_t)(3));
                        elect_commit_cg2_multicast(database_empty_addr, (uint16_t)(3));
                    }
                    elect_commit_cg2_multicast(query_empty_addr, (uint16_t)(3));
                }
            }
        }
    }

    // Cleanup
    asm volatile("barrier.cluster.arrive.release.aligned;");
    asm volatile("barrier.cluster.wait.acquire.aligned;");

    if (warp == 5) {
        asm volatile("tcgen05.dealloc.cta_group::2.sync.aligned.b32 %0, %1;" :: "r"(tmem_addr_storage[0]), "r"(64));
        asm volatile("tcgen05.relinquish_alloc_permit.cta_group::2.sync.aligned;");
    }
}

} // extern "C"

#undef BLOCK_M
#undef BLOCK_Q
#undef FEAT_D
#undef NUM_MAIN_STAGES
#undef SMEM_SMEM_DATABASE_OFF
#undef SMEM_SMEM_DATABASE_SQ_OFF
#undef SMEM_SMEM_DATABASE_SQ_STAGE_BYTES
#undef SMEM_SMEM_DATABASE_SQ_STRIDE
#undef SMEM_SMEM_DATABASE_STAGE_BYTES
#undef SMEM_SMEM_DATABASE_STRIDE
#undef SMEM_SMEM_QUERY_OFF
#undef SMEM_SMEM_QUERY_STAGE_BYTES
#undef SMEM_SMEM_QUERY_STRIDE
#undef SMEM_TOTAL
#undef THREADS
#undef TMEM_CROSS_OFFSET
#undef TMEM_NCOLS
#undef TOP_K_MAX
#undef database_empty_addr
#undef database_full_addr
#undef query_empty_addr
#undef query_full_addr
#undef score_empty_addr
#undef score_full_addr
#undef smem_database_addr
#undef smem_database_sq_addr
#undef smem_query_addr

#define NUM_MAIN_STAGES 1
#define THREADS 32
#define TOP_K_MAX 96
#define SPLIT_COUNT 8

extern "C" {

__global__ __launch_bounds__(32, 1) void
kernel_knn_build_k96_merge_s8_unordered_chunkprefill(float* __restrict__ partial_dists, int32_t* __restrict__ partial_indices, float* __restrict__ out_dists, int32_t* __restrict__ out_indices, int total_queries)
{
    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;


    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    const int warp_id = warp;
    const int lane_id = lane;

    // === Task calls (dependency order) ===
    int start_row = bid * 32 + tid;
    int stride = num_bids * 32;
    #pragma unroll 1
    for (int row = start_row; row < total_queries; row += stride) {
        int base_row = row * TOP_K_MAX;
        int split_stride = total_queries * TOP_K_MAX;
        float best_d[96];
        int best_i[96];
        #pragma unroll
        for (int cand_k = 0; cand_k < 96; cand_k++) {
            best_d[cand_k] = (float)partial_dists[base_row + cand_k];
            best_i[cand_k] = partial_indices[base_row + cand_k];
        }
        float chunk_worst_d[12];
        int chunk_worst_pos[12];
        #pragma unroll
        for (int chunk = 0; chunk < 12; chunk++) {
            int chunk_base = chunk * 8;
            chunk_worst_d[chunk] = best_d[chunk_base];
            chunk_worst_pos[chunk] = chunk_base;
            #pragma unroll
            for (int offset = 1; offset < 8; offset++) {
                int scan_pos = chunk_base + offset;
                if (best_d[scan_pos] > chunk_worst_d[chunk]) {
                    chunk_worst_d[chunk] = best_d[scan_pos];
                    chunk_worst_pos[chunk] = scan_pos;
                }
            }
        }
        float worst_d = chunk_worst_d[0];
        int worst_pos = chunk_worst_pos[0];
        int worst_chunk = 0;
        #pragma unroll
        for (int chunk = 1; chunk < 12; chunk++) {
            if (chunk_worst_d[chunk] > worst_d) {
                worst_d = chunk_worst_d[chunk];
                worst_pos = chunk_worst_pos[chunk];
                worst_chunk = chunk;
            }
        }
        #pragma unroll
        for (int split_idx = 1; split_idx < SPLIT_COUNT; split_idx++) {
            int partial_base = base_row + split_idx * split_stride;
            #pragma unroll
            for (int cand_k = 0; cand_k < 96; cand_k++) {
                float cand_d = (float)partial_dists[partial_base + cand_k];
                int cand_i = partial_indices[partial_base + cand_k];
                if (cand_d < worst_d) {
                    best_d[worst_pos] = cand_d;
                    best_i[worst_pos] = cand_i;
                    int refresh_base = worst_chunk * 8;
                    chunk_worst_d[worst_chunk] = best_d[refresh_base];
                    chunk_worst_pos[worst_chunk] = refresh_base;
                    #pragma unroll
                    for (int offset = 1; offset < 8; offset++) {
                        int scan_pos = refresh_base + offset;
                        if (best_d[scan_pos] > chunk_worst_d[worst_chunk]) {
                            chunk_worst_d[worst_chunk] = best_d[scan_pos];
                            chunk_worst_pos[worst_chunk] = scan_pos;
                        }
                    }
                    worst_d = chunk_worst_d[0];
                    worst_pos = chunk_worst_pos[0];
                    worst_chunk = 0;
                    #pragma unroll
                    for (int chunk = 1; chunk < 12; chunk++) {
                        if (chunk_worst_d[chunk] > worst_d) {
                            worst_d = chunk_worst_d[chunk];
                            worst_pos = chunk_worst_pos[chunk];
                            worst_chunk = chunk;
                        }
                    }
                }
            }
        }
        #pragma unroll
        for (int out_k = 0; out_k < 96; out_k++) {
            *((float*)(out_dists + base_row + out_k)) = best_d[out_k];
            *((int*)(out_indices + base_row + out_k)) = best_i[out_k];
        }
    }
}

} // extern "C"

#undef NUM_MAIN_STAGES
#undef SPLIT_COUNT
#undef THREADS
#undef TOP_K_MAX

#define NUM_MAIN_STAGES 1
#define THREADS 32
#define TOP_K_MAX 96
#define SPLIT_COUNT 8

extern "C" {

__global__ __launch_bounds__(32, 1) void
kernel_knn_build_k96_merge_s8_unordered_chunkprefill_k96over64s8chunkprefill(float* __restrict__ partial_dists, int32_t* __restrict__ partial_indices, float* __restrict__ out_dists, int32_t* __restrict__ out_indices, int total_queries)
{
    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;


    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    const int warp_id = warp;
    const int lane_id = lane;

    // === Task calls (dependency order) ===
    int start_row = bid * 32 + tid;
    int stride = num_bids * 32;
    #pragma unroll 1
    for (int row = start_row; row < total_queries; row += stride) {
        int base_row = row * TOP_K_MAX;
        int split_stride = total_queries * TOP_K_MAX;
        float best_d[96];
        int best_i[96];
        #pragma unroll
        for (int cand_k = 0; cand_k < 96; cand_k++) {
            best_d[cand_k] = (float)partial_dists[base_row + cand_k];
            best_i[cand_k] = partial_indices[base_row + cand_k];
        }
        float chunk_worst_d[12];
        int chunk_worst_pos[12];
        #pragma unroll
        for (int chunk = 0; chunk < 12; chunk++) {
            int chunk_base = chunk * 8;
            chunk_worst_d[chunk] = best_d[chunk_base];
            chunk_worst_pos[chunk] = chunk_base;
            #pragma unroll
            for (int offset = 1; offset < 8; offset++) {
                int scan_pos = chunk_base + offset;
                if (best_d[scan_pos] > chunk_worst_d[chunk]) {
                    chunk_worst_d[chunk] = best_d[scan_pos];
                    chunk_worst_pos[chunk] = scan_pos;
                }
            }
        }
        float worst_d = chunk_worst_d[0];
        int worst_pos = chunk_worst_pos[0];
        int worst_chunk = 0;
        #pragma unroll
        for (int chunk = 1; chunk < 12; chunk++) {
            if (chunk_worst_d[chunk] > worst_d) {
                worst_d = chunk_worst_d[chunk];
                worst_pos = chunk_worst_pos[chunk];
                worst_chunk = chunk;
            }
        }
        #pragma unroll
        for (int split_idx = 1; split_idx < SPLIT_COUNT; split_idx++) {
            int partial_base = base_row + split_idx * split_stride;
            #pragma unroll
            for (int cand_k = 0; cand_k < 96; cand_k++) {
                float cand_d = (float)partial_dists[partial_base + cand_k];
                int cand_i = partial_indices[partial_base + cand_k];
                if (cand_d < worst_d) {
                    best_d[worst_pos] = cand_d;
                    best_i[worst_pos] = cand_i;
                    int refresh_base = worst_chunk * 8;
                    chunk_worst_d[worst_chunk] = best_d[refresh_base];
                    chunk_worst_pos[worst_chunk] = refresh_base;
                    #pragma unroll
                    for (int offset = 1; offset < 8; offset++) {
                        int scan_pos = refresh_base + offset;
                        if (best_d[scan_pos] > chunk_worst_d[worst_chunk]) {
                            chunk_worst_d[worst_chunk] = best_d[scan_pos];
                            chunk_worst_pos[worst_chunk] = scan_pos;
                        }
                    }
                    worst_d = chunk_worst_d[0];
                    worst_pos = chunk_worst_pos[0];
                    worst_chunk = 0;
                    #pragma unroll
                    for (int chunk = 1; chunk < 12; chunk++) {
                        if (chunk_worst_d[chunk] > worst_d) {
                            worst_d = chunk_worst_d[chunk];
                            worst_pos = chunk_worst_pos[chunk];
                            worst_chunk = chunk;
                        }
                    }
                }
            }
        }
        #pragma unroll
        for (int out_k = 0; out_k < 96; out_k++) {
            *((float*)(out_dists + base_row + out_k)) = best_d[out_k];
            *((int*)(out_indices + base_row + out_k)) = best_i[out_k];
        }
    }
}

} // extern "C"

#undef NUM_MAIN_STAGES
#undef SPLIT_COUNT
#undef THREADS
#undef TOP_K_MAX

#define NUM_MAIN_STAGES 1
#define THREADS 32
#define TOP_K_MAX 32
#define SPLIT_COUNT 32

extern "C" {

__global__ __launch_bounds__(32, 1) void
kernel_knn_build_evolve_7bfc_k30_merge_s8_rowbase_cache_k32s32_4b5c(float* __restrict__ partial_dists, int32_t* __restrict__ partial_indices, float* __restrict__ out_dists, int32_t* __restrict__ out_indices, int total_queries)
{
    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;


    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    const int warp_id = warp;
    const int lane_id = lane;

    // === Task calls (dependency order) ===
    int start_row = bid * 32 + tid;
    int stride = num_bids * 32;
    #pragma unroll 1
    for (int row = start_row; row < total_queries; row += stride) {
        int base_row = row * TOP_K_MAX;
        int split_stride = total_queries * TOP_K_MAX;
        int out_base = base_row;
        int split_pos[SPLIT_COUNT];
        int split_base[SPLIT_COUNT];
        float cand_d[SPLIT_COUNT];
        int cand_i[SPLIT_COUNT];
        #pragma unroll
        for (int split_idx = 0; split_idx < SPLIT_COUNT; split_idx++) {
            split_pos[split_idx] = 0;
            split_base[split_idx] = base_row + split_idx * split_stride;
            cand_d[split_idx] = (float)partial_dists[split_base[split_idx]];
            cand_i[split_idx] = partial_indices[split_base[split_idx]];
        }
        #pragma unroll
        for (int out_k = 0; out_k < TOP_K_MAX; out_k++) {
            float best_d = cand_d[0];
            int best_i = cand_i[0];
            int best_split = 0;
            #pragma unroll
            for (int split_idx = 1; split_idx < SPLIT_COUNT; split_idx++) {
                if (cand_d[split_idx] < best_d) {
                    best_d = cand_d[split_idx];
                    best_i = cand_i[split_idx];
                    best_split = split_idx;
                }
            }
            *((float*)(out_dists + out_base + out_k)) = best_d;
            *((int*)(out_indices + out_base + out_k)) = best_i;
            split_pos[best_split] = split_pos[best_split] + 1;
            if (out_k + 1 < TOP_K_MAX) {
                int next_pos = split_pos[best_split];
                int next_addr = split_base[best_split] + next_pos;
                cand_d[best_split] = (float)partial_dists[next_addr];
                cand_i[best_split] = partial_indices[next_addr];
            }
        }
    }
}

} // extern "C"

#undef NUM_MAIN_STAGES
#undef SPLIT_COUNT
#undef THREADS
#undef TOP_K_MAX

#define TMEM_NCOLS 64
#define TMEM_CROSS_OFFSET 0
#define NUM_MAIN_STAGES 1
#define SMEM_SMEM_QUERY_OFF 1024
#define SMEM_SMEM_QUERY_STAGE_BYTES 32768
#define SMEM_SMEM_QUERY_STRIDE 32768
#define SMEM_SMEM_DATABASE_OFF 33792
#define SMEM_SMEM_DATABASE_STAGE_BYTES 16384
#define SMEM_SMEM_DATABASE_STRIDE 16384
#define SMEM_SMEM_DATABASE_SQ_OFF 50176
#define SMEM_SMEM_DATABASE_SQ_STAGE_BYTES 256
#define SMEM_SMEM_DATABASE_SQ_STRIDE 256
#define SMEM_TOTAL 50432
#define THREADS 192
#define BLOCK_Q 128
#define BLOCK_M 64
#define FEAT_D 128
#define TOP_K_MAX 32

extern "C" {

__global__ __launch_bounds__(192, 1) void
kernel_knn_build_rag_frontier_4fbf_v7_stage1_k32_sort4earlystop_tailinf(float* __restrict__ query_sq, float* __restrict__ database_sq, float* __restrict__ partial_dists, int32_t* __restrict__ partial_indices, const void* tmap_query, const void* tmap_database, int B, int Q, int M, int K, int num_q_tile_pairs, int db_tiles_per_split, int split_count, int total_work)
{
    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;

    extern __shared__ __align__(1024) char smem_raw[];
    int smem;
    smem = (int)(unsigned long long)__cvta_generic_to_shared(smem_raw);
    const int smem_smem_query = smem + 1024;
    const int smem_smem_database = smem + 33792;
    const int smem_smem_database_sq = smem + 50176;

    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;
    const unsigned int clusters_x = gridDim.x / 2;
    const unsigned int cluster_id = ((blockIdx.z * gridDim.y + blockIdx.y) * clusters_x) + blockIdx.x / 2;
    const unsigned int num_clusters = clusters_x * gridDim.y * gridDim.z;

    int cta_rank;
    asm volatile("mov.b32 %0, %%cluster_ctarank;" : "=r"(cta_rank));

    // Mbarrier init (6 groups, 6 barriers)
    // Mbarriers at smem_raw[0..48)

    if (warp == 0) {
        uint32_t leader = elect_sync();
        // query_full: 1 barriers, init_count=2
        mbarrier_init_pred(smem + 0, 2, leader);
        // query_empty: 1 barriers, init_count=1
        mbarrier_init_pred(smem + 8, 1, leader);
        // database_full: 1 barriers, init_count=2
        mbarrier_init_pred(smem + 16, 2, leader);
        // database_empty: 1 barriers, init_count=1
        mbarrier_init_pred(smem + 24, 1, leader);
        // score_full: 1 barriers, init_count=1
        mbarrier_init_pred(smem + 32, 1, leader);
        // score_empty: 1 barriers, init_count=8
        mbarrier_init_pred(smem + 40, 8, leader);
        asm volatile("fence.mbarrier_init.release.cluster;");
    }

    __syncthreads();

    // TMEM alloc (64 columns, 64 used)
    volatile int* tmem_addr_storage = (volatile int*)(smem_raw + 48);
    if (warp == 1) {
        int _tmem_hold = smem + 48;
        asm volatile("tcgen05.alloc.cta_group::2.sync.aligned.shared::cta.b32 [%0], %1;" :: "r"(_tmem_hold), "r"(64) : "memory");
    }

    asm volatile("barrier.cluster.arrive.release.aligned;");
    asm volatile("barrier.cluster.wait.acquire.aligned;");
    asm volatile("tcgen05.fence::after_thread_sync;");

    const int warp_id = warp;
    const int lane_id = lane;
    __nv_bfloat16* smem_query = (__nv_bfloat16*)(smem_raw + 1024);
    #define smem_query_addr (smem + 1024)
    __nv_bfloat16* smem_database = (__nv_bfloat16*)(smem_raw + 33792);
    #define smem_database_addr (smem + 33792)
    float* smem_database_sq = (float*)(smem_raw + 50176);
    #define smem_database_sq_addr (smem + 50176)
    const int mbar_base = smem;
    #define query_full_addr (mbar_base + 0)
    #define query_empty_addr (mbar_base + 8)
    #define database_full_addr (mbar_base + 16)
    #define database_empty_addr (mbar_base + 24)
    #define score_full_addr (mbar_base + 32)
    #define score_empty_addr (mbar_base + 40)
    const int taddr = tmem_addr_storage[0];

    // ---- Role: load ----
    if (warp == 0) {
        { // load_main
            uint32_t _phase_query_empty_0 = 1;
            uint32_t _phase_database_empty_0 = 1;
            if (warp_id == 0) {
                if (elect_sync()) {
                    #pragma unroll 1
                    for (unsigned int work_idx = cluster_id; work_idx < total_work; work_idx += num_clusters) {
                        int split_idx = work_idx % split_count;
                        int query_work = work_idx / split_count;
                        int batch_idx = query_work / num_q_tile_pairs;
                        int q_tile_pair = query_work % num_q_tile_pairs;
                        int q_tile = q_tile_pair * 2 + cta_rank;
                        int off_q = q_tile * BLOCK_Q;
                        int global_q = batch_idx * Q + off_q;
                        int db_tile_start = split_idx * db_tiles_per_split;
                        mbarrier_wait(query_empty_addr, _phase_query_empty_0);
                        _phase_query_empty_0 ^= 1;
                        asm volatile(
                            "mbarrier.arrive.expect_tx.release.cta.shared::cluster.b64 _, [%0], %1;"
                            :: "r"((query_full_addr) & 0xFEFFFFFF), "r"((uint32_t)(32768)) : "memory");
                        asm volatile(
                            "cp.async.bulk.tensor.3d.shared::cluster.global.mbarrier::complete_tx::bytes.cta_group::2"
                            " [%0], [%1, {%2, %3, %4}], [%5];"
                            :: "r"(smem_query_addr), "l"(tmap_query), "r"(0), "r"(global_q), "r"(0),
                               "r"(((query_full_addr) & 0xFEFFFFFF)) : "memory");
                        #pragma unroll 1
                        for (int local_db_tile = 0; local_db_tile < db_tiles_per_split; local_db_tile++) {
                            int db_tile = db_tile_start + local_db_tile;
                            int off_m = db_tile * BLOCK_M;
                            int global_m = batch_idx * M + off_m;
                            mbarrier_wait(database_empty_addr, _phase_database_empty_0);
                            _phase_database_empty_0 ^= 1;
                            asm volatile(
                                "mbarrier.arrive.expect_tx.release.cta.shared::cluster.b64 _, [%0], %1;"
                                :: "r"((database_full_addr) & 0xFEFFFFFF), "r"((uint32_t)(16384)) : "memory");
                            asm volatile(
                                "cp.async.bulk.tensor.3d.shared::cluster.global.mbarrier::complete_tx::bytes.cta_group::2"
                                " [%0], [%1, {%2, %3, %4}], [%5];"
                                :: "r"(smem_database_addr), "l"(tmap_database), "r"(0), "r"(global_m), "r"(0),
                                   "r"(((database_full_addr) & 0xFEFFFFFF)) : "memory");
                        }
                    }
                }
            }
        }
    // ---- Role: mma ----
    } else if (warp == 1) {
        { // mma_main
            uint32_t _phase_query_full_0 = 0;
            uint32_t _phase_score_empty_0 = 1;
            uint32_t _phase_database_full_0 = 0;
            if (cta_rank == 0) {
                #pragma unroll 1
                for (unsigned int work_idx = cluster_id; work_idx < total_work; work_idx += num_clusters) {
                    mbarrier_wait(query_full_addr, _phase_query_full_0);
                    _phase_query_full_0 ^= 1;
                    #pragma unroll 1
                    for (int _local_db_tile = 0; _local_db_tile < db_tiles_per_split; _local_db_tile++) {
                        mbarrier_wait(score_empty_addr, _phase_score_empty_0);
                        _phase_score_empty_0 ^= 1;
                        mbarrier_wait(database_full_addr, _phase_database_full_0);
                        _phase_database_full_0 ^= 1;
                        asm volatile("tcgen05.fence::after_thread_sync;");
                        int _mma_ss_a_lo_0 = (smem_query_addr >> 4) & 0x3FFF;
                        int _mma_ss_b_lo_0 = (smem_database_addr >> 4) & 0x3FFF;
                        asm volatile(
                    "{\n\t"
                    ".reg .pred leader, p0, p1;\n\t"
                    ".reg .b32 adhi, bdhi, alo, blo, id, m0, m1, m2, m3, m4, m5, m6, m7;\n\t"
                    ".reg .b64 da, db;\n\t"
                    "elect.sync _|leader, 0xFFFFFFFF;\n\t"
                    "setp.ne.b32 p0, %3, 0;\n\t"
                    "setp.ne.b32 p1, 1, 0;\n\t"
                    "mov.b32 m0, 0; mov.b32 m1, 0; mov.b32 m2, 0; mov.b32 m3, 0;\n\tmov.b32 m4, 0; mov.b32 m5, 0; mov.b32 m6, 0; mov.b32 m7, 0;\n\t"
                    "mov.b32 adhi, 0x40004040;\n\t"
                    "mov.b32 bdhi, 0x40004040;\n\t"
                    "mov.b32 id, 270533776;\n\t"
                    "mov.b32 alo, %0;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p0;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "add.u32 alo, alo, 1018;\n\t"
                    "add.u32 blo, blo, 506;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "}\n"
                    :: "r"(_mma_ss_a_lo_0), "r"(_mma_ss_b_lo_0), "r"(taddr), "r"(0));
                        elect_commit_cg2_multicast(score_full_addr, (uint16_t)(3));
                        elect_commit_cg2_multicast(database_empty_addr, (uint16_t)(3));
                    }
                    elect_commit_cg2_multicast(query_empty_addr, (uint16_t)(3));
                }
            }
        }
    // ---- Role: compute ----
    } else if (warp >= 2 && warp <= 5) {
        const int warp_id_in_wg = warp % 4;
        const int my_row = warp_id_in_wg * 32 + lane;
        const int tmem_row_base_v = (warp_id_in_wg * 32) << 16;
        const int tmem_cross = taddr + TMEM_CROSS_OFFSET;
        { // compute_main
            uint32_t _phase_score_full_0 = 0;
            #pragma unroll 1
            for (unsigned int work_idx = cluster_id; work_idx < total_work; work_idx += num_clusters) {
                int split_idx = work_idx % split_count;
                int query_work = work_idx / split_count;
                int batch_idx = query_work / num_q_tile_pairs;
                int q_tile_pair = query_work % num_q_tile_pairs;
                int q_tile = q_tile_pair * 2 + cta_rank;
                int off_q = q_tile * BLOCK_Q;
                int q_idx = off_q + my_row;
                int valid_q = ((q_idx < Q) ? 1 : 0);
                float q_sq_val = 0.0f;
                if (valid_q != 0) {
                    q_sq_val = (float)query_sq[batch_idx * Q + q_idx];
                }
                float best_d[TOP_K_MAX];
                int best_i[TOP_K_MAX];
                #pragma unroll
                for (int kk = 0; kk < TOP_K_MAX; kk++) {
                    best_d[kk] = 3.4e+38f;
                    best_i[kk] = -1;
                }
                float chunk_worst_d[4];
                int chunk_worst_pos[4];
                #pragma unroll
                for (int chunk = 0; chunk < 4; chunk++) {
                    int chunk_base = chunk * 8;
                    chunk_worst_d[chunk] = 3.4e+38f;
                    chunk_worst_pos[chunk] = chunk_base;
                }
                float worst_d = 3.4e+38f;
                int worst_pos = 0;
                int worst_chunk = 0;
                int db_tile_start = split_idx * db_tiles_per_split;
                #pragma unroll 1
                for (int local_db_tile = 0; local_db_tile < db_tiles_per_split; local_db_tile++) {
                    int db_tile = db_tile_start + local_db_tile;
                    int db_start = db_tile * BLOCK_M;
                    int db_sq_idx = db_start + my_row;
                    if (my_row < BLOCK_M) {
                        if (db_sq_idx < M) {
                            smem_database_sq[my_row] = (float)database_sq[batch_idx * M + db_sq_idx];
                        } else {
                            smem_database_sq[my_row] = 3.4e+38f;
                        }
                    }
                    asm volatile("barrier.sync 8, 128;");
                    mbarrier_wait(score_full_addr, _phase_score_full_0);
                    _phase_score_full_0 ^= 1;
                    int cross_addr = taddr + (cta_rank * BLOCK_Q + tmem_row_base_v << 16);
                    float dots[64];
                    asm volatile(
                        "tcgen05.ld.sync.aligned.32x32b.x64.b32"
                        " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63}, [%64];"
                        : "=f"(dots[0]), "=f"(dots[1]), "=f"(dots[2]), "=f"(dots[3]), "=f"(dots[4]), "=f"(dots[5]), "=f"(dots[6]), "=f"(dots[7]), "=f"(dots[8]), "=f"(dots[9]), "=f"(dots[10]), "=f"(dots[11]), "=f"(dots[12]), "=f"(dots[13]), "=f"(dots[14]), "=f"(dots[15]), "=f"(dots[16]), "=f"(dots[17]), "=f"(dots[18]), "=f"(dots[19]), "=f"(dots[20]), "=f"(dots[21]), "=f"(dots[22]), "=f"(dots[23]), "=f"(dots[24]), "=f"(dots[25]), "=f"(dots[26]), "=f"(dots[27]), "=f"(dots[28]), "=f"(dots[29]), "=f"(dots[30]), "=f"(dots[31]), "=f"(dots[32]), "=f"(dots[33]), "=f"(dots[34]), "=f"(dots[35]), "=f"(dots[36]), "=f"(dots[37]), "=f"(dots[38]), "=f"(dots[39]), "=f"(dots[40]), "=f"(dots[41]), "=f"(dots[42]), "=f"(dots[43]), "=f"(dots[44]), "=f"(dots[45]), "=f"(dots[46]), "=f"(dots[47]), "=f"(dots[48]), "=f"(dots[49]), "=f"(dots[50]), "=f"(dots[51]), "=f"(dots[52]), "=f"(dots[53]), "=f"(dots[54]), "=f"(dots[55]), "=f"(dots[56]), "=f"(dots[57]), "=f"(dots[58]), "=f"(dots[59]), "=f"(dots[60]), "=f"(dots[61]), "=f"(dots[62]), "=f"(dots[63])
                        : "r"(cross_addr)
                        : "memory");
                    asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                    asm volatile("barrier.sync 8, 128;");
                    if (elect_sync()) {
                        asm volatile(
                            "mbarrier.arrive.release.cta.shared::cluster.b64 _, [%0];"
                            :: "r"((score_empty_addr) & 0xFEFFFFFF) : "memory");
                    }
                    if (valid_q != 0) {
                        #pragma unroll 2
                        for (int col_base = 0; col_base < 64; col_base += 4) {
                            float dist_vec[4];
                            dist_vec[0] = dots[col_base];
                            dist_vec[1] = dots[col_base + 1];
                            dist_vec[2] = dots[col_base + 2];
                            dist_vec[3] = dots[col_base + 3];
                            const float2 _fma_b2_0 = {-2.0f, -2.0f};
                            const float2 _fma_c2_1 = {q_sq_val, q_sq_val};
                            #pragma unroll
                            for (int _lf = 0; _lf < 2; _lf++)
                                fma_f32x2_inplace(&reinterpret_cast<float2*>(dist_vec)[_lf], _fma_b2_0, _fma_c2_1);
                            float db_sq_vec[4];
                            db_sq_vec[0] = smem_database_sq[col_base];
                            db_sq_vec[1] = smem_database_sq[col_base + 1];
                            db_sq_vec[2] = smem_database_sq[col_base + 2];
                            db_sq_vec[3] = smem_database_sq[col_base + 3];
                            float _t0[4];
                            #pragma unroll
                            for (int _la = 0; _la < 2; _la++)
                                reinterpret_cast<float2*>(_t0)[_la] = add_f32x2(reinterpret_cast<float2*>(dist_vec)[_la], reinterpret_cast<const float2*>(db_sq_vec)[_la]);
                            float group_min = _t0[0];
                            if (_t0[1] < group_min) {
                                group_min = _t0[1];
                            }
                            if (_t0[2] < group_min) {
                                group_min = _t0[2];
                            }
                            if (_t0[3] < group_min) {
                                group_min = _t0[3];
                            }
                            if (group_min < worst_d) {
                                float sort_d0 = _t0[0];
                                float sort_d1 = _t0[1];
                                float sort_d2 = _t0[2];
                                float sort_d3 = _t0[3];
                                int sort_col0 = 0;
                                int sort_col1 = 1;
                                int sort_col2 = 2;
                                int sort_col3 = 3;
                                float tmp_d = 0.0f;
                                int tmp_col = 0;
                                if (sort_d1 < sort_d0) {
                                    tmp_d = sort_d0;
                                    sort_d0 = sort_d1;
                                    sort_d1 = tmp_d;
                                    tmp_col = sort_col0;
                                    sort_col0 = sort_col1;
                                    sort_col1 = tmp_col;
                                }
                                if (sort_d3 < sort_d2) {
                                    tmp_d = sort_d2;
                                    sort_d2 = sort_d3;
                                    sort_d3 = tmp_d;
                                    tmp_col = sort_col2;
                                    sort_col2 = sort_col3;
                                    sort_col3 = tmp_col;
                                }
                                if (sort_d2 < sort_d0) {
                                    tmp_d = sort_d0;
                                    sort_d0 = sort_d2;
                                    sort_d2 = tmp_d;
                                    tmp_col = sort_col0;
                                    sort_col0 = sort_col2;
                                    sort_col2 = tmp_col;
                                }
                                if (sort_d3 < sort_d1) {
                                    tmp_d = sort_d1;
                                    sort_d1 = sort_d3;
                                    sort_d3 = tmp_d;
                                    tmp_col = sort_col1;
                                    sort_col1 = sort_col3;
                                    sort_col3 = tmp_col;
                                }
                                if (sort_d2 < sort_d1) {
                                    tmp_d = sort_d1;
                                    sort_d1 = sort_d2;
                                    sort_d2 = tmp_d;
                                    tmp_col = sort_col1;
                                    sort_col1 = sort_col2;
                                    sort_col2 = tmp_col;
                                }
                                #pragma unroll
                                for (int visit = 0; visit < 4; visit++) {
                                    int vec_col = sort_col0;
                                    float dist = sort_d0;
                                    if (visit == 1) {
                                        vec_col = sort_col1;
                                        dist = sort_d1;
                                    }
                                    if (visit == 2) {
                                        vec_col = sort_col2;
                                        dist = sort_d2;
                                    }
                                    if (visit == 3) {
                                        vec_col = sort_col3;
                                        dist = sort_d3;
                                    }
                                    if (dist >= worst_d) {
                                        break;
                                    }
                                    int db_idx = db_start + col_base + vec_col;
                                    best_d[worst_pos] = dist;
                                    best_i[worst_pos] = db_idx;
                                    int refresh_base = worst_chunk * 8;
                                    chunk_worst_d[worst_chunk] = best_d[refresh_base];
                                    chunk_worst_pos[worst_chunk] = refresh_base;
                                    #pragma unroll
                                    for (int offset = 1; offset < 8; offset++) {
                                        int scan_pos = refresh_base + offset;
                                        if (best_d[scan_pos] > chunk_worst_d[worst_chunk]) {
                                            chunk_worst_d[worst_chunk] = best_d[scan_pos];
                                            chunk_worst_pos[worst_chunk] = scan_pos;
                                        }
                                    }
                                    worst_d = chunk_worst_d[0];
                                    worst_pos = chunk_worst_pos[0];
                                    worst_chunk = 0;
                                    #pragma unroll
                                    for (int chunk = 1; chunk < 4; chunk++) {
                                        if (chunk_worst_d[chunk] > worst_d) {
                                            worst_d = chunk_worst_d[chunk];
                                            worst_pos = chunk_worst_pos[chunk];
                                            worst_chunk = chunk;
                                        }
                                    }
                                }
                            }
                        }
                    }
                    asm volatile("barrier.sync 8, 128;");
                }
                if (valid_q != 0) {
                    int out_base = ((split_idx * B + batch_idx) * Q + q_idx) * K;
                    #pragma unroll
                    for (int out_k = 0; out_k < TOP_K_MAX; out_k++) {
                        float best_out_d = best_d[0];
                        int best_out_i = best_i[0];
                        int best_out_pos = 0;
                        #pragma unroll
                        for (int scan_pos = 1; scan_pos < TOP_K_MAX; scan_pos++) {
                            if (best_d[scan_pos] < best_out_d) {
                                best_out_d = best_d[scan_pos];
                                best_out_i = best_i[scan_pos];
                                best_out_pos = scan_pos;
                            }
                        }
                        if (out_k < K) {
                            *((float*)(partial_dists + out_base + out_k)) = best_out_d;
                            *((int*)(partial_indices + out_base + out_k)) = best_out_i;
                        }
                        best_d[best_out_pos] = 3.4e+38f;
                    }
                }
            }
        }
    }

    // Cleanup
    asm volatile("barrier.cluster.arrive.release.aligned;");
    asm volatile("barrier.cluster.wait.acquire.aligned;");

    if (warp == 1) {
        asm volatile("tcgen05.dealloc.cta_group::2.sync.aligned.b32 %0, %1;" :: "r"(tmem_addr_storage[0]), "r"(64));
        asm volatile("tcgen05.relinquish_alloc_permit.cta_group::2.sync.aligned;");
    }
}

} // extern "C"

#undef BLOCK_M
#undef BLOCK_Q
#undef FEAT_D
#undef NUM_MAIN_STAGES
#undef SMEM_SMEM_DATABASE_OFF
#undef SMEM_SMEM_DATABASE_SQ_OFF
#undef SMEM_SMEM_DATABASE_SQ_STAGE_BYTES
#undef SMEM_SMEM_DATABASE_SQ_STRIDE
#undef SMEM_SMEM_DATABASE_STAGE_BYTES
#undef SMEM_SMEM_DATABASE_STRIDE
#undef SMEM_SMEM_QUERY_OFF
#undef SMEM_SMEM_QUERY_STAGE_BYTES
#undef SMEM_SMEM_QUERY_STRIDE
#undef SMEM_TOTAL
#undef THREADS
#undef TMEM_CROSS_OFFSET
#undef TMEM_NCOLS
#undef TOP_K_MAX
#undef database_empty_addr
#undef database_full_addr
#undef query_empty_addr
#undef query_full_addr
#undef score_empty_addr
#undef score_full_addr
#undef smem_database_addr
#undef smem_database_sq_addr
#undef smem_query_addr

#define NUM_MAIN_STAGES 1
#define SMEM_GROUP_DISTS_OFF 0
#define SMEM_GROUP_DISTS_STAGE_BYTES 4096
#define SMEM_GROUP_DISTS_STRIDE 4096
#define SMEM_GROUP_INDICES_OFF 4096
#define SMEM_GROUP_INDICES_STAGE_BYTES 4096
#define SMEM_GROUP_INDICES_STRIDE 4096
#define SMEM_TOTAL 8192
#define THREADS 32
#define TOP_K_MAX 32
#define GROUP_COUNT 8
#define GROUP_SPLITS 9

extern "C" {

__global__ __launch_bounds__(32, 1) void
kernel_knn_build_rag_frontier_4fbf_v7_k32_fused_group_final_merge(float* __restrict__ partial_dists, int32_t* __restrict__ partial_indices, float* __restrict__ out_dists, int32_t* __restrict__ out_indices, int total_queries)
{
    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;

    extern __shared__ __align__(1024) char smem_raw[];
    int smem;
    smem = (int)(unsigned long long)__cvta_generic_to_shared(smem_raw);
    const int smem_group_dists = smem + 0;
    const int smem_group_indices = smem + 4096;

    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    const int warp_id = warp;
    const int lane_id = lane;
    float* group_dists = (float*)(smem_raw + 0);
    #define group_dists_addr (smem + 0)
    int* group_indices = (int*)(smem_raw + 4096);
    #define group_indices_addr (smem + 4096)

    // === Task calls (dependency order) ===
    int split_pos[GROUP_SPLITS];
    int split_base[GROUP_SPLITS];
    float group_cand_d[GROUP_SPLITS];
    int group_cand_i[GROUP_SPLITS];
    int final_pos[GROUP_COUNT];
    float final_cand_d[GROUP_COUNT];
    int final_cand_i[GROUP_COUNT];
    #pragma unroll 1
    for (int row = bid; row < total_queries; row += num_bids) {
        int base_row = row * TOP_K_MAX;
        int split_stride = total_queries * TOP_K_MAX;
        if (tid < GROUP_COUNT) {
            int group_idx = tid;
            int source_split0 = group_idx * GROUP_SPLITS;
            int shared_base = group_idx * TOP_K_MAX;
            #pragma unroll
            for (int local_split = 0; local_split < GROUP_SPLITS; local_split++) {
                split_pos[local_split] = 0;
                int split_id = source_split0 + local_split;
                split_base[local_split] = base_row + split_id * split_stride;
                group_cand_d[local_split] = (float)partial_dists[split_base[local_split]];
                group_cand_i[local_split] = partial_indices[split_base[local_split]];
            }
            #pragma unroll
            for (int out_k = 0; out_k < TOP_K_MAX; out_k++) {
                float best_d = group_cand_d[0];
                int best_i = group_cand_i[0];
                int best_split = 0;
                #pragma unroll
                for (int local_split = 1; local_split < GROUP_SPLITS; local_split++) {
                    if (group_cand_d[local_split] < best_d) {
                        best_d = group_cand_d[local_split];
                        best_i = group_cand_i[local_split];
                        best_split = local_split;
                    }
                }
                group_dists[shared_base + out_k] = best_d;
                group_indices[shared_base + out_k] = best_i;
                split_pos[best_split] = split_pos[best_split] + 1;
                if (out_k + 1 < TOP_K_MAX) {
                    int next_pos = split_pos[best_split];
                    int next_addr = split_base[best_split] + next_pos;
                    group_cand_d[best_split] = (float)partial_dists[next_addr];
                    group_cand_i[best_split] = partial_indices[next_addr];
                }
            }
        }
        __syncthreads();
        if (tid == 0) {
            #pragma unroll
            for (int group_idx = 0; group_idx < GROUP_COUNT; group_idx++) {
                final_pos[group_idx] = 0;
                int group_base = group_idx * TOP_K_MAX;
                final_cand_d[group_idx] = group_dists[group_base];
                final_cand_i[group_idx] = group_indices[group_base];
            }
            #pragma unroll
            for (int out_k = 0; out_k < TOP_K_MAX; out_k++) {
                float best_d = final_cand_d[0];
                int best_i = final_cand_i[0];
                int best_group = 0;
                #pragma unroll
                for (int group_idx = 1; group_idx < GROUP_COUNT; group_idx++) {
                    if (final_cand_d[group_idx] < best_d) {
                        best_d = final_cand_d[group_idx];
                        best_i = final_cand_i[group_idx];
                        best_group = group_idx;
                    }
                }
                *((float*)(out_dists + base_row + out_k)) = best_d;
                *((int*)(out_indices + base_row + out_k)) = best_i;
                final_pos[best_group] = final_pos[best_group] + 1;
                if (out_k + 1 < TOP_K_MAX) {
                    int next_pos = final_pos[best_group];
                    int next_addr = best_group * TOP_K_MAX + next_pos;
                    final_cand_d[best_group] = group_dists[next_addr];
                    final_cand_i[best_group] = group_indices[next_addr];
                }
            }
        }
        __syncthreads();
    }
}

} // extern "C"

#undef GROUP_COUNT
#undef GROUP_SPLITS
#undef NUM_MAIN_STAGES
#undef SMEM_GROUP_DISTS_OFF
#undef SMEM_GROUP_DISTS_STAGE_BYTES
#undef SMEM_GROUP_DISTS_STRIDE
#undef SMEM_GROUP_INDICES_OFF
#undef SMEM_GROUP_INDICES_STAGE_BYTES
#undef SMEM_GROUP_INDICES_STRIDE
#undef SMEM_TOTAL
#undef THREADS
#undef TOP_K_MAX
#undef group_dists_addr
#undef group_indices_addr

#define NUM_MAIN_STAGES 1
#define SMEM_GROUP_DISTS_OFF 0
#define SMEM_GROUP_DISTS_STAGE_BYTES 1024
#define SMEM_GROUP_DISTS_STRIDE 1024
#define SMEM_GROUP_INDICES_OFF 1024
#define SMEM_GROUP_INDICES_STAGE_BYTES 1024
#define SMEM_GROUP_INDICES_STRIDE 1024
#define SMEM_TOTAL 2048
#define THREADS 32
#define TOP_K_MAX 32
#define GROUP_COUNT 8
#define GROUP_SPLITS 9

extern "C" {

__global__ __launch_bounds__(32, 1) void
kernel_knn_build_rag_frontier_7399_k32_fused_group_final_merge(float* __restrict__ partial_dists, int32_t* __restrict__ partial_indices, float* __restrict__ out_dists, int32_t* __restrict__ out_indices, int total_queries)
{
    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;

    extern __shared__ __align__(1024) char smem_raw[];
    int smem;
    smem = (int)(unsigned long long)__cvta_generic_to_shared(smem_raw);
    const int smem_group_dists = smem + 0;
    const int smem_group_indices = smem + 1024;

    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    const int warp_id = warp;
    const int lane_id = lane;
    float* group_dists = (float*)(smem_raw + 0);
    #define group_dists_addr (smem + 0)
    int* group_indices = (int*)(smem_raw + 1024);
    #define group_indices_addr (smem + 1024)

    // === Task calls (dependency order) ===
    int split_pos[GROUP_SPLITS];
    int split_base[GROUP_SPLITS];
    float group_cand_d[GROUP_SPLITS];
    int group_cand_i[GROUP_SPLITS];
    int final_pos[GROUP_COUNT];
    float final_cand_d[GROUP_COUNT];
    int final_cand_i[GROUP_COUNT];
    #pragma unroll 1
    for (int row = bid; row < total_queries; row += num_bids) {
        int base_row = row * TOP_K_MAX;
        int split_stride = total_queries * TOP_K_MAX;
        if (tid < GROUP_COUNT) {
            int group_idx = tid;
            int source_split0 = group_idx * GROUP_SPLITS;
            int shared_base = group_idx * TOP_K_MAX;
            #pragma unroll
            for (int local_split = 0; local_split < GROUP_SPLITS; local_split++) {
                split_pos[local_split] = 0;
                int split_id = source_split0 + local_split;
                split_base[local_split] = base_row + split_id * split_stride;
                group_cand_d[local_split] = (float)partial_dists[split_base[local_split]];
                group_cand_i[local_split] = partial_indices[split_base[local_split]];
            }
            #pragma unroll
            for (int out_k = 0; out_k < TOP_K_MAX; out_k++) {
                float best_d = group_cand_d[0];
                int best_i = group_cand_i[0];
                int best_split = 0;
                #pragma unroll
                for (int local_split = 1; local_split < GROUP_SPLITS; local_split++) {
                    if (group_cand_d[local_split] < best_d) {
                        best_d = group_cand_d[local_split];
                        best_i = group_cand_i[local_split];
                        best_split = local_split;
                    }
                }
                group_dists[shared_base + out_k] = best_d;
                group_indices[shared_base + out_k] = best_i;
                split_pos[best_split] = split_pos[best_split] + 1;
                if (out_k + 1 < TOP_K_MAX) {
                    int next_pos = split_pos[best_split];
                    int next_addr = split_base[best_split] + next_pos;
                    group_cand_d[best_split] = (float)partial_dists[next_addr];
                    group_cand_i[best_split] = partial_indices[next_addr];
                }
            }
        }
        __syncthreads();
        if (tid == 0) {
            #pragma unroll
            for (int group_idx = 0; group_idx < GROUP_COUNT; group_idx++) {
                final_pos[group_idx] = 0;
                int group_base = group_idx * TOP_K_MAX;
                final_cand_d[group_idx] = group_dists[group_base];
                final_cand_i[group_idx] = group_indices[group_base];
            }
            #pragma unroll
            for (int out_k = 0; out_k < TOP_K_MAX; out_k++) {
                float best_d = final_cand_d[0];
                int best_i = final_cand_i[0];
                int best_group = 0;
                #pragma unroll
                for (int group_idx = 1; group_idx < GROUP_COUNT; group_idx++) {
                    if (final_cand_d[group_idx] < best_d) {
                        best_d = final_cand_d[group_idx];
                        best_i = final_cand_i[group_idx];
                        best_group = group_idx;
                    }
                }
                *((float*)(out_dists + base_row + out_k)) = best_d;
                *((int*)(out_indices + base_row + out_k)) = best_i;
                final_pos[best_group] = final_pos[best_group] + 1;
                if (out_k + 1 < TOP_K_MAX) {
                    int next_pos = final_pos[best_group];
                    int next_addr = best_group * TOP_K_MAX + next_pos;
                    final_cand_d[best_group] = group_dists[next_addr];
                    final_cand_i[best_group] = group_indices[next_addr];
                }
            }
        }
        __syncthreads();
    }
}

} // extern "C"

#undef GROUP_COUNT
#undef GROUP_SPLITS
#undef NUM_MAIN_STAGES
#undef SMEM_GROUP_DISTS_OFF
#undef SMEM_GROUP_DISTS_STAGE_BYTES
#undef SMEM_GROUP_DISTS_STRIDE
#undef SMEM_GROUP_INDICES_OFF
#undef SMEM_GROUP_INDICES_STAGE_BYTES
#undef SMEM_GROUP_INDICES_STRIDE
#undef SMEM_TOTAL
#undef THREADS
#undef TOP_K_MAX
#undef group_dists_addr
#undef group_indices_addr

#define NUM_MAIN_STAGES 1
#define SMEM_GROUP_DISTS_OFF 0
#define SMEM_GROUP_DISTS_STAGE_BYTES 512
#define SMEM_GROUP_DISTS_STRIDE 512
#define SMEM_GROUP_INDICES_OFF 512
#define SMEM_GROUP_INDICES_STAGE_BYTES 512
#define SMEM_GROUP_INDICES_STRIDE 512
#define SMEM_TOTAL 1024
#define THREADS 32
#define TOP_K_MAX 10
#define GROUP_COUNT 8
#define GROUP_SPLITS 9

extern "C" {

__global__ __launch_bounds__(32, 1) void
kernel_knn_build_rag_microbatch_4a72_k10_fused_group_final_merge(float* __restrict__ partial_dists, int32_t* __restrict__ partial_indices, float* __restrict__ out_dists, int32_t* __restrict__ out_indices, int total_queries)
{
    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;

    extern __shared__ __align__(1024) char smem_raw[];
    int smem;
    smem = (int)(unsigned long long)__cvta_generic_to_shared(smem_raw);
    const int smem_group_dists = smem + 0;
    const int smem_group_indices = smem + 512;

    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    const int warp_id = warp;
    const int lane_id = lane;
    float* group_dists = (float*)(smem_raw + 0);
    #define group_dists_addr (smem + 0)
    int* group_indices = (int*)(smem_raw + 512);
    #define group_indices_addr (smem + 512)

    // === Task calls (dependency order) ===
    int split_pos[GROUP_SPLITS];
    int split_base[GROUP_SPLITS];
    float group_cand_d[GROUP_SPLITS];
    int group_cand_i[GROUP_SPLITS];
    int final_pos[GROUP_COUNT];
    float final_cand_d[GROUP_COUNT];
    int final_cand_i[GROUP_COUNT];
    #pragma unroll 1
    for (int row = bid; row < total_queries; row += num_bids) {
        int base_row = row * TOP_K_MAX;
        int split_stride = total_queries * TOP_K_MAX;
        if (tid < GROUP_COUNT) {
            int group_idx = tid;
            int source_split0 = group_idx * GROUP_SPLITS;
            int shared_base = group_idx * TOP_K_MAX;
            #pragma unroll
            for (int local_split = 0; local_split < GROUP_SPLITS; local_split++) {
                split_pos[local_split] = 0;
                int split_id = source_split0 + local_split;
                split_base[local_split] = base_row + split_id * split_stride;
                group_cand_d[local_split] = (float)partial_dists[split_base[local_split]];
                group_cand_i[local_split] = partial_indices[split_base[local_split]];
            }
            #pragma unroll
            for (int out_k = 0; out_k < TOP_K_MAX; out_k++) {
                float best_d = group_cand_d[0];
                int best_i = group_cand_i[0];
                int best_split = 0;
                #pragma unroll
                for (int local_split = 1; local_split < GROUP_SPLITS; local_split++) {
                    if (group_cand_d[local_split] < best_d) {
                        best_d = group_cand_d[local_split];
                        best_i = group_cand_i[local_split];
                        best_split = local_split;
                    }
                }
                group_dists[shared_base + out_k] = best_d;
                group_indices[shared_base + out_k] = best_i;
                split_pos[best_split] = split_pos[best_split] + 1;
                if (out_k + 1 < TOP_K_MAX) {
                    int next_pos = split_pos[best_split];
                    int next_addr = split_base[best_split] + next_pos;
                    group_cand_d[best_split] = (float)partial_dists[next_addr];
                    group_cand_i[best_split] = partial_indices[next_addr];
                }
            }
        }
        __syncthreads();
        if (tid == 0) {
            #pragma unroll
            for (int group_idx = 0; group_idx < GROUP_COUNT; group_idx++) {
                final_pos[group_idx] = 0;
                int group_base = group_idx * TOP_K_MAX;
                final_cand_d[group_idx] = group_dists[group_base];
                final_cand_i[group_idx] = group_indices[group_base];
            }
            #pragma unroll
            for (int out_k = 0; out_k < TOP_K_MAX; out_k++) {
                float best_d = final_cand_d[0];
                int best_i = final_cand_i[0];
                int best_group = 0;
                #pragma unroll
                for (int group_idx = 1; group_idx < GROUP_COUNT; group_idx++) {
                    if (final_cand_d[group_idx] < best_d) {
                        best_d = final_cand_d[group_idx];
                        best_i = final_cand_i[group_idx];
                        best_group = group_idx;
                    }
                }
                *((float*)(out_dists + base_row + out_k)) = best_d;
                *((int*)(out_indices + base_row + out_k)) = best_i;
                final_pos[best_group] = final_pos[best_group] + 1;
                if (out_k + 1 < TOP_K_MAX) {
                    int next_pos = final_pos[best_group];
                    int next_addr = best_group * TOP_K_MAX + next_pos;
                    final_cand_d[best_group] = group_dists[next_addr];
                    final_cand_i[best_group] = group_indices[next_addr];
                }
            }
        }
        __syncthreads();
    }
}

} // extern "C"

#undef GROUP_COUNT
#undef GROUP_SPLITS
#undef NUM_MAIN_STAGES
#undef SMEM_GROUP_DISTS_OFF
#undef SMEM_GROUP_DISTS_STAGE_BYTES
#undef SMEM_GROUP_DISTS_STRIDE
#undef SMEM_GROUP_INDICES_OFF
#undef SMEM_GROUP_INDICES_STAGE_BYTES
#undef SMEM_GROUP_INDICES_STRIDE
#undef SMEM_TOTAL
#undef THREADS
#undef TOP_K_MAX
#undef group_dists_addr
#undef group_indices_addr

#define NUM_MAIN_STAGES 1
#define SMEM_GROUP_DISTS_OFF 0
#define SMEM_GROUP_DISTS_STAGE_BYTES 512
#define SMEM_GROUP_DISTS_STRIDE 512
#define SMEM_GROUP_INDICES_OFF 512
#define SMEM_GROUP_INDICES_STAGE_BYTES 512
#define SMEM_GROUP_INDICES_STRIDE 512
#define SMEM_TOTAL 1024
#define THREADS 32
#define TOP_K_MAX 10
#define GROUP_COUNT 8
#define GROUP_SPLITS 9

extern "C" {

__global__ __launch_bounds__(32, 1) void
kernel_knn_build_rag_microbatch_4a72_k10_fused_group_final_merge_s72g8_4a72_v1(float* __restrict__ partial_dists, int32_t* __restrict__ partial_indices, float* __restrict__ out_dists, int32_t* __restrict__ out_indices, int total_queries)
{
    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;

    extern __shared__ __align__(1024) char smem_raw[];
    int smem;
    smem = (int)(unsigned long long)__cvta_generic_to_shared(smem_raw);
    const int smem_group_dists = smem + 0;
    const int smem_group_indices = smem + 512;

    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    const int warp_id = warp;
    const int lane_id = lane;
    float* group_dists = (float*)(smem_raw + 0);
    #define group_dists_addr (smem + 0)
    int* group_indices = (int*)(smem_raw + 512);
    #define group_indices_addr (smem + 512)

    // === Task calls (dependency order) ===
    int split_pos[GROUP_SPLITS];
    int split_base[GROUP_SPLITS];
    float group_cand_d[GROUP_SPLITS];
    int group_cand_i[GROUP_SPLITS];
    int final_pos[GROUP_COUNT];
    float final_cand_d[GROUP_COUNT];
    int final_cand_i[GROUP_COUNT];
    #pragma unroll 1
    for (int row = bid; row < total_queries; row += num_bids) {
        int base_row = row * TOP_K_MAX;
        int split_stride = total_queries * TOP_K_MAX;
        if (tid < GROUP_COUNT) {
            int group_idx = tid;
            int source_split0 = group_idx * GROUP_SPLITS;
            int shared_base = group_idx * TOP_K_MAX;
            #pragma unroll
            for (int local_split = 0; local_split < GROUP_SPLITS; local_split++) {
                split_pos[local_split] = 0;
                int split_id = source_split0 + local_split;
                split_base[local_split] = base_row + split_id * split_stride;
                group_cand_d[local_split] = (float)partial_dists[split_base[local_split]];
                group_cand_i[local_split] = partial_indices[split_base[local_split]];
            }
            #pragma unroll
            for (int out_k = 0; out_k < TOP_K_MAX; out_k++) {
                float best_d = group_cand_d[0];
                int best_i = group_cand_i[0];
                int best_split = 0;
                #pragma unroll
                for (int local_split = 1; local_split < GROUP_SPLITS; local_split++) {
                    if (group_cand_d[local_split] < best_d) {
                        best_d = group_cand_d[local_split];
                        best_i = group_cand_i[local_split];
                        best_split = local_split;
                    }
                }
                group_dists[shared_base + out_k] = best_d;
                group_indices[shared_base + out_k] = best_i;
                split_pos[best_split] = split_pos[best_split] + 1;
                if (out_k + 1 < TOP_K_MAX) {
                    int next_pos = split_pos[best_split];
                    int next_addr = split_base[best_split] + next_pos;
                    group_cand_d[best_split] = (float)partial_dists[next_addr];
                    group_cand_i[best_split] = partial_indices[next_addr];
                }
            }
        }
        __syncthreads();
        if (tid == 0) {
            #pragma unroll
            for (int group_idx = 0; group_idx < GROUP_COUNT; group_idx++) {
                final_pos[group_idx] = 0;
                int group_base = group_idx * TOP_K_MAX;
                final_cand_d[group_idx] = group_dists[group_base];
                final_cand_i[group_idx] = group_indices[group_base];
            }
            #pragma unroll
            for (int out_k = 0; out_k < TOP_K_MAX; out_k++) {
                float best_d = final_cand_d[0];
                int best_i = final_cand_i[0];
                int best_group = 0;
                #pragma unroll
                for (int group_idx = 1; group_idx < GROUP_COUNT; group_idx++) {
                    if (final_cand_d[group_idx] < best_d) {
                        best_d = final_cand_d[group_idx];
                        best_i = final_cand_i[group_idx];
                        best_group = group_idx;
                    }
                }
                *((float*)(out_dists + base_row + out_k)) = best_d;
                *((int*)(out_indices + base_row + out_k)) = best_i;
                final_pos[best_group] = final_pos[best_group] + 1;
                if (out_k + 1 < TOP_K_MAX) {
                    int next_pos = final_pos[best_group];
                    int next_addr = best_group * TOP_K_MAX + next_pos;
                    final_cand_d[best_group] = group_dists[next_addr];
                    final_cand_i[best_group] = group_indices[next_addr];
                }
            }
        }
        __syncthreads();
    }
}

} // extern "C"

#undef GROUP_COUNT
#undef GROUP_SPLITS
#undef NUM_MAIN_STAGES
#undef SMEM_GROUP_DISTS_OFF
#undef SMEM_GROUP_DISTS_STAGE_BYTES
#undef SMEM_GROUP_DISTS_STRIDE
#undef SMEM_GROUP_INDICES_OFF
#undef SMEM_GROUP_INDICES_STAGE_BYTES
#undef SMEM_GROUP_INDICES_STRIDE
#undef SMEM_TOTAL
#undef THREADS
#undef TOP_K_MAX
#undef group_dists_addr
#undef group_indices_addr

#define TMEM_NCOLS 64
#define TMEM_CROSS_OFFSET 0
#define NUM_MAIN_STAGES 1
#define SMEM_SMEM_QUERY_OFF 1024
#define SMEM_SMEM_QUERY_STAGE_BYTES 32768
#define SMEM_SMEM_QUERY_STRIDE 32768
#define SMEM_SMEM_DATABASE_OFF 33792
#define SMEM_SMEM_DATABASE_STAGE_BYTES 16384
#define SMEM_SMEM_DATABASE_STRIDE 16384
#define SMEM_SMEM_DATABASE_SQ_OFF 50176
#define SMEM_SMEM_DATABASE_SQ_STAGE_BYTES 256
#define SMEM_SMEM_DATABASE_SQ_STRIDE 256
#define SMEM_TOTAL 50432
#define THREADS 192
#define BLOCK_Q 128
#define BLOCK_M 64
#define FEAT_D 128
#define TOP_K_MAX 10

extern "C" {

__global__ __launch_bounds__(192, 1) void
kernel_knn_build_rag_microbatch_4a72_v2_stage1_k10_cta1_maxtree(float* __restrict__ query_sq, float* __restrict__ database_sq, float* __restrict__ partial_dists, int32_t* __restrict__ partial_indices, const void* tmap_query, const void* tmap_database, int B, int Q, int M, int K, int num_q_tiles, int db_tiles_per_split, int split_count, int total_work)
{
    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;

    extern __shared__ __align__(1024) char smem_raw[];
    int smem;
    smem = (int)(unsigned long long)__cvta_generic_to_shared(smem_raw);
    const int smem_smem_query = smem + 1024;
    const int smem_smem_database = smem + 33792;
    const int smem_smem_database_sq = smem + 50176;

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
        // score_empty: 1 barriers, init_count=128
        mbarrier_init_pred(smem + 40, 128, leader);
        asm volatile("fence.mbarrier_init.release.cluster;");
    }

    __syncthreads();

    // TMEM alloc (64 columns, 64 used)
    volatile int* tmem_addr_storage = (volatile int*)(smem_raw + 48);
    if (warp == 1) {
        int _tmem_hold = smem + 48;
        asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;" :: "r"(_tmem_hold), "r"(64) : "memory");
    }

    __syncthreads();
    asm volatile("tcgen05.fence::after_thread_sync;");

    const int warp_id = warp;
    const int lane_id = lane;
    __nv_bfloat16* smem_query = (__nv_bfloat16*)(smem_raw + 1024);
    #define smem_query_addr (smem + 1024)
    __nv_bfloat16* smem_database = (__nv_bfloat16*)(smem_raw + 33792);
    #define smem_database_addr (smem + 33792)
    float* smem_database_sq = (float*)(smem_raw + 50176);
    #define smem_database_sq_addr (smem + 50176)
    const int mbar_base = smem;
    #define query_full_addr (mbar_base + 0)
    #define query_empty_addr (mbar_base + 8)
    #define database_full_addr (mbar_base + 16)
    #define database_empty_addr (mbar_base + 24)
    #define score_full_addr (mbar_base + 32)
    #define score_empty_addr (mbar_base + 40)
    const int taddr = tmem_addr_storage[0];

    // ---- Role: load ----
    if (warp == 0) {
        { // load_main
            uint32_t _phase_query_empty_0 = 1;
            uint32_t _phase_database_empty_0 = 1;
            if (warp_id == 0) {
                if (elect_sync()) {
                    #pragma unroll 1
                    for (unsigned int work_idx = bid; work_idx < total_work; work_idx += num_bids) {
                        int split_idx = work_idx % split_count;
                        int query_work = work_idx / split_count;
                        int batch_idx = query_work / num_q_tiles;
                        int q_tile = query_work % num_q_tiles;
                        int off_q = q_tile * BLOCK_Q;
                        int global_q = batch_idx * Q + off_q;
                        int db_tile_start = split_idx * db_tiles_per_split;
                        mbarrier_wait(query_empty_addr, _phase_query_empty_0);
                        _phase_query_empty_0 ^= 1;
                        mbarrier_arrive_expect_tx(query_full_addr, 32768);
                        tma_3d_gmem2smem(smem_query_addr, tmap_query, 0, global_q, 0, query_full_addr);
                        #pragma unroll 1
                        for (int local_db_tile = 0; local_db_tile < db_tiles_per_split; local_db_tile++) {
                            int db_tile = db_tile_start + local_db_tile;
                            int off_m = db_tile * BLOCK_M;
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
    } else if (warp == 1) {
        { // mma_main
            uint32_t _phase_query_full_0 = 0;
            uint32_t _phase_score_empty_0 = 1;
            uint32_t _phase_database_full_0 = 0;
            #pragma unroll 1
            for (unsigned int work_idx = bid; work_idx < total_work; work_idx += num_bids) {
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
                    "mov.b32 id, 135267472;\n\t"
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
    // ---- Role: compute ----
    } else if (warp >= 2 && warp <= 5) {
        const int warp_id_in_wg = warp % 4;
        const int my_row = warp_id_in_wg * 32 + lane;
        const int tmem_row_base_v = (warp_id_in_wg * 32) << 16;
        const int tmem_cross = taddr + TMEM_CROSS_OFFSET;
        { // compute_main
            uint32_t _phase_score_full_0 = 0;
            #pragma unroll 1
            for (unsigned int work_idx = bid; work_idx < total_work; work_idx += num_bids) {
                int split_idx = work_idx % split_count;
                int query_work = work_idx / split_count;
                int batch_idx = query_work / num_q_tiles;
                int q_tile = query_work % num_q_tiles;
                int off_q = q_tile * BLOCK_Q;
                int q_idx = off_q + my_row;
                int valid_q = ((q_idx < Q) ? 1 : 0);
                float q_sq_val = 0.0f;
                if (valid_q != 0) {
                    q_sq_val = (float)query_sq[batch_idx * Q + q_idx];
                }
                float best_d[TOP_K_MAX];
                int best_i[TOP_K_MAX];
                float worst_d = 3.4e+38f;
                int worst_pos = 0;
                #pragma unroll
                for (int kk = 0; kk < TOP_K_MAX; kk++) {
                    best_d[kk] = 3.4e+38f;
                    best_i[kk] = -1;
                }
                int db_tile_start = split_idx * db_tiles_per_split;
                #pragma unroll 1
                for (int local_db_tile = 0; local_db_tile < db_tiles_per_split; local_db_tile++) {
                    int db_tile = db_tile_start + local_db_tile;
                    int db_start = db_tile * BLOCK_M;
                    int db_sq_idx = db_start + my_row;
                    if (my_row < BLOCK_M) {
                        if (db_sq_idx < M) {
                            smem_database_sq[my_row] = (float)database_sq[batch_idx * M + db_sq_idx];
                        } else {
                            smem_database_sq[my_row] = 0.0f;
                        }
                    }
                    asm volatile("barrier.sync 8, 128;");
                    mbarrier_wait(score_full_addr, _phase_score_full_0);
                    _phase_score_full_0 ^= 1;
                    int cross_addr = taddr + tmem_row_base_v;
                    float dots[64];
                    asm volatile(
                        "tcgen05.ld.sync.aligned.32x32b.x64.b32"
                        " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63}, [%64];"
                        : "=f"(dots[0]), "=f"(dots[1]), "=f"(dots[2]), "=f"(dots[3]), "=f"(dots[4]), "=f"(dots[5]), "=f"(dots[6]), "=f"(dots[7]), "=f"(dots[8]), "=f"(dots[9]), "=f"(dots[10]), "=f"(dots[11]), "=f"(dots[12]), "=f"(dots[13]), "=f"(dots[14]), "=f"(dots[15]), "=f"(dots[16]), "=f"(dots[17]), "=f"(dots[18]), "=f"(dots[19]), "=f"(dots[20]), "=f"(dots[21]), "=f"(dots[22]), "=f"(dots[23]), "=f"(dots[24]), "=f"(dots[25]), "=f"(dots[26]), "=f"(dots[27]), "=f"(dots[28]), "=f"(dots[29]), "=f"(dots[30]), "=f"(dots[31]), "=f"(dots[32]), "=f"(dots[33]), "=f"(dots[34]), "=f"(dots[35]), "=f"(dots[36]), "=f"(dots[37]), "=f"(dots[38]), "=f"(dots[39]), "=f"(dots[40]), "=f"(dots[41]), "=f"(dots[42]), "=f"(dots[43]), "=f"(dots[44]), "=f"(dots[45]), "=f"(dots[46]), "=f"(dots[47]), "=f"(dots[48]), "=f"(dots[49]), "=f"(dots[50]), "=f"(dots[51]), "=f"(dots[52]), "=f"(dots[53]), "=f"(dots[54]), "=f"(dots[55]), "=f"(dots[56]), "=f"(dots[57]), "=f"(dots[58]), "=f"(dots[59]), "=f"(dots[60]), "=f"(dots[61]), "=f"(dots[62]), "=f"(dots[63])
                        : "r"(cross_addr)
                        : "memory");
                    asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                    mbarrier_arrive(score_empty_addr);
                    if (valid_q != 0) {
                        #pragma unroll 1
                        for (int col_base = 0; col_base < 64; col_base += 8) {
                            float dist_vec0[4];
                            dist_vec0[0] = dots[col_base];
                            dist_vec0[1] = dots[col_base + 1];
                            dist_vec0[2] = dots[col_base + 2];
                            dist_vec0[3] = dots[col_base + 3];
                            const float2 _fma_b2_0 = {-2.0f, -2.0f};
                            const float2 _fma_c2_1 = {q_sq_val, q_sq_val};
                            #pragma unroll
                            for (int _lf = 0; _lf < 2; _lf++)
                                fma_f32x2_inplace(&reinterpret_cast<float2*>(dist_vec0)[_lf], _fma_b2_0, _fma_c2_1);
                            float db_sq_vec0[4];
                            db_sq_vec0[0] = smem_database_sq[col_base];
                            db_sq_vec0[1] = smem_database_sq[col_base + 1];
                            db_sq_vec0[2] = smem_database_sq[col_base + 2];
                            db_sq_vec0[3] = smem_database_sq[col_base + 3];
                            float _t0[4];
                            #pragma unroll
                            for (int _la = 0; _la < 2; _la++)
                                reinterpret_cast<float2*>(_t0)[_la] = add_f32x2(reinterpret_cast<float2*>(dist_vec0)[_la], reinterpret_cast<const float2*>(db_sq_vec0)[_la]);
                            float dist_vec1[4];
                            dist_vec1[0] = dots[col_base + 4];
                            dist_vec1[1] = dots[col_base + 5];
                            dist_vec1[2] = dots[col_base + 6];
                            dist_vec1[3] = dots[col_base + 7];
                            const float2 _fma_b2_2 = {-2.0f, -2.0f};
                            const float2 _fma_c2_3 = {q_sq_val, q_sq_val};
                            #pragma unroll
                            for (int _lf = 0; _lf < 2; _lf++)
                                fma_f32x2_inplace(&reinterpret_cast<float2*>(dist_vec1)[_lf], _fma_b2_2, _fma_c2_3);
                            float db_sq_vec1[4];
                            db_sq_vec1[0] = smem_database_sq[col_base + 4];
                            db_sq_vec1[1] = smem_database_sq[col_base + 5];
                            db_sq_vec1[2] = smem_database_sq[col_base + 6];
                            db_sq_vec1[3] = smem_database_sq[col_base + 7];
                            float _t1[4];
                            #pragma unroll
                            for (int _la = 0; _la < 2; _la++)
                                reinterpret_cast<float2*>(_t1)[_la] = add_f32x2(reinterpret_cast<float2*>(dist_vec1)[_la], reinterpret_cast<const float2*>(db_sq_vec1)[_la]);
                            float _t0_min = _t0[0];
                            #pragma unroll
                            for (int _lr = 1; _lr < 4; _lr++) {
                                _t0_min = fminf(_t0_min, _t0[_lr]);
                            }
                            float group_min0 = _t0_min;
                            float _t1_min = _t1[0];
                            #pragma unroll
                            for (int _lr = 1; _lr < 4; _lr++) {
                                _t1_min = fminf(_t1_min, _t1[_lr]);
                            }
                            float group_min1 = _t1_min;
                            if (group_min0 < worst_d) {
                                #pragma unroll
                                for (int vec_col = 0; vec_col < 4; vec_col++) {
                                    int db_idx = db_start + col_base + vec_col;
                                    if (db_idx < M) {
                                        float dist = _t0[vec_col];
                                        if (dist < worst_d) {
                                            best_d[worst_pos] = dist;
                                            best_i[worst_pos] = db_idx;
                                            int cmp01 = ((best_d[1] > best_d[0]) ? 1 : 0);
                                            float max01_d = ((cmp01 != 0) ? best_d[1] : best_d[0]);
                                            int max01_p = ((cmp01 != 0) ? 1 : 0);
                                            int cmp23 = ((best_d[3] > best_d[2]) ? 1 : 0);
                                            float max23_d = ((cmp23 != 0) ? best_d[3] : best_d[2]);
                                            int max23_p = ((cmp23 != 0) ? 3 : 2);
                                            int cmp45 = ((best_d[5] > best_d[4]) ? 1 : 0);
                                            float max45_d = ((cmp45 != 0) ? best_d[5] : best_d[4]);
                                            int max45_p = ((cmp45 != 0) ? 5 : 4);
                                            int cmp67 = ((best_d[7] > best_d[6]) ? 1 : 0);
                                            float max67_d = ((cmp67 != 0) ? best_d[7] : best_d[6]);
                                            int max67_p = ((cmp67 != 0) ? 7 : 6);
                                            int cmp89 = ((best_d[9] > best_d[8]) ? 1 : 0);
                                            float max89_d = ((cmp89 != 0) ? best_d[9] : best_d[8]);
                                            int max89_p = ((cmp89 != 0) ? 9 : 8);
                                            int cmp0123 = ((max23_d > max01_d) ? 1 : 0);
                                            float max0123_d = ((cmp0123 != 0) ? max23_d : max01_d);
                                            int max0123_p = ((cmp0123 != 0) ? max23_p : max01_p);
                                            int cmp4567 = ((max67_d > max45_d) ? 1 : 0);
                                            float max4567_d = ((cmp4567 != 0) ? max67_d : max45_d);
                                            int max4567_p = ((cmp4567 != 0) ? max67_p : max45_p);
                                            int cmp0_7 = ((max4567_d > max0123_d) ? 1 : 0);
                                            float max0_7_d = ((cmp0_7 != 0) ? max4567_d : max0123_d);
                                            int max0_7_p = ((cmp0_7 != 0) ? max4567_p : max0123_p);
                                            int cmp_all = ((max89_d > max0_7_d) ? 1 : 0);
                                            worst_d = ((cmp_all != 0) ? max89_d : max0_7_d);
                                            worst_pos = ((cmp_all != 0) ? max89_p : max0_7_p);
                                        }
                                    }
                                }
                            }
                            if (group_min1 < worst_d) {
                                #pragma unroll
                                for (int vec_col = 0; vec_col < 4; vec_col++) {
                                    int db_idx = db_start + col_base + 4 + vec_col;
                                    if (db_idx < M) {
                                        float dist = _t1[vec_col];
                                        if (dist < worst_d) {
                                            best_d[worst_pos] = dist;
                                            best_i[worst_pos] = db_idx;
                                            int cmp01 = ((best_d[1] > best_d[0]) ? 1 : 0);
                                            float max01_d = ((cmp01 != 0) ? best_d[1] : best_d[0]);
                                            int max01_p = ((cmp01 != 0) ? 1 : 0);
                                            int cmp23 = ((best_d[3] > best_d[2]) ? 1 : 0);
                                            float max23_d = ((cmp23 != 0) ? best_d[3] : best_d[2]);
                                            int max23_p = ((cmp23 != 0) ? 3 : 2);
                                            int cmp45 = ((best_d[5] > best_d[4]) ? 1 : 0);
                                            float max45_d = ((cmp45 != 0) ? best_d[5] : best_d[4]);
                                            int max45_p = ((cmp45 != 0) ? 5 : 4);
                                            int cmp67 = ((best_d[7] > best_d[6]) ? 1 : 0);
                                            float max67_d = ((cmp67 != 0) ? best_d[7] : best_d[6]);
                                            int max67_p = ((cmp67 != 0) ? 7 : 6);
                                            int cmp89 = ((best_d[9] > best_d[8]) ? 1 : 0);
                                            float max89_d = ((cmp89 != 0) ? best_d[9] : best_d[8]);
                                            int max89_p = ((cmp89 != 0) ? 9 : 8);
                                            int cmp0123 = ((max23_d > max01_d) ? 1 : 0);
                                            float max0123_d = ((cmp0123 != 0) ? max23_d : max01_d);
                                            int max0123_p = ((cmp0123 != 0) ? max23_p : max01_p);
                                            int cmp4567 = ((max67_d > max45_d) ? 1 : 0);
                                            float max4567_d = ((cmp4567 != 0) ? max67_d : max45_d);
                                            int max4567_p = ((cmp4567 != 0) ? max67_p : max45_p);
                                            int cmp0_7 = ((max4567_d > max0123_d) ? 1 : 0);
                                            float max0_7_d = ((cmp0_7 != 0) ? max4567_d : max0123_d);
                                            int max0_7_p = ((cmp0_7 != 0) ? max4567_p : max0123_p);
                                            int cmp_all = ((max89_d > max0_7_d) ? 1 : 0);
                                            worst_d = ((cmp_all != 0) ? max89_d : max0_7_d);
                                            worst_pos = ((cmp_all != 0) ? max89_p : max0_7_p);
                                        }
                                    }
                                }
                            }
                        }
                    }
                    asm volatile("barrier.sync 8, 128;");
                }
                if (valid_q != 0) {
                    int out_base = ((split_idx * B + batch_idx) * Q + q_idx) * K;
                    #pragma unroll
                    for (int out_k = 0; out_k < TOP_K_MAX; out_k++) {
                        int cmp01_min = ((best_d[1] < best_d[0]) ? 1 : 0);
                        float min01_d = ((cmp01_min != 0) ? best_d[1] : best_d[0]);
                        int min01_i = ((cmp01_min != 0) ? best_i[1] : best_i[0]);
                        int min01_p = ((cmp01_min != 0) ? 1 : 0);
                        int cmp23_min = ((best_d[3] < best_d[2]) ? 1 : 0);
                        float min23_d = ((cmp23_min != 0) ? best_d[3] : best_d[2]);
                        int min23_i = ((cmp23_min != 0) ? best_i[3] : best_i[2]);
                        int min23_p = ((cmp23_min != 0) ? 3 : 2);
                        int cmp45_min = ((best_d[5] < best_d[4]) ? 1 : 0);
                        float min45_d = ((cmp45_min != 0) ? best_d[5] : best_d[4]);
                        int min45_i = ((cmp45_min != 0) ? best_i[5] : best_i[4]);
                        int min45_p = ((cmp45_min != 0) ? 5 : 4);
                        int cmp67_min = ((best_d[7] < best_d[6]) ? 1 : 0);
                        float min67_d = ((cmp67_min != 0) ? best_d[7] : best_d[6]);
                        int min67_i = ((cmp67_min != 0) ? best_i[7] : best_i[6]);
                        int min67_p = ((cmp67_min != 0) ? 7 : 6);
                        int cmp89_min = ((best_d[9] < best_d[8]) ? 1 : 0);
                        float min89_d = ((cmp89_min != 0) ? best_d[9] : best_d[8]);
                        int min89_i = ((cmp89_min != 0) ? best_i[9] : best_i[8]);
                        int min89_p = ((cmp89_min != 0) ? 9 : 8);
                        int cmp0123_min = ((min23_d < min01_d) ? 1 : 0);
                        float min0123_d = ((cmp0123_min != 0) ? min23_d : min01_d);
                        int min0123_i = ((cmp0123_min != 0) ? min23_i : min01_i);
                        int min0123_p = ((cmp0123_min != 0) ? min23_p : min01_p);
                        int cmp4567_min = ((min67_d < min45_d) ? 1 : 0);
                        float min4567_d = ((cmp4567_min != 0) ? min67_d : min45_d);
                        int min4567_i = ((cmp4567_min != 0) ? min67_i : min45_i);
                        int min4567_p = ((cmp4567_min != 0) ? min67_p : min45_p);
                        int cmp0_7_min = ((min4567_d < min0123_d) ? 1 : 0);
                        float min0_7_d = ((cmp0_7_min != 0) ? min4567_d : min0123_d);
                        int min0_7_i = ((cmp0_7_min != 0) ? min4567_i : min0123_i);
                        int min0_7_p = ((cmp0_7_min != 0) ? min4567_p : min0123_p);
                        int cmp_all_min = ((min89_d < min0_7_d) ? 1 : 0);
                        float selected_d = ((cmp_all_min != 0) ? min89_d : min0_7_d);
                        int selected_i = ((cmp_all_min != 0) ? min89_i : min0_7_i);
                        int selected_pos = ((cmp_all_min != 0) ? min89_p : min0_7_p);
                        if (out_k < K) {
                            *((float*)(partial_dists + out_base + out_k)) = selected_d;
                            *((int*)(partial_indices + out_base + out_k)) = selected_i;
                        }
                        best_d[selected_pos] = 3.4e+38f;
                        best_i[selected_pos] = -1;
                    }
                }
            }
        }
    }

    // Cleanup
    __syncthreads(); // barrier before TMEM dealloc

    if (warp == 1) {
        asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;" :: "r"(tmem_addr_storage[0]), "r"(64));
        asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;");
    }
}

} // extern "C"

#undef BLOCK_M
#undef BLOCK_Q
#undef FEAT_D
#undef NUM_MAIN_STAGES
#undef SMEM_SMEM_DATABASE_OFF
#undef SMEM_SMEM_DATABASE_SQ_OFF
#undef SMEM_SMEM_DATABASE_SQ_STAGE_BYTES
#undef SMEM_SMEM_DATABASE_SQ_STRIDE
#undef SMEM_SMEM_DATABASE_STAGE_BYTES
#undef SMEM_SMEM_DATABASE_STRIDE
#undef SMEM_SMEM_QUERY_OFF
#undef SMEM_SMEM_QUERY_STAGE_BYTES
#undef SMEM_SMEM_QUERY_STRIDE
#undef SMEM_TOTAL
#undef THREADS
#undef TMEM_CROSS_OFFSET
#undef TMEM_NCOLS
#undef TOP_K_MAX
#undef database_empty_addr
#undef database_full_addr
#undef query_empty_addr
#undef query_full_addr
#undef score_empty_addr
#undef score_full_addr
#undef smem_database_addr
#undef smem_database_sq_addr
#undef smem_query_addr

#define NUM_MAIN_STAGES 1
#define SMEM_GROUP_DISTS_OFF 0
#define SMEM_GROUP_DISTS_STAGE_BYTES 512
#define SMEM_GROUP_DISTS_STRIDE 512
#define SMEM_GROUP_INDICES_OFF 512
#define SMEM_GROUP_INDICES_STAGE_BYTES 512
#define SMEM_GROUP_INDICES_STRIDE 512
#define SMEM_TOTAL 1024
#define THREADS 32
#define TOP_K_MAX 10
#define GROUP_COUNT 8
#define GROUP_SPLITS 9

extern "C" {

__global__ __launch_bounds__(32, 1) void
kernel_knn_build_rag_microbatch_4a72_v2_k10_fused_group_final_merge(float* __restrict__ partial_dists, int32_t* __restrict__ partial_indices, float* __restrict__ out_dists, int32_t* __restrict__ out_indices, int total_queries)
{
    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;

    extern __shared__ __align__(1024) char smem_raw[];
    int smem;
    smem = (int)(unsigned long long)__cvta_generic_to_shared(smem_raw);
    const int smem_group_dists = smem + 0;
    const int smem_group_indices = smem + 512;

    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    const int warp_id = warp;
    const int lane_id = lane;
    float* group_dists = (float*)(smem_raw + 0);
    #define group_dists_addr (smem + 0)
    int* group_indices = (int*)(smem_raw + 512);
    #define group_indices_addr (smem + 512)

    // === Task calls (dependency order) ===
    int split_pos[GROUP_SPLITS];
    int split_base[GROUP_SPLITS];
    float group_cand_d[GROUP_SPLITS];
    int group_cand_i[GROUP_SPLITS];
    int final_pos[GROUP_COUNT];
    float final_cand_d[GROUP_COUNT];
    int final_cand_i[GROUP_COUNT];
    #pragma unroll 1
    for (int row = bid; row < total_queries; row += num_bids) {
        int base_row = row * TOP_K_MAX;
        int split_stride = total_queries * TOP_K_MAX;
        if (tid < GROUP_COUNT) {
            int group_idx = tid;
            int source_split0 = group_idx * GROUP_SPLITS;
            int shared_base = group_idx * TOP_K_MAX;
            #pragma unroll
            for (int local_split = 0; local_split < GROUP_SPLITS; local_split++) {
                split_pos[local_split] = 0;
                int split_id = source_split0 + local_split;
                split_base[local_split] = base_row + split_id * split_stride;
                group_cand_d[local_split] = (float)partial_dists[split_base[local_split]];
                group_cand_i[local_split] = partial_indices[split_base[local_split]];
            }
            #pragma unroll
            for (int out_k = 0; out_k < TOP_K_MAX; out_k++) {
                float best_d = group_cand_d[0];
                int best_i = group_cand_i[0];
                int best_split = 0;
                #pragma unroll
                for (int local_split = 1; local_split < GROUP_SPLITS; local_split++) {
                    if (group_cand_d[local_split] < best_d) {
                        best_d = group_cand_d[local_split];
                        best_i = group_cand_i[local_split];
                        best_split = local_split;
                    }
                }
                group_dists[shared_base + out_k] = best_d;
                group_indices[shared_base + out_k] = best_i;
                split_pos[best_split] = split_pos[best_split] + 1;
                if (out_k + 1 < TOP_K_MAX) {
                    int next_pos = split_pos[best_split];
                    int next_addr = split_base[best_split] + next_pos;
                    group_cand_d[best_split] = (float)partial_dists[next_addr];
                    group_cand_i[best_split] = partial_indices[next_addr];
                }
            }
        }
        __syncthreads();
        if (tid == 0) {
            #pragma unroll
            for (int group_idx = 0; group_idx < GROUP_COUNT; group_idx++) {
                final_pos[group_idx] = 0;
                int group_base = group_idx * TOP_K_MAX;
                final_cand_d[group_idx] = group_dists[group_base];
                final_cand_i[group_idx] = group_indices[group_base];
            }
            #pragma unroll
            for (int out_k = 0; out_k < TOP_K_MAX; out_k++) {
                float best_d = final_cand_d[0];
                int best_i = final_cand_i[0];
                int best_group = 0;
                #pragma unroll
                for (int group_idx = 1; group_idx < GROUP_COUNT; group_idx++) {
                    if (final_cand_d[group_idx] < best_d) {
                        best_d = final_cand_d[group_idx];
                        best_i = final_cand_i[group_idx];
                        best_group = group_idx;
                    }
                }
                *((float*)(out_dists + base_row + out_k)) = best_d;
                *((int*)(out_indices + base_row + out_k)) = best_i;
                final_pos[best_group] = final_pos[best_group] + 1;
                if (out_k + 1 < TOP_K_MAX) {
                    int next_pos = final_pos[best_group];
                    int next_addr = best_group * TOP_K_MAX + next_pos;
                    final_cand_d[best_group] = group_dists[next_addr];
                    final_cand_i[best_group] = group_indices[next_addr];
                }
            }
        }
        __syncthreads();
    }
}

} // extern "C"

#undef GROUP_COUNT
#undef GROUP_SPLITS
#undef NUM_MAIN_STAGES
#undef SMEM_GROUP_DISTS_OFF
#undef SMEM_GROUP_DISTS_STAGE_BYTES
#undef SMEM_GROUP_DISTS_STRIDE
#undef SMEM_GROUP_INDICES_OFF
#undef SMEM_GROUP_INDICES_STAGE_BYTES
#undef SMEM_GROUP_INDICES_STRIDE
#undef SMEM_TOTAL
#undef THREADS
#undef TOP_K_MAX
#undef group_dists_addr
#undef group_indices_addr

#define NUM_MAIN_STAGES 1
#define SMEM_GROUP_DISTS_OFF 0
#define SMEM_GROUP_DISTS_STAGE_BYTES 512
#define SMEM_GROUP_DISTS_STRIDE 512
#define SMEM_GROUP_INDICES_OFF 512
#define SMEM_GROUP_INDICES_STAGE_BYTES 512
#define SMEM_GROUP_INDICES_STRIDE 512
#define SMEM_TOTAL 1024
#define THREADS 32
#define TOP_K_MAX 10
#define GROUP_COUNT 12
#define GROUP_SPLITS 12

extern "C" {

__global__ __launch_bounds__(32, 1) void
kernel_knn_build_rag_microbatch_4a72_v2_k10_fused_group_final_merge_s144g12_4a72_v2(float* __restrict__ partial_dists, int32_t* __restrict__ partial_indices, float* __restrict__ out_dists, int32_t* __restrict__ out_indices, int total_queries)
{
    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;

    extern __shared__ __align__(1024) char smem_raw[];
    int smem;
    smem = (int)(unsigned long long)__cvta_generic_to_shared(smem_raw);
    const int smem_group_dists = smem + 0;
    const int smem_group_indices = smem + 512;

    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    const int warp_id = warp;
    const int lane_id = lane;
    float* group_dists = (float*)(smem_raw + 0);
    #define group_dists_addr (smem + 0)
    int* group_indices = (int*)(smem_raw + 512);
    #define group_indices_addr (smem + 512)

    // === Task calls (dependency order) ===
    int split_pos[GROUP_SPLITS];
    int split_base[GROUP_SPLITS];
    float group_cand_d[GROUP_SPLITS];
    int group_cand_i[GROUP_SPLITS];
    int final_pos[GROUP_COUNT];
    float final_cand_d[GROUP_COUNT];
    int final_cand_i[GROUP_COUNT];
    #pragma unroll 1
    for (int row = bid; row < total_queries; row += num_bids) {
        int base_row = row * TOP_K_MAX;
        int split_stride = total_queries * TOP_K_MAX;
        if (tid < GROUP_COUNT) {
            int group_idx = tid;
            int source_split0 = group_idx * GROUP_SPLITS;
            int shared_base = group_idx * TOP_K_MAX;
            #pragma unroll
            for (int local_split = 0; local_split < GROUP_SPLITS; local_split++) {
                split_pos[local_split] = 0;
                int split_id = source_split0 + local_split;
                split_base[local_split] = base_row + split_id * split_stride;
                group_cand_d[local_split] = (float)partial_dists[split_base[local_split]];
                group_cand_i[local_split] = partial_indices[split_base[local_split]];
            }
            #pragma unroll
            for (int out_k = 0; out_k < TOP_K_MAX; out_k++) {
                float best_d = group_cand_d[0];
                int best_i = group_cand_i[0];
                int best_split = 0;
                #pragma unroll
                for (int local_split = 1; local_split < GROUP_SPLITS; local_split++) {
                    if (group_cand_d[local_split] < best_d) {
                        best_d = group_cand_d[local_split];
                        best_i = group_cand_i[local_split];
                        best_split = local_split;
                    }
                }
                group_dists[shared_base + out_k] = best_d;
                group_indices[shared_base + out_k] = best_i;
                split_pos[best_split] = split_pos[best_split] + 1;
                if (out_k + 1 < TOP_K_MAX) {
                    int next_pos = split_pos[best_split];
                    int next_addr = split_base[best_split] + next_pos;
                    group_cand_d[best_split] = (float)partial_dists[next_addr];
                    group_cand_i[best_split] = partial_indices[next_addr];
                }
            }
        }
        __syncthreads();
        if (tid == 0) {
            #pragma unroll
            for (int group_idx = 0; group_idx < GROUP_COUNT; group_idx++) {
                final_pos[group_idx] = 0;
                int group_base = group_idx * TOP_K_MAX;
                final_cand_d[group_idx] = group_dists[group_base];
                final_cand_i[group_idx] = group_indices[group_base];
            }
            #pragma unroll
            for (int out_k = 0; out_k < TOP_K_MAX; out_k++) {
                float best_d = final_cand_d[0];
                int best_i = final_cand_i[0];
                int best_group = 0;
                #pragma unroll
                for (int group_idx = 1; group_idx < GROUP_COUNT; group_idx++) {
                    if (final_cand_d[group_idx] < best_d) {
                        best_d = final_cand_d[group_idx];
                        best_i = final_cand_i[group_idx];
                        best_group = group_idx;
                    }
                }
                *((float*)(out_dists + base_row + out_k)) = best_d;
                *((int*)(out_indices + base_row + out_k)) = best_i;
                final_pos[best_group] = final_pos[best_group] + 1;
                if (out_k + 1 < TOP_K_MAX) {
                    int next_pos = final_pos[best_group];
                    int next_addr = best_group * TOP_K_MAX + next_pos;
                    final_cand_d[best_group] = group_dists[next_addr];
                    final_cand_i[best_group] = group_indices[next_addr];
                }
            }
        }
        __syncthreads();
    }
}

} // extern "C"

#undef GROUP_COUNT
#undef GROUP_SPLITS
#undef NUM_MAIN_STAGES
#undef SMEM_GROUP_DISTS_OFF
#undef SMEM_GROUP_DISTS_STAGE_BYTES
#undef SMEM_GROUP_DISTS_STRIDE
#undef SMEM_GROUP_INDICES_OFF
#undef SMEM_GROUP_INDICES_STAGE_BYTES
#undef SMEM_GROUP_INDICES_STRIDE
#undef SMEM_TOTAL
#undef THREADS
#undef TOP_K_MAX
#undef group_dists_addr
#undef group_indices_addr

#define TMEM_NCOLS 128
#define TMEM_ACC_OFFSET 0
#define NUM_MAIN_STAGES 1
#define SMEM_SMEM_QUERY_OFF 1024
#define SMEM_SMEM_QUERY_STAGE_BYTES 16384
#define SMEM_SMEM_QUERY_STRIDE 16384
#define SMEM_SMEM_DATABASE_OFF 17408
#define SMEM_SMEM_DATABASE_STAGE_BYTES 32768
#define SMEM_SMEM_DATABASE_STRIDE 32768
#define SMEM_SMEM_LOCAL_D_OFF 50176
#define SMEM_SMEM_LOCAL_D_STAGE_BYTES 20480
#define SMEM_SMEM_LOCAL_D_STRIDE 20480
#define SMEM_SMEM_LOCAL_I_OFF 70656
#define SMEM_SMEM_LOCAL_I_STAGE_BYTES 20480
#define SMEM_SMEM_LOCAL_I_STRIDE 20480
#define SMEM_TOTAL 91392
#define THREADS 512

extern "C" {

__global__ __launch_bounds__(512, 1) void
kernel_knn_build_rag_microbatch_m64_d4f7_stage1(__nv_bfloat16* __restrict__ query, __nv_bfloat16* __restrict__ database, float* __restrict__ query_sq, float* __restrict__ database_sq, float* __restrict__ partial_dists, int32_t* __restrict__ partial_indices, int B, int Q, int M, int K, int num_q_tiles, int db_tiles_per_split, int split_count, int total_work)
{
    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;

    extern __shared__ __align__(1024) char smem_raw[];
    int smem;
    smem = (int)(unsigned long long)__cvta_generic_to_shared(smem_raw);
    const int smem_smem_query = smem + 1024;
    const int smem_smem_database = smem + 17408;
    const int smem_smem_local_d = smem + 50176;
    const int smem_smem_local_i = smem + 70656;

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
    __nv_bfloat16* smem_query = (__nv_bfloat16*)(smem_raw + 1024);
    #define smem_query_addr (smem + 1024)
    __nv_bfloat16* smem_database = (__nv_bfloat16*)(smem_raw + 17408);
    #define smem_database_addr (smem + 17408)
    float* smem_local_d = (float*)(smem_raw + 50176);
    #define smem_local_d_addr (smem + 50176)
    int* smem_local_i = (int*)(smem_raw + 70656);
    #define smem_local_i_addr (smem + 70656)
    const int mbar_base = smem;
    #define mma_done_addr (mbar_base + 0)
    const int taddr = tmem_addr_storage[0];

    const int tmem_row_base = (warp % 16) * 32;
    const int my_row = tmem_row_base + (lane / 4);
    // === Task calls (dependency order) ===
    int _desc_lo_0 = make_warp_uniform((smem_query_addr >> 4) & 0x3FFF);
    int _desc_lo_1 = make_warp_uniform((smem_database_addr >> 4) & 0x3FFF);
    uint32_t _phase_mma_done_0 = 0;
    #pragma unroll 1
    for (unsigned int work_idx = bid; work_idx < total_work; work_idx += num_bids) {
        int split_idx = work_idx % split_count;
        int query_work = work_idx / split_count;
        int batch_idx = query_work / num_q_tiles;
        int q_tile = query_work % num_q_tiles;
        int off_q = q_tile * 64;
        #pragma unroll 1
        for (int e_vec = tid; e_vec < 1024; e_vec += 512) {
            int q_elem = e_vec * 8;
            int q_row = q_elem / 128;
            int d_col = q_elem - q_row * 128;
            int q_idx = off_q + q_row;
            float q_vals[8];
            unsigned int q_pack[4];
            #pragma unroll
            for (int vi = 0; vi < 8; vi++) {
                q_vals[vi] = 0.0f;
            }
            if (q_idx < Q) {
                int q_addr = (batch_idx * Q + q_idx) * 128 + d_col;
                {
                    const uint4* _vptr_0 = reinterpret_cast<const uint4*>(query + (unsigned long long)q_addr);
                    uint4 _vld_0[1];
                    #pragma unroll
                    for (int _blk = 0; _blk < 1; _blk++) {
                        _vld_0[_blk] = _vptr_0[_blk];
                        __nv_bfloat16* _velems_0 = reinterpret_cast<__nv_bfloat16*>(&_vld_0[_blk]);
                        #pragma unroll
                        for (int _j = 0; _j < 8; _j++)
                            q_vals[0 + _blk * 8 + _j] = __bfloat162float(_velems_0[_j]);
                    }
                }
            }
            #pragma unroll
            for (int _lp = 0; _lp < 4; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(q_vals[_lp*2 + 0], q_vals[_lp*2+1 + 0]));
                q_pack[_lp] = *(uint32_t*)&_bf2;
            }
            int q_store_addr = (smem_query_addr + (d_col / 64 * 8192 + q_row * 128 + d_col % 64 * 2 ^ (d_col / 64 * 8192 + q_row * 128 + d_col % 64 * 2 >> 7 & 7) << 4));
            asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(q_store_addr), "r"(q_pack[0]), "r"(q_pack[1]), "r"(q_pack[2]), "r"(q_pack[3]) : "memory");
        }
        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
        __syncthreads();
        const int row_group = warp % 4;
        const int col_block = warp / 4;
        const int tmem_row_origin = row_group * 32;
        const int logical_row_origin = row_group * 16;
        int row_top = logical_row_origin + lane / 4;
        int row_bot = row_top + 8;
        const int lane_col = lane % 4;
        const int slot = col_block * 4 + lane_col;
        int q_top = off_q + row_top;
        int q_bot = off_q + row_bot;
        int valid_top = ((q_top < Q) ? 1 : 0);
        int valid_bot = ((q_bot < Q) ? 1 : 0);
        float q_sq_top = 0.0f;
        float q_sq_bot = 0.0f;
        if (valid_top != 0) {
            q_sq_top = (float)query_sq[batch_idx * Q + q_top];
        }
        if (valid_bot != 0) {
            q_sq_bot = (float)query_sq[batch_idx * Q + q_bot];
        }
        float best_top_d[10];
        float best_bot_d[10];
        int best_top_i[10];
        int best_bot_i[10];
        #pragma unroll
        for (int kk = 0; kk < 10; kk++) {
            best_top_d[kk] = 3.4e+38f;
            best_bot_d[kk] = 3.4e+38f;
            best_top_i[kk] = -1;
            best_bot_i[kk] = -1;
        }
        int db_tile_start = split_idx * db_tiles_per_split;
        #pragma unroll 1
        for (int local_db_tile = 0; local_db_tile < db_tiles_per_split; local_db_tile++) {
            int db_tile = db_tile_start + local_db_tile;
            int db_start = db_tile * 128;
            #pragma unroll 1
            for (int e_vec = tid; e_vec < 2048; e_vec += 512) {
                int db_elem = e_vec * 8;
                int db_row = db_elem / 128;
                int d_col = db_elem - db_row * 128;
                int db_idx = db_start + db_row;
                float db_vals[8];
                unsigned int db_pack[4];
                #pragma unroll
                for (int vi = 0; vi < 8; vi++) {
                    db_vals[vi] = 0.0f;
                }
                if (db_idx < M) {
                    int db_addr = (batch_idx * M + db_idx) * 128 + d_col;
                    {
                        const uint4* _vptr_1 = reinterpret_cast<const uint4*>(database + (unsigned long long)db_addr);
                        uint4 _vld_1[1];
                        #pragma unroll
                        for (int _blk = 0; _blk < 1; _blk++) {
                            _vld_1[_blk] = _vptr_1[_blk];
                            __nv_bfloat16* _velems_1 = reinterpret_cast<__nv_bfloat16*>(&_vld_1[_blk]);
                            #pragma unroll
                            for (int _j = 0; _j < 8; _j++)
                                db_vals[0 + _blk * 8 + _j] = __bfloat162float(_velems_1[_j]);
                        }
                    }
                }
                #pragma unroll
                for (int _lp = 0; _lp < 4; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(db_vals[_lp*2 + 0], db_vals[_lp*2+1 + 0]));
                    db_pack[_lp] = *(uint32_t*)&_bf2;
                }
                int b_store_addr = (smem_database_addr + (d_col / 64 * 16384 + db_row * 128 + d_col % 64 * 2 ^ (d_col / 64 * 16384 + db_row * 128 + d_col % 64 * 2 >> 7 & 7) << 4));
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(b_store_addr), "r"(db_pack[0]), "r"(db_pack[1]), "r"(db_pack[2]), "r"(db_pack[3]) : "memory");
            }
            asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
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
            if (warp < 8) {
                float dots[32];
                asm volatile(
                    "tcgen05.ld.sync.aligned.16x256b.x8.b32"
                    " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&dots[0])), "=r"(*reinterpret_cast<uint32_t*>(&dots[1])), "=r"(*reinterpret_cast<uint32_t*>(&dots[2])), "=r"(*reinterpret_cast<uint32_t*>(&dots[3])), "=r"(*reinterpret_cast<uint32_t*>(&dots[4])), "=r"(*reinterpret_cast<uint32_t*>(&dots[5])), "=r"(*reinterpret_cast<uint32_t*>(&dots[6])), "=r"(*reinterpret_cast<uint32_t*>(&dots[7])), "=r"(*reinterpret_cast<uint32_t*>(&dots[8])), "=r"(*reinterpret_cast<uint32_t*>(&dots[9])), "=r"(*reinterpret_cast<uint32_t*>(&dots[10])), "=r"(*reinterpret_cast<uint32_t*>(&dots[11])), "=r"(*reinterpret_cast<uint32_t*>(&dots[12])), "=r"(*reinterpret_cast<uint32_t*>(&dots[13])), "=r"(*reinterpret_cast<uint32_t*>(&dots[14])), "=r"(*reinterpret_cast<uint32_t*>(&dots[15])), "=r"(*reinterpret_cast<uint32_t*>(&dots[16])), "=r"(*reinterpret_cast<uint32_t*>(&dots[17])), "=r"(*reinterpret_cast<uint32_t*>(&dots[18])), "=r"(*reinterpret_cast<uint32_t*>(&dots[19])), "=r"(*reinterpret_cast<uint32_t*>(&dots[20])), "=r"(*reinterpret_cast<uint32_t*>(&dots[21])), "=r"(*reinterpret_cast<uint32_t*>(&dots[22])), "=r"(*reinterpret_cast<uint32_t*>(&dots[23])), "=r"(*reinterpret_cast<uint32_t*>(&dots[24])), "=r"(*reinterpret_cast<uint32_t*>(&dots[25])), "=r"(*reinterpret_cast<uint32_t*>(&dots[26])), "=r"(*reinterpret_cast<uint32_t*>(&dots[27])), "=r"(*reinterpret_cast<uint32_t*>(&dots[28])), "=r"(*reinterpret_cast<uint32_t*>(&dots[29])), "=r"(*reinterpret_cast<uint32_t*>(&dots[30])), "=r"(*reinterpret_cast<uint32_t*>(&dots[31]))
                    : "r"(taddr + (tmem_row_origin << 16) + col_block * 64)
                    : "memory");
                asm volatile("tcgen05.wait::ld.sync.aligned;");
                #pragma unroll
                for (int repeat = 0; repeat < 8; repeat++) {
                    const int reg_base = repeat * 4;
                    const int col_base = col_block * 64 + repeat * 8 + lane_col * 2;
                    int db_idx0 = db_start + col_base;
                    int db_idx1 = db_idx0 + 1;
                    float top_d0 = 3.4e+38f;
                    float top_d1 = 3.4e+38f;
                    if (valid_top != 0 & db_idx0 < M) {
                        top_d0 = max_noftz(q_sq_top + (float)database_sq[batch_idx * M + db_idx0] - 2.0f * dots[reg_base], 0.0f);
                    }
                    if (valid_top != 0 & db_idx1 < M) {
                        top_d1 = max_noftz(q_sq_top + (float)database_sq[batch_idx * M + db_idx1] - 2.0f * dots[reg_base + 1], 0.0f);
                    }
                    int top_take1 = ((top_d1 < top_d0) ? 1 : 0);
                    if (((top_take1 != 0) ? top_d1 : top_d0) < best_top_d[9]) {
                        best_top_d[9] = ((top_take1 != 0) ? top_d1 : top_d0);
                        best_top_i[9] = ((top_take1 != 0) ? db_idx1 : db_idx0);
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
                            best_top_i[9] = ((top_take1 != 0) ? db_idx0 : db_idx1);
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
                    float bot_d0 = 3.4e+38f;
                    float bot_d1 = 3.4e+38f;
                    if (valid_bot != 0 & db_idx0 < M) {
                        bot_d0 = max_noftz(q_sq_bot + (float)database_sq[batch_idx * M + db_idx0] - 2.0f * dots[reg_base + 2], 0.0f);
                    }
                    if (valid_bot != 0 & db_idx1 < M) {
                        bot_d1 = max_noftz(q_sq_bot + (float)database_sq[batch_idx * M + db_idx1] - 2.0f * dots[reg_base + 3], 0.0f);
                    }
                    int bot_take1 = ((bot_d1 < bot_d0) ? 1 : 0);
                    if (((bot_take1 != 0) ? bot_d1 : bot_d0) < best_bot_d[9]) {
                        best_bot_d[9] = ((bot_take1 != 0) ? bot_d1 : bot_d0);
                        best_bot_i[9] = ((bot_take1 != 0) ? db_idx1 : db_idx0);
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
                            best_bot_i[9] = ((bot_take1 != 0) ? db_idx0 : db_idx1);
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
            __syncthreads();
        }
        if (warp < 8) {
            int top_slot_base = (row_top * 8 + slot) * 10;
            int bot_slot_base = (row_bot * 8 + slot) * 10;
            #pragma unroll
            for (int kk = 0; kk < 10; kk++) {
                smem_local_d[top_slot_base + kk] = best_top_d[kk];
                smem_local_i[top_slot_base + kk] = best_top_i[kk];
                smem_local_d[bot_slot_base + kk] = best_bot_d[kk];
                smem_local_i[bot_slot_base + kk] = best_bot_i[kk];
            }
        }
        __syncthreads();
        if (tid < 64) {
            int row = tid;
            int q_idx = off_q + row;
            if (q_idx < Q) {
                float head_d[8];
                int head_i[8];
                int head_k[8];
                #pragma unroll
                for (int slot_idx = 0; slot_idx < 8; slot_idx++) {
                    int local_base = (row * 8 + slot_idx) * 10;
                    head_k[slot_idx] = 0;
                    head_d[slot_idx] = smem_local_d[local_base];
                    head_i[slot_idx] = smem_local_i[local_base];
                }
                int out_base = ((split_idx * B + batch_idx) * Q + q_idx) * K;
                #pragma unroll
                for (int out_k = 0; out_k < 10; out_k++) {
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
                    if (out_k < K) {
                        *((float*)(partial_dists + out_base + out_k)) = winner_d;
                        *((int*)(partial_indices + out_base + out_k)) = winner_i;
                    }
                    #pragma unroll
                    for (int slot_idx = 0; slot_idx < 8; slot_idx++) {
                        if (winner_slot == slot_idx) {
                            int next_head = head_k[slot_idx] + 1;
                            head_k[slot_idx] = next_head;
                            head_d[slot_idx] = 3.4e+38f;
                            head_i[slot_idx] = -1;
                            if (next_head < 10) {
                                int local_base = (row * 8 + slot_idx) * 10;
                                head_d[slot_idx] = smem_local_d[local_base + next_head];
                                head_i[slot_idx] = smem_local_i[local_base + next_head];
                            }
                        }
                    }
                }
            }
        }
        __syncthreads();
    }

    // Cleanup
    __syncthreads();

    if (warp == 0) {
        asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;" :: "r"(tmem_addr_storage[0]), "r"(128));
        asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;");
    }
}

} // extern "C"

#undef NUM_MAIN_STAGES
#undef SMEM_SMEM_DATABASE_OFF
#undef SMEM_SMEM_DATABASE_STAGE_BYTES
#undef SMEM_SMEM_DATABASE_STRIDE
#undef SMEM_SMEM_LOCAL_D_OFF
#undef SMEM_SMEM_LOCAL_D_STAGE_BYTES
#undef SMEM_SMEM_LOCAL_D_STRIDE
#undef SMEM_SMEM_LOCAL_I_OFF
#undef SMEM_SMEM_LOCAL_I_STAGE_BYTES
#undef SMEM_SMEM_LOCAL_I_STRIDE
#undef SMEM_SMEM_QUERY_OFF
#undef SMEM_SMEM_QUERY_STAGE_BYTES
#undef SMEM_SMEM_QUERY_STRIDE
#undef SMEM_TOTAL
#undef THREADS
#undef TMEM_ACC_OFFSET
#undef TMEM_NCOLS
#undef mma_done_addr
#undef smem_database_addr
#undef smem_local_d_addr
#undef smem_local_i_addr
#undef smem_query_addr

#define NUM_MAIN_STAGES 1
#define SMEM_GROUP_D_OFF 0
#define SMEM_GROUP_D_STAGE_BYTES 160
#define SMEM_GROUP_D_STRIDE 160
#define SMEM_GROUP_I_OFF 160
#define SMEM_GROUP_I_STAGE_BYTES 160
#define SMEM_GROUP_I_STRIDE 160
#define SMEM_TOTAL 512
#define THREADS 128
#define TOP_K_MAX 10
#define SPLIT_COUNT 72
#define MERGE_GROUPS 4
#define SPLITS_PER_GROUP 18

extern "C" {

__global__ __launch_bounds__(128, 1) void
kernel_knn_build_ragonline_mbucket_aa88_q1m_s72_k10_coop_merge(float* __restrict__ partial_dists, int32_t* __restrict__ partial_indices, float* __restrict__ out_dists, int32_t* __restrict__ out_indices, int total_queries)
{
    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;

    extern __shared__ __align__(1024) char smem_raw[];
    int smem;
    smem = (int)(unsigned long long)__cvta_generic_to_shared(smem_raw);
    const int smem_group_d = smem + 0;
    const int smem_group_i = smem + 160;

    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    const int warp_id = warp;
    const int lane_id = lane;
    float* group_d = (float*)(smem_raw + 0);
    #define group_d_addr (smem + 0)
    int* group_i = (int*)(smem_raw + 160);
    #define group_i_addr (smem + 160)

    // === Task calls (dependency order) ===
    int row = bid;
    int base_row = row * TOP_K_MAX;
    int split_stride = total_queries * TOP_K_MAX;
    int group = warp;
    int split_idx = group * SPLITS_PER_GROUP + lane;
    int split_pos = 0;
    float cand_d = 3.4e+38f;
    int cand_i = -1;
    if (row < total_queries) {
        if (lane < SPLITS_PER_GROUP) {
            if (split_idx < SPLIT_COUNT) {
                int split_base = base_row + split_idx * split_stride;
                cand_d = (float)partial_dists[split_base];
                cand_i = partial_indices[split_base];
            }
        }
        #pragma unroll
        for (int out_k = 0; out_k < TOP_K_MAX; out_k++) {
            float warp_min = cand_d;
            #pragma unroll
            for (int offset = 16; offset > 0; offset >>= 1)
                warp_min = fminf(warp_min, __shfl_xor_sync(0xFFFFFFFF, warp_min, offset));
            int _vote_0 = __ballot_sync(0xFFFFFFFF, cand_d == warp_min);
            int owner_ballot = _vote_0;
            int _ffs_0 = __ffs(owner_ballot);
            int winner_lane = _ffs_0 - 1;
            int _shfl_0 = __shfl_sync(0xFFFFFFFF, cand_i, winner_lane);
            int winner_i = _shfl_0;
            if (lane == 0) {
                int group_slot = group * TOP_K_MAX + out_k;
                group_d[group_slot] = warp_min;
                group_i[group_slot] = winner_i;
            }
            if (lane == winner_lane) {
                split_pos = split_pos + 1;
                if (split_pos < TOP_K_MAX) {
                    int next_addr = base_row + split_idx * split_stride + split_pos;
                    cand_d = (float)partial_dists[next_addr];
                    cand_i = partial_indices[next_addr];
                } else {
                    cand_d = 3.4e+38f;
                    cand_i = -1;
                }
            }
        }
    }
    asm volatile("barrier.sync 15, 128;");
    if (row < total_queries) {
        if (warp == 0) {
            int group_pos = 0;
            float final_d = 3.4e+38f;
            int final_i = -1;
            if (lane < MERGE_GROUPS) {
                final_d = group_d[lane * TOP_K_MAX];
                final_i = group_i[lane * TOP_K_MAX];
            }
            #pragma unroll
            for (int out_k = 0; out_k < TOP_K_MAX; out_k++) {
                float warp_min = final_d;
                #pragma unroll
                for (int offset = 16; offset > 0; offset >>= 1)
                    warp_min = fminf(warp_min, __shfl_xor_sync(0xFFFFFFFF, warp_min, offset));
                int _vote_1 = __ballot_sync(0xFFFFFFFF, final_d == warp_min);
                int owner_ballot = _vote_1;
                int _ffs_1 = __ffs(owner_ballot);
                int winner_lane = _ffs_1 - 1;
                int _shfl_1 = __shfl_sync(0xFFFFFFFF, final_i, winner_lane);
                int winner_i = _shfl_1;
                if (lane == 0) {
                    *((float*)(out_dists + base_row + out_k)) = warp_min;
                    *((int*)(out_indices + base_row + out_k)) = winner_i;
                }
                if (lane == winner_lane) {
                    group_pos = group_pos + 1;
                    if (group_pos < TOP_K_MAX) {
                        int next_slot = lane * TOP_K_MAX + group_pos;
                        final_d = group_d[next_slot];
                        final_i = group_i[next_slot];
                    } else {
                        final_d = 3.4e+38f;
                        final_i = -1;
                    }
                }
            }
        }
    }
}

} // extern "C"

#undef MERGE_GROUPS
#undef NUM_MAIN_STAGES
#undef SMEM_GROUP_D_OFF
#undef SMEM_GROUP_D_STAGE_BYTES
#undef SMEM_GROUP_D_STRIDE
#undef SMEM_GROUP_I_OFF
#undef SMEM_GROUP_I_STAGE_BYTES
#undef SMEM_GROUP_I_STRIDE
#undef SMEM_TOTAL
#undef SPLITS_PER_GROUP
#undef SPLIT_COUNT
#undef THREADS
#undef TOP_K_MAX
#undef group_d_addr
#undef group_i_addr

#define NUM_MAIN_STAGES 1
#define SMEM_GROUP_D_OFF 0
#define SMEM_GROUP_D_STAGE_BYTES 160
#define SMEM_GROUP_D_STRIDE 160
#define SMEM_GROUP_I_OFF 160
#define SMEM_GROUP_I_STAGE_BYTES 160
#define SMEM_GROUP_I_STRIDE 160
#define SMEM_TOTAL 512
#define THREADS 128
#define TOP_K_MAX 10
#define SPLIT_COUNT 74
#define MERGE_GROUPS 4
#define SPLITS_PER_GROUP 19

extern "C" {

__global__ __launch_bounds__(128, 1) void
kernel_knn_build_ragonline_mbucket_aa88_q1m_s72_k10_coop_merge_s74_m250(float* __restrict__ partial_dists, int32_t* __restrict__ partial_indices, float* __restrict__ out_dists, int32_t* __restrict__ out_indices, int total_queries)
{
    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;

    extern __shared__ __align__(1024) char smem_raw[];
    int smem;
    smem = (int)(unsigned long long)__cvta_generic_to_shared(smem_raw);
    const int smem_group_d = smem + 0;
    const int smem_group_i = smem + 160;

    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    const int warp_id = warp;
    const int lane_id = lane;
    float* group_d = (float*)(smem_raw + 0);
    #define group_d_addr (smem + 0)
    int* group_i = (int*)(smem_raw + 160);
    #define group_i_addr (smem + 160)

    // === Task calls (dependency order) ===
    int row = bid;
    int base_row = row * TOP_K_MAX;
    int split_stride = total_queries * TOP_K_MAX;
    int group = warp;
    int split_idx = group * SPLITS_PER_GROUP + lane;
    int split_pos = 0;
    float cand_d = 3.4e+38f;
    int cand_i = -1;
    if (row < total_queries) {
        if (lane < SPLITS_PER_GROUP) {
            if (split_idx < SPLIT_COUNT) {
                int split_base = base_row + split_idx * split_stride;
                cand_d = (float)partial_dists[split_base];
                cand_i = partial_indices[split_base];
            }
        }
        #pragma unroll
        for (int out_k = 0; out_k < TOP_K_MAX; out_k++) {
            float warp_min = cand_d;
            #pragma unroll
            for (int offset = 16; offset > 0; offset >>= 1)
                warp_min = fminf(warp_min, __shfl_xor_sync(0xFFFFFFFF, warp_min, offset));
            int _vote_0 = __ballot_sync(0xFFFFFFFF, cand_d == warp_min);
            int owner_ballot = _vote_0;
            int _ffs_0 = __ffs(owner_ballot);
            int winner_lane = _ffs_0 - 1;
            int _shfl_0 = __shfl_sync(0xFFFFFFFF, cand_i, winner_lane);
            int winner_i = _shfl_0;
            if (lane == 0) {
                int group_slot = group * TOP_K_MAX + out_k;
                group_d[group_slot] = warp_min;
                group_i[group_slot] = winner_i;
            }
            if (lane == winner_lane) {
                split_pos = split_pos + 1;
                if (split_pos < TOP_K_MAX) {
                    int next_addr = base_row + split_idx * split_stride + split_pos;
                    cand_d = (float)partial_dists[next_addr];
                    cand_i = partial_indices[next_addr];
                } else {
                    cand_d = 3.4e+38f;
                    cand_i = -1;
                }
            }
        }
    }
    asm volatile("barrier.sync 15, 128;");
    if (row < total_queries) {
        if (warp == 0) {
            int group_pos = 0;
            float final_d = 3.4e+38f;
            int final_i = -1;
            if (lane < MERGE_GROUPS) {
                final_d = group_d[lane * TOP_K_MAX];
                final_i = group_i[lane * TOP_K_MAX];
            }
            #pragma unroll
            for (int out_k = 0; out_k < TOP_K_MAX; out_k++) {
                float warp_min = final_d;
                #pragma unroll
                for (int offset = 16; offset > 0; offset >>= 1)
                    warp_min = fminf(warp_min, __shfl_xor_sync(0xFFFFFFFF, warp_min, offset));
                int _vote_1 = __ballot_sync(0xFFFFFFFF, final_d == warp_min);
                int owner_ballot = _vote_1;
                int _ffs_1 = __ffs(owner_ballot);
                int winner_lane = _ffs_1 - 1;
                int _shfl_1 = __shfl_sync(0xFFFFFFFF, final_i, winner_lane);
                int winner_i = _shfl_1;
                if (lane == 0) {
                    *((float*)(out_dists + base_row + out_k)) = warp_min;
                    *((int*)(out_indices + base_row + out_k)) = winner_i;
                }
                if (lane == winner_lane) {
                    group_pos = group_pos + 1;
                    if (group_pos < TOP_K_MAX) {
                        int next_slot = lane * TOP_K_MAX + group_pos;
                        final_d = group_d[next_slot];
                        final_i = group_i[next_slot];
                    } else {
                        final_d = 3.4e+38f;
                        final_i = -1;
                    }
                }
            }
        }
    }
}

} // extern "C"

#undef MERGE_GROUPS
#undef NUM_MAIN_STAGES
#undef SMEM_GROUP_D_OFF
#undef SMEM_GROUP_D_STAGE_BYTES
#undef SMEM_GROUP_D_STRIDE
#undef SMEM_GROUP_I_OFF
#undef SMEM_GROUP_I_STAGE_BYTES
#undef SMEM_GROUP_I_STRIDE
#undef SMEM_TOTAL
#undef SPLITS_PER_GROUP
#undef SPLIT_COUNT
#undef THREADS
#undef TOP_K_MAX
#undef group_d_addr
#undef group_i_addr

#define NUM_MAIN_STAGES 1
#define THREADS 8
#define TOP_K_MAX 10
#define SPLIT_COUNT 16

extern "C" {

__global__ __launch_bounds__(8, 1) void
kernel_knn_build_rect_d64_cf49_s16_cached_merge(float* __restrict__ partial_dists, int32_t* __restrict__ partial_indices, float* __restrict__ out_dists, int32_t* __restrict__ out_indices, int total_queries)
{
    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;


    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    const int warp_id = warp;
    const int lane_id = lane;

    // === Task calls (dependency order) ===
    int start_row = bid * 8 + tid;
    int stride = num_bids * 8;
    #pragma unroll 1
    for (int row = start_row; row < total_queries; row += stride) {
        int base_row = row * TOP_K_MAX;
        int split_stride = total_queries * TOP_K_MAX;
        int out_base = base_row;
        int split_pos[SPLIT_COUNT];
        int split_base[SPLIT_COUNT];
        float cand_d[SPLIT_COUNT];
        int cand_i[SPLIT_COUNT];
        #pragma unroll
        for (int split_idx = 0; split_idx < SPLIT_COUNT; split_idx++) {
            split_pos[split_idx] = 0;
            split_base[split_idx] = base_row + split_idx * split_stride;
            cand_d[split_idx] = (float)partial_dists[split_base[split_idx]];
            cand_i[split_idx] = partial_indices[split_base[split_idx]];
        }
        #pragma unroll
        for (int out_k = 0; out_k < TOP_K_MAX; out_k++) {
            float best_d = cand_d[0];
            int best_i = cand_i[0];
            int best_split = 0;
            #pragma unroll
            for (int split_idx = 1; split_idx < SPLIT_COUNT; split_idx++) {
                if (cand_d[split_idx] < best_d) {
                    best_d = cand_d[split_idx];
                    best_i = cand_i[split_idx];
                    best_split = split_idx;
                }
            }
            *((float*)(out_dists + out_base + out_k)) = best_d;
            *((int*)(out_indices + out_base + out_k)) = best_i;
            split_pos[best_split] = split_pos[best_split] + 1;
            if (out_k + 1 < TOP_K_MAX) {
                int next_pos = split_pos[best_split];
                int next_addr = split_base[best_split] + next_pos;
                cand_d[best_split] = (float)partial_dists[next_addr];
                cand_i[best_split] = partial_indices[next_addr];
            }
        }
    }
}

} // extern "C"

#undef NUM_MAIN_STAGES
#undef SPLIT_COUNT
#undef THREADS
#undef TOP_K_MAX

#define NUM_MAIN_STAGES 1
#define THREADS 32
#define TOP_K_MAX 10
#define SPLIT_COUNT 8

extern "C" {

__global__ __launch_bounds__(32, 1) void
kernel_knn_build_evolve_7bfc_k10_merge_s7_rowbase_cache_rect4452_s8(float* __restrict__ partial_dists, int32_t* __restrict__ partial_indices, float* __restrict__ out_dists, int32_t* __restrict__ out_indices, int total_queries)
{
    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;


    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    const int warp_id = warp;
    const int lane_id = lane;

    // === Task calls (dependency order) ===
    int start_row = bid * 32 + tid;
    int stride = num_bids * 32;
    #pragma unroll 1
    for (int row = start_row; row < total_queries; row += stride) {
        int base_row = row * TOP_K_MAX;
        int split_stride = total_queries * TOP_K_MAX;
        int out_base = base_row;
        int split_pos[SPLIT_COUNT];
        int split_base[SPLIT_COUNT];
        float cand_d[SPLIT_COUNT];
        int cand_i[SPLIT_COUNT];
        #pragma unroll
        for (int split_idx = 0; split_idx < SPLIT_COUNT; split_idx++) {
            split_pos[split_idx] = 0;
            split_base[split_idx] = base_row + split_idx * split_stride;
            cand_d[split_idx] = (float)partial_dists[split_base[split_idx]];
            cand_i[split_idx] = partial_indices[split_base[split_idx]];
        }
        #pragma unroll
        for (int out_k = 0; out_k < TOP_K_MAX; out_k++) {
            float best_d = cand_d[0];
            int best_i = cand_i[0];
            int best_split = 0;
            #pragma unroll
            for (int split_idx = 1; split_idx < SPLIT_COUNT; split_idx++) {
                if (cand_d[split_idx] < best_d) {
                    best_d = cand_d[split_idx];
                    best_i = cand_i[split_idx];
                    best_split = split_idx;
                }
            }
            *((float*)(out_dists + out_base + out_k)) = best_d;
            *((int*)(out_indices + out_base + out_k)) = best_i;
            split_pos[best_split] = split_pos[best_split] + 1;
            if (out_k + 1 < TOP_K_MAX) {
                int next_pos = split_pos[best_split];
                int next_addr = split_base[best_split] + next_pos;
                cand_d[best_split] = (float)partial_dists[next_addr];
                cand_i[best_split] = partial_indices[next_addr];
            }
        }
    }
}

} // extern "C"

#undef NUM_MAIN_STAGES
#undef SPLIT_COUNT
#undef THREADS
#undef TOP_K_MAX

#define NUM_MAIN_STAGES 1
#define THREADS 32
#define TOP_K_MAX 10
#define SPLIT_COUNT 12

extern "C" {

__global__ __launch_bounds__(32, 1) void
kernel_knn_build_evolve_7bfc_k10_merge_s7_rowbase_cache_rect4452_s12(float* __restrict__ partial_dists, int32_t* __restrict__ partial_indices, float* __restrict__ out_dists, int32_t* __restrict__ out_indices, int total_queries)
{
    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;


    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    const int warp_id = warp;
    const int lane_id = lane;

    // === Task calls (dependency order) ===
    int start_row = bid * 32 + tid;
    int stride = num_bids * 32;
    #pragma unroll 1
    for (int row = start_row; row < total_queries; row += stride) {
        int base_row = row * TOP_K_MAX;
        int split_stride = total_queries * TOP_K_MAX;
        int out_base = base_row;
        int split_pos[SPLIT_COUNT];
        int split_base[SPLIT_COUNT];
        float cand_d[SPLIT_COUNT];
        int cand_i[SPLIT_COUNT];
        #pragma unroll
        for (int split_idx = 0; split_idx < SPLIT_COUNT; split_idx++) {
            split_pos[split_idx] = 0;
            split_base[split_idx] = base_row + split_idx * split_stride;
            cand_d[split_idx] = (float)partial_dists[split_base[split_idx]];
            cand_i[split_idx] = partial_indices[split_base[split_idx]];
        }
        #pragma unroll
        for (int out_k = 0; out_k < TOP_K_MAX; out_k++) {
            float best_d = cand_d[0];
            int best_i = cand_i[0];
            int best_split = 0;
            #pragma unroll
            for (int split_idx = 1; split_idx < SPLIT_COUNT; split_idx++) {
                if (cand_d[split_idx] < best_d) {
                    best_d = cand_d[split_idx];
                    best_i = cand_i[split_idx];
                    best_split = split_idx;
                }
            }
            *((float*)(out_dists + out_base + out_k)) = best_d;
            *((int*)(out_indices + out_base + out_k)) = best_i;
            split_pos[best_split] = split_pos[best_split] + 1;
            if (out_k + 1 < TOP_K_MAX) {
                int next_pos = split_pos[best_split];
                int next_addr = split_base[best_split] + next_pos;
                cand_d[best_split] = (float)partial_dists[next_addr];
                cand_i[best_split] = partial_indices[next_addr];
            }
        }
    }
}

} // extern "C"

#undef NUM_MAIN_STAGES
#undef SPLIT_COUNT
#undef THREADS
#undef TOP_K_MAX

#define NUM_MAIN_STAGES 1
#define THREADS 32
#define TOP_K_MAX 10
#define SPLIT_COUNT 16

extern "C" {

__global__ __launch_bounds__(32, 1) void
kernel_knn_build_evolve_7bfc_k10_merge_s7_rowbase_cache_rect4452_s16(float* __restrict__ partial_dists, int32_t* __restrict__ partial_indices, float* __restrict__ out_dists, int32_t* __restrict__ out_indices, int total_queries)
{
    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;


    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    const int warp_id = warp;
    const int lane_id = lane;

    // === Task calls (dependency order) ===
    int start_row = bid * 32 + tid;
    int stride = num_bids * 32;
    #pragma unroll 1
    for (int row = start_row; row < total_queries; row += stride) {
        int base_row = row * TOP_K_MAX;
        int split_stride = total_queries * TOP_K_MAX;
        int out_base = base_row;
        int split_pos[SPLIT_COUNT];
        int split_base[SPLIT_COUNT];
        float cand_d[SPLIT_COUNT];
        int cand_i[SPLIT_COUNT];
        #pragma unroll
        for (int split_idx = 0; split_idx < SPLIT_COUNT; split_idx++) {
            split_pos[split_idx] = 0;
            split_base[split_idx] = base_row + split_idx * split_stride;
            cand_d[split_idx] = (float)partial_dists[split_base[split_idx]];
            cand_i[split_idx] = partial_indices[split_base[split_idx]];
        }
        #pragma unroll
        for (int out_k = 0; out_k < TOP_K_MAX; out_k++) {
            float best_d = cand_d[0];
            int best_i = cand_i[0];
            int best_split = 0;
            #pragma unroll
            for (int split_idx = 1; split_idx < SPLIT_COUNT; split_idx++) {
                if (cand_d[split_idx] < best_d) {
                    best_d = cand_d[split_idx];
                    best_i = cand_i[split_idx];
                    best_split = split_idx;
                }
            }
            *((float*)(out_dists + out_base + out_k)) = best_d;
            *((int*)(out_indices + out_base + out_k)) = best_i;
            split_pos[best_split] = split_pos[best_split] + 1;
            if (out_k + 1 < TOP_K_MAX) {
                int next_pos = split_pos[best_split];
                int next_addr = split_base[best_split] + next_pos;
                cand_d[best_split] = (float)partial_dists[next_addr];
                cand_i[best_split] = partial_indices[next_addr];
            }
        }
    }
}

} // extern "C"

#undef NUM_MAIN_STAGES
#undef SPLIT_COUNT
#undef THREADS
#undef TOP_K_MAX

#define NUM_MAIN_STAGES 1
#define THREADS 32
#define TOP_K_MAX 10
#define SPLIT_COUNT 24

extern "C" {

__global__ __launch_bounds__(32, 1) void
kernel_knn_build_evolve_7bfc_k10_merge_s7_rowbase_cache_rect4452_s24(float* __restrict__ partial_dists, int32_t* __restrict__ partial_indices, float* __restrict__ out_dists, int32_t* __restrict__ out_indices, int total_queries)
{
    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;


    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    const int warp_id = warp;
    const int lane_id = lane;

    // === Task calls (dependency order) ===
    int start_row = bid * 32 + tid;
    int stride = num_bids * 32;
    #pragma unroll 1
    for (int row = start_row; row < total_queries; row += stride) {
        int base_row = row * TOP_K_MAX;
        int split_stride = total_queries * TOP_K_MAX;
        int out_base = base_row;
        int split_pos[SPLIT_COUNT];
        int split_base[SPLIT_COUNT];
        float cand_d[SPLIT_COUNT];
        int cand_i[SPLIT_COUNT];
        #pragma unroll
        for (int split_idx = 0; split_idx < SPLIT_COUNT; split_idx++) {
            split_pos[split_idx] = 0;
            split_base[split_idx] = base_row + split_idx * split_stride;
            cand_d[split_idx] = (float)partial_dists[split_base[split_idx]];
            cand_i[split_idx] = partial_indices[split_base[split_idx]];
        }
        #pragma unroll
        for (int out_k = 0; out_k < TOP_K_MAX; out_k++) {
            float best_d = cand_d[0];
            int best_i = cand_i[0];
            int best_split = 0;
            #pragma unroll
            for (int split_idx = 1; split_idx < SPLIT_COUNT; split_idx++) {
                if (cand_d[split_idx] < best_d) {
                    best_d = cand_d[split_idx];
                    best_i = cand_i[split_idx];
                    best_split = split_idx;
                }
            }
            *((float*)(out_dists + out_base + out_k)) = best_d;
            *((int*)(out_indices + out_base + out_k)) = best_i;
            split_pos[best_split] = split_pos[best_split] + 1;
            if (out_k + 1 < TOP_K_MAX) {
                int next_pos = split_pos[best_split];
                int next_addr = split_base[best_split] + next_pos;
                cand_d[best_split] = (float)partial_dists[next_addr];
                cand_i[best_split] = partial_indices[next_addr];
            }
        }
    }
}

} // extern "C"

#undef NUM_MAIN_STAGES
#undef SPLIT_COUNT
#undef THREADS
#undef TOP_K_MAX

#define NUM_MAIN_STAGES 1
#define THREADS 32
#define TOP_K_MAX 10
#define SPLIT_COUNT 32

extern "C" {

__global__ __launch_bounds__(32, 1) void
kernel_knn_build_evolve_7bfc_k10_merge_s7_rowbase_cache_rect4452_s32(float* __restrict__ partial_dists, int32_t* __restrict__ partial_indices, float* __restrict__ out_dists, int32_t* __restrict__ out_indices, int total_queries)
{
    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;


    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    const int warp_id = warp;
    const int lane_id = lane;

    // === Task calls (dependency order) ===
    int start_row = bid * 32 + tid;
    int stride = num_bids * 32;
    #pragma unroll 1
    for (int row = start_row; row < total_queries; row += stride) {
        int base_row = row * TOP_K_MAX;
        int split_stride = total_queries * TOP_K_MAX;
        int out_base = base_row;
        int split_pos[SPLIT_COUNT];
        int split_base[SPLIT_COUNT];
        float cand_d[SPLIT_COUNT];
        int cand_i[SPLIT_COUNT];
        #pragma unroll
        for (int split_idx = 0; split_idx < SPLIT_COUNT; split_idx++) {
            split_pos[split_idx] = 0;
            split_base[split_idx] = base_row + split_idx * split_stride;
            cand_d[split_idx] = (float)partial_dists[split_base[split_idx]];
            cand_i[split_idx] = partial_indices[split_base[split_idx]];
        }
        #pragma unroll
        for (int out_k = 0; out_k < TOP_K_MAX; out_k++) {
            float best_d = cand_d[0];
            int best_i = cand_i[0];
            int best_split = 0;
            #pragma unroll
            for (int split_idx = 1; split_idx < SPLIT_COUNT; split_idx++) {
                if (cand_d[split_idx] < best_d) {
                    best_d = cand_d[split_idx];
                    best_i = cand_i[split_idx];
                    best_split = split_idx;
                }
            }
            *((float*)(out_dists + out_base + out_k)) = best_d;
            *((int*)(out_indices + out_base + out_k)) = best_i;
            split_pos[best_split] = split_pos[best_split] + 1;
            if (out_k + 1 < TOP_K_MAX) {
                int next_pos = split_pos[best_split];
                int next_addr = split_base[best_split] + next_pos;
                cand_d[best_split] = (float)partial_dists[next_addr];
                cand_i[best_split] = partial_indices[next_addr];
            }
        }
    }
}

} // extern "C"

#undef NUM_MAIN_STAGES
#undef SPLIT_COUNT
#undef THREADS
#undef TOP_K_MAX

#define NUM_MAIN_STAGES 1
#define THREADS 32
#define TOP_K_MAX 10
#define SPLIT_COUNT 8

extern "C" {

__global__ __launch_bounds__(32, 1) void
kernel_knn_build_evolve_7bfc_k10_merge_s7_rowbase_cache_rectd15e_s8(float* __restrict__ partial_dists, int32_t* __restrict__ partial_indices, float* __restrict__ out_dists, int32_t* __restrict__ out_indices, int total_queries)
{
    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;


    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    const int warp_id = warp;
    const int lane_id = lane;

    // === Task calls (dependency order) ===
    int start_row = bid * 32 + tid;
    int stride = num_bids * 32;
    #pragma unroll 1
    for (int row = start_row; row < total_queries; row += stride) {
        int base_row = row * TOP_K_MAX;
        int split_stride = total_queries * TOP_K_MAX;
        int out_base = base_row;
        int split_pos[SPLIT_COUNT];
        int split_base[SPLIT_COUNT];
        float cand_d[SPLIT_COUNT];
        int cand_i[SPLIT_COUNT];
        #pragma unroll
        for (int split_idx = 0; split_idx < SPLIT_COUNT; split_idx++) {
            split_pos[split_idx] = 0;
            split_base[split_idx] = base_row + split_idx * split_stride;
            cand_d[split_idx] = (float)partial_dists[split_base[split_idx]];
            cand_i[split_idx] = partial_indices[split_base[split_idx]];
        }
        #pragma unroll
        for (int out_k = 0; out_k < TOP_K_MAX; out_k++) {
            float best_d = cand_d[0];
            int best_i = cand_i[0];
            int best_split = 0;
            #pragma unroll
            for (int split_idx = 1; split_idx < SPLIT_COUNT; split_idx++) {
                if (cand_d[split_idx] < best_d) {
                    best_d = cand_d[split_idx];
                    best_i = cand_i[split_idx];
                    best_split = split_idx;
                }
            }
            *((float*)(out_dists + out_base + out_k)) = best_d;
            *((int*)(out_indices + out_base + out_k)) = best_i;
            split_pos[best_split] = split_pos[best_split] + 1;
            if (out_k + 1 < TOP_K_MAX) {
                int next_pos = split_pos[best_split];
                int next_addr = split_base[best_split] + next_pos;
                cand_d[best_split] = (float)partial_dists[next_addr];
                cand_i[best_split] = partial_indices[next_addr];
            }
        }
    }
}

} // extern "C"

#undef NUM_MAIN_STAGES
#undef SPLIT_COUNT
#undef THREADS
#undef TOP_K_MAX

#define NUM_MAIN_STAGES 1
#define THREADS 32
#define TOP_K_MAX 10
#define SPLIT_COUNT 16

extern "C" {

__global__ __launch_bounds__(32, 1) void
kernel_knn_build_evolve_7bfc_k10_merge_s7_rowbase_cache_rectd15e_s16(float* __restrict__ partial_dists, int32_t* __restrict__ partial_indices, float* __restrict__ out_dists, int32_t* __restrict__ out_indices, int total_queries)
{
    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;


    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    const int warp_id = warp;
    const int lane_id = lane;

    // === Task calls (dependency order) ===
    int start_row = bid * 32 + tid;
    int stride = num_bids * 32;
    #pragma unroll 1
    for (int row = start_row; row < total_queries; row += stride) {
        int base_row = row * TOP_K_MAX;
        int split_stride = total_queries * TOP_K_MAX;
        int out_base = base_row;
        int split_pos[SPLIT_COUNT];
        int split_base[SPLIT_COUNT];
        float cand_d[SPLIT_COUNT];
        int cand_i[SPLIT_COUNT];
        #pragma unroll
        for (int split_idx = 0; split_idx < SPLIT_COUNT; split_idx++) {
            split_pos[split_idx] = 0;
            split_base[split_idx] = base_row + split_idx * split_stride;
            cand_d[split_idx] = (float)partial_dists[split_base[split_idx]];
            cand_i[split_idx] = partial_indices[split_base[split_idx]];
        }
        #pragma unroll
        for (int out_k = 0; out_k < TOP_K_MAX; out_k++) {
            float best_d = cand_d[0];
            int best_i = cand_i[0];
            int best_split = 0;
            #pragma unroll
            for (int split_idx = 1; split_idx < SPLIT_COUNT; split_idx++) {
                if (cand_d[split_idx] < best_d) {
                    best_d = cand_d[split_idx];
                    best_i = cand_i[split_idx];
                    best_split = split_idx;
                }
            }
            *((float*)(out_dists + out_base + out_k)) = best_d;
            *((int*)(out_indices + out_base + out_k)) = best_i;
            split_pos[best_split] = split_pos[best_split] + 1;
            if (out_k + 1 < TOP_K_MAX) {
                int next_pos = split_pos[best_split];
                int next_addr = split_base[best_split] + next_pos;
                cand_d[best_split] = (float)partial_dists[next_addr];
                cand_i[best_split] = partial_indices[next_addr];
            }
        }
    }
}

} // extern "C"

#undef NUM_MAIN_STAGES
#undef SPLIT_COUNT
#undef THREADS
#undef TOP_K_MAX

#define NUM_MAIN_STAGES 1
#define THREADS 32
#define TOP_K_MAX 10
#define SPLIT_COUNT 32

extern "C" {

__global__ __launch_bounds__(32, 1) void
kernel_knn_build_evolve_7bfc_k10_merge_s7_rowbase_cache_rectd15e_s32(float* __restrict__ partial_dists, int32_t* __restrict__ partial_indices, float* __restrict__ out_dists, int32_t* __restrict__ out_indices, int total_queries)
{
    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;


    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    const int warp_id = warp;
    const int lane_id = lane;

    // === Task calls (dependency order) ===
    int start_row = bid * 32 + tid;
    int stride = num_bids * 32;
    #pragma unroll 1
    for (int row = start_row; row < total_queries; row += stride) {
        int base_row = row * TOP_K_MAX;
        int split_stride = total_queries * TOP_K_MAX;
        int out_base = base_row;
        int split_pos[SPLIT_COUNT];
        int split_base[SPLIT_COUNT];
        float cand_d[SPLIT_COUNT];
        int cand_i[SPLIT_COUNT];
        #pragma unroll
        for (int split_idx = 0; split_idx < SPLIT_COUNT; split_idx++) {
            split_pos[split_idx] = 0;
            split_base[split_idx] = base_row + split_idx * split_stride;
            cand_d[split_idx] = (float)partial_dists[split_base[split_idx]];
            cand_i[split_idx] = partial_indices[split_base[split_idx]];
        }
        #pragma unroll
        for (int out_k = 0; out_k < TOP_K_MAX; out_k++) {
            float best_d = cand_d[0];
            int best_i = cand_i[0];
            int best_split = 0;
            #pragma unroll
            for (int split_idx = 1; split_idx < SPLIT_COUNT; split_idx++) {
                if (cand_d[split_idx] < best_d) {
                    best_d = cand_d[split_idx];
                    best_i = cand_i[split_idx];
                    best_split = split_idx;
                }
            }
            *((float*)(out_dists + out_base + out_k)) = best_d;
            *((int*)(out_indices + out_base + out_k)) = best_i;
            split_pos[best_split] = split_pos[best_split] + 1;
            if (out_k + 1 < TOP_K_MAX) {
                int next_pos = split_pos[best_split];
                int next_addr = split_base[best_split] + next_pos;
                cand_d[best_split] = (float)partial_dists[next_addr];
                cand_i[best_split] = partial_indices[next_addr];
            }
        }
    }
}

} // extern "C"

#undef NUM_MAIN_STAGES
#undef SPLIT_COUNT
#undef THREADS
#undef TOP_K_MAX

