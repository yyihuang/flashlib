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

__device__ __forceinline__ void mbarrier_init_pred(int mbar_addr, uint32_t count, uint32_t pred) {
    asm volatile(
        "{\n\t"
        ".reg .pred p;\n\t"
        "setp.ne.b32 p, %2, 0;\n\t"
        "@p mbarrier.init.shared::cta.b64 [%0], %1;\n\t"
        "}\n" :: "r"(mbar_addr), "r"(count), "r"(pred));
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

__device__ __forceinline__ void tmem_ld_x16_wait(float* dst, int addr) {
    tmem_ld_x16(dst, addr);
    asm volatile("tcgen05.wait::ld.sync.aligned;");
}

__device__ __forceinline__ uint32_t make_warp_uniform(uint32_t val) {
    uint32_t result;
    asm volatile("shfl.sync.idx.b32 %0, %1, 0, 0x1f, 0xffffffff;"
        : "=r"(result) : "r"(val));
    return result;
}

#define TMEM_NCOLS 256
#define TMEM_SCORE_TMEM_OFFSET 0
#define NUM_MAIN_STAGES 1
#define SMEM_SMEM_X_OFF 1024
#define SMEM_SMEM_X_STAGE_BYTES 32768
#define SMEM_SMEM_X_STRIDE 32768
#define SMEM_SMEM_C_OFF 33792
#define SMEM_SMEM_C_STAGE_BYTES 65536
#define SMEM_SMEM_C_STRIDE 65536
#define SMEM_SMEM_CSQ_OFF 99328
#define SMEM_SMEM_CSQ_STAGE_BYTES 1024
#define SMEM_SMEM_CSQ_STRIDE 1024
#define SMEM_TOTAL 100352

extern "C" {

__global__ __launch_bounds__(192) void
kernel_flash_kmeans_assign_cleanroom_tcgen05_v10(float* __restrict__ x_sq, float* __restrict__ c_sq, int32_t* __restrict__ out, const void* x_tmap, const void* c_tmap, int B, int N, int D, int K, int num_n_tiles, int K_tiles)
{
    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;

    extern __shared__ __align__(1024) char smem_raw[];
    int smem;
    smem = (int)(unsigned long long)__cvta_generic_to_shared(smem_raw);
    const int smem_smem_x = smem + 1024;
    const int smem_smem_c = smem + 33792;
    const int smem_smem_csq = smem + 99328;

    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    // Mbarrier init (6 groups, 6 barriers)
    // Mbarriers at smem_raw[0..48)

    if (warp == 0) {
        uint32_t leader = elect_sync();
        // x_full: 1 barriers, init_count=1
        mbarrier_init_pred(smem + 0, 1, leader);
        // x_empty: 1 barriers, init_count=1
        mbarrier_init_pred(smem + 8, 1, leader);
        // c_full: 1 barriers, init_count=1
        mbarrier_init_pred(smem + 16, 1, leader);
        // c_empty: 1 barriers, init_count=1
        mbarrier_init_pred(smem + 24, 1, leader);
        // score_full: 1 barriers, init_count=1
        mbarrier_init_pred(smem + 32, 1, leader);
        // score_empty: 1 barriers, init_count=4
        mbarrier_init_pred(smem + 40, 4, leader);
        asm volatile("fence.mbarrier_init.release.cluster;");
    }

    __syncthreads();

    // TMEM alloc (256 columns, 256 used)
    volatile int* tmem_addr_storage = (volatile int*)(smem_raw + 48);
    if (warp == 1) {
        int _tmem_hold = smem + 48;
        asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;" :: "r"(_tmem_hold), "r"(256) : "memory");
    }

    __syncthreads();
    asm volatile("tcgen05.fence::after_thread_sync;");

    const int warp_id = warp;
    const int lane_id = lane;
    __nv_bfloat16* smem_x = (__nv_bfloat16*)(smem_raw + 1024);
    #define smem_x_addr (smem + 1024)
    __nv_bfloat16* smem_c = (__nv_bfloat16*)(smem_raw + 33792);
    #define smem_c_addr (smem + 33792)
    float* smem_csq = (float*)(smem_raw + 99328);
    #define smem_csq_addr (smem + 99328)
    const int mbar_base = smem;
    #define x_full_addr (mbar_base + 0)
    #define x_empty_addr (mbar_base + 8)
    #define c_full_addr (mbar_base + 16)
    #define c_empty_addr (mbar_base + 24)
    #define score_full_addr (mbar_base + 32)
    #define score_empty_addr (mbar_base + 40)
    const int taddr = tmem_addr_storage[0];

    // ---- Role: load ----
    if (warp == 0) {
        { // load_main
            int num_tiles = B * num_n_tiles;
            uint32_t _phase_x_empty_0 = 1;
            uint32_t _phase_c_empty_0 = 1;
            if (warp_id == 0) {
                if (elect_sync()) {
                    #pragma unroll 1
                    for (unsigned int tile_idx = bid; tile_idx < num_tiles; tile_idx += num_bids) {
                        int batch = tile_idx / num_n_tiles;
                        int n_tile = tile_idx % num_n_tiles;
                        int off_n = n_tile * 128;
                        int x_row = batch * N + off_n;
                        mbarrier_wait(x_empty_addr, _phase_x_empty_0);
                        _phase_x_empty_0 ^= 1;
                        tma_3d_gmem2smem(smem_x_addr, x_tmap, 0, x_row, 0, x_full_addr);
                        mbarrier_arrive_expect_tx(x_full_addr, 32768);
                        #pragma unroll 1
                        for (int iter_k = 0; iter_k < K_tiles; iter_k++) {
                            int off_k = iter_k * 256;
                            int c_row = batch * K + off_k;
                            mbarrier_wait(c_empty_addr, _phase_c_empty_0);
                            _phase_c_empty_0 ^= 1;
                            tma_3d_gmem2smem(smem_c_addr, c_tmap, 0, c_row, 0, c_full_addr);
                            mbarrier_arrive_expect_tx(c_full_addr, 65536);
                        }
                    }
                }
            }
        }
    // ---- Role: mma ----
    } else if (warp == 1) {
        const int tmem_score_tmem = taddr + TMEM_SCORE_TMEM_OFFSET;
        { // mma_main
            int num_tiles = B * num_n_tiles;
            uint32_t _phase_x_full_0 = 0;
            uint32_t _phase_score_empty_0 = 1;
            uint32_t _phase_c_full_0 = 0;
            #pragma unroll 1
            for (unsigned int tile_idx = bid; tile_idx < num_tiles; tile_idx += num_bids) {
                mbarrier_wait(x_full_addr, _phase_x_full_0);
                _phase_x_full_0 ^= 1;
                #pragma unroll 1
                for (int iter_k = 0; iter_k < K_tiles; iter_k++) {
                    mbarrier_wait(score_empty_addr, _phase_score_empty_0);
                    _phase_score_empty_0 ^= 1;
                    mbarrier_wait(c_full_addr, _phase_c_full_0);
                    _phase_c_full_0 ^= 1;
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    int _desc_addr_0 = smem_x_addr + 0 * 32768 + 16384;
                    int _desc_lo_0 = make_warp_uniform((_desc_addr_0 >> 4) & 0x3FFF);
                    int _desc_addr_1 = smem_c_addr + 0 * 65536 + 32768;
                    int _desc_lo_1 = make_warp_uniform((_desc_addr_1 >> 4) & 0x3FFF);
                    int _mma_ss_a_addr_2 = smem_x_addr + 0 * 32768;
                    int _mma_ss_a_lo_2 = make_warp_uniform((_mma_ss_a_addr_2 >> 4) & 0x3FFF);
                    int _mma_ss_b_addr_2 = smem_c_addr + 0 * 65536;
                    int _mma_ss_b_lo_2 = make_warp_uniform((_mma_ss_b_addr_2 >> 4) & 0x3FFF);
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
                    "mov.b32 id, 138413200;\n\t"
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
                    :: "r"(_mma_ss_a_lo_2), "r"(_mma_ss_b_lo_2), "r"(tmem_score_tmem), "r"(0));
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
                    "mov.b32 id, 138413200;\n\t"
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
                    :: "r"(_desc_lo_0), "r"(_desc_lo_1), "r"(tmem_score_tmem), "r"(1));
                    elect_commit(score_full_addr);
                    elect_commit(c_empty_addr);
                }
                elect_commit(x_empty_addr);
            }
        }
    // ---- Role: compute ----
    } else if (warp >= 2 && warp <= 5) {
        const int warp_id_in_wg = warp % 4;
        const int my_row = warp_id_in_wg * 32 + lane;
        const int tmem_row_base_v = (warp_id_in_wg * 32) << 16;
        { // compute_main
            int num_tiles = B * num_n_tiles;
            uint32_t _phase_score_full_0 = 0;
            #pragma unroll 1
            for (unsigned int tile_idx = bid; tile_idx < num_tiles; tile_idx += num_bids) {
                int batch = tile_idx / num_n_tiles;
                int n_tile = tile_idx % num_n_tiles;
                int off_n = n_tile * 128;
                int global_n = off_n + my_row;
                int out_offset = batch * N + global_n;
                int csq_smem_addr = smem_csq_addr + 0;
                float best_score = -3.4e+38f;
                int best_idx = 0;
                #pragma unroll 1
                for (int iter_k = 0; iter_k < K_tiles; iter_k++) {
                    int off_k = iter_k * 256;
                    if (my_row < 64.0f) {
                        int csq_base = my_row * 4;
                        float csq_pack[4];
                        {
                            float4 _v4 = *reinterpret_cast<const float4*>(c_sq + batch * K + off_k + csq_base);
                            csq_pack[0 + 0] = _v4.x;
                            csq_pack[0 + 1] = _v4.y;
                            csq_pack[0 + 2] = _v4.z;
                            csq_pack[0 + 3] = _v4.w;
                        }
                        float csq_half[4];
                        csq_half[0] = 0.5f * csq_pack[0];
                        csq_half[1] = 0.5f * csq_pack[1];
                        csq_half[2] = 0.5f * csq_pack[2];
                        csq_half[3] = 0.5f * csq_pack[3];
                        asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(csq_smem_addr + csq_base * 4), "f"(csq_half[0]), "f"(csq_half[1]), "f"(csq_half[2]), "f"(csq_half[3]) : "memory");
                    }
                    asm volatile("barrier.sync 8, 128;");
                    mbarrier_wait(score_full_addr, _phase_score_full_0);
                    _phase_score_full_0 ^= 1;
                    int score_base = 0;
                    float scores[128];
                    asm volatile(
                        "tcgen05.ld.sync.aligned.32x32b.x128.b32"
                        " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63, %64, %65, %66, %67, %68, %69, %70, %71, %72, %73, %74, %75, %76, %77, %78, %79, %80, %81, %82, %83, %84, %85, %86, %87, %88, %89, %90, %91, %92, %93, %94, %95, %96, %97, %98, %99, %100, %101, %102, %103, %104, %105, %106, %107, %108, %109, %110, %111, %112, %113, %114, %115, %116, %117, %118, %119, %120, %121, %122, %123, %124, %125, %126, %127}, [%128];"
                        : "=f"(scores[0]), "=f"(scores[1]), "=f"(scores[2]), "=f"(scores[3]), "=f"(scores[4]), "=f"(scores[5]), "=f"(scores[6]), "=f"(scores[7]), "=f"(scores[8]), "=f"(scores[9]), "=f"(scores[10]), "=f"(scores[11]), "=f"(scores[12]), "=f"(scores[13]), "=f"(scores[14]), "=f"(scores[15]), "=f"(scores[16]), "=f"(scores[17]), "=f"(scores[18]), "=f"(scores[19]), "=f"(scores[20]), "=f"(scores[21]), "=f"(scores[22]), "=f"(scores[23]), "=f"(scores[24]), "=f"(scores[25]), "=f"(scores[26]), "=f"(scores[27]), "=f"(scores[28]), "=f"(scores[29]), "=f"(scores[30]), "=f"(scores[31]), "=f"(scores[32]), "=f"(scores[33]), "=f"(scores[34]), "=f"(scores[35]), "=f"(scores[36]), "=f"(scores[37]), "=f"(scores[38]), "=f"(scores[39]), "=f"(scores[40]), "=f"(scores[41]), "=f"(scores[42]), "=f"(scores[43]), "=f"(scores[44]), "=f"(scores[45]), "=f"(scores[46]), "=f"(scores[47]), "=f"(scores[48]), "=f"(scores[49]), "=f"(scores[50]), "=f"(scores[51]), "=f"(scores[52]), "=f"(scores[53]), "=f"(scores[54]), "=f"(scores[55]), "=f"(scores[56]), "=f"(scores[57]), "=f"(scores[58]), "=f"(scores[59]), "=f"(scores[60]), "=f"(scores[61]), "=f"(scores[62]), "=f"(scores[63]), "=f"(scores[64]), "=f"(scores[65]), "=f"(scores[66]), "=f"(scores[67]), "=f"(scores[68]), "=f"(scores[69]), "=f"(scores[70]), "=f"(scores[71]), "=f"(scores[72]), "=f"(scores[73]), "=f"(scores[74]), "=f"(scores[75]), "=f"(scores[76]), "=f"(scores[77]), "=f"(scores[78]), "=f"(scores[79]), "=f"(scores[80]), "=f"(scores[81]), "=f"(scores[82]), "=f"(scores[83]), "=f"(scores[84]), "=f"(scores[85]), "=f"(scores[86]), "=f"(scores[87]), "=f"(scores[88]), "=f"(scores[89]), "=f"(scores[90]), "=f"(scores[91]), "=f"(scores[92]), "=f"(scores[93]), "=f"(scores[94]), "=f"(scores[95]), "=f"(scores[96]), "=f"(scores[97]), "=f"(scores[98]), "=f"(scores[99]), "=f"(scores[100]), "=f"(scores[101]), "=f"(scores[102]), "=f"(scores[103]), "=f"(scores[104]), "=f"(scores[105]), "=f"(scores[106]), "=f"(scores[107]), "=f"(scores[108]), "=f"(scores[109]), "=f"(scores[110]), "=f"(scores[111]), "=f"(scores[112]), "=f"(scores[113]), "=f"(scores[114]), "=f"(scores[115]), "=f"(scores[116]), "=f"(scores[117]), "=f"(scores[118]), "=f"(scores[119]), "=f"(scores[120]), "=f"(scores[121]), "=f"(scores[122]), "=f"(scores[123]), "=f"(scores[124]), "=f"(scores[125]), "=f"(scores[126]), "=f"(scores[127])
                        : "r"(taddr + tmem_row_base_v + score_base)
                        : "memory");
                    asm volatile("tcgen05.wait::ld.sync.aligned;");
                    #pragma unroll
                    for (int kk = 0; kk < 128; kk += 4) {
                        float csq_vals[4];
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&csq_vals[0])), "=r"(*reinterpret_cast<uint32_t*>(&csq_vals[1])), "=r"(*reinterpret_cast<uint32_t*>(&csq_vals[2])), "=r"(*reinterpret_cast<uint32_t*>(&csq_vals[3]))
                            : "r"(csq_smem_addr + (score_base + kk) * 4));
                        float d0 = scores[kk] - csq_vals[0];
                        float d1 = scores[kk + 1] - csq_vals[1];
                        float best_01 = d0;
                        int idx_01 = off_k + score_base + kk;
                        if (d1 > best_01) {
                            best_01 = d1;
                            idx_01 = off_k + score_base + kk + 1;
                        }
                        float d2 = scores[kk + 2] - csq_vals[2];
                        float d3 = scores[kk + 3] - csq_vals[3];
                        float best_23 = d2;
                        int idx_23 = off_k + score_base + kk + 2;
                        if (d3 > best_23) {
                            best_23 = d3;
                            idx_23 = off_k + score_base + kk + 3;
                        }
                        float best_group = best_01;
                        int idx_group = idx_01;
                        if (best_23 > best_group) {
                            best_group = best_23;
                            idx_group = idx_23;
                        }
                        if (best_group > best_score) {
                            best_score = best_group;
                            best_idx = idx_group;
                        }
                    }
                    score_base = 128;
                    asm volatile(
                        "tcgen05.ld.sync.aligned.32x32b.x128.b32"
                        " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63, %64, %65, %66, %67, %68, %69, %70, %71, %72, %73, %74, %75, %76, %77, %78, %79, %80, %81, %82, %83, %84, %85, %86, %87, %88, %89, %90, %91, %92, %93, %94, %95, %96, %97, %98, %99, %100, %101, %102, %103, %104, %105, %106, %107, %108, %109, %110, %111, %112, %113, %114, %115, %116, %117, %118, %119, %120, %121, %122, %123, %124, %125, %126, %127}, [%128];"
                        : "=f"(scores[0]), "=f"(scores[1]), "=f"(scores[2]), "=f"(scores[3]), "=f"(scores[4]), "=f"(scores[5]), "=f"(scores[6]), "=f"(scores[7]), "=f"(scores[8]), "=f"(scores[9]), "=f"(scores[10]), "=f"(scores[11]), "=f"(scores[12]), "=f"(scores[13]), "=f"(scores[14]), "=f"(scores[15]), "=f"(scores[16]), "=f"(scores[17]), "=f"(scores[18]), "=f"(scores[19]), "=f"(scores[20]), "=f"(scores[21]), "=f"(scores[22]), "=f"(scores[23]), "=f"(scores[24]), "=f"(scores[25]), "=f"(scores[26]), "=f"(scores[27]), "=f"(scores[28]), "=f"(scores[29]), "=f"(scores[30]), "=f"(scores[31]), "=f"(scores[32]), "=f"(scores[33]), "=f"(scores[34]), "=f"(scores[35]), "=f"(scores[36]), "=f"(scores[37]), "=f"(scores[38]), "=f"(scores[39]), "=f"(scores[40]), "=f"(scores[41]), "=f"(scores[42]), "=f"(scores[43]), "=f"(scores[44]), "=f"(scores[45]), "=f"(scores[46]), "=f"(scores[47]), "=f"(scores[48]), "=f"(scores[49]), "=f"(scores[50]), "=f"(scores[51]), "=f"(scores[52]), "=f"(scores[53]), "=f"(scores[54]), "=f"(scores[55]), "=f"(scores[56]), "=f"(scores[57]), "=f"(scores[58]), "=f"(scores[59]), "=f"(scores[60]), "=f"(scores[61]), "=f"(scores[62]), "=f"(scores[63]), "=f"(scores[64]), "=f"(scores[65]), "=f"(scores[66]), "=f"(scores[67]), "=f"(scores[68]), "=f"(scores[69]), "=f"(scores[70]), "=f"(scores[71]), "=f"(scores[72]), "=f"(scores[73]), "=f"(scores[74]), "=f"(scores[75]), "=f"(scores[76]), "=f"(scores[77]), "=f"(scores[78]), "=f"(scores[79]), "=f"(scores[80]), "=f"(scores[81]), "=f"(scores[82]), "=f"(scores[83]), "=f"(scores[84]), "=f"(scores[85]), "=f"(scores[86]), "=f"(scores[87]), "=f"(scores[88]), "=f"(scores[89]), "=f"(scores[90]), "=f"(scores[91]), "=f"(scores[92]), "=f"(scores[93]), "=f"(scores[94]), "=f"(scores[95]), "=f"(scores[96]), "=f"(scores[97]), "=f"(scores[98]), "=f"(scores[99]), "=f"(scores[100]), "=f"(scores[101]), "=f"(scores[102]), "=f"(scores[103]), "=f"(scores[104]), "=f"(scores[105]), "=f"(scores[106]), "=f"(scores[107]), "=f"(scores[108]), "=f"(scores[109]), "=f"(scores[110]), "=f"(scores[111]), "=f"(scores[112]), "=f"(scores[113]), "=f"(scores[114]), "=f"(scores[115]), "=f"(scores[116]), "=f"(scores[117]), "=f"(scores[118]), "=f"(scores[119]), "=f"(scores[120]), "=f"(scores[121]), "=f"(scores[122]), "=f"(scores[123]), "=f"(scores[124]), "=f"(scores[125]), "=f"(scores[126]), "=f"(scores[127])
                        : "r"(taddr + tmem_row_base_v + score_base)
                        : "memory");
                    asm volatile("tcgen05.wait::ld.sync.aligned;");
                    asm volatile("barrier.sync 9, 128;");
                    if (elect_sync()) {
                        mbarrier_arrive(score_empty_addr);
                    }
                    #pragma unroll
                    for (int kk = 0; kk < 128; kk += 4) {
                        float csq_vals[4];
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&csq_vals[0])), "=r"(*reinterpret_cast<uint32_t*>(&csq_vals[1])), "=r"(*reinterpret_cast<uint32_t*>(&csq_vals[2])), "=r"(*reinterpret_cast<uint32_t*>(&csq_vals[3]))
                            : "r"(csq_smem_addr + (score_base + kk) * 4));
                        float d0 = scores[kk] - csq_vals[0];
                        float d1 = scores[kk + 1] - csq_vals[1];
                        float best_01 = d0;
                        int idx_01 = off_k + score_base + kk;
                        if (d1 > best_01) {
                            best_01 = d1;
                            idx_01 = off_k + score_base + kk + 1;
                        }
                        float d2 = scores[kk + 2] - csq_vals[2];
                        float d3 = scores[kk + 3] - csq_vals[3];
                        float best_23 = d2;
                        int idx_23 = off_k + score_base + kk + 2;
                        if (d3 > best_23) {
                            best_23 = d3;
                            idx_23 = off_k + score_base + kk + 3;
                        }
                        float best_group = best_01;
                        int idx_group = idx_01;
                        if (best_23 > best_group) {
                            best_group = best_23;
                            idx_group = idx_23;
                        }
                        if (best_group > best_score) {
                            best_score = best_group;
                            best_idx = idx_group;
                        }
                    }
                }
                if (global_n < N) {
                    *((int*)(out + out_offset)) = best_idx;
                }
            }
        }
    }

    // Cleanup
    __syncthreads(); // barrier before TMEM dealloc

    if (warp == 1) {
        asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;" :: "r"(tmem_addr_storage[0]), "r"(256));
        asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;");
    }
}

} // extern "C"

#undef NUM_MAIN_STAGES
#undef SMEM_SMEM_CSQ_OFF
#undef SMEM_SMEM_CSQ_STAGE_BYTES
#undef SMEM_SMEM_CSQ_STRIDE
#undef SMEM_SMEM_C_OFF
#undef SMEM_SMEM_C_STAGE_BYTES
#undef SMEM_SMEM_C_STRIDE
#undef SMEM_SMEM_X_OFF
#undef SMEM_SMEM_X_STAGE_BYTES
#undef SMEM_SMEM_X_STRIDE
#undef SMEM_TOTAL
#undef TMEM_NCOLS
#undef TMEM_SCORE_TMEM_OFFSET
#undef c_empty_addr
#undef c_full_addr
#undef score_empty_addr
#undef score_full_addr
#undef smem_c_addr
#undef smem_csq_addr
#undef smem_x_addr
#undef x_empty_addr
#undef x_full_addr

#define TMEM_NCOLS 512
#define TMEM_SCORE_TMEM_OFFSET 0
#define NUM_MAIN_STAGES 1
#define SMEM_SMEM_X0_OFF 1024
#define SMEM_SMEM_X0_STAGE_BYTES 32768
#define SMEM_SMEM_X0_STRIDE 32768
#define SMEM_SMEM_X1_OFF 33792
#define SMEM_SMEM_X1_STAGE_BYTES 32768
#define SMEM_SMEM_X1_STRIDE 32768
#define SMEM_SMEM_C_OFF 66560
#define SMEM_SMEM_C_STAGE_BYTES 65536
#define SMEM_SMEM_C_STRIDE 65536
#define SMEM_SMEM_CSQ_OFF 132096
#define SMEM_SMEM_CSQ_STAGE_BYTES 1024
#define SMEM_SMEM_CSQ_STRIDE 1024
#define SMEM_TOTAL 133120

extern "C" {

__global__ __launch_bounds__(192) void
kernel_flash_kmeans_assign_cleanroom_tcgen05_v15(float* __restrict__ x_sq, float* __restrict__ c_sq, int32_t* __restrict__ out, const void* x_tmap, const void* c_tmap, int B, int N, int D, int K, int num_n_tiles, int K_tiles)
{
    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;

    extern __shared__ __align__(1024) char smem_raw[];
    int smem;
    smem = (int)(unsigned long long)__cvta_generic_to_shared(smem_raw);
    const int smem_smem_x0 = smem + 1024;
    const int smem_smem_x1 = smem + 33792;
    const int smem_smem_c = smem + 66560;
    const int smem_smem_csq = smem + 132096;

    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    // Mbarrier init (10 groups, 10 barriers)
    // Mbarriers at smem_raw[0..80)

    if (warp == 0) {
        uint32_t leader = elect_sync();
        // x0_full: 1 barriers, init_count=1
        mbarrier_init_pred(smem + 0, 1, leader);
        // x0_empty: 1 barriers, init_count=1
        mbarrier_init_pred(smem + 8, 1, leader);
        // x1_full: 1 barriers, init_count=1
        mbarrier_init_pred(smem + 16, 1, leader);
        // x1_empty: 1 barriers, init_count=1
        mbarrier_init_pred(smem + 24, 1, leader);
        // c_full: 1 barriers, init_count=1
        mbarrier_init_pred(smem + 32, 1, leader);
        // c_empty: 1 barriers, init_count=1
        mbarrier_init_pred(smem + 40, 1, leader);
        // score0_full: 1 barriers, init_count=1
        mbarrier_init_pred(smem + 48, 1, leader);
        // score0_empty: 1 barriers, init_count=4
        mbarrier_init_pred(smem + 56, 4, leader);
        // score1_full: 1 barriers, init_count=1
        mbarrier_init_pred(smem + 64, 1, leader);
        // score1_empty: 1 barriers, init_count=4
        mbarrier_init_pred(smem + 72, 4, leader);
        asm volatile("fence.mbarrier_init.release.cluster;");
    }

    __syncthreads();

    // TMEM alloc (512 columns, 512 used)
    volatile int* tmem_addr_storage = (volatile int*)(smem_raw + 80);
    if (warp == 1) {
        int _tmem_hold = smem + 80;
        asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;" :: "r"(_tmem_hold), "r"(512) : "memory");
    }

    __syncthreads();
    asm volatile("tcgen05.fence::after_thread_sync;");

    const int warp_id = warp;
    const int lane_id = lane;
    __nv_bfloat16* smem_x0 = (__nv_bfloat16*)(smem_raw + 1024);
    #define smem_x0_addr (smem + 1024)
    __nv_bfloat16* smem_x1 = (__nv_bfloat16*)(smem_raw + 33792);
    #define smem_x1_addr (smem + 33792)
    __nv_bfloat16* smem_c = (__nv_bfloat16*)(smem_raw + 66560);
    #define smem_c_addr (smem + 66560)
    float* smem_csq = (float*)(smem_raw + 132096);
    #define smem_csq_addr (smem + 132096)
    const int mbar_base = smem;
    #define x0_full_addr (mbar_base + 0)
    #define x0_empty_addr (mbar_base + 8)
    #define x1_full_addr (mbar_base + 16)
    #define x1_empty_addr (mbar_base + 24)
    #define c_full_addr (mbar_base + 32)
    #define c_empty_addr (mbar_base + 40)
    #define score0_full_addr (mbar_base + 48)
    #define score0_empty_addr (mbar_base + 56)
    #define score1_full_addr (mbar_base + 64)
    #define score1_empty_addr (mbar_base + 72)
    const int taddr = tmem_addr_storage[0];

    // ---- Role: load ----
    if (warp == 0) {
        { // load_main
            int pair_n_tiles = num_n_tiles / 2;
            int num_tiles = B * pair_n_tiles;
            uint32_t _phase_x0_empty_0 = 1;
            uint32_t _phase_x1_empty_0 = 1;
            uint32_t _phase_c_empty_0 = 1;
            if (warp_id == 0) {
                if (elect_sync()) {
                    #pragma unroll 1
                    for (unsigned int tile_idx = bid; tile_idx < num_tiles; tile_idx += num_bids) {
                        int batch = tile_idx / pair_n_tiles;
                        int pair_tile = tile_idx % pair_n_tiles;
                        int off_n0 = pair_tile * 256;
                        int off_n1 = off_n0 + 128;
                        int x_row0 = batch * N + off_n0;
                        int x_row1 = batch * N + off_n1;
                        mbarrier_wait(x0_empty_addr, _phase_x0_empty_0);
                        _phase_x0_empty_0 ^= 1;
                        tma_3d_gmem2smem(smem_x0_addr, x_tmap, 0, x_row0, 0, x0_full_addr);
                        mbarrier_arrive_expect_tx(x0_full_addr, 32768);
                        mbarrier_wait(x1_empty_addr, _phase_x1_empty_0);
                        _phase_x1_empty_0 ^= 1;
                        tma_3d_gmem2smem(smem_x1_addr, x_tmap, 0, x_row1, 0, x1_full_addr);
                        mbarrier_arrive_expect_tx(x1_full_addr, 32768);
                        #pragma unroll 1
                        for (int iter_k = 0; iter_k < K_tiles; iter_k++) {
                            int off_k = iter_k * 256;
                            int c_row = batch * K + off_k;
                            mbarrier_wait(c_empty_addr, _phase_c_empty_0);
                            _phase_c_empty_0 ^= 1;
                            tma_3d_gmem2smem(smem_c_addr, c_tmap, 0, c_row, 0, c_full_addr);
                            mbarrier_arrive_expect_tx(c_full_addr, 65536);
                        }
                    }
                }
            }
        }
    // ---- Role: mma ----
    } else if (warp == 1) {
        const int tmem_score_tmem = taddr + TMEM_SCORE_TMEM_OFFSET;
        { // mma_main
            int pair_n_tiles = num_n_tiles / 2;
            int num_tiles = B * pair_n_tiles;
            uint32_t _phase_x0_full_0 = 0;
            uint32_t _phase_x1_full_0 = 0;
            uint32_t _phase_score0_empty_0 = 1;
            uint32_t _phase_c_full_0 = 0;
            uint32_t _phase_score1_empty_0 = 1;
            #pragma unroll 1
            for (unsigned int tile_idx = bid; tile_idx < num_tiles; tile_idx += num_bids) {
                mbarrier_wait(x0_full_addr, _phase_x0_full_0);
                _phase_x0_full_0 ^= 1;
                mbarrier_wait(x1_full_addr, _phase_x1_full_0);
                _phase_x1_full_0 ^= 1;
                #pragma unroll 1
                for (int iter_k = 0; iter_k < K_tiles; iter_k++) {
                    mbarrier_wait(score0_empty_addr, _phase_score0_empty_0);
                    _phase_score0_empty_0 ^= 1;
                    mbarrier_wait(c_full_addr, _phase_c_full_0);
                    _phase_c_full_0 ^= 1;
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    int _desc_addr_0 = smem_x0_addr + 0 * 32768 + 16384;
                    int _desc_lo_0 = make_warp_uniform((_desc_addr_0 >> 4) & 0x3FFF);
                    int _desc_addr_1 = smem_c_addr + 0 * 65536 + 32768;
                    int _desc_lo_1 = make_warp_uniform((_desc_addr_1 >> 4) & 0x3FFF);
                    int _mma_ss_a_addr_2 = smem_x0_addr + 0 * 32768;
                    int _mma_ss_a_lo_2 = make_warp_uniform((_mma_ss_a_addr_2 >> 4) & 0x3FFF);
                    int _mma_ss_b_addr_2 = smem_c_addr + 0 * 65536;
                    int _mma_ss_b_lo_2 = make_warp_uniform((_mma_ss_b_addr_2 >> 4) & 0x3FFF);
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
                    "mov.b32 id, 138413200;\n\t"
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
                    :: "r"(_mma_ss_a_lo_2), "r"(_mma_ss_b_lo_2), "r"((tmem_score_tmem + (256))), "r"(0));
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
                    "mov.b32 id, 138413200;\n\t"
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
                    :: "r"(_desc_lo_0), "r"(_desc_lo_1), "r"((tmem_score_tmem + (256))), "r"(1));
                    elect_commit(score0_full_addr);
                    mbarrier_wait(score1_empty_addr, _phase_score1_empty_0);
                    _phase_score1_empty_0 ^= 1;
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    int _desc_addr_4 = smem_x1_addr + 0 * 32768 + 16384;
                    int _desc_lo_4 = make_warp_uniform((_desc_addr_4 >> 4) & 0x3FFF);
                    int _mma_ss_a_addr_5 = smem_x1_addr + 0 * 32768;
                    int _mma_ss_a_lo_5 = make_warp_uniform((_mma_ss_a_addr_5 >> 4) & 0x3FFF);
                    int _mma_ss_b_addr_5 = smem_c_addr + 0 * 65536;
                    int _mma_ss_b_lo_5 = make_warp_uniform((_mma_ss_b_addr_5 >> 4) & 0x3FFF);
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
                    "mov.b32 id, 138413200;\n\t"
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
                    :: "r"(_mma_ss_a_lo_5), "r"(_mma_ss_b_lo_5), "r"(tmem_score_tmem), "r"(0));
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
                    "mov.b32 id, 138413200;\n\t"
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
                    :: "r"(_desc_lo_4), "r"(_desc_lo_1), "r"(tmem_score_tmem), "r"(1));
                    elect_commit(score1_full_addr);
                    elect_commit(c_empty_addr);
                }
                elect_commit(x0_empty_addr);
                elect_commit(x1_empty_addr);
            }
        }
    // ---- Role: compute ----
    } else if (warp >= 2 && warp <= 5) {
        const int warp_id_in_wg = warp % 4;
        const int my_row = warp_id_in_wg * 32 + lane;
        const int tmem_row_base_v = (warp_id_in_wg * 32) << 16;
        { // compute_main
            int pair_n_tiles = num_n_tiles / 2;
            int num_tiles = B * pair_n_tiles;
            uint32_t _phase_score0_full_0 = 0;
            uint32_t _phase_score1_full_0 = 0;
            #pragma unroll 1
            for (unsigned int tile_idx = bid; tile_idx < num_tiles; tile_idx += num_bids) {
                int batch = tile_idx / pair_n_tiles;
                int pair_tile = tile_idx % pair_n_tiles;
                int off_n0 = pair_tile * 256;
                int off_n1 = off_n0 + 128;
                int global_n0 = off_n0 + my_row;
                int global_n1 = off_n1 + my_row;
                int out_offset0 = batch * N + global_n0;
                int out_offset1 = batch * N + global_n1;
                int csq_smem_addr = smem_csq_addr + 0;
                float best_score0 = -3.4e+38f;
                int best_idx0 = 0;
                float best_score1 = -3.4e+38f;
                int best_idx1 = 0;
                #pragma unroll 1
                for (int iter_k = 0; iter_k < K_tiles; iter_k++) {
                    int off_k = iter_k * 256;
                    if (my_row < 64.0f) {
                        int csq_base = my_row * 4;
                        float csq_pack[4];
                        {
                            float4 _v4 = *reinterpret_cast<const float4*>(c_sq + batch * K + off_k + csq_base);
                            csq_pack[0 + 0] = _v4.x;
                            csq_pack[0 + 1] = _v4.y;
                            csq_pack[0 + 2] = _v4.z;
                            csq_pack[0 + 3] = _v4.w;
                        }
                        float csq_half[4];
                        csq_half[0] = 0.5f * csq_pack[0];
                        csq_half[1] = 0.5f * csq_pack[1];
                        csq_half[2] = 0.5f * csq_pack[2];
                        csq_half[3] = 0.5f * csq_pack[3];
                        asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(csq_smem_addr + csq_base * 4), "f"(csq_half[0]), "f"(csq_half[1]), "f"(csq_half[2]), "f"(csq_half[3]) : "memory");
                    }
                    asm volatile("barrier.sync 8, 128;");
                    mbarrier_wait(score0_full_addr, _phase_score0_full_0);
                    _phase_score0_full_0 ^= 1;
                    int score_base = 0;
                    float scores[128];
                    asm volatile(
                        "tcgen05.ld.sync.aligned.32x32b.x128.b32"
                        " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63, %64, %65, %66, %67, %68, %69, %70, %71, %72, %73, %74, %75, %76, %77, %78, %79, %80, %81, %82, %83, %84, %85, %86, %87, %88, %89, %90, %91, %92, %93, %94, %95, %96, %97, %98, %99, %100, %101, %102, %103, %104, %105, %106, %107, %108, %109, %110, %111, %112, %113, %114, %115, %116, %117, %118, %119, %120, %121, %122, %123, %124, %125, %126, %127}, [%128];"
                        : "=f"(scores[0]), "=f"(scores[1]), "=f"(scores[2]), "=f"(scores[3]), "=f"(scores[4]), "=f"(scores[5]), "=f"(scores[6]), "=f"(scores[7]), "=f"(scores[8]), "=f"(scores[9]), "=f"(scores[10]), "=f"(scores[11]), "=f"(scores[12]), "=f"(scores[13]), "=f"(scores[14]), "=f"(scores[15]), "=f"(scores[16]), "=f"(scores[17]), "=f"(scores[18]), "=f"(scores[19]), "=f"(scores[20]), "=f"(scores[21]), "=f"(scores[22]), "=f"(scores[23]), "=f"(scores[24]), "=f"(scores[25]), "=f"(scores[26]), "=f"(scores[27]), "=f"(scores[28]), "=f"(scores[29]), "=f"(scores[30]), "=f"(scores[31]), "=f"(scores[32]), "=f"(scores[33]), "=f"(scores[34]), "=f"(scores[35]), "=f"(scores[36]), "=f"(scores[37]), "=f"(scores[38]), "=f"(scores[39]), "=f"(scores[40]), "=f"(scores[41]), "=f"(scores[42]), "=f"(scores[43]), "=f"(scores[44]), "=f"(scores[45]), "=f"(scores[46]), "=f"(scores[47]), "=f"(scores[48]), "=f"(scores[49]), "=f"(scores[50]), "=f"(scores[51]), "=f"(scores[52]), "=f"(scores[53]), "=f"(scores[54]), "=f"(scores[55]), "=f"(scores[56]), "=f"(scores[57]), "=f"(scores[58]), "=f"(scores[59]), "=f"(scores[60]), "=f"(scores[61]), "=f"(scores[62]), "=f"(scores[63]), "=f"(scores[64]), "=f"(scores[65]), "=f"(scores[66]), "=f"(scores[67]), "=f"(scores[68]), "=f"(scores[69]), "=f"(scores[70]), "=f"(scores[71]), "=f"(scores[72]), "=f"(scores[73]), "=f"(scores[74]), "=f"(scores[75]), "=f"(scores[76]), "=f"(scores[77]), "=f"(scores[78]), "=f"(scores[79]), "=f"(scores[80]), "=f"(scores[81]), "=f"(scores[82]), "=f"(scores[83]), "=f"(scores[84]), "=f"(scores[85]), "=f"(scores[86]), "=f"(scores[87]), "=f"(scores[88]), "=f"(scores[89]), "=f"(scores[90]), "=f"(scores[91]), "=f"(scores[92]), "=f"(scores[93]), "=f"(scores[94]), "=f"(scores[95]), "=f"(scores[96]), "=f"(scores[97]), "=f"(scores[98]), "=f"(scores[99]), "=f"(scores[100]), "=f"(scores[101]), "=f"(scores[102]), "=f"(scores[103]), "=f"(scores[104]), "=f"(scores[105]), "=f"(scores[106]), "=f"(scores[107]), "=f"(scores[108]), "=f"(scores[109]), "=f"(scores[110]), "=f"(scores[111]), "=f"(scores[112]), "=f"(scores[113]), "=f"(scores[114]), "=f"(scores[115]), "=f"(scores[116]), "=f"(scores[117]), "=f"(scores[118]), "=f"(scores[119]), "=f"(scores[120]), "=f"(scores[121]), "=f"(scores[122]), "=f"(scores[123]), "=f"(scores[124]), "=f"(scores[125]), "=f"(scores[126]), "=f"(scores[127])
                        : "r"(taddr + tmem_row_base_v + 256 + score_base)
                        : "memory");
                    asm volatile("tcgen05.wait::ld.sync.aligned;");
                    #pragma unroll
                    for (int kk = 0; kk < 128; kk += 16) {
                        float csq_vals[4];
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&csq_vals[0])), "=r"(*reinterpret_cast<uint32_t*>(&csq_vals[1])), "=r"(*reinterpret_cast<uint32_t*>(&csq_vals[2])), "=r"(*reinterpret_cast<uint32_t*>(&csq_vals[3]))
                            : "r"(csq_smem_addr + (score_base + kk) * 4));
                        float d0 = scores[kk] - csq_vals[0];
                        float d1 = scores[kk + 1] - csq_vals[1];
                        float best_01 = d0;
                        int idx_01 = off_k + score_base + kk;
                        if (d1 > best_01) {
                            best_01 = d1;
                            idx_01 = off_k + score_base + kk + 1;
                        }
                        float d2 = scores[kk + 2] - csq_vals[2];
                        float d3 = scores[kk + 3] - csq_vals[3];
                        float best_23 = d2;
                        int idx_23 = off_k + score_base + kk + 2;
                        if (d3 > best_23) {
                            best_23 = d3;
                            idx_23 = off_k + score_base + kk + 3;
                        }
                        float best_group = best_01;
                        int idx_group = idx_01;
                        if (best_23 > best_group) {
                            best_group = best_23;
                            idx_group = idx_23;
                        }
                        float csq_vals_hi[4];
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&csq_vals_hi[0])), "=r"(*reinterpret_cast<uint32_t*>(&csq_vals_hi[1])), "=r"(*reinterpret_cast<uint32_t*>(&csq_vals_hi[2])), "=r"(*reinterpret_cast<uint32_t*>(&csq_vals_hi[3]))
                            : "r"(csq_smem_addr + (score_base + kk + 4) * 4));
                        float d4 = scores[kk + 4] - csq_vals_hi[0];
                        float d5 = scores[kk + 5] - csq_vals_hi[1];
                        float best_45 = d4;
                        int idx_45 = off_k + score_base + kk + 4;
                        if (d5 > best_45) {
                            best_45 = d5;
                            idx_45 = off_k + score_base + kk + 5;
                        }
                        float d6 = scores[kk + 6] - csq_vals_hi[2];
                        float d7 = scores[kk + 7] - csq_vals_hi[3];
                        float best_67 = d6;
                        int idx_67 = off_k + score_base + kk + 6;
                        if (d7 > best_67) {
                            best_67 = d7;
                            idx_67 = off_k + score_base + kk + 7;
                        }
                        float best_hi = best_45;
                        int idx_hi = idx_45;
                        if (best_67 > best_hi) {
                            best_hi = best_67;
                            idx_hi = idx_67;
                        }
                        if (best_hi > best_group) {
                            best_group = best_hi;
                            idx_group = idx_hi;
                        }
                        float csq_vals_next[4];
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&csq_vals_next[0])), "=r"(*reinterpret_cast<uint32_t*>(&csq_vals_next[1])), "=r"(*reinterpret_cast<uint32_t*>(&csq_vals_next[2])), "=r"(*reinterpret_cast<uint32_t*>(&csq_vals_next[3]))
                            : "r"(csq_smem_addr + (score_base + kk + 8) * 4));
                        float d8 = scores[kk + 8] - csq_vals_next[0];
                        float d9 = scores[kk + 9] - csq_vals_next[1];
                        float best_89 = d8;
                        int idx_89 = off_k + score_base + kk + 8;
                        if (d9 > best_89) {
                            best_89 = d9;
                            idx_89 = off_k + score_base + kk + 9;
                        }
                        float d10 = scores[kk + 10] - csq_vals_next[2];
                        float d11 = scores[kk + 11] - csq_vals_next[3];
                        float best_1011 = d10;
                        int idx_1011 = off_k + score_base + kk + 10;
                        if (d11 > best_1011) {
                            best_1011 = d11;
                            idx_1011 = off_k + score_base + kk + 11;
                        }
                        float best_next = best_89;
                        int idx_next = idx_89;
                        if (best_1011 > best_next) {
                            best_next = best_1011;
                            idx_next = idx_1011;
                        }
                        float csq_vals_tail[4];
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&csq_vals_tail[0])), "=r"(*reinterpret_cast<uint32_t*>(&csq_vals_tail[1])), "=r"(*reinterpret_cast<uint32_t*>(&csq_vals_tail[2])), "=r"(*reinterpret_cast<uint32_t*>(&csq_vals_tail[3]))
                            : "r"(csq_smem_addr + (score_base + kk + 12) * 4));
                        float d12 = scores[kk + 12] - csq_vals_tail[0];
                        float d13 = scores[kk + 13] - csq_vals_tail[1];
                        float best_1213 = d12;
                        int idx_1213 = off_k + score_base + kk + 12;
                        if (d13 > best_1213) {
                            best_1213 = d13;
                            idx_1213 = off_k + score_base + kk + 13;
                        }
                        float d14 = scores[kk + 14] - csq_vals_tail[2];
                        float d15 = scores[kk + 15] - csq_vals_tail[3];
                        float best_1415 = d14;
                        int idx_1415 = off_k + score_base + kk + 14;
                        if (d15 > best_1415) {
                            best_1415 = d15;
                            idx_1415 = off_k + score_base + kk + 15;
                        }
                        float best_tail = best_1213;
                        int idx_tail = idx_1213;
                        if (best_1415 > best_tail) {
                            best_tail = best_1415;
                            idx_tail = idx_1415;
                        }
                        if (best_tail > best_next) {
                            best_next = best_tail;
                            idx_next = idx_tail;
                        }
                        if (best_next > best_group) {
                            best_group = best_next;
                            idx_group = idx_next;
                        }
                        if (best_group > best_score0) {
                            best_score0 = best_group;
                            best_idx0 = idx_group;
                        }
                    }
                    score_base = 128;
                    asm volatile(
                        "tcgen05.ld.sync.aligned.32x32b.x128.b32"
                        " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63, %64, %65, %66, %67, %68, %69, %70, %71, %72, %73, %74, %75, %76, %77, %78, %79, %80, %81, %82, %83, %84, %85, %86, %87, %88, %89, %90, %91, %92, %93, %94, %95, %96, %97, %98, %99, %100, %101, %102, %103, %104, %105, %106, %107, %108, %109, %110, %111, %112, %113, %114, %115, %116, %117, %118, %119, %120, %121, %122, %123, %124, %125, %126, %127}, [%128];"
                        : "=f"(scores[0]), "=f"(scores[1]), "=f"(scores[2]), "=f"(scores[3]), "=f"(scores[4]), "=f"(scores[5]), "=f"(scores[6]), "=f"(scores[7]), "=f"(scores[8]), "=f"(scores[9]), "=f"(scores[10]), "=f"(scores[11]), "=f"(scores[12]), "=f"(scores[13]), "=f"(scores[14]), "=f"(scores[15]), "=f"(scores[16]), "=f"(scores[17]), "=f"(scores[18]), "=f"(scores[19]), "=f"(scores[20]), "=f"(scores[21]), "=f"(scores[22]), "=f"(scores[23]), "=f"(scores[24]), "=f"(scores[25]), "=f"(scores[26]), "=f"(scores[27]), "=f"(scores[28]), "=f"(scores[29]), "=f"(scores[30]), "=f"(scores[31]), "=f"(scores[32]), "=f"(scores[33]), "=f"(scores[34]), "=f"(scores[35]), "=f"(scores[36]), "=f"(scores[37]), "=f"(scores[38]), "=f"(scores[39]), "=f"(scores[40]), "=f"(scores[41]), "=f"(scores[42]), "=f"(scores[43]), "=f"(scores[44]), "=f"(scores[45]), "=f"(scores[46]), "=f"(scores[47]), "=f"(scores[48]), "=f"(scores[49]), "=f"(scores[50]), "=f"(scores[51]), "=f"(scores[52]), "=f"(scores[53]), "=f"(scores[54]), "=f"(scores[55]), "=f"(scores[56]), "=f"(scores[57]), "=f"(scores[58]), "=f"(scores[59]), "=f"(scores[60]), "=f"(scores[61]), "=f"(scores[62]), "=f"(scores[63]), "=f"(scores[64]), "=f"(scores[65]), "=f"(scores[66]), "=f"(scores[67]), "=f"(scores[68]), "=f"(scores[69]), "=f"(scores[70]), "=f"(scores[71]), "=f"(scores[72]), "=f"(scores[73]), "=f"(scores[74]), "=f"(scores[75]), "=f"(scores[76]), "=f"(scores[77]), "=f"(scores[78]), "=f"(scores[79]), "=f"(scores[80]), "=f"(scores[81]), "=f"(scores[82]), "=f"(scores[83]), "=f"(scores[84]), "=f"(scores[85]), "=f"(scores[86]), "=f"(scores[87]), "=f"(scores[88]), "=f"(scores[89]), "=f"(scores[90]), "=f"(scores[91]), "=f"(scores[92]), "=f"(scores[93]), "=f"(scores[94]), "=f"(scores[95]), "=f"(scores[96]), "=f"(scores[97]), "=f"(scores[98]), "=f"(scores[99]), "=f"(scores[100]), "=f"(scores[101]), "=f"(scores[102]), "=f"(scores[103]), "=f"(scores[104]), "=f"(scores[105]), "=f"(scores[106]), "=f"(scores[107]), "=f"(scores[108]), "=f"(scores[109]), "=f"(scores[110]), "=f"(scores[111]), "=f"(scores[112]), "=f"(scores[113]), "=f"(scores[114]), "=f"(scores[115]), "=f"(scores[116]), "=f"(scores[117]), "=f"(scores[118]), "=f"(scores[119]), "=f"(scores[120]), "=f"(scores[121]), "=f"(scores[122]), "=f"(scores[123]), "=f"(scores[124]), "=f"(scores[125]), "=f"(scores[126]), "=f"(scores[127])
                        : "r"(taddr + tmem_row_base_v + 256 + score_base)
                        : "memory");
                    asm volatile("tcgen05.wait::ld.sync.aligned;");
                    asm volatile("barrier.sync 9, 128;");
                    if (elect_sync()) {
                        mbarrier_arrive(score0_empty_addr);
                    }
                    #pragma unroll
                    for (int kk = 0; kk < 128; kk += 16) {
                        float csq_vals[4];
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&csq_vals[0])), "=r"(*reinterpret_cast<uint32_t*>(&csq_vals[1])), "=r"(*reinterpret_cast<uint32_t*>(&csq_vals[2])), "=r"(*reinterpret_cast<uint32_t*>(&csq_vals[3]))
                            : "r"(csq_smem_addr + (score_base + kk) * 4));
                        float d0 = scores[kk] - csq_vals[0];
                        float d1 = scores[kk + 1] - csq_vals[1];
                        float best_01 = d0;
                        int idx_01 = off_k + score_base + kk;
                        if (d1 > best_01) {
                            best_01 = d1;
                            idx_01 = off_k + score_base + kk + 1;
                        }
                        float d2 = scores[kk + 2] - csq_vals[2];
                        float d3 = scores[kk + 3] - csq_vals[3];
                        float best_23 = d2;
                        int idx_23 = off_k + score_base + kk + 2;
                        if (d3 > best_23) {
                            best_23 = d3;
                            idx_23 = off_k + score_base + kk + 3;
                        }
                        float best_group = best_01;
                        int idx_group = idx_01;
                        if (best_23 > best_group) {
                            best_group = best_23;
                            idx_group = idx_23;
                        }
                        float csq_vals_hi[4];
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&csq_vals_hi[0])), "=r"(*reinterpret_cast<uint32_t*>(&csq_vals_hi[1])), "=r"(*reinterpret_cast<uint32_t*>(&csq_vals_hi[2])), "=r"(*reinterpret_cast<uint32_t*>(&csq_vals_hi[3]))
                            : "r"(csq_smem_addr + (score_base + kk + 4) * 4));
                        float d4 = scores[kk + 4] - csq_vals_hi[0];
                        float d5 = scores[kk + 5] - csq_vals_hi[1];
                        float best_45 = d4;
                        int idx_45 = off_k + score_base + kk + 4;
                        if (d5 > best_45) {
                            best_45 = d5;
                            idx_45 = off_k + score_base + kk + 5;
                        }
                        float d6 = scores[kk + 6] - csq_vals_hi[2];
                        float d7 = scores[kk + 7] - csq_vals_hi[3];
                        float best_67 = d6;
                        int idx_67 = off_k + score_base + kk + 6;
                        if (d7 > best_67) {
                            best_67 = d7;
                            idx_67 = off_k + score_base + kk + 7;
                        }
                        float best_hi = best_45;
                        int idx_hi = idx_45;
                        if (best_67 > best_hi) {
                            best_hi = best_67;
                            idx_hi = idx_67;
                        }
                        if (best_hi > best_group) {
                            best_group = best_hi;
                            idx_group = idx_hi;
                        }
                        float csq_vals_next[4];
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&csq_vals_next[0])), "=r"(*reinterpret_cast<uint32_t*>(&csq_vals_next[1])), "=r"(*reinterpret_cast<uint32_t*>(&csq_vals_next[2])), "=r"(*reinterpret_cast<uint32_t*>(&csq_vals_next[3]))
                            : "r"(csq_smem_addr + (score_base + kk + 8) * 4));
                        float d8 = scores[kk + 8] - csq_vals_next[0];
                        float d9 = scores[kk + 9] - csq_vals_next[1];
                        float best_89 = d8;
                        int idx_89 = off_k + score_base + kk + 8;
                        if (d9 > best_89) {
                            best_89 = d9;
                            idx_89 = off_k + score_base + kk + 9;
                        }
                        float d10 = scores[kk + 10] - csq_vals_next[2];
                        float d11 = scores[kk + 11] - csq_vals_next[3];
                        float best_1011 = d10;
                        int idx_1011 = off_k + score_base + kk + 10;
                        if (d11 > best_1011) {
                            best_1011 = d11;
                            idx_1011 = off_k + score_base + kk + 11;
                        }
                        float best_next = best_89;
                        int idx_next = idx_89;
                        if (best_1011 > best_next) {
                            best_next = best_1011;
                            idx_next = idx_1011;
                        }
                        float csq_vals_tail[4];
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&csq_vals_tail[0])), "=r"(*reinterpret_cast<uint32_t*>(&csq_vals_tail[1])), "=r"(*reinterpret_cast<uint32_t*>(&csq_vals_tail[2])), "=r"(*reinterpret_cast<uint32_t*>(&csq_vals_tail[3]))
                            : "r"(csq_smem_addr + (score_base + kk + 12) * 4));
                        float d12 = scores[kk + 12] - csq_vals_tail[0];
                        float d13 = scores[kk + 13] - csq_vals_tail[1];
                        float best_1213 = d12;
                        int idx_1213 = off_k + score_base + kk + 12;
                        if (d13 > best_1213) {
                            best_1213 = d13;
                            idx_1213 = off_k + score_base + kk + 13;
                        }
                        float d14 = scores[kk + 14] - csq_vals_tail[2];
                        float d15 = scores[kk + 15] - csq_vals_tail[3];
                        float best_1415 = d14;
                        int idx_1415 = off_k + score_base + kk + 14;
                        if (d15 > best_1415) {
                            best_1415 = d15;
                            idx_1415 = off_k + score_base + kk + 15;
                        }
                        float best_tail = best_1213;
                        int idx_tail = idx_1213;
                        if (best_1415 > best_tail) {
                            best_tail = best_1415;
                            idx_tail = idx_1415;
                        }
                        if (best_tail > best_next) {
                            best_next = best_tail;
                            idx_next = idx_tail;
                        }
                        if (best_next > best_group) {
                            best_group = best_next;
                            idx_group = idx_next;
                        }
                        if (best_group > best_score0) {
                            best_score0 = best_group;
                            best_idx0 = idx_group;
                        }
                    }
                    mbarrier_wait(score1_full_addr, _phase_score1_full_0);
                    _phase_score1_full_0 ^= 1;
                    score_base = 0;
                    asm volatile(
                        "tcgen05.ld.sync.aligned.32x32b.x128.b32"
                        " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63, %64, %65, %66, %67, %68, %69, %70, %71, %72, %73, %74, %75, %76, %77, %78, %79, %80, %81, %82, %83, %84, %85, %86, %87, %88, %89, %90, %91, %92, %93, %94, %95, %96, %97, %98, %99, %100, %101, %102, %103, %104, %105, %106, %107, %108, %109, %110, %111, %112, %113, %114, %115, %116, %117, %118, %119, %120, %121, %122, %123, %124, %125, %126, %127}, [%128];"
                        : "=f"(scores[0]), "=f"(scores[1]), "=f"(scores[2]), "=f"(scores[3]), "=f"(scores[4]), "=f"(scores[5]), "=f"(scores[6]), "=f"(scores[7]), "=f"(scores[8]), "=f"(scores[9]), "=f"(scores[10]), "=f"(scores[11]), "=f"(scores[12]), "=f"(scores[13]), "=f"(scores[14]), "=f"(scores[15]), "=f"(scores[16]), "=f"(scores[17]), "=f"(scores[18]), "=f"(scores[19]), "=f"(scores[20]), "=f"(scores[21]), "=f"(scores[22]), "=f"(scores[23]), "=f"(scores[24]), "=f"(scores[25]), "=f"(scores[26]), "=f"(scores[27]), "=f"(scores[28]), "=f"(scores[29]), "=f"(scores[30]), "=f"(scores[31]), "=f"(scores[32]), "=f"(scores[33]), "=f"(scores[34]), "=f"(scores[35]), "=f"(scores[36]), "=f"(scores[37]), "=f"(scores[38]), "=f"(scores[39]), "=f"(scores[40]), "=f"(scores[41]), "=f"(scores[42]), "=f"(scores[43]), "=f"(scores[44]), "=f"(scores[45]), "=f"(scores[46]), "=f"(scores[47]), "=f"(scores[48]), "=f"(scores[49]), "=f"(scores[50]), "=f"(scores[51]), "=f"(scores[52]), "=f"(scores[53]), "=f"(scores[54]), "=f"(scores[55]), "=f"(scores[56]), "=f"(scores[57]), "=f"(scores[58]), "=f"(scores[59]), "=f"(scores[60]), "=f"(scores[61]), "=f"(scores[62]), "=f"(scores[63]), "=f"(scores[64]), "=f"(scores[65]), "=f"(scores[66]), "=f"(scores[67]), "=f"(scores[68]), "=f"(scores[69]), "=f"(scores[70]), "=f"(scores[71]), "=f"(scores[72]), "=f"(scores[73]), "=f"(scores[74]), "=f"(scores[75]), "=f"(scores[76]), "=f"(scores[77]), "=f"(scores[78]), "=f"(scores[79]), "=f"(scores[80]), "=f"(scores[81]), "=f"(scores[82]), "=f"(scores[83]), "=f"(scores[84]), "=f"(scores[85]), "=f"(scores[86]), "=f"(scores[87]), "=f"(scores[88]), "=f"(scores[89]), "=f"(scores[90]), "=f"(scores[91]), "=f"(scores[92]), "=f"(scores[93]), "=f"(scores[94]), "=f"(scores[95]), "=f"(scores[96]), "=f"(scores[97]), "=f"(scores[98]), "=f"(scores[99]), "=f"(scores[100]), "=f"(scores[101]), "=f"(scores[102]), "=f"(scores[103]), "=f"(scores[104]), "=f"(scores[105]), "=f"(scores[106]), "=f"(scores[107]), "=f"(scores[108]), "=f"(scores[109]), "=f"(scores[110]), "=f"(scores[111]), "=f"(scores[112]), "=f"(scores[113]), "=f"(scores[114]), "=f"(scores[115]), "=f"(scores[116]), "=f"(scores[117]), "=f"(scores[118]), "=f"(scores[119]), "=f"(scores[120]), "=f"(scores[121]), "=f"(scores[122]), "=f"(scores[123]), "=f"(scores[124]), "=f"(scores[125]), "=f"(scores[126]), "=f"(scores[127])
                        : "r"(taddr + tmem_row_base_v + score_base)
                        : "memory");
                    asm volatile("tcgen05.wait::ld.sync.aligned;");
                    #pragma unroll
                    for (int kk = 0; kk < 128; kk += 16) {
                        float csq_vals[4];
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&csq_vals[0])), "=r"(*reinterpret_cast<uint32_t*>(&csq_vals[1])), "=r"(*reinterpret_cast<uint32_t*>(&csq_vals[2])), "=r"(*reinterpret_cast<uint32_t*>(&csq_vals[3]))
                            : "r"(csq_smem_addr + (score_base + kk) * 4));
                        float d0 = scores[kk] - csq_vals[0];
                        float d1 = scores[kk + 1] - csq_vals[1];
                        float best_01 = d0;
                        int idx_01 = off_k + score_base + kk;
                        if (d1 > best_01) {
                            best_01 = d1;
                            idx_01 = off_k + score_base + kk + 1;
                        }
                        float d2 = scores[kk + 2] - csq_vals[2];
                        float d3 = scores[kk + 3] - csq_vals[3];
                        float best_23 = d2;
                        int idx_23 = off_k + score_base + kk + 2;
                        if (d3 > best_23) {
                            best_23 = d3;
                            idx_23 = off_k + score_base + kk + 3;
                        }
                        float best_group = best_01;
                        int idx_group = idx_01;
                        if (best_23 > best_group) {
                            best_group = best_23;
                            idx_group = idx_23;
                        }
                        float csq_vals_hi[4];
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&csq_vals_hi[0])), "=r"(*reinterpret_cast<uint32_t*>(&csq_vals_hi[1])), "=r"(*reinterpret_cast<uint32_t*>(&csq_vals_hi[2])), "=r"(*reinterpret_cast<uint32_t*>(&csq_vals_hi[3]))
                            : "r"(csq_smem_addr + (score_base + kk + 4) * 4));
                        float d4 = scores[kk + 4] - csq_vals_hi[0];
                        float d5 = scores[kk + 5] - csq_vals_hi[1];
                        float best_45 = d4;
                        int idx_45 = off_k + score_base + kk + 4;
                        if (d5 > best_45) {
                            best_45 = d5;
                            idx_45 = off_k + score_base + kk + 5;
                        }
                        float d6 = scores[kk + 6] - csq_vals_hi[2];
                        float d7 = scores[kk + 7] - csq_vals_hi[3];
                        float best_67 = d6;
                        int idx_67 = off_k + score_base + kk + 6;
                        if (d7 > best_67) {
                            best_67 = d7;
                            idx_67 = off_k + score_base + kk + 7;
                        }
                        float best_hi = best_45;
                        int idx_hi = idx_45;
                        if (best_67 > best_hi) {
                            best_hi = best_67;
                            idx_hi = idx_67;
                        }
                        if (best_hi > best_group) {
                            best_group = best_hi;
                            idx_group = idx_hi;
                        }
                        float csq_vals_next[4];
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&csq_vals_next[0])), "=r"(*reinterpret_cast<uint32_t*>(&csq_vals_next[1])), "=r"(*reinterpret_cast<uint32_t*>(&csq_vals_next[2])), "=r"(*reinterpret_cast<uint32_t*>(&csq_vals_next[3]))
                            : "r"(csq_smem_addr + (score_base + kk + 8) * 4));
                        float d8 = scores[kk + 8] - csq_vals_next[0];
                        float d9 = scores[kk + 9] - csq_vals_next[1];
                        float best_89 = d8;
                        int idx_89 = off_k + score_base + kk + 8;
                        if (d9 > best_89) {
                            best_89 = d9;
                            idx_89 = off_k + score_base + kk + 9;
                        }
                        float d10 = scores[kk + 10] - csq_vals_next[2];
                        float d11 = scores[kk + 11] - csq_vals_next[3];
                        float best_1011 = d10;
                        int idx_1011 = off_k + score_base + kk + 10;
                        if (d11 > best_1011) {
                            best_1011 = d11;
                            idx_1011 = off_k + score_base + kk + 11;
                        }
                        float best_next = best_89;
                        int idx_next = idx_89;
                        if (best_1011 > best_next) {
                            best_next = best_1011;
                            idx_next = idx_1011;
                        }
                        float csq_vals_tail[4];
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&csq_vals_tail[0])), "=r"(*reinterpret_cast<uint32_t*>(&csq_vals_tail[1])), "=r"(*reinterpret_cast<uint32_t*>(&csq_vals_tail[2])), "=r"(*reinterpret_cast<uint32_t*>(&csq_vals_tail[3]))
                            : "r"(csq_smem_addr + (score_base + kk + 12) * 4));
                        float d12 = scores[kk + 12] - csq_vals_tail[0];
                        float d13 = scores[kk + 13] - csq_vals_tail[1];
                        float best_1213 = d12;
                        int idx_1213 = off_k + score_base + kk + 12;
                        if (d13 > best_1213) {
                            best_1213 = d13;
                            idx_1213 = off_k + score_base + kk + 13;
                        }
                        float d14 = scores[kk + 14] - csq_vals_tail[2];
                        float d15 = scores[kk + 15] - csq_vals_tail[3];
                        float best_1415 = d14;
                        int idx_1415 = off_k + score_base + kk + 14;
                        if (d15 > best_1415) {
                            best_1415 = d15;
                            idx_1415 = off_k + score_base + kk + 15;
                        }
                        float best_tail = best_1213;
                        int idx_tail = idx_1213;
                        if (best_1415 > best_tail) {
                            best_tail = best_1415;
                            idx_tail = idx_1415;
                        }
                        if (best_tail > best_next) {
                            best_next = best_tail;
                            idx_next = idx_tail;
                        }
                        if (best_next > best_group) {
                            best_group = best_next;
                            idx_group = idx_next;
                        }
                        if (best_group > best_score1) {
                            best_score1 = best_group;
                            best_idx1 = idx_group;
                        }
                    }
                    score_base = 128;
                    asm volatile(
                        "tcgen05.ld.sync.aligned.32x32b.x128.b32"
                        " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63, %64, %65, %66, %67, %68, %69, %70, %71, %72, %73, %74, %75, %76, %77, %78, %79, %80, %81, %82, %83, %84, %85, %86, %87, %88, %89, %90, %91, %92, %93, %94, %95, %96, %97, %98, %99, %100, %101, %102, %103, %104, %105, %106, %107, %108, %109, %110, %111, %112, %113, %114, %115, %116, %117, %118, %119, %120, %121, %122, %123, %124, %125, %126, %127}, [%128];"
                        : "=f"(scores[0]), "=f"(scores[1]), "=f"(scores[2]), "=f"(scores[3]), "=f"(scores[4]), "=f"(scores[5]), "=f"(scores[6]), "=f"(scores[7]), "=f"(scores[8]), "=f"(scores[9]), "=f"(scores[10]), "=f"(scores[11]), "=f"(scores[12]), "=f"(scores[13]), "=f"(scores[14]), "=f"(scores[15]), "=f"(scores[16]), "=f"(scores[17]), "=f"(scores[18]), "=f"(scores[19]), "=f"(scores[20]), "=f"(scores[21]), "=f"(scores[22]), "=f"(scores[23]), "=f"(scores[24]), "=f"(scores[25]), "=f"(scores[26]), "=f"(scores[27]), "=f"(scores[28]), "=f"(scores[29]), "=f"(scores[30]), "=f"(scores[31]), "=f"(scores[32]), "=f"(scores[33]), "=f"(scores[34]), "=f"(scores[35]), "=f"(scores[36]), "=f"(scores[37]), "=f"(scores[38]), "=f"(scores[39]), "=f"(scores[40]), "=f"(scores[41]), "=f"(scores[42]), "=f"(scores[43]), "=f"(scores[44]), "=f"(scores[45]), "=f"(scores[46]), "=f"(scores[47]), "=f"(scores[48]), "=f"(scores[49]), "=f"(scores[50]), "=f"(scores[51]), "=f"(scores[52]), "=f"(scores[53]), "=f"(scores[54]), "=f"(scores[55]), "=f"(scores[56]), "=f"(scores[57]), "=f"(scores[58]), "=f"(scores[59]), "=f"(scores[60]), "=f"(scores[61]), "=f"(scores[62]), "=f"(scores[63]), "=f"(scores[64]), "=f"(scores[65]), "=f"(scores[66]), "=f"(scores[67]), "=f"(scores[68]), "=f"(scores[69]), "=f"(scores[70]), "=f"(scores[71]), "=f"(scores[72]), "=f"(scores[73]), "=f"(scores[74]), "=f"(scores[75]), "=f"(scores[76]), "=f"(scores[77]), "=f"(scores[78]), "=f"(scores[79]), "=f"(scores[80]), "=f"(scores[81]), "=f"(scores[82]), "=f"(scores[83]), "=f"(scores[84]), "=f"(scores[85]), "=f"(scores[86]), "=f"(scores[87]), "=f"(scores[88]), "=f"(scores[89]), "=f"(scores[90]), "=f"(scores[91]), "=f"(scores[92]), "=f"(scores[93]), "=f"(scores[94]), "=f"(scores[95]), "=f"(scores[96]), "=f"(scores[97]), "=f"(scores[98]), "=f"(scores[99]), "=f"(scores[100]), "=f"(scores[101]), "=f"(scores[102]), "=f"(scores[103]), "=f"(scores[104]), "=f"(scores[105]), "=f"(scores[106]), "=f"(scores[107]), "=f"(scores[108]), "=f"(scores[109]), "=f"(scores[110]), "=f"(scores[111]), "=f"(scores[112]), "=f"(scores[113]), "=f"(scores[114]), "=f"(scores[115]), "=f"(scores[116]), "=f"(scores[117]), "=f"(scores[118]), "=f"(scores[119]), "=f"(scores[120]), "=f"(scores[121]), "=f"(scores[122]), "=f"(scores[123]), "=f"(scores[124]), "=f"(scores[125]), "=f"(scores[126]), "=f"(scores[127])
                        : "r"(taddr + tmem_row_base_v + score_base)
                        : "memory");
                    asm volatile("tcgen05.wait::ld.sync.aligned;");
                    asm volatile("barrier.sync 9, 128;");
                    if (elect_sync()) {
                        mbarrier_arrive(score1_empty_addr);
                    }
                    #pragma unroll
                    for (int kk = 0; kk < 128; kk += 16) {
                        float csq_vals[4];
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&csq_vals[0])), "=r"(*reinterpret_cast<uint32_t*>(&csq_vals[1])), "=r"(*reinterpret_cast<uint32_t*>(&csq_vals[2])), "=r"(*reinterpret_cast<uint32_t*>(&csq_vals[3]))
                            : "r"(csq_smem_addr + (score_base + kk) * 4));
                        float d0 = scores[kk] - csq_vals[0];
                        float d1 = scores[kk + 1] - csq_vals[1];
                        float best_01 = d0;
                        int idx_01 = off_k + score_base + kk;
                        if (d1 > best_01) {
                            best_01 = d1;
                            idx_01 = off_k + score_base + kk + 1;
                        }
                        float d2 = scores[kk + 2] - csq_vals[2];
                        float d3 = scores[kk + 3] - csq_vals[3];
                        float best_23 = d2;
                        int idx_23 = off_k + score_base + kk + 2;
                        if (d3 > best_23) {
                            best_23 = d3;
                            idx_23 = off_k + score_base + kk + 3;
                        }
                        float best_group = best_01;
                        int idx_group = idx_01;
                        if (best_23 > best_group) {
                            best_group = best_23;
                            idx_group = idx_23;
                        }
                        float csq_vals_hi[4];
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&csq_vals_hi[0])), "=r"(*reinterpret_cast<uint32_t*>(&csq_vals_hi[1])), "=r"(*reinterpret_cast<uint32_t*>(&csq_vals_hi[2])), "=r"(*reinterpret_cast<uint32_t*>(&csq_vals_hi[3]))
                            : "r"(csq_smem_addr + (score_base + kk + 4) * 4));
                        float d4 = scores[kk + 4] - csq_vals_hi[0];
                        float d5 = scores[kk + 5] - csq_vals_hi[1];
                        float best_45 = d4;
                        int idx_45 = off_k + score_base + kk + 4;
                        if (d5 > best_45) {
                            best_45 = d5;
                            idx_45 = off_k + score_base + kk + 5;
                        }
                        float d6 = scores[kk + 6] - csq_vals_hi[2];
                        float d7 = scores[kk + 7] - csq_vals_hi[3];
                        float best_67 = d6;
                        int idx_67 = off_k + score_base + kk + 6;
                        if (d7 > best_67) {
                            best_67 = d7;
                            idx_67 = off_k + score_base + kk + 7;
                        }
                        float best_hi = best_45;
                        int idx_hi = idx_45;
                        if (best_67 > best_hi) {
                            best_hi = best_67;
                            idx_hi = idx_67;
                        }
                        if (best_hi > best_group) {
                            best_group = best_hi;
                            idx_group = idx_hi;
                        }
                        float csq_vals_next[4];
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&csq_vals_next[0])), "=r"(*reinterpret_cast<uint32_t*>(&csq_vals_next[1])), "=r"(*reinterpret_cast<uint32_t*>(&csq_vals_next[2])), "=r"(*reinterpret_cast<uint32_t*>(&csq_vals_next[3]))
                            : "r"(csq_smem_addr + (score_base + kk + 8) * 4));
                        float d8 = scores[kk + 8] - csq_vals_next[0];
                        float d9 = scores[kk + 9] - csq_vals_next[1];
                        float best_89 = d8;
                        int idx_89 = off_k + score_base + kk + 8;
                        if (d9 > best_89) {
                            best_89 = d9;
                            idx_89 = off_k + score_base + kk + 9;
                        }
                        float d10 = scores[kk + 10] - csq_vals_next[2];
                        float d11 = scores[kk + 11] - csq_vals_next[3];
                        float best_1011 = d10;
                        int idx_1011 = off_k + score_base + kk + 10;
                        if (d11 > best_1011) {
                            best_1011 = d11;
                            idx_1011 = off_k + score_base + kk + 11;
                        }
                        float best_next = best_89;
                        int idx_next = idx_89;
                        if (best_1011 > best_next) {
                            best_next = best_1011;
                            idx_next = idx_1011;
                        }
                        float csq_vals_tail[4];
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&csq_vals_tail[0])), "=r"(*reinterpret_cast<uint32_t*>(&csq_vals_tail[1])), "=r"(*reinterpret_cast<uint32_t*>(&csq_vals_tail[2])), "=r"(*reinterpret_cast<uint32_t*>(&csq_vals_tail[3]))
                            : "r"(csq_smem_addr + (score_base + kk + 12) * 4));
                        float d12 = scores[kk + 12] - csq_vals_tail[0];
                        float d13 = scores[kk + 13] - csq_vals_tail[1];
                        float best_1213 = d12;
                        int idx_1213 = off_k + score_base + kk + 12;
                        if (d13 > best_1213) {
                            best_1213 = d13;
                            idx_1213 = off_k + score_base + kk + 13;
                        }
                        float d14 = scores[kk + 14] - csq_vals_tail[2];
                        float d15 = scores[kk + 15] - csq_vals_tail[3];
                        float best_1415 = d14;
                        int idx_1415 = off_k + score_base + kk + 14;
                        if (d15 > best_1415) {
                            best_1415 = d15;
                            idx_1415 = off_k + score_base + kk + 15;
                        }
                        float best_tail = best_1213;
                        int idx_tail = idx_1213;
                        if (best_1415 > best_tail) {
                            best_tail = best_1415;
                            idx_tail = idx_1415;
                        }
                        if (best_tail > best_next) {
                            best_next = best_tail;
                            idx_next = idx_tail;
                        }
                        if (best_next > best_group) {
                            best_group = best_next;
                            idx_group = idx_next;
                        }
                        if (best_group > best_score1) {
                            best_score1 = best_group;
                            best_idx1 = idx_group;
                        }
                    }
                }
                if (global_n0 < N) {
                    *((int*)(out + out_offset0)) = best_idx0;
                }
                if (global_n1 < N) {
                    *((int*)(out + out_offset1)) = best_idx1;
                }
            }
        }
    }

    // Cleanup
    __syncthreads(); // barrier before TMEM dealloc

    if (warp == 1) {
        asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;" :: "r"(tmem_addr_storage[0]), "r"(512));
        asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;");
    }
}

} // extern "C"

#undef NUM_MAIN_STAGES
#undef SMEM_SMEM_CSQ_OFF
#undef SMEM_SMEM_CSQ_STAGE_BYTES
#undef SMEM_SMEM_CSQ_STRIDE
#undef SMEM_SMEM_C_OFF
#undef SMEM_SMEM_C_STAGE_BYTES
#undef SMEM_SMEM_C_STRIDE
#undef SMEM_SMEM_X0_OFF
#undef SMEM_SMEM_X0_STAGE_BYTES
#undef SMEM_SMEM_X0_STRIDE
#undef SMEM_SMEM_X1_OFF
#undef SMEM_SMEM_X1_STAGE_BYTES
#undef SMEM_SMEM_X1_STRIDE
#undef SMEM_TOTAL
#undef TMEM_NCOLS
#undef TMEM_SCORE_TMEM_OFFSET
#undef c_empty_addr
#undef c_full_addr
#undef score0_empty_addr
#undef score0_full_addr
#undef score1_empty_addr
#undef score1_full_addr
#undef smem_c_addr
#undef smem_csq_addr
#undef smem_x0_addr
#undef smem_x1_addr
#undef x0_empty_addr
#undef x0_full_addr
#undef x1_empty_addr
#undef x1_full_addr

#define TMEM_NCOLS 256
#define TMEM_SCORE_TMEM_OFFSET 0
#define NUM_MAIN_STAGES 1
#define SMEM_SMEM_X_OFF 1024
#define SMEM_SMEM_X_STAGE_BYTES 16384
#define SMEM_SMEM_X_STRIDE 16384
#define SMEM_SMEM_C_OFF 17408
#define SMEM_SMEM_C_STAGE_BYTES 32768
#define SMEM_SMEM_C_STRIDE 32768
#define SMEM_SMEM_CSQ_OFF 50176
#define SMEM_SMEM_CSQ_STAGE_BYTES 1024
#define SMEM_SMEM_CSQ_STRIDE 1024
#define SMEM_TOTAL 51200

extern "C" {

__global__ __launch_bounds__(192) void
kernel_flash_kmeans_assign_d64_direct_1p2gap_9f2a_v1(float* __restrict__ x_sq, float* __restrict__ c_sq, int32_t* __restrict__ out, const void* x_tmap, const void* c_tmap, int B, int N, int D, int K, int num_n_tiles, int K_tiles)
{
    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;

    extern __shared__ __align__(1024) char smem_raw[];
    int smem;
    smem = (int)(unsigned long long)__cvta_generic_to_shared(smem_raw);
    const int smem_smem_x = smem + 1024;
    const int smem_smem_c = smem + 17408;
    const int smem_smem_csq = smem + 50176;

    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    // Mbarrier init (6 groups, 6 barriers)
    // Mbarriers at smem_raw[0..48)

    if (warp == 0) {
        uint32_t leader = elect_sync();
        // x_full: 1 barriers, init_count=1
        mbarrier_init_pred(smem + 0, 1, leader);
        // x_empty: 1 barriers, init_count=1
        mbarrier_init_pred(smem + 8, 1, leader);
        // c_full: 1 barriers, init_count=1
        mbarrier_init_pred(smem + 16, 1, leader);
        // c_empty: 1 barriers, init_count=1
        mbarrier_init_pred(smem + 24, 1, leader);
        // score_full: 1 barriers, init_count=1
        mbarrier_init_pred(smem + 32, 1, leader);
        // score_empty: 1 barriers, init_count=4
        mbarrier_init_pred(smem + 40, 4, leader);
        asm volatile("fence.mbarrier_init.release.cluster;");
    }

    __syncthreads();

    // TMEM alloc (256 columns, 256 used)
    volatile int* tmem_addr_storage = (volatile int*)(smem_raw + 48);
    if (warp == 1) {
        int _tmem_hold = smem + 48;
        asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;" :: "r"(_tmem_hold), "r"(256) : "memory");
    }

    __syncthreads();
    asm volatile("tcgen05.fence::after_thread_sync;");

    const int warp_id = warp;
    const int lane_id = lane;
    __nv_bfloat16* smem_x = (__nv_bfloat16*)(smem_raw + 1024);
    #define smem_x_addr (smem + 1024)
    __nv_bfloat16* smem_c = (__nv_bfloat16*)(smem_raw + 17408);
    #define smem_c_addr (smem + 17408)
    float* smem_csq = (float*)(smem_raw + 50176);
    #define smem_csq_addr (smem + 50176)
    const int mbar_base = smem;
    #define x_full_addr (mbar_base + 0)
    #define x_empty_addr (mbar_base + 8)
    #define c_full_addr (mbar_base + 16)
    #define c_empty_addr (mbar_base + 24)
    #define score_full_addr (mbar_base + 32)
    #define score_empty_addr (mbar_base + 40)
    const int taddr = tmem_addr_storage[0];

    // ---- Role: load ----
    if (warp == 0) {
        { // load_main
            int num_tiles = B * num_n_tiles;
            uint32_t _phase_x_empty_0 = 1;
            uint32_t _phase_c_empty_0 = 1;
            if (warp_id == 0) {
                if (elect_sync()) {
                    #pragma unroll 1
                    for (unsigned int tile_idx = bid; tile_idx < num_tiles; tile_idx += num_bids) {
                        int batch = tile_idx / num_n_tiles;
                        int n_tile = tile_idx % num_n_tiles;
                        int off_n = n_tile * 128;
                        int x_row = batch * N + off_n;
                        mbarrier_wait(x_empty_addr, _phase_x_empty_0);
                        _phase_x_empty_0 ^= 1;
                        tma_3d_gmem2smem(smem_x_addr, x_tmap, 0, x_row, 0, x_full_addr);
                        mbarrier_arrive_expect_tx(x_full_addr, 16384);
                        #pragma unroll 1
                        for (int iter_k = 0; iter_k < K_tiles; iter_k++) {
                            int off_k = iter_k * 256;
                            int c_row = batch * K + off_k;
                            mbarrier_wait(c_empty_addr, _phase_c_empty_0);
                            _phase_c_empty_0 ^= 1;
                            tma_3d_gmem2smem(smem_c_addr, c_tmap, 0, c_row, 0, c_full_addr);
                            mbarrier_arrive_expect_tx(c_full_addr, 32768);
                        }
                    }
                }
            }
        }
    // ---- Role: mma ----
    } else if (warp == 1) {
        const int tmem_score_tmem = taddr + TMEM_SCORE_TMEM_OFFSET;
        { // mma_main
            int num_tiles = B * num_n_tiles;
            uint32_t _phase_x_full_0 = 0;
            uint32_t _phase_score_empty_0 = 1;
            uint32_t _phase_c_full_0 = 0;
            #pragma unroll 1
            for (unsigned int _tile_idx = bid; _tile_idx < num_tiles; _tile_idx += num_bids) {
                mbarrier_wait(x_full_addr, _phase_x_full_0);
                _phase_x_full_0 ^= 1;
                #pragma unroll 1
                for (int _iter_k = 0; _iter_k < K_tiles; _iter_k++) {
                    mbarrier_wait(score_empty_addr, _phase_score_empty_0);
                    _phase_score_empty_0 ^= 1;
                    mbarrier_wait(c_full_addr, _phase_c_full_0);
                    _phase_c_full_0 ^= 1;
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    int _mma_ss_a_addr_0 = smem_x_addr + 0 * 16384;
                    int _mma_ss_a_lo_0 = make_warp_uniform((_mma_ss_a_addr_0 >> 4) & 0x3FFF);
                    int _mma_ss_b_addr_0 = smem_c_addr + 0 * 32768;
                    int _mma_ss_b_lo_0 = make_warp_uniform((_mma_ss_b_addr_0 >> 4) & 0x3FFF);
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
                    "mov.b32 id, 138413200;\n\t"
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
                    :: "r"(_mma_ss_a_lo_0), "r"(_mma_ss_b_lo_0), "r"(tmem_score_tmem), "r"(0));
                    elect_commit(score_full_addr);
                    elect_commit(c_empty_addr);
                }
                elect_commit(x_empty_addr);
            }
        }
    // ---- Role: compute ----
    } else if (warp >= 2 && warp <= 5) {
        const int warp_id_in_wg = warp % 4;
        const int my_row = warp_id_in_wg * 32 + lane;
        const int tmem_row_base_v = (warp_id_in_wg * 32) << 16;
        { // compute_main
            int num_tiles = B * num_n_tiles;
            uint32_t _phase_score_full_0 = 0;
            #pragma unroll 1
            for (unsigned int tile_idx = bid; tile_idx < num_tiles; tile_idx += num_bids) {
                int batch = tile_idx / num_n_tiles;
                int n_tile = tile_idx % num_n_tiles;
                int off_n = n_tile * 128;
                int global_n = off_n + my_row;
                int out_offset = batch * N + global_n;
                int csq_smem_addr = smem_csq_addr + 0;
                float best_score = -3.4e+38f;
                int best_idx = 0;
                #pragma unroll 1
                for (int iter_k = 0; iter_k < K_tiles; iter_k++) {
                    int off_k = iter_k * 256;
                    if (my_row < 64.0f) {
                        int csq_base = my_row * 4;
                        float csq_pack[4];
                        {
                            float4 _v4 = *reinterpret_cast<const float4*>(c_sq + batch * K + off_k + csq_base);
                            csq_pack[0 + 0] = _v4.x;
                            csq_pack[0 + 1] = _v4.y;
                            csq_pack[0 + 2] = _v4.z;
                            csq_pack[0 + 3] = _v4.w;
                        }
                        float csq_half[4];
                        csq_half[0] = 0.5f * csq_pack[0];
                        csq_half[1] = 0.5f * csq_pack[1];
                        csq_half[2] = 0.5f * csq_pack[2];
                        csq_half[3] = 0.5f * csq_pack[3];
                        asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(csq_smem_addr + csq_base * 4), "f"(csq_half[0]), "f"(csq_half[1]), "f"(csq_half[2]), "f"(csq_half[3]) : "memory");
                    }
                    asm volatile("barrier.sync 8, 128;");
                    mbarrier_wait(score_full_addr, _phase_score_full_0);
                    _phase_score_full_0 ^= 1;
                    int score_base = 0;
                    float scores[128];
                    asm volatile(
                        "tcgen05.ld.sync.aligned.32x32b.x128.b32"
                        " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63, %64, %65, %66, %67, %68, %69, %70, %71, %72, %73, %74, %75, %76, %77, %78, %79, %80, %81, %82, %83, %84, %85, %86, %87, %88, %89, %90, %91, %92, %93, %94, %95, %96, %97, %98, %99, %100, %101, %102, %103, %104, %105, %106, %107, %108, %109, %110, %111, %112, %113, %114, %115, %116, %117, %118, %119, %120, %121, %122, %123, %124, %125, %126, %127}, [%128];"
                        : "=f"(scores[0]), "=f"(scores[1]), "=f"(scores[2]), "=f"(scores[3]), "=f"(scores[4]), "=f"(scores[5]), "=f"(scores[6]), "=f"(scores[7]), "=f"(scores[8]), "=f"(scores[9]), "=f"(scores[10]), "=f"(scores[11]), "=f"(scores[12]), "=f"(scores[13]), "=f"(scores[14]), "=f"(scores[15]), "=f"(scores[16]), "=f"(scores[17]), "=f"(scores[18]), "=f"(scores[19]), "=f"(scores[20]), "=f"(scores[21]), "=f"(scores[22]), "=f"(scores[23]), "=f"(scores[24]), "=f"(scores[25]), "=f"(scores[26]), "=f"(scores[27]), "=f"(scores[28]), "=f"(scores[29]), "=f"(scores[30]), "=f"(scores[31]), "=f"(scores[32]), "=f"(scores[33]), "=f"(scores[34]), "=f"(scores[35]), "=f"(scores[36]), "=f"(scores[37]), "=f"(scores[38]), "=f"(scores[39]), "=f"(scores[40]), "=f"(scores[41]), "=f"(scores[42]), "=f"(scores[43]), "=f"(scores[44]), "=f"(scores[45]), "=f"(scores[46]), "=f"(scores[47]), "=f"(scores[48]), "=f"(scores[49]), "=f"(scores[50]), "=f"(scores[51]), "=f"(scores[52]), "=f"(scores[53]), "=f"(scores[54]), "=f"(scores[55]), "=f"(scores[56]), "=f"(scores[57]), "=f"(scores[58]), "=f"(scores[59]), "=f"(scores[60]), "=f"(scores[61]), "=f"(scores[62]), "=f"(scores[63]), "=f"(scores[64]), "=f"(scores[65]), "=f"(scores[66]), "=f"(scores[67]), "=f"(scores[68]), "=f"(scores[69]), "=f"(scores[70]), "=f"(scores[71]), "=f"(scores[72]), "=f"(scores[73]), "=f"(scores[74]), "=f"(scores[75]), "=f"(scores[76]), "=f"(scores[77]), "=f"(scores[78]), "=f"(scores[79]), "=f"(scores[80]), "=f"(scores[81]), "=f"(scores[82]), "=f"(scores[83]), "=f"(scores[84]), "=f"(scores[85]), "=f"(scores[86]), "=f"(scores[87]), "=f"(scores[88]), "=f"(scores[89]), "=f"(scores[90]), "=f"(scores[91]), "=f"(scores[92]), "=f"(scores[93]), "=f"(scores[94]), "=f"(scores[95]), "=f"(scores[96]), "=f"(scores[97]), "=f"(scores[98]), "=f"(scores[99]), "=f"(scores[100]), "=f"(scores[101]), "=f"(scores[102]), "=f"(scores[103]), "=f"(scores[104]), "=f"(scores[105]), "=f"(scores[106]), "=f"(scores[107]), "=f"(scores[108]), "=f"(scores[109]), "=f"(scores[110]), "=f"(scores[111]), "=f"(scores[112]), "=f"(scores[113]), "=f"(scores[114]), "=f"(scores[115]), "=f"(scores[116]), "=f"(scores[117]), "=f"(scores[118]), "=f"(scores[119]), "=f"(scores[120]), "=f"(scores[121]), "=f"(scores[122]), "=f"(scores[123]), "=f"(scores[124]), "=f"(scores[125]), "=f"(scores[126]), "=f"(scores[127])
                        : "r"(taddr + tmem_row_base_v + score_base)
                        : "memory");
                    asm volatile("tcgen05.wait::ld.sync.aligned;");
                    #pragma unroll
                    for (int kk = 0; kk < 128; kk += 4) {
                        float csq_vals[4];
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&csq_vals[0])), "=r"(*reinterpret_cast<uint32_t*>(&csq_vals[1])), "=r"(*reinterpret_cast<uint32_t*>(&csq_vals[2])), "=r"(*reinterpret_cast<uint32_t*>(&csq_vals[3]))
                            : "r"(csq_smem_addr + (score_base + kk) * 4));
                        float d0 = scores[kk] - csq_vals[0];
                        float d1 = scores[kk + 1] - csq_vals[1];
                        float best_01 = d0;
                        int idx_01 = off_k + score_base + kk;
                        if (d1 > best_01) {
                            best_01 = d1;
                            idx_01 = off_k + score_base + kk + 1;
                        }
                        float d2 = scores[kk + 2] - csq_vals[2];
                        float d3 = scores[kk + 3] - csq_vals[3];
                        float best_23 = d2;
                        int idx_23 = off_k + score_base + kk + 2;
                        if (d3 > best_23) {
                            best_23 = d3;
                            idx_23 = off_k + score_base + kk + 3;
                        }
                        float best_group = best_01;
                        int idx_group = idx_01;
                        if (best_23 > best_group) {
                            best_group = best_23;
                            idx_group = idx_23;
                        }
                        if (best_group > best_score) {
                            best_score = best_group;
                            best_idx = idx_group;
                        }
                    }
                    score_base = 128;
                    asm volatile(
                        "tcgen05.ld.sync.aligned.32x32b.x128.b32"
                        " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63, %64, %65, %66, %67, %68, %69, %70, %71, %72, %73, %74, %75, %76, %77, %78, %79, %80, %81, %82, %83, %84, %85, %86, %87, %88, %89, %90, %91, %92, %93, %94, %95, %96, %97, %98, %99, %100, %101, %102, %103, %104, %105, %106, %107, %108, %109, %110, %111, %112, %113, %114, %115, %116, %117, %118, %119, %120, %121, %122, %123, %124, %125, %126, %127}, [%128];"
                        : "=f"(scores[0]), "=f"(scores[1]), "=f"(scores[2]), "=f"(scores[3]), "=f"(scores[4]), "=f"(scores[5]), "=f"(scores[6]), "=f"(scores[7]), "=f"(scores[8]), "=f"(scores[9]), "=f"(scores[10]), "=f"(scores[11]), "=f"(scores[12]), "=f"(scores[13]), "=f"(scores[14]), "=f"(scores[15]), "=f"(scores[16]), "=f"(scores[17]), "=f"(scores[18]), "=f"(scores[19]), "=f"(scores[20]), "=f"(scores[21]), "=f"(scores[22]), "=f"(scores[23]), "=f"(scores[24]), "=f"(scores[25]), "=f"(scores[26]), "=f"(scores[27]), "=f"(scores[28]), "=f"(scores[29]), "=f"(scores[30]), "=f"(scores[31]), "=f"(scores[32]), "=f"(scores[33]), "=f"(scores[34]), "=f"(scores[35]), "=f"(scores[36]), "=f"(scores[37]), "=f"(scores[38]), "=f"(scores[39]), "=f"(scores[40]), "=f"(scores[41]), "=f"(scores[42]), "=f"(scores[43]), "=f"(scores[44]), "=f"(scores[45]), "=f"(scores[46]), "=f"(scores[47]), "=f"(scores[48]), "=f"(scores[49]), "=f"(scores[50]), "=f"(scores[51]), "=f"(scores[52]), "=f"(scores[53]), "=f"(scores[54]), "=f"(scores[55]), "=f"(scores[56]), "=f"(scores[57]), "=f"(scores[58]), "=f"(scores[59]), "=f"(scores[60]), "=f"(scores[61]), "=f"(scores[62]), "=f"(scores[63]), "=f"(scores[64]), "=f"(scores[65]), "=f"(scores[66]), "=f"(scores[67]), "=f"(scores[68]), "=f"(scores[69]), "=f"(scores[70]), "=f"(scores[71]), "=f"(scores[72]), "=f"(scores[73]), "=f"(scores[74]), "=f"(scores[75]), "=f"(scores[76]), "=f"(scores[77]), "=f"(scores[78]), "=f"(scores[79]), "=f"(scores[80]), "=f"(scores[81]), "=f"(scores[82]), "=f"(scores[83]), "=f"(scores[84]), "=f"(scores[85]), "=f"(scores[86]), "=f"(scores[87]), "=f"(scores[88]), "=f"(scores[89]), "=f"(scores[90]), "=f"(scores[91]), "=f"(scores[92]), "=f"(scores[93]), "=f"(scores[94]), "=f"(scores[95]), "=f"(scores[96]), "=f"(scores[97]), "=f"(scores[98]), "=f"(scores[99]), "=f"(scores[100]), "=f"(scores[101]), "=f"(scores[102]), "=f"(scores[103]), "=f"(scores[104]), "=f"(scores[105]), "=f"(scores[106]), "=f"(scores[107]), "=f"(scores[108]), "=f"(scores[109]), "=f"(scores[110]), "=f"(scores[111]), "=f"(scores[112]), "=f"(scores[113]), "=f"(scores[114]), "=f"(scores[115]), "=f"(scores[116]), "=f"(scores[117]), "=f"(scores[118]), "=f"(scores[119]), "=f"(scores[120]), "=f"(scores[121]), "=f"(scores[122]), "=f"(scores[123]), "=f"(scores[124]), "=f"(scores[125]), "=f"(scores[126]), "=f"(scores[127])
                        : "r"(taddr + tmem_row_base_v + score_base)
                        : "memory");
                    asm volatile("tcgen05.wait::ld.sync.aligned;");
                    asm volatile("barrier.sync 9, 128;");
                    if (elect_sync()) {
                        mbarrier_arrive(score_empty_addr);
                    }
                    #pragma unroll
                    for (int kk = 0; kk < 128; kk += 4) {
                        float csq_vals[4];
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&csq_vals[0])), "=r"(*reinterpret_cast<uint32_t*>(&csq_vals[1])), "=r"(*reinterpret_cast<uint32_t*>(&csq_vals[2])), "=r"(*reinterpret_cast<uint32_t*>(&csq_vals[3]))
                            : "r"(csq_smem_addr + (score_base + kk) * 4));
                        float d0 = scores[kk] - csq_vals[0];
                        float d1 = scores[kk + 1] - csq_vals[1];
                        float best_01 = d0;
                        int idx_01 = off_k + score_base + kk;
                        if (d1 > best_01) {
                            best_01 = d1;
                            idx_01 = off_k + score_base + kk + 1;
                        }
                        float d2 = scores[kk + 2] - csq_vals[2];
                        float d3 = scores[kk + 3] - csq_vals[3];
                        float best_23 = d2;
                        int idx_23 = off_k + score_base + kk + 2;
                        if (d3 > best_23) {
                            best_23 = d3;
                            idx_23 = off_k + score_base + kk + 3;
                        }
                        float best_group = best_01;
                        int idx_group = idx_01;
                        if (best_23 > best_group) {
                            best_group = best_23;
                            idx_group = idx_23;
                        }
                        if (best_group > best_score) {
                            best_score = best_group;
                            best_idx = idx_group;
                        }
                    }
                }
                if (global_n < N) {
                    *((int*)(out + out_offset)) = best_idx;
                }
            }
        }
    }

    // Cleanup
    __syncthreads(); // barrier before TMEM dealloc

    if (warp == 1) {
        asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;" :: "r"(tmem_addr_storage[0]), "r"(256));
        asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;");
    }
}

} // extern "C"

#undef NUM_MAIN_STAGES
#undef SMEM_SMEM_CSQ_OFF
#undef SMEM_SMEM_CSQ_STAGE_BYTES
#undef SMEM_SMEM_CSQ_STRIDE
#undef SMEM_SMEM_C_OFF
#undef SMEM_SMEM_C_STAGE_BYTES
#undef SMEM_SMEM_C_STRIDE
#undef SMEM_SMEM_X_OFF
#undef SMEM_SMEM_X_STAGE_BYTES
#undef SMEM_SMEM_X_STRIDE
#undef SMEM_TOTAL
#undef TMEM_NCOLS
#undef TMEM_SCORE_TMEM_OFFSET
#undef c_empty_addr
#undef c_full_addr
#undef score_empty_addr
#undef score_full_addr
#undef smem_c_addr
#undef smem_csq_addr
#undef smem_x_addr
#undef x_empty_addr
#undef x_full_addr

#define TMEM_NCOLS 256
#define TMEM_SCORE_TMEM_OFFSET 0
#define NUM_MAIN_STAGES 1
#define SMEM_SMEM_X_OFF 1024
#define SMEM_SMEM_X_STAGE_BYTES 16384
#define SMEM_SMEM_X_STRIDE 16384
#define SMEM_SMEM_C_OFF 17408
#define SMEM_SMEM_C_STAGE_BYTES 32768
#define SMEM_SMEM_C_STRIDE 32768
#define SMEM_SMEM_CSQ_OFF 50176
#define SMEM_SMEM_CSQ_STAGE_BYTES 1024
#define SMEM_SMEM_CSQ_STRIDE 1024
#define SMEM_TOTAL 51200

extern "C" {

__global__ __launch_bounds__(192) void
kernel_flash_kmeans_assign_microdim_direct_9c0d_v1(__nv_bfloat16* __restrict__ x, __nv_bfloat16* __restrict__ centroids, float* __restrict__ x_sq, float* __restrict__ c_sq, int32_t* __restrict__ out, int B, int N, int D, int K, int num_n_tiles, int K_tiles)
{
    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;

    extern __shared__ __align__(1024) char smem_raw[];
    int smem;
    smem = (int)(unsigned long long)__cvta_generic_to_shared(smem_raw);
    const int smem_smem_x = smem + 1024;
    const int smem_smem_c = smem + 17408;
    const int smem_smem_csq = smem + 50176;

    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    // Mbarrier init (6 groups, 6 barriers)
    // Mbarriers at smem_raw[0..48)

    if (warp == 0) {
        uint32_t leader = elect_sync();
        // x_full: 1 barriers, init_count=1
        mbarrier_init_pred(smem + 0, 1, leader);
        // x_empty: 1 barriers, init_count=1
        mbarrier_init_pred(smem + 8, 1, leader);
        // c_full: 1 barriers, init_count=1
        mbarrier_init_pred(smem + 16, 1, leader);
        // c_empty: 1 barriers, init_count=1
        mbarrier_init_pred(smem + 24, 1, leader);
        // score_full: 1 barriers, init_count=1
        mbarrier_init_pred(smem + 32, 1, leader);
        // score_empty: 1 barriers, init_count=4
        mbarrier_init_pred(smem + 40, 4, leader);
        asm volatile("fence.mbarrier_init.release.cluster;");
    }

    __syncthreads();

    // TMEM alloc (256 columns, 256 used)
    volatile int* tmem_addr_storage = (volatile int*)(smem_raw + 48);
    if (warp == 1) {
        int _tmem_hold = smem + 48;
        asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;" :: "r"(_tmem_hold), "r"(256) : "memory");
    }

    __syncthreads();
    asm volatile("tcgen05.fence::after_thread_sync;");

    const int warp_id = warp;
    const int lane_id = lane;
    __nv_bfloat16* smem_x = (__nv_bfloat16*)(smem_raw + 1024);
    #define smem_x_addr (smem + 1024)
    __nv_bfloat16* smem_c = (__nv_bfloat16*)(smem_raw + 17408);
    #define smem_c_addr (smem + 17408)
    float* smem_csq = (float*)(smem_raw + 50176);
    #define smem_csq_addr (smem + 50176)
    const int mbar_base = smem;
    #define x_full_addr (mbar_base + 0)
    #define x_empty_addr (mbar_base + 8)
    #define c_full_addr (mbar_base + 16)
    #define c_empty_addr (mbar_base + 24)
    #define score_full_addr (mbar_base + 32)
    #define score_empty_addr (mbar_base + 40)
    const int taddr = tmem_addr_storage[0];

    // ---- Role: load ----
    if (warp == 0) {
        { // load_main
            int num_tiles = B * num_n_tiles;
            int lane_start = tid;
            const int x_vecs = 1024;
            const int c_vecs = 2048;
            uint32_t _phase_x_empty_0 = 1;
            uint32_t _phase_c_empty_0 = 1;
            #pragma unroll 1
            for (unsigned int tile_idx = bid; tile_idx < num_tiles; tile_idx += num_bids) {
                int batch = tile_idx / num_n_tiles;
                int n_tile = tile_idx % num_n_tiles;
                int off_n = n_tile * 128;
                int x_row_base = batch * N + off_n;
                mbarrier_wait(x_empty_addr, _phase_x_empty_0);
                _phase_x_empty_0 ^= 1;
                #pragma unroll 1
                for (unsigned int vec_idx = lane_start; vec_idx < x_vecs; vec_idx += 32) {
                    int d_pad = vec_idx % 8 * 8;
                    int row = vec_idx / 8;
                    float vals[8];
                    unsigned int packed[4];
                    if (d_pad < D) {
                        {
                            const uint4* _vptr_0 = reinterpret_cast<const uint4*>(x + (x_row_base + row) * D + d_pad);
                            uint4 _vld_0[1];
                            #pragma unroll
                            for (int _blk = 0; _blk < 1; _blk++) {
                                _vld_0[_blk] = _vptr_0[_blk];
                                __nv_bfloat16* _velems_0 = reinterpret_cast<__nv_bfloat16*>(&_vld_0[_blk]);
                                #pragma unroll
                                for (int _j = 0; _j < 8; _j++)
                                    vals[0 + _blk * 8 + _j] = __bfloat162float(_velems_0[_j]);
                            }
                        }
                    } else {
                        #pragma unroll
                        for (int vi = 0; vi < 8; vi++) {
                            vals[vi] = 0.0f;
                        }
                    }
                    #pragma unroll
                    for (int _lp = 0; _lp < 4; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(vals[_lp*2 + 0], vals[_lp*2+1 + 0]));
                        packed[_lp] = *(uint32_t*)&_bf2;
                    }
                    int x_addr = (smem_x_addr + (d_pad / 64 * 16384 + row * 128 + d_pad % 64 * 2 ^ (d_pad / 64 * 16384 + row * 128 + d_pad % 64 * 2 >> 7 & 7) << 4));
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(x_addr), "r"(packed[0]), "r"(packed[1]), "r"(packed[2]), "r"(packed[3]) : "memory");
                }
                asm volatile("barrier.sync 8, 32;");
                if (warp_id == 0) {
                    if (elect_sync()) {
                        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                        mbarrier_arrive(x_full_addr);
                    }
                }
                #pragma unroll 1
                for (int iter_k = 0; iter_k < K_tiles; iter_k++) {
                    int off_k = iter_k * 256;
                    int c_row_base = batch * K + off_k;
                    mbarrier_wait(c_empty_addr, _phase_c_empty_0);
                    _phase_c_empty_0 ^= 1;
                    #pragma unroll 1
                    for (unsigned int vec_idx = lane_start; vec_idx < c_vecs; vec_idx += 32) {
                        int d_pad = vec_idx % 8 * 8;
                        int row = vec_idx / 8;
                        float vals[8];
                        unsigned int packed[4];
                        if (d_pad < D) {
                            {
                                const uint4* _vptr_1 = reinterpret_cast<const uint4*>(centroids + (c_row_base + row) * D + d_pad);
                                uint4 _vld_1[1];
                                #pragma unroll
                                for (int _blk = 0; _blk < 1; _blk++) {
                                    _vld_1[_blk] = _vptr_1[_blk];
                                    __nv_bfloat16* _velems_1 = reinterpret_cast<__nv_bfloat16*>(&_vld_1[_blk]);
                                    #pragma unroll
                                    for (int _j = 0; _j < 8; _j++)
                                        vals[0 + _blk * 8 + _j] = __bfloat162float(_velems_1[_j]);
                                }
                            }
                        } else {
                            #pragma unroll
                            for (int vi = 0; vi < 8; vi++) {
                                vals[vi] = 0.0f;
                            }
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 4; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(vals[_lp*2 + 0], vals[_lp*2+1 + 0]));
                            packed[_lp] = *(uint32_t*)&_bf2;
                        }
                        int c_addr = (smem_c_addr + (d_pad / 64 * 32768 + row * 128 + d_pad % 64 * 2 ^ (d_pad / 64 * 32768 + row * 128 + d_pad % 64 * 2 >> 7 & 7) << 4));
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(c_addr), "r"(packed[0]), "r"(packed[1]), "r"(packed[2]), "r"(packed[3]) : "memory");
                    }
                    asm volatile("barrier.sync 8, 32;");
                    if (warp_id == 0) {
                        if (elect_sync()) {
                            asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                            mbarrier_arrive(c_full_addr);
                        }
                    }
                }
            }
        }
    // ---- Role: mma ----
    } else if (warp == 1) {
        const int tmem_score_tmem = taddr + TMEM_SCORE_TMEM_OFFSET;
        { // mma_main
            int num_tiles = B * num_n_tiles;
            uint32_t _phase_x_full_0 = 0;
            uint32_t _phase_score_empty_0 = 1;
            uint32_t _phase_c_full_0 = 0;
            #pragma unroll 1
            for (unsigned int _tile_idx = bid; _tile_idx < num_tiles; _tile_idx += num_bids) {
                mbarrier_wait(x_full_addr, _phase_x_full_0);
                _phase_x_full_0 ^= 1;
                #pragma unroll 1
                for (int _iter_k = 0; _iter_k < K_tiles; _iter_k++) {
                    mbarrier_wait(score_empty_addr, _phase_score_empty_0);
                    _phase_score_empty_0 ^= 1;
                    mbarrier_wait(c_full_addr, _phase_c_full_0);
                    _phase_c_full_0 ^= 1;
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    int _mma_ss_a_addr_0 = smem_x_addr + 0 * 16384;
                    int _mma_ss_a_lo_0 = make_warp_uniform((_mma_ss_a_addr_0 >> 4) & 0x3FFF);
                    int _mma_ss_b_addr_0 = smem_c_addr + 0 * 32768;
                    int _mma_ss_b_lo_0 = make_warp_uniform((_mma_ss_b_addr_0 >> 4) & 0x3FFF);
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
                    "mov.b32 id, 138413200;\n\t"
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
                    :: "r"(_mma_ss_a_lo_0), "r"(_mma_ss_b_lo_0), "r"(tmem_score_tmem), "r"(0));
                    elect_commit(score_full_addr);
                    elect_commit(c_empty_addr);
                }
                elect_commit(x_empty_addr);
            }
        }
    // ---- Role: compute ----
    } else if (warp >= 2 && warp <= 5) {
        const int warp_id_in_wg = warp % 4;
        const int my_row = warp_id_in_wg * 32 + lane;
        const int tmem_row_base_v = (warp_id_in_wg * 32) << 16;
        { // compute_main
            int num_tiles = B * num_n_tiles;
            uint32_t _phase_score_full_0 = 0;
            #pragma unroll 1
            for (unsigned int tile_idx = bid; tile_idx < num_tiles; tile_idx += num_bids) {
                int batch = tile_idx / num_n_tiles;
                int n_tile = tile_idx % num_n_tiles;
                int off_n = n_tile * 128;
                int global_n = off_n + my_row;
                int out_offset = batch * N + global_n;
                int csq_smem_addr = smem_csq_addr + 0;
                float best_score = -3.4e+38f;
                int best_idx = 0;
                #pragma unroll 1
                for (int iter_k = 0; iter_k < K_tiles; iter_k++) {
                    int off_k = iter_k * 256;
                    if (my_row < 64.0f) {
                        int csq_base = my_row * 4;
                        float csq_pack[4];
                        {
                            float4 _v4 = *reinterpret_cast<const float4*>(c_sq + batch * K + off_k + csq_base);
                            csq_pack[0 + 0] = _v4.x;
                            csq_pack[0 + 1] = _v4.y;
                            csq_pack[0 + 2] = _v4.z;
                            csq_pack[0 + 3] = _v4.w;
                        }
                        float csq_half[4];
                        csq_half[0] = 0.5f * csq_pack[0];
                        csq_half[1] = 0.5f * csq_pack[1];
                        csq_half[2] = 0.5f * csq_pack[2];
                        csq_half[3] = 0.5f * csq_pack[3];
                        asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(csq_smem_addr + csq_base * 4), "f"(csq_half[0]), "f"(csq_half[1]), "f"(csq_half[2]), "f"(csq_half[3]) : "memory");
                    }
                    asm volatile("barrier.sync 9, 128;");
                    mbarrier_wait(score_full_addr, _phase_score_full_0);
                    _phase_score_full_0 ^= 1;
                    int score_base = 0;
                    float scores[128];
                    asm volatile(
                        "tcgen05.ld.sync.aligned.32x32b.x128.b32"
                        " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63, %64, %65, %66, %67, %68, %69, %70, %71, %72, %73, %74, %75, %76, %77, %78, %79, %80, %81, %82, %83, %84, %85, %86, %87, %88, %89, %90, %91, %92, %93, %94, %95, %96, %97, %98, %99, %100, %101, %102, %103, %104, %105, %106, %107, %108, %109, %110, %111, %112, %113, %114, %115, %116, %117, %118, %119, %120, %121, %122, %123, %124, %125, %126, %127}, [%128];"
                        : "=f"(scores[0]), "=f"(scores[1]), "=f"(scores[2]), "=f"(scores[3]), "=f"(scores[4]), "=f"(scores[5]), "=f"(scores[6]), "=f"(scores[7]), "=f"(scores[8]), "=f"(scores[9]), "=f"(scores[10]), "=f"(scores[11]), "=f"(scores[12]), "=f"(scores[13]), "=f"(scores[14]), "=f"(scores[15]), "=f"(scores[16]), "=f"(scores[17]), "=f"(scores[18]), "=f"(scores[19]), "=f"(scores[20]), "=f"(scores[21]), "=f"(scores[22]), "=f"(scores[23]), "=f"(scores[24]), "=f"(scores[25]), "=f"(scores[26]), "=f"(scores[27]), "=f"(scores[28]), "=f"(scores[29]), "=f"(scores[30]), "=f"(scores[31]), "=f"(scores[32]), "=f"(scores[33]), "=f"(scores[34]), "=f"(scores[35]), "=f"(scores[36]), "=f"(scores[37]), "=f"(scores[38]), "=f"(scores[39]), "=f"(scores[40]), "=f"(scores[41]), "=f"(scores[42]), "=f"(scores[43]), "=f"(scores[44]), "=f"(scores[45]), "=f"(scores[46]), "=f"(scores[47]), "=f"(scores[48]), "=f"(scores[49]), "=f"(scores[50]), "=f"(scores[51]), "=f"(scores[52]), "=f"(scores[53]), "=f"(scores[54]), "=f"(scores[55]), "=f"(scores[56]), "=f"(scores[57]), "=f"(scores[58]), "=f"(scores[59]), "=f"(scores[60]), "=f"(scores[61]), "=f"(scores[62]), "=f"(scores[63]), "=f"(scores[64]), "=f"(scores[65]), "=f"(scores[66]), "=f"(scores[67]), "=f"(scores[68]), "=f"(scores[69]), "=f"(scores[70]), "=f"(scores[71]), "=f"(scores[72]), "=f"(scores[73]), "=f"(scores[74]), "=f"(scores[75]), "=f"(scores[76]), "=f"(scores[77]), "=f"(scores[78]), "=f"(scores[79]), "=f"(scores[80]), "=f"(scores[81]), "=f"(scores[82]), "=f"(scores[83]), "=f"(scores[84]), "=f"(scores[85]), "=f"(scores[86]), "=f"(scores[87]), "=f"(scores[88]), "=f"(scores[89]), "=f"(scores[90]), "=f"(scores[91]), "=f"(scores[92]), "=f"(scores[93]), "=f"(scores[94]), "=f"(scores[95]), "=f"(scores[96]), "=f"(scores[97]), "=f"(scores[98]), "=f"(scores[99]), "=f"(scores[100]), "=f"(scores[101]), "=f"(scores[102]), "=f"(scores[103]), "=f"(scores[104]), "=f"(scores[105]), "=f"(scores[106]), "=f"(scores[107]), "=f"(scores[108]), "=f"(scores[109]), "=f"(scores[110]), "=f"(scores[111]), "=f"(scores[112]), "=f"(scores[113]), "=f"(scores[114]), "=f"(scores[115]), "=f"(scores[116]), "=f"(scores[117]), "=f"(scores[118]), "=f"(scores[119]), "=f"(scores[120]), "=f"(scores[121]), "=f"(scores[122]), "=f"(scores[123]), "=f"(scores[124]), "=f"(scores[125]), "=f"(scores[126]), "=f"(scores[127])
                        : "r"(taddr + tmem_row_base_v + score_base)
                        : "memory");
                    asm volatile("tcgen05.wait::ld.sync.aligned;");
                    #pragma unroll
                    for (int kk = 0; kk < 128; kk += 4) {
                        float csq_vals[4];
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&csq_vals[0])), "=r"(*reinterpret_cast<uint32_t*>(&csq_vals[1])), "=r"(*reinterpret_cast<uint32_t*>(&csq_vals[2])), "=r"(*reinterpret_cast<uint32_t*>(&csq_vals[3]))
                            : "r"(csq_smem_addr + (score_base + kk) * 4));
                        float d0 = scores[kk] - csq_vals[0];
                        float d1 = scores[kk + 1] - csq_vals[1];
                        float best_01 = d0;
                        int idx_01 = off_k + score_base + kk;
                        if (d1 > best_01) {
                            best_01 = d1;
                            idx_01 = off_k + score_base + kk + 1;
                        }
                        float d2 = scores[kk + 2] - csq_vals[2];
                        float d3 = scores[kk + 3] - csq_vals[3];
                        float best_23 = d2;
                        int idx_23 = off_k + score_base + kk + 2;
                        if (d3 > best_23) {
                            best_23 = d3;
                            idx_23 = off_k + score_base + kk + 3;
                        }
                        float best_group = best_01;
                        int idx_group = idx_01;
                        if (best_23 > best_group) {
                            best_group = best_23;
                            idx_group = idx_23;
                        }
                        if (best_group > best_score) {
                            best_score = best_group;
                            best_idx = idx_group;
                        }
                    }
                    score_base = 128;
                    asm volatile(
                        "tcgen05.ld.sync.aligned.32x32b.x128.b32"
                        " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63, %64, %65, %66, %67, %68, %69, %70, %71, %72, %73, %74, %75, %76, %77, %78, %79, %80, %81, %82, %83, %84, %85, %86, %87, %88, %89, %90, %91, %92, %93, %94, %95, %96, %97, %98, %99, %100, %101, %102, %103, %104, %105, %106, %107, %108, %109, %110, %111, %112, %113, %114, %115, %116, %117, %118, %119, %120, %121, %122, %123, %124, %125, %126, %127}, [%128];"
                        : "=f"(scores[0]), "=f"(scores[1]), "=f"(scores[2]), "=f"(scores[3]), "=f"(scores[4]), "=f"(scores[5]), "=f"(scores[6]), "=f"(scores[7]), "=f"(scores[8]), "=f"(scores[9]), "=f"(scores[10]), "=f"(scores[11]), "=f"(scores[12]), "=f"(scores[13]), "=f"(scores[14]), "=f"(scores[15]), "=f"(scores[16]), "=f"(scores[17]), "=f"(scores[18]), "=f"(scores[19]), "=f"(scores[20]), "=f"(scores[21]), "=f"(scores[22]), "=f"(scores[23]), "=f"(scores[24]), "=f"(scores[25]), "=f"(scores[26]), "=f"(scores[27]), "=f"(scores[28]), "=f"(scores[29]), "=f"(scores[30]), "=f"(scores[31]), "=f"(scores[32]), "=f"(scores[33]), "=f"(scores[34]), "=f"(scores[35]), "=f"(scores[36]), "=f"(scores[37]), "=f"(scores[38]), "=f"(scores[39]), "=f"(scores[40]), "=f"(scores[41]), "=f"(scores[42]), "=f"(scores[43]), "=f"(scores[44]), "=f"(scores[45]), "=f"(scores[46]), "=f"(scores[47]), "=f"(scores[48]), "=f"(scores[49]), "=f"(scores[50]), "=f"(scores[51]), "=f"(scores[52]), "=f"(scores[53]), "=f"(scores[54]), "=f"(scores[55]), "=f"(scores[56]), "=f"(scores[57]), "=f"(scores[58]), "=f"(scores[59]), "=f"(scores[60]), "=f"(scores[61]), "=f"(scores[62]), "=f"(scores[63]), "=f"(scores[64]), "=f"(scores[65]), "=f"(scores[66]), "=f"(scores[67]), "=f"(scores[68]), "=f"(scores[69]), "=f"(scores[70]), "=f"(scores[71]), "=f"(scores[72]), "=f"(scores[73]), "=f"(scores[74]), "=f"(scores[75]), "=f"(scores[76]), "=f"(scores[77]), "=f"(scores[78]), "=f"(scores[79]), "=f"(scores[80]), "=f"(scores[81]), "=f"(scores[82]), "=f"(scores[83]), "=f"(scores[84]), "=f"(scores[85]), "=f"(scores[86]), "=f"(scores[87]), "=f"(scores[88]), "=f"(scores[89]), "=f"(scores[90]), "=f"(scores[91]), "=f"(scores[92]), "=f"(scores[93]), "=f"(scores[94]), "=f"(scores[95]), "=f"(scores[96]), "=f"(scores[97]), "=f"(scores[98]), "=f"(scores[99]), "=f"(scores[100]), "=f"(scores[101]), "=f"(scores[102]), "=f"(scores[103]), "=f"(scores[104]), "=f"(scores[105]), "=f"(scores[106]), "=f"(scores[107]), "=f"(scores[108]), "=f"(scores[109]), "=f"(scores[110]), "=f"(scores[111]), "=f"(scores[112]), "=f"(scores[113]), "=f"(scores[114]), "=f"(scores[115]), "=f"(scores[116]), "=f"(scores[117]), "=f"(scores[118]), "=f"(scores[119]), "=f"(scores[120]), "=f"(scores[121]), "=f"(scores[122]), "=f"(scores[123]), "=f"(scores[124]), "=f"(scores[125]), "=f"(scores[126]), "=f"(scores[127])
                        : "r"(taddr + tmem_row_base_v + score_base)
                        : "memory");
                    asm volatile("tcgen05.wait::ld.sync.aligned;");
                    asm volatile("barrier.sync 10, 128;");
                    if (elect_sync()) {
                        mbarrier_arrive(score_empty_addr);
                    }
                    #pragma unroll
                    for (int kk = 0; kk < 128; kk += 4) {
                        float csq_vals[4];
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&csq_vals[0])), "=r"(*reinterpret_cast<uint32_t*>(&csq_vals[1])), "=r"(*reinterpret_cast<uint32_t*>(&csq_vals[2])), "=r"(*reinterpret_cast<uint32_t*>(&csq_vals[3]))
                            : "r"(csq_smem_addr + (score_base + kk) * 4));
                        float d0 = scores[kk] - csq_vals[0];
                        float d1 = scores[kk + 1] - csq_vals[1];
                        float best_01 = d0;
                        int idx_01 = off_k + score_base + kk;
                        if (d1 > best_01) {
                            best_01 = d1;
                            idx_01 = off_k + score_base + kk + 1;
                        }
                        float d2 = scores[kk + 2] - csq_vals[2];
                        float d3 = scores[kk + 3] - csq_vals[3];
                        float best_23 = d2;
                        int idx_23 = off_k + score_base + kk + 2;
                        if (d3 > best_23) {
                            best_23 = d3;
                            idx_23 = off_k + score_base + kk + 3;
                        }
                        float best_group = best_01;
                        int idx_group = idx_01;
                        if (best_23 > best_group) {
                            best_group = best_23;
                            idx_group = idx_23;
                        }
                        if (best_group > best_score) {
                            best_score = best_group;
                            best_idx = idx_group;
                        }
                    }
                }
                if (global_n < N) {
                    *((int*)(out + out_offset)) = best_idx;
                }
            }
        }
    }

    // Cleanup
    __syncthreads(); // barrier before TMEM dealloc

    if (warp == 1) {
        asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;" :: "r"(tmem_addr_storage[0]), "r"(256));
        asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;");
    }
}

} // extern "C"

#undef NUM_MAIN_STAGES
#undef SMEM_SMEM_CSQ_OFF
#undef SMEM_SMEM_CSQ_STAGE_BYTES
#undef SMEM_SMEM_CSQ_STRIDE
#undef SMEM_SMEM_C_OFF
#undef SMEM_SMEM_C_STAGE_BYTES
#undef SMEM_SMEM_C_STRIDE
#undef SMEM_SMEM_X_OFF
#undef SMEM_SMEM_X_STAGE_BYTES
#undef SMEM_SMEM_X_STRIDE
#undef SMEM_TOTAL
#undef TMEM_NCOLS
#undef TMEM_SCORE_TMEM_OFFSET
#undef c_empty_addr
#undef c_full_addr
#undef score_empty_addr
#undef score_full_addr
#undef smem_c_addr
#undef smem_csq_addr
#undef smem_x_addr
#undef x_empty_addr
#undef x_full_addr

#define NUM_MAIN_STAGES 1
#define THREADS 256

extern "C" {

__global__ __launch_bounds__(256) void
kernel_flash_kmeans_assign_microdim_pack_6cd2_v1(__nv_bfloat16* __restrict__ x, __nv_bfloat16* __restrict__ centroids, __nv_bfloat16* __restrict__ x_pad, __nv_bfloat16* __restrict__ c_pad, int B, int N, int D, int K, int total_x_pad, int total_c_pad)
{
    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;


    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    const int warp_id = warp;
    const int lane_id = lane;

    // === Task calls (dependency order) ===
    int grid_stride = num_bids * 256;
    int start = bid * 256 + tid;
    int total_x_vecs = total_x_pad / 8;
    int total_c_vecs = total_c_pad / 8;
    #pragma unroll 1
    for (unsigned int vec_idx = start; vec_idx < total_x_vecs; vec_idx += grid_stride) {
        int d_pad = vec_idx % 8 * 8;
        int row = vec_idx / 8;
        int dst_base = vec_idx * 8;
        if (d_pad < D) {
            float vals[8];
            {
                const uint4* _vptr_0 = reinterpret_cast<const uint4*>(x + row * D + d_pad);
                uint4 _vld_0[1];
                #pragma unroll
                for (int _blk = 0; _blk < 1; _blk++) {
                    _vld_0[_blk] = _vptr_0[_blk];
                    __nv_bfloat16* _velems_0 = reinterpret_cast<__nv_bfloat16*>(&_vld_0[_blk]);
                    #pragma unroll
                    for (int _j = 0; _j < 8; _j++)
                        vals[0 + _blk * 8 + _j] = __bfloat162float(_velems_0[_j]);
                }
            }
            {
                __nv_bfloat162 _pk[4];
                _pk[0] = __floats2bfloat162_rn(vals[0 + 0], vals[0 + 1]);
                _pk[1] = __floats2bfloat162_rn(vals[0 + 2], vals[0 + 3]);
                _pk[2] = __floats2bfloat162_rn(vals[0 + 4], vals[0 + 5]);
                _pk[3] = __floats2bfloat162_rn(vals[0 + 6], vals[0 + 7]);
                *reinterpret_cast<uint4*>(&((__nv_bfloat16*)(x_pad + dst_base))[0]) = *reinterpret_cast<uint4*>(&_pk[0]);
            }
        } else {
            float zeros[8];
            zeros[0] = 0.0f;
            zeros[1] = 0.0f;
            zeros[2] = 0.0f;
            zeros[3] = 0.0f;
            zeros[4] = 0.0f;
            zeros[5] = 0.0f;
            zeros[6] = 0.0f;
            zeros[7] = 0.0f;
            {
                __nv_bfloat162 _pk[4];
                _pk[0] = __floats2bfloat162_rn(zeros[0 + 0], zeros[0 + 1]);
                _pk[1] = __floats2bfloat162_rn(zeros[0 + 2], zeros[0 + 3]);
                _pk[2] = __floats2bfloat162_rn(zeros[0 + 4], zeros[0 + 5]);
                _pk[3] = __floats2bfloat162_rn(zeros[0 + 6], zeros[0 + 7]);
                *reinterpret_cast<uint4*>(&((__nv_bfloat16*)(x_pad + dst_base))[0]) = *reinterpret_cast<uint4*>(&_pk[0]);
            }
        }
    }
    #pragma unroll 1
    for (unsigned int vec_idx = start; vec_idx < total_c_vecs; vec_idx += grid_stride) {
        int d_pad = vec_idx % 8 * 8;
        int row = vec_idx / 8;
        int dst_base = vec_idx * 8;
        if (d_pad < D) {
            float vals[8];
            {
                const uint4* _vptr_1 = reinterpret_cast<const uint4*>(centroids + row * D + d_pad);
                uint4 _vld_1[1];
                #pragma unroll
                for (int _blk = 0; _blk < 1; _blk++) {
                    _vld_1[_blk] = _vptr_1[_blk];
                    __nv_bfloat16* _velems_1 = reinterpret_cast<__nv_bfloat16*>(&_vld_1[_blk]);
                    #pragma unroll
                    for (int _j = 0; _j < 8; _j++)
                        vals[0 + _blk * 8 + _j] = __bfloat162float(_velems_1[_j]);
                }
            }
            {
                __nv_bfloat162 _pk[4];
                _pk[0] = __floats2bfloat162_rn(vals[0 + 0], vals[0 + 1]);
                _pk[1] = __floats2bfloat162_rn(vals[0 + 2], vals[0 + 3]);
                _pk[2] = __floats2bfloat162_rn(vals[0 + 4], vals[0 + 5]);
                _pk[3] = __floats2bfloat162_rn(vals[0 + 6], vals[0 + 7]);
                *reinterpret_cast<uint4*>(&((__nv_bfloat16*)(c_pad + dst_base))[0]) = *reinterpret_cast<uint4*>(&_pk[0]);
            }
        } else {
            float zeros[8];
            zeros[0] = 0.0f;
            zeros[1] = 0.0f;
            zeros[2] = 0.0f;
            zeros[3] = 0.0f;
            zeros[4] = 0.0f;
            zeros[5] = 0.0f;
            zeros[6] = 0.0f;
            zeros[7] = 0.0f;
            {
                __nv_bfloat162 _pk[4];
                _pk[0] = __floats2bfloat162_rn(zeros[0 + 0], zeros[0 + 1]);
                _pk[1] = __floats2bfloat162_rn(zeros[0 + 2], zeros[0 + 3]);
                _pk[2] = __floats2bfloat162_rn(zeros[0 + 4], zeros[0 + 5]);
                _pk[3] = __floats2bfloat162_rn(zeros[0 + 6], zeros[0 + 7]);
                *reinterpret_cast<uint4*>(&((__nv_bfloat16*)(c_pad + dst_base))[0]) = *reinterpret_cast<uint4*>(&_pk[0]);
            }
        }
    }
}

} // extern "C"

#undef NUM_MAIN_STAGES
#undef THREADS

#define TMEM_NCOLS 256
#define TMEM_SCORE_TMEM_OFFSET 0
#define NUM_MAIN_STAGES 1
#define SMEM_SMEM_X_OFF 1024
#define SMEM_SMEM_X_STAGE_BYTES 16384
#define SMEM_SMEM_X_STRIDE 16384
#define SMEM_SMEM_C_OFF 17408
#define SMEM_SMEM_C_STAGE_BYTES 32768
#define SMEM_SMEM_C_STRIDE 32768
#define SMEM_SMEM_CSQ_OFF 50176
#define SMEM_SMEM_CSQ_STAGE_BYTES 1024
#define SMEM_SMEM_CSQ_STRIDE 1024
#define SMEM_TOTAL 51200

extern "C" {

__global__ __launch_bounds__(192) void
kernel_flash_kmeans_assign_microdim_6cd2_v1(float* __restrict__ x_sq, float* __restrict__ c_sq, int32_t* __restrict__ out, const void* x_tmap, const void* c_tmap, int B, int N, int D, int K, int num_n_tiles, int K_tiles)
{
    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;

    extern __shared__ __align__(1024) char smem_raw[];
    int smem;
    smem = (int)(unsigned long long)__cvta_generic_to_shared(smem_raw);
    const int smem_smem_x = smem + 1024;
    const int smem_smem_c = smem + 17408;
    const int smem_smem_csq = smem + 50176;

    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    // Mbarrier init (6 groups, 6 barriers)
    // Mbarriers at smem_raw[0..48)

    if (warp == 0) {
        uint32_t leader = elect_sync();
        // x_full: 1 barriers, init_count=1
        mbarrier_init_pred(smem + 0, 1, leader);
        // x_empty: 1 barriers, init_count=1
        mbarrier_init_pred(smem + 8, 1, leader);
        // c_full: 1 barriers, init_count=1
        mbarrier_init_pred(smem + 16, 1, leader);
        // c_empty: 1 barriers, init_count=1
        mbarrier_init_pred(smem + 24, 1, leader);
        // score_full: 1 barriers, init_count=1
        mbarrier_init_pred(smem + 32, 1, leader);
        // score_empty: 1 barriers, init_count=4
        mbarrier_init_pred(smem + 40, 4, leader);
        asm volatile("fence.mbarrier_init.release.cluster;");
    }

    __syncthreads();

    // TMEM alloc (256 columns, 256 used)
    volatile int* tmem_addr_storage = (volatile int*)(smem_raw + 48);
    if (warp == 1) {
        int _tmem_hold = smem + 48;
        asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;" :: "r"(_tmem_hold), "r"(256) : "memory");
    }

    __syncthreads();
    asm volatile("tcgen05.fence::after_thread_sync;");

    const int warp_id = warp;
    const int lane_id = lane;
    __nv_bfloat16* smem_x = (__nv_bfloat16*)(smem_raw + 1024);
    #define smem_x_addr (smem + 1024)
    __nv_bfloat16* smem_c = (__nv_bfloat16*)(smem_raw + 17408);
    #define smem_c_addr (smem + 17408)
    float* smem_csq = (float*)(smem_raw + 50176);
    #define smem_csq_addr (smem + 50176)
    const int mbar_base = smem;
    #define x_full_addr (mbar_base + 0)
    #define x_empty_addr (mbar_base + 8)
    #define c_full_addr (mbar_base + 16)
    #define c_empty_addr (mbar_base + 24)
    #define score_full_addr (mbar_base + 32)
    #define score_empty_addr (mbar_base + 40)
    const int taddr = tmem_addr_storage[0];

    // ---- Role: load ----
    if (warp == 0) {
        { // load_main
            int num_tiles = B * num_n_tiles;
            uint32_t _phase_x_empty_0 = 1;
            uint32_t _phase_c_empty_0 = 1;
            if (warp_id == 0) {
                if (elect_sync()) {
                    #pragma unroll 1
                    for (unsigned int tile_idx = bid; tile_idx < num_tiles; tile_idx += num_bids) {
                        int batch = tile_idx / num_n_tiles;
                        int n_tile = tile_idx % num_n_tiles;
                        int off_n = n_tile * 128;
                        int x_row = batch * N + off_n;
                        mbarrier_wait(x_empty_addr, _phase_x_empty_0);
                        _phase_x_empty_0 ^= 1;
                        tma_3d_gmem2smem(smem_x_addr, x_tmap, 0, x_row, 0, x_full_addr);
                        mbarrier_arrive_expect_tx(x_full_addr, 16384);
                        #pragma unroll 1
                        for (int iter_k = 0; iter_k < K_tiles; iter_k++) {
                            int off_k = iter_k * 256;
                            int c_row = batch * K + off_k;
                            mbarrier_wait(c_empty_addr, _phase_c_empty_0);
                            _phase_c_empty_0 ^= 1;
                            tma_3d_gmem2smem(smem_c_addr, c_tmap, 0, c_row, 0, c_full_addr);
                            mbarrier_arrive_expect_tx(c_full_addr, 32768);
                        }
                    }
                }
            }
        }
    // ---- Role: mma ----
    } else if (warp == 1) {
        const int tmem_score_tmem = taddr + TMEM_SCORE_TMEM_OFFSET;
        { // mma_main
            int num_tiles = B * num_n_tiles;
            uint32_t _phase_x_full_0 = 0;
            uint32_t _phase_score_empty_0 = 1;
            uint32_t _phase_c_full_0 = 0;
            #pragma unroll 1
            for (unsigned int _tile_idx = bid; _tile_idx < num_tiles; _tile_idx += num_bids) {
                mbarrier_wait(x_full_addr, _phase_x_full_0);
                _phase_x_full_0 ^= 1;
                #pragma unroll 1
                for (int _iter_k = 0; _iter_k < K_tiles; _iter_k++) {
                    mbarrier_wait(score_empty_addr, _phase_score_empty_0);
                    _phase_score_empty_0 ^= 1;
                    mbarrier_wait(c_full_addr, _phase_c_full_0);
                    _phase_c_full_0 ^= 1;
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    int _mma_ss_a_addr_0 = smem_x_addr + 0 * 16384;
                    int _mma_ss_a_lo_0 = make_warp_uniform((_mma_ss_a_addr_0 >> 4) & 0x3FFF);
                    int _mma_ss_b_addr_0 = smem_c_addr + 0 * 32768;
                    int _mma_ss_b_lo_0 = make_warp_uniform((_mma_ss_b_addr_0 >> 4) & 0x3FFF);
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
                    "mov.b32 id, 138413200;\n\t"
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
                    :: "r"(_mma_ss_a_lo_0), "r"(_mma_ss_b_lo_0), "r"(tmem_score_tmem), "r"(0));
                    elect_commit(score_full_addr);
                    elect_commit(c_empty_addr);
                }
                elect_commit(x_empty_addr);
            }
        }
    // ---- Role: compute ----
    } else if (warp >= 2 && warp <= 5) {
        const int warp_id_in_wg = warp % 4;
        const int my_row = warp_id_in_wg * 32 + lane;
        const int tmem_row_base_v = (warp_id_in_wg * 32) << 16;
        { // compute_main
            int num_tiles = B * num_n_tiles;
            uint32_t _phase_score_full_0 = 0;
            #pragma unroll 1
            for (unsigned int tile_idx = bid; tile_idx < num_tiles; tile_idx += num_bids) {
                int batch = tile_idx / num_n_tiles;
                int n_tile = tile_idx % num_n_tiles;
                int off_n = n_tile * 128;
                int global_n = off_n + my_row;
                int out_offset = batch * N + global_n;
                int csq_smem_addr = smem_csq_addr + 0;
                float best_score = -3.4e+38f;
                int best_idx = 0;
                #pragma unroll 1
                for (int iter_k = 0; iter_k < K_tiles; iter_k++) {
                    int off_k = iter_k * 256;
                    if (my_row < 64.0f) {
                        int csq_base = my_row * 4;
                        float csq_pack[4];
                        {
                            float4 _v4 = *reinterpret_cast<const float4*>(c_sq + batch * K + off_k + csq_base);
                            csq_pack[0 + 0] = _v4.x;
                            csq_pack[0 + 1] = _v4.y;
                            csq_pack[0 + 2] = _v4.z;
                            csq_pack[0 + 3] = _v4.w;
                        }
                        float csq_half[4];
                        csq_half[0] = 0.5f * csq_pack[0];
                        csq_half[1] = 0.5f * csq_pack[1];
                        csq_half[2] = 0.5f * csq_pack[2];
                        csq_half[3] = 0.5f * csq_pack[3];
                        asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(csq_smem_addr + csq_base * 4), "f"(csq_half[0]), "f"(csq_half[1]), "f"(csq_half[2]), "f"(csq_half[3]) : "memory");
                    }
                    asm volatile("barrier.sync 8, 128;");
                    mbarrier_wait(score_full_addr, _phase_score_full_0);
                    _phase_score_full_0 ^= 1;
                    int score_base = 0;
                    float scores[128];
                    asm volatile(
                        "tcgen05.ld.sync.aligned.32x32b.x128.b32"
                        " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63, %64, %65, %66, %67, %68, %69, %70, %71, %72, %73, %74, %75, %76, %77, %78, %79, %80, %81, %82, %83, %84, %85, %86, %87, %88, %89, %90, %91, %92, %93, %94, %95, %96, %97, %98, %99, %100, %101, %102, %103, %104, %105, %106, %107, %108, %109, %110, %111, %112, %113, %114, %115, %116, %117, %118, %119, %120, %121, %122, %123, %124, %125, %126, %127}, [%128];"
                        : "=f"(scores[0]), "=f"(scores[1]), "=f"(scores[2]), "=f"(scores[3]), "=f"(scores[4]), "=f"(scores[5]), "=f"(scores[6]), "=f"(scores[7]), "=f"(scores[8]), "=f"(scores[9]), "=f"(scores[10]), "=f"(scores[11]), "=f"(scores[12]), "=f"(scores[13]), "=f"(scores[14]), "=f"(scores[15]), "=f"(scores[16]), "=f"(scores[17]), "=f"(scores[18]), "=f"(scores[19]), "=f"(scores[20]), "=f"(scores[21]), "=f"(scores[22]), "=f"(scores[23]), "=f"(scores[24]), "=f"(scores[25]), "=f"(scores[26]), "=f"(scores[27]), "=f"(scores[28]), "=f"(scores[29]), "=f"(scores[30]), "=f"(scores[31]), "=f"(scores[32]), "=f"(scores[33]), "=f"(scores[34]), "=f"(scores[35]), "=f"(scores[36]), "=f"(scores[37]), "=f"(scores[38]), "=f"(scores[39]), "=f"(scores[40]), "=f"(scores[41]), "=f"(scores[42]), "=f"(scores[43]), "=f"(scores[44]), "=f"(scores[45]), "=f"(scores[46]), "=f"(scores[47]), "=f"(scores[48]), "=f"(scores[49]), "=f"(scores[50]), "=f"(scores[51]), "=f"(scores[52]), "=f"(scores[53]), "=f"(scores[54]), "=f"(scores[55]), "=f"(scores[56]), "=f"(scores[57]), "=f"(scores[58]), "=f"(scores[59]), "=f"(scores[60]), "=f"(scores[61]), "=f"(scores[62]), "=f"(scores[63]), "=f"(scores[64]), "=f"(scores[65]), "=f"(scores[66]), "=f"(scores[67]), "=f"(scores[68]), "=f"(scores[69]), "=f"(scores[70]), "=f"(scores[71]), "=f"(scores[72]), "=f"(scores[73]), "=f"(scores[74]), "=f"(scores[75]), "=f"(scores[76]), "=f"(scores[77]), "=f"(scores[78]), "=f"(scores[79]), "=f"(scores[80]), "=f"(scores[81]), "=f"(scores[82]), "=f"(scores[83]), "=f"(scores[84]), "=f"(scores[85]), "=f"(scores[86]), "=f"(scores[87]), "=f"(scores[88]), "=f"(scores[89]), "=f"(scores[90]), "=f"(scores[91]), "=f"(scores[92]), "=f"(scores[93]), "=f"(scores[94]), "=f"(scores[95]), "=f"(scores[96]), "=f"(scores[97]), "=f"(scores[98]), "=f"(scores[99]), "=f"(scores[100]), "=f"(scores[101]), "=f"(scores[102]), "=f"(scores[103]), "=f"(scores[104]), "=f"(scores[105]), "=f"(scores[106]), "=f"(scores[107]), "=f"(scores[108]), "=f"(scores[109]), "=f"(scores[110]), "=f"(scores[111]), "=f"(scores[112]), "=f"(scores[113]), "=f"(scores[114]), "=f"(scores[115]), "=f"(scores[116]), "=f"(scores[117]), "=f"(scores[118]), "=f"(scores[119]), "=f"(scores[120]), "=f"(scores[121]), "=f"(scores[122]), "=f"(scores[123]), "=f"(scores[124]), "=f"(scores[125]), "=f"(scores[126]), "=f"(scores[127])
                        : "r"(taddr + tmem_row_base_v + score_base)
                        : "memory");
                    asm volatile("tcgen05.wait::ld.sync.aligned;");
                    #pragma unroll
                    for (int kk = 0; kk < 128; kk += 4) {
                        float csq_vals[4];
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&csq_vals[0])), "=r"(*reinterpret_cast<uint32_t*>(&csq_vals[1])), "=r"(*reinterpret_cast<uint32_t*>(&csq_vals[2])), "=r"(*reinterpret_cast<uint32_t*>(&csq_vals[3]))
                            : "r"(csq_smem_addr + (score_base + kk) * 4));
                        float d0 = scores[kk] - csq_vals[0];
                        float d1 = scores[kk + 1] - csq_vals[1];
                        float best_01 = d0;
                        int idx_01 = off_k + score_base + kk;
                        if (d1 > best_01) {
                            best_01 = d1;
                            idx_01 = off_k + score_base + kk + 1;
                        }
                        float d2 = scores[kk + 2] - csq_vals[2];
                        float d3 = scores[kk + 3] - csq_vals[3];
                        float best_23 = d2;
                        int idx_23 = off_k + score_base + kk + 2;
                        if (d3 > best_23) {
                            best_23 = d3;
                            idx_23 = off_k + score_base + kk + 3;
                        }
                        float best_group = best_01;
                        int idx_group = idx_01;
                        if (best_23 > best_group) {
                            best_group = best_23;
                            idx_group = idx_23;
                        }
                        if (best_group > best_score) {
                            best_score = best_group;
                            best_idx = idx_group;
                        }
                    }
                    score_base = 128;
                    asm volatile(
                        "tcgen05.ld.sync.aligned.32x32b.x128.b32"
                        " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63, %64, %65, %66, %67, %68, %69, %70, %71, %72, %73, %74, %75, %76, %77, %78, %79, %80, %81, %82, %83, %84, %85, %86, %87, %88, %89, %90, %91, %92, %93, %94, %95, %96, %97, %98, %99, %100, %101, %102, %103, %104, %105, %106, %107, %108, %109, %110, %111, %112, %113, %114, %115, %116, %117, %118, %119, %120, %121, %122, %123, %124, %125, %126, %127}, [%128];"
                        : "=f"(scores[0]), "=f"(scores[1]), "=f"(scores[2]), "=f"(scores[3]), "=f"(scores[4]), "=f"(scores[5]), "=f"(scores[6]), "=f"(scores[7]), "=f"(scores[8]), "=f"(scores[9]), "=f"(scores[10]), "=f"(scores[11]), "=f"(scores[12]), "=f"(scores[13]), "=f"(scores[14]), "=f"(scores[15]), "=f"(scores[16]), "=f"(scores[17]), "=f"(scores[18]), "=f"(scores[19]), "=f"(scores[20]), "=f"(scores[21]), "=f"(scores[22]), "=f"(scores[23]), "=f"(scores[24]), "=f"(scores[25]), "=f"(scores[26]), "=f"(scores[27]), "=f"(scores[28]), "=f"(scores[29]), "=f"(scores[30]), "=f"(scores[31]), "=f"(scores[32]), "=f"(scores[33]), "=f"(scores[34]), "=f"(scores[35]), "=f"(scores[36]), "=f"(scores[37]), "=f"(scores[38]), "=f"(scores[39]), "=f"(scores[40]), "=f"(scores[41]), "=f"(scores[42]), "=f"(scores[43]), "=f"(scores[44]), "=f"(scores[45]), "=f"(scores[46]), "=f"(scores[47]), "=f"(scores[48]), "=f"(scores[49]), "=f"(scores[50]), "=f"(scores[51]), "=f"(scores[52]), "=f"(scores[53]), "=f"(scores[54]), "=f"(scores[55]), "=f"(scores[56]), "=f"(scores[57]), "=f"(scores[58]), "=f"(scores[59]), "=f"(scores[60]), "=f"(scores[61]), "=f"(scores[62]), "=f"(scores[63]), "=f"(scores[64]), "=f"(scores[65]), "=f"(scores[66]), "=f"(scores[67]), "=f"(scores[68]), "=f"(scores[69]), "=f"(scores[70]), "=f"(scores[71]), "=f"(scores[72]), "=f"(scores[73]), "=f"(scores[74]), "=f"(scores[75]), "=f"(scores[76]), "=f"(scores[77]), "=f"(scores[78]), "=f"(scores[79]), "=f"(scores[80]), "=f"(scores[81]), "=f"(scores[82]), "=f"(scores[83]), "=f"(scores[84]), "=f"(scores[85]), "=f"(scores[86]), "=f"(scores[87]), "=f"(scores[88]), "=f"(scores[89]), "=f"(scores[90]), "=f"(scores[91]), "=f"(scores[92]), "=f"(scores[93]), "=f"(scores[94]), "=f"(scores[95]), "=f"(scores[96]), "=f"(scores[97]), "=f"(scores[98]), "=f"(scores[99]), "=f"(scores[100]), "=f"(scores[101]), "=f"(scores[102]), "=f"(scores[103]), "=f"(scores[104]), "=f"(scores[105]), "=f"(scores[106]), "=f"(scores[107]), "=f"(scores[108]), "=f"(scores[109]), "=f"(scores[110]), "=f"(scores[111]), "=f"(scores[112]), "=f"(scores[113]), "=f"(scores[114]), "=f"(scores[115]), "=f"(scores[116]), "=f"(scores[117]), "=f"(scores[118]), "=f"(scores[119]), "=f"(scores[120]), "=f"(scores[121]), "=f"(scores[122]), "=f"(scores[123]), "=f"(scores[124]), "=f"(scores[125]), "=f"(scores[126]), "=f"(scores[127])
                        : "r"(taddr + tmem_row_base_v + score_base)
                        : "memory");
                    asm volatile("tcgen05.wait::ld.sync.aligned;");
                    asm volatile("barrier.sync 9, 128;");
                    if (elect_sync()) {
                        mbarrier_arrive(score_empty_addr);
                    }
                    #pragma unroll
                    for (int kk = 0; kk < 128; kk += 4) {
                        float csq_vals[4];
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&csq_vals[0])), "=r"(*reinterpret_cast<uint32_t*>(&csq_vals[1])), "=r"(*reinterpret_cast<uint32_t*>(&csq_vals[2])), "=r"(*reinterpret_cast<uint32_t*>(&csq_vals[3]))
                            : "r"(csq_smem_addr + (score_base + kk) * 4));
                        float d0 = scores[kk] - csq_vals[0];
                        float d1 = scores[kk + 1] - csq_vals[1];
                        float best_01 = d0;
                        int idx_01 = off_k + score_base + kk;
                        if (d1 > best_01) {
                            best_01 = d1;
                            idx_01 = off_k + score_base + kk + 1;
                        }
                        float d2 = scores[kk + 2] - csq_vals[2];
                        float d3 = scores[kk + 3] - csq_vals[3];
                        float best_23 = d2;
                        int idx_23 = off_k + score_base + kk + 2;
                        if (d3 > best_23) {
                            best_23 = d3;
                            idx_23 = off_k + score_base + kk + 3;
                        }
                        float best_group = best_01;
                        int idx_group = idx_01;
                        if (best_23 > best_group) {
                            best_group = best_23;
                            idx_group = idx_23;
                        }
                        if (best_group > best_score) {
                            best_score = best_group;
                            best_idx = idx_group;
                        }
                    }
                }
                if (global_n < N) {
                    *((int*)(out + out_offset)) = best_idx;
                }
            }
        }
    }

    // Cleanup
    __syncthreads(); // barrier before TMEM dealloc

    if (warp == 1) {
        asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;" :: "r"(tmem_addr_storage[0]), "r"(256));
        asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;");
    }
}

} // extern "C"

#undef NUM_MAIN_STAGES
#undef SMEM_SMEM_CSQ_OFF
#undef SMEM_SMEM_CSQ_STAGE_BYTES
#undef SMEM_SMEM_CSQ_STRIDE
#undef SMEM_SMEM_C_OFF
#undef SMEM_SMEM_C_STAGE_BYTES
#undef SMEM_SMEM_C_STRIDE
#undef SMEM_SMEM_X_OFF
#undef SMEM_SMEM_X_STAGE_BYTES
#undef SMEM_SMEM_X_STRIDE
#undef SMEM_TOTAL
#undef TMEM_NCOLS
#undef TMEM_SCORE_TMEM_OFFSET
#undef c_empty_addr
#undef c_full_addr
#undef score_empty_addr
#undef score_full_addr
#undef smem_c_addr
#undef smem_csq_addr
#undef smem_x_addr
#undef x_empty_addr
#undef x_full_addr

#define NUM_MAIN_STAGES 1
#define THREADS 256

extern "C" {

__global__ __launch_bounds__(256) void
kernel_flash_kmeans_assign_lowdim_pack_e50c_v1(__nv_bfloat16* __restrict__ x, __nv_bfloat16* __restrict__ centroids, __nv_bfloat16* __restrict__ x_pad, __nv_bfloat16* __restrict__ c_pad, int B, int N, int D, int K, int total_x_pad, int total_c_pad)
{
    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;


    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    const int warp_id = warp;
    const int lane_id = lane;

    // === Task calls (dependency order) ===
    int grid_stride = num_bids * 256;
    int start = bid * 256 + tid;
    int total_x_vecs = total_x_pad / 8;
    int total_c_vecs = total_c_pad / 8;
    #pragma unroll 1
    for (unsigned int vec_idx = start; vec_idx < total_x_vecs; vec_idx += grid_stride) {
        int d_pad = vec_idx % 16 * 8;
        int row = vec_idx / 16;
        int dst_base = vec_idx * 8;
        if (d_pad < D) {
            float vals[8];
            {
                const uint4* _vptr_0 = reinterpret_cast<const uint4*>(x + row * D + d_pad);
                uint4 _vld_0[1];
                #pragma unroll
                for (int _blk = 0; _blk < 1; _blk++) {
                    _vld_0[_blk] = _vptr_0[_blk];
                    __nv_bfloat16* _velems_0 = reinterpret_cast<__nv_bfloat16*>(&_vld_0[_blk]);
                    #pragma unroll
                    for (int _j = 0; _j < 8; _j++)
                        vals[0 + _blk * 8 + _j] = __bfloat162float(_velems_0[_j]);
                }
            }
            {
                __nv_bfloat162 _pk[4];
                _pk[0] = __floats2bfloat162_rn(vals[0 + 0], vals[0 + 1]);
                _pk[1] = __floats2bfloat162_rn(vals[0 + 2], vals[0 + 3]);
                _pk[2] = __floats2bfloat162_rn(vals[0 + 4], vals[0 + 5]);
                _pk[3] = __floats2bfloat162_rn(vals[0 + 6], vals[0 + 7]);
                *reinterpret_cast<uint4*>(&((__nv_bfloat16*)(x_pad + dst_base))[0]) = *reinterpret_cast<uint4*>(&_pk[0]);
            }
        } else {
            float zeros[8];
            zeros[0] = 0.0f;
            zeros[1] = 0.0f;
            zeros[2] = 0.0f;
            zeros[3] = 0.0f;
            zeros[4] = 0.0f;
            zeros[5] = 0.0f;
            zeros[6] = 0.0f;
            zeros[7] = 0.0f;
            {
                __nv_bfloat162 _pk[4];
                _pk[0] = __floats2bfloat162_rn(zeros[0 + 0], zeros[0 + 1]);
                _pk[1] = __floats2bfloat162_rn(zeros[0 + 2], zeros[0 + 3]);
                _pk[2] = __floats2bfloat162_rn(zeros[0 + 4], zeros[0 + 5]);
                _pk[3] = __floats2bfloat162_rn(zeros[0 + 6], zeros[0 + 7]);
                *reinterpret_cast<uint4*>(&((__nv_bfloat16*)(x_pad + dst_base))[0]) = *reinterpret_cast<uint4*>(&_pk[0]);
            }
        }
    }
    #pragma unroll 1
    for (unsigned int vec_idx = start; vec_idx < total_c_vecs; vec_idx += grid_stride) {
        int d_pad = vec_idx % 16 * 8;
        int row = vec_idx / 16;
        int dst_base = vec_idx * 8;
        if (d_pad < D) {
            float vals[8];
            {
                const uint4* _vptr_1 = reinterpret_cast<const uint4*>(centroids + row * D + d_pad);
                uint4 _vld_1[1];
                #pragma unroll
                for (int _blk = 0; _blk < 1; _blk++) {
                    _vld_1[_blk] = _vptr_1[_blk];
                    __nv_bfloat16* _velems_1 = reinterpret_cast<__nv_bfloat16*>(&_vld_1[_blk]);
                    #pragma unroll
                    for (int _j = 0; _j < 8; _j++)
                        vals[0 + _blk * 8 + _j] = __bfloat162float(_velems_1[_j]);
                }
            }
            {
                __nv_bfloat162 _pk[4];
                _pk[0] = __floats2bfloat162_rn(vals[0 + 0], vals[0 + 1]);
                _pk[1] = __floats2bfloat162_rn(vals[0 + 2], vals[0 + 3]);
                _pk[2] = __floats2bfloat162_rn(vals[0 + 4], vals[0 + 5]);
                _pk[3] = __floats2bfloat162_rn(vals[0 + 6], vals[0 + 7]);
                *reinterpret_cast<uint4*>(&((__nv_bfloat16*)(c_pad + dst_base))[0]) = *reinterpret_cast<uint4*>(&_pk[0]);
            }
        } else {
            float zeros[8];
            zeros[0] = 0.0f;
            zeros[1] = 0.0f;
            zeros[2] = 0.0f;
            zeros[3] = 0.0f;
            zeros[4] = 0.0f;
            zeros[5] = 0.0f;
            zeros[6] = 0.0f;
            zeros[7] = 0.0f;
            {
                __nv_bfloat162 _pk[4];
                _pk[0] = __floats2bfloat162_rn(zeros[0 + 0], zeros[0 + 1]);
                _pk[1] = __floats2bfloat162_rn(zeros[0 + 2], zeros[0 + 3]);
                _pk[2] = __floats2bfloat162_rn(zeros[0 + 4], zeros[0 + 5]);
                _pk[3] = __floats2bfloat162_rn(zeros[0 + 6], zeros[0 + 7]);
                *reinterpret_cast<uint4*>(&((__nv_bfloat16*)(c_pad + dst_base))[0]) = *reinterpret_cast<uint4*>(&_pk[0]);
            }
        }
    }
}

} // extern "C"

#undef NUM_MAIN_STAGES
#undef THREADS

#define TMEM_NCOLS 256
#define TMEM_SCORE_TMEM_OFFSET 0
#define NUM_MAIN_STAGES 1
#define SMEM_SMEM_X_OFF 1024
#define SMEM_SMEM_X_STAGE_BYTES 32768
#define SMEM_SMEM_X_STRIDE 32768
#define SMEM_SMEM_C_OFF 33792
#define SMEM_SMEM_C_STAGE_BYTES 65536
#define SMEM_SMEM_C_STRIDE 65536
#define SMEM_SMEM_CSQ_OFF 99328
#define SMEM_SMEM_CSQ_STAGE_BYTES 1024
#define SMEM_SMEM_CSQ_STRIDE 1024
#define SMEM_TOTAL 100352

extern "C" {

__global__ __launch_bounds__(192) void
kernel_flash_kmeans_assign_lowdim_e50c_v1(float* __restrict__ x_sq, float* __restrict__ c_sq, int32_t* __restrict__ out, const void* x_tmap, const void* c_tmap, int B, int N, int D, int K, int num_n_tiles, int K_tiles)
{
    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;

    extern __shared__ __align__(1024) char smem_raw[];
    int smem;
    smem = (int)(unsigned long long)__cvta_generic_to_shared(smem_raw);
    const int smem_smem_x = smem + 1024;
    const int smem_smem_c = smem + 33792;
    const int smem_smem_csq = smem + 99328;

    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    // Mbarrier init (6 groups, 6 barriers)
    // Mbarriers at smem_raw[0..48)

    if (warp == 0) {
        uint32_t leader = elect_sync();
        // x_full: 1 barriers, init_count=1
        mbarrier_init_pred(smem + 0, 1, leader);
        // x_empty: 1 barriers, init_count=1
        mbarrier_init_pred(smem + 8, 1, leader);
        // c_full: 1 barriers, init_count=1
        mbarrier_init_pred(smem + 16, 1, leader);
        // c_empty: 1 barriers, init_count=1
        mbarrier_init_pred(smem + 24, 1, leader);
        // score_full: 1 barriers, init_count=1
        mbarrier_init_pred(smem + 32, 1, leader);
        // score_empty: 1 barriers, init_count=4
        mbarrier_init_pred(smem + 40, 4, leader);
        asm volatile("fence.mbarrier_init.release.cluster;");
    }

    __syncthreads();

    // TMEM alloc (256 columns, 256 used)
    volatile int* tmem_addr_storage = (volatile int*)(smem_raw + 48);
    if (warp == 1) {
        int _tmem_hold = smem + 48;
        asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;" :: "r"(_tmem_hold), "r"(256) : "memory");
    }

    __syncthreads();
    asm volatile("tcgen05.fence::after_thread_sync;");

    const int warp_id = warp;
    const int lane_id = lane;
    __nv_bfloat16* smem_x = (__nv_bfloat16*)(smem_raw + 1024);
    #define smem_x_addr (smem + 1024)
    __nv_bfloat16* smem_c = (__nv_bfloat16*)(smem_raw + 33792);
    #define smem_c_addr (smem + 33792)
    float* smem_csq = (float*)(smem_raw + 99328);
    #define smem_csq_addr (smem + 99328)
    const int mbar_base = smem;
    #define x_full_addr (mbar_base + 0)
    #define x_empty_addr (mbar_base + 8)
    #define c_full_addr (mbar_base + 16)
    #define c_empty_addr (mbar_base + 24)
    #define score_full_addr (mbar_base + 32)
    #define score_empty_addr (mbar_base + 40)
    const int taddr = tmem_addr_storage[0];

    // ---- Role: load ----
    if (warp == 0) {
        { // load_main
            int num_tiles = B * num_n_tiles;
            uint32_t _phase_x_empty_0 = 1;
            uint32_t _phase_c_empty_0 = 1;
            if (warp_id == 0) {
                if (elect_sync()) {
                    #pragma unroll 1
                    for (unsigned int tile_idx = bid; tile_idx < num_tiles; tile_idx += num_bids) {
                        int batch = tile_idx / num_n_tiles;
                        int n_tile = tile_idx % num_n_tiles;
                        int off_n = n_tile * 128;
                        int x_row = batch * N + off_n;
                        mbarrier_wait(x_empty_addr, _phase_x_empty_0);
                        _phase_x_empty_0 ^= 1;
                        tma_3d_gmem2smem(smem_x_addr, x_tmap, 0, x_row, 0, x_full_addr);
                        mbarrier_arrive_expect_tx(x_full_addr, 32768);
                        #pragma unroll 1
                        for (int iter_k = 0; iter_k < K_tiles; iter_k++) {
                            int off_k = iter_k * 256;
                            int c_row = batch * K + off_k;
                            mbarrier_wait(c_empty_addr, _phase_c_empty_0);
                            _phase_c_empty_0 ^= 1;
                            tma_3d_gmem2smem(smem_c_addr, c_tmap, 0, c_row, 0, c_full_addr);
                            mbarrier_arrive_expect_tx(c_full_addr, 65536);
                        }
                    }
                }
            }
        }
    // ---- Role: mma ----
    } else if (warp == 1) {
        const int tmem_score_tmem = taddr + TMEM_SCORE_TMEM_OFFSET;
        { // mma_main
            int num_tiles = B * num_n_tiles;
            uint32_t _phase_x_full_0 = 0;
            uint32_t _phase_score_empty_0 = 1;
            uint32_t _phase_c_full_0 = 0;
            #pragma unroll 1
            for (unsigned int _tile_idx = bid; _tile_idx < num_tiles; _tile_idx += num_bids) {
                mbarrier_wait(x_full_addr, _phase_x_full_0);
                _phase_x_full_0 ^= 1;
                #pragma unroll 1
                for (int _iter_k = 0; _iter_k < K_tiles; _iter_k++) {
                    mbarrier_wait(score_empty_addr, _phase_score_empty_0);
                    _phase_score_empty_0 ^= 1;
                    mbarrier_wait(c_full_addr, _phase_c_full_0);
                    _phase_c_full_0 ^= 1;
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    int _desc_addr_0 = smem_x_addr + 0 * 32768 + 16384;
                    int _desc_lo_0 = make_warp_uniform((_desc_addr_0 >> 4) & 0x3FFF);
                    int _desc_addr_1 = smem_c_addr + 0 * 65536 + 32768;
                    int _desc_lo_1 = make_warp_uniform((_desc_addr_1 >> 4) & 0x3FFF);
                    int _mma_ss_a_addr_2 = smem_x_addr + 0 * 32768;
                    int _mma_ss_a_lo_2 = make_warp_uniform((_mma_ss_a_addr_2 >> 4) & 0x3FFF);
                    int _mma_ss_b_addr_2 = smem_c_addr + 0 * 65536;
                    int _mma_ss_b_lo_2 = make_warp_uniform((_mma_ss_b_addr_2 >> 4) & 0x3FFF);
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
                    "mov.b32 id, 138413200;\n\t"
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
                    :: "r"(_mma_ss_a_lo_2), "r"(_mma_ss_b_lo_2), "r"(tmem_score_tmem), "r"(0));
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
                    "mov.b32 id, 138413200;\n\t"
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
                    :: "r"(_desc_lo_0), "r"(_desc_lo_1), "r"(tmem_score_tmem), "r"(1));
                    elect_commit(score_full_addr);
                    elect_commit(c_empty_addr);
                }
                elect_commit(x_empty_addr);
            }
        }
    // ---- Role: compute ----
    } else if (warp >= 2 && warp <= 5) {
        const int warp_id_in_wg = warp % 4;
        const int my_row = warp_id_in_wg * 32 + lane;
        const int tmem_row_base_v = (warp_id_in_wg * 32) << 16;
        { // compute_main
            int num_tiles = B * num_n_tiles;
            uint32_t _phase_score_full_0 = 0;
            #pragma unroll 1
            for (unsigned int tile_idx = bid; tile_idx < num_tiles; tile_idx += num_bids) {
                int batch = tile_idx / num_n_tiles;
                int n_tile = tile_idx % num_n_tiles;
                int off_n = n_tile * 128;
                int global_n = off_n + my_row;
                int out_offset = batch * N + global_n;
                int csq_smem_addr = smem_csq_addr + 0;
                float best_score = -3.4e+38f;
                int best_idx = 0;
                #pragma unroll 1
                for (int iter_k = 0; iter_k < K_tiles; iter_k++) {
                    int off_k = iter_k * 256;
                    if (my_row < 64.0f) {
                        int csq_base = my_row * 4;
                        float csq_pack[4];
                        {
                            float4 _v4 = *reinterpret_cast<const float4*>(c_sq + batch * K + off_k + csq_base);
                            csq_pack[0 + 0] = _v4.x;
                            csq_pack[0 + 1] = _v4.y;
                            csq_pack[0 + 2] = _v4.z;
                            csq_pack[0 + 3] = _v4.w;
                        }
                        float csq_half[4];
                        csq_half[0] = 0.5f * csq_pack[0];
                        csq_half[1] = 0.5f * csq_pack[1];
                        csq_half[2] = 0.5f * csq_pack[2];
                        csq_half[3] = 0.5f * csq_pack[3];
                        asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(csq_smem_addr + csq_base * 4), "f"(csq_half[0]), "f"(csq_half[1]), "f"(csq_half[2]), "f"(csq_half[3]) : "memory");
                    }
                    asm volatile("barrier.sync 8, 128;");
                    mbarrier_wait(score_full_addr, _phase_score_full_0);
                    _phase_score_full_0 ^= 1;
                    int score_base = 0;
                    float scores[128];
                    asm volatile(
                        "tcgen05.ld.sync.aligned.32x32b.x128.b32"
                        " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63, %64, %65, %66, %67, %68, %69, %70, %71, %72, %73, %74, %75, %76, %77, %78, %79, %80, %81, %82, %83, %84, %85, %86, %87, %88, %89, %90, %91, %92, %93, %94, %95, %96, %97, %98, %99, %100, %101, %102, %103, %104, %105, %106, %107, %108, %109, %110, %111, %112, %113, %114, %115, %116, %117, %118, %119, %120, %121, %122, %123, %124, %125, %126, %127}, [%128];"
                        : "=f"(scores[0]), "=f"(scores[1]), "=f"(scores[2]), "=f"(scores[3]), "=f"(scores[4]), "=f"(scores[5]), "=f"(scores[6]), "=f"(scores[7]), "=f"(scores[8]), "=f"(scores[9]), "=f"(scores[10]), "=f"(scores[11]), "=f"(scores[12]), "=f"(scores[13]), "=f"(scores[14]), "=f"(scores[15]), "=f"(scores[16]), "=f"(scores[17]), "=f"(scores[18]), "=f"(scores[19]), "=f"(scores[20]), "=f"(scores[21]), "=f"(scores[22]), "=f"(scores[23]), "=f"(scores[24]), "=f"(scores[25]), "=f"(scores[26]), "=f"(scores[27]), "=f"(scores[28]), "=f"(scores[29]), "=f"(scores[30]), "=f"(scores[31]), "=f"(scores[32]), "=f"(scores[33]), "=f"(scores[34]), "=f"(scores[35]), "=f"(scores[36]), "=f"(scores[37]), "=f"(scores[38]), "=f"(scores[39]), "=f"(scores[40]), "=f"(scores[41]), "=f"(scores[42]), "=f"(scores[43]), "=f"(scores[44]), "=f"(scores[45]), "=f"(scores[46]), "=f"(scores[47]), "=f"(scores[48]), "=f"(scores[49]), "=f"(scores[50]), "=f"(scores[51]), "=f"(scores[52]), "=f"(scores[53]), "=f"(scores[54]), "=f"(scores[55]), "=f"(scores[56]), "=f"(scores[57]), "=f"(scores[58]), "=f"(scores[59]), "=f"(scores[60]), "=f"(scores[61]), "=f"(scores[62]), "=f"(scores[63]), "=f"(scores[64]), "=f"(scores[65]), "=f"(scores[66]), "=f"(scores[67]), "=f"(scores[68]), "=f"(scores[69]), "=f"(scores[70]), "=f"(scores[71]), "=f"(scores[72]), "=f"(scores[73]), "=f"(scores[74]), "=f"(scores[75]), "=f"(scores[76]), "=f"(scores[77]), "=f"(scores[78]), "=f"(scores[79]), "=f"(scores[80]), "=f"(scores[81]), "=f"(scores[82]), "=f"(scores[83]), "=f"(scores[84]), "=f"(scores[85]), "=f"(scores[86]), "=f"(scores[87]), "=f"(scores[88]), "=f"(scores[89]), "=f"(scores[90]), "=f"(scores[91]), "=f"(scores[92]), "=f"(scores[93]), "=f"(scores[94]), "=f"(scores[95]), "=f"(scores[96]), "=f"(scores[97]), "=f"(scores[98]), "=f"(scores[99]), "=f"(scores[100]), "=f"(scores[101]), "=f"(scores[102]), "=f"(scores[103]), "=f"(scores[104]), "=f"(scores[105]), "=f"(scores[106]), "=f"(scores[107]), "=f"(scores[108]), "=f"(scores[109]), "=f"(scores[110]), "=f"(scores[111]), "=f"(scores[112]), "=f"(scores[113]), "=f"(scores[114]), "=f"(scores[115]), "=f"(scores[116]), "=f"(scores[117]), "=f"(scores[118]), "=f"(scores[119]), "=f"(scores[120]), "=f"(scores[121]), "=f"(scores[122]), "=f"(scores[123]), "=f"(scores[124]), "=f"(scores[125]), "=f"(scores[126]), "=f"(scores[127])
                        : "r"(taddr + tmem_row_base_v + score_base)
                        : "memory");
                    asm volatile("tcgen05.wait::ld.sync.aligned;");
                    #pragma unroll
                    for (int kk = 0; kk < 128; kk += 4) {
                        float csq_vals[4];
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&csq_vals[0])), "=r"(*reinterpret_cast<uint32_t*>(&csq_vals[1])), "=r"(*reinterpret_cast<uint32_t*>(&csq_vals[2])), "=r"(*reinterpret_cast<uint32_t*>(&csq_vals[3]))
                            : "r"(csq_smem_addr + (score_base + kk) * 4));
                        float d0 = scores[kk] - csq_vals[0];
                        float d1 = scores[kk + 1] - csq_vals[1];
                        float best_01 = d0;
                        int idx_01 = off_k + score_base + kk;
                        if (d1 > best_01) {
                            best_01 = d1;
                            idx_01 = off_k + score_base + kk + 1;
                        }
                        float d2 = scores[kk + 2] - csq_vals[2];
                        float d3 = scores[kk + 3] - csq_vals[3];
                        float best_23 = d2;
                        int idx_23 = off_k + score_base + kk + 2;
                        if (d3 > best_23) {
                            best_23 = d3;
                            idx_23 = off_k + score_base + kk + 3;
                        }
                        float best_group = best_01;
                        int idx_group = idx_01;
                        if (best_23 > best_group) {
                            best_group = best_23;
                            idx_group = idx_23;
                        }
                        if (best_group > best_score) {
                            best_score = best_group;
                            best_idx = idx_group;
                        }
                    }
                    score_base = 128;
                    asm volatile(
                        "tcgen05.ld.sync.aligned.32x32b.x128.b32"
                        " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63, %64, %65, %66, %67, %68, %69, %70, %71, %72, %73, %74, %75, %76, %77, %78, %79, %80, %81, %82, %83, %84, %85, %86, %87, %88, %89, %90, %91, %92, %93, %94, %95, %96, %97, %98, %99, %100, %101, %102, %103, %104, %105, %106, %107, %108, %109, %110, %111, %112, %113, %114, %115, %116, %117, %118, %119, %120, %121, %122, %123, %124, %125, %126, %127}, [%128];"
                        : "=f"(scores[0]), "=f"(scores[1]), "=f"(scores[2]), "=f"(scores[3]), "=f"(scores[4]), "=f"(scores[5]), "=f"(scores[6]), "=f"(scores[7]), "=f"(scores[8]), "=f"(scores[9]), "=f"(scores[10]), "=f"(scores[11]), "=f"(scores[12]), "=f"(scores[13]), "=f"(scores[14]), "=f"(scores[15]), "=f"(scores[16]), "=f"(scores[17]), "=f"(scores[18]), "=f"(scores[19]), "=f"(scores[20]), "=f"(scores[21]), "=f"(scores[22]), "=f"(scores[23]), "=f"(scores[24]), "=f"(scores[25]), "=f"(scores[26]), "=f"(scores[27]), "=f"(scores[28]), "=f"(scores[29]), "=f"(scores[30]), "=f"(scores[31]), "=f"(scores[32]), "=f"(scores[33]), "=f"(scores[34]), "=f"(scores[35]), "=f"(scores[36]), "=f"(scores[37]), "=f"(scores[38]), "=f"(scores[39]), "=f"(scores[40]), "=f"(scores[41]), "=f"(scores[42]), "=f"(scores[43]), "=f"(scores[44]), "=f"(scores[45]), "=f"(scores[46]), "=f"(scores[47]), "=f"(scores[48]), "=f"(scores[49]), "=f"(scores[50]), "=f"(scores[51]), "=f"(scores[52]), "=f"(scores[53]), "=f"(scores[54]), "=f"(scores[55]), "=f"(scores[56]), "=f"(scores[57]), "=f"(scores[58]), "=f"(scores[59]), "=f"(scores[60]), "=f"(scores[61]), "=f"(scores[62]), "=f"(scores[63]), "=f"(scores[64]), "=f"(scores[65]), "=f"(scores[66]), "=f"(scores[67]), "=f"(scores[68]), "=f"(scores[69]), "=f"(scores[70]), "=f"(scores[71]), "=f"(scores[72]), "=f"(scores[73]), "=f"(scores[74]), "=f"(scores[75]), "=f"(scores[76]), "=f"(scores[77]), "=f"(scores[78]), "=f"(scores[79]), "=f"(scores[80]), "=f"(scores[81]), "=f"(scores[82]), "=f"(scores[83]), "=f"(scores[84]), "=f"(scores[85]), "=f"(scores[86]), "=f"(scores[87]), "=f"(scores[88]), "=f"(scores[89]), "=f"(scores[90]), "=f"(scores[91]), "=f"(scores[92]), "=f"(scores[93]), "=f"(scores[94]), "=f"(scores[95]), "=f"(scores[96]), "=f"(scores[97]), "=f"(scores[98]), "=f"(scores[99]), "=f"(scores[100]), "=f"(scores[101]), "=f"(scores[102]), "=f"(scores[103]), "=f"(scores[104]), "=f"(scores[105]), "=f"(scores[106]), "=f"(scores[107]), "=f"(scores[108]), "=f"(scores[109]), "=f"(scores[110]), "=f"(scores[111]), "=f"(scores[112]), "=f"(scores[113]), "=f"(scores[114]), "=f"(scores[115]), "=f"(scores[116]), "=f"(scores[117]), "=f"(scores[118]), "=f"(scores[119]), "=f"(scores[120]), "=f"(scores[121]), "=f"(scores[122]), "=f"(scores[123]), "=f"(scores[124]), "=f"(scores[125]), "=f"(scores[126]), "=f"(scores[127])
                        : "r"(taddr + tmem_row_base_v + score_base)
                        : "memory");
                    asm volatile("tcgen05.wait::ld.sync.aligned;");
                    asm volatile("barrier.sync 9, 128;");
                    if (elect_sync()) {
                        mbarrier_arrive(score_empty_addr);
                    }
                    #pragma unroll
                    for (int kk = 0; kk < 128; kk += 4) {
                        float csq_vals[4];
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&csq_vals[0])), "=r"(*reinterpret_cast<uint32_t*>(&csq_vals[1])), "=r"(*reinterpret_cast<uint32_t*>(&csq_vals[2])), "=r"(*reinterpret_cast<uint32_t*>(&csq_vals[3]))
                            : "r"(csq_smem_addr + (score_base + kk) * 4));
                        float d0 = scores[kk] - csq_vals[0];
                        float d1 = scores[kk + 1] - csq_vals[1];
                        float best_01 = d0;
                        int idx_01 = off_k + score_base + kk;
                        if (d1 > best_01) {
                            best_01 = d1;
                            idx_01 = off_k + score_base + kk + 1;
                        }
                        float d2 = scores[kk + 2] - csq_vals[2];
                        float d3 = scores[kk + 3] - csq_vals[3];
                        float best_23 = d2;
                        int idx_23 = off_k + score_base + kk + 2;
                        if (d3 > best_23) {
                            best_23 = d3;
                            idx_23 = off_k + score_base + kk + 3;
                        }
                        float best_group = best_01;
                        int idx_group = idx_01;
                        if (best_23 > best_group) {
                            best_group = best_23;
                            idx_group = idx_23;
                        }
                        if (best_group > best_score) {
                            best_score = best_group;
                            best_idx = idx_group;
                        }
                    }
                }
                if (global_n < N) {
                    *((int*)(out + out_offset)) = best_idx;
                }
            }
        }
    }

    // Cleanup
    __syncthreads(); // barrier before TMEM dealloc

    if (warp == 1) {
        asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;" :: "r"(tmem_addr_storage[0]), "r"(256));
        asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;");
    }
}

} // extern "C"

#undef NUM_MAIN_STAGES
#undef SMEM_SMEM_CSQ_OFF
#undef SMEM_SMEM_CSQ_STAGE_BYTES
#undef SMEM_SMEM_CSQ_STRIDE
#undef SMEM_SMEM_C_OFF
#undef SMEM_SMEM_C_STAGE_BYTES
#undef SMEM_SMEM_C_STRIDE
#undef SMEM_SMEM_X_OFF
#undef SMEM_SMEM_X_STAGE_BYTES
#undef SMEM_SMEM_X_STRIDE
#undef SMEM_TOTAL
#undef TMEM_NCOLS
#undef TMEM_SCORE_TMEM_OFFSET
#undef c_empty_addr
#undef c_full_addr
#undef score_empty_addr
#undef score_full_addr
#undef smem_c_addr
#undef smem_csq_addr
#undef smem_x_addr
#undef x_empty_addr
#undef x_full_addr

#define NUM_MAIN_STAGES 1
#define THREADS 256

extern "C" {

__global__ __launch_bounds__(256) void
kernel_flash_kmeans_assign_gap_pad_pack_v1(__nv_bfloat16* __restrict__ x, __nv_bfloat16* __restrict__ centroids, __nv_bfloat16* __restrict__ x_pad, __nv_bfloat16* __restrict__ c_pad, int B, int N, int D, int K, int D_PAD, int total_x_pad, int total_c_pad)
{
    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;


    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    const int warp_id = warp;
    const int lane_id = lane;

    // === Task calls (dependency order) ===
    int grid_stride = num_bids * 256;
    int start = bid * 256 + tid;
    int d_pad_vecs = D_PAD / 8;
    int total_x_vecs = total_x_pad / 8;
    int total_c_vecs = total_c_pad / 8;
    #pragma unroll 1
    for (unsigned int vec_idx = start; vec_idx < total_x_vecs; vec_idx += grid_stride) {
        int d_pad = vec_idx % d_pad_vecs * 8;
        int row = vec_idx / d_pad_vecs;
        int dst_base = vec_idx * 8;
        if (d_pad < D) {
            float vals[8];
            {
                const uint4* _vptr_0 = reinterpret_cast<const uint4*>(x + row * D + d_pad);
                uint4 _vld_0[1];
                #pragma unroll
                for (int _blk = 0; _blk < 1; _blk++) {
                    _vld_0[_blk] = _vptr_0[_blk];
                    __nv_bfloat16* _velems_0 = reinterpret_cast<__nv_bfloat16*>(&_vld_0[_blk]);
                    #pragma unroll
                    for (int _j = 0; _j < 8; _j++)
                        vals[0 + _blk * 8 + _j] = __bfloat162float(_velems_0[_j]);
                }
            }
            {
                __nv_bfloat162 _pk[4];
                _pk[0] = __floats2bfloat162_rn(vals[0 + 0], vals[0 + 1]);
                _pk[1] = __floats2bfloat162_rn(vals[0 + 2], vals[0 + 3]);
                _pk[2] = __floats2bfloat162_rn(vals[0 + 4], vals[0 + 5]);
                _pk[3] = __floats2bfloat162_rn(vals[0 + 6], vals[0 + 7]);
                *reinterpret_cast<uint4*>(&((__nv_bfloat16*)(x_pad + dst_base))[0]) = *reinterpret_cast<uint4*>(&_pk[0]);
            }
        } else {
            float zeros[8];
            zeros[0] = 0.0f;
            zeros[1] = 0.0f;
            zeros[2] = 0.0f;
            zeros[3] = 0.0f;
            zeros[4] = 0.0f;
            zeros[5] = 0.0f;
            zeros[6] = 0.0f;
            zeros[7] = 0.0f;
            {
                __nv_bfloat162 _pk[4];
                _pk[0] = __floats2bfloat162_rn(zeros[0 + 0], zeros[0 + 1]);
                _pk[1] = __floats2bfloat162_rn(zeros[0 + 2], zeros[0 + 3]);
                _pk[2] = __floats2bfloat162_rn(zeros[0 + 4], zeros[0 + 5]);
                _pk[3] = __floats2bfloat162_rn(zeros[0 + 6], zeros[0 + 7]);
                *reinterpret_cast<uint4*>(&((__nv_bfloat16*)(x_pad + dst_base))[0]) = *reinterpret_cast<uint4*>(&_pk[0]);
            }
        }
    }
    #pragma unroll 1
    for (unsigned int vec_idx = start; vec_idx < total_c_vecs; vec_idx += grid_stride) {
        int d_pad = vec_idx % d_pad_vecs * 8;
        int row = vec_idx / d_pad_vecs;
        int dst_base = vec_idx * 8;
        if (d_pad < D) {
            float vals[8];
            {
                const uint4* _vptr_1 = reinterpret_cast<const uint4*>(centroids + row * D + d_pad);
                uint4 _vld_1[1];
                #pragma unroll
                for (int _blk = 0; _blk < 1; _blk++) {
                    _vld_1[_blk] = _vptr_1[_blk];
                    __nv_bfloat16* _velems_1 = reinterpret_cast<__nv_bfloat16*>(&_vld_1[_blk]);
                    #pragma unroll
                    for (int _j = 0; _j < 8; _j++)
                        vals[0 + _blk * 8 + _j] = __bfloat162float(_velems_1[_j]);
                }
            }
            {
                __nv_bfloat162 _pk[4];
                _pk[0] = __floats2bfloat162_rn(vals[0 + 0], vals[0 + 1]);
                _pk[1] = __floats2bfloat162_rn(vals[0 + 2], vals[0 + 3]);
                _pk[2] = __floats2bfloat162_rn(vals[0 + 4], vals[0 + 5]);
                _pk[3] = __floats2bfloat162_rn(vals[0 + 6], vals[0 + 7]);
                *reinterpret_cast<uint4*>(&((__nv_bfloat16*)(c_pad + dst_base))[0]) = *reinterpret_cast<uint4*>(&_pk[0]);
            }
        } else {
            float zeros[8];
            zeros[0] = 0.0f;
            zeros[1] = 0.0f;
            zeros[2] = 0.0f;
            zeros[3] = 0.0f;
            zeros[4] = 0.0f;
            zeros[5] = 0.0f;
            zeros[6] = 0.0f;
            zeros[7] = 0.0f;
            {
                __nv_bfloat162 _pk[4];
                _pk[0] = __floats2bfloat162_rn(zeros[0 + 0], zeros[0 + 1]);
                _pk[1] = __floats2bfloat162_rn(zeros[0 + 2], zeros[0 + 3]);
                _pk[2] = __floats2bfloat162_rn(zeros[0 + 4], zeros[0 + 5]);
                _pk[3] = __floats2bfloat162_rn(zeros[0 + 6], zeros[0 + 7]);
                *reinterpret_cast<uint4*>(&((__nv_bfloat16*)(c_pad + dst_base))[0]) = *reinterpret_cast<uint4*>(&_pk[0]);
            }
        }
    }
}

} // extern "C"

#undef NUM_MAIN_STAGES
#undef THREADS

#define NUM_MAIN_STAGES 1
#define THREADS 256

extern "C" {

__global__ __launch_bounds__(256) void
kernel_flash_kmeans_assign_d160_pad192_pack_f9b2_v1(__nv_bfloat16* __restrict__ x, __nv_bfloat16* __restrict__ centroids, __nv_bfloat16* __restrict__ x_pad, __nv_bfloat16* __restrict__ c_pad, int B, int N, int D, int K, int total_x_pad, int total_c_pad)
{
    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;


    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    const int warp_id = warp;
    const int lane_id = lane;

    // === Task calls (dependency order) ===
    int grid_stride = num_bids * 256;
    int start = bid * 256 + tid;
    int total_x_vecs = total_x_pad / 8;
    int total_c_vecs = total_c_pad / 8;
    #pragma unroll 1
    for (unsigned int vec_idx = start; vec_idx < total_x_vecs; vec_idx += grid_stride) {
        int d_pad = vec_idx % 24 * 8;
        int row = vec_idx / 24;
        int dst_base = vec_idx * 8;
        if (d_pad < D) {
            float vals[8];
            {
                const uint4* _vptr_0 = reinterpret_cast<const uint4*>(x + row * D + d_pad);
                uint4 _vld_0[1];
                #pragma unroll
                for (int _blk = 0; _blk < 1; _blk++) {
                    _vld_0[_blk] = _vptr_0[_blk];
                    __nv_bfloat16* _velems_0 = reinterpret_cast<__nv_bfloat16*>(&_vld_0[_blk]);
                    #pragma unroll
                    for (int _j = 0; _j < 8; _j++)
                        vals[0 + _blk * 8 + _j] = __bfloat162float(_velems_0[_j]);
                }
            }
            {
                __nv_bfloat162 _pk[4];
                _pk[0] = __floats2bfloat162_rn(vals[0 + 0], vals[0 + 1]);
                _pk[1] = __floats2bfloat162_rn(vals[0 + 2], vals[0 + 3]);
                _pk[2] = __floats2bfloat162_rn(vals[0 + 4], vals[0 + 5]);
                _pk[3] = __floats2bfloat162_rn(vals[0 + 6], vals[0 + 7]);
                *reinterpret_cast<uint4*>(&((__nv_bfloat16*)(x_pad + dst_base))[0]) = *reinterpret_cast<uint4*>(&_pk[0]);
            }
        } else {
            float zeros[8];
            zeros[0] = 0.0f;
            zeros[1] = 0.0f;
            zeros[2] = 0.0f;
            zeros[3] = 0.0f;
            zeros[4] = 0.0f;
            zeros[5] = 0.0f;
            zeros[6] = 0.0f;
            zeros[7] = 0.0f;
            {
                __nv_bfloat162 _pk[4];
                _pk[0] = __floats2bfloat162_rn(zeros[0 + 0], zeros[0 + 1]);
                _pk[1] = __floats2bfloat162_rn(zeros[0 + 2], zeros[0 + 3]);
                _pk[2] = __floats2bfloat162_rn(zeros[0 + 4], zeros[0 + 5]);
                _pk[3] = __floats2bfloat162_rn(zeros[0 + 6], zeros[0 + 7]);
                *reinterpret_cast<uint4*>(&((__nv_bfloat16*)(x_pad + dst_base))[0]) = *reinterpret_cast<uint4*>(&_pk[0]);
            }
        }
    }
    #pragma unroll 1
    for (unsigned int vec_idx = start; vec_idx < total_c_vecs; vec_idx += grid_stride) {
        int d_pad = vec_idx % 24 * 8;
        int row = vec_idx / 24;
        int dst_base = vec_idx * 8;
        if (d_pad < D) {
            float vals[8];
            {
                const uint4* _vptr_1 = reinterpret_cast<const uint4*>(centroids + row * D + d_pad);
                uint4 _vld_1[1];
                #pragma unroll
                for (int _blk = 0; _blk < 1; _blk++) {
                    _vld_1[_blk] = _vptr_1[_blk];
                    __nv_bfloat16* _velems_1 = reinterpret_cast<__nv_bfloat16*>(&_vld_1[_blk]);
                    #pragma unroll
                    for (int _j = 0; _j < 8; _j++)
                        vals[0 + _blk * 8 + _j] = __bfloat162float(_velems_1[_j]);
                }
            }
            {
                __nv_bfloat162 _pk[4];
                _pk[0] = __floats2bfloat162_rn(vals[0 + 0], vals[0 + 1]);
                _pk[1] = __floats2bfloat162_rn(vals[0 + 2], vals[0 + 3]);
                _pk[2] = __floats2bfloat162_rn(vals[0 + 4], vals[0 + 5]);
                _pk[3] = __floats2bfloat162_rn(vals[0 + 6], vals[0 + 7]);
                *reinterpret_cast<uint4*>(&((__nv_bfloat16*)(c_pad + dst_base))[0]) = *reinterpret_cast<uint4*>(&_pk[0]);
            }
        } else {
            float zeros[8];
            zeros[0] = 0.0f;
            zeros[1] = 0.0f;
            zeros[2] = 0.0f;
            zeros[3] = 0.0f;
            zeros[4] = 0.0f;
            zeros[5] = 0.0f;
            zeros[6] = 0.0f;
            zeros[7] = 0.0f;
            {
                __nv_bfloat162 _pk[4];
                _pk[0] = __floats2bfloat162_rn(zeros[0 + 0], zeros[0 + 1]);
                _pk[1] = __floats2bfloat162_rn(zeros[0 + 2], zeros[0 + 3]);
                _pk[2] = __floats2bfloat162_rn(zeros[0 + 4], zeros[0 + 5]);
                _pk[3] = __floats2bfloat162_rn(zeros[0 + 6], zeros[0 + 7]);
                *reinterpret_cast<uint4*>(&((__nv_bfloat16*)(c_pad + dst_base))[0]) = *reinterpret_cast<uint4*>(&_pk[0]);
            }
        }
    }
}

} // extern "C"

#undef NUM_MAIN_STAGES
#undef THREADS

#define NUM_MAIN_STAGES 1
#define THREADS 256

extern "C" {

__global__ __launch_bounds__(256) void
kernel_flash_kmeans_assign_cleanroom_tcgen05_d160_pack_padded_b23d_v1(__nv_bfloat16* __restrict__ x, __nv_bfloat16* __restrict__ centroids, __nv_bfloat16* __restrict__ x_pad, __nv_bfloat16* __restrict__ c_pad, int B, int N, int D, int K, int total_x_pad, int total_c_pad)
{
    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;


    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    const int warp_id = warp;
    const int lane_id = lane;

    // === Task calls (dependency order) ===
    int grid_stride = num_bids * 256;
    int start = bid * 256 + tid;
    int total_x_vecs = total_x_pad / 8;
    int total_c_vecs = total_c_pad / 8;
    #pragma unroll 1
    for (unsigned int vec_idx = start; vec_idx < total_x_vecs; vec_idx += grid_stride) {
        int d_pad = vec_idx % 24 * 8;
        int row = vec_idx / 24;
        int dst_base = vec_idx * 8;
        if (d_pad < D) {
            float vals[8];
            {
                const uint4* _vptr_0 = reinterpret_cast<const uint4*>(x + row * D + d_pad);
                uint4 _vld_0[1];
                #pragma unroll
                for (int _blk = 0; _blk < 1; _blk++) {
                    _vld_0[_blk] = _vptr_0[_blk];
                    __nv_bfloat16* _velems_0 = reinterpret_cast<__nv_bfloat16*>(&_vld_0[_blk]);
                    #pragma unroll
                    for (int _j = 0; _j < 8; _j++)
                        vals[0 + _blk * 8 + _j] = __bfloat162float(_velems_0[_j]);
                }
            }
            {
                __nv_bfloat162 _pk[4];
                _pk[0] = __floats2bfloat162_rn(vals[0 + 0], vals[0 + 1]);
                _pk[1] = __floats2bfloat162_rn(vals[0 + 2], vals[0 + 3]);
                _pk[2] = __floats2bfloat162_rn(vals[0 + 4], vals[0 + 5]);
                _pk[3] = __floats2bfloat162_rn(vals[0 + 6], vals[0 + 7]);
                *reinterpret_cast<uint4*>(&((__nv_bfloat16*)(x_pad + dst_base))[0]) = *reinterpret_cast<uint4*>(&_pk[0]);
            }
        } else {
            float zeros[8];
            zeros[0] = 0.0f;
            zeros[1] = 0.0f;
            zeros[2] = 0.0f;
            zeros[3] = 0.0f;
            zeros[4] = 0.0f;
            zeros[5] = 0.0f;
            zeros[6] = 0.0f;
            zeros[7] = 0.0f;
            {
                __nv_bfloat162 _pk[4];
                _pk[0] = __floats2bfloat162_rn(zeros[0 + 0], zeros[0 + 1]);
                _pk[1] = __floats2bfloat162_rn(zeros[0 + 2], zeros[0 + 3]);
                _pk[2] = __floats2bfloat162_rn(zeros[0 + 4], zeros[0 + 5]);
                _pk[3] = __floats2bfloat162_rn(zeros[0 + 6], zeros[0 + 7]);
                *reinterpret_cast<uint4*>(&((__nv_bfloat16*)(x_pad + dst_base))[0]) = *reinterpret_cast<uint4*>(&_pk[0]);
            }
        }
    }
    #pragma unroll 1
    for (unsigned int vec_idx = start; vec_idx < total_c_vecs; vec_idx += grid_stride) {
        int d_pad = vec_idx % 24 * 8;
        int row = vec_idx / 24;
        int dst_base = vec_idx * 8;
        if (d_pad < D) {
            float vals[8];
            {
                const uint4* _vptr_1 = reinterpret_cast<const uint4*>(centroids + row * D + d_pad);
                uint4 _vld_1[1];
                #pragma unroll
                for (int _blk = 0; _blk < 1; _blk++) {
                    _vld_1[_blk] = _vptr_1[_blk];
                    __nv_bfloat16* _velems_1 = reinterpret_cast<__nv_bfloat16*>(&_vld_1[_blk]);
                    #pragma unroll
                    for (int _j = 0; _j < 8; _j++)
                        vals[0 + _blk * 8 + _j] = __bfloat162float(_velems_1[_j]);
                }
            }
            {
                __nv_bfloat162 _pk[4];
                _pk[0] = __floats2bfloat162_rn(vals[0 + 0], vals[0 + 1]);
                _pk[1] = __floats2bfloat162_rn(vals[0 + 2], vals[0 + 3]);
                _pk[2] = __floats2bfloat162_rn(vals[0 + 4], vals[0 + 5]);
                _pk[3] = __floats2bfloat162_rn(vals[0 + 6], vals[0 + 7]);
                *reinterpret_cast<uint4*>(&((__nv_bfloat16*)(c_pad + dst_base))[0]) = *reinterpret_cast<uint4*>(&_pk[0]);
            }
        } else {
            float zeros[8];
            zeros[0] = 0.0f;
            zeros[1] = 0.0f;
            zeros[2] = 0.0f;
            zeros[3] = 0.0f;
            zeros[4] = 0.0f;
            zeros[5] = 0.0f;
            zeros[6] = 0.0f;
            zeros[7] = 0.0f;
            {
                __nv_bfloat162 _pk[4];
                _pk[0] = __floats2bfloat162_rn(zeros[0 + 0], zeros[0 + 1]);
                _pk[1] = __floats2bfloat162_rn(zeros[0 + 2], zeros[0 + 3]);
                _pk[2] = __floats2bfloat162_rn(zeros[0 + 4], zeros[0 + 5]);
                _pk[3] = __floats2bfloat162_rn(zeros[0 + 6], zeros[0 + 7]);
                *reinterpret_cast<uint4*>(&((__nv_bfloat16*)(c_pad + dst_base))[0]) = *reinterpret_cast<uint4*>(&_pk[0]);
            }
        }
    }
}

} // extern "C"

#undef NUM_MAIN_STAGES
#undef THREADS

#define TMEM_NCOLS 512
#define TMEM_SCORE_TMEM_OFFSET 0
#define NUM_MAIN_STAGES 1
#define SMEM_SMEM_X0_OFF 1024
#define SMEM_SMEM_X0_STAGE_BYTES 40960
#define SMEM_SMEM_X0_STRIDE 40960
#define SMEM_SMEM_X1_OFF 41984
#define SMEM_SMEM_X1_STAGE_BYTES 40960
#define SMEM_SMEM_X1_STRIDE 40960
#define SMEM_SMEM_C_OFF 82944
#define SMEM_SMEM_C_STAGE_BYTES 81920
#define SMEM_SMEM_C_STRIDE 81920
#define SMEM_SMEM_CSQ_OFF 164864
#define SMEM_SMEM_CSQ_STAGE_BYTES 1024
#define SMEM_SMEM_CSQ_STRIDE 1024
#define SMEM_TOTAL 165888

extern "C" {

__global__ __launch_bounds__(192) void
kernel_flash_kmeans_assign_cleanroom_tcgen05_d160_splitd_v1(float* __restrict__ x_sq, float* __restrict__ c_sq, int32_t* __restrict__ out, const void* x_tmap, const void* c_tmap, int B, int N, int D, int K, int num_n_tiles, int K_tiles)
{
    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;

    extern __shared__ __align__(1024) char smem_raw[];
    int smem;
    smem = (int)(unsigned long long)__cvta_generic_to_shared(smem_raw);
    const int smem_smem_x0 = smem + 1024;
    const int smem_smem_x1 = smem + 41984;
    const int smem_smem_c = smem + 82944;
    const int smem_smem_csq = smem + 164864;

    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    // Mbarrier init (10 groups, 10 barriers)
    // Mbarriers at smem_raw[0..80)

    if (warp == 0) {
        uint32_t leader = elect_sync();
        // x0_full: 1 barriers, init_count=1
        mbarrier_init_pred(smem + 0, 1, leader);
        // x0_empty: 1 barriers, init_count=1
        mbarrier_init_pred(smem + 8, 1, leader);
        // x1_full: 1 barriers, init_count=1
        mbarrier_init_pred(smem + 16, 1, leader);
        // x1_empty: 1 barriers, init_count=1
        mbarrier_init_pred(smem + 24, 1, leader);
        // c_full: 1 barriers, init_count=1
        mbarrier_init_pred(smem + 32, 1, leader);
        // c_empty: 1 barriers, init_count=1
        mbarrier_init_pred(smem + 40, 1, leader);
        // score0_full: 1 barriers, init_count=1
        mbarrier_init_pred(smem + 48, 1, leader);
        // score0_empty: 1 barriers, init_count=4
        mbarrier_init_pred(smem + 56, 4, leader);
        // score1_full: 1 barriers, init_count=1
        mbarrier_init_pred(smem + 64, 1, leader);
        // score1_empty: 1 barriers, init_count=4
        mbarrier_init_pred(smem + 72, 4, leader);
        asm volatile("fence.mbarrier_init.release.cluster;");
    }

    __syncthreads();

    // TMEM alloc (512 columns, 512 used)
    volatile int* tmem_addr_storage = (volatile int*)(smem_raw + 80);
    if (warp == 1) {
        int _tmem_hold = smem + 80;
        asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;" :: "r"(_tmem_hold), "r"(512) : "memory");
    }

    __syncthreads();
    asm volatile("tcgen05.fence::after_thread_sync;");

    const int warp_id = warp;
    const int lane_id = lane;
    __nv_bfloat16* smem_x0 = (__nv_bfloat16*)(smem_raw + 1024);
    #define smem_x0_addr (smem + 1024)
    __nv_bfloat16* smem_x1 = (__nv_bfloat16*)(smem_raw + 41984);
    #define smem_x1_addr (smem + 41984)
    __nv_bfloat16* smem_c = (__nv_bfloat16*)(smem_raw + 82944);
    #define smem_c_addr (smem + 82944)
    float* smem_csq = (float*)(smem_raw + 164864);
    #define smem_csq_addr (smem + 164864)
    const int mbar_base = smem;
    #define x0_full_addr (mbar_base + 0)
    #define x0_empty_addr (mbar_base + 8)
    #define x1_full_addr (mbar_base + 16)
    #define x1_empty_addr (mbar_base + 24)
    #define c_full_addr (mbar_base + 32)
    #define c_empty_addr (mbar_base + 40)
    #define score0_full_addr (mbar_base + 48)
    #define score0_empty_addr (mbar_base + 56)
    #define score1_full_addr (mbar_base + 64)
    #define score1_empty_addr (mbar_base + 72)
    const int taddr = tmem_addr_storage[0];

    // ---- Role: load ----
    if (warp == 0) {
        { // load_main
            int pair_n_tiles = num_n_tiles / 2;
            int num_tiles = B * pair_n_tiles;
            uint32_t _phase_x0_empty_0 = 1;
            uint32_t _phase_x1_empty_0 = 1;
            uint32_t _phase_c_empty_0 = 1;
            if (warp_id == 0) {
                if (elect_sync()) {
                    #pragma unroll 1
                    for (unsigned int tile_idx = bid; tile_idx < num_tiles; tile_idx += num_bids) {
                        int batch = tile_idx / pair_n_tiles;
                        int pair_tile = tile_idx % pair_n_tiles;
                        int off_n0 = pair_tile * 256;
                        int off_n1 = off_n0 + 128;
                        int x_row0 = batch * N + off_n0;
                        int x_row1 = batch * N + off_n1;
                        mbarrier_wait(x0_empty_addr, _phase_x0_empty_0);
                        _phase_x0_empty_0 ^= 1;
                        tma_3d_gmem2smem(smem_x0_addr, x_tmap, 0, x_row0, 0, x0_full_addr);
                        mbarrier_arrive_expect_tx(x0_full_addr, 40960);
                        mbarrier_wait(x1_empty_addr, _phase_x1_empty_0);
                        _phase_x1_empty_0 ^= 1;
                        tma_3d_gmem2smem(smem_x1_addr, x_tmap, 0, x_row1, 0, x1_full_addr);
                        mbarrier_arrive_expect_tx(x1_full_addr, 40960);
                        #pragma unroll 1
                        for (int iter_k = 0; iter_k < K_tiles; iter_k++) {
                            int off_k = iter_k * 256;
                            int c_row = batch * K + off_k;
                            mbarrier_wait(c_empty_addr, _phase_c_empty_0);
                            _phase_c_empty_0 ^= 1;
                            tma_3d_gmem2smem(smem_c_addr, c_tmap, 0, c_row, 0, c_full_addr);
                            mbarrier_arrive_expect_tx(c_full_addr, 81920);
                        }
                    }
                }
            }
        }
    // ---- Role: mma ----
    } else if (warp == 1) {
        const int tmem_score_tmem = taddr + TMEM_SCORE_TMEM_OFFSET;
        { // mma_main
            int pair_n_tiles = num_n_tiles / 2;
            int num_tiles = B * pair_n_tiles;
            uint32_t _phase_x0_full_0 = 0;
            uint32_t _phase_x1_full_0 = 0;
            uint32_t _phase_score0_empty_0 = 1;
            uint32_t _phase_c_full_0 = 0;
            uint32_t _phase_score1_empty_0 = 1;
            #pragma unroll 1
            for (unsigned int tile_idx = bid; tile_idx < num_tiles; tile_idx += num_bids) {
                mbarrier_wait(x0_full_addr, _phase_x0_full_0);
                _phase_x0_full_0 ^= 1;
                mbarrier_wait(x1_full_addr, _phase_x1_full_0);
                _phase_x1_full_0 ^= 1;
                #pragma unroll 1
                for (int iter_k = 0; iter_k < K_tiles; iter_k++) {
                    mbarrier_wait(score0_empty_addr, _phase_score0_empty_0);
                    _phase_score0_empty_0 ^= 1;
                    mbarrier_wait(c_full_addr, _phase_c_full_0);
                    _phase_c_full_0 ^= 1;
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    int _desc_addr_0 = smem_x0_addr + 0 * 40960 + 16384;
                    int _desc_lo_0 = make_warp_uniform((_desc_addr_0 >> 4) & 0x3FFF);
                    int _desc_addr_1 = smem_c_addr + 0 * 81920 + 32768;
                    int _desc_lo_1 = make_warp_uniform((_desc_addr_1 >> 4) & 0x3FFF);
                    int _desc_addr_2 = smem_x0_addr + 0 * 40960 + 32768;
                    int _desc_lo_2 = make_warp_uniform((_desc_addr_2 >> 4) & 0x3FFF);
                    int _desc_addr_3 = smem_c_addr + 0 * 81920 + 65536;
                    int _desc_lo_3 = make_warp_uniform((_desc_addr_3 >> 4) & 0x3FFF);
                    int _mma_ss_a_addr_4 = smem_x0_addr + 0 * 40960;
                    int _mma_ss_a_lo_4 = make_warp_uniform((_mma_ss_a_addr_4 >> 4) & 0x3FFF);
                    int _mma_ss_b_addr_4 = smem_c_addr + 0 * 81920;
                    int _mma_ss_b_lo_4 = make_warp_uniform((_mma_ss_b_addr_4 >> 4) & 0x3FFF);
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
                    "mov.b32 id, 138413200;\n\t"
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
                    :: "r"(_mma_ss_a_lo_4), "r"(_mma_ss_b_lo_4), "r"((tmem_score_tmem + (256))), "r"(0));
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
                    "mov.b32 id, 138413200;\n\t"
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
                    :: "r"(_desc_lo_0), "r"(_desc_lo_1), "r"((tmem_score_tmem + (256))), "r"(1));
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
                    "mov.b32 id, 138413200;\n\t"
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
                    "}\n"
                    :: "r"(_desc_lo_2), "r"(_desc_lo_3), "r"((tmem_score_tmem + (256))), "r"(1));
                    elect_commit(score0_full_addr);
                    mbarrier_wait(score1_empty_addr, _phase_score1_empty_0);
                    _phase_score1_empty_0 ^= 1;
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    int _desc_addr_7 = smem_x1_addr + 0 * 40960 + 16384;
                    int _desc_lo_7 = make_warp_uniform((_desc_addr_7 >> 4) & 0x3FFF);
                    int _desc_addr_8 = smem_x1_addr + 0 * 40960 + 32768;
                    int _desc_lo_8 = make_warp_uniform((_desc_addr_8 >> 4) & 0x3FFF);
                    int _mma_ss_a_addr_9 = smem_x1_addr + 0 * 40960;
                    int _mma_ss_a_lo_9 = make_warp_uniform((_mma_ss_a_addr_9 >> 4) & 0x3FFF);
                    int _mma_ss_b_addr_9 = smem_c_addr + 0 * 81920;
                    int _mma_ss_b_lo_9 = make_warp_uniform((_mma_ss_b_addr_9 >> 4) & 0x3FFF);
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
                    "mov.b32 id, 138413200;\n\t"
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
                    :: "r"(_mma_ss_a_lo_9), "r"(_mma_ss_b_lo_9), "r"(tmem_score_tmem), "r"(0));
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
                    "mov.b32 id, 138413200;\n\t"
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
                    :: "r"(_desc_lo_7), "r"(_desc_lo_1), "r"(tmem_score_tmem), "r"(1));
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
                    "mov.b32 id, 138413200;\n\t"
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
                    "}\n"
                    :: "r"(_desc_lo_8), "r"(_desc_lo_3), "r"(tmem_score_tmem), "r"(1));
                    elect_commit(score1_full_addr);
                    elect_commit(c_empty_addr);
                }
                elect_commit(x0_empty_addr);
                elect_commit(x1_empty_addr);
            }
        }
    // ---- Role: compute ----
    } else if (warp >= 2 && warp <= 5) {
        const int warp_id_in_wg = warp % 4;
        const int my_row = warp_id_in_wg * 32 + lane;
        const int tmem_row_base_v = (warp_id_in_wg * 32) << 16;
        { // compute_main
            int pair_n_tiles = num_n_tiles / 2;
            int num_tiles = B * pair_n_tiles;
            uint32_t _phase_score0_full_0 = 0;
            uint32_t _phase_score1_full_0 = 0;
            #pragma unroll 1
            for (unsigned int tile_idx = bid; tile_idx < num_tiles; tile_idx += num_bids) {
                int batch = tile_idx / pair_n_tiles;
                int pair_tile = tile_idx % pair_n_tiles;
                int off_n0 = pair_tile * 256;
                int off_n1 = off_n0 + 128;
                int global_n0 = off_n0 + my_row;
                int global_n1 = off_n1 + my_row;
                int out_offset0 = batch * N + global_n0;
                int out_offset1 = batch * N + global_n1;
                int csq_smem_addr = smem_csq_addr + 0;
                float best_score0 = -3.4e+38f;
                int best_idx0 = 0;
                float best_score1 = -3.4e+38f;
                int best_idx1 = 0;
                #pragma unroll 1
                for (int iter_k = 0; iter_k < K_tiles; iter_k++) {
                    int off_k = iter_k * 256;
                    if (my_row < 64.0f) {
                        int csq_base = my_row * 4;
                        float csq_pack[4];
                        {
                            float4 _v4 = *reinterpret_cast<const float4*>(c_sq + batch * K + off_k + csq_base);
                            csq_pack[0 + 0] = _v4.x;
                            csq_pack[0 + 1] = _v4.y;
                            csq_pack[0 + 2] = _v4.z;
                            csq_pack[0 + 3] = _v4.w;
                        }
                        float csq_half[4];
                        csq_half[0] = 0.5f * csq_pack[0];
                        csq_half[1] = 0.5f * csq_pack[1];
                        csq_half[2] = 0.5f * csq_pack[2];
                        csq_half[3] = 0.5f * csq_pack[3];
                        asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(csq_smem_addr + csq_base * 4), "f"(csq_half[0]), "f"(csq_half[1]), "f"(csq_half[2]), "f"(csq_half[3]) : "memory");
                    }
                    asm volatile("barrier.sync 8, 128;");
                    mbarrier_wait(score0_full_addr, _phase_score0_full_0);
                    _phase_score0_full_0 ^= 1;
                    int score_base = 0;
                    float scores[128];
                    asm volatile(
                        "tcgen05.ld.sync.aligned.32x32b.x128.b32"
                        " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63, %64, %65, %66, %67, %68, %69, %70, %71, %72, %73, %74, %75, %76, %77, %78, %79, %80, %81, %82, %83, %84, %85, %86, %87, %88, %89, %90, %91, %92, %93, %94, %95, %96, %97, %98, %99, %100, %101, %102, %103, %104, %105, %106, %107, %108, %109, %110, %111, %112, %113, %114, %115, %116, %117, %118, %119, %120, %121, %122, %123, %124, %125, %126, %127}, [%128];"
                        : "=f"(scores[0]), "=f"(scores[1]), "=f"(scores[2]), "=f"(scores[3]), "=f"(scores[4]), "=f"(scores[5]), "=f"(scores[6]), "=f"(scores[7]), "=f"(scores[8]), "=f"(scores[9]), "=f"(scores[10]), "=f"(scores[11]), "=f"(scores[12]), "=f"(scores[13]), "=f"(scores[14]), "=f"(scores[15]), "=f"(scores[16]), "=f"(scores[17]), "=f"(scores[18]), "=f"(scores[19]), "=f"(scores[20]), "=f"(scores[21]), "=f"(scores[22]), "=f"(scores[23]), "=f"(scores[24]), "=f"(scores[25]), "=f"(scores[26]), "=f"(scores[27]), "=f"(scores[28]), "=f"(scores[29]), "=f"(scores[30]), "=f"(scores[31]), "=f"(scores[32]), "=f"(scores[33]), "=f"(scores[34]), "=f"(scores[35]), "=f"(scores[36]), "=f"(scores[37]), "=f"(scores[38]), "=f"(scores[39]), "=f"(scores[40]), "=f"(scores[41]), "=f"(scores[42]), "=f"(scores[43]), "=f"(scores[44]), "=f"(scores[45]), "=f"(scores[46]), "=f"(scores[47]), "=f"(scores[48]), "=f"(scores[49]), "=f"(scores[50]), "=f"(scores[51]), "=f"(scores[52]), "=f"(scores[53]), "=f"(scores[54]), "=f"(scores[55]), "=f"(scores[56]), "=f"(scores[57]), "=f"(scores[58]), "=f"(scores[59]), "=f"(scores[60]), "=f"(scores[61]), "=f"(scores[62]), "=f"(scores[63]), "=f"(scores[64]), "=f"(scores[65]), "=f"(scores[66]), "=f"(scores[67]), "=f"(scores[68]), "=f"(scores[69]), "=f"(scores[70]), "=f"(scores[71]), "=f"(scores[72]), "=f"(scores[73]), "=f"(scores[74]), "=f"(scores[75]), "=f"(scores[76]), "=f"(scores[77]), "=f"(scores[78]), "=f"(scores[79]), "=f"(scores[80]), "=f"(scores[81]), "=f"(scores[82]), "=f"(scores[83]), "=f"(scores[84]), "=f"(scores[85]), "=f"(scores[86]), "=f"(scores[87]), "=f"(scores[88]), "=f"(scores[89]), "=f"(scores[90]), "=f"(scores[91]), "=f"(scores[92]), "=f"(scores[93]), "=f"(scores[94]), "=f"(scores[95]), "=f"(scores[96]), "=f"(scores[97]), "=f"(scores[98]), "=f"(scores[99]), "=f"(scores[100]), "=f"(scores[101]), "=f"(scores[102]), "=f"(scores[103]), "=f"(scores[104]), "=f"(scores[105]), "=f"(scores[106]), "=f"(scores[107]), "=f"(scores[108]), "=f"(scores[109]), "=f"(scores[110]), "=f"(scores[111]), "=f"(scores[112]), "=f"(scores[113]), "=f"(scores[114]), "=f"(scores[115]), "=f"(scores[116]), "=f"(scores[117]), "=f"(scores[118]), "=f"(scores[119]), "=f"(scores[120]), "=f"(scores[121]), "=f"(scores[122]), "=f"(scores[123]), "=f"(scores[124]), "=f"(scores[125]), "=f"(scores[126]), "=f"(scores[127])
                        : "r"(taddr + tmem_row_base_v + 256 + score_base)
                        : "memory");
                    asm volatile("tcgen05.wait::ld.sync.aligned;");
                    #pragma unroll
                    for (int kk = 0; kk < 128; kk += 16) {
                        float csq_vals[4];
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&csq_vals[0])), "=r"(*reinterpret_cast<uint32_t*>(&csq_vals[1])), "=r"(*reinterpret_cast<uint32_t*>(&csq_vals[2])), "=r"(*reinterpret_cast<uint32_t*>(&csq_vals[3]))
                            : "r"(csq_smem_addr + (score_base + kk) * 4));
                        float d0 = scores[kk] - csq_vals[0];
                        float d1 = scores[kk + 1] - csq_vals[1];
                        float best_01 = d0;
                        int idx_01 = off_k + score_base + kk;
                        if (d1 > best_01) {
                            best_01 = d1;
                            idx_01 = off_k + score_base + kk + 1;
                        }
                        float d2 = scores[kk + 2] - csq_vals[2];
                        float d3 = scores[kk + 3] - csq_vals[3];
                        float best_23 = d2;
                        int idx_23 = off_k + score_base + kk + 2;
                        if (d3 > best_23) {
                            best_23 = d3;
                            idx_23 = off_k + score_base + kk + 3;
                        }
                        float best_group = best_01;
                        int idx_group = idx_01;
                        if (best_23 > best_group) {
                            best_group = best_23;
                            idx_group = idx_23;
                        }
                        float csq_vals_hi[4];
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&csq_vals_hi[0])), "=r"(*reinterpret_cast<uint32_t*>(&csq_vals_hi[1])), "=r"(*reinterpret_cast<uint32_t*>(&csq_vals_hi[2])), "=r"(*reinterpret_cast<uint32_t*>(&csq_vals_hi[3]))
                            : "r"(csq_smem_addr + (score_base + kk + 4) * 4));
                        float d4 = scores[kk + 4] - csq_vals_hi[0];
                        float d5 = scores[kk + 5] - csq_vals_hi[1];
                        float best_45 = d4;
                        int idx_45 = off_k + score_base + kk + 4;
                        if (d5 > best_45) {
                            best_45 = d5;
                            idx_45 = off_k + score_base + kk + 5;
                        }
                        float d6 = scores[kk + 6] - csq_vals_hi[2];
                        float d7 = scores[kk + 7] - csq_vals_hi[3];
                        float best_67 = d6;
                        int idx_67 = off_k + score_base + kk + 6;
                        if (d7 > best_67) {
                            best_67 = d7;
                            idx_67 = off_k + score_base + kk + 7;
                        }
                        float best_hi = best_45;
                        int idx_hi = idx_45;
                        if (best_67 > best_hi) {
                            best_hi = best_67;
                            idx_hi = idx_67;
                        }
                        if (best_hi > best_group) {
                            best_group = best_hi;
                            idx_group = idx_hi;
                        }
                        float csq_vals_next[4];
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&csq_vals_next[0])), "=r"(*reinterpret_cast<uint32_t*>(&csq_vals_next[1])), "=r"(*reinterpret_cast<uint32_t*>(&csq_vals_next[2])), "=r"(*reinterpret_cast<uint32_t*>(&csq_vals_next[3]))
                            : "r"(csq_smem_addr + (score_base + kk + 8) * 4));
                        float d8 = scores[kk + 8] - csq_vals_next[0];
                        float d9 = scores[kk + 9] - csq_vals_next[1];
                        float best_89 = d8;
                        int idx_89 = off_k + score_base + kk + 8;
                        if (d9 > best_89) {
                            best_89 = d9;
                            idx_89 = off_k + score_base + kk + 9;
                        }
                        float d10 = scores[kk + 10] - csq_vals_next[2];
                        float d11 = scores[kk + 11] - csq_vals_next[3];
                        float best_1011 = d10;
                        int idx_1011 = off_k + score_base + kk + 10;
                        if (d11 > best_1011) {
                            best_1011 = d11;
                            idx_1011 = off_k + score_base + kk + 11;
                        }
                        float best_next = best_89;
                        int idx_next = idx_89;
                        if (best_1011 > best_next) {
                            best_next = best_1011;
                            idx_next = idx_1011;
                        }
                        float csq_vals_tail[4];
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&csq_vals_tail[0])), "=r"(*reinterpret_cast<uint32_t*>(&csq_vals_tail[1])), "=r"(*reinterpret_cast<uint32_t*>(&csq_vals_tail[2])), "=r"(*reinterpret_cast<uint32_t*>(&csq_vals_tail[3]))
                            : "r"(csq_smem_addr + (score_base + kk + 12) * 4));
                        float d12 = scores[kk + 12] - csq_vals_tail[0];
                        float d13 = scores[kk + 13] - csq_vals_tail[1];
                        float best_1213 = d12;
                        int idx_1213 = off_k + score_base + kk + 12;
                        if (d13 > best_1213) {
                            best_1213 = d13;
                            idx_1213 = off_k + score_base + kk + 13;
                        }
                        float d14 = scores[kk + 14] - csq_vals_tail[2];
                        float d15 = scores[kk + 15] - csq_vals_tail[3];
                        float best_1415 = d14;
                        int idx_1415 = off_k + score_base + kk + 14;
                        if (d15 > best_1415) {
                            best_1415 = d15;
                            idx_1415 = off_k + score_base + kk + 15;
                        }
                        float best_tail = best_1213;
                        int idx_tail = idx_1213;
                        if (best_1415 > best_tail) {
                            best_tail = best_1415;
                            idx_tail = idx_1415;
                        }
                        if (best_tail > best_next) {
                            best_next = best_tail;
                            idx_next = idx_tail;
                        }
                        if (best_next > best_group) {
                            best_group = best_next;
                            idx_group = idx_next;
                        }
                        if (best_group > best_score0) {
                            best_score0 = best_group;
                            best_idx0 = idx_group;
                        }
                    }
                    score_base = 128;
                    asm volatile(
                        "tcgen05.ld.sync.aligned.32x32b.x128.b32"
                        " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63, %64, %65, %66, %67, %68, %69, %70, %71, %72, %73, %74, %75, %76, %77, %78, %79, %80, %81, %82, %83, %84, %85, %86, %87, %88, %89, %90, %91, %92, %93, %94, %95, %96, %97, %98, %99, %100, %101, %102, %103, %104, %105, %106, %107, %108, %109, %110, %111, %112, %113, %114, %115, %116, %117, %118, %119, %120, %121, %122, %123, %124, %125, %126, %127}, [%128];"
                        : "=f"(scores[0]), "=f"(scores[1]), "=f"(scores[2]), "=f"(scores[3]), "=f"(scores[4]), "=f"(scores[5]), "=f"(scores[6]), "=f"(scores[7]), "=f"(scores[8]), "=f"(scores[9]), "=f"(scores[10]), "=f"(scores[11]), "=f"(scores[12]), "=f"(scores[13]), "=f"(scores[14]), "=f"(scores[15]), "=f"(scores[16]), "=f"(scores[17]), "=f"(scores[18]), "=f"(scores[19]), "=f"(scores[20]), "=f"(scores[21]), "=f"(scores[22]), "=f"(scores[23]), "=f"(scores[24]), "=f"(scores[25]), "=f"(scores[26]), "=f"(scores[27]), "=f"(scores[28]), "=f"(scores[29]), "=f"(scores[30]), "=f"(scores[31]), "=f"(scores[32]), "=f"(scores[33]), "=f"(scores[34]), "=f"(scores[35]), "=f"(scores[36]), "=f"(scores[37]), "=f"(scores[38]), "=f"(scores[39]), "=f"(scores[40]), "=f"(scores[41]), "=f"(scores[42]), "=f"(scores[43]), "=f"(scores[44]), "=f"(scores[45]), "=f"(scores[46]), "=f"(scores[47]), "=f"(scores[48]), "=f"(scores[49]), "=f"(scores[50]), "=f"(scores[51]), "=f"(scores[52]), "=f"(scores[53]), "=f"(scores[54]), "=f"(scores[55]), "=f"(scores[56]), "=f"(scores[57]), "=f"(scores[58]), "=f"(scores[59]), "=f"(scores[60]), "=f"(scores[61]), "=f"(scores[62]), "=f"(scores[63]), "=f"(scores[64]), "=f"(scores[65]), "=f"(scores[66]), "=f"(scores[67]), "=f"(scores[68]), "=f"(scores[69]), "=f"(scores[70]), "=f"(scores[71]), "=f"(scores[72]), "=f"(scores[73]), "=f"(scores[74]), "=f"(scores[75]), "=f"(scores[76]), "=f"(scores[77]), "=f"(scores[78]), "=f"(scores[79]), "=f"(scores[80]), "=f"(scores[81]), "=f"(scores[82]), "=f"(scores[83]), "=f"(scores[84]), "=f"(scores[85]), "=f"(scores[86]), "=f"(scores[87]), "=f"(scores[88]), "=f"(scores[89]), "=f"(scores[90]), "=f"(scores[91]), "=f"(scores[92]), "=f"(scores[93]), "=f"(scores[94]), "=f"(scores[95]), "=f"(scores[96]), "=f"(scores[97]), "=f"(scores[98]), "=f"(scores[99]), "=f"(scores[100]), "=f"(scores[101]), "=f"(scores[102]), "=f"(scores[103]), "=f"(scores[104]), "=f"(scores[105]), "=f"(scores[106]), "=f"(scores[107]), "=f"(scores[108]), "=f"(scores[109]), "=f"(scores[110]), "=f"(scores[111]), "=f"(scores[112]), "=f"(scores[113]), "=f"(scores[114]), "=f"(scores[115]), "=f"(scores[116]), "=f"(scores[117]), "=f"(scores[118]), "=f"(scores[119]), "=f"(scores[120]), "=f"(scores[121]), "=f"(scores[122]), "=f"(scores[123]), "=f"(scores[124]), "=f"(scores[125]), "=f"(scores[126]), "=f"(scores[127])
                        : "r"(taddr + tmem_row_base_v + 256 + score_base)
                        : "memory");
                    asm volatile("tcgen05.wait::ld.sync.aligned;");
                    asm volatile("barrier.sync 9, 128;");
                    if (elect_sync()) {
                        mbarrier_arrive(score0_empty_addr);
                    }
                    #pragma unroll
                    for (int kk = 0; kk < 128; kk += 16) {
                        float csq_vals[4];
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&csq_vals[0])), "=r"(*reinterpret_cast<uint32_t*>(&csq_vals[1])), "=r"(*reinterpret_cast<uint32_t*>(&csq_vals[2])), "=r"(*reinterpret_cast<uint32_t*>(&csq_vals[3]))
                            : "r"(csq_smem_addr + (score_base + kk) * 4));
                        float d0 = scores[kk] - csq_vals[0];
                        float d1 = scores[kk + 1] - csq_vals[1];
                        float best_01 = d0;
                        int idx_01 = off_k + score_base + kk;
                        if (d1 > best_01) {
                            best_01 = d1;
                            idx_01 = off_k + score_base + kk + 1;
                        }
                        float d2 = scores[kk + 2] - csq_vals[2];
                        float d3 = scores[kk + 3] - csq_vals[3];
                        float best_23 = d2;
                        int idx_23 = off_k + score_base + kk + 2;
                        if (d3 > best_23) {
                            best_23 = d3;
                            idx_23 = off_k + score_base + kk + 3;
                        }
                        float best_group = best_01;
                        int idx_group = idx_01;
                        if (best_23 > best_group) {
                            best_group = best_23;
                            idx_group = idx_23;
                        }
                        float csq_vals_hi[4];
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&csq_vals_hi[0])), "=r"(*reinterpret_cast<uint32_t*>(&csq_vals_hi[1])), "=r"(*reinterpret_cast<uint32_t*>(&csq_vals_hi[2])), "=r"(*reinterpret_cast<uint32_t*>(&csq_vals_hi[3]))
                            : "r"(csq_smem_addr + (score_base + kk + 4) * 4));
                        float d4 = scores[kk + 4] - csq_vals_hi[0];
                        float d5 = scores[kk + 5] - csq_vals_hi[1];
                        float best_45 = d4;
                        int idx_45 = off_k + score_base + kk + 4;
                        if (d5 > best_45) {
                            best_45 = d5;
                            idx_45 = off_k + score_base + kk + 5;
                        }
                        float d6 = scores[kk + 6] - csq_vals_hi[2];
                        float d7 = scores[kk + 7] - csq_vals_hi[3];
                        float best_67 = d6;
                        int idx_67 = off_k + score_base + kk + 6;
                        if (d7 > best_67) {
                            best_67 = d7;
                            idx_67 = off_k + score_base + kk + 7;
                        }
                        float best_hi = best_45;
                        int idx_hi = idx_45;
                        if (best_67 > best_hi) {
                            best_hi = best_67;
                            idx_hi = idx_67;
                        }
                        if (best_hi > best_group) {
                            best_group = best_hi;
                            idx_group = idx_hi;
                        }
                        float csq_vals_next[4];
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&csq_vals_next[0])), "=r"(*reinterpret_cast<uint32_t*>(&csq_vals_next[1])), "=r"(*reinterpret_cast<uint32_t*>(&csq_vals_next[2])), "=r"(*reinterpret_cast<uint32_t*>(&csq_vals_next[3]))
                            : "r"(csq_smem_addr + (score_base + kk + 8) * 4));
                        float d8 = scores[kk + 8] - csq_vals_next[0];
                        float d9 = scores[kk + 9] - csq_vals_next[1];
                        float best_89 = d8;
                        int idx_89 = off_k + score_base + kk + 8;
                        if (d9 > best_89) {
                            best_89 = d9;
                            idx_89 = off_k + score_base + kk + 9;
                        }
                        float d10 = scores[kk + 10] - csq_vals_next[2];
                        float d11 = scores[kk + 11] - csq_vals_next[3];
                        float best_1011 = d10;
                        int idx_1011 = off_k + score_base + kk + 10;
                        if (d11 > best_1011) {
                            best_1011 = d11;
                            idx_1011 = off_k + score_base + kk + 11;
                        }
                        float best_next = best_89;
                        int idx_next = idx_89;
                        if (best_1011 > best_next) {
                            best_next = best_1011;
                            idx_next = idx_1011;
                        }
                        float csq_vals_tail[4];
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&csq_vals_tail[0])), "=r"(*reinterpret_cast<uint32_t*>(&csq_vals_tail[1])), "=r"(*reinterpret_cast<uint32_t*>(&csq_vals_tail[2])), "=r"(*reinterpret_cast<uint32_t*>(&csq_vals_tail[3]))
                            : "r"(csq_smem_addr + (score_base + kk + 12) * 4));
                        float d12 = scores[kk + 12] - csq_vals_tail[0];
                        float d13 = scores[kk + 13] - csq_vals_tail[1];
                        float best_1213 = d12;
                        int idx_1213 = off_k + score_base + kk + 12;
                        if (d13 > best_1213) {
                            best_1213 = d13;
                            idx_1213 = off_k + score_base + kk + 13;
                        }
                        float d14 = scores[kk + 14] - csq_vals_tail[2];
                        float d15 = scores[kk + 15] - csq_vals_tail[3];
                        float best_1415 = d14;
                        int idx_1415 = off_k + score_base + kk + 14;
                        if (d15 > best_1415) {
                            best_1415 = d15;
                            idx_1415 = off_k + score_base + kk + 15;
                        }
                        float best_tail = best_1213;
                        int idx_tail = idx_1213;
                        if (best_1415 > best_tail) {
                            best_tail = best_1415;
                            idx_tail = idx_1415;
                        }
                        if (best_tail > best_next) {
                            best_next = best_tail;
                            idx_next = idx_tail;
                        }
                        if (best_next > best_group) {
                            best_group = best_next;
                            idx_group = idx_next;
                        }
                        if (best_group > best_score0) {
                            best_score0 = best_group;
                            best_idx0 = idx_group;
                        }
                    }
                    mbarrier_wait(score1_full_addr, _phase_score1_full_0);
                    _phase_score1_full_0 ^= 1;
                    score_base = 0;
                    asm volatile(
                        "tcgen05.ld.sync.aligned.32x32b.x128.b32"
                        " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63, %64, %65, %66, %67, %68, %69, %70, %71, %72, %73, %74, %75, %76, %77, %78, %79, %80, %81, %82, %83, %84, %85, %86, %87, %88, %89, %90, %91, %92, %93, %94, %95, %96, %97, %98, %99, %100, %101, %102, %103, %104, %105, %106, %107, %108, %109, %110, %111, %112, %113, %114, %115, %116, %117, %118, %119, %120, %121, %122, %123, %124, %125, %126, %127}, [%128];"
                        : "=f"(scores[0]), "=f"(scores[1]), "=f"(scores[2]), "=f"(scores[3]), "=f"(scores[4]), "=f"(scores[5]), "=f"(scores[6]), "=f"(scores[7]), "=f"(scores[8]), "=f"(scores[9]), "=f"(scores[10]), "=f"(scores[11]), "=f"(scores[12]), "=f"(scores[13]), "=f"(scores[14]), "=f"(scores[15]), "=f"(scores[16]), "=f"(scores[17]), "=f"(scores[18]), "=f"(scores[19]), "=f"(scores[20]), "=f"(scores[21]), "=f"(scores[22]), "=f"(scores[23]), "=f"(scores[24]), "=f"(scores[25]), "=f"(scores[26]), "=f"(scores[27]), "=f"(scores[28]), "=f"(scores[29]), "=f"(scores[30]), "=f"(scores[31]), "=f"(scores[32]), "=f"(scores[33]), "=f"(scores[34]), "=f"(scores[35]), "=f"(scores[36]), "=f"(scores[37]), "=f"(scores[38]), "=f"(scores[39]), "=f"(scores[40]), "=f"(scores[41]), "=f"(scores[42]), "=f"(scores[43]), "=f"(scores[44]), "=f"(scores[45]), "=f"(scores[46]), "=f"(scores[47]), "=f"(scores[48]), "=f"(scores[49]), "=f"(scores[50]), "=f"(scores[51]), "=f"(scores[52]), "=f"(scores[53]), "=f"(scores[54]), "=f"(scores[55]), "=f"(scores[56]), "=f"(scores[57]), "=f"(scores[58]), "=f"(scores[59]), "=f"(scores[60]), "=f"(scores[61]), "=f"(scores[62]), "=f"(scores[63]), "=f"(scores[64]), "=f"(scores[65]), "=f"(scores[66]), "=f"(scores[67]), "=f"(scores[68]), "=f"(scores[69]), "=f"(scores[70]), "=f"(scores[71]), "=f"(scores[72]), "=f"(scores[73]), "=f"(scores[74]), "=f"(scores[75]), "=f"(scores[76]), "=f"(scores[77]), "=f"(scores[78]), "=f"(scores[79]), "=f"(scores[80]), "=f"(scores[81]), "=f"(scores[82]), "=f"(scores[83]), "=f"(scores[84]), "=f"(scores[85]), "=f"(scores[86]), "=f"(scores[87]), "=f"(scores[88]), "=f"(scores[89]), "=f"(scores[90]), "=f"(scores[91]), "=f"(scores[92]), "=f"(scores[93]), "=f"(scores[94]), "=f"(scores[95]), "=f"(scores[96]), "=f"(scores[97]), "=f"(scores[98]), "=f"(scores[99]), "=f"(scores[100]), "=f"(scores[101]), "=f"(scores[102]), "=f"(scores[103]), "=f"(scores[104]), "=f"(scores[105]), "=f"(scores[106]), "=f"(scores[107]), "=f"(scores[108]), "=f"(scores[109]), "=f"(scores[110]), "=f"(scores[111]), "=f"(scores[112]), "=f"(scores[113]), "=f"(scores[114]), "=f"(scores[115]), "=f"(scores[116]), "=f"(scores[117]), "=f"(scores[118]), "=f"(scores[119]), "=f"(scores[120]), "=f"(scores[121]), "=f"(scores[122]), "=f"(scores[123]), "=f"(scores[124]), "=f"(scores[125]), "=f"(scores[126]), "=f"(scores[127])
                        : "r"(taddr + tmem_row_base_v + score_base)
                        : "memory");
                    asm volatile("tcgen05.wait::ld.sync.aligned;");
                    #pragma unroll
                    for (int kk = 0; kk < 128; kk += 16) {
                        float csq_vals[4];
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&csq_vals[0])), "=r"(*reinterpret_cast<uint32_t*>(&csq_vals[1])), "=r"(*reinterpret_cast<uint32_t*>(&csq_vals[2])), "=r"(*reinterpret_cast<uint32_t*>(&csq_vals[3]))
                            : "r"(csq_smem_addr + (score_base + kk) * 4));
                        float d0 = scores[kk] - csq_vals[0];
                        float d1 = scores[kk + 1] - csq_vals[1];
                        float best_01 = d0;
                        int idx_01 = off_k + score_base + kk;
                        if (d1 > best_01) {
                            best_01 = d1;
                            idx_01 = off_k + score_base + kk + 1;
                        }
                        float d2 = scores[kk + 2] - csq_vals[2];
                        float d3 = scores[kk + 3] - csq_vals[3];
                        float best_23 = d2;
                        int idx_23 = off_k + score_base + kk + 2;
                        if (d3 > best_23) {
                            best_23 = d3;
                            idx_23 = off_k + score_base + kk + 3;
                        }
                        float best_group = best_01;
                        int idx_group = idx_01;
                        if (best_23 > best_group) {
                            best_group = best_23;
                            idx_group = idx_23;
                        }
                        float csq_vals_hi[4];
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&csq_vals_hi[0])), "=r"(*reinterpret_cast<uint32_t*>(&csq_vals_hi[1])), "=r"(*reinterpret_cast<uint32_t*>(&csq_vals_hi[2])), "=r"(*reinterpret_cast<uint32_t*>(&csq_vals_hi[3]))
                            : "r"(csq_smem_addr + (score_base + kk + 4) * 4));
                        float d4 = scores[kk + 4] - csq_vals_hi[0];
                        float d5 = scores[kk + 5] - csq_vals_hi[1];
                        float best_45 = d4;
                        int idx_45 = off_k + score_base + kk + 4;
                        if (d5 > best_45) {
                            best_45 = d5;
                            idx_45 = off_k + score_base + kk + 5;
                        }
                        float d6 = scores[kk + 6] - csq_vals_hi[2];
                        float d7 = scores[kk + 7] - csq_vals_hi[3];
                        float best_67 = d6;
                        int idx_67 = off_k + score_base + kk + 6;
                        if (d7 > best_67) {
                            best_67 = d7;
                            idx_67 = off_k + score_base + kk + 7;
                        }
                        float best_hi = best_45;
                        int idx_hi = idx_45;
                        if (best_67 > best_hi) {
                            best_hi = best_67;
                            idx_hi = idx_67;
                        }
                        if (best_hi > best_group) {
                            best_group = best_hi;
                            idx_group = idx_hi;
                        }
                        float csq_vals_next[4];
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&csq_vals_next[0])), "=r"(*reinterpret_cast<uint32_t*>(&csq_vals_next[1])), "=r"(*reinterpret_cast<uint32_t*>(&csq_vals_next[2])), "=r"(*reinterpret_cast<uint32_t*>(&csq_vals_next[3]))
                            : "r"(csq_smem_addr + (score_base + kk + 8) * 4));
                        float d8 = scores[kk + 8] - csq_vals_next[0];
                        float d9 = scores[kk + 9] - csq_vals_next[1];
                        float best_89 = d8;
                        int idx_89 = off_k + score_base + kk + 8;
                        if (d9 > best_89) {
                            best_89 = d9;
                            idx_89 = off_k + score_base + kk + 9;
                        }
                        float d10 = scores[kk + 10] - csq_vals_next[2];
                        float d11 = scores[kk + 11] - csq_vals_next[3];
                        float best_1011 = d10;
                        int idx_1011 = off_k + score_base + kk + 10;
                        if (d11 > best_1011) {
                            best_1011 = d11;
                            idx_1011 = off_k + score_base + kk + 11;
                        }
                        float best_next = best_89;
                        int idx_next = idx_89;
                        if (best_1011 > best_next) {
                            best_next = best_1011;
                            idx_next = idx_1011;
                        }
                        float csq_vals_tail[4];
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&csq_vals_tail[0])), "=r"(*reinterpret_cast<uint32_t*>(&csq_vals_tail[1])), "=r"(*reinterpret_cast<uint32_t*>(&csq_vals_tail[2])), "=r"(*reinterpret_cast<uint32_t*>(&csq_vals_tail[3]))
                            : "r"(csq_smem_addr + (score_base + kk + 12) * 4));
                        float d12 = scores[kk + 12] - csq_vals_tail[0];
                        float d13 = scores[kk + 13] - csq_vals_tail[1];
                        float best_1213 = d12;
                        int idx_1213 = off_k + score_base + kk + 12;
                        if (d13 > best_1213) {
                            best_1213 = d13;
                            idx_1213 = off_k + score_base + kk + 13;
                        }
                        float d14 = scores[kk + 14] - csq_vals_tail[2];
                        float d15 = scores[kk + 15] - csq_vals_tail[3];
                        float best_1415 = d14;
                        int idx_1415 = off_k + score_base + kk + 14;
                        if (d15 > best_1415) {
                            best_1415 = d15;
                            idx_1415 = off_k + score_base + kk + 15;
                        }
                        float best_tail = best_1213;
                        int idx_tail = idx_1213;
                        if (best_1415 > best_tail) {
                            best_tail = best_1415;
                            idx_tail = idx_1415;
                        }
                        if (best_tail > best_next) {
                            best_next = best_tail;
                            idx_next = idx_tail;
                        }
                        if (best_next > best_group) {
                            best_group = best_next;
                            idx_group = idx_next;
                        }
                        if (best_group > best_score1) {
                            best_score1 = best_group;
                            best_idx1 = idx_group;
                        }
                    }
                    score_base = 128;
                    asm volatile(
                        "tcgen05.ld.sync.aligned.32x32b.x128.b32"
                        " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63, %64, %65, %66, %67, %68, %69, %70, %71, %72, %73, %74, %75, %76, %77, %78, %79, %80, %81, %82, %83, %84, %85, %86, %87, %88, %89, %90, %91, %92, %93, %94, %95, %96, %97, %98, %99, %100, %101, %102, %103, %104, %105, %106, %107, %108, %109, %110, %111, %112, %113, %114, %115, %116, %117, %118, %119, %120, %121, %122, %123, %124, %125, %126, %127}, [%128];"
                        : "=f"(scores[0]), "=f"(scores[1]), "=f"(scores[2]), "=f"(scores[3]), "=f"(scores[4]), "=f"(scores[5]), "=f"(scores[6]), "=f"(scores[7]), "=f"(scores[8]), "=f"(scores[9]), "=f"(scores[10]), "=f"(scores[11]), "=f"(scores[12]), "=f"(scores[13]), "=f"(scores[14]), "=f"(scores[15]), "=f"(scores[16]), "=f"(scores[17]), "=f"(scores[18]), "=f"(scores[19]), "=f"(scores[20]), "=f"(scores[21]), "=f"(scores[22]), "=f"(scores[23]), "=f"(scores[24]), "=f"(scores[25]), "=f"(scores[26]), "=f"(scores[27]), "=f"(scores[28]), "=f"(scores[29]), "=f"(scores[30]), "=f"(scores[31]), "=f"(scores[32]), "=f"(scores[33]), "=f"(scores[34]), "=f"(scores[35]), "=f"(scores[36]), "=f"(scores[37]), "=f"(scores[38]), "=f"(scores[39]), "=f"(scores[40]), "=f"(scores[41]), "=f"(scores[42]), "=f"(scores[43]), "=f"(scores[44]), "=f"(scores[45]), "=f"(scores[46]), "=f"(scores[47]), "=f"(scores[48]), "=f"(scores[49]), "=f"(scores[50]), "=f"(scores[51]), "=f"(scores[52]), "=f"(scores[53]), "=f"(scores[54]), "=f"(scores[55]), "=f"(scores[56]), "=f"(scores[57]), "=f"(scores[58]), "=f"(scores[59]), "=f"(scores[60]), "=f"(scores[61]), "=f"(scores[62]), "=f"(scores[63]), "=f"(scores[64]), "=f"(scores[65]), "=f"(scores[66]), "=f"(scores[67]), "=f"(scores[68]), "=f"(scores[69]), "=f"(scores[70]), "=f"(scores[71]), "=f"(scores[72]), "=f"(scores[73]), "=f"(scores[74]), "=f"(scores[75]), "=f"(scores[76]), "=f"(scores[77]), "=f"(scores[78]), "=f"(scores[79]), "=f"(scores[80]), "=f"(scores[81]), "=f"(scores[82]), "=f"(scores[83]), "=f"(scores[84]), "=f"(scores[85]), "=f"(scores[86]), "=f"(scores[87]), "=f"(scores[88]), "=f"(scores[89]), "=f"(scores[90]), "=f"(scores[91]), "=f"(scores[92]), "=f"(scores[93]), "=f"(scores[94]), "=f"(scores[95]), "=f"(scores[96]), "=f"(scores[97]), "=f"(scores[98]), "=f"(scores[99]), "=f"(scores[100]), "=f"(scores[101]), "=f"(scores[102]), "=f"(scores[103]), "=f"(scores[104]), "=f"(scores[105]), "=f"(scores[106]), "=f"(scores[107]), "=f"(scores[108]), "=f"(scores[109]), "=f"(scores[110]), "=f"(scores[111]), "=f"(scores[112]), "=f"(scores[113]), "=f"(scores[114]), "=f"(scores[115]), "=f"(scores[116]), "=f"(scores[117]), "=f"(scores[118]), "=f"(scores[119]), "=f"(scores[120]), "=f"(scores[121]), "=f"(scores[122]), "=f"(scores[123]), "=f"(scores[124]), "=f"(scores[125]), "=f"(scores[126]), "=f"(scores[127])
                        : "r"(taddr + tmem_row_base_v + score_base)
                        : "memory");
                    asm volatile("tcgen05.wait::ld.sync.aligned;");
                    asm volatile("barrier.sync 9, 128;");
                    if (elect_sync()) {
                        mbarrier_arrive(score1_empty_addr);
                    }
                    #pragma unroll
                    for (int kk = 0; kk < 128; kk += 16) {
                        float csq_vals[4];
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&csq_vals[0])), "=r"(*reinterpret_cast<uint32_t*>(&csq_vals[1])), "=r"(*reinterpret_cast<uint32_t*>(&csq_vals[2])), "=r"(*reinterpret_cast<uint32_t*>(&csq_vals[3]))
                            : "r"(csq_smem_addr + (score_base + kk) * 4));
                        float d0 = scores[kk] - csq_vals[0];
                        float d1 = scores[kk + 1] - csq_vals[1];
                        float best_01 = d0;
                        int idx_01 = off_k + score_base + kk;
                        if (d1 > best_01) {
                            best_01 = d1;
                            idx_01 = off_k + score_base + kk + 1;
                        }
                        float d2 = scores[kk + 2] - csq_vals[2];
                        float d3 = scores[kk + 3] - csq_vals[3];
                        float best_23 = d2;
                        int idx_23 = off_k + score_base + kk + 2;
                        if (d3 > best_23) {
                            best_23 = d3;
                            idx_23 = off_k + score_base + kk + 3;
                        }
                        float best_group = best_01;
                        int idx_group = idx_01;
                        if (best_23 > best_group) {
                            best_group = best_23;
                            idx_group = idx_23;
                        }
                        float csq_vals_hi[4];
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&csq_vals_hi[0])), "=r"(*reinterpret_cast<uint32_t*>(&csq_vals_hi[1])), "=r"(*reinterpret_cast<uint32_t*>(&csq_vals_hi[2])), "=r"(*reinterpret_cast<uint32_t*>(&csq_vals_hi[3]))
                            : "r"(csq_smem_addr + (score_base + kk + 4) * 4));
                        float d4 = scores[kk + 4] - csq_vals_hi[0];
                        float d5 = scores[kk + 5] - csq_vals_hi[1];
                        float best_45 = d4;
                        int idx_45 = off_k + score_base + kk + 4;
                        if (d5 > best_45) {
                            best_45 = d5;
                            idx_45 = off_k + score_base + kk + 5;
                        }
                        float d6 = scores[kk + 6] - csq_vals_hi[2];
                        float d7 = scores[kk + 7] - csq_vals_hi[3];
                        float best_67 = d6;
                        int idx_67 = off_k + score_base + kk + 6;
                        if (d7 > best_67) {
                            best_67 = d7;
                            idx_67 = off_k + score_base + kk + 7;
                        }
                        float best_hi = best_45;
                        int idx_hi = idx_45;
                        if (best_67 > best_hi) {
                            best_hi = best_67;
                            idx_hi = idx_67;
                        }
                        if (best_hi > best_group) {
                            best_group = best_hi;
                            idx_group = idx_hi;
                        }
                        float csq_vals_next[4];
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&csq_vals_next[0])), "=r"(*reinterpret_cast<uint32_t*>(&csq_vals_next[1])), "=r"(*reinterpret_cast<uint32_t*>(&csq_vals_next[2])), "=r"(*reinterpret_cast<uint32_t*>(&csq_vals_next[3]))
                            : "r"(csq_smem_addr + (score_base + kk + 8) * 4));
                        float d8 = scores[kk + 8] - csq_vals_next[0];
                        float d9 = scores[kk + 9] - csq_vals_next[1];
                        float best_89 = d8;
                        int idx_89 = off_k + score_base + kk + 8;
                        if (d9 > best_89) {
                            best_89 = d9;
                            idx_89 = off_k + score_base + kk + 9;
                        }
                        float d10 = scores[kk + 10] - csq_vals_next[2];
                        float d11 = scores[kk + 11] - csq_vals_next[3];
                        float best_1011 = d10;
                        int idx_1011 = off_k + score_base + kk + 10;
                        if (d11 > best_1011) {
                            best_1011 = d11;
                            idx_1011 = off_k + score_base + kk + 11;
                        }
                        float best_next = best_89;
                        int idx_next = idx_89;
                        if (best_1011 > best_next) {
                            best_next = best_1011;
                            idx_next = idx_1011;
                        }
                        float csq_vals_tail[4];
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&csq_vals_tail[0])), "=r"(*reinterpret_cast<uint32_t*>(&csq_vals_tail[1])), "=r"(*reinterpret_cast<uint32_t*>(&csq_vals_tail[2])), "=r"(*reinterpret_cast<uint32_t*>(&csq_vals_tail[3]))
                            : "r"(csq_smem_addr + (score_base + kk + 12) * 4));
                        float d12 = scores[kk + 12] - csq_vals_tail[0];
                        float d13 = scores[kk + 13] - csq_vals_tail[1];
                        float best_1213 = d12;
                        int idx_1213 = off_k + score_base + kk + 12;
                        if (d13 > best_1213) {
                            best_1213 = d13;
                            idx_1213 = off_k + score_base + kk + 13;
                        }
                        float d14 = scores[kk + 14] - csq_vals_tail[2];
                        float d15 = scores[kk + 15] - csq_vals_tail[3];
                        float best_1415 = d14;
                        int idx_1415 = off_k + score_base + kk + 14;
                        if (d15 > best_1415) {
                            best_1415 = d15;
                            idx_1415 = off_k + score_base + kk + 15;
                        }
                        float best_tail = best_1213;
                        int idx_tail = idx_1213;
                        if (best_1415 > best_tail) {
                            best_tail = best_1415;
                            idx_tail = idx_1415;
                        }
                        if (best_tail > best_next) {
                            best_next = best_tail;
                            idx_next = idx_tail;
                        }
                        if (best_next > best_group) {
                            best_group = best_next;
                            idx_group = idx_next;
                        }
                        if (best_group > best_score1) {
                            best_score1 = best_group;
                            best_idx1 = idx_group;
                        }
                    }
                }
                if (global_n0 < N) {
                    *((int*)(out + out_offset0)) = best_idx0;
                }
                if (global_n1 < N) {
                    *((int*)(out + out_offset1)) = best_idx1;
                }
            }
        }
    }

    // Cleanup
    __syncthreads(); // barrier before TMEM dealloc

    if (warp == 1) {
        asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;" :: "r"(tmem_addr_storage[0]), "r"(512));
        asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;");
    }
}

} // extern "C"

#undef NUM_MAIN_STAGES
#undef SMEM_SMEM_CSQ_OFF
#undef SMEM_SMEM_CSQ_STAGE_BYTES
#undef SMEM_SMEM_CSQ_STRIDE
#undef SMEM_SMEM_C_OFF
#undef SMEM_SMEM_C_STAGE_BYTES
#undef SMEM_SMEM_C_STRIDE
#undef SMEM_SMEM_X0_OFF
#undef SMEM_SMEM_X0_STAGE_BYTES
#undef SMEM_SMEM_X0_STRIDE
#undef SMEM_SMEM_X1_OFF
#undef SMEM_SMEM_X1_STAGE_BYTES
#undef SMEM_SMEM_X1_STRIDE
#undef SMEM_TOTAL
#undef TMEM_NCOLS
#undef TMEM_SCORE_TMEM_OFFSET
#undef c_empty_addr
#undef c_full_addr
#undef score0_empty_addr
#undef score0_full_addr
#undef score1_empty_addr
#undef score1_full_addr
#undef smem_c_addr
#undef smem_csq_addr
#undef smem_x0_addr
#undef smem_x1_addr
#undef x0_empty_addr
#undef x0_full_addr
#undef x1_empty_addr
#undef x1_full_addr

#define TMEM_NCOLS 256
#define TMEM_SCORE_TMEM_OFFSET 0
#define NUM_MAIN_STAGES 1
#define SMEM_SMEM_X_OFF 1024
#define SMEM_SMEM_X_STAGE_BYTES 49152
#define SMEM_SMEM_X_STRIDE 49152
#define SMEM_SMEM_C_OFF 50176
#define SMEM_SMEM_C_STAGE_BYTES 98304
#define SMEM_SMEM_C_STRIDE 98304
#define SMEM_SMEM_CSQ_OFF 148480
#define SMEM_SMEM_CSQ_STAGE_BYTES 1024
#define SMEM_SMEM_CSQ_STRIDE 1024
#define SMEM_TOTAL 149504

extern "C" {

__global__ __launch_bounds__(192) void
kernel_flash_kmeans_assign_cleanroom_tcgen05_d192_single_v1(float* __restrict__ x_sq, float* __restrict__ c_sq, int32_t* __restrict__ out, const void* x_tmap, const void* c_tmap, int B, int N, int D, int K, int num_n_tiles, int K_tiles)
{
    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;

    extern __shared__ __align__(1024) char smem_raw[];
    int smem;
    smem = (int)(unsigned long long)__cvta_generic_to_shared(smem_raw);
    const int smem_smem_x = smem + 1024;
    const int smem_smem_c = smem + 50176;
    const int smem_smem_csq = smem + 148480;

    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    // Mbarrier init (6 groups, 6 barriers)
    // Mbarriers at smem_raw[0..48)

    if (warp == 0) {
        uint32_t leader = elect_sync();
        // x_full: 1 barriers, init_count=1
        mbarrier_init_pred(smem + 0, 1, leader);
        // x_empty: 1 barriers, init_count=1
        mbarrier_init_pred(smem + 8, 1, leader);
        // c_full: 1 barriers, init_count=1
        mbarrier_init_pred(smem + 16, 1, leader);
        // c_empty: 1 barriers, init_count=1
        mbarrier_init_pred(smem + 24, 1, leader);
        // score_full: 1 barriers, init_count=1
        mbarrier_init_pred(smem + 32, 1, leader);
        // score_empty: 1 barriers, init_count=4
        mbarrier_init_pred(smem + 40, 4, leader);
        asm volatile("fence.mbarrier_init.release.cluster;");
    }

    __syncthreads();

    // TMEM alloc (256 columns, 256 used)
    volatile int* tmem_addr_storage = (volatile int*)(smem_raw + 48);
    if (warp == 1) {
        int _tmem_hold = smem + 48;
        asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;" :: "r"(_tmem_hold), "r"(256) : "memory");
    }

    __syncthreads();
    asm volatile("tcgen05.fence::after_thread_sync;");

    const int warp_id = warp;
    const int lane_id = lane;
    __nv_bfloat16* smem_x = (__nv_bfloat16*)(smem_raw + 1024);
    #define smem_x_addr (smem + 1024)
    __nv_bfloat16* smem_c = (__nv_bfloat16*)(smem_raw + 50176);
    #define smem_c_addr (smem + 50176)
    float* smem_csq = (float*)(smem_raw + 148480);
    #define smem_csq_addr (smem + 148480)
    const int mbar_base = smem;
    #define x_full_addr (mbar_base + 0)
    #define x_empty_addr (mbar_base + 8)
    #define c_full_addr (mbar_base + 16)
    #define c_empty_addr (mbar_base + 24)
    #define score_full_addr (mbar_base + 32)
    #define score_empty_addr (mbar_base + 40)
    const int taddr = tmem_addr_storage[0];

    // ---- Role: load ----
    if (warp == 0) {
        { // load_main
            int num_tiles = B * num_n_tiles;
            uint32_t _phase_x_empty_0 = 1;
            uint32_t _phase_c_empty_0 = 1;
            if (warp_id == 0) {
                if (elect_sync()) {
                    #pragma unroll 1
                    for (unsigned int tile_idx = bid; tile_idx < num_tiles; tile_idx += num_bids) {
                        int batch = tile_idx / num_n_tiles;
                        int n_tile = tile_idx % num_n_tiles;
                        int off_n = n_tile * 128;
                        int x_row = batch * N + off_n;
                        mbarrier_wait(x_empty_addr, _phase_x_empty_0);
                        _phase_x_empty_0 ^= 1;
                        tma_3d_gmem2smem(smem_x_addr, x_tmap, 0, x_row, 0, x_full_addr);
                        mbarrier_arrive_expect_tx(x_full_addr, 49152);
                        #pragma unroll 1
                        for (int iter_k = 0; iter_k < K_tiles; iter_k++) {
                            int off_k = iter_k * 256;
                            int c_row = batch * K + off_k;
                            mbarrier_wait(c_empty_addr, _phase_c_empty_0);
                            _phase_c_empty_0 ^= 1;
                            tma_3d_gmem2smem(smem_c_addr, c_tmap, 0, c_row, 0, c_full_addr);
                            mbarrier_arrive_expect_tx(c_full_addr, 98304);
                        }
                    }
                }
            }
        }
    // ---- Role: mma ----
    } else if (warp == 1) {
        const int tmem_score_tmem = taddr + TMEM_SCORE_TMEM_OFFSET;
        { // mma_main
            int num_tiles = B * num_n_tiles;
            uint32_t _phase_x_full_0 = 0;
            uint32_t _phase_score_empty_0 = 1;
            uint32_t _phase_c_full_0 = 0;
            #pragma unroll 1
            for (unsigned int tile_idx = bid; tile_idx < num_tiles; tile_idx += num_bids) {
                mbarrier_wait(x_full_addr, _phase_x_full_0);
                _phase_x_full_0 ^= 1;
                #pragma unroll 1
                for (int iter_k = 0; iter_k < K_tiles; iter_k++) {
                    mbarrier_wait(score_empty_addr, _phase_score_empty_0);
                    _phase_score_empty_0 ^= 1;
                    mbarrier_wait(c_full_addr, _phase_c_full_0);
                    _phase_c_full_0 ^= 1;
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    int _desc_addr_0 = smem_x_addr + 0 * 49152 + 16384;
                    int _desc_lo_0 = make_warp_uniform((_desc_addr_0 >> 4) & 0x3FFF);
                    int _desc_addr_1 = smem_c_addr + 0 * 98304 + 32768;
                    int _desc_lo_1 = make_warp_uniform((_desc_addr_1 >> 4) & 0x3FFF);
                    int _desc_addr_2 = smem_x_addr + 0 * 49152 + 32768;
                    int _desc_lo_2 = make_warp_uniform((_desc_addr_2 >> 4) & 0x3FFF);
                    int _desc_addr_3 = smem_c_addr + 0 * 98304 + 65536;
                    int _desc_lo_3 = make_warp_uniform((_desc_addr_3 >> 4) & 0x3FFF);
                    int _mma_ss_a_addr_4 = smem_x_addr + 0 * 49152;
                    int _mma_ss_a_lo_4 = make_warp_uniform((_mma_ss_a_addr_4 >> 4) & 0x3FFF);
                    int _mma_ss_b_addr_4 = smem_c_addr + 0 * 98304;
                    int _mma_ss_b_lo_4 = make_warp_uniform((_mma_ss_b_addr_4 >> 4) & 0x3FFF);
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
                    "mov.b32 id, 138413200;\n\t"
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
                    :: "r"(_mma_ss_a_lo_4), "r"(_mma_ss_b_lo_4), "r"(tmem_score_tmem), "r"(0));
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
                    "mov.b32 id, 138413200;\n\t"
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
                    :: "r"(_desc_lo_0), "r"(_desc_lo_1), "r"(tmem_score_tmem), "r"(1));
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
                    "mov.b32 id, 138413200;\n\t"
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
                    :: "r"(_desc_lo_2), "r"(_desc_lo_3), "r"(tmem_score_tmem), "r"(1));
                    elect_commit(score_full_addr);
                    elect_commit(c_empty_addr);
                }
                elect_commit(x_empty_addr);
            }
        }
    // ---- Role: compute ----
    } else if (warp >= 2 && warp <= 5) {
        const int warp_id_in_wg = warp % 4;
        const int my_row = warp_id_in_wg * 32 + lane;
        const int tmem_row_base_v = (warp_id_in_wg * 32) << 16;
        { // compute_main
            int num_tiles = B * num_n_tiles;
            uint32_t _phase_score_full_0 = 0;
            #pragma unroll 1
            for (unsigned int tile_idx = bid; tile_idx < num_tiles; tile_idx += num_bids) {
                int batch = tile_idx / num_n_tiles;
                int n_tile = tile_idx % num_n_tiles;
                int off_n = n_tile * 128;
                int global_n = off_n + my_row;
                int out_offset = batch * N + global_n;
                int csq_smem_addr = smem_csq_addr + 0;
                float best_score = -3.4e+38f;
                int best_idx = 0;
                #pragma unroll 1
                for (int iter_k = 0; iter_k < K_tiles; iter_k++) {
                    int off_k = iter_k * 256;
                    if (my_row < 64.0f) {
                        int csq_base = my_row * 4;
                        float csq_pack[4];
                        {
                            float4 _v4 = *reinterpret_cast<const float4*>(c_sq + batch * K + off_k + csq_base);
                            csq_pack[0 + 0] = _v4.x;
                            csq_pack[0 + 1] = _v4.y;
                            csq_pack[0 + 2] = _v4.z;
                            csq_pack[0 + 3] = _v4.w;
                        }
                        float csq_half[4];
                        csq_half[0] = 0.5f * csq_pack[0];
                        csq_half[1] = 0.5f * csq_pack[1];
                        csq_half[2] = 0.5f * csq_pack[2];
                        csq_half[3] = 0.5f * csq_pack[3];
                        asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(csq_smem_addr + csq_base * 4), "f"(csq_half[0]), "f"(csq_half[1]), "f"(csq_half[2]), "f"(csq_half[3]) : "memory");
                    }
                    asm volatile("barrier.sync 8, 128;");
                    mbarrier_wait(score_full_addr, _phase_score_full_0);
                    _phase_score_full_0 ^= 1;
                    int score_base = 0;
                    float scores[128];
                    asm volatile(
                        "tcgen05.ld.sync.aligned.32x32b.x128.b32"
                        " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63, %64, %65, %66, %67, %68, %69, %70, %71, %72, %73, %74, %75, %76, %77, %78, %79, %80, %81, %82, %83, %84, %85, %86, %87, %88, %89, %90, %91, %92, %93, %94, %95, %96, %97, %98, %99, %100, %101, %102, %103, %104, %105, %106, %107, %108, %109, %110, %111, %112, %113, %114, %115, %116, %117, %118, %119, %120, %121, %122, %123, %124, %125, %126, %127}, [%128];"
                        : "=f"(scores[0]), "=f"(scores[1]), "=f"(scores[2]), "=f"(scores[3]), "=f"(scores[4]), "=f"(scores[5]), "=f"(scores[6]), "=f"(scores[7]), "=f"(scores[8]), "=f"(scores[9]), "=f"(scores[10]), "=f"(scores[11]), "=f"(scores[12]), "=f"(scores[13]), "=f"(scores[14]), "=f"(scores[15]), "=f"(scores[16]), "=f"(scores[17]), "=f"(scores[18]), "=f"(scores[19]), "=f"(scores[20]), "=f"(scores[21]), "=f"(scores[22]), "=f"(scores[23]), "=f"(scores[24]), "=f"(scores[25]), "=f"(scores[26]), "=f"(scores[27]), "=f"(scores[28]), "=f"(scores[29]), "=f"(scores[30]), "=f"(scores[31]), "=f"(scores[32]), "=f"(scores[33]), "=f"(scores[34]), "=f"(scores[35]), "=f"(scores[36]), "=f"(scores[37]), "=f"(scores[38]), "=f"(scores[39]), "=f"(scores[40]), "=f"(scores[41]), "=f"(scores[42]), "=f"(scores[43]), "=f"(scores[44]), "=f"(scores[45]), "=f"(scores[46]), "=f"(scores[47]), "=f"(scores[48]), "=f"(scores[49]), "=f"(scores[50]), "=f"(scores[51]), "=f"(scores[52]), "=f"(scores[53]), "=f"(scores[54]), "=f"(scores[55]), "=f"(scores[56]), "=f"(scores[57]), "=f"(scores[58]), "=f"(scores[59]), "=f"(scores[60]), "=f"(scores[61]), "=f"(scores[62]), "=f"(scores[63]), "=f"(scores[64]), "=f"(scores[65]), "=f"(scores[66]), "=f"(scores[67]), "=f"(scores[68]), "=f"(scores[69]), "=f"(scores[70]), "=f"(scores[71]), "=f"(scores[72]), "=f"(scores[73]), "=f"(scores[74]), "=f"(scores[75]), "=f"(scores[76]), "=f"(scores[77]), "=f"(scores[78]), "=f"(scores[79]), "=f"(scores[80]), "=f"(scores[81]), "=f"(scores[82]), "=f"(scores[83]), "=f"(scores[84]), "=f"(scores[85]), "=f"(scores[86]), "=f"(scores[87]), "=f"(scores[88]), "=f"(scores[89]), "=f"(scores[90]), "=f"(scores[91]), "=f"(scores[92]), "=f"(scores[93]), "=f"(scores[94]), "=f"(scores[95]), "=f"(scores[96]), "=f"(scores[97]), "=f"(scores[98]), "=f"(scores[99]), "=f"(scores[100]), "=f"(scores[101]), "=f"(scores[102]), "=f"(scores[103]), "=f"(scores[104]), "=f"(scores[105]), "=f"(scores[106]), "=f"(scores[107]), "=f"(scores[108]), "=f"(scores[109]), "=f"(scores[110]), "=f"(scores[111]), "=f"(scores[112]), "=f"(scores[113]), "=f"(scores[114]), "=f"(scores[115]), "=f"(scores[116]), "=f"(scores[117]), "=f"(scores[118]), "=f"(scores[119]), "=f"(scores[120]), "=f"(scores[121]), "=f"(scores[122]), "=f"(scores[123]), "=f"(scores[124]), "=f"(scores[125]), "=f"(scores[126]), "=f"(scores[127])
                        : "r"(taddr + tmem_row_base_v + score_base)
                        : "memory");
                    asm volatile("tcgen05.wait::ld.sync.aligned;");
                    #pragma unroll
                    for (int kk = 0; kk < 128; kk += 4) {
                        float csq_vals[4];
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&csq_vals[0])), "=r"(*reinterpret_cast<uint32_t*>(&csq_vals[1])), "=r"(*reinterpret_cast<uint32_t*>(&csq_vals[2])), "=r"(*reinterpret_cast<uint32_t*>(&csq_vals[3]))
                            : "r"(csq_smem_addr + (score_base + kk) * 4));
                        float d0 = scores[kk] - csq_vals[0];
                        float d1 = scores[kk + 1] - csq_vals[1];
                        float best_01 = d0;
                        int idx_01 = off_k + score_base + kk;
                        if (d1 > best_01) {
                            best_01 = d1;
                            idx_01 = off_k + score_base + kk + 1;
                        }
                        float d2 = scores[kk + 2] - csq_vals[2];
                        float d3 = scores[kk + 3] - csq_vals[3];
                        float best_23 = d2;
                        int idx_23 = off_k + score_base + kk + 2;
                        if (d3 > best_23) {
                            best_23 = d3;
                            idx_23 = off_k + score_base + kk + 3;
                        }
                        float best_group = best_01;
                        int idx_group = idx_01;
                        if (best_23 > best_group) {
                            best_group = best_23;
                            idx_group = idx_23;
                        }
                        if (best_group > best_score) {
                            best_score = best_group;
                            best_idx = idx_group;
                        }
                    }
                    score_base = 128;
                    asm volatile(
                        "tcgen05.ld.sync.aligned.32x32b.x128.b32"
                        " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63, %64, %65, %66, %67, %68, %69, %70, %71, %72, %73, %74, %75, %76, %77, %78, %79, %80, %81, %82, %83, %84, %85, %86, %87, %88, %89, %90, %91, %92, %93, %94, %95, %96, %97, %98, %99, %100, %101, %102, %103, %104, %105, %106, %107, %108, %109, %110, %111, %112, %113, %114, %115, %116, %117, %118, %119, %120, %121, %122, %123, %124, %125, %126, %127}, [%128];"
                        : "=f"(scores[0]), "=f"(scores[1]), "=f"(scores[2]), "=f"(scores[3]), "=f"(scores[4]), "=f"(scores[5]), "=f"(scores[6]), "=f"(scores[7]), "=f"(scores[8]), "=f"(scores[9]), "=f"(scores[10]), "=f"(scores[11]), "=f"(scores[12]), "=f"(scores[13]), "=f"(scores[14]), "=f"(scores[15]), "=f"(scores[16]), "=f"(scores[17]), "=f"(scores[18]), "=f"(scores[19]), "=f"(scores[20]), "=f"(scores[21]), "=f"(scores[22]), "=f"(scores[23]), "=f"(scores[24]), "=f"(scores[25]), "=f"(scores[26]), "=f"(scores[27]), "=f"(scores[28]), "=f"(scores[29]), "=f"(scores[30]), "=f"(scores[31]), "=f"(scores[32]), "=f"(scores[33]), "=f"(scores[34]), "=f"(scores[35]), "=f"(scores[36]), "=f"(scores[37]), "=f"(scores[38]), "=f"(scores[39]), "=f"(scores[40]), "=f"(scores[41]), "=f"(scores[42]), "=f"(scores[43]), "=f"(scores[44]), "=f"(scores[45]), "=f"(scores[46]), "=f"(scores[47]), "=f"(scores[48]), "=f"(scores[49]), "=f"(scores[50]), "=f"(scores[51]), "=f"(scores[52]), "=f"(scores[53]), "=f"(scores[54]), "=f"(scores[55]), "=f"(scores[56]), "=f"(scores[57]), "=f"(scores[58]), "=f"(scores[59]), "=f"(scores[60]), "=f"(scores[61]), "=f"(scores[62]), "=f"(scores[63]), "=f"(scores[64]), "=f"(scores[65]), "=f"(scores[66]), "=f"(scores[67]), "=f"(scores[68]), "=f"(scores[69]), "=f"(scores[70]), "=f"(scores[71]), "=f"(scores[72]), "=f"(scores[73]), "=f"(scores[74]), "=f"(scores[75]), "=f"(scores[76]), "=f"(scores[77]), "=f"(scores[78]), "=f"(scores[79]), "=f"(scores[80]), "=f"(scores[81]), "=f"(scores[82]), "=f"(scores[83]), "=f"(scores[84]), "=f"(scores[85]), "=f"(scores[86]), "=f"(scores[87]), "=f"(scores[88]), "=f"(scores[89]), "=f"(scores[90]), "=f"(scores[91]), "=f"(scores[92]), "=f"(scores[93]), "=f"(scores[94]), "=f"(scores[95]), "=f"(scores[96]), "=f"(scores[97]), "=f"(scores[98]), "=f"(scores[99]), "=f"(scores[100]), "=f"(scores[101]), "=f"(scores[102]), "=f"(scores[103]), "=f"(scores[104]), "=f"(scores[105]), "=f"(scores[106]), "=f"(scores[107]), "=f"(scores[108]), "=f"(scores[109]), "=f"(scores[110]), "=f"(scores[111]), "=f"(scores[112]), "=f"(scores[113]), "=f"(scores[114]), "=f"(scores[115]), "=f"(scores[116]), "=f"(scores[117]), "=f"(scores[118]), "=f"(scores[119]), "=f"(scores[120]), "=f"(scores[121]), "=f"(scores[122]), "=f"(scores[123]), "=f"(scores[124]), "=f"(scores[125]), "=f"(scores[126]), "=f"(scores[127])
                        : "r"(taddr + tmem_row_base_v + score_base)
                        : "memory");
                    asm volatile("tcgen05.wait::ld.sync.aligned;");
                    asm volatile("barrier.sync 9, 128;");
                    if (elect_sync()) {
                        mbarrier_arrive(score_empty_addr);
                    }
                    #pragma unroll
                    for (int kk = 0; kk < 128; kk += 4) {
                        float csq_vals[4];
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&csq_vals[0])), "=r"(*reinterpret_cast<uint32_t*>(&csq_vals[1])), "=r"(*reinterpret_cast<uint32_t*>(&csq_vals[2])), "=r"(*reinterpret_cast<uint32_t*>(&csq_vals[3]))
                            : "r"(csq_smem_addr + (score_base + kk) * 4));
                        float d0 = scores[kk] - csq_vals[0];
                        float d1 = scores[kk + 1] - csq_vals[1];
                        float best_01 = d0;
                        int idx_01 = off_k + score_base + kk;
                        if (d1 > best_01) {
                            best_01 = d1;
                            idx_01 = off_k + score_base + kk + 1;
                        }
                        float d2 = scores[kk + 2] - csq_vals[2];
                        float d3 = scores[kk + 3] - csq_vals[3];
                        float best_23 = d2;
                        int idx_23 = off_k + score_base + kk + 2;
                        if (d3 > best_23) {
                            best_23 = d3;
                            idx_23 = off_k + score_base + kk + 3;
                        }
                        float best_group = best_01;
                        int idx_group = idx_01;
                        if (best_23 > best_group) {
                            best_group = best_23;
                            idx_group = idx_23;
                        }
                        if (best_group > best_score) {
                            best_score = best_group;
                            best_idx = idx_group;
                        }
                    }
                }
                if (global_n < N) {
                    *((int*)(out + out_offset)) = best_idx;
                }
            }
        }
    }

    // Cleanup
    __syncthreads(); // barrier before TMEM dealloc

    if (warp == 1) {
        asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;" :: "r"(tmem_addr_storage[0]), "r"(256));
        asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;");
    }
}

} // extern "C"

#undef NUM_MAIN_STAGES
#undef SMEM_SMEM_CSQ_OFF
#undef SMEM_SMEM_CSQ_STAGE_BYTES
#undef SMEM_SMEM_CSQ_STRIDE
#undef SMEM_SMEM_C_OFF
#undef SMEM_SMEM_C_STAGE_BYTES
#undef SMEM_SMEM_C_STRIDE
#undef SMEM_SMEM_X_OFF
#undef SMEM_SMEM_X_STAGE_BYTES
#undef SMEM_SMEM_X_STRIDE
#undef SMEM_TOTAL
#undef TMEM_NCOLS
#undef TMEM_SCORE_TMEM_OFFSET
#undef c_empty_addr
#undef c_full_addr
#undef score_empty_addr
#undef score_full_addr
#undef smem_c_addr
#undef smem_csq_addr
#undef smem_x_addr
#undef x_empty_addr
#undef x_full_addr

#define TMEM_NCOLS 512
#define TMEM_SCORE_TMEM_OFFSET 0
#define NUM_MAIN_STAGES 1
#define SMEM_SMEM_X0_OFF 1024
#define SMEM_SMEM_X0_STAGE_BYTES 49152
#define SMEM_SMEM_X0_STRIDE 49152
#define SMEM_SMEM_X1_OFF 50176
#define SMEM_SMEM_X1_STAGE_BYTES 49152
#define SMEM_SMEM_X1_STRIDE 49152
#define SMEM_SMEM_C_OFF 99328
#define SMEM_SMEM_C_STAGE_BYTES 98304
#define SMEM_SMEM_C_STRIDE 98304
#define SMEM_SMEM_CSQ_OFF 197632
#define SMEM_SMEM_CSQ_STAGE_BYTES 1024
#define SMEM_SMEM_CSQ_STRIDE 1024
#define SMEM_TOTAL 198656

extern "C" {

__global__ __launch_bounds__(192) void
kernel_flash_kmeans_assign_cleanroom_tcgen05_d192_splitd_v1(float* __restrict__ x_sq, float* __restrict__ c_sq, int32_t* __restrict__ out, const void* x_tmap, const void* c_tmap, int B, int N, int D, int K, int num_n_tiles, int K_tiles)
{
    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;

    extern __shared__ __align__(1024) char smem_raw[];
    int smem;
    smem = (int)(unsigned long long)__cvta_generic_to_shared(smem_raw);
    const int smem_smem_x0 = smem + 1024;
    const int smem_smem_x1 = smem + 50176;
    const int smem_smem_c = smem + 99328;
    const int smem_smem_csq = smem + 197632;

    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    // Mbarrier init (10 groups, 10 barriers)
    // Mbarriers at smem_raw[0..80)

    if (warp == 0) {
        uint32_t leader = elect_sync();
        // x0_full: 1 barriers, init_count=1
        mbarrier_init_pred(smem + 0, 1, leader);
        // x0_empty: 1 barriers, init_count=1
        mbarrier_init_pred(smem + 8, 1, leader);
        // x1_full: 1 barriers, init_count=1
        mbarrier_init_pred(smem + 16, 1, leader);
        // x1_empty: 1 barriers, init_count=1
        mbarrier_init_pred(smem + 24, 1, leader);
        // c_full: 1 barriers, init_count=1
        mbarrier_init_pred(smem + 32, 1, leader);
        // c_empty: 1 barriers, init_count=1
        mbarrier_init_pred(smem + 40, 1, leader);
        // score0_full: 1 barriers, init_count=1
        mbarrier_init_pred(smem + 48, 1, leader);
        // score0_empty: 1 barriers, init_count=4
        mbarrier_init_pred(smem + 56, 4, leader);
        // score1_full: 1 barriers, init_count=1
        mbarrier_init_pred(smem + 64, 1, leader);
        // score1_empty: 1 barriers, init_count=4
        mbarrier_init_pred(smem + 72, 4, leader);
        asm volatile("fence.mbarrier_init.release.cluster;");
    }

    __syncthreads();

    // TMEM alloc (512 columns, 512 used)
    volatile int* tmem_addr_storage = (volatile int*)(smem_raw + 80);
    if (warp == 1) {
        int _tmem_hold = smem + 80;
        asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;" :: "r"(_tmem_hold), "r"(512) : "memory");
    }

    __syncthreads();
    asm volatile("tcgen05.fence::after_thread_sync;");

    const int warp_id = warp;
    const int lane_id = lane;
    __nv_bfloat16* smem_x0 = (__nv_bfloat16*)(smem_raw + 1024);
    #define smem_x0_addr (smem + 1024)
    __nv_bfloat16* smem_x1 = (__nv_bfloat16*)(smem_raw + 50176);
    #define smem_x1_addr (smem + 50176)
    __nv_bfloat16* smem_c = (__nv_bfloat16*)(smem_raw + 99328);
    #define smem_c_addr (smem + 99328)
    float* smem_csq = (float*)(smem_raw + 197632);
    #define smem_csq_addr (smem + 197632)
    const int mbar_base = smem;
    #define x0_full_addr (mbar_base + 0)
    #define x0_empty_addr (mbar_base + 8)
    #define x1_full_addr (mbar_base + 16)
    #define x1_empty_addr (mbar_base + 24)
    #define c_full_addr (mbar_base + 32)
    #define c_empty_addr (mbar_base + 40)
    #define score0_full_addr (mbar_base + 48)
    #define score0_empty_addr (mbar_base + 56)
    #define score1_full_addr (mbar_base + 64)
    #define score1_empty_addr (mbar_base + 72)
    const int taddr = tmem_addr_storage[0];

    // ---- Role: load ----
    if (warp == 0) {
        { // load_main
            int pair_n_tiles = num_n_tiles / 2;
            int num_tiles = B * pair_n_tiles;
            uint32_t _phase_x0_empty_0 = 1;
            uint32_t _phase_x1_empty_0 = 1;
            uint32_t _phase_c_empty_0 = 1;
            if (warp_id == 0) {
                if (elect_sync()) {
                    #pragma unroll 1
                    for (unsigned int tile_idx = bid; tile_idx < num_tiles; tile_idx += num_bids) {
                        int batch = tile_idx / pair_n_tiles;
                        int pair_tile = tile_idx % pair_n_tiles;
                        int off_n0 = pair_tile * 256;
                        int off_n1 = off_n0 + 128;
                        int x_row0 = batch * N + off_n0;
                        int x_row1 = batch * N + off_n1;
                        mbarrier_wait(x0_empty_addr, _phase_x0_empty_0);
                        _phase_x0_empty_0 ^= 1;
                        tma_3d_gmem2smem(smem_x0_addr, x_tmap, 0, x_row0, 0, x0_full_addr);
                        mbarrier_arrive_expect_tx(x0_full_addr, 49152);
                        mbarrier_wait(x1_empty_addr, _phase_x1_empty_0);
                        _phase_x1_empty_0 ^= 1;
                        tma_3d_gmem2smem(smem_x1_addr, x_tmap, 0, x_row1, 0, x1_full_addr);
                        mbarrier_arrive_expect_tx(x1_full_addr, 49152);
                        #pragma unroll 1
                        for (int iter_k = 0; iter_k < K_tiles; iter_k++) {
                            int off_k = iter_k * 256;
                            int c_row = batch * K + off_k;
                            mbarrier_wait(c_empty_addr, _phase_c_empty_0);
                            _phase_c_empty_0 ^= 1;
                            tma_3d_gmem2smem(smem_c_addr, c_tmap, 0, c_row, 0, c_full_addr);
                            mbarrier_arrive_expect_tx(c_full_addr, 98304);
                        }
                    }
                }
            }
        }
    // ---- Role: mma ----
    } else if (warp == 1) {
        const int tmem_score_tmem = taddr + TMEM_SCORE_TMEM_OFFSET;
        { // mma_main
            int pair_n_tiles = num_n_tiles / 2;
            int num_tiles = B * pair_n_tiles;
            uint32_t _phase_x0_full_0 = 0;
            uint32_t _phase_x1_full_0 = 0;
            uint32_t _phase_score0_empty_0 = 1;
            uint32_t _phase_c_full_0 = 0;
            uint32_t _phase_score1_empty_0 = 1;
            #pragma unroll 1
            for (unsigned int tile_idx = bid; tile_idx < num_tiles; tile_idx += num_bids) {
                mbarrier_wait(x0_full_addr, _phase_x0_full_0);
                _phase_x0_full_0 ^= 1;
                mbarrier_wait(x1_full_addr, _phase_x1_full_0);
                _phase_x1_full_0 ^= 1;
                #pragma unroll 1
                for (int iter_k = 0; iter_k < K_tiles; iter_k++) {
                    mbarrier_wait(score0_empty_addr, _phase_score0_empty_0);
                    _phase_score0_empty_0 ^= 1;
                    mbarrier_wait(c_full_addr, _phase_c_full_0);
                    _phase_c_full_0 ^= 1;
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    int _desc_addr_0 = smem_x0_addr + 0 * 49152 + 16384;
                    int _desc_lo_0 = make_warp_uniform((_desc_addr_0 >> 4) & 0x3FFF);
                    int _desc_addr_1 = smem_c_addr + 0 * 98304 + 32768;
                    int _desc_lo_1 = make_warp_uniform((_desc_addr_1 >> 4) & 0x3FFF);
                    int _desc_addr_2 = smem_x0_addr + 0 * 49152 + 32768;
                    int _desc_lo_2 = make_warp_uniform((_desc_addr_2 >> 4) & 0x3FFF);
                    int _desc_addr_3 = smem_c_addr + 0 * 98304 + 65536;
                    int _desc_lo_3 = make_warp_uniform((_desc_addr_3 >> 4) & 0x3FFF);
                    int _mma_ss_a_addr_4 = smem_x0_addr + 0 * 49152;
                    int _mma_ss_a_lo_4 = make_warp_uniform((_mma_ss_a_addr_4 >> 4) & 0x3FFF);
                    int _mma_ss_b_addr_4 = smem_c_addr + 0 * 98304;
                    int _mma_ss_b_lo_4 = make_warp_uniform((_mma_ss_b_addr_4 >> 4) & 0x3FFF);
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
                    "mov.b32 id, 138413200;\n\t"
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
                    :: "r"(_mma_ss_a_lo_4), "r"(_mma_ss_b_lo_4), "r"((tmem_score_tmem + (256))), "r"(0));
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
                    "mov.b32 id, 138413200;\n\t"
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
                    :: "r"(_desc_lo_0), "r"(_desc_lo_1), "r"((tmem_score_tmem + (256))), "r"(1));
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
                    "mov.b32 id, 138413200;\n\t"
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
                    :: "r"(_desc_lo_2), "r"(_desc_lo_3), "r"((tmem_score_tmem + (256))), "r"(1));
                    elect_commit(score0_full_addr);
                    mbarrier_wait(score1_empty_addr, _phase_score1_empty_0);
                    _phase_score1_empty_0 ^= 1;
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    int _desc_addr_7 = smem_x1_addr + 0 * 49152 + 16384;
                    int _desc_lo_7 = make_warp_uniform((_desc_addr_7 >> 4) & 0x3FFF);
                    int _desc_addr_8 = smem_x1_addr + 0 * 49152 + 32768;
                    int _desc_lo_8 = make_warp_uniform((_desc_addr_8 >> 4) & 0x3FFF);
                    int _mma_ss_a_addr_9 = smem_x1_addr + 0 * 49152;
                    int _mma_ss_a_lo_9 = make_warp_uniform((_mma_ss_a_addr_9 >> 4) & 0x3FFF);
                    int _mma_ss_b_addr_9 = smem_c_addr + 0 * 98304;
                    int _mma_ss_b_lo_9 = make_warp_uniform((_mma_ss_b_addr_9 >> 4) & 0x3FFF);
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
                    "mov.b32 id, 138413200;\n\t"
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
                    :: "r"(_mma_ss_a_lo_9), "r"(_mma_ss_b_lo_9), "r"(tmem_score_tmem), "r"(0));
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
                    "mov.b32 id, 138413200;\n\t"
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
                    :: "r"(_desc_lo_7), "r"(_desc_lo_1), "r"(tmem_score_tmem), "r"(1));
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
                    "mov.b32 id, 138413200;\n\t"
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
                    :: "r"(_desc_lo_8), "r"(_desc_lo_3), "r"(tmem_score_tmem), "r"(1));
                    elect_commit(score1_full_addr);
                    elect_commit(c_empty_addr);
                }
                elect_commit(x0_empty_addr);
                elect_commit(x1_empty_addr);
            }
        }
    // ---- Role: compute ----
    } else if (warp >= 2 && warp <= 5) {
        const int warp_id_in_wg = warp % 4;
        const int my_row = warp_id_in_wg * 32 + lane;
        const int tmem_row_base_v = (warp_id_in_wg * 32) << 16;
        { // compute_main
            int pair_n_tiles = num_n_tiles / 2;
            int num_tiles = B * pair_n_tiles;
            uint32_t _phase_score0_full_0 = 0;
            uint32_t _phase_score1_full_0 = 0;
            #pragma unroll 1
            for (unsigned int tile_idx = bid; tile_idx < num_tiles; tile_idx += num_bids) {
                int batch = tile_idx / pair_n_tiles;
                int pair_tile = tile_idx % pair_n_tiles;
                int off_n0 = pair_tile * 256;
                int off_n1 = off_n0 + 128;
                int global_n0 = off_n0 + my_row;
                int global_n1 = off_n1 + my_row;
                int out_offset0 = batch * N + global_n0;
                int out_offset1 = batch * N + global_n1;
                int csq_smem_addr = smem_csq_addr + 0;
                float best_score0 = -3.4e+38f;
                int best_idx0 = 0;
                float best_score1 = -3.4e+38f;
                int best_idx1 = 0;
                #pragma unroll 1
                for (int iter_k = 0; iter_k < K_tiles; iter_k++) {
                    int off_k = iter_k * 256;
                    if (my_row < 64.0f) {
                        int csq_base = my_row * 4;
                        float csq_pack[4];
                        {
                            float4 _v4 = *reinterpret_cast<const float4*>(c_sq + batch * K + off_k + csq_base);
                            csq_pack[0 + 0] = _v4.x;
                            csq_pack[0 + 1] = _v4.y;
                            csq_pack[0 + 2] = _v4.z;
                            csq_pack[0 + 3] = _v4.w;
                        }
                        float csq_half[4];
                        csq_half[0] = 0.5f * csq_pack[0];
                        csq_half[1] = 0.5f * csq_pack[1];
                        csq_half[2] = 0.5f * csq_pack[2];
                        csq_half[3] = 0.5f * csq_pack[3];
                        asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(csq_smem_addr + csq_base * 4), "f"(csq_half[0]), "f"(csq_half[1]), "f"(csq_half[2]), "f"(csq_half[3]) : "memory");
                    }
                    asm volatile("barrier.sync 8, 128;");
                    mbarrier_wait(score0_full_addr, _phase_score0_full_0);
                    _phase_score0_full_0 ^= 1;
                    int score_base = 0;
                    float scores[128];
                    asm volatile(
                        "tcgen05.ld.sync.aligned.32x32b.x128.b32"
                        " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63, %64, %65, %66, %67, %68, %69, %70, %71, %72, %73, %74, %75, %76, %77, %78, %79, %80, %81, %82, %83, %84, %85, %86, %87, %88, %89, %90, %91, %92, %93, %94, %95, %96, %97, %98, %99, %100, %101, %102, %103, %104, %105, %106, %107, %108, %109, %110, %111, %112, %113, %114, %115, %116, %117, %118, %119, %120, %121, %122, %123, %124, %125, %126, %127}, [%128];"
                        : "=f"(scores[0]), "=f"(scores[1]), "=f"(scores[2]), "=f"(scores[3]), "=f"(scores[4]), "=f"(scores[5]), "=f"(scores[6]), "=f"(scores[7]), "=f"(scores[8]), "=f"(scores[9]), "=f"(scores[10]), "=f"(scores[11]), "=f"(scores[12]), "=f"(scores[13]), "=f"(scores[14]), "=f"(scores[15]), "=f"(scores[16]), "=f"(scores[17]), "=f"(scores[18]), "=f"(scores[19]), "=f"(scores[20]), "=f"(scores[21]), "=f"(scores[22]), "=f"(scores[23]), "=f"(scores[24]), "=f"(scores[25]), "=f"(scores[26]), "=f"(scores[27]), "=f"(scores[28]), "=f"(scores[29]), "=f"(scores[30]), "=f"(scores[31]), "=f"(scores[32]), "=f"(scores[33]), "=f"(scores[34]), "=f"(scores[35]), "=f"(scores[36]), "=f"(scores[37]), "=f"(scores[38]), "=f"(scores[39]), "=f"(scores[40]), "=f"(scores[41]), "=f"(scores[42]), "=f"(scores[43]), "=f"(scores[44]), "=f"(scores[45]), "=f"(scores[46]), "=f"(scores[47]), "=f"(scores[48]), "=f"(scores[49]), "=f"(scores[50]), "=f"(scores[51]), "=f"(scores[52]), "=f"(scores[53]), "=f"(scores[54]), "=f"(scores[55]), "=f"(scores[56]), "=f"(scores[57]), "=f"(scores[58]), "=f"(scores[59]), "=f"(scores[60]), "=f"(scores[61]), "=f"(scores[62]), "=f"(scores[63]), "=f"(scores[64]), "=f"(scores[65]), "=f"(scores[66]), "=f"(scores[67]), "=f"(scores[68]), "=f"(scores[69]), "=f"(scores[70]), "=f"(scores[71]), "=f"(scores[72]), "=f"(scores[73]), "=f"(scores[74]), "=f"(scores[75]), "=f"(scores[76]), "=f"(scores[77]), "=f"(scores[78]), "=f"(scores[79]), "=f"(scores[80]), "=f"(scores[81]), "=f"(scores[82]), "=f"(scores[83]), "=f"(scores[84]), "=f"(scores[85]), "=f"(scores[86]), "=f"(scores[87]), "=f"(scores[88]), "=f"(scores[89]), "=f"(scores[90]), "=f"(scores[91]), "=f"(scores[92]), "=f"(scores[93]), "=f"(scores[94]), "=f"(scores[95]), "=f"(scores[96]), "=f"(scores[97]), "=f"(scores[98]), "=f"(scores[99]), "=f"(scores[100]), "=f"(scores[101]), "=f"(scores[102]), "=f"(scores[103]), "=f"(scores[104]), "=f"(scores[105]), "=f"(scores[106]), "=f"(scores[107]), "=f"(scores[108]), "=f"(scores[109]), "=f"(scores[110]), "=f"(scores[111]), "=f"(scores[112]), "=f"(scores[113]), "=f"(scores[114]), "=f"(scores[115]), "=f"(scores[116]), "=f"(scores[117]), "=f"(scores[118]), "=f"(scores[119]), "=f"(scores[120]), "=f"(scores[121]), "=f"(scores[122]), "=f"(scores[123]), "=f"(scores[124]), "=f"(scores[125]), "=f"(scores[126]), "=f"(scores[127])
                        : "r"(taddr + tmem_row_base_v + 256 + score_base)
                        : "memory");
                    asm volatile("tcgen05.wait::ld.sync.aligned;");
                    #pragma unroll
                    for (int kk = 0; kk < 128; kk += 16) {
                        float csq_vals[4];
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&csq_vals[0])), "=r"(*reinterpret_cast<uint32_t*>(&csq_vals[1])), "=r"(*reinterpret_cast<uint32_t*>(&csq_vals[2])), "=r"(*reinterpret_cast<uint32_t*>(&csq_vals[3]))
                            : "r"(csq_smem_addr + (score_base + kk) * 4));
                        float d0 = scores[kk] - csq_vals[0];
                        float d1 = scores[kk + 1] - csq_vals[1];
                        float best_01 = d0;
                        int idx_01 = off_k + score_base + kk;
                        if (d1 > best_01) {
                            best_01 = d1;
                            idx_01 = off_k + score_base + kk + 1;
                        }
                        float d2 = scores[kk + 2] - csq_vals[2];
                        float d3 = scores[kk + 3] - csq_vals[3];
                        float best_23 = d2;
                        int idx_23 = off_k + score_base + kk + 2;
                        if (d3 > best_23) {
                            best_23 = d3;
                            idx_23 = off_k + score_base + kk + 3;
                        }
                        float best_group = best_01;
                        int idx_group = idx_01;
                        if (best_23 > best_group) {
                            best_group = best_23;
                            idx_group = idx_23;
                        }
                        float csq_vals_hi[4];
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&csq_vals_hi[0])), "=r"(*reinterpret_cast<uint32_t*>(&csq_vals_hi[1])), "=r"(*reinterpret_cast<uint32_t*>(&csq_vals_hi[2])), "=r"(*reinterpret_cast<uint32_t*>(&csq_vals_hi[3]))
                            : "r"(csq_smem_addr + (score_base + kk + 4) * 4));
                        float d4 = scores[kk + 4] - csq_vals_hi[0];
                        float d5 = scores[kk + 5] - csq_vals_hi[1];
                        float best_45 = d4;
                        int idx_45 = off_k + score_base + kk + 4;
                        if (d5 > best_45) {
                            best_45 = d5;
                            idx_45 = off_k + score_base + kk + 5;
                        }
                        float d6 = scores[kk + 6] - csq_vals_hi[2];
                        float d7 = scores[kk + 7] - csq_vals_hi[3];
                        float best_67 = d6;
                        int idx_67 = off_k + score_base + kk + 6;
                        if (d7 > best_67) {
                            best_67 = d7;
                            idx_67 = off_k + score_base + kk + 7;
                        }
                        float best_hi = best_45;
                        int idx_hi = idx_45;
                        if (best_67 > best_hi) {
                            best_hi = best_67;
                            idx_hi = idx_67;
                        }
                        if (best_hi > best_group) {
                            best_group = best_hi;
                            idx_group = idx_hi;
                        }
                        float csq_vals_next[4];
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&csq_vals_next[0])), "=r"(*reinterpret_cast<uint32_t*>(&csq_vals_next[1])), "=r"(*reinterpret_cast<uint32_t*>(&csq_vals_next[2])), "=r"(*reinterpret_cast<uint32_t*>(&csq_vals_next[3]))
                            : "r"(csq_smem_addr + (score_base + kk + 8) * 4));
                        float d8 = scores[kk + 8] - csq_vals_next[0];
                        float d9 = scores[kk + 9] - csq_vals_next[1];
                        float best_89 = d8;
                        int idx_89 = off_k + score_base + kk + 8;
                        if (d9 > best_89) {
                            best_89 = d9;
                            idx_89 = off_k + score_base + kk + 9;
                        }
                        float d10 = scores[kk + 10] - csq_vals_next[2];
                        float d11 = scores[kk + 11] - csq_vals_next[3];
                        float best_1011 = d10;
                        int idx_1011 = off_k + score_base + kk + 10;
                        if (d11 > best_1011) {
                            best_1011 = d11;
                            idx_1011 = off_k + score_base + kk + 11;
                        }
                        float best_next = best_89;
                        int idx_next = idx_89;
                        if (best_1011 > best_next) {
                            best_next = best_1011;
                            idx_next = idx_1011;
                        }
                        float csq_vals_tail[4];
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&csq_vals_tail[0])), "=r"(*reinterpret_cast<uint32_t*>(&csq_vals_tail[1])), "=r"(*reinterpret_cast<uint32_t*>(&csq_vals_tail[2])), "=r"(*reinterpret_cast<uint32_t*>(&csq_vals_tail[3]))
                            : "r"(csq_smem_addr + (score_base + kk + 12) * 4));
                        float d12 = scores[kk + 12] - csq_vals_tail[0];
                        float d13 = scores[kk + 13] - csq_vals_tail[1];
                        float best_1213 = d12;
                        int idx_1213 = off_k + score_base + kk + 12;
                        if (d13 > best_1213) {
                            best_1213 = d13;
                            idx_1213 = off_k + score_base + kk + 13;
                        }
                        float d14 = scores[kk + 14] - csq_vals_tail[2];
                        float d15 = scores[kk + 15] - csq_vals_tail[3];
                        float best_1415 = d14;
                        int idx_1415 = off_k + score_base + kk + 14;
                        if (d15 > best_1415) {
                            best_1415 = d15;
                            idx_1415 = off_k + score_base + kk + 15;
                        }
                        float best_tail = best_1213;
                        int idx_tail = idx_1213;
                        if (best_1415 > best_tail) {
                            best_tail = best_1415;
                            idx_tail = idx_1415;
                        }
                        if (best_tail > best_next) {
                            best_next = best_tail;
                            idx_next = idx_tail;
                        }
                        if (best_next > best_group) {
                            best_group = best_next;
                            idx_group = idx_next;
                        }
                        if (best_group > best_score0) {
                            best_score0 = best_group;
                            best_idx0 = idx_group;
                        }
                    }
                    score_base = 128;
                    asm volatile(
                        "tcgen05.ld.sync.aligned.32x32b.x128.b32"
                        " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63, %64, %65, %66, %67, %68, %69, %70, %71, %72, %73, %74, %75, %76, %77, %78, %79, %80, %81, %82, %83, %84, %85, %86, %87, %88, %89, %90, %91, %92, %93, %94, %95, %96, %97, %98, %99, %100, %101, %102, %103, %104, %105, %106, %107, %108, %109, %110, %111, %112, %113, %114, %115, %116, %117, %118, %119, %120, %121, %122, %123, %124, %125, %126, %127}, [%128];"
                        : "=f"(scores[0]), "=f"(scores[1]), "=f"(scores[2]), "=f"(scores[3]), "=f"(scores[4]), "=f"(scores[5]), "=f"(scores[6]), "=f"(scores[7]), "=f"(scores[8]), "=f"(scores[9]), "=f"(scores[10]), "=f"(scores[11]), "=f"(scores[12]), "=f"(scores[13]), "=f"(scores[14]), "=f"(scores[15]), "=f"(scores[16]), "=f"(scores[17]), "=f"(scores[18]), "=f"(scores[19]), "=f"(scores[20]), "=f"(scores[21]), "=f"(scores[22]), "=f"(scores[23]), "=f"(scores[24]), "=f"(scores[25]), "=f"(scores[26]), "=f"(scores[27]), "=f"(scores[28]), "=f"(scores[29]), "=f"(scores[30]), "=f"(scores[31]), "=f"(scores[32]), "=f"(scores[33]), "=f"(scores[34]), "=f"(scores[35]), "=f"(scores[36]), "=f"(scores[37]), "=f"(scores[38]), "=f"(scores[39]), "=f"(scores[40]), "=f"(scores[41]), "=f"(scores[42]), "=f"(scores[43]), "=f"(scores[44]), "=f"(scores[45]), "=f"(scores[46]), "=f"(scores[47]), "=f"(scores[48]), "=f"(scores[49]), "=f"(scores[50]), "=f"(scores[51]), "=f"(scores[52]), "=f"(scores[53]), "=f"(scores[54]), "=f"(scores[55]), "=f"(scores[56]), "=f"(scores[57]), "=f"(scores[58]), "=f"(scores[59]), "=f"(scores[60]), "=f"(scores[61]), "=f"(scores[62]), "=f"(scores[63]), "=f"(scores[64]), "=f"(scores[65]), "=f"(scores[66]), "=f"(scores[67]), "=f"(scores[68]), "=f"(scores[69]), "=f"(scores[70]), "=f"(scores[71]), "=f"(scores[72]), "=f"(scores[73]), "=f"(scores[74]), "=f"(scores[75]), "=f"(scores[76]), "=f"(scores[77]), "=f"(scores[78]), "=f"(scores[79]), "=f"(scores[80]), "=f"(scores[81]), "=f"(scores[82]), "=f"(scores[83]), "=f"(scores[84]), "=f"(scores[85]), "=f"(scores[86]), "=f"(scores[87]), "=f"(scores[88]), "=f"(scores[89]), "=f"(scores[90]), "=f"(scores[91]), "=f"(scores[92]), "=f"(scores[93]), "=f"(scores[94]), "=f"(scores[95]), "=f"(scores[96]), "=f"(scores[97]), "=f"(scores[98]), "=f"(scores[99]), "=f"(scores[100]), "=f"(scores[101]), "=f"(scores[102]), "=f"(scores[103]), "=f"(scores[104]), "=f"(scores[105]), "=f"(scores[106]), "=f"(scores[107]), "=f"(scores[108]), "=f"(scores[109]), "=f"(scores[110]), "=f"(scores[111]), "=f"(scores[112]), "=f"(scores[113]), "=f"(scores[114]), "=f"(scores[115]), "=f"(scores[116]), "=f"(scores[117]), "=f"(scores[118]), "=f"(scores[119]), "=f"(scores[120]), "=f"(scores[121]), "=f"(scores[122]), "=f"(scores[123]), "=f"(scores[124]), "=f"(scores[125]), "=f"(scores[126]), "=f"(scores[127])
                        : "r"(taddr + tmem_row_base_v + 256 + score_base)
                        : "memory");
                    asm volatile("tcgen05.wait::ld.sync.aligned;");
                    asm volatile("barrier.sync 9, 128;");
                    if (elect_sync()) {
                        mbarrier_arrive(score0_empty_addr);
                    }
                    #pragma unroll
                    for (int kk = 0; kk < 128; kk += 16) {
                        float csq_vals[4];
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&csq_vals[0])), "=r"(*reinterpret_cast<uint32_t*>(&csq_vals[1])), "=r"(*reinterpret_cast<uint32_t*>(&csq_vals[2])), "=r"(*reinterpret_cast<uint32_t*>(&csq_vals[3]))
                            : "r"(csq_smem_addr + (score_base + kk) * 4));
                        float d0 = scores[kk] - csq_vals[0];
                        float d1 = scores[kk + 1] - csq_vals[1];
                        float best_01 = d0;
                        int idx_01 = off_k + score_base + kk;
                        if (d1 > best_01) {
                            best_01 = d1;
                            idx_01 = off_k + score_base + kk + 1;
                        }
                        float d2 = scores[kk + 2] - csq_vals[2];
                        float d3 = scores[kk + 3] - csq_vals[3];
                        float best_23 = d2;
                        int idx_23 = off_k + score_base + kk + 2;
                        if (d3 > best_23) {
                            best_23 = d3;
                            idx_23 = off_k + score_base + kk + 3;
                        }
                        float best_group = best_01;
                        int idx_group = idx_01;
                        if (best_23 > best_group) {
                            best_group = best_23;
                            idx_group = idx_23;
                        }
                        float csq_vals_hi[4];
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&csq_vals_hi[0])), "=r"(*reinterpret_cast<uint32_t*>(&csq_vals_hi[1])), "=r"(*reinterpret_cast<uint32_t*>(&csq_vals_hi[2])), "=r"(*reinterpret_cast<uint32_t*>(&csq_vals_hi[3]))
                            : "r"(csq_smem_addr + (score_base + kk + 4) * 4));
                        float d4 = scores[kk + 4] - csq_vals_hi[0];
                        float d5 = scores[kk + 5] - csq_vals_hi[1];
                        float best_45 = d4;
                        int idx_45 = off_k + score_base + kk + 4;
                        if (d5 > best_45) {
                            best_45 = d5;
                            idx_45 = off_k + score_base + kk + 5;
                        }
                        float d6 = scores[kk + 6] - csq_vals_hi[2];
                        float d7 = scores[kk + 7] - csq_vals_hi[3];
                        float best_67 = d6;
                        int idx_67 = off_k + score_base + kk + 6;
                        if (d7 > best_67) {
                            best_67 = d7;
                            idx_67 = off_k + score_base + kk + 7;
                        }
                        float best_hi = best_45;
                        int idx_hi = idx_45;
                        if (best_67 > best_hi) {
                            best_hi = best_67;
                            idx_hi = idx_67;
                        }
                        if (best_hi > best_group) {
                            best_group = best_hi;
                            idx_group = idx_hi;
                        }
                        float csq_vals_next[4];
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&csq_vals_next[0])), "=r"(*reinterpret_cast<uint32_t*>(&csq_vals_next[1])), "=r"(*reinterpret_cast<uint32_t*>(&csq_vals_next[2])), "=r"(*reinterpret_cast<uint32_t*>(&csq_vals_next[3]))
                            : "r"(csq_smem_addr + (score_base + kk + 8) * 4));
                        float d8 = scores[kk + 8] - csq_vals_next[0];
                        float d9 = scores[kk + 9] - csq_vals_next[1];
                        float best_89 = d8;
                        int idx_89 = off_k + score_base + kk + 8;
                        if (d9 > best_89) {
                            best_89 = d9;
                            idx_89 = off_k + score_base + kk + 9;
                        }
                        float d10 = scores[kk + 10] - csq_vals_next[2];
                        float d11 = scores[kk + 11] - csq_vals_next[3];
                        float best_1011 = d10;
                        int idx_1011 = off_k + score_base + kk + 10;
                        if (d11 > best_1011) {
                            best_1011 = d11;
                            idx_1011 = off_k + score_base + kk + 11;
                        }
                        float best_next = best_89;
                        int idx_next = idx_89;
                        if (best_1011 > best_next) {
                            best_next = best_1011;
                            idx_next = idx_1011;
                        }
                        float csq_vals_tail[4];
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&csq_vals_tail[0])), "=r"(*reinterpret_cast<uint32_t*>(&csq_vals_tail[1])), "=r"(*reinterpret_cast<uint32_t*>(&csq_vals_tail[2])), "=r"(*reinterpret_cast<uint32_t*>(&csq_vals_tail[3]))
                            : "r"(csq_smem_addr + (score_base + kk + 12) * 4));
                        float d12 = scores[kk + 12] - csq_vals_tail[0];
                        float d13 = scores[kk + 13] - csq_vals_tail[1];
                        float best_1213 = d12;
                        int idx_1213 = off_k + score_base + kk + 12;
                        if (d13 > best_1213) {
                            best_1213 = d13;
                            idx_1213 = off_k + score_base + kk + 13;
                        }
                        float d14 = scores[kk + 14] - csq_vals_tail[2];
                        float d15 = scores[kk + 15] - csq_vals_tail[3];
                        float best_1415 = d14;
                        int idx_1415 = off_k + score_base + kk + 14;
                        if (d15 > best_1415) {
                            best_1415 = d15;
                            idx_1415 = off_k + score_base + kk + 15;
                        }
                        float best_tail = best_1213;
                        int idx_tail = idx_1213;
                        if (best_1415 > best_tail) {
                            best_tail = best_1415;
                            idx_tail = idx_1415;
                        }
                        if (best_tail > best_next) {
                            best_next = best_tail;
                            idx_next = idx_tail;
                        }
                        if (best_next > best_group) {
                            best_group = best_next;
                            idx_group = idx_next;
                        }
                        if (best_group > best_score0) {
                            best_score0 = best_group;
                            best_idx0 = idx_group;
                        }
                    }
                    mbarrier_wait(score1_full_addr, _phase_score1_full_0);
                    _phase_score1_full_0 ^= 1;
                    score_base = 0;
                    asm volatile(
                        "tcgen05.ld.sync.aligned.32x32b.x128.b32"
                        " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63, %64, %65, %66, %67, %68, %69, %70, %71, %72, %73, %74, %75, %76, %77, %78, %79, %80, %81, %82, %83, %84, %85, %86, %87, %88, %89, %90, %91, %92, %93, %94, %95, %96, %97, %98, %99, %100, %101, %102, %103, %104, %105, %106, %107, %108, %109, %110, %111, %112, %113, %114, %115, %116, %117, %118, %119, %120, %121, %122, %123, %124, %125, %126, %127}, [%128];"
                        : "=f"(scores[0]), "=f"(scores[1]), "=f"(scores[2]), "=f"(scores[3]), "=f"(scores[4]), "=f"(scores[5]), "=f"(scores[6]), "=f"(scores[7]), "=f"(scores[8]), "=f"(scores[9]), "=f"(scores[10]), "=f"(scores[11]), "=f"(scores[12]), "=f"(scores[13]), "=f"(scores[14]), "=f"(scores[15]), "=f"(scores[16]), "=f"(scores[17]), "=f"(scores[18]), "=f"(scores[19]), "=f"(scores[20]), "=f"(scores[21]), "=f"(scores[22]), "=f"(scores[23]), "=f"(scores[24]), "=f"(scores[25]), "=f"(scores[26]), "=f"(scores[27]), "=f"(scores[28]), "=f"(scores[29]), "=f"(scores[30]), "=f"(scores[31]), "=f"(scores[32]), "=f"(scores[33]), "=f"(scores[34]), "=f"(scores[35]), "=f"(scores[36]), "=f"(scores[37]), "=f"(scores[38]), "=f"(scores[39]), "=f"(scores[40]), "=f"(scores[41]), "=f"(scores[42]), "=f"(scores[43]), "=f"(scores[44]), "=f"(scores[45]), "=f"(scores[46]), "=f"(scores[47]), "=f"(scores[48]), "=f"(scores[49]), "=f"(scores[50]), "=f"(scores[51]), "=f"(scores[52]), "=f"(scores[53]), "=f"(scores[54]), "=f"(scores[55]), "=f"(scores[56]), "=f"(scores[57]), "=f"(scores[58]), "=f"(scores[59]), "=f"(scores[60]), "=f"(scores[61]), "=f"(scores[62]), "=f"(scores[63]), "=f"(scores[64]), "=f"(scores[65]), "=f"(scores[66]), "=f"(scores[67]), "=f"(scores[68]), "=f"(scores[69]), "=f"(scores[70]), "=f"(scores[71]), "=f"(scores[72]), "=f"(scores[73]), "=f"(scores[74]), "=f"(scores[75]), "=f"(scores[76]), "=f"(scores[77]), "=f"(scores[78]), "=f"(scores[79]), "=f"(scores[80]), "=f"(scores[81]), "=f"(scores[82]), "=f"(scores[83]), "=f"(scores[84]), "=f"(scores[85]), "=f"(scores[86]), "=f"(scores[87]), "=f"(scores[88]), "=f"(scores[89]), "=f"(scores[90]), "=f"(scores[91]), "=f"(scores[92]), "=f"(scores[93]), "=f"(scores[94]), "=f"(scores[95]), "=f"(scores[96]), "=f"(scores[97]), "=f"(scores[98]), "=f"(scores[99]), "=f"(scores[100]), "=f"(scores[101]), "=f"(scores[102]), "=f"(scores[103]), "=f"(scores[104]), "=f"(scores[105]), "=f"(scores[106]), "=f"(scores[107]), "=f"(scores[108]), "=f"(scores[109]), "=f"(scores[110]), "=f"(scores[111]), "=f"(scores[112]), "=f"(scores[113]), "=f"(scores[114]), "=f"(scores[115]), "=f"(scores[116]), "=f"(scores[117]), "=f"(scores[118]), "=f"(scores[119]), "=f"(scores[120]), "=f"(scores[121]), "=f"(scores[122]), "=f"(scores[123]), "=f"(scores[124]), "=f"(scores[125]), "=f"(scores[126]), "=f"(scores[127])
                        : "r"(taddr + tmem_row_base_v + score_base)
                        : "memory");
                    asm volatile("tcgen05.wait::ld.sync.aligned;");
                    #pragma unroll
                    for (int kk = 0; kk < 128; kk += 16) {
                        float csq_vals[4];
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&csq_vals[0])), "=r"(*reinterpret_cast<uint32_t*>(&csq_vals[1])), "=r"(*reinterpret_cast<uint32_t*>(&csq_vals[2])), "=r"(*reinterpret_cast<uint32_t*>(&csq_vals[3]))
                            : "r"(csq_smem_addr + (score_base + kk) * 4));
                        float d0 = scores[kk] - csq_vals[0];
                        float d1 = scores[kk + 1] - csq_vals[1];
                        float best_01 = d0;
                        int idx_01 = off_k + score_base + kk;
                        if (d1 > best_01) {
                            best_01 = d1;
                            idx_01 = off_k + score_base + kk + 1;
                        }
                        float d2 = scores[kk + 2] - csq_vals[2];
                        float d3 = scores[kk + 3] - csq_vals[3];
                        float best_23 = d2;
                        int idx_23 = off_k + score_base + kk + 2;
                        if (d3 > best_23) {
                            best_23 = d3;
                            idx_23 = off_k + score_base + kk + 3;
                        }
                        float best_group = best_01;
                        int idx_group = idx_01;
                        if (best_23 > best_group) {
                            best_group = best_23;
                            idx_group = idx_23;
                        }
                        float csq_vals_hi[4];
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&csq_vals_hi[0])), "=r"(*reinterpret_cast<uint32_t*>(&csq_vals_hi[1])), "=r"(*reinterpret_cast<uint32_t*>(&csq_vals_hi[2])), "=r"(*reinterpret_cast<uint32_t*>(&csq_vals_hi[3]))
                            : "r"(csq_smem_addr + (score_base + kk + 4) * 4));
                        float d4 = scores[kk + 4] - csq_vals_hi[0];
                        float d5 = scores[kk + 5] - csq_vals_hi[1];
                        float best_45 = d4;
                        int idx_45 = off_k + score_base + kk + 4;
                        if (d5 > best_45) {
                            best_45 = d5;
                            idx_45 = off_k + score_base + kk + 5;
                        }
                        float d6 = scores[kk + 6] - csq_vals_hi[2];
                        float d7 = scores[kk + 7] - csq_vals_hi[3];
                        float best_67 = d6;
                        int idx_67 = off_k + score_base + kk + 6;
                        if (d7 > best_67) {
                            best_67 = d7;
                            idx_67 = off_k + score_base + kk + 7;
                        }
                        float best_hi = best_45;
                        int idx_hi = idx_45;
                        if (best_67 > best_hi) {
                            best_hi = best_67;
                            idx_hi = idx_67;
                        }
                        if (best_hi > best_group) {
                            best_group = best_hi;
                            idx_group = idx_hi;
                        }
                        float csq_vals_next[4];
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&csq_vals_next[0])), "=r"(*reinterpret_cast<uint32_t*>(&csq_vals_next[1])), "=r"(*reinterpret_cast<uint32_t*>(&csq_vals_next[2])), "=r"(*reinterpret_cast<uint32_t*>(&csq_vals_next[3]))
                            : "r"(csq_smem_addr + (score_base + kk + 8) * 4));
                        float d8 = scores[kk + 8] - csq_vals_next[0];
                        float d9 = scores[kk + 9] - csq_vals_next[1];
                        float best_89 = d8;
                        int idx_89 = off_k + score_base + kk + 8;
                        if (d9 > best_89) {
                            best_89 = d9;
                            idx_89 = off_k + score_base + kk + 9;
                        }
                        float d10 = scores[kk + 10] - csq_vals_next[2];
                        float d11 = scores[kk + 11] - csq_vals_next[3];
                        float best_1011 = d10;
                        int idx_1011 = off_k + score_base + kk + 10;
                        if (d11 > best_1011) {
                            best_1011 = d11;
                            idx_1011 = off_k + score_base + kk + 11;
                        }
                        float best_next = best_89;
                        int idx_next = idx_89;
                        if (best_1011 > best_next) {
                            best_next = best_1011;
                            idx_next = idx_1011;
                        }
                        float csq_vals_tail[4];
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&csq_vals_tail[0])), "=r"(*reinterpret_cast<uint32_t*>(&csq_vals_tail[1])), "=r"(*reinterpret_cast<uint32_t*>(&csq_vals_tail[2])), "=r"(*reinterpret_cast<uint32_t*>(&csq_vals_tail[3]))
                            : "r"(csq_smem_addr + (score_base + kk + 12) * 4));
                        float d12 = scores[kk + 12] - csq_vals_tail[0];
                        float d13 = scores[kk + 13] - csq_vals_tail[1];
                        float best_1213 = d12;
                        int idx_1213 = off_k + score_base + kk + 12;
                        if (d13 > best_1213) {
                            best_1213 = d13;
                            idx_1213 = off_k + score_base + kk + 13;
                        }
                        float d14 = scores[kk + 14] - csq_vals_tail[2];
                        float d15 = scores[kk + 15] - csq_vals_tail[3];
                        float best_1415 = d14;
                        int idx_1415 = off_k + score_base + kk + 14;
                        if (d15 > best_1415) {
                            best_1415 = d15;
                            idx_1415 = off_k + score_base + kk + 15;
                        }
                        float best_tail = best_1213;
                        int idx_tail = idx_1213;
                        if (best_1415 > best_tail) {
                            best_tail = best_1415;
                            idx_tail = idx_1415;
                        }
                        if (best_tail > best_next) {
                            best_next = best_tail;
                            idx_next = idx_tail;
                        }
                        if (best_next > best_group) {
                            best_group = best_next;
                            idx_group = idx_next;
                        }
                        if (best_group > best_score1) {
                            best_score1 = best_group;
                            best_idx1 = idx_group;
                        }
                    }
                    score_base = 128;
                    asm volatile(
                        "tcgen05.ld.sync.aligned.32x32b.x128.b32"
                        " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63, %64, %65, %66, %67, %68, %69, %70, %71, %72, %73, %74, %75, %76, %77, %78, %79, %80, %81, %82, %83, %84, %85, %86, %87, %88, %89, %90, %91, %92, %93, %94, %95, %96, %97, %98, %99, %100, %101, %102, %103, %104, %105, %106, %107, %108, %109, %110, %111, %112, %113, %114, %115, %116, %117, %118, %119, %120, %121, %122, %123, %124, %125, %126, %127}, [%128];"
                        : "=f"(scores[0]), "=f"(scores[1]), "=f"(scores[2]), "=f"(scores[3]), "=f"(scores[4]), "=f"(scores[5]), "=f"(scores[6]), "=f"(scores[7]), "=f"(scores[8]), "=f"(scores[9]), "=f"(scores[10]), "=f"(scores[11]), "=f"(scores[12]), "=f"(scores[13]), "=f"(scores[14]), "=f"(scores[15]), "=f"(scores[16]), "=f"(scores[17]), "=f"(scores[18]), "=f"(scores[19]), "=f"(scores[20]), "=f"(scores[21]), "=f"(scores[22]), "=f"(scores[23]), "=f"(scores[24]), "=f"(scores[25]), "=f"(scores[26]), "=f"(scores[27]), "=f"(scores[28]), "=f"(scores[29]), "=f"(scores[30]), "=f"(scores[31]), "=f"(scores[32]), "=f"(scores[33]), "=f"(scores[34]), "=f"(scores[35]), "=f"(scores[36]), "=f"(scores[37]), "=f"(scores[38]), "=f"(scores[39]), "=f"(scores[40]), "=f"(scores[41]), "=f"(scores[42]), "=f"(scores[43]), "=f"(scores[44]), "=f"(scores[45]), "=f"(scores[46]), "=f"(scores[47]), "=f"(scores[48]), "=f"(scores[49]), "=f"(scores[50]), "=f"(scores[51]), "=f"(scores[52]), "=f"(scores[53]), "=f"(scores[54]), "=f"(scores[55]), "=f"(scores[56]), "=f"(scores[57]), "=f"(scores[58]), "=f"(scores[59]), "=f"(scores[60]), "=f"(scores[61]), "=f"(scores[62]), "=f"(scores[63]), "=f"(scores[64]), "=f"(scores[65]), "=f"(scores[66]), "=f"(scores[67]), "=f"(scores[68]), "=f"(scores[69]), "=f"(scores[70]), "=f"(scores[71]), "=f"(scores[72]), "=f"(scores[73]), "=f"(scores[74]), "=f"(scores[75]), "=f"(scores[76]), "=f"(scores[77]), "=f"(scores[78]), "=f"(scores[79]), "=f"(scores[80]), "=f"(scores[81]), "=f"(scores[82]), "=f"(scores[83]), "=f"(scores[84]), "=f"(scores[85]), "=f"(scores[86]), "=f"(scores[87]), "=f"(scores[88]), "=f"(scores[89]), "=f"(scores[90]), "=f"(scores[91]), "=f"(scores[92]), "=f"(scores[93]), "=f"(scores[94]), "=f"(scores[95]), "=f"(scores[96]), "=f"(scores[97]), "=f"(scores[98]), "=f"(scores[99]), "=f"(scores[100]), "=f"(scores[101]), "=f"(scores[102]), "=f"(scores[103]), "=f"(scores[104]), "=f"(scores[105]), "=f"(scores[106]), "=f"(scores[107]), "=f"(scores[108]), "=f"(scores[109]), "=f"(scores[110]), "=f"(scores[111]), "=f"(scores[112]), "=f"(scores[113]), "=f"(scores[114]), "=f"(scores[115]), "=f"(scores[116]), "=f"(scores[117]), "=f"(scores[118]), "=f"(scores[119]), "=f"(scores[120]), "=f"(scores[121]), "=f"(scores[122]), "=f"(scores[123]), "=f"(scores[124]), "=f"(scores[125]), "=f"(scores[126]), "=f"(scores[127])
                        : "r"(taddr + tmem_row_base_v + score_base)
                        : "memory");
                    asm volatile("tcgen05.wait::ld.sync.aligned;");
                    asm volatile("barrier.sync 9, 128;");
                    if (elect_sync()) {
                        mbarrier_arrive(score1_empty_addr);
                    }
                    #pragma unroll
                    for (int kk = 0; kk < 128; kk += 16) {
                        float csq_vals[4];
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&csq_vals[0])), "=r"(*reinterpret_cast<uint32_t*>(&csq_vals[1])), "=r"(*reinterpret_cast<uint32_t*>(&csq_vals[2])), "=r"(*reinterpret_cast<uint32_t*>(&csq_vals[3]))
                            : "r"(csq_smem_addr + (score_base + kk) * 4));
                        float d0 = scores[kk] - csq_vals[0];
                        float d1 = scores[kk + 1] - csq_vals[1];
                        float best_01 = d0;
                        int idx_01 = off_k + score_base + kk;
                        if (d1 > best_01) {
                            best_01 = d1;
                            idx_01 = off_k + score_base + kk + 1;
                        }
                        float d2 = scores[kk + 2] - csq_vals[2];
                        float d3 = scores[kk + 3] - csq_vals[3];
                        float best_23 = d2;
                        int idx_23 = off_k + score_base + kk + 2;
                        if (d3 > best_23) {
                            best_23 = d3;
                            idx_23 = off_k + score_base + kk + 3;
                        }
                        float best_group = best_01;
                        int idx_group = idx_01;
                        if (best_23 > best_group) {
                            best_group = best_23;
                            idx_group = idx_23;
                        }
                        float csq_vals_hi[4];
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&csq_vals_hi[0])), "=r"(*reinterpret_cast<uint32_t*>(&csq_vals_hi[1])), "=r"(*reinterpret_cast<uint32_t*>(&csq_vals_hi[2])), "=r"(*reinterpret_cast<uint32_t*>(&csq_vals_hi[3]))
                            : "r"(csq_smem_addr + (score_base + kk + 4) * 4));
                        float d4 = scores[kk + 4] - csq_vals_hi[0];
                        float d5 = scores[kk + 5] - csq_vals_hi[1];
                        float best_45 = d4;
                        int idx_45 = off_k + score_base + kk + 4;
                        if (d5 > best_45) {
                            best_45 = d5;
                            idx_45 = off_k + score_base + kk + 5;
                        }
                        float d6 = scores[kk + 6] - csq_vals_hi[2];
                        float d7 = scores[kk + 7] - csq_vals_hi[3];
                        float best_67 = d6;
                        int idx_67 = off_k + score_base + kk + 6;
                        if (d7 > best_67) {
                            best_67 = d7;
                            idx_67 = off_k + score_base + kk + 7;
                        }
                        float best_hi = best_45;
                        int idx_hi = idx_45;
                        if (best_67 > best_hi) {
                            best_hi = best_67;
                            idx_hi = idx_67;
                        }
                        if (best_hi > best_group) {
                            best_group = best_hi;
                            idx_group = idx_hi;
                        }
                        float csq_vals_next[4];
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&csq_vals_next[0])), "=r"(*reinterpret_cast<uint32_t*>(&csq_vals_next[1])), "=r"(*reinterpret_cast<uint32_t*>(&csq_vals_next[2])), "=r"(*reinterpret_cast<uint32_t*>(&csq_vals_next[3]))
                            : "r"(csq_smem_addr + (score_base + kk + 8) * 4));
                        float d8 = scores[kk + 8] - csq_vals_next[0];
                        float d9 = scores[kk + 9] - csq_vals_next[1];
                        float best_89 = d8;
                        int idx_89 = off_k + score_base + kk + 8;
                        if (d9 > best_89) {
                            best_89 = d9;
                            idx_89 = off_k + score_base + kk + 9;
                        }
                        float d10 = scores[kk + 10] - csq_vals_next[2];
                        float d11 = scores[kk + 11] - csq_vals_next[3];
                        float best_1011 = d10;
                        int idx_1011 = off_k + score_base + kk + 10;
                        if (d11 > best_1011) {
                            best_1011 = d11;
                            idx_1011 = off_k + score_base + kk + 11;
                        }
                        float best_next = best_89;
                        int idx_next = idx_89;
                        if (best_1011 > best_next) {
                            best_next = best_1011;
                            idx_next = idx_1011;
                        }
                        float csq_vals_tail[4];
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&csq_vals_tail[0])), "=r"(*reinterpret_cast<uint32_t*>(&csq_vals_tail[1])), "=r"(*reinterpret_cast<uint32_t*>(&csq_vals_tail[2])), "=r"(*reinterpret_cast<uint32_t*>(&csq_vals_tail[3]))
                            : "r"(csq_smem_addr + (score_base + kk + 12) * 4));
                        float d12 = scores[kk + 12] - csq_vals_tail[0];
                        float d13 = scores[kk + 13] - csq_vals_tail[1];
                        float best_1213 = d12;
                        int idx_1213 = off_k + score_base + kk + 12;
                        if (d13 > best_1213) {
                            best_1213 = d13;
                            idx_1213 = off_k + score_base + kk + 13;
                        }
                        float d14 = scores[kk + 14] - csq_vals_tail[2];
                        float d15 = scores[kk + 15] - csq_vals_tail[3];
                        float best_1415 = d14;
                        int idx_1415 = off_k + score_base + kk + 14;
                        if (d15 > best_1415) {
                            best_1415 = d15;
                            idx_1415 = off_k + score_base + kk + 15;
                        }
                        float best_tail = best_1213;
                        int idx_tail = idx_1213;
                        if (best_1415 > best_tail) {
                            best_tail = best_1415;
                            idx_tail = idx_1415;
                        }
                        if (best_tail > best_next) {
                            best_next = best_tail;
                            idx_next = idx_tail;
                        }
                        if (best_next > best_group) {
                            best_group = best_next;
                            idx_group = idx_next;
                        }
                        if (best_group > best_score1) {
                            best_score1 = best_group;
                            best_idx1 = idx_group;
                        }
                    }
                }
                if (global_n0 < N) {
                    *((int*)(out + out_offset0)) = best_idx0;
                }
                if (global_n1 < N) {
                    *((int*)(out + out_offset1)) = best_idx1;
                }
            }
        }
    }

    // Cleanup
    __syncthreads(); // barrier before TMEM dealloc

    if (warp == 1) {
        asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;" :: "r"(tmem_addr_storage[0]), "r"(512));
        asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;");
    }
}

} // extern "C"

#undef NUM_MAIN_STAGES
#undef SMEM_SMEM_CSQ_OFF
#undef SMEM_SMEM_CSQ_STAGE_BYTES
#undef SMEM_SMEM_CSQ_STRIDE
#undef SMEM_SMEM_C_OFF
#undef SMEM_SMEM_C_STAGE_BYTES
#undef SMEM_SMEM_C_STRIDE
#undef SMEM_SMEM_X0_OFF
#undef SMEM_SMEM_X0_STAGE_BYTES
#undef SMEM_SMEM_X0_STRIDE
#undef SMEM_SMEM_X1_OFF
#undef SMEM_SMEM_X1_STAGE_BYTES
#undef SMEM_SMEM_X1_STRIDE
#undef SMEM_TOTAL
#undef TMEM_NCOLS
#undef TMEM_SCORE_TMEM_OFFSET
#undef c_empty_addr
#undef c_full_addr
#undef score0_empty_addr
#undef score0_full_addr
#undef score1_empty_addr
#undef score1_full_addr
#undef smem_c_addr
#undef smem_csq_addr
#undef smem_x0_addr
#undef smem_x1_addr
#undef x0_empty_addr
#undef x0_full_addr
#undef x1_empty_addr
#undef x1_full_addr

#define TMEM_NCOLS 256
#define TMEM_SCORE_TMEM_OFFSET 0
#define NUM_MAIN_STAGES 1
#define SMEM_SMEM_X_OFF 1024
#define SMEM_SMEM_X_STAGE_BYTES 65536
#define SMEM_SMEM_X_STRIDE 65536
#define SMEM_SMEM_C_OFF 66560
#define SMEM_SMEM_C_STAGE_BYTES 131072
#define SMEM_SMEM_C_STRIDE 131072
#define SMEM_SMEM_CSQ_OFF 197632
#define SMEM_SMEM_CSQ_STAGE_BYTES 1024
#define SMEM_SMEM_CSQ_STRIDE 1024
#define SMEM_TOTAL 198656

extern "C" {

__global__ __launch_bounds__(192) void
kernel_flash_kmeans_assign_cleanroom_tcgen05_d256_single_v1(float* __restrict__ x_sq, float* __restrict__ c_sq, int32_t* __restrict__ out, const void* x_tmap, const void* c_tmap, int B, int N, int D, int K, int num_n_tiles, int K_tiles)
{
    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;

    extern __shared__ __align__(1024) char smem_raw[];
    int smem;
    smem = (int)(unsigned long long)__cvta_generic_to_shared(smem_raw);
    const int smem_smem_x = smem + 1024;
    const int smem_smem_c = smem + 66560;
    const int smem_smem_csq = smem + 197632;

    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    // Mbarrier init (6 groups, 6 barriers)
    // Mbarriers at smem_raw[0..48)

    if (warp == 0) {
        uint32_t leader = elect_sync();
        // x_full: 1 barriers, init_count=1
        mbarrier_init_pred(smem + 0, 1, leader);
        // x_empty: 1 barriers, init_count=1
        mbarrier_init_pred(smem + 8, 1, leader);
        // c_full: 1 barriers, init_count=1
        mbarrier_init_pred(smem + 16, 1, leader);
        // c_empty: 1 barriers, init_count=1
        mbarrier_init_pred(smem + 24, 1, leader);
        // score_full: 1 barriers, init_count=1
        mbarrier_init_pred(smem + 32, 1, leader);
        // score_empty: 1 barriers, init_count=4
        mbarrier_init_pred(smem + 40, 4, leader);
        asm volatile("fence.mbarrier_init.release.cluster;");
    }

    __syncthreads();

    // TMEM alloc (256 columns, 256 used)
    volatile int* tmem_addr_storage = (volatile int*)(smem_raw + 48);
    if (warp == 1) {
        int _tmem_hold = smem + 48;
        asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;" :: "r"(_tmem_hold), "r"(256) : "memory");
    }

    __syncthreads();
    asm volatile("tcgen05.fence::after_thread_sync;");

    const int warp_id = warp;
    const int lane_id = lane;
    __nv_bfloat16* smem_x = (__nv_bfloat16*)(smem_raw + 1024);
    #define smem_x_addr (smem + 1024)
    __nv_bfloat16* smem_c = (__nv_bfloat16*)(smem_raw + 66560);
    #define smem_c_addr (smem + 66560)
    float* smem_csq = (float*)(smem_raw + 197632);
    #define smem_csq_addr (smem + 197632)
    const int mbar_base = smem;
    #define x_full_addr (mbar_base + 0)
    #define x_empty_addr (mbar_base + 8)
    #define c_full_addr (mbar_base + 16)
    #define c_empty_addr (mbar_base + 24)
    #define score_full_addr (mbar_base + 32)
    #define score_empty_addr (mbar_base + 40)
    const int taddr = tmem_addr_storage[0];

    // ---- Role: load ----
    if (warp == 0) {
        { // load_main
            int num_tiles = B * num_n_tiles;
            uint32_t _phase_x_empty_0 = 1;
            uint32_t _phase_c_empty_0 = 1;
            if (warp_id == 0) {
                if (elect_sync()) {
                    #pragma unroll 1
                    for (unsigned int tile_idx = bid; tile_idx < num_tiles; tile_idx += num_bids) {
                        int batch = tile_idx / num_n_tiles;
                        int n_tile = tile_idx % num_n_tiles;
                        int off_n = n_tile * 128;
                        int x_row = batch * N + off_n;
                        mbarrier_wait(x_empty_addr, _phase_x_empty_0);
                        _phase_x_empty_0 ^= 1;
                        tma_3d_gmem2smem(smem_x_addr, x_tmap, 0, x_row, 0, x_full_addr);
                        mbarrier_arrive_expect_tx(x_full_addr, 65536);
                        #pragma unroll 1
                        for (int iter_k = 0; iter_k < K_tiles; iter_k++) {
                            int off_k = iter_k * 256;
                            int c_row = batch * K + off_k;
                            mbarrier_wait(c_empty_addr, _phase_c_empty_0);
                            _phase_c_empty_0 ^= 1;
                            tma_3d_gmem2smem(smem_c_addr, c_tmap, 0, c_row, 0, c_full_addr);
                            mbarrier_arrive_expect_tx(c_full_addr, 131072);
                        }
                    }
                }
            }
        }
    // ---- Role: mma ----
    } else if (warp == 1) {
        const int tmem_score_tmem = taddr + TMEM_SCORE_TMEM_OFFSET;
        { // mma_main
            int num_tiles = B * num_n_tiles;
            uint32_t _phase_x_full_0 = 0;
            uint32_t _phase_score_empty_0 = 1;
            uint32_t _phase_c_full_0 = 0;
            #pragma unroll 1
            for (unsigned int tile_idx = bid; tile_idx < num_tiles; tile_idx += num_bids) {
                mbarrier_wait(x_full_addr, _phase_x_full_0);
                _phase_x_full_0 ^= 1;
                #pragma unroll 1
                for (int iter_k = 0; iter_k < K_tiles; iter_k++) {
                    mbarrier_wait(score_empty_addr, _phase_score_empty_0);
                    _phase_score_empty_0 ^= 1;
                    mbarrier_wait(c_full_addr, _phase_c_full_0);
                    _phase_c_full_0 ^= 1;
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    int _desc_addr_0 = smem_x_addr + 0 * 65536 + 16384;
                    int _desc_lo_0 = make_warp_uniform((_desc_addr_0 >> 4) & 0x3FFF);
                    int _desc_addr_1 = smem_c_addr + 0 * 131072 + 32768;
                    int _desc_lo_1 = make_warp_uniform((_desc_addr_1 >> 4) & 0x3FFF);
                    int _desc_addr_2 = smem_x_addr + 0 * 65536 + 32768;
                    int _desc_lo_2 = make_warp_uniform((_desc_addr_2 >> 4) & 0x3FFF);
                    int _desc_addr_3 = smem_c_addr + 0 * 131072 + 65536;
                    int _desc_lo_3 = make_warp_uniform((_desc_addr_3 >> 4) & 0x3FFF);
                    int _desc_addr_4 = smem_x_addr + 0 * 65536 + 49152;
                    int _desc_lo_4 = make_warp_uniform((_desc_addr_4 >> 4) & 0x3FFF);
                    int _desc_addr_5 = smem_c_addr + 0 * 131072 + 98304;
                    int _desc_lo_5 = make_warp_uniform((_desc_addr_5 >> 4) & 0x3FFF);
                    int _mma_ss_a_addr_6 = smem_x_addr + 0 * 65536;
                    int _mma_ss_a_lo_6 = make_warp_uniform((_mma_ss_a_addr_6 >> 4) & 0x3FFF);
                    int _mma_ss_b_addr_6 = smem_c_addr + 0 * 131072;
                    int _mma_ss_b_lo_6 = make_warp_uniform((_mma_ss_b_addr_6 >> 4) & 0x3FFF);
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
                    "mov.b32 id, 138413200;\n\t"
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
                    :: "r"(_mma_ss_a_lo_6), "r"(_mma_ss_b_lo_6), "r"(tmem_score_tmem), "r"(0));
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
                    "mov.b32 id, 138413200;\n\t"
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
                    :: "r"(_desc_lo_0), "r"(_desc_lo_1), "r"(tmem_score_tmem), "r"(1));
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
                    "mov.b32 id, 138413200;\n\t"
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
                    :: "r"(_desc_lo_2), "r"(_desc_lo_3), "r"(tmem_score_tmem), "r"(1));
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
                    "mov.b32 id, 138413200;\n\t"
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
                    :: "r"(_desc_lo_4), "r"(_desc_lo_5), "r"(tmem_score_tmem), "r"(1));
                    elect_commit(score_full_addr);
                    elect_commit(c_empty_addr);
                }
                elect_commit(x_empty_addr);
            }
        }
    // ---- Role: compute ----
    } else if (warp >= 2 && warp <= 5) {
        const int warp_id_in_wg = warp % 4;
        const int my_row = warp_id_in_wg * 32 + lane;
        const int tmem_row_base_v = (warp_id_in_wg * 32) << 16;
        { // compute_main
            int num_tiles = B * num_n_tiles;
            uint32_t _phase_score_full_0 = 0;
            #pragma unroll 1
            for (unsigned int tile_idx = bid; tile_idx < num_tiles; tile_idx += num_bids) {
                int batch = tile_idx / num_n_tiles;
                int n_tile = tile_idx % num_n_tiles;
                int off_n = n_tile * 128;
                int global_n = off_n + my_row;
                int out_offset = batch * N + global_n;
                int csq_smem_addr = smem_csq_addr + 0;
                float best_score = -3.4e+38f;
                int best_idx = 0;
                #pragma unroll 1
                for (int iter_k = 0; iter_k < K_tiles; iter_k++) {
                    int off_k = iter_k * 256;
                    if (my_row < 64.0f) {
                        int csq_base = my_row * 4;
                        float csq_pack[4];
                        {
                            float4 _v4 = *reinterpret_cast<const float4*>(c_sq + batch * K + off_k + csq_base);
                            csq_pack[0 + 0] = _v4.x;
                            csq_pack[0 + 1] = _v4.y;
                            csq_pack[0 + 2] = _v4.z;
                            csq_pack[0 + 3] = _v4.w;
                        }
                        float csq_half[4];
                        csq_half[0] = 0.5f * csq_pack[0];
                        csq_half[1] = 0.5f * csq_pack[1];
                        csq_half[2] = 0.5f * csq_pack[2];
                        csq_half[3] = 0.5f * csq_pack[3];
                        asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(csq_smem_addr + csq_base * 4), "f"(csq_half[0]), "f"(csq_half[1]), "f"(csq_half[2]), "f"(csq_half[3]) : "memory");
                    }
                    asm volatile("barrier.sync 8, 128;");
                    mbarrier_wait(score_full_addr, _phase_score_full_0);
                    _phase_score_full_0 ^= 1;
                    int score_base = 0;
                    float scores[128];
                    asm volatile(
                        "tcgen05.ld.sync.aligned.32x32b.x128.b32"
                        " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63, %64, %65, %66, %67, %68, %69, %70, %71, %72, %73, %74, %75, %76, %77, %78, %79, %80, %81, %82, %83, %84, %85, %86, %87, %88, %89, %90, %91, %92, %93, %94, %95, %96, %97, %98, %99, %100, %101, %102, %103, %104, %105, %106, %107, %108, %109, %110, %111, %112, %113, %114, %115, %116, %117, %118, %119, %120, %121, %122, %123, %124, %125, %126, %127}, [%128];"
                        : "=f"(scores[0]), "=f"(scores[1]), "=f"(scores[2]), "=f"(scores[3]), "=f"(scores[4]), "=f"(scores[5]), "=f"(scores[6]), "=f"(scores[7]), "=f"(scores[8]), "=f"(scores[9]), "=f"(scores[10]), "=f"(scores[11]), "=f"(scores[12]), "=f"(scores[13]), "=f"(scores[14]), "=f"(scores[15]), "=f"(scores[16]), "=f"(scores[17]), "=f"(scores[18]), "=f"(scores[19]), "=f"(scores[20]), "=f"(scores[21]), "=f"(scores[22]), "=f"(scores[23]), "=f"(scores[24]), "=f"(scores[25]), "=f"(scores[26]), "=f"(scores[27]), "=f"(scores[28]), "=f"(scores[29]), "=f"(scores[30]), "=f"(scores[31]), "=f"(scores[32]), "=f"(scores[33]), "=f"(scores[34]), "=f"(scores[35]), "=f"(scores[36]), "=f"(scores[37]), "=f"(scores[38]), "=f"(scores[39]), "=f"(scores[40]), "=f"(scores[41]), "=f"(scores[42]), "=f"(scores[43]), "=f"(scores[44]), "=f"(scores[45]), "=f"(scores[46]), "=f"(scores[47]), "=f"(scores[48]), "=f"(scores[49]), "=f"(scores[50]), "=f"(scores[51]), "=f"(scores[52]), "=f"(scores[53]), "=f"(scores[54]), "=f"(scores[55]), "=f"(scores[56]), "=f"(scores[57]), "=f"(scores[58]), "=f"(scores[59]), "=f"(scores[60]), "=f"(scores[61]), "=f"(scores[62]), "=f"(scores[63]), "=f"(scores[64]), "=f"(scores[65]), "=f"(scores[66]), "=f"(scores[67]), "=f"(scores[68]), "=f"(scores[69]), "=f"(scores[70]), "=f"(scores[71]), "=f"(scores[72]), "=f"(scores[73]), "=f"(scores[74]), "=f"(scores[75]), "=f"(scores[76]), "=f"(scores[77]), "=f"(scores[78]), "=f"(scores[79]), "=f"(scores[80]), "=f"(scores[81]), "=f"(scores[82]), "=f"(scores[83]), "=f"(scores[84]), "=f"(scores[85]), "=f"(scores[86]), "=f"(scores[87]), "=f"(scores[88]), "=f"(scores[89]), "=f"(scores[90]), "=f"(scores[91]), "=f"(scores[92]), "=f"(scores[93]), "=f"(scores[94]), "=f"(scores[95]), "=f"(scores[96]), "=f"(scores[97]), "=f"(scores[98]), "=f"(scores[99]), "=f"(scores[100]), "=f"(scores[101]), "=f"(scores[102]), "=f"(scores[103]), "=f"(scores[104]), "=f"(scores[105]), "=f"(scores[106]), "=f"(scores[107]), "=f"(scores[108]), "=f"(scores[109]), "=f"(scores[110]), "=f"(scores[111]), "=f"(scores[112]), "=f"(scores[113]), "=f"(scores[114]), "=f"(scores[115]), "=f"(scores[116]), "=f"(scores[117]), "=f"(scores[118]), "=f"(scores[119]), "=f"(scores[120]), "=f"(scores[121]), "=f"(scores[122]), "=f"(scores[123]), "=f"(scores[124]), "=f"(scores[125]), "=f"(scores[126]), "=f"(scores[127])
                        : "r"(taddr + tmem_row_base_v + score_base)
                        : "memory");
                    asm volatile("tcgen05.wait::ld.sync.aligned;");
                    #pragma unroll
                    for (int kk = 0; kk < 128; kk += 4) {
                        float csq_vals[4];
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&csq_vals[0])), "=r"(*reinterpret_cast<uint32_t*>(&csq_vals[1])), "=r"(*reinterpret_cast<uint32_t*>(&csq_vals[2])), "=r"(*reinterpret_cast<uint32_t*>(&csq_vals[3]))
                            : "r"(csq_smem_addr + (score_base + kk) * 4));
                        float d0 = scores[kk] - csq_vals[0];
                        float d1 = scores[kk + 1] - csq_vals[1];
                        float best_01 = d0;
                        int idx_01 = off_k + score_base + kk;
                        if (d1 > best_01) {
                            best_01 = d1;
                            idx_01 = off_k + score_base + kk + 1;
                        }
                        float d2 = scores[kk + 2] - csq_vals[2];
                        float d3 = scores[kk + 3] - csq_vals[3];
                        float best_23 = d2;
                        int idx_23 = off_k + score_base + kk + 2;
                        if (d3 > best_23) {
                            best_23 = d3;
                            idx_23 = off_k + score_base + kk + 3;
                        }
                        float best_group = best_01;
                        int idx_group = idx_01;
                        if (best_23 > best_group) {
                            best_group = best_23;
                            idx_group = idx_23;
                        }
                        if (best_group > best_score) {
                            best_score = best_group;
                            best_idx = idx_group;
                        }
                    }
                    score_base = 128;
                    asm volatile(
                        "tcgen05.ld.sync.aligned.32x32b.x128.b32"
                        " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63, %64, %65, %66, %67, %68, %69, %70, %71, %72, %73, %74, %75, %76, %77, %78, %79, %80, %81, %82, %83, %84, %85, %86, %87, %88, %89, %90, %91, %92, %93, %94, %95, %96, %97, %98, %99, %100, %101, %102, %103, %104, %105, %106, %107, %108, %109, %110, %111, %112, %113, %114, %115, %116, %117, %118, %119, %120, %121, %122, %123, %124, %125, %126, %127}, [%128];"
                        : "=f"(scores[0]), "=f"(scores[1]), "=f"(scores[2]), "=f"(scores[3]), "=f"(scores[4]), "=f"(scores[5]), "=f"(scores[6]), "=f"(scores[7]), "=f"(scores[8]), "=f"(scores[9]), "=f"(scores[10]), "=f"(scores[11]), "=f"(scores[12]), "=f"(scores[13]), "=f"(scores[14]), "=f"(scores[15]), "=f"(scores[16]), "=f"(scores[17]), "=f"(scores[18]), "=f"(scores[19]), "=f"(scores[20]), "=f"(scores[21]), "=f"(scores[22]), "=f"(scores[23]), "=f"(scores[24]), "=f"(scores[25]), "=f"(scores[26]), "=f"(scores[27]), "=f"(scores[28]), "=f"(scores[29]), "=f"(scores[30]), "=f"(scores[31]), "=f"(scores[32]), "=f"(scores[33]), "=f"(scores[34]), "=f"(scores[35]), "=f"(scores[36]), "=f"(scores[37]), "=f"(scores[38]), "=f"(scores[39]), "=f"(scores[40]), "=f"(scores[41]), "=f"(scores[42]), "=f"(scores[43]), "=f"(scores[44]), "=f"(scores[45]), "=f"(scores[46]), "=f"(scores[47]), "=f"(scores[48]), "=f"(scores[49]), "=f"(scores[50]), "=f"(scores[51]), "=f"(scores[52]), "=f"(scores[53]), "=f"(scores[54]), "=f"(scores[55]), "=f"(scores[56]), "=f"(scores[57]), "=f"(scores[58]), "=f"(scores[59]), "=f"(scores[60]), "=f"(scores[61]), "=f"(scores[62]), "=f"(scores[63]), "=f"(scores[64]), "=f"(scores[65]), "=f"(scores[66]), "=f"(scores[67]), "=f"(scores[68]), "=f"(scores[69]), "=f"(scores[70]), "=f"(scores[71]), "=f"(scores[72]), "=f"(scores[73]), "=f"(scores[74]), "=f"(scores[75]), "=f"(scores[76]), "=f"(scores[77]), "=f"(scores[78]), "=f"(scores[79]), "=f"(scores[80]), "=f"(scores[81]), "=f"(scores[82]), "=f"(scores[83]), "=f"(scores[84]), "=f"(scores[85]), "=f"(scores[86]), "=f"(scores[87]), "=f"(scores[88]), "=f"(scores[89]), "=f"(scores[90]), "=f"(scores[91]), "=f"(scores[92]), "=f"(scores[93]), "=f"(scores[94]), "=f"(scores[95]), "=f"(scores[96]), "=f"(scores[97]), "=f"(scores[98]), "=f"(scores[99]), "=f"(scores[100]), "=f"(scores[101]), "=f"(scores[102]), "=f"(scores[103]), "=f"(scores[104]), "=f"(scores[105]), "=f"(scores[106]), "=f"(scores[107]), "=f"(scores[108]), "=f"(scores[109]), "=f"(scores[110]), "=f"(scores[111]), "=f"(scores[112]), "=f"(scores[113]), "=f"(scores[114]), "=f"(scores[115]), "=f"(scores[116]), "=f"(scores[117]), "=f"(scores[118]), "=f"(scores[119]), "=f"(scores[120]), "=f"(scores[121]), "=f"(scores[122]), "=f"(scores[123]), "=f"(scores[124]), "=f"(scores[125]), "=f"(scores[126]), "=f"(scores[127])
                        : "r"(taddr + tmem_row_base_v + score_base)
                        : "memory");
                    asm volatile("tcgen05.wait::ld.sync.aligned;");
                    asm volatile("barrier.sync 9, 128;");
                    if (elect_sync()) {
                        mbarrier_arrive(score_empty_addr);
                    }
                    #pragma unroll
                    for (int kk = 0; kk < 128; kk += 4) {
                        float csq_vals[4];
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&csq_vals[0])), "=r"(*reinterpret_cast<uint32_t*>(&csq_vals[1])), "=r"(*reinterpret_cast<uint32_t*>(&csq_vals[2])), "=r"(*reinterpret_cast<uint32_t*>(&csq_vals[3]))
                            : "r"(csq_smem_addr + (score_base + kk) * 4));
                        float d0 = scores[kk] - csq_vals[0];
                        float d1 = scores[kk + 1] - csq_vals[1];
                        float best_01 = d0;
                        int idx_01 = off_k + score_base + kk;
                        if (d1 > best_01) {
                            best_01 = d1;
                            idx_01 = off_k + score_base + kk + 1;
                        }
                        float d2 = scores[kk + 2] - csq_vals[2];
                        float d3 = scores[kk + 3] - csq_vals[3];
                        float best_23 = d2;
                        int idx_23 = off_k + score_base + kk + 2;
                        if (d3 > best_23) {
                            best_23 = d3;
                            idx_23 = off_k + score_base + kk + 3;
                        }
                        float best_group = best_01;
                        int idx_group = idx_01;
                        if (best_23 > best_group) {
                            best_group = best_23;
                            idx_group = idx_23;
                        }
                        if (best_group > best_score) {
                            best_score = best_group;
                            best_idx = idx_group;
                        }
                    }
                }
                if (global_n < N) {
                    *((int*)(out + out_offset)) = best_idx;
                }
            }
        }
    }

    // Cleanup
    __syncthreads(); // barrier before TMEM dealloc

    if (warp == 1) {
        asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;" :: "r"(tmem_addr_storage[0]), "r"(256));
        asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;");
    }
}

} // extern "C"

#undef NUM_MAIN_STAGES
#undef SMEM_SMEM_CSQ_OFF
#undef SMEM_SMEM_CSQ_STAGE_BYTES
#undef SMEM_SMEM_CSQ_STRIDE
#undef SMEM_SMEM_C_OFF
#undef SMEM_SMEM_C_STAGE_BYTES
#undef SMEM_SMEM_C_STRIDE
#undef SMEM_SMEM_X_OFF
#undef SMEM_SMEM_X_STAGE_BYTES
#undef SMEM_SMEM_X_STRIDE
#undef SMEM_TOTAL
#undef TMEM_NCOLS
#undef TMEM_SCORE_TMEM_OFFSET
#undef c_empty_addr
#undef c_full_addr
#undef score_empty_addr
#undef score_full_addr
#undef smem_c_addr
#undef smem_csq_addr
#undef smem_x_addr
#undef x_empty_addr
#undef x_full_addr

#define TMEM_NCOLS 512
#define TMEM_SCORE_TMEM_OFFSET 0
#define NUM_MAIN_STAGES 1
#define SMEM_SMEM_X0_OFF 1024
#define SMEM_SMEM_X0_STAGE_BYTES 65536
#define SMEM_SMEM_X0_STRIDE 65536
#define SMEM_SMEM_X1_OFF 66560
#define SMEM_SMEM_X1_STAGE_BYTES 65536
#define SMEM_SMEM_X1_STRIDE 65536
#define SMEM_SMEM_C_OFF 132096
#define SMEM_SMEM_C_STAGE_BYTES 131072
#define SMEM_SMEM_C_STRIDE 131072
#define SMEM_SMEM_CSQ_OFF 263168
#define SMEM_SMEM_CSQ_STAGE_BYTES 1024
#define SMEM_SMEM_CSQ_STRIDE 1024
#define SMEM_TOTAL 264192

extern "C" {

__global__ __launch_bounds__(192) void
kernel_flash_kmeans_assign_cleanroom_tcgen05_d256_splitd_v1(float* __restrict__ x_sq, float* __restrict__ c_sq, int32_t* __restrict__ out, const void* x_tmap, const void* c_tmap, int B, int N, int D, int K, int num_n_tiles, int K_tiles)
{
    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;

    extern __shared__ __align__(1024) char smem_raw[];
    int smem;
    smem = (int)(unsigned long long)__cvta_generic_to_shared(smem_raw);
    const int smem_smem_x0 = smem + 1024;
    const int smem_smem_x1 = smem + 66560;
    const int smem_smem_c = smem + 132096;
    const int smem_smem_csq = smem + 263168;

    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    // Mbarrier init (10 groups, 10 barriers)
    // Mbarriers at smem_raw[0..80)

    if (warp == 0) {
        uint32_t leader = elect_sync();
        // x0_full: 1 barriers, init_count=1
        mbarrier_init_pred(smem + 0, 1, leader);
        // x0_empty: 1 barriers, init_count=1
        mbarrier_init_pred(smem + 8, 1, leader);
        // x1_full: 1 barriers, init_count=1
        mbarrier_init_pred(smem + 16, 1, leader);
        // x1_empty: 1 barriers, init_count=1
        mbarrier_init_pred(smem + 24, 1, leader);
        // c_full: 1 barriers, init_count=1
        mbarrier_init_pred(smem + 32, 1, leader);
        // c_empty: 1 barriers, init_count=1
        mbarrier_init_pred(smem + 40, 1, leader);
        // score0_full: 1 barriers, init_count=1
        mbarrier_init_pred(smem + 48, 1, leader);
        // score0_empty: 1 barriers, init_count=4
        mbarrier_init_pred(smem + 56, 4, leader);
        // score1_full: 1 barriers, init_count=1
        mbarrier_init_pred(smem + 64, 1, leader);
        // score1_empty: 1 barriers, init_count=4
        mbarrier_init_pred(smem + 72, 4, leader);
        asm volatile("fence.mbarrier_init.release.cluster;");
    }

    __syncthreads();

    // TMEM alloc (512 columns, 512 used)
    volatile int* tmem_addr_storage = (volatile int*)(smem_raw + 80);
    if (warp == 1) {
        int _tmem_hold = smem + 80;
        asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;" :: "r"(_tmem_hold), "r"(512) : "memory");
    }

    __syncthreads();
    asm volatile("tcgen05.fence::after_thread_sync;");

    const int warp_id = warp;
    const int lane_id = lane;
    __nv_bfloat16* smem_x0 = (__nv_bfloat16*)(smem_raw + 1024);
    #define smem_x0_addr (smem + 1024)
    __nv_bfloat16* smem_x1 = (__nv_bfloat16*)(smem_raw + 66560);
    #define smem_x1_addr (smem + 66560)
    __nv_bfloat16* smem_c = (__nv_bfloat16*)(smem_raw + 132096);
    #define smem_c_addr (smem + 132096)
    float* smem_csq = (float*)(smem_raw + 263168);
    #define smem_csq_addr (smem + 263168)
    const int mbar_base = smem;
    #define x0_full_addr (mbar_base + 0)
    #define x0_empty_addr (mbar_base + 8)
    #define x1_full_addr (mbar_base + 16)
    #define x1_empty_addr (mbar_base + 24)
    #define c_full_addr (mbar_base + 32)
    #define c_empty_addr (mbar_base + 40)
    #define score0_full_addr (mbar_base + 48)
    #define score0_empty_addr (mbar_base + 56)
    #define score1_full_addr (mbar_base + 64)
    #define score1_empty_addr (mbar_base + 72)
    const int taddr = tmem_addr_storage[0];

    // ---- Role: load ----
    if (warp == 0) {
        { // load_main
            int pair_n_tiles = num_n_tiles / 2;
            int num_tiles = B * pair_n_tiles;
            uint32_t _phase_x0_empty_0 = 1;
            uint32_t _phase_x1_empty_0 = 1;
            uint32_t _phase_c_empty_0 = 1;
            if (warp_id == 0) {
                if (elect_sync()) {
                    #pragma unroll 1
                    for (unsigned int tile_idx = bid; tile_idx < num_tiles; tile_idx += num_bids) {
                        int batch = tile_idx / pair_n_tiles;
                        int pair_tile = tile_idx % pair_n_tiles;
                        int off_n0 = pair_tile * 256;
                        int off_n1 = off_n0 + 128;
                        int x_row0 = batch * N + off_n0;
                        int x_row1 = batch * N + off_n1;
                        mbarrier_wait(x0_empty_addr, _phase_x0_empty_0);
                        _phase_x0_empty_0 ^= 1;
                        tma_3d_gmem2smem(smem_x0_addr, x_tmap, 0, x_row0, 0, x0_full_addr);
                        mbarrier_arrive_expect_tx(x0_full_addr, 65536);
                        mbarrier_wait(x1_empty_addr, _phase_x1_empty_0);
                        _phase_x1_empty_0 ^= 1;
                        tma_3d_gmem2smem(smem_x1_addr, x_tmap, 0, x_row1, 0, x1_full_addr);
                        mbarrier_arrive_expect_tx(x1_full_addr, 65536);
                        #pragma unroll 1
                        for (int iter_k = 0; iter_k < K_tiles; iter_k++) {
                            int off_k = iter_k * 256;
                            int c_row = batch * K + off_k;
                            mbarrier_wait(c_empty_addr, _phase_c_empty_0);
                            _phase_c_empty_0 ^= 1;
                            tma_3d_gmem2smem(smem_c_addr, c_tmap, 0, c_row, 0, c_full_addr);
                            mbarrier_arrive_expect_tx(c_full_addr, 131072);
                        }
                    }
                }
            }
        }
    // ---- Role: mma ----
    } else if (warp == 1) {
        const int tmem_score_tmem = taddr + TMEM_SCORE_TMEM_OFFSET;
        { // mma_main
            int pair_n_tiles = num_n_tiles / 2;
            int num_tiles = B * pair_n_tiles;
            uint32_t _phase_x0_full_0 = 0;
            uint32_t _phase_x1_full_0 = 0;
            uint32_t _phase_score0_empty_0 = 1;
            uint32_t _phase_c_full_0 = 0;
            uint32_t _phase_score1_empty_0 = 1;
            #pragma unroll 1
            for (unsigned int tile_idx = bid; tile_idx < num_tiles; tile_idx += num_bids) {
                mbarrier_wait(x0_full_addr, _phase_x0_full_0);
                _phase_x0_full_0 ^= 1;
                mbarrier_wait(x1_full_addr, _phase_x1_full_0);
                _phase_x1_full_0 ^= 1;
                #pragma unroll 1
                for (int iter_k = 0; iter_k < K_tiles; iter_k++) {
                    mbarrier_wait(score0_empty_addr, _phase_score0_empty_0);
                    _phase_score0_empty_0 ^= 1;
                    mbarrier_wait(c_full_addr, _phase_c_full_0);
                    _phase_c_full_0 ^= 1;
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    int _desc_addr_0 = smem_x0_addr + 0 * 65536 + 16384;
                    int _desc_lo_0 = make_warp_uniform((_desc_addr_0 >> 4) & 0x3FFF);
                    int _desc_addr_1 = smem_c_addr + 0 * 131072 + 32768;
                    int _desc_lo_1 = make_warp_uniform((_desc_addr_1 >> 4) & 0x3FFF);
                    int _desc_addr_2 = smem_x0_addr + 0 * 65536 + 32768;
                    int _desc_lo_2 = make_warp_uniform((_desc_addr_2 >> 4) & 0x3FFF);
                    int _desc_addr_3 = smem_c_addr + 0 * 131072 + 65536;
                    int _desc_lo_3 = make_warp_uniform((_desc_addr_3 >> 4) & 0x3FFF);
                    int _desc_addr_4 = smem_x0_addr + 0 * 65536 + 49152;
                    int _desc_lo_4 = make_warp_uniform((_desc_addr_4 >> 4) & 0x3FFF);
                    int _desc_addr_5 = smem_c_addr + 0 * 131072 + 98304;
                    int _desc_lo_5 = make_warp_uniform((_desc_addr_5 >> 4) & 0x3FFF);
                    int _mma_ss_a_addr_6 = smem_x0_addr + 0 * 65536;
                    int _mma_ss_a_lo_6 = make_warp_uniform((_mma_ss_a_addr_6 >> 4) & 0x3FFF);
                    int _mma_ss_b_addr_6 = smem_c_addr + 0 * 131072;
                    int _mma_ss_b_lo_6 = make_warp_uniform((_mma_ss_b_addr_6 >> 4) & 0x3FFF);
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
                    "mov.b32 id, 138413200;\n\t"
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
                    :: "r"(_mma_ss_a_lo_6), "r"(_mma_ss_b_lo_6), "r"((tmem_score_tmem + (256))), "r"(0));
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
                    "mov.b32 id, 138413200;\n\t"
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
                    :: "r"(_desc_lo_0), "r"(_desc_lo_1), "r"((tmem_score_tmem + (256))), "r"(1));
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
                    "mov.b32 id, 138413200;\n\t"
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
                    :: "r"(_desc_lo_2), "r"(_desc_lo_3), "r"((tmem_score_tmem + (256))), "r"(1));
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
                    "mov.b32 id, 138413200;\n\t"
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
                    :: "r"(_desc_lo_4), "r"(_desc_lo_5), "r"((tmem_score_tmem + (256))), "r"(1));
                    elect_commit(score0_full_addr);
                    mbarrier_wait(score1_empty_addr, _phase_score1_empty_0);
                    _phase_score1_empty_0 ^= 1;
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    int _desc_addr_10 = smem_x1_addr + 0 * 65536 + 16384;
                    int _desc_lo_10 = make_warp_uniform((_desc_addr_10 >> 4) & 0x3FFF);
                    int _desc_addr_11 = smem_x1_addr + 0 * 65536 + 32768;
                    int _desc_lo_11 = make_warp_uniform((_desc_addr_11 >> 4) & 0x3FFF);
                    int _desc_addr_12 = smem_x1_addr + 0 * 65536 + 49152;
                    int _desc_lo_12 = make_warp_uniform((_desc_addr_12 >> 4) & 0x3FFF);
                    int _mma_ss_a_addr_13 = smem_x1_addr + 0 * 65536;
                    int _mma_ss_a_lo_13 = make_warp_uniform((_mma_ss_a_addr_13 >> 4) & 0x3FFF);
                    int _mma_ss_b_addr_13 = smem_c_addr + 0 * 131072;
                    int _mma_ss_b_lo_13 = make_warp_uniform((_mma_ss_b_addr_13 >> 4) & 0x3FFF);
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
                    "mov.b32 id, 138413200;\n\t"
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
                    :: "r"(_mma_ss_a_lo_13), "r"(_mma_ss_b_lo_13), "r"(tmem_score_tmem), "r"(0));
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
                    "mov.b32 id, 138413200;\n\t"
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
                    :: "r"(_desc_lo_10), "r"(_desc_lo_1), "r"(tmem_score_tmem), "r"(1));
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
                    "mov.b32 id, 138413200;\n\t"
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
                    :: "r"(_desc_lo_11), "r"(_desc_lo_3), "r"(tmem_score_tmem), "r"(1));
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
                    "mov.b32 id, 138413200;\n\t"
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
                    :: "r"(_desc_lo_12), "r"(_desc_lo_5), "r"(tmem_score_tmem), "r"(1));
                    elect_commit(score1_full_addr);
                    elect_commit(c_empty_addr);
                }
                elect_commit(x0_empty_addr);
                elect_commit(x1_empty_addr);
            }
        }
    // ---- Role: compute ----
    } else if (warp >= 2 && warp <= 5) {
        const int warp_id_in_wg = warp % 4;
        const int my_row = warp_id_in_wg * 32 + lane;
        const int tmem_row_base_v = (warp_id_in_wg * 32) << 16;
        { // compute_main
            int pair_n_tiles = num_n_tiles / 2;
            int num_tiles = B * pair_n_tiles;
            uint32_t _phase_score0_full_0 = 0;
            uint32_t _phase_score1_full_0 = 0;
            #pragma unroll 1
            for (unsigned int tile_idx = bid; tile_idx < num_tiles; tile_idx += num_bids) {
                int batch = tile_idx / pair_n_tiles;
                int pair_tile = tile_idx % pair_n_tiles;
                int off_n0 = pair_tile * 256;
                int off_n1 = off_n0 + 128;
                int global_n0 = off_n0 + my_row;
                int global_n1 = off_n1 + my_row;
                int out_offset0 = batch * N + global_n0;
                int out_offset1 = batch * N + global_n1;
                int csq_smem_addr = smem_csq_addr + 0;
                float best_score0 = -3.4e+38f;
                int best_idx0 = 0;
                float best_score1 = -3.4e+38f;
                int best_idx1 = 0;
                #pragma unroll 1
                for (int iter_k = 0; iter_k < K_tiles; iter_k++) {
                    int off_k = iter_k * 256;
                    if (my_row < 64.0f) {
                        int csq_base = my_row * 4;
                        float csq_pack[4];
                        {
                            float4 _v4 = *reinterpret_cast<const float4*>(c_sq + batch * K + off_k + csq_base);
                            csq_pack[0 + 0] = _v4.x;
                            csq_pack[0 + 1] = _v4.y;
                            csq_pack[0 + 2] = _v4.z;
                            csq_pack[0 + 3] = _v4.w;
                        }
                        float csq_half[4];
                        csq_half[0] = 0.5f * csq_pack[0];
                        csq_half[1] = 0.5f * csq_pack[1];
                        csq_half[2] = 0.5f * csq_pack[2];
                        csq_half[3] = 0.5f * csq_pack[3];
                        asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(csq_smem_addr + csq_base * 4), "f"(csq_half[0]), "f"(csq_half[1]), "f"(csq_half[2]), "f"(csq_half[3]) : "memory");
                    }
                    asm volatile("barrier.sync 8, 128;");
                    mbarrier_wait(score0_full_addr, _phase_score0_full_0);
                    _phase_score0_full_0 ^= 1;
                    int score_base = 0;
                    float scores[128];
                    asm volatile(
                        "tcgen05.ld.sync.aligned.32x32b.x128.b32"
                        " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63, %64, %65, %66, %67, %68, %69, %70, %71, %72, %73, %74, %75, %76, %77, %78, %79, %80, %81, %82, %83, %84, %85, %86, %87, %88, %89, %90, %91, %92, %93, %94, %95, %96, %97, %98, %99, %100, %101, %102, %103, %104, %105, %106, %107, %108, %109, %110, %111, %112, %113, %114, %115, %116, %117, %118, %119, %120, %121, %122, %123, %124, %125, %126, %127}, [%128];"
                        : "=f"(scores[0]), "=f"(scores[1]), "=f"(scores[2]), "=f"(scores[3]), "=f"(scores[4]), "=f"(scores[5]), "=f"(scores[6]), "=f"(scores[7]), "=f"(scores[8]), "=f"(scores[9]), "=f"(scores[10]), "=f"(scores[11]), "=f"(scores[12]), "=f"(scores[13]), "=f"(scores[14]), "=f"(scores[15]), "=f"(scores[16]), "=f"(scores[17]), "=f"(scores[18]), "=f"(scores[19]), "=f"(scores[20]), "=f"(scores[21]), "=f"(scores[22]), "=f"(scores[23]), "=f"(scores[24]), "=f"(scores[25]), "=f"(scores[26]), "=f"(scores[27]), "=f"(scores[28]), "=f"(scores[29]), "=f"(scores[30]), "=f"(scores[31]), "=f"(scores[32]), "=f"(scores[33]), "=f"(scores[34]), "=f"(scores[35]), "=f"(scores[36]), "=f"(scores[37]), "=f"(scores[38]), "=f"(scores[39]), "=f"(scores[40]), "=f"(scores[41]), "=f"(scores[42]), "=f"(scores[43]), "=f"(scores[44]), "=f"(scores[45]), "=f"(scores[46]), "=f"(scores[47]), "=f"(scores[48]), "=f"(scores[49]), "=f"(scores[50]), "=f"(scores[51]), "=f"(scores[52]), "=f"(scores[53]), "=f"(scores[54]), "=f"(scores[55]), "=f"(scores[56]), "=f"(scores[57]), "=f"(scores[58]), "=f"(scores[59]), "=f"(scores[60]), "=f"(scores[61]), "=f"(scores[62]), "=f"(scores[63]), "=f"(scores[64]), "=f"(scores[65]), "=f"(scores[66]), "=f"(scores[67]), "=f"(scores[68]), "=f"(scores[69]), "=f"(scores[70]), "=f"(scores[71]), "=f"(scores[72]), "=f"(scores[73]), "=f"(scores[74]), "=f"(scores[75]), "=f"(scores[76]), "=f"(scores[77]), "=f"(scores[78]), "=f"(scores[79]), "=f"(scores[80]), "=f"(scores[81]), "=f"(scores[82]), "=f"(scores[83]), "=f"(scores[84]), "=f"(scores[85]), "=f"(scores[86]), "=f"(scores[87]), "=f"(scores[88]), "=f"(scores[89]), "=f"(scores[90]), "=f"(scores[91]), "=f"(scores[92]), "=f"(scores[93]), "=f"(scores[94]), "=f"(scores[95]), "=f"(scores[96]), "=f"(scores[97]), "=f"(scores[98]), "=f"(scores[99]), "=f"(scores[100]), "=f"(scores[101]), "=f"(scores[102]), "=f"(scores[103]), "=f"(scores[104]), "=f"(scores[105]), "=f"(scores[106]), "=f"(scores[107]), "=f"(scores[108]), "=f"(scores[109]), "=f"(scores[110]), "=f"(scores[111]), "=f"(scores[112]), "=f"(scores[113]), "=f"(scores[114]), "=f"(scores[115]), "=f"(scores[116]), "=f"(scores[117]), "=f"(scores[118]), "=f"(scores[119]), "=f"(scores[120]), "=f"(scores[121]), "=f"(scores[122]), "=f"(scores[123]), "=f"(scores[124]), "=f"(scores[125]), "=f"(scores[126]), "=f"(scores[127])
                        : "r"(taddr + tmem_row_base_v + 256 + score_base)
                        : "memory");
                    asm volatile("tcgen05.wait::ld.sync.aligned;");
                    #pragma unroll
                    for (int kk = 0; kk < 128; kk += 16) {
                        float csq_vals[4];
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&csq_vals[0])), "=r"(*reinterpret_cast<uint32_t*>(&csq_vals[1])), "=r"(*reinterpret_cast<uint32_t*>(&csq_vals[2])), "=r"(*reinterpret_cast<uint32_t*>(&csq_vals[3]))
                            : "r"(csq_smem_addr + (score_base + kk) * 4));
                        float d0 = scores[kk] - csq_vals[0];
                        float d1 = scores[kk + 1] - csq_vals[1];
                        float best_01 = d0;
                        int idx_01 = off_k + score_base + kk;
                        if (d1 > best_01) {
                            best_01 = d1;
                            idx_01 = off_k + score_base + kk + 1;
                        }
                        float d2 = scores[kk + 2] - csq_vals[2];
                        float d3 = scores[kk + 3] - csq_vals[3];
                        float best_23 = d2;
                        int idx_23 = off_k + score_base + kk + 2;
                        if (d3 > best_23) {
                            best_23 = d3;
                            idx_23 = off_k + score_base + kk + 3;
                        }
                        float best_group = best_01;
                        int idx_group = idx_01;
                        if (best_23 > best_group) {
                            best_group = best_23;
                            idx_group = idx_23;
                        }
                        float csq_vals_hi[4];
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&csq_vals_hi[0])), "=r"(*reinterpret_cast<uint32_t*>(&csq_vals_hi[1])), "=r"(*reinterpret_cast<uint32_t*>(&csq_vals_hi[2])), "=r"(*reinterpret_cast<uint32_t*>(&csq_vals_hi[3]))
                            : "r"(csq_smem_addr + (score_base + kk + 4) * 4));
                        float d4 = scores[kk + 4] - csq_vals_hi[0];
                        float d5 = scores[kk + 5] - csq_vals_hi[1];
                        float best_45 = d4;
                        int idx_45 = off_k + score_base + kk + 4;
                        if (d5 > best_45) {
                            best_45 = d5;
                            idx_45 = off_k + score_base + kk + 5;
                        }
                        float d6 = scores[kk + 6] - csq_vals_hi[2];
                        float d7 = scores[kk + 7] - csq_vals_hi[3];
                        float best_67 = d6;
                        int idx_67 = off_k + score_base + kk + 6;
                        if (d7 > best_67) {
                            best_67 = d7;
                            idx_67 = off_k + score_base + kk + 7;
                        }
                        float best_hi = best_45;
                        int idx_hi = idx_45;
                        if (best_67 > best_hi) {
                            best_hi = best_67;
                            idx_hi = idx_67;
                        }
                        if (best_hi > best_group) {
                            best_group = best_hi;
                            idx_group = idx_hi;
                        }
                        float csq_vals_next[4];
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&csq_vals_next[0])), "=r"(*reinterpret_cast<uint32_t*>(&csq_vals_next[1])), "=r"(*reinterpret_cast<uint32_t*>(&csq_vals_next[2])), "=r"(*reinterpret_cast<uint32_t*>(&csq_vals_next[3]))
                            : "r"(csq_smem_addr + (score_base + kk + 8) * 4));
                        float d8 = scores[kk + 8] - csq_vals_next[0];
                        float d9 = scores[kk + 9] - csq_vals_next[1];
                        float best_89 = d8;
                        int idx_89 = off_k + score_base + kk + 8;
                        if (d9 > best_89) {
                            best_89 = d9;
                            idx_89 = off_k + score_base + kk + 9;
                        }
                        float d10 = scores[kk + 10] - csq_vals_next[2];
                        float d11 = scores[kk + 11] - csq_vals_next[3];
                        float best_1011 = d10;
                        int idx_1011 = off_k + score_base + kk + 10;
                        if (d11 > best_1011) {
                            best_1011 = d11;
                            idx_1011 = off_k + score_base + kk + 11;
                        }
                        float best_next = best_89;
                        int idx_next = idx_89;
                        if (best_1011 > best_next) {
                            best_next = best_1011;
                            idx_next = idx_1011;
                        }
                        float csq_vals_tail[4];
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&csq_vals_tail[0])), "=r"(*reinterpret_cast<uint32_t*>(&csq_vals_tail[1])), "=r"(*reinterpret_cast<uint32_t*>(&csq_vals_tail[2])), "=r"(*reinterpret_cast<uint32_t*>(&csq_vals_tail[3]))
                            : "r"(csq_smem_addr + (score_base + kk + 12) * 4));
                        float d12 = scores[kk + 12] - csq_vals_tail[0];
                        float d13 = scores[kk + 13] - csq_vals_tail[1];
                        float best_1213 = d12;
                        int idx_1213 = off_k + score_base + kk + 12;
                        if (d13 > best_1213) {
                            best_1213 = d13;
                            idx_1213 = off_k + score_base + kk + 13;
                        }
                        float d14 = scores[kk + 14] - csq_vals_tail[2];
                        float d15 = scores[kk + 15] - csq_vals_tail[3];
                        float best_1415 = d14;
                        int idx_1415 = off_k + score_base + kk + 14;
                        if (d15 > best_1415) {
                            best_1415 = d15;
                            idx_1415 = off_k + score_base + kk + 15;
                        }
                        float best_tail = best_1213;
                        int idx_tail = idx_1213;
                        if (best_1415 > best_tail) {
                            best_tail = best_1415;
                            idx_tail = idx_1415;
                        }
                        if (best_tail > best_next) {
                            best_next = best_tail;
                            idx_next = idx_tail;
                        }
                        if (best_next > best_group) {
                            best_group = best_next;
                            idx_group = idx_next;
                        }
                        if (best_group > best_score0) {
                            best_score0 = best_group;
                            best_idx0 = idx_group;
                        }
                    }
                    score_base = 128;
                    asm volatile(
                        "tcgen05.ld.sync.aligned.32x32b.x128.b32"
                        " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63, %64, %65, %66, %67, %68, %69, %70, %71, %72, %73, %74, %75, %76, %77, %78, %79, %80, %81, %82, %83, %84, %85, %86, %87, %88, %89, %90, %91, %92, %93, %94, %95, %96, %97, %98, %99, %100, %101, %102, %103, %104, %105, %106, %107, %108, %109, %110, %111, %112, %113, %114, %115, %116, %117, %118, %119, %120, %121, %122, %123, %124, %125, %126, %127}, [%128];"
                        : "=f"(scores[0]), "=f"(scores[1]), "=f"(scores[2]), "=f"(scores[3]), "=f"(scores[4]), "=f"(scores[5]), "=f"(scores[6]), "=f"(scores[7]), "=f"(scores[8]), "=f"(scores[9]), "=f"(scores[10]), "=f"(scores[11]), "=f"(scores[12]), "=f"(scores[13]), "=f"(scores[14]), "=f"(scores[15]), "=f"(scores[16]), "=f"(scores[17]), "=f"(scores[18]), "=f"(scores[19]), "=f"(scores[20]), "=f"(scores[21]), "=f"(scores[22]), "=f"(scores[23]), "=f"(scores[24]), "=f"(scores[25]), "=f"(scores[26]), "=f"(scores[27]), "=f"(scores[28]), "=f"(scores[29]), "=f"(scores[30]), "=f"(scores[31]), "=f"(scores[32]), "=f"(scores[33]), "=f"(scores[34]), "=f"(scores[35]), "=f"(scores[36]), "=f"(scores[37]), "=f"(scores[38]), "=f"(scores[39]), "=f"(scores[40]), "=f"(scores[41]), "=f"(scores[42]), "=f"(scores[43]), "=f"(scores[44]), "=f"(scores[45]), "=f"(scores[46]), "=f"(scores[47]), "=f"(scores[48]), "=f"(scores[49]), "=f"(scores[50]), "=f"(scores[51]), "=f"(scores[52]), "=f"(scores[53]), "=f"(scores[54]), "=f"(scores[55]), "=f"(scores[56]), "=f"(scores[57]), "=f"(scores[58]), "=f"(scores[59]), "=f"(scores[60]), "=f"(scores[61]), "=f"(scores[62]), "=f"(scores[63]), "=f"(scores[64]), "=f"(scores[65]), "=f"(scores[66]), "=f"(scores[67]), "=f"(scores[68]), "=f"(scores[69]), "=f"(scores[70]), "=f"(scores[71]), "=f"(scores[72]), "=f"(scores[73]), "=f"(scores[74]), "=f"(scores[75]), "=f"(scores[76]), "=f"(scores[77]), "=f"(scores[78]), "=f"(scores[79]), "=f"(scores[80]), "=f"(scores[81]), "=f"(scores[82]), "=f"(scores[83]), "=f"(scores[84]), "=f"(scores[85]), "=f"(scores[86]), "=f"(scores[87]), "=f"(scores[88]), "=f"(scores[89]), "=f"(scores[90]), "=f"(scores[91]), "=f"(scores[92]), "=f"(scores[93]), "=f"(scores[94]), "=f"(scores[95]), "=f"(scores[96]), "=f"(scores[97]), "=f"(scores[98]), "=f"(scores[99]), "=f"(scores[100]), "=f"(scores[101]), "=f"(scores[102]), "=f"(scores[103]), "=f"(scores[104]), "=f"(scores[105]), "=f"(scores[106]), "=f"(scores[107]), "=f"(scores[108]), "=f"(scores[109]), "=f"(scores[110]), "=f"(scores[111]), "=f"(scores[112]), "=f"(scores[113]), "=f"(scores[114]), "=f"(scores[115]), "=f"(scores[116]), "=f"(scores[117]), "=f"(scores[118]), "=f"(scores[119]), "=f"(scores[120]), "=f"(scores[121]), "=f"(scores[122]), "=f"(scores[123]), "=f"(scores[124]), "=f"(scores[125]), "=f"(scores[126]), "=f"(scores[127])
                        : "r"(taddr + tmem_row_base_v + 256 + score_base)
                        : "memory");
                    asm volatile("tcgen05.wait::ld.sync.aligned;");
                    asm volatile("barrier.sync 9, 128;");
                    if (elect_sync()) {
                        mbarrier_arrive(score0_empty_addr);
                    }
                    #pragma unroll
                    for (int kk = 0; kk < 128; kk += 16) {
                        float csq_vals[4];
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&csq_vals[0])), "=r"(*reinterpret_cast<uint32_t*>(&csq_vals[1])), "=r"(*reinterpret_cast<uint32_t*>(&csq_vals[2])), "=r"(*reinterpret_cast<uint32_t*>(&csq_vals[3]))
                            : "r"(csq_smem_addr + (score_base + kk) * 4));
                        float d0 = scores[kk] - csq_vals[0];
                        float d1 = scores[kk + 1] - csq_vals[1];
                        float best_01 = d0;
                        int idx_01 = off_k + score_base + kk;
                        if (d1 > best_01) {
                            best_01 = d1;
                            idx_01 = off_k + score_base + kk + 1;
                        }
                        float d2 = scores[kk + 2] - csq_vals[2];
                        float d3 = scores[kk + 3] - csq_vals[3];
                        float best_23 = d2;
                        int idx_23 = off_k + score_base + kk + 2;
                        if (d3 > best_23) {
                            best_23 = d3;
                            idx_23 = off_k + score_base + kk + 3;
                        }
                        float best_group = best_01;
                        int idx_group = idx_01;
                        if (best_23 > best_group) {
                            best_group = best_23;
                            idx_group = idx_23;
                        }
                        float csq_vals_hi[4];
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&csq_vals_hi[0])), "=r"(*reinterpret_cast<uint32_t*>(&csq_vals_hi[1])), "=r"(*reinterpret_cast<uint32_t*>(&csq_vals_hi[2])), "=r"(*reinterpret_cast<uint32_t*>(&csq_vals_hi[3]))
                            : "r"(csq_smem_addr + (score_base + kk + 4) * 4));
                        float d4 = scores[kk + 4] - csq_vals_hi[0];
                        float d5 = scores[kk + 5] - csq_vals_hi[1];
                        float best_45 = d4;
                        int idx_45 = off_k + score_base + kk + 4;
                        if (d5 > best_45) {
                            best_45 = d5;
                            idx_45 = off_k + score_base + kk + 5;
                        }
                        float d6 = scores[kk + 6] - csq_vals_hi[2];
                        float d7 = scores[kk + 7] - csq_vals_hi[3];
                        float best_67 = d6;
                        int idx_67 = off_k + score_base + kk + 6;
                        if (d7 > best_67) {
                            best_67 = d7;
                            idx_67 = off_k + score_base + kk + 7;
                        }
                        float best_hi = best_45;
                        int idx_hi = idx_45;
                        if (best_67 > best_hi) {
                            best_hi = best_67;
                            idx_hi = idx_67;
                        }
                        if (best_hi > best_group) {
                            best_group = best_hi;
                            idx_group = idx_hi;
                        }
                        float csq_vals_next[4];
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&csq_vals_next[0])), "=r"(*reinterpret_cast<uint32_t*>(&csq_vals_next[1])), "=r"(*reinterpret_cast<uint32_t*>(&csq_vals_next[2])), "=r"(*reinterpret_cast<uint32_t*>(&csq_vals_next[3]))
                            : "r"(csq_smem_addr + (score_base + kk + 8) * 4));
                        float d8 = scores[kk + 8] - csq_vals_next[0];
                        float d9 = scores[kk + 9] - csq_vals_next[1];
                        float best_89 = d8;
                        int idx_89 = off_k + score_base + kk + 8;
                        if (d9 > best_89) {
                            best_89 = d9;
                            idx_89 = off_k + score_base + kk + 9;
                        }
                        float d10 = scores[kk + 10] - csq_vals_next[2];
                        float d11 = scores[kk + 11] - csq_vals_next[3];
                        float best_1011 = d10;
                        int idx_1011 = off_k + score_base + kk + 10;
                        if (d11 > best_1011) {
                            best_1011 = d11;
                            idx_1011 = off_k + score_base + kk + 11;
                        }
                        float best_next = best_89;
                        int idx_next = idx_89;
                        if (best_1011 > best_next) {
                            best_next = best_1011;
                            idx_next = idx_1011;
                        }
                        float csq_vals_tail[4];
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&csq_vals_tail[0])), "=r"(*reinterpret_cast<uint32_t*>(&csq_vals_tail[1])), "=r"(*reinterpret_cast<uint32_t*>(&csq_vals_tail[2])), "=r"(*reinterpret_cast<uint32_t*>(&csq_vals_tail[3]))
                            : "r"(csq_smem_addr + (score_base + kk + 12) * 4));
                        float d12 = scores[kk + 12] - csq_vals_tail[0];
                        float d13 = scores[kk + 13] - csq_vals_tail[1];
                        float best_1213 = d12;
                        int idx_1213 = off_k + score_base + kk + 12;
                        if (d13 > best_1213) {
                            best_1213 = d13;
                            idx_1213 = off_k + score_base + kk + 13;
                        }
                        float d14 = scores[kk + 14] - csq_vals_tail[2];
                        float d15 = scores[kk + 15] - csq_vals_tail[3];
                        float best_1415 = d14;
                        int idx_1415 = off_k + score_base + kk + 14;
                        if (d15 > best_1415) {
                            best_1415 = d15;
                            idx_1415 = off_k + score_base + kk + 15;
                        }
                        float best_tail = best_1213;
                        int idx_tail = idx_1213;
                        if (best_1415 > best_tail) {
                            best_tail = best_1415;
                            idx_tail = idx_1415;
                        }
                        if (best_tail > best_next) {
                            best_next = best_tail;
                            idx_next = idx_tail;
                        }
                        if (best_next > best_group) {
                            best_group = best_next;
                            idx_group = idx_next;
                        }
                        if (best_group > best_score0) {
                            best_score0 = best_group;
                            best_idx0 = idx_group;
                        }
                    }
                    mbarrier_wait(score1_full_addr, _phase_score1_full_0);
                    _phase_score1_full_0 ^= 1;
                    score_base = 0;
                    asm volatile(
                        "tcgen05.ld.sync.aligned.32x32b.x128.b32"
                        " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63, %64, %65, %66, %67, %68, %69, %70, %71, %72, %73, %74, %75, %76, %77, %78, %79, %80, %81, %82, %83, %84, %85, %86, %87, %88, %89, %90, %91, %92, %93, %94, %95, %96, %97, %98, %99, %100, %101, %102, %103, %104, %105, %106, %107, %108, %109, %110, %111, %112, %113, %114, %115, %116, %117, %118, %119, %120, %121, %122, %123, %124, %125, %126, %127}, [%128];"
                        : "=f"(scores[0]), "=f"(scores[1]), "=f"(scores[2]), "=f"(scores[3]), "=f"(scores[4]), "=f"(scores[5]), "=f"(scores[6]), "=f"(scores[7]), "=f"(scores[8]), "=f"(scores[9]), "=f"(scores[10]), "=f"(scores[11]), "=f"(scores[12]), "=f"(scores[13]), "=f"(scores[14]), "=f"(scores[15]), "=f"(scores[16]), "=f"(scores[17]), "=f"(scores[18]), "=f"(scores[19]), "=f"(scores[20]), "=f"(scores[21]), "=f"(scores[22]), "=f"(scores[23]), "=f"(scores[24]), "=f"(scores[25]), "=f"(scores[26]), "=f"(scores[27]), "=f"(scores[28]), "=f"(scores[29]), "=f"(scores[30]), "=f"(scores[31]), "=f"(scores[32]), "=f"(scores[33]), "=f"(scores[34]), "=f"(scores[35]), "=f"(scores[36]), "=f"(scores[37]), "=f"(scores[38]), "=f"(scores[39]), "=f"(scores[40]), "=f"(scores[41]), "=f"(scores[42]), "=f"(scores[43]), "=f"(scores[44]), "=f"(scores[45]), "=f"(scores[46]), "=f"(scores[47]), "=f"(scores[48]), "=f"(scores[49]), "=f"(scores[50]), "=f"(scores[51]), "=f"(scores[52]), "=f"(scores[53]), "=f"(scores[54]), "=f"(scores[55]), "=f"(scores[56]), "=f"(scores[57]), "=f"(scores[58]), "=f"(scores[59]), "=f"(scores[60]), "=f"(scores[61]), "=f"(scores[62]), "=f"(scores[63]), "=f"(scores[64]), "=f"(scores[65]), "=f"(scores[66]), "=f"(scores[67]), "=f"(scores[68]), "=f"(scores[69]), "=f"(scores[70]), "=f"(scores[71]), "=f"(scores[72]), "=f"(scores[73]), "=f"(scores[74]), "=f"(scores[75]), "=f"(scores[76]), "=f"(scores[77]), "=f"(scores[78]), "=f"(scores[79]), "=f"(scores[80]), "=f"(scores[81]), "=f"(scores[82]), "=f"(scores[83]), "=f"(scores[84]), "=f"(scores[85]), "=f"(scores[86]), "=f"(scores[87]), "=f"(scores[88]), "=f"(scores[89]), "=f"(scores[90]), "=f"(scores[91]), "=f"(scores[92]), "=f"(scores[93]), "=f"(scores[94]), "=f"(scores[95]), "=f"(scores[96]), "=f"(scores[97]), "=f"(scores[98]), "=f"(scores[99]), "=f"(scores[100]), "=f"(scores[101]), "=f"(scores[102]), "=f"(scores[103]), "=f"(scores[104]), "=f"(scores[105]), "=f"(scores[106]), "=f"(scores[107]), "=f"(scores[108]), "=f"(scores[109]), "=f"(scores[110]), "=f"(scores[111]), "=f"(scores[112]), "=f"(scores[113]), "=f"(scores[114]), "=f"(scores[115]), "=f"(scores[116]), "=f"(scores[117]), "=f"(scores[118]), "=f"(scores[119]), "=f"(scores[120]), "=f"(scores[121]), "=f"(scores[122]), "=f"(scores[123]), "=f"(scores[124]), "=f"(scores[125]), "=f"(scores[126]), "=f"(scores[127])
                        : "r"(taddr + tmem_row_base_v + score_base)
                        : "memory");
                    asm volatile("tcgen05.wait::ld.sync.aligned;");
                    #pragma unroll
                    for (int kk = 0; kk < 128; kk += 16) {
                        float csq_vals[4];
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&csq_vals[0])), "=r"(*reinterpret_cast<uint32_t*>(&csq_vals[1])), "=r"(*reinterpret_cast<uint32_t*>(&csq_vals[2])), "=r"(*reinterpret_cast<uint32_t*>(&csq_vals[3]))
                            : "r"(csq_smem_addr + (score_base + kk) * 4));
                        float d0 = scores[kk] - csq_vals[0];
                        float d1 = scores[kk + 1] - csq_vals[1];
                        float best_01 = d0;
                        int idx_01 = off_k + score_base + kk;
                        if (d1 > best_01) {
                            best_01 = d1;
                            idx_01 = off_k + score_base + kk + 1;
                        }
                        float d2 = scores[kk + 2] - csq_vals[2];
                        float d3 = scores[kk + 3] - csq_vals[3];
                        float best_23 = d2;
                        int idx_23 = off_k + score_base + kk + 2;
                        if (d3 > best_23) {
                            best_23 = d3;
                            idx_23 = off_k + score_base + kk + 3;
                        }
                        float best_group = best_01;
                        int idx_group = idx_01;
                        if (best_23 > best_group) {
                            best_group = best_23;
                            idx_group = idx_23;
                        }
                        float csq_vals_hi[4];
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&csq_vals_hi[0])), "=r"(*reinterpret_cast<uint32_t*>(&csq_vals_hi[1])), "=r"(*reinterpret_cast<uint32_t*>(&csq_vals_hi[2])), "=r"(*reinterpret_cast<uint32_t*>(&csq_vals_hi[3]))
                            : "r"(csq_smem_addr + (score_base + kk + 4) * 4));
                        float d4 = scores[kk + 4] - csq_vals_hi[0];
                        float d5 = scores[kk + 5] - csq_vals_hi[1];
                        float best_45 = d4;
                        int idx_45 = off_k + score_base + kk + 4;
                        if (d5 > best_45) {
                            best_45 = d5;
                            idx_45 = off_k + score_base + kk + 5;
                        }
                        float d6 = scores[kk + 6] - csq_vals_hi[2];
                        float d7 = scores[kk + 7] - csq_vals_hi[3];
                        float best_67 = d6;
                        int idx_67 = off_k + score_base + kk + 6;
                        if (d7 > best_67) {
                            best_67 = d7;
                            idx_67 = off_k + score_base + kk + 7;
                        }
                        float best_hi = best_45;
                        int idx_hi = idx_45;
                        if (best_67 > best_hi) {
                            best_hi = best_67;
                            idx_hi = idx_67;
                        }
                        if (best_hi > best_group) {
                            best_group = best_hi;
                            idx_group = idx_hi;
                        }
                        float csq_vals_next[4];
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&csq_vals_next[0])), "=r"(*reinterpret_cast<uint32_t*>(&csq_vals_next[1])), "=r"(*reinterpret_cast<uint32_t*>(&csq_vals_next[2])), "=r"(*reinterpret_cast<uint32_t*>(&csq_vals_next[3]))
                            : "r"(csq_smem_addr + (score_base + kk + 8) * 4));
                        float d8 = scores[kk + 8] - csq_vals_next[0];
                        float d9 = scores[kk + 9] - csq_vals_next[1];
                        float best_89 = d8;
                        int idx_89 = off_k + score_base + kk + 8;
                        if (d9 > best_89) {
                            best_89 = d9;
                            idx_89 = off_k + score_base + kk + 9;
                        }
                        float d10 = scores[kk + 10] - csq_vals_next[2];
                        float d11 = scores[kk + 11] - csq_vals_next[3];
                        float best_1011 = d10;
                        int idx_1011 = off_k + score_base + kk + 10;
                        if (d11 > best_1011) {
                            best_1011 = d11;
                            idx_1011 = off_k + score_base + kk + 11;
                        }
                        float best_next = best_89;
                        int idx_next = idx_89;
                        if (best_1011 > best_next) {
                            best_next = best_1011;
                            idx_next = idx_1011;
                        }
                        float csq_vals_tail[4];
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&csq_vals_tail[0])), "=r"(*reinterpret_cast<uint32_t*>(&csq_vals_tail[1])), "=r"(*reinterpret_cast<uint32_t*>(&csq_vals_tail[2])), "=r"(*reinterpret_cast<uint32_t*>(&csq_vals_tail[3]))
                            : "r"(csq_smem_addr + (score_base + kk + 12) * 4));
                        float d12 = scores[kk + 12] - csq_vals_tail[0];
                        float d13 = scores[kk + 13] - csq_vals_tail[1];
                        float best_1213 = d12;
                        int idx_1213 = off_k + score_base + kk + 12;
                        if (d13 > best_1213) {
                            best_1213 = d13;
                            idx_1213 = off_k + score_base + kk + 13;
                        }
                        float d14 = scores[kk + 14] - csq_vals_tail[2];
                        float d15 = scores[kk + 15] - csq_vals_tail[3];
                        float best_1415 = d14;
                        int idx_1415 = off_k + score_base + kk + 14;
                        if (d15 > best_1415) {
                            best_1415 = d15;
                            idx_1415 = off_k + score_base + kk + 15;
                        }
                        float best_tail = best_1213;
                        int idx_tail = idx_1213;
                        if (best_1415 > best_tail) {
                            best_tail = best_1415;
                            idx_tail = idx_1415;
                        }
                        if (best_tail > best_next) {
                            best_next = best_tail;
                            idx_next = idx_tail;
                        }
                        if (best_next > best_group) {
                            best_group = best_next;
                            idx_group = idx_next;
                        }
                        if (best_group > best_score1) {
                            best_score1 = best_group;
                            best_idx1 = idx_group;
                        }
                    }
                    score_base = 128;
                    asm volatile(
                        "tcgen05.ld.sync.aligned.32x32b.x128.b32"
                        " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63, %64, %65, %66, %67, %68, %69, %70, %71, %72, %73, %74, %75, %76, %77, %78, %79, %80, %81, %82, %83, %84, %85, %86, %87, %88, %89, %90, %91, %92, %93, %94, %95, %96, %97, %98, %99, %100, %101, %102, %103, %104, %105, %106, %107, %108, %109, %110, %111, %112, %113, %114, %115, %116, %117, %118, %119, %120, %121, %122, %123, %124, %125, %126, %127}, [%128];"
                        : "=f"(scores[0]), "=f"(scores[1]), "=f"(scores[2]), "=f"(scores[3]), "=f"(scores[4]), "=f"(scores[5]), "=f"(scores[6]), "=f"(scores[7]), "=f"(scores[8]), "=f"(scores[9]), "=f"(scores[10]), "=f"(scores[11]), "=f"(scores[12]), "=f"(scores[13]), "=f"(scores[14]), "=f"(scores[15]), "=f"(scores[16]), "=f"(scores[17]), "=f"(scores[18]), "=f"(scores[19]), "=f"(scores[20]), "=f"(scores[21]), "=f"(scores[22]), "=f"(scores[23]), "=f"(scores[24]), "=f"(scores[25]), "=f"(scores[26]), "=f"(scores[27]), "=f"(scores[28]), "=f"(scores[29]), "=f"(scores[30]), "=f"(scores[31]), "=f"(scores[32]), "=f"(scores[33]), "=f"(scores[34]), "=f"(scores[35]), "=f"(scores[36]), "=f"(scores[37]), "=f"(scores[38]), "=f"(scores[39]), "=f"(scores[40]), "=f"(scores[41]), "=f"(scores[42]), "=f"(scores[43]), "=f"(scores[44]), "=f"(scores[45]), "=f"(scores[46]), "=f"(scores[47]), "=f"(scores[48]), "=f"(scores[49]), "=f"(scores[50]), "=f"(scores[51]), "=f"(scores[52]), "=f"(scores[53]), "=f"(scores[54]), "=f"(scores[55]), "=f"(scores[56]), "=f"(scores[57]), "=f"(scores[58]), "=f"(scores[59]), "=f"(scores[60]), "=f"(scores[61]), "=f"(scores[62]), "=f"(scores[63]), "=f"(scores[64]), "=f"(scores[65]), "=f"(scores[66]), "=f"(scores[67]), "=f"(scores[68]), "=f"(scores[69]), "=f"(scores[70]), "=f"(scores[71]), "=f"(scores[72]), "=f"(scores[73]), "=f"(scores[74]), "=f"(scores[75]), "=f"(scores[76]), "=f"(scores[77]), "=f"(scores[78]), "=f"(scores[79]), "=f"(scores[80]), "=f"(scores[81]), "=f"(scores[82]), "=f"(scores[83]), "=f"(scores[84]), "=f"(scores[85]), "=f"(scores[86]), "=f"(scores[87]), "=f"(scores[88]), "=f"(scores[89]), "=f"(scores[90]), "=f"(scores[91]), "=f"(scores[92]), "=f"(scores[93]), "=f"(scores[94]), "=f"(scores[95]), "=f"(scores[96]), "=f"(scores[97]), "=f"(scores[98]), "=f"(scores[99]), "=f"(scores[100]), "=f"(scores[101]), "=f"(scores[102]), "=f"(scores[103]), "=f"(scores[104]), "=f"(scores[105]), "=f"(scores[106]), "=f"(scores[107]), "=f"(scores[108]), "=f"(scores[109]), "=f"(scores[110]), "=f"(scores[111]), "=f"(scores[112]), "=f"(scores[113]), "=f"(scores[114]), "=f"(scores[115]), "=f"(scores[116]), "=f"(scores[117]), "=f"(scores[118]), "=f"(scores[119]), "=f"(scores[120]), "=f"(scores[121]), "=f"(scores[122]), "=f"(scores[123]), "=f"(scores[124]), "=f"(scores[125]), "=f"(scores[126]), "=f"(scores[127])
                        : "r"(taddr + tmem_row_base_v + score_base)
                        : "memory");
                    asm volatile("tcgen05.wait::ld.sync.aligned;");
                    asm volatile("barrier.sync 9, 128;");
                    if (elect_sync()) {
                        mbarrier_arrive(score1_empty_addr);
                    }
                    #pragma unroll
                    for (int kk = 0; kk < 128; kk += 16) {
                        float csq_vals[4];
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&csq_vals[0])), "=r"(*reinterpret_cast<uint32_t*>(&csq_vals[1])), "=r"(*reinterpret_cast<uint32_t*>(&csq_vals[2])), "=r"(*reinterpret_cast<uint32_t*>(&csq_vals[3]))
                            : "r"(csq_smem_addr + (score_base + kk) * 4));
                        float d0 = scores[kk] - csq_vals[0];
                        float d1 = scores[kk + 1] - csq_vals[1];
                        float best_01 = d0;
                        int idx_01 = off_k + score_base + kk;
                        if (d1 > best_01) {
                            best_01 = d1;
                            idx_01 = off_k + score_base + kk + 1;
                        }
                        float d2 = scores[kk + 2] - csq_vals[2];
                        float d3 = scores[kk + 3] - csq_vals[3];
                        float best_23 = d2;
                        int idx_23 = off_k + score_base + kk + 2;
                        if (d3 > best_23) {
                            best_23 = d3;
                            idx_23 = off_k + score_base + kk + 3;
                        }
                        float best_group = best_01;
                        int idx_group = idx_01;
                        if (best_23 > best_group) {
                            best_group = best_23;
                            idx_group = idx_23;
                        }
                        float csq_vals_hi[4];
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&csq_vals_hi[0])), "=r"(*reinterpret_cast<uint32_t*>(&csq_vals_hi[1])), "=r"(*reinterpret_cast<uint32_t*>(&csq_vals_hi[2])), "=r"(*reinterpret_cast<uint32_t*>(&csq_vals_hi[3]))
                            : "r"(csq_smem_addr + (score_base + kk + 4) * 4));
                        float d4 = scores[kk + 4] - csq_vals_hi[0];
                        float d5 = scores[kk + 5] - csq_vals_hi[1];
                        float best_45 = d4;
                        int idx_45 = off_k + score_base + kk + 4;
                        if (d5 > best_45) {
                            best_45 = d5;
                            idx_45 = off_k + score_base + kk + 5;
                        }
                        float d6 = scores[kk + 6] - csq_vals_hi[2];
                        float d7 = scores[kk + 7] - csq_vals_hi[3];
                        float best_67 = d6;
                        int idx_67 = off_k + score_base + kk + 6;
                        if (d7 > best_67) {
                            best_67 = d7;
                            idx_67 = off_k + score_base + kk + 7;
                        }
                        float best_hi = best_45;
                        int idx_hi = idx_45;
                        if (best_67 > best_hi) {
                            best_hi = best_67;
                            idx_hi = idx_67;
                        }
                        if (best_hi > best_group) {
                            best_group = best_hi;
                            idx_group = idx_hi;
                        }
                        float csq_vals_next[4];
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&csq_vals_next[0])), "=r"(*reinterpret_cast<uint32_t*>(&csq_vals_next[1])), "=r"(*reinterpret_cast<uint32_t*>(&csq_vals_next[2])), "=r"(*reinterpret_cast<uint32_t*>(&csq_vals_next[3]))
                            : "r"(csq_smem_addr + (score_base + kk + 8) * 4));
                        float d8 = scores[kk + 8] - csq_vals_next[0];
                        float d9 = scores[kk + 9] - csq_vals_next[1];
                        float best_89 = d8;
                        int idx_89 = off_k + score_base + kk + 8;
                        if (d9 > best_89) {
                            best_89 = d9;
                            idx_89 = off_k + score_base + kk + 9;
                        }
                        float d10 = scores[kk + 10] - csq_vals_next[2];
                        float d11 = scores[kk + 11] - csq_vals_next[3];
                        float best_1011 = d10;
                        int idx_1011 = off_k + score_base + kk + 10;
                        if (d11 > best_1011) {
                            best_1011 = d11;
                            idx_1011 = off_k + score_base + kk + 11;
                        }
                        float best_next = best_89;
                        int idx_next = idx_89;
                        if (best_1011 > best_next) {
                            best_next = best_1011;
                            idx_next = idx_1011;
                        }
                        float csq_vals_tail[4];
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&csq_vals_tail[0])), "=r"(*reinterpret_cast<uint32_t*>(&csq_vals_tail[1])), "=r"(*reinterpret_cast<uint32_t*>(&csq_vals_tail[2])), "=r"(*reinterpret_cast<uint32_t*>(&csq_vals_tail[3]))
                            : "r"(csq_smem_addr + (score_base + kk + 12) * 4));
                        float d12 = scores[kk + 12] - csq_vals_tail[0];
                        float d13 = scores[kk + 13] - csq_vals_tail[1];
                        float best_1213 = d12;
                        int idx_1213 = off_k + score_base + kk + 12;
                        if (d13 > best_1213) {
                            best_1213 = d13;
                            idx_1213 = off_k + score_base + kk + 13;
                        }
                        float d14 = scores[kk + 14] - csq_vals_tail[2];
                        float d15 = scores[kk + 15] - csq_vals_tail[3];
                        float best_1415 = d14;
                        int idx_1415 = off_k + score_base + kk + 14;
                        if (d15 > best_1415) {
                            best_1415 = d15;
                            idx_1415 = off_k + score_base + kk + 15;
                        }
                        float best_tail = best_1213;
                        int idx_tail = idx_1213;
                        if (best_1415 > best_tail) {
                            best_tail = best_1415;
                            idx_tail = idx_1415;
                        }
                        if (best_tail > best_next) {
                            best_next = best_tail;
                            idx_next = idx_tail;
                        }
                        if (best_next > best_group) {
                            best_group = best_next;
                            idx_group = idx_next;
                        }
                        if (best_group > best_score1) {
                            best_score1 = best_group;
                            best_idx1 = idx_group;
                        }
                    }
                }
                if (global_n0 < N) {
                    *((int*)(out + out_offset0)) = best_idx0;
                }
                if (global_n1 < N) {
                    *((int*)(out + out_offset1)) = best_idx1;
                }
            }
        }
    }

    // Cleanup
    __syncthreads(); // barrier before TMEM dealloc

    if (warp == 1) {
        asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;" :: "r"(tmem_addr_storage[0]), "r"(512));
        asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;");
    }
}

} // extern "C"

#undef NUM_MAIN_STAGES
#undef SMEM_SMEM_CSQ_OFF
#undef SMEM_SMEM_CSQ_STAGE_BYTES
#undef SMEM_SMEM_CSQ_STRIDE
#undef SMEM_SMEM_C_OFF
#undef SMEM_SMEM_C_STAGE_BYTES
#undef SMEM_SMEM_C_STRIDE
#undef SMEM_SMEM_X0_OFF
#undef SMEM_SMEM_X0_STAGE_BYTES
#undef SMEM_SMEM_X0_STRIDE
#undef SMEM_SMEM_X1_OFF
#undef SMEM_SMEM_X1_STAGE_BYTES
#undef SMEM_SMEM_X1_STRIDE
#undef SMEM_TOTAL
#undef TMEM_NCOLS
#undef TMEM_SCORE_TMEM_OFFSET
#undef c_empty_addr
#undef c_full_addr
#undef score0_empty_addr
#undef score0_full_addr
#undef score1_empty_addr
#undef score1_full_addr
#undef smem_c_addr
#undef smem_csq_addr
#undef smem_x0_addr
#undef smem_x1_addr
#undef x0_empty_addr
#undef x0_full_addr
#undef x1_empty_addr
#undef x1_full_addr

#define TMEM_NCOLS 256
#define TMEM_SCORE_TMEM_OFFSET 0
#define NUM_MAIN_STAGES 1
#define SMEM_SMEM_X_OFF 1024
#define SMEM_SMEM_X_STAGE_BYTES 16384
#define SMEM_SMEM_X_STRIDE 16384
#define SMEM_SMEM_C_OFF 17408
#define SMEM_SMEM_C_STAGE_BYTES 32768
#define SMEM_SMEM_C_STRIDE 32768
#define SMEM_SMEM_CSQ_OFF 50176
#define SMEM_SMEM_CSQ_STAGE_BYTES 1024
#define SMEM_SMEM_CSQ_STRIDE 1024
#define SMEM_TOTAL 51200

extern "C" {

__global__ __launch_bounds__(192) void
kernel_flash_kmeans_assign_highd_splitd_6fcf_v1(float* __restrict__ x_sq, float* __restrict__ c_sq, int32_t* __restrict__ out, const void* x_tmap, const void* c_tmap, int B, int N, int D, int K, int num_n_tiles, int K_tiles)
{
    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;

    extern __shared__ __align__(1024) char smem_raw[];
    int smem;
    smem = (int)(unsigned long long)__cvta_generic_to_shared(smem_raw);
    const int smem_smem_x = smem + 1024;
    const int smem_smem_c = smem + 17408;
    const int smem_smem_csq = smem + 50176;

    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    // Mbarrier init (6 groups, 6 barriers)
    // Mbarriers at smem_raw[0..48)

    if (warp == 0) {
        uint32_t leader = elect_sync();
        // x_full: 1 barriers, init_count=1
        mbarrier_init_pred(smem + 0, 1, leader);
        // x_empty: 1 barriers, init_count=1
        mbarrier_init_pred(smem + 8, 1, leader);
        // c_full: 1 barriers, init_count=1
        mbarrier_init_pred(smem + 16, 1, leader);
        // c_empty: 1 barriers, init_count=1
        mbarrier_init_pred(smem + 24, 1, leader);
        // score_full: 1 barriers, init_count=1
        mbarrier_init_pred(smem + 32, 1, leader);
        // score_empty: 1 barriers, init_count=4
        mbarrier_init_pred(smem + 40, 4, leader);
        asm volatile("fence.mbarrier_init.release.cluster;");
    }

    __syncthreads();

    // TMEM alloc (256 columns, 256 used)
    volatile int* tmem_addr_storage = (volatile int*)(smem_raw + 48);
    if (warp == 1) {
        int _tmem_hold = smem + 48;
        asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;" :: "r"(_tmem_hold), "r"(256) : "memory");
    }

    __syncthreads();
    asm volatile("tcgen05.fence::after_thread_sync;");

    const int warp_id = warp;
    const int lane_id = lane;
    __nv_bfloat16* smem_x = (__nv_bfloat16*)(smem_raw + 1024);
    #define smem_x_addr (smem + 1024)
    __nv_bfloat16* smem_c = (__nv_bfloat16*)(smem_raw + 17408);
    #define smem_c_addr (smem + 17408)
    float* smem_csq = (float*)(smem_raw + 50176);
    #define smem_csq_addr (smem + 50176)
    const int mbar_base = smem;
    #define x_full_addr (mbar_base + 0)
    #define x_empty_addr (mbar_base + 8)
    #define c_full_addr (mbar_base + 16)
    #define c_empty_addr (mbar_base + 24)
    #define score_full_addr (mbar_base + 32)
    #define score_empty_addr (mbar_base + 40)
    const int taddr = tmem_addr_storage[0];

    // ---- Role: load ----
    if (warp == 0) {
        { // load_main
            int num_tiles = B * num_n_tiles;
            int feature_tiles = D / 64;
            uint32_t _phase_x_empty_0 = 1;
            uint32_t _phase_c_empty_0 = 1;
            if (warp_id == 0) {
                if (elect_sync()) {
                    #pragma unroll 1
                    for (unsigned int tile_idx = bid; tile_idx < num_tiles; tile_idx += num_bids) {
                        int batch = tile_idx / num_n_tiles;
                        int n_tile = tile_idx % num_n_tiles;
                        int off_n = n_tile * 128;
                        int x_row = batch * N + off_n;
                        #pragma unroll 1
                        for (int iter_k = 0; iter_k < K_tiles; iter_k++) {
                            int off_k = iter_k * 256;
                            int c_row = batch * K + off_k;
                            #pragma unroll 1
                            for (int feat_tile = 0; feat_tile < feature_tiles; feat_tile++) {
                                mbarrier_wait(x_empty_addr, _phase_x_empty_0);
                                _phase_x_empty_0 ^= 1;
                                tma_3d_gmem2smem(smem_x_addr, x_tmap, 0, x_row, feat_tile, x_full_addr);
                                mbarrier_arrive_expect_tx(x_full_addr, 16384);
                                mbarrier_wait(c_empty_addr, _phase_c_empty_0);
                                _phase_c_empty_0 ^= 1;
                                tma_3d_gmem2smem(smem_c_addr, c_tmap, 0, c_row, feat_tile, c_full_addr);
                                mbarrier_arrive_expect_tx(c_full_addr, 32768);
                            }
                        }
                    }
                }
            }
        }
    // ---- Role: mma ----
    } else if (warp == 1) {
        const int tmem_score_tmem = taddr + TMEM_SCORE_TMEM_OFFSET;
        { // mma_main
            int num_tiles = B * num_n_tiles;
            int feature_tiles = D / 64;
            uint32_t _phase_score_empty_0 = 1;
            uint32_t _phase_x_full_0 = 0;
            uint32_t _phase_c_full_0 = 0;
            #pragma unroll 1
            for (unsigned int tile_idx = bid; tile_idx < num_tiles; tile_idx += num_bids) {
                #pragma unroll 1
                for (int iter_k = 0; iter_k < K_tiles; iter_k++) {
                    mbarrier_wait(score_empty_addr, _phase_score_empty_0);
                    _phase_score_empty_0 ^= 1;
                    #pragma unroll 1
                    for (int feat_tile = 0; feat_tile < feature_tiles; feat_tile++) {
                        mbarrier_wait(x_full_addr, _phase_x_full_0);
                        _phase_x_full_0 ^= 1;
                        mbarrier_wait(c_full_addr, _phase_c_full_0);
                        _phase_c_full_0 ^= 1;
                        asm volatile("tcgen05.fence::after_thread_sync;");
                        int init_flag = ((feat_tile == 0) ? 1 : 0);
                        int _mma_ss_a_addr_0 = smem_x_addr + 0 * 16384;
                        int _mma_ss_a_lo_0 = make_warp_uniform((_mma_ss_a_addr_0 >> 4) & 0x3FFF);
                        int _mma_ss_b_addr_0 = smem_c_addr + 0 * 32768;
                        int _mma_ss_b_lo_0 = make_warp_uniform((_mma_ss_b_addr_0 >> 4) & 0x3FFF);
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
                    "mov.b32 id, 138413200;\n\t"
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
                    :: "r"(_mma_ss_a_lo_0), "r"(_mma_ss_b_lo_0), "r"(tmem_score_tmem), "r"(((init_flag) ? 0 : 1)));
                        elect_commit(x_empty_addr);
                        elect_commit(c_empty_addr);
                    }
                    elect_commit(score_full_addr);
                }
            }
        }
    // ---- Role: compute ----
    } else if (warp >= 2 && warp <= 5) {
        const int warp_id_in_wg = warp % 4;
        const int my_row = warp_id_in_wg * 32 + lane;
        const int tmem_row_base_v = (warp_id_in_wg * 32) << 16;
        { // compute_main
            int num_tiles = B * num_n_tiles;
            uint32_t _phase_score_full_0 = 0;
            #pragma unroll 1
            for (unsigned int tile_idx = bid; tile_idx < num_tiles; tile_idx += num_bids) {
                int batch = tile_idx / num_n_tiles;
                int n_tile = tile_idx % num_n_tiles;
                int off_n = n_tile * 128;
                int global_n = off_n + my_row;
                int out_offset = batch * N + global_n;
                int csq_smem_addr = smem_csq_addr + 0;
                float best_score = -3.4e+38f;
                int best_idx = 0;
                #pragma unroll 1
                for (int iter_k = 0; iter_k < K_tiles; iter_k++) {
                    int off_k = iter_k * 256;
                    if (my_row < 64.0f) {
                        int csq_base = my_row * 4;
                        float csq_pack[4];
                        {
                            float4 _v4 = *reinterpret_cast<const float4*>(c_sq + batch * K + off_k + csq_base);
                            csq_pack[0 + 0] = _v4.x;
                            csq_pack[0 + 1] = _v4.y;
                            csq_pack[0 + 2] = _v4.z;
                            csq_pack[0 + 3] = _v4.w;
                        }
                        float csq_half[4];
                        csq_half[0] = 0.5f * csq_pack[0];
                        csq_half[1] = 0.5f * csq_pack[1];
                        csq_half[2] = 0.5f * csq_pack[2];
                        csq_half[3] = 0.5f * csq_pack[3];
                        asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(csq_smem_addr + csq_base * 4), "f"(csq_half[0]), "f"(csq_half[1]), "f"(csq_half[2]), "f"(csq_half[3]) : "memory");
                    }
                    asm volatile("barrier.sync 8, 128;");
                    mbarrier_wait(score_full_addr, _phase_score_full_0);
                    _phase_score_full_0 ^= 1;
                    int score_base = 0;
                    float scores[128];
                    asm volatile(
                        "tcgen05.ld.sync.aligned.32x32b.x128.b32"
                        " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63, %64, %65, %66, %67, %68, %69, %70, %71, %72, %73, %74, %75, %76, %77, %78, %79, %80, %81, %82, %83, %84, %85, %86, %87, %88, %89, %90, %91, %92, %93, %94, %95, %96, %97, %98, %99, %100, %101, %102, %103, %104, %105, %106, %107, %108, %109, %110, %111, %112, %113, %114, %115, %116, %117, %118, %119, %120, %121, %122, %123, %124, %125, %126, %127}, [%128];"
                        : "=f"(scores[0]), "=f"(scores[1]), "=f"(scores[2]), "=f"(scores[3]), "=f"(scores[4]), "=f"(scores[5]), "=f"(scores[6]), "=f"(scores[7]), "=f"(scores[8]), "=f"(scores[9]), "=f"(scores[10]), "=f"(scores[11]), "=f"(scores[12]), "=f"(scores[13]), "=f"(scores[14]), "=f"(scores[15]), "=f"(scores[16]), "=f"(scores[17]), "=f"(scores[18]), "=f"(scores[19]), "=f"(scores[20]), "=f"(scores[21]), "=f"(scores[22]), "=f"(scores[23]), "=f"(scores[24]), "=f"(scores[25]), "=f"(scores[26]), "=f"(scores[27]), "=f"(scores[28]), "=f"(scores[29]), "=f"(scores[30]), "=f"(scores[31]), "=f"(scores[32]), "=f"(scores[33]), "=f"(scores[34]), "=f"(scores[35]), "=f"(scores[36]), "=f"(scores[37]), "=f"(scores[38]), "=f"(scores[39]), "=f"(scores[40]), "=f"(scores[41]), "=f"(scores[42]), "=f"(scores[43]), "=f"(scores[44]), "=f"(scores[45]), "=f"(scores[46]), "=f"(scores[47]), "=f"(scores[48]), "=f"(scores[49]), "=f"(scores[50]), "=f"(scores[51]), "=f"(scores[52]), "=f"(scores[53]), "=f"(scores[54]), "=f"(scores[55]), "=f"(scores[56]), "=f"(scores[57]), "=f"(scores[58]), "=f"(scores[59]), "=f"(scores[60]), "=f"(scores[61]), "=f"(scores[62]), "=f"(scores[63]), "=f"(scores[64]), "=f"(scores[65]), "=f"(scores[66]), "=f"(scores[67]), "=f"(scores[68]), "=f"(scores[69]), "=f"(scores[70]), "=f"(scores[71]), "=f"(scores[72]), "=f"(scores[73]), "=f"(scores[74]), "=f"(scores[75]), "=f"(scores[76]), "=f"(scores[77]), "=f"(scores[78]), "=f"(scores[79]), "=f"(scores[80]), "=f"(scores[81]), "=f"(scores[82]), "=f"(scores[83]), "=f"(scores[84]), "=f"(scores[85]), "=f"(scores[86]), "=f"(scores[87]), "=f"(scores[88]), "=f"(scores[89]), "=f"(scores[90]), "=f"(scores[91]), "=f"(scores[92]), "=f"(scores[93]), "=f"(scores[94]), "=f"(scores[95]), "=f"(scores[96]), "=f"(scores[97]), "=f"(scores[98]), "=f"(scores[99]), "=f"(scores[100]), "=f"(scores[101]), "=f"(scores[102]), "=f"(scores[103]), "=f"(scores[104]), "=f"(scores[105]), "=f"(scores[106]), "=f"(scores[107]), "=f"(scores[108]), "=f"(scores[109]), "=f"(scores[110]), "=f"(scores[111]), "=f"(scores[112]), "=f"(scores[113]), "=f"(scores[114]), "=f"(scores[115]), "=f"(scores[116]), "=f"(scores[117]), "=f"(scores[118]), "=f"(scores[119]), "=f"(scores[120]), "=f"(scores[121]), "=f"(scores[122]), "=f"(scores[123]), "=f"(scores[124]), "=f"(scores[125]), "=f"(scores[126]), "=f"(scores[127])
                        : "r"(taddr + tmem_row_base_v + score_base)
                        : "memory");
                    asm volatile("tcgen05.wait::ld.sync.aligned;");
                    #pragma unroll
                    for (int kk = 0; kk < 128; kk += 4) {
                        float csq_vals[4];
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&csq_vals[0])), "=r"(*reinterpret_cast<uint32_t*>(&csq_vals[1])), "=r"(*reinterpret_cast<uint32_t*>(&csq_vals[2])), "=r"(*reinterpret_cast<uint32_t*>(&csq_vals[3]))
                            : "r"(csq_smem_addr + (score_base + kk) * 4));
                        float d0 = scores[kk] - csq_vals[0];
                        float d1 = scores[kk + 1] - csq_vals[1];
                        float best_01 = d0;
                        int idx_01 = off_k + score_base + kk;
                        if (d1 > best_01) {
                            best_01 = d1;
                            idx_01 = off_k + score_base + kk + 1;
                        }
                        float d2 = scores[kk + 2] - csq_vals[2];
                        float d3 = scores[kk + 3] - csq_vals[3];
                        float best_23 = d2;
                        int idx_23 = off_k + score_base + kk + 2;
                        if (d3 > best_23) {
                            best_23 = d3;
                            idx_23 = off_k + score_base + kk + 3;
                        }
                        float best_group = best_01;
                        int idx_group = idx_01;
                        if (best_23 > best_group) {
                            best_group = best_23;
                            idx_group = idx_23;
                        }
                        if (best_group > best_score) {
                            best_score = best_group;
                            best_idx = idx_group;
                        }
                    }
                    score_base = 128;
                    asm volatile(
                        "tcgen05.ld.sync.aligned.32x32b.x128.b32"
                        " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63, %64, %65, %66, %67, %68, %69, %70, %71, %72, %73, %74, %75, %76, %77, %78, %79, %80, %81, %82, %83, %84, %85, %86, %87, %88, %89, %90, %91, %92, %93, %94, %95, %96, %97, %98, %99, %100, %101, %102, %103, %104, %105, %106, %107, %108, %109, %110, %111, %112, %113, %114, %115, %116, %117, %118, %119, %120, %121, %122, %123, %124, %125, %126, %127}, [%128];"
                        : "=f"(scores[0]), "=f"(scores[1]), "=f"(scores[2]), "=f"(scores[3]), "=f"(scores[4]), "=f"(scores[5]), "=f"(scores[6]), "=f"(scores[7]), "=f"(scores[8]), "=f"(scores[9]), "=f"(scores[10]), "=f"(scores[11]), "=f"(scores[12]), "=f"(scores[13]), "=f"(scores[14]), "=f"(scores[15]), "=f"(scores[16]), "=f"(scores[17]), "=f"(scores[18]), "=f"(scores[19]), "=f"(scores[20]), "=f"(scores[21]), "=f"(scores[22]), "=f"(scores[23]), "=f"(scores[24]), "=f"(scores[25]), "=f"(scores[26]), "=f"(scores[27]), "=f"(scores[28]), "=f"(scores[29]), "=f"(scores[30]), "=f"(scores[31]), "=f"(scores[32]), "=f"(scores[33]), "=f"(scores[34]), "=f"(scores[35]), "=f"(scores[36]), "=f"(scores[37]), "=f"(scores[38]), "=f"(scores[39]), "=f"(scores[40]), "=f"(scores[41]), "=f"(scores[42]), "=f"(scores[43]), "=f"(scores[44]), "=f"(scores[45]), "=f"(scores[46]), "=f"(scores[47]), "=f"(scores[48]), "=f"(scores[49]), "=f"(scores[50]), "=f"(scores[51]), "=f"(scores[52]), "=f"(scores[53]), "=f"(scores[54]), "=f"(scores[55]), "=f"(scores[56]), "=f"(scores[57]), "=f"(scores[58]), "=f"(scores[59]), "=f"(scores[60]), "=f"(scores[61]), "=f"(scores[62]), "=f"(scores[63]), "=f"(scores[64]), "=f"(scores[65]), "=f"(scores[66]), "=f"(scores[67]), "=f"(scores[68]), "=f"(scores[69]), "=f"(scores[70]), "=f"(scores[71]), "=f"(scores[72]), "=f"(scores[73]), "=f"(scores[74]), "=f"(scores[75]), "=f"(scores[76]), "=f"(scores[77]), "=f"(scores[78]), "=f"(scores[79]), "=f"(scores[80]), "=f"(scores[81]), "=f"(scores[82]), "=f"(scores[83]), "=f"(scores[84]), "=f"(scores[85]), "=f"(scores[86]), "=f"(scores[87]), "=f"(scores[88]), "=f"(scores[89]), "=f"(scores[90]), "=f"(scores[91]), "=f"(scores[92]), "=f"(scores[93]), "=f"(scores[94]), "=f"(scores[95]), "=f"(scores[96]), "=f"(scores[97]), "=f"(scores[98]), "=f"(scores[99]), "=f"(scores[100]), "=f"(scores[101]), "=f"(scores[102]), "=f"(scores[103]), "=f"(scores[104]), "=f"(scores[105]), "=f"(scores[106]), "=f"(scores[107]), "=f"(scores[108]), "=f"(scores[109]), "=f"(scores[110]), "=f"(scores[111]), "=f"(scores[112]), "=f"(scores[113]), "=f"(scores[114]), "=f"(scores[115]), "=f"(scores[116]), "=f"(scores[117]), "=f"(scores[118]), "=f"(scores[119]), "=f"(scores[120]), "=f"(scores[121]), "=f"(scores[122]), "=f"(scores[123]), "=f"(scores[124]), "=f"(scores[125]), "=f"(scores[126]), "=f"(scores[127])
                        : "r"(taddr + tmem_row_base_v + score_base)
                        : "memory");
                    asm volatile("tcgen05.wait::ld.sync.aligned;");
                    asm volatile("barrier.sync 9, 128;");
                    if (elect_sync()) {
                        mbarrier_arrive(score_empty_addr);
                    }
                    #pragma unroll
                    for (int kk = 0; kk < 128; kk += 4) {
                        float csq_vals[4];
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&csq_vals[0])), "=r"(*reinterpret_cast<uint32_t*>(&csq_vals[1])), "=r"(*reinterpret_cast<uint32_t*>(&csq_vals[2])), "=r"(*reinterpret_cast<uint32_t*>(&csq_vals[3]))
                            : "r"(csq_smem_addr + (score_base + kk) * 4));
                        float d0 = scores[kk] - csq_vals[0];
                        float d1 = scores[kk + 1] - csq_vals[1];
                        float best_01 = d0;
                        int idx_01 = off_k + score_base + kk;
                        if (d1 > best_01) {
                            best_01 = d1;
                            idx_01 = off_k + score_base + kk + 1;
                        }
                        float d2 = scores[kk + 2] - csq_vals[2];
                        float d3 = scores[kk + 3] - csq_vals[3];
                        float best_23 = d2;
                        int idx_23 = off_k + score_base + kk + 2;
                        if (d3 > best_23) {
                            best_23 = d3;
                            idx_23 = off_k + score_base + kk + 3;
                        }
                        float best_group = best_01;
                        int idx_group = idx_01;
                        if (best_23 > best_group) {
                            best_group = best_23;
                            idx_group = idx_23;
                        }
                        if (best_group > best_score) {
                            best_score = best_group;
                            best_idx = idx_group;
                        }
                    }
                }
                if (global_n < N) {
                    *((int*)(out + out_offset)) = best_idx;
                }
            }
        }
    }

    // Cleanup
    __syncthreads(); // barrier before TMEM dealloc

    if (warp == 1) {
        asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;" :: "r"(tmem_addr_storage[0]), "r"(256));
        asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;");
    }
}

} // extern "C"

#undef NUM_MAIN_STAGES
#undef SMEM_SMEM_CSQ_OFF
#undef SMEM_SMEM_CSQ_STAGE_BYTES
#undef SMEM_SMEM_CSQ_STRIDE
#undef SMEM_SMEM_C_OFF
#undef SMEM_SMEM_C_STAGE_BYTES
#undef SMEM_SMEM_C_STRIDE
#undef SMEM_SMEM_X_OFF
#undef SMEM_SMEM_X_STAGE_BYTES
#undef SMEM_SMEM_X_STRIDE
#undef SMEM_TOTAL
#undef TMEM_NCOLS
#undef TMEM_SCORE_TMEM_OFFSET
#undef c_empty_addr
#undef c_full_addr
#undef score_empty_addr
#undef score_full_addr
#undef smem_c_addr
#undef smem_csq_addr
#undef smem_x_addr
#undef x_empty_addr
#undef x_full_addr

#define TMEM_NCOLS 256
#define TMEM_SCORE_TMEM_OFFSET 0
#define NUM_MAIN_STAGES 1
#define SMEM_SMEM_X_OFF 1024
#define SMEM_SMEM_X_STAGE_BYTES 16384
#define SMEM_SMEM_X_STRIDE 16384
#define SMEM_SMEM_C_OFF 17408
#define SMEM_SMEM_C_STAGE_BYTES 32768
#define SMEM_SMEM_C_STRIDE 32768
#define SMEM_SMEM_CSQ_OFF 50176
#define SMEM_SMEM_CSQ_STAGE_BYTES 1024
#define SMEM_SMEM_CSQ_STRIDE 1024
#define SMEM_TOTAL 51200

extern "C" {

__global__ __launch_bounds__(192) void
kernel_flash_kmeans_assign_highd_splitk_partial_8de8_v1(float* __restrict__ c_sq, float* __restrict__ partial_scores, int32_t* __restrict__ partial_indices, const void* x_tmap, const void* c_tmap, int B, int N, int D, int K, int num_n_tiles, int K_tiles)
{
    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;

    extern __shared__ __align__(1024) char smem_raw[];
    int smem;
    smem = (int)(unsigned long long)__cvta_generic_to_shared(smem_raw);
    const int smem_smem_x = smem + 1024;
    const int smem_smem_c = smem + 17408;
    const int smem_smem_csq = smem + 50176;

    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    // Mbarrier init (6 groups, 6 barriers)
    // Mbarriers at smem_raw[0..48)

    if (warp == 0) {
        uint32_t leader = elect_sync();
        // x_full: 1 barriers, init_count=1
        mbarrier_init_pred(smem + 0, 1, leader);
        // x_empty: 1 barriers, init_count=1
        mbarrier_init_pred(smem + 8, 1, leader);
        // c_full: 1 barriers, init_count=1
        mbarrier_init_pred(smem + 16, 1, leader);
        // c_empty: 1 barriers, init_count=1
        mbarrier_init_pred(smem + 24, 1, leader);
        // score_full: 1 barriers, init_count=1
        mbarrier_init_pred(smem + 32, 1, leader);
        // score_empty: 1 barriers, init_count=4
        mbarrier_init_pred(smem + 40, 4, leader);
        asm volatile("fence.mbarrier_init.release.cluster;");
    }

    __syncthreads();

    // TMEM alloc (256 columns, 256 used)
    volatile int* tmem_addr_storage = (volatile int*)(smem_raw + 48);
    if (warp == 1) {
        int _tmem_hold = smem + 48;
        asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;" :: "r"(_tmem_hold), "r"(256) : "memory");
    }

    __syncthreads();
    asm volatile("tcgen05.fence::after_thread_sync;");

    const int warp_id = warp;
    const int lane_id = lane;
    __nv_bfloat16* smem_x = (__nv_bfloat16*)(smem_raw + 1024);
    #define smem_x_addr (smem + 1024)
    __nv_bfloat16* smem_c = (__nv_bfloat16*)(smem_raw + 17408);
    #define smem_c_addr (smem + 17408)
    float* smem_csq = (float*)(smem_raw + 50176);
    #define smem_csq_addr (smem + 50176)
    const int mbar_base = smem;
    #define x_full_addr (mbar_base + 0)
    #define x_empty_addr (mbar_base + 8)
    #define c_full_addr (mbar_base + 16)
    #define c_empty_addr (mbar_base + 24)
    #define score_full_addr (mbar_base + 32)
    #define score_empty_addr (mbar_base + 40)
    const int taddr = tmem_addr_storage[0];

    // ---- Role: load ----
    if (warp == 0) {
        { // load_main
            int total_work = B * num_n_tiles * K_tiles;
            int feature_tiles = D / 64;
            uint32_t _phase_x_empty_0 = 1;
            uint32_t _phase_c_empty_0 = 1;
            if (warp_id == 0) {
                if (elect_sync()) {
                    #pragma unroll 1
                    for (unsigned int work_idx = bid; work_idx < total_work; work_idx += num_bids) {
                        int iter_k = work_idx % K_tiles;
                        int point_tile_idx = work_idx / K_tiles;
                        int batch = point_tile_idx / num_n_tiles;
                        int n_tile = point_tile_idx % num_n_tiles;
                        int off_n = n_tile * 128;
                        int off_k = iter_k * 256;
                        int x_row = batch * N + off_n;
                        int c_row = batch * K + off_k;
                        #pragma unroll 1
                        for (int feat_tile = 0; feat_tile < feature_tiles; feat_tile++) {
                            mbarrier_wait(x_empty_addr, _phase_x_empty_0);
                            _phase_x_empty_0 ^= 1;
                            tma_3d_gmem2smem(smem_x_addr, x_tmap, 0, x_row, feat_tile, x_full_addr);
                            mbarrier_arrive_expect_tx(x_full_addr, 16384);
                            mbarrier_wait(c_empty_addr, _phase_c_empty_0);
                            _phase_c_empty_0 ^= 1;
                            tma_3d_gmem2smem(smem_c_addr, c_tmap, 0, c_row, feat_tile, c_full_addr);
                            mbarrier_arrive_expect_tx(c_full_addr, 32768);
                        }
                    }
                }
            }
        }
    // ---- Role: mma ----
    } else if (warp == 1) {
        const int tmem_score_tmem = taddr + TMEM_SCORE_TMEM_OFFSET;
        { // mma_main
            int total_work = B * num_n_tiles * K_tiles;
            int feature_tiles = D / 64;
            uint32_t _phase_score_empty_0 = 1;
            uint32_t _phase_x_full_0 = 0;
            uint32_t _phase_c_full_0 = 0;
            #pragma unroll 1
            for (unsigned int _work_idx = bid; _work_idx < total_work; _work_idx += num_bids) {
                mbarrier_wait(score_empty_addr, _phase_score_empty_0);
                _phase_score_empty_0 ^= 1;
                #pragma unroll 1
                for (int feat_tile = 0; feat_tile < feature_tiles; feat_tile++) {
                    mbarrier_wait(x_full_addr, _phase_x_full_0);
                    _phase_x_full_0 ^= 1;
                    mbarrier_wait(c_full_addr, _phase_c_full_0);
                    _phase_c_full_0 ^= 1;
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    int init_flag = ((feat_tile == 0) ? 1 : 0);
                    int _mma_ss_a_addr_0 = smem_x_addr + 0 * 16384;
                    int _mma_ss_a_lo_0 = make_warp_uniform((_mma_ss_a_addr_0 >> 4) & 0x3FFF);
                    int _mma_ss_b_addr_0 = smem_c_addr + 0 * 32768;
                    int _mma_ss_b_lo_0 = make_warp_uniform((_mma_ss_b_addr_0 >> 4) & 0x3FFF);
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
                    "mov.b32 id, 138413200;\n\t"
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
                    :: "r"(_mma_ss_a_lo_0), "r"(_mma_ss_b_lo_0), "r"(tmem_score_tmem), "r"(((init_flag) ? 0 : 1)));
                    elect_commit(x_empty_addr);
                    elect_commit(c_empty_addr);
                }
                elect_commit(score_full_addr);
            }
        }
    // ---- Role: compute ----
    } else if (warp >= 2 && warp <= 5) {
        const int warp_id_in_wg = warp % 4;
        const int my_row = warp_id_in_wg * 32 + lane;
        const int tmem_row_base_v = (warp_id_in_wg * 32) << 16;
        { // compute_main
            int total_work = B * num_n_tiles * K_tiles;
            uint32_t _phase_score_full_0 = 0;
            #pragma unroll 1
            for (unsigned int work_idx = bid; work_idx < total_work; work_idx += num_bids) {
                int iter_k = work_idx % K_tiles;
                int point_tile_idx = work_idx / K_tiles;
                int batch = point_tile_idx / num_n_tiles;
                int off_k = iter_k * 256;
                int csq_smem_addr = smem_csq_addr + 0;
                float best_score = -3.4e+38f;
                int best_idx = off_k;
                if (my_row < 64.0f) {
                    int csq_base = my_row * 4;
                    float csq_pack[4];
                    {
                        float4 _v4 = *reinterpret_cast<const float4*>(c_sq + batch * K + off_k + csq_base);
                        csq_pack[0 + 0] = _v4.x;
                        csq_pack[0 + 1] = _v4.y;
                        csq_pack[0 + 2] = _v4.z;
                        csq_pack[0 + 3] = _v4.w;
                    }
                    float csq_half[4];
                    csq_half[0] = 0.5f * csq_pack[0];
                    csq_half[1] = 0.5f * csq_pack[1];
                    csq_half[2] = 0.5f * csq_pack[2];
                    csq_half[3] = 0.5f * csq_pack[3];
                    asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(csq_smem_addr + csq_base * 4), "f"(csq_half[0]), "f"(csq_half[1]), "f"(csq_half[2]), "f"(csq_half[3]) : "memory");
                }
                asm volatile("barrier.sync 8, 128;");
                mbarrier_wait(score_full_addr, _phase_score_full_0);
                _phase_score_full_0 ^= 1;
                int score_base = 0;
                float scores[128];
                asm volatile(
                    "tcgen05.ld.sync.aligned.32x32b.x128.b32"
                    " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63, %64, %65, %66, %67, %68, %69, %70, %71, %72, %73, %74, %75, %76, %77, %78, %79, %80, %81, %82, %83, %84, %85, %86, %87, %88, %89, %90, %91, %92, %93, %94, %95, %96, %97, %98, %99, %100, %101, %102, %103, %104, %105, %106, %107, %108, %109, %110, %111, %112, %113, %114, %115, %116, %117, %118, %119, %120, %121, %122, %123, %124, %125, %126, %127}, [%128];"
                    : "=f"(scores[0]), "=f"(scores[1]), "=f"(scores[2]), "=f"(scores[3]), "=f"(scores[4]), "=f"(scores[5]), "=f"(scores[6]), "=f"(scores[7]), "=f"(scores[8]), "=f"(scores[9]), "=f"(scores[10]), "=f"(scores[11]), "=f"(scores[12]), "=f"(scores[13]), "=f"(scores[14]), "=f"(scores[15]), "=f"(scores[16]), "=f"(scores[17]), "=f"(scores[18]), "=f"(scores[19]), "=f"(scores[20]), "=f"(scores[21]), "=f"(scores[22]), "=f"(scores[23]), "=f"(scores[24]), "=f"(scores[25]), "=f"(scores[26]), "=f"(scores[27]), "=f"(scores[28]), "=f"(scores[29]), "=f"(scores[30]), "=f"(scores[31]), "=f"(scores[32]), "=f"(scores[33]), "=f"(scores[34]), "=f"(scores[35]), "=f"(scores[36]), "=f"(scores[37]), "=f"(scores[38]), "=f"(scores[39]), "=f"(scores[40]), "=f"(scores[41]), "=f"(scores[42]), "=f"(scores[43]), "=f"(scores[44]), "=f"(scores[45]), "=f"(scores[46]), "=f"(scores[47]), "=f"(scores[48]), "=f"(scores[49]), "=f"(scores[50]), "=f"(scores[51]), "=f"(scores[52]), "=f"(scores[53]), "=f"(scores[54]), "=f"(scores[55]), "=f"(scores[56]), "=f"(scores[57]), "=f"(scores[58]), "=f"(scores[59]), "=f"(scores[60]), "=f"(scores[61]), "=f"(scores[62]), "=f"(scores[63]), "=f"(scores[64]), "=f"(scores[65]), "=f"(scores[66]), "=f"(scores[67]), "=f"(scores[68]), "=f"(scores[69]), "=f"(scores[70]), "=f"(scores[71]), "=f"(scores[72]), "=f"(scores[73]), "=f"(scores[74]), "=f"(scores[75]), "=f"(scores[76]), "=f"(scores[77]), "=f"(scores[78]), "=f"(scores[79]), "=f"(scores[80]), "=f"(scores[81]), "=f"(scores[82]), "=f"(scores[83]), "=f"(scores[84]), "=f"(scores[85]), "=f"(scores[86]), "=f"(scores[87]), "=f"(scores[88]), "=f"(scores[89]), "=f"(scores[90]), "=f"(scores[91]), "=f"(scores[92]), "=f"(scores[93]), "=f"(scores[94]), "=f"(scores[95]), "=f"(scores[96]), "=f"(scores[97]), "=f"(scores[98]), "=f"(scores[99]), "=f"(scores[100]), "=f"(scores[101]), "=f"(scores[102]), "=f"(scores[103]), "=f"(scores[104]), "=f"(scores[105]), "=f"(scores[106]), "=f"(scores[107]), "=f"(scores[108]), "=f"(scores[109]), "=f"(scores[110]), "=f"(scores[111]), "=f"(scores[112]), "=f"(scores[113]), "=f"(scores[114]), "=f"(scores[115]), "=f"(scores[116]), "=f"(scores[117]), "=f"(scores[118]), "=f"(scores[119]), "=f"(scores[120]), "=f"(scores[121]), "=f"(scores[122]), "=f"(scores[123]), "=f"(scores[124]), "=f"(scores[125]), "=f"(scores[126]), "=f"(scores[127])
                    : "r"(taddr + tmem_row_base_v + score_base)
                    : "memory");
                asm volatile("tcgen05.wait::ld.sync.aligned;");
                #pragma unroll
                for (int kk = 0; kk < 128; kk += 4) {
                    float csq_vals[4];
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&csq_vals[0])), "=r"(*reinterpret_cast<uint32_t*>(&csq_vals[1])), "=r"(*reinterpret_cast<uint32_t*>(&csq_vals[2])), "=r"(*reinterpret_cast<uint32_t*>(&csq_vals[3]))
                        : "r"(csq_smem_addr + (score_base + kk) * 4));
                    float d0 = scores[kk] - csq_vals[0];
                    float d1 = scores[kk + 1] - csq_vals[1];
                    float best_01 = d0;
                    int idx_01 = off_k + score_base + kk;
                    if (d1 > best_01) {
                        best_01 = d1;
                        idx_01 = off_k + score_base + kk + 1;
                    }
                    float d2 = scores[kk + 2] - csq_vals[2];
                    float d3 = scores[kk + 3] - csq_vals[3];
                    float best_23 = d2;
                    int idx_23 = off_k + score_base + kk + 2;
                    if (d3 > best_23) {
                        best_23 = d3;
                        idx_23 = off_k + score_base + kk + 3;
                    }
                    float best_group = best_01;
                    int idx_group = idx_01;
                    if (best_23 > best_group) {
                        best_group = best_23;
                        idx_group = idx_23;
                    }
                    if (best_group > best_score) {
                        best_score = best_group;
                        best_idx = idx_group;
                    }
                }
                score_base = 128;
                asm volatile(
                    "tcgen05.ld.sync.aligned.32x32b.x128.b32"
                    " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63, %64, %65, %66, %67, %68, %69, %70, %71, %72, %73, %74, %75, %76, %77, %78, %79, %80, %81, %82, %83, %84, %85, %86, %87, %88, %89, %90, %91, %92, %93, %94, %95, %96, %97, %98, %99, %100, %101, %102, %103, %104, %105, %106, %107, %108, %109, %110, %111, %112, %113, %114, %115, %116, %117, %118, %119, %120, %121, %122, %123, %124, %125, %126, %127}, [%128];"
                    : "=f"(scores[0]), "=f"(scores[1]), "=f"(scores[2]), "=f"(scores[3]), "=f"(scores[4]), "=f"(scores[5]), "=f"(scores[6]), "=f"(scores[7]), "=f"(scores[8]), "=f"(scores[9]), "=f"(scores[10]), "=f"(scores[11]), "=f"(scores[12]), "=f"(scores[13]), "=f"(scores[14]), "=f"(scores[15]), "=f"(scores[16]), "=f"(scores[17]), "=f"(scores[18]), "=f"(scores[19]), "=f"(scores[20]), "=f"(scores[21]), "=f"(scores[22]), "=f"(scores[23]), "=f"(scores[24]), "=f"(scores[25]), "=f"(scores[26]), "=f"(scores[27]), "=f"(scores[28]), "=f"(scores[29]), "=f"(scores[30]), "=f"(scores[31]), "=f"(scores[32]), "=f"(scores[33]), "=f"(scores[34]), "=f"(scores[35]), "=f"(scores[36]), "=f"(scores[37]), "=f"(scores[38]), "=f"(scores[39]), "=f"(scores[40]), "=f"(scores[41]), "=f"(scores[42]), "=f"(scores[43]), "=f"(scores[44]), "=f"(scores[45]), "=f"(scores[46]), "=f"(scores[47]), "=f"(scores[48]), "=f"(scores[49]), "=f"(scores[50]), "=f"(scores[51]), "=f"(scores[52]), "=f"(scores[53]), "=f"(scores[54]), "=f"(scores[55]), "=f"(scores[56]), "=f"(scores[57]), "=f"(scores[58]), "=f"(scores[59]), "=f"(scores[60]), "=f"(scores[61]), "=f"(scores[62]), "=f"(scores[63]), "=f"(scores[64]), "=f"(scores[65]), "=f"(scores[66]), "=f"(scores[67]), "=f"(scores[68]), "=f"(scores[69]), "=f"(scores[70]), "=f"(scores[71]), "=f"(scores[72]), "=f"(scores[73]), "=f"(scores[74]), "=f"(scores[75]), "=f"(scores[76]), "=f"(scores[77]), "=f"(scores[78]), "=f"(scores[79]), "=f"(scores[80]), "=f"(scores[81]), "=f"(scores[82]), "=f"(scores[83]), "=f"(scores[84]), "=f"(scores[85]), "=f"(scores[86]), "=f"(scores[87]), "=f"(scores[88]), "=f"(scores[89]), "=f"(scores[90]), "=f"(scores[91]), "=f"(scores[92]), "=f"(scores[93]), "=f"(scores[94]), "=f"(scores[95]), "=f"(scores[96]), "=f"(scores[97]), "=f"(scores[98]), "=f"(scores[99]), "=f"(scores[100]), "=f"(scores[101]), "=f"(scores[102]), "=f"(scores[103]), "=f"(scores[104]), "=f"(scores[105]), "=f"(scores[106]), "=f"(scores[107]), "=f"(scores[108]), "=f"(scores[109]), "=f"(scores[110]), "=f"(scores[111]), "=f"(scores[112]), "=f"(scores[113]), "=f"(scores[114]), "=f"(scores[115]), "=f"(scores[116]), "=f"(scores[117]), "=f"(scores[118]), "=f"(scores[119]), "=f"(scores[120]), "=f"(scores[121]), "=f"(scores[122]), "=f"(scores[123]), "=f"(scores[124]), "=f"(scores[125]), "=f"(scores[126]), "=f"(scores[127])
                    : "r"(taddr + tmem_row_base_v + score_base)
                    : "memory");
                asm volatile("tcgen05.wait::ld.sync.aligned;");
                asm volatile("barrier.sync 9, 128;");
                if (elect_sync()) {
                    mbarrier_arrive(score_empty_addr);
                }
                #pragma unroll
                for (int kk = 0; kk < 128; kk += 4) {
                    float csq_vals[4];
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&csq_vals[0])), "=r"(*reinterpret_cast<uint32_t*>(&csq_vals[1])), "=r"(*reinterpret_cast<uint32_t*>(&csq_vals[2])), "=r"(*reinterpret_cast<uint32_t*>(&csq_vals[3]))
                        : "r"(csq_smem_addr + (score_base + kk) * 4));
                    float d0 = scores[kk] - csq_vals[0];
                    float d1 = scores[kk + 1] - csq_vals[1];
                    float best_01 = d0;
                    int idx_01 = off_k + score_base + kk;
                    if (d1 > best_01) {
                        best_01 = d1;
                        idx_01 = off_k + score_base + kk + 1;
                    }
                    float d2 = scores[kk + 2] - csq_vals[2];
                    float d3 = scores[kk + 3] - csq_vals[3];
                    float best_23 = d2;
                    int idx_23 = off_k + score_base + kk + 2;
                    if (d3 > best_23) {
                        best_23 = d3;
                        idx_23 = off_k + score_base + kk + 3;
                    }
                    float best_group = best_01;
                    int idx_group = idx_01;
                    if (best_23 > best_group) {
                        best_group = best_23;
                        idx_group = idx_23;
                    }
                    if (best_group > best_score) {
                        best_score = best_group;
                        best_idx = idx_group;
                    }
                }
                int partial_offset = work_idx * 128 + my_row;
                *((float*)(partial_scores + partial_offset)) = best_score;
                *((int*)(partial_indices + partial_offset)) = best_idx;
            }
        }
    }

    // Cleanup
    __syncthreads(); // barrier before TMEM dealloc

    if (warp == 1) {
        asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;" :: "r"(tmem_addr_storage[0]), "r"(256));
        asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;");
    }
}

} // extern "C"

#undef NUM_MAIN_STAGES
#undef SMEM_SMEM_CSQ_OFF
#undef SMEM_SMEM_CSQ_STAGE_BYTES
#undef SMEM_SMEM_CSQ_STRIDE
#undef SMEM_SMEM_C_OFF
#undef SMEM_SMEM_C_STAGE_BYTES
#undef SMEM_SMEM_C_STRIDE
#undef SMEM_SMEM_X_OFF
#undef SMEM_SMEM_X_STAGE_BYTES
#undef SMEM_SMEM_X_STRIDE
#undef SMEM_TOTAL
#undef TMEM_NCOLS
#undef TMEM_SCORE_TMEM_OFFSET
#undef c_empty_addr
#undef c_full_addr
#undef score_empty_addr
#undef score_full_addr
#undef smem_c_addr
#undef smem_csq_addr
#undef smem_x_addr
#undef x_empty_addr
#undef x_full_addr

#define NUM_MAIN_STAGES 1
#define THREADS 128

extern "C" {

__global__ __launch_bounds__(128) void
kernel_flash_kmeans_assign_highd_splitk_reduce_8de8_v1(float* __restrict__ partial_scores, int32_t* __restrict__ partial_indices, int32_t* __restrict__ out, int B, int N, int K, int num_n_tiles, int K_tiles)
{
    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;


    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    const int warp_id = warp;
    const int lane_id = lane;

    // === Task calls (dependency order) ===
    int row = tid;
    int total_point_tiles = B * num_n_tiles;
    #pragma unroll 1
    for (unsigned int point_tile_idx = bid; point_tile_idx < total_point_tiles; point_tile_idx += num_bids) {
        int batch = point_tile_idx / num_n_tiles;
        int n_tile = point_tile_idx % num_n_tiles;
        int global_n = n_tile * 128 + row;
        float best_score = -3.4e+38f;
        int best_idx = 0;
        #pragma unroll 1
        for (int iter_k = 0; iter_k < K_tiles; iter_k++) {
            int partial_offset = (point_tile_idx * K_tiles + iter_k) * 128 + row;
            float score = (float)partial_scores[partial_offset];
            int idx = partial_indices[partial_offset];
            if (score > best_score) {
                best_score = score;
                best_idx = idx;
            }
        }
        int out_offset = batch * N + global_n;
        *((int*)(out + out_offset)) = best_idx;
    }
}

} // extern "C"

#undef NUM_MAIN_STAGES
#undef THREADS

#define TMEM_NCOLS 256
#define TMEM_SCORE_TMEM_OFFSET 0
#define NUM_MAIN_STAGES 1
#define SMEM_SMEM_X_OFF 1024
#define SMEM_SMEM_X_STAGE_BYTES 8192
#define SMEM_SMEM_X_STRIDE 8192
#define SMEM_SMEM_C_OFF 9216
#define SMEM_SMEM_C_STAGE_BYTES 32768
#define SMEM_SMEM_C_STRIDE 32768
#define SMEM_SMEM_CSQ_OFF 41984
#define SMEM_SMEM_CSQ_STAGE_BYTES 1024
#define SMEM_SMEM_CSQ_STRIDE 1024
#define SMEM_TOTAL 43008

extern "C" {

__global__ __launch_bounds__(192) void
kernel_flash_kmeans_assign_highd_splitk_partial_blockn64_g2r4_b5a6_v1(float* __restrict__ c_sq, float* __restrict__ partial_scores, int32_t* __restrict__ partial_indices, const void* x_tmap, const void* c_tmap, int B, int N, int D, int K, int num_n_tiles, int K_tiles, int K_slices)
{
    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;

    extern __shared__ __align__(1024) char smem_raw[];
    int smem;
    smem = (int)(unsigned long long)__cvta_generic_to_shared(smem_raw);
    const int smem_smem_x = smem + 1024;
    const int smem_smem_c = smem + 9216;
    const int smem_smem_csq = smem + 41984;

    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    // Mbarrier init (6 groups, 6 barriers)
    // Mbarriers at smem_raw[0..48)

    if (warp == 0) {
        uint32_t leader = elect_sync();
        // x_full: 1 barriers, init_count=1
        mbarrier_init_pred(smem + 0, 1, leader);
        // x_empty: 1 barriers, init_count=1
        mbarrier_init_pred(smem + 8, 1, leader);
        // c_full: 1 barriers, init_count=1
        mbarrier_init_pred(smem + 16, 1, leader);
        // c_empty: 1 barriers, init_count=1
        mbarrier_init_pred(smem + 24, 1, leader);
        // score_full: 1 barriers, init_count=1
        mbarrier_init_pred(smem + 32, 1, leader);
        // score_empty: 1 barriers, init_count=4
        mbarrier_init_pred(smem + 40, 4, leader);
        asm volatile("fence.mbarrier_init.release.cluster;");
    }

    __syncthreads();

    // TMEM alloc (256 columns, 256 used)
    volatile int* tmem_addr_storage = (volatile int*)(smem_raw + 48);
    if (warp == 5) {
        int _tmem_hold = smem + 48;
        asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;" :: "r"(_tmem_hold), "r"(256) : "memory");
    }

    __syncthreads();
    asm volatile("tcgen05.fence::after_thread_sync;");

    const int warp_id = warp;
    const int lane_id = lane;
    __nv_bfloat16* smem_x = (__nv_bfloat16*)(smem_raw + 1024);
    #define smem_x_addr (smem + 1024)
    __nv_bfloat16* smem_c = (__nv_bfloat16*)(smem_raw + 9216);
    #define smem_c_addr (smem + 9216)
    float* smem_csq = (float*)(smem_raw + 41984);
    #define smem_csq_addr (smem + 41984)
    const int mbar_base = smem;
    #define x_full_addr (mbar_base + 0)
    #define x_empty_addr (mbar_base + 8)
    #define c_full_addr (mbar_base + 16)
    #define c_empty_addr (mbar_base + 24)
    #define score_full_addr (mbar_base + 32)
    #define score_empty_addr (mbar_base + 40)
    const int taddr = tmem_addr_storage[0];

    // ---- Role: compute ----
    if (warp <= 3) {
        const int tmem_row_base = (warp % 4) * 32;
        const int my_row = tmem_row_base + (lane / 4);
        { // compute_main
            int total_work = B * num_n_tiles * K_slices;
            int compute_warp = warp;
            int lane_pair = lane % 4;
            int row_origin = compute_warp * 16;
            int row_lane_base = row_origin + lane / 4;
            int row0 = row_lane_base;
            int row1 = row_lane_base + 8;
            int compute_tid = compute_warp * 32 + lane;
            uint32_t _phase_score_full_0 = 0;
            #pragma unroll 1
            for (unsigned int work_idx = bid; work_idx < total_work; work_idx += num_bids) {
                int iter_slice = work_idx % K_slices;
                int point_tile_idx = work_idx / K_slices;
                int batch = point_tile_idx / num_n_tiles;
                int slice_k_start = iter_slice * 2;
                int csq_smem_addr = smem_csq_addr + 0;
                float best0 = -3.4e+38f;
                float best1 = -3.4e+38f;
                int idx0 = slice_k_start * 256;
                int idx1 = idx0;
                #pragma unroll 1
                for (int local_k = 0; local_k < 2; local_k++) {
                    int iter_k = slice_k_start + local_k;
                    int off_k = iter_k * 256;
                    if (compute_tid < 64.0f) {
                        int csq_base = compute_tid * 4;
                        float csq_pack[4];
                        {
                            float4 _v4 = *reinterpret_cast<const float4*>(c_sq + batch * K + off_k + csq_base);
                            csq_pack[0 + 0] = _v4.x;
                            csq_pack[0 + 1] = _v4.y;
                            csq_pack[0 + 2] = _v4.z;
                            csq_pack[0 + 3] = _v4.w;
                        }
                        float csq_half[4];
                        csq_half[0] = 0.5f * csq_pack[0];
                        csq_half[1] = 0.5f * csq_pack[1];
                        csq_half[2] = 0.5f * csq_pack[2];
                        csq_half[3] = 0.5f * csq_pack[3];
                        asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(csq_smem_addr + csq_base * 4), "f"(csq_half[0]), "f"(csq_half[1]), "f"(csq_half[2]), "f"(csq_half[3]) : "memory");
                    }
                    asm volatile("barrier.sync 8, 128;");
                    mbarrier_wait(score_full_addr, _phase_score_full_0);
                    _phase_score_full_0 ^= 1;
                    #pragma unroll
                    for (int score_base = 0; score_base < 256; score_base += 128) {
                        float scores[64];
                        asm volatile(
                            "tcgen05.ld.sync.aligned.16x256b.x16.b32"
                            " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63}, [%64];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&scores[0])), "=r"(*reinterpret_cast<uint32_t*>(&scores[1])), "=r"(*reinterpret_cast<uint32_t*>(&scores[2])), "=r"(*reinterpret_cast<uint32_t*>(&scores[3])), "=r"(*reinterpret_cast<uint32_t*>(&scores[4])), "=r"(*reinterpret_cast<uint32_t*>(&scores[5])), "=r"(*reinterpret_cast<uint32_t*>(&scores[6])), "=r"(*reinterpret_cast<uint32_t*>(&scores[7])), "=r"(*reinterpret_cast<uint32_t*>(&scores[8])), "=r"(*reinterpret_cast<uint32_t*>(&scores[9])), "=r"(*reinterpret_cast<uint32_t*>(&scores[10])), "=r"(*reinterpret_cast<uint32_t*>(&scores[11])), "=r"(*reinterpret_cast<uint32_t*>(&scores[12])), "=r"(*reinterpret_cast<uint32_t*>(&scores[13])), "=r"(*reinterpret_cast<uint32_t*>(&scores[14])), "=r"(*reinterpret_cast<uint32_t*>(&scores[15])), "=r"(*reinterpret_cast<uint32_t*>(&scores[16])), "=r"(*reinterpret_cast<uint32_t*>(&scores[17])), "=r"(*reinterpret_cast<uint32_t*>(&scores[18])), "=r"(*reinterpret_cast<uint32_t*>(&scores[19])), "=r"(*reinterpret_cast<uint32_t*>(&scores[20])), "=r"(*reinterpret_cast<uint32_t*>(&scores[21])), "=r"(*reinterpret_cast<uint32_t*>(&scores[22])), "=r"(*reinterpret_cast<uint32_t*>(&scores[23])), "=r"(*reinterpret_cast<uint32_t*>(&scores[24])), "=r"(*reinterpret_cast<uint32_t*>(&scores[25])), "=r"(*reinterpret_cast<uint32_t*>(&scores[26])), "=r"(*reinterpret_cast<uint32_t*>(&scores[27])), "=r"(*reinterpret_cast<uint32_t*>(&scores[28])), "=r"(*reinterpret_cast<uint32_t*>(&scores[29])), "=r"(*reinterpret_cast<uint32_t*>(&scores[30])), "=r"(*reinterpret_cast<uint32_t*>(&scores[31])), "=r"(*reinterpret_cast<uint32_t*>(&scores[32])), "=r"(*reinterpret_cast<uint32_t*>(&scores[33])), "=r"(*reinterpret_cast<uint32_t*>(&scores[34])), "=r"(*reinterpret_cast<uint32_t*>(&scores[35])), "=r"(*reinterpret_cast<uint32_t*>(&scores[36])), "=r"(*reinterpret_cast<uint32_t*>(&scores[37])), "=r"(*reinterpret_cast<uint32_t*>(&scores[38])), "=r"(*reinterpret_cast<uint32_t*>(&scores[39])), "=r"(*reinterpret_cast<uint32_t*>(&scores[40])), "=r"(*reinterpret_cast<uint32_t*>(&scores[41])), "=r"(*reinterpret_cast<uint32_t*>(&scores[42])), "=r"(*reinterpret_cast<uint32_t*>(&scores[43])), "=r"(*reinterpret_cast<uint32_t*>(&scores[44])), "=r"(*reinterpret_cast<uint32_t*>(&scores[45])), "=r"(*reinterpret_cast<uint32_t*>(&scores[46])), "=r"(*reinterpret_cast<uint32_t*>(&scores[47])), "=r"(*reinterpret_cast<uint32_t*>(&scores[48])), "=r"(*reinterpret_cast<uint32_t*>(&scores[49])), "=r"(*reinterpret_cast<uint32_t*>(&scores[50])), "=r"(*reinterpret_cast<uint32_t*>(&scores[51])), "=r"(*reinterpret_cast<uint32_t*>(&scores[52])), "=r"(*reinterpret_cast<uint32_t*>(&scores[53])), "=r"(*reinterpret_cast<uint32_t*>(&scores[54])), "=r"(*reinterpret_cast<uint32_t*>(&scores[55])), "=r"(*reinterpret_cast<uint32_t*>(&scores[56])), "=r"(*reinterpret_cast<uint32_t*>(&scores[57])), "=r"(*reinterpret_cast<uint32_t*>(&scores[58])), "=r"(*reinterpret_cast<uint32_t*>(&scores[59])), "=r"(*reinterpret_cast<uint32_t*>(&scores[60])), "=r"(*reinterpret_cast<uint32_t*>(&scores[61])), "=r"(*reinterpret_cast<uint32_t*>(&scores[62])), "=r"(*reinterpret_cast<uint32_t*>(&scores[63]))
                            : "r"(taddr + score_base)
                            : "memory");
                        asm volatile("tcgen05.wait::ld.sync.aligned;");
                        #pragma unroll
                        for (int rep = 0; rep < 16.0f; rep++) {
                            int local_reg = rep * 4;
                            int col_base = score_base + rep * 8 + lane_pair * 2;
                            float csq0 = smem_csq[col_base];
                            float csq1 = smem_csq[col_base + 1];
                            float d0 = scores[local_reg] - csq0;
                            if (d0 > best0) {
                                best0 = d0;
                                idx0 = off_k + col_base;
                            }
                            float d1 = scores[local_reg + 1] - csq1;
                            if (d1 > best0) {
                                best0 = d1;
                                idx0 = off_k + col_base + 1;
                            }
                            float d2 = scores[local_reg + 2] - csq0;
                            if (d2 > best1) {
                                best1 = d2;
                                idx1 = off_k + col_base;
                            }
                            float d3 = scores[local_reg + 3] - csq1;
                            if (d3 > best1) {
                                best1 = d3;
                                idx1 = off_k + col_base + 1;
                            }
                        }
                    }
                    asm volatile("barrier.sync 9, 128;");
                    if (elect_sync()) {
                        mbarrier_arrive(score_empty_addr);
                    }
                }
                float _shfl_xor_0 = __shfl_xor_sync(0xFFFFFFFF, best0, 1);
                float peer0 = _shfl_xor_0;
                int _shfl_xor_1 = __shfl_xor_sync(0xFFFFFFFF, idx0, 1);
                int peer0_idx = _shfl_xor_1;
                if (peer0 > best0) {
                    best0 = peer0;
                    idx0 = peer0_idx;
                }
                peer0 = __shfl_xor_sync(0xFFFFFFFF, best0, 2);
                peer0_idx = __shfl_xor_sync(0xFFFFFFFF, idx0, 2);
                if (peer0 > best0) {
                    best0 = peer0;
                    idx0 = peer0_idx;
                }
                float _shfl_xor_2 = __shfl_xor_sync(0xFFFFFFFF, best1, 1);
                float peer1 = _shfl_xor_2;
                int _shfl_xor_3 = __shfl_xor_sync(0xFFFFFFFF, idx1, 1);
                int peer1_idx = _shfl_xor_3;
                if (peer1 > best1) {
                    best1 = peer1;
                    idx1 = peer1_idx;
                }
                peer1 = __shfl_xor_sync(0xFFFFFFFF, best1, 2);
                peer1_idx = __shfl_xor_sync(0xFFFFFFFF, idx1, 2);
                if (peer1 > best1) {
                    best1 = peer1;
                    idx1 = peer1_idx;
                }
                if (lane_pair == 0) {
                    int partial_offset0 = work_idx * 64 + row0;
                    *((float*)(partial_scores + partial_offset0)) = best0;
                    *((int*)(partial_indices + partial_offset0)) = idx0;
                    int partial_offset1 = work_idx * 64 + row1;
                    *((float*)(partial_scores + partial_offset1)) = best1;
                    *((int*)(partial_indices + partial_offset1)) = idx1;
                }
            }
        }
    // ---- Role: load ----
    } else if (warp == 4) {
        { // load_main
            int total_work = B * num_n_tiles * K_slices;
            int feature_tiles = D / 64;
            uint32_t _phase_x_empty_0 = 1;
            uint32_t _phase_c_empty_0 = 1;
            if (elect_sync()) {
                #pragma unroll 1
                for (unsigned int work_idx = bid; work_idx < total_work; work_idx += num_bids) {
                    int iter_slice = work_idx % K_slices;
                    int point_tile_idx = work_idx / K_slices;
                    int batch = point_tile_idx / num_n_tiles;
                    int n_tile = point_tile_idx % num_n_tiles;
                    int off_n = n_tile * 64;
                    int x_row = batch * N + off_n;
                    int slice_k_start = iter_slice * 2;
                    #pragma unroll 1
                    for (int local_k = 0; local_k < 2; local_k++) {
                        int iter_k = slice_k_start + local_k;
                        int off_k = iter_k * 256;
                        int c_row = batch * K + off_k;
                        #pragma unroll 1
                        for (int feat_tile = 0; feat_tile < feature_tiles; feat_tile++) {
                            mbarrier_wait(x_empty_addr, _phase_x_empty_0);
                            _phase_x_empty_0 ^= 1;
                            tma_3d_gmem2smem(smem_x_addr, x_tmap, 0, x_row, feat_tile, x_full_addr);
                            mbarrier_arrive_expect_tx(x_full_addr, 8192);
                            mbarrier_wait(c_empty_addr, _phase_c_empty_0);
                            _phase_c_empty_0 ^= 1;
                            tma_3d_gmem2smem(smem_c_addr, c_tmap, 0, c_row, feat_tile, c_full_addr);
                            mbarrier_arrive_expect_tx(c_full_addr, 32768);
                        }
                    }
                }
            }
        }
    // ---- Role: mma ----
    } else if (warp == 5) {
        const int tmem_score_tmem = taddr + TMEM_SCORE_TMEM_OFFSET;
        { // mma_main
            int total_work = B * num_n_tiles * K_slices;
            int feature_tiles = D / 64;
            uint32_t _phase_score_empty_0 = 1;
            uint32_t _phase_x_full_0 = 0;
            uint32_t _phase_c_full_0 = 0;
            #pragma unroll 1
            for (unsigned int work_idx = bid; work_idx < total_work; work_idx += num_bids) {
                int iter_slice = work_idx % K_slices;
                int slice_k_start = iter_slice * 2;
                #pragma unroll 1
                for (int local_k = 0; local_k < 2; local_k++) {
                    mbarrier_wait(score_empty_addr, _phase_score_empty_0);
                    _phase_score_empty_0 ^= 1;
                    #pragma unroll 1
                    for (int feat_tile = 0; feat_tile < feature_tiles; feat_tile++) {
                        mbarrier_wait(x_full_addr, _phase_x_full_0);
                        _phase_x_full_0 ^= 1;
                        mbarrier_wait(c_full_addr, _phase_c_full_0);
                        _phase_c_full_0 ^= 1;
                        asm volatile("tcgen05.fence::after_thread_sync;");
                        int init_flag = ((feat_tile == 0) ? 1 : 0);
                        int _mma_ss_a_addr_0 = smem_x_addr + 0 * 8192;
                        int _mma_ss_a_lo_0 = make_warp_uniform((_mma_ss_a_addr_0 >> 4) & 0x3FFF);
                        int _mma_ss_b_addr_0 = smem_c_addr + 0 * 32768;
                        int _mma_ss_b_lo_0 = make_warp_uniform((_mma_ss_b_addr_0 >> 4) & 0x3FFF);
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
                    "mov.b32 id, 71304336;\n\t"
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
                    :: "r"(_mma_ss_a_lo_0), "r"(_mma_ss_b_lo_0), "r"(tmem_score_tmem), "r"(((init_flag) ? 0 : 1)));
                        elect_commit(x_empty_addr);
                        elect_commit(c_empty_addr);
                    }
                    elect_commit(score_full_addr);
                }
            }
        }
    }

    // Cleanup
    __syncthreads(); // barrier before TMEM dealloc

    if (warp == 5) {
        asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;" :: "r"(tmem_addr_storage[0]), "r"(256));
        asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;");
    }
}

} // extern "C"

#undef NUM_MAIN_STAGES
#undef SMEM_SMEM_CSQ_OFF
#undef SMEM_SMEM_CSQ_STAGE_BYTES
#undef SMEM_SMEM_CSQ_STRIDE
#undef SMEM_SMEM_C_OFF
#undef SMEM_SMEM_C_STAGE_BYTES
#undef SMEM_SMEM_C_STRIDE
#undef SMEM_SMEM_X_OFF
#undef SMEM_SMEM_X_STAGE_BYTES
#undef SMEM_SMEM_X_STRIDE
#undef SMEM_TOTAL
#undef TMEM_NCOLS
#undef TMEM_SCORE_TMEM_OFFSET
#undef c_empty_addr
#undef c_full_addr
#undef score_empty_addr
#undef score_full_addr
#undef smem_c_addr
#undef smem_csq_addr
#undef smem_x_addr
#undef x_empty_addr
#undef x_full_addr

#define NUM_MAIN_STAGES 1
#define THREADS 256

extern "C" {

__global__ __launch_bounds__(256) void
kernel_flash_kmeans_assign_highd_splitk_reduce_blockn64_g2r4_b5a6_v1(float* __restrict__ partial_scores, int32_t* __restrict__ partial_indices, int32_t* __restrict__ out, int B, int N, int K, int num_n_tiles, int K_slices)
{
    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;


    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    const int warp_id = warp;
    const int lane_id = lane;

    // === Task calls (dependency order) ===
    int row = tid / 4;
    int row_lane = tid % 4;
    int total_point_tiles = B * num_n_tiles;
    #pragma unroll 1
    for (unsigned int point_tile_idx = bid; point_tile_idx < total_point_tiles; point_tile_idx += num_bids) {
        int batch = point_tile_idx / num_n_tiles;
        int n_tile = point_tile_idx % num_n_tiles;
        int global_n = n_tile * 64 + row;
        float best_score = -3.4e+38f;
        int best_idx = 0;
        #pragma unroll 1
        for (int iter_slice = row_lane; iter_slice < K_slices; iter_slice += 4) {
            int partial_offset = (point_tile_idx * K_slices + iter_slice) * 64 + row;
            float score = (float)partial_scores[partial_offset];
            int idx = partial_indices[partial_offset];
            if (score > best_score) {
                best_score = score;
                best_idx = idx;
            }
        }
        float _shfl_xor_0 = __shfl_xor_sync(0xFFFFFFFF, best_score, 1);
        float peer_score = _shfl_xor_0;
        int _shfl_xor_1 = __shfl_xor_sync(0xFFFFFFFF, best_idx, 1);
        int peer_idx = _shfl_xor_1;
        if (peer_score > best_score) {
            best_score = peer_score;
            best_idx = peer_idx;
        }
        peer_score = __shfl_xor_sync(0xFFFFFFFF, best_score, 2);
        peer_idx = __shfl_xor_sync(0xFFFFFFFF, best_idx, 2);
        if (peer_score > best_score) {
            best_score = peer_score;
            best_idx = peer_idx;
        }
        if (row_lane == 0) {
            int out_offset = batch * N + global_n;
            *((int*)(out + out_offset)) = best_idx;
        }
    }
}

} // extern "C"

#undef NUM_MAIN_STAGES
#undef THREADS

#define TMEM_NCOLS 256
#define TMEM_SCORE_TMEM_OFFSET 0
#define NUM_MAIN_STAGES 1
#define SMEM_SMEM_X_OFF 1024
#define SMEM_SMEM_X_STAGE_BYTES 8192
#define SMEM_SMEM_X_STRIDE 8192
#define SMEM_SMEM_C_OFF 9216
#define SMEM_SMEM_C_STAGE_BYTES 32768
#define SMEM_SMEM_C_STRIDE 32768
#define SMEM_SMEM_CSQ_OFF 41984
#define SMEM_SMEM_CSQ_STAGE_BYTES 1024
#define SMEM_SMEM_CSQ_STRIDE 1024
#define SMEM_TOTAL 43008

extern "C" {

__global__ __launch_bounds__(192) void
kernel_flash_kmeans_assign_highd_splitk_partial_blockn64_g1r4_streamdep_r63_v1(float* __restrict__ c_sq, float* __restrict__ partial_scores, int32_t* __restrict__ partial_indices, const void* x_tmap, const void* c_tmap, int B, int N, int D, int K, int num_n_tiles, int K_tiles, int K_slices)
{
    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;

    extern __shared__ __align__(1024) char smem_raw[];
    int smem;
    smem = (int)(unsigned long long)__cvta_generic_to_shared(smem_raw);
    const int smem_smem_x = smem + 1024;
    const int smem_smem_c = smem + 9216;
    const int smem_smem_csq = smem + 41984;

    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    // Mbarrier init (6 groups, 6 barriers)
    // Mbarriers at smem_raw[0..48)

    if (warp == 0) {
        uint32_t leader = elect_sync();
        // x_full: 1 barriers, init_count=1
        mbarrier_init_pred(smem + 0, 1, leader);
        // x_empty: 1 barriers, init_count=1
        mbarrier_init_pred(smem + 8, 1, leader);
        // c_full: 1 barriers, init_count=1
        mbarrier_init_pred(smem + 16, 1, leader);
        // c_empty: 1 barriers, init_count=1
        mbarrier_init_pred(smem + 24, 1, leader);
        // score_full: 1 barriers, init_count=1
        mbarrier_init_pred(smem + 32, 1, leader);
        // score_empty: 1 barriers, init_count=4
        mbarrier_init_pred(smem + 40, 4, leader);
        asm volatile("fence.mbarrier_init.release.cluster;");
    }

    __syncthreads();

    // TMEM alloc (256 columns, 256 used)
    volatile int* tmem_addr_storage = (volatile int*)(smem_raw + 48);
    if (warp == 5) {
        int _tmem_hold = smem + 48;
        asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;" :: "r"(_tmem_hold), "r"(256) : "memory");
    }

    __syncthreads();
    asm volatile("tcgen05.fence::after_thread_sync;");

    const int warp_id = warp;
    const int lane_id = lane;
    __nv_bfloat16* smem_x = (__nv_bfloat16*)(smem_raw + 1024);
    #define smem_x_addr (smem + 1024)
    __nv_bfloat16* smem_c = (__nv_bfloat16*)(smem_raw + 9216);
    #define smem_c_addr (smem + 9216)
    float* smem_csq = (float*)(smem_raw + 41984);
    #define smem_csq_addr (smem + 41984)
    const int mbar_base = smem;
    #define x_full_addr (mbar_base + 0)
    #define x_empty_addr (mbar_base + 8)
    #define c_full_addr (mbar_base + 16)
    #define c_empty_addr (mbar_base + 24)
    #define score_full_addr (mbar_base + 32)
    #define score_empty_addr (mbar_base + 40)
    const int taddr = tmem_addr_storage[0];

    // ---- Role: compute ----
    if (warp <= 3) {
        const int tmem_row_base = (warp % 4) * 32;
        const int my_row = tmem_row_base + (lane / 4);
        { // compute_main
            int total_work = B * num_n_tiles * K_slices;
            int compute_warp = warp;
            int lane_pair = lane % 4;
            int row_origin = compute_warp * 16;
            int row_lane_base = row_origin + lane / 4;
            int row0 = row_lane_base;
            int row1 = row_lane_base + 8;
            int compute_tid = compute_warp * 32 + lane;
            uint32_t _phase_score_full_0 = 0;
            #pragma unroll 1
            for (unsigned int work_idx = bid; work_idx < total_work; work_idx += num_bids) {
                int iter_slice = work_idx % K_slices;
                int point_tile_idx = work_idx / K_slices;
                int batch = point_tile_idx / num_n_tiles;
                int slice_k_start = iter_slice * 1;
                int csq_smem_addr = smem_csq_addr + 0;
                float best0 = -3.4e+38f;
                float best1 = -3.4e+38f;
                int idx0 = slice_k_start * 256;
                int idx1 = idx0;
                #pragma unroll 1
                for (int local_k = 0; local_k < 1; local_k++) {
                    int iter_k = slice_k_start + local_k;
                    int off_k = iter_k * 256;
                    if (compute_tid < 64.0f) {
                        int csq_base = compute_tid * 4;
                        float csq_pack[4];
                        {
                            float4 _v4 = *reinterpret_cast<const float4*>(c_sq + batch * K + off_k + csq_base);
                            csq_pack[0 + 0] = _v4.x;
                            csq_pack[0 + 1] = _v4.y;
                            csq_pack[0 + 2] = _v4.z;
                            csq_pack[0 + 3] = _v4.w;
                        }
                        float csq_half[4];
                        csq_half[0] = 0.5f * csq_pack[0];
                        csq_half[1] = 0.5f * csq_pack[1];
                        csq_half[2] = 0.5f * csq_pack[2];
                        csq_half[3] = 0.5f * csq_pack[3];
                        asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(csq_smem_addr + csq_base * 4), "f"(csq_half[0]), "f"(csq_half[1]), "f"(csq_half[2]), "f"(csq_half[3]) : "memory");
                    }
                    asm volatile("barrier.sync 8, 128;");
                    mbarrier_wait(score_full_addr, _phase_score_full_0);
                    _phase_score_full_0 ^= 1;
                    #pragma unroll
                    for (int score_base = 0; score_base < 256; score_base += 128) {
                        float scores[64];
                        asm volatile(
                            "tcgen05.ld.sync.aligned.16x256b.x16.b32"
                            " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63}, [%64];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&scores[0])), "=r"(*reinterpret_cast<uint32_t*>(&scores[1])), "=r"(*reinterpret_cast<uint32_t*>(&scores[2])), "=r"(*reinterpret_cast<uint32_t*>(&scores[3])), "=r"(*reinterpret_cast<uint32_t*>(&scores[4])), "=r"(*reinterpret_cast<uint32_t*>(&scores[5])), "=r"(*reinterpret_cast<uint32_t*>(&scores[6])), "=r"(*reinterpret_cast<uint32_t*>(&scores[7])), "=r"(*reinterpret_cast<uint32_t*>(&scores[8])), "=r"(*reinterpret_cast<uint32_t*>(&scores[9])), "=r"(*reinterpret_cast<uint32_t*>(&scores[10])), "=r"(*reinterpret_cast<uint32_t*>(&scores[11])), "=r"(*reinterpret_cast<uint32_t*>(&scores[12])), "=r"(*reinterpret_cast<uint32_t*>(&scores[13])), "=r"(*reinterpret_cast<uint32_t*>(&scores[14])), "=r"(*reinterpret_cast<uint32_t*>(&scores[15])), "=r"(*reinterpret_cast<uint32_t*>(&scores[16])), "=r"(*reinterpret_cast<uint32_t*>(&scores[17])), "=r"(*reinterpret_cast<uint32_t*>(&scores[18])), "=r"(*reinterpret_cast<uint32_t*>(&scores[19])), "=r"(*reinterpret_cast<uint32_t*>(&scores[20])), "=r"(*reinterpret_cast<uint32_t*>(&scores[21])), "=r"(*reinterpret_cast<uint32_t*>(&scores[22])), "=r"(*reinterpret_cast<uint32_t*>(&scores[23])), "=r"(*reinterpret_cast<uint32_t*>(&scores[24])), "=r"(*reinterpret_cast<uint32_t*>(&scores[25])), "=r"(*reinterpret_cast<uint32_t*>(&scores[26])), "=r"(*reinterpret_cast<uint32_t*>(&scores[27])), "=r"(*reinterpret_cast<uint32_t*>(&scores[28])), "=r"(*reinterpret_cast<uint32_t*>(&scores[29])), "=r"(*reinterpret_cast<uint32_t*>(&scores[30])), "=r"(*reinterpret_cast<uint32_t*>(&scores[31])), "=r"(*reinterpret_cast<uint32_t*>(&scores[32])), "=r"(*reinterpret_cast<uint32_t*>(&scores[33])), "=r"(*reinterpret_cast<uint32_t*>(&scores[34])), "=r"(*reinterpret_cast<uint32_t*>(&scores[35])), "=r"(*reinterpret_cast<uint32_t*>(&scores[36])), "=r"(*reinterpret_cast<uint32_t*>(&scores[37])), "=r"(*reinterpret_cast<uint32_t*>(&scores[38])), "=r"(*reinterpret_cast<uint32_t*>(&scores[39])), "=r"(*reinterpret_cast<uint32_t*>(&scores[40])), "=r"(*reinterpret_cast<uint32_t*>(&scores[41])), "=r"(*reinterpret_cast<uint32_t*>(&scores[42])), "=r"(*reinterpret_cast<uint32_t*>(&scores[43])), "=r"(*reinterpret_cast<uint32_t*>(&scores[44])), "=r"(*reinterpret_cast<uint32_t*>(&scores[45])), "=r"(*reinterpret_cast<uint32_t*>(&scores[46])), "=r"(*reinterpret_cast<uint32_t*>(&scores[47])), "=r"(*reinterpret_cast<uint32_t*>(&scores[48])), "=r"(*reinterpret_cast<uint32_t*>(&scores[49])), "=r"(*reinterpret_cast<uint32_t*>(&scores[50])), "=r"(*reinterpret_cast<uint32_t*>(&scores[51])), "=r"(*reinterpret_cast<uint32_t*>(&scores[52])), "=r"(*reinterpret_cast<uint32_t*>(&scores[53])), "=r"(*reinterpret_cast<uint32_t*>(&scores[54])), "=r"(*reinterpret_cast<uint32_t*>(&scores[55])), "=r"(*reinterpret_cast<uint32_t*>(&scores[56])), "=r"(*reinterpret_cast<uint32_t*>(&scores[57])), "=r"(*reinterpret_cast<uint32_t*>(&scores[58])), "=r"(*reinterpret_cast<uint32_t*>(&scores[59])), "=r"(*reinterpret_cast<uint32_t*>(&scores[60])), "=r"(*reinterpret_cast<uint32_t*>(&scores[61])), "=r"(*reinterpret_cast<uint32_t*>(&scores[62])), "=r"(*reinterpret_cast<uint32_t*>(&scores[63]))
                            : "r"(taddr + score_base)
                            : "memory");
                        asm volatile("tcgen05.wait::ld.sync.aligned;");
                        #pragma unroll
                        for (int rep = 0; rep < 16.0f; rep++) {
                            int local_reg = rep * 4;
                            int col_base = score_base + rep * 8 + lane_pair * 2;
                            float csq0 = smem_csq[col_base];
                            float csq1 = smem_csq[col_base + 1];
                            float d0 = scores[local_reg] - csq0;
                            if (d0 > best0) {
                                best0 = d0;
                                idx0 = off_k + col_base;
                            }
                            float d1 = scores[local_reg + 1] - csq1;
                            if (d1 > best0) {
                                best0 = d1;
                                idx0 = off_k + col_base + 1;
                            }
                            float d2 = scores[local_reg + 2] - csq0;
                            if (d2 > best1) {
                                best1 = d2;
                                idx1 = off_k + col_base;
                            }
                            float d3 = scores[local_reg + 3] - csq1;
                            if (d3 > best1) {
                                best1 = d3;
                                idx1 = off_k + col_base + 1;
                            }
                        }
                    }
                    asm volatile("barrier.sync 9, 128;");
                    if (elect_sync()) {
                        mbarrier_arrive(score_empty_addr);
                    }
                }
                float _shfl_xor_0 = __shfl_xor_sync(0xFFFFFFFF, best0, 1);
                float peer0 = _shfl_xor_0;
                int _shfl_xor_1 = __shfl_xor_sync(0xFFFFFFFF, idx0, 1);
                int peer0_idx = _shfl_xor_1;
                if (peer0 > best0) {
                    best0 = peer0;
                    idx0 = peer0_idx;
                }
                peer0 = __shfl_xor_sync(0xFFFFFFFF, best0, 2);
                peer0_idx = __shfl_xor_sync(0xFFFFFFFF, idx0, 2);
                if (peer0 > best0) {
                    best0 = peer0;
                    idx0 = peer0_idx;
                }
                float _shfl_xor_2 = __shfl_xor_sync(0xFFFFFFFF, best1, 1);
                float peer1 = _shfl_xor_2;
                int _shfl_xor_3 = __shfl_xor_sync(0xFFFFFFFF, idx1, 1);
                int peer1_idx = _shfl_xor_3;
                if (peer1 > best1) {
                    best1 = peer1;
                    idx1 = peer1_idx;
                }
                peer1 = __shfl_xor_sync(0xFFFFFFFF, best1, 2);
                peer1_idx = __shfl_xor_sync(0xFFFFFFFF, idx1, 2);
                if (peer1 > best1) {
                    best1 = peer1;
                    idx1 = peer1_idx;
                }
                if (lane_pair == 0) {
                    int partial_offset0 = work_idx * 64 + row0;
                    *((float*)(partial_scores + partial_offset0)) = best0;
                    *((int*)(partial_indices + partial_offset0)) = idx0;
                    int partial_offset1 = work_idx * 64 + row1;
                    *((float*)(partial_scores + partial_offset1)) = best1;
                    *((int*)(partial_indices + partial_offset1)) = idx1;
                }
            }
        }
    // ---- Role: load ----
    } else if (warp == 4) {
        { // load_main
            int total_work = B * num_n_tiles * K_slices;
            int feature_tiles = D / 64;
            uint32_t _phase_x_empty_0 = 1;
            uint32_t _phase_c_empty_0 = 1;
            if (elect_sync()) {
                #pragma unroll 1
                for (unsigned int work_idx = bid; work_idx < total_work; work_idx += num_bids) {
                    int iter_slice = work_idx % K_slices;
                    int point_tile_idx = work_idx / K_slices;
                    int batch = point_tile_idx / num_n_tiles;
                    int n_tile = point_tile_idx % num_n_tiles;
                    int off_n = n_tile * 64;
                    int x_row = batch * N + off_n;
                    int slice_k_start = iter_slice * 1;
                    #pragma unroll 1
                    for (int local_k = 0; local_k < 1; local_k++) {
                        int iter_k = slice_k_start + local_k;
                        int off_k = iter_k * 256;
                        int c_row = batch * K + off_k;
                        #pragma unroll 1
                        for (int feat_tile = 0; feat_tile < feature_tiles; feat_tile++) {
                            mbarrier_wait(x_empty_addr, _phase_x_empty_0);
                            _phase_x_empty_0 ^= 1;
                            tma_3d_gmem2smem(smem_x_addr, x_tmap, 0, x_row, feat_tile, x_full_addr);
                            mbarrier_arrive_expect_tx(x_full_addr, 8192);
                            mbarrier_wait(c_empty_addr, _phase_c_empty_0);
                            _phase_c_empty_0 ^= 1;
                            tma_3d_gmem2smem(smem_c_addr, c_tmap, 0, c_row, feat_tile, c_full_addr);
                            mbarrier_arrive_expect_tx(c_full_addr, 32768);
                        }
                    }
                }
            }
        }
    // ---- Role: mma ----
    } else if (warp == 5) {
        const int tmem_score_tmem = taddr + TMEM_SCORE_TMEM_OFFSET;
        { // mma_main
            int total_work = B * num_n_tiles * K_slices;
            int feature_tiles = D / 64;
            uint32_t _phase_score_empty_0 = 1;
            uint32_t _phase_x_full_0 = 0;
            uint32_t _phase_c_full_0 = 0;
            #pragma unroll 1
            for (unsigned int work_idx = bid; work_idx < total_work; work_idx += num_bids) {
                int iter_slice = work_idx % K_slices;
                int slice_k_start = iter_slice * 1;
                #pragma unroll 1
                for (int local_k = 0; local_k < 1; local_k++) {
                    mbarrier_wait(score_empty_addr, _phase_score_empty_0);
                    _phase_score_empty_0 ^= 1;
                    #pragma unroll 1
                    for (int feat_tile = 0; feat_tile < feature_tiles; feat_tile++) {
                        mbarrier_wait(x_full_addr, _phase_x_full_0);
                        _phase_x_full_0 ^= 1;
                        mbarrier_wait(c_full_addr, _phase_c_full_0);
                        _phase_c_full_0 ^= 1;
                        asm volatile("tcgen05.fence::after_thread_sync;");
                        int init_flag = ((feat_tile == 0) ? 1 : 0);
                        int _mma_ss_a_addr_0 = smem_x_addr + 0 * 8192;
                        int _mma_ss_a_lo_0 = make_warp_uniform((_mma_ss_a_addr_0 >> 4) & 0x3FFF);
                        int _mma_ss_b_addr_0 = smem_c_addr + 0 * 32768;
                        int _mma_ss_b_lo_0 = make_warp_uniform((_mma_ss_b_addr_0 >> 4) & 0x3FFF);
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
                    "mov.b32 id, 71304336;\n\t"
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
                    :: "r"(_mma_ss_a_lo_0), "r"(_mma_ss_b_lo_0), "r"(tmem_score_tmem), "r"(((init_flag) ? 0 : 1)));
                        elect_commit(x_empty_addr);
                        elect_commit(c_empty_addr);
                    }
                    elect_commit(score_full_addr);
                }
            }
        }
    }

    // Cleanup
    __syncthreads(); // barrier before TMEM dealloc

    if (warp == 5) {
        asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;" :: "r"(tmem_addr_storage[0]), "r"(256));
        asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;");
    }
}

} // extern "C"

#undef NUM_MAIN_STAGES
#undef SMEM_SMEM_CSQ_OFF
#undef SMEM_SMEM_CSQ_STAGE_BYTES
#undef SMEM_SMEM_CSQ_STRIDE
#undef SMEM_SMEM_C_OFF
#undef SMEM_SMEM_C_STAGE_BYTES
#undef SMEM_SMEM_C_STRIDE
#undef SMEM_SMEM_X_OFF
#undef SMEM_SMEM_X_STAGE_BYTES
#undef SMEM_SMEM_X_STRIDE
#undef SMEM_TOTAL
#undef TMEM_NCOLS
#undef TMEM_SCORE_TMEM_OFFSET
#undef c_empty_addr
#undef c_full_addr
#undef score_empty_addr
#undef score_full_addr
#undef smem_c_addr
#undef smem_csq_addr
#undef smem_x_addr
#undef x_empty_addr
#undef x_full_addr

#define NUM_MAIN_STAGES 1
#define THREADS 256

extern "C" {

__global__ __launch_bounds__(256) void
kernel_flash_kmeans_assign_highd_splitk_reduce_blockn64_g1r4_streamdep_r63_v1(float* __restrict__ partial_scores, int32_t* __restrict__ partial_indices, int32_t* __restrict__ out, int B, int N, int K, int num_n_tiles, int K_slices)
{
    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;


    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    const int warp_id = warp;
    const int lane_id = lane;

    // === Task calls (dependency order) ===
    int row = tid / 4;
    int row_lane = tid % 4;
    int total_point_tiles = B * num_n_tiles;
    #pragma unroll 1
    for (unsigned int point_tile_idx = bid; point_tile_idx < total_point_tiles; point_tile_idx += num_bids) {
        int batch = point_tile_idx / num_n_tiles;
        int n_tile = point_tile_idx % num_n_tiles;
        int global_n = n_tile * 64 + row;
        float best_score = -3.4e+38f;
        int best_idx = 0;
        #pragma unroll 1
        for (int iter_slice = row_lane; iter_slice < K_slices; iter_slice += 4) {
            int partial_offset = (point_tile_idx * K_slices + iter_slice) * 64 + row;
            float score = (float)partial_scores[partial_offset];
            int idx = partial_indices[partial_offset];
            if (score > best_score) {
                best_score = score;
                best_idx = idx;
            }
        }
        float _shfl_xor_0 = __shfl_xor_sync(0xFFFFFFFF, best_score, 1);
        float peer_score = _shfl_xor_0;
        int _shfl_xor_1 = __shfl_xor_sync(0xFFFFFFFF, best_idx, 1);
        int peer_idx = _shfl_xor_1;
        if (peer_score > best_score) {
            best_score = peer_score;
            best_idx = peer_idx;
        }
        peer_score = __shfl_xor_sync(0xFFFFFFFF, best_score, 2);
        peer_idx = __shfl_xor_sync(0xFFFFFFFF, best_idx, 2);
        if (peer_score > best_score) {
            best_score = peer_score;
            best_idx = peer_idx;
        }
        if (row_lane == 0) {
            int out_offset = batch * N + global_n;
            *((int*)(out + out_offset)) = best_idx;
        }
    }
}

} // extern "C"

#undef NUM_MAIN_STAGES
#undef THREADS

#define TMEM_NCOLS 512
#define TMEM_SCORE_TMEM0_OFFSET 0
#define TMEM_SCORE_TMEM1_OFFSET 256
#define NUM_MAIN_STAGES 1
#define SMEM_SMEM_X_OFF 1024
#define SMEM_SMEM_X_STAGE_BYTES 8192
#define SMEM_SMEM_X_STRIDE 8192
#define SMEM_SMEM_C0_OFF 9216
#define SMEM_SMEM_C0_STAGE_BYTES 32768
#define SMEM_SMEM_C0_STRIDE 32768
#define SMEM_SMEM_C1_OFF 41984
#define SMEM_SMEM_C1_STAGE_BYTES 32768
#define SMEM_SMEM_C1_STRIDE 32768
#define SMEM_SMEM_CSQ_OFF 74752
#define SMEM_SMEM_CSQ_STAGE_BYTES 1024
#define SMEM_SMEM_CSQ_STRIDE 1024
#define SMEM_TOTAL 75776

extern "C" {

__global__ __launch_bounds__(192) void
kernel_flash_kmeans_assign_highd_paired_xreuse_dualtmem_producer_r47_v1(float* __restrict__ c_sq, uint64_t* __restrict__ partial_keys, const void* x_tmap, const void* c_tmap, int B, int N, int D, int K, int num_n_tiles, int K_tiles, int K_slices)
{
    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;

    extern __shared__ __align__(1024) char smem_raw[];
    int smem;
    smem = (int)(unsigned long long)__cvta_generic_to_shared(smem_raw);
    const int smem_smem_x = smem + 1024;
    const int smem_smem_c0 = smem + 9216;
    const int smem_smem_c1 = smem + 41984;
    const int smem_smem_csq = smem + 74752;

    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    // Mbarrier init (8 groups, 8 barriers)
    // Mbarriers at smem_raw[0..64)

    if (warp == 0) {
        uint32_t leader = elect_sync();
        // x_full: 1 barriers, init_count=1
        mbarrier_init_pred(smem + 0, 1, leader);
        // x_empty: 1 barriers, init_count=1
        mbarrier_init_pred(smem + 8, 1, leader);
        // c0_full: 1 barriers, init_count=1
        mbarrier_init_pred(smem + 16, 1, leader);
        // c0_empty: 1 barriers, init_count=1
        mbarrier_init_pred(smem + 24, 1, leader);
        // c1_full: 1 barriers, init_count=1
        mbarrier_init_pred(smem + 32, 1, leader);
        // c1_empty: 1 barriers, init_count=1
        mbarrier_init_pred(smem + 40, 1, leader);
        // score_full: 1 barriers, init_count=1
        mbarrier_init_pred(smem + 48, 1, leader);
        // score_empty: 1 barriers, init_count=4
        mbarrier_init_pred(smem + 56, 4, leader);
        asm volatile("fence.mbarrier_init.release.cluster;");
    }

    __syncthreads();

    // TMEM alloc (512 columns, 512 used)
    volatile int* tmem_addr_storage = (volatile int*)(smem_raw + 64);
    if (warp == 5) {
        int _tmem_hold = smem + 64;
        asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;" :: "r"(_tmem_hold), "r"(512) : "memory");
    }

    __syncthreads();
    asm volatile("tcgen05.fence::after_thread_sync;");

    const int warp_id = warp;
    const int lane_id = lane;
    __nv_bfloat16* smem_x = (__nv_bfloat16*)(smem_raw + 1024);
    #define smem_x_addr (smem + 1024)
    __nv_bfloat16* smem_c0 = (__nv_bfloat16*)(smem_raw + 9216);
    #define smem_c0_addr (smem + 9216)
    __nv_bfloat16* smem_c1 = (__nv_bfloat16*)(smem_raw + 41984);
    #define smem_c1_addr (smem + 41984)
    float* smem_csq = (float*)(smem_raw + 74752);
    #define smem_csq_addr (smem + 74752)
    const int mbar_base = smem;
    #define x_full_addr (mbar_base + 0)
    #define x_empty_addr (mbar_base + 8)
    #define c0_full_addr (mbar_base + 16)
    #define c0_empty_addr (mbar_base + 24)
    #define c1_full_addr (mbar_base + 32)
    #define c1_empty_addr (mbar_base + 40)
    #define score_full_addr (mbar_base + 48)
    #define score_empty_addr (mbar_base + 56)
    const int taddr = tmem_addr_storage[0];

    // ---- Role: compute ----
    if (warp <= 3) {
        const int tmem_row_base = (warp % 4) * 32;
        const int my_row = tmem_row_base + (lane / 4);
        { // compute_main
            int total_work = B * num_n_tiles * K_slices;
            int compute_warp = warp;
            int lane_pair = lane % 4;
            int row_origin = compute_warp * 16;
            int row_lane_base = row_origin + lane / 4;
            int row0 = row_lane_base;
            int row1 = row_lane_base + 8;
            int compute_tid = compute_warp * 32 + lane;
            uint32_t _phase_score_full_0 = 0;
            #pragma unroll 1
            for (unsigned int work_idx = bid; work_idx < total_work; work_idx += num_bids) {
                int iter_slice = work_idx % K_slices;
                int point_tile_idx = work_idx / K_slices;
                int batch = point_tile_idx / num_n_tiles;
                int slice_k_start = iter_slice * 2;
                int csq_smem_addr = smem_csq_addr + 0;
                float best0 = -3.4e+38f;
                float best1 = -3.4e+38f;
                int idx0 = slice_k_start * 256;
                int idx1 = idx0;
                int off_k0 = slice_k_start * 256;
                int off_k1 = off_k0 + 256;
                if (compute_tid < 64.0f) {
                    int csq_base0 = compute_tid * 4;
                    float csq_pack0[4];
                    {
                        float4 _v4 = *reinterpret_cast<const float4*>(c_sq + batch * K + off_k0 + csq_base0);
                        csq_pack0[0 + 0] = _v4.x;
                        csq_pack0[0 + 1] = _v4.y;
                        csq_pack0[0 + 2] = _v4.z;
                        csq_pack0[0 + 3] = _v4.w;
                    }
                    float csq_half0[4];
                    csq_half0[0] = 0.5f * csq_pack0[0];
                    csq_half0[1] = 0.5f * csq_pack0[1];
                    csq_half0[2] = 0.5f * csq_pack0[2];
                    csq_half0[3] = 0.5f * csq_pack0[3];
                    asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(csq_smem_addr + csq_base0 * 4), "f"(csq_half0[0]), "f"(csq_half0[1]), "f"(csq_half0[2]), "f"(csq_half0[3]) : "memory");
                }
                asm volatile("barrier.sync 8, 128;");
                mbarrier_wait(score_full_addr, _phase_score_full_0);
                _phase_score_full_0 ^= 1;
                #pragma unroll
                for (int score_base = 0; score_base < 256; score_base += 128) {
                    float scores0[64];
                    asm volatile(
                        "tcgen05.ld.sync.aligned.16x256b.x16.b32"
                        " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63}, [%64];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&scores0[0])), "=r"(*reinterpret_cast<uint32_t*>(&scores0[1])), "=r"(*reinterpret_cast<uint32_t*>(&scores0[2])), "=r"(*reinterpret_cast<uint32_t*>(&scores0[3])), "=r"(*reinterpret_cast<uint32_t*>(&scores0[4])), "=r"(*reinterpret_cast<uint32_t*>(&scores0[5])), "=r"(*reinterpret_cast<uint32_t*>(&scores0[6])), "=r"(*reinterpret_cast<uint32_t*>(&scores0[7])), "=r"(*reinterpret_cast<uint32_t*>(&scores0[8])), "=r"(*reinterpret_cast<uint32_t*>(&scores0[9])), "=r"(*reinterpret_cast<uint32_t*>(&scores0[10])), "=r"(*reinterpret_cast<uint32_t*>(&scores0[11])), "=r"(*reinterpret_cast<uint32_t*>(&scores0[12])), "=r"(*reinterpret_cast<uint32_t*>(&scores0[13])), "=r"(*reinterpret_cast<uint32_t*>(&scores0[14])), "=r"(*reinterpret_cast<uint32_t*>(&scores0[15])), "=r"(*reinterpret_cast<uint32_t*>(&scores0[16])), "=r"(*reinterpret_cast<uint32_t*>(&scores0[17])), "=r"(*reinterpret_cast<uint32_t*>(&scores0[18])), "=r"(*reinterpret_cast<uint32_t*>(&scores0[19])), "=r"(*reinterpret_cast<uint32_t*>(&scores0[20])), "=r"(*reinterpret_cast<uint32_t*>(&scores0[21])), "=r"(*reinterpret_cast<uint32_t*>(&scores0[22])), "=r"(*reinterpret_cast<uint32_t*>(&scores0[23])), "=r"(*reinterpret_cast<uint32_t*>(&scores0[24])), "=r"(*reinterpret_cast<uint32_t*>(&scores0[25])), "=r"(*reinterpret_cast<uint32_t*>(&scores0[26])), "=r"(*reinterpret_cast<uint32_t*>(&scores0[27])), "=r"(*reinterpret_cast<uint32_t*>(&scores0[28])), "=r"(*reinterpret_cast<uint32_t*>(&scores0[29])), "=r"(*reinterpret_cast<uint32_t*>(&scores0[30])), "=r"(*reinterpret_cast<uint32_t*>(&scores0[31])), "=r"(*reinterpret_cast<uint32_t*>(&scores0[32])), "=r"(*reinterpret_cast<uint32_t*>(&scores0[33])), "=r"(*reinterpret_cast<uint32_t*>(&scores0[34])), "=r"(*reinterpret_cast<uint32_t*>(&scores0[35])), "=r"(*reinterpret_cast<uint32_t*>(&scores0[36])), "=r"(*reinterpret_cast<uint32_t*>(&scores0[37])), "=r"(*reinterpret_cast<uint32_t*>(&scores0[38])), "=r"(*reinterpret_cast<uint32_t*>(&scores0[39])), "=r"(*reinterpret_cast<uint32_t*>(&scores0[40])), "=r"(*reinterpret_cast<uint32_t*>(&scores0[41])), "=r"(*reinterpret_cast<uint32_t*>(&scores0[42])), "=r"(*reinterpret_cast<uint32_t*>(&scores0[43])), "=r"(*reinterpret_cast<uint32_t*>(&scores0[44])), "=r"(*reinterpret_cast<uint32_t*>(&scores0[45])), "=r"(*reinterpret_cast<uint32_t*>(&scores0[46])), "=r"(*reinterpret_cast<uint32_t*>(&scores0[47])), "=r"(*reinterpret_cast<uint32_t*>(&scores0[48])), "=r"(*reinterpret_cast<uint32_t*>(&scores0[49])), "=r"(*reinterpret_cast<uint32_t*>(&scores0[50])), "=r"(*reinterpret_cast<uint32_t*>(&scores0[51])), "=r"(*reinterpret_cast<uint32_t*>(&scores0[52])), "=r"(*reinterpret_cast<uint32_t*>(&scores0[53])), "=r"(*reinterpret_cast<uint32_t*>(&scores0[54])), "=r"(*reinterpret_cast<uint32_t*>(&scores0[55])), "=r"(*reinterpret_cast<uint32_t*>(&scores0[56])), "=r"(*reinterpret_cast<uint32_t*>(&scores0[57])), "=r"(*reinterpret_cast<uint32_t*>(&scores0[58])), "=r"(*reinterpret_cast<uint32_t*>(&scores0[59])), "=r"(*reinterpret_cast<uint32_t*>(&scores0[60])), "=r"(*reinterpret_cast<uint32_t*>(&scores0[61])), "=r"(*reinterpret_cast<uint32_t*>(&scores0[62])), "=r"(*reinterpret_cast<uint32_t*>(&scores0[63]))
                        : "r"(taddr + score_base)
                        : "memory");
                    asm volatile("tcgen05.wait::ld.sync.aligned;");
                    #pragma unroll
                    for (int rep = 0; rep < 16.0f; rep++) {
                        int local_reg = rep * 4;
                        int col_base = score_base + rep * 8 + lane_pair * 2;
                        float csq0 = smem_csq[col_base];
                        float csq1 = smem_csq[col_base + 1];
                        float d0 = scores0[local_reg] - csq0;
                        if (d0 > best0) {
                            best0 = d0;
                            idx0 = off_k0 + col_base;
                        }
                        float d1 = scores0[local_reg + 1] - csq1;
                        if (d1 > best0) {
                            best0 = d1;
                            idx0 = off_k0 + col_base + 1;
                        }
                        float d2 = scores0[local_reg + 2] - csq0;
                        if (d2 > best1) {
                            best1 = d2;
                            idx1 = off_k0 + col_base;
                        }
                        float d3 = scores0[local_reg + 3] - csq1;
                        if (d3 > best1) {
                            best1 = d3;
                            idx1 = off_k0 + col_base + 1;
                        }
                    }
                }
                asm volatile("barrier.sync 9, 128;");
                if (compute_tid < 64.0f) {
                    int csq_base1 = compute_tid * 4;
                    float csq_pack1[4];
                    {
                        float4 _v4 = *reinterpret_cast<const float4*>(c_sq + batch * K + off_k1 + csq_base1);
                        csq_pack1[0 + 0] = _v4.x;
                        csq_pack1[0 + 1] = _v4.y;
                        csq_pack1[0 + 2] = _v4.z;
                        csq_pack1[0 + 3] = _v4.w;
                    }
                    float csq_half1[4];
                    csq_half1[0] = 0.5f * csq_pack1[0];
                    csq_half1[1] = 0.5f * csq_pack1[1];
                    csq_half1[2] = 0.5f * csq_pack1[2];
                    csq_half1[3] = 0.5f * csq_pack1[3];
                    asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(csq_smem_addr + csq_base1 * 4), "f"(csq_half1[0]), "f"(csq_half1[1]), "f"(csq_half1[2]), "f"(csq_half1[3]) : "memory");
                }
                asm volatile("barrier.sync 8, 128;");
                #pragma unroll
                for (int score_base = 0; score_base < 256; score_base += 128) {
                    float scores1[64];
                    asm volatile(
                        "tcgen05.ld.sync.aligned.16x256b.x16.b32"
                        " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63}, [%64];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&scores1[0])), "=r"(*reinterpret_cast<uint32_t*>(&scores1[1])), "=r"(*reinterpret_cast<uint32_t*>(&scores1[2])), "=r"(*reinterpret_cast<uint32_t*>(&scores1[3])), "=r"(*reinterpret_cast<uint32_t*>(&scores1[4])), "=r"(*reinterpret_cast<uint32_t*>(&scores1[5])), "=r"(*reinterpret_cast<uint32_t*>(&scores1[6])), "=r"(*reinterpret_cast<uint32_t*>(&scores1[7])), "=r"(*reinterpret_cast<uint32_t*>(&scores1[8])), "=r"(*reinterpret_cast<uint32_t*>(&scores1[9])), "=r"(*reinterpret_cast<uint32_t*>(&scores1[10])), "=r"(*reinterpret_cast<uint32_t*>(&scores1[11])), "=r"(*reinterpret_cast<uint32_t*>(&scores1[12])), "=r"(*reinterpret_cast<uint32_t*>(&scores1[13])), "=r"(*reinterpret_cast<uint32_t*>(&scores1[14])), "=r"(*reinterpret_cast<uint32_t*>(&scores1[15])), "=r"(*reinterpret_cast<uint32_t*>(&scores1[16])), "=r"(*reinterpret_cast<uint32_t*>(&scores1[17])), "=r"(*reinterpret_cast<uint32_t*>(&scores1[18])), "=r"(*reinterpret_cast<uint32_t*>(&scores1[19])), "=r"(*reinterpret_cast<uint32_t*>(&scores1[20])), "=r"(*reinterpret_cast<uint32_t*>(&scores1[21])), "=r"(*reinterpret_cast<uint32_t*>(&scores1[22])), "=r"(*reinterpret_cast<uint32_t*>(&scores1[23])), "=r"(*reinterpret_cast<uint32_t*>(&scores1[24])), "=r"(*reinterpret_cast<uint32_t*>(&scores1[25])), "=r"(*reinterpret_cast<uint32_t*>(&scores1[26])), "=r"(*reinterpret_cast<uint32_t*>(&scores1[27])), "=r"(*reinterpret_cast<uint32_t*>(&scores1[28])), "=r"(*reinterpret_cast<uint32_t*>(&scores1[29])), "=r"(*reinterpret_cast<uint32_t*>(&scores1[30])), "=r"(*reinterpret_cast<uint32_t*>(&scores1[31])), "=r"(*reinterpret_cast<uint32_t*>(&scores1[32])), "=r"(*reinterpret_cast<uint32_t*>(&scores1[33])), "=r"(*reinterpret_cast<uint32_t*>(&scores1[34])), "=r"(*reinterpret_cast<uint32_t*>(&scores1[35])), "=r"(*reinterpret_cast<uint32_t*>(&scores1[36])), "=r"(*reinterpret_cast<uint32_t*>(&scores1[37])), "=r"(*reinterpret_cast<uint32_t*>(&scores1[38])), "=r"(*reinterpret_cast<uint32_t*>(&scores1[39])), "=r"(*reinterpret_cast<uint32_t*>(&scores1[40])), "=r"(*reinterpret_cast<uint32_t*>(&scores1[41])), "=r"(*reinterpret_cast<uint32_t*>(&scores1[42])), "=r"(*reinterpret_cast<uint32_t*>(&scores1[43])), "=r"(*reinterpret_cast<uint32_t*>(&scores1[44])), "=r"(*reinterpret_cast<uint32_t*>(&scores1[45])), "=r"(*reinterpret_cast<uint32_t*>(&scores1[46])), "=r"(*reinterpret_cast<uint32_t*>(&scores1[47])), "=r"(*reinterpret_cast<uint32_t*>(&scores1[48])), "=r"(*reinterpret_cast<uint32_t*>(&scores1[49])), "=r"(*reinterpret_cast<uint32_t*>(&scores1[50])), "=r"(*reinterpret_cast<uint32_t*>(&scores1[51])), "=r"(*reinterpret_cast<uint32_t*>(&scores1[52])), "=r"(*reinterpret_cast<uint32_t*>(&scores1[53])), "=r"(*reinterpret_cast<uint32_t*>(&scores1[54])), "=r"(*reinterpret_cast<uint32_t*>(&scores1[55])), "=r"(*reinterpret_cast<uint32_t*>(&scores1[56])), "=r"(*reinterpret_cast<uint32_t*>(&scores1[57])), "=r"(*reinterpret_cast<uint32_t*>(&scores1[58])), "=r"(*reinterpret_cast<uint32_t*>(&scores1[59])), "=r"(*reinterpret_cast<uint32_t*>(&scores1[60])), "=r"(*reinterpret_cast<uint32_t*>(&scores1[61])), "=r"(*reinterpret_cast<uint32_t*>(&scores1[62])), "=r"(*reinterpret_cast<uint32_t*>(&scores1[63]))
                        : "r"(taddr + 256 + score_base)
                        : "memory");
                    asm volatile("tcgen05.wait::ld.sync.aligned;");
                    #pragma unroll
                    for (int rep = 0; rep < 16.0f; rep++) {
                        int local_reg = rep * 4;
                        int col_base = score_base + rep * 8 + lane_pair * 2;
                        float csq0 = smem_csq[col_base];
                        float csq1 = smem_csq[col_base + 1];
                        float d0 = scores1[local_reg] - csq0;
                        if (d0 > best0) {
                            best0 = d0;
                            idx0 = off_k1 + col_base;
                        }
                        float d1 = scores1[local_reg + 1] - csq1;
                        if (d1 > best0) {
                            best0 = d1;
                            idx0 = off_k1 + col_base + 1;
                        }
                        float d2 = scores1[local_reg + 2] - csq0;
                        if (d2 > best1) {
                            best1 = d2;
                            idx1 = off_k1 + col_base;
                        }
                        float d3 = scores1[local_reg + 3] - csq1;
                        if (d3 > best1) {
                            best1 = d3;
                            idx1 = off_k1 + col_base + 1;
                        }
                    }
                }
                asm volatile("barrier.sync 9, 128;");
                if (elect_sync()) {
                    mbarrier_arrive(score_empty_addr);
                }
                float _shfl_xor_0 = __shfl_xor_sync(0xFFFFFFFF, best0, 1);
                float peer0 = _shfl_xor_0;
                int _shfl_xor_1 = __shfl_xor_sync(0xFFFFFFFF, idx0, 1);
                int peer0_idx = _shfl_xor_1;
                if (peer0 > best0) {
                    best0 = peer0;
                    idx0 = peer0_idx;
                }
                peer0 = __shfl_xor_sync(0xFFFFFFFF, best0, 2);
                peer0_idx = __shfl_xor_sync(0xFFFFFFFF, idx0, 2);
                if (peer0 > best0) {
                    best0 = peer0;
                    idx0 = peer0_idx;
                }
                float _shfl_xor_2 = __shfl_xor_sync(0xFFFFFFFF, best1, 1);
                float peer1 = _shfl_xor_2;
                int _shfl_xor_3 = __shfl_xor_sync(0xFFFFFFFF, idx1, 1);
                int peer1_idx = _shfl_xor_3;
                if (peer1 > best1) {
                    best1 = peer1;
                    idx1 = peer1_idx;
                }
                peer1 = __shfl_xor_sync(0xFFFFFFFF, best1, 2);
                peer1_idx = __shfl_xor_sync(0xFFFFFFFF, idx1, 2);
                if (peer1 > best1) {
                    best1 = peer1;
                    idx1 = peer1_idx;
                }
                if (lane_pair == 0) {
                    unsigned long long shift32 = 32;
                    unsigned long long mask64 = 4294967295;
                    uint32_t _amf_u_0 = __float_as_uint(best0);
                    uint32_t _amf_mask_0 = -int32_t(_amf_u_0 >> 31) | 0x80000000u;
                    unsigned int enc0 = _amf_u_0 ^ _amf_mask_0;
                    unsigned long long key0 = (unsigned long long)enc0 << shift32 | mask64 - (unsigned long long)idx0;
                    int partial_offset0 = work_idx * 64 + row0;
                    *((unsigned long long*)(partial_keys + partial_offset0)) = key0;
                    uint32_t _amf_u_1 = __float_as_uint(best1);
                    uint32_t _amf_mask_1 = -int32_t(_amf_u_1 >> 31) | 0x80000000u;
                    unsigned int enc1 = _amf_u_1 ^ _amf_mask_1;
                    unsigned long long key1 = (unsigned long long)enc1 << shift32 | mask64 - (unsigned long long)idx1;
                    int partial_offset1 = work_idx * 64 + row1;
                    *((unsigned long long*)(partial_keys + partial_offset1)) = key1;
                }
            }
        }
    // ---- Role: load ----
    } else if (warp == 4) {
        { // load_main
            int total_work = B * num_n_tiles * K_slices;
            uint32_t _phase_x_empty_0 = 1;
            uint32_t _phase_c0_empty_0 = 1;
            uint32_t _phase_c1_empty_0 = 1;
            if (elect_sync()) {
                #pragma unroll 1
                for (unsigned int work_idx = bid; work_idx < total_work; work_idx += num_bids) {
                    int iter_slice = work_idx % K_slices;
                    int point_tile_idx = work_idx / K_slices;
                    int batch = point_tile_idx / num_n_tiles;
                    int n_tile = point_tile_idx % num_n_tiles;
                    int off_n = n_tile * 64;
                    int x_row = batch * N + off_n;
                    int slice_k_start = iter_slice * 2;
                    int iter_k0 = slice_k_start;
                    int iter_k1 = slice_k_start + 1;
                    int c_row0 = batch * K + iter_k0 * 256;
                    int c_row1 = batch * K + iter_k1 * 256;
                    #pragma unroll 7
                    for (int feat_tile = 0; feat_tile < 7; feat_tile++) {
                        mbarrier_wait(x_empty_addr, _phase_x_empty_0);
                        _phase_x_empty_0 ^= 1;
                        tma_3d_gmem2smem(smem_x_addr, x_tmap, 0, x_row, feat_tile, x_full_addr);
                        mbarrier_arrive_expect_tx(x_full_addr, 8192);
                        mbarrier_wait(c0_empty_addr, _phase_c0_empty_0);
                        _phase_c0_empty_0 ^= 1;
                        tma_3d_gmem2smem(smem_c0_addr, c_tmap, 0, c_row0, feat_tile, c0_full_addr);
                        mbarrier_arrive_expect_tx(c0_full_addr, 32768);
                        mbarrier_wait(c1_empty_addr, _phase_c1_empty_0);
                        _phase_c1_empty_0 ^= 1;
                        tma_3d_gmem2smem(smem_c1_addr, c_tmap, 0, c_row1, feat_tile, c1_full_addr);
                        mbarrier_arrive_expect_tx(c1_full_addr, 32768);
                    }
                }
            }
        }
    // ---- Role: mma ----
    } else if (warp == 5) {
        const int tmem_score_tmem0 = taddr + TMEM_SCORE_TMEM0_OFFSET;
        const int tmem_score_tmem1 = taddr + TMEM_SCORE_TMEM1_OFFSET;
        { // mma_main
            int total_work = B * num_n_tiles * K_slices;
            uint32_t _phase_score_empty_0 = 1;
            uint32_t _phase_x_full_0 = 0;
            uint32_t _phase_c0_full_0 = 0;
            uint32_t _phase_c1_full_0 = 0;
            #pragma unroll 1
            for (unsigned int work_idx = bid; work_idx < total_work; work_idx += num_bids) {
                mbarrier_wait(score_empty_addr, _phase_score_empty_0);
                _phase_score_empty_0 ^= 1;
                #pragma unroll 7
                for (int feat_tile = 0; feat_tile < 7; feat_tile++) {
                    mbarrier_wait(x_full_addr, _phase_x_full_0);
                    _phase_x_full_0 ^= 1;
                    mbarrier_wait(c0_full_addr, _phase_c0_full_0);
                    _phase_c0_full_0 ^= 1;
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    int init_flag = ((feat_tile == 0) ? 1 : 0);
                    int _mma_ss_a_addr_0 = smem_x_addr + 0 * 8192;
                    int _mma_ss_a_lo_0 = make_warp_uniform((_mma_ss_a_addr_0 >> 4) & 0x3FFF);
                    int _mma_ss_b_addr_0 = smem_c0_addr + 0 * 32768;
                    int _mma_ss_b_lo_0 = make_warp_uniform((_mma_ss_b_addr_0 >> 4) & 0x3FFF);
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
                    "mov.b32 id, 71304336;\n\t"
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
                    :: "r"(_mma_ss_a_lo_0), "r"(_mma_ss_b_lo_0), "r"(tmem_score_tmem0), "r"(((init_flag) ? 0 : 1)));
                    elect_commit(c0_empty_addr);
                    mbarrier_wait(c1_full_addr, _phase_c1_full_0);
                    _phase_c1_full_0 ^= 1;
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    int _mma_ss_a_addr_1 = smem_x_addr + 0 * 8192;
                    int _mma_ss_a_lo_1 = make_warp_uniform((_mma_ss_a_addr_1 >> 4) & 0x3FFF);
                    int _mma_ss_b_addr_1 = smem_c1_addr + 0 * 32768;
                    int _mma_ss_b_lo_1 = make_warp_uniform((_mma_ss_b_addr_1 >> 4) & 0x3FFF);
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
                    "mov.b32 id, 71304336;\n\t"
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
                    :: "r"(_mma_ss_a_lo_1), "r"(_mma_ss_b_lo_1), "r"(tmem_score_tmem1), "r"(((init_flag) ? 0 : 1)));
                    elect_commit(c1_empty_addr);
                    elect_commit(x_empty_addr);
                }
                elect_commit(score_full_addr);
            }
        }
    }

    // Cleanup
    __syncthreads(); // barrier before TMEM dealloc

    if (warp == 5) {
        asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;" :: "r"(tmem_addr_storage[0]), "r"(512));
        asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;");
    }
}

} // extern "C"

#undef NUM_MAIN_STAGES
#undef SMEM_SMEM_C0_OFF
#undef SMEM_SMEM_C0_STAGE_BYTES
#undef SMEM_SMEM_C0_STRIDE
#undef SMEM_SMEM_C1_OFF
#undef SMEM_SMEM_C1_STAGE_BYTES
#undef SMEM_SMEM_C1_STRIDE
#undef SMEM_SMEM_CSQ_OFF
#undef SMEM_SMEM_CSQ_STAGE_BYTES
#undef SMEM_SMEM_CSQ_STRIDE
#undef SMEM_SMEM_X_OFF
#undef SMEM_SMEM_X_STAGE_BYTES
#undef SMEM_SMEM_X_STRIDE
#undef SMEM_TOTAL
#undef TMEM_NCOLS
#undef TMEM_SCORE_TMEM0_OFFSET
#undef TMEM_SCORE_TMEM1_OFFSET
#undef c0_empty_addr
#undef c0_full_addr
#undef c1_empty_addr
#undef c1_full_addr
#undef score_empty_addr
#undef score_full_addr
#undef smem_c0_addr
#undef smem_c1_addr
#undef smem_csq_addr
#undef smem_x_addr
#undef x_empty_addr
#undef x_full_addr

#define NUM_MAIN_STAGES 1
#define THREADS 64

extern "C" {

__global__ __launch_bounds__(64) void
kernel_flash_kmeans_assign_highd_paired_ownerreduce_r39_reduce1_unroll_v1(uint64_t* __restrict__ partial_keys, int32_t* __restrict__ out, int B, int N, int K, int num_n_tiles, int K_slices)
{
    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;


    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    const int warp_id = warp;
    const int lane_id = lane;

    // === Task calls (dependency order) ===
    int row = tid;
    int total_point_tiles = B * num_n_tiles;
    #pragma unroll 1
    for (unsigned int point_tile_idx = bid; point_tile_idx < total_point_tiles; point_tile_idx += num_bids) {
        int batch = point_tile_idx / num_n_tiles;
        int n_tile = point_tile_idx % num_n_tiles;
        int global_n = n_tile * 64 + row;
        unsigned long long best_key = 0;
        #pragma unroll 8
        for (int iter_slice = 0; iter_slice < 8; iter_slice++) {
            int partial_offset = (point_tile_idx * K_slices + iter_slice) * 64 + row;
            unsigned long long key = partial_keys[partial_offset];
            if (key > best_key) {
                best_key = key;
            }
        }
        unsigned long long mask64 = 4294967295;
        unsigned long long inv_idx = best_key & mask64;
        unsigned long long idx_u64 = mask64 - inv_idx;
        int idx = (int)idx_u64;
        int out_offset = batch * N + global_n;
        *((int*)(out + out_offset)) = idx;
    }
}

} // extern "C"

#undef NUM_MAIN_STAGES
#undef THREADS

#define TMEM_NCOLS 256
#define TMEM_SCORE_TMEM_OFFSET 0
#define NUM_MAIN_STAGES 1
#define SMEM_SMEM_X_OFF 1024
#define SMEM_SMEM_X_STAGE_BYTES 8192
#define SMEM_SMEM_X_STRIDE 8192
#define SMEM_SMEM_C_OFF 9216
#define SMEM_SMEM_C_STAGE_BYTES 32768
#define SMEM_SMEM_C_STRIDE 32768
#define SMEM_SMEM_CSQ_OFF 41984
#define SMEM_SMEM_CSQ_STAGE_BYTES 1024
#define SMEM_SMEM_CSQ_STRIDE 1024
#define SMEM_TOTAL 43008

extern "C" {

__global__ __launch_bounds__(192) void
kernel_flash_kmeans_assign_highd_paired_packedpartial_producer_7b3c_v1(float* __restrict__ c_sq, uint64_t* __restrict__ partial_keys, const void* x_tmap, const void* c_tmap, int B, int N, int D, int K, int num_n_tiles, int K_tiles, int K_slices)
{
    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;

    extern __shared__ __align__(1024) char smem_raw[];
    int smem;
    smem = (int)(unsigned long long)__cvta_generic_to_shared(smem_raw);
    const int smem_smem_x = smem + 1024;
    const int smem_smem_c = smem + 9216;
    const int smem_smem_csq = smem + 41984;

    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    // Mbarrier init (6 groups, 6 barriers)
    // Mbarriers at smem_raw[0..48)

    if (warp == 0) {
        uint32_t leader = elect_sync();
        // x_full: 1 barriers, init_count=1
        mbarrier_init_pred(smem + 0, 1, leader);
        // x_empty: 1 barriers, init_count=1
        mbarrier_init_pred(smem + 8, 1, leader);
        // c_full: 1 barriers, init_count=1
        mbarrier_init_pred(smem + 16, 1, leader);
        // c_empty: 1 barriers, init_count=1
        mbarrier_init_pred(smem + 24, 1, leader);
        // score_full: 1 barriers, init_count=1
        mbarrier_init_pred(smem + 32, 1, leader);
        // score_empty: 1 barriers, init_count=4
        mbarrier_init_pred(smem + 40, 4, leader);
        asm volatile("fence.mbarrier_init.release.cluster;");
    }

    __syncthreads();

    // TMEM alloc (256 columns, 256 used)
    volatile int* tmem_addr_storage = (volatile int*)(smem_raw + 48);
    if (warp == 5) {
        int _tmem_hold = smem + 48;
        asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;" :: "r"(_tmem_hold), "r"(256) : "memory");
    }

    __syncthreads();
    asm volatile("tcgen05.fence::after_thread_sync;");

    const int warp_id = warp;
    const int lane_id = lane;
    __nv_bfloat16* smem_x = (__nv_bfloat16*)(smem_raw + 1024);
    #define smem_x_addr (smem + 1024)
    __nv_bfloat16* smem_c = (__nv_bfloat16*)(smem_raw + 9216);
    #define smem_c_addr (smem + 9216)
    float* smem_csq = (float*)(smem_raw + 41984);
    #define smem_csq_addr (smem + 41984)
    const int mbar_base = smem;
    #define x_full_addr (mbar_base + 0)
    #define x_empty_addr (mbar_base + 8)
    #define c_full_addr (mbar_base + 16)
    #define c_empty_addr (mbar_base + 24)
    #define score_full_addr (mbar_base + 32)
    #define score_empty_addr (mbar_base + 40)
    const int taddr = tmem_addr_storage[0];

    // ---- Role: compute ----
    if (warp <= 3) {
        const int tmem_row_base = (warp % 4) * 32;
        const int my_row = tmem_row_base + (lane / 4);
        { // compute_main
            int total_work = B * num_n_tiles * K_slices;
            int compute_warp = warp;
            int lane_pair = lane % 4;
            int row_origin = compute_warp * 16;
            int row_lane_base = row_origin + lane / 4;
            int row0 = row_lane_base;
            int row1 = row_lane_base + 8;
            int compute_tid = compute_warp * 32 + lane;
            uint32_t _phase_score_full_0 = 0;
            #pragma unroll 1
            for (unsigned int work_idx = bid; work_idx < total_work; work_idx += num_bids) {
                int iter_slice = work_idx % K_slices;
                int point_tile_idx = work_idx / K_slices;
                int batch = point_tile_idx / num_n_tiles;
                int slice_k_start = iter_slice * 2;
                int csq_smem_addr = smem_csq_addr + 0;
                float best0 = -3.4e+38f;
                float best1 = -3.4e+38f;
                int idx0 = slice_k_start * 256;
                int idx1 = idx0;
                #pragma unroll 1
                for (int local_k = 0; local_k < 2; local_k++) {
                    int iter_k = slice_k_start + local_k;
                    int off_k = iter_k * 256;
                    if (compute_tid < 64.0f) {
                        int csq_base = compute_tid * 4;
                        float csq_pack[4];
                        {
                            float4 _v4 = *reinterpret_cast<const float4*>(c_sq + batch * K + off_k + csq_base);
                            csq_pack[0 + 0] = _v4.x;
                            csq_pack[0 + 1] = _v4.y;
                            csq_pack[0 + 2] = _v4.z;
                            csq_pack[0 + 3] = _v4.w;
                        }
                        float csq_half[4];
                        csq_half[0] = 0.5f * csq_pack[0];
                        csq_half[1] = 0.5f * csq_pack[1];
                        csq_half[2] = 0.5f * csq_pack[2];
                        csq_half[3] = 0.5f * csq_pack[3];
                        asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(csq_smem_addr + csq_base * 4), "f"(csq_half[0]), "f"(csq_half[1]), "f"(csq_half[2]), "f"(csq_half[3]) : "memory");
                    }
                    asm volatile("barrier.sync 8, 128;");
                    mbarrier_wait(score_full_addr, _phase_score_full_0);
                    _phase_score_full_0 ^= 1;
                    #pragma unroll
                    for (int score_base = 0; score_base < 256; score_base += 128) {
                        float scores[64];
                        asm volatile(
                            "tcgen05.ld.sync.aligned.16x256b.x16.b32"
                            " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63}, [%64];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&scores[0])), "=r"(*reinterpret_cast<uint32_t*>(&scores[1])), "=r"(*reinterpret_cast<uint32_t*>(&scores[2])), "=r"(*reinterpret_cast<uint32_t*>(&scores[3])), "=r"(*reinterpret_cast<uint32_t*>(&scores[4])), "=r"(*reinterpret_cast<uint32_t*>(&scores[5])), "=r"(*reinterpret_cast<uint32_t*>(&scores[6])), "=r"(*reinterpret_cast<uint32_t*>(&scores[7])), "=r"(*reinterpret_cast<uint32_t*>(&scores[8])), "=r"(*reinterpret_cast<uint32_t*>(&scores[9])), "=r"(*reinterpret_cast<uint32_t*>(&scores[10])), "=r"(*reinterpret_cast<uint32_t*>(&scores[11])), "=r"(*reinterpret_cast<uint32_t*>(&scores[12])), "=r"(*reinterpret_cast<uint32_t*>(&scores[13])), "=r"(*reinterpret_cast<uint32_t*>(&scores[14])), "=r"(*reinterpret_cast<uint32_t*>(&scores[15])), "=r"(*reinterpret_cast<uint32_t*>(&scores[16])), "=r"(*reinterpret_cast<uint32_t*>(&scores[17])), "=r"(*reinterpret_cast<uint32_t*>(&scores[18])), "=r"(*reinterpret_cast<uint32_t*>(&scores[19])), "=r"(*reinterpret_cast<uint32_t*>(&scores[20])), "=r"(*reinterpret_cast<uint32_t*>(&scores[21])), "=r"(*reinterpret_cast<uint32_t*>(&scores[22])), "=r"(*reinterpret_cast<uint32_t*>(&scores[23])), "=r"(*reinterpret_cast<uint32_t*>(&scores[24])), "=r"(*reinterpret_cast<uint32_t*>(&scores[25])), "=r"(*reinterpret_cast<uint32_t*>(&scores[26])), "=r"(*reinterpret_cast<uint32_t*>(&scores[27])), "=r"(*reinterpret_cast<uint32_t*>(&scores[28])), "=r"(*reinterpret_cast<uint32_t*>(&scores[29])), "=r"(*reinterpret_cast<uint32_t*>(&scores[30])), "=r"(*reinterpret_cast<uint32_t*>(&scores[31])), "=r"(*reinterpret_cast<uint32_t*>(&scores[32])), "=r"(*reinterpret_cast<uint32_t*>(&scores[33])), "=r"(*reinterpret_cast<uint32_t*>(&scores[34])), "=r"(*reinterpret_cast<uint32_t*>(&scores[35])), "=r"(*reinterpret_cast<uint32_t*>(&scores[36])), "=r"(*reinterpret_cast<uint32_t*>(&scores[37])), "=r"(*reinterpret_cast<uint32_t*>(&scores[38])), "=r"(*reinterpret_cast<uint32_t*>(&scores[39])), "=r"(*reinterpret_cast<uint32_t*>(&scores[40])), "=r"(*reinterpret_cast<uint32_t*>(&scores[41])), "=r"(*reinterpret_cast<uint32_t*>(&scores[42])), "=r"(*reinterpret_cast<uint32_t*>(&scores[43])), "=r"(*reinterpret_cast<uint32_t*>(&scores[44])), "=r"(*reinterpret_cast<uint32_t*>(&scores[45])), "=r"(*reinterpret_cast<uint32_t*>(&scores[46])), "=r"(*reinterpret_cast<uint32_t*>(&scores[47])), "=r"(*reinterpret_cast<uint32_t*>(&scores[48])), "=r"(*reinterpret_cast<uint32_t*>(&scores[49])), "=r"(*reinterpret_cast<uint32_t*>(&scores[50])), "=r"(*reinterpret_cast<uint32_t*>(&scores[51])), "=r"(*reinterpret_cast<uint32_t*>(&scores[52])), "=r"(*reinterpret_cast<uint32_t*>(&scores[53])), "=r"(*reinterpret_cast<uint32_t*>(&scores[54])), "=r"(*reinterpret_cast<uint32_t*>(&scores[55])), "=r"(*reinterpret_cast<uint32_t*>(&scores[56])), "=r"(*reinterpret_cast<uint32_t*>(&scores[57])), "=r"(*reinterpret_cast<uint32_t*>(&scores[58])), "=r"(*reinterpret_cast<uint32_t*>(&scores[59])), "=r"(*reinterpret_cast<uint32_t*>(&scores[60])), "=r"(*reinterpret_cast<uint32_t*>(&scores[61])), "=r"(*reinterpret_cast<uint32_t*>(&scores[62])), "=r"(*reinterpret_cast<uint32_t*>(&scores[63]))
                            : "r"(taddr + score_base)
                            : "memory");
                        asm volatile("tcgen05.wait::ld.sync.aligned;");
                        #pragma unroll
                        for (int rep = 0; rep < 16.0f; rep++) {
                            int local_reg = rep * 4;
                            int col_base = score_base + rep * 8 + lane_pair * 2;
                            float csq0 = smem_csq[col_base];
                            float csq1 = smem_csq[col_base + 1];
                            float d0 = scores[local_reg] - csq0;
                            if (d0 > best0) {
                                best0 = d0;
                                idx0 = off_k + col_base;
                            }
                            float d1 = scores[local_reg + 1] - csq1;
                            if (d1 > best0) {
                                best0 = d1;
                                idx0 = off_k + col_base + 1;
                            }
                            float d2 = scores[local_reg + 2] - csq0;
                            if (d2 > best1) {
                                best1 = d2;
                                idx1 = off_k + col_base;
                            }
                            float d3 = scores[local_reg + 3] - csq1;
                            if (d3 > best1) {
                                best1 = d3;
                                idx1 = off_k + col_base + 1;
                            }
                        }
                    }
                    asm volatile("barrier.sync 9, 128;");
                    if (elect_sync()) {
                        mbarrier_arrive(score_empty_addr);
                    }
                }
                float _shfl_xor_0 = __shfl_xor_sync(0xFFFFFFFF, best0, 1);
                float peer0 = _shfl_xor_0;
                int _shfl_xor_1 = __shfl_xor_sync(0xFFFFFFFF, idx0, 1);
                int peer0_idx = _shfl_xor_1;
                if (peer0 > best0) {
                    best0 = peer0;
                    idx0 = peer0_idx;
                }
                peer0 = __shfl_xor_sync(0xFFFFFFFF, best0, 2);
                peer0_idx = __shfl_xor_sync(0xFFFFFFFF, idx0, 2);
                if (peer0 > best0) {
                    best0 = peer0;
                    idx0 = peer0_idx;
                }
                float _shfl_xor_2 = __shfl_xor_sync(0xFFFFFFFF, best1, 1);
                float peer1 = _shfl_xor_2;
                int _shfl_xor_3 = __shfl_xor_sync(0xFFFFFFFF, idx1, 1);
                int peer1_idx = _shfl_xor_3;
                if (peer1 > best1) {
                    best1 = peer1;
                    idx1 = peer1_idx;
                }
                peer1 = __shfl_xor_sync(0xFFFFFFFF, best1, 2);
                peer1_idx = __shfl_xor_sync(0xFFFFFFFF, idx1, 2);
                if (peer1 > best1) {
                    best1 = peer1;
                    idx1 = peer1_idx;
                }
                if (lane_pair == 0) {
                    unsigned long long shift32 = 32;
                    unsigned long long mask64 = 4294967295;
                    uint32_t _amf_u_0 = __float_as_uint(best0);
                    uint32_t _amf_mask_0 = -int32_t(_amf_u_0 >> 31) | 0x80000000u;
                    unsigned int enc0 = _amf_u_0 ^ _amf_mask_0;
                    unsigned long long key0 = (unsigned long long)enc0 << shift32 | mask64 - (unsigned long long)idx0;
                    int partial_offset0 = work_idx * 64 + row0;
                    *((unsigned long long*)(partial_keys + partial_offset0)) = key0;
                    uint32_t _amf_u_1 = __float_as_uint(best1);
                    uint32_t _amf_mask_1 = -int32_t(_amf_u_1 >> 31) | 0x80000000u;
                    unsigned int enc1 = _amf_u_1 ^ _amf_mask_1;
                    unsigned long long key1 = (unsigned long long)enc1 << shift32 | mask64 - (unsigned long long)idx1;
                    int partial_offset1 = work_idx * 64 + row1;
                    *((unsigned long long*)(partial_keys + partial_offset1)) = key1;
                }
            }
        }
    // ---- Role: load ----
    } else if (warp == 4) {
        { // load_main
            int total_work = B * num_n_tiles * K_slices;
            int feature_tiles = D / 64;
            uint32_t _phase_x_empty_0 = 1;
            uint32_t _phase_c_empty_0 = 1;
            if (elect_sync()) {
                #pragma unroll 1
                for (unsigned int work_idx = bid; work_idx < total_work; work_idx += num_bids) {
                    int iter_slice = work_idx % K_slices;
                    int point_tile_idx = work_idx / K_slices;
                    int batch = point_tile_idx / num_n_tiles;
                    int n_tile = point_tile_idx % num_n_tiles;
                    int off_n = n_tile * 64;
                    int x_row = batch * N + off_n;
                    int slice_k_start = iter_slice * 2;
                    #pragma unroll 1
                    for (int local_k = 0; local_k < 2; local_k++) {
                        int iter_k = slice_k_start + local_k;
                        int off_k = iter_k * 256;
                        int c_row = batch * K + off_k;
                        #pragma unroll 1
                        for (int feat_tile = 0; feat_tile < feature_tiles; feat_tile++) {
                            mbarrier_wait(x_empty_addr, _phase_x_empty_0);
                            _phase_x_empty_0 ^= 1;
                            tma_3d_gmem2smem(smem_x_addr, x_tmap, 0, x_row, feat_tile, x_full_addr);
                            mbarrier_arrive_expect_tx(x_full_addr, 8192);
                            mbarrier_wait(c_empty_addr, _phase_c_empty_0);
                            _phase_c_empty_0 ^= 1;
                            tma_3d_gmem2smem(smem_c_addr, c_tmap, 0, c_row, feat_tile, c_full_addr);
                            mbarrier_arrive_expect_tx(c_full_addr, 32768);
                        }
                    }
                }
            }
        }
    // ---- Role: mma ----
    } else if (warp == 5) {
        const int tmem_score_tmem = taddr + TMEM_SCORE_TMEM_OFFSET;
        { // mma_main
            int total_work = B * num_n_tiles * K_slices;
            int feature_tiles = D / 64;
            uint32_t _phase_score_empty_0 = 1;
            uint32_t _phase_x_full_0 = 0;
            uint32_t _phase_c_full_0 = 0;
            #pragma unroll 1
            for (unsigned int work_idx = bid; work_idx < total_work; work_idx += num_bids) {
                int iter_slice = work_idx % K_slices;
                int slice_k_start = iter_slice * 2;
                #pragma unroll 1
                for (int local_k = 0; local_k < 2; local_k++) {
                    mbarrier_wait(score_empty_addr, _phase_score_empty_0);
                    _phase_score_empty_0 ^= 1;
                    #pragma unroll 1
                    for (int feat_tile = 0; feat_tile < feature_tiles; feat_tile++) {
                        mbarrier_wait(x_full_addr, _phase_x_full_0);
                        _phase_x_full_0 ^= 1;
                        mbarrier_wait(c_full_addr, _phase_c_full_0);
                        _phase_c_full_0 ^= 1;
                        asm volatile("tcgen05.fence::after_thread_sync;");
                        int init_flag = ((feat_tile == 0) ? 1 : 0);
                        int _mma_ss_a_addr_0 = smem_x_addr + 0 * 8192;
                        int _mma_ss_a_lo_0 = make_warp_uniform((_mma_ss_a_addr_0 >> 4) & 0x3FFF);
                        int _mma_ss_b_addr_0 = smem_c_addr + 0 * 32768;
                        int _mma_ss_b_lo_0 = make_warp_uniform((_mma_ss_b_addr_0 >> 4) & 0x3FFF);
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
                    "mov.b32 id, 71304336;\n\t"
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
                    :: "r"(_mma_ss_a_lo_0), "r"(_mma_ss_b_lo_0), "r"(tmem_score_tmem), "r"(((init_flag) ? 0 : 1)));
                        elect_commit(x_empty_addr);
                        elect_commit(c_empty_addr);
                    }
                    elect_commit(score_full_addr);
                }
            }
        }
    }

    // Cleanup
    __syncthreads(); // barrier before TMEM dealloc

    if (warp == 5) {
        asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;" :: "r"(tmem_addr_storage[0]), "r"(256));
        asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;");
    }
}

} // extern "C"

#undef NUM_MAIN_STAGES
#undef SMEM_SMEM_CSQ_OFF
#undef SMEM_SMEM_CSQ_STAGE_BYTES
#undef SMEM_SMEM_CSQ_STRIDE
#undef SMEM_SMEM_C_OFF
#undef SMEM_SMEM_C_STAGE_BYTES
#undef SMEM_SMEM_C_STRIDE
#undef SMEM_SMEM_X_OFF
#undef SMEM_SMEM_X_STAGE_BYTES
#undef SMEM_SMEM_X_STRIDE
#undef SMEM_TOTAL
#undef TMEM_NCOLS
#undef TMEM_SCORE_TMEM_OFFSET
#undef c_empty_addr
#undef c_full_addr
#undef score_empty_addr
#undef score_full_addr
#undef smem_c_addr
#undef smem_csq_addr
#undef smem_x_addr
#undef x_empty_addr
#undef x_full_addr

#define NUM_MAIN_STAGES 1
#define THREADS 128

extern "C" {

__global__ __launch_bounds__(128) void
kernel_flash_kmeans_assign_highd_paired_packedpartial_reduce_r2_7b3c_v1(uint64_t* __restrict__ partial_keys, int32_t* __restrict__ out, int B, int N, int K, int num_n_tiles, int K_slices)
{
    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;


    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    const int warp_id = warp;
    const int lane_id = lane;

    // === Task calls (dependency order) ===
    int row = tid / 2;
    int row_lane = tid % 2;
    int total_point_tiles = B * num_n_tiles;
    #pragma unroll 1
    for (unsigned int point_tile_idx = bid; point_tile_idx < total_point_tiles; point_tile_idx += num_bids) {
        int batch = point_tile_idx / num_n_tiles;
        int n_tile = point_tile_idx % num_n_tiles;
        int global_n = n_tile * 64 + row;
        unsigned long long best_key = 0;
        #pragma unroll 1
        for (int iter_slice = row_lane; iter_slice < K_slices; iter_slice += 2) {
            int partial_offset = (point_tile_idx * K_slices + iter_slice) * 64 + row;
            unsigned long long key = partial_keys[partial_offset];
            if (key > best_key) {
                best_key = key;
            }
        }
        unsigned long long _shfl_xor_0 = __shfl_xor_sync(0xFFFFFFFF, best_key, 1);
        unsigned long long peer_key = _shfl_xor_0;
        if (peer_key > best_key) {
            best_key = peer_key;
        }
        if (row_lane == 0) {
            unsigned long long mask64 = 4294967295;
            unsigned long long inv_idx = best_key & mask64;
            unsigned long long idx_u64 = mask64 - inv_idx;
            int idx = (int)idx_u64;
            int out_offset = batch * N + global_n;
            *((int*)(out + out_offset)) = idx;
        }
    }
}

} // extern "C"

#undef NUM_MAIN_STAGES
#undef THREADS

