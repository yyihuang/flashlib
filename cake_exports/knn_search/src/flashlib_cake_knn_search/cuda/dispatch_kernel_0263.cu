typedef unsigned char      uint8_t;
typedef unsigned short     uint16_t;
typedef unsigned int       uint32_t;
typedef unsigned long long uint64_t;
typedef signed int         int32_t;
typedef short int          int16_t;

#include <cuda_bf16.h>

__device__ __forceinline__ int make_warp_uniform(int x) {
    int result;
    asm volatile("shfl.sync.idx.b32 %0, %1, 0, 0x1F, 0xFFFFFFFF;"
                 : "=r"(result) : "r"(x));
    return result;
}

#define NUM_MAIN_STAGES 1
#define SMEM_ROW_DIST_OFF 0
#define SMEM_ROW_DIST_STAGE_BYTES 128
#define SMEM_ROW_DIST_STRIDE 128
#define SMEM_ROW_IDX_OFF 128
#define SMEM_ROW_IDX_STAGE_BYTES 128
#define SMEM_ROW_IDX_STRIDE 128
#define SMEM_TOTAL 256
#define THREADS 32
#define K_MAX_ 10
#define M_MAX_ 32

#include <math_constants.h>
#define LOOM_INF CUDART_INF_F

extern "C" {

__global__ __launch_bounds__(32) void
kernel_knn_search_lowd_ivf_direct_dispatch0610_r2_v1(__nv_bfloat16* __restrict__ queries, __nv_bfloat16* __restrict__ database, float* __restrict__ out_distances, int32_t* __restrict__ out_indices, int B, int Q, int M, int K, int D)
{
    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;

    extern __shared__ __align__(1024) char smem_raw[];
    int smem;
    smem = (int)(unsigned long long)__cvta_generic_to_shared(smem_raw);
    const int smem_row_dist = smem + 0;
    const int smem_row_idx = smem + 128;

    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    const int warp_id = warp;
    const int lane_id = lane;
    float* row_dist = (float*)(smem_raw + 0);
    #define row_dist_addr (smem + 0)
    int* row_idx = (int*)(smem_raw + 128);
    #define row_idx_addr (smem + 128)

    // === Task calls (dependency order) ===
    int q_linear = bid;
    int batch_id = q_linear / Q;
    int q_row = q_linear - batch_id * Q;
    if (tid < M_MAX_) {
        float dist = LOOM_INF;
        int idx = -1;
        if (batch_id < B) {
            if (tid < M) {
                unsigned long long q_base = (unsigned long long)((batch_id * Q + q_row) * D);
                unsigned long long db_base = (unsigned long long)((batch_id * M + tid) * D);
                dist = 0.0f;
                #pragma unroll 1
                for (int d_vec = 0; d_vec < D; d_vec += 8) {
                    float q_val[8];
                    {
                        const uint4* _vptr_0 = reinterpret_cast<const uint4*>(queries + q_base + (unsigned long long)d_vec);
                        uint4 _vld_0[1];
                        #pragma unroll
                        for (int _blk = 0; _blk < 1; _blk++) {
                            _vld_0[_blk] = _vptr_0[_blk];
                            __nv_bfloat16* _velems_0 = reinterpret_cast<__nv_bfloat16*>(&_vld_0[_blk]);
                            #pragma unroll
                            for (int _j = 0; _j < 8; _j++)
                                q_val[0 + _blk * 8 + _j] = __bfloat162float(_velems_0[_j]);
                        }
                    }
                    float db_val[8];
                    {
                        const uint4* _vptr_1 = reinterpret_cast<const uint4*>(database + db_base + (unsigned long long)d_vec);
                        uint4 _vld_1[1];
                        #pragma unroll
                        for (int _blk = 0; _blk < 1; _blk++) {
                            _vld_1[_blk] = _vptr_1[_blk];
                            __nv_bfloat16* _velems_1 = reinterpret_cast<__nv_bfloat16*>(&_vld_1[_blk]);
                            #pragma unroll
                            for (int _j = 0; _j < 8; _j++)
                                db_val[0 + _blk * 8 + _j] = __bfloat162float(_velems_1[_j]);
                        }
                    }
                    #pragma unroll
                    for (int jj = 0; jj < 8; jj++) {
                        float diff = q_val[jj] - db_val[jj];
                        dist += diff * diff;
                    }
                }
                idx = tid;
            }
        }
        row_dist[tid] = dist;
        row_idx[tid] = idx;
    }
    __syncthreads();
    if (batch_id < B) {
        if (tid == 0) {
            float final_d[10];
            int final_i[10];
            #pragma unroll
            for (int kk = 0; kk < K_MAX_; kk++) {
                final_d[kk] = LOOM_INF;
                final_i[kk] = -1;
            }
            #pragma unroll
            for (int src_m = 0; src_m < M_MAX_; src_m++) {
                int cand_i = row_idx[src_m];
                if (cand_i >= 0) {
                    float cand_d = row_dist[src_m];
                    if (cand_d < final_d[K_MAX_ - 1]) {
                        float carry_d = cand_d;
                        int carry_i = cand_i;
                        #pragma unroll
                        for (int kk = 0; kk < K_MAX_; kk++) {
                            float old_d = final_d[kk];
                            int old_i = final_i[kk];
                            int take = ((carry_d < old_d) ? 1 : 0);
                            if (carry_d == old_d) {
                                if (carry_i >= 0) {
                                    if (old_i < 0) {
                                        take = 1;
                                    } else if (carry_i < old_i) {
                                        take = 1;
                                    }
                                }
                            }
                            final_d[kk] = ((take != 0) ? carry_d : old_d);
                            final_i[kk] = ((take != 0) ? carry_i : old_i);
                            carry_d = ((take != 0) ? old_d : carry_d);
                            carry_i = ((take != 0) ? old_i : carry_i);
                        }
                    }
                }
            }
            unsigned long long out_base = (unsigned long long)((batch_id * Q + q_row) * K);
            #pragma unroll
            for (int kk = 0; kk < K_MAX_; kk++) {
                if (kk < K) {
                    out_distances[out_base + kk] = final_d[kk];
                    out_indices[out_base + kk] = final_i[kk];
                }
            }
        }
    }
}

} // extern "C"

