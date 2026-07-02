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
#define SMEM_LOCAL_DIST_OFF 0
#define SMEM_LOCAL_DIST_STAGE_BYTES 10240
#define SMEM_LOCAL_DIST_STRIDE 10240
#define SMEM_LOCAL_IDX_OFF 10240
#define SMEM_LOCAL_IDX_STAGE_BYTES 10240
#define SMEM_LOCAL_IDX_STRIDE 10240
#define SMEM_TOTAL 20608
#define THREADS 256
#define K_MAX_ 10
#define LOCAL_LIST_CAP_ 10
#define ROWS_PER_THREAD_ 256

#include <math_constants.h>
#define LOOM_INF CUDART_INF_F

extern "C" {

__global__ __launch_bounds__(256) void
kernel_knn_search_dynamic_lowd_d1_serialmerge_0624_3676_v1(__nv_bfloat16* __restrict__ queries, __nv_bfloat16* __restrict__ database, float* __restrict__ out_distances, int32_t* __restrict__ out_indices, int B, int Q, int M, int K)
{
    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;

    extern __shared__ __align__(1024) char smem_raw[];
    int smem;
    smem = (int)(unsigned long long)__cvta_generic_to_shared(smem_raw);
    const int smem_local_dist = smem + 0;
    const int smem_local_idx = smem + 10240;

    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    const int warp_id = warp;
    const int lane_id = lane;
    float* local_dist = (float*)(smem_raw + 0);
    #define local_dist_addr (smem + 0)
    int* local_idx = (int*)(smem_raw + 10240);
    #define local_idx_addr (smem + 10240)

    // === Task calls (dependency order) ===
    int q_linear = bid;
    int batch_id = q_linear / Q;
    int q_row = q_linear - batch_id * Q;
    float best_d[10];
    int best_i[10];
    #pragma unroll
    for (int kk = 0; kk < LOCAL_LIST_CAP_; kk++) {
        best_d[kk] = LOOM_INF;
        best_i[kk] = -1;
    }
    if (batch_id < B) {
        unsigned long long q_base = (unsigned long long)(batch_id * Q + q_row);
        float q0_v[1];
        {
            __nv_bfloat16 _bf16_0 = *reinterpret_cast<const __nv_bfloat16*>(queries + q_base + 0);
            q0_v[0] = __bfloat162float(_bf16_0);
        }
        float q0 = q0_v[0];
        #pragma unroll
        for (int local_slot = 0; local_slot < ROWS_PER_THREAD_; local_slot++) {
            int m_row = tid + local_slot * 256;
            if (m_row < M) {
                unsigned long long db_base = (unsigned long long)(batch_id * M + m_row);
                float db0_v[1];
                {
                    __nv_bfloat16 _bf16_1 = *reinterpret_cast<const __nv_bfloat16*>(database + db_base + 0);
                    db0_v[0] = __bfloat162float(_bf16_1);
                }
                float diff0 = q0 - db0_v[0];
                float dist = diff0 * diff0;
                int should_insert = ((dist < best_d[LOCAL_LIST_CAP_ - 1]) ? 1 : 0);
                if (dist == best_d[LOCAL_LIST_CAP_ - 1]) {
                    if (m_row < best_i[LOCAL_LIST_CAP_ - 1]) {
                        should_insert = 1;
                    }
                }
                if (should_insert != 0) {
                    float carry_d = dist;
                    int carry_i = m_row;
                    #pragma unroll
                    for (int rank = 0; rank < LOCAL_LIST_CAP_; rank++) {
                        float old_d = best_d[rank];
                        int old_i = best_i[rank];
                        int take = ((carry_d < old_d) ? 1 : 0);
                        if (carry_d == old_d) {
                            if (carry_i < old_i) {
                                take = 1;
                            }
                        }
                        best_d[rank] = ((take != 0) ? carry_d : old_d);
                        best_i[rank] = ((take != 0) ? carry_i : old_i);
                        carry_d = ((take != 0) ? old_d : carry_d);
                        carry_i = ((take != 0) ? old_i : carry_i);
                    }
                }
            }
        }
    }
    int local_base = tid * LOCAL_LIST_CAP_;
    #pragma unroll
    for (int kk = 0; kk < LOCAL_LIST_CAP_; kk++) {
        local_dist[local_base + kk] = best_d[kk];
        local_idx[local_base + kk] = best_i[kk];
    }
    __syncthreads();
    if (tid == 0) {
        float final_d[10];
        int final_i[10];
        #pragma unroll
        for (int kk = 0; kk < LOCAL_LIST_CAP_; kk++) {
            final_d[kk] = LOOM_INF;
            final_i[kk] = -1;
        }
        #pragma unroll 1
        for (int src_thread = 0; src_thread < 256; src_thread++) {
            #pragma unroll
            for (int src_k = 0; src_k < LOCAL_LIST_CAP_; src_k++) {
                int src_off = src_thread * LOCAL_LIST_CAP_ + src_k;
                int cand_i = local_idx[src_off];
                if (cand_i >= 0) {
                    float cand_d = local_dist[src_off];
                    int accept_tail = ((cand_d < final_d[LOCAL_LIST_CAP_ - 1]) ? 1 : 0);
                    if (cand_d == final_d[LOCAL_LIST_CAP_ - 1]) {
                        if (cand_i < final_i[LOCAL_LIST_CAP_ - 1]) {
                            accept_tail = 1;
                        }
                    }
                    if (accept_tail != 0) {
                        float carry_d = cand_d;
                        int carry_i = cand_i;
                        #pragma unroll
                        for (int kk = 0; kk < LOCAL_LIST_CAP_; kk++) {
                            float old_d = final_d[kk];
                            int old_i = final_i[kk];
                            int take = ((carry_d < old_d) ? 1 : 0);
                            if (carry_d == old_d) {
                                if (carry_i < old_i) {
                                    take = 1;
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
        }
        unsigned long long out_base = (unsigned long long)((batch_id * Q + q_row) * K);
        #pragma unroll
        for (int kk = 0; kk < LOCAL_LIST_CAP_; kk++) {
            if (kk < K) {
                out_distances[out_base + kk] = final_d[kk];
                out_indices[out_base + kk] = final_i[kk];
            }
        }
    }
}

} // extern "C"

