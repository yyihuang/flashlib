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
#define GROUP_SPLITS 2

extern "C" {

__global__ __launch_bounds__(32, 1) void
kernel_knn_build_non128_frontier_4be7_d768fused_merge_s16g8_4be7_d768fused_v1(float* __restrict__ partial_dists, int32_t* __restrict__ partial_indices, float* __restrict__ out_dists, int32_t* __restrict__ out_indices, int total_queries)
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

