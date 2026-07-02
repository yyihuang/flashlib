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
#define THREADS 32
#define TOP_K_MAX 10
#define GROUP_COUNT 21
#define GROUP_SPLITS 7

extern "C" {

__global__ __launch_bounds__(32, 1) void
kernel_knn_build_q1m524_workfeed_s147_g21_register_merge(float* __restrict__ partial_dists, int32_t* __restrict__ partial_indices, float* __restrict__ out_dists, int32_t* __restrict__ out_indices, int total_queries)
{
    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;


    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    const int warp_id = warp;
    const int lane_id = lane;

    // === Task calls (dependency order) ===
    int split_pos[GROUP_SPLITS];
    int split_base[GROUP_SPLITS];
    float group_cand_d[GROUP_SPLITS];
    int group_cand_i[GROUP_SPLITS];
    #pragma unroll 1
    for (int row = bid; row < total_queries; row += num_bids) {
        int base_row = row * TOP_K_MAX;
        int split_stride = total_queries * TOP_K_MAX;
        float group_best_d = 3.4e+38f;
        int group_best_i = -1;
        int group_best_split = 0;
        if (tid < GROUP_COUNT) {
            int group_idx = tid;
            int source_split0 = group_idx * GROUP_SPLITS;
            #pragma unroll
            for (int local_split = 0; local_split < GROUP_SPLITS; local_split++) {
                split_pos[local_split] = 0;
                int split_id = source_split0 + local_split;
                split_base[local_split] = base_row + split_id * split_stride;
                group_cand_d[local_split] = (float)partial_dists[split_base[local_split]];
                group_cand_i[local_split] = partial_indices[split_base[local_split]];
            }
            group_best_d = group_cand_d[0];
            group_best_i = group_cand_i[0];
            #pragma unroll
            for (int local_split = 1; local_split < GROUP_SPLITS; local_split++) {
                if (group_cand_d[local_split] < group_best_d) {
                    group_best_d = group_cand_d[local_split];
                    group_best_i = group_cand_i[local_split];
                    group_best_split = local_split;
                }
            }
        }
        #pragma unroll
        for (int out_k = 0; out_k < TOP_K_MAX; out_k++) {
            float warp_min = group_best_d;
            #pragma unroll
            for (int offset = 16; offset > 0; offset >>= 1)
                warp_min = fminf(warp_min, __shfl_xor_sync(0xFFFFFFFF, warp_min, offset));
            int _vote_0 = __ballot_sync(0xFFFFFFFF, group_best_d == warp_min);
            int owner_ballot = _vote_0;
            int _ffs_0 = __ffs(owner_ballot);
            int winner_lane = _ffs_0 - 1;
            int _shfl_0 = __shfl_sync(0xFFFFFFFF, group_best_i, winner_lane);
            int winner_i = _shfl_0;
            if (tid == 0) {
                *((float*)(out_dists + base_row + out_k)) = warp_min;
                *((int*)(out_indices + base_row + out_k)) = winner_i;
            }
            if (tid == winner_lane) {
                if (out_k + 1 < TOP_K_MAX) {
                    split_pos[group_best_split] = split_pos[group_best_split] + 1;
                    int next_pos = split_pos[group_best_split];
                    int next_addr = split_base[group_best_split] + next_pos;
                    group_cand_d[group_best_split] = (float)partial_dists[next_addr];
                    group_cand_i[group_best_split] = partial_indices[next_addr];
                    group_best_d = group_cand_d[0];
                    group_best_i = group_cand_i[0];
                    group_best_split = 0;
                    #pragma unroll
                    for (int local_split = 1; local_split < GROUP_SPLITS; local_split++) {
                        if (group_cand_d[local_split] < group_best_d) {
                            group_best_d = group_cand_d[local_split];
                            group_best_i = group_cand_i[local_split];
                            group_best_split = local_split;
                        }
                    }
                }
            }
        }
    }
}

} // extern "C"

