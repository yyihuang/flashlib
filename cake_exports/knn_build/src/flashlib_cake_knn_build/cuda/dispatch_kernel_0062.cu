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
#define THREADS 128
#define TOP_K_MAX 48
#define SPLIT_COUNT 4

extern "C" {

__global__ __launch_bounds__(128, 1) void
kernel_knn_build_k48_merge_s4_unordered_warp_select(float* __restrict__ partial_dists, int32_t* __restrict__ partial_indices, float* __restrict__ out_dists, int32_t* __restrict__ out_indices, int total_queries)
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
        float cand_d[8];
        int cand_i[8];
        #pragma unroll
        for (int slot = 0; slot < 8; slot++) {
            cand_d[slot] = 3.4e+38f;
            cand_i[slot] = -1;
        }
        #pragma unroll
        for (int split_idx = 0; split_idx < SPLIT_COUNT; split_idx++) {
            int split_base = base_row + split_idx * split_stride;
            int lo_slot = split_idx * 2;
            int hi_slot = lo_slot + 1;
            cand_d[lo_slot] = (float)partial_dists[split_base + cand_lo];
            cand_i[lo_slot] = partial_indices[split_base + cand_lo];
            if (cand_hi < TOP_K_MAX) {
                cand_d[hi_slot] = (float)partial_dists[split_base + cand_hi];
                cand_i[hi_slot] = partial_indices[split_base + cand_hi];
            }
        }
        #pragma unroll
        for (int out_k = 0; out_k < TOP_K_MAX; out_k++) {
            float winner_d = cand_d[0];
            int winner_i = cand_i[0];
            int winner_slot = 0;
            #pragma unroll
            for (int slot = 1; slot < 8; slot++) {
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
                for (int slot = 0; slot < 8; slot++) {
                    if (winner_slot == slot) {
                        cand_d[slot] = 3.4e+38f;
                    }
                }
            }
        }
    }
}

} // extern "C"

