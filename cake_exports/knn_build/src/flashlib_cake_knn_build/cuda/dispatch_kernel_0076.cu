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
#define TOP_K 10
#define SPLITS 72
#define LANESLOTS 3
#define ROWS 4

extern "C" {

__global__ __launch_bounds__(128, 1) void
kernel_knn_build_rag_stream_k10_s72_warp_row_merge_34da(float* __restrict__ partial_dists, int32_t* __restrict__ partial_indices, float* __restrict__ out_dists, int32_t* __restrict__ out_indices, int total_queries)
{
    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;


    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    const int warp_id = warp;
    const int lane_id = lane;

    // === Task calls (dependency order) ===
    int row = bid * ROWS + warp;
    int base_row = row * TOP_K;
    int split_stride = total_queries * TOP_K;
    if (warp < ROWS & row < total_queries) {
        int split_pos[LANESLOTS];
        int split_id_for_slot[LANESLOTS];
        float cand_d[LANESLOTS];
        int cand_i[LANESLOTS];
        #pragma unroll
        for (int slot = 0; slot < LANESLOTS; slot++) {
            int split_id = slot * 32 + lane;
            split_id_for_slot[slot] = split_id;
            split_pos[slot] = 0;
            cand_d[slot] = 3.4e+38f;
            cand_i[slot] = -1;
            if (split_id < SPLITS) {
                int source_addr = base_row + split_id * split_stride;
                cand_d[slot] = (float)partial_dists[source_addr];
                cand_i[slot] = partial_indices[source_addr];
            }
        }
        #pragma unroll
        for (int out_k = 0; out_k < TOP_K; out_k++) {
            float lane_best_d = cand_d[0];
            int lane_best_i = cand_i[0];
            int lane_best_slot = 0;
            #pragma unroll
            for (int slot = 1; slot < LANESLOTS; slot++) {
                if (cand_d[slot] < lane_best_d) {
                    lane_best_d = cand_d[slot];
                    lane_best_i = cand_i[slot];
                    lane_best_slot = slot;
                }
            }
            float warp_min = lane_best_d;
            #pragma unroll
            for (int offset = 16; offset > 0; offset >>= 1)
                warp_min = fminf(warp_min, __shfl_xor_sync(0xFFFFFFFF, warp_min, offset));
            int _vote_0 = __ballot_sync(0xFFFFFFFF, lane_best_d == warp_min);
            int owner_ballot = _vote_0;
            int _ffs_0 = __ffs(owner_ballot);
            int winner_lane = _ffs_0 - 1;
            int _shfl_0 = __shfl_sync(0xFFFFFFFF, lane_best_i, winner_lane);
            int winner_i = _shfl_0;
            int _shfl_1 = __shfl_sync(0xFFFFFFFF, lane_best_slot, winner_lane);
            int winner_slot = _shfl_1;
            if (lane == 0) {
                *((float*)(out_dists + base_row + out_k)) = warp_min;
                *((int*)(out_indices + base_row + out_k)) = winner_i;
            }
            if (lane == winner_lane) {
                #pragma unroll
                for (int slot = 0; slot < LANESLOTS; slot++) {
                    if (winner_slot == slot) {
                        int next_pos = split_pos[slot] + 1;
                        split_pos[slot] = next_pos;
                        cand_d[slot] = 3.4e+38f;
                        cand_i[slot] = -1;
                        if (next_pos < TOP_K) {
                            int next_addr = base_row + split_id_for_slot[slot] * split_stride + next_pos;
                            cand_d[slot] = (float)partial_dists[next_addr];
                            cand_i[slot] = partial_indices[next_addr];
                        }
                    }
                }
            }
        }
    }
}

} // extern "C"

