/*
 * Copyright (c) 2023-2024, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

// Modified by Sukjin Kim, 2025

#include <pybind11/pybind11.h>
#include <torch/extension.h>

#include <cstdint>
#include <iostream>
#include <algorithm>
#include <vector>
#include <assert.h>
#include <limits>

#include <cuda.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>

namespace py = pybind11;

// #define _CLK_BREAKDOWN
// #define PRINT

typedef float DATA_T;
typedef uint32_t INDEX_T;
typedef float DISTANCE_T;

#define NO_PRUNE 0
#define RANDOM_PRUNE 1
#define SIGN_BIT_PRUNE 2

#define NO_THRESHOLD 0
#define USE_THRESHOLD 1
#define AGRESSIVE_THRESHOLD 2

#define NO_SEED 0
#define USE_SEED 1

/////////////////////////////////////////////////////// debug ///////////////////////////////////////////////////////

template <typename T>
__device__ inline void print_buffer(T* buffer, uint32_t buffer_size)
{
    if constexpr (std::is_same<T, INDEX_T>::value) {
        for (uint32_t i = 0; i < buffer_size; i++) {
            printf("%d, ", buffer[i]);
        }
    } else if constexpr (std::is_same<T, DISTANCE_T>::value || std::is_same<T, DATA_T>::value) {
        for (uint32_t i = 0; i < buffer_size; i++) {
            printf("%f, ", buffer[i]);
        }
    } else {
        printf("Unsupported type\n");
        assert(false);
    }
    printf("\n");
}

/////////////////////////////////////////////////////// hash table ///////////////////////////////////////////////////////

__host__ __device__ inline uint32_t hashtable_getsize(const uint32_t BITLEN)
{
    return 1 << BITLEN;
}

__device__ inline void hashtable_init(INDEX_T* const table, const unsigned BITLEN, unsigned FIRST_TID = 0)
{
    if (threadIdx.x < FIRST_TID) return;
    for (uint32_t i = threadIdx.x - FIRST_TID; i < hashtable_getsize(BITLEN); i += blockDim.x - FIRST_TID)
    {
        table[i] = std::numeric_limits<INDEX_T>::max();
    }
}

__device__ inline uint32_t hashtable_insert(INDEX_T* const table, const unsigned BITLEN, const INDEX_T key)
{
    // Open addressing is used for collision resolution
    const uint32_t size     = hashtable_getsize(BITLEN);
    const uint32_t bit_mask = size - 1;

    // Linear probing
    INDEX_T index                = (key ^ (key >> BITLEN)) & bit_mask;
    constexpr uint32_t stride = 1;

    for (unsigned i = 0; i < size; i++)
    {
        const INDEX_T old = atomicCAS(&table[index], ~static_cast<INDEX_T>(0), key);
        if (old == ~static_cast<INDEX_T>(0))
        {
            return 1;
        }
        else if (old == key)
        {
            return 0;
        }
        index = (index + stride) & bit_mask;
    }
  return 0;
}

__device__ inline void hashtable_restore
(   
    INDEX_T* const table, 
    const unsigned BITLEN,
    const INDEX_T* itopk_indices,
    const uint32_t itopk_size,
    const uint32_t first_tid = 0 
)
{
    constexpr INDEX_T index_msb_1_mask = 0x80000000;
    if (threadIdx.x < first_tid) return;
    for (unsigned i = threadIdx.x - first_tid; i < itopk_size; i += blockDim.x - first_tid)
    {
        auto key = itopk_indices[i] & ~index_msb_1_mask;  // clear most significant bit
        hashtable_insert(table, BITLEN, key);
    }
}


/////////////////////////////////////////////////////// sort ///////////////////////////////////////////////////////

template <class K, class V>
__device__ inline void swap_if_needed(K& k0, V& v0, const unsigned lane_offset, const bool asc)
{
  auto k1 = __shfl_xor_sync(~0u, k0, lane_offset);
  auto v1 = __shfl_xor_sync(~0u, v0, lane_offset);
  if ((k0 != k1) && ((k0 < k1) != asc)) {
    k0 = k1;
    v0 = v1;
  }
}

template <class K, class V>
__device__ inline void swap_if_needed(K& k0, V& v0, K& k1, V& v1, const bool asc)
{
  if ((k0 != k1) && ((k0 < k1) != asc)) {
    const auto tmp_k = k0;
    k0               = k1;
    k1               = tmp_k;
    const auto tmp_v = v0;
    v0               = v1;
    v1               = tmp_v;
  }
}

template <class K, class V, unsigned _N, unsigned warp_size>
__device__ inline void warp_merge_core(K k[2], V v[2], const std::uint32_t range, const bool asc)
{
    if (_N == 2)
    {
        constexpr unsigned N = 2;
        const auto lane_id   = threadIdx.x % warp_size;

        if (range == 1) {
            const auto p = ((lane_id & 1) == 0);
            swap_if_needed<float, uint32_t>(k[0], v[0], k[1], v[1], p);
            return;
        }

        const std::uint32_t b = range;
        for (std::uint32_t c = b / 2; c >= 1; c >>= 1) {
            const auto p = static_cast<bool>(lane_id & b) == static_cast<bool>(lane_id & c);
        #pragma unroll
            for (std::uint32_t i = 0; i < N; i++) {
            swap_if_needed<float, uint32_t>(k[i], v[i], c, p);
            }
        }
        const auto p = ((lane_id & b) == 0);
        swap_if_needed<float, uint32_t>(k[0], v[0], k[1], v[1], p);
    }

    else if (_N == 4)
    {
        constexpr unsigned N = 4;
        const auto lane_id = threadIdx.x % warp_size;

        if (range == 1) {
            for (std::uint32_t b = 2; b <= N; b <<= 1) {
                for (std::uint32_t c = b / 2; c >= 1; c >>= 1) {
        #pragma unroll
                for (std::uint32_t i = 0; i < N; i++) {
                    std::uint32_t j = i ^ c;
                    if (i >= j) continue;
                    const auto line_id = i + (N * lane_id);
                    const auto p       = static_cast<bool>(line_id & b) == static_cast<bool>(line_id & c);
                    swap_if_needed(k[i], v[i], k[j], v[j], p);
                }
                }
            }
            return;
        }

        const std::uint32_t b = range;
        for (std::uint32_t c = b / 2; c >= 1; c >>= 1) {
            const auto p = static_cast<bool>(lane_id & b) == static_cast<bool>(lane_id & c);
        #pragma unroll
            for (std::uint32_t i = 0; i < N; i++) {
                swap_if_needed(k[i], v[i], c, p);
            }
        }
        const auto p = ((lane_id & b) == 0);
        for (std::uint32_t c = N / 2; c >= 1; c >>= 1) {
        #pragma unroll
            for (std::uint32_t i = 0; i < N; i++) {
                std::uint32_t j = i ^ c;
                if (i >= j) continue;
                swap_if_needed(k[i], v[i], k[j], v[j], p);
            }
        }
    }
}


template <class K, class V, unsigned N, unsigned warp_size = 32>
__device__ void warp_merge(K k[N], V v[N], unsigned range, const bool asc = true)
{
  warp_merge_core<K, V, N, warp_size>(k, v, range, asc);
}

template <class K, class V, unsigned N, unsigned warp_size = 32>
__device__ void warp_sort(K k[N], V v[N], const bool asc = true)
{
  for (std::uint32_t range = 1; range <= warp_size; range <<= 1) {
    warp_merge<K, V, N, warp_size>(k, v, range, asc);
  }
}

template <unsigned N_1, unsigned N_2>
__device__ void candidate_by_bitonic_sort
(
    INDEX_T* candidate_indices,
    DISTANCE_T* candidate_distances,
    uint32_t CANDIDATE_BUFFER_SIZE
)
{
    const unsigned lane_id = threadIdx.x % 32;
    const unsigned warp_id = threadIdx.x / 32;

// sort 1
    if (warp_id > 0) { return; }
    if (CANDIDATE_BUFFER_SIZE != 64) 
    {
        printf("CANDIDATE_BUFFER_SIZE must be 64\n");
        assert(false);
    }
    // constexpr unsigned N_1 = 2;
    DISTANCE_T key_1[N_1];
    INDEX_T val_1[N_1];

    /* Candidates -> Reg */
    for (unsigned i = 0; i < N_1; i++)
    {
        unsigned j = lane_id + (32 * i);
        if (j < CANDIDATE_BUFFER_SIZE)
        {
            key_1[i] = candidate_distances[j];
            val_1[i] = candidate_indices[j];
        }
        else
        {
            key_1[i] = std::numeric_limits<DISTANCE_T>::max();
            val_1[i] = std::numeric_limits<INDEX_T>::max();
        }
    }
    /* Sort */
    warp_sort<float, uint32_t, N_1>(key_1, val_1);
    /* Reg -> Temp_itopk */
    for (unsigned i = 0; i < N_1; i++)
    {
        unsigned j = (N_1 * lane_id) + i;
        if (j < CANDIDATE_BUFFER_SIZE){
            candidate_distances[j] = key_1[i];
            candidate_indices[j]   = val_1[i];
        }
    }
}

template <unsigned N_1, unsigned N_2>
__device__ void candidate_by_bitonic_sort_inverse
(
    INDEX_T* candidate_indices,
    DISTANCE_T* candidate_distances,
    uint32_t CANDIDATE_BUFFER_SIZE
)
{
    const unsigned lane_id = threadIdx.x % 32;
    const unsigned warp_id = threadIdx.x / 32;

// sort 1
    if (warp_id > 0) { return; }
    if (CANDIDATE_BUFFER_SIZE != 64) 
    {
        printf("CANDIDATE_BUFFER_SIZE must be 64\n");
        assert(false);
    }
    // constexpr unsigned N_1 = 2;
    DISTANCE_T key_1[N_1];
    INDEX_T val_1[N_1];

    /* Candidates -> Reg */
    for (unsigned i = 0; i < N_1; i++)
    {
        unsigned j = lane_id + (32 * i);
        if (j < CANDIDATE_BUFFER_SIZE)
        {
            key_1[i] = candidate_distances[j];
            val_1[i] = candidate_indices[j];
        }
        else
        {
            key_1[i] = std::numeric_limits<DISTANCE_T>::max();
            val_1[i] = std::numeric_limits<INDEX_T>::max();
        }
    }
    /* Sort */
    warp_sort<float, uint32_t, N_1>(key_1, val_1);
    /* Reg -> Temp_itopk */
    for (unsigned i = 0; i < N_1; i++)
    {
        unsigned j = CANDIDATE_BUFFER_SIZE - 1 - ( (N_1 * lane_id) + i );
        if (j < CANDIDATE_BUFFER_SIZE){
            candidate_distances[j] = key_1[i];
            candidate_indices[j]   = val_1[i];
        }
    }
}

template <unsigned N_1, unsigned N_2>
__device__ void topk_by_bitonic_sort
(
    INDEX_T* result_indices_ptr,
    DISTANCE_T* result_distances_ptr,
    uint32_t RESULT_BUFFER_SIZE,
    uint32_t CANDIDATE_BUFFER_SIZE,
    uint32_t INTERNAL_TOPK,
    bool first
)
{
    const unsigned lane_id = threadIdx.x % 32;
    const unsigned warp_id = threadIdx.x / 32;

// sort 1
    if (warp_id > 0) { return; }
    if (CANDIDATE_BUFFER_SIZE != 64) 
    {
        printf("CANDIDATE_BUFFER_SIZE must be 64\n");
        assert(false);
    }
    // constexpr unsigned N_1 = 2;
    DISTANCE_T key_1[N_1];
    INDEX_T val_1[N_1];
    auto candidate_distances = result_distances_ptr + INTERNAL_TOPK;
    auto candidate_indices = result_indices_ptr + INTERNAL_TOPK;
    /* Candidates -> Reg */
    for (unsigned i = 0; i < N_1; i++)
    {
        unsigned j = lane_id + (32 * i);
        if (j < CANDIDATE_BUFFER_SIZE)
        {
            key_1[i] = candidate_distances[j];
            val_1[i] = candidate_indices[j];
        }
        else
        {
            key_1[i] = std::numeric_limits<DISTANCE_T>::max();
            val_1[i] = std::numeric_limits<INDEX_T>::max();
        }
    }
    /* Sort */
    warp_sort<float, uint32_t, N_1>(key_1, val_1);
    /* Reg -> Temp_itopk */
    for (unsigned i = 0; i < N_1; i++)
    {
        unsigned j = (N_1 * lane_id) + i;
        if (j < CANDIDATE_BUFFER_SIZE && j < INTERNAL_TOPK){
            candidate_distances[j] = key_1[i];
            candidate_indices[j]   = val_1[i];
        }
    }

// sort 2
    // constexpr unsigned N_2 = 4;
    DISTANCE_T key_2[N_2];
    INDEX_T val_2[N_2];
    if (first) {
      /* Load itopk results */
        for (unsigned i = 0; i < N_2; i++) {
            unsigned j = lane_id + (32 * i);
            if (j < INTERNAL_TOPK) {
                key_2[i] = result_distances_ptr[j];
                val_2[i] = result_indices_ptr[j];
            } else {
                key_2[i] = std::numeric_limits<DISTANCE_T>::max();
                val_2[i] = std::numeric_limits<INDEX_T>::max();
            }
        }
        /* Warp Sort */
        warp_sort<float, uint32_t, N_2>(key_2, val_2);
    }
    else
    {
        /* Load itopk results */
        for (unsigned i = 0; i < N_2; i++) {
            unsigned j = (N_2 * lane_id) + i;
            if (j < INTERNAL_TOPK) {
                key_2[i] = result_distances_ptr[j];
                val_2[i] = result_indices_ptr[j];
            } else {
                key_2[i] = std::numeric_limits<DISTANCE_T>::max();
                val_2[i] = std::numeric_limits<INDEX_T>::max();
            }
        }
    }
    /* Merge candidates */
    for (unsigned i = 0; i < N_2; i++) {
      unsigned j = (N_2 * lane_id) + i;  // [0:MAX_ITOPK-1]
      unsigned k = INTERNAL_TOPK - 1 - j;
      if (k >= INTERNAL_TOPK || k >= CANDIDATE_BUFFER_SIZE) continue;
      auto candidate_key = candidate_distances[k];
      if (key_2[i] > candidate_key) {
        key_2[i] = candidate_key;
        val_2[i] = candidate_indices[k];
      }
    }
    /* Warp Merge */
    warp_merge<float, uint32_t, N_2>(key_2, val_2, 32);
    /* Store new itopk results */
    for (unsigned i = 0; i < N_2; i++) {
      unsigned j = (N_2 * lane_id) + i;
      if (j < INTERNAL_TOPK) {
        result_distances_ptr[j] = key_2[i];
        result_indices_ptr[j]   = val_2[i];
      }
    }

}

/////////////////////////////////////////////////////// distance calculation ///////////////////////////////////////////////////////

__device__ void pickup_next_parents
(
    uint32_t* terminate_flag,
    INDEX_T* const parent_list_buffer,
    INDEX_T* const result_indices_buffer,
    const uint32_t INTERNAL_TOPK,
    const uint32_t SEARCH_WIDTH
)
{
    constexpr INDEX_T index_msb_1_mask = 0x80000000;
    // if (threadIdx.x >= 32) return;

    for (std::uint32_t i = threadIdx.x; i < SEARCH_WIDTH; i += 32)
    {
        parent_list_buffer[i] = std::numeric_limits<INDEX_T>::max();
    }
    std::uint32_t itopk_max = INTERNAL_TOPK;
    if (itopk_max % 32)
    {
        itopk_max += 32 - (itopk_max % 32);
    }
    std::uint32_t num_new_parents = 0;
    
    for (std::uint32_t j = threadIdx.x; j < itopk_max; j += 32) 
    {
        INDEX_T index;
        int new_parent = 0;
        if (j < INTERNAL_TOPK)
        {
            index = result_indices_buffer[j];
            if ((index & index_msb_1_mask) == 0)
            {   // check if most significant bit is set
                new_parent = 1;
            }
        }

        const std::uint32_t ballot_mask = __ballot_sync(0xffffffff, new_parent);
        if (new_parent)
        {
            const auto i = __popc(ballot_mask & ((1 << threadIdx.x) - 1)) + num_new_parents;
            if (i < SEARCH_WIDTH)
            {
                parent_list_buffer[i] = j;
                // set most significant bit as used node
                result_indices_buffer[j] |= index_msb_1_mask;
            }
        }

        num_new_parents += __popc(ballot_mask);
        if (num_new_parents >= SEARCH_WIDTH) { break; }
    }
    if (threadIdx.x == 0 && (num_new_parents == 0)) { *terminate_flag = 1; }

}


template<uint32_t VECTOR_DIM, uint32_t TEAM_SIZE>
__device__ DISTANCE_T compute_similarity_vector_load_full
(
    DATA_T* d_dataset_ptr,
    DATA_T* query_ptr,
    INDEX_T child_id,
    bool valid_child
)
{
    auto child_data_ptr  = d_dataset_ptr + child_id * VECTOR_DIM;
    unsigned lane_id  = threadIdx.x % TEAM_SIZE;

    constexpr unsigned vlen = 16 / sizeof(DATA_T);  //128bit = 16Byte
    constexpr unsigned reg_nelem = (VECTOR_DIM + TEAM_SIZE * vlen - 1) / (TEAM_SIZE * vlen);
    float4 dl_buff[reg_nelem];
    unsigned int full_mask = 0xffffffff;


    DISTANCE_T norm2 = 0;

    if (valid_child)
    {
        // Load
        #pragma unroll
        for (uint32_t e = 0; e < reg_nelem; e++) 
        {
            const uint32_t k = (lane_id + (TEAM_SIZE * e)) * vlen;
            if (k >= VECTOR_DIM) break;
            dl_buff[e] = reinterpret_cast<float4*>(child_data_ptr)[k / 4];
        }
        
        // Compute
        #pragma unroll
        for (uint32_t e = 0; e < reg_nelem; e++) 
        {
            const uint32_t k = (lane_id + (TEAM_SIZE * e)) * vlen;
            if (k >= VECTOR_DIM) break;

            DISTANCE_T d = query_ptr[k];
            norm2 += (d - dl_buff[e].x) * (d - dl_buff[e].x);
            d = query_ptr[k + 1];
            norm2 += (d - dl_buff[e].y) * (d - dl_buff[e].y);
            d = query_ptr[k + 2];
            norm2 += (d - dl_buff[e].z) * (d - dl_buff[e].z);
            d = query_ptr[k + 3];
            norm2 += (d - dl_buff[e].w) * (d - dl_buff[e].w);

        }
    }

    unsigned team_mask = __ballot_sync(0xffffffff, valid_child);

    for (uint32_t offset = TEAM_SIZE / 2; offset > 0; offset >>= 1) {
        norm2 += __shfl_xor_sync(full_mask, norm2, offset);
    }

    return norm2;
}

template<uint32_t VECTOR_DIM, uint32_t TEAM_SIZE, uint32_t RATIO = 2>
__device__ DISTANCE_T compute_similarity_vector_load_partial
(
    DATA_T* d_dataset_ptr,
    DATA_T* query_ptr,
    INDEX_T child_id,
    bool valid_child,
    DISTANCE_T threshold = 9999999999
)
{
    auto child_data_ptr  = d_dataset_ptr + child_id * VECTOR_DIM;
    unsigned lane_id  = threadIdx.x % TEAM_SIZE;

    constexpr unsigned vlen = 16 / sizeof(DATA_T);  //128bit = 16Byte, vlen == 4
    constexpr unsigned reg_nelem = std::max(1U, ((VECTOR_DIM + TEAM_SIZE * vlen - 1) / (TEAM_SIZE * vlen) + RATIO - 1) / RATIO * (RATIO - 1));
    float4 dl_buff[reg_nelem];
    unsigned int full_mask = 0xffffffff;


    DISTANCE_T norm2 = 0;

    if (valid_child)
    {
        // Load
        #pragma unroll
        for (uint32_t e = 0; e < reg_nelem; e++) 
        {
            const uint32_t k = (lane_id + (TEAM_SIZE * e)) * vlen;
            if (k >= VECTOR_DIM) break;
            dl_buff[e] = reinterpret_cast<float4*>(child_data_ptr)[k / 4];
        }
        
        // Compute
        #pragma unroll
        for (uint32_t e = 0; e < reg_nelem; e++) 
        {
            const uint32_t k = (lane_id + (TEAM_SIZE * e)) * vlen;
            if (k >= VECTOR_DIM) break;

            DISTANCE_T d = query_ptr[k];
            norm2 += (d - dl_buff[e].x) * (d - dl_buff[e].x);
            d = query_ptr[k + 1];
            norm2 += (d - dl_buff[e].y) * (d - dl_buff[e].y);
            d = query_ptr[k + 2];
            norm2 += (d - dl_buff[e].z) * (d - dl_buff[e].z);
            d = query_ptr[k + 3];
            norm2 += (d - dl_buff[e].w) * (d - dl_buff[e].w);

        }
    }

    unsigned team_mask = __ballot_sync(0xffffffff, valid_child);

    for (uint32_t offset = TEAM_SIZE / 2; offset > 0; offset >>= 1) {
        norm2 += __shfl_xor_sync(full_mask, norm2, offset);
    }

    return norm2;
}

template<uint32_t VECTOR_DIM, uint32_t TEAM_SIZE>
__device__ DISTANCE_T compute_similarity_vector_load_by_team
(
    DATA_T* d_dataset_ptr,
    DATA_T* query_ptr,
    INDEX_T child_id,
    unsigned team_mask
)
{
    auto child_data_ptr  = d_dataset_ptr + child_id * VECTOR_DIM;
    unsigned lane_id  = threadIdx.x % TEAM_SIZE;
    unsigned team_id  = threadIdx.x / TEAM_SIZE;

    constexpr unsigned vlen = 16 / sizeof(DATA_T);  //128bit = 16Byte
    constexpr unsigned reg_nelem = (VECTOR_DIM + TEAM_SIZE * vlen - 1) / (TEAM_SIZE * vlen);
    float4 dl_buff[reg_nelem];

    DISTANCE_T norm2 = 0;
    
    // Load
    #pragma unroll
    for (uint32_t e = 0; e < reg_nelem; e++) 
    {
        const uint32_t k = (lane_id + (TEAM_SIZE * e)) * vlen;
        if (k >= VECTOR_DIM) break;
        dl_buff[e] = reinterpret_cast<float4*>(child_data_ptr)[k / 4];
    }
    
    // Compute
    #pragma unroll
    for (uint32_t e = 0; e < reg_nelem; e++) 
    {
        const uint32_t k = (lane_id + (TEAM_SIZE * e)) * vlen;
        if (k >= VECTOR_DIM) break;

        DISTANCE_T d = query_ptr[k];
        norm2 += (d - dl_buff[e].x) * (d - dl_buff[e].x);
        d = query_ptr[k + 1];
        norm2 += (d - dl_buff[e].y) * (d - dl_buff[e].y);
        d = query_ptr[k + 2];
        norm2 += (d - dl_buff[e].z) * (d - dl_buff[e].z);
        d = query_ptr[k + 3];
        norm2 += (d - dl_buff[e].w) * (d - dl_buff[e].w);

    }

    for (uint32_t offset = TEAM_SIZE / 2; offset > 0; offset >>= 1) {
        norm2 += __shfl_xor_sync(team_mask, norm2, offset);
    }

    return norm2;
}

template<uint32_t VECTOR_DIM, uint32_t TEAM_SIZE>
__device__ DISTANCE_T compute_similarity_vector_load_by_team_partial_float4
(
    DATA_T* d_dataset_ptr,
    DATA_T* query_ptr,
    INDEX_T child_id,
    unsigned team_mask,
    uint32_t e
)
{

    auto child_data_ptr  = d_dataset_ptr + child_id * VECTOR_DIM;
    unsigned lane_id  = threadIdx.x % TEAM_SIZE;

    constexpr unsigned vlen = 16 / sizeof(DATA_T);  //128bit = 16Byte
    // constexpr unsigned reg_nelem = (VECTOR_DIM + TEAM_SIZE * vlen - 1) / (TEAM_SIZE * vlen);
    float4 dl_buff;

    DISTANCE_T norm2 = 0;
    
    const uint32_t k = (lane_id + (TEAM_SIZE * e)) * vlen;
    if (k < VECTOR_DIM)
    {
        // Load
        dl_buff = reinterpret_cast<float4*>(child_data_ptr)[k / 4];
        // Compute
        DISTANCE_T d = query_ptr[k];
        norm2 += (d - dl_buff.x) * (d - dl_buff.x);
        d = query_ptr[k + 1];
        norm2 += (d - dl_buff.y) * (d - dl_buff.y);
        d = query_ptr[k + 2];
        norm2 += (d - dl_buff.z) * (d - dl_buff.z);
        d = query_ptr[k + 3];
        norm2 += (d - dl_buff.w) * (d - dl_buff.w);
    }

    // Reduce
    for (uint32_t offset = TEAM_SIZE / 2; offset > 0; offset >>= 1) {
        norm2 += __shfl_xor_sync(team_mask, norm2, offset);
    }

    return norm2;
}

template<uint32_t VECTOR_DIM, uint32_t TEAM_SIZE>
__device__ DISTANCE_T compute_similarity_vector_load_by_team_partial_float2
(
    DATA_T* d_dataset_ptr,
    DATA_T* query_ptr,
    INDEX_T child_id,
    unsigned team_mask,
    uint32_t e
)
{

    auto child_data_ptr  = d_dataset_ptr + child_id * VECTOR_DIM;
    unsigned lane_id  = threadIdx.x % TEAM_SIZE;
    unsigned team_id  = threadIdx.x / TEAM_SIZE;

    float2 dl_buff;

    DISTANCE_T norm2 = 0;
    
    const uint32_t k = (lane_id + (TEAM_SIZE * e)) * 2;
    if (k < VECTOR_DIM)
    {
        // Load
        dl_buff = reinterpret_cast<float2*>(child_data_ptr)[k / 2];
        // Compute
        DISTANCE_T d = query_ptr[k];
        norm2 += (d - dl_buff.x) * (d - dl_buff.x);
        d = query_ptr[k + 1];
        norm2 += (d - dl_buff.y) * (d - dl_buff.y);
    }

    // Reduce
    for (uint32_t offset = TEAM_SIZE / 2; offset > 0; offset >>= 1) {
        norm2 += __shfl_xor_sync(team_mask, norm2, offset);
    }

    return norm2;
}

// map top10 nodes and fetch neighbors and calculate distance
template<uint32_t VECTOR_DIM, uint32_t TEAM_SIZE>
__device__ void compute_distance_to_maped_top10_nodes
(
    INDEX_T* candidate_indices_ptr,
    DISTANCE_T* candidate_distances_ptr,
    DATA_T* query_ptr,
    DATA_T* dataset_ptr,
    INDEX_T* graph_ptr,
    INDEX_T* top10_ptr,
    INDEX_T* seed_map_ptr,
    uint32_t GRAPH_DEGREE,
    uint32_t DATASET_SIZE,
    uint32_t CANDIDATE_BUFFER_SIZE,
    INDEX_T* visited_hash_ptr,
    uint32_t BITLEN,
    uint32_t SEED_CONFIG,
    uint32_t HASH_TABLE_CONFIG,
    uint32_t SEED_TOPK_SIZE
)
{
    const INDEX_T invalid_index        = std::numeric_limits<INDEX_T>::max();

    // Load top N nodes
    INDEX_T top1 = top10_ptr[blockIdx.y * SEED_TOPK_SIZE];

    // Map top1 nodes to dataset
    INDEX_T parent_id = seed_map_ptr[top1];

    // fetch neighbors and insert to hash table
    for (uint32_t i = threadIdx.x; i < CANDIDATE_BUFFER_SIZE; i += blockDim.x)
    {
        INDEX_T child_id = graph_ptr[(static_cast<int64_t>(GRAPH_DEGREE) * parent_id) + i];

        if (HASH_TABLE_CONFIG)
        {
            if (child_id != invalid_index)
            {
                if (hashtable_insert(visited_hash_ptr, BITLEN, child_id) == 0)
                {
                    child_id = invalid_index;
                    candidate_distances_ptr[i] = FLT_MAX;
                }
                else
                {
                    candidate_distances_ptr[i] = 0;
                }
            }
            candidate_indices_ptr[i] = child_id;
        }
    }

    __syncthreads();

    // calculate distance
    for (std::uint32_t tid = threadIdx.x; tid < CANDIDATE_BUFFER_SIZE * TEAM_SIZE; tid += blockDim.x)
    {
        const auto i       = tid / TEAM_SIZE;
        const bool valid_i = (i < (CANDIDATE_BUFFER_SIZE));
        INDEX_T child_id   = invalid_index;
        if (valid_i) { child_id = candidate_indices_ptr[i]; }

        DISTANCE_T norm2 = compute_similarity_vector_load_full<VECTOR_DIM, TEAM_SIZE>
                                (
                                    dataset_ptr, 
                                    query_ptr, 
                                    child_id, 
                                    child_id != invalid_index
                                );

        // Store the distance
        const unsigned lane_id = threadIdx.x % TEAM_SIZE;
        if (valid_i && lane_id == 0)
        {
            if (child_id != invalid_index)
            {
                candidate_distances_ptr[i] = norm2;
            } 
            else
            {
                candidate_distances_ptr[i] = FLT_MAX;
            }
        }
    }    

}

template <uint32_t VECTOR_DIM, uint32_t TEAM_SIZE>
__device__ void compute_distance_to_random_nodes
(
    INDEX_T* candidate_indices_ptr,
    DISTANCE_T* candidate_distances_ptr,
    DATA_T* query_ptr,
    DATA_T* dataset_ptr,
    uint32_t DATASET_SIZE,
    uint32_t CANDIDATE_BUFFER_SIZE,
    uint32_t NUM_DIST,
    INDEX_T* visited_hash_ptr,
    uint32_t BITLEN,
    uint32_t HASH_TABLE_CONFIG
)
{   
    uint32_t max_i = CANDIDATE_BUFFER_SIZE;
    if (max_i % (32 / TEAM_SIZE)) { max_i += (32 / TEAM_SIZE) - (max_i % (32 / TEAM_SIZE)); }

    for (uint32_t i = threadIdx.x / TEAM_SIZE; i < max_i; i += blockDim.x / TEAM_SIZE)
    {
        const bool valid_i = (i < CANDIDATE_BUFFER_SIZE);

        INDEX_T best_index_team_local;
        DISTANCE_T best_norm2_team_local = std::numeric_limits<DISTANCE_T>::max();
        for (uint32_t j = 0; j < NUM_DIST; j++)
        {
            // Select a node randomly and compute the distance to it
            INDEX_T seed_index;
            if (valid_i)
            {
                uint32_t gid = threadIdx.x + (blockDim.x * (i + (CANDIDATE_BUFFER_SIZE * j)));
                curandState state;
                curand_init(clock64(), gid, 0, &state);
                seed_index = curand_uniform(&state) * DATASET_SIZE;
                
            }

            DISTANCE_T norm2 = compute_similarity_vector_load_full<VECTOR_DIM, 8>
                                                (
                                                    dataset_ptr,
                                                    query_ptr,
                                                    seed_index,
                                                    valid_i
                                                );

            if (valid_i && (norm2 < best_norm2_team_local)) {
                best_norm2_team_local = norm2;
                best_index_team_local = seed_index;
            }
        }

        const unsigned lane_id = threadIdx.x % TEAM_SIZE;
        if (valid_i && lane_id == 0)
        {
            if (HASH_TABLE_CONFIG)
            {
                if (hashtable_insert(visited_hash_ptr, BITLEN, best_index_team_local))
                {
                    candidate_distances_ptr[i] = best_norm2_team_local;
                    candidate_indices_ptr[i]   = best_index_team_local;
                } 
                else
                {
                    candidate_distances_ptr[i] = std::numeric_limits<DISTANCE_T>::max();
                    candidate_indices_ptr[i]   = std::numeric_limits<INDEX_T>::max();
                }
            }
            else
            {
                candidate_distances_ptr[i] = best_norm2_team_local;
                candidate_indices_ptr[i]   = best_index_team_local;
            }
        }
    }
}

template <uint32_t VECTOR_DIM, uint32_t TEAM_SIZE, uint32_t INTERNAL_TOPK>
__device__ void compute_distance_to_child_nodes
(
    INDEX_T* parent_buffer,
    INDEX_T* candidate_indices_ptr,
    DISTANCE_T* candidate_distances_ptr,
    INDEX_T* internal_topk_list,
    DATA_T* query_ptr,
    DATA_T* d_dataset_ptr,
    INDEX_T* d_graph_ptr,
    uint32_t GRAPH_DEGREE,
    uint32_t SEARCH_WIDTH,
    uint32_t CANDIDATE_BUFFER_SIZE,
    uint32_t NUM_DIST,
    INDEX_T* visited_hash_ptr,
    uint32_t BITLEN,
    uint32_t* counter,
    uint32_t THRESHOLD_CONFIG,
    uint32_t FULL_COMPUTE_RATIO,
    uint32_t HASH_TABLE_CONFIG
)
{
    constexpr INDEX_T index_msb_1_mask = 0x80000000;
    const INDEX_T invalid_index        = std::numeric_limits<INDEX_T>::max();

    // Read child indices of parents from knn graph and check if the distance
    // computaiton is necessary.
    for (uint32_t i = threadIdx.x; i < CANDIDATE_BUFFER_SIZE; i += blockDim.x)
    {
        const INDEX_T smem_parent_id = parent_buffer[0];
        const auto parent_id = internal_topk_list[smem_parent_id] & ~index_msb_1_mask;
        INDEX_T child_id = d_graph_ptr[(static_cast<int64_t>(GRAPH_DEGREE) * parent_id) + i];
       
        if(!HASH_TABLE_CONFIG)
        {
            // check with internal topk
            for (int32_t internal_topk_ptr = 0; internal_topk_ptr < INTERNAL_TOPK; internal_topk_ptr++)
            {
                INDEX_T internal_topk_id = internal_topk_list[internal_topk_ptr] & ~index_msb_1_mask;
                if(child_id == internal_topk_id)
                {
                    child_id = invalid_index;
                    break;
                }
            }
            if (child_id != invalid_index)
            {
                candidate_distances_ptr[i] = 0;
            }
            candidate_indices_ptr[i] = child_id;
        }
        else
        {
            // const INDEX_T smem_parent_id = parent_buffer[i / GRAPH_DEGREE];
            // INDEX_T child_id             = invalid_index;
            // if (smem_parent_id != invalid_index)
            // {
            //     const auto parent_id = internal_topk_list[smem_parent_id] & ~index_msb_1_mask;
            //     child_id             = d_graph_ptr[(i % GRAPH_DEGREE) + (static_cast<int64_t>(GRAPH_DEGREE) * parent_id)];
            // }
            if (child_id != invalid_index)
            {
                if (hashtable_insert(visited_hash_ptr, BITLEN, child_id) == 0)
                {
                    child_id = invalid_index;
                    candidate_distances_ptr[i] = FLT_MAX;
                }
                else
                {
                    candidate_distances_ptr[i] = 0;
                }
            }
            candidate_indices_ptr[i] = child_id;
        }

    }

    // Compute half of the distance to child nodes
    __syncthreads();

    if (THRESHOLD_CONFIG == NO_THRESHOLD)
    {
        std::uint32_t max_i = GRAPH_DEGREE * SEARCH_WIDTH;
        for (std::uint32_t tid = threadIdx.x; tid < max_i * TEAM_SIZE; tid += blockDim.x)
        {
            const auto i       = tid / TEAM_SIZE;
            const bool valid_i = (i < (GRAPH_DEGREE * SEARCH_WIDTH));
            INDEX_T child_id   = invalid_index;
            if (valid_i) { child_id = candidate_indices_ptr[i]; }

            DISTANCE_T norm2 = compute_similarity_vector_load_full<VECTOR_DIM, TEAM_SIZE>
                                    (
                                        d_dataset_ptr, 
                                        query_ptr, 
                                        child_id, 
                                        child_id != invalid_index
                                    );

            // Store the distance
            const unsigned lane_id = threadIdx.x % TEAM_SIZE;
            if (valid_i && lane_id == 0)
            {
                if (child_id != invalid_index)
                {
                    candidate_distances_ptr[i] = norm2;
                } 
                else
                {
                    candidate_distances_ptr[i] = FLT_MAX;
                }
            }
        }

    }
    else
    {

        DISTANCE_T threshold = candidate_distances_ptr[-1];
        std::uint32_t max_i = GRAPH_DEGREE * SEARCH_WIDTH;
        for (std::uint32_t tid = threadIdx.x; tid < max_i * TEAM_SIZE; tid += blockDim.x)
        {
            const auto i       = tid / TEAM_SIZE;
            const bool valid_i = (i < (GRAPH_DEGREE * SEARCH_WIDTH));
            INDEX_T child_id   = invalid_index;
            if (valid_i) { child_id = candidate_indices_ptr[i]; }

            DISTANCE_T norm2 = compute_similarity_vector_load_partial<VECTOR_DIM, TEAM_SIZE, 4/*compute 75%*/>
                                    (
                                        d_dataset_ptr, 
                                        query_ptr, 
                                        child_id, 
                                        child_id != invalid_index
                                    );

            // Store the distance
            const unsigned lane_id = threadIdx.x % TEAM_SIZE;
            if (valid_i && lane_id == 0)
            {
                if (child_id != invalid_index && norm2 < threshold)
                {
                    candidate_distances_ptr[i] = norm2;
                } 
                else
                {
                    candidate_distances_ptr[i] = FLT_MAX;
                }
            }
        }

        // atomic add
        __syncthreads();
        const uint32_t _TEAM_SIZE = TEAM_SIZE;
        uint8_t _lane_id = threadIdx.x % _TEAM_SIZE;
        uint8_t _team_id = threadIdx.x / _TEAM_SIZE;
        unsigned _team_mask = 0xff << (_team_id * _TEAM_SIZE);
        uint32_t _offset_e_maxi = ((VECTOR_DIM + _TEAM_SIZE * 4 - 1) / (_TEAM_SIZE * 4) + 4 -1) / 4;
        uint32_t _max_i = _offset_e_maxi * CANDIDATE_BUFFER_SIZE;
        // uint32_t _max_i = ((VECTOR_DIM + _TEAM_SIZE * 4 - 1) / (_TEAM_SIZE * 4/*float4*/) + FULL_COMPUTE_RATIO - 1)/ FULL_COMPUTE_RATIO /*75% is alreadly computed*/ * CANDIDATE_BUFFER_SIZE;
        while(true)
        {   
            uint32_t i = 0;
            if(_lane_id == 0)
            {
                while(true)
                {
                    i =  atomicAdd(counter, 1);
                    if(i >= _max_i)
                    {
                        break;
                    }
                    if(candidate_distances_ptr[i % CANDIDATE_BUFFER_SIZE] != FLT_MAX)
                    {
                        break;
                    }
                }
            }
            i = __shfl_sync(_team_mask, i, _team_id * _TEAM_SIZE);
            
            if(i >= _max_i)
            {
                break;
            }
            
            uint32_t vector_offset = i / CANDIDATE_BUFFER_SIZE;
            i = i % CANDIDATE_BUFFER_SIZE;
     
            INDEX_T child_id = candidate_indices_ptr[i];

            // Distance calculation
            uint32_t e = _offset_e_maxi * (4 - 1);
            // uint32_t e =((VECTOR_DIM + _TEAM_SIZE * 4 - 1) / (_TEAM_SIZE * 4) + FULL_COMPUTE_RATIO -1) / FULL_COMPUTE_RATIO * (FULL_COMPUTE_RATIO -1);
            DISTANCE_T partial_norm2 = compute_similarity_vector_load_by_team_partial_float4<VECTOR_DIM, _TEAM_SIZE>
                                        (
                                            d_dataset_ptr,
                                            query_ptr,
                                            child_id,
                                            _team_mask,
                                            e + vector_offset               // vector offset
                                        );
            
            if (_lane_id == 0)
            {
                DISTANCE_T saved_norm2 = candidate_distances_ptr[i];
                if(saved_norm2 + partial_norm2 > threshold)
                {
                    candidate_distances_ptr[i] = FLT_MAX;
                }
                else
                {
                    candidate_distances_ptr[i] = saved_norm2 + partial_norm2;
                }
            }  
        }

    }

}


template <uint32_t VECTOR_DIM, uint32_t TEAM_SIZE, uint32_t INTERNAL_TOPK>
__device__ inline void compute_distance_to_child_nodes_team_with_direction
(
    INDEX_T* parent_buffer,
    INDEX_T* internal_topk_list,
    INDEX_T* candidate_indices_ptr,
    DISTANCE_T* candidate_distances_ptr,
    DATA_T* query_ptr,
    DATA_T* d_dataset_ptr,
    INDEX_T* d_graph_ptr,
    uint32_t* d_sign_bit_ptr,
    uint32_t GRAPH_DEGREE,
    uint32_t CANDIDATE_BUFFER_SIZE,
    uint32_t PRUNE_CONFIG,
    float PRUNE_RATIO
)
{
    constexpr INDEX_T index_msb_1_mask = 0x80000000;
    const INDEX_T invalid_index        = std::numeric_limits<INDEX_T>::max();

    const INDEX_T smem_parent_id = parent_buffer[0];
    const auto parent_id = internal_topk_list[smem_parent_id] & ~index_msb_1_mask;
    
    int8_t lane_id = threadIdx.x % 32;
    uint32_t sign_bit_vector_size = VECTOR_DIM * GRAPH_DEGREE / 32; // 256 for sift-128
    constexpr uint32_t packed_vector_dim_size = (VECTOR_DIM + 31) / 32;

    // uint32_t query_sign_bit[VECTOR_DIM / 32];
    int32_t query_sign_bit[packed_vector_dim_size];

    uint32_t VECTOR_DIM_32 = (VECTOR_DIM + 31) & ~31;
    for (uint32_t i = threadIdx.x; i < VECTOR_DIM_32; i += blockDim.x)
    {
        const bool valid_i = i < VECTOR_DIM;
        uint32_t sign_bit = query_ptr[i] > d_dataset_ptr[parent_id * VECTOR_DIM + i];

        sign_bit <<= (31 - lane_id);
        unsigned active_threads = __ballot_sync(0xffffffff, valid_i);
        __syncwarp(active_threads);

        for (int offset = 16; offset > 0; offset /= 2) 
        {
            sign_bit |= __shfl_xor_sync(0xffffffff, sign_bit, offset);
        }
        
     
        // query_sign_bit[i / (VECTOR_DIM / 4)] = sign_bit;
        // query_sign_bit[i / (VECTOR_DIM / packed_vector_dim_size)] = sign_bit;
        query_sign_bit[i / 32] = sign_bit;

        __syncwarp(active_threads);
    }

    for (uint32_t i = threadIdx.x; i < CANDIDATE_BUFFER_SIZE; i += blockDim.x)
    {

        // const auto parent_id = internal_topk_list[smem_parent_id] & ~index_msb_1_mask;
        INDEX_T child_id = d_graph_ptr[(static_cast<int64_t>(GRAPH_DEGREE) * parent_id) + i];
        // check with internal topk
        for (int32_t internal_topk_ptr = 0; internal_topk_ptr < INTERNAL_TOPK; internal_topk_ptr++)
        {
            INDEX_T internal_topk_id = internal_topk_list[internal_topk_ptr] & ~index_msb_1_mask;
            if(child_id == internal_topk_id)
            {
                child_id = invalid_index;
                break;
            }
        }

        
        if (child_id != invalid_index)
        {
            if(PRUNE_CONFIG == RANDOM_PRUNE)
            {
                uint32_t gid = threadIdx.x + blockDim.x * i;
                curandState state;
                curand_init(clock64(), gid, 0, &state);
                DISTANCE_T direction = (curand_uniform(&state) - 0.5) * VECTOR_DIM;
                candidate_distances_ptr[i] = direction;
                
                // int direction = 0; 
                // for (int j = 0; j < packed_vector_dim_size; j++)
                // {
                //     if (j < (packed_vector_dim_size / 4))
                //         direction -= __popcll(query_sign_bit[j] ^ d_sign_bit_ptr[parent_id * sign_bit_vector_size + i * packed_vector_dim_size + j]);
                //     else
                //         direction = 10;
                // }
                // candidate_distances_ptr[i] = static_cast<float>(direction);
            }
            else if(PRUNE_CONFIG == SIGN_BIT_PRUNE)
            {
                int direction = 0; 
                for (int j = 0; j < packed_vector_dim_size; j++)
                {
                    direction -= __popcll(query_sign_bit[j] ^ d_sign_bit_ptr[parent_id * sign_bit_vector_size + i * packed_vector_dim_size + j]);
                }
                candidate_distances_ptr[i] = static_cast<float>(direction);
            }
    
        }
        else
        {
            candidate_distances_ptr[i] = -99999999.0;
        }
        candidate_indices_ptr[i] = child_id;

    }

    __syncwarp(0xffffffff);

    // sort
    candidate_by_bitonic_sort_inverse<2, INTERNAL_TOPK / 32>
        (
            candidate_indices_ptr,
            candidate_distances_ptr,
            CANDIDATE_BUFFER_SIZE
        );

    __syncwarp(0xffffffff);

    uint32_t max_tid = CANDIDATE_BUFFER_SIZE * TEAM_SIZE * PRUNE_RATIO + CANDIDATE_BUFFER_SIZE - CANDIDATE_BUFFER_SIZE * PRUNE_RATIO;
    for (uint32_t tid = threadIdx.x; tid < max_tid ; tid += blockDim.x)
    {
        if (tid < CANDIDATE_BUFFER_SIZE * TEAM_SIZE * PRUNE_RATIO)
        {
            uint32_t i = tid / TEAM_SIZE;
            INDEX_T child_id = candidate_indices_ptr[i];

            // Distance calculation
            DISTANCE_T norm2 = compute_similarity_vector_load_full<VECTOR_DIM, TEAM_SIZE>
                                        (
                                            d_dataset_ptr,
                                            query_ptr,
                                            child_id,
                                            child_id != invalid_index
                                        );
            
            uint8_t lane_id = threadIdx.x % TEAM_SIZE;
            if (lane_id == 0)
            {
                if(child_id != invalid_index)
                {
                    candidate_distances_ptr[i] = norm2;
                }
                else
                {
                    candidate_distances_ptr[i] = std::numeric_limits<DISTANCE_T>::max();
                }
            }
        }
        else
        {
            uint32_t i = tid - CANDIDATE_BUFFER_SIZE * TEAM_SIZE * PRUNE_RATIO + CANDIDATE_BUFFER_SIZE * PRUNE_RATIO;
            candidate_distances_ptr[i] = std::numeric_limits<DISTANCE_T>::max();
        }

    }

}

/////////////////////////////////////////////////////// main search kernel ///////////////////////////////////////////////////////

template<uint32_t VECTOR_DIM, uint32_t INTERNAL_TOPK>
__global__ void search_kernel
(
    INDEX_T* d_graph_ptr,
    DATA_T* d_dataset_ptr,
    DATA_T* d_queries_ptr,
    INDEX_T* d_top10_ptr,
    INDEX_T* d_seed_map_ptr,
    uint32_t* d_sign_bit_ptr,
    INDEX_T* d_results_ptr,
    DISTANCE_T* d_distances_ptr,
    uint32_t NUM_QUERIES,
    uint32_t TOPK,
    uint32_t SEARCH_WIDTH,
    uint32_t MAX_ITER,
    uint32_t MIN_ITER,
    uint32_t DATASET_SIZE,
    uint32_t TEAM_SIZE,
    uint32_t GRAPH_DEGREE,
    uint32_t NUM_DIST,
    uint32_t BLOCK_SIZE,
    uint32_t BITLEN,
    uint32_t SMALL_HASH_RESET_INTERVAL,
    uint32_t PRUNE_CONFIG,
    float ITERATION_DIRECTION_RATIO,
    float PRUNE_RATIO,
    uint32_t THRESHOLD_CONFIG,
    uint32_t FULL_COMPUTE_RATIO,
    uint32_t SEED_CONFIG,
    uint32_t HASH_TABLE_CONFIG,
    uint32_t SEED_TOPK_SIZE
)
{

    if(TEAM_SIZE != 8)
    {
        printf("Invalid TEAM_SIZE\n");
        assert(false);
    }

#ifdef _CLK_BREAKDOWN
  std::uint64_t clk_init                 = 0;
  std::uint64_t clk_compute_1st_distance = 0;
  std::uint64_t clk_sort                 = 0;
  std::uint64_t clk_reset_hash           = 0;
  std::uint64_t clk_pickup_parents       = 0;
  std::uint64_t clk_restore_hash         = 0;
  std::uint64_t clk_compute_distance     = 0;
  std::uint64_t clk_start;
#define _CLK_START() clk_start = clock64()
#define _CLK_REC(V)  V += clock64() - clk_start;
#else
#define _CLK_START()
#define _CLK_REC(V)
#endif

    _CLK_START();
    const uint32_t query_id = blockIdx.y;

    // Shared memory initialization
    extern __shared__ uint32_t smem[];

    uint32_t RESULT_BUFFER_SIZE = INTERNAL_TOPK + SEARCH_WIDTH * GRAPH_DEGREE;
    uint32_t CANDIDATE_BUFFER_SIZE = SEARCH_WIDTH * GRAPH_DEGREE;
    uint32_t QUERY_BUFFER_SIZE = VECTOR_DIM;

    uint32_t hash_table_size = 1 << BITLEN;

    // buffer
    auto query_buffer = reinterpret_cast<DATA_T*>(smem);
    auto result_indices_buffer = reinterpret_cast<INDEX_T*>(query_buffer + QUERY_BUFFER_SIZE);
    auto result_distances_buffer = reinterpret_cast<DISTANCE_T*>(result_indices_buffer + RESULT_BUFFER_SIZE);
    auto visited_hash_buffer = reinterpret_cast<INDEX_T*>(result_distances_buffer + RESULT_BUFFER_SIZE);
    auto parent_list_buffer = reinterpret_cast<INDEX_T*>(visited_hash_buffer + hash_table_size);
    auto terminate_flag = reinterpret_cast<uint32_t*>(parent_list_buffer + SEARCH_WIDTH);
    // flags
    terminate_flag[0] = 0;
    // counter
    auto counter = reinterpret_cast<uint32_t*>(terminate_flag + 1);
    
    if(HASH_TABLE_CONFIG)
    {
        // Hash table initialization
        hashtable_init(visited_hash_buffer, BITLEN);
    }

    // Load query
    for (uint32_t i = threadIdx.x; i < VECTOR_DIM; i += blockDim.x)
    {   
        if (i < VECTOR_DIM)
        {
            query_buffer[i] = d_queries_ptr[query_id * VECTOR_DIM + i];
        }
    }

    // Initialize result buffer
    for (uint32_t i = threadIdx.x; i < RESULT_BUFFER_SIZE; i += blockDim.x)
    {
        if (i < RESULT_BUFFER_SIZE)
        {
            result_indices_buffer[i] = std::numeric_limits<INDEX_T>::max();
            result_distances_buffer[i] = std::numeric_limits<DISTANCE_T>::max();
        }
    }

    __syncthreads();
    _CLK_REC(clk_init);

    _CLK_START();

    if (SEED_CONFIG == NO_SEED)
    {
        // Compute distance to randomly selecting nodes
        compute_distance_to_random_nodes<VECTOR_DIM, 8>
        (
            result_indices_buffer + INTERNAL_TOPK,
            result_distances_buffer + INTERNAL_TOPK,
            query_buffer,
            d_dataset_ptr,
            DATASET_SIZE,
            CANDIDATE_BUFFER_SIZE,
            NUM_DIST,
            visited_hash_buffer,
            BITLEN,
            HASH_TABLE_CONFIG
        );
    }
    else
    {
        // map top1 nodes and fetch neighbors and calculate distance
        compute_distance_to_maped_top10_nodes<VECTOR_DIM, 8>
        (
            result_indices_buffer + INTERNAL_TOPK,
            result_distances_buffer + INTERNAL_TOPK,
            query_buffer,
            d_dataset_ptr,
            d_graph_ptr,
            d_top10_ptr,
            d_seed_map_ptr,
            GRAPH_DEGREE,
            DATASET_SIZE,
            CANDIDATE_BUFFER_SIZE,
            visited_hash_buffer,
            BITLEN,
            SEED_CONFIG,
            HASH_TABLE_CONFIG,
            SEED_TOPK_SIZE
        );
    }
    __syncthreads();
    _CLK_REC(clk_compute_1st_distance);

    uint32_t iter = 0;
    while (1)
    {
        
        if(HASH_TABLE_CONFIG)
        {
            _CLK_START();
            if ((iter + 1) % SMALL_HASH_RESET_INTERVAL == 0)
            {
                hashtable_init(visited_hash_buffer, BITLEN);
            }
            __syncthreads();
            _CLK_REC(clk_reset_hash);
        }

        // Sort
        _CLK_START();
        topk_by_bitonic_sort<2, INTERNAL_TOPK / 32>
            (
                result_indices_buffer,
                result_distances_buffer,
                RESULT_BUFFER_SIZE,
                CANDIDATE_BUFFER_SIZE,
                INTERNAL_TOPK,
                (iter == 0)
            );

        __syncthreads();
        _CLK_REC(clk_sort);

        if(iter + 1 == MAX_ITER)
        {
            break;
        }

        // Pick up next parents
        if (threadIdx.x < 32)
        {
            _CLK_START();
            pickup_next_parents
                (
                    terminate_flag,
                    parent_list_buffer,
                    result_indices_buffer,
                    INTERNAL_TOPK,
                    SEARCH_WIDTH
                );
            _CLK_REC(clk_pickup_parents);
        }
        

        if(HASH_TABLE_CONFIG)
        {
            // Restore hash table by putting internal-topk indices in it
            _CLK_START();
            if ((iter + 1) % SMALL_HASH_RESET_INTERVAL == 0)
            {
                const unsigned first_tid = ((blockDim.x <= 32) ? 0 : 32);
                hashtable_restore
                    (
                        visited_hash_buffer,
                        BITLEN,
                        result_indices_buffer,
                        INTERNAL_TOPK,
                        first_tid
                    );
            }
            __syncthreads();
            _CLK_REC(clk_restore_hash);
        }

        if (*terminate_flag && iter >= MIN_ITER)
        {
            break;
        }

        // Compute distance to child nodes and query node
        _CLK_START();
        if (threadIdx.x == 0) {
            counter[0] = 0;
        }
        __syncthreads();

        if (PRUNE_CONFIG != NO_PRUNE && iter < MAX_ITER * ITERATION_DIRECTION_RATIO)
        {
            compute_distance_to_child_nodes_team_with_direction<VECTOR_DIM, 8, INTERNAL_TOPK>
            (
                parent_list_buffer,
                result_indices_buffer,
                result_indices_buffer + INTERNAL_TOPK,
                result_distances_buffer + INTERNAL_TOPK,
                query_buffer,
                d_dataset_ptr,
                d_graph_ptr,
                d_sign_bit_ptr,
                GRAPH_DEGREE,
                CANDIDATE_BUFFER_SIZE,
                PRUNE_CONFIG,
                PRUNE_RATIO
            );
        }
        else
        {
            compute_distance_to_child_nodes<VECTOR_DIM, 8, INTERNAL_TOPK>
            (
                parent_list_buffer,
                result_indices_buffer + INTERNAL_TOPK,
                result_distances_buffer + INTERNAL_TOPK,
                result_indices_buffer,
                query_buffer,
                d_dataset_ptr,
                d_graph_ptr,
                GRAPH_DEGREE,
                SEARCH_WIDTH,
                CANDIDATE_BUFFER_SIZE,
                NUM_DIST,
                visited_hash_buffer,
                BITLEN,
                counter,
                THRESHOLD_CONFIG,
                FULL_COMPUTE_RATIO,
                HASH_TABLE_CONFIG
            );  
        }

        __syncthreads();
        _CLK_REC(clk_compute_distance);

        iter++;
    }

    // Sorting
    topk_by_bitonic_sort<2, INTERNAL_TOPK / 32>
            (
                result_indices_buffer,
                result_distances_buffer,
                RESULT_BUFFER_SIZE,
                CANDIDATE_BUFFER_SIZE,
                INTERNAL_TOPK,
                false
            );
    __syncthreads();

    // Write results
    for (uint32_t i = threadIdx.x; i < TOPK; i += blockDim.x)
    {
        d_results_ptr[query_id * TOPK + i] = result_indices_buffer[i] & ~0x80000000;
        d_distances_ptr[query_id * TOPK + i] = result_distances_buffer[i];
    }

    #ifdef _CLK_BREAKDOWN
    if (threadIdx.x == 0 && query_id == 0)
    {
        printf(
        "query, %d, thread, %d"
        ", init, %lu"
        ", 1st_distance, %lu"
        ", topk, %lu"
        ", reset_hash, %lu"
        ", pickup_parents, %lu"
        ", restore_hash, %lu"
        ", distance, %lu"
        "\n",
        query_id,
        threadIdx.x,
        clk_init,
        clk_compute_1st_distance,
        clk_sort,
        clk_reset_hash,
        clk_pickup_parents,
        clk_restore_hash,
        clk_compute_distance);
    }
    #endif
}

torch::Tensor search
(
    torch::Tensor graph, 
    torch::Tensor dataset, 
    torch::Tensor queries, 
    torch::Tensor top10,
    torch::Tensor seed_map,
    torch::Tensor sign_bit, 
    torch::Tensor configs,
    torch::Tensor results_distances
) 
{
    
    if (!graph.is_cuda())
    {
        throw std::runtime_error("Input tensor 'graph' must be on GPU.");
    }
    if (!dataset.is_cuda())
    {
        throw std::runtime_error("Input tensor 'dataset' must be on GPU.");
    }
    if (!queries.is_cuda())
    {
        throw std::runtime_error("Input tensor 'queries' must be on GPU.");
    }
    if (!top10.is_cuda())
    {
        throw std::runtime_error("Input tensor 'top10' must be on GPU.");
    }
    if (!seed_map.is_cuda())
    {
        throw std::runtime_error("Input tensor 'seed_map' must be on GPU.");
    }
    if (!sign_bit.is_cuda())
    {
        throw std::runtime_error("Input tensor 'sign_bit' must be on GPU.");
    }
    if (!configs.is_cuda())
    {
        throw std::runtime_error("Input tensor 'configs' must be on GPU.");
    }
    
    uint32_t NUM_QUERIES = configs[0].item<int>();
    uint32_t TOPK = configs[1].item<int>();
    uint32_t SEARCH_WIDTH = configs[2].item<int>();
    uint32_t MAX_ITER = configs[3].item<int>();
    uint32_t MIN_ITER = configs[4].item<int>();
    uint32_t INTERNAL_TOPK = configs[5].item<int>();
    uint32_t VECTOR_DIM = configs[6].item<int>();
    uint32_t DATASET_SIZE = configs[7].item<int>();
    uint32_t TEAM_SIZE = configs[8].item<int>();
    uint32_t GRAPH_DEGREE = configs[9].item<int>();
    uint32_t NUM_DIST = configs[10].item<int>();
    uint32_t BLOCK_SIZE = configs[11].item<int>();
    uint32_t BITLEN = configs[12].item<int>();
    uint32_t SMALL_HASH_RESET_INTERVAL = configs[13].item<int>();
    uint32_t SHARED_MEM_SIZE = configs[14].item<int>();
    // DIRECTION
    uint32_t PRUNE_CONFIG = configs[15].item<int>(); 
    float ITERATION_DIRECTION_RATIO = configs[16].item<float>();
    float PRUNE_RATIO = configs[17].item<float>();
    // THRESHOLD
    uint32_t THRESHOLD_CONFIG = configs[18].item<int>();
    uint32_t FULL_COMPUTE_RATIO = configs[19].item<int>();
    // SEED
    uint32_t SEED_CONFIG = configs[20].item<int>();
    // HASH TABLE
    uint32_t HASH_TABLE_CONFIG = configs[21].item<int>();
    // SEED_COUPLING
    uint32_t SEED_TOPK_SIZE;
    try {
        SEED_TOPK_SIZE = configs[22].item<int>();
    } catch (const std::exception& e) {
        SEED_TOPK_SIZE = 1;
    }

    torch::Tensor results_indices = torch::zeros({NUM_QUERIES, TOPK}, torch::device(queries.device()).dtype(torch::kUInt32));

    // pointer
    INDEX_T* d_graph_ptr = graph.data_ptr<INDEX_T>();
    DATA_T* d_dataset_ptr = dataset.data_ptr<DATA_T>();
    DATA_T* d_queries_ptr = queries.data_ptr<DATA_T>();
    INDEX_T* d_top10_ptr = top10.data_ptr<INDEX_T>();
    INDEX_T* d_seed_map_ptr = seed_map.data_ptr<INDEX_T>();
    uint32_t* d_sign_bit_ptr = sign_bit.data_ptr<uint32_t>();
    INDEX_T* d_results_indices_ptr = results_indices.data_ptr<INDEX_T>();
    DISTANCE_T* d_results_distances_ptr = results_distances.data_ptr<DISTANCE_T>();

    // Calculate shaerd mem size
    uint32_t shared_size = SHARED_MEM_SIZE;

    dim3 thread_dims(BLOCK_SIZE, 1, 1);
    dim3 block_dims(1, NUM_QUERIES, 1);
    
    switch (VECTOR_DIM)
    {
        case 128:
            // sift-128-euclidean
            if(INTERNAL_TOPK == 64)
            {
                search_kernel<128, 64><<<block_dims, thread_dims, shared_size>>>
                (
                    d_graph_ptr,
                    d_dataset_ptr,
                    d_queries_ptr,
                    d_top10_ptr,
                    d_seed_map_ptr,
                    d_sign_bit_ptr,
                    d_results_indices_ptr,
                    d_results_distances_ptr,
                    NUM_QUERIES,
                    TOPK,
                    SEARCH_WIDTH,
                    MAX_ITER,
                    MIN_ITER,
                    DATASET_SIZE,
                    TEAM_SIZE,
                    GRAPH_DEGREE,
                    NUM_DIST,
                    BLOCK_SIZE,
                    BITLEN,
                    SMALL_HASH_RESET_INTERVAL,
                    PRUNE_CONFIG,
                    ITERATION_DIRECTION_RATIO,
                    PRUNE_RATIO,
                    THRESHOLD_CONFIG,
                    FULL_COMPUTE_RATIO,
                    SEED_CONFIG,
                    HASH_TABLE_CONFIG,
                    SEED_TOPK_SIZE
                );
            }
            else if(INTERNAL_TOPK == 128)
            {
                search_kernel<128, 128><<<block_dims, thread_dims, shared_size>>>
                (
                    d_graph_ptr,
                    d_dataset_ptr,
                    d_queries_ptr,
                    d_top10_ptr,
                    d_seed_map_ptr,
                    d_sign_bit_ptr,
                    d_results_indices_ptr,
                    d_results_distances_ptr,
                    NUM_QUERIES,
                    TOPK,
                    SEARCH_WIDTH,
                    MAX_ITER,
                    MIN_ITER,
                    DATASET_SIZE,
                    TEAM_SIZE,
                    GRAPH_DEGREE,
                    NUM_DIST,
                    BLOCK_SIZE,
                    BITLEN,
                    SMALL_HASH_RESET_INTERVAL,
                    PRUNE_CONFIG,
                    ITERATION_DIRECTION_RATIO,
                    PRUNE_RATIO,
                    THRESHOLD_CONFIG,
                    FULL_COMPUTE_RATIO,
                    SEED_CONFIG,
                    HASH_TABLE_CONFIG,
                    SEED_TOPK_SIZE
                );
            }
            else
            {
                printf("Invalid INTERNAL_TOPK\n");
                assert(false);
            }
            break;
        case 960:
            // gist-960-euclidean
            if(INTERNAL_TOPK == 64)
            {
                search_kernel<960, 64><<<block_dims, thread_dims, shared_size>>>
                (
                    d_graph_ptr,
                    d_dataset_ptr,
                    d_queries_ptr,
                    d_top10_ptr,
                    d_seed_map_ptr,
                    d_sign_bit_ptr,
                    d_results_indices_ptr,
                    d_results_distances_ptr,
                    NUM_QUERIES,
                    TOPK,
                    SEARCH_WIDTH,
                    MAX_ITER,
                    MIN_ITER,
                    DATASET_SIZE,
                    TEAM_SIZE,
                    GRAPH_DEGREE,
                    NUM_DIST,
                    BLOCK_SIZE,
                    BITLEN,
                    SMALL_HASH_RESET_INTERVAL,
                    PRUNE_CONFIG,
                    ITERATION_DIRECTION_RATIO,
                    PRUNE_RATIO,
                    THRESHOLD_CONFIG,
                    FULL_COMPUTE_RATIO,
                    SEED_CONFIG,
                    HASH_TABLE_CONFIG,
                    SEED_TOPK_SIZE
                );
            }
            else if(INTERNAL_TOPK == 128)
            {
                search_kernel<960, 128><<<block_dims, thread_dims, shared_size>>>
                (
                    d_graph_ptr,
                    d_dataset_ptr,
                    d_queries_ptr,
                    d_top10_ptr,
                    d_seed_map_ptr,
                    d_sign_bit_ptr,
                    d_results_indices_ptr,
                    d_results_distances_ptr,
                    NUM_QUERIES,
                    TOPK,
                    SEARCH_WIDTH,
                    MAX_ITER,
                    MIN_ITER,
                    DATASET_SIZE,
                    TEAM_SIZE,
                    GRAPH_DEGREE,
                    NUM_DIST,
                    BLOCK_SIZE,
                    BITLEN,
                    SMALL_HASH_RESET_INTERVAL,
                    PRUNE_CONFIG,
                    ITERATION_DIRECTION_RATIO,
                    PRUNE_RATIO,
                    THRESHOLD_CONFIG,
                    FULL_COMPUTE_RATIO,
                    SEED_CONFIG,
                    HASH_TABLE_CONFIG,
                    SEED_TOPK_SIZE
                );
            }
            else
            {
                printf("Invalid INTERNAL_TOPK\n");
                assert(false);
            }
            break;
        case 100:
            // glove-100-inner            
            if(INTERNAL_TOPK == 64)
            {
                search_kernel<100, 64><<<block_dims, thread_dims, shared_size>>>
                (
                    d_graph_ptr,
                    d_dataset_ptr,
                    d_queries_ptr,
                    d_top10_ptr,
                    d_seed_map_ptr,
                    d_sign_bit_ptr,
                    d_results_indices_ptr,
                    d_results_distances_ptr,
                    NUM_QUERIES,
                    TOPK,
                    SEARCH_WIDTH,
                    MAX_ITER,
                    MIN_ITER,
                    DATASET_SIZE,
                    TEAM_SIZE,
                    GRAPH_DEGREE,
                    NUM_DIST,
                    BLOCK_SIZE,
                    BITLEN,
                    SMALL_HASH_RESET_INTERVAL,
                    PRUNE_CONFIG,
                    ITERATION_DIRECTION_RATIO,
                    PRUNE_RATIO,
                    THRESHOLD_CONFIG,
                    FULL_COMPUTE_RATIO,
                    SEED_CONFIG,
                    HASH_TABLE_CONFIG,
                    SEED_TOPK_SIZE
                );
            }
            else if(INTERNAL_TOPK == 128)
            {
                search_kernel<100, 128><<<block_dims, thread_dims, shared_size>>>
                (
                    d_graph_ptr,
                    d_dataset_ptr,
                    d_queries_ptr,
                    d_top10_ptr,
                    d_seed_map_ptr,
                    d_sign_bit_ptr,
                    d_results_indices_ptr,
                    d_results_distances_ptr,
                    NUM_QUERIES,
                    TOPK,
                    SEARCH_WIDTH,
                    MAX_ITER,
                    MIN_ITER,
                    DATASET_SIZE,
                    TEAM_SIZE,
                    GRAPH_DEGREE,
                    NUM_DIST,
                    BLOCK_SIZE,
                    BITLEN,
                    SMALL_HASH_RESET_INTERVAL,
                    PRUNE_CONFIG,
                    ITERATION_DIRECTION_RATIO,
                    PRUNE_RATIO,
                    THRESHOLD_CONFIG,
                    FULL_COMPUTE_RATIO,
                    SEED_CONFIG,
                    HASH_TABLE_CONFIG,
                    SEED_TOPK_SIZE
                );
            }
            else
            {
                printf("Invalid INTERNAL_TOPK\n");
                assert(false);
            }
            break;
        case 256:
            // nytimes-256-inner
            if(INTERNAL_TOPK == 64)
            {
                search_kernel<256, 64><<<block_dims, thread_dims, shared_size>>>
                (
                    d_graph_ptr,
                    d_dataset_ptr,
                    d_queries_ptr,
                    d_top10_ptr,
                    d_seed_map_ptr,
                    d_sign_bit_ptr,
                    d_results_indices_ptr,
                    d_results_distances_ptr,
                    NUM_QUERIES,
                    TOPK,
                    SEARCH_WIDTH,
                    MAX_ITER,
                    MIN_ITER,
                    DATASET_SIZE,
                    TEAM_SIZE,
                    GRAPH_DEGREE,
                    NUM_DIST,
                    BLOCK_SIZE,
                    BITLEN,
                    SMALL_HASH_RESET_INTERVAL,
                    PRUNE_CONFIG,
                    ITERATION_DIRECTION_RATIO,
                    PRUNE_RATIO,
                    THRESHOLD_CONFIG,
                    FULL_COMPUTE_RATIO,
                    SEED_CONFIG,
                    HASH_TABLE_CONFIG,
                    SEED_TOPK_SIZE
                );
            }
            else if(INTERNAL_TOPK == 128)
            {
                search_kernel<256, 128><<<block_dims, thread_dims, shared_size>>>
                (
                    d_graph_ptr,
                    d_dataset_ptr,
                    d_queries_ptr,
                    d_top10_ptr,
                    d_seed_map_ptr,
                    d_sign_bit_ptr,
                    d_results_indices_ptr,
                    d_results_distances_ptr,
                    NUM_QUERIES,
                    TOPK,
                    SEARCH_WIDTH,
                    MAX_ITER,
                    MIN_ITER,
                    DATASET_SIZE,
                    TEAM_SIZE,
                    GRAPH_DEGREE,
                    NUM_DIST,
                    BLOCK_SIZE,
                    BITLEN,
                    SMALL_HASH_RESET_INTERVAL,
                    PRUNE_CONFIG,
                    ITERATION_DIRECTION_RATIO,
                    PRUNE_RATIO,
                    THRESHOLD_CONFIG,
                    FULL_COMPUTE_RATIO,
                    SEED_CONFIG,
                    HASH_TABLE_CONFIG,
                    SEED_TOPK_SIZE
                );
            }
            else
            {
                printf("Invalid INTERNAL_TOPK\n");
                assert(false);
            }
            break;
        case 96:
            // deep-image-96-inner
            if(INTERNAL_TOPK == 64)
            {
                search_kernel<96, 64><<<block_dims, thread_dims, shared_size>>>
                (
                    d_graph_ptr,
                    d_dataset_ptr,
                    d_queries_ptr,
                    d_top10_ptr,
                    d_seed_map_ptr,
                    d_sign_bit_ptr,
                    d_results_indices_ptr,
                    d_results_distances_ptr,
                    NUM_QUERIES,
                    TOPK,
                    SEARCH_WIDTH,
                    MAX_ITER,
                    MIN_ITER,
                    DATASET_SIZE,
                    TEAM_SIZE,
                    GRAPH_DEGREE,
                    NUM_DIST,
                    BLOCK_SIZE,
                    BITLEN,
                    SMALL_HASH_RESET_INTERVAL,
                    PRUNE_CONFIG,
                    ITERATION_DIRECTION_RATIO,
                    PRUNE_RATIO,
                    THRESHOLD_CONFIG,
                    FULL_COMPUTE_RATIO,
                    SEED_CONFIG,
                    HASH_TABLE_CONFIG,
                    SEED_TOPK_SIZE
                );
            }
            else if(INTERNAL_TOPK == 128)
            {
                search_kernel<96, 128><<<block_dims, thread_dims, shared_size>>>
                (
                    d_graph_ptr,
                    d_dataset_ptr,
                    d_queries_ptr,
                    d_top10_ptr,
                    d_seed_map_ptr,
                    d_sign_bit_ptr,
                    d_results_indices_ptr,
                    d_results_distances_ptr,
                    NUM_QUERIES,
                    TOPK,
                    SEARCH_WIDTH,
                    MAX_ITER,
                    MIN_ITER,
                    DATASET_SIZE,
                    TEAM_SIZE,
                    GRAPH_DEGREE,
                    NUM_DIST,
                    BLOCK_SIZE,
                    BITLEN,
                    SMALL_HASH_RESET_INTERVAL,
                    PRUNE_CONFIG,
                    ITERATION_DIRECTION_RATIO,
                    PRUNE_RATIO,
                    THRESHOLD_CONFIG,
                    FULL_COMPUTE_RATIO,
                    SEED_CONFIG,
                    HASH_TABLE_CONFIG,
                    SEED_TOPK_SIZE
                );
            }
            else
            {
                printf("Invalid INTERNAL_TOPK\n");
                assert(false);
            }
            break;
        case 768:
            // wiki-all-10M
            if(INTERNAL_TOPK == 64)
            {
                search_kernel<768, 64><<<block_dims, thread_dims, shared_size>>>
                (
                    d_graph_ptr,
                    d_dataset_ptr,
                    d_queries_ptr,
                    d_top10_ptr,
                    d_seed_map_ptr,
                    d_sign_bit_ptr,
                    d_results_indices_ptr,
                    d_results_distances_ptr,
                    NUM_QUERIES,
                    TOPK,
                    SEARCH_WIDTH,
                    MAX_ITER,
                    MIN_ITER,
                    DATASET_SIZE,
                    TEAM_SIZE,
                    GRAPH_DEGREE,
                    NUM_DIST,
                    BLOCK_SIZE,
                    BITLEN,
                    SMALL_HASH_RESET_INTERVAL,
                    PRUNE_CONFIG,
                    ITERATION_DIRECTION_RATIO,
                    PRUNE_RATIO,
                    THRESHOLD_CONFIG,
                    FULL_COMPUTE_RATIO,
                    SEED_CONFIG,
                    HASH_TABLE_CONFIG,
                    SEED_TOPK_SIZE
                );
            }
            else if(INTERNAL_TOPK == 128)
            {
                search_kernel<768, 128><<<block_dims, thread_dims, shared_size>>>
                (
                    d_graph_ptr,
                    d_dataset_ptr,
                    d_queries_ptr,
                    d_top10_ptr,
                    d_seed_map_ptr,
                    d_sign_bit_ptr,
                    d_results_indices_ptr,
                    d_results_distances_ptr,
                    NUM_QUERIES,
                    TOPK,
                    SEARCH_WIDTH,
                    MAX_ITER,
                    MIN_ITER,
                    DATASET_SIZE,
                    TEAM_SIZE,
                    GRAPH_DEGREE,
                    NUM_DIST,
                    BLOCK_SIZE,
                    BITLEN,
                    SMALL_HASH_RESET_INTERVAL,
                    PRUNE_CONFIG,
                    ITERATION_DIRECTION_RATIO,
                    PRUNE_RATIO,
                    THRESHOLD_CONFIG,
                    FULL_COMPUTE_RATIO,
                    SEED_CONFIG,
                    HASH_TABLE_CONFIG,
                    SEED_TOPK_SIZE
                );
            }
            else
            {
                printf("Invalid INTERNAL_TOPK\n");
                assert(false);
            }
            break;
        default:
            printf("Invalid VECTOR_DIM\n");
            assert(false);
        
    }

    return results_indices;
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("search", &search, "A function that performs search on the graph");
}
