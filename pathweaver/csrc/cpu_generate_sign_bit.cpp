#include <pybind11/pybind11.h>
#include <torch/extension.h>

#include <cstdint>
#include <iostream>
#include <algorithm>
#include <vector>
#include <assert.h>
#include <limits>
#include <omp.h>

#include <cuda.h>
#include <cuda_runtime.h>

namespace py = pybind11;

typedef float DATA_T;
typedef uint32_t INDEX_T;
typedef float DISTANCE_T;

torch::Tensor generate_uint32_compressed(torch::Tensor graph, torch::Tensor dataset)
{
  
    uint32_t DATASET_SIZE = dataset.size(0);
    uint32_t VECTOR_DIM = dataset.size(1);
    uint32_t GRAPH_DEGREE = graph.size(1);
    printf("DATASET_SIZE: %u\n", DATASET_SIZE);
    printf("VECTOR_DIM: %u\n", VECTOR_DIM);
    printf("GRAPH_DEGREE: %u\n", GRAPH_DEGREE);

    uint32_t sign_bit_vector_size = VECTOR_DIM * GRAPH_DEGREE / 32;     // VECTOR_DIM * GRAPH_DEGREE / 32 = 256
    uint32_t packed_vector_dim_size = (VECTOR_DIM + 31) / 32;
    torch::Tensor results = torch::zeros({DATASET_SIZE, sign_bit_vector_size}, torch::dtype(torch::kUInt32));
    
    uint32_t* graph_ptr = graph.data_ptr<uint32_t>();
    float* dataset_ptr = dataset.data_ptr<float>();
    uint32_t* results_ptr = results.data_ptr<uint32_t>();
    
    #pragma omp parallel for
    for (uint32_t i = 0; i < DATASET_SIZE; i++)
    {
        for (uint32_t j = 0; j < GRAPH_DEGREE; j++)
        {
            uint32_t neighbor_index = graph_ptr[i * GRAPH_DEGREE + j];
  
            for (uint32_t k = 0; k < VECTOR_DIM; k++)
            {   
                if (dataset_ptr[neighbor_index * VECTOR_DIM + k] > dataset_ptr[i * VECTOR_DIM + k])
                {
                    // uint32_t current_value = results_ptr[i * 256 + j * 4 + k / 32];
                    // uint32_t updated_value = current_value | (1 << (31 - (k % 32)));
                    // results_ptr[i * 256 + j * 4 + k / 32] = updated_value;
                    uint32_t current_value = results_ptr[i * sign_bit_vector_size + j * packed_vector_dim_size + k / 32];
                    uint32_t updated_value = current_value | (1 << (31 - (k % 32)));
                    results_ptr[i * sign_bit_vector_size + j * packed_vector_dim_size + k / 32] = updated_value;
                }
            }
        }
    }

    return results;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("generate_uint32_compressed", &generate_uint32_compressed, "A function that performs search on the graph");
}