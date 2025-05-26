import argparse
import os
import sys
import numpy as np
import torch
import time
import pickle
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from util import load_dataset, load_graph, generate_sign_bit, convert_to_torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../bin/pathweaver')))
import cpu_generate_sign_bit
# import cpu_generate_sign_bit

def save_sign_bit_1(sign_bit_file_path, dataset, graph):

    # memory leak or something else -> OOM
    print("Memory leak or something else -> OOM"); sys.exit(1)
    
    dataset_size = dataset.shape[0]
    vector_dim = dataset.shape[1]
    graph_degree = graph.shape[1]

    # graph_expanded_with_dataset: (dataset_size, vector_dim * graph_degree)
    graph_expanded_with_dataset = np.zeros((dataset_size, vector_dim * graph_degree), dtype=dataset.dtype)

    for i in range(dataset_size):
        neighbors = dataset[graph[i]]  # shape: (graph_degree, vector_dim)
        graph_expanded_with_dataset[i] = neighbors.flatten()

    # dataset: (dataset_size, vector_dim) -> (dataset_size, vector_dim * graph_degree)
    dataset_expanded_with_graph_out_degree = np.tile(dataset, (1, graph_degree))
    direction = graph_expanded_with_dataset - dataset_expanded_with_graph_out_degree

    # slice and concat unpacked sign bit
    num_slices = direction.shape[1] // vector_dim
    slices = [direction[:, i * vector_dim: (i + 1) * vector_dim] for i in range(num_slices)]
    sign_bit_unpacked_and_concat = np.concatenate(slices, axis=0)

    # pack sign bit
    final_unpacked = sign_bit_unpacked_and_concat.reshape(dataset_size, vector_dim * num_slices)

    total_bits = final_unpacked.shape[1]
    n_uint32 = (total_bits + 31) // 32
    padded_length = n_uint32 * 32

    if padded_length > total_bits:
        pad_width = padded_length - total_bits
        final_unpacked = np.pad(final_unpacked, ((0, 0), (0, pad_width)), mode='constant', constant_values=0)

    final_unpacked = final_unpacked.reshape(dataset_size, n_uint32, 32)

    weights = (2 ** np.arange(31, -1, -1)).astype(np.uint32)
    compressed_sign_bit = np.dot(final_unpacked, weights)

    with open(sign_bit_file_path, 'wb') as f:
        pickle.dump(graph, f)
        print("File successfully saved")
    
    return

def save_sign_bit_2(sign_bit_file_path, dataset, graph):

    graph = convert_to_torch(graph)
    dataset = convert_to_torch(dataset)

    start = time.time()
    sign_bit = cpu_generate_sign_bit.generate_uint32_compressed(graph, dataset)
    end = time.time()

    print(f"Sign bit generation time: {end-start}")

    sign_bit = sign_bit.to(torch.uint32)
    sign_bit = sign_bit.cpu().numpy()

    with open(sign_bit_file_path, "wb") as f: f.write(sign_bit.tobytes())

    return

def main():
    parser = argparse.ArgumentParser(description='graph name')
    parser.add_argument('--dataset', type=str, default='nytimes-256-inner')
    parser.add_argument('--dataset_size', type=str, default='1M')
    parser.add_argument('--num_shard', type=int)
    parser.add_argument('--i', type=str)
    args = parser.parse_args()

    dataset_path = f'../../_datasets/{args.dataset_size}'
    graph_path = f'../../_graph/{args.dataset_size}'
    sign_path = f'../../_sign/{args.dataset_size}'

    # shard
    num_shard = args.num_shard
    if num_shard == 1:
        # no shard
        build_config = 'cagra-64-128-1'
        graph = load_graph(graph_path, build_config, args.dataset)
        dataset, queries, ground_truth, _ = load_dataset(dataset_path, args.dataset)
        
        graph = convert_to_torch(graph)
        dataset = convert_to_torch(dataset)

        start = time.time()
        sign_bit = cpu_generate_sign_bit.generate_uint32_compressed(graph, dataset)
        end = time.time()

        print(f"Sign bit generation time: {end-start}")

        sign_bit = sign_bit.to(torch.uint32)
        sign_bit = sign_bit.cpu().numpy()

        sign_dir = os.path.join(sign_path, build_config)
        sign_file = os.path.join(sign_dir, args.dataset + ".bin")

        os.makedirs(sign_dir, exist_ok=True)

        with open(sign_file, "wb") as f:
            f.write(sign_bit.tobytes())
    else:
        build_config = f"_shard{num_shard}"
        dataset, queries, ground_truth, _ = load_dataset(dataset_path, args.dataset)
        shard_size = dataset.shape[0] // num_shard
        
        # for i in range(num_shard):
        i = int(args.i)

        shard_name = f"{args.dataset}-{i}"
        graph = load_graph(graph_path, build_config, shard_name)
        dataset_chunk = dataset[i*shard_size:(i+1)*shard_size]
        graph = convert_to_torch(graph)
        dataset_chunk = convert_to_torch(dataset_chunk)


        start = time.time()
        sign_bit = cpu_generate_sign_bit.generate_uint32_compressed(graph, dataset_chunk)
        end = time.time()
        print(f"Sign bit generation time: {end-start}")

        sign_bit = sign_bit.to(torch.uint32)
        sign_bit = sign_bit.cpu().numpy()
        dataset_size = dataset_path.split('/')[-1]
        sign_dir = os.path.join(sign_path, dataset_size ,build_config)
        sign_file = os.path.join(sign_dir, shard_name + ".bin")

        os.makedirs(sign_dir, exist_ok=True)

        with open(sign_file, "wb") as f:
            f.write(sign_bit.tobytes())
        
if __name__ == '__main__':
    main()

