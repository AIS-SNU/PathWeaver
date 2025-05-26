import argparse
import cupy as cp
import pickle
import time
import torch
import os

from util import load_dataset

from cuvs.neighbors import cagra

def save_checkpoint_map_1(checkpoint_map_file_path, current_shard, query_shard):

    current_shard_gpu = cp.array(current_shard)
    query_shard_gpu = cp.array(query_shard)

    build_params = cagra.IndexParams(
        intermediate_graph_degree = 64,
        graph_degree = 64
    )

    graph = cagra.build(build_params, current_shard_gpu)

    topk = 1
    search_params = cagra.SearchParams(
                    max_queries = 10000,
                    itopk_size = 64,
                    max_iterations = 71,
                    algo = "single_cta",
                    team_size = 8,
                    search_width = 1,
                    min_iterations = 0,
                    thread_block_size = 64
                )
    
    torch.cuda.synchronize()
    search_start_time = time.time()
    _, neighbors = cagra.search(search_params, graph, query_shard_gpu, topk)
    torch.cuda.synchronize()
    search_end_time = time.time()

    print(f"Checkpoint generation time: {search_end_time - search_start_time}")

    neighbors_numpy = neighbors.copy_to_host()

    with open(checkpoint_map_file_path, "wb") as file:
        pickle.dump(neighbors_numpy, file)

    cp.get_default_memory_pool().free_all_blocks()
    cp.get_default_pinned_memory_pool().free_all_blocks()

    return



def main():

    parser = argparse.ArgumentParser(description='graph name')
    parser.add_argument('--dataset', type=str, default='sift-128-euclidean')
    parser.add_argument('--dataset_size', type=str, default='100M')
    parser.add_argument('--num_shard', type=int)
    parser.add_argument('--i', type=int)
    args = parser.parse_args()

    dataset_path = f'../_datasets/{args.dataset_size}'

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.i)
    print(f"Using GPU ID: {args.i}")
    
    # multi gpu
    num_shard = args.num_shard
    shard_id = args.i

    # load dataset
    dataset, _, _, _ = load_dataset(dataset_path, args.dataset)
    dataset_size = dataset.shape[0]
    vector_dim = dataset.shape[1]
    graph_degree = 64
    if dataset_size % num_shard != 0:
        raise Exception("Dataset size should be divisible by the number of GPUs")
    shard_size = dataset_size // num_shard
    queries_size = shard_size

    saved_file_path = f"../_checkpoint_map/{args.dataset_size}/_shard{num_shard}/{args.dataset}-{shard_id}.pkl"
    print(f"Saved file path: {saved_file_path}")

    query_shard_id = (shard_id + 1) % num_shard
    current_shard = dataset[shard_id*shard_size:(shard_id+1)*shard_size]
    query_shard = dataset[query_shard_id*shard_size:(query_shard_id+1)*shard_size]

    current_shard_gpu = cp.array(current_shard)
    query_shard_gpu = cp.array(query_shard)

    build_params = cagra.IndexParams(
        intermediate_graph_degree = 64,
        graph_degree = 64
    )

    graph = cagra.build(build_params, current_shard_gpu)

    topk = 1
    search_params = cagra.SearchParams(
                    max_queries = 10000,
                    itopk_size = 64,
                    max_iterations = 71,
                    algo = "single_cta",
                    team_size = 8,
                    search_width = 1,
                    min_iterations = 0,
                    thread_block_size = 64
                )
    
    torch.cuda.synchronize()
    search_start_time = time.time()
    _, neighbors = cagra.search(search_params, graph, query_shard_gpu, topk)
    torch.cuda.synchronize()
    search_end_time = time.time()

    print(f"Search time: {search_end_time - search_start_time}")

    neighbors_numpy = neighbors.copy_to_host()

    with open(saved_file_path, "wb") as file:
        pickle.dump(neighbors_numpy, file)

    cp.get_default_memory_pool().free_all_blocks()
    cp.get_default_pinned_memory_pool().free_all_blocks()




if __name__ == "__main__":
    main()