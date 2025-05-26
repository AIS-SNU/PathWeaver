import os
import sys
import logging
import argparse
import pickle
import numpy as np
import torch
import time

from util import load_dataset, load_graph_1, load_sign_bit_1, load_checkpoint_map_1
from util import convert_to_torch, calculate_recall, calculate_shared_mem_size, is_valid_prune_ratio

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), './bin/pathweaver')))
import pathweaver
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), './bin/cagra')))
import cagra_wrapper

NO_PROFILE = 0
PROFILE = 1

NO_PRUNE = 0
RANDOM_PRUNE = 1
SIGN_BIT_PRUNE = 2

NO_THRESHOLD = 0
USE_THRESHOLD = 1
AGRESSIVE_THRESHOLD = 2

NO_SEED = 0
USE_SEED = 1

NO_HASH_TABLE = 0
USE_HASH_TABLE = 1

def main():


    parser = argparse.ArgumentParser(description='')

    parser.add_argument('--dataset_name', type=str, required=True)
    parser.add_argument('--dataset_path', type=str, required=True)
    parser.add_argument('--graph_save_path', type=str, required=True)
    parser.add_argument('--dgs_save_path', type=str, required=True)
    parser.add_argument('--build_graph_degree', type=int, required=True)
    parser.add_argument('--build_intermediate_graph_degree', type=int, required=True)

    #============== Search Parameters ==============
    parser.add_argument('--test_iteration', type=int, required=True)
    parser.add_argument('--file_save', type=bool, required=True)

    parser.add_argument('--ghost_scale_factor', type=int, required=True)

    parser.add_argument('--ghost_internal_topk', type=int, required=True)
    parser.add_argument('--base_internal_topk', type=int, required=True)

    parser.add_argument('--ghost_max_iter', type=int, required=True)
    parser.add_argument('--base_max_iter', type=int, required=True)

    parser.add_argument('--base_prune_config', type=int, required=True)
    parser.add_argument('--base_iteration_prune_ratio', type=float, required=True)
    parser.add_argument('--base_neighbor_prune_ratio', type=float, required=True)

    args = parser.parse_args()

    if not is_valid_prune_ratio(args.base_neighbor_prune_ratio):
        print("Error: --base_neighbor_prune_ratio must be between 0.0 and 1.0 and a multiple of 0.125.")
        sys.exit(1)

    #============== Common Parameters ==============
    min_iter = 0
    search_width = 1
    num_dist = 1
    team_size = 8
    block_size = 32
    bitlen = 10
    hash_reset_interval = 16
    seed_config = 1
    hash_table_config = 1
    threshold_config = 0
    #===============================================

    #=============================== Log file path ===============================
    output_dir = "../_log" + "/single" + "/pathweaver"
    if not os.path.exists(output_dir): os.makedirs(output_dir)
    output_file = f"{output_dir}/{args.dataset_name}_{args.build_graph_degree}_{args.build_intermediate_graph_degree}.log"
    #=============================================================================

#======================================================================================================================

    dataset_dir_path = f"{args.dataset_path}"
    graph_path = f"{args.graph_save_path}/shard1_{args.dataset_name}_{args.build_graph_degree}_{args.build_intermediate_graph_degree}/{args.dataset_name}-0.graph"
    sign_bit_path = f"{args.dgs_save_path}/shard1_{args.dataset_name}_{args.build_graph_degree}_{args.build_intermediate_graph_degree}/{args.dataset_name}-0.sign"
    
    dataset, queries, ground_truth, _ = load_dataset(dataset_dir_path, args.dataset_name)
    graph = load_graph_1(graph_path)
    sign_bit = load_sign_bit_1(sign_bit_path)

    num_device = torch.cuda.device_count()
    if(num_device == 0):
        raise Exception("No CUDA device found")
    device = torch.device("cuda:0")

    dataset_size = dataset.shape[0]
    vector_dim = dataset.shape[1]
    graph_degree = graph.shape[1]
    query_size = queries.shape[0]
    ghost_dataset_size = dataset_size // args.ghost_scale_factor
    print(f">>> Dataset: {args.dataset_name}, Dataset size: {dataset_size}, Ghost dataset size: {ghost_dataset_size}, Graph degree: {graph_degree}, Vector dimension: {vector_dim}, Query size: {query_size}")

    # build graph out of seed
    ghost_index = np.random.randint(0, dataset.shape[0], ghost_dataset_size, dtype=np.uint32)
    ghost_dataset = dataset[ghost_index]
    cagra_instance = cagra_wrapper.CagraWrapper()
    build_params = cagra_wrapper.IndexParams()
    build_params.graph_degree = args.build_graph_degree
    build_params.intermediate_graph_degree = args.build_intermediate_graph_degree
    ghost_graph = cagra_instance.build(ghost_dataset, build_params)
    empty_sign_bit = np.zeros((0, 0), dtype=np.uint32)
    initial_starting_point = np.zeros((query_size), dtype=np.uint32)
    identity_map = np.arange(ghost_dataset_size, dtype=np.uint32)
 
    # convert to torch
    ghost_graph = convert_to_torch(ghost_graph).to(device)
    ghost_dataset = convert_to_torch(ghost_dataset).to(device)
    queries = convert_to_torch(queries).to(device)
    empty_sign_bit = torch.from_numpy(empty_sign_bit.copy()).to(device)
    initial_starting_point = convert_to_torch(initial_starting_point).to(device)
    identity_map = convert_to_torch(identity_map).to(device)
    # convert to torch
    graph = convert_to_torch(graph).to(device)
    dataset = convert_to_torch(dataset).to(device)
    sign_bit = torch.from_numpy(sign_bit.copy()).to(device)

#======================================================================================================================

    # config
    # seed graph search configs
    ghost_topk = 1
    ghost_shared_mem_size = calculate_shared_mem_size(vector_dim, args.ghost_internal_topk, graph_degree, 1, 10)
    ghost_configs = np.array(
        [
            queries.shape[0],
            ghost_topk,
            search_width,
            args.ghost_max_iter,
            min_iter,
            args.ghost_internal_topk,
            vector_dim,
            ghost_dataset_size,
            team_size,
            graph_degree,
            num_dist,
            block_size,
            bitlen,
            hash_reset_interval,
            ghost_shared_mem_size, 
            NO_PRUNE,
            1.0, 
            0.0, 
            threshold_config, 
            0, 
            seed_config,
            hash_table_config
        ]
    )
    ghost_configs = convert_to_torch(ghost_configs).to(device)
    ghost_results_distances = torch.zeros((queries.shape[0], ghost_topk), dtype=torch.float32).to(device)

    # original graph search configs
    base_topk = 10
    base_shared_mem_size = calculate_shared_mem_size(vector_dim, args.base_internal_topk, graph_degree, 1, 10)
    base_configs = np.array(
        [
            queries.shape[0],
            base_topk,
            search_width,
            args.base_max_iter,
            min_iter,
            args.base_internal_topk,
            vector_dim,
            dataset_size,
            team_size,
            graph_degree,
            num_dist,
            block_size,
            bitlen,
            hash_reset_interval,
            base_shared_mem_size, 
            args.base_prune_config,                   
            args.base_iteration_prune_ratio, 
            args.base_neighbor_prune_ratio, 
            threshold_config, 
            0, 
            seed_config, 
            hash_table_config
        ]
    )
    base_configs = convert_to_torch(base_configs).to(device)
    base_results_distances = torch.zeros((queries.shape[0], base_topk), dtype=torch.float32).to(device)

    # # warm up
    pathweaver.search(ghost_graph, ghost_dataset, queries, initial_starting_point, identity_map, empty_sign_bit, ghost_configs, ghost_results_distances)
    torch.cuda.synchronize()

    recalls = []
    exec_times = []

    # search
    seed_map = convert_to_torch(ghost_index).to(device)
    for iter in range(args.test_iteration):
        
        start_time = time.time()
        torch.cuda.synchronize()
        top1 = pathweaver.search(ghost_graph, ghost_dataset, queries, initial_starting_point, identity_map, empty_sign_bit, ghost_configs, base_results_distances)            
        torch.cuda.synchronize()  
        full_results = pathweaver.search(graph, dataset, queries, top1, seed_map, sign_bit, base_configs, base_results_distances)
        torch.cuda.synchronize()
        end_time = time.time()

        recall = calculate_recall(full_results.to('cpu'), ground_truth, base_topk)
        execution_time = (end_time - start_time) * 1000

        recalls.append(recall)
        exec_times.append(execution_time)

    mean_recall = np.mean(recalls)
    median_execution_time = np.median(exec_times)
    print(f"Ghost_scale_factor: {args.ghost_scale_factor}, Ghost_internal_topk: {args.ghost_internal_topk}, "
        f"Base_internal_topk: {args.base_internal_topk}, Ghost_max_iter: {args.ghost_max_iter}, "
        f"Base_max_iter: {args.base_max_iter}, Base_prune_config: {args.base_prune_config}, "
        f"Base_iteration_prune_ratio: {args.base_iteration_prune_ratio}, base_neighbor_prune_ratio: {args.base_neighbor_prune_ratio}, "
        f"Mean Recall: {mean_recall:.4f}, Median Execution Time: {median_execution_time:.4f}ms")
    
    # save to file
    if args.file_save:
        with open(output_file, "a") as f:
            f.write(
                f"Ghost_scale_factor: {args.ghost_scale_factor}, "
                f"Ghost_internal_topk: {args.ghost_internal_topk}, "
                f"Base_internal_topk: {args.base_internal_topk}, "
                f"Ghost_max_iter: {args.ghost_max_iter}, "
                f"Base_max_iter: {args.base_max_iter}, "
                f"Base_prune_config: {args.base_prune_config}, "
                f"Base_iteration_prune_ratio: {args.base_iteration_prune_ratio}, "
                f"base_neighbor_prune_ratio: {args.base_neighbor_prune_ratio}, "
                f"Mean Recall: {mean_recall:.4f}, "
                f"Median Execution Time: {median_execution_time:.4f}ms\n"
            )


if __name__ == "__main__":
    main()