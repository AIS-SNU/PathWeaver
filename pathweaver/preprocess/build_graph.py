import logging
import argparse
import numpy as np
import torch
import os
import sys
import pickle
import time
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from util import load_dataset

def save_graph(base_path, graph_build_config, dataset_name, dataset, cagra_wrapper, graph_size):

    algorithm, intermediate_graph_degree, graph_degree, num_shard = graph_build_config.split('-')

    cagra_instance = cagra_wrapper.CagraWrapper()

    build_params = cagra_wrapper.IndexParams()
    build_params.graph_degree = int(graph_degree)
    build_params.intermediate_graph_degree = int(intermediate_graph_degree)

    graph_path = f'./{base_path}/{dataset_name}.pkl'
    graph_dir = f'./{base_path}'

    os.makedirs(graph_dir, exist_ok=True)

    if os.path.exists(graph_path):
        print('Already exists')
    else:
        print(f"Building and Saving graph to {graph_path}")
        torch.cuda.synchronize()
        start = time.time()
        graph = cagra_instance.build(dataset, build_params)
        # graph = cagra_instance.build(dataset)
        torch.cuda.synchronize()
        end = time.time()
        print(f"Graph build time: {end-start}")
        torch.cuda.empty_cache()

        with open(graph_path, 'wb') as f:

            import code; code.interact(local=locals())
            pickle.dump(graph, f)
            print("File successfully saved")
    
    return

def build_and_save_graph_1(graph_file_path, shard, intermediate_graph_degree, graph_degree, cagra_wrapper):
    
    cagra_instance = cagra_wrapper.CagraWrapper()

    build_params = cagra_wrapper.IndexParams()
    build_params.graph_degree = graph_degree
    build_params.intermediate_graph_degree = intermediate_graph_degree

    torch.cuda.synchronize()
    start = time.time()
    graph = cagra_instance.build(shard, build_params)
    # graph = cagra_instance.build(shard)
    torch.cuda.synchronize()
    end = time.time()
    print(f"Graph build time: {end-start}")

    with open(graph_file_path, 'wb') as f:
        pickle.dump(graph, f)
        print("File successfully saved")

    return graph

def main():

    parser = argparse.ArgumentParser(description='graph name')
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--dataset_size', type=str)
    parser.add_argument('--num_shard', type=int, default=1)
    parser.add_argument('--graph_degree', type=int, default=64)
    parser.add_argument('--internal_graph_degree', type=int, default=128)
    parser.add_argument('--i', type=int)
    args = parser.parse_args()

    if args.dataset is None or args.dataset_size is None or args.num_shard is None or args.graph_degree is None or args.internal_graph_degree is None or args.i is None:
        raise ValueError("Please provide all the arguments")

    dataset_path = f'../../_datasets/{args.dataset_size}'
    graph_path = f'../../_graph/{args.dataset_size}'

    dataset, _, _, _ = load_dataset(dataset_path, args.dataset)

    # src_path = os.path.abspath('/nfs/home/daisy1212/2024_OSDI/fast_search/raft_graph_build/cpp/template/build')
    # if os.path.exists(src_path) and os.path.isdir(src_path):
    #     sys.path.append(src_path)
    #     print(f"Added to sys.path: {src_path}")
    # else:
    #     raise ValueError(f"Invalid path provided for --src: {src_path}")

    try:
        import cagra_wrapper
    except ImportError as e:
        print("Cannot import cagra_wrapper")
        sys.exit(1)
    
    # build_config = f"shard{args.num_shard}-{args.internal_graph_degree}-{args.graph_degree}"
    build_config = f"cagra-{args.internal_graph_degree}-{args.graph_degree}-{args.num_shard}"

    dataset_size = dataset.shape[0]
    shard_size = dataset_size // args.num_shard
    print(f"{dataset_size} Dataset split into {args.num_shard} shards of size {shard_size}")

    for i in range(args.num_shard):
        dataset_chunk = dataset[i*shard_size: (i+1)*shard_size]
        save_graph(graph_path, build_config, args.dataset, dataset_chunk, cagra_wrapper, args.dataset_size)

    if args.num_shard == 1:
        save_graph(graph_path, build_config, args.dataset, dataset, cagra_wrapper, args.dataset_size)
    else:
        dataset_size = dataset.shape[0]
        chunk_size = dataset_size // args.num_shard
        print(f"dataset_size: {dataset_size}")
        print(f"chunk_size: {chunk_size}")

        shard_graph_path = f'{graph_path}/_shard{args.num_shard}'
        # for i in range(num_shard):
        i = int(args.i)
        chunk = dataset[i*chunk_size: (i+1)*chunk_size]
        save_graph(shard_graph_path, build_config, f'{args.dataset}-{i}', chunk, cagra_wrapper, args.dataset_size)


if __name__ == "__main__":
    main()