import os
import sys
import numpy as np
import torch
import argparse

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../bin/cagra')))
import cagra_wrapper

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from .build_graph import build_and_save_graph_1
from .generate_checkpoint_map import save_checkpoint_map_1
from .generate_sign_bit import save_sign_bit_2
from util import load_dataset, load_graph_1

def main():
    parser = argparse.ArgumentParser(description='')

    parser.add_argument('--dataset_name', type=str, required=True)
    parser.add_argument('--dataset_path', type=str, required=True)
    parser.add_argument('--graph_save_path', type=str, required=True)
    parser.add_argument('--dgs_save_path', type=str, required=True)
    parser.add_argument('--pbpe_save_path', type=str, required=True)
    parser.add_argument('--build_graph_degree', type=int, required=True)
    parser.add_argument('--build_intermediate_graph_degree', type=int, required=True)
    parser.add_argument('--num_shard', type=int,required=True)
    args = parser.parse_args()

    dataset_dir = f"{args.dataset_path}"
    graph_dir = f"{args.graph_save_path}/shard{args.num_shard}_{args.dataset_name}_{args.build_graph_degree}_{args.build_intermediate_graph_degree}"
    sign_bit_dir = f"{args.dgs_save_path}/shard{args.num_shard}_{args.dataset_name}_{args.build_graph_degree}_{args.build_intermediate_graph_degree}"
    checkpoint_map_dir = f"{args.pbpe_save_path}/shard{args.num_shard}_{args.dataset_name}_{args.build_graph_degree}_{args.build_intermediate_graph_degree}"


    dataset, _, _, _ = load_dataset(dataset_dir, args.dataset_name)
    dataset_size = dataset.shape[0]
    shard_size = dataset_size // args.num_shard
    print(f"{dataset_size} Dataset split into {args.num_shard} shards of size {shard_size}")

    if not os.path.exists(graph_dir): 
        os.makedirs(graph_dir)
    if not os.path.exists(sign_bit_dir):
        os.makedirs(sign_bit_dir)    
    if args.num_shard != 1:
        if not os.path.exists(checkpoint_map_dir):
            os.makedirs(checkpoint_map_dir)
    
    for i in range (int(args.num_shard)):
        ############################## BUILD GRAPH ##########################################
        dataset_per_device = dataset[i*shard_size: (i+1)*shard_size]
        
        graph_file_path = f"{graph_dir}/{args.dataset_name}-{i}.graph"
        graph = None
        if os.path.exists(graph_file_path):
            print(f"Graph file already exists: {graph_file_path}")
            graph = load_graph_1(graph_file_path)
        else:
            graph = build_and_save_graph_1(graph_file_path, dataset_per_device, args.build_intermediate_graph_degree, args.build_graph_degree, cagra_wrapper)

        ############################## GENERATE SIGN BITS ###################################
        dataset_per_device = dataset[i*shard_size: (i+1)*shard_size]

        sign_bit_file_path = f"{sign_bit_dir}/{args.dataset_name}-{i}.sign"
        if os.path.exists(sign_bit_file_path):
            print(f"Sign bit file already exists: {sign_bit_file_path}")
        else:
            save_sign_bit_2(sign_bit_file_path, dataset_per_device, graph)

        del dataset_per_device

        if args.num_shard != 1:
        ############################## GENERATE CHECKPOINT MAP ##############################
            dataset_per_device = dataset[i*shard_size: (i+1)*shard_size]
        
            query_shard_id = (i + 1) % args.num_shard
            current_shard = dataset_per_device
            query_shard = dataset[query_shard_id*shard_size:(query_shard_id+1)*shard_size]

            save_checkpoint_map_path = f"{checkpoint_map_dir}/{args.dataset_name}-{i}.checkpoint"
            if os.path.exists(save_checkpoint_map_path):
                print(f"Checkpoint map file already exists: {save_checkpoint_map_path}")
            else:
                save_checkpoint_map_1(save_checkpoint_map_path, current_shard, query_shard)
            


if __name__ == "__main__":
    main()
