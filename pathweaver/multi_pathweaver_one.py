import os
import sys
import argparse
import pickle
import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import time
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
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

def search_on_gpu(
    gpu_id,
    num_gpu,
    ghost_dataset_pool,
    ghost_graph_pool,
    ghost_query_pool,
    ghost_seed_pool,
    ghost_identity_map_pool,
    ghost_sign_bit_pool,
    ghost_distance_pool,
    ghost_config_pool,
    base_dataset_pool,
    base_graph_pool,
    base_query_pool,
    base_seed_pool,
    base_seed_map_pool,
    base_sign_bit_pool,
    base_distance_pool,
    base_config_pool,
    base_checkpoint_map_pool,
    base_234_config_pool,
    query_chunk_size,
    shard_size,
    result_dict,
    time_buffer
):
    device = torch.device(f'cuda:{gpu_id}')
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '17775'
    dist.init_process_group(backend="nccl", init_method="env://", rank=gpu_id, world_size=num_gpu)
    torch.cuda.set_device(gpu_id)

    with torch.cuda.device(device):
        dist.barrier()

        #####################################
        # Warm-Up Phase (Optional)
        #####################################
        # A warm-up search is performed to prepare the GPU and cache.
        # We call a single stage of search for entry and full graph.
        warmup_offset = gpu_id * query_chunk_size
        ghost_result_warm = pathweaver.search(
            ghost_graph_pool[gpu_id],
            ghost_dataset_pool[gpu_id],
            ghost_query_pool[gpu_id][warmup_offset: warmup_offset + query_chunk_size],
            ghost_seed_pool[gpu_id],
            ghost_identity_map_pool[gpu_id],
            ghost_sign_bit_pool[gpu_id],
            ghost_config_pool[gpu_id],
            ghost_distance_pool[gpu_id]
        )
        torch.cuda.synchronize()
        _ = pathweaver.search(
            base_graph_pool[gpu_id],
            base_dataset_pool[gpu_id],
            base_query_pool[gpu_id][warmup_offset: warmup_offset + query_chunk_size],
            ghost_result_warm,
            base_seed_map_pool[gpu_id],
            base_sign_bit_pool[gpu_id],
            base_config_pool[gpu_id],
            base_distance_pool[gpu_id][warmup_offset: warmup_offset + query_chunk_size]
        )
        dist.barrier()
        
        #####################################
        # Pipeline Search – Loop over rounds
        #####################################
        # We define a pipeline with num_rounds equal to the number of GPUs.
        # In round 0 we perform the initial entry and full search stage.
        # In subsequent rounds we perform a send/recv exchange of the previous round's result,
        # adjust the query offset and perform the full search using the checkpoint map and pipeline config.
        #
        # Note: The distance pool base_distance_pool is preallocated with shape:
        #   (query_chunk_size * num_gpu, base_topk)
        
        result_list = []  # collect results from each round
        current_query_offset = gpu_id * query_chunk_size

        # Round 0: initial entry search then full search
        first_stage_result = pathweaver.search(
            ghost_graph_pool[gpu_id],
            ghost_dataset_pool[gpu_id],
            ghost_query_pool[gpu_id][current_query_offset: current_query_offset + query_chunk_size],
            ghost_seed_pool[gpu_id],
            ghost_identity_map_pool[gpu_id],
            ghost_sign_bit_pool[gpu_id],
            ghost_config_pool[gpu_id],
            ghost_distance_pool[gpu_id]
        )
        torch.cuda.synchronize()
        current_result = pathweaver.search(
            base_graph_pool[gpu_id],
            base_dataset_pool[gpu_id],
            base_query_pool[gpu_id][current_query_offset: current_query_offset + query_chunk_size],
            first_stage_result,
            base_seed_map_pool[gpu_id],
            base_sign_bit_pool[gpu_id],
            base_config_pool[gpu_id],
            base_distance_pool[gpu_id][current_query_offset: current_query_offset + query_chunk_size]
        )
        result_list.append(current_result)
        
        # Pipeline rounds: for rounds 1 .. num_gpu-1
        dist.barrier()
        torch.cuda.synchronize()
        start = time.time()
        for stage in range(1, num_gpu):
            # Synchronize and exchange previous round's result with neighbors.
            dist.barrier()
            current_result_int = current_result.to(torch.int32)
            tmp_result_recv = torch.zeros_like(current_result_int)
            send_op = dist.P2POp(dist.isend, current_result_int, (gpu_id - 1) % num_gpu)
            recv_op = dist.P2POp(dist.irecv, tmp_result_recv, (gpu_id + 1) % num_gpu)
            reqs = dist.batch_isend_irecv([send_op, recv_op])
            for req in reqs:
                req.wait()
            dist.barrier()

            # Prepare for the next round.
            current_result = tmp_result_recv.to(torch.uint32)
            # Compute new query offset: rotate based on stage
            current_query_offset = ((gpu_id + stage) % num_gpu) * query_chunk_size

            # In subsequent rounds, use the checkpoint map and pipeline configuration (base_234_config_pool)
            current_result = pathweaver.search(
                base_graph_pool[gpu_id],
                base_dataset_pool[gpu_id],
                base_query_pool[gpu_id][current_query_offset: current_query_offset + query_chunk_size],
                current_result,
                base_checkpoint_map_pool[gpu_id],
                base_sign_bit_pool[gpu_id],
                base_234_config_pool[gpu_id],
                base_distance_pool[gpu_id][current_query_offset: current_query_offset + query_chunk_size]
            )
            result_list.append(current_result)
        
        # End of pipeline; ensure synchronization and record time.
        dist.barrier()
        torch.cuda.synchronize()
        end = time.time()
        elapsed_time = (end - start) * 1000

        # Send the results from all rounds to the host.
        # Concatenate the results from each round along the 0-th axis.
        # (Note that each round produced a tensor of shape (query_chunk_size, ...),
        # so the full tensor is (query_chunk_size * num_gpu, ...).
        result_cpu_list = [res.to('cpu') for res in result_list]
        # results_indices = torch.cat(result_cpu_list, dim=0) + gpu_id * shard_size
        results_indices = torch.cat(result_cpu_list, dim=0).to(torch.int64) + gpu_id * shard_size
        results_distances = base_distance_pool[gpu_id].to('cpu').numpy()



        # Store results in the shared dictionary.
        result_dict[gpu_id] = {
            'indices': results_indices,
            'distances': results_distances
        }

        # print(f"GPU {gpu_id} finished in {elapsed_time:.4f} ms")
        time_buffer[gpu_id] = elapsed_time  # Note: you might want to record (end - start) instead.
        dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser(description='Multi-GPU ANN Search with Pipeline Sharding')
 
    parser.add_argument('--dataset_name', type=str, required=True)
    parser.add_argument('--dataset_path', type=str, required=True)
    parser.add_argument('--graph_save_path', type=str, required=True)
    parser.add_argument('--dgs_save_path', type=str, required=True)
    parser.add_argument('--pbpe_save_path', type=str, required=True)
    parser.add_argument('--build_graph_degree', type=int,required=True)
    parser.add_argument('--build_intermediate_graph_degree', type=int, required=True)

    #============== Search Parameters ==============
    parser.add_argument('--num_gpu', type=int, required=True)
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

    parser.add_argument('--base_234_iteration_adventage', type=int, required=True)

    args = parser.parse_args()

    if not is_valid_prune_ratio(args.base_neighbor_prune_ratio):
        print("Error: base_neighbor_prune_ratio must be between 0.0 and 1.0 and a multiple of 0.125.")
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
    output_dir = "../_log" + "/multi" + "/pathweaver"
    if not os.path.exists(output_dir): os.makedirs(output_dir)
    output_file = f"{output_dir}/{args.dataset_name}_{args.build_graph_degree}_{args.build_intermediate_graph_degree}.log"
    #=============================================================================

#======================================================================================================================

    dataset_dir_path = f"{args.dataset_path}"
    graph_dir_path = f"{args.graph_save_path}/shard{args.num_gpu}_{args.dataset_name}_{args.build_graph_degree}_{args.build_intermediate_graph_degree}"
    sign_bit_dir_path = f"{args.dgs_save_path}/shard{args.num_gpu}_{args.dataset_name}_{args.build_graph_degree}_{args.build_intermediate_graph_degree}"
    checkpoint_map_dir_path = f"{args.pbpe_save_path}/shard{args.num_gpu}_{args.dataset_name}_{args.build_graph_degree}_{args.build_intermediate_graph_degree}"    

    # Multi GPU: Use the provided number (default 4)
    print(f"# of GPUS: {torch.cuda.device_count()}")
    num_gpu = args.num_gpu

    # load dataset
    dataset, queries, ground_truth, _ = load_dataset(dataset_dir_path, args.dataset_name)
    num_duplicate = 6
    queries = np.concatenate([queries for _ in range(num_duplicate)], axis=0)

    dataset_size = dataset.shape[0]
    vector_dim = dataset.shape[1]
    graph_degree = args.build_graph_degree

    queries_size = queries.shape[0]
    if dataset_size % num_gpu != 0:
        raise Exception("Dataset size should be divisible by the number of GPUs")
    shard_size = dataset_size // num_gpu
    ghost_dataset_size = shard_size // args.ghost_scale_factor
    if shard_size % args.ghost_scale_factor != 0:
        raise Exception("Shard size should be divisible by seed graph scale factor")
    
    # Pipeline sharding
    query_chunk_size = queries_size // num_gpu
    if queries_size % num_gpu != 0:
        raise Exception("Query size should be divisible by the number of GPUs")
    
    print(f"Available GPUs: {torch.cuda.device_count()}, Requested GPUs: {num_gpu}")
    print(f"Dataset: {args.dataset_name}, Dataset_size: {dataset_size}, Shard_size: {shard_size}")
    print(f"Query_size: {queries_size}, Query_chunk_size: {query_chunk_size}")
    print(f"Graph degree: {graph_degree}, vector_dim: {vector_dim}, ghost_dataset_size: {ghost_dataset_size}")

    ##################################################### CONFIGURATION #####################################################
    base_topk = 10
    base_shared_mem_size = calculate_shared_mem_size(vector_dim, 64, graph_degree, 1, 10)
    ghost_topk = 1
    ghost_shared_mem_size = calculate_shared_mem_size(vector_dim, 64, graph_degree, 1, 10)
    
    base_configs = np.array(
        [
            query_chunk_size,
            base_topk,
            search_width,
            args.base_max_iter,
            min_iter,
            args.base_internal_topk,
            vector_dim,
            shard_size,
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
            hash_table_config,
            ghost_topk                     # seed_topk
        ]
    )

    base_234_configs = np.array(
        [
            query_chunk_size,
            base_topk,
            search_width,
            args.base_max_iter - args.base_234_iteration_adventage,
            min_iter,
            args.base_internal_topk,
            vector_dim,
            shard_size,
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
            hash_table_config,
            base_topk
        ]
    )

    ghost_configs = np.array(
        [
            query_chunk_size,
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
            0.0, 
            1.0, 
            threshold_config, 
            0, 
            seed_config,
            hash_table_config,
            ghost_topk
        ]
    )

    ##################################################### MEMORY POOL #####################################################
    # For full search
    base_dataset_pool = [None] * num_gpu
    base_graph_pool = [None] * num_gpu
    base_query_pool = [None] * num_gpu
    base_seed_pool = [None] * num_gpu
    base_seed_map_pool = [None] * num_gpu
    base_sign_bit_pool = [None] * num_gpu
    # Preallocate distance tensor with shape (query_chunk_size * num_gpu, topk)
    base_distance_pool = [np.zeros((query_chunk_size * num_gpu, base_topk), dtype=np.float32) for _ in range(num_gpu)]
    base_config_pool = [None] * num_gpu

    # For entry search
    ghost_dataset_pool = [None] * num_gpu
    ghost_graph_pool = [None] * num_gpu
    ghost_query_pool = [None] * num_gpu
    ghost_seed_pool = [None] * num_gpu
    ghost_identity_map_pool = [None] * num_gpu
    ghost_sign_bit_pool = [None] * num_gpu
    ghost_distance_pool = [None] * num_gpu
    ghost_config_pool = [None] * num_gpu

    # For checkpoint propagation
    base_checkpoint_map_pool = [None] * num_gpu
    # For pipeline search configuration
    base_234_config_pool = [None] * num_gpu

    ##################################################### FULL GRAPH #####################################################
    for i in range(num_gpu):
        shard_name = f"{args.dataset_name}-{i}"
        build_config = f"_shard{num_gpu}"
        base_dataset_pool[i] = dataset[i*shard_size:(i+1)*shard_size].copy()
        base_graph_pool[i] = load_graph_1(f"{graph_dir_path}/{args.dataset_name}-{i}.graph").copy()
        base_query_pool[i] = queries.copy()
        # base_seed_pool[i] = load_seed(args.seed_path, shard_name, base_dataset_pool[i].shape[0], queries.shape[0], base_graph_pool[i].shape[1], 1, 1)
        base_seed_map_pool[i] = np.random.randint(0, shard_size, ghost_dataset_size, dtype=np.uint32).copy()
        base_sign_bit_pool[i] = load_sign_bit_1(f"{sign_bit_dir_path}/{args.dataset_name}-{i}.sign").copy()
        base_config_pool[i] = base_configs.copy()
        base_checkpoint_map_pool[i] = load_checkpoint_map_1(f"{checkpoint_map_dir_path}/{args.dataset_name}-{i}.checkpoint").copy()
        base_234_config_pool[i] = base_234_configs.copy()

    ##################################################### ENTRY GRAPH #####################################################
    cagra_instance = cagra_wrapper.CagraWrapper()
    build_params = cagra_wrapper.IndexParams()
    build_params.graph_degree = 64
    build_params.intermediate_graph_degree = 128

    for i in range(num_gpu):
        ghost_dataset_pool[i] = base_dataset_pool[i][base_seed_map_pool[i]].copy()
        start_seed_graph_time = time.time()
        ghost_graph_pool[i] = cagra_instance.build(ghost_dataset_pool[i], build_params).copy()
        torch.cuda.synchronize()
        end_seed_graph_time = time.time()
        print(f"Seed graph build time (GPU {i}): {end_seed_graph_time - start_seed_graph_time}")
        ghost_query_pool[i] = queries.copy()  # redundant copy
        ghost_seed_pool[i] = np.zeros(query_chunk_size, dtype=np.uint32).copy()
        ghost_identity_map_pool[i] = np.arange(ghost_dataset_size, dtype=np.uint32).copy()
        ghost_sign_bit_pool[i] = np.zeros((0, 0), dtype=np.uint32).copy()
        ghost_distance_pool[i] = np.zeros((query_chunk_size, ghost_topk), dtype=np.float32).copy()
        ghost_config_pool[i] = ghost_configs.copy()

    ##################################################### TO TORCH AND TO GPU #####################################################
    for i in range(num_gpu):
        device = torch.device(f'cuda:{i}')
        with torch.cuda.device(i):
            base_graph_pool[i] = convert_to_torch(base_graph_pool[i]).to(device)
            base_dataset_pool[i] = convert_to_torch(base_dataset_pool[i]).to(device)
            base_query_pool[i] = convert_to_torch(base_query_pool[i]).to(device)
            base_seed_map_pool[i] = convert_to_torch(base_seed_map_pool[i]).to(device)
            base_sign_bit_pool[i] = convert_to_torch(base_sign_bit_pool[i]).to(device)
            base_distance_pool[i] = convert_to_torch(base_distance_pool[i]).to(device)
            base_config_pool[i] = convert_to_torch(base_config_pool[i]).to(device)
            base_checkpoint_map_pool[i] = convert_to_torch(base_checkpoint_map_pool[i]).to(device)
            base_234_config_pool[i] = convert_to_torch(base_234_config_pool[i]).to(device)

            ghost_graph_pool[i] = convert_to_torch(ghost_graph_pool[i]).to(device)
            ghost_dataset_pool[i] = convert_to_torch(ghost_dataset_pool[i]).to(device)
            ghost_query_pool[i] = convert_to_torch(ghost_query_pool[i]).to(device)
            ghost_seed_pool[i] = convert_to_torch(ghost_seed_pool[i]).to(device)
            ghost_identity_map_pool[i] = convert_to_torch(ghost_identity_map_pool[i]).to(device)
            ghost_sign_bit_pool[i] = convert_to_torch(ghost_sign_bit_pool[i]).to(device)
            ghost_distance_pool[i] = convert_to_torch(ghost_distance_pool[i]).to(device)
            ghost_config_pool[i] = convert_to_torch(ghost_config_pool[i]).to(device)

    ##################################################### SEARCH #####################################################
    recalls = []
    exec_times = []
    for i in range(args.test_iteration):
        # Launch a process for each GPU.
        mp.set_start_method('spawn', force=True)
        manager = mp.Manager()
        result_dict = manager.dict()
        time_buffer = manager.dict()

        processes = []
        for gpu_id in range(num_gpu):
            p = mp.Process(
                target=search_on_gpu,
                args=(
                    gpu_id,
                    num_gpu,
                    ghost_dataset_pool,
                    ghost_graph_pool,
                    ghost_query_pool,
                    ghost_seed_pool,
                    ghost_identity_map_pool,
                    ghost_sign_bit_pool,
                    ghost_distance_pool,
                    ghost_config_pool,
                    base_dataset_pool,
                    base_graph_pool,
                    base_query_pool,
                    base_seed_pool,           # not used in search
                    base_seed_map_pool,
                    base_sign_bit_pool,
                    base_distance_pool,
                    base_config_pool,
                    base_checkpoint_map_pool,
                    base_234_config_pool,
                    query_chunk_size,
                    shard_size,
                    result_dict,
                    time_buffer
                )
            )
            p.start()
            processes.append(p)

        for p in processes:
            p.join()

        # CPU-side merging.
        # For each GPU, we rotate the concatenated results by a "cut offset" defined by:
        #   cut_offset = query_chunk_size * (num_rounds - gpu_id)
        # where num_rounds equals num_gpu (the number of pipeline rounds).
        merged_results_indices = []
        merged_results_distances = []
        num_rounds = num_gpu  # number of rounds in the pipeline
        for gpu_id in range(num_gpu):
            cut_offset = query_chunk_size * (num_rounds - gpu_id)
            reordered_indices = np.concatenate(
                [result_dict[gpu_id]['indices'][cut_offset:], result_dict[gpu_id]['indices'][:cut_offset]],
                axis=0
            )
            reordered_distances = result_dict[gpu_id]['distances']
            merged_results_indices.append(reordered_indices)
            merged_results_distances.append(reordered_distances)

        merged_results_indices = np.concatenate(merged_results_indices, axis=1)
        merged_results_distances = np.concatenate(merged_results_distances, axis=1)

        # Merge sort along each row.
        sorted_indices = np.argsort(merged_results_distances, axis=1)
        sorted_merged_results_indices = np.take_along_axis(merged_results_indices, sorted_indices, axis=1)

        ground_truth = np.concatenate([ground_truth for _ in range(num_duplicate)], axis=0)
        mean_recall = calculate_recall(sorted_merged_results_indices, ground_truth, base_topk)
        max_time = max(time_buffer.values())
        # print(f"Recall: {mean_recall}")
        # print(f"Max time: {max_time:4f} ms")

        recalls.append(mean_recall)
        exec_times.append(max_time)

    mean_recall = np.mean(recalls)
    median_execution_time = np.median(exec_times)
    print(f"Ghost_scale_factor: {args.ghost_scale_factor}, Ghost_internal_topk: {args.ghost_internal_topk}, "
        f"Base_internal_topk: {args.base_internal_topk}, Ghost_max_iter: {args.ghost_max_iter}, "
        f"Base_max_iter: {args.base_max_iter}, Base_prune_config: {args.base_prune_config}, "
        f"Base_iteration_prune_ratio: {args.base_iteration_prune_ratio}, Base_neighbor_prune_ratio: {args.base_neighbor_prune_ratio}, "
        f"Base_234_iteration_adventage: {args.base_234_iteration_adventage}, "
        f"Mean Recall: {mean_recall:.4f}, Median Execution Time: {median_execution_time:.4f}ms")


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
                f"Base_neighbor_prune_ratio: {args.base_neighbor_prune_ratio}, "
                f"Base_234_iteration_adventage: {args.base_234_iteration_adventage}, "
                f"Mean Recall: {mean_recall:.4f}, "
                f"Median Execution Time: {median_execution_time:.4f}ms\n"
            )



if __name__ == "__main__":
    main()
