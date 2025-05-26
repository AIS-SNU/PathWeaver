import os
import numpy as np
import pymetis
import torch
import pickle
import time

def read_fbin(file_path):
    with open(file_path, 'rb') as f:
        num_vectors = np.fromfile(f, dtype=np.uint32, count=1)[0]
        dimension = np.fromfile(f, dtype=np.uint32, count=1)[0]
        
        vectors = np.fromfile(f, dtype=np.float32).reshape(num_vectors, dimension)
        
    return vectors

def read_ibin(file_path):
    with open(file_path, 'rb') as f:
        num_vectors = np.fromfile(f, dtype=np.uint32, count=1)[0]
        dimension = np.fromfile(f, dtype=np.uint32, count=1)[0]
        
        vectors = np.fromfile(f, dtype=np.int32).reshape(num_vectors, dimension)
        
    return vectors

def load_dataset(base_dir, dataset_name):

    base_dir = os.path.join(base_dir, dataset_name)
    
    dataset_path = os.path.join(base_dir, 'base.fbin')
    query_path = os.path.join(base_dir, 'query.fbin')
    gt_neighbors_path = os.path.join(base_dir, 'groundtruth.neighbors.ibin')
    gt_distances_path = os.path.join(base_dir, 'groundtruth.distances.fbin')

    dataset_data = read_fbin(dataset_path)
    queries_data = read_fbin(query_path)
    gt_neighbors = read_ibin(gt_neighbors_path)
    gt_distances = read_fbin(gt_distances_path)

    return dataset_data, queries_data, gt_neighbors, gt_distances

def load_dataset_1(dataset_path):

    base_path = os.path.join(dataset_path, 'base.fbin')
    query_path = os.path.join(dataset_path, 'query.fbin')
    gt_neighbors_path = os.path.join(dataset_path, 'groundtruth.neighbors.ibin')
    gt_distances_path = os.path.join(dataset_path, 'groundtruth.distances.fbin')

    dataset_data = read_fbin(base_path)
    queries_data = read_fbin(query_path)
    gt_neighbors = read_ibin(gt_neighbors_path)
    gt_distances = read_fbin(gt_distances_path)

    return dataset_data, queries_data, gt_neighbors, gt_distances

def calculate_recall(r, gt, topk):

    n_queries = r.shape[0]
    if(topk == None):
        topk = r.shape[1]

    recalls = np.zeros(n_queries)
    
    for i in range(n_queries):
        gt_topk = gt[i][0:topk]

        for j in range(topk):
            if r[i][j] in gt_topk:
                recalls[i] += 1
        
        recalls[i] = recalls[i] / topk
        
    mean_recall = np.mean(recalls)
    
    return mean_recall

def get_torch_dtype(np_dtype):
    if np_dtype == np.float32:
        return torch.float32
    elif np_dtype == np.float64:
        return torch.float64
    elif np_dtype == np.int32:
        return torch.int32
    elif np_dtype == np.int64:
        return torch.int64
    elif np_dtype == np.uint32:
        return torch.uint32
    else:
        raise TypeError(f"Unsupported NumPy dtype: {np_dtype}")
    
def convert_to_torch(tensor):
    torch_dtype = get_torch_dtype(tensor.dtype)
    return torch.from_numpy(tensor).to(torch_dtype)

# def load_graph(base_path, graph_build_config, dataset):
#     graph = None
#     if not os.path.exists(base_path):
#         with open(f'{base_path}/{graph_build_config}/{dataset}.pkl', 'rb') as f:
#             graph = pickle.load(f)
#     else:
#         print(f"Graph file not found: {base_path}/{graph_build_config}/{dataset}.pkl")
#     return graph

def load_graph(base_path, algorithm, dataset):
    if os.path.exists(base_path):
        with open(f'{base_path}/{algorithm}/{dataset}.pkl', 'rb') as f:
            graph = pickle.load(f)
    else:
        print(f"Graph file not found: {base_path}/{algorithm}/{dataset}.pkl")

    return graph

def load_graph_1(graph_file_path):
    with open(graph_file_path, 'rb') as f:
        graph = pickle.load(f)
    return graph

def partition_graph(graph, num_parts):
    
    graph = graph.to(torch.int64)

    adjacency_list = []
    for node_idx in range(graph.size(0)):
        adjacency_list.append(graph[node_idx].tolist())

    cut, membership = pymetis.part_graph(num_parts, adjacency = adjacency_list)

    partitions = [[] for _ in range(num_parts)]

    for node_idx in range(len(membership)):
        for partition_idx in range(num_parts):
            if membership[node_idx] == partition_idx:
                partitions[partition_idx].append(node_idx)
    assert(sum([len(partition) for partition in partitions]) == graph.size(0))

    return cut, partitions

def generate_sign_bit(sign_path, build_config, dataset_name, dataset, graph):
    
    full_compressed_sign_bit_list = []
    
    # for node_index in tqdm(range(graph.shape[0])):
    for node_index in range(2):
        
        parent_vector = dataset[node_index]
        
        compressed_sign_bit_list = []
        for edge in graph[node_index]:
            child_vector = dataset[edge]
            direction = child_vector - parent_vector
            
            sign_bit_list = []
            for element in direction:
                if element > 0:
                    sign_bit = 1
                else:
                    sign_bit = 0
                sign_bit_list.append(sign_bit)
            
            for i in range(0, len(sign_bit_list), 32):
                bits = sign_bit_list[i:i+32]
                while len(bits) < 32:
                    print("Need padding!")
                value = int(''.join(map(str, bits)), 2)
                compressed_sign_bit_list.append(value)
            
        full_compressed_sign_bit_list.append(compressed_sign_bit_list)
    import code; code.interact(local=dict(globals(), **locals()))

    full_compressed_sign_bit_list = np.array(full_compressed_sign_bit_list)

    sign_dir = os.path.join(sign_path, build_config)
    sign_file = os.path.join(sign_dir, dataset_name + ".pkl")
    with open(sign_file, "wb") as f:
        pickle.dump(full_compressed_sign_bit_list, f)

    return full_compressed_sign_bit_list    

def load_sign_bit(sign_path, build_config, dataset_name):

    with open(f"{sign_path}/{build_config}/{dataset_name}.bin", "rb") as f:
        raw = f.read()
    sign_bit = np.frombuffer(raw, dtype=np.uint32)

    return sign_bit

def load_sign_bit_1(sign_file_path):

    with open(f"{sign_file_path}", "rb") as f:
        raw = f.read()
    sign_bit = np.frombuffer(raw, dtype=np.uint32)

    return sign_bit

def load_seed(seed_path, dataset_name, dataset_size, num_queries, graph_degree, search_width, num_dist):

    file_path = f"{seed_path}/{dataset_name}.pkl"
    if os.path.exists(file_path):
        with open(file_path, "rb") as f:
            seed = pickle.load(f)

    else:
        with open(file_path, "wb") as f:
            start_time = time.time()
            size_of_seed = num_queries * (graph_degree * search_width) * num_dist
            seed = np.random.randint(0, dataset_size, size_of_seed, dtype=np.uint32)
            end_time = time.time()
            print(f"Seed generation time: {end_time - start_time}")
            pickle.dump(seed, f)
    
    return seed

def calculate_shared_mem_size(VECTOR_DIM, INTERNAL_TOPK, GRAPH_DEGREE, SEARCH_WIDTH, BITLEN):
    RESULT_BUFFER_SIZE = INTERNAL_TOPK + SEARCH_WIDTH * GRAPH_DEGREE
    QUERY_BUFFER_SIZE = VECTOR_DIM
    
    hash_table_size = 1 << BITLEN

    query_buffer_size = QUERY_BUFFER_SIZE
    result_indices_buffer_size = RESULT_BUFFER_SIZE
    result_distances_buffer_size = RESULT_BUFFER_SIZE
    parent_buffer_size = SEARCH_WIDTH
    terminate_flag_size = 1
    counter_size = 1

    shared_size =  hash_table_size * 4\
                    + query_buffer_size * 4\
                    + result_indices_buffer_size * 4\
                    + result_distances_buffer_size * 4\
                    + parent_buffer_size * 4\
                    + terminate_flag_size * 4\
                    + counter_size * 4\

    return shared_size

def load_checkpoint_map_link(checkpoint_map_path, build_config, shard_name):
    with open(f"{checkpoint_map_path}/{build_config}/{shard_name}.pkl", "rb") as f:
        checkpoint_map = pickle.load(f)
    return checkpoint_map

def load_checkpoint_map_1(checkpoint_map_file_path):
    with open(checkpoint_map_file_path, "rb") as f:
        checkpoint_map = pickle.load(f)
    return checkpoint_map

def is_valid_prune_ratio(value):
    """Check if the value is between 0 and 1, and a multiple of 0.125."""
    return 0.0 <= value <= 1.0 and (value * 8).is_integer()