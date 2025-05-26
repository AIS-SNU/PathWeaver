#!/bin/bash

#============================ Single GPU Preprocessing ============================

python -m preprocess \
    --dataset_name sift-128-euclidean \
    --dataset_path ../_datasets \
    --graph_save_path ./_auxiliary_structure/graphs \
    --dgs_save_path ./_auxiliary_structure/direction_guided_selection \
    --pbpe_save_path ./_auxiliary_structure/pipeline_based_path_extension \
    --build_graph_degree 64 \
    --build_intermediate_graph_degree 128 \
    --num_shard 1 \

python -m preprocess \
    --dataset_name gist-960-euclidean \
    --dataset_path ../_datasets \
    --graph_save_path ./_auxiliary_structure/graphs \
    --dgs_save_path ./_auxiliary_structure/direction_guided_selection \
    --pbpe_save_path ./_auxiliary_structure/pipeline_based_path_extension \
    --build_graph_degree 64 \
    --build_intermediate_graph_degree 128 \
    --num_shard 1 \

python -m preprocess \
    --dataset_name deep-image-96-inner \
    --dataset_path ../_datasets \
    --graph_save_path ./_auxiliary_structure/graphs \
    --dgs_save_path ./_auxiliary_structure/direction_guided_selection \
    --pbpe_save_path ./_auxiliary_structure/pipeline_based_path_extension \
    --build_graph_degree 64 \
    --build_intermediate_graph_degree 128 \
    --num_shard 1 \

#============================ Multi GPU Preprocessing ============================

python -m preprocess \
    --dataset_name deep-image-96-inner \
    --dataset_path ../_datasets \
    --graph_save_path ./_auxiliary_structure/graphs \
    --dgs_save_path ./_auxiliary_structure/direction_guided_selection \
    --pbpe_save_path ./_auxiliary_structure/pipeline_based_path_extension \
    --build_graph_degree 64 \
    --build_intermediate_graph_degree 64 \
    --num_shard 4 \

python -m preprocess \
    --dataset_name wiki_all_10M \
    --dataset_path ../_datasets \
    --graph_save_path ./_auxiliary_structure/graphs \
    --dgs_save_path ./_auxiliary_structure/direction_guided_selection \
    --pbpe_save_path ./_auxiliary_structure/pipeline_based_path_extension \
    --build_graph_degree 64 \
    --build_intermediate_graph_degree 64 \
    --num_shard 4 \

python -m preprocess \
    --dataset_name deep_50M \
    --dataset_path ../_datasets \
    --graph_save_path ./_auxiliary_structure/graphs \
    --dgs_save_path ./_auxiliary_structure/direction_guided_selection \
    --pbpe_save_path ./_auxiliary_structure/pipeline_based_path_extension \
    --build_graph_degree 64 \
    --build_intermediate_graph_degree 64 \
    --num_shard 4 \


