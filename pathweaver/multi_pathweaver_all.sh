#!/bin/bash

# Dataset and path configs
DATASET_NAME=deep-image-96-inner
DATASET_PATH=../_datasets
GRAPH_PATH=./_auxiliary_structure/graphs
DGS_PATH=./_auxiliary_structure/direction_guided_selection
PBPE_PATH=./_auxiliary_structure/pipeline_based_path_extension
BUILD_GRAPH_DEGREE=64
BUILD_INTERMEDIATE_GRAPH_DEGREE=64

# Search parameters
TEST_ITERATION=4
FILE_SAVE=False
GHOST_SCALE_FACTOR=10
NUM_GPU=4

GHOST_INTERNAL_TOPK=64
BASE_INTERNAL_TOPK=64

GHOST_MAX_ITERS=(1 2 3)
BASE_MAX_ITERS=(10 15 20 25 30 35 40 45 50 55 60 65 70 75 80)

BASE_PRUNE_CONFIG=2
NEIGHBOR_PRUNE_RATIOS=(0.125 0.25 0.375 0.5 0.625)
ITERATION_PRUNE_RATIOS=(0.3 0.4 0.5 0.6 0.7 0.8 0.9)
ITERATION_ADVENTAGES=(1 2 3 4 5)

# Run
for ghost_max_iter in "${GHOST_MAX_ITERS[@]}"; do
    for base_max_iter in "${BASE_MAX_ITERS[@]}"; do
        for iteration_prune_ratio in "${ITERATION_PRUNE_RATIOS[@]}"; do
            for neighbor_prune_ratio in "${NEIGHBOR_PRUNE_RATIOS[@]}"; do
                for iteration_adventage in "${ITERATION_ADVENTAGES[@]}"; do
                    
                    python multi_pathweaver_one.py \
                        --dataset_name $DATASET_NAME \
                        --dataset_path $DATASET_PATH \
                        --graph_save_path $GRAPH_PATH \
                        --dgs_save_path $DGS_PATH \
                        --pbpe_save_path $PBPE_PATH \
                        --build_graph_degree $BUILD_GRAPH_DEGREE \
                        --build_intermediate_graph_degree $BUILD_INTERMEDIATE_GRAPH_DEGREE \
                        --num_gpu $NUM_GPU \
                        --test_iteration $TEST_ITERATION \
                        --file_save $FILE_SAVE \
                        --ghost_scale_factor $GHOST_SCALE_FACTOR \
                        --ghost_internal_topk $GHOST_INTERNAL_TOPK \
                        --base_internal_topk $BASE_INTERNAL_TOPK \
                        --ghost_max_iter $ghost_max_iter \
                        --base_max_iter $base_max_iter \
                        --base_prune_config $BASE_PRUNE_CONFIG \
                        --base_iteration_prune_ratio $iteration_prune_ratio \
                        --base_neighbor_prune_ratio $neighbor_prune_ratio \
                        --base_234_iteration_adventage $iteration_adventage

                done
            done
        done
    done
done
