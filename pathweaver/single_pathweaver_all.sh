#!/bin/bash

DATASET_NAME=sift-128-euclidean
DATASET_PATH=../_datasets
GRAPH_PATH=./_auxiliary_structure/graphs
DGS_PATH=./_auxiliary_structure/direction_guided_selection
BUILD_GRAPH_DEGREE=64
BUILD_INTERMEDIATE_GRAPH_DEGREE=128

TEST_ITERATION=4
FILE_SAVE=False
GHOST_SCALE_FACTOR=100

GHOST_INTERNAL_TOPK=64
BASE_INTERNAL_TOPK_64=64
BASE_INTERNAL_TOPK_128=128

GHOST_MAX_ITERS=(1 2 3)
BASE_MAX_ITERS_64=(32 48 64 80)
BASE_MAX_ITERS_128=(64 80 96 112 128 144 160 176 192)

BASE_PRUNE_CONFIG=2
NEIGHBOR_PRUNE_RATIOS=(0.125 0.25 0.375 0.5 0.625)
ITERATION_PRUNE_RATIOS=(0.3 0.4 0.5 0.6 0.7 0.8 0.9)

# BASE_INTERNAL_TOPK_64
for ghost_max_iter in "${GHOST_MAX_ITERS[@]}"; do
    for base_max_iter_64 in "${BASE_MAX_ITERS_64[@]}"; do
        for base_prune_ratio in "${NEIGHBOR_PRUNE_RATIOS[@]}"; do
            for iteration_prune_ratio in "${ITERATION_PRUNE_RATIOS[@]}"; do
                echo "Running: ghost_iter=$ghost_max_iter, base_iter=$base_max_iter_64, prune=$base_prune_ratio, iter_prune=$iteration_prune_ratio"

                python single_pathweaver_one.py \
                    --dataset_name $DATASET_NAME \
                    --dataset_path $DATASET_PATH \
                    --graph_save_path $GRAPH_PATH \
                    --dgs_save_path $DGS_PATH \
                    --build_graph_degree $BUILD_GRAPH_DEGREE \
                    --build_intermediate_graph_degree $BUILD_INTERMEDIATE_GRAPH_DEGREE \
                    --test_iteration $TEST_ITERATION \
                    --file_save $FILE_SAVE \
                    --ghost_scale_factor $GHOST_SCALE_FACTOR \
                    --ghost_internal_topk $GHOST_INTERNAL_TOPK \
                    --base_internal_topk $BASE_INTERNAL_TOPK_64 \
                    --ghost_max_iter $ghost_max_iter \
                    --base_max_iter $base_max_iter_64 \
                    --base_prune_config $BASE_PRUNE_CONFIG \
                    --base_neighbor_prune_ratio $base_prune_ratio \
                    --base_iteration_prune_ratio $iteration_prune_ratio

            done
        done
    done
done

# BASE_INTERNAL_TOPK_128
for ghost_max_iter in "${GHOST_MAX_ITERS[@]}"; do
    for base_max_iter_128 in "${BASE_MAX_ITERS_128[@]}"; do
        for base_prune_ratio in "${NEIGHBOR_PRUNE_RATIOS[@]}"; do
            for iteration_prune_ratio in "${ITERATION_PRUNE_RATIOS[@]}"; do
                echo "Running: ghost_iter=$ghost_max_iter, base_iter=$base_max_iter_64, prune=$base_prune_ratio, iter_prune=$iteration_prune_ratio"

                python single_pathweaver_one.py \
                    --dataset_name $DATASET_NAME \
                    --dataset_path $DATASET_PATH \
                    --graph_save_path $GRAPH_PATH \
                    --dgs_save_path $DGS_PATH \
                    --build_graph_degree $BUILD_GRAPH_DEGREE \
                    --build_intermediate_graph_degree $BUILD_INTERMEDIATE_GRAPH_DEGREE \
                    --test_iteration $TEST_ITERATION \
                    --file_save $FILE_SAVE \
                    --ghost_scale_factor $GHOST_SCALE_FACTOR \
                    --ghost_internal_topk $GHOST_INTERNAL_TOPK \
                    --base_internal_topk $BASE_INTERNAL_TOPK_128 \
                    --ghost_max_iter $ghost_max_iter \
                    --base_max_iter $base_max_iter_128 \
                    --base_prune_config $BASE_PRUNE_CONFIG \
                    --base_neighbor_prune_ratio $base_prune_ratio \
                    --base_iteration_prune_ratio $iteration_prune_ratio
                    
            done
        done
    done
done