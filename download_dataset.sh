#!/bin/bash

DATASETS_PATH=_datasets
mkdir -p $DATASETS_PATH

cd prepare_dataset

#=========================== Single-GPU Datasets ===========================
# download
python download_and_convert_to_bin_format.py --dataset sift-128-euclidean --dataset-path ../$DATASETS_PATH --normalize
python download_and_convert_to_bin_format.py --dataset deep-image-96-angular --dataset-path ../$DATASETS_PATH --normalize
python download_and_convert_to_bin_format.py --dataset gist-960-euclidean --dataset-path ../$DATASETS_PATH --normalize

# convert to vecs format
python convert_to_vecs_format.py --dataset_name sift-128-euclidean --base_path ../$DATASETS_PATH
python convert_to_vecs_format.py --dataset_name deep-image-96-inner --base_path ../$DATASETS_PATH
python convert_to_vecs_format.py --dataset_name gist-960-euclidean --base_path ../$DATASETS_PATH

#=========================== Multi-GPU Datasets ===========================
# download
cd ../$DATASETS_PATH
mkdir -p wiki_all_10M
curl -s https://data.rapids.ai/raft/datasets/wiki_all_10M/wiki_all_10M.tar -o ./wiki_all_10M.tar
tar -xf ./wiki_all_10M.tar -C ./wiki_all_10M
mv ./wiki_all_10M/base.10M.fbin ./wiki_all_10M/base.fbin
mv ./wiki_all_10M/queries.fbin ./wiki_all_10M/query.fbin
mv ./wiki_all_10M/groundtruth.10M.distances.fbin ./wiki_all_10M/groundtruth.distances.fbin
mv ./wiki_all_10M/groundtruth.10M.neighbors.ibin ./wiki_all_10M/groundtruth.neighbors.ibin

mkdir -p deep_50M
