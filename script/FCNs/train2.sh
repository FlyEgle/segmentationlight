#!/bin/bash 
OMP_NUM_THREADS=1
MKL_NUM_THREADS=1
export OMP_NUM_THREADS
export MKL_NUM_THREADS
cd /data/jiangmingchao/data/code/SegmentationLight;
CUDA_VISIBLE_DEVICES=1 python -W ignore -m torch.distributed.launch  --master_port 29501 --nproc_per_node 1 main.py \
--hyp /data/jiangmingchao/data/code/SegmentationLight/hyparam/FCNs/fcn_mbv2_8s.yaml