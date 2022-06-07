#!/bin/bash 
OMP_NUM_THREADS=1
MKL_NUM_THREADS=1
export OMP_NUM_THREADS
export MKL_NUM_THREADS
cd /data/jiangmingchao/data/code/SegmentationLight;
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -W ignore -m torch.distributed.launch \
--master_port 2953 \
--nproc_per_node 8 main.py \
--hyp /data/jiangmingchao/data/code/SegmentationLight/hyparam/U2Net/baseline_bce_dice_pretrain_320_data.yaml