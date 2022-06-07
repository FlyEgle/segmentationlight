#!/bin/bash 
OMP_NUM_THREADS=1
MKL_NUM_THREADS=1
export OMP_NUM_THREADS
export MKL_NUM_THREADS
cd /data/jiangmingchao/data/code/SegmentationLight;
CUDA_VISIBLE_DEVICES=0,1 python -W ignore -m torch.distributed.launch --master_port 2953 --nproc_per_node 2 main.py \
--hyp /data/jiangmingchao/data/code/SegmentationLight/hyparam/UNet/unet_resnet50.yaml