# #!/bin/bash 
# OMP_NUM_THREADS=1
# MKL_NUM_THREADS=1
# export OMP_NUM_THREADS
# export MKL_NUM_THREADS
# cd /data/jiangmingchao/data/code/SegmentationLight;
# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -W ignore -m torch.distributed.launch --nproc_per_node 8 main.py \
# --batch-size 32 \
# --num-workers 32 \
# --lr 1e-3 \
# --optim-name "sgd" \
# --cosine 1 \
# --fix 0 \
# --max-epochs 300 \
# --warmup-epochs 0 \
# --num-classes 21 \
# --crop-size 520 \
# --weight-decay 5e-4 \
# --train-file /data/jiangmingchao/data/code/SegmentationLight/datasets/voc2012/data/segclass/seg_train.log \
# --val-file /data/jiangmingchao/data/code/SegmentationLight/datasets/voc2012/data/segclass/seg_val.log \
# --ckpt-path /data/jiangmingchao/data/AICutDataset/Segmentation/FCN/fcn_baseline_lr_1e-3_cosine_aug_rotate_300epoch_flatten_ce_modify_fcn_520/checkpoints \
# --log-dir /data/jiangmingchao/data/AICutDataset/Segmentation/FCN/fcn_baseline_lr_1e-3_cosine_aug_rotate_300epoch_flatten_ce_modify_fcn_520/log_dir

#!/bin/bash 
OMP_NUM_THREADS=1
MKL_NUM_THREADS=1
export OMP_NUM_THREADS
export MKL_NUM_THREADS
cd /data/jiangmingchao/data/code/SegmentationLight;
CUDA_VISIBLE_DEVICES=0 python -W ignore -m torch.distributed.launch --nproc_per_node 1 main.py \
--batch-size 32 \
--num-workers 32 \
--lr 7e-3 \
--optim-name "sgd" \
--cosine 1 \
--fix 0 \
--max-epochs 50 \
--warmup-epochs 0 \
--num-classes 21 \
--crop-size 500 \
--weight-decay 5e-4 \
--train-file /data/jiangmingchao/data/code/SegmentationLight/datasets/voc2012/data/voc_aug/seg_train.log \
--val-file /data/jiangmingchao/data/code/SegmentationLight/datasets/voc2012/data/voc_aug/seg_val.log \
--ckpt-path /data/jiangmingchao/data/AICutDataset/Segmentation/deeplab/deeplab_ddp_8nodes/checkpoints \
--log-dir /data/jiangmingchao/data/AICutDataset/Segmentation/deeplab/deeplab_ddp_8nodes/log_dir