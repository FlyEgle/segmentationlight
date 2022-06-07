"""Traing model 
@author: FlyEgle
@datetime: 2022-01-20
"""
import warnings
warnings.filterwarnings("ignore")

import os 
import math 
import torch 
import numpy as np 
import torch.nn as nn  
 
import torch.distributed as dist

from torch.cuda.amp import autocast as autocast 
from torch.nn.parallel  import DistributedDataParallel as DistParallel
from torch.utils.data import DistributedSampler, DataLoader

# model
from config.config import build_argparse, parse_yaml
# Optimizer
from utils.Optim import BuildOptim
# Metric
from utils.Metirc import SegmentationMetric
# loss 
from utils.Loss import LossBar
# DataSet
from datasets.voc2012.dataset import VocSemanticSegDataSet, build_transformers, build_val_transformers
# Model 
from models.model_factory import ModelFactory
# Complex Aug
from utils.FuseAugments import MixUP, MixCriterion
# Train 
from train import train_one_epoch, val_one_epoch
# Summary
from utils.Summary import LoggerRecord, LoggerInfo, CkptRecord
# Utils function
from utils.utils import Load_state_dict


# random init seed 
def random_init(seed=100):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False


def translate_state_dict(state_dict):
    new_state_dict = {}
    for key, value in state_dict.items():
        if 'module' in key:
            new_state_dict[key[7:]] = value
        else:
            new_state_dict[key] = value
    return new_state_dict


def main_worker(args, opt):
    total_rank = opt.DIST.NGPU * torch.cuda.device_count()
    print('rank: {} / {}'.format(args.local_rank, total_rank))
    dist.init_process_group(backend=opt.DIST.DIST_BACKEND)
    torch.cuda.set_device(args.local_rank)

    ngpus_per_node = total_rank

    if opt.DATASET_TYPE.lower() ==  "voc":
        NUM_CLASSES = opt.VOC_DATASET.NUM_CLASSES
        TRAIN_DATA = opt.VOC_DATASET.TRAIN_FILE
        VAL_DATA  = opt.VOC_DATASET.VAL_FILE
    elif opt.DATASET_TYPE.lower() ==  "voc_aug":
        NUM_CLASSES = opt.VOC_DATASET.NUM_CLASSES
        TRAIN_DATA = opt.VOC_DATASET.TRAIN_FILE
        VAL_DATA  = opt.VOC_DATASET.VAL_FILE  
    elif opt.DATASET_TYPE.lower() == "custom":
        NUM_CLASSES = opt.CUSTOM_DATASET.NUM_CLASSES
        TRAIN_DATA = opt.CUSTOM_DATASET.TRAIN_FILE
        VAL_DATA = opt.CUSTOM_DATASET.VAL_FILE

    # training metric
    train_metric = SegmentationMetric("Train", NUM_CLASSES)
    val_metric = SegmentationMetric("Val", NUM_CLASSES)

    # model
    model_factory = ModelFactory()
    net = model_factory.getattr(opt.MODEL_NAME)(num_classes=NUM_CLASSES)
    if opt.PRETRAIN:
        state = Load_state_dict(opt.PRETRAIN_WEIGHTS, net)
        net.load_state_dict(state)

        print("Load the pretrain from real domain dataset!!!")

    # resume
    if opt.RESUME:
        ckpt = torch.load(opt.RESUME_CHECKPOINTS, map_location="cpu")
        resume_start_epoch = ckpt["epoch"]
        optim_state_dict = ckpt["optimizer"]

        state = Load_state_dict(opt.RESUME_CHECKPOINTS, net)
        net.load_state_dict(state)
        print("Load the resume checkpoints for follow training!!!")


    if args.local_rank == 0:
        print(f"===============model arch ===============")
        print(net)

    if torch.cuda.is_available():
        net.cuda(args.local_rank)
    
    # build loss function
    criterion = LossBar(opt.LOSS.lower())()
    if "mutil" in opt.MODEL_NAME.lower():
        aux_criterion = nn.L1Loss()
    else:
        aux_criterion = None 

    # mixup 
    if opt.BATCH_AUG.MIXUP:
        mixup = MixUP(alpha=1.0, cuda=torch.cuda.is_available())
        mix_criterion = MixCriterion(criterion)
    else:
        mixup = None
        mix_criterion = criterion

    # build Optim 
    optim = BuildOptim(
        opt.OPTIMIZER.OPTIM_NAME,
        opt.OPTIMIZER.LEARNING_RATE,
        opt.OPTIMIZER.WEIGHT_DECAY,
        opt.OPTIMIZER.MOMENTUM
    )(net.parameters())

    if opt.DIST.DISTRIBUTED and opt.SYNCBN:
        net = nn.SyncBatchNorm.convert_sync_batchnorm(net)

    if opt.DIST.DISTRIBUTED:
        net = DistParallel(net, device_ids=[args.local_rank], find_unused_parameters=False)
    # dataset & dataloader
    TrainDataset = VocSemanticSegDataSet(
        TRAIN_DATA, 
        transformers=build_transformers(opt.DATA.CROP_SIZE),   
        # transformers=build_val_transformers(opt.DATA.CROP_SIZE),  # used for the fixres
        train_phase=True 
        )
    ValidationDataset = VocSemanticSegDataSet(
        VAL_DATA,
        transformers=build_val_transformers(opt.DATA.CROP_SIZE),
        train_phase=False
    )

    if args.local_rank == 0:
        print("Training Dataset length: ", len(TrainDataset))
        print("Validation Dataset length: ", len(ValidationDataset))

    if opt.DIST.DISTRIBUTED:
        TrainSampler = DistributedSampler(TrainDataset)
        ValidationSampler = DistributedSampler(ValidationDataset)
    else:
        TrainSampler = None 
        ValidationSampler = None 

    # dataloader
    TrainLoader =  DataLoader(
        dataset = TrainDataset,
        batch_size = opt.OPTIMIZER.BATCH_SIZE, 
        shuffle = (TrainSampler is None),
        num_workers = opt.OPTIMIZER.NUM_WORKERS,
        pin_memory = True,
        sampler = TrainSampler,  
        drop_last = True
    )
    ValidationLoader = DataLoader(
        dataset = ValidationDataset,
        batch_size = opt.OPTIMIZER.BATCH_SIZE, 
        shuffle = (ValidationSampler is None),
        num_workers  = opt.OPTIMIZER.NUM_WORKERS,
        pin_memory = True,
        sampler = ValidationSampler,
        drop_last = False
    )

    # log & ckpt
    if args.local_rank == 0:
        logger_writter = LoggerRecord(os.path.join(opt.SUMMARY.SAVE_PATH, opt.SUMMARY.LOG))     # log
        logger_info = LoggerInfo(os.path.join(opt.SUMMARY.SAVE_PATH, opt.SUMMARY.LOG))          # logger
        ckpt_saver = CkptRecord(os.path.join(opt.SUMMARY.SAVE_PATH, opt.SUMMARY.CHECKPOINTS))   # ckpt
    else:
        logger_writter = None
        logger_info = None 
        ckpt_saver = None 
    
    # training params
    if opt.RESUME:
        start_epoch = resume_start_epoch
        optim.load_state_dict(optim_state_dict)
        batch_iter = train_batch * start_epoch    
    else:
        start_epoch = 1 
        batch_iter = 0

    # train_batch = math.ceil(len(TrainLoader) / (args.batch_size * ngpus_per_node))
    train_batch = len(TrainLoader)
    total_batch = train_batch * opt.MAX_EPOCHS
    print("train_batch: ", train_batch)

    val_batch = math.ceil(len(ValidationDataset) / (opt.OPTIMIZER.BATCH_SIZE * ngpus_per_node))

    scaler = torch.cuda.amp.GradScaler()
    # training loop
    for epoch in range(start_epoch, opt.MAX_EPOCHS + 1):
        if opt.DIST.DISTRIBUTED:
            TrainSampler.set_epoch(epoch)
        # train
        batch_iter, scaler = train_one_epoch(
            args, opt, scaler, net, TrainLoader, 
            mixup, mix_criterion, aux_criterion, 
            optim, epoch, batch_iter, total_batch, 
            train_batch, logger_writter, logger_info,
            train_metric
            )
        # val 
        if epoch % opt.FREQENCE == 0:
            val_losses, val_pa, val_mpa, val_miou, val_fwiou = val_one_epoch(
                args, opt, ValidationLoader, net, 
                criterion, epoch, val_batch,
                logger_writter, logger_info, 
                val_metric)
            # save ckpt 
            if args.local_rank == 0:
                model_state = translate_state_dict(net.state_dict())
                state_dict = {
                    'epoch': epoch,
                    'state_dict': model_state,
                    'optimizer': optim.state_dict()
                }
                ckpt_saver.SaveBestCkpt(state_dict, epoch, val_losses, val_miou)
            
        net.train()
            

if __name__ == "__main__":
    args = build_argparse()
    
    opt = parse_yaml(args.hyp)
    print(opt)
    random_init()

    main_worker(args, opt)