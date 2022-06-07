"""LR sheduler
@author: FlyEgle
@datetime: 2022-01-21
"""
import math
 

def StepLR(opt, epoch, batch_iter, optimizer, train_batch):
    """Sets the learning rate
    # Adapted from PyTorch Imagenet example:
    # https://github.com/pytorch/examples/blob/master/imagenet/main.py
    """
    total_epochs = opt.MAX_EPOCHS
    warm_epochs = opt.WARMUP_EPOCHS
    if epoch <= warm_epochs:
        lr_adj = (batch_iter + 1) / (warm_epochs * train_batch)
    elif epoch < int(0.3 * total_epochs):
        lr_adj = 1.
    elif epoch < int(0.6 * total_epochs):
        lr_adj = 1e-1
    elif epoch < int(0.8 * total_epochs):
        lr_adj = 1e-2
    else:
        lr_adj = 1e-3

    for param_group in optimizer.param_groups:
        param_group['lr'] = opt.OPTIMIZER.LEARNING_RATE * lr_adj
    return opt.OPTIMIZER.LEARNING_RATE * lr_adj


def CosineLR(opt, epoch, batch_iter, optimizer, train_batch):
    """Cosine Learning rate 
    """
    total_epochs = opt.MAX_EPOCHS
    warm_epochs = opt.WARMUP_EPOCHS
    if epoch <= warm_epochs:
        lr_adj = (batch_iter + 1) / (warm_epochs * train_batch) + 1e-6
    else:
        lr_adj = 1/2 * (1 + math.cos(batch_iter * math.pi / ((total_epochs - warm_epochs) * train_batch))) + 1e-6

    for param_group in optimizer.param_groups:
        param_group['lr'] = opt.OPTIMIZER.LEARNING_RATE * lr_adj
    return opt.OPTIMIZER.LEARNING_RATE * lr_adj


def FixLR(opt):
    """Fix Learning Rate
    """
    lr_adj = 1
    return opt.OPTIMIZER.LEARNING_RATE * lr_adj


def PolyLR(opt, epoch, batch_iter, optimizer, train_batch):
    """Poly LR follow the Rethinking Atrous Convolution for Semantic Image Segmentation
    """
    total_batch = train_batch * opt.MAX_EPOCHS
    lr_adj = 1 - math.pow(batch_iter, total_batch) ** 0.9
    
    for param_group in optimizer.param_groups:
        param_group['lr'] = opt.OPTIMIZER.LEARNING_RATE * lr_adj 

    return opt.OPTIMIZER.LEARNING_RATE * lr_adj


# make the cricle need iter param
def make_iter(opt, train_batch):
    total_epochs = opt.MAX_EPOCHS
    warm_epochs = opt.WARMUP_EPOCHS
    cricle = opt.OPTIMIZER.CRICLE_STEPS

    warm_iter = warm_epochs*train_batch
    epochs_list = [x+1 for x in range(total_epochs * train_batch)]
    # print(len(epochs_list))
    cricle_epochs = int((total_epochs - warm_epochs) * train_batch / cricle)
    # print(cricle_epochs)
    cricle_epochs_list = [None for _ in range(cricle)]

    for i in range(cricle):
        if i == 0:
            cricle_epochs_list[i] = epochs_list[warm_iter: cricle_epochs + warm_iter]
        elif i == cricle - 1:
            cricle_epochs_list[i] = epochs_list[cricle_epochs*i + warm_iter: ]
        else:
            cricle_epochs_list[i] = epochs_list[cricle_epochs*i + warm_iter: cricle_epochs*(i+1)+ warm_iter]
        
    return cricle_epochs_list


def CirCleLR(opt, epoch, batch_iter, optimizer, train_batch,  cricle_epochs_list):
    """Circle Cosine LR + WarmUp
    """
    warm_epochs = opt.WARMUP_EPOCHS

    warm_iter = warm_epochs * train_batch

    if epoch <= warm_epochs:
        lr_adj = (batch_iter + 1) / (warm_epochs * train_batch) + 1e-6
    else:
        for i in range(len(cricle_epochs_list)):
            # print(batch_iter)
            # print(cricle_epochs_list[0])
            # restart the batchidx
            if i == 0:
                if (batch_iter+1) in cricle_epochs_list[i]:
                    batch_idx = batch_iter + 1- warm_iter 
                    # print("batch_idx: ", batch_idx)
                    lr_adj = 1/2 * (1 + math.cos(batch_idx * math.pi / (len(cricle_epochs_list[i])))) + 1e-6
            else:
                if (batch_iter+1) in cricle_epochs_list[i]:
                    batch_idx = batch_iter + 1 - warm_iter - len(cricle_epochs_list[i]) * i 
                    # print("batch_idx: ", batch_idx)
                    lr_adj = 1/2 * (1 + math.cos(batch_idx * math.pi / (len(cricle_epochs_list[i])))) + 1e-6

    for param_group in optimizer.param_groups:
        param_group['lr'] = opt.OPTIMIZER.LEARNING_RATE * lr_adj
    return opt.OPTIMIZER.LEARNING_RATE * lr_adj
