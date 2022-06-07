"""Trainig & Validation on an epoch
@author: FlyEgle
@datetime: 2022-01-21
""" 
import sys 
import time 
import torch 
from datetime import datetime
from utils.LrSheduler import CosineLR, StepLR, FixLR, CirCleLR, make_iter
from utils.utils import check_tensor
from torch.cuda.amp import autocast as autocast 


def train_one_epoch(
        args,  
        opt,
        scaler,
        net,
        loader,
        mixup, 
        criterion,
        aux_criterion,
        optimizer,
        epoch,
        batch_iter,
        total_batch,
        train_batch,
        log_writter,
        log_info,
        metric
    ):
    """
    Trainig model on one epoch, record the each batch result and epoch result 
    Args:
        args: dist rank
        opt: hyp parameters
        scaler: amp scaler for loss and bp gradients
        net   : model
        loader : dataloader
        mixup  : mixup 
        criterion:  celoss or other loss function 
        aux_criterion: aux loss for mutil task or None 
        optimizer: sgd or adam and so on
        epoch: current epoch
        batch_iter: all batch iter accumulate 
        total_batch: dataset all batch iter of max epochs
        train_batch: each epoch batch iter
        log_writter: logger 
        log_info : logger training log
        metric : calculate the  confusion matrix and miou
    Returns:
        batch_iter : batch iter
        scaler     : amp scaler
    """
    net.train()
    # parse the loss function
    if "+" in opt.LOSS.lower():
        loss_name = opt.LOSS.lower().split("+")
        if not "ce" in loss_name:
        # if "bce" in opt.LOSS.lower() or "dice" in opt.LOSS.lower():
            bce_flag = True
        else:
            bce_flag = False
    else:
        if opt.LOSS.lower() != "ce":
            bce_flag = True
        else:
            bce_flag = False 
    
    data_length = len(loader)


    # TODO: add the cricle params
    if opt.OPTIMIZER.CRICLE:
        cricle_epochs_list = make_iter(opt, train_batch)

    # loop batch iter
    for batch_idx, data in enumerate(loader):
        start_time = time.time()
        if opt.OPTIMIZER.COSINE:
            lr = CosineLR(opt, epoch, batch_iter, optimizer, train_batch)
        elif opt.OPTIMIZER.FIX:
            lr = FixLR(opt)
        elif opt.OPTIMIZER.CRICLE:
            lr = CirCleLR(opt, epoch, batch_iter, optimizer, train_batch, cricle_epochs_list)
        else:
            lr = StepLR(opt, epoch, batch_iter, optimizer)
        
        # data
        images, targets = data[0], data[1]
        images = images.cuda()
        targets = targets.cuda()
        # mixup
        if mixup is not None:
            images, targets_a, targets_b, lam = mixup(images, targets)
        # forward with amp
        with autocast():
            outputs = net(images)
            # check tensor
            if batch_idx == 0 or batch_idx == data_length - 1:
                if (check_tensor(outputs[0])): # only for the u2net tensor
                    if args.local_rank == 0:
                        print("outputs have nan or inf, need stop the training!!!")
                    sys.exit(1)

            MUTIL_OUTS = False
            if isinstance(outputs, list) or isinstance(outputs, tuple):
                MUTIL_OUTS = True
                losses = 0.0
                losses_first = 0.0

                # mutil task u2net match function
                for i in range(len(outputs)):
                    if i <= 6: # only match the mutil task U2Net
                        if mixup is not None:
                            loss = criterion(outputs[i], targets_a, targets_b, lam)
                        else:
                            loss = criterion(outputs[i], targets)
                        # record the first outputs map losses
                        if i == 0:
                            losses_first = loss 
                        losses += loss 
                    else:
                        if mixup is not None:
                            #TODO: Need to impelemenation this switch with the mixup 
                            aux_loss = 0.0
                        else:
                            aux_loss = aux_criterion(outputs[i], targets.unsqueeze(1) * images)
                        losses += aux_loss 
            else:
                if  mixup is not None:
                    losses = criterion(outputs[i], targets_a, targets_b, lam)
                else:
                    losses = criterion(outputs, targets)
        # accumulate
        # loss regularization
        if opt.ACCUMULATE:
            losses = losses / opt.ACCUMULATE_STEPS
            scaler.scale(losses).backward()
            if ((batch_idx + 1) % opt.ACCUMULATE_STEPS) == 0:
                scaler.step(optimizer) 
                scaler.update()
                optimizer.zero_grad()
        else:
            # backward
            optimizer.zero_grad()
            scaler.scale(losses).backward()
            scaler.step(optimizer)          # if train loss is nan skip step 
            scaler.update()

        # metric
        if MUTIL_OUTS:
            # default the first order map is the best results
            metric.update(outputs[0], targets, losses, bce_flag)
        else:
            metric.update(outputs, targets, losses, bce_flag)
        batch_pa, batch_mpa, batch_miou, batch_fwiou = metric.batch_metric

        batch_time = time.time() - start_time
        batch_iter += 1

        if args.local_rank == 0:
            if MUTIL_OUTS:
                result = "[\033[1;35mTraining\033[0m][{}] Epoch: [{}/{}] batch_idx: [{}/{}] batch_iter: [{}/{}] Losses: {:.4f} Losses1:{:.4f} PA: {:.4f} MPA: {:.4f} MIOU: {:.4f} FWIOU:{:.4f} LR: {:.6f} BatchTime: {:.4f}".format(
                    datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    epoch,
                    opt.MAX_EPOCHS,
                    batch_idx,
                    train_batch,
                    batch_iter,
                    total_batch,
                    losses.data.item(),
                    losses_first.data.item(),
                    batch_pa,
                    batch_mpa,
                    batch_miou,
                    batch_fwiou,
                    lr,
                    batch_time
                )
                print(result)
                log_info.write(result)
            else:
                result = "[\033[1;35mTraining\033[0m] Epoch: [{}/{}] batch_idx: [{}/{}] batch_iter: [{}/{}] Losses: {:.4f}  PA: {:.4f} MPA: {:.4f} MIOU: {:.4f} FWIOU:{:.4f} LR: {:.6f} BatchTime: {:.4f}".format(
                    datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    epoch,
                    opt.MAX_EPOCHS,
                    batch_idx,
                    train_batch,
                    batch_iter,
                    total_batch,
                    losses.data.item(),
                    batch_pa,
                    batch_mpa,
                    batch_miou,
                    batch_fwiou,
                    lr,
                    batch_time
                )
                print(result)
                log_info.write(result)

        # batch log writter
        if args.local_rank == 0:
            log_writter.BatchMetricWritter(
                losses,
                batch_pa,
                batch_mpa,
                batch_miou,
                batch_fwiou,
                lr,
                batch_time,
                batch_iter,
                "Train"
            )
    
    epoch_pa, epoch_mpa, epoch_miou, epoch_fwiou = metric.epoch_metric
    epoch_losses = metric.average
    
    # epoch log writter
    if opt.DIST.LOCAL_RANK == 0:
        log_writter.EpochMetricWritter(
            epoch_losses,
            epoch_pa,
            epoch_mpa,
            epoch_miou,
            epoch_fwiou,
            epoch,
            "Train"
        )

    # metric reset
    metric.reset()

    return batch_iter, scaler

def val_one_epoch(args,
                  opt,
                  loader,
                  net,
                  criterion,
                  epoch,
                  val_batch, 
                  log_writter,
                  log_info,
                  metric):
    """
    Validation model on one epoch, record the each batch result and epoch result 
    Args:
        args: dist local rank
        opt: base config parameters
        net   : model
        loader : dataloader
        criterion:  celoss or other loss function 
        epoch: current epoch
        val_batch: val loader batch number for one epoch
        log_writter: logger 
        log_info: log for training log
        metric : calculate the  confusion matrix and miou
    Returns:
        losses:  float
        pa:   float
        mpa:  float
        miou: float
        
    """
    net.eval()
    #  parser the loss function
    if "+" in opt.LOSS.lower():
        loss_name = opt.LOSS.lower().split("+")
        if not "ce" in loss_name:
        # if "bce" in opt.LOSS.lower() or "dice" in opt.LOSS.lower():
            bce_flag = True
        else:
            bce_flag = False
    else:
        if opt.LOSS.lower() != "ce":
            bce_flag = True
        else:
            bce_flag = False  

    data_length = len(loader)

    with torch.no_grad():
        for batch_idx, data in enumerate(loader):
            images, targets = data[0], data[1]
            images = images.cuda()
            targets = targets.cuda()

            # forward with amp
            with autocast():
                outputs = net(images)
                # check tensor
                if batch_idx == 0 or batch_idx == data_length - 1:
                    if (check_tensor(outputs[0])):
                        if args.local_rank == 0:
                            print("outputs have nan or inf, need stop the training!!!")
                            print("save the current checkpoints to check value!!!")
                            model_state = net.state_dict()
                            state_dict = {
                                    'epoch': epoch,
                                    'state_dict': model_state,
                            }
                            torch.save(state_dict, "./exception/nan_ckpt.pth")
                        sys.exit(1)

                MUTIL_OUTS = False
                if isinstance(outputs, list) or isinstance(outputs, tuple):
                    MUTIL_OUTS = True
                    losses = 0.0
                    losses_first = 0.0
                    for i in range(len(outputs)):
                        if i <= 6: # only for u2net mutil task
                            loss = criterion(outputs[i], targets)
                            # record the first outputs map losses
                            if i == 0:
                                losses_first = loss 
                            losses += loss 
                else:
                    losses = criterion(outputs, targets)
            
            # metric
            if MUTIL_OUTS:
                metric.update(outputs[0], targets, losses, bce_flag)
            else:
                metric.update(outputs, targets, losses, bce_flag)
            batch_pa, batch_mpa, batch_miou, batch_fwiou = metric.batch_metric

            # batch print
            if args.local_rank == 0:
                if MUTIL_OUTS:
                    result = "[\033[1;34mValidation\033[0m][{}] Epoch: [{}/{}] batch_idx: [{}/{}] Losses: {:.4f} Losses0:{:4f} PA: {:.4f} MPA: {:.4f} MIOU: {:.4f} FWIOU:{:.4f}".format(
                    datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    epoch,
                    opt.MAX_EPOCHS,
                    batch_idx + 1,
                    val_batch,
                    losses.data.item(),
                    losses_first.data.item(),
                    batch_pa,
                    batch_mpa,
                    batch_miou,
                    batch_fwiou
                    )
                    print(result)
                    log_info.write(result)
                else:
                    result = "[\033[1;34mValidation\033[0m][{}] Epoch: [{}/{}] batch_idx: [{}/{}] Losses: {:.4f}  PA: {:.4f} MPA: {:.4f} MIOU: {:.4f} FWIOU:{:.4f}".format(
                        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        epoch,
                        opt.MAX_EPOCHS,
                        batch_idx + 1,
                        val_batch,
                        losses.data.item(),
                        batch_pa,
                        batch_mpa,
                        batch_miou,
                        batch_fwiou
                    )
                    print(result)
                    log_info.write(result)

        # epoch metric 
        epoch_pa, epoch_mpa, epoch_miou, epoch_fwiou = metric.epoch_metric
        epoch_losses = metric.average

        # epoch print & record
        if args.local_rank == 0:
            result = "[\033[1;34mValidation\033[0m][{}]|Epoch: [{}/{}]|Losses: {:.4f}|PA: {:.4f}|MPA: {:.4f}|MIOU: {:.4f}|FWIOU:{:.4f}".format(
                    datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    epoch,
                    opt.MAX_EPOCHS,
                    epoch_losses,
                    epoch_pa,
                    epoch_mpa,
                    epoch_miou,
                    epoch_fwiou
                )
            print(result)
            log_info.write(result)
            # epoch log writter
            log_writter.EpochMetricWritter(
                epoch_losses,
                epoch_pa,
                epoch_mpa,
                epoch_miou,
                epoch_fwiou,
                epoch,
                "Val"
            )

    # reset
    metric.reset()
    
    return epoch_losses, epoch_pa, epoch_mpa, epoch_miou, epoch_fwiou