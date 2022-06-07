"""Record the model Training log or image generate
@author: FlyEgle
@datetime: 2022-01-21
"""
import os 
import json 
import torch 
import numpy as np 
from tensorboardX import SummaryWriter


class LoggerRecord:
    def __init__(self, log_path) -> None:
        if os.path.exists(log_path):
            if not os.path.isdir(log_path):
                raise FileExistsError(f"{log_path} must be a folder!!!")
        else:
            os.makedirs(log_path)
        
        self.logger = SummaryWriter(log_path)

    def BatchMetricWritter(self, losses, pa, mpa, miou, fwiou, lr, batch_time, batch_iter, flag):
        self.logger.add_scalar(f"{flag}/batch/Loss", losses.data.item(), batch_iter)
        self.logger.add_scalar(f"{flag}/batch/PA", pa, batch_iter)
        self.logger.add_scalar(f"{flag}/batch/MPA", mpa, batch_iter)
        self.logger.add_scalar(f"{flag}/batch/MIoU", miou, batch_iter)
        self.logger.add_scalar(f"{flag}/batch/FwIoU", miou, batch_iter)
        self.logger.add_scalar(f"{flag}/batch/LearningRate", lr, batch_iter)
        self.logger.add_scalar(f"{flag}/batch/Times", batch_time, batch_iter)

    def EpochMetricWritter(self, losses, pa, mpa, miou, fwiou, epoch, flag):
        self.logger.add_scalar(f"{flag}/epoch/Loss", losses, epoch)
        self.logger.add_scalar(f"{flag}/epoch/PA", pa, epoch)
        self.logger.add_scalar(f"{flag}/epoch/MPA", mpa, epoch)
        self.logger.add_scalar(f"{flag}/epoch/MIoU", miou, epoch)
        self.logger.add_scalar(f"{flag}/epoch/FwIoU", miou, epoch)


    # todo
    def ImageCallback(self):
        raise NotImplementedError
    

class CkptRecord:
    def __init__(self, ckpt_path):
        if os.path.exists(ckpt_path):
            if not os.path.isdir(ckpt_path):
                raise FileExistsError("f{ckpt_path} must be a folder!!!")
        else:
            os.makedirs(ckpt_path)

        self.best_losses = np.inf
        self.best_pa = 0.0
        self.best_mpa =  0.0 
        self.best_miou = 0.0

        self.ckpt_queue = []
        self.ckpt_path = ckpt_path

    def SaveBestMIOU(self, state_dict, losses, miou, maxNum=10):
        """save only final 5 best ckpt"""
        
        if miou > self.best_miou:
            self.best_miou = miou
            output_name = f"best_ckpt_losses_{losses}_miou_{miou}.pth"
            save_path = os.path.join(self.ckpt_path, output_name)
            torch.save(state_dict, save_path)

            if len(self.ckpt_queue) <= maxNum:
                if os.path.exists(save_path):
                    self.ckpt_queue.append(save_path)
                else:
                    print(f"{save_path} is not exists!!!")
            else:
                save_path = self.ckpt_queue.pop(0)
                os.remove(save_path)

    def SaveBestCkpt(self, state_dict, epoch, losses, miou):
        loss_postive = False
        miou_postive = False

        if losses < self.best_losses:
            self.best_losses = losses
            loss_postive = True 
    
        if miou > self.best_miou:
            self.best_miou = miou
            miou_postive = True 

        if loss_postive or miou_postive:
            output_name = f"best_ckpt_epoch_{epoch}_losses_{losses}_miou_{miou}.pth"
            torch.save(state_dict, os.path.join(self.ckpt_path, output_name))
        

    def SaveCkpt(self, state_dict, losses, pa, mpa, miou):
        output_name = f"ckpt_losses_{losses}_pa_{pa}_mpa_{mpa}_miou_{miou}.pth"
        torch.save(state_dict, os.path.join(self.ckpt_path, output_name))


# TODO: draw confusion matrix
def draw_confusion_matrix(confusion_matrix):
    pass


class LoggerInfo:
    """save the training log
    """
    def __init__(self, save_path):
        self.save_path = save_path
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)
        self.logger_name = "train.log"
        self.logger_path = os.path.join(self.save_path, self.logger_name)
    
    def write(self, result):
        with open(self.logger_path, "a") as file:
            file.write(json.dumps(result) + '\n')
