"""
@author: FlyEgle
@datetime: 2022-04-07
@describe: loss function manage
"""
import torch.nn as nn 
# general loss
from losses.loss import CELoss, BCELoss, DiceWithBCELoss, Dice_Bce_L1, Dice_Ce_L1, IouWithDiceWithBCELoss, IOUWithBCELoss
# generate loss
from losses.generatorLoss import TVLoss


# ------------------------ BCE + 3* DICE + 0.5 * TV -------------------------
class Bce_Dice_TvLoss(nn.Module):
    def __init__(self):
        super(Bce_Dice_TvLoss, self).__init__()
        self.weights = {
            'bce': 1.0,
            'dice': 3.0,
            'tv': 0.1
        }
        self.dice_bce = DiceWithBCELoss(self.weights)
        self.tv = TVLoss()

    def forward(self, pred, tgts):
        if len(tgts.shape) == 3:
            tgts = tgts.unsqueeze(1).float()
        else:
            tgts =  tgts.float()

        losses = self.dice_bce(pred, tgts) +  self.weights['tv'] * self.tv(pred)
        return losses


class LossBar:
    """Loss Bar
    """
    def __init__(self, loss_name):
        self.loss_name = loss_name.lower()

    def __call__(self):
        weights = {
            'bce': 1.0,
            'dice': 1.0,
            'iou': 1.0
        }
        if self.loss_name == "bce":
            criterion = BCELoss()
        elif self.loss_name == "ce":
            criterion = CELoss(flatten=False)
        elif self.loss_name == "bce+dice":
            weights = {'bce':1.0, 'dice': 1.0}
            criterion = DiceWithBCELoss(weights, mining=False)
        elif self.loss_name == "bce+dice+iou":
            weights = {'bce':1.0, 'dice':1.0, 'iou': 1.0}
            criterion = IouWithDiceWithBCELoss(weights, mining=False)
        elif self.loss_name == "bce+iou":
            criterion = IOUWithBCELoss(weights, mining=False)
        elif self.loss_name == "bce+dice+l1":
            weights = {'bce':1.0, 'dice': 3.0}
            criterion = Dice_Bce_L1(weights)
        elif self.loss_name == "ce+dice+l1":
            criterion = Dice_Ce_L1()
        elif self.loss_name == "bce+dice+tv":
            criterion = Bce_Dice_TvLoss()
        return criterion
