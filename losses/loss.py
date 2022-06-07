"""loss function for segmentation
@author:  FlyEgle
@datetime: 2021-01-20
"""
import torch 
import torch.nn as nn 
import torch.nn.functional as F 

class CELoss(nn.Module):
    """CrossEntropyLoss for segmentation 
    """
    def __init__(self, flatten=False, weights=None):
        super(CELoss, self).__init__()
        self.flatten = flatten
        if weights is not None:
            if not isinstance(weights, torch.Tensor):
                raise TypeError("weights must be tensor")
            elif weights.ndim != 1:
                raise ValueError("weights must be shape [classes]")
            self.weights = weights 
        else:
            self.weights = None
        
    def forward(self, inputs, targets):
        assert len(inputs.shape)  == 4, "inputs shape must be 4 dims, NXCXHXW"
        assert len(targets.shape) == 3, "targets shape must be 3 dims, NXHXW"
        
        N, C, H, W =  inputs.shape
        # flatten NCHW->(NHW)C
        if self.flatten:
            log_p = F.log_softmax(inputs, dim=1)
            # log_p: (n*h*w, c)
            log_p = log_p.transpose(1, 2).transpose(2, 3).contiguous()
            log_p = log_p[targets.view(N, H, W, 1).repeat(1, 1, 1, C) >= 0]
            log_p = log_p.view(-1, C)
            # target: (n*h*w,)
            targets = targets.view(-1)

            mask = targets >= 0
            targets = targets[mask]
            losses = F.nll_loss(log_p, targets, weight=self.weights, reduction='sum')
            losses /= mask.data.sum()
        else:
            losses = F.cross_entropy(inputs, targets)
        
        return losses 


# Weights Cross Entropy Loss for each batch
class BatchWeightCELoss(nn.Module):
    def  __init__(self, flatten, num_classes):
        super(BatchWeightCELoss, self).__init__()
        self.flatten = flatten 
        self.num_classes = num_classes

    def forward(self, inputs, targets):
        assert len(inputs.shape)  == 4, "inputs shape must be 4 dims, NXCXHXW"
        assert len(targets.shape) == 3, "targets shape must be 3 dims, NXHXW"

        batch_nums = torch.bincount(targets.contiguous().view(-1))
        total_nums = torch.sum(batch_nums)
        if total_nums != 0:
            weights = torch.tensor([(total_nums - x) / total_nums for x in batch_nums])
        else:
            weights = torch.ones(batch_nums.shape)

        N, C, H, W =  inputs.shape
        # flatten NCHW->(NHW)C
        if self.flatten:
            log_p = F.log_softmax(inputs, dim=1)
            # log_p: (n*h*w, c)
            log_p = log_p.transpose(1, 2).transpose(2, 3).contiguous()
            log_p = log_p[targets.view(N, H, W, 1).repeat(1, 1, 1, C) >= 0]
            log_p = log_p.view(-1, C)
            # target: (n*h*w,)
            targets = targets.view(-1)

            mask = targets >= 0
            targets = targets[mask]
            losses = F.nll_loss(log_p, targets, weight=weights, reduction='sum')
            losses /= mask.data.sum()
        else:
            losses = F.cross_entropy(inputs, targets, weight=weights)
        
        return losses 


class BalanceCrossEntropyLoss(nn.Module):
    '''
    Balanced cross entropy loss.
    Shape:
        - Input: :math:`(N, 1, H, W)`
        - GT: :math:`(N, 1, H, W)`, same shape as the input
        - Mask: :math:`(N, H, W)`, same spatial shape as the input
        - Output: scalar.
    Examples::
        >>> m = nn.Sigmoid()
        >>> loss = nn.BCELoss()
        >>> input = torch.randn(3, requires_grad=True)
        >>> target = torch.empty(3).random_(2)
        >>> output = loss(m(input), target)
        >>> output.backward()
    '''
    def __init__(self, negative_ratio=3.0, eps=1e-6):
        super(BalanceCrossEntropyLoss, self).__init__()
        self.negative_ratio = negative_ratio
        self.eps = eps

    def forward(self,
                pred: torch.Tensor,
                gt: torch.Tensor,
                return_origin=False):
        '''
        Args:
            pred: shape :math:`(N, 1, H, W)`, the prediction of network
            gt: shape :math:`(N, 1, H, W)`, the target
        '''
        assert len(pred.shape) == 4, "inputs shape must be NCHW"
        if len(gt.shape) != 4:
            gt = gt.unsqueeze(1).float()
        else:
            gt = gt.float()

        positive = gt.byte()
        negative = (1 - gt).byte()
        positive_count = int(positive.float().sum())
        negative_count = min(int(negative.float().sum()), int(positive_count * self.negative_ratio))
        loss = F.binary_cross_entropy_with_logits(pred, gt, reduction='none')
        positive_loss = loss * positive.float()
        negative_loss = loss * negative.float()
        # negative_loss, _ = torch.topk(negative_loss.view(-1).contiguous(), negative_count)
        negative_loss, _ = negative_loss.view(-1).topk(negative_count)

        balance_loss = (positive_loss.sum() + negative_loss.sum()) / (positive_count + negative_count + self.eps)

        if return_origin:
            return balance_loss, loss
        return balance_loss


# -------------------- BCELoss -----------------------
class BCELoss(nn.Module):
    """binary bceloss with sigmoid"""
    def __init__(self):
        super(BCELoss, self).__init__()

    def forward(self, inputs, targets, weights=None, mask=False):
        assert len(inputs.shape) == 4, "inputs shape must be NCHW"
        if len(targets.shape) != 4:
            targets = targets.unsqueeze(1).float()
        else:
            targets = targets.float()
        if mask:
            inputs  = inputs * targets
        losses = F.binary_cross_entropy_with_logits(inputs, targets, weights)
        return losses


# ----------------- DICE Loss--------------------
class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()
 
    def forward(self, logits, targets, mask=False):
        num = targets.size(0)
        smooth = 1.

        probs = torch.sigmoid(logits)
        m1 = probs.view(num, -1)
        m2 = targets.view(num, -1)
        intersection = (m1 * m2)
 
        score = 2. * (intersection.sum(1) + smooth) / (m1.sum(1) + m2.sum(1) + smooth)
        score = 1 - score.sum() / num
        return score

# ------------------ IOU Loss ------------------------
class IOULoss(nn.Module):
    def __init__(self):
        super(IOULoss, self).__init__()

    def forward(self, logits, targets, mask=None):
        num = targets.size(0)
        smooth = 1.

        probs = torch.sigmoid(logits)
        m1 = probs.view(num, -1)
        m2 = targets.view(num, -1)
        intersection = (m1 * m2)
 
        score = (intersection.sum(1) + smooth) / (m1.sum(1) + m2.sum(1)  - intersection.sum(1) + smooth)
        score = 1 - score.sum() / num
        return score 


# ----------------- IOU+DICE+BCE Loss--------------------
class IouWithDiceWithBCELoss(nn.Module):
    def __init__(self, weights, mining=False):
        super(IouWithDiceWithBCELoss, self).__init__()
        self.dice_loss = DiceLoss()
        if mining:
            self.bce_loss = BalanceCrossEntropyLoss() 
        else:
            self.bce_loss = BCELoss()
        
        self.iou_loss = IOULoss()
        self.weights = weights

    def forward(self, preds, targets):
        iouloss = self.iou_loss(preds, targets)
        bceloss = self.bce_loss(preds, targets)
        diceloss = self.dice_loss(preds, targets)
        return self.weights['bce'] * bceloss + self.weights['dice']*diceloss + self.weights['iou']*iouloss


# ----------------- IOU+BCE Loss --------------------
class IOUWithBCELoss(nn.Module):
    def __init__(self, weights, mining=False) -> None:
        super(IOUWithBCELoss, self).__init__()
        self.iou_loss = IOULoss()
        if mining:
            self.bce_loss = BalanceCrossEntropyLoss() 
        else:
            self.bce_loss = BCELoss()

        self.weights = weights

    def forward(self, preds, targets):
        iouloss = self.iou_loss(preds, targets)
        bceloss = self.bce_loss(preds, targets)
        return self.weights['bce']*bceloss + self.weights['iou']*iouloss


# ----------------- DICE+BCE Loss--------------------
class DiceWithBCELoss(nn.Module):
    def __init__(self, weights, mining=False):
        super(DiceWithBCELoss, self).__init__()
        self.dice_loss = DiceLoss()
        if mining:
            self.bce_loss = BalanceCrossEntropyLoss() 
        else:
            self.bce_loss = BCELoss()
        self.weights = weights

    def forward(self, preds, targets):
        bceloss = self.bce_loss(preds, targets)
        diceloss = self.dice_loss(preds, targets)
        return self.weights['bce'] * bceloss + self.weights['dice']*diceloss


# ----------------- Charbonnier L1 loss -----------
class L1_Charbonnier(nn.Module):
    """L1 Charbonnierloss."""

    def __init__(self, eps=1e-3):
        super(L1_Charbonnier, self).__init__()
        '''MSRNet uses 1e-3 as default'''
        self.eps = eps

    def forward(self, X, Y):
        if len(Y.shape) != 4:
            Y = Y.unsqueeze(1).float()
        else:
            Y = Y.float()

        diff = torch.add(X, -Y)
        error = torch.sqrt(diff * diff + self.eps * self.eps)
        loss = torch.mean(error)
        return loss


# --------------- bce+dice+L1 ----------------
class Dice_Bce_L1(nn.Module):
    def __init__(self, weights):
        super(Dice_Bce_L1, self).__init__()
        self.dice_bce = DiceWithBCELoss(weights)
        self.l1 = L1_Charbonnier()
        
    def forward(self, preds, targets):
        return self.dice_bce(preds, targets) + self.l1(F.sigmoid(preds), targets)


# ---------------- ce + dice + l1 ---------------
class Dice_Ce_L1(nn.Module):
    """defautl the ce with dice is 1: 3
    """
    def __init__(self):
        super(Dice_Ce_L1, self).__init__()
        self.dice = DiceLoss()
        self.ce = CELoss()
        self.l1 = nn.SmoothL1Loss()

    def forward(self, preds, targets):
        ce_loss = self.ce(preds, targets)

        # targets for bce & dice & l1
        targets = targets.float()

        # use the fround image
        dice_loss = self.dice(preds[:,1,:,:], targets)
        l1_loss = self.l1(preds[:,1,:,:], targets)

        return ce_loss + 3*dice_loss + l1_loss

            
# ---------------- mutil classes bce & dice loss --------------------
"""Because the loss is used for 0-1 labels, So we need use loss function 
on each labels. Beacuse each targets have include the bg, so we only need 
classes-1 channels for each fg labels. Targets is the NX(CLASS-1)XHXW, outputs
shape is same with targets.
"""
class MutilClassBCELoss(nn.Module):
    """Mutil class bce losss with sigmoid"""
    def __init__(self, num_classes):
        super(MutilClassBCELoss, self).__init__()
        self.num_classes = num_classes
        self.bceloss = BCELoss()

    def forward(self, inputs, targets, weights=None):
        C = inputs.shape[1]
        total_loss = 0.0
        for i in range(C):
            losses = self.bceloss(inputs[:,i], targets[:,i])
            if weights is not None:
                losses *= weights[i]
            total_loss += losses 

        return total_loss


class MutilClassDiceLoss(nn.Module):
    """Mutil class dice loss, cal the loss with each classes, 
        targets need translate to one-hot for loss
    """
    def __init__(self, smooth=1.0, num_classes=21):
        super(MutilClassDiceLoss, self).__init__()
        self.smooth = smooth 
        self.num_classes = num_classes
        self.dice = DiceLoss(self.smooth)

    def forward(self, inputs, targets, weights=None):
        C = targets.shape[1]
        total_loss = 0.0

        for i in range(C):
            losses = self.dice(inputs[:,i], targets[:,i])
            if weights is not None:
                losses *= weights[i]
            total_loss += losses 

        return total_loss
        
# translate to onehot
def toOnehot(x, num_classes):
    if len(x.shape) == 3:
        max_num = torch.max(x)
        if max_num == num_classes - 1:
            new_shape = (x.shape[0], num_classes-1, x.shape[1], x.shape[2])
            x1 = torch.zeros(new_shape).long()
            for i in range(1, num_classes):
                x1[:,i-1][torch.where(x==i)] = 1
            return x1
        else:
            raise ValueError("num classes not match the targets, need check the targets or classes number")
    else:
        return x 


if __name__ == "__main__":
    # loss = DiceLoss()
    inputs = torch.sigmoid(torch.randn(4,2,32,32).float())
    targets = torch.empty(4,32,32).random_(1).long()
    # losses = loss(inputs, targets)
    # print(losses)

    losses = Dice_Ce_L1()

    print(losses(inputs, targets))
    

    losses = BCELoss()
    inputs = torch.zeros(1,1,4,4)
    inputs[:,:,0,0] = torch.nan 

    targets = torch.ones_like(inputs)
    # targets[:,:,0,0] = torch.nan
    outs = losses(inputs, targets)
    print(outs)