from torch import nn
import torch


class DFLoss(nn.Module):
    """Criterion class for computing DFL losses during training."""

    def __init__(self):
        """Initialize the DFL module."""
        super().__init__()

    def __call__(self, pred_dist, target):
        """
        Return sum of left and right DFL losses.

        Distribution Focal Loss (DFL) proposed in Generalized Focal Loss
        https://ieeexplore.ieee.org/document/9792391
        """
        target = target.clamp_(0, self.reg_max - 1 - 0.01)
        tl = target.long()  # target left
        tr = tl + 1  # target right
        wl = tr - target  # weight left
        wr = 1 - wl  # weight right
        return (
            F.cross_entropy(pred_dist, tl.view(-1), reduction="none").view(tl.shape) * wl
            + F.cross_entropy(pred_dist, tr.view(-1), reduction="none").view(tl.shape) * wr
        ).mean(-1, keepdim=True)

class BboxLoss(nn.Module):
    """Criterion class for computing training losses during training."""

    def __init__(self):
        """Initialize the BboxLoss module with regularization maximum and DFL settings."""
        super().__init__()
        self.dfl_loss = DFLoss()

    def forward(self, preds, labels):
        """IoU loss."""
        weight = 
        iou = 
        loss_iou = 

        # DFL loss
        loss_dfl = 

        return loss_iou, loss_dfl
    
class DetectionLoss:
    """Criterion class for computing training losses."""

    def __init__(self, h, device):  # model must be de-paralleled
        """Initializes v8DetectionLoss with the model, defining model-related properties and BCE loss function."""
        self.bce = nn.BCEWithLogitsLoss(reduction="none")
        self.hyp = h
        self.device = device
        self.bbox_loss = BboxLoss().to(device)


    def __call__(self, preds, labels):
        """Calculate the sum of the loss for box, cls and dfl multiplied by batch size."""
        loss = torch.zeros(3, device=self.device)  # box, cls, dfl
        # Cls loss
        # loss[1] = self.varifocal_loss(pred_scores, target_scores, target_labels) / target_scores_sum  # VFL way
        loss[1] = self.bce()

 
        loss[0], loss[2] = self.bbox_loss(
            )

        loss[0] *= self.hyp.box  # box gain
        loss[1] *= self.hyp.cls  # cls gain
        loss[2] *= self.hyp.dfl  # dfl gain

        return # loss(box, cls, dfl)
