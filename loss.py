import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    def __init__(self, pos_weight):
        super().__init__()
        self.pos_weight = pos_weight

    def forward(self, input, target):
        bce = F.binary_cross_entropy_with_logits(input, target, reduction='none', pos_weight=self.pos_weight)
        probs = torch.sigmoid(input)
        focal = torch.where(target.bool(), 1.0 - probs, probs) ** 2
        return (focal * bce).mean()


class YOLOLoss(nn.Module):
    def __init__(self, spatial_coef=5.0, positive_coef=2.0):
        super().__init__()
        self.spatial_coef = spatial_coef
        self.class_loss = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(positive_coef))

    def forward(self, input, target):
        obj_mask = target[:, :1]
        spatial_loss = F.mse_loss(input[:, 1:] * obj_mask, target[:, 1:] * obj_mask)
        class_loss = self.class_loss(input[:, :1], obj_mask)
        self.last_losses = (spatial_loss.item(), class_loss.item())
        return self.spatial_coef * spatial_loss + class_loss


class YOLOFocalLoss(YOLOLoss):
    def __init__(self, spatial_coef=5.0, positive_coef=2.0):
        super().__init__()
        self.class_loss = FocalLoss(pos_weight=torch.tensor(positive_coef))

