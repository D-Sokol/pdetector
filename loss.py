import torch
import torch.nn as nn
import torch.nn.functional as F


class YOLOLoss(nn.Module):
    def __init__(self, spatial_coef=5.0, positive_coef=2.0):
        super().__init__()
        self.spatial_coef = spatial_coef
        self.class_loss = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(positive_coef))

    def forward(self, input, target):
        obj_mask = target[:, 0]
        spatial_loss = F.mse_loss(input[:, 1:] * obj_mask, target[:, 1:] * obj_mask)
        class_loss = self.class_loss(input[:, 0], obj_mask)
        return self.spatial_coef * spatial_loss + class_loss

