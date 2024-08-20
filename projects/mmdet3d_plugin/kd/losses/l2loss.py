import torch.nn as nn
import torch.nn.functional as F
import torch
from ..builder import DISTILL_LOSSES


@DISTILL_LOSSES.register_module()
class L2Loss(nn.Module):
    def __init__(self,
                 student_channels,
                 teacher_channels,
                 weight=1.0,
                 use_adapter=False,
                 **kwargs,
                 ):
        super(L2Loss, self).__init__()
        self.loss_weight = weight
    
        if student_channels != teacher_channels:
            self.align = nn.Conv2d(student_channels, teacher_channels, kernel_size=1, stride=1, padding=0)
        else:
            self.align = None
        if use_adapter:
            self.generation = nn.Sequential(
                nn.Conv2d(teacher_channels, teacher_channels, kernel_size=3, padding=1),
                nn.ReLU(inplace=True), 
                nn.Conv2d(teacher_channels, teacher_channels, kernel_size=3, padding=1))
        else:
            self.generation = None

    def forward(self, preds_S, preds_T, gt_bboxes, image_metas):
        """Forward function."""
        assert preds_S.shape[-2:] == preds_T.shape[-2:],'the output dim of teacher and student differ'
        preds_T = preds_T.to(dtype=torch.float32)
        preds_S = preds_S.to(dtype=torch.float32)
        N, C, H, W = preds_S.shape
        if self.align is not None:
            preds_S = self.align(preds_S)
        if self.generation is not None:
            preds_T = self.generation(preds_T)
        loss_mse = nn.MSELoss(reduction='mean')(preds_S, preds_T)/N
        return loss_mse * self.loss_weight



