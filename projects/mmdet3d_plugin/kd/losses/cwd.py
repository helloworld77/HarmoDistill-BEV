import torch.nn as nn
import torch.nn.functional as F
import torch
from ..builder import DISTILL_LOSSES


@DISTILL_LOSSES.register_module()
class ChannelWiseDivergence(nn.Module):

    """PyTorch version of `Channel-wise Distillation for Semantic Segmentation
     <https://arxiv.org/abs/2011.13256>`_.
   
    Args:
        student_channels(int): Number of channels in the student's feature map.
        teacher_channels(int): Number of channels in the teacher's feature map.
        name(str): 
        tau (float, optional): Temperature coefficient. Defaults to 1.0.
        weight (float, optional): Weight of loss.Defaults to 1.0.
        
    """
    def __init__(self,
                 student_channels,
                 teacher_channels,
                 name,
                 tau=4.0,
                 weight=3.0,
                 ):
        super(ChannelWiseDivergence, self).__init__()
        self.tau = tau
        self.loss_weight = weight
    
        if student_channels != teacher_channels:
            self.align = nn.Sequential(
                nn.BatchNorm2d(num_features=student_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(student_channels, teacher_channels, kernel_size=1, stride=1, padding=0),
            )
        else:
            self.align = None

    def forward(self, preds_S, preds_T, gt_bboxes, image_metas):
        """Forward function."""
        if preds_S.shape[-2:] != preds_T.shape[-2:]:
            preds_T = F.interpolate(preds_T, size=(preds_S.shape[-2], preds_S.shape[-1]), mode='bilinear', align_corners=True)
        N,C,W,H = preds_S.shape
        if self.align is not None:
            preds_S = self.align(preds_S)
        softmax_t = torch.nn.functional.softmax(preds_T.view(-1, W * H) / self.tau, dim=1)
        logsoftmax_s = torch.nn.functional.log_softmax(preds_S.view(-1, W * H) / self.tau, dim=1)
        loss = torch.nn.functional.kl_div(logsoftmax_s, softmax_t , size_average=False) * (self.tau**2 / (C * N))
        loss = self.loss_weight * loss
        return loss