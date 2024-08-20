import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.nn.init as init
from ..builder import DISTILL_LOSSES


@DISTILL_LOSSES.register_module()
class BalanceKLDivergence(nn.Module):

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
                temp_stu=1.0,
                temp_tea=1.0,
                weight=1.0,
                reverse=0.0,
                ):
        super(BalanceKLDivergence, self).__init__()
        self.temp_f = temp_stu
        self.temp_r = temp_tea
        self.loss_weight = weight
        self.reverse = reverse
        self.name = name
    
        if student_channels != teacher_channels:
            self.align = nn.Sequential(
                nn.BatchNorm2d(num_features=student_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(student_channels, teacher_channels, kernel_size=1, stride=1, padding=0),
            )
        else:
            self.align = None
        if "bev" in name:
            self.adapter_layer_s = nn.Sequential(
                nn.AvgPool2d(kernel_size=3, stride=1, padding=0),
                nn.Conv2d(teacher_channels, teacher_channels, kernel_size=3, padding=0),
            )
            self.adapter_layer_t = nn.Sequential(
                nn.AvgPool2d(kernel_size=3, stride=1, padding=0),
                nn.Conv2d(teacher_channels, teacher_channels, kernel_size=3, padding=0),
            )
            self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.fill_(1.0)
                # nn.init.kaiming_normal_(m.weight)
                
    def forward(self, preds_S, preds_T, gt_bboxes, image_metas):
        """Forward function."""
        # assert preds_S.shape[-2:] == preds_T.shape[-2:],'the output dim of teacher and student differ'
        preds_T = preds_T.to(dtype=torch.float32)
        preds_S = preds_S.to(dtype=torch.float32)
        if preds_S.shape[-2:] != preds_T.shape[-2:]:
            preds_T = F.interpolate(preds_T, size=(preds_S.shape[-2], preds_S.shape[-1]), mode='bilinear', align_corners=True)
            # preds_S = self.generation_up(preds_S.float()).half()
            
        n,c,h,w = preds_S.shape
        if self.align is not None:
            preds_S = self.align(preds_S)
        if 'bev' in self.name:
            preds_S = self.adapter_layer_s(preds_S)
            preds_T = self.adapter_layer_t(preds_T)
            
        self.temperature_stu = self.temp_f
        self.temperature_tea = self.temp_f #if self.temp_tea > 0 else self.temperature_stu
        norm_s_log = F.log_softmax(preds_S.reshape((n, c, -1)) / self.temperature_stu, dim=-1)
        norm_t = F.softmax(preds_T.reshape((n, c, -1)) / self.temperature_tea, dim=-1)
        if type(image_metas) == float:
            # loss = self.reverse * (1 - image_metas) * F.kl_div(norm_s_log, norm_t , size_average=False) * (self.temperature_stu**2) / (n * c)
            loss = self.reverse * image_metas * F.kl_div(norm_s_log, norm_t , size_average=False) * (self.temperature_stu**2) / (n * c)
        else:
            loss = F.kl_div(norm_s_log, norm_t , size_average=False) * (self.temperature_stu**2) / (n * c)
        if self.reverse > 0:
            self.temperature_stu = self.temp_r #10 - self.temperature_stu
            self.temperature_tea = self.temp_r #self.temp_tea if self.temp_tea > 0 else self.temperature_stu
            assert self.temperature_stu > 0
            norm_s = F.softmax(preds_S.reshape((n, c, -1)) / self.temperature_stu, dim=-1)
            norm_t_log = F.log_softmax(preds_T.reshape((n, c, -1)) / self.temperature_tea, dim=-1)
            if type(image_metas) == float:
                # loss += self.reverse * image_metas * F.kl_div(norm_t_log, norm_s , size_average=False) * (self.temperature_stu**2) / (n * c)
                loss += self.reverse * (1.0 - image_metas) * F.kl_div(norm_t_log, norm_s , size_average=False) * (self.temperature_stu**2) / (n * c)
            else:
                loss += self.reverse * F.kl_div(norm_t_log, norm_s , size_average=False) * (self.temperature_stu**2) / (n * c)
        return loss * self.loss_weight
