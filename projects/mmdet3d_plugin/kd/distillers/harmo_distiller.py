import torch.nn as nn
import torch.nn.functional as F
import torch
from mmdet.models.detectors.base import BaseDetector
from mmdet.models import build_detector
from mmcv.runner import  load_checkpoint
from ..builder import DISTILLER, build_distill_loss
import os
import cv2
import copy
import numpy as np
from nuscenes.utils.data_classes import Box as NuScenesBox
from pyquaternion import Quaternion
#from tools.analysis_tools.visual import *

def get_global_pos(points, pc_range):
    pc_range = torch.tensor(pc_range).to(points.device)
    points = points * (pc_range[3:6] - pc_range[0:3]) + pc_range[0:3]
    return points
            
def get_attnmap_sum(target_point, target_feature, pc_range):
    bs, num_query, num_ref, num_dim = target_point.shape
    num_points = num_query * num_ref
    feature_dim = target_feature.shape[2]
    attnmap_size = 128
    attnmap = torch.zeros((bs, attnmap_size, attnmap_size, feature_dim)).to(target_feature.device)
    x_coords = target_point[:, :, :, 0] / (pc_range[3] - pc_range[0]) * (attnmap_size - 1) + attnmap_size / 2
    y_coords = target_point[:, :, :, 1] / (pc_range[4] - pc_range[1]) * (attnmap_size - 1) + attnmap_size / 2
    x_coords = torch.clamp(x_coords, 0, attnmap_size - 1)
    y_coords = torch.clamp(y_coords, 0, attnmap_size - 1)
    x_coords = x_coords.unsqueeze(3)
    y_coords = y_coords.unsqueeze(3)
    # coords: [4, 428x13, 2]
    coords = torch.cat([x_coords, y_coords], dim=3).reshape(bs, -1, 2).long()
    # target_feature: [4, 428x13, 256]
    target_feature = target_feature.unsqueeze(2).repeat(1, 1, num_ref, 1).reshape(bs, -1, feature_dim)
    batch_indices = torch.arange(bs).view(bs, 1, 1).expand(-1, num_points, -1).to(coords.device)
    full_indices = torch.cat((batch_indices, coords), dim=2).long()
    full_indices = full_indices.permute(2, 0, 1).contiguous().view(3, -1)
    target_feature = target_feature.reshape(-1, feature_dim)
    attnmap.index_put_(tuple(full_indices), target_feature, accumulate=True)
    return attnmap.permute(0, 3, 1, 2)

def get_attnmap(target_point, target_feature, pc_range):
    bs, num_query, num_ref, num_dim = target_point.shape
    num_points = target_feature.shape[1]
    feature_dim = target_feature.shape[2]
    attnmap_size = 128
    attnmap = torch.zeros((bs, attnmap_size, attnmap_size, feature_dim)).to(target_feature.device)
    x_coords = target_point[:, :, :, 0] / (pc_range[3] - pc_range[0]) * (attnmap_size - 1) + attnmap_size / 2
    y_coords = target_point[:, :, :, 1] / (pc_range[4] - pc_range[1]) * (attnmap_size - 1) + attnmap_size / 2
    x_coords = torch.clamp(x_coords, 0, attnmap_size - 1)
    y_coords = torch.clamp(y_coords, 0, attnmap_size - 1)
    x_coords = x_coords.unsqueeze(3)
    y_coords = y_coords.unsqueeze(3)
    # coords: [4, 428x13, 2]
    coords = torch.cat([x_coords, y_coords], dim=3).reshape(bs, -1, 2).long()
    # target_feature: [4, 428x13, 256]
    target_feature = target_feature.unsqueeze(2).repeat(1, 1, num_ref, 1).reshape(bs, -1, feature_dim)
    # attnmap: [4, 256, 128, 128]; coords: [4, 428x13, 2]
    attnmap[np.arange(attnmap.shape[0])[:, None], coords[..., 0], coords[..., 1], :] = target_feature            
    return attnmap.permute(0, 3, 1, 2)

@DISTILLER.register_module()
class HarmoDistiller(BaseDetector):
    """Base distiller for detectors.
    It typically consists of teacher_model and student_model.
    """
    def __init__(self,
                 teacher_cfg,
                 student_cfg,
                 distill_on_FPN = True,
                 distill_on_TFPN = False,
                 distill_on_ATTN = False,
                 distill_on_DN = False,
                 distill_on_BEVATTN = False,
                 distill_on_spatial_w = 0,
                 distill_on_outputs_w = 0,
                 distill_on_diff_dim = False,
                 update_by_iter = False,
                 distiller_fpn = None,
                 distiller_tfpn = None,
                 distiller_attn = None,
                 distiller_bevattn = None,
                 teacher_pretrained=None,
                 branch_distill_queries_cfg = None,
                 duplicate_student_head = False,
                 ):

        super(HarmoDistiller, self).__init__()
        self.teacher_cfg = teacher_cfg
        self.student_cfg = student_cfg
        self.teacher = build_detector(teacher_cfg.model,
                                        train_cfg=teacher_cfg.get('train_cfg'),
                                        test_cfg=teacher_cfg.get('test_cfg'))
        self.channel_tch = teacher_cfg.get('_dim_')
        self.init_weights_teacher(teacher_pretrained)
        for (name, param) in self.teacher.named_parameters():
            param.requires_grad = False
        self.teacher.eval()
        self.student = build_detector(student_cfg.model,
                                        train_cfg=student_cfg.get('train_cfg'),
                                        test_cfg=student_cfg.get('test_cfg'))

        if duplicate_student_head:
            print('duplicate_student_head')
            self.init_weights_student_head(teacher_pretrained)
        
        self.channel_stu = student_cfg.get('_dim_')
        self.flag_distill_on_FPN = distill_on_FPN if distiller_fpn is not None else False
        self.flag_distill_on_TFPN = distill_on_TFPN if distiller_tfpn is not None else False
        self.flag_distill_on_ATTN = distill_on_ATTN if distiller_attn is not None else False
        self.flag_distill_on_BEVATTN = distill_on_BEVATTN if distiller_bevattn is not None else False
        self.flag_distill_on_DN = distill_on_DN
        self.distill_on_spatial_w = distill_on_spatial_w
        self.distill_on_outputs_w = distill_on_outputs_w
        self.update_by_iter = update_by_iter
        self.distill_on_diff_dim = distill_on_diff_dim
        if self.update_by_iter:
            self.max_loss = -1
            self.max_feat_wh = -1

        def regitster_hooks(student_module, teacher_module):
            def hook_teacher_forward(module, input, output):
                #self.register_buffer(teacher_module, output)
                setattr(self, teacher_module, output)
            def hook_student_forward(module, input, output):
                #self.register_buffer(student_module, output)
                setattr(self, student_module, output)
            return hook_teacher_forward, hook_student_forward
        
        modules_stu = dict(self.student.named_modules())
        modules_tch = dict(self.teacher.named_modules())

        # loss items
        self.distill_losses = nn.ModuleDict()

        self.distiller_fpn = distiller_fpn
        self.distiller_tfpn = distiller_tfpn
        self.distiller_attn = distiller_attn
        self.distiller_bevattn = distiller_bevattn

        if self.flag_distill_on_FPN:
            for item_loc in distiller_fpn:
                student_module = 'student_' + item_loc.student_module.replace('.','_')
                teacher_module = 'teacher_' + item_loc.teacher_module.replace('.','_')
                self.register_buffer(student_module, None)
                self.register_buffer(teacher_module, None)
                hook_teacher_forward, hook_student_forward = regitster_hooks(student_module, teacher_module)
                modules_tch[item_loc.teacher_module].register_forward_hook(hook_teacher_forward)
                modules_stu[item_loc.student_module].register_forward_hook(hook_student_forward)
                for item_loss in item_loc.methods:
                    loss_name = item_loss.name
                    self.distill_losses[loss_name] = build_distill_loss(item_loss)
        
        if self.flag_distill_on_BEVATTN:
            for item_loc in distiller_bevattn:
                student_module = 'student_' + item_loc.student_module.replace('.','_')
                teacher_module = 'teacher_' + item_loc.teacher_module.replace('.','_')
                self.register_buffer(student_module, None)
                self.register_buffer(teacher_module, None)
                hook_teacher_forward, hook_student_forward = regitster_hooks(student_module, teacher_module)
                modules_tch[item_loc.teacher_module].register_forward_hook(hook_teacher_forward)
                modules_stu[item_loc.student_module].register_forward_hook(hook_student_forward)
                for item_loss in item_loc.methods:
                    loss_name = item_loss.name
                    self.distill_losses[loss_name] = build_distill_loss(item_loss)

        # BEV features alignment and distillation
        if self.flag_distill_on_TFPN:
            for item_loc in self.distiller_tfpn:
                for item_loss in item_loc.methods:
                    loss_name = item_loss.name
                    self.distill_losses[loss_name] = build_distill_loss(item_loss)
                    
        if self.flag_distill_on_ATTN:
            for item_loc in self.distiller_attn:
                for item_loss in item_loc.methods:
                    loss_name = item_loss.name
                    self.distill_losses[loss_name] = build_distill_loss(item_loss)       


    def base_parameters(self):
        return nn.ModuleList([self.student, self.distill_losses])

    def discriminator_parameters(self):
        return self.discriminator

    @property
    def with_neck(self):
        """bool: whether the detector has a neck"""
        return hasattr(self.student, 'neck') and self.student.neck is not None

    # TODO: these properties need to be carefully handled
    # for both single stage & two stage detectors
    @property
    def with_shared_head(self):
        """bool: whether the detector has a shared head in the RoI Head"""
        return hasattr(self.student, 'roi_head') and self.student.roi_head.with_shared_head

    @property
    def with_bbox(self):
        """bool: whether the detector has a bbox head"""
        return ((hasattr(self.student, 'roi_head') and self.student.roi_head.with_bbox)
                or (hasattr(self.student, 'bbox_head') and self.student.bbox_head is not None))

    @property
    def with_mask(self):
        """bool: whether the detector has a mask head"""
        return ((hasattr(self.student, 'roi_head') and self.student.roi_head.with_mask)
                or (hasattr(self.student, 'mask_head') and self.student.mask_head is not None))

    def init_weights_teacher(self, path=None):
        """Load the pretrained model in teacher detector.
        Args:
            pretrained (str, optional): Path to pre-trained weights.
                Defaults to None.
        """
        checkpoint = load_checkpoint(self.teacher, path, map_location='cpu')

    def init_weights_student_head(self, path=None):
        """Load the pretrained model in teacher detector.
        Args:
            pretrained (str, optional): Path to pre-trained weights.
                Defaults to None.
        """
        checkpoint = load_checkpoint(self.student.pts_bbox_head, path, map_location='cpu', \
            revise_keys=[(r'^module\.', ''), (r'^pts_bbox_head\.', '')])

    def forward(self, return_loss=True, **kwargs):
        """Calls either forward_train or forward_test depending on whether
        return_loss=True.
        Note this setting will change the expected inputs. When
        `return_loss=True`, img and img_metas are single-nested (i.e.
        torch.Tensor and list[dict]), and when `resturn_loss=False`, img and
        img_metas should be double nested (i.e.  list[torch.Tensor],
        list[list[dict]]), with the outer list indicating test time
        augmentations.
        """

        if return_loss:
            return self.forward_train(**kwargs)
        else:
            return self.forward_test(**kwargs)

    def _make_deconv_layers(self, inplanes, outplanes, num_deconv_layers=1):
        def _get_deconv_cfg(deconv_kernel):
            if deconv_kernel == 4:
                padding = 1
                output_padding = 0
            elif deconv_kernel == 3:
                padding = 1
                output_padding = 1
            elif deconv_kernel == 2:
                padding = 0
                output_padding = 0

            return deconv_kernel, padding, output_padding
        
        layers = []
        BN_MOMENTUM = 0.1

        for idx in range(num_deconv_layers):
            kernel, padding, output_padding = _get_deconv_cfg(4)
            layers.append(
                nn.ConvTranspose2d(
                    in_channels = inplanes,
                    out_channels = outplanes,
                    kernel_size=kernel,
                    stride=2,
                    padding=padding,
                    output_padding=output_padding,
                    bias = False))
            layers.append(nn.BatchNorm2d(outplanes, momentum=BN_MOMENTUM))
            layers.append(nn.ReLU(inplace=True))

        layers.append(
                nn.Conv2d(
                    in_channels=outplanes,
                    out_channels=outplanes,
                    kernel_size=3,
                    stride=1,
                    padding=1))

        return nn.Sequential(*layers) 

    def forward_train(self, **kwargs):
        """
        Args:
            img (Tensor): Input images of shape (N, C, H, W).
                Typically these should be mean centered and std scaled.
            img_metas (list[dict]): A List of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                :class:`mmdet.datasets.pipelines.Collect`.
        Returns:
            dict[str, Tensor]: A dictionary of loss components(student's losses and distiller's losses).
        """
        bs, seq_len, N_cam, c, h, w = kwargs['img'].shape
        kwargs_T = copy.deepcopy(kwargs)
        kwargs_S = copy.deepcopy(kwargs)
        _scale = self.teacher_cfg['ida_aug_conf']['final_dim'][1] // self.student_cfg['ida_aug_conf']['final_dim'][1]
        # reshape input image shape for teacher model
        if _scale > 1:
            kwargs_T['img'] = F.interpolate(kwargs['img'].view(bs * seq_len * N_cam, c, h, w), scale_factor=_scale, \
                    mode='bilinear', align_corners=False).view(bs, seq_len, N_cam, c, h*_scale, w*_scale)
            scale_factor = np.eye(4)
            scale_factor[0, 0] *= _scale
            scale_factor[1, 1] *= _scale
            for i_batch in range(len(kwargs_T['img_metas'])):
                for i_seq in range(len(kwargs_T['img_metas'][0])):
                    raw_img_metas = kwargs['img_metas'][i_batch][i_seq]
                    N_cam = len(raw_img_metas['filename'])
                    kwargs_T['img_metas'][i_batch][i_seq]['img_shape'] = [(img_shapes[0]*_scale, img_shapes[1]*_scale) for img_shapes in raw_img_metas['img_shape']]
                    kwargs_T['img_metas'][i_batch][i_seq]['pad_shape'] = [(img_shapes[0]*_scale, img_shapes[1]*_scale) for img_shapes in raw_img_metas['pad_shape']]

        # align student feature with teacher feature
        with torch.no_grad():
            self.teacher.eval()
            if self.flag_distill_on_FPN or self.flag_distill_on_TFPN or self.flag_distill_on_ATTN or self.flag_distill_on_DN or self.flag_distill_on_BEVATTN:
                data_T = {}
                kwargs_T['img_feats'] = self.teacher.extract_img_feat(kwargs_T['img'], 1)
                for key in ['gt_bboxes_3d', 'gt_labels_3d', 'gt_bboxes', 'gt_labels', 'centers2d', 'depths', 'img_metas']:
                    kwargs_T[key] = list(zip(*kwargs_T[key]))
                # teacher_outs_roi = self.teacher.forward_roi_head(**kwargs_T)
                for key in kwargs_T.keys() - ['gt_bboxes_3d', 'gt_labels_3d', 'gt_bboxes', 'gt_labels', 'centers2d', 'depths', 'img_metas', 'img_feats']:
                    data_T[key] = kwargs_T[key][:, 0]
                data_T['img_feats'] = kwargs_T['img_feats']
                if self.flag_distill_on_ATTN or self.flag_distill_on_DN:
                    teacher_outs = self.teacher.pts_bbox_head.forward_teacher_test(kwargs_T['img_metas'][0], **data_T)
                else:
                    teacher_outs = self.teacher.pts_bbox_head(kwargs_T['img_metas'][0], **data_T)

        if self.flag_distill_on_DN:            
            kwargs_S['teacher_outs'] = teacher_outs

        for key in ['gt_bboxes_3d', 'gt_labels_3d', 'gt_bboxes', 'gt_labels', 'centers2d', 'depths', 'img_metas']:
            kwargs_S[key] = list(zip(*kwargs_S[key]))
        student_loss, student_outs_roi, student_outs = self.student.forward_distill(**kwargs_S)
        if self.update_by_iter:
            student_loss_sum = sum(student_loss.values())
            self.max_loss = max(self.max_loss, student_loss_sum.item())
            self.alpha_adapter_loss = student_loss_sum.item() / self.max_loss 
            if self.max_feat_wh < 0:
                self.max_feat_wh = student_outs['feat_shape'][0].sum()
        
        buffer_dict = dict(self.named_buffers())

        # distill on FPN features
        if self.flag_distill_on_FPN:
            for idx, item_loc in enumerate(self.distiller_fpn):
                student_module = 'student_' + item_loc.student_module.replace('.', '_')
                teacher_module = 'teacher_' + item_loc.teacher_module.replace('.', '_')
                student_feat = buffer_dict[student_module]
                teacher_feat = buffer_dict[teacher_module]
                ## 24 256 h, w -> 4, 256, h*3, w*2
                # student_feat = student_feat.reshape(bs, N_cam, -1, student_feat.shape[2], student_feat.shape[3]).permute(0, 2, 1, 3, 4).reshape(bs, -1, student_feat.shape[2]*3, student_feat.shape[3]*2)
                # teacher_feat = teacher_feat.reshape(bs, N_cam, -1, teacher_feat.shape[2], teacher_feat.shape[3]).permute(0, 2, 1, 3, 4).reshape(bs, -1, teacher_feat.shape[2]*3, teacher_feat.shape[3]*2)
                for item_loss in item_loc.methods:
                    loss_name = item_loss.name
                    ## distill on 2D feature: 4, 6, H, W, C -> 24, 256, 32, 88 (distill on N, C, H, W)
                    if self.update_by_iter:
                        # alpha_adapter = (self.alpha_adapter_loss + (student_feat.shape[2] + student_feat.shape[3]) / self.max_feat_wh) / 2
                        student_loss[loss_name] = self.distill_losses[loss_name](student_feat, teacher_feat, kwargs['gt_bboxes'], self.iter )
                    else:
                        student_loss[loss_name] = self.distill_losses[loss_name](student_feat, teacher_feat, kwargs['gt_bboxes'], kwargs['img_metas'])
                    if self.distill_on_diff_dim:
                        ## distill on N, H, W, C
                        student_loss[loss_name] += self.distill_losses[loss_name](student_feat.permute(0,2,3,1), teacher_feat.permute(0,2,3,1), kwargs['gt_bboxes'], kwargs['img_metas'])
            if self.distill_on_outputs_w > 0:
                ## distill on cls and box outputs ## torch.Size([6, 4, 428, 10]) * torch.Size([6, 4, 428, 10]) -> torch.Size([24, 428, 428])
                s_cls = student_outs["all_cls_scores"]
                t_cls = teacher_outs["all_cls_scores"]
                s_bbox = student_outs["all_bbox_preds"]
                t_bbox = teacher_outs["all_bbox_preds"]
                # torch.einsum('bcn,bchw->bnhw', [query_feat, intpu_feat.float()])
                s_relation_cls_bbox = torch.einsum('ijab,ijcb->ijac', [s_cls, s_bbox])
                t_relation_cls_bbox = torch.einsum('ijab,ijcb->ijac', [t_cls, t_bbox])
                student_loss[loss_name+"_output_relation"] += self.distill_on_outputs_w * self.distill_losses[loss_name](s_relation_cls_bbox, t_relation_cls_bbox, kwargs['gt_bboxes'], kwargs['img_metas'])
            
        if self.flag_distill_on_TFPN or self.flag_distill_on_ATTN:
            teacher_feat_index = 0
            student_feat_index = 0
            for idx, teacher_feat_shape in enumerate(teacher_outs['feat_shape']):
                teacher_feat_H, teacher_feat_W = teacher_feat_shape
                student_feat_H, student_feat_W = student_outs['feat_shape'][idx]
                student_feat_flatten = student_outs["feat_flatten"][:, student_feat_index:student_feat_H * student_feat_W+student_feat_index, :].transpose(1,2).reshape(bs*N_cam, -1, student_feat_H, student_feat_W).contiguous()
                teacher_feat_flatten = teacher_outs["feat_flatten"][:, teacher_feat_index:teacher_feat_H * teacher_feat_W+teacher_feat_index, :].transpose(1,2).reshape(bs*N_cam, -1, teacher_feat_H, teacher_feat_W).contiguous()
                student_feat_index += student_feat_H * student_feat_W
                teacher_feat_index += teacher_feat_H * teacher_feat_W
                if self.flag_distill_on_TFPN:
                    for item_loss in self.distiller_tfpn[0].methods:
                        loss_name = item_loss.name
                        student_loss[loss_name+"_layer"+str(idx)] = self.distill_losses[loss_name](student_feat_flatten, teacher_feat_flatten, kwargs['gt_bboxes'], kwargs['img_metas'])
                        if idx == 3:
                            student_loss[loss_name+"_layer"+str(idx)] *= 0.1
                            
                if self.flag_distill_on_ATTN:
                    def get_correlation_matrix(intpu_feat, query_feat):
                        bn_, c_, h_, w_ = intpu_feat.size()
                        attn_dec = query_feat.sum(dim=0)
                        attn_dec = attn_dec.softmax(-1)
                        attn_dec = attn_dec.unsqueeze(1).repeat(1, 6, 1, 1).reshape(bn_, -1, attn_dec.size(-1)).permute(0, 2, 1)
                        att_mat = torch.einsum('bcn,bchw->bnhw', [attn_dec, intpu_feat.float()])
                        return att_mat
                    teacher_attn = get_correlation_matrix(teacher_feat_flatten, teacher_outs['querys'])
                    student_attn = get_correlation_matrix(student_feat_flatten, student_outs['querys'])
                    for item_loss in self.distiller_attn[0].methods:
                        loss_name = item_loss.name
                        student_loss[loss_name+"_layer"+str(idx)] = self.distill_losses[loss_name](student_attn, teacher_attn, kwargs['gt_bboxes'], kwargs['img_metas'])
                        if idx == 3:
                            student_loss[loss_name+"_layer"+str(idx)] *= 0.1 #1.0 #0.5 #0.003
        
    
        if self.flag_distill_on_BEVATTN:
            pc_range = self.student_cfg.point_cloud_range
            student_querys = student_outs["querys"]#.sum(-1).unsqueeze(-1)
            teacher_querys = teacher_outs["querys"]#.sum(-1).unsqueeze(-1)

            for i, item_loc in enumerate(self.distiller_bevattn):
                loss_name = item_loc.methods[0].name
                student_module = 'student_' + item_loc.student_module.replace('.', '_')
                teacher_module = 'teacher_' + item_loc.teacher_module.replace('.', '_')
                student_learnable_fc = buffer_dict[student_module][:, -428:, :]
                teacher_learnable_fc = buffer_dict[teacher_module]
                student_reference_points = student_outs["reference_points"]
                teacher_reference_points = teacher_outs["reference_points"]
                bs, num_anchor = student_reference_points.shape[:2]
                student_reference_points = get_global_pos(student_reference_points, pc_range)
                student_key_points = student_reference_points.unsqueeze(-2) + student_learnable_fc.reshape(bs, num_anchor, -1, 3)
                bs, num_anchor = teacher_reference_points.shape[:2]
                teacher_reference_points = get_global_pos(teacher_reference_points, pc_range)
                teacher_key_points = teacher_reference_points.unsqueeze(-2) + teacher_learnable_fc.reshape(bs, num_anchor, -1, 3)
                
                # teacher_attnmap = get_attnmap_sum(teacher_key_points[:, -428:, :, :], teacher_querys[i][:, -428:, :], pc_range)
                teacher_attnmap = get_attnmap(teacher_key_points[:, -428:, :, :], teacher_querys[i][:, -428:, :], pc_range)
                # student_attnmap = get_attnmap_sum(student_key_points[:, -428:, :, :], student_querys[i][:, -428:, :], pc_range)
                student_attnmap = get_attnmap(student_key_points[:, -428:, :, :], student_querys[i][:, -428:, :], pc_range)
                student_loss[loss_name] = (1 - self.iter) * self.distill_losses[loss_name](student_attnmap, teacher_attnmap, kwargs['gt_bboxes'], kwargs['img_metas'])

        return student_loss
    
    def forward_test(self, **kwargs):
        return self.student.forward_test(**kwargs)
    
    def extract_feat(self, imgs):
        """Extract features from images."""
        return self.student.extract_feat(imgs)

    def simple_test(self, img, img_metas, **kwargs):
        return self.student.simple_test(img, img_metas, **kwargs)

    def aug_test(self, imgs, img_metas, **kwargs):
        return self.student.aug_test(imgs, img_metas, **kwargs)