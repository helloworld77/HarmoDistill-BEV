import torch
import torch.nn as nn 
from mmcv.cnn import Linear, bias_init_with_prob, Scale

from mmcv.runner import force_fp32
from mmdet.core import (build_assigner, build_sampler, multi_apply,
                        reduce_mean)
from mmdet.models.utils import build_transformer
from mmdet.models import HEADS, build_loss
from mmdet.models.dense_heads.anchor_free_head import AnchorFreeHead
from mmdet.models.utils.transformer import inverse_sigmoid
from mmdet3d.core.bbox.coders import build_bbox_coder
from mmdet3d.core.bbox import LiDARInstance3DBoxes
from projects.mmdet3d_plugin.core.bbox.util import normalize_bbox, denormalize_bbox
from projects.mmdet3d_plugin.models.utils.positional_encoding import pos2posemb3d, pos2posemb1d, nerf_positional_encoding
from projects.mmdet3d_plugin.models.utils.misc import MLN, topk_gather, transform_reference_points, memory_refresh, SELayer_Linear
import copy
from mmdet.models.utils import NormedLinear
from torch.distributions import Beta

@HEADS.register_module()
class HarmoDistillHead(AnchorFreeHead):
    def __init__(self,
                 num_classes,
                 in_channels=256,
                 stride=[16],
                 embed_dims=256,
                 num_query=100,
                 num_reg_fcs=2,
                 memory_len=1024,
                 topk_proposals=256,
                 num_propagated=256,
                 with_dn=True,
                 with_ego_pos=True,
                 match_with_velo=True,
                 match_costs=None,
                 transformer=None,
                 sync_cls_avg_factor=False,
                 code_weights=None,
                 bbox_coder=None,
                 loss_cls=dict(
                     type='CrossEntropyLoss',
                     bg_cls_weight=0.1,
                     use_sigmoid=False,
                     loss_weight=1.0,
                     class_weight=1.0),
                 loss_bbox=dict(type='L1Loss', loss_weight=5.0),
                 loss_iou=dict(type='GIoULoss', loss_weight=2.0),
                 train_cfg=dict(
                     assigner=dict(
                         type='HungarianAssigner3D',
                         cls_cost=dict(type='ClassificationCost', weight=1.),
                         reg_cost=dict(type='BBoxL1Cost', weight=5.0),
                         iou_cost=dict(
                             type='IoUCost', iou_mode='giou', weight=2.0)),),
                 test_cfg=dict(max_per_img=100),
                 scalar = 5,
                 noise_scale = 0.4,
                 noise_trans = 0.0,
                 dn_weight = 1.0,
                 split = 0.5,
                 raydn_group=1,
                 raydn_num=5,
                 raydn_alpha=8,
                 raydn_beta=2,
                 raydn_radius=3,
                 init_cfg=None,
                 normedlinear=False,
                 **kwargs):
        # NOTE here use `AnchorFreeHead` instead of `TransformerHead`,
        # since it brings inconvenience when the initialization of
        # `AnchorFreeHead` is called.
        if 'code_size' in kwargs:
            self.code_size = kwargs['code_size']
        else:
            self.code_size = 10
        if code_weights is not None:
            self.code_weights = code_weights
        else:
            self.code_weights = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.2, 0.2]

        self.code_weights = self.code_weights[:self.code_size]

        if match_costs is not None:
            self.match_costs = match_costs
        else:
            self.match_costs = self.code_weights
            
        self.bg_cls_weight = 0
        self.sync_cls_avg_factor = sync_cls_avg_factor
        class_weight = loss_cls.get('class_weight', None)
        if class_weight is not None and (self.__class__ is HarmoDistillHead):
            assert isinstance(class_weight, float), 'Expected ' \
                'class_weight to have type float. Found ' \
                f'{type(class_weight)}.'
            # NOTE following the official DETR rep0, bg_cls_weight means
            # relative classification weight of the no-object class.
            bg_cls_weight = loss_cls.get('bg_cls_weight', class_weight)
            assert isinstance(bg_cls_weight, float), 'Expected ' \
                'bg_cls_weight to have type float. Found ' \
                f'{type(bg_cls_weight)}.'
            class_weight = torch.ones(num_classes + 1) * class_weight
            # set background class as the last indice
            class_weight[num_classes] = bg_cls_weight
            loss_cls.update({'class_weight': class_weight})
            if 'bg_cls_weight' in loss_cls:
                loss_cls.pop('bg_cls_weight')
            self.bg_cls_weight = bg_cls_weight

        if train_cfg:
            assert 'assigner' in train_cfg, 'assigner should be provided '\
                'when train_cfg is set.'
            assigner = train_cfg['assigner']


            self.assigner = build_assigner(assigner)
            # DETR sampling=False, so use PseudoSampler
            sampler_cfg = dict(type='PseudoSampler')
            self.sampler = build_sampler(sampler_cfg, context=self)

        self.num_query = num_query
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.memory_len = memory_len
        self.topk_proposals = topk_proposals
        self.num_propagated = num_propagated
        self.with_dn = with_dn
        self.with_ego_pos = with_ego_pos
        self.match_with_velo = match_with_velo
        self.num_reg_fcs = num_reg_fcs
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.fp16_enabled = False
        self.embed_dims = embed_dims
        self.with_dn = with_dn
        self.stride=stride

        self.scalar = scalar
        self.bbox_noise_scale = noise_scale
        self.bbox_noise_trans = noise_trans
        self.dn_weight = dn_weight
        self.split = split 

        self.raydn_group=raydn_group
        self.raydn_num=raydn_num
        self.raydn_alpha=raydn_alpha
        self.raydn_beta=raydn_beta
        self.raydn_radius=raydn_radius
        self.raydn_sampler = Beta(raydn_alpha, raydn_beta)

        self.act_cfg = transformer.get('act_cfg',
                                       dict(type='ReLU', inplace=True))
        self.num_pred = 6
        self.normedlinear = normedlinear
        super(HarmoDistillHead, self).__init__(num_classes, in_channels, init_cfg = init_cfg)

        self.loss_cls = build_loss(loss_cls)
        self.loss_bbox = build_loss(loss_bbox)
        self.loss_iou = build_loss(loss_iou)

        if self.loss_cls.use_sigmoid:
            self.cls_out_channels = num_classes
        else:
            self.cls_out_channels = num_classes + 1

        self.transformer = build_transformer(transformer)

        self.code_weights = nn.Parameter(torch.tensor(
            self.code_weights), requires_grad=False)

        self.match_costs = nn.Parameter(torch.tensor(
            self.match_costs), requires_grad=False)

        self.bbox_coder = build_bbox_coder(bbox_coder)

        self.pc_range = nn.Parameter(torch.tensor(
            self.bbox_coder.pc_range), requires_grad=False)


        self._init_layers()
        self.reset_memory()

    def _init_layers(self):
        """Initialize layers of the transformer head."""
        cls_branch = []
        for _ in range(self.num_reg_fcs):
            cls_branch.append(Linear(self.embed_dims, self.embed_dims))
            cls_branch.append(nn.LayerNorm(self.embed_dims))
            cls_branch.append(nn.ReLU(inplace=True))
        if self.normedlinear:
            cls_branch.append(NormedLinear(self.embed_dims, self.cls_out_channels))
        else:
            cls_branch.append(Linear(self.embed_dims, self.cls_out_channels))
        fc_cls = nn.Sequential(*cls_branch)

        reg_branch = []
        for _ in range(self.num_reg_fcs):
            reg_branch.append(Linear(self.embed_dims, self.embed_dims))
            reg_branch.append(nn.ReLU())
        reg_branch.append(Linear(self.embed_dims, self.code_size))
        reg_branch = nn.Sequential(*reg_branch)

        self.cls_branches = nn.ModuleList(
            [fc_cls for _ in range(self.num_pred)])
        self.reg_branches = nn.ModuleList(
            [reg_branch for _ in range(self.num_pred)])

        self.reference_points = nn.Embedding(self.num_query, 3)
        if self.num_propagated > 0:
            self.pseudo_reference_points = nn.Embedding(self.num_propagated, 3)


        self.query_embedding = nn.Sequential(
            nn.Linear(self.embed_dims*3//2, self.embed_dims),
            nn.ReLU(),
            nn.Linear(self.embed_dims, self.embed_dims),
        )

        self.spatial_alignment = MLN(14, use_ln=False)

        self.time_embedding = nn.Sequential(
            nn.Linear(self.embed_dims, self.embed_dims),
            nn.LayerNorm(self.embed_dims)
        )

        # encoding ego pose
        if self.with_ego_pos:
            self.ego_pose_pe = MLN(180)
            self.ego_pose_memory = MLN(180)

    def temporal_alignment(self, query_pos, tgt, reference_points):
        B = query_pos.size(0)

        temp_reference_point = (self.memory_reference_point - self.pc_range[:3]) / (self.pc_range[3:6] - self.pc_range[0:3])
        temp_pos = self.query_embedding(pos2posemb3d(temp_reference_point)) 
        temp_memory = self.memory_embedding
        rec_ego_pose = torch.eye(4, device=query_pos.device).unsqueeze(0).unsqueeze(0).repeat(B, query_pos.size(1), 1, 1)
        
        if self.with_ego_pos:
            rec_ego_motion = torch.cat([torch.zeros_like(reference_points[...,:3]), rec_ego_pose[..., :3, :].flatten(-2)], dim=-1)
            rec_ego_motion = nerf_positional_encoding(rec_ego_motion) #通过sin/cos将reference_points和eog_pose进行编码，扩张了12倍。[4, 1020, 3+12] * 12 -> [4, 1020, 180]
            tgt = self.ego_pose_memory(tgt, rec_ego_motion)
            query_pos = self.ego_pose_pe(query_pos, rec_ego_motion)
            memory_ego_motion = torch.cat([self.memory_velo, self.memory_timestamp, self.memory_egopose[..., :3, :].flatten(-2)], dim=-1).float()
            memory_ego_motion = nerf_positional_encoding(memory_ego_motion)
            temp_pos = self.ego_pose_pe(temp_pos, memory_ego_motion)
            temp_memory = self.ego_pose_memory(temp_memory, memory_ego_motion)

        query_pos += self.time_embedding(pos2posemb1d(torch.zeros_like(reference_points[...,:1])))
        temp_pos += self.time_embedding(pos2posemb1d(self.memory_timestamp).float())

        if self.num_propagated > 0:
            tgt = torch.cat([tgt, temp_memory[:, :self.num_propagated]], dim=1)
            query_pos = torch.cat([query_pos, temp_pos[:, :self.num_propagated]], dim=1)
            reference_points = torch.cat([reference_points, temp_reference_point[:, :self.num_propagated]], dim=1)
            rec_ego_pose = torch.eye(4, device=query_pos.device).unsqueeze(0).unsqueeze(0).repeat(B, query_pos.shape[1]+self.num_propagated, 1, 1)
            temp_memory = temp_memory[:, self.num_propagated:]
            temp_pos = temp_pos[:, self.num_propagated:]
            
        return tgt, query_pos, reference_points, temp_memory, temp_pos, rec_ego_pose

    def prepare_for_dn(self, batch_size, reference_points, img_metas, data):
        if self.training and self.with_dn:
            targets = [torch.cat((img_meta['gt_bboxes_3d']._data.gravity_center, img_meta['gt_bboxes_3d']._data.tensor[:, 3:]),dim=1) for img_meta in img_metas ]
            labels = [img_meta['gt_labels_3d']._data for img_meta in img_metas ]
            # add teacher pseudo label from teacher_outs
            if "teacher_outs" in data.keys():
                targets = [ torch.cat((t, data['teacher_outs']['pseudo_bboxes_3d'][i].tensor.to(t.device)), dim=0) for i, t in enumerate(targets) ]
                labels = [ torch.cat((t, data['teacher_outs']['pseudo_label_3d'][i].to(t.device)), dim=0) for i, t in enumerate(labels) ]

            known = [(torch.ones_like(t)).to(reference_points.device) for t in labels]
            know_idx = known
            unmask_bbox = unmask_label = torch.cat(known)
            #gt_num
            known_num = [t.size(0) for t in targets]

            labels = torch.cat([t for t in labels])
            boxes = torch.cat([t for t in targets])
            batch_idx = torch.cat([torch.full((t.size(0), ), i) for i, t in enumerate(targets)])
        
            known_indice = torch.nonzero(unmask_label + unmask_bbox)
            known_indice = known_indice.view(-1)
            # add noise
            total_raydn_num = self.raydn_num * self.raydn_group
            known_indice = known_indice.repeat(self.scalar+total_raydn_num, 1).view(-1)
            known_labels = labels.repeat(self.scalar, 1).view(-1).long().to(reference_points.device)
            known_bid = batch_idx.repeat(self.scalar+total_raydn_num, 1).view(-1)
            known_bboxs = boxes.repeat(self.scalar, 1).to(reference_points.device)
            known_bbox_center = known_bboxs[:, :3].clone()
            known_bbox_scale = known_bboxs[:, 3:6].clone()
            # center是不是上下不应该动？
            if self.bbox_noise_scale > 0:
                diff = known_bbox_scale / 2 + self.bbox_noise_trans
                rand_prob = torch.rand_like(known_bbox_center) * 2 - 1.0
                known_bbox_center += torch.mul(rand_prob, diff) * self.bbox_noise_scale
                known_bbox_center[..., 0:3] = (known_bbox_center[..., 0:3] - self.pc_range[0:3]) / (self.pc_range[3:6] - self.pc_range[0:3])

                known_bbox_center = known_bbox_center.clamp(min=0.0, max=1.0)
                mask = torch.norm(rand_prob, 2, 1) > self.split
                known_labels[mask] = self.num_classes

           # Ray Denoising
            for g_id in range(self.raydn_group):
                raydn_known_labels = labels.repeat(self.raydn_num, 1).view(-1).long().to(reference_points.device)
                raydn_known_bboxs = boxes.repeat(self.raydn_num, 1).to(reference_points.device)
                raydn_known_bbox_center = raydn_known_bboxs[:, :3].clone()
                raydn_known_bbox_scale = raydn_known_bboxs[:, 3:6].clone()
                noise_scale = raydn_known_bbox_scale[:, :].mean(dim=-1) / 2
                noise_step = (self.raydn_sampler.sample([noise_scale.shape[0]]).to(reference_points.device) * 2 - 1.0) * self.raydn_radius

                noise_scale = noise_scale.view(self.raydn_num, -1)
                noise_step = noise_step.view(self.raydn_num, -1)
                min_value, min_index = noise_step.abs().min(dim=0)
                reset_mask = min_value.abs() > self.split
                reset_value = (torch.rand(reset_mask.sum()).to(reference_points.device) * 2 - 1) * self.split     
                min_value[reset_mask] = reset_value           
                noise_step.scatter_(0, min_index.unsqueeze(0), min_value.unsqueeze(0))
                mask = torch.zeros_like(noise_step)
                mask.scatter_(0, min_index.unsqueeze(0), 1)
                mask = mask < 1
                mask = mask.view(-1)
                raydn_known_labels[mask] = self.num_classes

                raydn_known_bbox_center = raydn_known_bbox_center.view(self.raydn_num, -1, 3)
                ori_raydn_known_bbox_center = raydn_known_bbox_center.clone()
                for view_id in range(data['lidar2img'].shape[1]):
                    raydn_known_bbox_center_copy = torch.cat([ori_raydn_known_bbox_center.clone(), ori_raydn_known_bbox_center.new_ones((ori_raydn_known_bbox_center.shape[0], ori_raydn_known_bbox_center.shape[1], 1))], dim=-1)
                    tmp_p = raydn_known_bbox_center_copy.new_zeros(raydn_known_bbox_center_copy.shape)
                    for batch_id in range(data['lidar2img'].shape[0]):
                        tmp_p[:, sum(known_num[:batch_id]): sum(known_num[:batch_id+1])] = (data['lidar2img'][batch_id][view_id] @ raydn_known_bbox_center_copy[:, sum(known_num[:batch_id]): sum(known_num[:batch_id+1])].permute(0, 2, 1)).permute(0, 2, 1)

                    z_mask = tmp_p[..., 2] > 0 # depth > 0
                    tmp_p[..., :2] = tmp_p[..., :2] / (tmp_p[..., 2:3] + z_mask.unsqueeze(-1) * 1e-6 - (~z_mask).unsqueeze(-1) * 1e-6)
                    pad_h, pad_w = img_metas[0]['pad_shape'][0][:2] #(320, 800) #(640, 1600)
                    hw_mask = (
                        (tmp_p[..., 0] < pad_w)
                        & (tmp_p[..., 0] >= 0)
                        & (tmp_p[..., 1] < pad_h)
                        & (tmp_p[..., 1] >= 0)
                    ) # 0 < u < h and 0 < v < w
                    valid_mask = torch.logical_and(hw_mask, z_mask)
                    tmp_p[..., 2] += noise_scale*noise_step
                    tmp_p[..., :2] = tmp_p[..., :2] * tmp_p[..., 2:3]
                    proj_back = raydn_known_bbox_center_copy.new_zeros(raydn_known_bbox_center_copy.shape)
                    for batch_id in range(data['lidar2img'].shape[0]):
                        proj_back[:, sum(known_num[:batch_id]): sum(known_num[:batch_id+1])] = (data['lidar2img'][batch_id][view_id].inverse() @ tmp_p[:, sum(known_num[:batch_id]): sum(known_num[:batch_id+1])].permute(0, 2, 1)).permute(0, 2, 1)
                    raydn_known_bbox_center[valid_mask.unsqueeze(-1).repeat(1, 1, 3)] = proj_back[..., :3][valid_mask.unsqueeze(-1).repeat(1, 1, 3)]
                raydn_known_bbox_center = raydn_known_bbox_center.view(-1, 3)
                raydn_known_bbox_center[..., 0:3] = (raydn_known_bbox_center[..., 0:3] - self.pc_range[0:3]) / (self.pc_range[3:6] - self.pc_range[0:3])
                raydn_known_bbox_center = raydn_known_bbox_center.clamp(min=0.0, max=1.0)
                known_labels = torch.cat([known_labels, raydn_known_labels], dim=0)
                known_bbox_center = torch.cat([known_bbox_center, raydn_known_bbox_center], dim=0)
            known_bboxs = boxes.repeat(self.scalar+total_raydn_num, 1).to(reference_points.device)

            single_pad = int(max(known_num))
            pad_size = int(single_pad * (self.scalar+total_raydn_num))
            padding_bbox = torch.zeros(pad_size, 3).to(reference_points.device)
            padded_reference_points = torch.cat([padding_bbox, reference_points], dim=0).unsqueeze(0).repeat(batch_size, 1, 1)
            if len(known_num):
                map_known_indice = torch.cat([torch.tensor(range(num)) for num in known_num])  # [1,2, 1,2,3]
                map_known_indice = torch.cat([map_known_indice + single_pad * i for i in range(self.scalar+total_raydn_num)]).long()
            if len(known_bid):
                padded_reference_points[(known_bid.long(), map_known_indice)] = known_bbox_center.to(reference_points.device)

            tgt_size = pad_size + self.num_query
            attn_mask = torch.ones(tgt_size, tgt_size).to(reference_points.device) < 0
            # match query cannot see the reconstruct
            attn_mask[pad_size:, :pad_size] = True
            # reconstruct cannot see each other
            for i in range(self.scalar):
                if i == 0:
                    attn_mask[single_pad * i:single_pad * (i + 1), single_pad * (i + 1):pad_size] = True
                # if i == self.scalar - 1:
                #     attn_mask[single_pad * i:single_pad * (i + 1), :single_pad * i] = True
                else:
                    attn_mask[single_pad * i:single_pad * (i + 1), single_pad * (i + 1):pad_size] = True
                    attn_mask[single_pad * i:single_pad * (i + 1), :single_pad * i] = True
            for i in range(self.raydn_group):
                attn_mask[single_pad * (self.scalar + i*self.raydn_num):single_pad * (self.scalar + (i + 1)*self.raydn_num), single_pad * (self.scalar + (i + 1)*self.raydn_num):pad_size] = True
                attn_mask[single_pad * (self.scalar + i*self.raydn_num):single_pad * (self.scalar + (i + 1)*self.raydn_num), :single_pad * (self.scalar + i*self.raydn_num)] = True

            # update dn mask for temporal modeling
            query_size = pad_size + self.num_query + self.num_propagated
            tgt_size = pad_size + self.num_query + self.memory_len
            temporal_attn_mask = torch.ones(query_size, tgt_size).to(reference_points.device) < 0
            temporal_attn_mask[:attn_mask.size(0), :attn_mask.size(1)] = attn_mask 
            temporal_attn_mask[pad_size:, :pad_size] = True
            attn_mask = temporal_attn_mask

            mask_dict = {
                'known_indice': torch.as_tensor(known_indice).long(),
                'batch_idx': torch.as_tensor(batch_idx).long(),
                'map_known_indice': torch.as_tensor(map_known_indice).long(),
                'known_lbs_bboxes': (known_labels, known_bboxs),
                'know_idx': know_idx,
                'pad_size': pad_size
            }
            
        else:
            padded_reference_points = reference_points.unsqueeze(0).repeat(batch_size, 1, 1)
            attn_mask = None
            mask_dict = None

        return padded_reference_points, attn_mask, mask_dict


    def init_weights(self):
        """Initialize weights of the transformer head."""
        # The initialization for transformer is important
        nn.init.uniform_(self.reference_points.weight.data, 0, 1)
        if self.num_propagated > 0:
            nn.init.uniform_(self.pseudo_reference_points.weight.data, 0, 1)
            self.pseudo_reference_points.weight.requires_grad = False
        self.transformer.init_weights()
        if self.loss_cls.use_sigmoid:
            bias_init = bias_init_with_prob(0.01)
            for m in self.cls_branches:
                nn.init.constant_(m[-1].bias, bias_init)


    def reset_memory(self):
        self.memory_embedding = None
        self.memory_reference_point = None
        self.memory_timestamp = None
        self.memory_egopose = None
        self.memory_velo = None

    def pre_update_memory(self, data):
        x = data['prev_exists']
        B = x.size(0)
        # refresh the memory when the scene changes
        if self.memory_embedding is None:
            self.memory_embedding = x.new_zeros(B, self.memory_len, self.embed_dims)
            self.memory_reference_point = x.new_zeros(B, self.memory_len, 3)
            self.memory_timestamp = x.new_zeros(B, self.memory_len, 1)
            self.memory_egopose = x.new_zeros(B, self.memory_len, 4, 4)
            self.memory_velo = x.new_zeros(B, self.memory_len, 2)
        else:
            self.memory_timestamp += data['timestamp'].unsqueeze(-1).unsqueeze(-1)
            self.memory_egopose = data['ego_pose_inv'].unsqueeze(1) @ self.memory_egopose
            self.memory_reference_point = transform_reference_points(self.memory_reference_point, data['ego_pose_inv'], reverse=False)
            self.memory_timestamp = memory_refresh(self.memory_timestamp[:, :self.memory_len], x)
            self.memory_reference_point = memory_refresh(self.memory_reference_point[:, :self.memory_len], x)
            self.memory_embedding = memory_refresh(self.memory_embedding[:, :self.memory_len], x)
            self.memory_egopose = memory_refresh(self.memory_egopose[:, :self.memory_len], x)
            self.memory_velo = memory_refresh(self.memory_velo[:, :self.memory_len], x)
        
        # for the first frame, padding pseudo_reference_points (non-learnable)
        if self.num_propagated > 0:
            pseudo_reference_points = self.pseudo_reference_points.weight * (self.pc_range[3:6] - self.pc_range[0:3]) + self.pc_range[0:3]
            self.memory_reference_point[:, :self.num_propagated]  = self.memory_reference_point[:, :self.num_propagated] + (1 - x).view(B, 1, 1) * pseudo_reference_points
            self.memory_egopose[:, :self.num_propagated]  = self.memory_egopose[:, :self.num_propagated] + (1 - x).view(B, 1, 1, 1) * torch.eye(4, device=x.device)

    def post_update_memory(self, data, rec_ego_pose, all_cls_scores, all_bbox_preds, outs_dec, mask_dict):
        if self.training and mask_dict and mask_dict['pad_size'] > 0:
            rec_reference_points = all_bbox_preds[:, :, mask_dict['pad_size']:, :3][-1]
            rec_velo = all_bbox_preds[:, :, mask_dict['pad_size']:, -2:][-1]
            rec_memory = outs_dec[:, :, mask_dict['pad_size']:, :][-1]
            rec_score = all_cls_scores[:, :, mask_dict['pad_size']:, :][-1].sigmoid().topk(1, dim=-1).values[..., 0:1]
            rec_timestamp = torch.zeros_like(rec_score, dtype=torch.float64)
        else:
            rec_reference_points = all_bbox_preds[..., :3][-1]
            rec_velo = all_bbox_preds[..., -2:][-1]
            rec_memory = outs_dec[-1]
            rec_score = all_cls_scores[-1].sigmoid().topk(1, dim=-1).values[..., 0:1]
            rec_timestamp = torch.zeros_like(rec_score, dtype=torch.float64)
        
        # topk proposals
        _, topk_indexes = torch.topk(rec_score, self.topk_proposals, dim=1)
        rec_timestamp = topk_gather(rec_timestamp, topk_indexes)
        rec_reference_points = topk_gather(rec_reference_points, topk_indexes).detach()
        rec_memory = topk_gather(rec_memory, topk_indexes).detach()
        rec_ego_pose = topk_gather(rec_ego_pose, topk_indexes)
        rec_velo = topk_gather(rec_velo, topk_indexes).detach()

        self.memory_embedding = torch.cat([rec_memory, self.memory_embedding], dim=1)
        self.memory_timestamp = torch.cat([rec_timestamp, self.memory_timestamp], dim=1)
        self.memory_egopose= torch.cat([rec_ego_pose, self.memory_egopose], dim=1)
        self.memory_reference_point = torch.cat([rec_reference_points, self.memory_reference_point], dim=1)
        self.memory_velo = torch.cat([rec_velo, self.memory_velo], dim=1)
        self.memory_reference_point = transform_reference_points(self.memory_reference_point, data['ego_pose'], reverse=False)
        self.memory_timestamp -= data['timestamp'].unsqueeze(-1).unsqueeze(-1)
        self.memory_egopose = data['ego_pose'].unsqueeze(1) @ self.memory_egopose
        
    def get_spatial_alignment(self, **data):
        self.pre_update_memory(data)
        mlvl_feats = data['img_feats'] 
        # B = mlvl_feats[0].size(0) # [B, 6, 3, 256, 704] -> [B, 6, 256, 32, 88] [B, 6, 256, 16, 44] [B, 6, 256, 8, 22] [B, 6, 256, 4, 11]  
        intrinsics = data['intrinsics'] / 1e3 # [B, 6, 4x4]
        extrinsics = data['extrinsics'][..., :3, :] #[B, 6, 3x4]
        mln_input = torch.cat([intrinsics[..., 0,0:1], intrinsics[..., 1,1:2], extrinsics.flatten(-2)], dim=-1) # [B, 6, 14]
        mln_input = mln_input.flatten(0, 1).unsqueeze(1) # [Bx6, 1, 14]
        feat_flatten = []
        spatial_flatten = []
        for i in range(len(mlvl_feats)):
            B, N, C, H, W = mlvl_feats[i].shape
            mlvl_feat = mlvl_feats[i].reshape(B * N, C, -1).transpose(1, 2) # [Bx6, 2816, 256] [Bx6, 704, 256] [Bx6, 176, 256] [Bx6, 44, 256]
            mlvl_feat = self.spatial_alignment(mlvl_feat, mln_input) # [Bx6, 2816, 256] [Bx6, 704, 256] [Bx6, 176, 256] [Bx6, 44, 256]
            feat_flatten.append(mlvl_feat)
            spatial_flatten.append((H, W))
        feat_flatten = torch.cat(feat_flatten, dim=1) # [Bx6, 3740, 256]
        # self.distill_feat_flatten = feat_flatten
        # self.distill_spatial_flatten = spatial_flatten
        return feat_flatten, spatial_flatten
        
    def forward(self, img_metas, **data):
        """Forward function.
        Args:
            mlvl_feats (tuple[Tensor]): Features from the upstream
                network, each is a 5D-tensor with shape
                (B, N, C, H, W).
        Returns:
            all_cls_scores (Tensor): Outputs from the classification head, \
                shape [nb_dec, bs, num_query, cls_out_channels]. Note \
                cls_out_channels should includes background.
            all_bbox_preds (Tensor): Sigmoid outputs from the regression \
                head with normalized coordinate format (cx, cy, w, l, cz, h, theta, vx, vy). \
                Shape [nb_dec, bs, num_query, 9].
        """
        self.pre_update_memory(data)
        mlvl_feats = data['img_feats'] 
        B = mlvl_feats[0].size(0) # [B, 6, 3, 256, 704] -> [B, 6, 256, 32, 88] [B, 6, 256, 16, 44] [B, 6, 256, 8, 22] [B, 6, 256, 4, 11] 
        reference_points = self.reference_points.weight
        dtype = reference_points.dtype
        intrinsics = data['intrinsics'] / 1e3 # [B, 6, 4x4]
        extrinsics = data['extrinsics'][..., :3, :] #[B, 6, 3x4]
        mln_input = torch.cat([intrinsics[..., 0,0:1], intrinsics[..., 1,1:2], extrinsics.flatten(-2)], dim=-1) # [B, 6, 14]
        mln_input = mln_input.flatten(0, 1).unsqueeze(1) # [Bx6, 1, 14]
        feat_flatten = []
        spatial_flatten = []
        for i in range(len(mlvl_feats)):
            B, N, C, H, W = mlvl_feats[i].shape
            mlvl_feat = mlvl_feats[i].reshape(B * N, C, -1).transpose(1, 2) # [Bx6, 2816, 256] [Bx6, 704, 256] [Bx6, 176, 256] [Bx6, 44, 256]
            mlvl_feat = self.spatial_alignment(mlvl_feat, mln_input) # [[Bx6, 2816, 256] [Bx6, 704, 256] [Bx6, 176, 256] [Bx6, 44, 256]] [Bx6, 1, 14] -- [Bx6, 44, 256]
            feat_flatten.append(mlvl_feat.to(dtype))
            spatial_flatten.append((H, W))
        feat_flatten = torch.cat(feat_flatten, dim=1) # [Bx6, 3740, 256]
        spatial_flatten = torch.as_tensor(spatial_flatten, dtype=torch.long, device=mlvl_feats[0].device)
        level_start_index = torch.cat((spatial_flatten.new_zeros((1, )), spatial_flatten.prod(1).cumsum(0)[:-1])) # [0, 2816, 3520, 3696]
        reference_points, attn_mask, mask_dict = self.prepare_for_dn(B, reference_points, img_metas, data)
        query_pos = self.query_embedding(pos2posemb3d(reference_points))
        tgt = torch.zeros_like(query_pos)  # [4, 1020, 256]

        # prepare for the tgt and query_pos using mln.
        tgt, query_pos, reference_points, temp_memory, temp_pos, rec_ego_pose = self.temporal_alignment(query_pos, tgt, reference_points)  # [4, 1148, 256] [4, 1148, 256] [4, 1148, 3] [4, 384, 256] [4, 384, 256] [4, 1276, 4, 4]

        outs_dec = self.transformer(tgt, query_pos, feat_flatten, spatial_flatten, level_start_index, temp_memory, 
                                    temp_pos, attn_mask, reference_points, self.pc_range, data, img_metas)

        outs_dec = torch.nan_to_num(outs_dec)  # [6, 4, 1148, 256]
        outputs_classes = []
        outputs_coords = []
        for lvl in range(outs_dec.shape[0]):
            reference = inverse_sigmoid(reference_points.clone()) # [4, 1148, 3]
            assert reference.shape[-1] == 3
            outputs_class = self.cls_branches[lvl](outs_dec[lvl]) # [4, 1148, 10]
            tmp = self.reg_branches[lvl](outs_dec[lvl])           # [4, 1148, 10]

            tmp[..., 0:3] += reference[..., 0:3]                  # 加 reference 形成 coordinate
            tmp[..., 0:3] = tmp[..., 0:3].sigmoid()

            outputs_coord = tmp
            outputs_classes.append(outputs_class)
            outputs_coords.append(outputs_coord)

        all_cls_scores = torch.stack(outputs_classes)   # 6x [4, 1148, 10]
        all_bbox_preds = torch.stack(outputs_coords)    # 6x [4, 1148, 10]
        all_bbox_preds[..., 0:3] = (all_bbox_preds[..., 0:3] * (self.pc_range[3:6] - self.pc_range[0:3]) + self.pc_range[0:3])
        
        # update the memory bank
        self.post_update_memory(data, rec_ego_pose, all_cls_scores, all_bbox_preds, outs_dec, mask_dict)


        if mask_dict and mask_dict['pad_size'] > 0:  # mask_dict['pad_size']=720
            output_known_class = all_cls_scores[:, :, :mask_dict['pad_size'], :]    # [6, 4, 720, 10]
            output_known_coord = all_bbox_preds[:, :, :mask_dict['pad_size'], :]    # [6, 4, 720, 10]
            outputs_class = all_cls_scores[:, :, mask_dict['pad_size']:, :]         # [6, 4, 428, 10]
            outputs_coord = all_bbox_preds[:, :, mask_dict['pad_size']:, :]         # [6, 4, 428, 10]
            outputs_query = outs_dec[:, :, mask_dict['pad_size']:, :]               # [6, 4, 428, 256]
            reference_points = reference_points[:, mask_dict['pad_size']:, :]       # [4, 428, 3]
            mask_dict['output_known_lbs_bboxes']=(output_known_class, output_known_coord)
            outs = {
                'all_cls_scores': outputs_class, # 6, 4, 428, 10
                'all_bbox_preds': outputs_coord, # 6, 4, 428, 10
                'dn_mask_dict': mask_dict,        # 
                'feat_flatten': feat_flatten,
                'querys': outputs_query,
                'reference_points': reference_points,
                'feat_shape': spatial_flatten,
            }
        else:
            outs = {
                'all_cls_scores': all_cls_scores, # 6, 1, 428, 10
                'all_bbox_preds': all_bbox_preds, # 6, 1, 428, 10
                'dn_mask_dict': None,
                'feat_flatten': feat_flatten,
                'querys': outs_dec,
                'reference_points': reference_points,
                'feat_shape': spatial_flatten,
            }

        return outs


    def prepare_for_loss(self, mask_dict):
        """
        prepare dn components to calculate loss
        Args:
            mask_dict: a dict that contains dn information
        """
        output_known_class, output_known_coord = mask_dict['output_known_lbs_bboxes']
        known_labels, known_bboxs = mask_dict['known_lbs_bboxes']
        map_known_indice = mask_dict['map_known_indice'].long()
        known_indice = mask_dict['known_indice'].long().cpu()
        batch_idx = mask_dict['batch_idx'].long()
        bid = batch_idx[known_indice]
        if len(output_known_class) > 0:
            output_known_class = output_known_class.permute(1, 2, 0, 3)[(bid, map_known_indice)].permute(1, 0, 2)
            output_known_coord = output_known_coord.permute(1, 2, 0, 3)[(bid, map_known_indice)].permute(1, 0, 2)
        num_tgt = known_indice.numel()
        return known_labels, known_bboxs, output_known_class, output_known_coord, num_tgt


    def _get_target_single(self,
                           cls_score,
                           bbox_pred,
                           gt_labels,
                           gt_bboxes,
                           gt_bboxes_ignore=None):
        """"Compute regression and classification targets for one image.
        Outputs from a single decoder layer of a single feature level are used.
        Args:
            cls_score (Tensor): Box score logits from a single decoder layer
                for one image. Shape [num_query, cls_out_channels].
            bbox_pred (Tensor): Sigmoid outputs from a single decoder layer
                for one image, with normalized coordinate (cx, cy, w, h) and
                shape [num_query, 4].
            gt_bboxes (Tensor): Ground truth bboxes for one image with
                shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (Tensor): Ground truth class indexes for one image
                with shape (num_gts, ).
            gt_bboxes_ignore (Tensor, optional): Bounding boxes
                which can be ignored. Default None.
        Returns:
            tuple[Tensor]: a tuple containing the following for one image.
                - labels (Tensor): Labels of each image.
                - label_weights (Tensor]): Label weights of each image.
                - bbox_targets (Tensor): BBox targets of each image.
                - bbox_weights (Tensor): BBox weights of each image.
                - pos_inds (Tensor): Sampled positive indexes for each image.
                - neg_inds (Tensor): Sampled negative indexes for each image.
        """

        num_bboxes = bbox_pred.size(0)
        # assigner and sampler

        assign_result = self.assigner.assign(bbox_pred, cls_score, gt_bboxes,
                                                gt_labels, gt_bboxes_ignore, self.match_costs, self.match_with_velo)
        sampling_result = self.sampler.sample(assign_result, bbox_pred,
                                              gt_bboxes)
        pos_inds = sampling_result.pos_inds
        neg_inds = sampling_result.neg_inds

        # label targets
        labels = gt_bboxes.new_full((num_bboxes, ),
                                    self.num_classes,
                                    dtype=torch.long)
        label_weights = gt_bboxes.new_ones(num_bboxes)

        # bbox targets
        code_size = gt_bboxes.size(1)
        bbox_targets = torch.zeros_like(bbox_pred)[..., :code_size]
        bbox_weights = torch.zeros_like(bbox_pred)
        # print(gt_bboxes.size(), bbox_pred.size())
        # DETR
        if sampling_result.num_gts > 0:
            bbox_targets[pos_inds] = sampling_result.pos_gt_bboxes
            bbox_weights[pos_inds] = 1.0
            labels[pos_inds] = gt_labels[sampling_result.pos_assigned_gt_inds]
        return (labels, label_weights, bbox_targets, bbox_weights, 
                pos_inds, neg_inds)

    def get_targets(self,
                    cls_scores_list,
                    bbox_preds_list,
                    gt_bboxes_list,
                    gt_labels_list,
                    gt_bboxes_ignore_list=None):
        """"Compute regression and classification targets for a batch image.
        Outputs from a single decoder layer of a single feature level are used.
        Args:
            cls_scores_list (list[Tensor]): Box score logits from a single
                decoder layer for each image with shape [num_query,
                cls_out_channels].
            bbox_preds_list (list[Tensor]): Sigmoid outputs from a single
                decoder layer for each image, with normalized coordinate
                (cx, cy, w, h) and shape [num_query, 4].
            gt_bboxes_list (list[Tensor]): Ground truth bboxes for each image
                with shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels_list (list[Tensor]): Ground truth class indexes for each
                image with shape (num_gts, ).
            gt_bboxes_ignore_list (list[Tensor], optional): Bounding
                boxes which can be ignored for each image. Default None.
        Returns:
            tuple: a tuple containing the following targets.
                - labels_list (list[Tensor]): Labels for all images.
                - label_weights_list (list[Tensor]): Label weights for all \
                    images.
                - bbox_targets_list (list[Tensor]): BBox targets for all \
                    images.
                - bbox_weights_list (list[Tensor]): BBox weights for all \
                    images.
                - num_total_pos (int): Number of positive samples in all \
                    images.
                - num_total_neg (int): Number of negative samples in all \
                    images.
        """
        assert gt_bboxes_ignore_list is None, \
            'Only supports for gt_bboxes_ignore setting to None.'
        num_imgs = len(cls_scores_list)
        gt_bboxes_ignore_list = [
            gt_bboxes_ignore_list for _ in range(num_imgs)
        ]

        (labels_list, label_weights_list, bbox_targets_list,
         bbox_weights_list, pos_inds_list, neg_inds_list) = multi_apply(
             self._get_target_single, cls_scores_list, bbox_preds_list,
             gt_labels_list, gt_bboxes_list, gt_bboxes_ignore_list)
        num_total_pos = sum((inds.numel() for inds in pos_inds_list))
        num_total_neg = sum((inds.numel() for inds in neg_inds_list))
        return (labels_list, label_weights_list, bbox_targets_list,
                bbox_weights_list, num_total_pos, num_total_neg)

    def loss_single(self,
                    cls_scores,
                    bbox_preds,
                    gt_bboxes_list,
                    gt_labels_list,
                    gt_bboxes_ignore_list=None):
        """"Loss function for outputs from a single decoder layer of a single
        feature level.
        Args:
            cls_scores (Tensor): Box score logits from a single decoder layer
                for all images. Shape [bs, num_query, cls_out_channels].
            bbox_preds (Tensor): Sigmoid outputs from a single decoder layer
                for all images, with normalized coordinate (cx, cy, w, h) and
                shape [bs, num_query, 4].
            gt_bboxes_list (list[Tensor]): Ground truth bboxes for each image
                with shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels_list (list[Tensor]): Ground truth class indexes for each
                image with shape (num_gts, ).
            gt_bboxes_ignore_list (list[Tensor], optional): Bounding
                boxes which can be ignored for each image. Default None.
        Returns:
            dict[str, Tensor]: A dictionary of loss components for outputs from
                a single decoder layer.
        """
        num_imgs = cls_scores.size(0)
        cls_scores_list = [cls_scores[i] for i in range(num_imgs)]
        bbox_preds_list = [bbox_preds[i] for i in range(num_imgs)]
        cls_reg_targets = self.get_targets(cls_scores_list, bbox_preds_list,
                                           gt_bboxes_list, gt_labels_list, 
                                           gt_bboxes_ignore_list)
        (labels_list, label_weights_list, bbox_targets_list, bbox_weights_list,
         num_total_pos, num_total_neg) = cls_reg_targets
        labels = torch.cat(labels_list, 0)
        label_weights = torch.cat(label_weights_list, 0)
        bbox_targets = torch.cat(bbox_targets_list, 0)
        bbox_weights = torch.cat(bbox_weights_list, 0)

        # classification loss
        cls_scores = cls_scores.reshape(-1, self.cls_out_channels)
        # construct weighted avg_factor to match with the official DETR repo
        cls_avg_factor = num_total_pos * 1.0 + \
            num_total_neg * self.bg_cls_weight
        if self.sync_cls_avg_factor:
            cls_avg_factor = reduce_mean(
                cls_scores.new_tensor([cls_avg_factor]))

        cls_avg_factor = max(cls_avg_factor, 1)
        loss_cls = self.loss_cls(cls_scores, labels, label_weights, avg_factor=cls_avg_factor)

        # Compute the average number of gt boxes accross all gpus, for
        # normalization purposes
        num_total_pos = loss_cls.new_tensor([num_total_pos])
        num_total_pos = torch.clamp(reduce_mean(num_total_pos), min=1).item()

        # regression L1 loss
        bbox_preds = bbox_preds.reshape(-1, bbox_preds.size(-1))
        normalized_bbox_targets = normalize_bbox(bbox_targets, self.pc_range)
        isnotnan = torch.isfinite(normalized_bbox_targets).all(dim=-1)
        bbox_weights = bbox_weights * self.code_weights

        loss_bbox = self.loss_bbox(
                bbox_preds[isnotnan, :10], normalized_bbox_targets[isnotnan, :10], bbox_weights[isnotnan, :10], avg_factor=num_total_pos)

        loss_cls = torch.nan_to_num(loss_cls)
        loss_bbox = torch.nan_to_num(loss_bbox)
        return loss_cls, loss_bbox

   
    def dn_loss_single(self,
                    cls_scores,
                    bbox_preds,
                    known_bboxs,
                    known_labels,
                    num_total_pos=None):
        """"Loss function for outputs from a single decoder layer of a single
        feature level.
        Args:
            cls_scores (Tensor): Box score logits from a single decoder layer
                for all images. Shape [bs, num_query, cls_out_channels].
            bbox_preds (Tensor): Sigmoid outputs from a single decoder layer
                for all images, with normalized coordinate (cx, cy, w, h) and
                shape [bs, num_query, 4].
            gt_bboxes_list (list[Tensor]): Ground truth bboxes for each image
                with shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels_list (list[Tensor]): Ground truth class indexes for each
                image with shape (num_gts, ).
            gt_bboxes_ignore_list (list[Tensor], optional): Bounding
                boxes which can be ignored for each image. Default None.
        Returns:
            dict[str, Tensor]: A dictionary of loss components for outputs from
                a single decoder layer.
        """
        # classification loss
        cls_scores = cls_scores.reshape(-1, self.cls_out_channels)
        # construct weighted avg_factor to match with the official DETR repo
        cls_avg_factor = num_total_pos * 3.14159 / 6 * self.split * self.split  * self.split ### positive rate
        if self.sync_cls_avg_factor:
            cls_avg_factor = reduce_mean(
                cls_scores.new_tensor([cls_avg_factor]))
        bbox_weights = torch.ones_like(bbox_preds)
        label_weights = torch.ones_like(known_labels)
        cls_avg_factor = max(cls_avg_factor, 1)
        loss_cls = self.loss_cls(
            cls_scores, known_labels.long(), label_weights, avg_factor=cls_avg_factor)

        # Compute the average number of gt boxes accross all gpus, for
        # normalization purposes
        num_total_pos = loss_cls.new_tensor([num_total_pos])
        num_total_pos = torch.clamp(reduce_mean(num_total_pos), min=1).item()

        # regression L1 loss
        bbox_preds = bbox_preds.reshape(-1, bbox_preds.size(-1))
        normalized_bbox_targets = normalize_bbox(known_bboxs, self.pc_range)
        isnotnan = torch.isfinite(normalized_bbox_targets).all(dim=-1)

        bbox_weights = bbox_weights * self.code_weights

        
        loss_bbox = self.loss_bbox(
                bbox_preds[isnotnan, :10], normalized_bbox_targets[isnotnan, :10], bbox_weights[isnotnan, :10], avg_factor=num_total_pos)

        loss_cls = torch.nan_to_num(loss_cls)
        loss_bbox = torch.nan_to_num(loss_bbox)
        
        return self.dn_weight * loss_cls, self.dn_weight * loss_bbox
    
    @force_fp32(apply_to=('preds_dicts'))
    def loss(self,
             gt_bboxes_list,
             gt_labels_list,
             preds_dicts,
             gt_bboxes_ignore=None):
        """"Loss function.
        Args:
            gt_bboxes_list (list[Tensor]): Ground truth bboxes for each image
                with shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels_list (list[Tensor]): Ground truth class indexes for each
                image with shape (num_gts, ).
            preds_dicts:
                all_cls_scores (Tensor): Classification score of all
                    decoder layers, has shape
                    [nb_dec, bs, num_query, cls_out_channels].
                all_bbox_preds (Tensor): Sigmoid regression
                    outputs of all decode layers. Each is a 4D-tensor with
                    normalized coordinate format (cx, cy, w, h) and shape
                    [nb_dec, bs, num_query, 4].
                enc_cls_scores (Tensor): Classification scores of
                    points on encode feature map , has shape
                    (N, h*w, num_classes). Only be passed when as_two_stage is
                    True, otherwise is None.
                enc_bbox_preds (Tensor): Regression results of each points
                    on the encode feature map, has shape (N, h*w, 4). Only be
                    passed when as_two_stage is True, otherwise is None.
            gt_bboxes_ignore (list[Tensor], optional): Bounding boxes
                which can be ignored for each image. Default None.
        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        assert gt_bboxes_ignore is None, \
            f'{self.__class__.__name__} only supports ' \
            f'for gt_bboxes_ignore setting to None.'

        all_cls_scores = preds_dicts['all_cls_scores']
        all_bbox_preds = preds_dicts['all_bbox_preds']

        num_dec_layers = len(all_cls_scores)
        device = gt_labels_list[0].device
        gt_bboxes_list = [torch.cat(
            (gt_bboxes.gravity_center, gt_bboxes.tensor[:, 3:]),
            dim=1).to(device) for gt_bboxes in gt_bboxes_list]

        all_gt_bboxes_list = [gt_bboxes_list for _ in range(num_dec_layers)]
        all_gt_labels_list = [gt_labels_list for _ in range(num_dec_layers)]
        all_gt_bboxes_ignore_list = [
            gt_bboxes_ignore for _ in range(num_dec_layers)
        ]

        losses_cls, losses_bbox = multi_apply(
            self.loss_single, all_cls_scores, all_bbox_preds,
            all_gt_bboxes_list, all_gt_labels_list, 
            all_gt_bboxes_ignore_list)

        loss_dict = dict()

        # loss_dict['size_loss'] = size_loss
        # loss from the last decoder layer
        loss_dict['loss_cls'] = losses_cls[-1]
        loss_dict['loss_bbox'] = losses_bbox[-1]

        # loss from other decoder layers
        num_dec_layer = 0
        for loss_cls_i, loss_bbox_i in zip(losses_cls[:-1],
                                           losses_bbox[:-1]):
            loss_dict[f'd{num_dec_layer}.loss_cls'] = loss_cls_i
            loss_dict[f'd{num_dec_layer}.loss_bbox'] = loss_bbox_i
            num_dec_layer += 1
        
        if preds_dicts['dn_mask_dict'] is not None:
            known_labels, known_bboxs, output_known_class, output_known_coord, num_tgt = self.prepare_for_loss(preds_dicts['dn_mask_dict'])
            all_known_bboxs_list = [known_bboxs for _ in range(num_dec_layers)]
            all_known_labels_list = [known_labels for _ in range(num_dec_layers)]
            all_num_tgts_list = [
                num_tgt for _ in range(num_dec_layers)
            ]
            
            dn_losses_cls, dn_losses_bbox = multi_apply(
                self.dn_loss_single, output_known_class, output_known_coord,
                all_known_bboxs_list, all_known_labels_list, 
                all_num_tgts_list)
            loss_dict['dn_loss_cls'] = dn_losses_cls[-1]
            loss_dict['dn_loss_bbox'] = dn_losses_bbox[-1]
            num_dec_layer = 0
            for loss_cls_i, loss_bbox_i in zip(dn_losses_cls[:-1],
                                            dn_losses_bbox[:-1]):
                loss_dict[f'd{num_dec_layer}.dn_loss_cls'] = loss_cls_i
                loss_dict[f'd{num_dec_layer}.dn_loss_bbox'] = loss_bbox_i
                num_dec_layer += 1
                
        elif self.with_dn:
            dn_losses_cls, dn_losses_bbox = multi_apply(
                self.loss_single, all_cls_scores, all_bbox_preds,
                all_gt_bboxes_list, all_gt_labels_list, 
                all_gt_bboxes_ignore_list)
            loss_dict['dn_loss_cls'] = dn_losses_cls[-1].detach()
            loss_dict['dn_loss_bbox'] = dn_losses_bbox[-1].detach()     
            num_dec_layer = 0
            for loss_cls_i, loss_bbox_i in zip(dn_losses_cls[:-1],
                                            dn_losses_bbox[:-1]):
                loss_dict[f'd{num_dec_layer}.dn_loss_cls'] = loss_cls_i.detach()     
                loss_dict[f'd{num_dec_layer}.dn_loss_bbox'] = loss_bbox_i.detach()     
                num_dec_layer += 1

        return loss_dict


    @force_fp32(apply_to=('preds_dicts'))
    def get_bboxes(self, preds_dicts, img_metas, rescale=False):
        """Generate bboxes from bbox head predictions.
        Args:
            preds_dicts (tuple[list[dict]]): Prediction results.
            img_metas (list[dict]): Point cloud and image's meta info.
        Returns:
            list[dict]: Decoded bbox, scores and labels after nms.
        """
        preds_dicts = self.bbox_coder.decode(preds_dicts)
        num_samples = len(preds_dicts)

        ret_list = []
        for i in range(num_samples):
            preds = preds_dicts[i]
            bboxes = preds['bboxes']
            bboxes[:, 2] = bboxes[:, 2] - bboxes[:, 5] * 0.5
            bboxes = img_metas[i]['box_type_3d'](bboxes, bboxes.size(-1))
            scores = preds['scores']
            labels = preds['labels']
            ret_list.append([bboxes, scores, labels])
        return ret_list
    
    def forward_teacher_test(self, img_metas, **data):
        """Forward teacher function, no need to calculate loss.
        Args:
            mlvl_feats (tuple[Tensor]): Features from the upstream
                network, each is a 5D-tensor with shape
                (B, N, C, H, W).
        Returns:
            all_cls_scores (Tensor): Outputs from the classification head, \
                shape [nb_dec, bs, num_query, cls_out_channels]. Note \
                cls_out_channels should includes background.
            all_bbox_preds (Tensor): Sigmoid outputs from the regression \
                head with normalized coordinate format (cx, cy, w, l, cz, h, theta, vx, vy). \
                Shape [nb_dec, bs, num_query, 9].
        """
        self.pre_update_memory(data)
        mlvl_feats = data['img_feats'] 
        B = mlvl_feats[0].size(0)
        reference_points = self.reference_points.weight
        intrinsics = data['intrinsics'] / 1e3
        extrinsics = data['extrinsics'][..., :3, :]
        mln_input = torch.cat([intrinsics[..., 0,0:1], intrinsics[..., 1,1:2], extrinsics.flatten(-2)], dim=-1)
        mln_input = mln_input.flatten(0, 1).unsqueeze(1) # [Bx6, 1, 14]
        feat_flatten = []
        spatial_flatten = []
        for i in range(len(mlvl_feats)):
            B, N, C, H, W = mlvl_feats[i].shape
            mlvl_feat = mlvl_feats[i].reshape(B * N, C, -1).transpose(1, 2)
            # align the 2d feature into 3d spatial features by MLN
            mlvl_feat = self.spatial_alignment(mlvl_feat, mln_input)
            feat_flatten.append(mlvl_feat.to(reference_points.dtype))
            spatial_flatten.append((H, W))
        feat_flatten = torch.cat(feat_flatten, dim=1)
        spatial_flatten = torch.as_tensor(spatial_flatten, dtype=torch.long, device=mlvl_feats[0].device)
        level_start_index = torch.cat((spatial_flatten.new_zeros((1, )), spatial_flatten.prod(1).cumsum(0)[:-1]))
        # import pdb; pdb.set_trace()
        reference_points, attn_mask, mask_dict = self.prepare_for_dn(B, reference_points, img_metas, data)
        query_pos = self.query_embedding(pos2posemb3d(reference_points))
        tgt = torch.zeros_like(query_pos)
        # prepare for the tgt and query_pos using mln.
        tgt, query_pos, reference_points, temp_memory, temp_pos, rec_ego_pose = self.temporal_alignment(query_pos, tgt, reference_points)
        # transformer forward
        outs_dec = self.transformer(tgt, query_pos, feat_flatten, spatial_flatten, level_start_index, temp_memory, 
                                    temp_pos, attn_mask, reference_points, self.pc_range, data, img_metas)
        outs_dec = torch.nan_to_num(outs_dec)
        outputs_classes = []
        outputs_coords = []
        for lvl in range(outs_dec.shape[0]):
            reference = inverse_sigmoid(reference_points.clone())
            assert reference.shape[-1] == 3
            outputs_class = self.cls_branches[lvl](outs_dec[lvl])
            tmp = self.reg_branches[lvl](outs_dec[lvl])          
            tmp[..., 0:3] += reference[..., 0:3]                 
            tmp[..., 0:3] = tmp[..., 0:3].sigmoid()
            outputs_coord = tmp
            outputs_classes.append(outputs_class)
            outputs_coords.append(outputs_coord)

        all_cls_scores = torch.stack(outputs_classes)
        all_bbox_preds = torch.stack(outputs_coords)
        all_bbox_preds[..., 0:3] = (all_bbox_preds[..., 0:3] * (self.pc_range[3:6] - self.pc_range[0:3]) + self.pc_range[0:3])
        # update the memory bank
        self.post_update_memory(data, rec_ego_pose, all_cls_scores, all_bbox_preds, outs_dec, mask_dict)
        pseudo_bboxes_3d, pseudo_label_3d, pseudo_label_score_3d = self._get_teacher_results(all_cls_scores, all_bbox_preds, img_metas)
        outs = {
            'all_cls_scores': all_cls_scores, # 6, 1, 428, 10
            'all_bbox_preds': all_bbox_preds, # 6, 1, 428, 10
            'dn_mask_dict': None,
            'feat_flatten': feat_flatten,
            'querys': outs_dec,
            'reference_points': reference_points,
            'feat_shape': spatial_flatten,
            'pseudo_bboxes_3d': pseudo_bboxes_3d,
            'pseudo_label_3d': pseudo_label_3d,
            'pseudo_label_score_3d': pseudo_label_score_3d,
        }
        if 0:
            from projects.mmdet3d_plugin.core.visualizer import draw_bbox3d_on_img, draw_bbox3d_on_img_3x2, plot_raydn_known_bbox_center
            import matplotlib.pyplot as plt
            for i in range(B):
                print('scene_token:', img_metas[i]['scene_token'])
                plt.figure(figsize=(15, 10))
                plt.imshow(draw_bbox3d_on_img_3x2(data, pseudo_bboxes_3d, img_metas, i, color_pv=(255, 100, 0), color_bev=(0, 0, 230), background=190, bev_range=100))
                plt.savefig('known_bbox_center_2%s.png'%img_metas[i]['scene_token'])
            
        return outs
    
    def _get_teacher_results(self, all_cls_scores, all_bbox_preds, img_metas, topk=0):
        layer_iter, batch_size, query_num, num_cls = all_cls_scores.shape
        t_all_cls_scores = all_cls_scores.transpose(0, 1).reshape(batch_size, -1, self.num_classes)
        t_all_bbox_preds = all_bbox_preds.transpose(0, 1).reshape(batch_size, -1, 10)
        # labels = [img_meta['gt_labels_3d']._data for img_meta in img_metas ]
        pseudo_label_3d = []
        pseudo_label_score_3d = []
        pseudo_bboxes_3d = []
        for i in range(batch_size):
            t_all_cls_scores[i] = t_all_cls_scores[i].sigmoid()
            topk = topk if topk else img_metas[i]['gt_labels_3d']._data.size(0)
            scores, indexs = t_all_cls_scores[i].view(-1).topk(topk)
            # random sample gt_num of the topk
            # random_indices = torch.randperm(gt_num*5)[:gt_num]
            # indexs = indexs[random_indices]
            # scores = scores[random_indices]
            labels = indexs % self.num_classes
            bbox_index = torch.div(indexs, self.num_classes, rounding_mode='floor')
            bbox_preds = t_all_bbox_preds[i][bbox_index]
            final_box_preds = denormalize_bbox(bbox_preds, self.pc_range)  
            final_scores = scores 
            final_preds = labels
            post_center_range = torch.tensor([-61.2, -61.2, -10.0, 61.2, 61.2, 10.0], device=scores.device)
            mask = (final_box_preds[..., :3] >= post_center_range[:3]).all(1)
            mask &= (final_box_preds[..., :3] <= post_center_range[3:]).all(1)
            boxes3d = final_box_preds[mask]
            scores = final_scores[mask]
            labels = final_preds[mask]
            pseudo_label_3d.append(labels)
            pseudo_label_score_3d.append(scores)  
            # 将t_boxes3d转换为 LiDARInstance3DBoxes
            pseudo_bboxes_3d.append(LiDARInstance3DBoxes(boxes3d, box_dim=boxes3d.shape[-1]))
            
        return pseudo_bboxes_3d, pseudo_label_3d, pseudo_label_score_3d
    
    def draw_featmaps(self, c_feats, data, img_metas, idx=0, h=0, w=0, fig_title="fig", fid=0):
        import matplotlib.pyplot as plt
        import numpy as np; import cv2
        img_h, img_w = data['img'].shape[-2:]
        def resize_tensor(c_feat, h=0, w=0):
            if h == 0:
                return c_feat
            return torch.nn.functional.interpolate(c_feat.unsqueeze(0).unsqueeze(0), size=(img_h, img_w), mode='bilinear', align_corners=False).squeeze(0).squeeze(0)
        def plot_image(ax, image, title):
            ax.imshow(image)
            ax.set_title(title)
            ax.axis('off')
            
        fig = plt.figure(figsize=(30, 3))
        fig.canvas.set_window_title(fig_title)
        ii = idx // 6
        jj = idx % 6
        img = data['img'][ii, jj].permute(1, 2, 0).cpu().numpy() * img_metas[0]['img_norm_cfg']['std'] + img_metas[0]['img_norm_cfg']['mean']
        ax0 = plt.subplot2grid((2, 7), (0, 0))
        plot_image(ax0, img.astype(np.uint8), "input image")
        # 按路径读取 #img_metas[0]['filename'][0], resize 到img_h, img_w
        original_img = cv2.cvtColor(cv2.imread(img_metas[ii]['filename'][jj]), cv2.COLOR_BGR2RGB)
        ax1 = plt.subplot2grid((2, 7), (1, 0))
        plot_image(ax1, original_img.astype(np.uint8), "orig image")
        transformations = [
            (resize_tensor(c_feats[idx][fid]).cpu().numpy(), "mean"),
            (resize_tensor(c_feats[idx][fid]).reshape(-1).div(0.01).softmax(-1).cpu().numpy().reshape(h, w), "mean Softmax t=(0.01"),
            (resize_tensor(c_feats[idx][fid]).reshape(-1).div(0.1).softmax(-1).cpu().numpy().reshape(h, w), "mean Softmax t=0.1"),
            (resize_tensor(c_feats[idx][fid]).reshape(-1).div(1).softmax(-1).cpu().numpy().reshape(h, w), "mean Softmax t=1"),
            (resize_tensor(c_feats[idx][fid]).reshape(-1).div(4).softmax(-1).cpu().numpy().reshape(h, w), "mean Softmax t=4"),
            (resize_tensor(c_feats[idx][fid]).reshape(-1).div(40).softmax(-1).cpu().numpy().reshape(h, w), "mean Softmax t=40"),
            (resize_tensor(c_feats[idx][fid]).sigmoid().cpu().numpy(), "sigmoid"),
            (resize_tensor(c_feats[idx][fid]).sigmoid().div(0.01).reshape(-1).softmax(-1).cpu().numpy().reshape(h, w), "sigmoid Softmax t=0.01"),
            (resize_tensor(c_feats[idx][fid]).sigmoid().div(0.1).reshape(-1).softmax(-1).cpu().numpy().reshape(h, w), "sigmoid Softmax t=0.1"),
            (resize_tensor(c_feats[idx][fid]).sigmoid().div(1).reshape(-1).softmax(-1).cpu().numpy().reshape(h, w), "sigmoid Softmax t=1"),
            (resize_tensor(c_feats[idx][fid]).sigmoid().div(4).reshape(-1).softmax(-1).cpu().numpy().reshape(h, w), "sigmoid Softmax t=4"),
            (resize_tensor(c_feats[idx][fid]).sigmoid().div(40).reshape(-1).softmax(-1).cpu().numpy().reshape(h, w), "sigmoid Softmax t=40")
        ]
        for i, (img_trans, title) in enumerate(transformations):
            ax = plt.subplot2grid((2, 7), (i // 6, 1 + i % 6))
            plot_image(ax, img_trans, title)
        plt.tight_layout()
        plt.show()
            

class MLN(nn.Module):
    ''' 
    Args:
        c_dim (int): dimension of latent code c
        f_dim (int): feature dimension
    '''

    def __init__(self, c_dim, f_dim=256, use_ln=True):
        super().__init__()
        self.c_dim = c_dim
        self.f_dim = f_dim
        self.use_ln = use_ln

        self.reduce = nn.Sequential(
            nn.Linear(c_dim, f_dim),
            nn.ReLU(),
        )
        self.gamma = nn.Linear(f_dim, f_dim)
        self.beta = nn.Linear(f_dim, f_dim)
        if self.use_ln:
            self.ln = nn.LayerNorm(f_dim, elementwise_affine=False)
        self.init_weight()

    def init_weight(self):
        nn.init.zeros_(self.gamma.weight)
        nn.init.zeros_(self.beta.weight)
        nn.init.ones_(self.gamma.bias)
        nn.init.zeros_(self.beta.bias)

    def forward(self, x, c):
        if self.use_ln:
            x = self.ln(x)
        c = self.reduce(c)
        gamma = self.gamma(c)
        beta = self.beta(c)
        out = gamma * x + beta

        return out