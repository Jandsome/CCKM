# ------------------------------------------------------------------------
# Novel Scenes & Classes: Towards Adaptive Open-set Object Detection
# Modified by Wuyang Li
# ------------------------------------------------------------------------
# Modified by Wei-Jie Huang
# ------------------------------------------------------------------------
# Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------

"""
Deformable DETR model and criterion classes.
"""
import torch
import torch.nn.functional as F
from torch import nn
import math
import numpy as np
import sklearn.cluster as cluster
from util import box_ops
from util.misc import (NestedTensor, nested_tensor_from_tensor_list,
                       accuracy, get_world_size, interpolate,
                       is_dist_avail_and_initialized, inverse_sigmoid)

from .backbone import build_backbone
from .matcher import build_matcher
from .segmentation import (DETRsegm, PostProcessPanoptic, PostProcessSegm,
                           dice_loss, sigmoid_focal_loss)
from .deformable_transformer import build_deforamble_transformer
from .utils import GradientReversal
import copy
from copy import deepcopy

def sim_matrix(a, b, eps=1e-8):
    """
    added eps for numerical stability
    """
    a_n, b_n = a.norm(dim=1)[:, None], b.norm(dim=1)[:, None]
    a_norm = a / torch.clamp(a_n, min=eps)
    b_norm = b / torch.clamp(b_n, min=eps)
    sim_mt = torch.mm(a_norm, b_norm.transpose(0, 1))
    return sim_mt

def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


class DeformableDETR(nn.Module):
    """ This is the Deformable DETR module that performs object detection """
    def __init__(self, backbone, transformer, num_classes, num_queries, num_feature_levels,
                 aux_loss=True, with_box_refine=False, two_stage=False, from_cfg=None):
        """ Initializes the model.
        Parameters:
            backbone: torch module of the backbone to be used. See backbone.py
            transformer: torch module of the transformer architecture. See transformer.py
            num_classes: number of object classes
            num_queries: number of object queries, ie detection slot. This is the maximal number of objects
                         DETR can detect in a single image. For COCO, we recommend 100 queries.
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
            with_box_refine: iterative bounding box refinement
            two_stage: two-stage Deformable DETR
        """
        super().__init__()

        self.from_cfg = from_cfg
        self.num_queries = num_queries
        self.transformer = transformer
        hidden_dim = transformer.d_model
        self.class_embed = nn.Linear(hidden_dim, num_classes)
        self.bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)
        self.num_feature_levels = num_feature_levels
        if not two_stage:
            self.query_embed = nn.Embedding(num_queries, hidden_dim*2)
        if num_feature_levels > 1:
            num_backbone_outs = len(backbone.strides)
            input_proj_list = []
            for _ in range(num_backbone_outs):
                in_channels = backbone.num_channels[_]
                input_proj_list.append(nn.Sequential(
                    nn.Conv2d(in_channels, hidden_dim, kernel_size=1),
                    nn.GroupNorm(32, hidden_dim),
                ))
            for _ in range(num_feature_levels - num_backbone_outs):
                input_proj_list.append(nn.Sequential(
                    nn.Conv2d(in_channels, hidden_dim, kernel_size=3, stride=2, padding=1),
                    nn.GroupNorm(32, hidden_dim),
                ))
                in_channels = hidden_dim
            self.input_proj = nn.ModuleList(input_proj_list)
        else:
            self.input_proj = nn.ModuleList([
                nn.Sequential(
                    nn.Conv2d(backbone.num_channels[0], hidden_dim, kernel_size=1),
                    nn.GroupNorm(32, hidden_dim),
                )])
        self.backbone = backbone
        self.aux_loss = aux_loss
        self.with_box_refine = with_box_refine
        self.two_stage = two_stage
        
        self.da = self.from_cfg['da']
        self.backbone_align = self.from_cfg['backbone_align']
        self.space_align = self.from_cfg['space_align']
        self.channel_align = self.from_cfg['channel_align']
        self.instance_align = self.from_cfg['instance_align']
        
        self.register_buffer('cls_means', torch.zeros(num_classes, 256))
        self.register_buffer('cls_stds', torch.zeros(num_classes, 256))
        self.register_buffer('ext_1', torch.zeros(num_classes, 256))
        self.register_buffer('ext_2', torch.zeros(num_classes, 256))

        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        self.class_embed.bias.data = torch.ones(num_classes) * bias_value
        nn.init.constant_(self.bbox_embed.layers[-1].weight.data, 0)
        nn.init.constant_(self.bbox_embed.layers[-1].bias.data, 0)
        for proj in self.input_proj:
            nn.init.xavier_uniform_(proj[0].weight, gain=1)
            nn.init.constant_(proj[0].bias, 0)

        # if two-stage, the last class_embed and bbox_embed is for region proposal generation
        num_pred = (transformer.decoder.num_layers + 1) if two_stage else transformer.decoder.num_layers

        if with_box_refine:
            self.class_embed = _get_clones(self.class_embed, num_pred)
            self.bbox_embed = _get_clones(self.bbox_embed, num_pred)
            nn.init.constant_(self.bbox_embed[0].layers[-1].bias.data[2:], -2.0)
            # hack implementation for iterative bounding box refinement
            self.transformer.decoder.bbox_embed = self.bbox_embed
        else:
            nn.init.constant_(self.bbox_embed.layers[-1].bias.data[2:], -2.0)
            self.class_embed = nn.ModuleList([self.class_embed for _ in range(num_pred)])
            self.bbox_embed = nn.ModuleList([self.bbox_embed for _ in range(num_pred)])
            self.transformer.decoder.bbox_embed = None
        if two_stage:
            # hack implementation for two-stage
            self.transformer.decoder.class_embed = self.class_embed
            for box_embed in self.bbox_embed:
                nn.init.constant_(box_embed.layers[-1].bias.data[2:], 0.0)
        if self.backbone_align:
            self.grl = GradientReversal(lambda_=self.from_cfg['backbone_adv_lambda'])
            self.backbone_D = MLP(hidden_dim, hidden_dim, 1, 3)
            for layer in self.backbone_D.layers:
                nn.init.xavier_uniform_(layer.weight, gain=1)
                nn.init.constant_(layer.bias, 0)
        if self.space_align:
            self.space_D = MLP(hidden_dim, hidden_dim, 1, 3)
            for layer in self.space_D.layers:
                nn.init.xavier_uniform_(layer.weight, gain=1)
                nn.init.constant_(layer.bias, 0)
        if self.channel_align:
            self.channel_D = MLP(hidden_dim, hidden_dim, 1, 3)
            for layer in self.channel_D.layers:
                nn.init.xavier_uniform_(layer.weight, gain=1)
                nn.init.constant_(layer.bias, 0)
        if self.instance_align:
            self.instance_D = MLP(hidden_dim, hidden_dim, 1, 3)
            for layer in self.instance_D.layers:
                nn.init.xavier_uniform_(layer.weight, gain=1)
                nn.init.constant_(layer.bias, 0)

    def forward(self, samples: NestedTensor, targets=None):
        """ The forward expects a NestedTensor, which consists of:
               - samples.tensor: batched images, of shape [batch_size x 3 x H x W]
               - samples.mask: a binary mask of shape [batch_size x H x W], containing 1 on padded pixels

            It returns a dict with the following elements:
               - "pred_logits": the classification logits (including no-object) for all queries.
                                Shape= [batch_size x num_queries x (num_classes + 1)]
               - "pred_boxes": The normalized boxes coordinates for all queries, represented as
                               (center_x, center_y, height, width). These values are normalized in [0, 1],
                               relative to the size of each individual image (disregarding possible padding).
                               See PostProcess for information on how to retrieve the unnormalized bounding box.
               - "aux_outputs": Optional, only returned when auxilary losses are activated. It is a list of
                                dictionnaries containing the two above keys for each decoder layer.
        """
        out = {}
        if not isinstance(samples, NestedTensor):
            samples = nested_tensor_from_tensor_list(samples)
        features, pos = self.backbone(samples)

        srcs = []
        masks = []
        for l, feat in enumerate(features):
            src, mask = feat.decompose()
            srcs.append(self.input_proj[l](src))
            masks.append(mask)
            assert mask is not None
        if self.num_feature_levels > len(srcs):
            _len_srcs = len(srcs)
            for l in range(_len_srcs, self.num_feature_levels):
                if l == _len_srcs:
                    src = self.input_proj[l](features[-1].tensors)
                else:
                    src = self.input_proj[l](srcs[-1])
                m = samples.mask
                mask = F.interpolate(m[None].float(), size=src.shape[-2:]).to(torch.bool)[0]
                pos_l = self.backbone[1](NestedTensor(src, mask)).to(src.dtype)
                srcs.append(src)
                masks.append(mask)
                pos.append(pos_l)

        query_embeds = None
        if not self.two_stage:
            query_embeds = self.query_embed.weight

        # send to def-transformer 
        hs, init_reference, inter_references, enc_outputs_class, enc_outputs_coord_unact, da_output, memory = self.transformer(srcs, masks, pos, query_embeds)
        # hs: lvl, bs, 100, 256
        
        outputs_classes = []
        outputs_coords = []
        for lvl in range(hs.shape[0]):
            if lvl == 0:
                reference = init_reference
            else:
                reference = inter_references[lvl - 1]
            reference = inverse_sigmoid(reference)
            outputs_class = self.class_embed[lvl](hs[lvl])
            tmp = self.bbox_embed[lvl](hs[lvl])
            if reference.shape[-1] == 4:
                tmp += reference
            else:
                assert reference.shape[-1] == 2
                tmp[..., :2] += reference
            outputs_coord = tmp.sigmoid()
            outputs_classes.append(outputs_class)
            outputs_coords.append(outputs_coord)
        outputs_class = torch.stack(outputs_classes)
        outputs_coord = torch.stack(outputs_coords)

        out['pred_logits_both'] = outputs_class[-1]
        out['is_training'] = self.training

        out['cls_means'] = self.cls_means
        out['cls_stds'] = self.cls_stds
        out['ext_1'] = self.ext_1  
        out['ext_2'] = self.ext_2
        out['final_classifier'] = self.class_embed[-1]
        out['first_classifier'] = self.class_embed[0]

        if self.training and self.da:
            B = outputs_class.shape[1]
            outputs_class = outputs_class[:, :B//2]
            outputs_coord = outputs_coord[:, :B//2]
            if self.two_stage:
                enc_outputs_class = enc_outputs_class[:B//2]
                enc_outputs_coord_unact = enc_outputs_coord_unact[:B//2]
            if self.backbone_align:
                da_output['backbone'] = torch.cat([self.backbone_D(self.grl(src.flatten(2).transpose(1, 2))) for src in srcs], dim=1)
            if self.space_align:
                da_output['space_query'] = self.space_D(da_output['space_query'])
            if self.channel_align:
                da_output['channel_query'] = self.channel_D(da_output['channel_query'])
            if self.instance_align:
                da_output['instance_query'] = self.instance_D(da_output['instance_query'])


        out.update({'pred_logits': outputs_class[-1], 'pred_boxes': outputs_coord[-1], 'object_embedding': hs[-1], 'first_embedding': hs[0]})
        if self.aux_loss:
            out['aux_outputs'] = self._set_aux_loss(outputs_class, outputs_coord)

        if self.two_stage:
            enc_outputs_coord = enc_outputs_coord_unact.sigmoid()
            out['enc_outputs'] = {'pred_logits': enc_outputs_class, 'pred_boxes': enc_outputs_coord}
        if self.training and self.da:
            out['da_output'] = da_output
        return out

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_coord):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        return [{'pred_logits': a, 'pred_boxes': b}
                for a, b in zip(outputs_class[:-1], outputs_coord[:-1])]


class SetCriterion(nn.Module):
    """ This class computes the loss for DETR.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """
    def __init__(self, num_classes, matcher, weight_dict, losses, focal_alpha=0.25, da_gamma=2, from_cfg = None):
        """ Create the criterion.
        Parameters:
            num_classes: number of object categories, omitting the special no-object category
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            losses: list of all the losses to be applied. See get_loss for list of available losses.
            focal_alpha: alpha in Focal Loss
        """
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.losses = losses
        self.focal_alpha = focal_alpha
        self.da_gamma = da_gamma
        self.from_cfg = from_cfg
        self.unk_prob = from_cfg['unk_prob']
        self.bce_loss = nn.BCELoss()
        self.pretrain_th = from_cfg['pretrain_th']
        self.std_scaling = from_cfg['std_scaling']
        self.alpha = from_cfg['alpha']
        self.with_openset = from_cfg['with_openset']

    def loss_labels(self, outputs, targets, indices, num_boxes, log=True):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        assert 'pred_logits' in outputs
        src_logits = outputs['pred_logits']

        idx = self._get_src_permutation_idx(indices)

        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(src_logits.shape[:2], self.num_classes,
                                    dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o

        target_classes_onehot = torch.zeros([src_logits.shape[0], src_logits.shape[1], src_logits.shape[2] + 1],
                                            dtype=src_logits.dtype, layout=src_logits.layout, device=src_logits.device)
        target_classes_onehot.scatter_(2, target_classes.unsqueeze(-1), 1)
        target_classes_onehot = target_classes_onehot[:,:,:-1]

        if self.unk_prob > 0 and self.with_openset:
            obj_idx = target_classes_onehot.sum(-1) > 0
            tmp = target_classes_onehot[obj_idx]   
            tmp[:,-1] = self.unk_prob
            target_classes_onehot[obj_idx] = tmp

        loss_ce = sigmoid_focal_loss(src_logits, target_classes_onehot, num_boxes, alpha=self.focal_alpha, gamma=2) * src_logits.shape[1]
        losses = {'loss_ce': loss_ce}

        if log:
            # TODO this should probably be a separate loss, not hacked in this one here
            losses['class_error'] = 100 - accuracy(src_logits[idx], target_classes_o)[0]
        return losses

    @torch.no_grad()
    def loss_cardinality(self, outputs, targets, indices, num_boxes):
        """ Compute the cardinality error, ie the absolute error in the number of predicted non-empty boxes
        This is not really a loss, it is intended for logging purposes only. It doesn't propagate gradients
        """
        pred_logits = outputs['pred_logits']
        device = pred_logits.device
        tgt_lengths = torch.as_tensor([len(v["labels"]) for v in targets], device=device)
        # Count the number of predictions that are NOT "no-object" (which is the last class)
        card_pred = (pred_logits.argmax(-1) != pred_logits.shape[-1] - 1).sum(1)
        card_err = F.l1_loss(card_pred.float(), tgt_lengths.float())
        losses = {'cardinality_error': card_err}
        return losses

    def loss_boxes(self, outputs, targets, indices, num_boxes):
        """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
           targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
           The target boxes are expected in format (center_x, center_y, h, w), normalized by the image size.
        """
        assert 'pred_boxes' in outputs
        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs['pred_boxes'][idx]
        target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)

        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='none')

        losses = {}
        losses['loss_bbox'] = loss_bbox.sum() / num_boxes

        loss_giou = 1 - torch.diag(box_ops.generalized_box_iou(
            box_ops.box_cxcywh_to_xyxy(src_boxes),
            box_ops.box_cxcywh_to_xyxy(target_boxes)))
        losses['loss_giou'] = loss_giou.sum() / num_boxes
        return losses

    def loss_masks(self, outputs, targets, indices, num_boxes):
        """Compute the losses related to the masks: the focal loss and the dice loss.
           targets dicts must contain the key "masks" containing a tensor of dim [nb_target_boxes, h, w]
        """
        assert "pred_masks" in outputs

        src_idx = self._get_src_permutation_idx(indices)
        tgt_idx = self._get_tgt_permutation_idx(indices)

        src_masks = outputs["pred_masks"]

        # TODO use valid to mask invalid areas due to padding in loss
        target_masks, valid = nested_tensor_from_tensor_list([t["masks"] for t in targets]).decompose()
        target_masks = target_masks.to(src_masks)

        src_masks = src_masks[src_idx]
        # upsample predictions to the target size
        src_masks = interpolate(src_masks[:, None], size=target_masks.shape[-2:],
                                mode="bilinear", align_corners=False)
        src_masks = src_masks[:, 0].flatten(1)

        target_masks = target_masks[tgt_idx].flatten(1)

        losses = {
            "loss_mask": sigmoid_focal_loss(src_masks, target_masks, num_boxes),
            "loss_dice": dice_loss(src_masks, target_masks, num_boxes),
        }
        return losses

    def loss_da(self, outputs, use_focal=False):

        B = outputs.shape[0]
        assert B % 2 == 0

        targets = torch.empty_like(outputs)
        targets[:B//2] = 0
        targets[B//2:] = 1

        loss = F.binary_cross_entropy_with_logits(outputs, targets, reduction='none')
        if use_focal:
            prob = outputs.sigmoid()
            p_t = prob * targets + (1 - prob) * (1 - targets)
            loss = loss * ((1 - p_t) ** self.da_gamma)

        return loss.mean()

    def loss_openset(self, outputs, indices, targets):
        
        ctrs = outputs['cls_means'][:-1]
        obj_emb = outputs['object_embedding'] # bs, 100, 256
        
        ctrs_labels = torch.arange(self.num_classes-1).to(ctrs.device)
        # mtch_idx = self._get_src_permutation_idx(indices) # bs, idx
        unmtch_idx = self._get_src_unmatched_permutation_idx(indices, num_query=100)
        unmtch_emb = obj_emb[unmtch_idx]

        pair_dis = self.eu_dis(ctrs, ctrs)
        top_k_idx = torch.sort(pair_dis, descending=True, dim=-1)[1][:,0] # k far nei

        ctrs_1 = ctrs
        ctrs_2 = ctrs_1[top_k_idx]
    
        ctrs_1_labels = ctrs_labels
        ctrs_2_labels = ctrs_labels[top_k_idx]

        motif_embeds_list = []
        angle_list = []
        calss_1 = []
        calss_2 = []

        motif_records = []

        center_point = (ctrs_1 + ctrs_1)/2

        # dis_base = (ctrs_2 - ctrs_1).norm(dim=-1)
        # L = beta * dis_base
        # offset_magnitude = torch.sqrt(L ** 2 - (dis_base / 2) ** 2)
        for i in range(len(unmtch_emb)):

            vct1 = (ctrs_1 - unmtch_emb[i])
            vct2 = (ctrs_2 - unmtch_emb[i])
            vct3 = (center_point - unmtch_emb[i])

            dis1 = (ctrs_1 - unmtch_emb[i]).norm(dim=-1)
            dis2 = (ctrs_2 - unmtch_emb[i]).norm(dim=-1)
            dis3 = (center_point - unmtch_emb[i]).norm(dim=-1)
            delta_dis = (dis1-dis2).abs()
            dis_base = (ctrs_1 - ctrs_2).norm(dim=-1)
            dis4 = (dis3 - dis_base / 3).abs()  
            angle = dis4 / dis_base + delta_dis / dis_base  



            motif_idx = angle.argmin()
            angle_list.append(angle.min().unsqueeze(dim=0))

            min_angle = angle.min() 

            motif_emb = torch.stack([ctrs_1[motif_idx], ctrs_2[motif_idx], unmtch_emb[i]], dim=0)
            calss_1.append(ctrs_1_labels[motif_idx].unsqueeze(0))
            calss_2.append(ctrs_2_labels[motif_idx].unsqueeze(0))

            motif_embeds_list.append(motif_emb.mean(dim=0)[None,:])

            motif_mean = motif_emb.mean(dim=0)

            record = {
                "motif_mean": motif_mean,
                "unmtch_emb": unmtch_emb[i],
                "angle": min_angle,
            }
            motif_records.append(record)

        angles = torch.tensor([r["angle"] for r in motif_records])  # (N,)
        neg_angles = -angles

        # 按neg_angles选取top-k
        select_idx_1 = neg_angles.topk(self.from_cfg['os_KNN'])[1]
        top_k_records = [motif_records[i] for i in select_idx_1]

        # 整理输出
        motif_embeds_topk_1 = torch.stack([r["motif_mean"] for r in top_k_records])
        unmtch_embs_topk = torch.stack([r["unmtch_emb"] for r in top_k_records])


        motif_embeds = torch.cat(motif_embeds_list,dim=0)
        neg_angles = -torch.cat(angle_list)

        assert motif_embeds.size(0) == neg_angles.size(0)

        select_idx = neg_angles.topk(self.from_cfg['os_KNN'])[1]

        calss_1 = torch.cat(calss_1)[select_idx].unsqueeze(-1)
        calss_2 = torch.cat(calss_2)[select_idx].unsqueeze(-1)

        motif_embeds_topk = motif_embeds[select_idx]

        classifier = outputs['final_classifier']
        #motif_prob = classifier(motif_embeds_topk).sigmoid() #optimal
        motif_prob = classifier(unmtch_embs_topk).sigmoid()
        
        target = torch.full_like(motif_prob, 0.0).detach()
        target[:,-1]=1.0

        loss = self.bce_loss(motif_prob, target)

        # update memory bank
        with torch.no_grad():
            ctrs = outputs['cls_means']
            stds = outputs['cls_stds']

            ema = self.alpha
            #avg_emb_base = motif_embeds_topk.mean(0)
            avg_emb_base = unmtch_embs_topk.mean(0)
            sim_ctrs = F.cosine_similarity(ctrs[-1], avg_emb_base, dim=0)
            ctrs[-1] = (1. - 0.005 * (sim_ctrs + 1)) * ctrs[-1] + 0.005 * (sim_ctrs + 1) * avg_emb_base

            #std_emb_base = motif_embeds_topk.std(0)
            std_emb_base = unmtch_embs_topk.std(0)
            sim_stds = F.cosine_similarity(stds[-1], std_emb_base, dim=0)
            stds[-1] = (1. - 0.005 * (sim_stds + 1)) * stds[-1] + 0.005 * (sim_stds + 1) * std_emb_base

            outputs['cls_means'] = ctrs
            outputs['cls_stds'] = stds

        return loss

    def loss_crossdomain(self, outputs, targets, indices):
        q_embs = outputs['object_embedding'] # bs, 100, 256 # source: 0: bs//2; target: bs//2: 

        B = q_embs.shape[0]
        assert B % 2 == 0

        q_tg_pred = outputs['pred_logits_both'][B//2:]
        q_tg_scores = q_tg_pred.view(-1, q_tg_pred.size(-1)).sigmoid()

        ctrs = outputs['cls_means']
        stds = outputs['cls_stds']
        a = outputs['ext_1'] 
        b = outputs['ext_2']

        ctrs_labels = torch.arange(self.num_classes).to(ctrs.device)


        # add centers
        ctrs_1 = torch.cat([a, b], dim=0)
        ctrs_1_labels = torch.cat([ctrs_labels, ctrs_labels])
        ctrs_2 = torch.cat([b, a], dim=0)
        center_point = (ctrs_1 +ctrs_2)/2
        dis_base = (ctrs_1 - ctrs_2).norm(dim=-1)


        q_tg_raw = q_embs[B//2:].view(-1, q_embs.size(-1))
        # score_mask = q_tg_scores.max(-1)[0] > self.pretrain_th
        score_mask = q_tg_scores.sum(-1) > self.pretrain_th
        q_tg = q_tg_raw[score_mask]

        if len(q_tg)< self.from_cfg['da_KNN']:
            return q_tg_scores.sum()*0

        sr_label = []
        motif_embeds_list =[]
        q_act = []
        for i in range(len(q_tg)):        
            vct1 = (ctrs_1 - q_tg[i]) 
            vct2 = (ctrs_2 - q_tg[i])
            dis1 = (center_point - q_tg[i]).norm(dim=-1)
            angle = dis1 / dis_base
            motif_idx = angle.argmin(-1)

            ctr_1 = ctrs_1[motif_idx]
            ctr_2 = ctrs_2[motif_idx]

            motif_emb = torch.stack([ctr_1, q_tg[i], ctr_2], dim=0)
            sr_label.append(ctrs_1_labels[motif_idx].unsqueeze(dim=0))
            motif_embeds_list.append(motif_emb.mean(dim=0)[None,:])

            if ctrs_1_labels[motif_idx].item() == 0:
                q_act.append(q_tg)


        motif_embeds = torch.cat(motif_embeds_list, dim=0)
        sr_label = torch.cat(sr_label)
        tg_label = self.eu_dis(q_tg, ctrs).argmin(-1)

        if len(q_act) != 0:
            q_act = torch.cat(q_act)
            with torch.no_grad():
                ctrs = outputs['cls_means']
                stds = outputs['cls_stds']

                avg_emb_base = q_act.mean(0)
                sim_ctrs = F.cosine_similarity(ctrs[-1], avg_emb_base, dim=0)
                ctrs[-1] = (1. - 0.005 * (sim_ctrs + 1)) * ctrs[-1] + 0.005 * (sim_ctrs + 1) * avg_emb_base
                std_emb_base = q_act.std(0)
                sim_stds = F.cosine_similarity(stds[-1], std_emb_base, dim=0)
                stds[-1] = (1. - 0.005 * (sim_stds + 1)) * stds[-1] + 0.005 * (sim_stds + 1) * std_emb_base
                outputs['cls_means'] = ctrs
                outputs['cls_stds'] = stds


        prob = outputs['final_classifier'](motif_embeds)
        target_motif = torch.zeros(prob.size()).to(prob.device)
        prob_tmp = 0.5
        tg = torch.full_like(sr_label[:, None].float(), prob_tmp)
        target_motif.scatter_(1, sr_label[:, None], tg)
        target_motif.scatter_(1, tg_label[:, None], tg)
        target_motif[target_motif.sum(-1) == 0.5] *= 2

        loss = self.bce_loss(prob.sigmoid(), target_motif.detach())


        return loss
    
    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_src_unmatched_permutation_idx(self, indices, num_query=100):
        # permute predictions following indices
        bs = len(indices)
        queries = torch.arange(num_query)
        batch_idx = []
        src_idx = []
        for  i, (src, _) in enumerate(indices):
            combined = torch.cat(
                (queries, src))  
            uniques, counts = combined.unique(return_counts=True)
            unmatched_box = uniques[counts == 1]
            batch_idx.append(torch.full_like(unmatched_box, i))
            src_idx.append(unmatched_box)
        batch_idx = torch.cat(batch_idx)
        src_idx = torch.cat(src_idx)
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def eu_dis(self, a,b,p=2):
        return torch.norm(a[:,None]-b,dim=2,p=p)

    def get_loss(self, loss, outputs, targets, indices, num_boxes, **kwargs):
        loss_map = {
            'labels': self.loss_labels,
            'cardinality': self.loss_cardinality,
            'boxes': self.loss_boxes,
            'masks': self.loss_masks
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, indices, num_boxes, **kwargs)

    @torch.no_grad()
    def update_class_centers(self, outputs, targets, indices):

        ctrs = outputs['cls_means']
        stds = outputs['cls_stds']
        ext1 = outputs['ext_1']  
        ext2 = outputs['ext_2']
        q_embs = outputs['object_embedding']  # bs, 100, 256   object_embedding是Deformable DETR最后一个decoder的输出
        matched_idx = self._get_src_permutation_idx(indices)  # bs, idx
        matched_q = q_embs[matched_idx]  # 得到匹配到的proposal的向量空间上的坐标
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in
                                      zip(targets, indices)])  # 根据indices中GT的标号在GT中寻找相应的label，形成一个与matched_q对应的label向量
        k = 20
        scaling_factor = ctrs.new_ones(ctrs.size(0)) * self.std_scaling
        scaling_factor = scaling_factor[:, None]

        for i in target_classes_o.unique():
            per_cls_q = matched_q[target_classes_o == i]  # 提取所有属于当前类的matched_q
            bs_ctr = matched_q[target_classes_o == i].detach()  # ext更新
            avg_emb = per_cls_q.mean(dim=0)  # 计算属于当前类的matched_q的中心点
            # ctrs[i] = (1. - ema) * ctrs[i] + ema * avg_emb.detach()     #根据本轮计算的matched_q的中心点进行中心点的更新，其中avg_emb不参与反向传播

            if per_cls_q.size(0) > 2:
                std_emb = per_cls_q.std(dim=0)
                sim_stds = F.cosine_similarity(stds[i], std_emb, dim=0)
                stds[i] = (1. - 0.005 * (sim_stds + 1)) * stds[i] + 0.005 * (sim_stds + 1) * std_emb.detach()

            if len(bs_ctr) > k:
                sp = cluster.SpectralClustering(3, affinity='nearest_neighbors', n_jobs=-1, assign_labels='kmeans',
                                                random_state=3141, n_neighbors=len(bs_ctr) // 3)
                # mid = F.cosine_similarity(avg_emb, ctrs[i], dim=0)
                # seed_ctrs = (1. - 0.005 * (mid + 1)) * ctrs[i] + 0.005 * (mid + 1) * avg_emb.detach()
                seed_ctrs = ctrs[i]
                indx = sp.fit_predict(torch.cat([seed_ctrs[None, :], bs_ctr]).cpu().numpy())
                indx_ext1, indx_ext2 = self.ext_clusters_indx(indx)
                indx_ctr = (indx == indx[0])[1:]  # ext更新
                bs_ext1 = bs_ctr[indx_ext1].mean(0)
                bs_ext2 = bs_ctr[indx_ext2].mean(0)
                bs_ctr = bs_ctr[indx_ctr].mean(0)
            else:
                bs_ctr = avg_emb
                bs_ext1 = ctrs[i] + scaling_factor[i] * stds[i]
                bs_ext2 = ctrs[i] - scaling_factor[i] * stds[i]

            momentum = F.cosine_similarity(bs_ctr, ctrs[i], dim=0)
            ctrs[i] = 0.005 * (momentum + 1) * ctrs[i] + (1. - 0.005 * (momentum + 1)) * bs_ctr

            ext1[i] = bs_ext1
            ext2[i] = bs_ext2

        avg_emb_base = ctrs[:-1].mean(0)
        sim_ctr = F.cosine_similarity(ctrs[-1], avg_emb_base, dim=0)
        ctrs[-1] = (1. - 0.005 * (sim_ctr + 1)) * ctrs[-1] + 0.005 * (sim_ctr + 1) * avg_emb_base

        std_emb_base = stds[:-1].mean(0)
        sim_std = F.cosine_similarity(stds[-1], std_emb_base, dim=0)
        stds[-1] = (1. - 0.005 * (sim_std + 1)) * stds[-1] + 0.005 * (sim_std + 1) * std_emb_base

        ext1[-1] = ctrs[-1] + scaling_factor[-1] * stds[-1]
        ext2[-1] = ctrs[-1] - scaling_factor[-1] * stds[-1]

        outputs['cls_means'] = ctrs
        outputs['cls_stds'] = stds
        outputs['ext_1'] = ext1  
        outputs['ext_2'] = ext2

        return outputs



    def ext_clusters_indx(self, indx):

        indx1 = np.zeros_like(indx, dtype=bool)  
        indx2 = np.zeros_like(indx, dtype=bool)  # 初始化 tensor1 和 tensor2，与原 indx 同样长度，初始为 False

        # 记录第一个与 indx[0] 不同的值
        temp = None

        # 第一次遍历，找到第一个与 indx[0] 不同的值
        for j in range(indx.size):
            if indx[j] != indx[0] and temp is None:
                temp = indx[j]
                indx1[j] = True  # 设置 tensor1 中当前元素为 True
                break

        # 如果找到了 temp，则进行第二次遍历
        if temp is not None:
            for j in range(indx.size):
                if indx[j] != indx[0]:
                    if indx[j] == temp:
                        indx1[j] = True  # 与 temp 相同的值在 tensor1 中为 True
                    indx2[j] = ~indx1[j]

        indx1 = indx1[1:]
        indx2 = indx2[1:]

        return indx1, indx2

    def forward(self, samples, outputs, targets, epoch=0):
        """ This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        outputs_without_aux = {k: v for k, v in outputs.items() if k != 'aux_outputs' and k != 'enc_outputs'}
        # Compute all the requested losses
        losses = {}
        # Retrieve the matching between the outputs of the last layer and the targets
        indices = self.matcher(outputs_without_aux, targets)
        if self.training:
            #outputs = self.update_class_centers(outputs, targets, indices, ema=self.alpha)
            outputs = self.update_class_centers(outputs, targets, indices)
        # Compute the average number of target boxes accross all nodes, for normalization purposes
        num_boxes = sum(len(t["labels"]) for t in targets)
        num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=next(iter(outputs.values())).device)

        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_boxes)
        num_boxes = torch.clamp(num_boxes / get_world_size(), min=1).item()


        for loss in self.losses:
            kwargs = {}
            losses.update(self.get_loss(loss, outputs, targets, indices, num_boxes, **kwargs))

        if self.training and self.from_cfg['with_openset'] and epoch > self.from_cfg['warm_up_epoch']:
        #if self.training and self.from_cfg['with_openset']:
            losses['loss_openset'] = self.loss_openset(outputs, indices, targets)

        if self.training and self.from_cfg['with_crossdomain'] and epoch > self.from_cfg['warm_up_epoch']:
        #if self.training and self.from_cfg['with_crossdomain']:
            losses['loss_crossdomain'] = self.loss_crossdomain(outputs, targets, indices)  

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if 'aux_outputs' in outputs:
            for i, aux_outputs in enumerate(outputs['aux_outputs']):
                indices = self.matcher(aux_outputs, targets)
                for loss in self.losses:
                    if loss == 'masks':
                        # Intermediate masks losses are too costly to compute, we ignore them.
                        continue
                    kwargs = {}
                    if loss == 'labels':
                        # Logging is enabled only for the last layer
                        kwargs['log'] = False
       
                    l_dict = self.get_loss(loss, aux_outputs, targets, indices, num_boxes, **kwargs)
                    l_dict = {k + f'_{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)

        if 'enc_outputs' in outputs:
            enc_outputs = outputs['enc_outputs']
            bin_targets = copy.deepcopy(targets)
            for bt in bin_targets:
                bt['labels'] = torch.zeros_like(bt['labels'])
            indices = self.matcher(enc_outputs, bin_targets)
            for loss in self.losses:
                if loss == 'masks':
                    # Intermediate masks losses are too costly to compute, we ignore them.
                    continue
                kwargs = {}
                if loss == 'labels':
                    # Logging is enabled only for the last layer
                    kwargs['log'] = False
                l_dict = self.get_loss(loss, enc_outputs, bin_targets, indices, num_boxes, **kwargs)
                l_dict = {k + f'_enc': v for k, v in l_dict.items()}
                losses.update(l_dict)
                
        if 'da_output' in outputs:
            for k, v in outputs['da_output'].items():
                losses[f'loss_{k}'] = self.loss_da(v, use_focal='query' in k)

        return losses


class PostProcess(nn.Module):
    """ This module converts the model's output into the format expected by the coco api"""

    @torch.no_grad()
    def forward(self, outputs, target_sizes, show_box=False):
        """ Perform the computation
        Parameters:
            outputs: raw outputs of the model
            target_sizes: tensor of dimension [batch_size x 2] containing the size of each images of the batch
                          For evaluation, this must be the original image size (before any data augmentation)
                          For visualization, this should be the image size after data augment, but before padding
        """
        out_logits, out_bbox = outputs['pred_logits'], outputs['pred_boxes']

        assert len(out_logits) == len(target_sizes)
        assert target_sizes.shape[1] == 2
    
        prob = out_logits.sigmoid()
        if show_box:
            # for qualitative visualization to surpress unk preds 
            #TODO may be different from the old implementation, need to check
            bs, num_q, num_class = prob.size()
            unk_mask = prob.argmax(-1) != num_class - 1
            prob[unk_mask] = 0.0

        topk_values, topk_indexes = torch.topk(prob.view(out_logits.shape[0], -1), 100, dim=1)
        scores = topk_values
        topk_boxes = topk_indexes // out_logits.shape[2]
        labels = topk_indexes % out_logits.shape[2]
        boxes = box_ops.box_cxcywh_to_xyxy(out_bbox)
        boxes = torch.gather(boxes, 1, topk_boxes.unsqueeze(-1).repeat(1,1,4))

        # and from relative [0, 1] to absolute [0, height] coordinates
        img_h, img_w = target_sizes.unbind(1)
        scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1)
        boxes = boxes * scale_fct[:, None, :]
        results = [{'scores': s, 'labels': l, 'boxes': b} for s, l, b in zip(scores, labels, boxes)]
        return results

class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x

def build(cfg):

    device = torch.device(cfg.DEVICE)
    backbone = build_backbone(cfg)
    transformer = build_deforamble_transformer(cfg)

    from_cfg = dict(
        backbone_align=cfg.MODEL.BACKBONE_ALIGN,
        space_align=cfg.MODEL.SPACE_ALIGN,
        channel_align=cfg.MODEL.CHANNEL_ALIGN,
        instance_align=cfg.MODEL.INSTANCE_ALIGN,
        da=cfg.DATASET.DA_MODE == 'uda' or cfg.DATASET.DA_MODE == 'aood',
        batch_size=cfg.TRAIN.BATCH_SIZE,
        with_openset=cfg.AOOD.OPEN_SET.MOTIF_ON,
        os_KNN=cfg.AOOD.OPEN_SET.KNN,
        pretrain_th=cfg.AOOD.OPEN_SET.TH,
        with_crossdomain=cfg.AOOD.CROSS_DOMAIN.MOTIF_ON,
        da_KNN=cfg.AOOD.CROSS_DOMAIN.KNN,
        unk_prob=cfg.AOOD.OPEN_SET.UNK_PROB,
        backbone_adv_lambda=cfg.AOOD.CROSS_DOMAIN.BACKBONE_LAMBDA,
        warm_up_epoch=cfg.AOOD.OPEN_SET.WARM_UP,
        std_scaling=cfg.AOOD.CROSS_DOMAIN.BETA,
        motif_update=cfg.AOOD.OPEN_SET.MOTIF_UPDATE,
        alpha=cfg.AOOD.OPEN_SET.ALPHA,
    )
    print(from_cfg)

    model = DeformableDETR(
        backbone,
        transformer,
        num_classes=cfg.DATASET.NUM_CLASSES,
        num_queries=cfg.MODEL.NUM_QUERIES,
        num_feature_levels=cfg.MODEL.NUM_FEATURE_LEVELS,
        aux_loss=cfg.LOSS.AUX_LOSS,
        with_box_refine=cfg.MODEL.WITH_BOX_REFINE,
        two_stage=cfg.MODEL.TWO_STAGE,
        from_cfg = from_cfg,
    )
    if cfg.MODEL.MASKS:
        model = DETRsegm(model, freeze_detr=(cfg.MODEL.FROZEN_WEIGHTS is not None))
    matcher = build_matcher(cfg)
    weight_dict = {'loss_ce': cfg.LOSS.CLS_LOSS_COEF, 'loss_bbox': cfg.LOSS.BBOX_LOSS_COEF}
    weight_dict['loss_giou'] = cfg.LOSS.GIOU_LOSS_COEF
    if cfg.MODEL.MASKS:
        weight_dict["loss_mask"] = cfg.LOSS.MASK_LOSS_COEF
        weight_dict["loss_dice"] = cfg.LOSS.DICE_LOSS_COEF
    # TODO this is a hack
    if cfg.LOSS.AUX_LOSS:
        aux_weight_dict = {}
        for i in range(cfg.MODEL.DEC_LAYERS - 1):
            aux_weight_dict.update({k + f'_{i}': v for k, v in weight_dict.items()})
        aux_weight_dict.update({k + f'_enc': v for k, v in weight_dict.items()})
        weight_dict.update(aux_weight_dict)

    weight_dict['loss_backbone'] = cfg.LOSS.BACKBONE_LOSS_COEF
    weight_dict['loss_space_query'] = cfg.LOSS.SPACE_QUERY_LOSS_COEF
    weight_dict['loss_channel_query'] = cfg.LOSS.CHANNEL_QUERY_LOSS_COEF
    weight_dict['loss_instance_query'] = cfg.LOSS.INSTANCE_QUERY_LOSS_COEF

    weight_dict['loss_crossdomain'] = cfg.AOOD.CROSS_DOMAIN.MOTIF_LOSS_COEF
    weight_dict['loss_openset'] = cfg.AOOD.OPEN_SET.MOTIF_LOSS_COEF

    losses = ['labels', 'boxes']
    if cfg.MODEL.MASKS:
        losses += ["masks"]
    # num_classes, matcher, weight_dict, losses, focal_alpha=0.25
    criterion = SetCriterion(
        cfg.DATASET.NUM_CLASSES,
        matcher,
        weight_dict,
        losses,
        focal_alpha=cfg.LOSS.FOCAL_ALPHA,
        da_gamma=cfg.LOSS.DA_GAMMA,
        from_cfg=from_cfg,

    )
    criterion.to(device)
    postprocessors = {'bbox': PostProcess()}
    if cfg.MODEL.MASKS:
        postprocessors['segm'] = PostProcessSegm()
        if cfg.DATASET.DATASET_FILE == "coco_panoptic":
            is_thing_map = {i: i <= 90 for i in range(201)}
            postprocessors["panoptic"] = PostProcessPanoptic(is_thing_map, threshold=0.85)

    return model, criterion, postprocessors
