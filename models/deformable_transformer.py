import copy
from typing import Optional, List
import math

import torch
import torch.nn.functional as F
from torch import nn, Tensor
from torch.nn.init import xavier_uniform_, constant_, uniform_, normal_
from models.ops.modules import MSDeformAttn
from models.utils import GradientReversal


class DeformableTransformer(nn.Module):
    def __init__(self, d_model=256, nhead=8, num_encoder_layers=6,
                dropout=0.1, activation="relu",num_feature_levels=4, enc_n_points=4, use_grl=False):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead

        encoder_layer = DeformableTransformerEncoderLayer(d_model, dropout, activation, num_feature_levels, nhead, enc_n_points, use_grl)

        self.encoder = DeformableTransformerEncoder(encoder_layer, num_encoder_layers)

        self.level_embed = nn.Parameter(torch.Tensor(num_feature_levels, d_model))

        self._reset_parameters()


    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        for m in self.modules():
            if isinstance(m, MSDeformAttn):
                m._reset_parameters()
        normal_(self.level_embed)

    def get_proposal_pos_embed(self, proposals):
        num_pos_feats = 128
        temperature = 10000
        scale = 2 * math.pi

        dim_t = torch.arange(num_pos_feats, dtype=torch.float32, device=proposals.device)
        dim_t = temperature ** (2 * (dim_t // 2) / num_pos_feats)
        # N, L, 4
        proposals = proposals.sigmoid() * scale
        # N, L, 4, 128
        pos = proposals[:, :, :, None] / dim_t
        # N, L, 4, 64, 2
        pos = torch.stack((pos[:, :, :, 0::2].sin(), pos[:, :, :, 1::2].cos()), dim=4).flatten(2)
        return pos

    def gen_encoder_output_proposals(self, memory, memory_padding_mask, spatial_shapes):
        N_, S_, C_ = memory.shape
        base_scale = 4.0
        proposals = []
        _cur = 0
        for lvl, (H_, W_) in enumerate(spatial_shapes):
            mask_flatten_ = memory_padding_mask[:, _cur:(_cur + H_ * W_)].view(N_, H_, W_, 1)
            valid_H = torch.sum(~mask_flatten_[:, :, 0, 0], 1)
            valid_W = torch.sum(~mask_flatten_[:, 0, :, 0], 1)

            grid_y, grid_x = torch.meshgrid(torch.linspace(0, H_ - 1, H_, dtype=torch.float32, device=memory.device),
                                            torch.linspace(0, W_ - 1, W_, dtype=torch.float32, device=memory.device))
            grid = torch.cat([grid_x.unsqueeze(-1), grid_y.unsqueeze(-1)], -1)

            scale = torch.cat([valid_W.unsqueeze(-1), valid_H.unsqueeze(-1)], 1).view(N_, 1, 1, 2)
            grid = (grid.unsqueeze(0).expand(N_, -1, -1, -1) + 0.5) / scale
            wh = torch.ones_like(grid) * 0.05 * (2.0 ** lvl)
            proposal = torch.cat((grid, wh), -1).view(N_, -1, 4)
            proposals.append(proposal)
            _cur += (H_ * W_)
        output_proposals = torch.cat(proposals, 1)
        output_proposals_valid = ((output_proposals > 0.01) & (output_proposals < 0.99)).all(-1, keepdim=True)
        output_proposals = torch.log(output_proposals / (1 - output_proposals))
        output_proposals = output_proposals.masked_fill(memory_padding_mask.unsqueeze(-1), float('inf'))
        output_proposals = output_proposals.masked_fill(~output_proposals_valid, float('inf'))

        output_memory = memory
        output_memory = output_memory.masked_fill(memory_padding_mask.unsqueeze(-1), float(0))
        output_memory = output_memory.masked_fill(~output_proposals_valid, float(0))
        output_memory = self.enc_output_norm(self.enc_output(output_memory))
        return output_memory, output_proposals

    def get_valid_ratio(self, mask):
        _, H, W = mask.shape
        valid_H = torch.sum(~mask[:, :, 0], 1)
        valid_W = torch.sum(~mask[:, 0, :], 1)
        valid_ratio_h = valid_H.float() / H
        valid_ratio_w = valid_W.float() / W
        valid_ratio = torch.stack([valid_ratio_w, valid_ratio_h], -1)
        return valid_ratio

    def forward(self, features, pos_embed):
        """
        Input:
            - srcs: List([bs, c, h, w])
        """

        # prepare input for encoder



        src_flatten = []
        mask_flatten = []
        lvl_pos_embed_flatten = []
        spatial_shapes = []
        masks = []

        bs, c, h, w = features.shape
        spatial_shape = (h, w)
        spatial_shapes.append(spatial_shape)

        features = features.flatten(2).transpose(1, 2)
        pos_embed = pos_embed.flatten(2).transpose(1, 2)
        mask = torch.zeros((bs, h, w), dtype=torch.bool, device=features.device)
        masks.append(mask)
        mask = mask.flatten(1)
        lvl_pos_embed = pos_embed + self.level_embed.view(1, 1, -1)  # 后面那部分可以更新参数，前面的是backbone得到的，不能更新参数
        lvl_pos_embed_flatten.append(lvl_pos_embed)
        src_flatten.append(features)
        mask_flatten.append(mask)

        # for lvl, (src, pos_embed) in enumerate(zip(features, pos_embeds)):
        #     bs, c, h, w = src.shape
        #     spatial_shape = (h, w)
        #     spatial_shapes.append(spatial_shape)
        #
        #     src = src.flatten(2).transpose(1, 2)  # bs, hw, c
        #     pos_embed = pos_embed.flatten(2).transpose(1, 2)  # bs, hw, c
        #     mask = torch.zeros((bs, h, w), dtype=torch.bool, device=src.device)
        #     masks.append(mask)
        #     mask = mask.flatten(1)
        #     lvl_pos_embed = pos_embed + self.level_embed[lvl].view(1, 1, -1)  # 后面那部分可以更新参数，前面的是backbone得到的，不能更新参数
        #     lvl_pos_embed_flatten.append(lvl_pos_embed)
        #     src_flatten.append(src)
        #     mask_flatten.append(mask)
        src_flatten = torch.cat(src_flatten, 1)  # bs, \sum{hxw}, c
        mask_flatten = torch.cat(mask_flatten, 1)  # bs, \sum{hxw}
        lvl_pos_embed_flatten = torch.cat(lvl_pos_embed_flatten, 1)
        spatial_shapes = torch.as_tensor(spatial_shapes, dtype=torch.long, device=src_flatten.device)
        level_start_index = torch.cat((spatial_shapes.new_zeros((1,)), spatial_shapes.prod(1).cumsum(0)[:-1]))
        valid_ratios = torch.stack([self.get_valid_ratio(m) for m in masks], 1)

        # encoder
        memory, memorys = self.encoder(src_flatten, spatial_shapes, level_start_index, valid_ratios, lvl_pos_embed_flatten, mask_flatten)

        # lvl_srcs = memorys
        lvl_srcs = [src_flatten] + memorys
        return memory, lvl_srcs

class DeformableTransformerEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers

    @staticmethod
    def get_reference_points(spatial_shapes, valid_ratios, device):
        reference_points_list = []
        for lvl, (H_, W_) in enumerate(spatial_shapes):
            ref_y, ref_x = torch.meshgrid(torch.linspace(0.5, H_ - 0.5, H_, dtype=torch.float32, device=device),
                                          torch.linspace(0.5, W_ - 0.5, W_, dtype=torch.float32, device=device))
            ref_y = ref_y.reshape(-1)[None] / (valid_ratios[:, None, lvl, 1] * H_)
            ref_x = ref_x.reshape(-1)[None] / (valid_ratios[:, None, lvl, 0] * W_)
            ref = torch.stack((ref_x, ref_y), -1)
            reference_points_list.append(ref)
        reference_points = torch.cat(reference_points_list, 1)
        reference_points = reference_points[:, :, None] * valid_ratios[:, None]
        return reference_points

    def forward(self, src, spatial_shapes, level_start_index, valid_ratios, pos=None, padding_mask=None):
        """
        Input:
            - src: [bs, sum(hi*wi), 256]
            - spatial_shapes: h,w of each level [num_level, 2]
            - level_start_index: [num_level] start point of level in sum(hi*wi).
            - valid_ratios: [bs, num_level, 2]
            - pos: pos embed for src. [bs, sum(hi*wi), 256]
        """
        output = src
        outputs = []
        # bs, sum(hi*wi), 256
        # import ipdb; ipdb.set_trace()
        reference_points = self.get_reference_points(spatial_shapes, valid_ratios, device=src.device)
        for _, layer in enumerate(self.layers):
            output = layer(output, pos, reference_points, spatial_shapes, level_start_index, padding_mask)
            outputs.append(output)

        return output, outputs


class DeformableTransformerEncoderLayer(nn.Module):
    def __init__(self,
                 d_model=256,dropout=0.1, activation="relu",
                 n_levels=4, n_heads=8, n_points=4, use_grl=False):
        super().__init__()

        unit_dim = d_model // 2

        # self attention
        self.self_attn = MSDeformAttn(d_model, n_levels, n_heads, n_points, cross_domain=True, use_grl=use_grl)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout1z = nn.Dropout(dropout)
        self.norm1_xy = nn.LayerNorm(unit_dim * 2)
        self.norm1_zd = nn.LayerNorm(unit_dim * 1)

        self.ffn_xy = FFN(unit_dim * 2, unit_dim * 2, activation, dropout)
        self.ffn_zd = FFN(unit_dim * 1, unit_dim * 1, activation, dropout)

        if use_grl:
            self.grl = GradientReversal()
            self.grl0 = GradientReversal()
        else:
            self.grl = nn.Identity()
            self.grl0 = nn.Identity()

    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos

    @staticmethod
    def with_pos_domain_embed(tensor, pos):
        if pos is None:
            return tensor
        unit_dim = tensor.shape[-1] // 3
        res = torch.cat((tensor[..., :unit_dim * 2] + pos[..., :unit_dim * 2],
                         tensor[..., unit_dim * 2:] * pos[..., unit_dim * 2:]), dim=-1)
        return res

    @staticmethod
    def with_grl_pos_embed(tensor, pos, grl0):
        if pos is None:
            return tensor
        unit_dim = tensor.shape[-1] // 3
        res = torch.cat((grl0(tensor[..., :unit_dim * 2]) + pos[..., :unit_dim * 2],
                         tensor[..., unit_dim * 2:] + pos[..., unit_dim * 2:]), dim=-1)
        return res

    def forward_ffn(self, src):
        src2 = self.linear2(self.dropout2(self.activation(self.linear1(src))))
        src = src + self.dropout3(src2)
        src = self.norm2(src)
        return src

    def forward(self, src, pos, reference_points, spatial_shapes, level_start_index, padding_mask=None):

        # self attention
        unit_dim = src.shape[-1] // 2
        src2 = self.self_attn.cd_forward(self.with_pos_embed(src, pos), reference_points, src, spatial_shapes, level_start_index, padding_mask)
        src_xy = src[..., :unit_dim * 2] + self.dropout1(src2)[..., :unit_dim * 2] # 残差结构
        src_xy = self.norm1_xy(src_xy)
        # ffn
        src = self.ffn_xy(src_xy)

        return src


class FFN(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, d_model, d_ffn, activation, dropout):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ffn)
        self.activation = _get_activation_fn(activation)
        self.dropout3 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.dropout4 = nn.Dropout(dropout)
        self.norm3 = nn.LayerNorm(d_model)
        self._reset_parameters()

    def _reset_parameters(self):
        xavier_uniform_(self.linear1.weight)
        constant_(self.linear1.bias, 0.)

        xavier_uniform_(self.linear2.weight)
        constant_(self.linear2.bias, 0.)

    def forward(self, tgt):
        tgt2 = self.linear2(self.dropout3(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout4(tgt2)
        tgt = self.norm3(tgt)
        return tgt

def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")

def build_deforamble_transformer(config):
    return DeformableTransformer(
        d_model=config['arch_kwargs']['hidden_dim'],
        nhead=config['arch_kwargs']['nheads'],
        num_encoder_layers=config['arch_kwargs']['enc_layers'],
        dropout=config['arch_kwargs']['dropout'],
        activation="relu",
        num_feature_levels=1,
        enc_n_points=config['arch_kwargs']['enc_n_points'],
        use_grl=config['arch_kwargs']['use_grl']
        )