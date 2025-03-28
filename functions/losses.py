import os
import torch
import torch.nn.functional as F
from torch import nn
import math

def get_imbalance_mask(sigmoid_logits, labels, nclass, threshold=0.7, imbalance_scale=-1):
    if imbalance_scale == -1:
        imbalance_scale = 1 / nclass

    mask = torch.ones_like(sigmoid_logits) * imbalance_scale

    # wan to activate the output
    mask[labels == 1] = 1

    # if predicted wrong, and not the same as labels, minimize it
    correct = (sigmoid_logits >= threshold) == (labels == 1)
    mask[~correct] = 1

    multiclass_acc = correct.float().mean()

    # the rest maintain "imbalance_scale"
    return mask, multiclass_acc

class Mutual_loss(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, x_input_dim=128, x_output_dim=128, y_input_dim=128, y_output_dim=128):
        super().__init__()
        self.fc1_x = nn.Linear(x_input_dim, x_output_dim)
        self.fc1_y = nn.Linear(y_input_dim, y_output_dim)
        self.fc2 = nn.Linear(x_input_dim, x_input_dim)
        # self.fc2_t = nn.Linear(unit_dim*1, unit_dim)
        self.fc3 = nn.Linear(x_input_dim, 1)
        self._reset_parameters()

    def mine(self, x, y):
        h1 = F.leaky_relu(self.fc1_x(x) + self.fc1_y(y))
        h2 = F.leaky_relu(self.fc2(h1))
        h4 = self.fc3(h2)
        return h4

    def forward(self, xy, zd):
        """
        xy: bs, n, 256; domain invariant
        zd: bs, n, 128; domain specific
        """
        # xy = xy0.detach()
        joint = self.mine(xy, zd)
        # shuffled_xy = torch.index_select(xy, 1, torch.randperm(xy.shape[1]).cuda())
        # shuffled_zd
        # marginal = self.mine(shuffled_xy, zd)
        shuffled_zd = torch.index_select(zd, 1, torch.randperm(zd.shape[1]).cuda())
        marginal2 = self.mine(xy, shuffled_zd)
        # loss = torch.mean(joint, dim=1, keepdim=True) * 1  - torch.log(torch.mean(torch.exp(marginal), dim=1, keepdim=True)) #- torch.log(torch.mean(torch.exp(marginal2), dim=1, keepdim=True))
        loss = torch.mean(joint, dim=1, keepdim=True) * 1 - torch.log(
            torch.mean(torch.exp(marginal2), dim=1, keepdim=True))
        return loss.abs().mean()  # * 0.5
        # return loss.mean() * (-1)

    def _reset_parameters(self):
        # xavier_uniform_(self.linear1.weight.data)

        nn.init.xavier_uniform_(self.fc1_x.weight, gain=1)
        nn.init.xavier_uniform_(self.fc1_y.weight, gain=1)
        nn.init.xavier_uniform_(self.fc2.weight, gain=1)
        nn.init.xavier_uniform_(self.fc3.weight, gain=1)

        # nn.init.normal_(self.fc1_x.weight,std=0.02)
        # nn.init.normal_(self.fc1_y.weight,std=0.02)
        # nn.init.normal_(self.fc2.weight,std=0.02)
        # nn.init.normal_(self.fc3.weight,std=0.02)

        nn.init.constant_(self.fc1_x.bias.data, 0.)
        nn.init.constant_(self.fc1_y.bias.data, 0.)
        nn.init.constant_(self.fc2.bias.data, 0.)
        nn.init.constant_(self.fc3.bias.data, 0.)


class SetCriterion(nn.Module):
    def __init__(self, config):
        super().__init__()

        nclass = config['arch_kwargs']['nclass']
        u_dim = config['arch_kwargs']['hidden_dim'] // 2

        self.final_feature_MI = Mutual_loss(x_input_dim=u_dim, x_output_dim=u_dim, y_input_dim=u_dim, y_output_dim=u_dim)
        self.target_f_l_MI = Mutual_loss(x_input_dim=nclass, x_output_dim=nclass, y_input_dim=u_dim, y_output_dim=nclass) # feature-label mutual information
        self.sensitive_f_l_MI = Mutual_loss(x_input_dim=2, x_output_dim=2, y_input_dim=u_dim, y_output_dim=2)
        self.m_type = config['m_type']
        self.m = config['m']
        self.s = config['s']

        self.quan = 0
        self.multiclass = False

        self.losses = {}

        # # for label continuation
        # torch.manual_seed(42)
        self.u_dim = u_dim
        self.target_embedding = nn.Embedding(config['arch_kwargs']['nclass'], u_dim)
        self.sensitive_embedding = nn.Embedding(2, u_dim)
        self.target_embedding.weight.data.normal_(mean=0.0, std=0.1)
        self.sensitive_embedding.weight.data.normal_(mean=0.0, std=0.1)
        self.target_embedding.weight.requires_grad = False
        self.sensitive_embedding.weight.requires_grad = False

        # loss hyperparameters
        self.ce = config['loss_params']['ce']
        self.final_feature_MI_weight = config['loss_params']['final_feature']
        self.lvl = config['loss_params']['lvl']
        self.attributes_feature_MI_weight = config['loss_params']['attributes_feature']

    def compute_margin_logits(self, logits, labels):
        if self.m_type == 'cos':
            if self.multiclass:
                y_onehot = labels * self.m
                margin_logits = self.s * (logits - y_onehot)
            else:
                y_onehot = torch.zeros_like(logits)
                y_onehot.scatter_(1, torch.unsqueeze(labels, dim=-1), self.m)
                margin_logits = self.s * (logits - y_onehot)
        else:
            if self.multiclass:
                y_onehot = labels * self.m
                arc_logits = torch.acos(logits.clamp(-0.99999, 0.99999))
                logits = torch.cos(arc_logits + y_onehot)
                margin_logits = self.s * logits
            else:
                y_onehot = torch.zeros_like(logits)
                y_onehot.scatter_(1, torch.unsqueeze(labels, dim=-1), self.m)
                arc_logits = torch.acos(logits.clamp(-0.99999, 0.99999))
                logits = torch.cos(arc_logits + y_onehot)
                margin_logits = self.s * logits

        return margin_logits

    def forward(self, logits, ta, sa, lvl_srcs, code, onehot=True):


        target_feature, sensitive_features = code[..., :self.u_dim], code[..., self.u_dim:]

        # original CE loss, 用于常规分类
        ta_label = ta.argmax(dim=1)
        margin_logits = self.compute_margin_logits(logits, ta_label)
        loss_ce = F.cross_entropy(margin_logits, ta_label)

        # 特征之间的互信息，decrease the mutual information between features
        loss_final_feature_MI = self.final_feature_MI(target_feature, sensitive_features)

        ## 各层之间的特征做二分类，分开taraget feature和sensitive feature
        loss_lvls = []
        for one_lvl_src in lvl_srcs:
            target = torch.empty_like(one_lvl_src)
            target[..., :self.u_dim] = 0
            target[..., self.u_dim:] = 1
            loss_lvls.append(F.binary_cross_entropy_with_logits(one_lvl_src, target))
        loss_lvl = torch.stack(loss_lvls).mean()

        # 标签与特征做互信息，increase the mutual information between label and features
        # label continuity
        # target_indices = torch.argmax(ta, dim=1)
        # sensitive_indices = torch.argmax(sa, dim=1)
        # target_label_embedding = self.target_embedding(target_indices)
        # sensitive_label_embedding = self.sensitive_embedding(sensitive_indices)
        #
        # loss_target_f_l_MI = self.target_f_l_MI(target_label_embedding, target_feature)
        # loss_sensitive_f_l_MI = self.sensitive_f_l_MI(sensitive_label_embedding, sensitive_features)
        # loss_attributes_feature_MI = loss_target_f_l_MI + loss_sensitive_f_l_MI
        loss_target_f_l_MI = self.target_f_l_MI(ta.float(), target_feature)
        loss_sensitive_f_l_MI = self.sensitive_f_l_MI(sa.float(), sensitive_features)
        loss_attributes_feature_MI = loss_target_f_l_MI + loss_sensitive_f_l_MI

        # 最终loss
        loss_all = (self.ce * loss_ce + self.final_feature_MI_weight * loss_final_feature_MI +
                    self.lvl * loss_lvl - self.attributes_feature_MI_weight * loss_attributes_feature_MI)
        # loss_all = self.ce * loss_ce + self.lvl * loss_lvl  # 33.81
        # loss_all = self.ce * loss_ce - self.attributes_feature_MI_weight * loss_attributes_feature_MI

        self.losses['loss_ce'] = loss_ce
        self.losses['loss_final_feature_MI'] = loss_final_feature_MI
        self.losses['loss_lvl'] = loss_lvl
        self.losses['loss_attributes_feature_MI'] = loss_attributes_feature_MI

        return loss_all