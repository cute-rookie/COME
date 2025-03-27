import os
import torch
import torch.nn.functional as F
from torch import nn
import math
from models.position_encoding import build_position_encoding


class CosSim(nn.Module):
    def __init__(self, nfeat, nclass, codebook=None, learn_cent=True):
        super(CosSim, self).__init__()
        self.nfeat = nfeat
        self.nclass = nclass
        self.learn_cent = learn_cent

        if codebook is None:  # if no centroids, by default just usual weight
            codebook = torch.randn(nclass, nfeat)

        self.centroids = nn.Parameter(codebook.clone())
        if not learn_cent:
            self.centroids.requires_grad_(False)

    def forward(self, x):
        norms = torch.norm(x, p=2, dim=-1, keepdim=True)
        nfeat = torch.div(x, norms)

        norms_c = torch.norm(self.centroids, p=2, dim=-1, keepdim=True)
        ncenters = torch.div(self.centroids, norms_c)
        logits = torch.matmul(nfeat, torch.transpose(ncenters, 0, 1))

        return logits

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

class RetrievalModel(nn.Module):
    def __init__(self, config, backbone, transformer, codebook, num_classes):
        super(RetrievalModel, self).__init__()
        self.backbone = backbone
        self.transformer = transformer
        self.codebook = codebook
        self.num_classes = num_classes
        hidden_dim = transformer.d_model
        self.hidden_dim = hidden_dim

        unit_dim = hidden_dim // 2
        self.unit_dim = unit_dim
        nbit = config['arch_kwargs']['nbit']
        self.input_proj = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(256, hidden_dim, kernel_size=1),
                nn.GroupNorm(32, hidden_dim),
            )])
        input_feature_dim = hidden_dim * 6 * 6
        self.process_for_feature = nn.Sequential(
            nn.Linear(input_feature_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(config['arch_kwargs']['dropout']),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(config['arch_kwargs']['dropout']),
            nn.ReLU(inplace=True),
            )

        self.process_for_codebook = nn.Sequential(
            nn.Linear(hidden_dim, nbit),
            # nn.LayerNorm(nbit),
            nn.ReLU(inplace=True),
            nn.Dropout(config['arch_kwargs']['dropout'])
        )

        ##### channel process #####
        self.channel_process = nn.Sequential(
            nn.Conv2d(256, self.hidden_dim, kernel_size=1),
            nn.GroupNorm(32, self.hidden_dim),
        )
        ##### Positional Encoding #####
        self.position_encoding = build_position_encoding(config)


        if codebook is None:  # usual CE
            self.ce_fc = nn.Linear(nbit, num_classes)
        else:
            # not learning cent, we are doing codebook learning
            self.ce_fc = CosSim(nbit, num_classes, codebook, learn_cent=False)

    def forward(self, image):
        features = self.backbone(image)
        features = self.channel_process(features)
        pos = self.position_encoding(features).to(features.dtype)
        lvl_feature, lvl_features = self.transformer(features, pos)

        lvl_feature = lvl_feature.view(lvl_feature.size(0), -1)
        code = self.process_for_feature(lvl_feature)
        v = self.process_for_codebook(code)
        logits = self.ce_fc(v)
        return logits, code, lvl_features





def build_retrieval_model(config,backbone, transformer, codebook, num_classes):
    return RetrievalModel(
        config,
        backbone,
        transformer,
        codebook,
        num_classes)