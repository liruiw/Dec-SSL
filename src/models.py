#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6
import torch
from torch import nn
import torch.nn.functional as F
from utils import get_backbone


class MLP(nn.Module):
    def __init__(self, dim_in, dim_hidden, dim_out):
        super(MLP, self).__init__()
        self.layer_input = nn.Linear(dim_in, dim_hidden)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout()
        self.layer_hidden = nn.Linear(dim_hidden, dim_out)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = x.view(-1, x.shape[1] * x.shape[-2] * x.shape[-1])
        x = self.layer_input(x)
        x = self.dropout(x)
        x = self.relu(x)
        x = self.layer_hidden(x)
        return self.softmax(x)


class Flatten(torch.nn.Module):
    def forward(self, x):
        batch_size = x.shape[0]
        return x.view(batch_size, -1)


class prediction_MLP(nn.Module):
    def __init__(
        self, in_dim=2048, hidden_dim=512, out_dim=2048
    ):  # bottleneck structure
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
        )
        self.layer2 = nn.Linear(hidden_dim, out_dim)

    def forward(self, x):
        x = torch.flatten(x, start_dim=1)
        x = self.layer1(x)
        x = self.layer2(x)
        return x


class SimCLR(nn.Module):
    def __init__(self, args, feature_dim=128, input_dim=32):
        super(SimCLR, self).__init__()
        self.input_dim = input_dim

        # encoder
        self.f, feat_dim = get_backbone(args.backbone, full_size=args.full_size)
        # projection head
        self.g = prediction_MLP(feat_dim, 512, 512)
        self.output_dim = feat_dim

    def get_encoder(self):
        return nn.Sequential(self.f, Flatten(), self.g)

    def forward(self, pos1, pos2=None, add_feat=None, scale=1.0, return_feat=False):
        temperature = 0.5
        batch_size = len(pos1)
        pos1 = self.f(pos1)

        feature1 = torch.flatten(pos1, start_dim=1)
        out1 = self.g(feature1)

        pos2 = self.f(pos2)

        feature2 = torch.flatten(pos2, start_dim=1)
        out2 = self.g(feature2)

        out_1, out_2 = F.normalize(out1, dim=-1), F.normalize(out2, dim=-1)
        out = torch.cat([out_1, out_2], dim=0)

        # [2*B, 2*B]
        sim_matrix = torch.exp(torch.mm(out, out.t().contiguous()) / temperature)
        mask = (
            torch.ones_like(sim_matrix)
            - torch.eye(2 * batch_size, device=sim_matrix.device)
        ).bool()

        # [2*B, 2*B-1]
        sim_matrix = sim_matrix.masked_select(mask).view(2 * batch_size, -1)

        # compute loss
        pos_sim = torch.exp(torch.sum(out_1 * out_2, dim=-1) / temperature)

        # [2*B]
        pos_sim = torch.cat([pos_sim, pos_sim], dim=0)
        loss = (-torch.log(pos_sim / sim_matrix.sum(dim=-1))).mean()

        if add_feat is not None:
            if type(add_feat) is list:
                add_feat = [add_feat[0].float().detach(), add_feat[1].float().detach()]
            else:
                add_feat = add_feat.float().detach()
                add_feat = [add_feat.float().detach(), add_feat.float().detach()]
            reg_loss = -0.5 * (
                (add_feat[0] * out_1).sum(-1).mean()
                + (add_feat[1] * out_2).sum(-1).mean()
            )
            loss = loss + scale * reg_loss

        if return_feat:
            return loss, feature1
        return loss

    def save_model(self, model_dir, suffix=None, step=None):
        model_name = (
            "model_{}.pth".format(suffix) if suffix is not None else "model.pth"
        )
        torch.save(
            {"model": self.state_dict(), "step": step},
            "{}/{}".format(model_dir, model_name),
        )


class ResNetCifarClassifier(nn.Module):
    def __init__(self, args, num_class=10, feature_dim=128, data_dim=32):
        super(ResNetCifarClassifier, self).__init__()

        # encoder
        self.data_dim = data_dim
        num_class = 10

        if "imagenet100" in args.dataset:
            num_class = 100
        elif "imagenet" in args.dataset:
            num_class = 1000

        model = SimCLR(args, feature_dim, data_dim)
        self.f = model.f
        # classifier
        self.fc = nn.Linear(model.output_dim, num_class, bias=True)

    def forward(self, x, return_feat=False):
        x = self.f(x)
        feature = torch.flatten(x, start_dim=1)
        out = self.fc(feature)
        if return_feat:
            return F.log_softmax(out, dim=1), feature
        return F.log_softmax(out, dim=1)

    def save_model(self, model_dir, suffix=None, step=None):
        model_name = (
            "model_{}.pth".format(suffix) if suffix is not None else "model.pth"
        )
        torch.save(
            {"model": self.state_dict(), "step": step},
            "{}/{}".format(model_dir, model_name),
        )
