import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import get_backbone


def D(p, z, version="simplified"):  # negative cosine similarity
    return 1.0 - F.cosine_similarity(p, z.detach(), dim=-1).mean()


class Flatten(torch.nn.Module):
    def forward(self, x):
        batch_size = x.shape[0]
        return x.view(batch_size, -1)


class SimSiam(nn.Module):
    def __init__(self, args=None, backbone=None):
        super(SimSiam, self).__init__()

        self.backbone, backbone_in_channels = get_backbone(
            args.backbone,
            full_size=args.full_size,
        )
        self.f = self.backbone
        proj_hid, proj_out = 2048, 512
        pred_hid, pred_out = 512, 512

        self.projection = nn.Sequential(
            Flatten(),
            nn.Linear(backbone_in_channels, proj_hid, bias=False),
            nn.BatchNorm1d(proj_hid),
            nn.ReLU(inplace=True),  # first layer
            nn.Linear(proj_hid, proj_hid, bias=False),
            nn.BatchNorm1d(proj_hid),
            nn.ReLU(inplace=True),  # second layer
            nn.Linear(proj_hid, proj_out, bias=False),
            nn.BatchNorm1d(proj_out, affine=False),
        )  # output layer

        self.encoder = nn.Sequential(self.f, self.projection)
        # build a 2-layer predictor
        self.prediction = nn.Sequential(
            nn.Linear(proj_out, pred_hid, bias=False),
            nn.BatchNorm1d(pred_hid),
            nn.ReLU(inplace=True),  # hidden layer
            nn.Linear(pred_hid, pred_out),
        )  # output layer

    def get_encoder(self):
        return self.encoder

    def forward(self, x1, x2, add_feat=None, scale=1.0, return_feat=False):
        z1 = self.encoder(x1)
        p1 = self.prediction(z1)
        z2 = self.encoder(x2)
        p2 = self.prediction(z2)

        d1 = D(p1, z2) / 2.0
        d2 = D(p2, z1) / 2.0
        loss = d1 + d2

        if add_feat is not None:
            out_1 = F.normalize(z1, dim=-1)
            out_2 = F.normalize(z2, dim=-1)

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
            return loss, p1
        return loss

    def save_model(self, model_dir, suffix=None, step=None):
        model_name = (
            "model_{}.pth".format(suffix) if suffix is not None else "model.pth"
        )
        torch.save(
            {"model": self.state_dict(), "step": step},
            "{}/{}".format(model_dir, model_name),
        )
