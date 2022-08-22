import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader


class wrapper(nn.Module):

    def __init__(self, module, feat_dim, class_num=100):

        super(wrapper, self).__init__()

        self.backbone = module
        in_features = list(module.children())[-1].in_features
        self.proj_head = nn.Sequential(
            nn.Linear(in_features, feat_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feat_dim, feat_dim)
            )
        self.l2norm = Normalize(2)
        self.classifier = nn.Linear(feat_dim, class_num)
        self.weight_check = None

    def forward(self, x, bb_grad=True):
        with torch.no_grad():    
            feats, out = self.backbone(x, is_feat=True)
        feat = feats[-1].view(feats[-1].size(0), -1)
        if not bb_grad:
            feat = feat.detach()
        proj_x = self.proj_head(feat)
        proj_x  = self.l2norm(proj_x)
        proj_x = proj_x * 16.0
        proj_logit = self.classifier(proj_x)
        return feat, out, proj_x, proj_logit

    def forward_proj(self, x, bb_grad=False):
        if self.weight_check is None:
            self.weight_check = self.classifier.weight.norm()
        weight_norm = self.classifier.weight.norm()
        assert weight_norm.item() == self.weight_check.item()
        x = x.view(x.size(0), -1)
        x = self.l2norm(x)
        x = x * 16.0
        proj_logit = self.classifier(x)
        return proj_logit


class Normalize(nn.Module):
    """normalization layer"""

    def __init__(self, power=2):
        super(Normalize, self).__init__()
        self.power = power

    def forward(self, x):
        norm = x.pow(self.power).sum(1, keepdim=True).pow(1. / self.power)
        out = x.div(norm)
        return out
