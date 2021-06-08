import torch.nn as nn
import torch

from lanedet.models.registry import NET
from ..registry import build_backbones, build_aggregator, build_heads, build_necks


@NET.register_module
class Detector(nn.Module):
    def __init__(self, cfg):
        super(Detector, self).__init__()
        self.cfg = cfg
        self.backbone = build_backbones(cfg)
        self.aggregator = build_aggregator(cfg) if cfg.aggregator else None
        self.neck = build_necks(cfg) if cfg.neck else None
        self.heads = build_heads(cfg)

    def forward(self, batch):
        output = {}
        fea = self.backbone(batch['img'])

        if self.aggregator:
            fea[-1] = self.aggregator(fea[-1])

        if self.neck:
            fea = self.neck(fea)

        if self.training:
            out = self.heads(fea, batch=batch)
            output.update(self.heads.loss(out, batch))
        else:
            output = self.heads(fea)
            output = self.heads.get_lanes(output)

        return output
