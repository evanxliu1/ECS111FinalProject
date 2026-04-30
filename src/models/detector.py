"""Composes encoder + classification head + optional contrastive/adversarial heads.

The same module supports all 4 variants by toggling λ weights in the config:
  - base:        λ_c = 0, λ_a = 0
  - contrastive: λ_c > 0, λ_a = 0
  - adversarial: λ_c = 0, λ_a > 0
  - both:        λ_c > 0, λ_a > 0
"""
from dataclasses import dataclass

import torch
from torch import nn
import torch.nn.functional as F

from src.models.encoder import Encoder
from src.models.grl import grad_reverse
from src.models.losses import SupConLoss


@dataclass
class DetectorOutput:
    cls_logits: torch.Tensor                 # [B, 2]
    proj: torch.Tensor | None = None         # [B, D] (L2-normalized) for contrastive
    gen_logits: torch.Tensor | None = None   # [B, num_generators] for adversarial


class Detector(nn.Module):
    def __init__(
        self,
        encoder_name: str = "microsoft/deberta-v3-base",
        projection_dim: int = 256,
        num_generators: int = 4,
        use_contrastive: bool = False,
        use_adversarial: bool = False,
    ):
        super().__init__()
        self.encoder = Encoder(encoder_name)
        H = self.encoder.hidden_size

        self.cls_head = nn.Linear(H, 2)
        self.use_contrastive = use_contrastive
        self.use_adversarial = use_adversarial

        if use_contrastive:
            self.proj_head = nn.Sequential(
                nn.Linear(H, H), nn.ReLU(), nn.Linear(H, projection_dim)
            )
        if use_adversarial:
            self.gen_head = nn.Sequential(
                nn.Linear(H, H), nn.ReLU(), nn.Linear(H, num_generators)
            )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        grl_lambda: float = 1.0,
    ) -> DetectorOutput:
        h = self.encoder(input_ids, attention_mask)
        out = DetectorOutput(cls_logits=self.cls_head(h))

        if self.use_contrastive:
            z = self.proj_head(h)
            out.proj = F.normalize(z, dim=-1)
        if self.use_adversarial:
            h_rev = grad_reverse(h, grl_lambda)
            out.gen_logits = self.gen_head(h_rev)
        return out

    def head_parameters(self):
        params = list(self.cls_head.parameters())
        if self.use_contrastive:
            params += list(self.proj_head.parameters())
        if self.use_adversarial:
            params += list(self.gen_head.parameters())
        return params


def compute_loss(
    out: DetectorOutput,
    labels: torch.Tensor,
    gen_labels: torch.Tensor | None,
    lambda_ce: float,
    lambda_contrastive: float,
    lambda_adversarial: float,
    supcon: SupConLoss | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    parts: dict[str, float] = {}
    loss = torch.zeros((), device=out.cls_logits.device)

    if lambda_ce > 0:
        l_ce = F.cross_entropy(out.cls_logits, labels)
        loss = loss + lambda_ce * l_ce
        parts["ce"] = float(l_ce.detach())

    if lambda_contrastive > 0 and out.proj is not None and supcon is not None:
        l_c = supcon(out.proj, labels)
        loss = loss + lambda_contrastive * l_c
        parts["contrastive"] = float(l_c.detach())

    if lambda_adversarial > 0 and out.gen_logits is not None and gen_labels is not None:
        l_a = F.cross_entropy(out.gen_logits, gen_labels)
        loss = loss + lambda_adversarial * l_a
        parts["adversarial"] = float(l_a.detach())

    parts["total"] = float(loss.detach())
    return loss, parts
