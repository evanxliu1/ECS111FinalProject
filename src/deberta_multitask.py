"""DeBERTa encoder + CE / supervised contrastive / generator-adversarial (GRL) heads."""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, PretrainedConfig
from transformers.modeling_outputs import SequenceClassifierOutput


class GradientReversalFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, lambda_: float) -> torch.Tensor:
        ctx.lambda_ = lambda_
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        return -ctx.lambda_ * grad_output, None


def grad_reverse(x: torch.Tensor, lambda_: float) -> torch.Tensor:
    return GradientReversalFn.apply(x, lambda_)


def mean_pool(hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).type_as(hidden)
    summed = (hidden * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1e-9)
    return summed / counts


def supervised_contrastive_loss(
    features: torch.Tensor,
    labels: torch.Tensor,
    temperature: float = 0.07,
) -> torch.Tensor:
    """Supervised contrastive loss; needs >=2 samples per class in batch when possible."""
    device = features.device
    labels = labels.view(-1, 1)
    mask = torch.eq(labels, labels.T).float().to(device)

    logits = torch.matmul(features, features.T) / temperature
    logits = logits - torch.max(logits, dim=1, keepdim=True)[0].detach()

    logits_mask = torch.ones_like(mask) - torch.eye(mask.size(0), device=device)
    mask = mask * logits_mask

    exp_logits = torch.exp(logits) * logits_mask
    log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True) + 1e-12)

    pos_per_row = mask.sum(1)
    # rows with no positive pair (singleton class in batch) are skipped
    valid = pos_per_row > 0
    if not valid.any():
        return features.new_tensor(0.0)

    mean_log_prob_pos = (mask * log_prob).sum(1) / pos_per_row.clamp(min=1.0)
    return -mean_log_prob_pos[valid].mean()


def grl_schedule(step: int, total_steps: int, gamma: float = 10.0, max_lambda: float = 1.0) -> float:
    """DANN-style ramp from 0 → max_lambda over training."""
    import math

    if total_steps <= 0:
        return 0.0
    p = float(step) / float(total_steps)
    return float(max_lambda * (2.0 / (1.0 + math.exp(-gamma * p)) - 1.0))


@dataclass
class MultitaskConfig:
    model_name: str = "microsoft/deberta-v3-base"
    variant: str = "both"
    num_labels: int = 2
    num_sources: int = 4
    proj_dim: int = 128
    lambda_c: float = 0.5
    lambda_a: float = 0.5
    supcon_temperature: float = 0.07
    grl_gamma: float = 10.0
    grl_max_lambda: float = 1.0
    source_map: dict[str, int] | None = None

    def use_supcon(self) -> bool:
        return self.variant in ("contrastive", "both")

    def use_adversarial(self) -> bool:
        return self.variant in ("adversarial", "both")

    def save(self, directory: str) -> None:
        os.makedirs(directory, exist_ok=True)
        with open(os.path.join(directory, "multitask_config.json"), "w", encoding="utf-8") as f:
            json.dump(asdict(self), f, indent=2)

    @classmethod
    def load(cls, directory: str) -> MultitaskConfig:
        path = os.path.join(directory, "multitask_config.json")
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        return cls(**data)


class DebertaMultitaskModel(nn.Module):
    def __init__(self, config: MultitaskConfig):
        super().__init__()
        self.config = config
        try:
            self.encoder = AutoModel.from_pretrained(
                config.model_name,
                dtype=torch.float32,
                local_files_only=True,
            )
        except OSError:
            self.encoder = AutoModel.from_pretrained(
                config.model_name,
                dtype=torch.float32,
            )
        hidden = self.encoder.config.hidden_size
        self.classifier = nn.Linear(hidden, config.num_labels)
        self.projection = nn.Linear(hidden, config.proj_dim)
        self.domain_classifier = nn.Linear(hidden, config.num_sources)

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        source_ids: torch.Tensor | None = None,
        grl_lambda: float = 1.0,
        return_dict: bool = True,
        **kwargs,
    ):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask, **kwargs)
        h = mean_pool(outputs.last_hidden_state, attention_mask)
        logits = self.classifier(h)

        loss = None
        loss_ce = None
        loss_supcon = None
        loss_adv = None

        if labels is not None:
            loss_ce = F.cross_entropy(logits, labels)
            loss = loss_ce

            if self.config.use_supcon():
                proj = F.normalize(self.projection(h), dim=-1)
                loss_supcon = supervised_contrastive_loss(
                    proj, labels, temperature=self.config.supcon_temperature
                )
                loss = loss + self.config.lambda_c * loss_supcon

            if self.config.use_adversarial() and source_ids is not None:
                h_rev = grad_reverse(h, grl_lambda)
                adv_logits = self.domain_classifier(h_rev)
                loss_adv = F.cross_entropy(adv_logits, source_ids)
                loss = loss + self.config.lambda_a * loss_adv

        if not return_dict:
            return (loss, logits) if loss is not None else (logits,)

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states if hasattr(outputs, "hidden_states") else None,
        )

    def save_pretrained(self, directory: str) -> None:
        os.makedirs(directory, exist_ok=True)
        self.config.save(directory)
        torch.save(self.state_dict(), os.path.join(directory, "pytorch_model.bin"))

    @classmethod
    def from_pretrained(cls, directory: str, map_location: str | torch.device = "cpu"):
        config = MultitaskConfig.load(directory)
        model = cls(config)
        state_path = os.path.join(directory, "pytorch_model.bin")
        state = torch.load(state_path, map_location=map_location, weights_only=True)
        model.load_state_dict(state)
        return model


def build_source_map(train_csv_path: str) -> tuple[dict[str, int], list[str]]:
    import pandas as pd

    df = pd.read_csv(train_csv_path)
    generators = sorted(df["generator"].astype(str).str.strip().unique())
    source_map = {g: i for i, g in enumerate(generators)}
    return source_map, generators


VARIANTS = ("base", "contrastive", "adversarial", "both")
