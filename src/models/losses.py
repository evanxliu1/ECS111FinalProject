"""Supervised Contrastive Loss (Khosla et al. 2020), minimal implementation."""
import torch
from torch import nn
import torch.nn.functional as F


class SupConLoss(nn.Module):
    """SupCon: pull together same-class embeddings, push apart different-class.

    For our use, classes = {human, AI}. With a moderately sized batch and
    balanced sampling, this is well-defined.
    """

    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        features: [B, D] — already L2-normalized projection embeddings
        labels:   [B]    — class indices
        """
        device = features.device
        B = features.shape[0]

        # Pairwise similarity logits [B, B]
        sim = torch.matmul(features, features.T) / self.temperature

        # For numerical stability, subtract max per row
        sim = sim - sim.max(dim=1, keepdim=True).values.detach()

        # Mask of positive pairs (same label, excluding self)
        labels = labels.view(-1, 1)
        pos_mask = (labels == labels.T).float().to(device)
        self_mask = torch.eye(B, device=device)
        pos_mask = pos_mask - self_mask
        pos_mask = pos_mask.clamp(min=0.0)

        # log_prob for all non-self pairs
        exp_sim = torch.exp(sim) * (1 - self_mask)
        log_prob = sim - torch.log(exp_sim.sum(dim=1, keepdim=True) + 1e-12)

        # Mean log-prob over positive pairs per anchor
        pos_count = pos_mask.sum(dim=1)
        # Avoid div-by-zero for anchors with no positives in the batch
        valid = pos_count > 0
        if valid.sum() == 0:
            return torch.tensor(0.0, device=device, requires_grad=True)
        mean_log_prob_pos = (pos_mask * log_prob).sum(dim=1)[valid] / pos_count[valid]

        return -mean_log_prob_pos.mean()
