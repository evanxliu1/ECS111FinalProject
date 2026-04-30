"""DeBERTa encoder wrapper with mean-pooling."""
import torch
from torch import nn
from transformers import AutoModel, AutoTokenizer


def mean_pool(last_hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).float()
    summed = (last_hidden * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1e-6)
    return summed / counts


class Encoder(nn.Module):
    def __init__(self, model_name: str = "microsoft/deberta-v3-base"):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(model_name)
        self.hidden_size = self.backbone.config.hidden_size

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        out = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        return mean_pool(out.last_hidden_state, attention_mask)


def make_tokenizer(model_name: str = "microsoft/deberta-v3-base"):
    return AutoTokenizer.from_pretrained(model_name)
