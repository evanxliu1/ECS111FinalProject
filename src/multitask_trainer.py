"""Hugging Face Trainer wrapper for DebertaMultitaskModel."""

from __future__ import annotations

from transformers import Trainer

from deberta_multitask import grl_schedule


class MultitaskTrainer(Trainer):
    def __init__(self, *args, grl_gamma: float = 10.0, grl_max_lambda: float = 1.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.grl_gamma = grl_gamma
        self.grl_max_lambda = grl_max_lambda

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        source_ids = inputs.pop("source_ids", None)
        total_steps = max(self.state.max_steps, 1)
        grl_lambda = grl_schedule(
            self.state.global_step,
            total_steps,
            gamma=self.grl_gamma,
            max_lambda=self.grl_max_lambda,
        )
        outputs = model(
            **inputs,
            labels=labels,
            source_ids=source_ids,
            grl_lambda=grl_lambda,
        )
        return (outputs.loss, outputs) if return_outputs else outputs.loss
