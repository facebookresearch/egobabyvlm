# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Objective / loss plumbing for the Flamingo trainer.

The primary objective is the teacher-forced **caption log-likelihood** (needed
for the generative eval + layerwise logit-lens). Two optional, metric-aligned
auxiliary terms are wired in **config-gated and default-OFF** (weights 0), so
they can be turned on later without a rewrite (see
``FLAMINGO_c_options_and_ab_sketches.md`` §Q4):

* **(a) hard-negative caption margin** — the correct caption should score higher
  than a mismatched (in-batch) caption for the *same* image:
  ``relu(margin - (LL_pos - LL_neg))``. No new parameters.
* **(b) ITM + contrastive** — an image-text-matching classifier over
  matched/shuffled pairs, plus an in-batch InfoNCE between pooled image latents
  and pooled caption states. Uses :class:`FlamingoAuxHeads`.

:class:`FlamingoTrainingModule` wraps a :class:`FlamingoVLM` and returns the
scalar total loss from a single ``forward`` — so it can be DDP-wrapped directly
(all sub-forwards live inside one wrapper forward → one backward → correct DDP).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch
import torch.nn.functional as F
from torch import nn

from apps.baselines.flamingo.constants import IGNORE_INDEX
from apps.baselines.flamingo.modeling.aux_heads import masked_mean

if TYPE_CHECKING:
    from apps.baselines.flamingo.modeling.flamingo_model import FlamingoVLM


@dataclass
class ObjectiveWeights:
    """Runtime copy of the objective weights (plain floats)."""

    caption_loss_weight: float = 1.0
    hard_negative_loss_weight: float = 0.0
    hard_negative_margin: float = 1.0
    itm_loss_weight: float = 0.0
    contrastive_loss_weight: float = 0.0

    @property
    def uses_aux_heads(self) -> bool:
        """Whether ITM / contrastive (which need aux heads) are active."""
        return self.itm_loss_weight > 0 or self.contrastive_loss_weight > 0


def sequence_ll(
    model: FlamingoVLM,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    labels: torch.Tensor,
    image_embeds: torch.Tensor,
) -> torch.Tensor:
    """Per-example caption log-likelihood (sum of token log-probs), ``(B,)``."""
    out = model(input_ids=input_ids, attention_mask=attention_mask, image_embeds=image_embeds)
    shift_logits = out.logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    logp = F.log_softmax(shift_logits, dim=-1)
    mask = (shift_labels != IGNORE_INDEX).float()
    tgt = shift_labels.clamp_min(0).unsqueeze(-1)
    tok_ll = logp.gather(-1, tgt).squeeze(-1) * mask
    return tok_ll.sum(dim=1)


class FlamingoTrainingModule(nn.Module):
    """DDP-wrappable module: ``forward(batch) -> (loss, log_dict)``."""

    def __init__(self, model: FlamingoVLM, weights: ObjectiveWeights) -> None:
        super().__init__()
        self.model = model
        self.weights = weights

    def forward(self, batch: dict[str, Any]) -> tuple[torch.Tensor, dict[str, float]]:
        """Compute the weighted total loss for a batch."""
        w = self.weights
        pixel_values = batch["pixel_values"]
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]

        # Encode the image once; reuse the latents for every sub-forward.
        image_embeds = self.model.encode_images(pixel_values)

        out = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            image_embeds=image_embeds,
        )
        caption_loss = out.loss
        total = w.caption_loss_weight * caption_loss
        log: dict[str, float] = {"caption_loss": float(caption_loss.item())}

        # (a) Hard-negative caption margin — same image, wrong caption.
        if w.hard_negative_loss_weight > 0:
            ll_pos = sequence_ll(self.model, input_ids, attention_mask, labels, image_embeds)
            ll_neg = sequence_ll(
                self.model,
                batch["neg_input_ids"],
                batch["neg_attention_mask"],
                batch["neg_labels"],
                image_embeds,
            )
            margin_loss = F.relu(w.hard_negative_margin - (ll_pos - ll_neg)).mean()
            total = total + w.hard_negative_loss_weight * margin_loss
            log["hard_negative_loss"] = float(margin_loss.item())

        # (b) ITM + contrastive — need aux heads.
        if w.uses_aux_heads:
            aux = self.model.aux_heads
            if aux is None:
                msg = "itm/contrastive loss weight > 0 but model.aux_heads is None (set model use_aux_heads)."
                raise RuntimeError(msg)
            txt_states = out.hidden_states[-1]

            if w.itm_loss_weight > 0:
                h_pos = masked_mean(txt_states, attention_mask)
                shuffle = torch.randperm(image_embeds.shape[0], device=image_embeds.device)
                neg_out = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    image_embeds=image_embeds[shuffle],
                )
                h_neg = masked_mean(neg_out.hidden_states[-1], attention_mask)
                logits_itm = aux.itm_head(torch.cat([h_pos, h_neg], dim=0))
                tgt = (
                    torch.cat(
                        [torch.ones(h_pos.shape[0]), torch.zeros(h_neg.shape[0])],
                    )
                    .long()
                    .to(logits_itm.device)
                )
                itm_loss = F.cross_entropy(logits_itm, tgt)
                total = total + w.itm_loss_weight * itm_loss
                log["itm_loss"] = float(itm_loss.item())

            if w.contrastive_loss_weight > 0:
                img_sum = image_embeds.mean(dim=(1, 2))  # (B, H)
                txt_sum = masked_mean(txt_states, attention_mask)
                zi = F.normalize(aux.img_proj(img_sum), dim=-1)
                zt = F.normalize(aux.txt_proj(txt_sum), dim=-1)
                # Temperature divides the logits: bound it below (CLIP caps logit scale at 100).
                scale = aux.log_temp.exp().clamp(min=0.01)
                logits_c = zi @ zt.t() / scale
                tgt_c = torch.arange(zi.shape[0], device=zi.device)
                contrastive_loss = 0.5 * (F.cross_entropy(logits_c, tgt_c) + F.cross_entropy(logits_c.t(), tgt_c))
                total = total + w.contrastive_loss_weight * contrastive_loss
                log["contrastive_loss"] = float(contrastive_loss.item())

        log["loss"] = float(total.item())
        return total, log
