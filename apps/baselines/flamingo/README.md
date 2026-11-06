<!--
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
-->

# Flamingo baseline

A small-scale [Flamingo](https://arxiv.org/abs/2204.14198) (Alayrac et al.,
2022) generative VLM, built from the same encoders as our LLaVA baseline:

- **Vision encoder**: a frozen DINOv2 ViT-B/14 pretrained on the training corpus
  ([`apps/baselines/dinov2/`](../dinov2/)).
- **Language model**: a GPT-2 Small pretrained on the same corpus (Phase 0,
  [`apps/baselines/lm_training/`](../lm_training/)), trained (not frozen)
  during Flamingo training.

```
frozen DINOv2 ViT-B/14 (256 patch tokens)
    -> Perceiver Resampler (64 latents, depth 2)
    -> GPT-2 Small with a gated cross-attention block before each of its 12 layers
    -> caption log-likelihood
```

The Perceiver Resampler and gated cross-attention are adapted from
[lucidrains/flamingo-pytorch](https://github.com/lucidrains/flamingo-pytorch)
(MIT); see file headers for attribution.

> [!NOTE]
> **Causal mask.** The decoder calls GPT-2's blocks directly, so it builds the
> attention mask itself. It always passes a `(B, 1, S, S)` causal + padding mask
> (`modeling/decoder.py::_causal_attention_mask`). An earlier version passed only
> a padding mask, which made attention non-causal under PyTorch sdpa (every
> token could see the future). The paper's numbers come from the fixed code;
> `tests/apps/baselines/flamingo/test_decoder_causal_mask.py` guards it.

### Layout

```
apps/baselines/flamingo/
├── modeling/
│   ├── perceiver_resampler.py        # Perceiver Resampler (adapted from flamingo-pytorch)
│   ├── gated_cross_attention.py      # gated cross-attention (adapted from flamingo-pytorch)
│   ├── vision_tower.py               # frozen DINOv2 (shares the LLaVA tower + loader)
│   ├── decoder.py                    # GPT-2 + interleaved gated cross-attn + causal mask
│   ├── aux_heads.py                  # optional ITM / contrastive heads (off in the paper)
│   └── flamingo_model.py             # FlamingoConfig, FlamingoVLM
├── model/builder.py                  # load_pretrained_model -> (tokenizer, model, image processor, ctx)
├── data/                             # prompt + label masking around the shared COCO-MC / BabyView readers
├── training/                         # Hydra dataclasses, AdamW param groups, losses, DDP trainer, entry point
├── configs/                          # Hydra YAML tree (defaults = paper settings)
├── scripts/train_flamingo.sh         # SLURM launcher: one (condition, seed)
├── multimodal_feature_extractor.py   # eval adapter: caption log-likelihood scoring
├── run_mdb_eval.py                   # Machine-DevBench eval
└── run_devbench_eval.py              # DevBench eval
```

### Encoders

Each training corpus uses the DINOv2 and Phase-0 GPT-2 pretrained on that
corpus, the same pair as the corresponding LLaVA run. The GPT-2's own tokenizer
is loaded from its directory, plus an added `<image>` token. Train them with
the [DINOv2](../dinov2/README.md) and [LM](../lm_training/README.md) baselines.

### Data

The COCO-MC and BabyView readers are shared with the EgoVLP baseline: see
[`apps/baselines/egovlp/README.md`](../egovlp/README.md#data) for the
`COCO_MC_*`, `COCO_IMAGE_DIR` and `BABYVIEW_ROOT` variables and the one-time
MetaCLIP staging (`EXTRACTED_ROOT`). COCO-MC shuffle conditions use the same
seed-42 permutation files as the CLIP+ and LLaVA runs.

### Training

The paper reports 6 conditions × 3 seeds (42, 43, 44): `p0` (COCO-MC aligned),
`p25`, `p50`, `p75`, `p100` (fully shuffled) and `babyview`. Each run uses
4 GPUs (one process per GPU). The launcher has no `#SBATCH` defaults:

```bash
CONDITION=p50 SEED=42 \
COCOMC_VISION_TOWER_PATH=/path/to/coco_mc_dinov2/teacher_checkpoint.pth \
COCOMC_DECODER_INIT_PATH=/path/to/coco_mc_phase0_gpt2 \
EXTRACTED_ROOT=/path/to/scratch/coco_metaclip/extracted \
EGOBABYVLM_CKPT_DIR=/path/to/checkpoints \
sbatch --nodes=1 --ntasks-per-node=4 --gpus-per-node=4 --cpus-per-task=8 \
    --mem=256G --time=12:00:00 --qos=<your_qos> --account=<your_account> \
    apps/baselines/flamingo/scripts/train_flamingo.sh

# BabyView: CONDITION=babyview with BABYVIEW_{VISION_TOWER,DECODER_INIT}_PATH and BABYVIEW_ROOT.
# All 18 paper runs as one array (condition = idx % 6, seed = idx / 6):
sbatch --array=0-17 ... apps/baselines/flamingo/scripts/train_flamingo.sh
```

The config defaults are the paper's settings:

| | |
|---|---|
| Perceiver | 64 latents, depth 2, 8 heads × 64 |
| Gated cross-attention | before every GPT-2 layer (`cross_attn_every=1`); tanh gates pinned open at tanh(3) ≈ 0.995 (`gate_mode=frozen_open`) |
| Objective | teacher-forced caption cross-entropy, prompt `<image> {caption}`, loss on caption tokens only |
| Optimizer | AdamW, lr 1e-4, cosine decay to 0 per step, no warmup, grad clip 1.0; weight decay 0.01 except biases, LayerNorm parameters, gates and the Perceiver `latents` / `time_pos_emb` |
| Batch / epochs | 32 per GPU × 4 GPUs = 128 / 10 |
| Input | 224×224 resize, ImageNet normalization, no augmentation; prompt + caption ≤ 64 tokens; fp32 |

Direct Hydra entry point (`egobabyvlm-train-flamingo`), e.g. for a smoke test:

```bash
python -m apps.baselines.flamingo.training.train \
    name=flamingo_smoke data=babyview max_steps=3 data.batch_size=2 \
    model.vision_tower_path=/path/to/dinov2/teacher_checkpoint.pth \
    model.decoder_init_path=/path/to/phase0_gpt2 \
    checkpoint.save_dir=/tmp/flamingo_smoke
```

### Checkpoints

Each epoch writes `epoch_XXXX.pt` (last 3 kept), `latest.pt`, and `best.pt`
(lowest validation loss). Every `.pt` embeds the `FlamingoConfig`.
**The paper evaluates the final epoch** (`epoch_0009.pt`, identical to
`latest.pt` after a full run), not `best.pt`.

### Evaluation

Flamingo is generative, so image–text pairs are scored by the mean caption-token
log-likelihood under the LLaVA scoring prompt `<image>\nDescribe this
image. {caption}` (`evaluation/configs/model/flamingo.yaml`). A trial is
correct when the matching pair scores highest, using the same rule as LLaVA:
2AFC on Machine-DevBench; 2- or 4-way choice (and the Winoground text score) on
DevBench.

```bash
CKPT=/path/to/checkpoints/flamingo/flamingo_cocomc_p50_seed42/checkpoints/epoch_0009.pt

pixi run -e dev python -m apps.baselines.flamingo.run_mdb_eval \
    --model_path "$CKPT" --data_root "$MACHINE_DEVBENCH_DATA_ROOT" --out results/mdb.json

pixi run -e dev python -m apps.baselines.flamingo.run_devbench_eval \
    --model_path "$CKPT" --data_root "$DEVBENCH_DATA_ROOT" --out results/devbench.json
```

Both scripts reuse the repo's scoring and aggregation code
(`evaluation/multimodal/{machine_devbench,devbench}`) and load the model once
per checkpoint. The checkpoint config records where the DINOv2 tower and
tokenizer were loaded from; set `vision_tower_path` in the model config if
those paths moved.

### Citation

```bibtex
@inproceedings{alayrac2022flamingo,
  title     = {Flamingo: a Visual Language Model for Few-Shot Learning},
  author    = {Alayrac, Jean-Baptiste and Donahue, Jeff and Luc, Pauline and Miech, Antoine and Barr, Iain and Hasson, Yana and Lenc, Karel and Mensch, Arthur and Millican, Katherine and Reynolds, Malcolm and others},
  booktitle = {Advances in Neural Information Processing Systems},
  year      = {2022}
}
```

The Perceiver Resampler and gated cross-attention implementations are adapted
from [lucidrains/flamingo-pytorch](https://github.com/lucidrains/flamingo-pytorch)
by Phil Wang (MIT License).
