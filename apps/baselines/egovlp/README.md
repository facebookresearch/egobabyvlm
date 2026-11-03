<!--
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
-->

# EgoVLP baseline

Fine-tunes the egocentric video-language model
[EgoVLP](https://github.com/showlab/EgoVLP) (Lin et al., 2022) on our training
corpora and evaluates it on Machine-DevBench. EgoVLP is a
[Frozen-in-Time](https://github.com/m-bain/frozen-in-time) (Bain et al., 2021)
dual encoder: a SpaceTimeTransformer video encoder and a DistilBERT text
encoder, projected to a shared 256-d space. We initialize from the released
EgoClip-pretrained checkpoint, feed still images as 1-frame clips, and score
image–text pairs by cosine similarity, the same path as the CLIP+ dual encoder.

Code originally derived from EgoVLP (MIT) and Frozen-in-Time (MIT); see file
headers for attribution.

> [!NOTE]
> The EgoClip checkpoint was pretrained on Ego4D, i.e. on external egocentric
> data that the other baselines never see. We do not redistribute it: download
> it from the EgoVLP repo. EgoVLP's code is MIT-licensed; the Ego4D data it was
> pretrained on is provided under the Ego4D license agreement (see
> [ego4d-data.org](https://ego4d-data.org/docs/start-here/)).

### Layout

```
apps/baselines/egovlp/
├── model/
│   ├── video_transformer.py   # vendored EgoVLP SpaceTimeTransformer + FrozenInTime
│   └── frozen.py              # EgoVLPModel wrapper, EgoVLPConfig, checkpoint save/load
├── data/
│   ├── coco_metaclip.py       # COCO-MC pairs + permutation variants
│   ├── stage_metaclip.py      # one-time MetaCLIP jpeg staging (COCO-MC)
│   ├── babyview.py            # BabyView (utterance, frame) pairs
│   ├── transforms.py          # 224x224 resize + ImageNet normalization
│   └── collate.py             # images -> (B, 1, C, H, W) clips
├── training/train.py          # contrastive fine-tuning entrypoint
├── trained_extractor.py       # eval-pipeline MultiModalFeatureExtractor
└── scripts/train_egovlp.sh    # SLURM launcher: one (setting, seed)
```

The model is vendored (ported to the repo's torch / timm / transformers
versions) with parameter names that match the released checkpoint, which loads
with `strict=True`. The upstream repo is not needed at runtime.

### Setup

1. **EgoClip checkpoint**: download `EgoVLP_PT_BEST` (EgoClip pretraining
   with EgoNCE) from the "Pretrained Weights" section of the
   [EgoVLP README](https://github.com/showlab/EgoVLP#-pretrained-weights) and
   set `EGOVLP_CHECKPOINT=/path/to/egovlp.pth`.
2. **DistilBERT**: `distilbert-base-uncased` is fetched from the Hugging Face
   hub. Without internet access, download it once and pass
   `--text_model /path/to/distilbert-base-uncased`.

### Data

Data locations are read from environment variables:

| Variable | Contents | Settings |
|---|---|---|
| `COCO_MC_MANIFEST` | `coco_to_metaclip.jsonl`: one row per COCO caption with its 5 retrieved MetaCLIP `matches` (`uuid`, `tar_path`) | COCO-MC |
| `COCO_MC_SPLIT_DIR` | `coco_metaclip_val.txt`: held-out validation captions (every other caption is train) | COCO-MC |
| `COCO_MC_PERMUTATION_DIR` | `permutation_metadata_cocomc_train_p{25,50,75,100}_seed42.json` | COCO-MC |
| `COCO_IMAGE_DIR` | COCO images as `{coco_id:012d}.jpg` | COCO-MC |
| `BABYVIEW_ROOT` | `{train,val}.json` manifests (from `egobabyvlm-build-clip-manifest`) + frames under `frames/resize_256/frames` | BabyView |

Every COCO caption is paired with its COCO image and each of its 5 MetaCLIP
images. The train split must contain exactly 495,380 pairs (asserted), so that
the permutation files, shared with the CLIP+ and LLaVA COCO-MC runs, line up.
For `coco_pX`, X% of the training pairs get another pair's caption; images
are never moved.

**Stage the MetaCLIP images once.** Reading MetaCLIP jpegs straight from their
tars starves the GPU, so extract them first (about 40 GB, 483k files). The
image set is the same for all COCO-MC settings:

```bash
python -m apps.baselines.egovlp.data.stage_metaclip \
    --manifest "$COCO_MC_MANIFEST" --scratch_root /path/to/scratch/coco_metaclip
# then pass EXTRACTED_ROOT=/path/to/scratch/coco_metaclip/extracted
```

### Training

The paper reports 6 settings × 3 seeds (42, 43, 44): `coco_p0` (aligned),
`coco_p25`, `coco_p50`, `coco_p75`, `coco_p100` (fully shuffled) and
`babyview`. The launcher has no `#SBATCH` defaults; pass your own resources on
the command line:

```bash
SETTING=coco_p50 SEED=42 \
EGOVLP_CHECKPOINT=/path/to/egovlp.pth \
EXTRACTED_ROOT=/path/to/scratch/coco_metaclip/extracted \
EGOBABYVLM_CKPT_DIR=/path/to/checkpoints \
sbatch --gpus=1 --cpus-per-gpu=8 --mem=128G --time=12:00:00 \
    --qos=<your_qos> --account=<your_account> \
    apps/baselines/egovlp/scripts/train_egovlp.sh

# All 18 paper runs as one array (setting = idx % 6, seed = idx / 6):
sbatch --array=0-17 ... apps/baselines/egovlp/scripts/train_egovlp.sh
```

The `training.train` defaults are the paper's hyperparameters:

| | |
|---|---|
| Init | released EgoClip checkpoint (`--pretrained_checkpoint`) |
| Objective | symmetric image–text InfoNCE, learnable temperature (init 0.07, logit scale ≤ 100) |
| Optimizer | AdamW, lr 3e-5 (constant), weight decay 0.2 |
| Batch / epochs | 32 on one GPU / 10 |
| Input | 224×224 resize, ImageNet normalization, no augmentation; 1-frame clips; text ≤ 64 tokens |

`torchrun` also works (`--batch_size` is per process), but the paper runs used
a single GPU.

### Checkpoints

`training/train.py` writes, after every epoch:

```
$EGOBABYVLM_CKPT_DIR/egovlp/<setting>/seed<seed>/
├── latest.pth   # end of the most recent epoch
└── best.pth     # epoch whose last training batch had the lowest loss
```

> [!NOTE]
> No validation pass is run. `best.pth` is chosen by the loss of each epoch's
> final training batch. **The paper's EgoVLP numbers were evaluated on
> `best.pth`.** (The other baselines use different conventions; e.g. LLaVA
> evaluates the final checkpoint.)

Each `.pth` holds `{"state_dict", "egovlp_config", "epoch", "step", "args"}`.
`load_egovlp_model(path)` rebuilds the model from `egovlp_config`, so no sidecar
files are needed.

### Evaluating a trained checkpoint

`evaluation/configs/model/egovlp.yaml` exposes the fine-tuned model to the eval
pipeline through `EgoVLPFeatureExtractor`:

```bash
pixi run -e dev python -m evaluation.eval_launcher \
    eval=multimodal/machine_devbench_pipeline \
    model=egovlp \
    model.kwargs.checkpoint_path=/path/to/checkpoints/egovlp/coco_p50/seed42/best.pth \
    name=egovlp_coco_p50_seed42
```

As a sanity check of the eval setup, off-the-shelf CLIP ViT-B/16
(`model=clip_example`) scores 80.1% Overall on Machine-DevBench.

### Citation

```bibtex
@article{kevin2022egovlp,
  title={Egocentric Video-Language Pretraining},
  author={Lin, Kevin Qinghong and Wang, Alex Jinpeng and Soldan, Mattia and Wray, Michael and Yan, Rui and Xu, Eric Zhongcong and Gao, Difei and Tu, Rongcheng and Zhao, Wenzhe and Kong, Weijie and others},
  journal={arXiv preprint arXiv:2206.01670},
  year={2022}
}

@InProceedings{Bain21,
  author       = "Max Bain and Arsha Nagrani and G{\"u}l Varol and Andrew Zisserman",
  title        = "Frozen in Time: A Joint Video and Image Encoder for End-to-End Retrieval",
  booktitle    = "IEEE International Conference on Computer Vision",
  year         = "2021",
}
```
