# EgoBabyLLaVA — minimal LLaVA baseline

A small-scale [LLaVA](https://github.com/haotian-liu/LLaVA) implementation
built around:

- **Language model**: GPT-2 (trainable from scratch via [`apps/baselines/lm_training/`](../lm_training/))
- **Vision encoder**: DINOv2 ViT-B/14 (off-the-shelf from `torch.hub`, or
  your own custom-trained checkpoint)

Code originally derived from haotian-liu's LLaVA (Apache-2.0); see file
headers for attribution.

### Layout

```
apps/baselines/llava/
├── model/
│   ├── language_model/llava_gpt2.py         # GPT-2 + LLaVA arch glue
│   ├── multimodal_encoder/dinov2_encoder.py # DINOv2 vision tower
│   ├── multimodal_projector/builder.py      # MLP projector (vision → LLM dim)
│   ├── llava_arch.py                        # core multimodal arch with freeze flags
│   └── builder.py                           # load_pretrained_model() helper
├── train/
│   └── train.py                             # Phase 1 / Phase 2 trainer
├── feature_extractor.py                     # eval-pipeline ImageFeatureExtractor wrapper
├── scripts/
│   ├── phase1_pretrain.sh                   # SLURM launcher: Phase 1
│   ├── phase2_finetune.sh                   # SLURM launcher: Phase 2
│   ├── deepspeed_zero2_config.json
│   └── deepspeed_zero3_offload_config.json
├── constants.py                             # IMAGE_TOKEN_INDEX, IGNORE_INDEX, etc.
└── utils/                                   # logging helpers
```

> [!NOTE]
> Phase 0 (GPT-2 from scratch) lives one level up at
> [`apps/baselines/lm_training/`](../lm_training/) so the LM trainer can
> be reused for non-LLaVA work.

### Training pipeline

The full training is three phases. Each script is SLURM-ready (no
`#SBATCH --qos` or `#SBATCH --account` defaults — pass them on the
`sbatch` command line).

#### Phase 0: GPT-2 from scratch

See [`apps/baselines/lm_training/README.md`](../lm_training/README.md).
Produces the GPT-2 backbone consumed by Phase 1 / Phase 2.

#### Phase 1: projector pretraining

Trains only the multimodal projector with the vision tower and LLM frozen.

```bash
GPT2_MODEL=/path/to/phase0/output \
DATA_PATH=/path/to/coco_llava_train.json \
IMAGE_FOLDER=/path/to/coco/images \
sbatch --gpus=1 --mem=64G --time=12:00:00 \
    --qos=<your_qos> --account=<your_account> \
    apps/baselines/llava/scripts/phase1_pretrain.sh
```

#### Phase 2: full finetune

Unfreezes the LLM (vision tower stays frozen) and finetunes against the
same data using the Phase 1 projector as initialisation.

```bash
GPT2_MODEL=/path/to/phase0/output \
PRETRAIN_PROJECTOR=/path/to/phase1/output/mm_projector.bin \
DATA_PATH=/path/to/coco_llava_train.json \
IMAGE_FOLDER=/path/to/coco/images \
sbatch --gpus=1 --mem=64G --time=12:00:00 \
    --qos=<your_qos> --account=<your_account> \
    apps/baselines/llava/scripts/phase2_finetune.sh
```

### Vision tower options

Both Phase 1 and Phase 2 accept either:

1. **Off-the-shelf DINOv2** from `torch.hub` (ImageNet pretrained):
   ```bash
   VISION_TOWER=dinov2_vitb14 ./scripts/phase1_pretrain.sh
   ```
2. **A custom DINOv2 checkpoint** trained via the EgoBabyVLM DINOv2 stack
   (under `apps/baselines/dinov2/third_party/dinov2/`). The checkpoint
   directory must contain a `config.yaml`:
   ```bash
   VISION_TOWER=dinov2_vitb14 \
   VISION_TOWER_PATH=/path/to/dinov2/teacher_checkpoint.pth \
       ./scripts/phase1_pretrain.sh
   ```

### Evaluating a trained checkpoint

The vision tower of a trained LLaVA checkpoint is exposed via
`LlavaVisionFeatureExtractor` so the existing
[`evaluation/`](../../../evaluation/) pipeline (KNN, linear, ABX, depth,
semantic segmentation) can score it directly:

```bash
pixi run -e dev python -m evaluation.eval_launcher \
    eval=vision/knn_imagenet \
    model=llava_vision \
    +model.kwargs.model_path=/path/to/phase2/output \
    eval.output_dir=$HOME/egobabyvlm_eval
```

The `model=llava_vision` selection picks
[`evaluation/configs/model/llava_vision.yaml`](../../../evaluation/configs/model/llava_vision.yaml),
which Hydra-instantiates `LlavaVisionFeatureExtractor`.

### Data format

LLaVA-format JSON: a list of dicts, each with an `image` (path, relative
to `IMAGE_FOLDER`) and a `conversations` list (LLaVA-style turns with
`<image>` placeholders). See the upstream LLaVA repo for the exact schema.

The Phase 0 GPT-2 trainer expects a plain-text file with one caption per
line.
