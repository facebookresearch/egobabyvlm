# apps/baselines/clip/

Contrastive trainer and downstream feature extractor for CLIP-style
image–text alignment. The trainer learns a shared embedding space for a
vision tower (DINOv2 or random ViT) and a text tower (BERT) using
InfoNCE, optionally co-trained with BERT masked language modelling on a
separate text corpus and/or DINOv2 self-supervised learning on the same
images.

### Trainer modes

| Mode               | Losses                       | Notes                                                                                                                                       |
|--------------------|------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------|
| `contrastive`      | InfoNCE                      | Vanilla CLIP-style; one optimizer.                                                                                                          |
| `interleaved_lm`   | InfoNCE + BERT MLM           | Alternates contrastive batches with MLM batches from a separate text-only corpus.                                                           |
| `interleaved_dino` | InfoNCE + DINOv2 SSL         | Alternates contrastive batches with DINOv2 SSL on the same images; teacher backbone is copied into the CLIP vision encoder after each SSL block. |
| `triple`           | InfoNCE + BERT MLM + DINOv2  | Round-robin all three.                                                                                                                      |

Schedule is set under `mode.interleave`, e.g.
`mode.interleave={contrastive: 4, mlm: 1}` runs four contrastive steps
then one MLM step.

### Layout

```
apps/baselines/clip/
├── configs/
│   ├── config.yaml         # default composition (mode=contrastive)
│   ├── mode/               # contrastive / interleaved_lm / interleaved_dino / triple
│   ├── model/              # embedding_dim, temperature, normalize_features
│   ├── text_encoder/       # bert_base.yaml (BERT-base TextEncoder)
│   ├── vision_encoder/     # hub_dinov2_{vits,vitb,vitl}14, custom_dinov2, random_vit_b14
│   ├── data/               # coco / ego4d / howto manifest schemas
│   ├── optim/              # AdamW + cosine schedule
│   ├── text_only_data/     # text-only corpus for the MLM head
│   ├── dinov2/             # vitb14_coco.yaml — bundled DINOv2 SSL config
│   └── checkpoint/, wandb/
├── modeling/               # text + vision encoders, multimodal model, MLM head, DINOv2 SSL wrapper
├── data/                   # caption datasets, text-only dataset, transforms, collate
├── training/               # trainer loop, optimizer factories, interleave scheduler, checkpoint I/O
├── scripts/                # checkpoint conversion utilities (e.g. egobabyvlm-export-text-encoder-to-hf)
└── extractor.py            # downstream ImageFeatureExtractor wrapper
```

### Quickstart

#### Single-GPU contrastive on COCO

```bash
pixi run -e dev egobabyvlm-train-contrastive \
    name=coco_baseline \
    data=coco \
    data.train_dataset.manifest_path=/data/coco/karpathy_train.json \
    data.train_dataset.image_root=/data/coco/all_images \
    data.val_dataset.manifest_path=/data/coco/karpathy_val.json \
    checkpoint.save_dir=$HOME/runs/coco_baseline
```

#### Multi-GPU triple mode on Ego4D

```bash
pixi run -e dev torchrun --standalone --nproc-per-node=4 \
    -m apps.baselines.clip.training.train \
    name=ego4d_triple mode=triple \
    data=ego4d \
    data.train_dataset.manifest_path=/data/ego4d/train.json \
    data.train_dataset.image_root=/data/ego4d/frames_1fps \
    data.val_dataset.manifest_path=/data/ego4d/val.json \
    +text_only_data=default text_only_data.train_file=/data/ego4d/narrations.txt \
    +dinov2=vitb14_coco \
    checkpoint.save_dir=$HOME/runs/ego4d_triple
```

The `data=` selector resolves to `configs/data/{coco,ego4d,howto}.yaml`.
COCO uses the Karpathy schema (`images[*].sentences`); Ego4D / HowTo100M
use the multi-frame-per-utterance schema produced by
`egobabyvlm-build-clip-manifest` (see
[`apps/data_preprocessing/`](../../data_preprocessing/README.md)).

#### Resume from a checkpoint

```bash
pixi run -e dev egobabyvlm-train-contrastive \
    name=coco_baseline \
    ... \
    checkpoint.resume_from=$HOME/runs/coco_baseline/latest.pt
```

### Checkpoint format

Each checkpoint is a single `.pt` file containing the multimodal model
state, optionally an MLM head + DINOv2 SSL state, all optimizer and
scheduler state, the resolved Hydra config, and metadata (epoch, step,
best validation loss). The embedded `config` carries the full `_target_`
for both encoders, so loading does **not** need any sidecar config files.

Default layout:

```
$checkpoint.save_dir/
├── latest.pt       # last completed step
├── epoch_0000.pt   # one per epoch
├── epoch_0001.pt
├── ...
└── best.pt         # best validation loss
```

Set `checkpoint.keep_last=N` to retain only the most recent N `epoch_*`
files.

<details>
<summary>Loading a trained checkpoint in Python</summary>

```python
import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf
from apps.baselines.clip.modeling import MultiModalModel

payload = torch.load("/path/to/checkpoint.pt", weights_only=False)
cfg = OmegaConf.create(payload["config"])

text_encoder = instantiate(cfg.model.text_encoder)
vision_encoder = instantiate(cfg.model.vision_encoder)
model = MultiModalModel(
    vision_encoder, text_encoder,
    normalize_features=cfg.model.normalize_features,
    temperature=cfg.model.temperature,
    fix_temperature=cfg.model.fix_temperature,
)
model.load_state_dict(payload["model_state_dict"])
model.eval()
```
</details>

### Datasets supported

COCO (Karpathy split), Ego4D, and HowTo100M ship as concrete
instantiations. Adding a new caption dataset is one Python class + one
YAML — see `apps/baselines/clip/data/captions.py`.

### Exporting the text encoder for `lm_eval`

The text-side evals (Zorro, LT-Swap) load HuggingFace
`BertForMaskedLM` from a directory on disk. To convert a trained
contrastive checkpoint:

```bash
pixi run -e dev egobabyvlm-export-text-encoder-to-hf \
    --checkpoint /path/to/contrastive.pt \
    --output-dir /path/to/hf_bert_dir
```
