# DINOv2 baseline

Self-supervised vision pretraining (DINO + iBOT objectives) and feature
extraction. Code forked from
[facebookresearch/dinov2](https://github.com/facebookresearch/dinov2) and
adapted for OSS distribution.

### Layout

```
apps/baselines/dinov2/
├── extractor.py                    # ImageFeatureExtractor wrapper for the eval pipeline
├── training/
│   ├── train.py                    # SSL training entry point
│   └── submit.py                   # Upstream Submitit-based SLURM submission
├── scripts/
│   └── train_dinov2.sh             # SLURM launcher (preferred)
└── third_party/dinov2/             # Upstream DINOv2 library + per-dataset configs
```

`third_party/dinov2/` carries its own README documenting the upstream
provenance and the local refresh procedure.

### Training

#### Shipped configs

The configs under `third_party/dinov2/configs/train/` cover the four
corpora used in the paper:

| Config                | Dataset env vars                                  |
|-----------------------|---------------------------------------------------|
| `vitb14_babyview.yaml`| `BABYVIEW_DATA_ROOT`, `BABYVIEW_EXTRA_ROOT`        |
| `vitb14_coco.yaml`    | `COCO_DATA_ROOT`, `COCO_EXTRA_ROOT`                |
| `vitb14_coco_mc.yaml` | `COCO_MC_DATA_ROOT`, `COCO_MC_EXTRA_ROOT`          |
| `vitb14_ego4d.yaml`   | `EGO4D_DATA_ROOT`, `EGO4D_EXTRA_ROOT`              |
| `vitb14_howto.yaml`   | `HOWTO_DATA_ROOT`, `HOWTO_EXTRA_ROOT`              |

Each YAML references the dataset via DINOv2's `dataset_path` string:

```yaml
dataset_path: Ego4D;split=TRAIN;root=${oc.env:EGO4D_DATA_ROOT};extra=${oc.env:EGO4D_EXTRA_ROOT}
```

The bundled dataset registry exposes `BabyView`, `CocoMc`, `Ego4D`,
`HowToSubset` (used as `HowTo`), `MSCOCO`, and `ImageNet`; adding a new
corpus is one class under `third_party/dinov2/data/datasets/` plus a
config YAML.

#### Submitting via SLURM (preferred)

```bash
CONFIG_FILE=apps/baselines/dinov2/third_party/dinov2/configs/train/vitb14_ego4d.yaml \
EGOBABYVLM_CKPT_DIR=/path/to/checkpoints \
EGO4D_DATA_ROOT=/path/to/ego4d/frames \
EGO4D_EXTRA_ROOT=/path/to/ego4d/extra \
sbatch --qos=<your_qos> --account=<your_account> \
    apps/baselines/dinov2/scripts/train_dinov2.sh
```

Tunables (env vars): `OUTPUT_DIR`, `OPTS` (extra Hydra `key=value`
overrides forwarded to the trainer). See the script header for defaults.

#### Running directly

```bash
pixi run -e dev python -m apps.baselines.dinov2.training.train \
    --config-file apps/baselines/dinov2/third_party/dinov2/configs/train/vitb14_ego4d.yaml \
    train.output_dir=/path/to/output \
    train.batch_size_per_gpu=64
```

<details>
<summary>Submitting via the bundled Submitit driver</summary>

The upstream DINOv2 Submitit entry point ships at `training/submit.py`.
Prefer the SLURM script above for new runs since it threads our env-var
conventions cleanly.

```bash
pixi run -e dev python -m apps.baselines.dinov2.training.submit \
    --config-file apps/baselines/dinov2/third_party/dinov2/configs/train/vitb14_ego4d.yaml \
    --partition <slurm_partition> \
    --output-dir /path/to/output \
    --ngpus 8
```
</details>

### Output layout

```
<OUTPUT_DIR>/
├── config.yaml                    # resolved DINOv2 config used for the run
├── eval/
│   └── training_<step>/
│       └── teacher_checkpoint.pth # periodic teacher snapshots
├── model_<step>.rank_<r>.pth      # FSDP-sharded student/teacher
└── logs/
    └── log.txt
```

The `teacher_checkpoint.pth` files plug directly into the
`DINOv2FeatureExtractor` at `apps/baselines/dinov2/extractor.py`.

### Evaluating a trained checkpoint

```bash
pixi run -e dev python -m evaluation.eval_launcher \
    eval=vision/knn_imagenet \
    model=dino \
    +model.kwargs.pretrained_weights=/path/to/teacher_checkpoint.pth \
    +model.kwargs.config_file=/path/to/output/config.yaml \
    eval.output_dir=$HOME/dinov2_eval
```

### License

Apache License 2.0 (inherited from facebookresearch/dinov2; see per-file
headers).
