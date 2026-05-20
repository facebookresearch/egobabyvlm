# DINOv2

A modified copy of [facebookresearch/dinov2](https://github.com/facebookresearch/dinov2)
adapted for OSS distribution under EgoBabyVLM.

## Why a copy lives here

This copy has diverged substantially from upstream
facebookresearch/dinov2 (BabyView / HowTo / Ego4D dataset support,
custom training-side hooks, ~3K LOC of additions and ~900 LOC of
deletions). The fork is the source of truth; it does not track a
specific upstream commit SHA.

Keeping the code under `apps/baselines/dinov2/third_party/dinov2/`
namespaces it so it doesn't shadow any pip-installed `dinov2` package,
and lets us pin exact versions of `fvcore`, `xformers`, etc. ourselves
(see `pixi.toml`).

## What's here

- The library + training-loss core (`train/ssl_meta_arch.py`,
  `models/vision_transformer.py`, `loss/`, `layers/`, `data/{augmentations,
  collate,masking}.py`, `fsdp/`, `distributed/`, `utils/{utils,param_groups,
  cluster,dtype,config}.py`, `configs/`).
- The training entrypoint (`train/train.py`, `run/train/train.py`,
  `run/submit.py`).
- Five dataset iterators (`data/datasets/{image_net,mscoco,ego4d,howto,
  babyview,extended,decoders}.py`).
- `hub/` + `hubconf.py` — needed at runtime by
  `torch.hub.load("facebookresearch/dinov2", ...)`. Upstream's hubconf
  does ``from dinov2.hub.X import ...``, which resolves into this tree
  once `dinov2` is pinned in `sys.modules`.

## What was stripped or replaced

- `run/eval/` — eval lives under `evaluation/` in this repository.
- `data/datasets/image_net_22k.py` — not used by the OSS pipeline;
  ImageNet goes through `image_net.py`.
- `logging/` — the original streamed logs to remote storage; replaced
  with plain `logging.FileHandler` writes.

## License

Apache License 2.0 (inherited from facebookresearch/dinov2).
