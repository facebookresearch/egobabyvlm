# perception_models

In-tree snapshot of
[facebookresearch/perception_models](https://github.com/facebookresearch/perception_models)
needed by `apps/alignment_scoring/modeling/plm.py` to load Perception-LM
checkpoints for captioning and VQA scoring.

**Upstream revision**: `3e352cca660658d4b5c90f42a7808b11469e4c66`

## Why a copy lives here

perception_models has no `pyproject.toml` and pins old, conflicting
versions of numpy / opencv / transformers in `setup.py`. Worse, its
top-level packages are named `core/` and `apps/` — the same names as
our top-level packages, so we can't install it editably from PyPI/git
without an import collision.

Shipping it in-tree at
`apps/alignment_scoring/third_party/perception_models/` under our own
namespace avoids the shadowing and lets us pick exact pins ourselves
(see `pixi.toml`).

## What's here

Closure transitively reachable from `plm.py`'s direct imports plus the
PE-Core scoring loader at `apps/benchmark_creation/utils/vision_scoring.py`:

- `apps/plm/{tokenizer.py, transformer.py}` — PLM entry points
- `core/{args.py, checkpoint.py, transformer.py, tokenizer.py, utils.py}`
- `core/data/conversation.py`
- `core/transforms/{image_transform.py, video_transform.py}`
- `core/vision_encoder/{config.py, pe.py, rope.py, tokenizer.py, transforms.py}`
- `core/vision_encoder/bpe_simple_vocab_16e6.txt.gz` — BPE vocabulary
  loaded by `tokenizer.py` via `__file__`-relative path
- `core/vision_projector/{base.py, mlp.py}`

Two files are stubbed because the originals pull in heavy unused tooling:

- `core/distributed.py` — only `get_is_master()` is needed
- `core/probe.py` — only `log_stats()` (a no-op for inference) is needed

## Refreshing

Run `apps/alignment_scoring/third_party/perception_models/refresh.py`
after bumping the upstream SHA in this README. The script copies the
closure with `from core.X` imports rewritten to
`from apps.alignment_scoring.third_party.perception_models.core.X`,
plus any non-Python data files listed in `BUNDLED_DATA_FILES` (shipped
unchanged next to the modules that load them).

## Licenses

- `LICENSE.PE` — Perception Encoder weights and code: FAIR Noncommercial Research License
- `LICENSE.PLM` — Perception Language Model weights and code: FAIR Noncommercial Research License

These licenses are non-commercial; downstream commercial use is not permitted.
