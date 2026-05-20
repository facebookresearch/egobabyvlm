# apps/alignment_scoring/

Pipelines for measuring image / video – text alignment with off-the-shelf
and fine-tuned models. Used to filter and re-caption the contrastive
training data described in the EgoBabyVLM paper.

### Pipelines

| CLI                              | Model family                                                       | What it does                                                                                  |
|----------------------------------|--------------------------------------------------------------------|-----------------------------------------------------------------------------------------------|
| `alignment-clip-scoring`         | CLIP / [Perception Encoder](https://arxiv.org/abs/2504.13181)      | Cosine sim on matched vs shuffled (image, caption) pairs, then bootstrap JS divergence.       |
| `alignment-sts-scoring`          | [SONAR](https://arxiv.org/abs/2308.11466)                          | Same matched-vs-shuffled aggregation on text-only pairs (e.g. originals vs PLM re-captions).  |
| `alignment-captioning`           | [Perception-LM](https://arxiv.org/abs/2504.13180)                  | Re-caption a manifest with PLM and write the result back into a manifest copy.                |
| `alignment-vqa-scoring`          | Perception-LM                                                      | Score matched vs shuffled by `P("Yes")` for `"Does this figure show '{caption}'?"`; same JS.  |
| `alignment-finetune-lora`        | CLIP / PE                                                          | LoRA-finetune a CLIP-style encoder on (image, caption) pairs.                                 |
| `alignment-create-shuffled-manifest` | —                                                              | Build the shuffled manifest required by the scoring pipelines.                                |

The three scoring pipelines schedule **matched** + **shuffled**
processors in parallel via
[Stopes](https://github.com/facebookresearch/stopes) and aggregate the
two distributions into JS-divergence + KL stats. The output
`results.yaml` includes the divergence summary plus per-pair CSVs.

### Layout

```
apps/alignment_scoring/
├── configs/
│   ├── pipeline/         # one YAML per pipeline (clip_scoring, sts_scoring, ...)
│   ├── dataset/          # full datasets returning (media, text, media_id)
│   ├── dataset_path/     # path-only variants (for STS, captioning, VQA)
│   └── model/            # vit_b16_openai, pe_core_bigg, plm_1b, plm_8b
├── data/                 # COCO / video / text-pair datasets + collate
├── modeling/             # PLMGenerationModule + PackedCausalTransformerGenerator
├── pipelines/            # one Hydra entry point per CLI
├── scripts/              # manifest tooling (create_shuffled_manifest)
└── third_party/perception_models/   # FAIR Noncommercial Research License
```

### Models

| Config YAML        | Used by                                          | Notes                                              |
|--------------------|--------------------------------------------------|----------------------------------------------------|
| `vit_b16_openai`   | clip_scoring, finetune_lora (default)            | Off-the-shelf OpenAI CLIP ViT-B/16, useful for smoke tests. |
| `pe_core_bigg`     | clip_scoring (default), finetune_lora            | Perception Encoder Core bigG-14 at 448 px (paper). |
| `plm_1b`           | captioning, vqa_scoring                          | Perception-LM 1B, smaller variant for iteration.   |
| `plm_8b`           | captioning, vqa_scoring (default)                | Perception-LM 8B (paper).                          |

STS scoring uses SONAR's `text_sonar_basic_encoder` directly (set inside
`configs/pipeline/sts_scoring.yaml`, no model YAML).

### Datasets

Two manifest formats, distinguished by extension:

- **JSON** — COCO captions (top-level `images` and `annotations` arrays)
  or Karpathy split (`images[*].sentences`).
- **CSV** — must contain `clip_filename` and `utterance` columns.

For matched-vs-shuffled scoring you need two manifests over the same
media: the original ("matched") and one with captions shuffled across
other media. Generate the shuffled side deterministically:

```bash
pixi run -e dev alignment-create-shuffled-manifest \
    --manifest-path /data/coco/captions_train2017.json \
    --output-path   /data/coco/captions_train2017_shuffled.json \
    --type json \
    --random-seed 42
```

Available `--type` values: `json` (COCO `images`+`annotations`),
`karpathy_json` (Karpathy `images[*].sentences`), `csv` (CSV with
`clip_filename` + `utterance`), and `karpathy_json_with_permutation`
(replays an externally-provided permutation map).

### Quickstart — CLIP scoring on COCO

```bash
pixi run -e dev alignment-clip-scoring \
    name=coco_smoke \
    matched_processor.data.dataset.manifest_path=/data/coco/captions_train2017.json \
    matched_processor.data.dataset.dataset_dir=/data/coco/train2017 \
    shuffled_processor.data.dataset.manifest_path=/data/coco/captions_train2017_shuffled.json \
    shuffled_processor.data.dataset.dataset_dir=/data/coco/train2017 \
    model@matched_processor.model=vit_b16_openai \
    model@shuffled_processor.model=vit_b16_openai
```

To submit on SLURM, add `launcher.cluster=slurm` plus the per-cluster
overrides (e.g. `launcher.update_parameters.slurm_qos=...`,
`launcher.update_parameters.slurm_account=...`). Stopes job arrays
auto-shard by `num_items_per_chunk` (default 2000).

### Output

Each pipeline writes to `output_dir`:

```
<output_dir>/
├── results.yaml                       # JSD + KL + per-side mean/std
├── js_bootstrap_distribution.npy      # bootstrap JS samples
├── similarity_histogram.png           # KDE plot (matched vs shuffled)
├── cosine_similarities.csv            # CLIP scoring
├── sts_results_{matched,shuffled}.csv # STS scoring
├── vqa_results_{matched,shuffled}.csv # VQA scoring
└── recaptioned.json                   # captioning
```

### Adding a new dataset

1. Subclass `CaptionsMediaDataset` in `apps/alignment_scoring/data/` and
   return `(media, text, media_id)` triples.
2. Add `configs/dataset/<name>.yaml` with `_target_` pointing at the
   class.
3. For STS / captioning / VQA, also add a path-only variant returning
   `(media_path, text, media_id)` plus `configs/dataset_path/<name>.yaml`.

No pipeline changes needed.

### Tests

```bash
pixi run -e dev pytest -q tests/apps/alignment_scoring/         # unit tests
pixi run -e dev pytest -m gpu tests/apps/alignment_scoring/     # GPU smoke
```

GPU smoke tests download ViT-B/16 + SONAR-text on first run.

### perception_models

`third_party/perception_models/` ships an in-tree snapshot of
[FAIR's perception_models](https://github.com/facebookresearch/perception_models)
needed for PLM captioning + VQA scoring; released under the FAIR
Noncommercial Research License. See
`third_party/perception_models/README.md` for what's included and the
refresh script.
