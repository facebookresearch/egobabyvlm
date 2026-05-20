# Usage Guide

A detailed companion to [`README.md`](README.md): per-stage Python entry
points, SLURM submission patterns, and config reference. The README
covers the end-to-end `run_pipeline.sh` orchestrator; this guide covers
what you need when you want to run or override stages individually.

### Stage-by-stage commands

Each module below is a Hydra-free Python script with `argparse` flags.
The full pipeline (`run_pipeline.sh`) wires them together with vLLM
lifecycle + parallelism; you only need these directly for local
debugging or surgical re-runs.

<details>
<summary>Stage 1 — Vocabulary curation</summary>

POS-tags, frequency-bins, and filters the raw word frequencies.

```bash
pixi run -e dev python -m apps.benchmark_creation.pipeline.create_vocabulary \
    --vocab-csv path/to/vocab_sorted.csv \
    --output-dir MachineDevBench --name COCO
```

Output: `MachineDevBench/COCO_<timestamp>/longtail_wordlist.csv`.
</details>

<details>
<summary>Stage 2A — Lexical word lists</summary>

Nouns (WordNet semantic categorisation + LLM filtering for
child-appropriateness + stratified sampling) and adjectives (LLM
contrastive phrases). Both require a running vLLM server.

```bash
pixi run -e dev python -m apps.benchmark_creation.pipeline.lexical.build_nouns \
    --vocab-dir MachineDevBench/COCO_<timestamp> --name COCO \
    --api-base http://localhost:8000/v1 --model google/gemma-4-26B-A4B-it

pixi run -e dev python -m apps.benchmark_creation.pipeline.lexical.build_adjectives \
    --vocab-dir MachineDevBench/COCO_<timestamp> --name COCO \
    --api-base http://localhost:8000/v1 --model google/gemma-4-26B-A4B-it
```

Output: `Lexical/{Nouns,Adjectives}/word_list.json`.
</details>

<details>
<summary>Stage 2B — Lexical image generation</summary>

Generates images with Flux. Multi-GPU batched; resumable (existing
images are skipped on re-run).

```bash
pixi run -e dev python -m apps.benchmark_creation.pipeline.lexical.generate_noun_images \
    --data-dir MachineDevBench/COCO_<timestamp> \
    --model-id black-forest-labs/FLUX.2-klein-4B \
    --styles realistic cartoon --num-gpus 4

pixi run -e dev python -m apps.benchmark_creation.pipeline.lexical.generate_adj_images \
    --data-dir MachineDevBench/COCO_<timestamp> \
    --model-id black-forest-labs/FLUX.2-klein-4B \
    --styles realistic cartoon
```

Output: `Lexical/{Nouns,Adjectives}/{style}/`.
</details>

<details>
<summary>Stage 3A — Grammatical sentence pairs</summary>

3-LLM-call pipeline per category: word selection → pair generation →
LLM validation. Categories: `subject_verb`, `subject_adjective`,
`negation`, `order_matters`, `prepositions`, `comparatives`,
`counting`, `embedded_relative`.

```bash
pixi run -e dev python -m apps.benchmark_creation.pipeline.grammatical.build_benchmark \
    --vocab-dir MachineDevBench/COCO_<timestamp> --name COCO \
    --api-base http://localhost:8000/v1 --model google/gemma-4-26B-A4B-it
```

Output: `Grammatical/gram_{category}/sentence_list.json`.
</details>

<details>
<summary>Stage 3B — Grammatical image generation</summary>

Two images per trial with category-specific contrastive prompts.

```bash
pixi run -e dev python -m apps.benchmark_creation.pipeline.grammatical.generate_images \
    --data-dir MachineDevBench/COCO_<timestamp> \
    --model-id black-forest-labs/FLUX.2-klein-4B \
    --styles realistic cartoon --num-gpus 4
```

Output: `Grammatical/gram_{category}/imgs/{style}/seq_NN/img_{0,1}.png`.
</details>

<details>
<summary>Stage 4 — Post-filtering</summary>

Scores generated images against captions, drops poorly-aligned trials.
Lexical uses SigLIP2 image–caption alignment; grammatical uses a VLM
for depiction quality + distinguishability.

```bash
pixi run -e dev python -m apps.benchmark_creation.pipeline.filtering.post_filter_lexical \
    --data-dir MachineDevBench/COCO_<timestamp> --write-filtered

pixi run -e dev python -m apps.benchmark_creation.pipeline.filtering.post_filter_grammatical \
    --data-dir MachineDevBench/COCO_<timestamp> --write-filtered

# Optional: inspect score distributions
pixi run -e dev python -m apps.benchmark_creation.pipeline.filtering.compute_distributions \
    --data-dir MachineDevBench/COCO_<timestamp>
```

Output: `siglip2_scores_{style}.json`, `word_list_filtered_{style}.json`,
`vlm_scores_{style}.json`.
</details>

<details>
<summary>Stage 5 — Manifest generation</summary>

Assembles word lists, sentence lists, and image paths into
evaluation-ready manifests.

```bash
pixi run -e dev python -m apps.benchmark_creation.pipeline.manifests.generate_lexical \
    --data-dir MachineDevBench/COCO_<timestamp> --tasks nouns adjectives \
    --styles realistic cartoon

pixi run -e dev python -m apps.benchmark_creation.pipeline.manifests.generate_grammatical \
    --data-dir MachineDevBench/COCO_<timestamp> --styles realistic cartoon
```

Output: `manifest_{task}_{style}.json`.
</details>

### Configuration

#### `configs/paths.yaml`

Path configuration for the Python package. Override via
`BENCHMARK_CREATION_PATHS=<path/to/your_paths.yaml>`.

```yaml
outputs_root: ./MachineDevBench

# Dataset manifests (one path per corpus)
howto100m_manifest: path/to/howto100m_manifest.json
ego4d_manifest:     path/to/ego4d_manifest.json
coco_captions:      path/to/coco_captions.txt
babyview_manifest:  path/to/babyview_manifest.txt
```

#### `configs/styles.yaml`

Style prefixes prepended to image-generation prompts.

```yaml
cartoon:
  prefix: >-
    A simple children's book illustration, clean lines, bright colors,
    white background, no text, no watermark.

realistic:
  prefix: >-
    A clear, realistic photo of the following scene.
    Plain simple background, well-lit, easy to understand.
```

### SLURM submission

> [!IMPORTANT]
> Submit with `sbatch`, not `bash`. Every launcher script in `scripts/`
> (including `run_pipeline.sh`) starts with `#SBATCH` directives that
> request partition, GPUs, memory, and wall-time. Those directives only
> take effect under `sbatch`; running with `bash` ignores them and
> executes on the login node with no GPU allocation.

The shipped scripts intentionally omit `--qos`, `--account`, and
`--partition` since those are cluster-specific. Pass them on the
`sbatch` command line, or set `SBATCH_*` env vars
(e.g. `SBATCH_QOS`, `SBATCH_ACCOUNT`, `SBATCH_PARTITION`) before
submitting.

```bash
# Full pipeline — SLURM allocates GPUs, logs go to MachineDevBench_logs/
sbatch --qos=<your_qos> --account=<your_account> \
    apps/benchmark_creation/scripts/run_pipeline.sh --dataset coco --name COCO

# Per-stage submission works the same way
sbatch --qos=<your_qos> apps/benchmark_creation/scripts/02_Create_Lexical/run_build_nouns.sh \
    --vocab-dir MachineDevBench/COCO_<timestamp> --name COCO
```

#### Overriding SBATCH defaults

Flags passed before the script path override the in-script `#SBATCH`
headers:

```bash
# Short test run on 2 GPUs with a 4h wall-clock
sbatch --time=4:00:00 --gpus=2 \
    apps/benchmark_creation/scripts/run_pipeline.sh --test --dataset coco --name COCO
```

#### Logs

Each script writes to
`MachineDevBench_logs/slurm-<jobid>-<stage>.{out,err}`, relative to your
**submission directory** — submit from the repo root.

> [!WARNING]
> If the log directory does not exist, SLURM silently drops
> stdout/stderr. Always create it before your first submission:
>
> ```bash
> mkdir -p MachineDevBench_logs
> ```

#### Monitoring & cancelling

```bash
squeue -u $USER                                  # queued / running jobs
scontrol show job <jobid>                        # detailed job info
tail -f MachineDevBench_logs/slurm-<jobid>-*.out # live log
scancel <jobid>                                  # cancel
```

#### Running interactively (no SLURM)

Inside an existing interactive GPU allocation, `bash`-invocation is
fine — the `#SBATCH` lines are just comments to the shell:

```bash
srun --gpus=4 --time=2:00:00 --pty bash
bash apps/benchmark_creation/scripts/run_pipeline.sh \
    --dataset coco --name COCO --test
```

### Output layout

```
MachineDevBench/COCO_<timestamp>/
├── longtail_wordlist.csv                              # Stage 1
├── frequency_report.txt
├── Lexical/
│   ├── Nouns/
│   │   ├── word_list.json                             # Stage 2A
│   │   ├── word_list_filtered_<style>.json            # Stage 4
│   │   ├── siglip2_scores_<style>.json                # Stage 4
│   │   ├── manifest_nouns_<style>.json                # Stage 5
│   │   └── {style}/{category}/{word}.png              # Stage 2B
│   └── Adjectives/
│       └── ... (same shape with pos.png / neg.png per word)
├── Grammatical/
│   ├── gram_subject_verb/
│   │   ├── sentence_list.json                         # Stage 3A
│   │   ├── vlm_scores_<style>.json                    # Stage 4
│   │   └── imgs/{style}/seq_<NN>/{metadata.json,img_0.png,img_1.png}
│   ├── gram_negation/ ...
└── db_stats.json
```

### Extending

#### New grammatical category

1. Add an entry to `GRAMMATICAL_TEMPLATES` in
   `pipeline/grammatical/prompts.py` with `pos`, `template`, and
   `pair_mode` (`llm` = both captions LLM-generated, `deterministic` =
   LLM generates one, code derives the other).
2. (If needed) add word filters in
   `pipeline/grammatical/word_filters.py` and image rewriters in
   `pipeline/grammatical/rewriters.py`.

No other changes — `build_benchmark.py` iterates over the templates.

#### New lexical task type

1. New builder under `pipeline/lexical/` (template: `build_nouns.py`).
2. New image generator (template: `generate_noun_images.py`).
3. Register in `task_registry.py`.
4. Add or extend `manifests/generate_lexical.py`.

#### New image style

Add an entry to `configs/styles.yaml`; all image generators read styles
from there automatically.

#### New training corpus

Register its manifest path in `configs/paths.yaml` and add a
`get_<name>_manifest()` accessor in `paths.py`.
