# Machine-DevBench — Benchmark Creation

A scalable, corpus-grounded pipeline for generating developmental
benchmarks that evaluate lexical and grammatical competence in
vision-language models. Machine-DevBench draws its vocabulary directly
from a model's training corpus, eliminating confounds between vocabulary
coverage and linguistic competence. Words are sampled across logarithmic
frequency bins covering the full long-tail distribution.

This package (`benchmark_creation/`) contains the **generation pipeline
only**. For evaluation, see the `evaluation/` package.

### Tasks

| Task | ID | What it tests |
|------|----|---------------|
| **Nouns** | `lex_nouns` | Word-to-image recognition, same-category distractor |
| **Adjectives** | `lex_adjectives` | Property recognition, antonym contrast |
| **Subject–Verb** | `gram_subject_verb` | Agent–action binding |
| **Subject–Adjective** | `gram_subject_adjective` | Property–object binding |
| **Negation** | `gram_negation` | "is X" vs. "is not X" |
| **Word Order** | `gram_order_matters` | Thematic role assignment (who does what to whom) |
| **Prepositions** | `gram_prepositions` | Spatial relation understanding |
| **Comparatives** | `gram_comparatives` | Comparative constructions |
| **Counting** | `gram_counting` | Numeral comprehension |
| **Embedded Relative** | `gram_embedded_relative` | Relative clause attachment |

All stimuli are generated in two visual styles (photorealistic and
cartoon).

### Installation

The full environment is pinned in [`pixi.toml`](../../pixi.toml).
Install the two pixi envs this pipeline needs from the repo root:

```bash
pixi install -e dev                  # main env
pixi install -e vllm                 # vLLM env (only needed for local LLM serving)
```

The launcher (`scripts/run_pipeline.sh`) resolves the env Pythons
directly at `.pixi/envs/{dev,vllm}/bin/python` — no `pixi run -e ...`
wrapper needed when invoking the orchestrator. For ad-hoc Python
commands, prefix them with `pixi run -e dev`.

### Quick start

> [!IMPORTANT]
> Submit with `sbatch`, not `bash`. The launcher scripts carry
> `#SBATCH` directives that only take effect under `sbatch`; running
> with `bash` executes on the login node with no GPU allocation.

```bash
# Verify installation
pixi run -e dev python -c "import apps.benchmark_creation; print(apps.benchmark_creation.__version__)"

# Check available tasks
pixi run -e dev python -c "from apps.benchmark_creation.task_registry import list_tasks; print(list_tasks())"

# One-time: SLURM logs land in MachineDevBench_logs/ (must exist beforehand)
mkdir -p MachineDevBench_logs

# Submit the full pipeline as a SLURM job
sbatch apps/benchmark_creation/scripts/run_pipeline.sh \
    --dataset coco --name COCO

# Check status / tail logs
squeue -u $USER
tail -f MachineDevBench_logs/slurm-<jobid>-pipeline.out
```

Run a single Python module without SLURM (for local debugging):

```bash
pixi run -e dev python -m apps.benchmark_creation.pipeline.create_vocabulary \
    --vocab-csv path/to/vocab_sorted.csv --output-dir data/coco --name COCO
```

See [`USAGE_GUIDE.md`](USAGE_GUIDE.md#slurm-submission) for the full
SLURM submission reference (per-stage examples, log management,
overriding SBATCH defaults).

Path overrides: set `BENCHMARK_CREATION_PATHS` to a YAML file whose
keys override `configs/paths.yaml`.

### Pipeline

The benchmark is built through a 5-stage pipeline, each with a launcher
script in `scripts/`:

```
Stage 1  Vocabulary curation     scripts/01_Create_Vocabulary/
Stage 2  Lexical tasks           scripts/02_Create_Lexical/
Stage 3  Grammatical tasks       scripts/03_Create_Grammatical/
Stage 4  Post-filtering          scripts/04_Post_Filtering/
Stage 5  Manifest generation     scripts/05_Manifest_Generation/
```

Configuration lives in `configs/paths.yaml` and `configs/styles.yaml`.
See [`USAGE_GUIDE.md`](USAGE_GUIDE.md) for the full walkthrough.

### Package structure

```
apps/benchmark_creation/
├── paths.py / task_registry.py     # path + task registry
├── configs/                        # paths.yaml + styles.yaml (image-gen prefixes)
├── utils/                          # vocabulary, FluxPipeline, SigLIP2/CLIP scoring, vLLM server
├── pipeline/                       # per-stage Python entry points
│   ├── create_vocabulary.py        # Stage 1
│   ├── lexical/                    # Stage 2 (build_nouns, build_adjectives, generate_*_images)
│   ├── grammatical/                # Stage 3 (build_benchmark, generate_images, prompts, filters)
│   ├── filtering/                  # Stage 4 (post_filter_{lexical,lexical_hard,grammatical})
│   └── manifests/                  # Stage 5 (generate_{lexical,grammatical})
└── scripts/                        # 5 numbered SLURM launcher directories + run_pipeline.sh
```

See [`USAGE_GUIDE.md`](USAGE_GUIDE.md) for the per-stage Python commands
and the full SLURM submission reference.
