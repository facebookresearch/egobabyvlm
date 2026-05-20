# SwapBench: minimal-pair language and visual-property probes

Two corpus-grounded benchmark generators for evaluating language and
visual-property knowledge from minimal pairs:

| Benchmark | Origin | What it tests |
|---|---|---|
| **LT-Swap** (WordSwap, InflectionSwap, AgreementSwap) | Built around [`facebookresearch/lt-swap`](https://github.com/facebookresearch/lt-swap) | Lexical and morphological knowledge of long-tail words sampled from the model's training corpus (see [Algayres et al.](https://arxiv.org/abs/2502.10075)). |
| **VP-Swap** (4 properties: color, material, relative size, shape) | New, paper App. methods:vp-swap | Grounded knowledge of physical-object properties from the same long-tail vocabulary. |

Both pipelines drive an OpenAI-compatible LLM endpoint (a local
[vLLM](https://github.com/vllm-project/vllm) server is the canonical
choice) via the same async worker pool
(`apps/swapbench/utils/llm_runner.py`), so the model behind each step is
swappable from the CLI.

> [!NOTE]
> Pre-generated pair files for the four training corpora the paper uses
> (BabyView, Ego4D, HowTo, COCO-MC) ship as a GitHub release tarball;
> download them with
> `python -m scripts.eval_data.download_ltswap` (see
> [`docs/eval_data.md`](../../docs/eval_data.md)). The pipelines below
> are needed when training on a different corpus, a different snapshot
> of one of these, or with different preprocessing — pair files are
> vocabulary-conditional, so the long-tail bins must come from the
> training corpus your model actually saw.

### Layout

```
apps/swapbench/
├── third_party/
│   └── lt_swap/                  # Upstream LT-Swap (CC-BY-NC; see VERSION + LICENSE)
│       ├── generate_task/        # Per-stage prep + filter scripts (shipped unchanged from upstream)
│       ├── eval/                 # Upstream evaluator scripts (not on the OSS critical path)
│       ├── LICENSE
│       ├── README.upstream.md    # Upstream README, kept for reference
│       └── VERSION               # Upstream URL + pinned commit SHA
├── longtail_swap/
│   ├── build_word_lists.py       # Stage-0 corpus → wordlist + inflpairs + visualwords
│   └── generate.py               # End-to-end runner (task=wordswap or task=syntax)
├── visual_property_swap/
│   ├── prompts.py                # Per-property prompt templates
│   └── generate.py               # End-to-end VP-Swap runner (one or all four properties)
└── utils/
    └── llm_runner.py             # Async worker pool that drives the LT-Swap pipeline against a vLLM endpoint
```

The upstream code under `third_party/lt_swap/generate_task/` and
`third_party/lt_swap/eval/` is shipped unchanged from the source repo at
the SHA pinned in `VERSION`; refresh is a re-copy + bump. The Hydra
runners under `longtail_swap/` and the VP-Swap pipeline under
`visual_property_swap/` are first-party, drive the upstream scripts as
subprocesses, and use the async worker pool in `utils/llm_runner.py`
instead of the upstream `mp_main.py` orchestrator.

### CLI entry points

```text
egobabyvlm-swapbench-build-word-lists   # corpus → wordlist + inflpairs + visualwords
egobabyvlm-swapbench-lt-swap            # WordSwap or syntax (InflectionSwap + AgreementSwap)
egobabyvlm-swapbench-vp-swap            # VP-Swap, one property or all four
```

### Quickstart

The pipelines assume an OpenAI-compatible inference endpoint. Start a
vLLM server with your judge model first; the example below uses
[Llama-3.1-405B-Instruct](https://huggingface.co/meta-llama/Llama-3.1-405B-Instruct),
the model used in the EgoBabyVLM paper. The server lives in its own
pixi env (`vllm`) because vLLM's pinned torch / numpy differ from the
main `dev` env.

```bash
# Start a vLLM server (from the vllm pixi env; --local runs as a subprocess).
pixi run -e vllm python -m apps.benchmark_creation.utils.vllm_server \
    --local --model meta-llama/Llama-3.1-405B-Instruct --port 8000
```

Omit `--local` to submit a SLURM job instead (see `--help` for the
SLURM-only flags `--qos`, `--gpus`, `--cpus`, `--mem-gb`,
`--timeout-min`).

#### 1. Build the per-corpus word lists (one-time per corpus)

```bash
pixi run -e dev egobabyvlm-swapbench-build-word-lists \
    processor.data_dir=/path/to/corpus_text/ \
    processor.output_dir=/path/to/wordlists/
```

The corpus directory should contain plain `.txt` files (one shard per
file; the upstream `get_word_lists.py` parallelises over shards). Output:

```
/path/to/wordlists/
├── wordlists/             # per-shard intermediate JSONs
├── longtail_wordlist      # WordSwap candidate words
├── longtail_inflpairs     # InflectionSwap / AgreementSwap candidate pairs
├── longtail_visualnouns   # VP-Swap candidate words (word,freq per row, nouns only)
└── vocabulary             # corpus vocabulary with raw frequency counts
```

#### 2. Run LT-Swap

```bash
# WordSwap
pixi run -e dev egobabyvlm-swapbench-lt-swap \
    processor.task=wordswap \
    processor.wordlists_dir=/path/to/wordlists/ \
    processor.output_dir=/path/to/swapbench/wordswap/ \
    processor.model=meta-llama/Llama-3.1-405B-Instruct

# InflectionSwap + AgreementSwap (shared pipeline; produces both files)
pixi run -e dev egobabyvlm-swapbench-lt-swap \
    processor.task=syntax \
    processor.wordlists_dir=/path/to/wordlists/ \
    processor.output_dir=/path/to/swapbench/syntax/ \
    processor.model=meta-llama/Llama-3.1-405B-Instruct
```

Each stage writes to disk and is restartable — re-running the same
command picks up where the last one left off. The stages are documented
in the `apps/swapbench/longtail_swap/generate.py` module docstring.

#### 3. Run VP-Swap

```bash
# All four properties sequentially
pixi run -e dev egobabyvlm-swapbench-vp-swap \
    processor.visualnouns_path=/path/to/wordlists/longtail_visualnouns \
    processor.output_dir=/path/to/swapbench/vp_swap/ \
    processor.visual_property=all \
    processor.model=meta-llama/Llama-3.1-405B-Instruct

# Single property
pixi run -e dev egobabyvlm-swapbench-vp-swap \
    processor.visualnouns_path=/path/to/wordlists/longtail_visualnouns \
    processor.output_dir=/path/to/swapbench/vp_swap/ \
    processor.visual_property=color \
    processor.model=meta-llama/Llama-3.1-405B-Instruct
```

The first stage (the "is this word physical?" gate) is shared across
properties and only runs once.

### Output schema

LT-Swap pair files match the upstream format documented in
`apps/swapbench/third_party/lt_swap/README.upstream.md`. VP-Swap pair
files use the LT-Swap `visualswap` row layout
(`bin|VISUAL|w1|s1|i1|w2|s2|i2`), one file per property.

To evaluate a model on these pair files, set `LTSWAP_DATA_ROOT` to the
directory containing the four pair files and run the LT-Swap evaluator
(`evaluation/configs/eval/text/ltswap.yaml`) — see
[`docs/eval_data.md`](../../docs/eval_data.md).

### License

The upstream LT-Swap code under `third_party/lt_swap/` is CC-BY-NC and
retains all [upstream copyright headers](third_party/lt_swap/LICENSE).
Code under `apps/swapbench/longtail_swap/`,
`apps/swapbench/visual_property_swap/`, and `apps/swapbench/utils/` is
part of EgoBabyVLM and inherits the top-level [CC-BY-NC](../../LICENSE)
license.

Generated pair files are derivative works of the LLM that produced them
and are subject to that LLM's license. For Llama-class models, see the
[Llama 3.1 license](https://github.com/meta-llama/llama-models/blob/main/models/llama3_1/LICENSE).
