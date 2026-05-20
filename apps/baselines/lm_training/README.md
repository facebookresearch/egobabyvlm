# Language-model pretraining baselines

Train language models from scratch on a plain-text corpus (one utterance
per line). The trained checkpoints feed downstream stacks: GPT-2 backs
the LLaVA multimodal baseline at `apps/baselines/llava/`, BERT backs the
contrastive trainer's `interleaved_lm` mode at
`apps/baselines/clip/training/`.

| Model | Trainer                                       | Tokenizer training                                                |
|-------|-----------------------------------------------|-------------------------------------------------------------------|
| GPT-2 | [`train/train_gpt2.py`](train/train_gpt2.py)  | bundled (retrains byte-level BPE from `gpt2`)                     |
| BERT  | [`train/train_bert.py`](train/train_bert.py)  | [`scripts/train_bert_tokenizer.py`](scripts/train_bert_tokenizer.py) (WordPiece) |

### GPT-2 from scratch

Trains GPT-2 and (by default) retrains a byte-level BPE tokenizer from
the standard GPT-2 base on your training corpus. Pass
`--tokenizer_name <hf_id_or_path>` to use an existing tokenizer instead.

#### SLURM (preferred)

```bash
EGOBABYVLM_DATA_DIR=/path/to/your/data \
EGOBABYVLM_CKPT_DIR=/path/to/your/checkpoints \
sbatch --qos=<your_qos> --account=<your_account> \
    apps/baselines/lm_training/scripts/phase0_train_gpt2.sh
```

The script reads `${EGOBABYVLM_DATA_DIR}/coco_captions_{train,val}.txt`
by default (or `coco_captions_{format}_{train,val}.txt` when
`PHASE0_FORMAT` is set). Tunable env vars: `PHASE0_FORMAT`,
`TOKENIZER_MODE` (`custom` / `mistral`), `SEED`, `LR`, `BS`, `GACC`,
`EPOCHS`. See the script header for the full list.

#### Running directly

```bash
pixi run -e dev python -m apps.baselines.lm_training.train.train_gpt2 \
    --train_file /path/to/corpus_train.txt \
    --validation_file /path/to/corpus_val.txt \
    --output_dir /path/to/output \
    --vocab_size 52000 --do_train --do_eval --bf16 \
    --load_best_model_at_end
```

`--load_best_model_at_end` saves the best-eval-loss checkpoint as the
root `model.safetensors`. Without it, the saved weights are the final
overfit epoch and downstream Zorro / LT-Swap scores drop by 5–10 points.
See `python -m apps.baselines.lm_training.train.train_gpt2 --help` for
every HF `Trainer` flag; defaults match `phase0_train_gpt2.sh`.

> [!IMPORTANT]
> When the GPT-2 you're training will back LLaVA Phase 1 / 2, prepend
> the LLaVA prompt prefix to each training line — e.g.
> `"Describe this image. <utterance>"`. The downstream LLaVA model
> conditions on that exact prefix at every multimodal step, so the LM
> needs to see the same distribution during phase 0.

### BERT from scratch (MLM)

Three steps, each with its own script.

#### 1. Train a WordPiece tokenizer on your corpus

```bash
pixi run -e dev python -m apps.baselines.lm_training.scripts.train_bert_tokenizer \
    /path/to/output/tokenizers/bert_corpus \
    --train_file /path/to/corpus_train.txt \
    --val_file   /path/to/corpus_val.txt \
    --vocab_size 30522
```

Inherits algorithm + special-token layout from `bert-base-cased` by
default; override with `--base_tokenizer <hf_id>`.

#### 2. Build a fresh BERT config

```bash
pixi run -e dev python -m apps.baselines.lm_training.scripts.create_bert_config \
    /path/to/output/configs/bert_base
```

Emits a `BertConfig` matching `bert-base-cased` (12 layers, 768 hidden,
30522 vocab). Override architecture knobs with `--hidden_size`,
`--num_hidden_layers`, `--num_attention_heads`, `--intermediate_size`,
`--max_position_embeddings`, etc.

#### 3. Run the MLM trainer

```bash
TRAIN_FILE=/path/to/corpus_train.txt \
VAL_FILE=/path/to/corpus_val.txt \
TOKENIZER_FOLDER=/path/to/output/tokenizers/bert_corpus \
CONFIG_FOLDER=/path/to/output/configs/bert_base \
EGOBABYVLM_CKPT_DIR=/path/to/output \
sbatch --qos=<your_qos> --account=<your_account> \
    apps/baselines/lm_training/scripts/train_bert.sh
```

Tunables (env vars): `MODEL_DIR`, `LR`, `NUM_TRAIN_EPOCHS`,
`PER_GPU_BATCH_SIZE`, `MLM_PROBABILITY`, `SEED`, `NUM_GPUS` (enables
multi-GPU DDP), `EGOBABYVLM_LOG_DIR`. See the script header.

The trained BERT plugs into the contrastive trainer's
`text_encoder.hf_model_name` config — point that at the output dir.

### Output layout

```
<EGOBABYVLM_CKPT_DIR>/
├── phase0_gpt2/gpt2_<tok_tag>_<format_tag>/   # GPT-2 trainer
│   ├── config.json
│   ├── model.safetensors
│   ├── tokenizer/                              # retrained BPE
│   └── ...
└── bert_mlm/                                   # BERT trainer
    ├── config.json
    ├── model.safetensors
    └── ...
```

- GPT-2 output → LLaVA Phase 1 / Phase 2's `GPT2_MODEL` env var.
- BERT output → the contrastive trainer's `text_encoder.hf_model_name`
  config.
