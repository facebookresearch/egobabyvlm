# Evaluation datasets

The evaluation pipeline resolves dataset paths at runtime via Hydra
environment variables. To run any eval, download the data once (using
the scripts in `scripts/eval_data/`), point the corresponding env var
at your cache, and Hydra will substitute it into the YAML.

### Cache layout

Default cache root is `~/.cache/egobabyvlm/eval_data/`. Override with
`EGOBABYVLM_CACHE` if you want it elsewhere.

```
~/.cache/egobabyvlm/eval_data/
├── mnist/                          # MNIST_ROOT
├── countbench/                     # COUNTBENCH_ROOT (HF-cache layout)
├── zorro/                          # ZORRO_DATA_ROOT (per-task .json JSONL files)
├── devbench/                       # DEVBENCH_DATA_ROOT
│   ├── sem-things/
│   ├── gram-trog/
│   ├── gram-winoground/
│   ├── lex-lwl/
│   ├── lex-viz_vocab/
│   └── sem-viz_obj_cat/
├── machine_devbench/               # MACHINE_DEVBENCH_DATA_ROOT
│   ├── Lexical/{Nouns,Verbs,Adjectives}/manifest_<pos>_<style>.json + images
│   └── Grammatical/gram_<category>/manifest_grammatical_<category>_<style>.json + images
├── ltswap/                         # parent of LTSWAP_DATA_ROOT (point env var
│   │                               # at one of the per-corpus subdirs below)
│   ├── babyview/{wordswap,agrswap,inflswap,vp_swap_combined}_pairs.txt
│   ├── ego4d/   (same 4 files)
│   ├── howto/   (same 4 files)
│   └── coco_mc/ (same 4 files)
└── cocostuff/                      # COCOSTUFF_ROOT
    ├── train2017/                  # 118k RGB JPEGs
    ├── val2017/                    # 5k RGB JPEGs
    └── annotations/
        ├── stuff_{train,val}2017.json         # 91 stuff cats
        ├── stuffthings_{train,val}2017.json   # 171 stuff+thing cats (built locally)
        ├── instances_{train,val}2017.json     # COCO things
        └── captions_{train,val}2017.json      # COCO captions (incidental)
```

### Automatable downloads

Each script accepts `--cache-dir <path>` and `--force` (re-download
even if cached). All write a `.downloaded` marker on completion so the
next run is a no-op.

| Dataset | Command | Env var the YAMLs read |
|---|---|---|
| MNIST | `pixi run -e dev python -m scripts.eval_data.download_mnist` | `MNIST_ROOT` |
| CountBench | `pixi run -e dev python -m scripts.eval_data.download_countbench` | `COUNTBENCH_ROOT` |
| Zorro | `pixi run -e dev python -m scripts.eval_data.download_zorro` | `ZORRO_DATA_ROOT` |
| DevBench | `pixi run -e dev python -m scripts.eval_data.download_devbench` | `DEVBENCH_DATA_ROOT` |
| Machine-DevBench | `pixi run -e dev python -m scripts.eval_data.download_machine_devbench` | `MACHINE_DEVBENCH_DATA_ROOT` |
| LT-Swap | `pixi run -e dev python -m scripts.eval_data.download_ltswap` | `LTSWAP_DATA_ROOT` (per-corpus subdir) |
| COCO-Stuff (~20 GB) | `pixi run -e dev python -m scripts.eval_data.download_cocostuff` | `COCOSTUFF_ROOT` |

Or grab everything at once:

```sh
pixi run -e dev python -m scripts.eval_data.download_all
```

#### Notes

- **DevBench**: the `gram-winoground` task is gated on HuggingFace
  (`facebook/winoground`). Run `huggingface-cli login` once and accept
  the dataset's terms before running the download. The other five
  DevBench tasks are open-access.
- **Machine-DevBench** is shipped as a single tarball on the
  `facebookresearch/egobabyvlm` GitHub release. The download script
  fetches it from `releases/download/Eval-Data/MachineDevBench.tar` and
  extracts the top-level `Lexical/` + `Grammatical/` tree the eval
  pipeline expects. Point `MACHINE_DEVBENCH_DATA_ROOT` at the
  extracted directory. Pass `--archive <path/to/MachineDevBench.tar>`
  if you've already staged the tarball locally.
- **LT-Swap** is shipped as one tarball
  (`releases/download/Eval-Data/LTSwap.tar`) containing per-corpus subdirs
  for the four training corpora the paper uses (`babyview/`, `ego4d/`,
  `howto/`, `coco_mc/`). Each subdir has the four pair files the eval
  reads: `wordswap_pairs.txt`, `agrswap_pairs.txt`,
  `inflswap_pairs.txt`, and `vp_swap_combined_pairs.txt` (matched by
  the eval's `vp_swap_*_pairs.txt` glob). Point `LTSWAP_DATA_ROOT` at
  **one** of the per-corpus subdirs (e.g.
  `~/.cache/egobabyvlm/eval_data/ltswap/babyview`), not the cache
  root. **Reproducibility caveat:** these pair files were generated
  against the paper's exact training-corpus snapshots and are intended
  primarily for reproducing / comparing to the paper. If your training
  data differs (different snapshot, subset, preprocessing,
  tokenization), the long-tail vocabulary may not match what your model
  actually saw — regenerate the pair files from your own corpus via the
  `apps/swapbench/` pipelines (see
  [`apps/swapbench/README.md`](../apps/swapbench/README.md)).
- **COCO-Stuff** is large; the script streams to a temporary
  `_downloads/` dir inside your cache and removes it after extraction.
  The eval (`semantic_seg_coco_171.yaml`) needs
  `stuffthings_{train,val}2017.json`, which the script builds at the
  end by merging the upstream COCO-Stuff JSON with the COCO
  `instances_*` JSON (no extra step required).
- **Zorro** is downloaded as raw paradigm `.txt` files from the
  [phueb/Zorro](https://github.com/phueb/Zorro) repo and converted to
  per-task BLiMP-style JSONLs the pipeline expects.

### Manual setup required

These datasets have access restrictions or non-trivial preprocessing.
Each YAML that depends on one references an env var the operator must
set to point at a pre-prepared dataset directory.

#### ImageNet (`IMAGENET_ROOT`)

ImageNet-1k is gated. Two prep paths are supported, both producing the
same `<IMAGENET_ROOT>/{train,val,labels.txt}` + `<IMAGENET_EXTRA>/`
layout that DINOv2's loader expects:

- **HuggingFace mirror (recommended)** — easier auth, fully scripted
  end-to-end.
- **Official ILSVRC2012 tarballs** — byte-identical filenames to legacy
  / published numbers, but more manual.

<details>
<summary>Recommended: HuggingFace recipe</summary>

1. Go to <https://huggingface.co/datasets/ILSVRC/imagenet-1k> and accept
   the ImageNet ToS (auto-approved).
2. `pixi run -e dev hf auth login` and paste a token with read access
   (create one at <https://huggingface.co/settings/tokens>).
3. Run the converter:

   ```sh
   pixi run -e dev python -m scripts.eval_data.prepare_imagenet_from_hf \
       --root /path/to/imagenet
   ```

   This downloads parquet shards (val 7 GB, train 140 GB), then writes
   the raw JPEG bytes from each row directly into
   `train/<wnid>/<wnid>_<idx>.JPEG` and
   `val/<wnid>/ILSVRC2012_val_<8d>.JPEG` — byte-identical to what the
   official tar route produces. Shard extraction parallelises over 8
   workers by default (override with `--workers N`). Disk usage:
   ~150 GB peak (HF cache + final layout); pass `--purge-hf-cache`
   after to reclaim the cache.

4. Generate DINOv2's `extra/` sidecar (see the shared step below).
5. Validate:

   ```sh
   pixi run -e dev python -m scripts.eval_data.validate_imagenet \
       --root $IMAGENET_ROOT --extra $IMAGENET_EXTRA
   ```
</details>

<details>
<summary>Alternative: official ILSVRC2012 tarballs</summary>

The cleanest path for byte-identical filenames. Grab the tars from
[image-net.org](https://image-net.org/download.php) (register, accept
the license, download from your account's links). Expected layout:

```
<IMAGENET_ROOT>/
├── labels.txt                                  # CSV "<wnid>,<class_name>" per line
├── train/<wnid>/<wnid>_<idx>.JPEG              # 1000 dirs (from per-class tars)
├── val/<wnid>/ILSVRC2012_val_<8d>.JPEG         # 1000 dirs (after valprep)
└── test/ILSVRC2012_test_<8d>.JPEG              # optional; only for the TEST split
```

**Step 1: train images.** `ILSVRC2012_img_train.tar` (138 GB) is a tar
of 1000 per-class tars. Unpack the outer tar, then unpack each inner
tar into a directory named after its WNID:

```sh
mkdir -p $IMAGENET_ROOT/train && cd $IMAGENET_ROOT/train
tar -xf /path/to/ILSVRC2012_img_train.tar
for f in *.tar; do
    wnid="${f%.tar}"
    mkdir -p "$wnid" && tar -xf "$f" -C "$wnid" && rm "$f"
done
```

**Step 2: val images.** `ILSVRC2012_img_val.tar` (6.3 GB) is a flat
list of 50k JPEGs. The standard PyTorch reorganizer regroups them into
per-WNID dirs using the official ground-truth file:

```sh
mkdir -p $IMAGENET_ROOT/val && cd $IMAGENET_ROOT/val
tar -xf /path/to/ILSVRC2012_img_val.tar
# Soumith's one-liner — fetches and runs the canonical valprep.sh.
wget -qO- https://raw.githubusercontent.com/soumith/imagenetloader.torch/master/valprep.sh | bash
```

(`valprep.sh` is a 50k-line `mv` script; safe to inspect before
running.)

**Step 3: `labels.txt`.** DINOv2 expects `<IMAGENET_ROOT>/labels.txt`
with one `<wnid>,<class_name>` row per class. Generate it from the
standard Caffe `synset_words.txt` (shipped with the ILSVRC2012 dev kit,
also mirrored at
[BVLC/caffe](https://github.com/BVLC/caffe/blob/master/data/ilsvrc12/get_ilsvrc_aux.sh)):

```sh
awk '{
    split($0, w, " ");
    wnid = w[1];
    rest = substr($0, length(wnid) + 2);
    split(rest, names, ",");
    print wnid "," names[1];
}' synset_words.txt > $IMAGENET_ROOT/labels.txt
```
</details>

##### Build the `extra/` index (both recipes)

DINOv2's loader requires a sidecar directory of `.npy` index files:

```sh
export IMAGENET_ROOT=/path/to/imagenet
export IMAGENET_EXTRA=/path/to/imagenet_extra
pixi run -e dev python -m scripts.eval_data.build_imagenet_extra \
    --root $IMAGENET_ROOT --extra $IMAGENET_EXTRA
```

This wraps DINOv2's own `ImageNet(...).dump_extra()` and writes
`class-ids-{TRAIN,VAL}.npy`, `class-names-{TRAIN,VAL}.npy`, and
`entries-{TRAIN,VAL}.npy`. Add `--splits train val test` if you also
prepared the test split.

After that, the KNN/Linear/ABX ImageNet evals will run against the
cache.

#### NYUv2 depth (`NYU_ROOT`)

The depth-estimation eval expects the standard NYUv2 monocular-depth
layout used by AdaBins, NeWCRFs, ZoeDepth, DPT, DepthAnything, and
DINOv3's `DATASETS.md`:

```
<NYU_ROOT>/
├── nyu_train.txt          # one line per sample: rgb_path depth_path focal_length
├── nyu_test.txt
├── <scene_instance>/      # train scenes (e.g. basement_0001a)
│   ├── rgb_<NNNNN>.jpg
│   └── sync_depth_<NNNNN>.png   # uint16 depth in millimeters
└── <scene_type>/          # test scenes (e.g. bathroom)
    ├── rgb_<NNNNN>.jpg
    └── sync_depth_<NNNNN>.png
```

<details>
<summary>Option 1 (recommended): prepare locally (~400 GB + Octave)</summary>

Download
[`nyu_depth_v2_raw.zip`](http://horatio.cs.nyu.edu/mit/silberman/nyu_depth_v2/nyu_depth_v2_raw.zip)
(~399 GB) from the NYU lab, then:

```sh
pixi run -e dev python -m scripts.eval_data.prepare_nyu \
    --raw-zip /path/to/nyu_depth_v2_raw.zip \
    --out $NYU_ROOT \
    --workers 8 \
    --write-splits
```

The script downloads the NYU Depth Toolbox V2 on first run (the
depth-projection math runs through Octave). With `--write-splits` it
additionally downloads `nyu_depth_v2_labeled.mat` (~3 GB) for the
official 654-image labeled-test extraction and writes `nyu_train.txt` +
`nyu_test.txt`. Pass `--raw-dir <extracted_dir>` instead of
`--raw-zip` if you already have the scenes extracted.
</details>

<details>
<summary>Option 2: BinsFormer's pre-prepared dump (~50 GB)</summary>

If 400 GB of disk + Octave is too much, the BinsFormer authors host a
[pre-prepared NYU dataset](https://drive.google.com/file/d/1xI9ksHzCC_kUz6Z4FL_b1ttgj3RVHGwW/view)
(~50 GB) with the same layout. Mirrored by Meta's DINOv3
[`DATASETS.md`](https://github.com/facebookresearch/dinov3/blob/main/DATASETS.md).
Extract it into `$NYU_ROOT` and you're done. (License unspecified;
verify it suits your use case.)
</details>

##### Validate

```sh
pixi run -e dev python -m scripts.eval_data.validate_nyu --nyu-root $NYU_ROOT
```

### Running an eval against your cache

After exporting the relevant env vars, the eval YAMLs Just Work:

```sh
export MNIST_ROOT=~/.cache/egobabyvlm/eval_data/mnist
pixi run -e dev python -m evaluation.eval_launcher \
    eval=vision/knn_mnist \
    model=dino \
    eval.output_dir=/tmp/knn_mnist_run \
    launcher.cluster=local
```
