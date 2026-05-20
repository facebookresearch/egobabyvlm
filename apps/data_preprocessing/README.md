# Data preprocessing

Four pipelines for turning raw video data into the contrastive training
manifests consumed by `apps/baselines/clip/`:

| Pipeline                       | Entry point               | What it does                                                                                  |
| ------------------------------ | ------------------------- | --------------------------------------------------------------------------------------------- |
| **Frame extraction**           | `egobabyvlm-extract-frames`  | Sample frames from video files at a fixed FPS (ffmpeg).                                       |
| **WhisperX transcription**     | `egobabyvlm-transcribe-whisperx` | Word-level audio transcription with [WhisperX](https://github.com/m-bain/whisperX), one JSON per video. |
| **VTC + word-confidence filter** | `egobabyvlm-filter-vtc`     | Drop transcript segments overlapping the key child (KCHI) and/or below a mean word-confidence threshold. |
| **Manifest builder**           | `egobabyvlm-build-clip-manifest` | Pair WhisperX-format transcripts with extracted frames and emit train/val/test JSONs in the trainer schema. |

Each pipeline is a single Hydra entry point. Frame extraction and
WhisperX fan out via [`stopes`](https://github.com/facebookresearch/stopes)
job arrays — set `launcher.cluster=slurm` to dispatch on a cluster, or
leave the default (`local`) for a single-host run. The VTC filter is
CPU-bound JSON parsing and runs as `joblib.Parallel` within a single
process. The manifest builder is single-process and finishes in seconds
even on ~50k-video corpora.

### Layout

```
apps/data_preprocessing/
├── frames/
│   └── extract_frames.py            # ffmpeg frame extraction (stopes job array)
├── transcription/
│   ├── whisperx_transcribe.py       # WhisperX inference (stopes job array, GPU)
│   └── filter_with_vtc.py           # KCHI + word-confidence filter (joblib)
└── manifests/
    └── build_manifest.py            # Pair transcripts + frames → train/val/test JSON
```

### Quickstart

All four entry points take Hydra overrides. Each example below uses
`pixi run -e dev`; drop the prefix if you've activated the env.

#### 1. Extract frames

```bash
pixi run -e dev egobabyvlm-extract-frames \
    processor.data_dir=/path/to/videos \
    processor.output_dir=/path/to/output \
    processor.fps=1 \
    processor.videos_per_chunk=100
```

Frames land at `<output_dir>/frames/<video_name>/<video_name>_<idx>.jpg`
plus a per-run summary JSON. Override `processor.video_extensions` to
include other container formats. The 1-indexed `<idx>` is what the
manifest builder expects (step 4).

#### 2. Transcribe with WhisperX

```bash
pixi run -e dev egobabyvlm-transcribe-whisperx \
    processor.data_dir=/path/to/videos \
    processor.output_dir=/path/to/output \
    processor.whisperx_model=large-v2 \
    processor.batch_size=16 \
    processor.language=en
```

Per-video transcripts land at
`<output_dir>/transcriptions/<video_name>.json` with WhisperX's standard
`segments` + `words` schema. A `transcription_summary.json` is written
at the output root summarising success counts, per-language distribution,
and total word counts.

#### 3. Filter transcripts (VTC + word confidence)

> [!IMPORTANT]
> VTC filtering is BabyView-specific (key-child speech removal). Skip
> this step for HowTo100M / Ego4D.

```bash
pixi run -e dev egobabyvlm-filter-vtc \
    processor.transcripts_dir=/path/to/whisperx_output/transcriptions \
    processor.vtc_annotations_dir=/path/to/vtc_rttms \
    processor.output_dir=/path/to/filtered \
    processor.num_workers=8 \
    processor.min_avg_word_score=0.5
```

Transcripts and RTTM files are matched by filename stem. For each pair
the filter (a) drops every segment whose interval overlaps a `KCHI`
(key child) annotation, then (b) drops every remaining segment whose
mean WhisperX word-confidence is below `min_avg_word_score`. The
filtered transcript is written to `<output_dir>/<original_filename>.json`
with extra metadata (`kchi_filtered`, `kchi_segments_removed`,
`low_confidence_segments_removed`, `min_avg_word_score`). A
`filter_summary.json` aggregates per-VTC-label confidence statistics.

#### 4. Build the train/val/test manifest

```bash
pixi run -e dev egobabyvlm-build-clip-manifest \
    processor.transcripts_dir=/path/to/transcripts \
    processor.frames_dir=/path/to/output/frames \
    processor.output_dir=/path/to/manifests \
    processor.frames_fps=1 \
    processor.train_frac=0.85 \
    processor.val_frac=0.10 \
    processor.min_frames_per_utterance=1 \
    processor.seed=42
```

Pairs each transcript JSON (WhisperX or VTC-filtered output, same
schema) with the corresponding `<frames_dir>/<video_name>/` directory
and writes three flat JSON lists — `train.json`, `val.json`,
`test.json` — plus a `manifest_build_summary.json`. The frame index → time
mapping uses the midpoint convention `t = (idx − 0.5) / frames_fps`, so
make sure `frames_fps` matches the `processor.fps` you passed to
`egobabyvlm-extract-frames`.

<details>
<summary>Output manifest schema</summary>

The output schema is exactly what
`apps.baselines.clip.data.HowToCaptionsDataset` and
`Ego4DCaptionsDataset` consume:

```json
[
  {
    "utterance": "the cat sat on the mat",
    "frame_filenames": ["vid_a/vid_a_3.jpg", "vid_a/vid_a_4.jpg"],
    "timestamps": [2.5, 3.5],
    "utterance_num": 1,
    "video_filename": "vid_a.mp4",
    "transcript_filename": "vid_a.json",
    "num_frames": 2
  }
]
```

The trainer override is:

```bash
data=ego4d \
data.train_dataset.manifest_path=/path/to/manifests/train.json \
data.train_dataset.image_root=/path/to/output/frames \
data.val_dataset.manifest_path=/path/to/manifests/val.json
```
</details>

### Submitting on SLURM

Both stopes-driven pipelines (`egobabyvlm-extract-frames`,
`egobabyvlm-transcribe-whisperx`) accept Hydra launcher overrides:

```bash
pixi run -e dev egobabyvlm-transcribe-whisperx \
    processor.data_dir=/path/to/videos \
    processor.output_dir=/path/to/output \
    launcher.cluster=slurm \
    +launcher.update_parameters.slurm_qos=<your_qos> \
    +launcher.update_parameters.slurm_account=<your_account>
```

The job array slices `processor.videos_per_chunk` videos per task; tune
that and the per-task `Requirements` (in `extract_frames.py` /
`whisperx_transcribe.py`) for your cluster.

### Per-dataset notes

The 4-step Quickstart above applies to every corpus. Only the
dataset-specific bits differ.

#### BabyView

[BabyView](https://databrary.org/volume/1882) is a longitudinal corpus
of head-mounted-camera footage from young children.

1. **Get the videos**: download from
   [Databrary volume 1882](https://www.databrary.org/volume/1882)
   (account + data-use agreement required).
2. **Speaker diarization with VTC** before transcription — run
   [LAAC-LSCP/VTC](https://github.com/LAAC-LSCP/VTC) and save the RTTM
   outputs to a directory mirroring the video filenames. Step 3 of the
   Quickstart drops segments overlapping `KCHI` (key child) and any
   below a mean WhisperX word-confidence threshold (the paper used
   `min_avg_word_score=0.5`).
3. Run **all four** Quickstart steps. Trains with `data=ego4d` (BabyView
   shares the multi-frame-per-utterance schema).

#### Ego4D

[Ego4D](https://ego4d-data.org/) is a large egocentric video dataset.

1. **Request access** at <https://ego4d-data.org/> and install the
   official downloader.
2. **Download the full-scale videos**:
   ```bash
   pip install ego4d
   ego4d --output_directory=/path/to/ego4d --datasets full_scale
   ```
   Ego4D ships its own narration JSONs (`--datasets annotations`) but
   we re-transcribe with WhisperX so the schema and word-level
   timestamps match the rest of the stack.
3. Run Quickstart steps **1, 2, and 4** (skip VTC — Ego4D videos are
   not centred on a target child). Trains with `data=ego4d`.

#### COCO

COCO already ships with hand-written captions, so no preprocessing
pipeline is needed. Point the trainer at the
[Karpathy split](https://cs.stanford.edu/people/karpathy/deepimagesent/coco.zip)
and the COCO 2014 images:

```bash
pixi run -e dev torchrun --standalone --nproc-per-node=4 \
    -m apps.baselines.clip.training.train \
    name=coco_baseline data=coco \
    data.train_dataset.manifest_path=/path/to/dataset_coco.json \
    data.train_dataset.image_root=/path/to/coco/all_images \
    data.val_dataset.manifest_path=/path/to/dataset_coco.json
```

The COCO loader (`apps.baselines.clip.data.CocoCaptionsDataset`) reads
the standard Karpathy schema
(`{"images": [{"filename", "sentences": [{"raw"|"tokens"}]}]}`). For a
held-out validation manifest, pre-split the Karpathy JSON into
train-only / val-only files and point each `*_dataset.manifest_path` at
the matching split.
