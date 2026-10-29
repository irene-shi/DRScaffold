# DRScaffold training

Four-stage supervised fine-tuning of **Qwen2.5-VL-3B-Instruct** (default),
**InternVL2.5-2B**, and **Phi-4-multimodal-instruct** with ms-swift.
The full structured response stays in the input throughout training; binary loss
masks progressively unlock entities, relations, reasoning, and the answer.
Only attention-projection LoRA parameters in the language decoder are trained.

## Install

Use Python 3.11 on a Linux CUDA machine. From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
pip install -r train/requirements.txt
```

For InternVL or Phi-4, create a separate environment and replace the last install
command with the corresponding requirements file:

```bash
# InternVL2.5-2B
pip install -r train/requirements/internvl.txt
# Phi-4-multimodal (in its own environment)
pip install -r train/requirements/phi4.txt
```

These two backbones use Transformers 4.48.3 for their custom model code.
Qwen and Phi-4 use SDPA; InternVL uses eager attention. The paper uses
two NVIDIA A6000 GPUs. Single-GPU and multi-GPU launch commands are provided below.

## Input data

See [data/README.md](../data/README.md) for dataset construction and annotation
export. This training package consumes **already exported** ms-swift JSONL:
one `messages` / `images` object per line. It does not download datasets.

Each DRBench record has `system`, `user`, and `assistant` messages, with one
`<image>` placeholder in the user message and one path in `images`. Use
`prompts/system.txt` and `prompts/user.txt` for the original training prompts;
replace `{question}` with the question including any options.
The assistant content is a JSON **string** with this schema:

```json
{
  "thought_process": {
    "detected_entities": ["cup", "table"],
    "spatial_graph": [["cup", "on", "table"]],
    "reasoning_steps": [
      "1. [Perception]: A cup and a table are visible.",
      "2. [Relation]: The cup is on the table.",
      "3. [Logic]: The table supports the cup.",
      "4. [Conclusion]: The cup is on the table."
    ]
  },
  "answer": "table"
}
```

This is an illustrative annotation, not a training sample. The implementation's
`detected_entities`, `spatial_graph`, and `reasoning_steps` correspond to the
paper's `key_objects`, `scene_graph`, and `reasoning_chain`. Preserve this schema
when exporting annotations. Missing reasoning is rejected rather than fabricated.

LLaVA replay uses the same outer `messages` / `images` format but ordinary
assistant text and, optionally, multiple user/assistant turns. Convert original
LLaVA `conversations` to this format in the data pipeline. Use training splits
only, with at least 15,000 replay records.

```bash
python -m train.prepare \
  --train data/drbench_train.jsonl \
  --val data/drbench_val.jsonl \
  --replay data/llava_train.jsonl \
  --image-root /path/to/drbench/images \
  --replay-image-root /path/to/coco/images \
  --output data/prepared
```

Relative image paths resolve against the supplied root, or against the input
JSONL directory if no root is supplied. Absolute paths are preserved. Export
portable relative paths when moving data between machines. Preparation checks
image existence and train/validation image-path overlap, canonicalizes field
order, and rejects per-message loss overrides. It does not validate image
contents or detect identical images stored at different paths.

P1–P3 use the unchanged DRBench training set. P4 retains every DRBench training
row once and appends 15,000 replay rows, sampled without replacement if the
replay file contains more rows. An already selected 15,000-row replay file is
appended unchanged. DRBench is neither repeated nor subsampled to match replay.
The verified 15,000-replay runs use 11,894 DRBench rows plus 15,000 LLaVA rows
(26,894 total), an approximately 1:1 mixture. The default sampling seed is 42;
`--seed` overrides it. `--replay-count` changes only the replay count, and
`summary.json` records the actual counts. Validation remains DRBench-only.

## Train

```bash
# Inspect all four commands without loading data or a model.
python -m train.run --dry-run

# Full curriculum; use --model /path/to/Qwen2.5-VL-3B-Instruct for local weights.
CUDA_VISIBLE_DEVICES=0,1 NPROC_PER_NODE=2 \
  python -m train.run --data-dir data/prepared --output-dir outputs/qwen
```

For one GPU, set `CUDA_VISIBLE_DEVICES=0 NPROC_PER_NODE=1`. With per-device
batch size 1 and gradient accumulation 4, the effective batch size is 8 on two
GPUs and 4 on one GPU.
Set `MASTER_PORT` if needed for concurrent distributed jobs.

| Phase | Supervised fields | Epochs | Learning rate | Schedule | Warmup |
|---|---|---:|---:|---|---|
| P1 | Entities | 0.5 | 5e-5 | Cosine | 10% |
| P2 | + Relations | 1 | 4e-5 | Constant with warmup | 20 steps |
| P3 | + Reasoning | 1 | 3e-5 | Constant with warmup | 30 steps |
| P4 | + Answer; LLaVA replay | 1 | 3e-6 | Cosine | 5% |

Select a backbone with `--backbone`; use `--model /path/to/model` for local weights:

```bash
CUDA_VISIBLE_DEVICES=0,1 NPROC_PER_NODE=2 python -m train.run --backbone internvl
CUDA_VISIBLE_DEVICES=0,1 NPROC_PER_NODE=2 python -m train.run --backbone phi4
```

| Backbone | Model ID | Decoder attention LoRA targets |
|---|---|---|
| `qwen` | `Qwen/Qwen2.5-VL-3B-Instruct` | `q_proj`, `k_proj`, `v_proj`, `o_proj` |
| `internvl` | `OpenGVLab/InternVL2_5-2B` | `attention.wqkv`, `attention.wo` |
| `phi4` | `microsoft/Phi-4-multimodal-instruct` | `self_attn.qkv_proj`, `self_attn.o_proj` |

Phi-4's pretrained vision/speech adapters remain frozen. The Phi-4 tuner keeps
its separate task adapter active alongside the relevant pretrained modality
adapter, including after checkpoint loading. When using Swift directly for
Phi-4 inference, also pass `--external_plugins train/swift_plugin.py` and
`--tuner_type drscaffold_phi4`; do not merge the modality adapters.

Edit the backbone JSON under `configs/` or provide `--config` to change settings.
InternVL and Phi-4 inherit the common training schedule from the Qwen configuration.
The default LoRA rank/alpha are 16/16, dropout is 0.2, and maximum length is
8192 tokens. The backbone-specific target regex selects only decoder attention projections; the vision encoder, merger, embeddings, and output head
stay frozen.

Runtime defaults are explicit in the configuration: BF16, gradient checkpointing,
training/data seeds of 42, per-device train/eval batch sizes of 1, evaluation and
checkpoint saving every 300 steps, logging every 5 steps, and 4 data-loading
workers and preprocessing processes. Experiment-tracker reporting is disabled.
These are execution settings, separate from the paper's reported hyperparameters.
Overlength samples are dropped rather than truncated; inspect preprocessing logs
for the number dropped. Strict data processing is enabled, cached preprocessing
is disabled, and an explicit validation set is used without another random split.

Each stage initializes from the previous stage's **last** saved adapter and
starts a fresh optimizer/scheduler. It does not merge the adapter or restart from
the base model. Default output roots are `outputs/<config-name>/<experiment>/`.
For the full curriculum, stages live in `p1/` through `p4/`; each includes launch metadata,
Swift checkpoints, and `last_adapter.txt`. Existing output directories are not
silently reused. Automatic version subdirectories are disabled. The last two
checkpoints per stage are retained, including optimizer state for resumption.
`last_adapter.txt` records the actual checkpoint path.

```bash
# Start P3 from a completed P2 adapter (fresh optimizer).
python -m train.run --start-phase 3 \
  --adapter outputs/previous/p2/checkpoint-1000 \
  --output-dir outputs/continued

# Resume an interrupted P2 with its optimizer, then continue P3/P4.
python -m train.run --start-phase 2 \
  --resume outputs/qwen/p2/checkpoint-300 \
  --output-dir outputs/qwen
```

Use real checkpoint paths from your run; the step numbers above are examples.
`--end-phase` stops after a selected stage. A resume checkpoint must belong to
the starting stage's output directory. Do not change data, GPU count, or training
configuration when resuming an interrupted stage.

## Stage ablations

The paper's stage-removal and ordering ablations use Qwen2.5-VL-3B. All
experiments start independently from the base model and use the same prepared
training and validation files. Epochs removed with a stage are redistributed
**proportionally** across the retained stages, preserving a total of 3.5 epochs.

| `--experiment` | Stage order | Epochs in execution order |
|---|---|---|
| `full` | S1 → S2 → S3 → S4 | 0.5, 1, 1, 1 |
| `s4` | S4 | 3.5 |
| `s34` | S3 → S4 | 1.75, 1.75 |
| `s234` | S2 → S3 → S4 | 7/6, 7/6, 7/6 |
| `s134` | S1 → S3 → S4 | 0.7, 1.4, 1.4 |
| `reverse` | S4 → S3 → S2 → S1 | 1, 1, 1, 0.5 |
| `random_1324` | S1 → S3 → S2 → S4 | 0.5, 1, 1, 1 |
| `random_3412` | S3 → S4 → S1 → S2 | 1, 1, 0.5, 1 |
| `random_4123` | S4 → S1 → S2 → S3 | 1, 0.5, 1, 1 |

S1–S4 correspond to P1–P4 above. Each stage retains its own cumulative loss
mask, learning rate, and scheduler regardless of position. S4 always uses the
DRBench-plus-replay mixture, including when it runs first. Thus `s4` supervises **all
structured fields simultaneously**, rather than only the answer. Equal epoch
budgets do not imply equal optimizer-step counts because S4 uses a mixed set.

```bash
# One experiment.
CUDA_VISIBLE_DEVICES=0,1 NPROC_PER_NODE=2 \
  python -m train.run --experiment reverse --output-dir outputs/reverse

# Run the full reference and all eight ablation runs sequentially.
CUDA_VISIBLE_DEVICES=0,1 NPROC_PER_NODE=2 \
  python -m train.ablations --data-dir data/prepared --output-dir outputs/ablations

# Run only the three fixed random orderings from the appendix.
python -m train.ablations --experiments random_1324 random_3412 random_4123

# Inspect commands without launching training.
python -m train.ablations --dry-run
```

Report “Random (avg.)” as the arithmetic mean of the three separately evaluated
random-order runs; their adapters are not averaged. Intermediate checkpoints
from `full` provide the stage-wise comparison.

Ablation output folders identify both execution position and stage, such as
`step1_p4/` and `step2_p3/` for reverse order. Use `--start-step` and `--end-step`
for positions in the selected schedule (`--start-phase` and `--end-phase` remain
aliases for the full-curriculum commands above):

```bash
python -m train.run --experiment reverse --start-step 2 \
  --resume outputs/reverse/step2_p3/checkpoint-300 \
  --output-dir outputs/reverse
```

## Tests

```bash
python -m unittest discover -s train/tests -v
```

The checks cover cumulative masking, invalid annotations, seeded mixing,
validation overlap, checkpoint handoff, and the actual ms-swift loss-scale API
when installed. They also check equal-budget ablations and Phi-4 modality
adapter gradients, freezing, and checkpoint round trips with a tiny CPU decoder.
They require no downloaded model or GPU.
