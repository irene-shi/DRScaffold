# DRBench data construction

This package implements the DRBench construction pipeline: question generation,
visual-dependency filtering, scene-matching review, human-exemplar-guided
structured annotation, consistency verification, and export to `train/`.

Configure your models, endpoints, and API keys in `data/.env`, prepare the source
images and GT annotations, and supply completed human examples as described below.

| Step | Implementation | Paper prompt |
|---|---|---|
| 1 | Generate 15 indoor or 8–10 street QA pairs | 1 |
| 2 | Two models answer without images; remove only if both are correct | 2a, 2b |
| 3A | Room type and camera tilt summary; Hypersim only | 3a |
| 3B | Conservative text-only scene-context screening | 3b |
| 3C | Image + GT review: pass, small revision, or delete | 3c |
| Human preparation | Select tasks; user supplies completed examples | Human exemplars |
| 4 | Generate structured fields; verify/revise up to five rounds | 4a, 4b |
| Export | Preserve accepted annotations and write Swift JSONL | Training interface |

## 1. Environment and model roles

Use Python 3.11 and run commands from the repository root:

```bash
pip install -r data/requirements.txt
cp data/.env.example data/.env
```

Edit `data/.env`, replacing every `xxx` used by your stage. Each role needs a
`BASE_URL`, `API_KEY`, and `MODEL`. Endpoints must implement an OpenAI-compatible
`POST /chat/completions` API; `BASE_URL` is the prefix, usually ending in `/v1`.
Existing environment variables take precedence over the file. Assignments are
read as text, never executed as shell commands.

| Role | Purpose | Input |
|---|---|---|
| `VISION` | Generation, scene summary, visual review, structured fields | Text and images |
| `BLIND_A`, `BLIND_B` | Independent blind answering | Question/type/options only |
| `JUDGE` | Compare blind answers with reference answers | Text only |
| `REVIEWER` | Scene-context screening and annotation consistency | Text only |

Choose two genuinely different blind models, not aliases for one model. Blind
answerers receive neither reference answers nor category labels. The judge and
reviewer may share a backend; their requests remain separate. Vision endpoints
must support PNG/JPEG/WebP data URLs and multiple images per conversation.
No provider SDK or model weights are downloaded by this package.

Defaults in `configs/pipeline.json`: 15 Hypersim questions, 10 Cityscapes
questions, three retrieved human examples per question, and five **total**
generation/check rounds, including the initial attempt.

## 2. Images, GT annotations, and splits

Obtain images and corresponding ground-truth files from Hypersim and Cityscapes
under their dataset terms. Select the image list yourself. Use annotated
Cityscapes images with `gtFine`; source test images without GT cannot pass the
required visual review. Your DRBench `split` is separate from the source
dataset's directory name.

Prepare GT assets for each image:

```bash
python -m data.ground_truth hypersim \
  --image /path/to/preview.png \
  --semantic /path/to/frame.semantic.hdf5 \
  --instances /path/to/frame.semantic_instance.hdf5 \
  --positions /path/to/frame.position.hdf5 \
  --output data/work/gt/your_image_id

python -m data.ground_truth cityscapes \
  --image /path/to/image_leftImg8bit.png \
  --semantic /path/to/image_gtFine_labelIds.png \
  --color /path/to/image_gtFine_color.png \
  --output data/work/gt/another_image_id
```

Hypersim `--instances` and `--positions` are optional; the semantic map is
required. Position coordinates retain source asset units. Cityscapes input is
**labelIds**, not trainIds. Outputs are `overlay.png` and `context.json` with
visible classes and normalized image positions. Hypersim overlay colors are
custom display colors with an explicit legend. Class-level boxes can contain
multiple instances; missing coarse labels do not prove accessories absent.

Create `data/inputs/images.jsonl`, one object per image, using the blank
`templates/image_manifest.template.json` as the field reference:

| Field | Fill in |
|---|---|
| `id` | Unique image identifier |
| `domain` | `hypersim` or `cityscapes` |
| `split` | `train`, `val`, or `test` |
| `group_id` | Shared scene/sequence ID; keep related views in one split |
| `image` | Source image path |
| `gt_image` | Generated semantic overlay path |
| `gt_context` | Generated context JSON path |

Paths may be absolute or relative to **the manifest's directory**. Generated
records use absolute paths. Regenerate GT context after moving source images.
The reader prevents duplicate image paths and scene groups crossing splits; it
cannot detect the same image copied under different names. Assign splits in the
manifest, keeping all images from the same scene or sequence in one split.

```bash
python -m data.run check --manifest data/inputs/images.jsonl
```

This checks paths, GT metadata, and split assignments without any model calls.

## 3. Generate and filter QA

```bash
python -m data.run generate --manifest data/inputs/images.jsonl
python -m data.run blind
python -m data.run review
```

Default outputs under `data/work/`:

```text
01_questions.jsonl       # QA only; no structured fields yet
02_filtered.jsonl       # Dual-model text-blind filtering
03_reviewed.jsonl       # Scene review, including corrected QA
```

Records retain image metadata, surviving questions, and rejected questions with
stage/reason. Blind predictions and scene-review decisions remain available.
Stage A is skipped for Cityscapes. Stage C requires the original image, aligned
GT overlay, and GT context; it never silently falls back to image-only review.

False Premise Rejection is exempt from checks requiring its intentionally absent
target to be visible, large, or locatable. Deliberate absence is not itself a
deletion reason. Other contradictions and unsupported answers can still fail.

## 4. Supply human annotations

```bash
python -m data.run human-templates --count 300
```

This selects tasks across available domain/category buckets using **training
questions only** and writes `data/human_samples/to_annotate.jsonl`. It requires
at least that many reviewed training questions. For development, explicitly
choose a smaller count. The paper uses 300 examples across both domains and all
five categories; inspect the selected pool's coverage yourself.

The tasks retain the image and reviewed QA, but structured annotation lists are
empty and `approved` is false. Inspect each image, complete the three fields,
and save `data/human_samples/approved.jsonl` with `approved: true`. See
[human_samples/README.md](human_samples/README.md) for the exact contract.
No completed human examples are distributed.

The pool can contain more records than fit in one API call. Three examples are
selected deterministically by default, preferring the same domain and category.
Their images, QA and annotations are sent as demonstration turns. Empty or
unfinished pools are rejected instead of silently running without exemplars.

## 5. Generate structured annotations

```bash
python -m data.run annotate --human-samples data/human_samples/approved.jsonl
```

For every accepted QA, generate `key_objects`, `scene_graph`, and four tagged
`reasoning_chain` steps using the image and verified answer. The QA answer is
immutable at this stage. Approved human records are preserved for their own
questions. Remaining records go through text-only consistency checking.

A failed check returns previous fields and feedback to the vision model.
Malformed structured fields also generate repair feedback. Five unsuccessful
rounds mark the question `discarded`; it is excluded from export. History is
retained in `04_annotated.jsonl` for troubleshooting. Consistency verification
checks agreement between the structured fields and the verified answer.

## 6. Export to training

```bash
python -m data.export drbench \
  --input data/work/04_annotated.jsonl --output data/work/export
```

Outputs: `drbench_train.jsonl`, `drbench_val.jsonl`, `drbench_test.jsonl`, and
`drbench_annotations.jsonl` retaining categories and original annotations.
Only accepted records are exported. Empty splits remain empty; assign suitable
splits before training. Conversion uses the existing training prompts:

| Annotation | Training JSON |
|---|---|
| `key_objects` | `thought_process.detected_entities` |
| `scene_graph`: `A --predicate--> B` | `thought_process.spatial_graph`: `[A, predicate, B]` |
| Four tagged reasoning strings | Numbered `thought_process.reasoning_steps` |
| Verified answer | `answer` |

For replay, obtain original LLaVA instruction annotations and COCO images
locally, then convert a seeded subset:

```bash
python -m data.export llava \
  --input /path/to/llava_instruct_150k.json \
  --images /path/to/coco/train2017 \
  --count 15000 --output data/work/llava_train.jsonl

python -m train.prepare \
  --train data/work/export/drbench_train.jsonl \
  --val data/work/export/drbench_val.jsonl \
  --replay data/work/llava_train.jsonl \
  --output data/prepared
```

The test split is never included in training/replay. See [train/README.md](../train/README.md)
for training commands. LLaVA conversion preserves all natural-language dialogue
turns; it does not impose the structured answer schema on replay conversations.

## Resume and validation

Repeat the same stage with `--resume` after interruption:

```bash
python -m data.run annotate --human-samples data/human_samples/approved.jsonl --resume
```

Stages save completed image records atomically in a sibling `.parts/` directory,
then assemble the JSONL output. Responses are cached by exact request contents,
endpoint and model, allowing reuse of completed subrequests within an interrupted
image. Transient failures and malformed responses are retried; authentication
errors stop immediately. API failures never count as successful verification or
scientific rejection.

Resume requires matching inputs, asset contents, code, prompts, configuration,
models and examples. After changes, choose a new `--output`/`--work-dir`; use
`--input` when the prior stage is outside that work directory. Execution is
sequential by default. Start with a small manifest to measure your provider's
cost. No paid request is made by tests or the `check` command.

```bash
python -m unittest discover -s data/tests -v
```

Tests use synthetic fixtures and mocked requests to exercise the complete
pipeline, training compatibility, revision limits, text-only input isolation,
GT preparation, resume and credential handling.
