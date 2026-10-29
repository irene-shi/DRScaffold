# Human exemplar handoff

No completed samples are supplied. `../templates/human_sample.template.json`
contains only empty fields and is a format reference, not a usable example.

1. Run generation, blind filtering, and scene review.
2. Run `python -m data.run human-templates --count 300` from the repository root.
3. Open each task's image and reviewed QA in `data/human_samples/to_annotate.jsonl`.
4. Complete `annotation` with grounded labels:
   - `key_objects`: at least three distinct task-relevant entity strings.
   - `scene_graph`: at least three distinct `Entity --predicate--> Entity` strings.
     Every endpoint must match a key object exactly; predicates use lowercase
     letters/digits/underscores.
   - `reasoning_chain`: exactly four nonempty strings, beginning with
     `[Perception]: `, `[Relation]: `, `[Logic]: `, and `[Conclusion]: ` in order.
     The conclusion supports the supplied answer, including its option letters.
     Never assert an intentionally absent FPR referent as visible.
5. Set `approved` to true after inspecting the image and completing all fields.
   Save the pool as `data/human_samples/approved.jsonl`.
6. Continue generation:

```bash
python -m data.run annotate --human-samples data/human_samples/approved.jsonl
```

Keep identifiers, image paths, split, and reviewed QA unchanged. If QA itself
needs correction, update it upstream and rerun review into a new output before
regenerating tasks. Do not pad graphs with invented objects to pass the minimum
complexity check; remove unsuitable questions upstream.

Only training examples may serve as demonstrations. The loader checks membership
against reviewed records and rejects unfinished samples, duplicates, and QA
mismatches. The paper uses 300 examples covering both domains and all five
categories. Selection balances available buckets, but cannot manufacture missing
categories; inspect coverage before the full run.

Approved examples become final human annotations for their own questions.
For other questions, a deterministic subset is sent with images as in-context
demonstrations. `few_shot_count` controls the subset size, not the full pool size.

Annotation JSONL files in this directory are ignored by Git.
