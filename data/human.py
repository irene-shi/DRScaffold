"""Select empty annotation tasks and load only explicitly approved human examples."""

from collections import defaultdict
from pathlib import Path
import random

from data.io import read_jsonl, resolve_file, write_jsonl
from data.schema import require, validate_annotation, validate_qa


def make_tasks(records, output, count=300, seed=42):
    require(not Path(output).exists(), f"Refusing to replace existing human annotations: {output}")
    buckets = defaultdict(list)
    for row in records:
        if row["split"] != "train":
            continue
        for qa in row["questions"]:
            buckets[(row["domain"], qa["category"])].append({
                "image_id": row["id"], "domain": row["domain"], "split": "train",
                "group_id": row["group_id"], "image": row["image"], "qa": qa,
                "annotation": {"key_objects": [], "scene_graph": [], "reasoning_chain": []}, "approved": False})
    require(count > 0 and sum(map(len, buckets.values())) >= count, f"Need at least {count} reviewed training questions")
    rng = random.Random(seed)
    for bucket in buckets.values():
        rng.shuffle(bucket)
    selected = []
    while len(selected) < count:
        for key in sorted(buckets):
            if buckets[key] and len(selected) < count:
                selected.append(buckets[key].pop())
    write_jsonl(output, selected)
    return selected


def load_samples(path, config, records):
    samples = read_jsonl(path)
    by_id = {row["id"]: row for row in records}
    seen = set()
    for sample in samples:
        require(sample.get("approved") is True, "All human samples must be completed and approved=true")
        require(sample.get("split") == "train", "Human exemplars must belong to the training split")
        validate_qa(sample.get("qa"))
        validate_annotation(sample.get("annotation"), config["min_objects"], config["min_relations"])
        sample["image"] = resolve_file(sample.get("image"), Path(path).resolve().parent)
        row = by_id.get(sample.get("image_id"))
        require(row is not None, "Human image_id must belong to the reviewed dataset")
        require(row["split"] == "train" and row["domain"] == sample.get("domain")
                and row["image"] == sample["image"], "Human exemplar metadata differs from reviewed data")
        require(sample["qa"] in row["questions"], "Human QA must match the reviewed QA exactly")
        key = (sample["image_id"], sample["qa"]["id"])
        require(key not in seen, "Duplicate human example")
        seen.add(key)
    return samples
