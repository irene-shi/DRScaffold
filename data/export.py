"""Export accepted annotations to the existing train/ interface, without API calls."""

import argparse
import json
from pathlib import Path
import random

from data.io import read_jsonl, resolve_file, write_json, write_jsonl
from data.schema import require, triplet, validate_annotation, validate_qa
from train.schema import canonical_response

ROOT = Path(__file__).resolve().parents[1]


def swift_record(row, qa, annotation):
    validate_qa(qa)
    validate_annotation(annotation, min_objects=1, min_relations=1)
    answer = canonical_response(json.dumps({
        "thought_process": {
            "detected_entities": annotation["key_objects"],
            "spatial_graph": [triplet(line) for line in annotation["scene_graph"]],
            "reasoning_steps": [f"{i}. {step}" for i, step in enumerate(annotation["reasoning_chain"], 1)],
        }, "answer": qa["answer"]}, ensure_ascii=False))
    question = qa["question"]
    if qa["options"]:
        question += "\n" + "\n".join(qa["options"])
        question += "\nAnswer with " + ("one option letter." if qa["type"] == "single_choice" else "comma-separated option letters (multiple answers allowed).")
    user = (ROOT / "train/prompts/user.txt").read_text(encoding="utf-8").format(question=question)
    return {"messages": [
        {"role": "system", "content": (ROOT / "train/prompts/system.txt").read_text(encoding="utf-8")},
        {"role": "user", "content": "<image>\n" + user},
        {"role": "assistant", "content": answer}], "images": [row["image"]]}


def export_drbench(records, output):
    output = Path(output)
    require(not output.exists() or not any(output.iterdir()), "Export directory must be empty")
    splits = {split: [] for split in ("train", "val", "test")}
    benchmark, discarded, seen, groups, images = [], 0, set(), {}, {}
    for row in records:
        require(row.get("stage") == "annotate", "Expected annotate-stage records")
        split = row["split"]
        require(split in splits, "Unknown split")
        group = (row["domain"], row["group_id"])
        require(groups.get(group, split) == split, "A group crosses splits")
        require(images.get(row["image"], split) == split, "An image crosses splits")
        groups[group], images[row["image"]] = split, split
        for item in row["annotations"]:
            if item["status"] != "accepted":
                discarded += 1
                continue
            qa, annotation = item["qa"], item["annotation"]
            key = (row["id"], qa["id"])
            require(key not in seen, "Duplicate exported question")
            seen.add(key)
            splits[split].append(swift_record(row, qa, annotation))
            benchmark.append({"image_id": row["id"], "domain": row["domain"], "group_id": row["group_id"],
                              "split": split, "image": row["image"], "qa": qa, "annotation": annotation,
                              "source": item["source"]})
    require(bool(benchmark), "No accepted annotations to export")
    for split, rows in splits.items():
        write_jsonl(output / f"drbench_{split}.jsonl", rows)
    write_jsonl(output / "drbench_annotations.jsonl", benchmark)
    summary = {**{split: len(rows) for split, rows in splits.items()}, "discarded_annotations": discarded}
    write_json(output / "summary.json", summary)
    return summary


def export_llava(source, images, output, count=15000, seed=42):
    require(not Path(output).exists(), f"Output already exists: {output}")
    records = json.loads(Path(source).read_text(encoding="utf-8"))
    require(isinstance(records, list) and count > 0 and len(records) >= count, "Not enough LLaVA records")
    selected = random.Random(seed).sample(records, count)
    rows = []
    for record in selected:
        path = resolve_file(record.get("image"), Path(images).resolve())
        turns = record.get("conversations")
        require(isinstance(turns, list) and len(turns) > 0 and len(turns) % 2 == 0, "Invalid LLaVA conversations")
        messages = [{"role": "system", "content": "You are a helpful language and vision assistant. You are able to understand the visual content that the user provides, and assist the user with a variety of tasks using natural language."}]
        for i, turn in enumerate(turns):
            expected = "human" if i % 2 == 0 else "gpt"
            require(turn.get("from") == expected and isinstance(turn.get("value"), str) and turn["value"].strip(), "Invalid LLaVA turn")
            messages.append({"role": "user" if expected == "human" else "assistant", "content": turn["value"]})
        require(sum(m["content"].count("<image>") for m in messages if m["role"] == "user") == 1,
                "Expected one image placeholder in LLaVA user turns")
        rows.append({"images": [path], "messages": messages})
    write_jsonl(output, rows)
    return {"replay_records": len(rows), "seed": seed}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="kind", required=True)
    drbench = commands.add_parser("drbench")
    drbench.add_argument("--input", type=Path, required=True)
    drbench.add_argument("--output", type=Path, default=ROOT / "data/work/export")
    llava = commands.add_parser("llava")
    llava.add_argument("--input", type=Path, required=True)
    llava.add_argument("--images", type=Path, required=True)
    llava.add_argument("--output", type=Path, default=ROOT / "data/work/llava_train.jsonl")
    llava.add_argument("--count", type=int, default=15000)
    llava.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    try:
        result = export_drbench(read_jsonl(args.input), args.output) if args.kind == "drbench" else export_llava(
            args.input, args.images, args.output, args.count, args.seed)
        print(json.dumps(result, indent=2))
    except (ValueError, OSError) as exc:
        parser.exit(2, f"error: {exc}\n")


if __name__ == "__main__":
    main()
