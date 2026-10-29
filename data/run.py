"""Run individual construction stages; resume reuses only matching inputs/configuration."""

import argparse
import hashlib
import json
from pathlib import Path

from data.client import Client, load_env
from data.human import load_samples, make_tasks
from data.io import digest, write_json, write_jsonl
from data.pipeline import annotate, blind_filter, generate, review
from data.schema import load_manifest, require, validate_qa

HERE = Path(__file__).resolve().parent
FILES = {"generate": "01_questions.jsonl", "blind": "02_filtered.jsonl",
         "review": "03_reviewed.jsonl", "annotate": "04_annotated.jsonl"}
PREVIOUS = {"blind": "generate", "review": "blind", "annotate": "review", "human-templates": "review"}
ROLES = {"generate": ("VISION",), "blind": ("BLIND_A", "BLIND_B", "JUDGE"),
         "review": ("VISION", "REVIEWER"), "annotate": ("VISION", "REVIEWER")}


def file_hash(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def validate_config(config):
    require(config["hypersim_questions"] == 15, "Paper recipe requires 15 Hypersim questions")
    require(config["cityscapes_questions"] in (8, 9, 10), "Cityscapes uses 8-10 questions")
    for key in ("few_shot_count", "max_annotation_rounds", "min_objects", "min_relations",
                "max_tokens", "timeout_seconds", "request_attempts"):
        require(type(config[key]) is int and config[key] > 0, f"{key} must be a positive integer")
    require(config["max_annotation_rounds"] <= 5, "At most five annotation rounds")
    require(config["min_objects"] >= 3 and config["min_relations"] >= 3, "Paper complexity minimum is three objects and relations")
    require(isinstance(config["temperature"], (int, float)) and 0 <= config["temperature"] <= 2, "Invalid temperature")


def run_stage(stage, records, client, config, output, resume=False, samples=()):
    output = Path(output)
    parts = output.with_name(output.name + ".parts")
    meta_path = parts / "run.json"
    paths = sorted({row[k] for row in records for k in ("image", "gt_image", "gt_context") if row.get(k)})
    code = {str(p.relative_to(HERE)): file_hash(p) for p in sorted(HERE.glob("*.py"))}
    prompts = {p.name: file_hash(p) for p in sorted((HERE / "prompts").glob("*.txt"))}
    metadata = {"stage": stage, "input_digest": digest(records), "config": config,
                "models": client.public_settings(), "examples_digest": digest(samples),
                "assets": {path: file_hash(path) for path in paths}, "code": code, "prompts": prompts}
    if parts.exists() or output.exists():
        require(resume, f"Output exists: {output}; use --resume with unchanged inputs or a new output")
        require(meta_path.is_file(), "Resume metadata missing")
        require(json.loads(meta_path.read_text(encoding="utf-8")) == metadata,
                "Inputs, assets, code, prompts, models or configuration changed; use a new output path")
    else:
        write_json(meta_path, metadata)
    results = []
    function = {"generate": generate, "blind": blind_filter, "review": review, "annotate": annotate}[stage]
    for index, row in enumerate(records, 1):
        part = parts / f"{digest(row['id'])}.json"
        if part.is_file():
            result = json.loads(part.read_text(encoding="utf-8"))
            action = "cached"
        else:
            result = function(row, client, config, samples) if stage == "annotate" else function(row, client, config)
            write_json(part, result)
            action = "completed"
        results.append(result)
        print(f"{stage} {index}/{len(records)}: {row['id']} {action}", flush=True)
    write_jsonl(output, results)
    summary = {"images": len(results), "questions": sum(len(r["questions"]) for r in results)}
    if stage == "annotate":
        summary.update({status: sum(a["status"] == status for r in results for a in r["annotations"])
                        for status in ("accepted", "discarded")})
    write_json(output.with_suffix(".summary.json"), summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("check", "generate", "blind", "review", "human-templates", "annotate"))
    parser.add_argument("--manifest", type=Path, default=HERE / "inputs/images.jsonl")
    parser.add_argument("--input", type=Path, help="Previous stage's JSONL (defaults under data/work)")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--work-dir", type=Path, default=HERE / "work")
    parser.add_argument("--config", type=Path, default=HERE / "configs/pipeline.json")
    parser.add_argument("--env-file", type=Path, default=HERE / ".env")
    parser.add_argument("--human-samples", type=Path, default=HERE / "human_samples/approved.jsonl")
    parser.add_argument("--count", type=int, default=300, help="Number of human annotation tasks")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    try:
        config = json.loads(args.config.read_text(encoding="utf-8"))
        validate_config(config)
        input_path = args.manifest if args.stage in ("check", "generate") else args.input or args.work_dir / FILES[PREVIOUS[args.stage]]
        records = load_manifest(input_path)
        if args.stage in PREVIOUS:
            require(all(row.get("stage") == PREVIOUS[args.stage] for row in records), "Unexpected upstream stage")
            for row in records:
                require(isinstance(row.get("questions"), list), "Missing question list")
                for qa in row["questions"]:
                    validate_qa(qa)
                require(len({q["id"] for q in row["questions"]}) == len(row["questions"]), "Duplicate question id")
        if args.stage in ("check", "review"):
            for row in records:
                require(row.get("gt_image") and row.get("gt_context"), f"{row['id']}: GT overlay and context are required for Stage C")
                gt = json.loads(Path(row["gt_context"]).read_text(encoding="utf-8"))
                require(isinstance(gt, dict) and gt.get("domain") == row["domain"], "GT context domain mismatch")
                require(gt.get("image") and str(Path(gt["image"]).resolve()) == row["image"],
                        "GT context must identify the same image; regenerate GT assets after moving images")
                require(isinstance(gt.get("classes"), list) and bool(gt["classes"]), "GT context has no semantic classes")
        if args.stage == "check":
            print(f"Checked {len(records)} images, GT paths and split assignments. No API calls made.")
            return
        if args.stage == "human-templates":
            output = args.output or HERE / "human_samples/to_annotate.jsonl"
            tasks = make_tasks(records, output, args.count, config["seed"])
            print(f"Wrote {len(tasks)} tasks with empty structured fields to {output}")
            return
        samples = load_samples(args.human_samples, config, records) if args.stage == "annotate" else []
        load_env(args.env_file)
        client = Client(config, args.work_dir / "api_cache", ROLES[args.stage])
        output = args.output or args.work_dir / FILES[args.stage]
        print(json.dumps(run_stage(args.stage, records, client, config, output, args.resume, samples), indent=2))
    except (ValueError, OSError, RuntimeError, KeyError) as exc:
        parser.exit(2, f"error: {exc}\n")


if __name__ == "__main__":
    main()
