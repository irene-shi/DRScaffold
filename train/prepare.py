"""Validate Swift JSONL inputs and prepare an explicit, seeded P4 replay mix."""

import argparse
import json
import random
from pathlib import Path

from train.schema import canonical_response


def read_records(path, image_root=None, structured=True):
    records = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            try:
                record = json.loads(line)
                messages = record["messages"]
                if not isinstance(messages, list) or not messages:
                    raise ValueError("messages must be a nonempty list")
                roles = [m["role"] for m in messages]
                dialogue = roles[1:] if roles[0] == "system" else roles
                if not dialogue or len(dialogue) % 2 or dialogue != ["user", "assistant"] * (len(dialogue) // 2):
                    raise ValueError("Expected alternating user/assistant turns, optionally preceded by system")
                if structured and roles != ["system", "user", "assistant"]:
                    raise ValueError("DRBench rows must have system, user, assistant messages")
                for message in messages:
                    if not isinstance(message["content"], str) or not message["content"].strip():
                        raise ValueError("Message content must be a nonempty string")
                    if "loss" in message:
                        raise ValueError("Per-message loss overrides are not supported")
                    if structured and message["role"] == "assistant":
                        message["content"] = canonical_response(message["content"])
                images = record["images"]
                if not isinstance(images, list) or len(images) != 1 or not isinstance(images[0], str):
                    raise ValueError("Expected exactly one image path")
                if sum(m["content"].count("<image>") for m in messages if m["role"] == "user") != 1:
                    raise ValueError("Expected one <image> placeholder in user messages")
                image = Path(images[0]).expanduser()
                if not image.is_absolute():
                    image = (image_root or path.parent) / image
                image = image.resolve()
                if not image.is_file():
                    raise ValueError(f"Image not found: {image}")
                records.append({"messages": messages, "images": [str(image)]})
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"{path}:{line_number}: {exc}") from exc
    if not records:
        raise ValueError(f"Empty dataset: {path}")
    return records


def mix_replay(domain, replay, count, seed):
    """Keep every domain row once and append a replay subset without replacement."""
    if count <= 0 or count > len(replay):
        raise ValueError(f"Need at least {count} replay rows; found {len(replay)}")
    if not domain:
        raise ValueError("Empty DRBench training set")
    # Preserve an already selected replay file, matching the original concatenation.
    selected = list(replay) if count == len(replay) else random.Random(seed).sample(replay, count)
    return list(domain) + selected


def prepare(train, val, replay, output, image_root=None, replay_image_root=None,
            replay_count=15000, seed=42):
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"Output directory must be empty: {output}")
    domain = read_records(train, image_root)
    validation = read_records(val, image_root)
    general = read_records(replay, replay_image_root, structured=False)
    train_images = {r["images"][0] for r in domain}
    val_images = {r["images"][0] for r in validation}
    if train_images & val_images:
        raise ValueError("Training and validation contain overlapping image paths")
    if val_images & {r["images"][0] for r in general}:
        raise ValueError("Replay and validation contain overlapping image paths")
    mixed = mix_replay(domain, general, replay_count, seed)
    output.mkdir(parents=True, exist_ok=True)
    for name, rows in (("train", domain), ("val", validation), ("p4_train", mixed)):
        with (output / f"{name}.jsonl").open("w", encoding="utf-8") as stream:
            for row in rows:
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    summary = {"train": len(domain), "val": len(validation), "p4_domain": len(domain),
               "p4_replay": replay_count, "seed": seed}
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--val", type=Path, required=True)
    parser.add_argument("--replay", type=Path, required=True, help="LLaVA conversations in Swift messages/images JSONL")
    parser.add_argument("--output", type=Path, default=Path("data/prepared"))
    parser.add_argument("--image-root", type=Path)
    parser.add_argument("--replay-image-root", type=Path)
    parser.add_argument("--replay-count", type=int, default=15000,
                        help="Number of replay rows to append; all DRBench training rows are retained")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    try:
        print(json.dumps(prepare(**vars(args)), indent=2))
    except ValueError as exc:
        parser.exit(2, f"error: {exc}\n")


if __name__ == "__main__":
    main()
