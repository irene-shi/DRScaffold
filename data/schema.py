"""Explicit schemas shared by generation, human annotation and training export."""

import re
from collections import Counter
from pathlib import Path

from data.io import read_jsonl, resolve_file

CATEGORIES = ("Perception", "Spatial Reasoning", "Affordance Reasoning",
              "Anomaly Detection", "False Premise Rejection")
TAGS = ("Perception", "Relation", "Logic", "Conclusion")
QA_KEYS = {"id", "type", "category", "question", "options", "answer"}
ANNOTATION_KEYS = {"key_objects", "scene_graph", "reasoning_chain"}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def nonempty(value):
    return isinstance(value, str) and bool(value.strip())


def boolean_result(value, key):
    require(isinstance(value, dict) and type(value.get(key)) is bool, f"{key} must be a JSON boolean")
    return value


def validate_qa(qa):
    require(isinstance(qa, dict) and set(qa) == QA_KEYS, f"QA must contain exactly {sorted(QA_KEYS)}")
    require(all(nonempty(qa[k]) for k in ("id", "question", "answer")), "Empty QA id/question/answer")
    require(qa["category"] in CATEGORIES, "Unknown category")
    require(qa["type"] in ("single_choice", "multi_select", "open_ended"), "Unknown question type")
    options, answer = qa["options"], qa["answer"]
    if qa["type"] == "open_ended":
        require(options is None, "open_ended options must be null")
    else:
        require(isinstance(options, list) and len(options) == 4, "Choice questions need four A-D options")
        require(all(nonempty(option) and re.match(rf"^{letter}[.)]\s+\S", option)
                    for letter, option in zip("ABCD", options)), "Options must be labeled A. through D.")
        letters = answer.split(",")
        require(all(letter in "ABCD" and len(letter) == 1 for letter in letters), "Invalid answer letters")
        require(letters == sorted(set(letters)), "Answer letters must be unique and sorted, without spaces")
        if qa["type"] == "single_choice":
            require(len(letters) == 1, "single_choice requires one answer")
    return qa


def validate_questions(result, domain, count):
    require(isinstance(result, dict) and set(result) == {"questions"}, "Expected questions object")
    questions = result["questions"]
    require(isinstance(questions, list) and len(questions) == count, f"Expected {count} questions")
    for question in questions:
        validate_qa(question)
    require(len({q["id"] for q in questions}) == len(questions), "Duplicate question ids")
    counts = Counter(q["category"] for q in questions)
    minimum = 2 if domain == "hypersim" else 1
    require(all(counts[c] >= minimum for c in CATEGORIES), "Missing category coverage")
    if domain == "hypersim":
        require(sum(q["type"] == "open_ended" for q in questions) >= 4, "Hypersim needs at least four open questions")
    require(all(q["type"] != "open_ended" for q in questions if q["category"] == "Perception"),
            "Perception must use choice questions")
    return result


def triplet(line):
    require(isinstance(line, str), "Scene graph entries must be strings")
    match = re.fullmatch(r"\s*(.+?)\s+--([a-z][a-z0-9_]*)-->\s+(.+?)\s*", line)
    require(match is not None, "Expected scene-graph relation: Entity --predicate--> Entity")
    return list(match.groups())


def validate_annotation(value, min_objects=3, min_relations=3):
    require(isinstance(value, dict) and set(value) == ANNOTATION_KEYS, "Expected only the three structured fields")
    objects, graph, chain = (value[k] for k in ("key_objects", "scene_graph", "reasoning_chain"))
    require(isinstance(objects, list) and all(nonempty(v) for v in objects), "Invalid key_objects")
    require(len(set(objects)) == len(objects) and len(objects) >= min_objects, "Too few distinct key_objects")
    require(isinstance(graph, list) and len(graph) >= min_relations, "Too few scene_graph relations")
    triples = [triplet(line) for line in graph]
    require(len({tuple(t) for t in triples}) == len(triples), "Duplicate scene_graph relations")
    require(all(s in objects and o in objects for s, _, o in triples), "All graph endpoints must occur in key_objects")
    require(isinstance(chain, list) and len(chain) == 4, "reasoning_chain must contain four stages")
    require(all(isinstance(line, str) and line.startswith(f"[{tag}]: ")
                and line[len(f"[{tag}]: "):].strip() for tag, line in zip(TAGS, chain)), "Invalid reasoning stage order or empty step")
    return value


def validate_review(value, questions, visual=False):
    require(isinstance(value, dict) and isinstance(value.get("results"), list), "Expected results list")
    results = value["results"]
    require(all(isinstance(row, dict) and isinstance(row.get("id"), str) for row in results), "Invalid review row")
    require(Counter(row["id"] for row in results) == Counter(q["id"] for q in questions), "Review must cover every input id exactly once")
    originals = {q["id"]: q for q in questions}
    for row in results:
        boolean_result(row, "bad_question")
        require(nonempty(row.get("reason")), "Missing review reason")
        if visual:
            boolean_result(row, "is_modified")
            require(not (row["bad_question"] and row["is_modified"]), "Deleted questions cannot also be modified")
            if not row["bad_question"]:
                updated = {**originals[row["id"]], **{k: row.get(k) for k in ("question", "answer", "options")}}
                validate_qa(updated)
                require(row["is_modified"] or updated == originals[row["id"]], "Unchanged review changed QA content")
    return value


def load_manifest(path):
    rows = read_jsonl(path)
    ids, images, groups = set(), {}, {}
    for row in rows:
        require(nonempty(row.get("id")) and row["id"] not in ids, "Missing/duplicate image id")
        ids.add(row["id"])
        require(row.get("domain") in ("hypersim", "cityscapes"), "Invalid image domain")
        require(row.get("split") in ("train", "val", "test"), "Assign train/val/test explicitly in the manifest")
        row["group_id"] = row.get("group_id") or row["id"]
        require(nonempty(row["group_id"]), "Invalid group_id")
        row["image"] = resolve_file(row.get("image"), Path(path).resolve().parent)
        for key in ("gt_image", "gt_context"):
            if row.get(key):
                row[key] = resolve_file(row[key], Path(path).resolve().parent)
        group = (row["domain"], row["group_id"])
        require(groups.get(group, row["split"]) == row["split"], "A scene group crosses splits")
        require(row["image"] not in images, "The same image path occurs more than once")
        groups[group] = row["split"]
        images[row["image"]] = row["split"]
    return rows
