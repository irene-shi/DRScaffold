"""Paper stages: QA -> text-blind filter -> scene review -> structured supervision."""

import copy
import json
from pathlib import Path

from data.io import digest
from data.schema import (boolean_result, nonempty, require, validate_annotation,
                         validate_qa, validate_questions, validate_review)

PROMPTS = Path(__file__).resolve().parent / "prompts"


def prompt(name, **values):
    text = (PROMPTS / f"{name}.txt").read_text(encoding="utf-8")
    for key, value in values.items():
        text = text.replace("{{" + key + "}}", str(value))
    require("{{" not in text, f"Unfilled template in {name}")
    return text


def generate(row, client, config):
    domain = row["domain"]
    count = config[f"{domain}_questions"]
    system = prompt("01_questions", domain="indoor" if domain == "hypersim" else "urban street",
                    count=count, coverage="at least 2 per category" if domain == "hypersim" else "at least 1 per category",
                    domain_rules="- At least 4 out of 15 questions must be open-ended."
                    if domain == "hypersim" else
                    "- Use clearly identifiable vehicles, pedestrians and infrastructure as anchors. Avoid tiny distant objects, ambiguous colors, or unclear lane/location references. Cover all five categories within the 8-10 questions.")
    result = client.ask("VISION", system, {"task": "Generate the question set for this image."},
                        images=[row["image"]], validator=lambda v: validate_questions(v, domain, count))
    return {**row, "questions": result["questions"], "rejected": [], "stage": "generate"}


def blind_filter(row, client, config):
    result = copy.deepcopy(row)
    kept, decisions = [], []
    for qa in row["questions"]:
        # Neither the category (especially FPR) nor the reference answer is exposed.
        question = {k: qa[k] for k in ("type", "question", "options")}
        answers, correct = {}, {}
        for role in ("BLIND_A", "BLIND_B"):
            prediction = client.ask(role, prompt("02a_blind_answer"), question,
                                    validator=lambda v: require(nonempty(v.get("answer")), "Missing blind answer"))
            judgment = client.ask("JUDGE", prompt("02b_correctness"),
                                  {**question, "reference_answer": qa["answer"], "model_answer": prediction["answer"]},
                                  validator=lambda v: boolean_result(v, "similar"))
            answers[role], correct[role] = prediction["answer"], judgment["similar"]
        remove = correct["BLIND_A"] and correct["BLIND_B"]
        decisions.append({"id": qa["id"], "answers": answers, "correct": correct, "removed": remove})
        if remove:
            result["rejected"].append({"qa": qa, "stage": "blind", "reason": "Both text-only models answered correctly"})
        else:
            kept.append(qa)
    result.update(questions=kept, blind_decisions=decisions, stage="blind")
    return result


def validate_scene(value):
    require(all(nonempty(value.get(k)) for k in ("room_type", "scene_summary")), "Missing scene summary")
    require(value.get("camera_tilt") in ("none", "slight", "strong"), "Invalid camera_tilt")
    return value


def review(row, client, config):
    result = copy.deepcopy(row)
    if not row["questions"]:
        return {**result, "stage": "review"}
    if row["domain"] == "hypersim":
        scene = client.ask("VISION", prompt("03a_scene"), {"task": "Describe this scene."},
                           images=[row["image"]], validator=validate_scene)
        b_rules = "Use the supplied indoor room type and summary. Defer uncertain visual details to Stage C."
        c_rules = "Camera may be tilted; mentally re-level the scene first."
    else:
        scene = {"room_type": "outdoor urban street", "scene_summary": "An ego-centric urban driving image."}
        b_rules = "Cityscapes adaptation: all images are urban street scenes. Reject clearly meaningless forced choices, scene-incompatible assumptions, or blatant internal contradictions. Do not guess object visibility, color or position from text; those are Stage C's responsibility."
        c_rules = "Cityscapes: delete or modify when the target is too small/far away, its spatial location is ambiguous, or its color/identity is unclear due to lighting, shadow, motion blur or compression. Prefer tightening the referring expression when safe; otherwise delete. This target-object check does not apply to the deliberately absent referent in False Premise Rejection. Do NOT discuss camera tilt; treat the view as the driver would see it."
    context = client.ask("REVIEWER", prompt("03b_context", domain_rules=b_rules),
                         {"scene": scene, "questions": row["questions"]},
                         validator=lambda v: validate_review(v, row["questions"]))
    b_by_id = {v["id"]: v for v in context["results"]}
    remaining = []
    for qa in row["questions"]:
        if b_by_id[qa["id"]]["bad_question"]:
            result["rejected"].append({"qa": qa, "stage": "review_b", "reason": b_by_id[qa["id"]]["reason"]})
        else:
            remaining.append(qa)
    result.update(scene=scene, review_b=context["results"], review_c=[])
    if remaining:
        gt = json.loads(Path(row["gt_context"]).read_text(encoding="utf-8"))
        visual = client.ask("VISION", prompt("03c_visual", domain_rules=c_rules),
                            {"questions": remaining, "ground_truth": gt}, images=[row["image"], row["gt_image"]],
                            validator=lambda v: validate_review(v, remaining, visual=True))
        c_by_id = {v["id"]: v for v in visual["results"]}
        kept = []
        for qa in remaining:
            decision = c_by_id[qa["id"]]
            if decision["bad_question"]:
                result["rejected"].append({"qa": qa, "stage": "review_c", "reason": decision["reason"]})
            else:
                kept.append({**qa, **{k: decision[k] for k in ("question", "options", "answer")}})
        result.update(questions=kept, review_c=visual["results"])
    else:
        result["questions"] = []
    result["stage"] = "review"
    return result


def pick_examples(samples, row, qa, count, seed):
    eligible = [s for s in samples if not (s["image_id"] == row["id"] and s["qa"]["id"] == qa["id"])]
    # Stable, relevance-first selection; the paper does not specify a retrieval rule.
    eligible.sort(key=lambda s: (s["domain"] != row["domain"], s["qa"]["category"] != qa["category"],
                                 digest([seed, row["id"], qa["id"], s["image_id"], s["qa"]["id"]])))
    require(bool(eligible), "No eligible human exemplar for this question")
    return eligible[:count]


def annotate(row, client, config, samples):
    result = copy.deepcopy(row)
    annotations = []
    human = {(s["image_id"], s["qa"]["id"]): s for s in samples}
    for qa in row["questions"]:
        key = (row["id"], qa["id"])
        if key in human:
            sample = human[key]
            require(sample["qa"] == qa and sample["image"] == row["image"], "Human QA differs from the reviewed record; rerun review for revised QA")
            annotations.append({"qa": qa, "annotation": sample["annotation"], "status": "accepted",
                                "source": "human", "history": []})
            continue
        examples = pick_examples(samples, row, qa, config["few_shot_count"], config["seed"])
        previous, feedback, history = None, None, []
        for round_number in range(1, config["max_annotation_rounds"] + 1):
            payload = {"qa": qa}
            if previous is not None:
                payload.update(previous=previous, feedback=feedback)
            fields = client.ask("VISION", prompt("04a_structure", min_objects=config["min_objects"],
                                                 min_relations=config["min_relations"]), payload,
                                images=[row["image"]], examples=examples)
            try:
                validate_annotation(fields, config["min_objects"], config["min_relations"])
            except ValueError as exc:
                verdict = {"pass": False, "feedback": str(exc), "hint": "Repair the structured format using only grounded facts."}
            else:
                verdict = client.ask("REVIEWER", prompt("04b_consistency"), {"qa": qa, "annotation": fields},
                                     validator=validate_consistency)
            history.append({"round": round_number, "annotation": fields, "verdict": verdict})
            if verdict["pass"]:
                break
            previous, feedback = fields, verdict
        annotations.append({"qa": qa, "annotation": fields, "status": "accepted" if verdict["pass"] else "discarded",
                            "source": "generated", "history": history,
                            "exemplar_ids": [[s["image_id"], s["qa"]["id"]] for s in examples]})
    result.update(annotations=annotations, stage="annotate")
    return result


def validate_consistency(value):
    boolean_result(value, "pass")
    if not value["pass"]:
        require(nonempty(value.get("feedback")) and nonempty(value.get("hint")), "Failed review needs feedback and hint")
    return value
