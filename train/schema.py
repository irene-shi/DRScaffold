"""The original training serialization and progressive supervision boundaries."""

import json

FIELDS = ("detected_entities", "spatial_graph", "reasoning_steps")
TAGS = ("Perception", "Relation", "Logic", "Conclusion")


def canonical_response(text):
    """Validate annotations without inventing entities or missing reasoning steps."""
    obj = json.loads(text)
    if not isinstance(obj, dict) or set(obj) != {"thought_process", "answer"}:
        raise ValueError("Expected thought_process and answer")
    thought = obj["thought_process"]
    if not isinstance(thought, dict) or set(thought) != set(FIELDS):
        raise ValueError(f"thought_process must contain {FIELDS}")
    entities, graph, steps = (thought[k] for k in FIELDS)
    if not isinstance(entities, list) or not all(isinstance(x, str) and x.strip() for x in entities):
        raise ValueError("detected_entities must be a list of nonempty strings")
    if not isinstance(graph, list) or not all(
        isinstance(t, list) and len(t) == 3
        and all(isinstance(x, str) and x.strip() for x in t) for t in graph
    ):
        raise ValueError("spatial_graph must contain [subject, relation, object] triples")
    if not isinstance(steps, list) or len(steps) != 4 or not all(
        isinstance(s, str) and s.startswith(f"{i}. [{tag}]: ")
        and s[len(f"{i}. [{tag}]: "):].strip()
        for i, (tag, s) in enumerate(zip(TAGS, steps), 1)
    ):
        raise ValueError("reasoning_steps must have four numbered Perception/Relation/Logic/Conclusion steps")
    if not isinstance(obj["answer"], str) or not obj["answer"].strip():
        raise ValueError("answer must be a nonempty string")
    ordered = {"thought_process": {key: thought[key] for key in FIELDS}, "answer": obj["answer"]}
    return json.dumps(ordered, ensure_ascii=False, separators=(",", ":"))


def split_response(text, phase):
    """Keep the complete response as context; mask the not-yet-unlocked suffix."""
    if phase not in (1, 2, 3, 4):
        raise ValueError("phase must be 1, 2, 3 or 4")
    if phase == 4:
        return [text], [1.0]
    if canonical_response(text) != text:
        raise ValueError("Noncanonical DRScaffold response; run the dataset preparation step")
    marker = ('"spatial_graph":', '"reasoning_steps":', '"answer":')[phase - 1]
    boundary = text.index(marker)
    return [text[:boundary], text[boundary:]], [1.0, 0.0]
