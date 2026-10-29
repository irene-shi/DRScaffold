"""Offline synthetic tests. No real human examples, images, or API calls are included."""

import base64
import copy
import importlib.util
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from data.client import Client
from data.export import export_drbench, export_llava
from data.human import load_samples, make_tasks
from data.io import read_jsonl, write_json, write_jsonl
from data.pipeline import annotate, blind_filter, generate, review
from data.run import HERE, run_stage
from data.schema import CATEGORIES, load_manifest, validate_annotation, validate_review
from train.prepare import read_records

CONFIG = json.loads((HERE / "configs/pipeline.json").read_text())


def qa(index=0):
    category = CATEGORIES[index % 5]
    choice = category == "Perception"
    return {"id": f"Q{index}", "type": "single_choice" if choice else "open_ended", "category": category,
            "question": f"Synthetic test query {index}",
            "options": [f"{letter}. synthetic {letter}" for letter in "ABCD"] if choice else None,
            "answer": "A" if choice else "synthetic result"}


def fields(answer="A"):
    return {"key_objects": ["synthetic_a", "synthetic_b", "synthetic_c"],
            "scene_graph": ["synthetic_a --left_of--> synthetic_b", "synthetic_b --on--> synthetic_c",
                            "synthetic_a --near--> synthetic_c"],
            "reasoning_chain": ["[Perception]: Synthetic observation.", "[Relation]: Synthetic relation.",
                                "[Logic]: Synthetic deduction.", f"[Conclusion]: {answer}"]}


class FakeClient:
    def __init__(self, domain="hypersim", both_correct=False, fail_consistency=False):
        self.domain, self.both_correct, self.fail_consistency = domain, both_correct, fail_consistency
        self.calls = []

    def public_settings(self):
        return {"test": {"model": "offline-synthetic"}}

    def ask(self, role, prompt, payload, images=(), examples=(), validator=None):
        self.calls.append((role, prompt, copy.deepcopy(payload), list(images), list(examples)))
        if "Generate exactly" in prompt:
            result = {"questions": [qa(i) for i in range(CONFIG[f"{self.domain}_questions"])]}
        elif "CANNOT see the image" in prompt:
            assert not images and not examples
            assert set(payload) == {"type", "question", "options"}
            result = {"answer": "A"}
        elif "semantically equivalent" in prompt:
            assert not images
            result = {"similar": self.both_correct}
        elif "TYPE of scene/room" in prompt:
            result = {"room_type": "synthetic", "scene_summary": "Synthetic test scene", "camera_tilt": "none"}
        elif "strict text-only reviewer" in prompt:
            assert not images
            result = {"results": [{"id": q["id"], "bad_question": False, "reason": "synthetic pass"} for q in payload["questions"]]}
        elif "careful visual QA reviewer" in prompt:
            assert len(images) == 2 and "ground_truth" in payload
            result = {"results": [{**{k: q[k] for k in ("id", "question", "options", "answer")},
                                   "bad_question": False, "is_modified": False, "reason": "synthetic pass"}
                                  for q in payload["questions"]]}
        elif "fill structured supervision" in prompt:
            assert images and examples
            result = fields(payload["qa"]["answer"])
        elif "strict logic reviewer" in prompt:
            assert not images and not examples
            result = {"pass": False, "feedback": "synthetic failure", "hint": "synthetic revision"} if self.fail_consistency else {"pass": True}
        else:
            raise AssertionError(prompt)
        if validator:
            validator(result)
        return result


def make_image(root, name, domain="hypersim", split="train"):
    root = root.resolve()
    image = root / f"{name}.png"
    # A tiny synthetic PNG; core tests only need path existence / base64 transport.
    image.write_bytes(base64.b64decode("iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+aKdEAAAAASUVORK5CYII="))
    gt = root / f"{name}_gt.json"
    write_json(gt, {"domain": domain, "image": str(image), "classes": [{"class": "synthetic"}]})
    return {"id": name, "domain": domain, "split": split, "group_id": name,
            "image": str(image), "gt_image": str(image), "gt_context": str(gt)}


class PipelineTests(unittest.TestCase):
    def test_offline_end_to_end_and_train_compatibility(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rows = [make_image(root, "train"), make_image(root, "val", "cityscapes", "val")]
            reviewed = []
            for row in rows:
                client = FakeClient(row["domain"])
                current = review(blind_filter(generate(row, client, CONFIG), client, CONFIG), client, CONFIG)
                reviewed.append(current)
                stage_a_calls = sum("TYPE of scene/room" in call[1] for call in client.calls)
                self.assertEqual(stage_a_calls, int(row["domain"] == "hypersim"))
            sample_path = root / "human.jsonl"
            tasks = make_tasks(reviewed, sample_path, count=1)
            self.assertFalse(tasks[0]["approved"])
            self.assertEqual(tasks[0]["annotation"]["key_objects"], [])
            with self.assertRaises(ValueError):
                load_samples(sample_path, CONFIG, reviewed)
            tasks[0].update(approved=True, annotation=fields(tasks[0]["qa"]["answer"]))
            write_jsonl(sample_path, tasks)
            samples = load_samples(sample_path, CONFIG, reviewed)
            completed = [annotate(row, FakeClient(row["domain"]), CONFIG, samples) for row in reviewed]
            output = root / "export"
            result = export_drbench(completed, output)
            self.assertEqual((result["train"], result["val"]), (15, 10))
            self.assertEqual(len(read_records(output / "drbench_train.jsonl")), 15)
            self.assertEqual(len(read_records(output / "drbench_val.jsonl")), 10)
            assistant = json.loads(read_jsonl(output / "drbench_train.jsonl")[0]["messages"][-1]["content"])
            self.assertEqual(assistant["thought_process"]["detected_entities"], fields()["key_objects"])

    def test_blind_filter_requires_both_correct(self):
        row = {"questions": [qa()], "rejected": []}
        self.assertEqual(len(blind_filter(row, FakeClient(both_correct=False), CONFIG)["questions"]), 1)
        self.assertEqual(len(blind_filter(row, FakeClient(both_correct=True), CONFIG)["questions"]), 0)
        class OneCorrect(FakeClient):
            def ask(self, role, prompt, payload, **kwargs):
                result = super().ask(role, prompt, payload, **kwargs)
                if role == "JUDGE":
                    result["similar"] = sum(call[0] == "JUDGE" for call in self.calls) == 1
                return result
        self.assertEqual(len(blind_filter(row, OneCorrect(), CONFIG)["questions"]), 1)

    def test_five_round_failure_discarded_and_feedback_returned(self):
        with tempfile.TemporaryDirectory() as tmp:
            row = make_image(Path(tmp), "target")
            row.update(questions=[qa()], rejected=[])
            sample = {"image_id": "another", "domain": "hypersim", "image": row["image"], "qa": qa(1), "annotation": fields()}
            client = FakeClient(fail_consistency=True)
            output = annotate(row, client, CONFIG, [sample])
            self.assertEqual(output["annotations"][0]["status"], "discarded")
            calls = [call for call in client.calls if "fill structured supervision" in call[1]]
            self.assertEqual(len(calls), 5)
            self.assertNotIn("feedback", calls[0][2])
            self.assertIn("feedback", calls[1][2])
            with self.assertRaisesRegex(ValueError, "No accepted"):
                export_drbench([output], Path(tmp) / "export")

    def test_review_preserves_fpr_and_rejects_missing_or_changed_results(self):
        q = qa(4)
        with self.assertRaises(ValueError):
            validate_review({"results": []}, [q])
        decision = {**q, "bad_question": False, "is_modified": False, "reason": "test"}
        decision["answer"] = "changed"
        with self.assertRaises(ValueError):
            validate_review({"results": [decision]}, [q], visual=True)
        self.assertIn("exempt", (HERE / "prompts/03c_visual.txt").read_text())
        invalid = fields()
        invalid["scene_graph"][0] = "missing --on--> synthetic_c"
        with self.assertRaises(ValueError):
            validate_annotation(invalid)

    def test_visual_revision_and_deletion_are_applied_before_annotation(self):
        class RevisingClient(FakeClient):
            def ask(self, role, prompt, payload, **kwargs):
                validator = kwargs.pop("validator", None)
                result = super().ask(role, prompt, payload, **kwargs)
                if "careful visual QA reviewer" in prompt:
                    result["results"][0].update(is_modified=True, answer="B")
                    result["results"][1].update(bad_question=True)
                if validator:
                    validator(result)
                return result
        with tempfile.TemporaryDirectory() as tmp:
            row = make_image(Path(tmp), "revision")
            row.update(questions=[qa(0), qa(1)], rejected=[], stage="blind")
            result = review(row, RevisingClient(), CONFIG)
            self.assertEqual(len(result["questions"]), 1)
            self.assertEqual(result["questions"][0]["answer"], "B")
            self.assertEqual(result["questions"][0]["category"], "Perception")
            self.assertEqual(result["rejected"][0]["qa"]["id"], "Q1")
            self.assertEqual(row["questions"][0]["answer"], "A")

    def test_resume_checks_assets_and_reuses_completed_rows(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            row = make_image(root, "source")
            output = root / "generated.jsonl"
            client = FakeClient()
            run_stage("generate", [row], client, CONFIG, output)
            count = len(client.calls)
            run_stage("generate", [row], client, CONFIG, output, resume=True)
            self.assertEqual(len(client.calls), count)
            Path(row["image"]).write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "changed"):
                run_stage("generate", [row], client, CONFIG, output, resume=True)

    def test_split_leakage_and_unfilled_env_fail(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rows = [make_image(root, "a"), make_image(root, "b", split="test")]
            rows[1]["group_id"] = rows[0]["group_id"]
            manifest = root / "images.jsonl"
            write_jsonl(manifest, rows)
            with self.assertRaisesRegex(ValueError, "crosses splits"):
                load_manifest(manifest)
            with patch.dict(os.environ, {}, clear=True):
                with self.assertRaisesRegex(ValueError, "VISION_BASE_URL"):
                    Client(CONFIG, root / "cache", ["VISION"])

    def test_transport_cache_and_no_credential_persistence(self):
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, {
            "VISION_BASE_URL": "https://example.invalid/v1", "VISION_API_KEY": "synthetic-test-secret", "VISION_MODEL": "test-model"}):
            client = Client(CONFIG, Path(tmp), ["VISION"])
            envelope = json.dumps({"choices": [{"finish_reason": "stop", "message": {"content": '{"pass":true}'}}]}).encode()
            with patch("urllib.request.urlopen", return_value=io.BytesIO(envelope)) as mocked:
                self.assertEqual(client.ask("VISION", "test", {}), {"pass": True})
                self.assertEqual(client.ask("VISION", "test", {}), {"pass": True})
                self.assertEqual(mocked.call_count, 1)
            self.assertNotIn("synthetic-test-secret", json.dumps(client.public_settings()))
            self.assertTrue(all("synthetic-test-secret" not in p.read_text() for p in Path(tmp).glob("*.json")))

    def test_llava_conversion(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            row = make_image(root, "replay")
            source = root / "llava.json"
            write_json(source, [{"image": Path(row["image"]).name, "conversations": [
                {"from": "human", "value": "<image> Synthetic question"}, {"from": "gpt", "value": "Synthetic response"}]}])
            output = root / "replay.jsonl"
            export_llava(source, root, output, count=1)
            self.assertEqual(len(read_records(output, structured=False)), 1)

    def test_invalid_api_responses_stop_and_disk_errors_do_not_retry(self):
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, {
            "VISION_BASE_URL": "https://example.invalid/v1", "VISION_API_KEY": "synthetic-test-secret", "VISION_MODEL": "test-model"}):
            client = Client({**CONFIG, "request_attempts": 2}, Path(tmp), ["VISION"])
            with patch("urllib.request.urlopen", side_effect=lambda *a, **k: io.BytesIO(b'{}')) as mocked, patch("time.sleep"):
                with self.assertRaises(RuntimeError):
                    client.ask("VISION", "test", {})
                self.assertEqual(mocked.call_count, 2)
                self.assertFalse(list(Path(tmp).glob("*.json")))
            envelope = json.dumps({"choices": [{"message": {"content": '{"pass":true}'}}]}).encode()
            with patch("urllib.request.urlopen", return_value=io.BytesIO(envelope)) as mocked, patch("data.client.write_json", side_effect=PermissionError("disk")):
                with self.assertRaises(PermissionError):
                    client.ask("VISION", "test", {})
                self.assertEqual(mocked.call_count, 1)


@unittest.skipUnless(all(importlib.util.find_spec(name) for name in ("PIL", "numpy", "h5py")), "GT dependencies not installed")
class GroundTruthTests(unittest.TestCase):
    def test_hypersim_and_cityscapes_assets(self):
        import h5py
        import numpy as np
        from PIL import Image
        from data.ground_truth import prepare_gt
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image = root / "image.png"
            Image.new("RGB", (4, 4), "white").save(image)
            semantic = root / "semantic.hdf5"
            with h5py.File(semantic, "w") as handle:
                handle["dataset"] = np.full((4, 4), 1, dtype=np.int32)
            out = prepare_gt("hypersim", image, semantic, root / "hs")
            self.assertEqual(json.loads(Path(out["gt_context"]).read_text())["classes"][0]["class"], "wall")
            city = root / "labels.png"
            color = root / "color.png"
            Image.fromarray(np.full((4, 4), 7, dtype=np.uint8)).save(city)
            Image.new("RGB", (4, 4), (128, 64, 128)).save(color)
            out = prepare_gt("cityscapes", image, city, root / "cs", color=color)
            self.assertEqual(json.loads(Path(out["gt_context"]).read_text())["classes"][0]["class"], "road")


if __name__ == "__main__":
    unittest.main()
