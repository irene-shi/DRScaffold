"""CPU checks for supervision, preparation and stage handoff; no model download."""

import importlib.util
import json
from pathlib import Path
import re
import tempfile
import unittest

from train.prepare import mix_replay, prepare
from train.run import DEFAULT_CONFIG, last_checkpoint, phase_command
from train.schema import canonical_response, split_response


def response():
    return canonical_response(json.dumps({
        "answer": "A",
        "thought_process": {
            "reasoning_steps": [f'{i}. [{tag}]: Evidence includes "answer": and café.'
                                for i, tag in enumerate(("Perception", "Relation", "Logic", "Conclusion"), 1)],
            "spatial_graph": [["cup", "on", "table"]],
            "detected_entities": ["cup", "table"],
        },
    }))


class TrainingTests(unittest.TestCase):
    def test_masks_preserve_context_and_progress(self):
        text = response()
        lengths = []
        for phase, marker in enumerate(('"spatial_graph":', '"reasoning_steps":', '"answer":'), 1):
            parts, weights = split_response(text, phase)
            self.assertEqual("".join(parts), text)
            self.assertEqual(weights, [1.0, 0.0])
            self.assertTrue(parts[1].startswith(marker))
            lengths.append(len(parts[0]))
        self.assertTrue(lengths[0] < lengths[1] < lengths[2] < len(text))
        self.assertEqual(split_response(text, 4), ([text], [1.0]))
        self.assertEqual(split_response("Ordinary replay answer", 4), (["Ordinary replay answer"], [1.0]))

    def test_invalid_annotations_never_fall_back_to_full_loss(self):
        for text in ('{"answer":"A"}', "plain text", response().replace('"answer":', '"wrong":')):
            with self.assertRaises(ValueError):
                split_response(text, 1)
        obj = json.loads(response())
        obj["thought_process"]["reasoning_steps"].pop()
        with self.assertRaises(ValueError):
            canonical_response(json.dumps(obj))

    def test_mix_preserves_domain_and_samples_replay_without_replacement(self):
        replay = [{"kind": "replay", "id": i} for i in range(8)]
        for domain_count in (3, 7):
            with self.subTest(domain_count=domain_count):
                domain = [{"kind": "domain", "id": i} for i in range(domain_count)]
                mixed = mix_replay(domain, replay, 5, 42)
                self.assertEqual(mixed, mix_replay(domain, replay, 5, 42))
                self.assertEqual([r for r in mixed if r["kind"] == "domain"], domain)
                self.assertEqual(len(mixed), domain_count + 5)
                self.assertEqual(len({r["id"] for r in mixed if r["kind"] == "replay"}), 5)
                self.assertEqual(mix_replay(domain, replay, 8, 42), domain + replay)
        with self.assertRaises(ValueError):
            mix_replay(domain, replay, 9, 42)
        with self.assertRaises(ValueError):
            mix_replay([], replay, 5, 42)

    def test_preparation_and_image_split_check(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for name in ("train", "val", "replay"):
                (root / f"{name}.png").touch()
                row = {"images": [f"{name}.png"], "messages": [
                    {"role": "system", "content": "Follow the schema."},
                    {"role": "user", "content": "<image> What is on the table?"},
                    {"role": "assistant", "content": response() if name != "replay" else "A cup."}]}
                (root / f"{name}.jsonl").write_text(json.dumps(row) + "\n")
            paths = [root / f"{name}.jsonl" for name in ("train", "val", "replay")]
            replay_row = json.loads(paths[2].read_text())
            extra_replay_row = json.loads(paths[2].read_text())
            extra_replay_row["messages"][-1]["content"] = "A table."
            paths[2].write_text(json.dumps(replay_row) + "\n" + json.dumps(extra_replay_row) + "\n")
            summary = prepare(*paths, root / "prepared", replay_count=2)
            self.assertEqual(summary["p4_replay"], 2)
            self.assertEqual(summary["p4_domain"], 1)
            self.assertEqual(json.loads((root / "prepared/summary.json").read_text()), summary)
            mixed = (root / "prepared/p4_train.jsonl").read_text().splitlines()
            self.assertEqual(len(mixed), 3)
            self.assertTrue(Path(json.loads(mixed[0])["images"][0]).is_absolute())
            paths[1].write_text(paths[0].read_text())
            with self.assertRaisesRegex(ValueError, "overlapping"):
                prepare(*paths, root / "bad", replay_count=1)

    def test_commands_and_numeric_checkpoint_handoff(self):
        config = json.loads(DEFAULT_CONFIG.read_text())
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for step in (9, 100):
                checkpoint = root / f"checkpoint-{step}"
                checkpoint.mkdir()
                (checkpoint / "adapter_config.json").touch()
                (checkpoint / "adapter_model.safetensors").touch()
            adapter = last_checkpoint(root)
            self.assertEqual(adapter.name, "checkpoint-100")
            for phase in range(1, 5):
                cmd = phase_command(config, phase, root, root / f"p{phase}", adapter if phase > 1 else None)
                self.assertIn(f"drscaffold_p{phase}", cmd)
                self.assertIn(str(root / ("p4_train.jsonl" if phase == 4 else "train.jsonl")), cmd)
                self.assertEqual("--adapters" in cmd, phase > 1)
                self.assertNotIn("--resume_from_checkpoint", cmd)
            cmd = phase_command(config, 2, root, root / "p2", resume=adapter)
            self.assertIn("--resume_from_checkpoint", cmd)
            self.assertNotIn("--adapters", cmd)

    def test_lora_target_excludes_vision_and_mlp(self):
        pattern = json.loads(DEFAULT_CONFIG.read_text())["common"]["target_regex"]
        self.assertIsNotNone(re.fullmatch(pattern, "model.language_model.layers.0.self_attn.q_proj"))
        self.assertIsNotNone(re.fullmatch(pattern, "model.layers.0.self_attn.o_proj"))
        for name in ("model.visual.blocks.0.attn.qkv", "model.layers.0.mlp.up_proj", "lm_head"):
            self.assertIsNone(re.fullmatch(pattern, name))


@unittest.skipUnless(importlib.util.find_spec("swift"), "ms-swift is not installed")
class SwiftIntegrationTests(unittest.TestCase):
    def test_real_template_context_masks_prompts_and_suffix(self):
        import train.swift_plugin  # registers the four strategies
        from swift.loss_scale import get_loss_scale
        from swift.template import ContextType
        text = response()
        messages = [{"role": "user", "content": "Question"}, {"role": "assistant", "content": text}]
        contexts = ["Question", text, "<|im_end|>"]
        kinds = [ContextType.OTHER, ContextType.RESPONSE, ContextType.SUFFIX]
        for phase in range(1, 5):
            scale = get_loss_scale(f"drscaffold_p{phase}")
            parts, weights = scale(contexts, kinds, messages)
            self.assertEqual("".join(parts), "".join(contexts))
            self.assertEqual(weights[0], 0.0)
            self.assertEqual(weights[-1], 1.0 if phase == 4 else 0.0)
            self.assertEqual(weights[1], 1.0)


if __name__ == "__main__":
    unittest.main()
