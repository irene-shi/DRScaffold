"""Budget, stage identity and decoder target checks for the paper experiments."""

import re
from pathlib import Path
import subprocess
import sys
import unittest

from train.experiments import BACKBONES, CONFIG_DIR, EXPERIMENTS, experiment_config, load_config
from train.run import phase_command


class ExperimentTests(unittest.TestCase):
    def test_every_schedule_preserves_budget_and_stage_identity(self):
        base = load_config(CONFIG_DIR / BACKBONES["qwen"])
        for name, stages in EXPERIMENTS.items():
            config = experiment_config(base, name)
            self.assertAlmostEqual(sum(config["phases"][p - 1]["num_train_epochs"] for p in stages), 3.5)
            for phase in stages:
                cmd = phase_command(config, phase, Path("data"), Path("output"))
                self.assertEqual(cmd[cmd.index("--loss_scale") + 1], f"drscaffold_p{phase}")
                self.assertEqual(cmd[cmd.index("--dataset") + 1],
                                 "data/p4_train.jsonl" if phase == 4 else "data/train.jsonl")
                self.assertEqual(config["phases"][phase - 1]["learning_rate"],
                                 base["phases"][phase - 1]["learning_rate"])
        self.assertEqual(base["phases"][0]["num_train_epochs"], 0.5)
        removed = experiment_config(base, "s134")
        self.assertAlmostEqual(removed["phases"][0]["num_train_epochs"], 0.7)
        self.assertAlmostEqual(removed["phases"][2]["num_train_epochs"], 1.4)

    def test_backbone_targets_select_decoder_attention_only(self):
        positives = {
            "qwen": "model.language_model.layers.0.self_attn.q_proj",
            "internvl": "language_model.model.layers.0.attention.wqkv",
            "phi4": "model.layers.0.self_attn.qkv_proj",
        }
        for backbone, positive in positives.items():
            config = load_config(CONFIG_DIR / BACKBONES[backbone])
            pattern = config["common"]["target_regex"]
            self.assertIsNotNone(re.fullmatch(pattern, positive))
            for negative in ("vision_model.encoder.layers.0.self_attn.qkv_proj",
                             "model.layers.0.mlp.gate_up_proj", "lm_head",
                             "model.embed_tokens", "model.visual.blocks.0.attn.qkv"):
                self.assertIsNone(re.fullmatch(pattern, negative), (backbone, negative))

    def test_reordered_handoff_and_resume_steps(self):
        root = Path(__file__).resolve().parents[2]
        command = [sys.executable, "-m", "train.run", "--experiment", "reverse", "--dry-run"]
        output = subprocess.check_output(command, cwd=root, text=True)
        lines = output.strip().splitlines()
        self.assertEqual(len(lines), 4)
        self.assertNotIn("--adapters", lines[0])
        for i, line in enumerate(lines[1:], 2):
            self.assertIn(f"step{i - 1}_p{6 - i}/checkpoint-LAST", line)
        output = subprocess.check_output(command + ["--start-step", "2", "--end-step", "2",
                                                    "--adapter", "/tmp/previous-adapter"], cwd=root, text=True)
        self.assertIn("Step 2 / P3", output)
        self.assertIn(f"--adapters {Path('/tmp/previous-adapter').resolve()}", output)


if __name__ == "__main__":
    unittest.main()
