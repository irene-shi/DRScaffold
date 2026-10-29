"""Exercise Phi modality routing with real PEFT layers and a tiny CPU decoder."""

import importlib.util
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest


@unittest.skipUnless(importlib.util.find_spec("swift"), "ms-swift is not installed")
class Phi4AdapterTests(unittest.TestCase):
    def test_frozen_modality_adapters_gradient_and_checkpoint_roundtrip(self):
        import torch
        from peft import LoraConfig, get_peft_model
        from peft.tuners.lora.layer import LoraLayer
        from transformers import LlamaConfig, LlamaForCausalLM
        from train.phi4 import Phi4Tuner

        class TinyPhi(LlamaForCausalLM):
            def __init__(self):
                super().__init__(LlamaConfig(vocab_size=32, hidden_size=16, intermediate_size=32,
                                            num_hidden_layers=1, num_attention_heads=2,
                                            num_key_value_heads=2))
                pretrained = get_peft_model(self.model, LoraConfig(r=2, lora_alpha=2,
                                            target_modules=["q_proj", "o_proj", "up_proj"]),
                                            adapter_name="vision")
                pretrained.add_adapter("speech", LoraConfig(r=2, lora_alpha=2,
                                       target_modules=["q_proj", "o_proj", "up_proj"]))

            def set_lora_adapter(self, name):
                raise AssertionError("Modality routing was not configured")

            def unset_lora_adapter(self):
                raise AssertionError("Text routing was not configured")

        def fresh():
            torch.manual_seed(42)
            return TinyPhi()

        args = SimpleNamespace(target_regex=r"model\.layers\.\d+\.self_attn\.(q_proj|o_proj)",
                               lora_rank=2, lora_alpha=2, lora_dropout=0.0)
        model = Phi4Tuner.prepare_model(args, fresh())
        tokens = torch.tensor([[1, 2, 3, 4]])
        for modality in ("vision", "speech", None):
            base = model.get_base_model()
            if modality:
                base.set_lora_adapter(modality)
            else:
                base.unset_lora_adapter()
            for name, parameter in model.named_parameters():
                self.assertEqual(parameter.requires_grad, ".default." in name)
            for layer in model.modules():
                if isinstance(layer, LoraLayer) and "default" in layer.lora_A:
                    self.assertIn("default", layer.active_adapters)
                    self.assertFalse(layer.disable_adapters)
            model.zero_grad(set_to_none=True)
            model(input_ids=tokens, labels=tokens).loss.backward()
            gradients = [p.grad for n, p in model.named_parameters() if ".lora_B.default." in n]
            self.assertTrue(all(g is not None for g in gradients))
            self.assertTrue(any(torch.count_nonzero(g) for g in gradients))
            self.assertTrue(all(p.grad is None for n, p in model.named_parameters() if ".default." not in n))
        with torch.no_grad():
            for name, parameter in model.named_parameters():
                if ".lora_B.default." in name:
                    parameter.add_(0.1)
        model.eval()
        model.get_base_model().set_lora_adapter("vision")
        expected = model(input_ids=tokens).logits.detach()
        with tempfile.TemporaryDirectory() as directory:
            Phi4Tuner.save_pretrained(model, directory)
            self.assertTrue((Path(directory) / "adapter_model.safetensors").is_file())
            restored = Phi4Tuner.from_pretrained(fresh(), directory, is_trainable=True).eval()
            torch.testing.assert_close(restored(input_ids=tokens).logits, expected)
            frozen = Phi4Tuner.from_pretrained(fresh(), directory, is_trainable=False)
            frozen.get_base_model().set_lora_adapter("vision")
            self.assertFalse(any(p.requires_grad for p in frozen.parameters()))


if __name__ == "__main__":
    unittest.main()
