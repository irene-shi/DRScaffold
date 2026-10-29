"""Keep Phi-4's pretrained modality adapters frozen alongside task LoRA."""

from types import MethodType

from peft import LoraConfig, PeftModel, get_peft_model
from peft.tuners.lora.layer import LoraLayer
from swift.tuner_plugin.base import PeftTuner


def configure_modality_adapters(model, is_trainable=True):
    base = model.get_base_model()
    if not hasattr(base, "set_lora_adapter") or not hasattr(base, "unset_lora_adapter"):
        raise ValueError("The Phi-4 tuner requires a Phi4MM model with modality adapters")
    layers = [module for module in base.modules() if isinstance(module, LoraLayer)]
    if not any("default" in layer.lora_A for layer in layers):
        raise ValueError("No task LoRA projections matched the Phi-4 target regex")

    def select(self, modality):
        names = [modality] if isinstance(modality, str) else list(modality)
        for layer in layers:
            if layer.merged:
                layer.unmerge()
            active = [name for name in (*names, "default") if name in layer.lora_A]
            layer.set_adapter(active)
            layer.enable_adapters(bool(active))
            # PEFT set_adapter enables gradients; explicitly freeze pretrained LoRA.
            for attr in layer.adapter_layer_names:
                for name, parameters in getattr(layer, attr).items():
                    parameters.requires_grad_(is_trainable and name == "default")

    def text_only(self):
        select(self, [])

    base.set_lora_adapter = MethodType(select, base)
    base.unset_lora_adapter = MethodType(text_only, base)
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(is_trainable and (
            ".lora_A.default." in name or ".lora_B.default." in name))
    base.set_lora_adapter("vision")
    return model


class Phi4Tuner(PeftTuner):
    @staticmethod
    def prepare_model(args, model):
        model.requires_grad_(False)
        config = LoraConfig(
            task_type="CAUSAL_LM", target_modules=args.target_regex,
            r=args.lora_rank, lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout, bias="none",
        )
        return configure_modality_adapters(get_peft_model(model, config))

    @staticmethod
    def from_pretrained(model, model_id, **kwargs):
        model.requires_grad_(False)
        adapted = PeftModel.from_pretrained(model, model_id, **kwargs)
        return configure_modality_adapters(adapted, kwargs.get("is_trainable", False))
