"""ms-swift 4.0.3 plugin: cumulative binary loss masks for P1 through P4."""

from swift.loss_scale.base import LossScale
from swift.loss_scale.mapping import loss_scale_map
from swift.template import ContextType

from train.schema import split_response


class ScaffoldLossScale(LossScale):
    phase = 1

    def __call__(self, context_list, context_types, messages, **kwargs):
        # Reject malformed responses before Swift handles template suffix tokens.
        if self.phase < 4:
            for context, kind in zip(context_list, context_types):
                if kind == ContextType.RESPONSE:
                    split_response(context, self.phase)
        return super().__call__(context_list, context_types, messages, **kwargs)

    def get_loss_scale(self, context, **kwargs):
        if self.phase == 4:
            return [context], [1.0]
        if isinstance(context, str) and context.startswith('{"thought_process":'):
            return split_response(context, self.phase)
        # Swift also invokes this method for template suffixes (e.g. im_end).
        # They occur after the masked fields and must not re-enable supervision.
        return [context], [0.0]


class EntityLossScale(ScaffoldLossScale):
    phase = 1


class RelationLossScale(ScaffoldLossScale):
    phase = 2


class ReasoningLossScale(ScaffoldLossScale):
    phase = 3


class AnswerLossScale(ScaffoldLossScale):
    phase = 4


for phase, cls in enumerate(
    (EntityLossScale, RelationLossScale, ReasoningLossScale, AnswerLossScale), 1
):
    loss_scale_map[f"drscaffold_p{phase}"] = cls

# Loaded before Swift resolves the requested tuner, for both training and inference.
from swift.tuner_plugin.mapping import tuners_map
from train.phi4 import Phi4Tuner

tuners_map["drscaffold_phi4"] = Phi4Tuner
