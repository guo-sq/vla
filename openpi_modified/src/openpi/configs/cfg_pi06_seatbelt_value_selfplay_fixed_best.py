"""Best config: destroyer=both + no_exclude_failures + cross_negative + fixed valid-mask.

Combines four fixes on top of ``_fixed.py``:

1. ``state_confirmation_by_role``: destroyer "end_only" -> "both"
   (from ``_destroyer_both.py`` — clean 0->-1 GT transition for destroyer episodes)

2. ``exclude_failures``: True -> False
   (from ``_destroyer_both_no_exclude_failures.py`` — activates FAILURE_FP/FN split)

3. Cross-negative mechanism (from ``_cross_negative.py``):
   - Distinct prompts: positive="Hang...", negative="Take off..."
   - ``cross_negative_rate=0.5`` in value_net_cfg (50% random prompt flip + GT inversion
     at ``rl_dataset.py:402``)

4. Valid-mask fix (new here):
   - Remove the two ``GripperCountRule(count=-1, invalidate="after")`` rules for
     ``self_play_record_cleaned``. Those rules invalidated every frame after the
     last gripper open, which erroneously dropped the "gripper-open -> return to
     home" motion segment — frames the model still needs to learn to value.
   - Keep ``PruneHeadTailStaticValidMaskPreprocessor`` but raise
     ``trailing_margin_s`` from 0.0 to 0.3s so the last ~9 frames immediately
     before the static home-position segment stay valid. PruneHeadTail already
     only prunes the trailing *static* run, so the robot-returning-to-home
     (non-static) frames were never wrongly pruned by it.

Result: model trains on the full task-execution + return-to-home motion and
only drops the post-home static dwell.

Benchmark evidence (batch.7+8 of v4 cleaned test split, hang prompt):
  - ``destroyer_both_no_exclude_failures @ 10k`` (Run 2): TN Pearson +0.931,
    Builder AUC 0.828, TP Tail MSE 0.0012 — clear improvement over the
    published v4 ``cross_neg_10k`` baseline (TN Pearson +0.943, Builder AUC
    0.566, TP Tail MSE 0.065).
  - ``cross_negative @ 10k`` (Run 1): TN Pearson -0.185, Builder AUC 0.495 —
    collapsed due to inherited ``destroyer="end_only"`` producing an impulse
    (not ramp) GT that is incompatible with the cross_negative flip.

This config adopts the Run-2 winning ``state_confirmation_by_role`` and the
Run-1 cross-negative training signal at the same time, after removing the
valid-mask over-filter that clipped the return-to-home phase for all three
runs.
"""

import dataclasses

from openpi.configs.cfg_pi06_seatbelt_value_selfplay_fixed_destroyer_both_no_exclude_failures import (
    cfg as _base_cfg,
)
from openpi.training.frame_attributes_preprocessors import GripperCountValidMaskPreprocessor
from openpi.training.frame_attributes_preprocessors import PruneHeadTailStaticValidMaskPreprocessor
from openpi.training.frame_attributes_preprocessors import ValidMaskGroupParams
from openpi.training.frame_attributes_preprocessors import ValueReturnsPreprocessor

POSITIVE_PROMPT = "Hang the seatbelt with right hand under 20 seconds."
NEGATIVE_PROMPT = "Take the seatbelt off under 20 seconds."
TRAILING_MARGIN_S = 0.3


def _best_override(data_config):
    """Stack cross_negative prompts + drop over-filter gripper rules + add tail margin."""
    new_preprocessors = []
    for p in data_config.base_config.frame_attributes_preprocessors:
        if isinstance(p, ValueReturnsPreprocessor):
            new_preprocessors.append(
                dataclasses.replace(
                    p,
                    positive_prompt=POSITIVE_PROMPT,
                    negative_prompt=NEGATIVE_PROMPT,
                )
            )
        elif isinstance(p, PruneHeadTailStaticValidMaskPreprocessor):
            new_groups = [dataclasses.replace(g, trailing_margin_s=TRAILING_MARGIN_S) for g in p.groups]
            new_preprocessors.append(dataclasses.replace(p, groups=new_groups))
        elif isinstance(p, GripperCountValidMaskPreprocessor):
            kept_rules = [
                r
                for r in p.rules
                if not (
                    r.batch_contains == "self_play_record_cleaned"
                    and r.event == "open"
                    and r.count == -1
                    and r.invalidate == "after"
                )
            ]
            if kept_rules:
                new_preprocessors.append(dataclasses.replace(p, rules=kept_rules))
            # Drop the processor entirely if all rules were the over-filter ones.
        else:
            new_preprocessors.append(p)

    return dataclasses.replace(
        data_config,
        base_config=dataclasses.replace(
            data_config.base_config,
            frame_attributes_preprocessors=new_preprocessors,
            value_net_cfg={
                **data_config.base_config.value_net_cfg,
                "cross_negative_rate": 0.5,
            },
        ),
    )


cfg = dataclasses.replace(
    _base_cfg,
    exp_name="fixed_best_exp",
    data=_best_override(_base_cfg.data),
)
