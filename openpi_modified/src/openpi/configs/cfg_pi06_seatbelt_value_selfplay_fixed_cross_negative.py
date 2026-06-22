"""Cross-negative ablation: distinct prompts + random prompt flip.

Based on _fixed.py (exclude_failures=True, destroyer=end_only), adds:
1. Distinct prompts: positive="Hang...", negative="Take off..."
2. cross_negative_rate=0.5 in value_net_cfg

At __getitem__ time, with 50% probability:
  - Swap prompt to opposite (Hang <-> Take off)
  - Flip GT returns: -(1 + v) inverts [-1, 0] range

This forces the model to learn prompt-conditioned value prediction:
the same trajectory under different prompts gets opposite GT.

Validated by dual-prompt benchmark: old cross_neg_10k was the ONLY
prompt-conditioned model (hang Pearson 0.971 + takeoff Pearson 0.971).
See insight_dual_prompt_is_load_bearing.md for full evidence.

Differences vs _fixed.py:
  - positive_prompt: "Hang..." (unchanged)
  - negative_prompt: "Hang..." -> "Take the seatbelt off under 20 seconds."
  - state_confirmation_by_role["destroyer"]: "end_only" -> "both"
    (end_only produces impulse GT that stays impulse after cross_negative's
    -(1+returns) flip, giving no ramp signal; "both" yields a clean 0->-1
    ramp for destroyer that flips into a -1->0 ramp — the shape the model
    must learn.)
  - value_net_cfg: added cross_negative_rate=0.5
"""

import dataclasses

from openpi.configs.cfg_pi06_seatbelt_value_selfplay_fixed import cfg as _fixed_cfg
from openpi.training.frame_attributes_preprocessors import ValueReturnsPreprocessor

POSITIVE_PROMPT = "Hang the seatbelt with right hand under 20 seconds."
NEGATIVE_PROMPT = "Take the seatbelt off under 20 seconds."


def _override_prompts_and_rate(data_config):
    """Override prompts to be distinct, force destroyer="both", add cross_negative_rate.

    destroyer="both" (was inherited "end_only" from _fixed.py) is required for
    cross_negative to work: end_only produces a flat GT=-1 with a single 0 at
    the end frame (impulse), and cross_negative's -(1+returns) flip on such a
    shape still yields an impulse — the model gets no gradient signal for
    prompt-conditioned ramp learning.
    """
    new_preprocessors = []
    for p in data_config.base_config.frame_attributes_preprocessors:
        if isinstance(p, ValueReturnsPreprocessor):
            new_preprocessors.append(
                dataclasses.replace(
                    p,
                    positive_prompt=POSITIVE_PROMPT,
                    negative_prompt=NEGATIVE_PROMPT,
                    state_confirmation_by_role={
                        "builder": "both",
                        "destroyer": "both",
                    },
                )
            )
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
    _fixed_cfg,
    exp_name="fixed_cross_negative_exp",
    data=_override_prompts_and_rate(_fixed_cfg.data),
)
