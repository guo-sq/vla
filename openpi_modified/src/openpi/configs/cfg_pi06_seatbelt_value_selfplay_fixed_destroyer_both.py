"""Ablation: destroyer state_confirmation = "both" (was "end_only").

Purpose: test whether forcing destroyer episodes to use BOTH_CONFIRMED
(norm_length = total_steps, clean GT 0 -> -1) improves training signal
vs the default END_CONFIRMED (which uses per_task norm_length and can
produce a flat GT=-1 segment when episode length > norm_length).

Effect on GT:
  builder  BOTH_CONFIRMED: norm_length = total_steps, GT [0, +1] clean (unchanged)
  destroyer BOTH_CONFIRMED: norm_length = total_steps, GT [0, -1] clean (NEW)
  destroyer END_CONFIRMED: norm_length = per_task_percentile, GT may clamp to -1 early

Differences vs _fixed.py:
  - state_confirmation_by_role: {"builder": "both", "destroyer": "end_only"}
    -> {"builder": "both", "destroyer": "both"}
  - exp_name: "..._exp" -> "fixed_destroyer_both_exp"
  - name: unchanged (reuses assets/norm_stats from _fixed)
"""

import dataclasses

from openpi.configs.cfg_pi06_seatbelt_value_selfplay_fixed import cfg as _fixed_cfg
from openpi.training.frame_attributes_preprocessors import ValueReturnsPreprocessor


def _override_state_confirmation(data_config):
    """Replace ValueReturnsPreprocessor with destroyer=both."""
    new_preprocessors = []
    for p in data_config.base_config.frame_attributes_preprocessors:
        if isinstance(p, ValueReturnsPreprocessor):
            new_preprocessors.append(
                dataclasses.replace(
                    p,
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
        ),
    )


cfg = dataclasses.replace(
    _fixed_cfg,
    exp_name="fixed_destroyer_both_exp",
    data=_override_state_confirmation(_fixed_cfg.data),
)
