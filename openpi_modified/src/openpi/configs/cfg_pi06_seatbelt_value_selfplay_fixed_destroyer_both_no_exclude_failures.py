"""Ablation: destroyer BOTH + exclude_failures=False (FP/FN split active).

Combines two orthogonal changes vs _fixed.py:
1. state_confirmation_by_role: destroyer "end_only" -> "both"
   → destroyer GT uses norm_length=total_steps for clean [0, -1] transition
2. exclude_failures: True -> False
   → failure episodes included in training, FP/FN split activated:
     FAILURE_FP (builder-failure) → GT=-1 (UNCONFIRMED_NEGATIVE_END)
     FAILURE_FN (destroyer-failure) → GT=0 (UNCONFIRMED_POSITIVE_END)

Differences vs _destroyer_both.py:
  - exclude_failures: True -> False (failure episodes included)
  - FP/FN split activated (FAILURE_FN GT changes from -1 to 0)
"""

import dataclasses

from openpi.configs.cfg_pi06_seatbelt_value_selfplay_fixed_destroyer_both import (
    cfg as _destroyer_both_cfg,
)
from openpi.training.frame_attributes_preprocessors import ValueReturnsPreprocessor


def _override_exclude_failures(data_config):
    """Replace ValueReturnsPreprocessor with exclude_failures=False."""
    new_preprocessors = []
    for p in data_config.base_config.frame_attributes_preprocessors:
        if isinstance(p, ValueReturnsPreprocessor):
            new_preprocessors.append(dataclasses.replace(p, exclude_failures=False))
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
    _destroyer_both_cfg,
    exp_name="fixed_destroyer_both_no_exclude_failures_exp",
    data=_override_exclude_failures(_destroyer_both_cfg.data),
)
