"""Ablation: fixed selfplay config without exclude_failures.

Purpose: verify whether `exclude_failures=True` is a necessary defense line
when the other 4 fixes are already in place (state_confirmation_by_role,
unified prompt, dirty-batch exclusion, train/val split).

Interpretation:
- If collapses like v3  -> exclude_failures is *necessary*
- If trains like fixed -> other fixes are sufficient without exclude_failures
  (and diagnose's GT-distribution evidence should be re-interpreted)

Scope:
- Single-factor ablation. Cannot quantify individual contributions because
  the 4 remaining fixes may have interactions.
- Cannot identify the root cause of v3 collapse (that needs a reverse
  experiment matrix: v3 + single fix).

Differences vs _fixed.py:
- exclude_failures: True -> False
- exp_name: "..._exp" -> "fixed_ablation_no_exclude_failures_exp"
- name: unchanged (reuses assets/norm_stats from _fixed)
"""

import dataclasses

from openpi.configs.cfg_pi06_seatbelt_value_selfplay_fixed import cfg as _fixed_cfg
from openpi.training.frame_attributes_preprocessors import ValueReturnsPreprocessor


def _override_value_returns(data_config):
    """Replace ValueReturnsPreprocessor in the pipeline with exclude_failures=False."""
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
    _fixed_cfg,
    # keep `name` identical to _fixed so assets/norm_stats are reused
    exp_name="fixed_ablation_no_exclude_failures_exp",
    data=_override_value_returns(_fixed_cfg.data),
)
