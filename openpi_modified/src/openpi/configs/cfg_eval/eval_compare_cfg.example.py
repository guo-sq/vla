from openpi.training.base_cfg import EvalConfig

cfg = EvalConfig(
    targets=(
        "checkpoints/cfg_pi0.5_28_dim.bridge_orig/cfg_pi0.5_28_dim.bridge_orig_exp/10000::src/openpi/configs/cfg_opensource/cfg_pi0.5_28_dim.bridge_orig.py",
        "checkpoints/cfg_pi0.5_28_dim.bridge_orig/cfg_pi0.5_28_dim.bridge_orig_exp/20000::src/openpi/configs/cfg_opensource/cfg_pi0.5_28_dim.bridge_orig.py",
    ),
    dataset_root="/mnt/workspace/heyuan/openpi_modified/datasets",
    eval_split="val",
    num_batches=1,
    batch_size=1,
    num_workers=0,
    report_out="outputs/eval_compare/report.json",
)
