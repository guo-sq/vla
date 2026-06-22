from openpi.training.base_cfg import TestConfig

test_cfg = TestConfig(
    checkpoint_dir="checkpoints/cfg_pi0.5_28_dim.bridge_orig/cfg_pi0.5_28_dim.bridge_orig_exp/10000",
    dataset_root="/mnt/workspace/heyuan/openpi_modified/datasets",
    config="src/openpi/configs/cfg_opensource/cfg_pi0.5_28_dim.bridge_orig.py",
    repo_id="IPEC-COMMUNITY/bridge_orig_lerobot",
    num_batches=1,
    batch_size=1,
    num_workers=0,
    eval_split="val",
    vis_dir="checkpoints/cfg_pi0.5_28_dim.bridge_orig/cfg_pi0.5_28_dim.bridge_orig_exp",
)
