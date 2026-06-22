import os
# import yaml
import torch
import matplotlib.pyplot as plt

import json
import logging
import threading
import time
from collections.abc import Callable
from contextlib import nullcontext
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
from pprint import pformat

# import einops
# import gymnasium as gym
import logging
import time
from contextlib import nullcontext
from pprint import pformat
from typing import Any

import torch
from termcolor import colored
from torch.amp import GradScaler
from torch.optim import Optimizer

from lerobot.configs import parser
from lerobot.configs.train import TrainPipelineConfig
from lerobot.datasets.factory import make_dataset
from lerobot.datasets.sampler import EpisodeAwareSampler
from lerobot.datasets.utils import cycle
from lerobot.envs.factory import make_env
from lerobot.optim.factory import make_optimizer_and_scheduler
from lerobot.policies.factory import make_policy
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.policies.utils import get_device_from_parameters
from lerobot.scripts.eval import eval_policy
from lerobot.scripts.visualize_dataset import EpisodeSampler
from lerobot.utils.logging_utils import AverageMeter, MetricsTracker
from lerobot.utils.random_utils import set_seed
from lerobot.utils.train_utils import (
    get_step_checkpoint_dir,
    get_step_identifier,
    load_training_state,
    save_checkpoint,
    update_last_checkpoint,
)
from lerobot.utils.utils import (
    format_big_number,
    get_safe_torch_device,
    has_method,
    init_logging,
)
from lerobot.utils.wandb_utils import WandBLogger


tolerance_s = 1e-4

@parser.wrap()
def eval_openloop(cfg: TrainPipelineConfig):

    cfg.validate()

    save_dir = cfg.output_dir
    dataset = make_dataset(cfg)

    model = make_policy(
        cfg=cfg.policy,
        ds_meta=dataset.meta,
    )
    model.eval()

    device = get_safe_torch_device(cfg.policy.device, log=True)
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True

    logging.info("Creating dataset")
    logging.info("Loading dataset")


    logging.info(f"{dataset.num_frames=} ({format_big_number(dataset.num_frames)})")
    logging.info(f"{dataset.num_episodes=}")
    logging.info("Loading dataloader")
    num_episodes = dataset.num_episodes

    for episode_index in range(dataset.num_episodes):
        plot_name= f"eval_{episode_index}"
        model.reset()

        episode_sampler = EpisodeSampler(dataset, episode_index)
        dataloader = torch.utils.data.DataLoader(
            dataset,
            num_workers=cfg.num_workers,
            batch_size=cfg.batch_size,
            sampler=episode_sampler,
        )
        if len(dataloader) < 1:
            continue

        total_frames = len(dataloader)
        # import pdb; pdb.set_trace()
        pred_horizon = cfg.policy.chunk_size
        action_dim = cfg.policy.output_features["action"].shape[0]
        # n_action_steps = cfg.policy.n_action_steps
        gt_traj = torch.zeros((total_frames, action_dim))
        pred_traj = torch.zeros((total_frames, action_dim))
        # Initialize state trajectory storage
        state_traj = torch.zeros((total_frames, action_dim))
        print("total_frames: ", total_frames, "pred_horizon: ", pred_horizon)

        for idx, batch in enumerate(dataloader):
            # Extract state trajectory
            if "state" in batch:
                # Ensure state dimension matches action dimension for simplicity
                if batch["state"].shape[-1] != action_dim:
                    logging.warning(
                        f"State dimension {batch['state'].shape[-1]} does not match action dimension {action_dim}. Truncating or padding as necessary."
                    )
                    # if batch["state"].shape[-1] > action_dim:
                    #     state_traj[idx : idx + pred_horizon] = batch["state"][0, :action_dim].detach().cpu()
                    # else:
                    #     padded_state = torch.zeros((action_dim,), dtype=batch["state"].dtype)
                    #     padded_state[: batch["state"].shape[-1]] = batch["state"][0].detach().cpu()
                    #     state_traj[idx : idx + pred_horizon] = padded_state
                import pdb; pdb.set_trace()
                state_traj[idx, ...] = batch["state"][0, -1, ...].detach().cpu()

            if idx % pred_horizon == 0 and idx + pred_horizon < total_frames:
                print(idx, pred_horizon, total_frames)

                for key in batch:
                    if isinstance(batch[key], torch.Tensor):
                        batch[key] = batch[key].to(device, non_blocking=device.type == "cuda")
                        print(key, batch[key].shape, batch[key].dtype)
                with (
                    torch.no_grad(),
                    torch.autocast(device_type=device.type) if cfg.policy.use_amp else nullcontext(),
                ):
                    # select_action
                    action = model.predict_action_chunk(
                        batch
                    )
                    print("predict_action_chunk::action.shape: ", action.shape)
                    print("predict_action_chunk::batch action.shape: ", batch["action"].shape)


                    pred_traj[idx : idx + pred_horizon] = action[0, :pred_horizon, ...].detach().cpu()
                    gt_traj[idx : idx + pred_horizon] = batch["action"][0, :pred_horizon, ...].detach().cpu()


                    gt_traj_np = gt_traj.numpy()
                    pred_traj_np = pred_traj.numpy()
                    state_traj_np = state_traj.numpy()

        timesteps = gt_traj.shape[0]

        fig, axs = plt.subplots(action_dim, 1, figsize=(15, 5 * action_dim), sharex=True)
        fig.suptitle("Action and State Comparison for lerobot", fontsize=16)

        for i in range(action_dim):
            axs[i].plot(range(timesteps), gt_traj_np[:, i], label="Ground Truth")
            axs[i].plot(range(timesteps), pred_traj_np[:, i], label="Prediction")
            axs[i].plot(range(timesteps), state_traj_np[:, i], label="State", linestyle="--", alpha=0.7)
            axs[i].set_ylabel(f"Action/State Dim {i+1}")
            axs[i].legend()
            axs[i].grid(True)

        axs[-1].set_xlabel("Timestep")
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(os.path.join(save_dir, plot_name))
        plt.close()


def main():
    init_logging()
    eval_openloop()

if __name__ == "__main__":
    main()