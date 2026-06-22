import dataclasses

import einops
import numpy as np
import torch

from openpi import transforms
from openpi.models import model as _model
import openpi.training.utils as _utils


# 说是gr00t，其实面向的是用lerobot-record采的数据格式
def make_gr00t_lerobot_example() -> dict:
    """Creates a random input example for the Libero policy."""
    return {
        "observation/state": np.random.rand(8),
        "observation/front_image": np.random.randint(
            256, size=(224, 224, 3), dtype=np.uint8
        ),
        "observation/wrist_image": np.random.randint(
            256, size=(224, 224, 3), dtype=np.uint8
        ),
        "observation/wrist_image_lf": np.random.randint(
            256, size=(224, 224, 3), dtype=np.uint8
        ),
        "prompt": "do something",
    }


def _parse_image(image) -> np.ndarray:
    if image is None:
        return np.random.randint(256, size=(224, 224, 3), dtype=np.uint8)
    else:
        image = np.asarray(image)
        if np.issubdtype(image.dtype, np.floating):
            image = (255 * image).astype(np.uint8)
        if image.shape[0] == 3:
            image = einops.rearrange(image, "c h w -> h w c")
        return image


@dataclasses.dataclass(frozen=True)
class Gr00tLerobotInputs(transforms.DataTransformFn):
    """
    This class is used to convert inputs to the model to the expected format. It is used for both training and inference.

    For your own dataset, you can copy this class and modify the keys based on the comments below to pipe
    the correct elements of your dataset into the model.
    """

    # Determines which model will be used.
    # Do not change this for your own dataset.
    model_type: _model.ModelType
    unify_action_mode: bool = False
    align_dim: int = 28
    robot_type: str = "bi_piper_follower"

    def update_state(self, ori_state):
        # define state& action
        state = torch.zeros(
            self.align_dim,
            dtype=torch.float32,
            # device=ori_state.device,
        )

        align_info = _utils.TEST_ROBOT_ALIGN_INFO[self.robot_type]
        for (
            robot_part_name,
            target_dof,
        ) in align_info.get_state_name_dict().items():
            for state_index, tgt_dof in target_dof.items():
                state[tgt_dof] = torch.tensor(ori_state[state_index])

        # update joint state& action
        return state

    def __call__(self, data: dict) -> dict:
        # Possibly need to parse images to uint8 (H,W,C) since LeRobot automatically
        # stores as float32 (C,H,W), gets skipped for policy inference.
        # Keep this for your own dataset, but if your dataset stores the images
        # in a different key than "observation/image" or "observation/wrist_image",
        # you should change it below.
        # Pi0 models support three image inputs at the moment: one third-person view,
        # and two wrist views (left and right). If your dataset does not have a particular type
        # of image, e.g. wrist images, you can comment it out here and replace it with zeros like we do for the
        # right wrist image below.
        # print("------gr00t_policy-------")
        # print(data.keys())
        head_img = _parse_image(data.get("observation/front_image", None))
        # left是一个固定的不动的视频
        lf_wrist_img = _parse_image(data.get("observation/wrist_image_lf", None))
        rt_wrist_img = _parse_image(data.get("observation/wrist_image", None))
        third_view_img = _parse_image(data.get("observation/third_view_image", None))

        # Create inputs dict. Do not change the keys in the dict below.
        if not self.unify_action_mode:
            state = data["observation/state"]
        else:
            # 从14维映射为align_dim
            assert (
                self.robot_type in _utils.TEST_ROBOT_ALIGN_INFO
            ), f"Robot type {self.robot_type} not in robot align info"
            state = self.update_state(data["observation/state"])

        inputs = {
            "state": state,
            "image": {
                "base_0_rgb": head_img,
                "left_wrist_0_rgb": lf_wrist_img,
                # Pad any non-existent images with zero-arrays of the appropriate shape.
                "right_wrist_0_rgb": rt_wrist_img,
                "third_view_0_rgb": third_view_img,
            },
            "image_mask": {
                "base_0_rgb": (
                    np.True_
                    if data.get("observation/front_image", None) is not None
                    else np.False_
                ),
                "left_wrist_0_rgb": (
                    np.True_
                    if data.get("observation/wrist_image_lf", None) is not None
                    else np.False_
                ),
                # We only mask padding images for pi0 model, not pi0-FAST. Do not change this for your own dataset.
                # "right_wrist_0_rgb": np.True_ if self.model_type == _model.ModelType.PI0_FAST else np.False_,
                "right_wrist_0_rgb": (
                    np.True_
                    if data.get("observation/wrist_image", None) is not None
                    else np.False_
                ),
                "third_view_0_rgb": (
                    np.True_
                    if data.get("observation/third_view_image", None) is not None
                    else np.False_
                ),
            },
        }

        # Pad actions to the model action dimension. Keep this for your own dataset.
        # Actions are only available during training.
        if "action" in data:
            inputs["actions"] = data["action"]
        if "action_mask" in data:
            # TODO(heyuan): check if action_mask is properly processed
            inputs["actions_mask"] = data["action_mask"]
            if "joint_eef_dof_mask" in data:
                inputs["joint_eef_dof_mask"] = data["joint_eef_dof_mask"]

        # Pass the prompt (aka language instruction) to the model.
        # Keep this for your own dataset (but modify the key if the instruction is not
        # stored in "prompt"; the output dict always needs to have the key "prompt").
        # print(f"--prompt in data:{'prompt' in data}")
        if "prompt" in data:
            inputs["prompt"] = data["prompt"]
        # print(f"---inputs:{inputs.keys()}")
        if "returns" in data:
            inputs["returns"] = data["returns"]
        if "frame_index" in data:
            inputs["frame_index"] = data["frame_index"]
        if "episode_index" in data:
            inputs["episode_index"] = data["episode_index"]
        if "subtask" in data:
            inputs["subtask"] = data["subtask"]
        if "robot_type" in data:
            inputs["robot_type"] = data["robot_type"]
        if "episode_index" in data:
            inputs["episode_index"] = data["episode_index"]
        if "optimality" in data:
            inputs["optimality"] = data["optimality"]
        return inputs


@dataclasses.dataclass(frozen=True)
class Gr00tLerobotOutputs(transforms.DataTransformFn):
    """
    This class is used to convert outputs from the model back the the dataset specific format. It is
    used for inference only.

    For your own dataset, you can copy this class and modify the action dimension based on the comments below.
    """

    target_action_dim: list = range(14)

    def __call__(self, data: dict) -> dict:
        # Only return the first N actions -- since we padded actions above to fit the model action
        # dimension, we need to now parse out the correct number of actions in the return dict.
        # For Libero, we only return the first 7 actions (since the rest is padding).
        # For your own dataset, replace `7` with the action dimension of your dataset.
        return {"actions": np.asarray(data["actions"][..., self.target_action_dim])}
