import dataclasses

import einops
import numpy as np

from openpi import transforms

G1_STATE_DIM = 30
G1_ACTION_DIM = 34
G1_ARM_ACTION_DIM = 10
G1_ACTION_GROUP_RANGES = {
    "left_arm": (0, 5),
    "right_arm": (5, 10),
    "left_hand": (10, 20),
    "right_hand": (20, 30),
    "navigate_command": (30, 33),
    "base_height_command": (33, 34),
}


def split_action_groups(actions: np.ndarray) -> dict[str, np.ndarray]:
    """Split flat RoboJuDo actions into the groups consumed by the robot runtime."""
    actions = np.asarray(actions)
    if actions.shape[-1] != G1_ACTION_DIM:
        raise ValueError(f"Expected {G1_ACTION_DIM} action dimensions, got shape {actions.shape}")
    return {name: actions[..., start:end] for name, (start, end) in G1_ACTION_GROUP_RANGES.items()}


def make_g1_example() -> dict:
    """Create an example observation in the format expected during inference."""
    return {
        "observation/head_rgb": np.random.randint(256, size=(480, 640, 3), dtype=np.uint8),
        "observation/left_wrist_rgb": np.random.randint(256, size=(480, 640, 3), dtype=np.uint8),
        "observation/right_wrist_rgb": np.random.randint(256, size=(480, 640, 3), dtype=np.uint8),
        "observation/state": np.random.rand(G1_STATE_DIM).astype(np.float32),
        "prompt": "pick up the red cup",
    }


def _parse_image(image: np.ndarray) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        # LeRobot decodes videos as float CHW images in [0, 1].
        image = (255 * image).clip(0, 255).astype(np.uint8)
    if image.ndim != 3:
        raise ValueError(f"Expected a 3D image, got shape {image.shape}")
    if image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    if image.shape[-1] != 3:
        raise ValueError(f"Expected an RGB image, got shape {image.shape}")
    return image


@dataclasses.dataclass(frozen=True)
class G1Inputs(transforms.DataTransformFn):
    """Map G1 observations and training actions into the common pi model format."""

    def __call__(self, data: dict) -> dict:
        state = np.asarray(data["observation/state"], dtype=np.float32)
        if state.shape[-1] != G1_STATE_DIM:
            raise ValueError(f"Expected a {G1_STATE_DIM}-D G1 state, got shape {state.shape}")

        inputs = {
            "state": state,
            "image": {
                "base_0_rgb": _parse_image(data["observation/head_rgb"]),
                "left_wrist_0_rgb": _parse_image(data["observation/left_wrist_rgb"]),
                "right_wrist_0_rgb": _parse_image(data["observation/right_wrist_rgb"]),
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.True_,
                "right_wrist_0_rgb": np.True_,
            },
        }

        if "actions" in data:
            actions = np.asarray(data["actions"], dtype=np.float32)
            if actions.shape[-1] != G1_ACTION_DIM:
                raise ValueError(f"Expected {G1_ACTION_DIM} action dimensions, got shape {actions.shape}")
            inputs["actions"] = actions

        if "prompt" in data:
            prompt = data["prompt"]
            inputs["prompt"] = prompt.decode("utf-8") if isinstance(prompt, bytes) else prompt

        return inputs


@dataclasses.dataclass(frozen=True)
class G1Outputs(transforms.DataTransformFn):
    """Return the complete 34-D RoboJuDo action in its original order."""

    def __call__(self, data: dict) -> dict:
        actions = np.asarray(data["actions"], dtype=np.float32)
        if actions.shape[-1] != G1_ACTION_DIM:
            raise ValueError(f"Expected {G1_ACTION_DIM} model action dimensions, got shape {actions.shape}")
        return {"actions": actions}
