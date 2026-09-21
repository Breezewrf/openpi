import numpy as np

from openpi.policies import g1_policy


def test_g1_inputs_for_inference() -> None:
    transformed = g1_policy.G1Inputs()(g1_policy.make_g1_example())

    assert transformed["state"].shape == (g1_policy.G1_STATE_DIM,)
    assert transformed["state"].dtype == np.float32
    assert set(transformed["image"]) == {"base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb"}
    assert all(image.shape == (480, 640, 3) for image in transformed["image"].values())
    assert all(image.dtype == np.uint8 for image in transformed["image"].values())
    assert all(transformed["image_mask"].values())
    assert "actions" not in transformed


def test_g1_inputs_for_training() -> None:
    data = {
        "observation/head_rgb": np.random.rand(3, 480, 640).astype(np.float32),
        "observation/left_wrist_rgb": np.random.rand(3, 480, 640).astype(np.float32),
        "observation/right_wrist_rgb": np.random.rand(3, 480, 640).astype(np.float32),
        "observation/state": np.zeros(30, dtype=np.float32),
        "actions": np.arange(g1_policy.G1_ACTION_HORIZON * 34, dtype=np.float32).reshape(
            g1_policy.G1_ACTION_HORIZON, 34
        ),
        "prompt": b"pick up the red cup",
    }

    transformed = g1_policy.G1Inputs()(data)

    assert transformed["actions"].shape == (g1_policy.G1_ACTION_HORIZON, 34)
    np.testing.assert_array_equal(transformed["actions"], data["actions"])
    assert transformed["prompt"] == "pick up the red cup"
    assert all(image.shape == (480, 640, 3) for image in transformed["image"].values())


def test_g1_outputs_preserve_complete_action() -> None:
    actions = np.arange(g1_policy.G1_ACTION_HORIZON * 34, dtype=np.float32).reshape(g1_policy.G1_ACTION_HORIZON, 34)
    transformed = g1_policy.G1Outputs()({"actions": actions})

    assert transformed["actions"].shape == (g1_policy.G1_ACTION_HORIZON, 34)
    np.testing.assert_array_equal(transformed["actions"], actions)


def test_split_action_groups_matches_robojudo_layout() -> None:
    actions = np.arange(2 * 34, dtype=np.float32).reshape(2, 34)

    groups = g1_policy.split_action_groups(actions)

    assert {key: value.shape for key, value in groups.items()} == {
        "left_arm": (2, 5),
        "right_arm": (2, 5),
        "left_hand": (2, 10),
        "right_hand": (2, 10),
        "navigate_command": (2, 3),
        "base_height_command": (2, 1),
    }
    np.testing.assert_array_equal(groups["navigate_command"], actions[:, 30:33])
    np.testing.assert_array_equal(groups["base_height_command"], actions[:, 33:34])
