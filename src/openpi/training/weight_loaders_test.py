import jax
import numpy as np
import pytest

from openpi.training import weight_loaders


def test_zero_extend_array() -> None:
    source = np.arange(6, dtype=np.float32).reshape(2, 3)

    result = weight_loaders._zero_extend_array(source, (4, 5))  # noqa: SLF001

    assert result.shape == (4, 5)
    np.testing.assert_array_equal(result[:2, :3], source)
    np.testing.assert_array_equal(result[2:, :], 0)
    np.testing.assert_array_equal(result[:, 3:], 0)


def test_merge_params_adapts_only_selected_shapes() -> None:
    loaded = {
        "action_in_proj": {"kernel": np.arange(6, dtype=np.float32).reshape(2, 3)},
        "other": {"kernel": np.ones((2, 2), dtype=np.float32)},
    }
    reference = {
        "action_in_proj": {"kernel": jax.ShapeDtypeStruct((4, 3), np.float32)},
        "other": {"kernel": jax.ShapeDtypeStruct((2, 2), np.float32)},
    }

    result = weight_loaders._merge_params(  # noqa: SLF001
        loaded,
        reference,
        missing_regex=".*lora.*",
        adapt_shape_regex=r"action_in_proj/kernel",
    )

    assert result["action_in_proj"]["kernel"].shape == (4, 3)
    np.testing.assert_array_equal(result["action_in_proj"]["kernel"][:2], loaded["action_in_proj"]["kernel"])
    np.testing.assert_array_equal(result["action_in_proj"]["kernel"][2:], 0)
    np.testing.assert_array_equal(result["other"]["kernel"], loaded["other"]["kernel"])


def test_merge_params_expands_pi05_action_projections_from_32_to_34() -> None:
    loaded = {
        "action_in_proj": {
            "kernel": np.ones((32, 4), dtype=np.float32),
            "bias": np.ones((4,), dtype=np.float32),
        },
        "action_out_proj": {
            "kernel": np.ones((4, 32), dtype=np.float32),
            "bias": np.ones((32,), dtype=np.float32),
        },
    }
    reference = {
        "action_in_proj": {
            "kernel": jax.ShapeDtypeStruct((34, 4), np.float32),
            "bias": jax.ShapeDtypeStruct((4,), np.float32),
        },
        "action_out_proj": {
            "kernel": jax.ShapeDtypeStruct((4, 34), np.float32),
            "bias": jax.ShapeDtypeStruct((34,), np.float32),
        },
    }

    result = weight_loaders._merge_params(  # noqa: SLF001
        loaded,
        reference,
        missing_regex=r".*lora.*",
        adapt_shape_regex=r"(?:action_in_proj/kernel|action_out_proj/(?:kernel|bias))",
    )

    np.testing.assert_array_equal(result["action_in_proj"]["kernel"][:32], 1)
    np.testing.assert_array_equal(result["action_in_proj"]["kernel"][32:], 0)
    np.testing.assert_array_equal(result["action_out_proj"]["kernel"][:, :32], 1)
    np.testing.assert_array_equal(result["action_out_proj"]["kernel"][:, 32:], 0)
    np.testing.assert_array_equal(result["action_out_proj"]["bias"][:32], 1)
    np.testing.assert_array_equal(result["action_out_proj"]["bias"][32:], 0)
    np.testing.assert_array_equal(result["action_in_proj"]["bias"], 1)


def test_zero_extend_array_rejects_smaller_target() -> None:
    with pytest.raises(ValueError, match="smaller shape"):
        weight_loaders._zero_extend_array(np.ones((4, 4), dtype=np.float32), (3, 4))  # noqa: SLF001
