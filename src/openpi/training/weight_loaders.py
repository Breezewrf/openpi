import dataclasses
import logging
import re
from typing import Protocol, runtime_checkable

import flax.traverse_util
import numpy as np

import openpi.models.model as _model
import openpi.shared.array_typing as at
import openpi.shared.download as download

logger = logging.getLogger(__name__)


@runtime_checkable
class WeightLoader(Protocol):
    def load(self, params: at.Params) -> at.Params:
        """Loads the model weights.

        Args:
            params: Parameters of the model. This is a nested structure of array-like objects that
                represent the model's parameters.

        Returns:
            Loaded parameters. The structure must be identical to `params`. If returning a subset of
            the parameters the loader must merge the loaded parameters with `params`.
        """


@dataclasses.dataclass(frozen=True)
class NoOpWeightLoader(WeightLoader):
    def load(self, params: at.Params) -> at.Params:
        return params


@dataclasses.dataclass(frozen=True)
class CheckpointWeightLoader(WeightLoader):
    """Loads an entire set of weights from a checkpoint.

    Compatible with:
      trained checkpoints:
        example: "./checkpoints/<config>/<exp>/<step>/params"
      released checkpoints:
        example: "gs://openpi-assets/checkpoints/<model>/params"
    """

    params_path: str

    def load(self, params: at.Params) -> at.Params:
        # We are loading np.ndarray and relying on the training code to properly convert and shard the params.
        loaded_params = _model.restore_params(download.maybe_download(self.params_path), restore_type=np.ndarray)
        # Add all missing LoRA weights.
        return _merge_params(loaded_params, params, missing_regex=".*lora.*")


@dataclasses.dataclass(frozen=True)
class ShapeAdaptedCheckpointWeightLoader(WeightLoader):
    """Load a checkpoint while zero-extending selected parameters to new shapes.

    This is useful when an embodiment needs a larger state/action dimension than the released checkpoint. Existing
    dimensions retain their pretrained values, while newly added rows or columns start at zero and remain trainable.
    Shape adaptation is deliberately restricted by a full-match regular expression so unrelated architecture
    mismatches still fail loudly.
    """

    params_path: str
    adapt_shape_regex: str

    def load(self, params: at.Params) -> at.Params:
        loaded_params = _model.restore_params(download.maybe_download(self.params_path), restore_type=np.ndarray)
        return _merge_params(
            loaded_params,
            params,
            missing_regex=".*lora.*",
            adapt_shape_regex=self.adapt_shape_regex,
        )


@dataclasses.dataclass(frozen=True)
class PaliGemmaWeightLoader(WeightLoader):
    """Loads weights from the official PaliGemma checkpoint.

    This will overwrite existing weights with similar names while keeping all extra weights intact.
    This allows us to support the action expert which is used by the Pi0 model.
    """

    def load(self, params: at.Params) -> at.Params:
        path = download.maybe_download(
            "gs://vertex-model-garden-paligemma-us/paligemma/pt_224.npz", gs={"token": "anon"}
        )
        with path.open("rb") as f:
            flat_params = dict(np.load(f, allow_pickle=False))
        loaded_params = {"PaliGemma": flax.traverse_util.unflatten_dict(flat_params, sep="/")["params"]}
        # Add all missing weights.
        return _merge_params(loaded_params, params, missing_regex=".*")


def _merge_params(
    loaded_params: at.Params,
    params: at.Params,
    *,
    missing_regex: str,
    adapt_shape_regex: str | None = None,
) -> at.Params:
    """Merges the loaded parameters with the reference parameters.

    Args:
        loaded_params: The parameters to merge.
        params: The reference parameters.
        missing_regex: A regex pattern for all missing keys that should be merged from the reference parameters.
        adapt_shape_regex: A full-match regex for checkpoint arrays that may be copied into a differently shaped
            reference array. The overlapping values are preserved and any newly introduced entries are zero-filled.

    Returns:
        A new dictionary with the merged parameters.
    """
    flat_ref = flax.traverse_util.flatten_dict(params, sep="/")
    flat_loaded = flax.traverse_util.flatten_dict(loaded_params, sep="/")

    shape_pattern = re.compile(adapt_shape_regex) if adapt_shape_regex is not None else None

    # First, take all weights that are a subset of the reference weights.
    result = {}
    for k, v in flat_loaded.items():
        if k in flat_ref:
            ref = flat_ref[k]
            loaded_value = v
            if v.shape != ref.shape and shape_pattern is not None and shape_pattern.fullmatch(k):
                logger.info("Adapting checkpoint parameter %s from %s to %s", k, v.shape, ref.shape)
                loaded_value = _zero_extend_array(v, ref.shape)
            result[k] = (
                loaded_value.astype(flat_ref[k].dtype) if loaded_value.dtype != flat_ref[k].dtype else loaded_value
            )

    flat_loaded.clear()

    # Then, merge any missing weights as defined by the missing regex.
    pattern = re.compile(missing_regex)
    for k in {k for k in flat_ref if pattern.fullmatch(k)}:
        if k not in result:
            result[k] = flat_ref[k]

    return flax.traverse_util.unflatten_dict(result, sep="/")


def _zero_extend_array(value: np.ndarray, target_shape: tuple[int, ...]) -> np.ndarray:
    """Copy the overlapping part of an array into a zero-initialized target shape."""
    if value.ndim != len(target_shape):
        raise ValueError(f"Cannot adapt rank-{value.ndim} array to shape {target_shape}")
    if any(target < source for source, target in zip(value.shape, target_shape, strict=True)):
        raise ValueError(f"Cannot zero-extend shape {value.shape} to smaller shape {target_shape}")
    result = np.zeros(target_shape, dtype=value.dtype)
    overlap = tuple(slice(0, min(source, target)) for source, target in zip(value.shape, target_shape, strict=True))
    result[overlap] = value[overlap]
    return result
