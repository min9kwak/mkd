from __future__ import division

import random
import typing
import warnings
from collections import defaultdict

import numpy as np
import json
import typing
import warnings
from abc import ABC, ABCMeta, abstractmethod
from typing import IO, Any, Callable, Dict, Optional, Tuple, Type, Union


class ReplayCompose(Compose):
    def __init__(
        self,
        transforms,
        bbox_params: typing.Optional[typing.Union[dict, "BboxParams"]] = None,
        keypoint_params: typing.Optional[typing.Union[dict, "KeypointParams"]] = None,
        additional_targets: typing.Optional[typing.Dict[str, str]] = None,
        p: float = 1.0,
        save_key: str = "replay",
    ):
        super(ReplayCompose, self).__init__(transforms, bbox_params, keypoint_params, additional_targets, p)
        self.set_deterministic(True, save_key=save_key)
        self.save_key = save_key

    def __call__(self, *args, force_apply: bool = False, **kwargs) -> typing.Dict[str, typing.Any]:
        kwargs[self.save_key] = defaultdict(dict)
        result = super(ReplayCompose, self).__call__(force_apply=force_apply, **kwargs)
        serialized = self.get_dict_with_id()
        self.fill_with_params(serialized, result[self.save_key])
        self.fill_applied(serialized)
        result[self.save_key] = serialized
        return result

    @staticmethod
    def replay(saved_augmentations: typing.Dict[str, typing.Any], **kwargs) -> typing.Dict[str, typing.Any]:
        augs = ReplayCompose._restore_for_replay(saved_augmentations)
        return augs(force_apply=True, **kwargs)

    @staticmethod
    def _restore_for_replay(
        transform_dict: typing.Dict[str, typing.Any], lambda_transforms: typing.Optional[dict] = None
    ):
        """
        Args:
            lambda_transforms (dict): A dictionary that contains lambda transforms, that
            is instances of the Lambda class.
                This dictionary is required when you are restoring a pipeline that contains lambda transforms. Keys
                in that dictionary should be named same as `name` arguments in respective lambda transforms from
                a serialized pipeline.
        """
        applied = transform_dict["applied"]
        params = transform_dict["params"]
        lmbd = instantiate_nonserializable(transform_dict, lambda_transforms)
        if lmbd:
            transform = lmbd
        else:
            name = transform_dict["__class_fullname__"]
            args = {k: v for k, v in transform_dict.items() if k not in ["__class_fullname__", "applied", "params"]}
            cls = SERIALIZABLE_REGISTRY[name]
            if "transforms" in args:
                args["transforms"] = [
                    ReplayCompose._restore_for_replay(t, lambda_transforms=lambda_transforms)
                    for t in args["transforms"]
                ]
            transform = cls(**args)

        transform = typing.cast(BasicTransform, transform)
        if isinstance(transform, BasicTransform):
            transform.params = params
        transform.replay_mode = True
        transform.applied_in_replay = applied
        return transform

    def fill_with_params(self, serialized: dict, all_params: dict) -> None:
        params = all_params.get(serialized.get("id"))
        serialized["params"] = params
        del serialized["id"]
        for transform in serialized.get("transforms", []):
            self.fill_with_params(transform, all_params)

    def fill_applied(self, serialized: typing.Dict[str, typing.Any]) -> bool:
        if "transforms" in serialized:
            applied = [self.fill_applied(t) for t in serialized["transforms"]]
            serialized["applied"] = any(applied)
        else:
            serialized["applied"] = serialized.get("params") is not None
        return serialized["applied"]

    def _to_dict(self) -> typing.Dict[str, typing.Any]:
        dictionary = super(ReplayCompose, self)._to_dict()
        dictionary.update({"save_key": self.save_key})
        return dictionary


NON_SERIALIZABLE_REGISTRY: Dict[str, "SerializableMeta"] = {}


def instantiate_nonserializable(
    transform: Dict[str, Any], nonserializable: Optional[Dict[str, Any]] = None
):
    if transform.get("__class_fullname__") in NON_SERIALIZABLE_REGISTRY:
        name = transform["__name__"]
        if nonserializable is None:
            raise ValueError(
                "To deserialize a non-serializable transform with name {name} you need to pass a dict with"
                "this transform as the `lambda_transforms` argument".format(name=name)
            )
        result_transform = nonserializable.get(name)
        if transform is None:
            raise ValueError(
                "Non-serializable transform with {name} was not found in `nonserializable`".format(name=name)
            )
        return result_transform
    return None
