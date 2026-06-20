# SPDX-License-Identifier: Apache-2.0
"""Utilities for loading diffusers-style component configs."""

import json
import os
from typing import Any


def get_diffusers_config(model: str) -> dict[str, Any]:
    """Load a local diffusers component config."""
    config_name = "config.json"
    if "scheduler" in model:
        config_name = "scheduler_config.json"

    if not os.path.exists(model):
        raise RuntimeError(f"Diffusers config file not found at {model}")

    config_file = os.path.join(model, config_name)
    if not os.path.exists(config_file):
        raise RuntimeError(f"Config file not found at {config_file}")

    try:
        with open(config_file) as f:
            config_dict: dict[str, Any] = json.load(f)
    except Exception as e:
        raise RuntimeError(
            f"Failed to load diffusers config from {config_file}: {e}"
        ) from e

    config_dict.pop("_diffusers_version", None)
    return config_dict
