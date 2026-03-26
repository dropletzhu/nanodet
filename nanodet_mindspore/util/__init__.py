# Copyright 2021 RangiLyu.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""MindSpore utility functions."""

import yaml


def load_config(config_path):
    """Load config from yaml file."""
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
    return cfg


def mkdir(path):
    """Create directory."""
    import os
    os.makedirs(path, exist_ok=True)


__all__ = ["load_config", "mkdir"]
