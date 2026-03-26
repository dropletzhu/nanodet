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

"""MindSpore activation layers."""

import mindspore.nn as nn


def act_layers(name):
    """Create activation layer by name."""
    if name == "ReLU":
        return nn.ReLU()
    elif name == "LeakyReLU":
        return nn.LeakyReLU(alpha=0.1)
    elif name == "ReLU6":
        return nn.ReLU6()
    elif name == "SELU" or name == "SeLU":
        return nn.SeLU()
    elif name == "ELU":
        return nn.ELU()
    elif name == "GELU":
        return nn.GELU()
    elif name == "PReLU":
        return nn.PReLU()
    elif name == "SiLU":
        return nn.SiLU()
    elif name == "HardSwish" or name == "Hardswish":
        return nn.HSwish()
    elif name is None:
        return nn.Identity()
    else:
        raise ValueError(f"Unknown activation: {name}")
