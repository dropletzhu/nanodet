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

"""MindSpore weight initialization."""

import mindspore.nn as nn
from mindspore.common import initializer


def kaiming_init(module, a=0, mode="fan_out", nonlinearity="relu", bias=0, distribution="normal"):
    """Kaiming initialization using HeNormal."""
    if distribution == "uniform":
        module.weight.set_data(
            initializer.initializer(initializer.XavierUniform(), module.weight.shape)
        )
    else:
        module.weight.set_data(
            initializer.initializer(initializer.HeNormal(), module.weight.shape)
        )
    if hasattr(module, "bias") and module.bias is not None:
        module.bias.set_data(initializer.initializer(initializer.Constant(bias), module.bias.shape))


def xavier_init(module, gain=1, bias=0, distribution="normal"):
    """Xavier initialization."""
    if distribution == "uniform":
        module.weight.set_data(
            initializer.initializer(initializer.XavierUniform(gain=gain), module.weight.shape)
        )
    else:
        module.weight.set_data(
            initializer.initializer(initializer.XavierNormal(gain=gain), module.weight.shape)
        )
    if hasattr(module, "bias") and module.bias is not None:
        module.bias.set_data(initializer.initializer(initializer.Constant(bias), module.bias.shape))


def normal_init(module, mean=0, std=1, bias=0):
    """Normal initialization."""
    module.weight.set_data(
        initializer.initializer(initializer.Normal(mean=mean, sigma=std), module.weight.shape)
    )
    if hasattr(module, "bias") and module.bias is not None:
        module.bias.set_data(initializer.initializer(initializer.Constant(bias), module.bias.shape))


def constant_init(module, val, bias=0):
    """Constant initialization."""
    if hasattr(module, "weight") and module.weight is not None:
        module.weight.set_data(initializer.initializer(initializer.Constant(val), module.weight.shape))
    if hasattr(module, "bias") and module.bias is not None:
        module.bias.set_data(initializer.initializer(initializer.Constant(bias), module.bias.shape))
