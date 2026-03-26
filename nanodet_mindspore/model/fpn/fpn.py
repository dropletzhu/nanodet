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

"""MindSpore FPN module."""

import mindspore.nn as nn
import mindspore.ops as ops

from ..module.conv import ConvModule
from ..module.init_weights import xavier_init


class FPN(nn.Cell):
    def __init__(
        self,
        in_channels,
        out_channels,
        num_outs,
        start_level=0,
        end_level=-1,
        conv_cfg=None,
        norm_cfg=None,
        activation=None,
    ):
        super(FPN, self).__init__()
        assert isinstance(in_channels, list)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_ins = len(in_channels)
        self.num_outs = num_outs

        if end_level == -1:
            self.backbone_end_level = self.num_ins
            assert num_outs >= self.num_ins - start_level
        else:
            self.backbone_end_level = end_level
            assert end_level <= len(in_channels)
            assert num_outs == end_level - start_level
        self.start_level = start_level
        self.end_level = end_level

        self.lateral_convs = nn.CellList()
        for i in range(self.start_level, self.backbone_end_level):
            l_conv = ConvModule(
                in_channels[i],
                out_channels,
                1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                activation=activation,
                inplace=False,
            )
            self.lateral_convs.append(l_conv)

        self.init_weights()

    def init_weights(self):
        for m in self.cells():
            if isinstance(m, nn.Conv2d):
                xavier_init(m, distribution="uniform")

    def construct(self, inputs):
        assert len(inputs) == len(self.in_channels)

        laterals = [
            self.lateral_convs[i](inputs[i + self.start_level])
            for i in range(len(self.lateral_convs))
        ]

        used_backbone_levels = len(laterals)
        for i in range(used_backbone_levels - 1, 0, -1):
            target_size = (laterals[i - 1].shape[2], laterals[i - 1].shape[3])
            upsampled = ops.ResizeBilinearV2(laterals[i], target_size=target_size)
            laterals[i - 1] = laterals[i - 1] + upsampled

        outs = [laterals[i] for i in range(used_backbone_levels)]
        return tuple(outs)


__all__ = ["FPN"]
