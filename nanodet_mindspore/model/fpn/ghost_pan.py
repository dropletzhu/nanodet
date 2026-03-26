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

"""MindSpore GhostPAN module."""

import mindspore.nn as nn
import mindspore.ops as ops

from ..module.conv import ConvModule, DepthwiseConvModule


class GhostBlocks(nn.Cell):
    """Stack of GhostBlocks for MindSpore."""

    def __init__(
        self,
        in_channels,
        out_channels,
        expand=1,
        kernel_size=5,
        num_blocks=1,
        use_res=False,
        activation="LeakyReLU",
    ):
        super(GhostBlocks, self).__init__()
        self.use_res = use_res
        if use_res:
            self.reduce_conv = ConvModule(
                in_channels,
                out_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                activation=activation,
            )

        self.blocks = nn.CellList()
        for _ in range(num_blocks):
            self.blocks.append(
                ConvModule(
                    in_channels,
                    out_channels,
                    kernel_size,
                    padding=kernel_size // 2,
                    activation=activation,
                )
            )

    def construct(self, x):
        out = x
        for block in self.blocks:
            out = block(out)
        if self.use_res:
            out = out + self.reduce_conv(x)
        return out


class GhostPAN(nn.Cell):
    """Path Aggregation Network with Ghost block for MindSpore."""

    def __init__(
        self,
        in_channels,
        out_channels,
        use_depthwise=False,
        kernel_size=5,
        expand=1,
        num_blocks=1,
        use_res=False,
        num_extra_level=0,
        upsample_cfg=dict(scale_factor=2.0, mode="bilinear"),
        norm_cfg=dict(type="BN"),
        activation="LeakyReLU",
    ):
        super(GhostPAN, self).__init__()
        assert num_extra_level >= 0
        assert num_blocks >= 1
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.upsample_scale = upsample_cfg.get("scale_factor", 2.0)
        conv = DepthwiseConvModule if use_depthwise else ConvModule

        self.reduce_layers = nn.CellList()
        for idx in range(len(in_channels)):
            self.reduce_layers.append(
                ConvModule(
                    in_channels[idx],
                    out_channels,
                    1,
                    norm_cfg=norm_cfg,
                    activation=activation,
                )
            )

        self.top_down_blocks = nn.CellList()
        for idx in range(len(in_channels) - 1, 0, -1):
            self.top_down_blocks.append(
                GhostBlocks(
                    out_channels * 2,
                    out_channels,
                    expand,
                    kernel_size=kernel_size,
                    num_blocks=num_blocks,
                    use_res=use_res,
                    activation=activation,
                )
            )

        self.downsamples = nn.CellList()
        self.bottom_up_blocks = nn.CellList()
        for idx in range(len(in_channels) - 1):
            self.downsamples.append(
                conv(
                    out_channels,
                    out_channels,
                    kernel_size,
                    stride=2,
                    padding=kernel_size // 2,
                    norm_cfg=norm_cfg,
                    activation=activation,
                )
            )
            self.bottom_up_blocks.append(
                GhostBlocks(
                    out_channels * 2,
                    out_channels,
                    expand,
                    kernel_size=kernel_size,
                    num_blocks=num_blocks,
                    use_res=use_res,
                    activation=activation,
                )
            )

        self.extra_lvl_in_conv = nn.CellList()
        self.extra_lvl_out_conv = nn.CellList()
        for i in range(num_extra_level):
            self.extra_lvl_in_conv.append(
                conv(
                    out_channels,
                    out_channels,
                    kernel_size,
                    stride=2,
                    padding=kernel_size // 2,
                    norm_cfg=norm_cfg,
                    activation=activation,
                )
            )
            self.extra_lvl_out_conv.append(
                conv(
                    out_channels,
                    out_channels,
                    kernel_size,
                    stride=2,
                    padding=kernel_size // 2,
                    norm_cfg=norm_cfg,
                    activation=activation,
                )
            )

    def construct(self, inputs):
        assert len(inputs) == len(self.in_channels)

        inputs = [
            reduce(input_x) for input_x, reduce in zip(inputs, self.reduce_layers)
        ]

        inner_outs = [inputs[-1]]
        for idx in range(len(self.in_channels) - 1, 0, -1):
            feat_heigh = inner_outs[0]
            feat_low = inputs[idx - 1]
            inner_outs[0] = feat_heigh

            upsample_feat = ops.interpolate(feat_heigh, scale_factor=self.upsample_scale, mode="bilinear")

            inner_out = self.top_down_blocks[len(self.in_channels) - 1 - idx](
                ops.concat([upsample_feat, feat_low], axis=1)
            )
            inner_outs.insert(0, inner_out)

        outs = [inner_outs[0]]
        for idx in range(len(self.in_channels) - 1):
            feat_low = outs[-1]
            feat_height = inner_outs[idx + 1]
            downsample_feat = self.downsamples[idx](feat_low)
            out = self.bottom_up_blocks[idx](
                ops.concat([downsample_feat, feat_height], axis=1)
            )
            outs.append(out)

        for extra_in_layer, extra_out_layer in zip(
            self.extra_lvl_in_conv, self.extra_lvl_out_conv
        ):
            outs.append(extra_in_layer(inputs[-1]) + extra_out_layer(outs[-1]))

        return tuple(outs)


__all__ = ["GhostPAN", "GhostBlocks"]
