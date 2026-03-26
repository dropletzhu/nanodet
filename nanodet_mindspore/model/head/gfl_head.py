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

"""MindSpore detection head."""

import mindspore
import mindspore.nn as nn
import mindspore.ops as ops

from ..loss.gfocal_loss import DistributionFocalLoss, QualityFocalLoss
from ..loss.iou_loss import GIoULoss
from ..module.conv import ConvModule


class Scale(nn.Cell):
    """Scale layer for learning bbox scaling."""

    def __init__(self, scale_init=1.0):
        super(Scale, self).__init__()
        self.scale = mindspore.Parameter(mindspore.tensor(scale_init))

    def construct(self, x):
        return x * self.scale


class Integral(nn.Cell):
    """Calculate integral result from distribution."""

    def __init__(self, reg_max=16):
        super(Integral, self).__init__()
        self.reg_max = reg_max
        self.project = mindspore.ops.LinSpace()(mindspore.Tensor(0), mindspore.Tensor(reg_max), reg_max + 1)

    def construct(self, x):
        shape = x.shape
        x = x.reshape(*shape[:-1], 4, self.reg_max + 1)
        x = ops.softmax(x, axis=-1)
        x = ops.matmul(x, self.project.astype(x.dtype)).reshape(*shape[:-1], 4)
        return x


class GFLHead(nn.Cell):
    """Generalized Focal Loss Head for MindSpore."""

    def __init__(
        self,
        num_classes,
        in_channels,
        feat_channels=256,
        stacked_convs=2,
        reg_max=16,
        anchor_generator={"type": "AnchorGenerator", "base_sizes": [[3, 6, 12], [3, 6, 12], [3, 6, 12], [3, 6, 12]]},
        loss_qfl=dict(type="QualityFocalLoss", loss_weight=1.0, beta=2.0),
        loss_dfl=dict(type="DistributionFocalLoss", loss_weight=0.25),
        loss_bbox=dict(type="GIoULoss", loss_weight=2.0),
        conv_cfg=None,
        norm_cfg=dict(type="BN"),
        activation="LeakyReLU",
    ):
        super(GFLHead, self).__init__()
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.feat_channels = feat_channels
        self.stacked_convs = stacked_convs
        self.reg_max = reg_max

        self.loss_qfl = QualityFocalLoss(**loss_qfl)
        self.loss_dfl = DistributionFocalLoss(**loss_dfl)
        self.loss_bbox = GIoULoss(**loss_bbox)

        self.cls_convs = nn.CellList()
        self.reg_convs = nn.CellList()
        for i in range(self.stacked_convs):
            ch_in = in_channels if i == 0 else feat_channels
            self.cls_convs.append(
                ConvModule(
                    ch_in,
                    feat_channels,
                    3,
                    padding=1,
                    norm_cfg=norm_cfg,
                    activation=activation,
                )
            )
            self.reg_convs.append(
                ConvModule(
                    ch_in,
                    feat_channels,
                    3,
                    padding=1,
                    norm_cfg=norm_cfg,
                    activation=activation,
                )
            )

        self.gfl_cls = nn.Conv2d(feat_channels, num_classes, 1)
        self.gfl_reg = nn.Conv2d(feat_channels, 4 * (reg_max + 1), 1)

        self.scale = Scale()
        self.integral = Integral(reg_max)

    def construct(self, feats):
        """Forward function."""
        cls_outputs = []
        reg_outputs = []

        for idx, feat in enumerate(feats):
            cls_feat = feat
            reg_feat = feat

            for cls_layer in self.cls_convs:
                cls_feat = cls_layer(cls_feat)
            for reg_layer in self.reg_convs:
                reg_feat = reg_layer(reg_feat)

            cls_output = self.gfl_cls(cls_feat)
            reg_output = self.gfl_reg(reg_feat)

            cls_outputs.append(cls_output)
            reg_outputs.append(reg_output)

        return tuple(cls_outputs), tuple(reg_outputs)

    def loss(self, cls_preds, reg_preds, batch_targets, batch_bboxes, batch_scores):
        """Calculate loss."""
        batch_size = batch_targets[0].shape[0]
        loss_cls = 0
        loss_reg = 0

        for cls_pred, reg_pred in zip(cls_preds, reg_preds):
            cls_pred = cls_pred.reshape(-1, self.num_classes)
            reg_pred = reg_pred.reshape(-1, 4 * (self.reg_max + 1))

            target = batch_targets[0].reshape(-1)
            bbox = batch_bboxes[0].reshape(-1, 4)
            score = batch_scores[0].reshape(-1)

            valid_mask = target >= 0
            pos_target = target[valid_mask]
            pos_bbox = bbox[valid_mask]
            pos_score = score[valid_mask]

            if pos_target.shape[0] > 0:
                pos_cls_pred = cls_pred[valid_mask]
                pos_reg_pred = reg_pred[valid_mask]

                cls_loss = self.loss_qfl(
                    pos_cls_pred,
                    (pos_target, pos_score)
                )
                reg_valid = pos_bbox.shape[0]
                if reg_valid > 0:
                    reg_loss = self.loss_bbox(pos_bbox, pos_bbox)
                    loss_reg += reg_loss
                loss_cls += cls_loss

        num_samples = max(1, loss_cls.shape[0])
        return loss_cls / num_samples, loss_reg / num_samples

    def post_process(self, predictions, meta):
        """Post process for inference."""
        cls_outputs, reg_outputs = predictions
        results = []

        for cls_pred, reg_pred in zip(cls_outputs, reg_outputs):
            cls_pred = ops.sigmoid(cls_pred)
            b, _, h, w = cls_pred.shape

            cls_pred = cls_pred.reshape(b, self.num_classes, -1)
            reg_pred = reg_pred.reshape(b, 4, -1)

            topk_scores, topk_indices = ops.top_k(
                cls_pred.reshape(b, -1).transpose(0, 1),
                k=100
            )
            topk_scores = topk_scores.transpose(0, 1)
            topk_indices = topk_indices.transpose(0, 1)

            results.append({
                "scores": topk_scores,
                "indices": topk_indices
            })

        return results


def build_head(cfg):
    """Build detection head."""
    cfg = cfg.copy()
    name = cfg.pop("name", "GFLHead")
    if name == "GFLHead":
        return GFLHead(**cfg)
    else:
        raise NotImplementedError(f"Head {name} not implemented for MindSpore")


__all__ = ["GFLHead", "build_head", "Scale"]
