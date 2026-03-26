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

"""MindSpore Quality Focal Loss and Distribution Focal Loss."""

import mindspore
import mindspore.nn as nn
import mindspore.ops as ops
import mindspore.numpy as np_ms


def quality_focal_loss(pred, target, beta=2.0):
    """Quality Focal Loss."""
    assert len(target) == 2, "target for QFL must be a tuple of two elements"

    label, score = target
    pred_sigmoid = ops.sigmoid(pred)
    scale_factor = pred_sigmoid
    zerolabel = ops.zeros_like(pred)

    loss = ops.binary_cross_entropy_with_logits(
        pred, zerolabel, reduction="none"
    ) * scale_factor.pow(beta)

    bg_class_ind = pred.shape[1]
    pos = ops.nonzero((label >= 0) & (label < bg_class_ind)).squeeze(1)
    pos_label = label[pos].astype(mindspore.int64)

    scale_factor = score[pos] - pred_sigmoid[pos, pos_label]
    loss[pos, pos_label] = ops.binary_cross_entropy_with_logits(
        pred[pos, pos_label], score[pos], reduction="none"
    ) * scale_factor.abs().pow(beta)

    loss = loss.sum(axis=1, keepdims=False)
    return loss


def distribution_focal_loss(pred, label):
    """Distribution Focal Loss."""
    dis_left = label.astype(mindspore.int32)
    dis_right = dis_left + 1
    weight_left = dis_right.float() - label.float()
    weight_right = label.float() - dis_left.float()

    loss = (
        ops.cross_entropy(pred, dis_left, reduction="none") * weight_left
        + ops.cross_entropy(pred, dis_right, reduction="none") * weight_right
    )
    return loss


class QualityFocalLoss(nn.Cell):
    """Quality Focal Loss for MindSpore."""

    def __init__(self, use_sigmoid=True, beta=2.0, reduction="mean", loss_weight=1.0):
        super(QualityFocalLoss, self).__init__()
        assert use_sigmoid is True, "Only sigmoid in QFL supported now."
        self.use_sigmoid = use_sigmoid
        self.beta = beta
        self.reduction = reduction
        self.loss_weight = loss_weight

    def construct(self, pred, target, weight=None, avg_factor=None, reduction_override=None):
        """Forward function."""
        assert reduction_override in (None, "none", "mean", "sum")
        reduction = reduction_override if reduction_override else self.reduction

        if self.use_sigmoid:
            loss_cls = self.loss_weight * quality_focal_loss(
                pred,
                target,
                beta=self.beta,
            )
        else:
            raise NotImplementedError

        if reduction == "mean":
            return loss_cls.mean()
        elif reduction == "sum":
            return loss_cls.sum()
        return loss_cls


class DistributionFocalLoss(nn.Cell):
    """Distribution Focal Loss for MindSpore."""

    def __init__(self, reduction="mean", loss_weight=1.0):
        super(DistributionFocalLoss, self).__init__()
        self.reduction = reduction
        self.loss_weight = loss_weight

    def construct(self, pred, target, weight=None, avg_factor=None, reduction_override=None):
        """Forward function."""
        assert reduction_override in (None, "none", "mean", "sum")
        reduction = reduction_override if reduction_override else self.reduction
        loss_cls = self.loss_weight * distribution_focal_loss(pred, target)

        if reduction == "mean":
            return loss_cls.mean()
        elif reduction == "sum":
            return loss_cls.sum()
        return loss_cls


__all__ = ["QualityFocalLoss", "DistributionFocalLoss", "quality_focal_loss", "distribution_focal_loss"]
