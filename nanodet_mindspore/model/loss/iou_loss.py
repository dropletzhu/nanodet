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

"""MindSpore IoU Loss."""

import mindspore
import mindspore.nn as nn
import mindspore.ops as ops


def bbox_overlaps(bboxes1, bboxes2, mode="iou", is_aligned=False, eps=1e-6):
    """Calculate bbox overlaps."""
    assert mode in ["iou", "iof", "giou"], f"Unsupported mode {mode}"

    if bboxes1.shape[-1] != 4 or bboxes1.shape[0] == 0:
        raise ValueError("bboxes1 shape error")
    if bboxes2.shape[-1] != 4 or bboxes2.shape[0] == 0:
        raise ValueError("bboxes2 shape error")

    batch_shape = bboxes1.shape[:-2]
    rows = bboxes1.shape[-2]
    cols = bboxes2.shape[-2]

    if rows * cols == 0:
        if is_aligned:
            return mindspore.zeros(batch_shape + (rows,))
        else:
            return mindspore.zeros(batch_shape + (rows, cols))

    area1 = (bboxes1[..., 2] - bboxes1[..., 0]) * (bboxes1[..., 3] - bboxes1[..., 1])
    area2 = (bboxes2[..., 2] - bboxes2[..., 0]) * (bboxes2[..., 3] - bboxes2[..., 1])

    if is_aligned:
        lt = ops.maximum(bboxes1[..., :2], bboxes2[..., :2])
        rb = ops.minimum(bboxes1[..., 2:], bboxes2[..., 2:])
        wh = (rb - lt).clamp(min=0)
        inter = wh[..., 0] * wh[..., 1]

        if mode == "iou":
            union = area1 + area2 - inter
            return (inter / (union + eps)).squeeze(-1)
        else:
            return (inter / (area1 + eps)).squeeze(-1)
    else:
        lt = ops.maximum(bboxes1[..., None, :2], bboxes2[..., :2])
        rb = ops.minimum(bboxes1[..., None, 2:], bboxes2[..., 2:])
        wh = (rb - lt).clamp(min=0)
        inter = wh[..., 0] * wh[..., 1]

        if mode == "iou":
            union = area1[..., None] + area2 - inter
            return (inter / (union + eps))
        else:
            return (inter / (area1[..., None] + eps))


class IoULoss(nn.Cell):
    """IoU Loss for MindSpore."""

    def __init__(self, loss_weight=1.0, eps=1e-6, reduction="mean", loss_name="loss_iou"):
        super(IoULoss, self).__init__()
        self.loss_weight = loss_weight
        self.eps = eps
        self.reduction = reduction
        self.loss_name = loss_name

    def construct(self, pred, target, weight=None, avg_factor=None):
        """Forward function."""
        loss = self.loss_weight * self.bbox_loss(pred, target, weight, avg_factor)
        return loss

    def bbox_loss(self, pred, target, weight=None, avg_factor=None):
        """Calculate IoU loss."""
        ious = bbox_overlaps(pred, target, mode="iou")
        loss = 1 - ious
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


class GIoULoss(nn.Cell):
    """GIoU Loss for MindSpore."""

    def __init__(self, loss_weight=1.0, eps=1e-6, reduction="mean", loss_name="loss_giou"):
        super(GIoULoss, self).__init__()
        self.loss_weight = loss_weight
        self.eps = eps
        self.reduction = reduction
        self.loss_name = loss_name

    def construct(self, pred, target, weight=None, avg_factor=None):
        """Forward function."""
        loss = self.loss_weight * self.bbox_loss(pred, target, weight, avg_factor)
        return loss

    def bbox_loss(self, pred, target, weight=None, avg_factor=None):
        """Calculate GIoU loss."""
        ious = bbox_overlaps(pred, target, mode="giou")
        loss = 1 - ious
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


class DIoULoss(nn.Cell):
    """DIoU Loss for MindSpore."""

    def __init__(self, loss_weight=1.0, eps=1e-6, reduction="mean", loss_name="loss_diou"):
        super(DIoULoss, self).__init__()
        self.loss_weight = loss_weight
        self.eps = eps
        self.reduction = reduction
        self.loss_name = loss_name

    def construct(self, pred, target, weight=None, avg_factor=None):
        """Forward function."""
        loss = self.loss_weight * self.bbox_loss(pred, target, weight, avg_factor)
        return loss

    def bbox_loss(self, pred, target, weight=None, avg_factor=None):
        """Calculate DIoU loss."""
        ious = bbox_overlaps(pred, target, mode="iou")
        iou = ious.clamp(min=self.eps)

        lt = ops.maximum(pred[..., :2], target[..., :2])
        rb = ops.minimum(pred[..., 2:], target[..., 2:])
        wh = (rb - lt).clamp(min=0)
        diag_len = wh[..., 0] ** 2 + wh[..., 1] ** 2

        c2 = (pred[..., 2] - pred[..., 0]) ** 2 + (pred[..., 3] - pred[..., 1]) ** 2
        c2 = c2.clamp(min=self.eps)

        loss = 1 - iou + diag_len / c2
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


__all__ = ["IoULoss", "GIoULoss", "DIoULoss", "bbox_overlaps"]
