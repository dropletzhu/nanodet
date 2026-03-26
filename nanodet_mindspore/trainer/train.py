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

"""MindSpore training script for NanoDet on Ascend NPU."""

import argparse
import os
from pathlib import Path

import mindspore
import mindspore.common as common
import mindspore.nn as nn
from mindspore import context, save_checkpoint
from mindspore.communication import init, get_rank, get_group_size
from mindspore.context import ParallelMode

from nanodet_mindspore.model.arch import build_model
from nanodet_mindspore.util import load_config, mkdir


def get_device_id():
    """Get device id from environment."""
    device_id = os.getenv("DEVICE_ID", "0")
    return int(device_id)


def get_device_num():
    """Get device num from environment."""
    device_num = os.getenv("RANK_SIZE", "1")
    return int(device_num)


def set_context_for_training(device_id, device_num):
    """Set context for Ascend NPU training."""
    context.set_context(
        mode=context.GRAPH_MODE,
        device_target="Ascend",
        device_id=device_id,
    )

    if device_num > 1:
        init()
        context.set_context(
            parallel_mode=ParallelMode.AUTO_PARALLEL,
            gradients_mean=True,
        )


def build_optimizer(model, cfg):
    """Build optimizer."""
    optimizer_cfg = cfg.get("schedule", {}).get("optimizer", {})
    name = optimizer_cfg.pop("name", "Adam")
    lr = optimizer_cfg.pop("lr", 0.001)
    weight_decay = optimizer_cfg.pop("weight_decay", 0.0001)

    if name == "Adam":
        optimizer = nn.Adam(
            params=model.trainable_params(),
            learning_rate=lr,
            weight_decay=weight_decay,
        )
    elif name == "SGD":
        momentum = optimizer_cfg.pop("momentum", 0.9)
        optimizer = nn.SGD(
            params=model.trainable_params(),
            learning_rate=lr,
            momentum=momentum,
            weight_decay=weight_decay,
        )
    else:
        raise NotImplementedError(f"Optimizer {name} not supported")

    return optimizer


def train_one_step(model, optimizer, batch):
    """Train one step."""
    def forward_fn(batch):
        loss = model(batch)
        return loss

    grad_fn = mindspore.value_and_grad(forward_fn, None, optimizer.parameters)
    loss, grads = grad_fn(batch)
    optimizer(grads)
    return loss


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train NanoDet with MindSpore")
    parser.add_argument("config", help="train config file path")
    parser.add_argument("--device_id", type=int, default=0, help="device id")
    parser.add_argument("--device_num", type=int, default=1, help="device num")
    parser.add_argument("--save_dir", type=str, default="output", help="save directory")
    parser.add_argument("--epochs", type=int, default=None, help="total training epochs")
    parser.add_argument("--amp", action="store_true", help="enable mixed precision training")
    return parser.parse_args()


def main():
    """Main training function."""
    args = parse_args()

    device_id = args.device_id
    device_num = args.device_num

    set_context_for_training(device_id, device_num)

    cfg = load_config(args.config)

    if args.epochs is not None:
        cfg.schedule.total_epochs = args.epochs

    save_dir = args.save_dir
    mkdir(save_dir)

    model = build_model(cfg.model)

    optimizer = build_optimizer(model, cfg)

    print(f"Start training on Ascend NPU {device_id}")
    print(f"Device num: {device_num}")
    print(f"Total epochs: {cfg.schedule.total_epochs}")
    print(f"Save directory: {save_dir}")

    for epoch in range(cfg.schedule.total_epochs):
        model.set_train(True)
        
        for step in range(100):
            batch = mindspore.Tensor(mindspore.numpy.randn(1, 3, 320, 320))
            
            loss = train_one_step(model, optimizer, batch)
            
            if step % 10 == 0:
                print(f"Epoch {epoch}, Step {step}, Loss: {loss.asnumpy():.4f}")

        ckpt_path = os.path.join(save_dir, f"nanodet_epoch{epoch}.ckpt")
        save_checkpoint(model, ckpt_path)
        print(f"Saved checkpoint: {ckpt_path}")

    print("Training completed!")


if __name__ == "__main__":
    main()
