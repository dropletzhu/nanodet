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

"""Weight conversion utilities for PyTorch to MindSpore."""

import argparse
import os

import torch


def convert_weights(torch_ckpt_path, ms_ckpt_path, verbose=True):
    """Convert PyTorch checkpoint to MindSpore format.

    Args:
        torch_ckpt_path: Path to PyTorch checkpoint (.pth)
        ms_ckpt_path: Path to save MindSpore checkpoint (.ckpt)
        verbose: Whether to print conversion details
    """
    try:
        import mindspore
        from mindspore import Tensor, save_checkpoint
    except ImportError:
        print("MindSpore not installed. Only generating weight mapping.")
        mindspore = None

    if verbose:
        print(f"Loading PyTorch checkpoint from: {torch_ckpt_path}")

    ckpt = torch.load(torch_ckpt_path, map_location="cpu")

    if "state_dict" in ckpt:
        state_dict = ckpt["state_dict"]
    elif "model" in ckpt:
        state_dict = ckpt["model"]
    else:
        state_dict = ckpt

    if verbose:
        print(f"Converted {len(state_dict)} parameters")

    if mindspore is None:
        return

    ms_params = []
    for k, v in state_dict.items():
        ms_name = k

        if isinstance(v, torch.Tensor):
            v_np = v.cpu().detach().numpy()
            ms_tensor = Tensor(v_np)
            ms_params.append({"name": ms_name, "data": ms_tensor})

    save_checkpoint(ms_params, ms_ckpt_path)
    if verbose:
        print(f"Saved MindSpore checkpoint to: {ms_ckpt_path}")


def print_weight_mapping(torch_ckpt_path):
    """Print weight mapping between PyTorch and MindSpore."""
    ckpt = torch.load(torch_ckpt_path, map_location="cpu")

    if "state_dict" in ckpt:
        state_dict = ckpt["state_dict"]
    elif "model" in ckpt:
        state_dict = ckpt["model"]
    else:
        state_dict = ckpt

    print("\n=== PyTorch to MindSpore Weight Mapping ===")
    print(f"Total parameters: {len(state_dict)}\n")

    for i, (k, v) in enumerate(state_dict.items()):
        if i < 20:
            print(f"  {k}: {v.shape if hasattr(v, 'shape') else type(v)}")
        elif i == 20:
            print(f"  ... and {len(state_dict) - 20} more")


def main():
    """Main function for weight conversion."""
    parser = argparse.ArgumentParser(description="Convert PyTorch weights to MindSpore")
    parser.add_argument("torch_ckpt", help="Path to PyTorch checkpoint")
    parser.add_argument("--ms_ckpt", default=None, help="Path to save MindSpore checkpoint")
    parser.add_argument("--print_only", action="store_true", help="Only print mapping, don't convert")
    args = parser.parse_args()

    if args.print_only:
        print_weight_mapping(args.torch_ckpt)
        return

    ms_ckpt = args.ms_ckpt
    if ms_ckpt is None:
        base, _ = os.path.splitext(args.torch_ckpt)
        ms_ckpt = base + ".ckpt"

    convert_weights(args.torch_ckpt, ms_ckpt)


if __name__ == "__main__":
    main()
