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

import argparse
import os
import warnings
import datetime

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from nanodet.data.batch_process import stack_batch_img
from nanodet.data.collate import naive_collate
from nanodet.data.dataset import build_dataset
from nanodet.model.arch import build_model
from nanodet.util import cfg, load_config, mkdir


def get_device():
    """Get available device: npu, cuda, or cpu."""
    if hasattr(torch, "npu") and torch.npu.is_available():
        return torch.device("npu")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def setup_distributed():
    """Setup distributed training."""
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
    else:
        rank = 0
        world_size = 1
        local_rank = 0

    if world_size > 1:
        dist.init_process_group(backend="nccl" if torch.cuda.is_available() else "gloo")
    
    return rank, world_size, local_rank


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("config", help="train config file path")
    parser.add_argument("--local_rank", default=-1, type=int, help="node rank for distributed training")
    parser.add_argument("--seed", type=int, default=None, help="random seed")
    parser.add_argument("--device", type=str, default="npu", choices=["cpu", "npu", "cuda"], help="device to use")
    parser.add_argument("--epochs", type=int, default=None, help="number of training epochs")
    args = parser.parse_args()
    return args


def main(args):
    load_config(cfg, args.config)
    cfg.defrost()
    
    # Override epochs if specified
    if args.epochs is not None:
        cfg.schedule.total_epochs = args.epochs
    
    rank, world_size, local_rank = setup_distributed()
    device = get_device()
    
    if rank == 0:
        print(f"Using device: {device}")
        mkdir(rank, cfg.save_dir)
    
    if args.seed is not None:
        torch.manual_seed(args.seed)
    
    # Build dataset
    train_dataset = build_dataset(cfg.data.train, "train")
    val_dataset = build_dataset(cfg.data.val, "test")
    
    # Build dataloader
    train_sampler = DistributedSampler(train_dataset, shuffle=True) if world_size > 1 else None
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=cfg.device.batchsize_per_gpu,
        shuffle=(train_sampler is None),
        num_workers=cfg.device.workers_per_gpu,
        pin_memory=True,
        collate_fn=naive_collate,
        drop_last=True,
        sampler=train_sampler,
    )
    
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=cfg.device.batchsize_per_gpu,
        shuffle=False,
        num_workers=cfg.device.workers_per_gpu,
        pin_memory=True,
        collate_fn=naive_collate,
        drop_last=False,
    )
    
    # Build model
    model = build_model(cfg.model)
    model = model.to(device)
    
    if world_size > 1:
        model = DDP(model, device_ids=[local_rank])
    
    # Build optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.schedule.optimizer.lr,
        weight_decay=cfg.schedule.optimizer.weight_decay,
    )
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, 
        T_max=cfg.schedule.lr_schedule.T_max,
        eta_min=cfg.schedule.lr_schedule.eta_min,
    )
    
    # Training loop
    for epoch in range(cfg.schedule.total_epochs):
        model.train()
        
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        
        epoch_loss = 0.0
        
        for batch_idx, batch in enumerate(train_dataloader):
            # Move data to device
            if isinstance(batch, dict):
                if isinstance(batch.get("img"), list):
                    batch["img"] = [img.to(device) for img in batch["img"]]
                    batch["img"] = stack_batch_img(batch["img"], divisible=32)
                else:
                    batch["img"] = batch["img"].to(device)
                for k in batch:
                    if k != "img" and isinstance(batch[k], torch.Tensor):
                        batch[k] = batch[k].to(device)
            elif isinstance(batch, (list, tuple)):
                batch = [b.to(device) if isinstance(b, torch.Tensor) else b for b in batch]
            
            # Forward
            optimizer.zero_grad()
            preds, loss, loss_states = model.forward_train(batch)
            
            # Backward
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
            if rank == 0 and batch_idx % cfg.log.interval == 0:
                lr = optimizer.param_groups[0]["lr"]
                print(f"Epoch [{epoch+1}/{cfg.schedule.total_epochs}] "
                      f"Iter [{batch_idx+1}/{len(train_dataloader)}] "
                      f"Loss: {loss.item():.4f} LR: {lr:.6f}")
        
        scheduler.step()
        
        # Validation
        if (epoch + 1) % cfg.schedule.val_intervals == 0:
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for batch in val_dataloader:
                    if isinstance(batch, dict):
                        if isinstance(batch.get("img"), list):
                            batch["img"] = [img.to(device) for img in batch["img"]]
                            batch["img"] = stack_batch_img(batch["img"], divisible=32)
                        else:
                            batch["img"] = batch["img"].to(device)
                        for k in batch:
                            if k != "img" and isinstance(batch[k], torch.Tensor):
                                batch[k] = batch[k].to(device)
                    elif isinstance(batch, (list, tuple)):
                        batch = [b.to(device) if isinstance(b, torch.Tensor) else b for b in batch]
                    
                    preds, loss, loss_states = model.forward_train(batch)
                    val_loss += loss.item()
            
            if rank == 0:
                avg_val_loss = val_loss / len(val_dataloader)
                print(f"Validation Epoch [{epoch+1}] Loss: {avg_val_loss:.4f}")
                
                # Save checkpoint
                ckpt_path = os.path.join(cfg.save_dir, f"epoch_{epoch+1}.pth")
                if world_size > 1:
                    torch.save(model.module.state_dict(), ckpt_path)
                else:
                    torch.save(model.state_dict(), ckpt_path)
                print(f"Saved checkpoint to {ckpt_path}")
    
    if rank == 0:
        # Save final model
        final_path = os.path.join(cfg.save_dir, "nanodet_model_final.pth")
        if world_size > 1:
            torch.save(model.module.state_dict(), final_path)
        else:
            torch.save(model.state_dict(), final_path)
        print(f"Training complete! Final model saved to {final_path}")
    
    if world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    args = parse_args()
    main(args)
