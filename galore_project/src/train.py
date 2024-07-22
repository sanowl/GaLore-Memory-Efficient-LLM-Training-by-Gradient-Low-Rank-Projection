import argparse
import os
import time
from typing import Dict, Any

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler
from torch.cuda.amp import GradScaler, autocast
from transformers import GPT2Tokenizer, GPT2LMHeadModel, get_linear_schedule_with_warmup

from src.galore import GaLoreProjection, apply_galore
from src.dataset import LargeLanguageModelDataset
from src.utils import setup_logging, save_checkpoint, load_checkpoint

def parse_args():
    parser = argparse.ArgumentParser(description="Train a large language model with GaLore")
    parser.add_argument("--data_path", type=str, required=True, help="Path to the dataset")
    parser.add_argument("--model_name", type=str, default="gpt2", help="Name of the pretrained model to use")
    parser.add_argument("--output_dir", type=str, default="./output", help="Directory to save checkpoints and logs")
    parser.add_argument("--num_epochs", type=int, default=3, help="Number of epochs to train")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training")
    parser.add_argument("--learning_rate", type=float, default=5e-5, help="Learning rate")
    parser.add_argument("--warmup_steps", type=int, default=1000, help="Number of warmup steps for learning rate scheduler")
    parser.add_argument("--max_grad_norm", type=float, default=1.0, help="Maximum gradient norm for clipping")
    parser.add_argument("--galore_rank", type=int, default=4, help="Rank for GaLore projection")
    parser.add_argument("--galore_update_freq", type=int, default=200, help="Update frequency for GaLore projection")
    parser.add_argument("--fp16", action="store_true", help="Use mixed precision training")
    parser.add_argument("--local_rank", type=int, default=-1, help="Local rank for distributed training")
    return parser.parse_args()

def setup_distributed():
    dist.init_process_group(backend="nccl")
    torch.cuda.set_device(args.local_rank)

def get_model_and_tokenizer(args):
    tokenizer = GPT2Tokenizer.from_pretrained(args.model_name)
    model = GPT2LMHeadModel.from_pretrained(args.model_name)
    return model, tokenizer

def train_epoch(args, model, dataloader, optimizer, scheduler, scaler, galore):
    model.train()
    total_loss = 0
    for step, batch in enumerate(dataloader):
        inputs, labels = batch
        inputs = inputs.to(args.device)
        labels = labels.to(args.device)

        optimizer.zero_grad()

        with autocast(enabled=args.fp16):
            outputs = model(inputs, labels=labels)
            loss = outputs.loss

        if args.fp16:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
        else:
            loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

        if args.fp16:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()

        scheduler.step()
        total_loss += loss.item()

        if step % 100 == 0 and args.local_rank in (0, -1):
            print(f"Step {step}/{len(dataloader)}, Loss: {loss.item():.4f}")

    return total_loss / len(dataloader)

def evaluate(args, model, dataloader):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch in dataloader:
            inputs, labels = batch
            inputs = inputs.to(args.device)
            labels = labels.to(args.device)

            with autocast(enabled=args.fp16):
                outputs = model(inputs, labels=labels)
                loss = outputs.loss

            total_loss += loss.item()

    return total_loss / len(dataloader)

def main(args):
    setup_logging(args.output_dir)
    if args.local_rank != -1:
        setup_distributed()

    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model, tokenizer = get_model_and_tokenizer(args)
    model = model.to(args.device)

    if args.local_rank != -1:
        model = DistributedDataParallel(model, device_ids=[args.local_rank], output_device=args.local_rank)

    train_dataset = LargeLanguageModelDataset(args.data_path, tokenizer, split="train")
    val_dataset = LargeLanguageModelDataset(args.data_path, tokenizer, split="val")

    train_sampler = DistributedSampler(train_dataset) if args.local_rank != -1 else None
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=train_sampler, num_workers=4)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, num_workers=4)

    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=len(train_dataloader) * args.num_epochs
    )

    galore = GaLoreProjection(rank=args.galore_rank, update_freq=args.galore_update_freq)
    apply_galore(model, optimizer, galore)

    scaler = GradScaler(enabled=args.fp16)

    best_val_loss = float('inf')
    for epoch in range(args.num_epochs):
        if args.local_rank != -1:
            train_sampler.set_epoch(epoch)

        train_loss = train_epoch(args, model, train_dataloader, optimizer, scheduler, scaler, galore)
        val_loss = evaluate(args, model, val_dataloader)

        if args.local_rank in (0, -1):
            print(f"Epoch {epoch+1}/{args.num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
            
        if val_loss < best_val_loss:
           best_val_loss = val_loss
    save_checkpoint({
        'epoch': epoch + 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'best_val_loss': best_val_loss,
    }, is_best=True, filename=os.path.join(args.output_dir, f'checkpoint_epoch_{epoch+1}.pth'))

    if args.local_rank in (0, -1):
        print("Training completed!")

if __name__ == "__main__":
    args = parse_args()
    main(args)
