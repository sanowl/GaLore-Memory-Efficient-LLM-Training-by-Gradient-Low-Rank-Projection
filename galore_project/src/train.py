import argparse
import logging
import os
from typing import Tuple, Dict, Any

import torch
import torch.optim as optim
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import GPT2Tokenizer, GPT2LMHeadModel, get_linear_schedule_with_warmup
from tqdm import tqdm
from torch.cuda.amp import GradScaler, autocast

CONFIG = {
    "dataset": "wikitext",
    "dataset_config": "wikitext-2-raw-v1",
    "model_name": "gpt2",
    "output_dir": "./output",
    "num_epochs": 3,
    "batch_size": 32,
    "learning_rate": 5e-5,
    "warmup_steps": 1000,
    "max_grad_norm": 1.0,
    "early_stopping_patience": 3,
    "eval_steps": 500,
    "gradient_accumulation_steps": 2,
    "fp16": True,
    "logging_steps": 100,
}

def setup_logging(output_dir: str) -> None:
    os.makedirs(output_dir, exist_ok=True)
    logging.basicConfig(
        filename=os.path.join(output_dir, 'training.log'),
        filemode='a',
        format='%(asctime)s - %(levelname)s - %(message)s',
        level=logging.INFO
    )
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    logging.getLogger('').addHandler(console)
    logging.info("Logging setup complete.")

def save_checkpoint(state: Dict[str, Any], is_best: bool, filename: str) -> None:
    torch.save(state, filename)
    if is_best:
        best_path = os.path.join(os.path.dirname(filename), 'best_checkpoint.pth')
        torch.save(state, best_path)
    logging.info(f"Checkpoint saved: {filename}")

def load_checkpoint(filename: str, model: torch.nn.Module, optimizer: optim.Optimizer = None,
                    scheduler: Any = None) -> Tuple[torch.nn.Module, optim.Optimizer, Any, int, float]:
    if os.path.isfile(filename):
        logging.info(f"Loading checkpoint: {filename}")
        checkpoint = torch.load(filename, map_location=torch.device('cpu'))
        model.load_state_dict(checkpoint['model_state_dict'])
        if optimizer and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if scheduler and 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        epoch = checkpoint['epoch']
        best_val_loss = checkpoint['best_val_loss']
        return model, optimizer, scheduler, epoch, best_val_loss
    else:
        raise FileNotFoundError(f"No checkpoint found at '{filename}'")

def get_model_and_tokenizer() -> Tuple[GPT2LMHeadModel, GPT2Tokenizer]:
    tokenizer = GPT2Tokenizer.from_pretrained(CONFIG['model_name'])
    model = GPT2LMHeadModel.from_pretrained(CONFIG['model_name'])
    return model, tokenizer

def prepare_data(tokenizer: GPT2Tokenizer) -> Tuple[Any, Any]:
    try:
        dataset = load_dataset(CONFIG['dataset'], CONFIG['dataset_config'])
    except Exception as e:
        logging.error(f"Failed to load dataset: {e}")
        raise

    def tokenize_function(examples):
        return tokenizer(examples["text"], truncation=True, max_length=512)

    tokenized_datasets = dataset.map(tokenize_function, batched=True, num_proc=4, remove_columns=["text"])

    train_dataset = tokenized_datasets["train"]
    val_dataset = tokenized_datasets["validation"]

    return train_dataset, val_dataset

def train_step(model: GPT2LMHeadModel, batch: Dict[str, torch.Tensor], optimizer: optim.Optimizer,
               scheduler: Any, scaler: GradScaler) -> float:
    model.train()
    inputs = batch["input_ids"].to(CONFIG['device'])
    labels = inputs.clone()

    with autocast(enabled=CONFIG['fp16']):
        outputs = model(inputs, labels=labels)
        loss = outputs.loss / CONFIG['gradient_accumulation_steps']

    scaler.scale(loss).backward()
    return loss.item()

def evaluate(model: GPT2LMHeadModel, dataloader: DataLoader) -> float:
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            inputs = batch["input_ids"].to(CONFIG['device'])
            labels = inputs.clone()

            outputs = model(inputs, labels=labels)
            loss = outputs.loss

            total_loss += loss.item()

    return total_loss / len(dataloader)

def train(model: GPT2LMHeadModel, train_dataloader: DataLoader, val_dataloader: DataLoader,
          optimizer: optim.Optimizer, scheduler: Any) -> None:
    scaler = GradScaler(enabled=CONFIG['fp16'])
    best_val_loss = float('inf')
    early_stopping_counter = 0
    global_step = 0

    for epoch in range(CONFIG['num_epochs']):
        model.train()
        epoch_loss = 0
        for step, batch in enumerate(tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{CONFIG['num_epochs']}")):
            loss = train_step(model, batch, optimizer, scheduler, scaler)
            epoch_loss += loss

            if (step + 1) % CONFIG['gradient_accumulation_steps'] == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), CONFIG['max_grad_norm'])
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                if global_step % CONFIG['logging_steps'] == 0:
                    logging.info(f"Step {global_step}: loss = {loss:.4f}")

                if global_step % CONFIG['eval_steps'] == 0:
                    val_loss = evaluate(model, val_dataloader)
                    logging.info(f"Step {global_step}: val_loss = {val_loss:.4f}")

                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        early_stopping_counter = 0
                        save_checkpoint({
                            'epoch': epoch + 1,
                            'global_step': global_step,
                            'model_state_dict': model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'scheduler_state_dict': scheduler.state_dict(),
                            'best_val_loss': best_val_loss,
                        }, is_best=True, filename=os.path.join(CONFIG['output_dir'], f'checkpoint_step_{global_step}.pth'))
                    else:
                        early_stopping_counter += 1

                    if early_stopping_counter >= CONFIG['early_stopping_patience']:
                        logging.info(f"Early stopping triggered after {global_step} steps")
                        return

        epoch_loss /= len(train_dataloader)
        logging.info(f"Epoch {epoch+1}/{CONFIG['num_epochs']}, Train Loss: {epoch_loss:.4f}")

    logging.info("Training completed!")

def main() -> None:
    setup_logging(CONFIG['output_dir'])

    if torch.backends.mps.is_available():
        CONFIG['device'] = torch.device("mps")
    elif torch.cuda.is_available():
        CONFIG['device'] = torch.device("cuda")
    else:
        CONFIG['device'] = torch.device("cpu")

    logging.info(f"Using device: {CONFIG['device']}")

    model, tokenizer = get_model_and_tokenizer()
    model = model.to(CONFIG['device'])

    train_dataset, val_dataset = prepare_data(tokenizer)

    train_dataloader = DataLoader(train_dataset, batch_size=CONFIG['batch_size'], shuffle=True, num_workers=4)
    val_dataloader = DataLoader(val_dataset, batch_size=CONFIG['batch_size'], num_workers=4)

    optimizer = optim.AdamW(model.parameters(), lr=CONFIG['learning_rate'])
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=CONFIG['warmup_steps'],
        num_training_steps=len(train_dataloader) * CONFIG['num_epochs']
    )

    train(model, train_dataloader, val_dataloader, optimizer, scheduler)

if __name__ == "__main__":
    main()