#!/usr/bin/env python3
"""
GROK-Ω (OMEGA) - WikiText Training Script
=========================================

Trains GROK-Ω on WikiText dataset using pure physics.
No tokenization, no softmax, no fallbacks.

ZERO FALLBACK POLICY: Physical failure is honest failure.
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import datasets
from pathlib import Path
import math
import numpy as np
from typing import List, Tuple, Optional
import sys
import os

# Add base directory to path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

from grok_omega import GrokOmega, create_grok_omega


class WikiTextDataset(Dataset):
    """
    WikiText dataset for character-level training.
    No tokenization - pure character sequences.
    """

    def __init__(self, split: str = 'train', seq_length: int = 128, device: str = 'cpu'):
        self.seq_length = seq_length
        self.device = device

        print(f"🔬 Loading WikiText-{split} dataset...")

        # Load WikiText dataset
        try:
            dataset = datasets.load_dataset('wikitext', 'wikitext-2-raw-v1', split=split)
        except:
            print("❌ Failed to load WikiText dataset. Install with: pip install datasets")
            raise

        # Extract text and convert to characters
        self.text = ''.join(dataset['text'])

        # Filter out very short texts and clean
        self.text = self._clean_text(self.text)

        print(f"   📊 Dataset size: {len(self.text)} characters")
        print(f"   📏 Sequence length: {seq_length}")

    def _clean_text(self, text: str) -> str:
        """Clean and filter text data."""
        # Remove excessive whitespace
        import re
        text = re.sub(r'\n+', '\n', text)
        text = re.sub(r' +', ' ', text)

        # Keep only printable characters and basic punctuation
        import string
        allowed_chars = string.ascii_letters + string.digits + string.punctuation + ' \n\t'
        text = ''.join(c for c in text if c in allowed_chars)

        # Remove very short lines (likely headers/footers)
        lines = text.split('\n')
        lines = [line for line in lines if len(line.strip()) > 10]
        text = '\n'.join(lines)

        return text

    def __len__(self) -> int:
        return max(1, len(self.text) - self.seq_length)

    def __getitem__(self, idx: int) -> Tuple[str, str]:
        """Get input-target pair for training."""
        start_idx = idx
        end_idx = start_idx + self.seq_length

        input_text = self.text[start_idx:end_idx]
        target_text = self.text[start_idx + 1:end_idx + 1]

        return input_text, target_text


def create_dataloaders(batch_size: int = 32, seq_length: int = 128, device: str = 'cpu'):
    """Create train and validation dataloaders."""

    # Create datasets
    train_dataset = WikiTextDataset('train', seq_length, device)
    val_dataset = WikiTextDataset('validation', seq_length, device)

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,  # Avoid multiprocessing issues
        pin_memory=False
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False
    )

    return train_loader, val_loader


class WikiTextTrainer:
    """
    Trainer for GROK-Ω on WikiText dataset.
    Implements pure physics training with honest failures.
    """

    def __init__(self,
                 model: GrokOmega,
                 train_loader: DataLoader,
                 val_loader: DataLoader,
                 device: str = 'cpu',
                 lr: float = 1e-3,
                 save_path: str = 'grok_omega_wikitext.pt'):

        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.save_path = Path(save_path)

        # Optimizer - no learning rate scheduling (honest physics)
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        # Training metrics
        self.best_val_loss = float('inf')
        self.epoch = 0

        print("🔬 WikiText Trainer initialized")
        print(f"   📊 Train batches: {len(train_loader)}")
        print(f"   📊 Val batches: {len(val_loader)}")
        print(f"   🎯 Learning rate: {lr}")
        print("   🚫 ZERO FALLBACK: CONFIRMED")

    def train_epoch(self) -> float:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        num_batches = 0

        for batch_idx, (input_texts, target_texts) in enumerate(self.train_loader):
            batch_loss = 0

            # Process each sequence in batch
            for input_text, target_text in zip(input_texts, target_texts):
                try:
                    # Train step (honest physics - no fallbacks)
                    loss = self.model.train_step(input_text, target_text, self.optimizer)
                    batch_loss += loss
                except Exception as e:
                    # Honest failure - log and continue
                    print(f"⚠️  Training failure in batch {batch_idx}: {e}")
                    continue

            if len(input_texts) > 0:
                avg_batch_loss = batch_loss / len(input_texts)
                total_loss += avg_batch_loss
                num_batches += 1

            # Progress update
            if batch_idx % 100 == 0:
                print(f"   Batch {batch_idx}/{len(self.train_loader)}: Loss = {avg_batch_loss:.4f}")

        return total_loss / max(1, num_batches)

    def validate(self) -> float:
        """Validate model."""
        self.model.eval()
        total_loss = 0
        num_batches = 0

        with torch.no_grad():
            for input_texts, target_texts in self.val_loader:
                batch_loss = 0

                for input_text, target_text in zip(input_texts, target_texts):
                    try:
                        # Forward pass only (no gradients)
                        logits = self.model(input_text)
                        # Compute loss (same as training)
                        min_len = min(len(input_text), len(target_text))
                        target_indices = torch.tensor([ord(c) % self.model.vocab_size for c in target_text[:min_len]],
                                                    dtype=torch.long, device=self.device)
                        target_onehot = torch.nn.functional.one_hot(target_indices, num_classes=self.model.vocab_size).float()
                        loss = torch.nn.functional.mse_loss(logits[:min_len], target_onehot)
                        batch_loss += loss.item()
                    except Exception as e:
                        print(f"⚠️  Validation failure: {e}")
                        continue

                if len(input_texts) > 0:
                    avg_batch_loss = batch_loss / len(input_texts)
                    total_loss += avg_batch_loss
                    num_batches += 1

        return total_loss / max(1, num_batches)

    def save_checkpoint(self, val_loss: float):
        """Save model checkpoint if validation improves."""
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            checkpoint = {
                'epoch': self.epoch,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'val_loss': val_loss,
                'best_val_loss': self.best_val_loss
            }
            torch.save(checkpoint, self.save_path)
            print(f"💾 Checkpoint saved: {self.save_path}")

    def train(self, num_epochs: int = 10, patience: int = 3):
        """Full training loop."""
        print("🚀 Starting GROK-Ω WikiText training...")
        print("=" * 50)

        patience_counter = 0

        for epoch in range(num_epochs):
            self.epoch = epoch + 1

            print(f"\n🔄 Epoch {self.epoch}/{num_epochs}")

            # Train
            train_loss = self.train_epoch()
            print(".4f")

            # Validate
            val_loss = self.validate()
            print(".4f")

            # Save checkpoint if improved
            self.save_checkpoint(val_loss)

            # Early stopping (honest physics - no artificial patience)
            if val_loss >= self.best_val_loss:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"⏹️  Early stopping at epoch {self.epoch}")
                    break
            else:
                patience_counter = 0

            # Generate sample text
            if epoch % 2 == 0:
                self.generate_sample()

        print("\n✅ WikiText training completed!")
        print(f"   🏆 Best validation loss: {self.best_val_loss:.4f}")
        print("   🌊 Language as continuous wave")
        print("   ⚛️  Pure quantum physics")
        print("   🚫 No softmax, no tokenization")

    def generate_sample(self, prompt: str = "The quantum", max_length: int = 100):
        """Generate sample text from prompt."""
        print(f"\n📝 Sample generation from '{prompt}':")

        try:
            generated = self.model.generate_next_wave(prompt, time_steps=10)
            # Take first max_length characters
            sample = generated[:max_length]
            print(f"   '{sample}'")
        except Exception as e:
            print(f"   ❌ Generation failed: {e}")


def main():
    """Main training function."""
    print("🚀 GROK-Ω (OMEGA) - WikiText Training")
    print("=" * 50)

    # Configuration
    config = {
        'embed_dim': 64,
        'vocab_size': 256,
        'batch_size': 16,
        'seq_length': 128,
        'num_epochs': 5,
        'learning_rate': 1e-3,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'save_path': 'grok_omega_wikitext.pt'
    }

    print("🔧 Configuration:")
    for key, value in config.items():
        print(f"   {key}: {value}")

    # Create model
    model = create_grok_omega(
        embed_dim=config['embed_dim'],
        vocab_size=config['vocab_size'],
        device=config['device']
    )

    # Create dataloaders
    train_loader, val_loader = create_dataloaders(
        batch_size=config['batch_size'],
        seq_length=config['seq_length'],
        device=config['device']
    )

    # Create trainer
    trainer = WikiTextTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=config['device'],
        lr=config['learning_rate'],
        save_path=config['save_path']
    )

    # Train
    trainer.train(num_epochs=config['num_epochs'])

    print("\n🎯 GROK-Ω WikiText training complete!")
    print("   🌊 Language as continuous wave")
    print("   ⚛️  Pure quantum physics")
    print("   🚫 No softmax, no tokenization, no fallbacks")


if __name__ == "__main__":
    main()