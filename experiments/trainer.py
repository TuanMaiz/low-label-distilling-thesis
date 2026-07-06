"""Training loop with optional W&B logging for seq2seq ER students."""

from typing import Optional, Dict, List, Callable
from pathlib import Path
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import get_scheduler
from tqdm import tqdm

try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False
    print("Warning: wandb not installed. Install with: pip install wandb")


class Trainer:
    """
    Trainer for seq2seq models with W&B logging.
    """

    def __init__(
        self,
        model,
        tokenizer,
        device: str = "cuda",
        learning_rate: float = 5e-5,
        weight_decay: float = 0.01,
        warmup_steps: int = 100,
        label_smoothing: float = 0.1,
        wandb_project: Optional[str] = "distiller-wdc-er",
        wandb_entity: Optional[str] = None,
        wandb_run_name: Optional[str] = None,
    ):
        """
        Initialize the trainer.

        Args:
            model: The seq2seq model (HuggingFace)
            tokenizer: The tokenizer
            device: Device to use
            learning_rate: Learning rate
            weight_decay: Weight decay for AdamW
            warmup_steps: Number of warmup steps
            label_smoothing: Label smoothing factor
            wandb_project: W&B project name
            wandb_entity: W&B entity/username
            wandb_run_name: W&B run name
        """
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.warmup_steps = warmup_steps
        self.label_smoothing = label_smoothing

        # Move model to device
        self.model.to(self.device)

        # Setup optimizer
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )

        self.wandb_project = wandb_project
        self.wandb_entity = wandb_entity
        self.wandb_run_name = wandb_run_name
        self.use_wandb = HAS_WANDB and bool(wandb_project)
        self.scheduler = None

        self.global_step = 0
        self.best_val_loss = float("inf")

    def setup_wandb(self, config: Dict) -> None:
        """Initialize W&B logging."""
        if not self.use_wandb:
            print("W&B not available. Skipping logging.")
            return

        wandb.init(
            project=self.wandb_project,
            entity=self.wandb_entity,
            name=self.wandb_run_name,
            config=config
        )
        wandb.watch(self.model, log="all")

    def train_epoch(
        self,
        train_loader: DataLoader,
        epoch: int,
        log_every: int = 10
    ) -> float:
        """
        Train for one epoch.

        Args:
            train_loader: Training data loader
            epoch: Current epoch number
            log_every: Log every N steps

        Returns:
            Average loss for the epoch
        """
        self.model.train()
        total_loss = 0
        num_batches = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}")

        for batch_idx, batch in enumerate(pbar):
            # Move batch to device
            batch = {k: v.to(self.device) for k, v in batch.items()}

            # Forward pass
            outputs = self.model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=batch["labels"]
            )

            loss = outputs.loss
            total_loss += loss.item()
            num_batches += 1

            # Backward pass
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            # Optimizer step
            self.optimizer.step()
            if self.scheduler is not None:
                self.scheduler.step()
            self.optimizer.zero_grad()

            self.global_step += 1

            # Update progress bar
            pbar.set_postfix({"loss": loss.item()})

            # Log to W&B
            if self.use_wandb and self.global_step % log_every == 0:
                wandb.log({
                    "train/loss": loss.item(),
                    "train/epoch": epoch,
                    "train/global_step": self.global_step
                })

        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0

        if self.use_wandb:
            wandb.log({"train/epoch_loss": avg_loss, "epoch": epoch})

        return avg_loss

    @torch.no_grad()
    def evaluate(self, val_loader: DataLoader, epoch: int = 0) -> float:
        """
        Evaluate on validation set.

        Args:
            val_loader: Validation data loader
            epoch: Current epoch number

        Returns:
            Average validation loss
        """
        self.model.eval()
        total_loss = 0
        num_batches = 0

        for batch in tqdm(val_loader, desc="Validation"):
            batch = {k: v.to(self.device) for k, v in batch.items()}

            outputs = self.model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=batch["labels"]
            )

            loss = outputs.loss
            total_loss += loss.item()
            num_batches += 1

        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0

        if self.use_wandb:
            wandb.log({"val/loss": avg_loss, "epoch": epoch})

        return avg_loss

    def train(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        num_epochs: int = 10,
        save_dir: str = "checkpoints",
        early_stopping_patience: int = 3,
    ) -> Dict[str, List[float]]:
        """
        Full training loop.

        Args:
            train_loader: Training data loader
            val_loader: Optional validation data loader
            num_epochs: Number of epochs to train
            save_dir: Directory to save checkpoints
            early_stopping_patience: Patience for early stopping

        Returns:
            Dictionary with training history
        """
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        history = {
            "train_loss": [],
            "val_loss": []
        }

        patience_counter = 0

        for epoch in range(1, num_epochs + 1):
            print(f"\n{'=' * 50}")
            print(f"Epoch {epoch}/{num_epochs}")
            print(f"{'=' * 50}")

            # Train
            train_loss = self.train_epoch(train_loader, epoch)
            history["train_loss"].append(train_loss)

            print(f"Train Loss: {train_loss:.4f}")

            # Validate
            if val_loader is not None:
                val_loss = self.evaluate(val_loader, epoch)
                history["val_loss"].append(val_loss)
                print(f"Val Loss: {val_loss:.4f}")

                # Check for improvement
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    patience_counter = 0

                    # Save best model
                    checkpoint_path = save_dir / "best_model"
                    self.model.save_pretrained(checkpoint_path)
                    self.tokenizer.save_pretrained(checkpoint_path)
                    print(f"Saved best model to {checkpoint_path}")
                else:
                    patience_counter += 1

                # Early stopping
                if patience_counter >= early_stopping_patience:
                    print(f"Early stopping triggered after {epoch} epochs")
                    break
            else:
                # Save checkpoint every epoch if no validation
                checkpoint_path = save_dir / f"checkpoint_epoch_{epoch}"
                self.model.save_pretrained(checkpoint_path)
                self.tokenizer.save_pretrained(checkpoint_path)

        if self.use_wandb:
            wandb.finish()

        return history

    def save_checkpoint(self, path: str) -> None:
        """Save model checkpoint."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)

        # Save training state
        state = {
            "optimizer": self.optimizer.state_dict(),
            "global_step": self.global_step,
            "best_val_loss": self.best_val_loss
        }
        torch.save(state, path / "trainer_state.pt")


def create_training_config(
    model_name: str,
    learning_rate: float,
    batch_size: int,
    num_epochs: int,
    warmup_steps: int,
    **kwargs
) -> Dict:
    """Create a configuration dict for W&B logging."""
    config = {
        "model_name": model_name,
        "learning_rate": learning_rate,
        "batch_size": batch_size,
        "num_epochs": num_epochs,
        "warmup_steps": warmup_steps,
        **kwargs
    }
    return config


if __name__ == "__main__":
    print("Trainer module loaded successfully")
    print(f"W&B available: {HAS_WANDB}")
