"""Training loop with optional W&B logging for Hugging Face ER students."""

import math
from typing import Callable, Optional, Dict, List
from pathlib import Path
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm

from utils.torch_runtime import autocast_context, create_grad_scaler, resolve_precision
from utils.classification_threshold import (
    THRESHOLD_FILENAME,
    select_decision_threshold,
    write_decision_threshold,
)

try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False
    print("Warning: wandb not installed. Install with: pip install wandb")


class Trainer:
    """
    Trainer for Hugging Face models that return a scalar supervised loss.
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
        precision: str = "auto",
        optimizer_param_groups: Optional[list[dict]] = None,
        checkpoint_metric: str = "val_loss",
        unfreeze_after_epoch: int | None = None,
        unfreeze_callback: Callable[[], int] | None = None,
        positive_label_id: int = 1,
    ):
        """
        Initialize the trainer.

        Args:
            model: A Hugging Face model accepting input IDs, masks, and labels
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
        self.precision_requested = precision
        self.precision = resolve_precision(device, precision)
        if checkpoint_metric not in {"val_loss", "macro_f1"}:
            raise ValueError("checkpoint_metric must be 'val_loss' or 'macro_f1'")
        self.checkpoint_metric = checkpoint_metric
        self.unfreeze_after_epoch = unfreeze_after_epoch
        self.unfreeze_callback = unfreeze_callback
        self.unfrozen_encoder_layers = 0
        self.positive_label_id = positive_label_id

        # Move model to device
        self.model.to(self.device)

        # Setup optimizer
        self.optimizer = AdamW(
            optimizer_param_groups or self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        self.grad_scaler = create_grad_scaler(self.device, self.precision)

        self.wandb_project = wandb_project
        self.wandb_entity = wandb_entity
        self.wandb_run_name = wandb_run_name
        self.use_wandb = HAS_WANDB and bool(wandb_project)
        self.scheduler = None

        self.global_step = 0
        self.best_val_loss = float("inf")
        self.best_macro_f1 = float("-inf")
        self.best_same_f1 = float("-inf")
        self.best_decision_threshold: float | None = None

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
            with autocast_context(self.device, self.precision):
                outputs = self.model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    labels=batch["labels"]
                )

            loss = outputs.loss
            total_loss += loss.item()
            num_batches += 1

            # Backward pass
            if self.grad_scaler is not None:
                self.grad_scaler.scale(loss).backward()
                self.grad_scaler.unscale_(self.optimizer)
            else:
                loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            # Optimizer step
            optimizer_stepped = True
            if self.grad_scaler is not None:
                scale_before_step = self.grad_scaler.get_scale()
                self.grad_scaler.step(self.optimizer)
                self.grad_scaler.update()
                optimizer_stepped = self.grad_scaler.get_scale() >= scale_before_step
            else:
                self.optimizer.step()
            if self.scheduler is not None and optimizer_stepped:
                self.scheduler.step()
            self.optimizer.zero_grad(set_to_none=True)

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
    def evaluate(
        self,
        val_loader: DataLoader,
        epoch: int = 0,
        collect_classification_metrics: bool = False,
    ) -> float | dict:
        """
        Evaluate on validation set.

        Args:
            val_loader: Validation data loader
            epoch: Current epoch number

        Returns:
            Average validation loss
        """
        self.model.eval()
        total_weighted_loss = 0.0
        total_label_tokens = 0
        match_probabilities: list[float] = []
        classification_labels: list[bool] = []

        for batch in tqdm(val_loader, desc="Validation"):
            batch = {k: v.to(self.device) for k, v in batch.items()}

            with autocast_context(self.device, self.precision):
                outputs = self.model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    labels=batch["labels"]
                )

            loss_value = outputs.loss.item()
            if not math.isfinite(loss_value):
                raise RuntimeError(
                    f"Non-finite validation loss under precision={self.precision}; "
                    "retry with PRECISION=fp32 or inspect the checkpoint inputs"
                )
            label_tokens = int((batch["labels"] != -100).sum().item())
            total_weighted_loss += loss_value * label_tokens
            total_label_tokens += label_tokens
            if collect_classification_metrics:
                if outputs.logits.ndim != 2 or outputs.logits.shape[-1] != 2:
                    raise ValueError("macro-F1 checkpointing requires binary classifier logits")
                probabilities = torch.softmax(outputs.logits.float(), dim=-1)[
                    :, self.positive_label_id
                ]
                match_probabilities.extend(probabilities.cpu().tolist())
                classification_labels.extend(
                    (batch["labels"] == self.positive_label_id).cpu().tolist()
                )

        avg_loss = (
            total_weighted_loss / total_label_tokens
            if total_label_tokens > 0
            else 0.0
        )

        result: float | dict = avg_loss
        if collect_classification_metrics:
            threshold_result = select_decision_threshold(
                match_probabilities,
                classification_labels,
            )
            result = {"val_loss": avg_loss, **threshold_result}

        if self.use_wandb:
            log_payload = {"val/loss": avg_loss, "epoch": epoch}
            if isinstance(result, dict):
                validation_metrics = result["validation_metrics"]
                log_payload.update(
                    {
                        "val/macro_f1": validation_metrics["macro_f1"],
                        "val/same_f1": validation_metrics["same_f1"],
                        "val/decision_threshold": result["decision_threshold"],
                    }
                )
            wandb.log(log_payload)

        return result

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
            "val_loss": [],
            "val_macro_f1": [],
            "val_same_f1": [],
            "val_decision_threshold": [],
        }

        patience_counter = 0

        for epoch in range(1, num_epochs + 1):
            print(f"\n{'=' * 50}")
            print(f"Epoch {epoch}/{num_epochs}")
            print(f"{'=' * 50}")

            if (
                self.unfreeze_after_epoch is not None
                and epoch == self.unfreeze_after_epoch + 1
            ):
                if self.unfreeze_callback is None:
                    raise ValueError("unfreeze_after_epoch requires an unfreeze_callback")
                self.unfrozen_encoder_layers = self.unfreeze_callback()
                print(f"Unfroze {self.unfrozen_encoder_layers} final encoder layers")

            # Train
            train_loss = self.train_epoch(train_loader, epoch)
            history["train_loss"].append(train_loss)

            print(f"Train Loss: {train_loss:.4f}")

            # Validate
            if val_loader is not None:
                evaluation = self.evaluate(
                    val_loader,
                    epoch,
                    collect_classification_metrics=self.checkpoint_metric == "macro_f1",
                )
                if isinstance(evaluation, dict):
                    val_loss = evaluation["val_loss"]
                    validation_metrics = evaluation["validation_metrics"]
                    macro_f1 = validation_metrics["macro_f1"]
                    same_f1 = validation_metrics["same_f1"]
                    decision_threshold = evaluation["decision_threshold"]
                    history["val_macro_f1"].append(macro_f1)
                    history["val_same_f1"].append(same_f1)
                    history["val_decision_threshold"].append(decision_threshold)
                    print(
                        f"Val Macro F1: {macro_f1:.4f}; Match F1: {same_f1:.4f}; "
                        f"Threshold: {decision_threshold:.6f}"
                    )
                else:
                    val_loss = evaluation
                    macro_f1 = None
                    same_f1 = None
                    decision_threshold = None
                history["val_loss"].append(val_loss)
                print(f"Val Loss: {val_loss:.4f}")

                # Check for improvement
                if self.checkpoint_metric == "macro_f1":
                    improved = (macro_f1, same_f1) > (
                        self.best_macro_f1,
                        self.best_same_f1,
                    )
                else:
                    improved = val_loss < self.best_val_loss
                if improved:
                    self.best_val_loss = val_loss
                    if macro_f1 is not None and same_f1 is not None:
                        self.best_macro_f1 = macro_f1
                        self.best_same_f1 = same_f1
                        self.best_decision_threshold = decision_threshold
                    patience_counter = 0

                    # Save best model
                    checkpoint_path = save_dir / "best_model"
                    self.model.save_pretrained(checkpoint_path)
                    self.tokenizer.save_pretrained(checkpoint_path)
                    if isinstance(evaluation, dict):
                        threshold_payload = {
                            **evaluation,
                            "checkpoint_metric": self.checkpoint_metric,
                            "epoch": epoch,
                        }
                        write_decision_threshold(
                            checkpoint_path / THRESHOLD_FILENAME,
                            threshold_payload,
                        )
                        write_decision_threshold(
                            save_dir / THRESHOLD_FILENAME,
                            threshold_payload,
                        )
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
