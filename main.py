"""
Main entry point for multilingual name entity matching.

Usage:
    python main.py --mode train --model mbart
    python main.py --mode evaluate --checkpoint checkpoints/best_model
    python main.py --mode test
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from models.architectures.mbart import MBartModel
from utils.data_loader import DataLoader
from experiments.trainer import Trainer, create_training_config
from experiments.evaluate import evaluate_on_splits


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Multilingual Name Entity Matching"
    )

    parser.add_argument(
        "--mode",
        type=str,
        choices=["train", "evaluate", "test", "prepare"],
        default="test",
        help="Mode to run in"
    )

    parser.add_argument(
        "--model",
        type=str,
        choices=["mbart", "nllb", "mt5"],
        default="mbart",
        help="Model architecture to use"
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to checkpoint for evaluation"
    )

    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/raw",
        help="Directory containing raw data"
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Batch size for training/evaluation"
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="Number of training epochs"
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=5e-5,
        help="Learning rate"
    )

    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use (cuda/cpu, auto if None)"
    )

    parser.add_argument(
        "--wandb-project",
        type=str,
        default="multilingual-ner",
        help="W&B project name"
    )

    return parser.parse_args()


def create_model(model_name: str, device: str = None) -> MBartModel:
    """Create a model instance."""
    if model_name == "mbart":
        return MBartModel(device=device)
    elif model_name == "nllb":
        raise NotImplementedError("NLLB model not yet implemented")
    elif model_name == "mt5":
        raise NotImplementedError("mT5 model not yet implemented")
    else:
        raise ValueError(f"Unknown model: {model_name}")


def mode_test(args):
    """Test mode: verify setup without training."""
    print("=" * 60)
    print("TEST MODE - Verifying Setup")
    print("=" * 60)

    # Test data loader
    print("\n1. Testing data loader...")
    data_loader = DataLoader(args.data_dir)
    dataset = data_loader.load_dataset()

    stats = data_loader.get_statistics()
    print(f"   Records: {stats['total_records']}")
    print(f"   Pairs: {stats['total_pairs']}")
    print(f"   Languages: {stats['languages']}")
    print(f"   Families: {stats['families']}")

    # Test model (without loading weights)
    print("\n2. Testing model setup...")
    model = create_model(args.model, args.device)
    print(f"   Model: {model.model_name}")
    print(f"   Device: {model.device}")

    # Test formatting
    input_text = model.format_input(
        name="Владимир Путин",
        source_lang="ru",
        target_lang="en",
        age=68,
        gender="M"
    )
    target_text = model.format_target("Vladimir Putin", "en")
    print(f"   Input format: {input_text}")
    print(f"   Target format: {target_text}")

    # Test metrics
    print("\n3. Testing metrics...")
    from utils.metrics import combined_similarity, compute_metrics

    sim = combined_similarity("Vladimir Putin", "Vladimir Vladimirovich Putin")
    print(f"   Similarity test: {sim:.4f}")

    # Test pairs
    print("\n4. Testing pair data...")
    pairs = data_loader.get_pairs_with_records(dataset)
    train_pairs = data_loader.get_split_pairs("train", dataset)
    val_pairs = data_loader.get_split_pairs("val", dataset)
    test_pairs = data_loader.get_split_pairs("test", dataset)

    print(f"   Total pairs: {len(pairs)}")
    print(f"   Train: {len(train_pairs)}")
    print(f"   Val: {len(val_pairs)}")
    print(f"   Test: {len(test_pairs)}")

    print("\n" + "=" * 60)
    print("All tests passed! Ready for training.")
    print("=" * 60)
    print("\nTo start training:")
    print(f"  python main.py --mode train --model {args.model}")
    print("\nNote: To actually train, you'll need to:")
    print("  1. Install full dependencies: uv pip install -r requirements.txt")
    print("  2. Run in Colab or local machine with GPU")


def mode_prepare(args):
    """Prepare mode: create training datasets from data."""
    print("=" * 60)
    print("PREPARE MODE - Creating Training Datasets")
    print("=" * 60)

    data_loader = DataLoader(args.data_dir)
    dataset = data_loader.load_dataset()

    # Get pairs by split
    train_pairs = data_loader.get_split_pairs("train", dataset)
    val_pairs = data_loader.get_split_pairs("val", dataset)
    test_pairs = data_loader.get_split_pairs("test", dataset)

    print(f"\nDataset splits:")
    print(f"  Train: {len(train_pairs)} pairs")
    print(f"  Val: {len(val_pairs)} pairs")
    print(f"  Test: {len(test_pairs)} pairs")

    # Show sample pairs
    print(f"\nSample training pairs:")
    for pair in train_pairs[:3]:
        match = "MATCH" if pair.label else "NO MATCH"
        print(f"  {pair.record_a.name} <-> {pair.record_b.name}: {match}")

    print("\nPreparation complete!")


def mode_train(args):
    """Train mode: train a model."""
    print("=" * 60)
    print("TRAIN MODE")
    print("=" * 60)

    # Check if we're on a GPU-enabled machine
    import torch
    if not torch.cuda.is_available():
        print("\nWARNING: CUDA not available. Training will be very slow.")
        print("Consider running in Google Colab with GPU.")
        response = input("\nContinue anyway? (y/n): ")
        if response.lower() != 'y':
            return

    # Create model
    model = create_model(args.model, args.device)
    print(f"\nLoading model: {model.model_name}")
    model.load_model()
    print(f"Parameters: {model.get_num_parameters():,}")

    # Load data
    print(f"\nLoading data from {args.data_dir}")
    data_loader = DataLoader(args.data_dir)
    dataset = data_loader.load_dataset()

    # Get train/val pairs
    train_pairs = data_loader.get_split_pairs("train", dataset)
    val_pairs = data_loader.get_split_pairs("val", dataset)

    if not train_pairs:
        print("ERROR: No training pairs found!")
        return

    print(f"Train pairs: {len(train_pairs)}")
    print(f"Val pairs: {len(val_pairs)}")

    # Format data for training
    source_lang = "ru"
    target_lang = "en"

    train_inputs = []
    train_targets = []
    for pair in train_pairs:
        if pair.record_a.language == source_lang and pair.record_b.language == target_lang:
            inp = model.format_input(
                pair.record_a.name, source_lang, target_lang,
                pair.record_a.age, pair.record_a.gender
            )
            tgt = model.format_target(pair.record_b.name, target_lang)
            train_inputs.append(inp)
            train_targets.append(tgt)

    print(f"\nFormatted {len(train_inputs)} training examples")

    # Create datasets
    train_dataset = model.create_dataset(train_inputs, train_targets)
    train_loader = model.create_dataloader(train_dataset, batch_size=args.batch_size)

    val_loader = None
    if val_pairs:
        val_inputs = []
        val_targets = []
        for pair in val_pairs:
            if pair.record_a.language == source_lang and pair.record_b.language == target_lang:
                inp = model.format_input(
                    pair.record_a.name, source_lang, target_lang,
                    pair.record_a.age, pair.record_a.gender
                )
                tgt = model.format_target(pair.record_b.name, target_lang)
                val_inputs.append(inp)
                val_targets.append(tgt)

        if val_inputs:
            val_dataset = model.create_dataset(val_inputs, val_targets)
            val_loader = model.create_dataloader(val_dataset, batch_size=args.batch_size, shuffle=False)

    # Create trainer
    config = create_training_config(
        model_name=args.model,
        learning_rate=args.lr,
        batch_size=args.batch_size,
        num_epochs=args.epochs,
        warmup_steps=100
    )

    trainer = Trainer(
        model=model.model,
        tokenizer=model.tokenizer,
        device=model.device,
        learning_rate=args.lr,
        wandb_project=args.wandb_project
    )

    trainer.setup_wandb(config)

    # Train
    print(f"\nStarting training for {args.epochs} epochs...")
    history = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=args.epochs,
        save_dir="checkpoints"
    )

    print("\nTraining complete!")
    print(f"Best val loss: {trainer.best_val_loss:.4f}")


def mode_evaluate(args):
    """Evaluate mode: evaluate a trained model."""
    print("=" * 60)
    print("EVALUATE MODE")
    print("=" * 60)

    if args.checkpoint is None:
        print("ERROR: --checkpoint required for evaluation")
        return

    # Create model
    model = create_model(args.model, args.device)
    print(f"\nLoading checkpoint: {args.checkpoint}")
    model.load_from_checkpoint(args.checkpoint)
    print(f"Parameters: {model.get_num_parameters():,}")

    # Load data
    print(f"\nLoading data from {args.data_dir}")
    data_loader = DataLoader(args.data_dir)

    # Evaluate
    results = evaluate_on_splits(
        model=model.model,
        tokenizer=model.tokenizer,
        data_loader=data_loader,
        device=model.device,
        find_threshold=True
    )

    print("\n" + "=" * 60)
    print("Evaluation complete!")
    print("=" * 60)


def main():
    """Main entry point."""
    args = parse_args()

    if args.mode == "test":
        mode_test(args)
    elif args.mode == "prepare":
        mode_prepare(args)
    elif args.mode == "train":
        mode_train(args)
    elif args.mode == "evaluate":
        mode_evaluate(args)


if __name__ == "__main__":
    main()
