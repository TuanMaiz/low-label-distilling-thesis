"""
Evaluation module for multilingual name entity matching.

Implements evaluation matching Paper 1's format:
- Precision, Recall, F1 for both classes (Same/Different)
- Macro-average and Micro-average F1
"""

import torch
from typing import List, Dict, Optional, Tuple
from tqdm import tqdm

from utils.metrics import (
    jaro_winkler_similarity,
    combined_similarity,
    predict_match,
    compute_metrics,
    format_metrics_output,
    find_optimal_threshold
)
from utils.data_loader import RecordPairWithRecords, DataLoader as DataLoader


class Evaluator:
    """
    Evaluator for multilingual entity matching models.
    """

    def __init__(
        self,
        model,
        tokenizer,
        device: str = "cuda",
        similarity_fn: str = "combined",
        default_threshold: float = 0.8
    ):
        """
        Initialize evaluator.

        Args:
            model: The seq2seq model
            tokenizer: The tokenizer
            device: Device to use
            similarity_fn: Which similarity function to use
            default_threshold: Default threshold for binary classification
        """
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.similarity_fn = similarity_fn
        self.default_threshold = default_threshold

        self.model.to(self.device)

    @torch.no_grad()
    def generate_names(
        self,
        pairs: List[RecordPairWithRecords],
        source_lang: str = "ru",
        target_lang: str = "en",
        batch_size: int = 8,
        show_progress: bool = True
    ) -> List[str]:
        """
        Generate target language names from source records.

        Args:
            pairs: List of record pairs
            source_lang: Source language code
            target_lang: Target language code
            batch_size: Batch size for generation
            show_progress: Show progress bar

        Returns:
            List of generated names
        """
        self.model.eval()

        # Format inputs
        input_texts = []
        for pair in pairs:
            # Use the source language record
            if pair.record_a.language == source_lang:
                record = pair.record_a
            elif pair.record_b.language == source_lang:
                record = pair.record_b
            else:
                raise ValueError(f"Neither record is in source language {source_lang}")

            input_text = self._format_input(
                record.name, source_lang, target_lang,
                record.age, record.gender
            )
            input_texts.append(input_text)

        # Generate in batches
        generated_names = []
        iterator = range(0, len(input_texts), batch_size)

        if show_progress:
            iterator = tqdm(iterator, desc="Generating names")

        for i in iterator:
            batch = input_texts[i:i + batch_size]

            inputs = self.tokenizer(
                batch,
                max_length=128,
                padding=True,
                truncation=True,
                return_tensors="pt"
            ).to(self.device)

            outputs = self.model.generate(
                **inputs,
                max_length=128,
                num_beams=5
            )

            batch_generated = self.tokenizer.batch_decode(
                outputs, skip_special_tokens=True
            )

            # Clean up language prefixes
            for name in batch_generated:
                clean_name = self._clean_output(name)
                generated_names.append(clean_name)

        return generated_names

    def _format_input(
        self,
        name: str,
        source_lang: str,
        target_lang: str,
        age: Optional[int] = None,
        gender: Optional[str] = None
    ) -> str:
        """Format input for the model."""
        parts = []
        parts.append("[TRANSLATE]")
        parts.append(f"[{source_lang.upper()}→{target_lang.upper()}]")

        if age is not None:
            parts.append(f"[AGE:{age}]")
        if gender is not None:
            parts.append(f"[GENDER:{gender}]")

        parts.append(name)

        return " ".join(parts)

    def _clean_output(self, text: str) -> str:
        """Remove language prefix from generated text."""
        text = text.strip()

        for lang in ["en", "ru", "de", "fr", "es", "it", "zh", "ja"]:
            prefix = f"[{lang.upper()}] "
            if text.startswith(prefix):
                text = text[len(prefix):]
                break

        return text.strip()

    def evaluate_pairs(
        self,
        pairs: List[RecordPairWithRecords],
        source_lang: str = "ru",
        target_lang: str = "en",
        threshold: Optional[float] = None,
        batch_size: int = 8
    ) -> Dict:
        """
        Evaluate on a set of record pairs.

        Args:
            pairs: List of record pairs
            source_lang: Source language code
            target_lang: Target language code
            threshold: Similarity threshold (uses default if None)
            batch_size: Batch size for generation

        Returns:
            Dictionary with evaluation results
        """
        if threshold is None:
            threshold = self.default_threshold

        # Generate names
        generated_names = self.generate_names(
            pairs, source_lang, target_lang, batch_size
        )

        # Get actual target names
        actual_names = []
        for pair in pairs:
            if pair.record_a.language == target_lang:
                actual_names.append(pair.record_a.name)
            elif pair.record_b.language == target_lang:
                actual_names.append(pair.record_b.name)
            else:
                raise ValueError(f"Neither record is in target language {target_lang}")

        # Get true labels
        true_labels = [pair.label for pair in pairs]

        # Compute similarities
        similarities = []
        for gen, act in zip(generated_names, actual_names):
            if self.similarity_fn == "combined":
                sim = combined_similarity(gen, act)
            elif self.similarity_fn == "jaro_winkler":
                sim = jaro_winkler_similarity(gen, act)
            else:
                sim = combined_similarity(gen, act)  # default

            similarities.append(sim)

        # Make predictions
        pred_labels = [s >= threshold for s in similarities]

        # Compute metrics
        metrics = compute_metrics(pred_labels, true_labels)

        # Add additional info
        metrics["threshold"] = threshold
        metrics["similarities"] = similarities
        metrics["generated_names"] = generated_names
        metrics["actual_names"] = actual_names

        return metrics

    def find_best_threshold(
        self,
        pairs: List[RecordPairWithRecords],
        source_lang: str = "ru",
        target_lang: str = "en",
        split_name: str = "val",
        batch_size: int = 8
    ) -> float:
        """
        Find optimal threshold on a validation set.

        Args:
            pairs: List of record pairs
            source_lang: Source language code
            target_lang: Target language code
            split_name: Name of the split (for logging)
            batch_size: Batch size for generation

        Returns:
            Optimal threshold value
        """
        # Generate without thresholding
        generated_names = self.generate_names(
            pairs, source_lang, target_lang, batch_size
        )

        # Get actual names and labels
        actual_names = []
        true_labels = []
        for pair in pairs:
            if pair.record_a.language == target_lang:
                actual_names.append(pair.record_a.name)
            elif pair.record_b.language == target_lang:
                actual_names.append(pair.record_b.name)
            else:
                raise ValueError(f"Neither record is in target language {target_lang}")
            true_labels.append(pair.label)

        # Compute all similarities
        similarities = []
        for gen, act in zip(generated_names, actual_names):
            sim = combined_similarity(gen, act)
            similarities.append(sim)

        # Find optimal threshold
        optimal_threshold = find_optimal_threshold(
            similarities, true_labels, metric="f1"
        )

        print(f"Optimal threshold on {split_name}: {optimal_threshold:.2f}")

        return optimal_threshold

    def print_results(self, metrics: Dict) -> None:
        """Print evaluation results in Paper 1 format."""
        print(format_metrics_output(metrics))

    def get_error_analysis(
        self,
        pairs: List[RecordPairWithRecords],
        metrics: Dict,
        num_examples: int = 10
    ) -> Dict:
        """
        Get examples of correct and incorrect predictions for error analysis.

        Args:
            pairs: List of record pairs
            metrics: Metrics dictionary (must contain generated_names, actual_names)
            num_examples: Number of examples to return for each category

        Returns:
            Dictionary with error analysis examples
        """
        generated_names = metrics["generated_names"]
        actual_names = metrics["actual_names"]
        similarities = metrics["similarities"]

        # Reconstruct predictions
        threshold = metrics.get("threshold", self.default_threshold)
        pred_labels = [s >= threshold for s in similarities]
        true_labels = [pair.label for pair in pairs]

        # Find examples
        true_positives = []
        false_positives = []
        false_negatives = []
        true_negatives = []

        for i, (pred, true) in enumerate(zip(pred_labels, true_labels)):
            example = {
                "index": i,
                "generated": generated_names[i],
                "actual": actual_names[i],
                "similarity": similarities[i],
                "record_a": pairs[i].record_a.name,
                "record_b": pairs[i].record_b.name,
                "true_label": true
            }

            if pred and true:
                true_positives.append(example)
            elif pred and not true:
                false_positives.append(example)
            elif not pred and true:
                false_negatives.append(example)
            else:
                true_negatives.append(example)

        return {
            "true_positives": true_positives[:num_examples],
            "false_positives": false_positives[:num_examples],
            "false_negatives": false_negatives[:num_examples],
            "true_negatives": true_negatives[:num_examples]
        }


def evaluate_on_splits(
    model,
    tokenizer,
    data_loader: DataLoader,
    device: str = "cuda",
    find_threshold: bool = True
) -> Dict:
    """
    Evaluate model on all data splits.

    Args:
        model: The seq2seq model
        tokenizer: The tokenizer
        data_loader: Data loader with dataset
        device: Device to use
        find_threshold: Whether to find optimal threshold on val set

    Returns:
        Dictionary with results for all splits
    """
    evaluator = Evaluator(model, tokenizer, device)

    results = {}

    # Load dataset
    dataset = data_loader.load_dataset()

    # Get pairs by split
    splits = ["train", "val", "test"]

    if find_threshold:
        # First find optimal threshold on validation set
        print("\n" + "=" * 60)
        print("Finding optimal threshold on validation set...")
        print("=" * 60)
        val_pairs = data_loader.get_split_pairs("val", dataset)
        if val_pairs:
            optimal_threshold = evaluator.find_best_threshold(
                val_pairs, source_lang="ru", target_lang="en", split_name="val"
            )
            evaluator.default_threshold = optimal_threshold
        else:
            print("No validation set found, using default threshold")

    # Evaluate on each split
    for split in splits:
        pairs = data_loader.get_split_pairs(split, dataset)

        if not pairs:
            continue

        print(f"\n{'=' * 60}")
        print(f"Evaluating on {split.upper()} set ({len(pairs)} pairs)")
        print(f"{'=' * 60}")

        split_metrics = evaluator.evaluate_pairs(
            pairs, source_lang="ru", target_lang="en",
            threshold=evaluator.default_threshold
        )

        evaluator.print_results(split_metrics)
        results[split] = split_metrics

    return results


if __name__ == "__main__":
    print("Evaluation module loaded successfully")
    print("Use evaluate_on_splits() for full evaluation")
