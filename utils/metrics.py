"""
Metrics and evaluation for multilingual entity matching.

Approach: Generation Quality
1. Model generates target language name from source
2. Compare generated vs actual using string similarity
3. Apply threshold for binary classification (MATCH / NO MATCH)
"""

from typing import List, Tuple
import numpy as np


# ============================================================================
# String Similarity Functions
# ============================================================================

def jaro_winkler_similarity(s1: str, s2: str, p: float = 0.1) -> float:
    """
    Compute Jaro-Winkler similarity between two strings.

    The Jaro-Winkler similarity gives more favorable ratings to strings
    that match from the beginning. Used in Paper 1 for name matching.

    Args:
        s1: First string
        s2: Second string
        p: Prefix scaling factor (default 0.1, max 0.25)

    Returns:
        Similarity score between 0 and 1
    """
    def jaro_similarity(str1: str, str2: str) -> float:
        len1, len2 = len(str1), len(str2)

        if len1 == 0 or len2 == 0:
            return 0.0

        match_distance = max(len1, len2) // 2 - 1
        if match_distance < 0:
            match_distance = 0

        str1_matches = [False] * len1
        str2_matches = [False] * len2

        matches = 0
        transpositions = 0

        for i in range(len1):
            start = max(0, i - match_distance)
            end = min(i + match_distance + 1, len2)

            for j in range(start, end):
                if str2_matches[j] or str1[i] != str2[j]:
                    continue
                str1_matches[i] = str2_matches[j] = True
                matches += 1
                break

        if matches == 0:
            return 0.0

        k = 0
        for i in range(len1):
            if not str1_matches[i]:
                continue
            while not str2_matches[k]:
                k += 1
            if str1[i] != str2[k]:
                transpositions += 1
            k += 1

        return (
            matches / len1 +
            matches / len2 +
            (matches - transpositions / 2) / matches
        ) / 3

    jaro = jaro_similarity(s1.lower(), s2.lower())

    # Find common prefix length (up to 4 characters)
    prefix = 0
    for c1, c2 in zip(s1.lower(), s2.lower()):
        if c1 == c2 and prefix < 4:
            prefix += 1
        else:
            break

    p = min(p, 0.25)
    return jaro + prefix * p * (1 - jaro)


def levenshtein_ratio(s1: str, s2: str) -> float:
    """
    Compute Levenshtein distance ratio (normalized similarity).

    Args:
        s1: First string
        s2: Second string

    Returns:
        Similarity score between 0 and 1
    """
    len1, len2 = len(s1), len(s2)

    if len1 == 0 or len2 == 0:
        return 0.0 if len1 != len2 else 1.0

    # Dynamic programming for Levenshtein distance
    dp = [[0] * (len2 + 1) for _ in range(len1 + 1)]

    for i in range(len1 + 1):
        dp[i][0] = i
    for j in range(len2 + 1):
        dp[0][j] = j

    for i in range(1, len1 + 1):
        for j in range(1, len2 + 1):
            cost = 0 if s1[i - 1] == s2[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,      # deletion
                dp[i][j - 1] + 1,      # insertion
                dp[i - 1][j - 1] + cost  # substitution
            )

    distance = dp[len1][len2]
    max_len = max(len1, len2)

    return 1 - (distance / max_len) if max_len > 0 else 1.0


def token_f1_score(pred: str, target: str) -> float:
    """
    Compute token-level F1 score.

    Splits by whitespace and computes precision, recall, F1.
    Good for names with multiple parts.

    Args:
        pred: Predicted string
        target: Target string

    Returns:
        F1 score between 0 and 1
    """
    pred_tokens = set(pred.lower().split())
    target_tokens = set(target.lower().split())

    if not pred_tokens and not target_tokens:
        return 1.0
    if not pred_tokens or not target_tokens:
        return 0.0

    intersection = pred_tokens & target_tokens

    precision = len(intersection) / len(pred_tokens)
    recall = len(intersection) / len(target_tokens)

    if precision + recall == 0:
        return 0.0

    return 2 * precision * recall / (precision + recall)


def character_ngram_f1(pred: str, target: str, n: int = 3) -> float:
    """
    Compute character n-gram F1 score.

    More robust for partial matches and transliterations.

    Args:
        pred: Predicted string
        target: Target string
        n: N-gram size (default 3 for trigrams)

    Returns:
        F1 score between 0 and 1
    """
    def get_ngrams(text: str, n: int) -> set:
        text = text.lower().replace(" ", "")
        if len(text) < n:
            return {text}
        return {text[i:i+n] for i in range(len(text) - n + 1)}

    pred_ngrams = get_ngrams(pred, n)
    target_ngrams = get_ngrams(target, n)

    if not pred_ngrams and not target_ngrams:
        return 1.0
    if not pred_ngrams or not target_ngrams:
        return 0.0

    intersection = pred_ngrams & target_ngrams

    precision = len(intersection) / len(pred_ngrams)
    recall = len(intersection) / len(target_ngrams)

    if precision + recall == 0:
        return 0.0

    return 2 * precision * recall / (precision + recall)


def combined_similarity(
    pred: str,
    target: str,
    weights: dict = None
) -> float:
    """
    Combined similarity score using multiple metrics.

    Args:
        pred: Predicted string
        target: Target string
        weights: Dictionary of weights for each metric
            Default: {"jaro_winkler": 0.4, "levenshtein": 0.3, "token_f1": 0.2, "char_ngram": 0.1}

    Returns:
        Combined similarity score between 0 and 1
    """
    if weights is None:
        weights = {
            "jaro_winkler": 0.4,
            "levenshtein": 0.3,
            "token_f1": 0.2,
            "char_ngram": 0.1
        }

    scores = {
        "jaro_winkler": jaro_winkler_similarity(pred, target),
        "levenshtein": levenshtein_ratio(pred, target),
        "token_f1": token_f1_score(pred, target),
        "char_ngram": character_ngram_f1(pred, target, n=3)
    }

    combined = sum(weights.get(k, 0) * scores[k] for k in scores.keys())
    return combined


# ============================================================================
# Prediction and Classification
# ============================================================================

def predict_match(
    pred_name: str,
    actual_name: str,
    threshold: float = 0.8,
    similarity_fn: str = "combined"
) -> Tuple[bool, float]:
    """
    Predict if two names match based on similarity threshold.

    Args:
        pred_name: Predicted name from the model
        actual_name: Actual target name
        threshold: Similarity threshold for matching (default 0.8)
        similarity_fn: Which similarity function to use
            Options: "jaro_winkler", "levenshtein", "token_f1", "char_ngram", "combined"

    Returns:
        Tuple of (is_match: bool, similarity_score: float)
    """
    similarity_functions = {
        "jaro_winkler": jaro_winkler_similarity,
        "levenshtein": levenshtein_ratio,
        "token_f1": token_f1_score,
        "char_ngram": lambda x, y: character_ngram_f1(x, y, n=3),
        "combined": combined_similarity
    }

    fn = similarity_functions.get(similarity_fn, combined_similarity)
    score = fn(pred_name, actual_name)

    return score >= threshold, score


def compute_all_similarities(pred_name: str, actual_name: str) -> dict:
    """
    Compute all similarity metrics for a pair of names.

    Useful for analysis and ablation studies.
    """
    return {
        "jaro_winkler": jaro_winkler_similarity(pred_name, actual_name),
        "levenshtein": levenshtein_ratio(pred_name, actual_name),
        "token_f1": token_f1_score(pred_name, actual_name),
        "char_ngram": character_ngram_f1(pred_name, actual_name),
        "combined": combined_similarity(pred_name, actual_name)
    }


# ============================================================================
# Classification Metrics (Paper 1 compatible)
# ============================================================================

def compute_metrics(
    predictions: List[bool],
    labels: List[bool]
) -> dict:
    """
    Compute classification metrics (Precision, Recall, F1).

    Matches Paper 1's evaluation format.

    Args:
        predictions: List of predicted labels (True = match, False = no match)
        labels: List of true labels

    Returns:
        Dictionary with precision, recall, f1 for both classes
    """
    predictions = np.array(predictions)
    labels = np.array(labels)

    # True positives, false positives, true negatives, false negatives
    tp = np.sum((predictions == True) & (labels == True))
    fp = np.sum((predictions == True) & (labels == False))
    tn = np.sum((predictions == False) & (labels == False))
    fn = np.sum((predictions == False) & (labels == True))

    # Metrics for "Same" class (positive class)
    same_precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    same_recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    same_f1 = 2 * same_precision * same_recall / (same_precision + same_recall) if (same_precision + same_recall) > 0 else 0.0

    # Metrics for "Different" class (negative class)
    different_precision = tn / (tn + fn) if (tn + fn) > 0 else 0.0
    different_recall = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    different_f1 = 2 * different_precision * different_recall / (different_precision + different_recall) if (different_precision + different_recall) > 0 else 0.0

    # Macro and micro averages
    macro_precision = (same_precision + different_precision) / 2
    macro_recall = (same_recall + different_recall) / 2
    macro_f1 = (same_f1 + different_f1) / 2

    # Overall accuracy
    accuracy = (tp + tn) / (tp + tn + fp + fn) if len(predictions) > 0 else 0.0

    return {
        "same_precision": same_precision,
        "same_recall": same_recall,
        "same_f1": same_f1,
        "different_precision": different_precision,
        "different_recall": different_recall,
        "different_f1": different_f1,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
        "accuracy": accuracy,
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn)
    }


def format_metrics_output(metrics: dict) -> str:
    """Format metrics for display, matching Paper 1's table format."""
    output = []
    output.append("=" * 60)
    output.append("Classification Results (Paper 1 Format)")
    output.append("=" * 60)
    output.append(f"{'Class':<15} {'Precision':<12} {'Recall':<12} {'F1':<12}")
    output.append("-" * 60)
    output.append(f"{'Same':<15} {metrics['same_precision']:<12.4f} {metrics['same_recall']:<12.4f} {metrics['same_f1']:<12.4f}")
    output.append(f"{'Different':<15} {metrics['different_precision']:<12.4f} {metrics['different_recall']:<12.4f} {metrics['different_f1']:<12.4f}")
    output.append("-" * 60)
    output.append(f"{'Macro Avg':<15} {metrics['macro_precision']:<12.4f} {metrics['macro_recall']:<12.4f} {metrics['macro_f1']:<12.4f}")
    output.append(f"{'Accuracy':<15} {metrics['accuracy']:<12.4f}")
    output.append("=" * 60)
    return "\n".join(output)


def find_optimal_threshold(
    similarities: List[float],
    labels: List[bool],
    metric: str = "f1"
) -> float:
    """
    Find optimal threshold for binary classification.

    Searches over threshold values to maximize the specified metric.

    Args:
        similarities: List of similarity scores
        labels: List of true labels (True = match, False = no match)
        metric: Which metric to optimize ("f1", "precision", "recall")

    Returns:
        Optimal threshold value
    """
    similarities = np.array(similarities)
    labels = np.array(labels)

    best_threshold = 0.5
    best_score = 0.0

    for threshold in np.arange(0.0, 1.0, 0.01):
        predictions = similarities >= threshold
        metrics = compute_metrics(predictions.tolist(), labels.tolist())

        if metric == "f1":
            score = metrics["same_f1"]
        elif metric == "precision":
            score = metrics["same_precision"]
        elif metric == "recall":
            score = metrics["same_recall"]
        else:
            score = metrics["macro_f1"]

        if score > best_score:
            best_score = score
            best_threshold = threshold

    return best_threshold


# ============================================================================
# Generation-specific metrics
# ============================================================================

def exact_match_rate(pred_names: List[str], target_names: List[str]) -> float:
    """
    Compute exact match rate (string equality).

    Args:
        pred_names: List of predicted names
        target_names: List of target names

    Returns:
        Fraction of exact matches
    """
    if len(pred_names) != len(target_names):
        raise ValueError("pred_names and target_names must have same length")

    matches = sum(1 for p, t in zip(pred_names, target_names) if p.lower().strip() == t.lower().strip())
    return matches / len(pred_names) if pred_names else 0.0


def average_similarity(pred_names: List[str], target_names: List[str], similarity_fn: str = "combined") -> float:
    """
    Compute average similarity score across all pairs.

    Args:
        pred_names: List of predicted names
        target_names: List of target names
        similarity_fn: Which similarity function to use

    Returns:
        Average similarity score
    """
    if len(pred_names) != len(target_names):
        raise ValueError("pred_names and target_names must have same length")

    similarity_functions = {
        "jaro_winkler": jaro_winkler_similarity,
        "levenshtein": levenshtein_ratio,
        "token_f1": token_f1_score,
        "char_ngram": lambda x, y: character_ngram_f1(x, y, n=3),
        "combined": combined_similarity
    }

    fn = similarity_functions.get(similarity_fn, combined_similarity)

    scores = [fn(p, t) for p, t in zip(pred_names, target_names)]
    return sum(scores) / len(scores) if scores else 0.0


# ============================================================================
# Test code
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Similarity Function Tests")
    print("=" * 60)

    test_cases = [
        ("Vladimir Putin", "Vladimir Putin", "Exact match"),
        ("Vladimir Putin", "Vladimir Vladimirovich Putin", "Partial match (shortened)"),
        ("Vladimir Putin", "Joseph Stalin", "No match"),
        ("Vladimir Putin", "Vova Putin", "Nickname"),
        ("Ivan Petrov", "Ivan Sidorov", "Same first, different last"),
        ("Vladimir Put", "Vladimir Putin", "Typo/shortened"),
    ]

    for pred, target, desc in test_cases:
        jw = jaro_winkler_similarity(pred, target)
        lev = levenshtein_ratio(pred, target)
        tf1 = token_f1_score(pred, target)
        cg = character_ngram_f1(pred, target)
        combined = combined_similarity(pred, target)
        is_match, score = predict_match(pred, target, threshold=0.8)

        print(f"\n[{desc}]")
        print(f"  Pred:    '{pred}'")
        print(f"  Target:  '{target}'")
        print(f"  Jaro-Winkler:  {jw:.4f}")
        print(f"  Levenshtein:   {lev:.4f}")
        print(f"  Token F1:      {tf1:.4f}")
        print(f"  Char Ngram:    {cg:.4f}")
        print(f"  Combined:      {combined:.4f}")
        print(f"  Prediction:    {'MATCH' if is_match else 'NO MATCH'} (threshold=0.8)")

    # Test classification metrics
    print("\n" + "=" * 60)
    print("Classification Metrics Test")
    print("=" * 60)

    # Simulated predictions based on similarity
    predictions = [True, True, False, True, False, False, True, True, False]
    labels =       [True, False, False, True, True, False, True, False, False]

    metrics = compute_metrics(predictions, labels)
    print(format_metrics_output(metrics))

    # Test threshold finding
    print("\n" + "=" * 60)
    print("Optimal Threshold Test")
    print("=" * 60)

    similarities = [0.95, 0.85, 0.30, 0.92, 0.45, 0.20, 0.88, 0.75, 0.15]
    labels =      [True,  True, False, True, False, False, True,  False, False]

    optimal = find_optimal_threshold(similarities, labels, metric="f1")
    print(f"Optimal threshold (maximizing F1): {optimal:.2f}")

    # Apply optimal threshold
    predictions = [s >= optimal for s in similarities]
    metrics = compute_metrics(predictions, labels)
    print(f"\nMetrics at optimal threshold:")
    print(f"  Same F1: {metrics['same_f1']:.4f}")
    print(f"  Macro F1: {metrics['macro_f1']:.4f}")
