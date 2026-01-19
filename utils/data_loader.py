"""
Data loading utilities for multilingual entity matching.
"""

import csv
from pathlib import Path
from typing import List, Optional, Tuple

from data.schema import (
    PersonRecord,
    RecordPair,
    RecordPairWithRecords,
    Dataset
)


class DataLoader:
    """
    Load and manage multilingual entity matching dataset.
    """

    def __init__(self, data_dir: str = "data/raw"):
        self.data_dir = Path(data_dir)
        self._dataset: Optional[Dataset] = None

    def load_records(self, filename: str = "fake_dataset.csv") -> List[PersonRecord]:
        """
        Load person records from CSV file.

        Expected CSV format:
        record_id,person_id,family_id,name,language,age,gender

        Args:
            filename: Name of the CSV file in data_dir

        Returns:
            List of PersonRecord objects
        """
        filepath = self.data_dir / filename
        records = []

        with open(filepath, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Convert age to int if present
                age = int(row["age"]) if row["age"] else None
                # Gender can be empty
                gender = row["gender"] if row["gender"] else None

                record = PersonRecord(
                    record_id=row["record_id"],
                    person_id=row["person_id"],
                    family_id=row["family_id"],
                    name=row["name"],
                    language=row["language"],
                    age=age,
                    gender=gender
                )
                records.append(record)

        return records

    def load_pairs(
        self,
        filename: str = "fake_pairs.csv"
    ) -> List[RecordPair]:
        """
        Load record pairs from CSV file.

        Expected CSV format:
        record_a_id,record_b_id,label,split

        Args:
            filename: Name of the CSV file in data_dir

        Returns:
            List of RecordPair objects
        """
        filepath = self.data_dir / filename
        pairs = []

        with open(filepath, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                pair = RecordPair(
                    record_a_id=row["record_a_id"],
                    record_b_id=row["record_b_id"],
                    label=row["label"].lower() == "true",
                    split=row["split"]
                )
                pairs.append(pair)

        return pairs

    def load_dataset(
        self,
        records_file: str = "fake_dataset.csv",
        pairs_file: str = "fake_pairs.csv"
    ) -> Dataset:
        """
        Load complete dataset (records + pairs).

        Args:
            records_file: Name of the records CSV file
            pairs_file: Name of the pairs CSV file

        Returns:
            Dataset object with records and pairs
        """
        records = self.load_records(records_file)
        pairs = self.load_pairs(pairs_file)

        # Get unique languages
        languages = sorted(set(r.language for r in records))

        self._dataset = Dataset(
            records=records,
            pairs=pairs,
            languages=languages
        )

        return self._dataset

    def get_pairs_with_records(
        self,
        dataset: Optional[Dataset] = None
    ) -> List[RecordPairWithRecords]:
        """
        Get pairs with full record data attached.

        Args:
            dataset: Dataset to use (loads from disk if None)

        Returns:
            List of RecordPairWithRecords
        """
        if dataset is None:
            dataset = self.load_dataset()

        # Create lookup dict
        records_dict = {r.record_id: r for r in dataset.records}

        pairs_with_records = []
        for pair in dataset.pairs:
            record_a = records_dict.get(pair.record_a_id)
            record_b = records_dict.get(pair.record_b_id)

            if record_a is None or record_b is None:
                raise ValueError(
                    f"Record not found for pair: {pair.record_a_id}, {pair.record_b_id}"
                )

            pair_with_records = RecordPairWithRecords(
                record_a_id=pair.record_a_id,
                record_b_id=pair.record_b_id,
                label=pair.label,
                split=pair.split,
                record_a=record_a,
                record_b=record_b
            )
            pairs_with_records.append(pair_with_records)

        return pairs_with_records

    def get_split_pairs(
        self,
        split: str,
        dataset: Optional[Dataset] = None
    ) -> List[RecordPairWithRecords]:
        """
        Get pairs for a specific split.

        Args:
            split: One of "train", "val", "test", "threshold"
            dataset: Dataset to use (loads from disk if None)

        Returns:
            List of RecordPairWithRecords for the specified split
        """
        all_pairs = self.get_pairs_with_records(dataset)
        return [p for p in all_pairs if p.split == split]

    def get_cross_lingual_pairs(
        self,
        lang_a: str,
        lang_b: str,
        dataset: Optional[Dataset] = None
    ) -> List[RecordPairWithRecords]:
        """
        Get pairs where records are in specific languages.

        Args:
            lang_a: Language of record_a (e.g., "ru")
            lang_b: Language of record_b (e.g., "en")
            dataset: Dataset to use (loads from disk if None)

        Returns:
            List of RecordPairWithRecords matching the language criteria
        """
        all_pairs = self.get_pairs_with_records(dataset)
        return [
            p for p in all_pairs
            if p.record_a.language == lang_a and p.record_b.language == lang_b
        ]

    def get_statistics(self, dataset: Optional[Dataset] = None) -> dict:
        """
        Get dataset statistics.

        Args:
            dataset: Dataset to use (loads from disk if None)

        Returns:
            Dictionary with dataset statistics
        """
        if dataset is None:
            dataset = self.load_dataset()

        pairs_with_records = self.get_pairs_with_records(dataset)

        # Count by split
        splits = {}
        for pair in pairs_with_records:
            splits[pair.split] = splits.get(pair.split, 0) + 1

        # Count positive/negative
        positive = sum(1 for p in pairs_with_records if p.label)
        negative = sum(1 for p in pairs_with_records if not p.label)

        # Count by family
        families = {}
        for record in dataset.records:
            families[record.family_id] = families.get(record.family_id, 0) + 1

        return {
            "total_records": len(dataset.records),
            "total_pairs": len(pairs_with_records),
            "positive_pairs": positive,
            "negative_pairs": negative,
            "positive_ratio": positive / len(pairs_with_records) if pairs_with_records else 0,
            "splits": splits,
            "languages": dataset.languages,
            "families": len(families),
            "family_counts": families
        }


def format_for_generation(
    pair: RecordPairWithRecords,
    target_language: str
) -> Tuple[str, str]:
    """
    Format a record pair for the generative task.

    Given a pair of records, create an input string that asks the model
    to translate/transliterate the name from source language to target language.

    Args:
        pair: Record pair with full record data
        target_language: Target language code (e.g., "en")

    Returns:
        Tuple of (input_string, target_string)
        - input_string: Prompt for the model
        - target_string: Expected output (the actual name in target language)
    """
    source_record = pair.record_a
    target_record = pair.record_b

    # Build input prompt
    parts = []
    parts.append(f"[TRANSLATE]")
    parts.append(f"[{source_record.language.upper()}→{target_language.upper()}]")

    if source_record.age is not None:
        parts.append(f"[AGE:{source_record.age}]")
    if source_record.gender is not None:
        parts.append(f"[GENDER:{source_record.gender}]")

    parts.append(source_record.name)

    input_string = " ".join(parts)
    target_string = target_record.name

    return input_string, target_string


if __name__ == "__main__":
    # Test the data loader
    loader = DataLoader()
    dataset = loader.load_dataset()

    print("=" * 50)
    print("Dataset Statistics")
    print("=" * 50)
    stats = loader.get_statistics()
    for key, value in stats.items():
        if key != "family_counts":
            print(f"{key}: {value}")

    print("\n" + "=" * 50)
    print("Sample Records")
    print("=" * 50)
    for record in dataset.records[:3]:
        print(f"  {record.record_id}: {record.name} ({record.language})")

    print("\n" + "=" * 50)
    print("Sample Pairs")
    print("=" * 50)
    pairs = loader.get_pairs_with_records(dataset)
    for pair in pairs[:5]:
        match = "MATCH" if pair.label else "NO MATCH"
        print(f"  {pair.record_a_id} <-> {pair.record_b_id}: {match}")
        print(f"    {pair.record_a.name} ({pair.record_a.language})")
        print(f"    {pair.record_b.name} ({pair.record_b.language})")

    print("\n" + "=" * 50)
    print("Sample Generation Format")
    print("=" * 50)
    sample_pair = pairs[0]  # Should be a positive pair
    inp, target = format_for_generation(sample_pair, "en")
    print(f"Input:  {inp}")
    print(f"Target: {target}")
