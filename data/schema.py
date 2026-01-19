"""
Data schema definitions for multilingual entity matching.
Based on Paper 1: "Multilingual Entity Matching"
"""

from typing import Optional, List
from pydantic import BaseModel, Field


class PersonRecord(BaseModel):
    """
    A single person record in one language.

    Attributes:
        record_id: Unique identifier for this specific record
        person_id: Real-world entity ID (e.g., Wikidata Q-number)
        family_id: Family grouping identifier
        name: Full name as a single string
        language: Language code ("ru" or "en")
        age: Age at time of record (optional)
        gender: Gender ("M" or "F", optional)
    """
    record_id: str = Field(..., description="Unique record identifier")
    person_id: str = Field(..., description="Real-world entity ID (e.g., Q123)")
    family_id: str = Field(..., description="Family grouping identifier")
    name: str = Field(..., description="Full name as single string")
    language: str = Field(..., description="Language code: 'ru' or 'en'")
    age: Optional[int] = Field(None, description="Age at time of record")
    gender: Optional[str] = Field(None, description="Gender: 'M' or 'F'")


class Relationships(BaseModel):
    """
    Family relationships for a person.

    These are used for collective entity resolution in PSL.
    For our generative approach, they may be used as additional context.
    """
    mother_id: Optional[str] = None
    father_id: Optional[str] = None
    spouse_ids: List[str] = Field(default_factory=list)
    sister_ids: List[str] = Field(default_factory=list)
    brother_ids: List[str] = Field(default_factory=list)
    daughter_ids: List[str] = Field(default_factory=list)
    son_ids: List[str] = Field(default_factory=list)


class PersonWithRelationships(PersonRecord):
    """
    Extended person record including family relationships.
    """
    relationships: Optional[Relationships] = None


class RecordPair(BaseModel):
    """
    A pair of records for entity matching.

    This is what the model actually predicts on: given two records,
    determine if they refer to the same real-world entity.

    Attributes:
        record_a_id: ID of first record
        record_b_id: ID of second record
        label: True if same person, False if different
        split: Dataset split ("train", "val", "test", "threshold")
    """
    record_a_id: str = Field(..., description="ID of first record")
    record_b_id: str = Field(..., description="ID of second record")
    label: bool = Field(..., description="True = same person, False = different")
    split: str = Field(..., description="Dataset split: 'train', 'val', 'test', 'threshold'")


class RecordPairWithRecords(RecordPair):
    """
    RecordPair that includes the full record data for convenience.
    Useful for training and evaluation.
    """
    record_a: PersonRecord
    record_b: PersonRecord


class Dataset(BaseModel):
    """
    Complete dataset container.

    Attributes:
        records: List of all person records
        pairs: List of all record pairs (positive and negative examples)
        languages: List of language codes in the dataset
    """
    records: List[PersonRecord] = Field(default_factory=list)
    pairs: List[RecordPair] = Field(default_factory=list)
    languages: List[str] = Field(default_factory=list)

    def get_records_by_id(self, record_id: str) -> Optional[PersonRecord]:
        """Get a record by its ID."""
        for record in self.records:
            if record.record_id == record_id:
                return record
        return None

    def get_records_by_person(self, person_id: str) -> List[PersonRecord]:
        """Get all records (language variants) for a person."""
        return [r for r in self.records if r.person_id == person_id]

    def get_family_members(self, family_id: str) -> List[PersonRecord]:
        """Get all records for a family."""
        return [r for r in self.records if r.family_id == family_id]
