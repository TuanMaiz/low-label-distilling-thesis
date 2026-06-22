"""
FEBRL record schema for patient record entity resolution.

FEBRL (Freely Extensible Biomedical Record Linkage) datasets contain
person-level fields typical of healthcare/customer records. This schema
mirrors the columns returned by the recordlinkage package loaders.
"""
from typing import Optional, List
from pydantic import BaseModel, Field


class FebrlRecord(BaseModel):
    """
    A single FEBRL patient/person record.

    Fields map directly to recordlinkage.datasets FEBRL columns:
        given_name, surname, street_number, address_1, address_2,
        suburb, postcode, state, date_of_birth, soc_sec_id
    """
    record_id: str = Field(..., description="FEBRL index label (e.g., 'rec-1-org')")
    given_name: Optional[str] = Field(None, description="First name")
    surname: Optional[str] = Field(None, description="Last name")
    street_number: Optional[str] = Field(None, description="Street number")
    address_1: Optional[str] = Field(None, description="Address line 1")
    address_2: Optional[str] = Field(None, description="Address line 2")
    suburb: Optional[str] = Field(None, description="Suburb/city")
    postcode: Optional[str] = Field(None, description="Postal code")
    state: Optional[str] = Field(None, description="State/region")
    date_of_birth: Optional[str] = Field(None, description="DOB as YYYYMMDD string")
    soc_sec_id: Optional[str] = Field(None, description="Social security number")

    def to_comparison_string(self) -> str:
        """
        Serialize record to a single string for string-similarity baselines.

        Empty/missing fields are skipped to avoid noise. The soc_sec_id is
        intentionally excluded because baselines that rely on it become
        trivially perfect.
        """
        parts = [
            self.given_name, self.surname,
            self.street_number, self.address_1, self.address_2,
            self.suburb, self.postcode, self.state,
            self.date_of_birth,
        ]
        return " ".join(p for p in parts if p)


class FebrlPair(BaseModel):
    """A pair of FEBRL records with a ground-truth match label."""
    record_a: FebrlRecord
    record_b: FebrlRecord
    label: bool = Field(..., description="True = same person, False = different")
