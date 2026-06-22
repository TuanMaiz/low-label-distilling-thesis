"""FEBRL dataset module for patient record entity resolution."""
from data.febrl.schema import FebrlRecord, FebrlPair
from data.febrl.loader import FebrlLoader, load_febrl_dataset

__all__ = ["FebrlRecord", "FebrlPair", "FebrlLoader", "load_febrl_dataset"]
