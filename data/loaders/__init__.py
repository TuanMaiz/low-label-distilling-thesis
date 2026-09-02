"""Dataset-specific adapters for the generic ER pair schema."""

from data.loaders.dblp_acm import DblpAcmLoadResult, audit_and_load_dblp_acm

__all__ = ["DblpAcmLoadResult", "audit_and_load_dblp_acm"]
