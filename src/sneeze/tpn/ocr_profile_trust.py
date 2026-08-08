"""Reviewed execution-profile allowlist.

This registry is intentionally separate from the attested OCR implementation
bundle. Qualified profiles bind that implementation digest, so including
their hashes in the implementation digest would create a circular identity:
the profile would depend on a digest that itself depended on the profile.

Adding an entry remains a source-controlled review decision. The attested
validation and selection logic lives in :mod:`sneeze.tpn.ocr_profiles`.
"""

from __future__ import annotations

TRUSTED_QUALIFIED_PROFILE_SHA256S: frozenset[str] = frozenset(
    {
        "0b15775d6931f451b4efa9bb77a38ca2c546a707932e4ddee33da11955ac7f58",
    }
)


__all__ = ["TRUSTED_QUALIFIED_PROFILE_SHA256S"]
