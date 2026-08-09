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
        "e3e1df959942b96141d0dbbfa65b47ded5bedede446a141461d0d77ef15b3aef",
        "368531b003418d2cb0749949e8e842a6db929d8bb7e71b62ddaac0e3d2af2c51",
    }
)


__all__ = ["TRUSTED_QUALIFIED_PROFILE_SHA256S"]
