"""
Standardised result envelope – Issue #609.

Every AI endpoint response must carry:

  result          – a concise, domain-specific summary of the AI decision
                    (e.g. "ocr_complete", "real_person", "fraud_detected").
  confidence      – float in [0, 1] representing model certainty.
  reasons         – ordered list of human-readable explanation strings.
  anchor_metadata – structured caller-correlating context (via AnchorMetadata).
  trace_id        – UUID-style request identifier for end-to-end tracing.

All fields are Optional so that callers that have not yet migrated continue to
work (no breaking change), while new consumers can rely on the full contract.
"""

import uuid
from typing import List, Optional

from pydantic import BaseModel, Field

from schemas.common import AnchorMetadata


def _new_trace_id() -> str:
    """Generate a fresh UUID4 trace identifier."""
    return str(uuid.uuid4())


class ResultEnvelope(BaseModel):
    """Mixin that adds the standard result envelope fields to any response model."""

    result: Optional[str] = Field(
        default=None,
        description="Concise AI decision label (e.g. 'ocr_complete', 'real_person').",
    )
    confidence: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Model certainty in [0, 1].",
    )
    reasons: Optional[List[str]] = Field(
        default=None,
        description="Ordered list of human-readable explanation strings.",
    )
    anchor_metadata: Optional[AnchorMetadata] = Field(
        default=None,
        description="Structured caller-correlating context.",
    )
    trace_id: Optional[str] = Field(
        default_factory=_new_trace_id,
        description="UUID-style request identifier for end-to-end tracing.",
    )
