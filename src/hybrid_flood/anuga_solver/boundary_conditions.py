"""Apply the project's final all-reflective ANUGA boundary policy.

The three supplied boundary segments are deliberately treated as closed,
no-flow boundaries.  An investigation of the raw ``BndTypeNo`` and
``ConstValue`` fields found no authoritative schema, legend, metadata, or
recoverable source documentation.  The all-reflective treatment is therefore
a permanent modelling assumption for this project and a known limitation,
not an unresolved implementation task.  The evidence and decision are recorded
in ``docs/decisions_log.md``.
"""

from __future__ import annotations

import logging
from typing import Any

import anuga

LOGGER = logging.getLogger(__name__)

FINAL_REFLECTIVE_POLICY = "all_reflective_final"
REFLECTIVE_ASSUMPTION = (
    "All supplied boundary segments and remaining exterior edges are reflective "
    "by final documented modelling choice; see docs/decisions_log.md."
)


def configure_boundary_conditions(
    domain: anuga.Domain,
    *,
    policy: str = FINAL_REFLECTIVE_POLICY,
) -> dict[str, Any]:
    """Apply reflective conditions to every exterior tag by final design."""
    if policy != FINAL_REFLECTIVE_POLICY:
        raise ValueError(f"Only the final {FINAL_REFLECTIVE_POLICY!r} policy is supported.")
    LOGGER.info(REFLECTIVE_ASSUMPTION)

    tags = sorted(set(domain.boundary.values()))
    reflective = anuga.Reflective_boundary(domain)
    domain.set_boundary({tag: reflective for tag in tags})
    return {
        "policy": policy,
        "boundary_tags": tags,
        "condition_by_tag": {tag: "Reflective_boundary" for tag in tags},
        "modeling_assumption": REFLECTIVE_ASSUMPTION,
    }
