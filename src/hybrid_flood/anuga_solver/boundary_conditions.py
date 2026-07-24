"""Configure conservative ANUGA boundary conditions."""

from __future__ import annotations

import logging
from typing import Any

import anuga

LOGGER = logging.getLogger(__name__)

# TODO: verify BndTypeNo=2 semantics before the final report and replace this
# all-reflective policy only after the supplied boundary schema is confirmed.
UNVERIFIED_BOUNDARY_MESSAGE = (
    "Boundary BndTypeNo semantics are unverified; applying reflective conditions "
    "to every supplied segment and all remaining exterior edges. In particular, "
    "2Dboundary_1 (BndTypeNo=2, ConstValue=216) is NOT treated as an outflow."
)


def configure_boundary_conditions(
    domain: anuga.Domain,
    *,
    policy: str = "all_reflective_unverified",
) -> dict[str, Any]:
    """Apply the user-confirmed safe all-reflective interim policy."""
    if policy != "all_reflective_unverified":
        raise ValueError(
            "Only the confirmed 'all_reflective_unverified' policy is currently supported."
        )
    LOGGER.warning(UNVERIFIED_BOUNDARY_MESSAGE)

    tags = sorted(set(domain.boundary.values()))
    reflective = anuga.Reflective_boundary(domain)
    domain.set_boundary({tag: reflective for tag in tags})
    return {
        "policy": policy,
        "boundary_tags": tags,
        "condition_by_tag": {tag: "Reflective_boundary" for tag in tags},
        "warning": UNVERIFIED_BOUNDARY_MESSAGE,
    }
