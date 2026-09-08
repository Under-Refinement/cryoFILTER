"""Shared JSON protocol package for local orchestrator <-> bridge communication.

Model classes live in ``cryofilter.cryosparc.protocol.models``. They are not
imported eagerly here so engaging-side transport/deploy commands can run before
the local environment has pydantic installed.
"""

from __future__ import annotations

__all__: list[str] = []
