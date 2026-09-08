"""Remote bridge package for CryoSPARC access."""

from __future__ import annotations

from cryofilter import __version__ as CRYOFILTER_VERSION
from cryofilter.cryosparc import PROTOCOL_VERSION

BRIDGE_VERSION = CRYOFILTER_VERSION

__all__ = ["BRIDGE_VERSION", "PROTOCOL_VERSION"]
