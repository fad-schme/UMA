"""
UMA-3 Feature Attachment Interface (Production Version)
=======================================================

This module defines:
    - UMAFeature: Abstract base class for pluggable UMA-3 features

This remains part of the UMA-3 CORE because optional capabilities
must share a consistent attachment interface.

Coding Agent Instructions
-------------------------
- Each optional feature MUST subclass UMAFeature.
- Features MUST implement .attach(memory_client) safely.
- Exceptions in attach() MUST BE logged AND must NOT break UMA-3.
- Optional features attach directly to UMAMemory.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


# ======================================================================
#  UMAFeature (ABC)
# ======================================================================

class UMAFeature(ABC):
    """
    Abstract base class for UMA-3 optional plugin features.

    Requirements
    ------------
    - Must set class attribute `name` (string)
    - Must implement .attach(memory_client)
    - Must NOT raise exceptions during attach()

    attach(memory_client):
        Use this to extend UMAMemory with additional helper functions,
        hook registrations, or specialized subsystems.
    """

    #: Name used to reference this feature inside UMAMemory.features
    name: str = "unnamed_feature"

    @abstractmethod
    def attach(self, memory_client: Any) -> None:
        """
        Attach feature to UMAMemory instance.

        Must:
          - Register itself under memory_client.features[self.name]
          - Initialize internal resources
          - Not raise exceptions (log instead)

        Parameters
        ----------
        memory_client : UMAMemory
            The runtime memory container object.
        """
        raise NotImplementedError
