from abc import ABC, abstractmethod
from typing import List

from environment.entities.Resource import Resource

class ResourceAllocationPolicy(ABC):

    @abstractmethod
    def select_resource(self, activity, case=None) -> "Resource":
        """
        Returns a Resource object (domain resource).
        Must not block or wait.
        """
        pass
