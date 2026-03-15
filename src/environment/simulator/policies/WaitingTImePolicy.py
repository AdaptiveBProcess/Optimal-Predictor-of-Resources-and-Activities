from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.environment.entities.Activity import Activity
    from src.environment.entities.Resource import Resource





class WaitingTimePolicy(ABC):
    """
    Abstract base class for waiting
    time policies. Defines the interface for getting waiting times for events in the simulation.
    """

    @abstractmethod
    def get_waiting_time(self, activity: "Activity", resource: "Resource") -> float:
        """
        Gets the waiting time for a specific event of a case.

        :param case: The case for which to get the waiting time.
        :return: The waiting time of the event in simulation time units.
        """
        pass
