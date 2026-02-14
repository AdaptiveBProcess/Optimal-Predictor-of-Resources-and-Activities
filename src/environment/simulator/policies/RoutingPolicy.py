from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.environment.entities.Case import Case
    from src.environment.entities.Activity import Activity


class RoutingPolicy(ABC):
    """
    An abstract base class for routing policies.
    A routing policy determines the next activity in a process instance (case).

    Implementations of this class can range from simple rule-based logic
    (e.g., probabilistic routing from a process model) to complex machine
    learning models that predict the next step based on the case history
    and other contextual data.
    """

    @abstractmethod
    def  get_next_activity(self, case: "Case", current_activity: "Activity" = None) -> "Activity":
        """
        Determines the next activity for a given case.

        :param case: The case for which to determine the next activity.
        :param current_activity: The activity that was just completed. Can be None if it's the start of the process.
        :return: The next Activity object, or None if the process has completed.
        """
        pass