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
    def get_next_activity(self, case: "Case") -> "Activity":
        """
        Determines the next activity for a given case.

        :param case: The case for which to determine the next activity. Policies
                     should extract whatever context they need from the case
                     (e.g. case.activity_history for the current or previous activities).
        :return: The next Activity object, or None if the process has completed.
        """
        pass