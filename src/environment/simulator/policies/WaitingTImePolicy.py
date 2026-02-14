from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.environment.entities.Activity import Activity
    from src.environment.entities.Resource import Resource


class WaitingTimePolicy(ABC):
    """
    An abstract base class for waiting time policies.
    A waiting time policy determines the duration of an activity, often referred to
    as the processing time.

    Implementations can draw duration samples from a fixed statistical
    distribution (e.g., exponential, normal, log-normal) derived from
    historical data. Alternatively, they can use machine learning models
    (e.g., regression models) to predict the duration based on features of
    the activity, case, or assigned resource.
    """

    @abstractmethod
    def get_activity_duration(self, activity: "Activity", resource: "Resource" = None) -> float:
        """
        Gets the duration for a specific activity.

        :param activity: The activity to be performed.
        :param resource: The resource that will perform the activity.
        :return: The duration of the activity in simulation time units.
        """
        pass
