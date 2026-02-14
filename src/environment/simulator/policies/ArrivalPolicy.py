from abc import ABC, abstractmethod


class ArrivalPolicy(ABC):
    """
    An abstract base class for arrival policies.
    An arrival policy determines when new cases enter the simulation. This is
    often referred to as the case arrival process.

    Implementations can model arrivals using statistical distributions (e.g.,
    a Poisson process for random arrivals), a fixed schedule from a dataset,
    or more complex patterns predicted by a machine learning model.
    """

    @abstractmethod
    def get_next_arrival_time(self, current_time: float) -> float:
        """
        Determines the time of the next case arrival.

        :param current_time: The current simulation time.
        :return: The absolute timestamp of the next arrival.
        """
        pass
