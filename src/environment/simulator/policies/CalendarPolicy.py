from abc import ABC, abstractmethod

class CalendarPolicy(ABC):

    @abstractmethod
    def is_working_time(self, t: float) -> bool:
        pass

    @abstractmethod
    def next_working_time(self, t: float) -> float:
        pass
