from abc import ABC, abstractmethod
import pandas as pd

from environment.simulator.core.setup import SimulationSetup
from environment.simulator.core.log_names import LogColumnNames


class Initializer(ABC):

    @abstractmethod
    def build(
        self,
        log: pd.DataFrame,
        log_names: LogColumnNames
    ) -> SimulationSetup:
        pass
