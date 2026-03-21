import numpy as np
import random
from datetime import datetime, timedelta
from environment.simulator.policies.ArrivalPolicy import ArrivalPolicy

class WeeklyArrivalPolicy(ArrivalPolicy):
    """
    Non-homogeneous Poisson arrival process with a weekly rate profile.
    
    The rate matrix (7 x 24) stores the mean number of arrivals per hour
    for each (weekday, hour) slot, estimated directly from the log.
    At simulation time, the current slot's rate is used to sample an
    exponential inter-arrival time — consistent with a Poisson process.
    """

    def __init__(self, rate_matrix: np.ndarray, start_timestamp: str, time_unit: str = "seconds", ):
        # rate_matrix[weekday, hour] = mean arrivals per hour in that slot
        self.rate_matrix = rate_matrix
        self.time_unit = time_unit
        self.start_ts = datetime.fromisoformat(start_timestamp).timestamp()

    def get_next_arrival_time(self, current_time: float) -> float:
        # Determine current position in the weekly cycle
        absolute_time = self.start_ts + current_time
        seconds_per_unit = {"seconds": 1, "minutes": 60, "hours": 3600}[self.time_unit]
        total_seconds = current_time * seconds_per_unit
        dt = datetime.fromtimestamp(absolute_time)
        weekday = dt.weekday()
        hour = dt.hour

        rate = self.rate_matrix[weekday, hour]  # arrivals per hour

        if rate <= 0:
            # Outside working hours — fall back to mean rate to avoid deadlock
            nonzero = self.rate_matrix[self.rate_matrix > 0]
            rate = float(nonzero.mean()) if nonzero.size > 0 else 1.0

        # Convert rate to time-unit scale and sample exponential inter-arrival
        rate_per_unit = rate / (3600 / seconds_per_unit)
        return float(np.random.exponential(1.0 / rate_per_unit))

    def __str__(self) -> str:
        days = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
        bars = " ▁▂▃▄▅▆▇█"
        max_rate = self.rate_matrix.max()
        lines = ["WeeklyArrivalPolicy"]

        for d in range(7):
            row = []
            for h in range(24):
                rate = self.rate_matrix[d, h]
                if max_rate > 0:
                    idx = int(round(rate / max_rate * (len(bars) - 1)))
                else:
                    idx = 0
                row.append(bars[idx])
            lines.append(f"{days[d]}: " + " ".join(row))

        return "\n".join(lines)