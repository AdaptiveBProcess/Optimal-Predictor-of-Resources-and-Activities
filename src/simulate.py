import pandas as pd

from environment.simulator.adapters.event_log_to_csv import export_event_log_to_csv
from initializer.implementations.DESInitializer import DESInitializer
from environment.simulator.core.setup import SimulationSetup
from environment.simulator.core.log_names import LogColumnNames

from environment.simulator.core.engine import SimulatorEngine

import random
import numpy as np

# Set seeds for reproducibility
random.seed(42)
np.random.seed(42)

def run_basic_simulation():
    """
    Runs a basic Discrete Event Simulation (DES) using the OPRA framework.
    This script initializes the simulator with data from a CSV event log,
    runs the simulation for a specified number of cases, and exports the
    resulting simulated event log to a new CSV file.

    To run this script:
    python src/simulate.py
    """
    log = pd.read_csv("data/logs/PurchasingExample/PurchasingExample.csv")

    initializer = DESInitializer()

    log_names = LogColumnNames(
        case_id="caseid",
        activity="Activity_1",
        resource="Resource_1",
        start_timestamp="start_timestamp",
        end_timestamp="end_timestamp",
    )

    start_timestamp = log[log_names.start_timestamp].min()
    time_unit = "seconds"

    setup: SimulationSetup = initializer.build(log, log_names, start_timestamp, time_unit)
    simulator = SimulatorEngine(setup)
    event_log = simulator.simulate(max_cases=200)

    export_event_log_to_csv(event_log, "data/simulated_logs/PurchasingExample/PurchasingExample.csv")
    print(f"Basic DES simulation finished. Simulated event log exported to data/simulated_logs/PurchasingExample/PurchasingExample.csv")

if __name__ == "__main__":
    run_basic_simulation()


