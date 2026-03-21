import pandas as pd

from environment.simulator.adapters.event_log_to_csv import export_event_log_to_csv
from initializer.implementations.DDPSInitializer import DDPSInitializer
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
    Runs a basic Discrete Event Simulation (DDPS) using the OPRA framework.
    This script initializes the simulator with data from a CSV event log,
    runs the simulation for a specified number of cases, and exports the
    resulting simulated event log to a new CSV file.

    To run this script:
    python src/simulate.py
    """
    log = pd.read_csv("data/logs/AcademicCredentials/AcademicCredentials_train.csv")

    initializer = DDPSInitializer()

    log_names = LogColumnNames(
        case_id="case_id",
        activity="activity",
        resource="resource",
        start_timestamp="start_time",
        end_timestamp="end_time",
    )

    start_timestamp = log[log_names.start_timestamp].min()
    time_unit = "seconds"

    setup: SimulationSetup = initializer.build(log, log_names, start_timestamp, time_unit)
    simulator = SimulatorEngine(setup)
    #print(setup.routing_policy)
    # get cases
    ncases = len(log[log_names.case_id].unique())
    print(f"Running basic DDPS simulation with {ncases} cases...")
    event_log = simulator.simulate(max_cases=ncases, convert_to_absolute_time=True)
    
    path = "data/simulated_logs/AcademicCredentials/AcademicCredentials_DDPS_v2.csv"
    export_event_log_to_csv(event_log, path)
    print(f"Basic DDPS simulation finished. Simulated event log exported to {path}")

if __name__ == "__main__":
    run_basic_simulation()


