import pandas as pd
import numpy as np

from environment.simulator.adapters.event_log_to_csv import export_event_log_to_csv
from initializer.implementations.DESInitializer import DESInitializer
from environment.simulator.core.setup import SimulationSetup
from environment.environment import BusinessProcessEnvironment
from environment.simulator.core.log_names import LogColumnNames

from environment.simulator.core.engine import SimulatorEngine

import random
random.seed(42)


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

# Group by case_id and calculate cycle time for each case
cycle_time = []
for case_id, group in log.groupby(log_names.case_id):
    start_time = group[log_names.start_timestamp].min()
    end_time = group[log_names.end_timestamp].max()
    cycle_time.append((end_time - start_time).total_seconds())
# calcualte p75
p75_cycle_time = np.percentile(cycle_time, 75)
print("p75 cycle time", p75_cycle_time)

environment = BusinessProcessEnvironment(simulator, sla_threshold=p75_cycle_time, max_cases=10)

event_log = simulator.simulate(max_cases=10)





export_event_log_to_csv(event_log, "data/simulated_logs/PurchasingExample/PurchasingExample.csv")

