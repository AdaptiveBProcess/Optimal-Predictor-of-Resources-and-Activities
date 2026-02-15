import pandas as pd

from environment.simulator.adapters.event_log_to_csv import export_event_log_to_csv
from initializer.implementations.DESInitializer import DESInitializer
from environment.simulator.core.setup import SimulationSetup
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

all_activities = set(log[log_names.activity].unique())
timed_activities = set(setup.processing_time_policy.samples.keys())

print("Activities with NO processing times:")
for act in sorted(all_activities - timed_activities):
    print(" ", act)


print("Routing Policy :", setup.routing_policy)
print("Processing Time Policy:", setup.processing_time_policy)
print("Calendar Policy:", setup.calendar_policy)

simulator = SimulatorEngine(setup)
event_log = simulator.simulate(max_cases=200)

export_event_log_to_csv(event_log, "data/simulated_logs/PurchasingExample/PurchasingExample.csv")

