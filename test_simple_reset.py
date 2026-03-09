#!/usr/bin/env python3
"""Quick diagnostic: test if env.reset() hangs."""

import random
import numpy as np
import pandas as pd
import sys

from initializer.implementations.DESInitializer import DESInitializer
from environment.core.env import BusinessProcessEnvironment
from environment.core.mask import NucleusMaskFunction
from environment.simulator.core.log_names import LogColumnNames
from environment.simulator.core.engine import SimulatorEngine

def main():
    random.seed(42)
    np.random.seed(42)

    # Load log
    log = pd.read_csv("data/logs/PurchasingExample/PurchasingExample.csv")
    log_names = LogColumnNames(
        case_id="case_id",
        activity="activity",
        resource="resource",
        start_timestamp="start_time",
        end_timestamp="end_time",
    )

    # Build setup
    print("Building setup...")
    initializer = DESInitializer()
    start_timestamp = log[log_names.start_timestamp].min()
    setup = initializer.build(log, log_names, start_timestamp, "seconds")
    print(f"  Activities: {setup.activities}")
    print(f"  Resources: {len(setup.resources)} resources")
    print(f"  Arrival policy: {type(setup.arrival_policy).__name__}")

    # Create engine
    print("\nCreating engine...")
    simulator = SimulatorEngine(setup)
    print(f"  Engine created: {simulator}")
    print(f"  All activities: {simulator.all_activities}")
    print(f"  All resources: {len(simulator.all_resources)} resources")

    # Create environment
    print("\nCreating environment...")
    env = BusinessProcessEnvironment(
        simulator,
        sla_threshold=7074216.0,
        max_cases=5,
        activity_mask_function=NucleusMaskFunction(k=2, p=0.5),
    )
    print("  Environment created")

    # Try to reset (this is where it hangs)
    print("\nCalling env.reset()...")
    sys.stdout.flush()
    try:
        obs, info = env.reset()
        print(f"  Reset succeeded! obs shape: {obs.shape}")
        print(f"  Pending decisions: {len(simulator.pending_decisions)}")
        if simulator.pending_decisions:
            case = simulator.get_case_needing_decision()
            print(f"  Case needing decision: {case.case_id if case else 'None'}")
    except KeyboardInterrupt:
        print("  INTERRUPTED (hung)")
        sys.exit(1)
    except Exception as e:
        print(f"  ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
