import pandas as pd
from log_distance_measures.config import EventLogIDs
from log_distance_measures.control_flow_log_distance import control_flow_log_distance
from log_distance_measures.absolute_event_distribution import absolute_event_distribution_distance
from log_distance_measures.case_arrival_distribution import case_arrival_distribution_distance
from log_distance_measures.config import AbsoluteTimestampType, discretize_to_hour
import matplotlib.pyplot as plt


def main():

    log1 = pd.read_csv("data/logs/PurchasingExample/PurchasingExample.csv")
    log2 = pd.read_csv("data/logs/simulated_event_log.csv")

    original_ids = EventLogIDs(
        case="caseid",
        activity="Activity_1",
        start_time="start_timestamp",
        end_time="end_timestamp",
        resource="Resource_1",
    )

    simulated_ids = EventLogIDs(
        case="case",
        activity="activity",
        start_time="start",
        end_time="end",
        resource="resource",
    )

    log1[original_ids.start_time] = pd.to_datetime(log1[original_ids.start_time], utc=True)
    log1[original_ids.end_time]   = pd.to_datetime(log1[original_ids.end_time], utc=True)

    log2[simulated_ids.start_time] = pd.to_datetime(log2[simulated_ids.start_time], utc=True)
    log2[simulated_ids.end_time]   = pd.to_datetime(log2[simulated_ids.end_time], utc=True)

    cfld_value = control_flow_log_distance(
        log1, original_ids,
        log2, simulated_ids
    )

    print("CFLD:", cfld_value)

    abs_dist = absolute_event_distribution_distance(
        log1, original_ids,
        log2, simulated_ids,
        discretize_type=AbsoluteTimestampType.BOTH,
        discretize_event=discretize_to_hour
    )

    print("Absolute Event Distribution Distance:", abs_dist)

    cad_dist = case_arrival_distribution_distance(
        log1, original_ids,
        log2, simulated_ids,
        discretize_event=discretize_to_hour
    )

    print("Case Arrival Distribution Distance:", cad_dist)

    log1["hour"] = log1[original_ids.start_time].dt.hour
    log2["hour"] = log2[simulated_ids.start_time].dt.hour

    plt.hist(log1["hour"], bins=24, alpha=0.5, label="Original")
    plt.hist(log2["hour"], bins=24, alpha=0.5, label="Simulated")

    plt.xlabel("Hour of Day")
    plt.ylabel("Count")
    plt.legend()
    plt.title("Hour-Based Event Distribution")
    plt.show()


if __name__ == "__main__":
    main()
