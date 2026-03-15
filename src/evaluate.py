import pandas as pd
from log_distance_measures.config import EventLogIDs
from log_distance_measures.control_flow_log_distance import control_flow_log_distance
from log_distance_measures.absolute_event_distribution import absolute_event_distribution_distance
from log_distance_measures.case_arrival_distribution import case_arrival_distribution_distance
from log_distance_measures.config import AbsoluteTimestampType, discretize_to_hour
from log_distance_measures.cycle_time_distribution import cycle_time_distribution_distance
import matplotlib.pyplot as plt


def main():
    log_name = "AcademicCredentials"
    log1 = pd.read_csv(f"data/logs/{log_name}/{log_name}_train.csv")
    log2 = pd.read_csv(f"data/simulated_logs/{log_name}/{log_name}_DFG.csv")
    log3 = pd.read_csv(f"data/simulated_logs/{log_name}/{log_name}_DDPS.csv")
    tag = "DDPS"
    original_ids = EventLogIDs(
        case="case_id",
        activity="activity",
        start_time="start_time",
        end_time="end_time",
        resource="resource",
    )    

    simulated_ids = EventLogIDs(
        case="case",
        activity="activity",
        start_time="start",
        end_time="end",
        resource="resource",
    )

    log1[original_ids.start_time] = pd.to_datetime(log1[original_ids.start_time], utc=True, format="mixed")
    log1[original_ids.end_time]   = pd.to_datetime(log1[original_ids.end_time], utc=True, format="mixed")

    log2[simulated_ids.start_time] = pd.to_datetime(log2[simulated_ids.start_time], utc=True, format="mixed")
    log2[simulated_ids.end_time]   = pd.to_datetime(log2[simulated_ids.end_time], utc=True, format="mixed")

    log3[simulated_ids.start_time] = pd.to_datetime(log3[simulated_ids.start_time], utc=True, format="mixed")
    log3[simulated_ids.end_time]   = pd.to_datetime(log3[simulated_ids.end_time], utc=True, format="mixed")

    cfld_value = control_flow_log_distance(
        log1, original_ids,
        log2, simulated_ids
    )

    cfld_value_rl = control_flow_log_distance(
        log1, original_ids,
        log3, simulated_ids
    )

    print("CFLD (Normal):", cfld_value)
    print(f"CFLD ({tag}):", cfld_value_rl)

    abs_dist = absolute_event_distribution_distance(
        log1, original_ids,
        log2, simulated_ids,
        discretize_type=AbsoluteTimestampType.BOTH,
        discretize_event=discretize_to_hour
    )

    abs_dist_rl = absolute_event_distribution_distance(
        log1, original_ids,
        log3, simulated_ids,
        discretize_type=AbsoluteTimestampType.BOTH,
        discretize_event=discretize_to_hour
    )
    

    print("Absolute Event Distribution Distance:", abs_dist)
    print(f"Absolute Event Distribution Distance ({tag}):", abs_dist_rl)

    cad_dist = case_arrival_distribution_distance(
        log1, original_ids,
        log2, simulated_ids,
        discretize_event=discretize_to_hour
    )

    cad_dist_rl = case_arrival_distribution_distance(
        log1, original_ids,
        log3, simulated_ids,
        discretize_event=discretize_to_hour
    )

    print("Case Arrival Distribution Distance:", cad_dist)
    print(f"Case Arrival Distribution Distance ({tag}):", cad_dist_rl)


    cad_dist = cycle_time_distribution_distance(
        log1, original_ids,
        log2, simulated_ids,
        bin_size=pd.Timedelta(hours=1)
    )

    cad_dist_rl = cycle_time_distribution_distance(
        log1, original_ids,
        log3, simulated_ids,
        bin_size=pd.Timedelta(hours=1)
    )

    print("Cycle Time Distribution Distance:", cad_dist)
    print(f"Cycle Time Distribution Distance ({tag}):", cad_dist_rl)

    # Compare end-to-end Cycle Time Distributions (per case)
    def case_cycle_times(log, ids):
        grouped = log.groupby(ids.case)
        return (grouped[ids.end_time].max() - grouped[ids.start_time].min()).dt.total_seconds() / 3600

    ct1 = case_cycle_times(log1, original_ids)
    ct2 = case_cycle_times(log2, simulated_ids)
    ct3 = case_cycle_times(log3, simulated_ids)

    plt.figure(figsize=(10, 5))
    plt.hist(ct1, bins=30, alpha=0.5, label="Original")
    plt.hist(ct2, bins=30, alpha=0.5, label="Simulated DFG")
    plt.hist(ct3, bins=30, alpha=0.5, label=f"Simulated {tag}")
    plt.xlabel("End-to-End Cycle Time per Case (hours)")
    plt.ylabel("Number of Cases")
    plt.legend()
    plt.title("End-to-End Cycle Time Distribution per Case")
    plt.tight_layout()
    plt.show()

    # Processing time & waiting time distributions per activity
    def add_times(log, ids):
        log = log.sort_values([ids.case, ids.start_time]).copy()
        log["processing_time"] = (log[ids.end_time] - log[ids.start_time]).dt.total_seconds() / 3600
        log["prev_end"] = log.groupby(ids.case)[ids.end_time].shift(1)
        log["waiting_time"] = ((log[ids.start_time] - log["prev_end"]).dt.total_seconds() / 3600).clip(lower=0)
        return log

    log1 = add_times(log1, original_ids)
    log2 = add_times(log2, simulated_ids)
    log3 = add_times(log3, simulated_ids)

    def boxplot_per_activity(logs_labels, value_col, title):
        n = len(logs_labels)
        fig, axes = plt.subplots(n, 1, figsize=(10, n * 4))
        if n == 1:
            axes = [axes]
        for ax, (log, ids, label) in zip(axes, logs_labels):
            activities = log.groupby(ids.activity)[value_col].median().sort_values(ascending=False).index.tolist()
            grouped = [log.loc[log[ids.activity] == a, value_col].dropna().values for a in activities]
            ax.boxplot(grouped, vert=False, tick_labels=activities,
                       flierprops=dict(marker=".", markersize=2, alpha=0.4),
                       patch_artist=True, boxprops=dict(facecolor="steelblue", alpha=0.6))
            ax.set_xlabel("Hours")
            ax.set_title(label)
        fig.suptitle(title, fontsize=14, y=1.01)
        plt.tight_layout()
        plt.show()

    logs_labels = [
        (log1, original_ids, "Original"),
        (log2, simulated_ids, "Simulated DFG"),
        (log3, simulated_ids, f"Simulated {tag}"),
    ]
    boxplot_per_activity(logs_labels, "processing_time", "Processing Time Distribution per Activity")
    boxplot_per_activity(logs_labels, "waiting_time", "Waiting Time Distribution per Activity")

    # log1["hour"] = log1[original_ids.start_time].dt.hour
    # log2["hour"] = log2[simulated_ids.start_time].dt.hour
    # log3["hour"] = log3[simulated_ids.start_time].dt.hour
    # plt.hist(log1["hour"], bins=24, alpha=0.5, label="Original")
    # plt.hist(log2["hour"], bins=24, alpha=0.5, label="Simulated")
    # plt.hist(log3["hour"], bins=24, alpha=0.5, label=f"Simulated {tag}")

    # plt.xlabel("Hour of Day")
    # plt.ylabel("Count")
    # plt.legend()
    # plt.title("Hour-Based Event Distribution")
    # plt.show()


if __name__ == "__main__":
    main()
