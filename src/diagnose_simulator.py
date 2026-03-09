"""
Diagnostic: Where does the 4x cycle time inflation come from?

Breaks down cycle time into:
  - Processing time (activity execution)
  - Waiting time (queue for resource)
  - Calendar gaps (waiting for working hours)
  - Inter-activity gaps (time between one activity ending and the next starting)

Also compares resource pool and activity distributions.

Run from project root:
    python src/diagnose_simulator.py
"""

import random
import numpy as np
import pandas as pd
from collections import defaultdict

from initializer.implementations.DESInitializer import DESInitializer
from environment.simulator.core.setup import SimulationSetup
from environment.simulator.core.log_names import LogColumnNames
from environment.simulator.core.engine import SimulatorEngine


def analyze_event_log(events, label="Log"):
    """Break down timing from a list of event dicts with numeric times."""
    cases = defaultdict(list)
    for e in events:
        cases[e["case"]].append(e)

    processing_times = []
    inter_activity_gaps = []
    case_cycle_times = []
    total_processing_per_case = []
    total_gap_per_case = []
    events_per_case = []
    activity_counts = defaultdict(int)
    resource_counts = defaultdict(int)
    activity_durations = defaultdict(list)

    for case_id, case_events in cases.items():
        case_events.sort(key=lambda x: x["start"])
        events_per_case.append(len(case_events))

        case_start = min(e["start"] for e in case_events)
        case_end = max(e["end"] for e in case_events)
        case_ct = case_end - case_start
        case_cycle_times.append(case_ct)

        total_proc = 0
        total_gap = 0

        for i, e in enumerate(case_events):
            duration = e["end"] - e["start"]
            processing_times.append(duration)
            total_proc += duration
            activity_counts[e["activity"]] += 1
            resource_counts[e["resource"]] += 1
            activity_durations[e["activity"]].append(duration)

            if i > 0:
                gap = e["start"] - case_events[i - 1]["end"]
                inter_activity_gaps.append(gap)
                total_gap += gap

        total_processing_per_case.append(total_proc)
        total_gap_per_case.append(total_gap)

    ct = np.array(case_cycle_times)
    proc = np.array(processing_times)
    gaps = np.array(inter_activity_gaps) if inter_activity_gaps else np.array([0])
    total_proc_case = np.array(total_processing_per_case)
    total_gap_case = np.array(total_gap_per_case)

    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")
    print(f"  Cases: {len(cases)}")
    print(f"  Total events: {sum(events_per_case)}")
    print(f"  Avg events/case: {np.mean(events_per_case):.1f}")

    print(f"\n  CYCLE TIME (end-to-end per case):")
    print(f"    Mean:   {np.mean(ct):>10.0f}s  ({np.mean(ct)/3600:>6.1f}h)")
    print(f"    Median: {np.median(ct):>10.0f}s  ({np.median(ct)/3600:>6.1f}h)")
    print(f"    p25:    {np.percentile(ct,25):>10.0f}s  ({np.percentile(ct,25)/3600:>6.1f}h)")
    print(f"    p75:    {np.percentile(ct,75):>10.0f}s  ({np.percentile(ct,75)/3600:>6.1f}h)")
    print(f"    p95:    {np.percentile(ct,95):>10.0f}s  ({np.percentile(ct,95)/3600:>6.1f}h)")

    print(f"\n  PROCESSING TIME (per activity execution):")
    print(f"    Mean:   {np.mean(proc):>10.0f}s  ({np.mean(proc)/3600:>6.1f}h)")
    print(f"    Median: {np.median(proc):>10.0f}s  ({np.median(proc)/3600:>6.1f}h)")
    print(f"    Min:    {np.min(proc):>10.0f}s")
    print(f"    Max:    {np.max(proc):>10.0f}s  ({np.max(proc)/3600:>6.1f}h)")

    print(f"\n  INTER-ACTIVITY GAP (waiting + calendar between activities):")
    print(f"    Mean:   {np.mean(gaps):>10.0f}s  ({np.mean(gaps)/3600:>6.1f}h)")
    print(f"    Median: {np.median(gaps):>10.0f}s  ({np.median(gaps)/3600:>6.1f}h)")
    print(f"    Max:    {np.max(gaps):>10.0f}s  ({np.max(gaps)/3600:>6.1f}h)")
    if len(gaps) > 0:
        large_gaps = np.sum(gaps > 3600)
        print(f"    Gaps > 1h: {large_gaps} ({large_gaps/len(gaps)*100:.1f}%)")
        large_gaps_8h = np.sum(gaps > 28800)
        print(f"    Gaps > 8h: {large_gaps_8h} ({large_gaps_8h/len(gaps)*100:.1f}%)")

    print(f"\n  TIME BREAKDOWN PER CASE (avg):")
    avg_proc = np.mean(total_proc_case)
    avg_gap = np.mean(total_gap_case)
    avg_ct = np.mean(ct)
    print(f"    Processing: {avg_proc:>10.0f}s  ({avg_proc/3600:>6.1f}h)  ({avg_proc/avg_ct*100:>5.1f}% of CT)")
    print(f"    Gaps:       {avg_gap:>10.0f}s  ({avg_gap/3600:>6.1f}h)  ({avg_gap/avg_ct*100:>5.1f}% of CT)")
    other = avg_ct - avg_proc - avg_gap
    if other > 0:
        print(f"    Other:      {other:>10.0f}s  ({other/3600:>6.1f}h)  ({other/avg_ct*100:>5.1f}% of CT)")

    print(f"\n  ACTIVITY DURATIONS (mean per activity):")
    for act in sorted(activity_durations.keys(), key=lambda a: str(a)):
        durs = activity_durations[act]
        print(f"    {str(act):>30s}: mean={np.mean(durs):>8.0f}s ({np.mean(durs)/3600:.1f}h), "
              f"median={np.median(durs):>8.0f}s, count={len(durs)}")

    print(f"\n  TOP RESOURCES BY USAGE:")
    sorted_res = sorted(resource_counts.items(), key=lambda x: -x[1])[:10]
    for rid, count in sorted_res:
        print(f"    {str(rid):>30s}: {count} events")

    return {
        "cycle_times": ct,
        "processing_times": proc,
        "gaps": gaps,
        "activity_durations": activity_durations,
    }


def analyze_original_log(log_df, log_names):
    """Analyze the original CSV log."""
    events = []
    for _, row in log_df.iterrows():
        st = pd.to_datetime(row[log_names.start_timestamp])
        et = pd.to_datetime(row[log_names.end_timestamp])
        events.append({
            "case": row[log_names.case_id],
            "activity": row[log_names.activity],
            "resource": row[log_names.resource],
            "start": st.timestamp(),
            "end": et.timestamp(),
        })
    return analyze_event_log(events, label="ORIGINAL LOG")


def analyze_calendar(setup):
    """Inspect the calendar policy."""
    print(f"\n{'='*60}")
    print(f"  CALENDAR POLICY")
    print(f"{'='*60}")
    cal = setup.calendar_policy
    if hasattr(cal, 'weekly_schedule'):
        schedule = cal.weekly_schedule
        days = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
        for i, day in enumerate(days):
            if i < len(schedule):
                hours = schedule[i]
                active = sum(hours) if isinstance(hours, (list, np.ndarray)) else "unknown"
                print(f"    {day}: {active} active hours")
    elif hasattr(cal, 'schedule'):
        print(f"    Schedule type: {type(cal.schedule)}")
        print(f"    Schedule: {cal.schedule}")
    else:
        print(f"    Calendar type: {type(cal).__name__}")
        print(f"    Attributes: {[a for a in dir(cal) if not a.startswith('_')]}")


def analyze_resources(setup):
    """Inspect resource pool."""
    print(f"\n{'='*60}")
    print(f"  RESOURCE POOL")
    print(f"{'='*60}")
    resources = setup.resource_policy.resources if hasattr(setup.resource_policy, 'resources') else []
    print(f"  Total resources: {len(resources)}")
    
    total_capacity = sum(r.capacity for r in resources)
    print(f"  Total capacity: {total_capacity}")
    
    skills_count = defaultdict(int)
    for r in resources:
        if hasattr(r, 'skills'):
            for s in r.skills:
                skills_count[s] += 1
        if hasattr(r, 'capacity') and r.capacity > 1:
            print(f"    Resource {r.id}: capacity={r.capacity}")
    
    print(f"\n  Resources per activity (skill coverage):")
    for act in sorted(skills_count.keys(), key=lambda a: str(a)):
        print(f"    {str(act):>30s}: {skills_count[act]} resources")


def analyze_arrival_policy(setup, n_samples=100):
    """Sample from the arrival policy to check inter-arrival times."""
    print(f"\n{'='*60}")
    print(f"  ARRIVAL POLICY")
    print(f"{'='*60}")
    arrivals = []
    for _ in range(n_samples):
        t = setup.arrival_policy.get_next_arrival_time()
        arrivals.append(t)
    arrivals = np.array(arrivals)
    print(f"  Type: {type(setup.arrival_policy).__name__}")
    print(f"  Sample inter-arrival times ({n_samples} samples):")
    print(f"    Mean:   {np.mean(arrivals):>8.0f}s ({np.mean(arrivals)/3600:.1f}h)")
    print(f"    Median: {np.median(arrivals):>8.0f}s ({np.median(arrivals)/3600:.1f}h)")
    print(f"    Min:    {np.min(arrivals):>8.0f}s")
    print(f"    Max:    {np.max(arrivals):>8.0f}s ({np.max(arrivals)/3600:.1f}h)")


def main():
    # --- Load original log ---
    log = pd.read_csv("data/logs/PurchasingExample/PurchasingExample.csv")
    log_names = LogColumnNames(
        case_id="case_id", activity="activity", resource="resource",
        start_timestamp="start_time", end_timestamp="end_time",
    )

    # --- Build setup ---
    initializer = DESInitializer()
    start_timestamp = log[log_names.start_timestamp].min()
    setup = initializer.build(log, log_names, start_timestamp, "seconds")

    # --- Analyze original log ---
    orig_stats = analyze_original_log(log, log_names)

    # --- Analyze simulator components ---
    analyze_calendar(setup)
    analyze_resources(setup)
    analyze_arrival_policy(setup)

    # --- Run normal simulation and analyze ---
    random.seed(42)
    np.random.seed(42)
    simulator = SimulatorEngine(setup)
    event_log = simulator.simulate(max_cases=200)
    sim_stats = analyze_event_log(event_log, label="SIMULATED LOG (DES, 200 cases)")

    # --- Compare activity durations ---
    print(f"\n{'='*60}")
    print(f"  ACTIVITY DURATION COMPARISON (Original vs Simulated)")
    print(f"{'='*60}")
    all_acts = set(list(orig_stats["activity_durations"].keys()) +
                   list(sim_stats["activity_durations"].keys()))
    for act in sorted(all_acts, key=lambda a: str(a)):
        orig_durs = orig_stats["activity_durations"].get(act, [])
        sim_durs = sim_stats["activity_durations"].get(act, [])
        orig_mean = np.mean(orig_durs) if orig_durs else 0
        sim_mean = np.mean(sim_durs) if sim_durs else 0
        ratio = sim_mean / orig_mean if orig_mean > 0 else float('inf')
        flag = " *** INFLATED" if ratio > 2.0 else ""
        print(f"  {str(act):>30s}:  orig={orig_mean:>8.0f}s  sim={sim_mean:>8.0f}s  ratio={ratio:>5.1f}x{flag}")


if __name__ == "__main__":
    main()