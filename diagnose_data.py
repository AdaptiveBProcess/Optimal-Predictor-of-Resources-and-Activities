#!/usr/bin/env python3
"""Diagnose PurchasingExample dataset issues."""

import pandas as pd
from collections import defaultdict, Counter

log = pd.read_csv("data/logs/PurchasingExample/PurchasingExample.csv")

print("=== DATA SHAPE ===")
print(f"Rows: {len(log)}")
print(f"Columns: {list(log.columns)}")

print("\n=== UNIQUE VALUES ===")
print(f"Unique cases: {log['case_id'].nunique()}")
print(f"Unique activities: {log['activity'].nunique()}")
print(f"Activities: {sorted(log['activity'].unique())}")
print(f"Unique resources: {log['resource'].nunique()}")
print(f"Resources (first 10): {sorted(log['resource'].unique())[:10]}")

print("\n=== CASE STRUCTURE ===")
case_lengths = log.groupby('case_id').size()
print(f"Activities per case:")
print(f"  Min: {case_lengths.min()}")
print(f"  Max: {case_lengths.max()}")
print(f"  Mean: {case_lengths.mean():.1f}")
print(f"  Median: {case_lengths.median():.0f}")

# Check for cases with 1 activity
single_activity_cases = (case_lengths == 1).sum()
print(f"  Cases with only 1 activity: {single_activity_cases}")

print("\n=== ROUTING ===")
# Build routing transitions
transitions = defaultdict(Counter)
for case_id, group in log.groupby('case_id'):
    acts = group.sort_values('start_time')['activity'].tolist()
    prev = None
    for act in acts:
        transitions[prev][act] += 1
        prev = act
    transitions[prev][None] += 1

print(f"Unique from-activities (including START): {len(transitions)}")
print(f"All transitions:")
for from_act in sorted([str(a) for a in list(transitions.keys())[:5]]):
    key = None if from_act == 'None' else from_act
    if key in transitions:
        to_acts = transitions[key]
        print(f"  {str(from_act):30s} -> {dict(to_acts)}")

print("\n=== TIME ISSUES ===")
log['start'] = pd.to_datetime(log['start_time'])
log['end'] = pd.to_datetime(log['end_time'])
log['duration'] = (log['end'] - log['start']).dt.total_seconds()

print(f"Duration stats (seconds):")
print(f"  Min: {log['duration'].min():.0f}s")
print(f"  Max: {log['duration'].max():.0f}s")
print(f"  Mean: {log['duration'].mean():.0f}s")
print(f"  Median: {log['duration'].median():.0f}s")
print(f"  Zero or negative duration: {(log['duration'] <= 0).sum()} rows")

if (log['duration'] <= 0).sum() > 0:
    print("\n  Examples of zero/negative durations:")
    print(log[log['duration'] <= 0][['case_id', 'activity', 'start_time', 'end_time', 'duration']].head())

print("\n=== CASE TIMING ===")
case_timings = []
for case_id, group in log.groupby('case_id'):
    g_start = pd.to_datetime(group['start_time']).min()
    g_end = pd.to_datetime(group['end_time']).max()
    case_timings.append((g_end - g_start).total_seconds())

import numpy as np
ct = np.array(case_timings)
print(f"Case cycle times (seconds):")
print(f"  Mean: {np.mean(ct):.0f}s ({np.mean(ct)/3600:.1f}h)")
print(f"  Median: {np.median(ct):.0f}s")
print(f"  p90: {np.percentile(ct, 90):.0f}s")
