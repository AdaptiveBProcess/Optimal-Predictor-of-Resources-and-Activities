import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sympy as sp



original_log = pd.read_csv('data\logs\pizzeria_event_log.csv')

activities = original_log['activity'].unique()
resources = original_log['resource'].unique()


print(original_log.head())


calendar = np.zeros((7, 24))
calendar_resources = np.zeros((len(resources),7, 24))
calendar_activities = np.zeros((len(activities),7, 24))
total_events = 0

for _, row in original_log.iterrows():
    ts = pd.to_datetime(row['start_timestamp'])
    calendar[ts.weekday(), ts.hour] += 1
    activity_idx = np.where(activities == row['activity'])[0][0]
    resource_idx = np.where(resources == row['resource'])[0][0]
    calendar_activities[activity_idx, ts.weekday(), ts.hour] += 1
    calendar_resources[resource_idx, ts.weekday(), ts.hour] += 1

# normalize per weekday
calendar = calendar / calendar.sum(axis=1, keepdims=True)
calendar = calendar*100  # convert to percentage

calendar_activities = calendar_activities / calendar_activities.sum(axis=2, keepdims=True)
calendar_resources = calendar_resources / calendar_resources.sum(axis=2, keepdims=True)

calendar_activities = calendar_activities*100  # convert to percentage
calendar_resources = calendar_resources*100  # convert to percentage

# percentile thresholding
percentile = 50
threshold = np.percentile(calendar.flatten(),percentile)

print(f"{percentile}th percentile threshold:", threshold)    

binary_calendar = (calendar > threshold).astype(int)

plt.imshow(binary_calendar, cmap='gray_r', interpolation='nearest')
plt.colorbar(label='Activity Presence')
plt.xlabel('Hour of Day')
plt.ylabel('Day of Week')
plt.title(f'Activity Calendar ({percentile}th percentile, Threshold: {threshold:.2f})')
plt.show()

plt.imshow(calendar, cmap='coolwarm', interpolation='nearest')
plt.colorbar(label='Activity Presence')
plt.xlabel('Hour of Day')
plt.ylabel('Day of Week')
plt.title(f'Activity Calendar (probability %)')
plt.show()

for i, activity in enumerate(activities):
    plt.imshow(calendar_activities[i], cmap='coolwarm', interpolation='nearest')
    plt.colorbar(label='Activity Presence')
    plt.xlabel('Hour of Day')
    plt.ylabel('Day of Week')
    plt.title(f'Activity Calendar for {activity} (probability %)')
    plt.show()

for i, resource in enumerate(resources):
    plt.imshow(calendar_resources[i], cmap='coolwarm', interpolation='nearest')
    plt.colorbar(label='Resource Presence')
    plt.xlabel('Hour of Day')
    plt.ylabel('Day of Week')
    plt.title(f'Resource Calendar for {resource} (probability %)')
    plt.show()