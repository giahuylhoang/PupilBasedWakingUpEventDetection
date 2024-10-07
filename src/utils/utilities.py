# Utility functions that can be used across modules
import numpy as np
import pandas as pd
import os
from scipy.signal import find_peaks
from scipy.stats import sem


def find_folders_with_csv(root_folder):
    csv_folders = []
    for dirpath, dirnames, filenames in os.walk(root_folder):
        for file in filenames:
            if file.endswith('.csv'):
                print(f"Folder containing CSV file: {dirpath}")
                csv_folders.append(dirpath)
                break
    return csv_folders

def detect_and_interpolate_sudden_changes(df, threshold_quantile, window_size):
    df['change'] = df['pupil_size'].diff()
    threshold = df['change'].quantile(threshold_quantile)
    sudden_drops = df[df['change'] < threshold]
    time_points_to_interpolate = []

    for index in sudden_drops.index:
        time_point = df.loc[index, 'time']
        time_points_to_interpolate.extend(
            df[(df['time'] >= time_point - window_size) & (df['time'] <= time_point + window_size)].index.tolist()
        )

    time_points_to_interpolate = list(set(time_points_to_interpolate))
    filtered_df = df.drop(index=time_points_to_interpolate).reset_index(drop=True)
    interpolated_df = filtered_df.interpolate()
    interpolated_df = interpolated_df[['time', 'pupil_size']]
    return interpolated_df

def normalize_mean_std(df):
    cols = df.columns
    mean = df.iloc[:, 1].median()
    std = df.iloc[:, 1].std()
    df.iloc[:, 1] = (df.iloc[:, 1] - mean) / std
    df.columns = cols
    return df

def moving_average(data, window_size):
    return np.convolve(data, np.ones(window_size) / window_size, mode='valid') if window_size != 0 else data

def normalize_series(series, threshold=1):
    min_val = np.percentile(series, threshold)
    max_val = np.percentile(series, 100 - threshold)
    return (series - min_val) / (max_val - min_val)

def change_shape(df, time_dim):
    whisker_shape = df[0].shape[0]
    diff_shape = time_dim - whisker_shape
    value_appended = df[0].values[-1]
    appended = np.full(diff_shape, value_appended)
    df = pd.concat([df[0], pd.DataFrame(appended)])
    df.reset_index(inplace=True)
    df.drop(['index'], axis=1, inplace=True)
    return df

def detect_sudden_change_events(pupil_diameter, padding=None, pre_event_window=20, event_window=20, threshold=3, step=1):
    if padding is not None:
        pupil_diameter = np.concatenate([
            np.full(padding, pupil_diameter[0]),
            pupil_diameter,
            np.full(padding, pupil_diameter[-1])
        ])

    events = []
    events_indices = []

    for i in range(0, len(pupil_diameter) - pre_event_window - event_window, step):
        pre_event_mean = np.mean(pupil_diameter[i:i + pre_event_window])
        pre_event_std = np.std(pupil_diameter[i:i + pre_event_window])
        event_mean = np.mean(pupil_diameter[i + pre_event_window:i + pre_event_window + event_window])

        if event_mean - pre_event_mean > pre_event_std * threshold:
            events.append((i, 'increase'))
            events_indices.append(1)
        elif event_mean - pre_event_mean < -pre_event_std * threshold:
            events.append((i, 'decrease'))
            events_indices.append(2)
        else:
            events_indices.append(0)

    return events, events_indices

def fill_false_between_trues(mask, threshold):
    mask = mask.copy()
    n = len(mask)
    false_count = 0
    start_false_index = -1

    for i in range(n):
        if not mask[i]:
            if false_count == 0:
                start_false_index = i
            false_count += 1
        else:
            if false_count > 0 and false_count <= threshold:
                mask[start_false_index:i] = True
            false_count = 0

    if false_count > 0 and false_count <= threshold:
        mask[start_false_index:] = True

    return mask

def find_consecutive_true_blocks(mask, pupil_sampling_rate=40):
    blocks = []
    n = len(mask)
    in_block = False
    start_idx = 0

    for i in range(n):
        if mask[i]:
            if not in_block:
                start_idx = i
                in_block = True
        else:
            if in_block:
                blocks.append((start_idx, i - 1))
                in_block = False

    if in_block:
        blocks.append((min(start_idx, abs(start_idx - 5 * pupil_sampling_rate)), n - 1))

    return blocks

def check_cross_midline(segment, midline=0.5):
    above = segment > midline
    below = segment < midline
    crosses = np.any(np.diff(above.astype(int)) != 0) or np.any(np.diff(below.astype(int)) != 0)
    return crosses

def calculate_properties_possible_events(block, signal, time, step=0.25, baseline_window=5, event_window=15):
    start_idx, end_idx = block
    time_segment = time[start_idx:end_idx]
    time_step = np.mean(np.diff(time_segment))
    baseline_size = int(baseline_window / time_step)
    event_size = int(event_window / time_step)
    pupil_segment = signal[start_idx:end_idx]
    step_size = int(step / time_step)
    threshold = 0.5
    event_properties = []

    event_indices = np.arange(0, pupil_segment.shape[0], step_size)
    event_indices = event_indices[pupil_segment[event_indices] < threshold]

    for idx in event_indices:
        baseline_values = signal[start_idx + idx - baseline_size:start_idx + idx]
        if start_idx + idx + event_size > signal.shape[0]:
            continue
        event_values = signal[start_idx + idx:start_idx + idx + event_size]
        downward_values = signal[start_idx + idx:start_idx + idx + int(event_size / 3)]

        baseline_mean = np.mean(baseline_values)
        baseline_std = np.std(baseline_values)
        event_mean = np.mean(event_values)
        event_std = np.std(event_values)

        diffs = np.diff(downward_values)
        num_downward_movements = np.sum(diffs < 0)
        total_downward_magnitude = np.sum(diffs[diffs < 0])

        event_properties.append((
            idx, baseline_mean, baseline_std, event_mean, event_std,
            num_downward_movements, total_downward_magnitude
        ))

    return event_properties

def calculate_derivative(arr, times):
    arr = np.array(arr)
    times = np.array(times)
    sorted_indices = np.argsort(times)
    arr = arr[sorted_indices]
    times = times[sorted_indices]
    delta_arr = np.diff(arr)
    delta_times = np.diff(times)
    derivatives = delta_arr / delta_times
    return derivatives