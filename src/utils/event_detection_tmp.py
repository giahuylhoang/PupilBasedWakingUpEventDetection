import numpy as np
from src.utils.utilities import (
    detect_sudden_change_events, fill_false_between_trues,
    find_consecutive_true_blocks, check_cross_midline, calculate_derivative, normalize_series
)
from src.visualization.plotter import find_best_events

def detect_events(normalized_smoothed_pupil_size, smoothed_time_series, whisker_time, whisker_angle, pupil_sampling_rate, whisker_sampling_rate, bsline_length, event_length):
    events = detect_pupil_events(normalized_smoothed_pupil_size, pupil_sampling_rate, bsline_length, event_length)
    cross_midline_blocks = filter_cross_midline_blocks(events, normalized_smoothed_pupil_size)
    final_blocks = filter_final_blocks(cross_midline_blocks, normalized_smoothed_pupil_size)
    
    normalized_whisker_velocity = calculate_normalized_whisker_velocity(whisker_angle, whisker_time)
    
    final_events = find_best_events_for_blocks(final_blocks, normalized_smoothed_pupil_size, smoothed_time_series, whisker_time, normalized_whisker_velocity)
    waking_up_events = filter_waking_up_events(final_events, normalized_whisker_velocity, smoothed_time_series, whisker_time, whisker_sampling_rate, bsline_length, event_length)
    
    return waking_up_events

def detect_pupil_events(normalized_smoothed_pupil_size, pupil_sampling_rate, bsline_length, event_length):
    pre_event_window = bsline_length * pupil_sampling_rate
    event_window = event_length * pupil_sampling_rate
    events, event_indices = detect_sudden_change_events(normalized_smoothed_pupil_size, 5, pre_event_window, event_window, 3, 1)
    
    mask = create_event_mask(event_indices, normalized_smoothed_pupil_size.shape, event_window, pre_event_window)
    filled_mask = fill_false_between_trues(mask > 0.5, 10 * pupil_sampling_rate)
    consecutive_blocks = find_consecutive_true_blocks(filled_mask, pupil_sampling_rate)
    
    return consecutive_blocks

def create_event_mask(event_indices, shape, event_window, pre_event_window):
    mask = np.zeros(shape)
    for idx, event_idx in enumerate(event_indices):
        zero_array = np.zeros(shape)
        if event_idx == 1:
            zero_array[idx:idx + event_window + pre_event_window] = 1
        elif event_idx == 2:
            zero_array[idx:idx + event_window + pre_event_window] = -1
        mask += zero_array
    return mask

def filter_cross_midline_blocks(blocks, normalized_smoothed_pupil_size):
    return [block for block in blocks if check_cross_midline(normalized_smoothed_pupil_size[block[0]:block[1]])]

def filter_final_blocks(cross_midline_blocks, normalized_smoothed_pupil_size):
    final_ranges = [np.max(normalized_smoothed_pupil_size[block[0]:block[1]]) - np.min(normalized_smoothed_pupil_size[block[0]:block[1]]) for block in cross_midline_blocks]
    return [cross_midline_blocks[i] for i in range(len(cross_midline_blocks)) if final_ranges[i] > 0.5]

def calculate_normalized_whisker_velocity(whisker_angle, whisker_time):
    return normalize_series(np.power(calculate_derivative(whisker_angle, whisker_time), 2))

def find_best_events_for_blocks(blocks, normalized_smoothed_pupil_size, smoothed_time_series, whisker_time, normalized_whisker_velocity):
    final_events = []
    for block in blocks:
        best_event = find_best_events(block, normalized_smoothed_pupil_size, smoothed_time_series, whisker_time, normalized_whisker_velocity, print_result=False, plot_result=False)
        if best_event is not None:
            final_events.append(best_event)
    return final_events

def filter_waking_up_events(events, normalized_whisker_velocity, smoothed_time_series, whisker_time, whisker_sampling_rate, bsline_length, event_length):
    integral_data = calculate_integral_data(events, normalized_whisker_velocity, smoothed_time_series, whisker_time, whisker_sampling_rate, bsline_length, event_length)
    return [event for idx, event in enumerate(events) if integral_data[idx] > 1.5]

def calculate_integral_data(events, normalized_whisker_velocity, smoothed_time_series, whisker_time, whisker_sampling_rate, bsline_length, event_length):
    integral_data = []
    for event in events:
        time = smoothed_time_series[event]
        event_whisker_idx = (whisker_time < time).sum()
        baseline_whisker = normalized_whisker_velocity[event_whisker_idx - whisker_sampling_rate * bsline_length:event_whisker_idx]
        waking_up_whisker = normalized_whisker_velocity[event_whisker_idx:event_whisker_idx + whisker_sampling_rate * event_length]
        integral_baseline_whisker = np.trapz(baseline_whisker, whisker_time[event_whisker_idx - whisker_sampling_rate * bsline_length:event_whisker_idx]).mean()
        integral_waking_up_whisker = np.trapz(waking_up_whisker, whisker_time[event_whisker_idx:event_whisker_idx + whisker_sampling_rate * event_length]).mean()
        integral_ratio = integral_waking_up_whisker / integral_baseline_whisker
        integral_data.append(integral_ratio)
    return integral_data