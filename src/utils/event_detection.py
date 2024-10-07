import numpy as np
from src.utils.utilities import (
    detect_sudden_change_events, fill_false_between_trues,
    find_consecutive_true_blocks, check_cross_midline, calculate_derivative, normalize_series
)
from src.visualization.plotter import find_best_events

def detect_events(normalized_smoothed_pupil_size, smoothed_time_series, normalized_whisker_velocity, whisker_velocity_time, pupil_sampling_rate, whisker_sampling_rate, bsline_length, event_length):
    # Detect events
    pre_event_window = bsline_length * pupil_sampling_rate
    event_window = event_length * pupil_sampling_rate
    events, event_indices = detect_sudden_change_events(normalized_smoothed_pupil_size, 5, pre_event_window, event_window, 3, 1)
    mask = np.zeros(normalized_smoothed_pupil_size.shape)

    for idx, event_idx in enumerate(event_indices):
        zero_array = np.zeros(normalized_smoothed_pupil_size.shape)
        if event_idx == 1:
            zero_array[idx:idx + event_window + pre_event_window] = 1
        elif event_idx == 2:
            zero_array[idx:idx + event_window + pre_event_window] = -1
        mask += zero_array

    filled_mask = fill_false_between_trues(mask > 0.5, 10 * pupil_sampling_rate)
    consecutive_blocks = find_consecutive_true_blocks(filled_mask, pupil_sampling_rate)

    cross_midline_blocks = [block for block in consecutive_blocks if check_cross_midline(normalized_smoothed_pupil_size[block[0]:block[1]])]
    final_ranges = [np.max(normalized_smoothed_pupil_size[block[0]:block[1]]) - np.min(normalized_smoothed_pupil_size[block[0]:block[1]]) for block in cross_midline_blocks]
    final_blocks = [cross_midline_blocks[i] for i in range(len(cross_midline_blocks)) if final_ranges[i] > 0.5]

    final_events = []
    for block in final_blocks:
        best_event = find_best_events(block, normalized_smoothed_pupil_size, smoothed_time_series, whisker_velocity_time, normalized_whisker_velocity, print_result=False, plot_result=False)
        if best_event is not None:
            final_events.append(best_event)

    integral_data = []
    for event in final_events:
        time = smoothed_time_series[event]
        event_whisker_idx = (whisker_velocity_time < time).sum()
        baseline_whisker = normalized_whisker_velocity[event_whisker_idx - whisker_sampling_rate * bsline_length:event_whisker_idx]
        waking_up_whisker = normalized_whisker_velocity[event_whisker_idx:event_whisker_idx + whisker_sampling_rate * event_length]
        integral_baseline_whisker = np.trapz(baseline_whisker, whisker_velocity_time[event_whisker_idx - whisker_sampling_rate * bsline_length:event_whisker_idx]).mean()
        integral_waking_up_whisker = np.trapz(waking_up_whisker, whisker_velocity_time[event_whisker_idx:event_whisker_idx + whisker_sampling_rate * event_length]).mean()
        integral_ratio = integral_waking_up_whisker / integral_baseline_whisker
        integral_data.append(integral_ratio)

    waking_up_events = [event for idx, event in enumerate(final_events) if integral_data[idx] > 1.5]

    return waking_up_events