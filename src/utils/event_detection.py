import numpy as np
from src.utils.utilities import (
    detect_sudden_change_events, fill_false_between_trues,
    find_consecutive_true_blocks, check_cross_midline, calculate_derivative, normalize_series
)
from src.visualization.plotter import find_best_events

def detect_events(normalized_smoothed_pupil_size, smoothed_time_series, normalized_whisker_velocity, whisker_velocity_time, pupil_sampling_rate, whisker_sampling_rate, bsline_length, event_length, wakeup=False):
    # Detect events
    pre_event_window = bsline_length * pupil_sampling_rate
    event_window = event_length * pupil_sampling_rate
    events, event_indices = detect_sudden_change_events(normalized_smoothed_pupil_size, 5, pre_event_window, event_window, 2, 1)
    mask = np.zeros(normalized_smoothed_pupil_size.shape)

    for idx, event_idx in enumerate(event_indices):
        zero_array = np.zeros(normalized_smoothed_pupil_size.shape)
        if event_idx == 1:
            zero_array[idx:idx + event_window + pre_event_window] = 1
        elif event_idx == 2:
            zero_array[idx:idx + event_window + pre_event_window] = -1
        mask += zero_array

    if wakeup == False:
        filled_mask = fill_false_between_trues(mask < -0.5, 10 * pupil_sampling_rate)
    else:
        filled_mask = fill_false_between_trues(mask > 0.5, 10 * pupil_sampling_rate)
        print("Wake to sleep here")
        input("Wake to sleep event detection is not implemented yet. Please implement it in the detect_events function.")
    
    # Plotting the mask and normalized smoothed pupil size
    # Uncomment the following lines to enable plotting

    # import matplotlib
    # matplotlib.use('Agg')  # Use a non-interactive backend
    
    import matplotlib.pyplot as plt
    import os

    # Create the tmp directory if it doesn't exist
    os.makedirs('tmp', exist_ok=True)

    # # Plot the mask
    # plt.figure(figsize=(12, 6))  # Wider than tall
    # plt.plot(filled_mask, label='Event Mask')
    # plt.title('Event Detection Mask')
    # plt.xlabel('Time (samples)')
    # plt.ylabel('Mask Value')
    # plt.axhline(0, color='gray', lw=0.5, ls='--')
    # plt.legend()
    # plt.grid()

    # # Save the plot to the tmp folder
    # plt.savefig('tmp/event_detection_filled_mask_negative.png')
    # plt.close()

    # plt.figure(figsize=(12, 6))  # Wider than tall
    # plt.plot(filled_mask, label='Event Mask', alpha=0.5)
    # plt.plot(normalized_smoothed_pupil_size, label='Normalized Smoothed Pupil Size', alpha=0.75)
    # plt.title('Event Detection Mask and Normalized Smoothed Pupil Size')
    # plt.xlabel('Time (samples)')
    # plt.ylabel('Value')
    # plt.axhline(0, color='gray', lw=0.5, ls='--')
    # plt.legend()
    # plt.grid()

    # # Save the combined plot to the tmp folder
    # plt.savefig('tmp/event_detection_combined_plot.png')
    # plt.close()

    consecutive_blocks = find_consecutive_true_blocks(filled_mask, pupil_sampling_rate)

    cross_midline_blocks = [block for block in consecutive_blocks if check_cross_midline(normalized_smoothed_pupil_size[block[0]:block[1]])]

    final_ranges = [np.max(normalized_smoothed_pupil_size[block[0]:block[1]]) - np.min(normalized_smoothed_pupil_size[block[0]:block[1]]) for block in cross_midline_blocks]
    final_blocks = [cross_midline_blocks[i] for i in range(len(cross_midline_blocks)) if final_ranges[i] > 0.5]

    # # Plot the final blocks
    # plt.figure(figsize=(12, 6))  # Wider than tall

    # index_start = final_blocks[1][0]
    # index_end =final_blocks[1][1]
    # pupil_seg = normalized_smoothed_pupil_size[index_start:index_end]
    # time_seg = smoothed_time_series[index_start:index_end]

    # plt.plot(time_seg, pupil_seg, label='Normalized Smoothed Pupil Size', alpha=0.75)
    # plt.axhline(0, color='gray', lw=0.5, ls='--')
    # plt.axvline(time_seg[0], color='red', lw=0.5, ls='--', label='Event Start')
    # plt.axvline(time_seg[-1], color='blue', lw=0.5, ls='--', label='Event End')
    # plt.title('Event Detection Final Blocks')
    # plt.xlabel('Time (samples)')
    # plt.ylabel('Value')
    # plt.savefig('tmp/event_detection_final_blocks.png')
    # plt.close()
    # print(final_blocks)
    # return

    final_events = []
    for block in final_blocks:
        best_event = find_best_events(block, normalized_smoothed_pupil_size, smoothed_time_series, whisker_velocity_time, normalized_whisker_velocity, print_result=True, plot_result=True, wakeup=False)
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