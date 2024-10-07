import matplotlib.pyplot as plt
from ipywidgets import interact, FloatSlider
import numpy as np
from src.utils.utilities import (detect_sudden_change_events, calculate_properties_possible_events)

def plot_data(data):
    plt.plot(data)
    plt.show()

def plot_detected_events(window_size, pre_event_window, pupil_diameter, pupil_times, event_window, threshold, step, pupil_sampling_rate):
    window_size, pre_event_window, event_window, step = (int(window_size * pupil_sampling_rate),
                                                        int(pre_event_window * pupil_sampling_rate),
                                                        int(event_window * pupil_sampling_rate),
                                                        int(step * pupil_sampling_rate))
    padding = 5

    events, events_indices = detect_sudden_change_events(
        pupil_diameter,
        padding,
        pre_event_window,
        event_window,
        threshold,
        step
    )

    event_or_not = np.zeros(pupil_diameter.shape)
    for idx, event_idx in enumerate(events_indices):
        zero_arr = np.zeros(pupil_diameter.shape)
        if event_idx == 1:
            zero_arr[step*idx:step*idx+event_window+pre_event_window] = 1
        elif event_idx == 2:
            zero_arr[step*idx:step*idx+event_window+pre_event_window] = -1
        event_or_not += zero_arr

    interact(
        plot_detected_events,
        window_size=FloatSlider(value=1, min=0, max=60, step=1, description='Smoothing'),
        pre_event_window=FloatSlider(value=5, min=1, max=20, step=0.5, description='Baseline (s)'),
        event_window=FloatSlider(value=5, min=1, max=20, step=0.5, description='Event Window (s)'),
        threshold=FloatSlider(value=3, min=2, max=10, step=0.5, description='Threshold (SD)'),
        step=FloatSlider(value=0.5, min=0.5, max=10, step=0.25, description='Step Size (s)')
    )

def find_best_events(block, pupil_diameter, time, whisker_time, whisker_velocity, print_result=True, plot_result=True):
    pupil_segment = pupil_diameter[block[0]:block[1]]
    time_segment = time[block[0]:block[1]]

    analysis_results = calculate_properties_possible_events(block, pupil_diameter, time)
    filtered_results = [res for res in analysis_results if res[4] > res[2] * 3 and res[1] < 0.5 and res[3] - res[1] > 0.2]
    if print_result:
        for result in filtered_results:
            print(
                f"Start index: {result[0]}, Baseline mean: {result[1]:.4f}, Baseline std: {result[2]:.4f}, "
                f"Event mean: {result[3]:.4f}, Event std: {result[4]:.4f}, Total deflection: {result[5]:.4f}, "
                f"Total deflection value: {result[6]:.4f}"
            )

    if plot_result:
        plt.plot(time_segment, pupil_segment, label='Pupil Segmentation')
        for result in filtered_results:
            plt.axvline(time_segment[result[0]], color='red', linestyle='--')
        plt.xlabel('Time (seconds)')
        plt.ylabel('Pupil Segmentation')
        plt.title('Pupil Segmentation Over Time')
        plt.legend()

        start_index = (whisker_time < time_segment[0]).sum()
        end_index = (whisker_time < time_segment[-1]).sum()
        plt.plot(whisker_time[start_index:end_index], whisker_velocity[start_index:end_index])
        plt.ylim(0, 1)
        plt.xlabel('Time (seconds)')
        plt.ylabel('Normalized Whisker Velocity')
        plt.title('Whisker Velocity Over Time')
        plt.show()

    optimal_event = max(filtered_results, key=lambda x: x[6], default=None)

    if plot_result and optimal_event:
        print("\nBest Event:")
        print(
            f"Start index: {optimal_event[0]}, Baseline mean: {optimal_event[1]:.4f}, Baseline std: {optimal_event[2]:.4f}, "
            f"Event mean: {optimal_event[3]:.4f}, Event std: {optimal_event[4]:.4f}, Downward movements: {optimal_event[5]}, "
            f"Total downward magnitude: {optimal_event[6]:.4f}"
        )

        plt.plot(time_segment, pupil_segment, label='Pupil Segmentation')
        plt.axvline(time_segment[optimal_event[0]], color='red', linestyle='--', label='Best Event')
        plt.xlabel('Time (seconds)')
        plt.ylabel('Pupil Segmentation')
        plt.title('Pupil Segmentation Over Time')
        plt.legend()
        plt.show()

    return block[0] + optimal_event[0] if optimal_event else None