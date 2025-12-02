import numpy as np
import pandas as pd
from scipy.stats import sem

import matplotlib.pyplot as plt
from pathlib import Path

def process_calcium_data(calcium, smoothed_times, final_events, save_path, calcium_sampling_rate, normalize=True, save_files=True, event_length=15, bsline_length=5, interactive_plots=False):
    calcium_data = calcium['calcium'].values
    calcium_time = calcium['time'].values
    windows = []

    for event in final_events:
        time = smoothed_times[event]
        event_calcium_idx = (calcium_time < time).sum()
        window = calcium_data[event_calcium_idx - calcium_sampling_rate*bsline_length:event_calcium_idx + calcium_sampling_rate*event_length]
        baseline = calcium_data[event_calcium_idx - calcium_sampling_rate*bsline_length:event_calcium_idx]
        if normalize:
            window = 100 * (window - np.mean(baseline)) / np.mean(baseline)
        windows.append(window)

    time_event = calcium_time[0:calcium_sampling_rate*(event_length + bsline_length)] - bsline_length

    fig1 = plt.figure()
    for window in windows:
        plt.plot(time_event, window)
    plt.title("Calcium Data Windows")
    plt.xlabel("Time (s)")
    plt.ylabel("Calcium Level")
    if interactive_plots:
        plt.show(block=True)  # Block until window is closed
    else:
        plt.close(fig1)

    mean_window = np.mean(windows, axis=0)
    ci = 1.96 * sem(windows, axis=0)

    fig2 = plt.figure()
    plt.plot(time_event, mean_window, label='Mean')
    plt.fill_between(time_event, mean_window - ci, mean_window + ci, color='b', alpha=0.2, label='95% CI')
    plt.title("Average Calcium Data Window with 95% CI")
    plt.xlabel("Time (s)")
    plt.ylabel("Calcium Level")
    plt.legend()
    if interactive_plots:
        plt.show(block=True)  # Block until window is closed
    else:
        plt.close(fig2)

    calcium_mean_df = pd.DataFrame({'Time (s)': time_event, 'Calcium Level': mean_window})
    if save_files:
        calcium_mean_df.to_csv(Path(save_path) / 'calcium_mean.csv', index=False)

    calcium_windows_df = pd.DataFrame(windows).T
    calcium_windows_df.insert(0, 'Time (s)', time_event)
    if save_files:
        calcium_windows_df.to_csv(Path(save_path) / 'calcium_windows.csv', index=False)
    return calcium_windows_df

def process_arteriole_data(arteriole_diameter, smoothed_times, final_events, save_path, arteriole_sampling_rate, normalize=True, save_files=True, bsline_length=5, event_length=15, interactive_plots=False):
    arteriole_data = arteriole_diameter['arteriole_diameter'].values
    arteriole_time = arteriole_diameter['time'].values
    windows = []

    for event in final_events:
        time = smoothed_times[event]
        event_arteriole_idx = (arteriole_time < time).sum()
        window = arteriole_data[event_arteriole_idx - arteriole_sampling_rate*bsline_length:event_arteriole_idx + arteriole_sampling_rate*event_length]
        baseline = arteriole_data[event_arteriole_idx - arteriole_sampling_rate*bsline_length:event_arteriole_idx]
        if normalize:
            window = 100 * (window - np.mean(baseline)) / np.mean(baseline)
        windows.append(window)

    time_event = arteriole_time[0:arteriole_sampling_rate*(event_length + bsline_length)] - bsline_length

    fig1 = plt.figure()
    for window in windows:
        plt.plot(time_event, window)
    plt.title("Arteriole Diameter Data Windows")
    plt.xlabel("Time (s)")
    plt.ylabel("Arteriole Diameter")
    if interactive_plots:
        plt.show(block=True)  # Block until window is closed
    else:
        plt.close(fig1)

    mean_window = np.mean(windows, axis=0)
    ci = 1.96 * sem(windows, axis=0)

    fig2 = plt.figure()
    plt.plot(time_event, mean_window, label='Mean')
    plt.fill_between(time_event, mean_window - ci, mean_window + ci, color='b', alpha=0.2, label='95% CI')
    plt.title("Average Arteriole Diameter Data Window with 95% CI")
    plt.xlabel("Time (s)")
    plt.ylabel("Arteriole Diameter")
    plt.legend()
    if interactive_plots:
        plt.show(block=True)  # Block until window is closed
    else:
        plt.close(fig2)

    arteriole_mean_df = pd.DataFrame({'Time (s)': time_event, 'Arteriole Diameter': mean_window})

    if save_files:
        arteriole_mean_df.to_csv(Path(save_path) / 'arteriole_mean.csv', index=False)

    arteriole_windows_df = pd.DataFrame(windows).T
    arteriole_windows_df.insert(0, 'Time (s)', time_event)
    if save_files:
        arteriole_windows_df.to_csv(Path(save_path) / 'arteriole_windows.csv', index=False)

    return arteriole_windows_df

def process_whisker_data(normalize_whisker_gradient, whisker_time, smoothed_times,
                           final_events, save_path, whisker_sampling_rate,
                           save_files=True, normalize=True,
                           bsline_length=5, event_length=15, interactive_plots=False):
    windows_whisker = []
    time_event_whisker = whisker_time[0:(bsline_length + event_length) * whisker_sampling_rate] - bsline_length

    for event in final_events:
        time = smoothed_times[event]
        event_whisker_idx = (whisker_time < time).sum()
        start = event_whisker_idx - bsline_length * whisker_sampling_rate
        end   = event_whisker_idx + event_length * whisker_sampling_rate
        # skip out-of-bounds
        if start < 0 or end > len(normalize_whisker_gradient):
            continue

        window = normalize_whisker_gradient[start:end]
        baseline = normalize_whisker_gradient[start:event_whisker_idx]
        if normalize:
            window = 100 * (window - np.mean(baseline)) / np.mean(baseline)
        windows_whisker.append(window)

    # Compute mean and 95% CI
    mean_window_whisker = np.mean(windows_whisker, axis=0)
    ci_whisker = 1.96 * sem(windows_whisker, axis=0)

    # Plot individual windows
    fig1 = plt.figure()
    for window in windows_whisker:
        plt.plot(time_event_whisker, window)
    plt.title("Whisker Gradient Data Windows")
    plt.xlabel("Time (s)")
    plt.ylabel("Whisker Gradient")
    if interactive_plots:
        plt.show(block=True)  # Block until window is closed
    else:
        plt.close(fig1)

    # Plot mean with CI
    fig2 = plt.figure()
    plt.plot(time_event_whisker, mean_window_whisker, label='Mean')
    plt.fill_between(
        time_event_whisker,
        mean_window_whisker - ci_whisker,
        mean_window_whisker + ci_whisker,
        alpha=0.2,
        label='95% CI'
    )
    plt.title("Average Whisker Gradient Window with 95% CI")
    plt.xlabel("Time (s)")
    plt.ylabel("Whisker Gradient")
    plt.legend()
    if interactive_plots:
        plt.show(block=True)  # Block until window is closed
    else:
        plt.close(fig2)

    # Save to CSV
    whisker_mean_df = pd.DataFrame({'Time (s)': time_event_whisker,
                                    'Whisker Gradient': mean_window_whisker})
    if save_files:
        whisker_mean_df.to_csv(Path(save_path) / 'whisker_mean.csv', index=False)

    whisker_windows_df = pd.DataFrame(windows_whisker).T
    whisker_windows_df.insert(0, 'Time (s)', time_event_whisker)
    if save_files:
        whisker_windows_df.to_csv(Path(save_path) / 'whisker_windows.csv', index=False)

    return whisker_windows_df


def process_pupil_data(pupil_size, pupil_time, smoothed_times_series, final_events, save_path, pupil_sampling_rate, exclude_threshold=6, save_files=True, normalize=True, event_length=15, bsline_length=5, interactive_plots=False):
    windows_pupil = []
    clean_events = []
    for event in final_events:
        time = smoothed_times_series[event]
        event_pupil_idx = (pupil_time < time).sum()
        start_idx = event_pupil_idx - pupil_sampling_rate * bsline_length
        end_idx   = event_pupil_idx + pupil_sampling_rate * event_length

        # if the slice is out of bounds or empty, skip
        if start_idx < 0 or end_idx > len(pupil_size) or end_idx - start_idx == 0:
            continue

        window   = pupil_size[start_idx:end_idx]
        baseline = pupil_size[start_idx:event_pupil_idx]

        if normalize:
            window = 100 * (window - np.mean(baseline)) / np.mean(baseline)

        if (np.percentile(window, 80) > exclude_threshold) or (np.percentile(window, 20) < -exclude_threshold):
            continue

        clean_events.append(event)
        windows_pupil.append(window)

    time_event_pupil = pupil_time[0:pupil_sampling_rate * (event_length + bsline_length)] - bsline_length

    fig1 = plt.figure()
    for window in windows_pupil:
        plt.plot(time_event_pupil, window)
    plt.title("Pupil Size Data Windows")
    plt.xlabel("Time (s)")
    plt.ylabel("Pupil Size")
    if interactive_plots:
        plt.show(block=True)  # Block until window is closed
    else:
        plt.close(fig1)

    mean_window_pupil = np.mean(windows_pupil, axis=0)
    ci_pupil = 1.96 * sem(windows_pupil, axis=0)

    fig2 = plt.figure()
    plt.plot(time_event_pupil, mean_window_pupil, label='Mean')
    plt.fill_between(time_event_pupil, mean_window_pupil - ci_pupil, mean_window_pupil + ci_pupil, color='b', alpha=0.2, label='95% CI')
    plt.title("Average Pupil Size Data Window with 95% CI")
    plt.xlabel("Time (s)")
    plt.ylabel("Pupil Size")
    plt.legend()
    if interactive_plots:
        plt.show(block=True)  # Block until window is closed
    else:
        plt.close(fig2)

    pupil_mean_df = pd.DataFrame({'Time (s)': time_event_pupil, 'Pupil Size': mean_window_pupil})
    if save_files:
        pupil_mean_df.to_csv(Path(save_path) / 'pupil_mean.csv', index=False)

    pupil_windows_df = pd.DataFrame(windows_pupil).T
    pupil_windows_df.insert(0, 'Time (s)', time_event_pupil)
    if save_files:
        pupil_windows_df.to_csv(Path(save_path) / 'pupil_windows.csv', index=False)

    return pupil_windows_df, clean_events