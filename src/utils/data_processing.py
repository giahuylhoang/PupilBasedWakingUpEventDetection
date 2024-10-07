import os
import numpy as np
import matplotlib.pyplot as plt
import logging
from src.data.data_loader import load_arteriole_data, load_calcium_data, load_pupil_data, load_whisker_data
from src.utils.utilities import (
    detect_and_interpolate_sudden_changes, normalize_mean_std, normalize_series,
    moving_average, calculate_derivative,
)
from src.utils.processing import process_calcium_data, process_arteriole_data, process_whisker_data, process_pupil_data
from src.utils.event_detection import detect_events

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def process_data(data_folder_path, threshold_to_exclude_from_min_max=1, threshold_to_exclude_base_on_pupil=2, plot_traces=False, save_trace_plot=True, clear_output=True, bsline_length=5, event_length=15, results_folder=None):
    try:
        if results_folder is None:
            results_path = data_folder_path.replace('test', 'results')
            results_path = data_folder_path.replace('data', 'results')
        else:
            results_path = data_folder_path.replace('test', results_folder)
            results_path = data_folder_path.replace('data', results_folder)
        os.makedirs(results_path, exist_ok=True)

        logging.info(f"Processing data for folder: {data_folder_path}")

        # Load data
        logging.info("Loading arteriole data")
        arteriole_data = load_arteriole_data(data_folder_path)

        logging.info("Loading calcium data")
        calcium_data = load_calcium_data(data_folder_path)

        logging.info("Loading pupil data")
        pupil_data = load_pupil_data(data_folder_path)

        logging.info("Loading whisker data")
        whisker_data = load_whisker_data(data_folder_path)

        # Normalize and interpolate pupil data
        logging.info("Normalizing and interpolating pupil data")
        pupil_data_normalized = normalize_mean_std(pupil_data)
        interpolated_pupil_data = detect_and_interpolate_sudden_changes(pupil_data_normalized, 0.001, 1)

        # Calculate sampling rates
        logging.info("Calculating sampling rates")
        arteriole_sampling_rate = int(round(1 / np.mean(np.diff(arteriole_data['time']))))
        calcium_sampling_rate = int(round(1 / np.mean(np.diff(calcium_data['time']))))
        pupil_sampling_rate = int(round(1 / np.mean(np.diff(pupil_data['time']))))
        whisker_sampling_rate = int(round(1 / np.mean(np.diff(whisker_data['time']))))

        # Process pupil data
        logging.info("Processing pupil data")
        pupil_size_normalized = normalize_series(pupil_data['pupil_size'].values, threshold_to_exclude_from_min_max)
        smoothed_pupil_size = moving_average(interpolated_pupil_data['pupil_size'], pupil_sampling_rate)
        normalized_smoothed_pupil_size = normalize_series(smoothed_pupil_size, threshold_to_exclude_from_min_max)
        smoothed_time_series = interpolated_pupil_data['time'].values[int(pupil_sampling_rate / 2) - 1:-int(pupil_sampling_rate / 2)][:smoothed_pupil_size.shape[0]]

        # Detect events
        logging.info("Detecting events")
        whisker_time = whisker_data['time'].values
        whisker_angle = whisker_data['whisker_angle'].values
        normalized_whisker_velocity = normalize_series(np.power(calculate_derivative(whisker_angle, whisker_time), 2))
        whisker_velocity_time = whisker_time[:-1]

        waking_up_events = detect_events(normalized_smoothed_pupil_size, smoothed_time_series, normalized_whisker_velocity, whisker_velocity_time, pupil_sampling_rate, whisker_sampling_rate, bsline_length, event_length)

        # Process and save data
        logging.info("Processing and saving pupil data")
        pupil_traces_df, clean_events = process_pupil_data(pupil_size_normalized, pupil_data['time'].values, smoothed_time_series, waking_up_events, results_path, pupil_sampling_rate, exclude_threshold=threshold_to_exclude_base_on_pupil, normalize=False, bsline_length=bsline_length, event_length=event_length)
        pupil_traces_df.to_csv(os.path.join(results_path, 'pupil_traces.csv'), index=False)
        waking_up_events = clean_events

        logging.info("Processing and saving calcium data")
        calcium_traces_df = process_calcium_data(calcium_data, smoothed_time_series, waking_up_events, results_path, calcium_sampling_rate, bsline_length=bsline_length, event_length=event_length)
        calcium_traces_df.to_csv(os.path.join(results_path, 'calcium_traces.csv'), index=False)

        logging.info("Processing and saving arteriole data")
        arteriole_traces = process_arteriole_data(arteriole_data, smoothed_time_series, waking_up_events, results_path, arteriole_sampling_rate, bsline_length=bsline_length, event_length=event_length)
        arteriole_traces.to_csv(os.path.join(results_path, 'arteriole_traces.csv'), index=False)

        logging.info("Processing and saving whisker data")
        whisker_traces = process_whisker_data(normalized_whisker_velocity, whisker_velocity_time, smoothed_time_series, waking_up_events, results_path, whisker_sampling_rate, bsline_length=bsline_length, event_length=event_length)
        whisker_traces.to_csv(os.path.join(results_path, 'whisker_traces.csv'), index=False)

        if save_trace_plot:
            logging.info("Saving trace plots")
            figure = plt.figure(figsize=(14, 8))
            for col in pupil_traces_df.columns[1:]:
                plt.plot(pupil_traces_df['Time (s)'], pupil_traces_df[col], label=col)
                plt.legend()
            figure.savefig(os.path.join(results_path, 'pupil_traces.png'))
            plt.close(figure)

            figure_2 = plt.figure(figsize=(14, 8))
            for col in calcium_traces_df.columns[1:]:
                plt.plot(calcium_traces_df['Time (s)'], calcium_traces_df[col], label=col)
                plt.legend()
            figure_2.savefig(os.path.join(results_path, 'calcium_traces.png'))
            plt.close(figure_2)

            figure_3 = plt.figure(figsize=(14, 8))
            for col in arteriole_traces.columns[1:]:
                plt.plot(arteriole_traces['Time (s)'], arteriole_traces[col], label=col)
                plt.legend()
            figure_3.savefig(os.path.join(results_path, 'arteriole_traces.png'))
            plt.close(figure_3)

        if clear_output:
            from IPython.display import clear_output
            clear_output()

        logging.info("Data processing completed successfully")

    except Exception as e:
        logging.error(f"An error occurred: {e}")
        raise