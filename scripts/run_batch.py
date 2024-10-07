import os
import sys
import logging
import argparse
from datetime import datetime
from src.utils.data_processing import process_data
from src.utils.utilities import find_folders_with_csv

# Add the project root to the PYTHONPATH
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Configure logging
log_file = f'logs/batch_log_{datetime.now().strftime("%Y%m%d_%H%M%S")}.txt'
os.makedirs(os.path.dirname(log_file), exist_ok=True)
logging.basicConfig(
    filename=log_file,
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def run_batch(root_folder=None, list_of_folders=None, default_result_path=None,
              threshold_to_exclude_from_min_max=1, threshold_to_exclude_base_on_pupil=2,
              plot_traces=True, save_trace_plot=True, clear_output=False,
              bsline_length=5, event_length=15):
    try:
        logging.info("Starting batch data processing")

        if root_folder:
            folders = find_folders_with_csv(root_folder)
        elif list_of_folders:
            folders = list_of_folders
        else:
            raise ValueError("Either root_folder or list_of_folders must be provided")

        if not default_result_path:
            raise ValueError("default_result_path must be provided")

        for folder in folders:
            try:
                logging.info(f"Processing folder: {folder}")

                # Construct the results folder path
                base_folder = os.path.join(*folder.split(os.sep)[-2:])  # This will give "2023.06.01/cycle_8"
                results_folder = os.path.join(default_result_path, base_folder)

                # Ensure the results directory exists
                os.makedirs(results_folder, exist_ok=True)

                process_data(
                    folder,
                    threshold_to_exclude_from_min_max=threshold_to_exclude_from_min_max,
                    threshold_to_exclude_base_on_pupil=threshold_to_exclude_base_on_pupil,
                    plot_traces=plot_traces,
                    save_trace_plot=save_trace_plot,
                    clear_output=clear_output,
                    bsline_length=bsline_length,
                    event_length=event_length,
                    results_folder=results_folder
                )
                logging.info(f"Successfully processed folder: {folder}")
            except Exception as e:
                logging.error(f"Error processing folder {folder}: {e}")

        logging.info("Batch data processing completed")

    except Exception as e:
        logging.error(f"An error occurred during batch processing: {e}")
        raise

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process batch data folders.')
    parser.add_argument('--root_folder', type=str, help='Root folder to search for CSV files')
    parser.add_argument('--folders', nargs='+', help='List of specific folders to process')
    parser.add_argument('--default_result_path', type=str, required=True, help='Default path to save results')
    parser.add_argument('--threshold_to_exclude_from_min_max', type=int, default=1, help='Threshold to exclude from min max')
    parser.add_argument('--threshold_to_exclude_base_on_pupil', type=int, default=2, help='Threshold to exclude based on pupil')
    parser.add_argument('--plot_traces', type=bool, default=True, help='Whether to plot traces')
    parser.add_argument('--save_trace_plot', type=bool, default=True, help='Whether to save trace plot')
    parser.add_argument('--clear_output', type=bool, default=False, help='Whether to clear output')
    parser.add_argument('--bsline_length', type=int, default=5, help='Baseline length')
    parser.add_argument('--event_length', type=int, default=15, help='Event length')

    args = parser.parse_args()

    run_batch(
        root_folder=args.root_folder,
        list_of_folders=args.folders,
        default_result_path=args.default_result_path,
        threshold_to_exclude_from_min_max=args.threshold_to_exclude_from_min_max,
        threshold_to_exclude_base_on_pupil=args.threshold_to_exclude_base_on_pupil,
        plot_traces=args.plot_traces,
        save_trace_plot=args.save_trace_plot,
        clear_output=args.clear_output,
        bsline_length=args.bsline_length,
        event_length=args.event_length
    )