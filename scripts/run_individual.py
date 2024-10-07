import os
import sys
import logging
import argparse
from src.utils.data_processing import process_data

# Add the project root to the PYTHONPATH
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Configure logging to write to a file
from datetime import datetime

log_file = f'logs/log_{datetime.now().strftime("%Y%m%d_%H%M%S")}.txt'
os.makedirs(os.path.dirname(log_file), exist_ok=True)
logging.basicConfig(
    filename=log_file,
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def run_individual(data_folder_path, results_folder=None):
    try:
        logging.info("Starting individual data processing")

        # Determine the results folder
        if results_folder is None:
            # Create a default results folder based on the data folder name
            folder_name = os.path.basename(data_folder_path.rstrip('/'))
            results_folder = os.path.join(os.path.dirname(data_folder_path), f"{folder_name}_results")
        
        # Ensure the results directory exists
        os.makedirs(results_folder, exist_ok=True)

        # Process the data for the specified folder
        process_data(
            data_folder_path,
            threshold_to_exclude_from_min_max=1,
            threshold_to_exclude_base_on_pupil=2,
            plot_traces=True,
            save_trace_plot=True,
            clear_output=False,
            bsline_length=5,
            event_length=15,
            results_folder=results_folder
        )

        logging.info("Individual data processing completed successfully")

    except Exception as e:
        logging.error(f"An error occurred during individual data processing: {e}")
        raise

if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Process individual data folder.')
    parser.add_argument('data_folder_path', type=str, help='Path to the data folder to process')
    parser.add_argument('--results_folder', type=str, help='Optional path to the results folder')

    # Parse arguments
    args = parser.parse_args()

    # Run the individual processing with the provided arguments
    run_individual(args.data_folder_path, args.results_folder)