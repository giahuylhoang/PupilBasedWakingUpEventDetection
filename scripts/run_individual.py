import os
import sys
import logging
import argparse
from src.utils.data_processing import process_data
from datetime import datetime
def str2bool(v):
    if isinstance(v, bool):
        return v
    v = v.lower()
    if v in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
    

# Add the project root to the PYTHONPATH
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Configure logging to write to a file
log_file = f'logs/log_{datetime.now().strftime("%Y%m%d_%H%M%S")}.txt'
os.makedirs(os.path.dirname(log_file), exist_ok=True)
logging.basicConfig(
    filename=log_file,
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def run_individual(data_folder_path, results_folder=None, threshold_to_exclude_from_min_max=1,
                   threshold_to_exclude_base_on_pupil=2, plot_traces=True, save_trace_plot=True,
                   clear_output=False, bsline_length=5, event_length=15, wakeup=False):
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
            data_folder_path=data_folder_path,
            threshold_to_exclude_from_min_max=threshold_to_exclude_from_min_max,
            threshold_to_exclude_base_on_pupil=threshold_to_exclude_base_on_pupil,
            plot_traces=plot_traces,
            save_trace_plot=save_trace_plot,
            clear_output=clear_output,
            bsline_length=bsline_length,
            event_length=event_length,
            results_folder=results_folder,
            wakeup=wakeup
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
    parser.add_argument('--threshold_to_exclude_from_min_max', type=int, default=1, help='Threshold to exclude from min max')
    parser.add_argument('--threshold_to_exclude_base_on_pupil', type=int, default=2, help='Threshold to exclude based on pupil')
    parser.add_argument('--plot_traces', type=bool, default=True, help='Whether to plot traces')
    parser.add_argument('--save_trace_plot', type=bool, default=True, help='Whether to save trace plot')
    parser.add_argument('--clear_output', type=bool, default=False, help='Whether to clear output')
    parser.add_argument('--bsline_length', type=int, default=5, help='Baseline length')
    parser.add_argument('--event_length', type=int, default=15, help='Event length')
    parser.add_argument(
    '--wake_up',
    type=str2bool,
    default=False,
    help='sleep-to-wake or wake-to-sleep event detection (true|false)'
)

    # Parse arguments
    args = parser.parse_args()

    # Run the individual processing with the provided arguments
    run_individual(
        data_folder_path=args.data_folder_path, 
        results_folder=args.results_folder, 
        threshold_to_exclude_from_min_max=args.threshold_to_exclude_from_min_max,
        threshold_to_exclude_base_on_pupil=args.threshold_to_exclude_base_on_pupil,
        plot_traces=args.plot_traces,
        save_trace_plot=args.save_trace_plot,
        clear_output=args.clear_output, 
        bsline_length=args.bsline_length, 
        event_length=args.event_length, 
        wakeup=args.wake_up
    )