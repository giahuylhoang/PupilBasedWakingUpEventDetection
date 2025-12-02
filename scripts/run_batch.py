import os
import sys
import logging
import argparse
from datetime import datetime
from src.utils.data_processing import process_data
from src.utils.utilities import find_folders_with_csv

# Add the project root to the PYTHONPATH
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

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

# Configure logging
log_file = f'logs/batch_log_{datetime.now().strftime("%Y%m%d_%H%M%S")}.txt'
os.makedirs(os.path.dirname(log_file), exist_ok=True)
logging.basicConfig(
    filename=log_file,
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def run_batch(root_folder=None, list_of_folders=None, results_path=None,
              threshold_to_exclude_from_min_max=1, threshold_to_exclude_base_on_pupil=2,
              plot_traces=True, save_trace_plot=True, clear_output=False,
              bsline_length=5, event_length=15, wakeup=False, interactive_plots=False):
    try:
        logging.info("Starting batch data processing")

        # Find folders containing CSV files
        if root_folder:
            if not os.path.isdir(root_folder):
                raise ValueError(f"Root folder does not exist: {root_folder}")
            folders = find_folders_with_csv(root_folder)
            logging.info(f"Found {len(folders)} folders with CSV files in {root_folder}")
            if len(folders) == 0:
                raise ValueError(f"No folders with CSV files found in {root_folder}")
        elif list_of_folders:
            # Verify folders exist and contain CSV files
            verified_folders = []
            for folder in list_of_folders:
                if not os.path.isdir(folder):
                    logging.warning(f"Folder does not exist: {folder}, skipping")
                    continue
                # Check if folder contains CSV files
                has_csv = any(f.endswith('.csv') for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f)))
                if has_csv:
                    verified_folders.append(folder)
                else:
                    logging.warning(f"Folder does not contain CSV files: {folder}, skipping")
            folders = verified_folders
            logging.info(f"Processing {len(folders)} specified folders")
            if len(folders) == 0:
                raise ValueError("No valid folders with CSV files found in the provided list")
        else:
            raise ValueError("Either root_folder or list_of_folders must be provided")

        if not results_path:
            raise ValueError("results_path must be provided")

        # Ensure results_path is an absolute path and separate from data folders
        results_path = os.path.abspath(results_path)
        logging.info(f"Results will be saved to: {results_path}")

        for folder in folders:
            try:
                folder_abs = os.path.abspath(folder)
                logging.info(f"Processing folder: {folder_abs}")

                # Create a unique result folder name based on the folder path
                # Use the last 2 directory components to create meaningful structure
                folder_parts = folder_abs.split(os.sep)
                if len(folder_parts) >= 2:
                    # Use last 2 parts (e.g., "2023.06.01/cycle_8")
                    folder_name = os.path.join(*folder_parts[-2:])
                else:
                    # If only one level, just use the folder name
                    folder_name = folder_parts[-1]
                
                # Replace any spaces or special characters that might cause issues
                folder_name = folder_name.replace(' ', '_')
                
                results_folder = os.path.join(results_path, folder_name)
                logging.info(f"Results folder for {folder}: {results_folder}")

                # Ensure the results directory exists
                os.makedirs(results_folder, exist_ok=True)

                process_data(
                    data_folder_path=folder,
                    threshold_to_exclude_from_min_max=threshold_to_exclude_from_min_max,
                    threshold_to_exclude_base_on_pupil=threshold_to_exclude_base_on_pupil,
                    plot_traces=plot_traces,
                    save_trace_plot=save_trace_plot,
                    clear_output=clear_output,
                    bsline_length=bsline_length,
                    event_length=event_length,
                    results_folder=results_folder,
                    wakeup=wakeup,
                    interactive_plots=interactive_plots
                )
                logging.info(f"Successfully processed folder: {folder}")
            except Exception as e:
                logging.error(f"Error processing folder {folder}: {e}")

        logging.info("Batch data processing completed")

    except Exception as e:
        logging.error(f"An error occurred during batch processing: {e}")
        raise

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process batch data folders. Finds all folders containing CSV files.')
    parser.add_argument('-r', '--root_folder', type=str, help='Parent folder to search for all subfolders containing CSV files')
    parser.add_argument('--folders', nargs='+', help='List of specific folders to process (alternative to --root_folder)')
    parser.add_argument('-o', '--output', '--results_path', '--default_result_path', dest='results_path', 
                       type=str, required=True, help='Output folder for results (must be separate from data folders)')
    parser.add_argument('--threshold_to_exclude_from_min_max', type=int, default=1, help='Threshold to exclude from min max')
    parser.add_argument('--threshold_to_exclude_base_on_pupil', type=int, default=2, help='Threshold to exclude based on pupil')
    parser.add_argument('--plot_traces', type=str2bool, default=True, help='Whether to plot traces')
    parser.add_argument('--save_trace_plot', type=str2bool, default=True, help='Whether to save trace plot')
    parser.add_argument('--clear_output', type=str2bool, default=False, help='Whether to clear output')
    parser.add_argument('--bsline_length', type=int, default=5, help='Baseline length')
    parser.add_argument('--event_length', type=int, default=15, help='Event length')
    parser.add_argument('--wake_up', type=str2bool, default=False, help='If sleep-to-wake transition detection, set it to False, otherwise if wake-to-sleep transition detection, set it to True')
    parser.add_argument('--interactive_plots', type=str2bool, default=False, help='Whether to show plots interactively (true) or just save them (false)')

    args = parser.parse_args()

    run_batch(
        root_folder=args.root_folder,
        list_of_folders=args.folders,
        results_path=args.results_path,
        threshold_to_exclude_from_min_max=args.threshold_to_exclude_from_min_max,
        threshold_to_exclude_base_on_pupil=args.threshold_to_exclude_base_on_pupil,
        plot_traces=args.plot_traces,
        save_trace_plot=args.save_trace_plot,
        clear_output=args.clear_output,
        bsline_length=args.bsline_length,
        event_length=args.event_length,
        wakeup=args.wake_up,
        interactive_plots=args.interactive_plots
    )