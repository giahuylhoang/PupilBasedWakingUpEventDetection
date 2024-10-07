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

def run_batch(root_folder=None, list_of_folders=None):
    try:
        logging.info("Starting batch data processing")

        if root_folder:
            folders = find_folders_with_csv(root_folder)
        elif list_of_folders:
            folders = list_of_folders
        else:
            raise ValueError("Either root_folder or list_of_folders must be provided")

        for folder in folders:
            try:
                logging.info(f"Processing folder: {folder}")
                process_data(folder)
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

    args = parser.parse_args()

    if args.root_folder:
        run_batch(root_folder=args.root_folder)
    elif args.folders:
        run_batch(list_of_folders=args.folders)
    else:
        print("Please provide either --root_folder or --folders argument.")