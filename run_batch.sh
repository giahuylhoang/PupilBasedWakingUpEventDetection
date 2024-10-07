#!/bin/bash

#------------------------------------------------------------------------------
# Default values for optional arguments
ROOT_FOLDER="$(pwd)" # Use this to automatically detect all folders/subdirectories containing CSV files
LIST_OF_FOLDERS=() # Use this to specify a list of folders to process
DEFAULT_RESULT_PATH="results" # Default path to save results


#------------------------------------------------------------------------------
# Default values for optional arguments
BSLINE_LENGTH=5 # Baseline length in seconds before the event
EVENT_LENGTH=15 # Event length in seconds after the event
THRESHOLD_TO_EXCLUDE_FROM_MIN_MAX=1 # Percentile threshold for outlier exclusion in min/max calculations
# Example: If set to 1, values outside 1st-99th percentiles are excluded

THRESHOLD_TO_EXCLUDE_BASE_ON_PUPIL=2 # Threshold for excluding events based on pupil trace outliers
# Example: If set to 2, events where the top 20% of pupil values > 2 or bottom 20% < -2 are excluded
# Set to a very high value to effectively disable this exclusion

PLOT_TRACES=true # Whether to generate plots of the traces
SAVE_TRACE_PLOT=true # Whether to save the generated trace plots
CLEAR_OUTPUT=false # Whether to clear output after processing (useful in interactive environments)

# Run init.sh to set up the environment
source ./init.sh

# Function to display help message
usage() {
    echo "Usage: $0 [-r <root_folder>] [--folders <folder1 folder2 ...>] --default_result_path <path>"
    echo "          [--threshold_min_max <value>] [--threshold_pupil <value>]"
    echo "          [--plot_traces <true|false>] [--save_trace_plot <true|false>] [--clear_output <true|false>]"
    echo "          [--bsline_length <value>] [--event_length <value>]"
    exit 1
}

# Parse command-line arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        -r|--root_folder) ROOT_FOLDER="$2"; shift ;;
        --folders) shift; LIST_OF_FOLDERS=("$@"); break ;;
        --default_result_path) DEFAULT_RESULT_PATH="$2"; shift ;;
        --threshold_min_max) THRESHOLD_TO_EXCLUDE_FROM_MIN_MAX="$2"; shift ;;
        --threshold_pupil) THRESHOLD_TO_EXCLUDE_BASE_ON_PUPIL="$2"; shift ;;
        --plot_traces) PLOT_TRACES="$2"; shift ;;
        --save_trace_plot) SAVE_TRACE_PLOT="$2"; shift ;;
        --clear_output) CLEAR_OUTPUT="$2"; shift ;;
        --bsline_length) BSLINE_LENGTH="$2"; shift ;;
        --event_length) EVENT_LENGTH="$2"; shift ;;
        *) usage ;;
    esac
    shift
done

# Check if the required argument is provided
if [ -z "$DEFAULT_RESULT_PATH" ]; then
    echo "Error: default_result_path is required."
    usage
fi

# Check if at least one of the required arguments is provided
if [ -z "$ROOT_FOLDER" ] && [ ${#LIST_OF_FOLDERS[@]} -eq 0 ]; then
    echo "Error: Either root_folder or list_of_folders must be provided."
    usage
fi

# Run the Python script with the provided arguments
if [ -n "$ROOT_FOLDER" ]; then
    python scripts/run_batch.py --root_folder "$ROOT_FOLDER" --default_result_path "$DEFAULT_RESULT_PATH" \
        --threshold_to_exclude_from_min_max "$THRESHOLD_TO_EXCLUDE_FROM_MIN_MAX" \
        --threshold_to_exclude_base_on_pupil "$THRESHOLD_TO_EXCLUDE_BASE_ON_PUPIL" \
        --plot_traces "$PLOT_TRACES" \
        --save_trace_plot "$SAVE_TRACE_PLOT" \
        --clear_output "$CLEAR_OUTPUT" \
        --bsline_length "$BSLINE_LENGTH" \
        --event_length "$EVENT_LENGTH"
elif [ ${#LIST_OF_FOLDERS[@]} -gt 0 ]; then
    python scripts/run_batch.py --folders "${LIST_OF_FOLDERS[@]}" --default_result_path "$DEFAULT_RESULT_PATH" \
        --threshold_to_exclude_from_min_max "$THRESHOLD_TO_EXCLUDE_FROM_MIN_MAX" \
        --threshold_to_exclude_base_on_pupil "$THRESHOLD_TO_EXCLUDE_BASE_ON_PUPIL" \
        --plot_traces "$PLOT_TRACES" \
        --save_trace_plot "$SAVE_TRACE_PLOT" \
        --clear_output "$CLEAR_OUTPUT" \
        --bsline_length "$BSLINE_LENGTH" \
        --event_length "$EVENT_LENGTH"
fi