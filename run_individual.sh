#!/bin/bash

# Run init.sh to set up the environment
source ./init.sh


# Please input the data path here
DATA_FOLDER_PATH="data/test/cycle_8"
RESULTS_FOLDER="data/test/results/cycle_8"

# Default values for optional arguments
BSLINE_LENGTH=5 # Baseline length in seconds before the event
EVENT_LENGTH=15 # Event length in seconds after the event
RESULTS_FOLDER="data/test/cycle_8" # Path to save results; if empty, a default path will be used
THRESHOLD_TO_EXCLUDE_FROM_MIN_MAX=1 # Percentile threshold for outlier exclusion in min/max calculations
# Example: If set to 1, values outside 1st-99th percentiles are excluded

THRESHOLD_TO_EXCLUDE_BASE_ON_PUPIL=2 # Threshold for excluding events based on pupil trace outliers
# Example: If set to 2, events where the top 20% of pupil values > 2 or bottom 20% < -2 are excluded
# Set to a very high value to effectively disable this exclusion

PLOT_TRACES=true # Whether to generate plots of the traces
SAVE_TRACE_PLOT=true # Whether to save the generated trace plots
CLEAR_OUTPUT=false # Whether to clear output after processing (useful in interactive environments)


# Function to display help message
usage() {
    echo "Usage: $0 -d <data_folder_path> [-r <results_folder>] [--threshold_min_max <value>] [--threshold_pupil <value>]"
    echo "          [--plot_traces <true|false>] [--save_trace_plot <true|false>] [--clear_output <true|false>]"
    echo "          [--bsline_length <value>] [--event_length <value>]"
    exit 1
}

# Parse command-line arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        -d|--data_folder_path) DATA_FOLDER_PATH="$2"; shift ;;
        -r|--results_folder) RESULTS_FOLDER="$2"; shift ;;
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
if [ -z "$DATA_FOLDER_PATH" ]; then
    echo "Error: data_folder_path is required."
    usage
fi

# Run the Python script with the provided arguments
python scripts/run_individual.py "$DATA_FOLDER_PATH" \
    --results_folder "$RESULTS_FOLDER" \
    --threshold_to_exclude_from_min_max "$THRESHOLD_TO_EXCLUDE_FROM_MIN_MAX" \
    --threshold_to_exclude_base_on_pupil "$THRESHOLD_TO_EXCLUDE_BASE_ON_PUPIL" \
    --plot_traces "$PLOT_TRACES" \
    --save_trace_plot "$SAVE_TRACE_PLOT" \
    --clear_output "$CLEAR_OUTPUT" \
    --bsline_length "$BSLINE_LENGTH" \
    --event_length "$EVENT_LENGTH"