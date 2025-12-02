#!/bin/bash

# Run init.sh to set up the environment
source ./init.sh

# Default values for optional arguments
ROOT_FOLDER="data/raw/test" # Parent folder to search for all subfolders containing CSV files 
LIST_OF_FOLDERS=() # Use this to specify a list of folders to process (alternative to ROOT_FOLDER)
RESULTS_PATH="data/results" # Path to save results (must be completely separate from data folders)

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
WAKE_UP=false # If sleep-to-wake transition detection, set it to False, otherwise,if wake-to-sleep transition detection, set it to True
INTERACTIVE_PLOTS=true # Whether to show plots interactively (true) or just save them (false)

# Function to display help message
usage() {
    echo "Usage: $0 [-r <root_folder>] [--folders <folder1 folder2 ...>] -o <results_path>"
    echo ""
    echo "Options:"
    echo "  -r, --root_folder <path>       Parent folder to search for all folders containing CSV files"
    echo "  --folders <folder1 folder2 ...> List of specific folders to process (alternative to -r)"
    echo "  -o, --output <path>            Output folder for results (must be separate from data folders)"
    echo ""
    echo "  [--threshold_min_max <value>] [--threshold_pupil <value>]"
    echo "  [--plot_traces <true|false>] [--save_trace_plot <true|false>] [--clear_output <true|false>]"
    echo "  [--bsline_length <value>] [--event_length <value>] [--wake_up <true|false>]"
    echo "  [--interactive_plots <true|false>]"
    echo ""
    echo "Examples:"
    echo "  # Process all folders with CSV files in data/raw/test:"
    echo "  $0 -r data/raw/test -o data/results/batch_run"
    echo ""
    echo "  # Process specific folders:"
    echo "  $0 --folders data/raw/test/cycle_9 data/raw/test/cycle_10 -o data/results/batch_run"
    exit 1
}

# Parse command-line arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        -r|--root_folder) ROOT_FOLDER="$2"; shift ;;
        --folders) shift; LIST_OF_FOLDERS=("$@"); break ;;
        -o|--output|--results_path|--default_result_path) RESULTS_PATH="$2"; shift ;;
        --threshold_min_max) THRESHOLD_TO_EXCLUDE_FROM_MIN_MAX="$2"; shift ;;
        --threshold_pupil) THRESHOLD_TO_EXCLUDE_BASE_ON_PUPIL="$2"; shift ;;
        --plot_traces) PLOT_TRACES="$2"; shift ;;
        --save_trace_plot) SAVE_TRACE_PLOT="$2"; shift ;;
        --clear_output) CLEAR_OUTPUT="$2"; shift ;;
        --bsline_length) BSLINE_LENGTH="$2"; shift ;;
        --event_length) EVENT_LENGTH="$2"; shift ;;
        --wake_up) WAKE_UP="$2"; shift ;;
        --interactive_plots) INTERACTIVE_PLOTS="$2"; shift ;;
        *) usage ;;
    esac
    shift
done

# Check if the required argument is provided
if [ -z "$RESULTS_PATH" ]; then
    echo "Error: Output path (-o or --output) is required."
    echo ""
    usage
fi

# Check if at least one of the required arguments is provided
if [ -z "$ROOT_FOLDER" ] && [ ${#LIST_OF_FOLDERS[@]} -eq 0 ]; then
    echo "Error: Either --root_folder (-r) or --folders must be provided."
    echo ""
    usage
fi

# Warn if results path might overlap with data folders
if [ -n "$ROOT_FOLDER" ] && [[ "$RESULTS_PATH" == "$ROOT_FOLDER"* ]]; then
    echo "Warning: Results path '$RESULTS_PATH' appears to be inside or overlap with data folder '$ROOT_FOLDER'"
    echo "This may cause issues. Consider using a completely separate location (e.g., 'data/results/batch_run')"
    echo ""
fi

# Run the Python script with the provided arguments
if [ -n "$ROOT_FOLDER" ]; then
    python3 scripts/run_batch.py --root_folder "$ROOT_FOLDER" --results_path "$RESULTS_PATH" \
        --threshold_to_exclude_from_min_max "$THRESHOLD_TO_EXCLUDE_FROM_MIN_MAX" \
        --threshold_to_exclude_base_on_pupil "$THRESHOLD_TO_EXCLUDE_BASE_ON_PUPIL" \
        --plot_traces "$PLOT_TRACES" \
        --save_trace_plot "$SAVE_TRACE_PLOT" \
        --clear_output "$CLEAR_OUTPUT" \
        --bsline_length "$BSLINE_LENGTH" \
        --event_length "$EVENT_LENGTH" \
        --wake_up "$WAKE_UP" \
        --interactive_plots "$INTERACTIVE_PLOTS"
elif [ ${#LIST_OF_FOLDERS[@]} -gt 0 ]; then
    python3 scripts/run_batch.py --folders "${LIST_OF_FOLDERS[@]}" --results_path "$RESULTS_PATH" \
        --threshold_to_exclude_from_min_max "$THRESHOLD_TO_EXCLUDE_FROM_MIN_MAX" \
        --threshold_to_exclude_base_on_pupil "$THRESHOLD_TO_EXCLUDE_BASE_ON_PUPIL" \
        --plot_traces "$PLOT_TRACES" \
        --save_trace_plot "$SAVE_TRACE_PLOT" \
        --clear_output "$CLEAR_OUTPUT" \
        --bsline_length "$BSLINE_LENGTH" \
        --event_length "$EVENT_LENGTH" \
        --wake_up "$WAKE_UP" \
        --interactive_plots "$INTERACTIVE_PLOTS"
fi