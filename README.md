# Pupil-Based Waking Up Event Detection

A Python-based analysis tool for detecting waking up events based on pupil data and other physiological measurements.

## Prerequisites

- **Python 3.7 or higher** (Python 3.8+ recommended)
- **macOS or Linux** operating system
- **Bash shell** (default on macOS and most Linux distributions)

## Quick Setup

### For New Users (First Time Setup)

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd PupilBasedWakingUpEventDetection
   ```

2. **Run the setup script:**
   ```bash
   chmod +x setup.sh
   ./setup.sh
   ```

   This will:
   - Check for Python 3 installation
   - Create a virtual environment
   - Install all required dependencies
   - Set up the environment

3. **Test the setup:**
   ```bash
   chmod +x test_setup.sh
   ./test_setup.sh
   ```
   This will verify that everything is installed and working correctly.

4. **You're ready to go!** Proceed to "Running the Analysis" below.

### Alternative Manual Setup

If you prefer to set up manually:

1. **Initialize the environment:**
   ```bash
   source init.sh
   ```

   This script will:
   - Create a virtual environment if it doesn't exist
   - Install dependencies from `requirements.txt`
   - Set up PYTHONPATH automatically

2. **Verify installation:**
   ```bash
   python3 --version
   pip list
   ```

## Running the Analysis

### Run Individual Analysis

The simplest way to run an analysis:

```bash
./run_individual.sh -d "data/raw/test/cycle_9"
```

**Full usage:**
```bash
./run_individual.sh -d <data_folder_path> [options]
```

**Required arguments:**
- `-d` or `--data_folder_path`: Path to the folder containing input data files

**Optional arguments:**
- `-r` or `--results_folder`: Path to save results (default: `data/results/test/cycle_9`)
- `--threshold_min_max <value>`: Percentile threshold for outlier exclusion (default: 1)
- `--threshold_pupil <value>`: Threshold for excluding events based on pupil trace (default: 2)
- `--plot_traces <true|false>`: Whether to generate plots (default: false)
- `--save_trace_plot <true|false>`: Whether to save trace plots (default: true)
- `--clear_output <true|false>`: Whether to clear output after processing (default: false)
- `--bsline_length <value>`: Baseline length in seconds before event (default: 10)
- `--event_length <value>`: Event length in seconds after event (default: 20)
- `--wake_up <true|false>`: True for wake-to-sleep, False for sleep-to-wake (default: false)

**Example:**
```bash
./run_individual.sh \
  -d "data/raw/test/cycle_9" \
  -r "data/results/test/cycle_9" \
  --bsline_length 10 \
  --event_length 20 \
  --wake_up false
```

### Using Python Script Directly

If you prefer to run the Python script directly:

1. **Activate the environment first:**
   ```bash
   source init.sh
   ```

2. **Run the script:**
   ```bash
   python3 scripts/run_individual.py "data/raw/test/cycle_9" \
     --results_folder "data/results/test/cycle_9" \
     --bsline_length 10 \
     --event_length 20
   ```

## Input Data Format

Your data folder should contain the following CSV files (with flexible naming):

- **Calcium data**: Any file containing "calcium" in the filename (e.g., `calcium.csv`)
- **Pupil data**: Any file containing "pupil" in the filename (e.g., `pupil_size.csv`)
- **Arteriole data**: Any file containing "arteriole" in the filename (e.g., `arteriole_diameter.csv`)
- **Whisker data**: Any file containing "whisker" in the filename (e.g., `whisker_gradients.csv`)

### Expected CSV Format:

- **Calcium/Pupil/Arteriole files**: Should have two columns (time and value), with or without headers
- **Whisker files**: Can have multiple columns; only the first column will be used

## Output

Results are saved to the specified results folder and include:

- `pupil_traces.csv` - Processed pupil trace data
- `calcium_traces.csv` - Processed calcium trace data
- `arteriole_traces.csv` - Processed arteriole trace data
- `whisker_traces.csv` - Processed whisker trace data
- `*_mean.csv` - Mean trace data for each measurement type
- `*_windows.csv` - Windowed event data
- `*_traces.png` - Visualization plots (if enabled)

## Quick Test on Another Computer

To quickly test if the setup works on another computer after pulling from git:

```bash
# 1. Make scripts executable (if needed)
chmod +x setup.sh test_setup.sh init.sh run_individual.sh

# 2. Run the setup
./setup.sh

# 3. Test everything is working
./test_setup.sh
```

The `test_setup.sh` script will check:
- ✓ Python 3 installation
- ✓ Virtual environment creation
- ✓ All required packages
- ✓ Project module imports
- ✓ Basic functionality

If all tests pass, you're good to go!

## Troubleshooting

### Python 3 Not Found

If you get an error about Python 3 not being found:

**On macOS:**
```bash
# Install using Homebrew
brew install python3

# Or download from python.org
```

**On Linux:**
```bash
# Ubuntu/Debian
sudo apt-get update
sudo apt-get install python3 python3-pip python3-venv

# Fedora/CentOS
sudo dnf install python3 python3-pip
```

### Virtual Environment Issues

If you encounter issues with the virtual environment:

```bash
# Remove the existing venv
rm -rf venv

# Reinitialize
source init.sh
```

### Permission Denied

If you get "Permission denied" errors:

```bash
# Make scripts executable
chmod +x init.sh
chmod +x run_individual.sh
chmod +x setup.sh
```

### Dependencies Installation Fails

If package installation fails:

1. Make sure you have an internet connection
2. Try upgrading pip first:
   ```bash
   source init.sh
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

3. On macOS, if you get compilation errors, you may need Xcode command line tools:
   ```bash
   xcode-select --install
   ```

## Project Structure

```
PupilBasedWakingUpEventDetection/
├── init.sh                  # Environment initialization script
├── setup.sh                 # First-time setup script
├── run_individual.sh        # Main analysis script
├── requirements.txt         # Python dependencies
├── README.md               # This file
├── scripts/
│   ├── run_individual.py   # Main Python analysis script
│   └── run_batch.py        # Batch processing script
├── src/
│   ├── data/
│   │   └── data_loader.py  # Data loading utilities
│   ├── utils/
│   │   ├── data_processing.py
│   │   ├── event_detection.py
│   │   └── utilities.py
│   └── visualization/
│       └── plotter.py
└── data/
    ├── raw/                # Input data files
    └── results/            # Output results
```

## Contributing

When contributing to this project:

1. Make sure your code works on both macOS and Linux
2. Test the setup process from a fresh clone
3. Update this README if you add new features or change the setup process

## License

[Add your license information here]

## Support

For issues or questions, please [open an issue on GitHub] or contact the project maintainers.

