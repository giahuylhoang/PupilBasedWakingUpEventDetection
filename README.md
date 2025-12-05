# Pupil-Based Waking Up Event Detection

A Python-based analysis tool for detecting waking up events based on pupil data and other physiological measurements.

## Prerequisites

- **Python 3.7 or higher** (Python 3.8+ recommended)
- **macOS, Linux, or Windows** operating system
- **Bash shell** (macOS/Linux) or **Command Prompt/PowerShell** (Windows)

### Check Prerequisites

Before setting up, check if you have everything needed:

**On macOS/Linux:**
```bash
chmod +x check_prerequisites.sh
./check_prerequisites.sh
```

**On Windows:**
```cmd
check_prerequisites.bat
```

If Python is not installed:
- **macOS/Linux**: See **[INSTALL_PYTHON.md](INSTALL_PYTHON.md)** for detailed installation instructions
- **Windows**: See **[INSTALL_WINDOWS.md](INSTALL_WINDOWS.md)** for Windows-specific installation instructions

## Quick Setup

### For New Users (First Time Setup)

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd PupilBasedWakingUpEventDetection
   ```

2. **Run the setup script (it handles everything):**

   **On macOS/Linux:**
   ```bash
   chmod +x setup.sh
   ./setup.sh
   ```

   **On Windows:**
   ```cmd
   setup.bat
   ```
   
   Or using PowerShell (better error handling):
   ```powershell
   powershell -ExecutionPolicy Bypass -File setup.ps1
   ```

   The setup script automatically:
   - ✓ Checks if Python 3 is installed (if not, shows installation instructions)
   - ✓ Checks for pip and venv (tries to auto-install if missing)
   - ✓ Creates a virtual environment
   - ✓ Installs all required dependencies from `requirements.txt`
   - ✓ Verifies all packages are working correctly
   - ✓ Provides clear error messages if anything is missing
   
   **Note:** If Python 3 is not installed, the script will detect your OS and show you exactly how to install it. After installing Python, just run the setup script again.

3. **Test the setup:**
   ```bash
   chmod +x test_setup.sh
   ./test_setup.sh
   ```
   This will verify that everything is installed and working correctly.

4. **You're ready to go!** Proceed to "Running the Analysis" below.

### Alternative Manual Setup

If you prefer to set up manually:

**On macOS/Linux:**
```bash
source init.sh
```

**On Windows:**
```cmd
init.bat
```

This script will:
- Create a virtual environment if it doesn't exist
- Install dependencies from `requirements.txt`
- Set up PYTHONPATH automatically

**Verify installation:**
```bash
# macOS/Linux
python3 --version
pip list

# Windows
python --version
pip list
```

## Running the Analysis

### Web Interface (Recommended for Easy Use)

The easiest way to run analyses is through the web interface:

1. **Start the web server:**
   
   **On macOS/Linux:**
   ```bash
   chmod +x run_webapp.sh
   ./run_webapp.sh
   ```
   
   **On Windows:**
   ```cmd
   run_webapp.bat
   ```

2. **Open your web browser:**
   - The app will start on `http://localhost:5000` (or the next available port)
   - The terminal will show the exact URL to use

3. **Use the web interface:**
   - **Individual Processing**: Process a single folder with interactive folder browser
   - **Batch Processing**: Process multiple folders at once
   - **No command line needed!** Just click "Browse" to select folders and fill in parameters

**Features:**
- ✅ Interactive folder browser (no need to type paths manually)
- ✅ Visual indicators for folders containing CSV files
- ✅ Real-time processing status updates
- ✅ Easy parameter configuration through forms
- ✅ Supports both individual and batch processing

### Run Individual Analysis (Command Line)

The simplest way to run an analysis:

**On macOS/Linux:**
```bash
./run_individual.sh -d "data/raw/test/cycle_9"
```

**On Windows:**
```cmd
run_individual.bat -d "data\raw\test\cycle_9"
```

**Full usage:**
```bash
# macOS/Linux
./run_individual.sh -d <data_folder_path> [options]

# Windows
run_individual.bat -d <data_folder_path> [options]
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

**On macOS/Linux:**
```bash
./run_individual.sh \
  -d "data/raw/test/cycle_9" \
  -r "data/results/test/cycle_9" \
  --bsline_length 10 \
  --event_length 20 \
  --wake_up false
```

**On Windows:**
```cmd
run_individual.bat -d "data\raw\test\cycle_9" -r "data\results\test\cycle_9" --bsline_length 10 --event_length 20 --wake_up false
```

### Run Batch Analysis (Command Line)

Process multiple folders at once:

**On macOS/Linux:**
```bash
# Process all folders with CSV files in a directory
./run_batch.sh -r "data/raw/test" -o "data/results/batch_run"

# Process specific folders
./run_batch.sh --folders "data/raw/test/cycle_9" "data/raw/test/cycle_10" -o "data/results/batch_run"
```

**On Windows:**
```cmd
REM Process all folders with CSV files in a directory
run_batch.bat -r "data\raw\test" -o "data\results\batch_run"

REM Process specific folders
run_batch.bat --folders "data\raw\test\cycle_9" "data\raw\test\cycle_10" -o "data\results\batch_run"
```

**Batch processing options:**
- `-r` or `--root_folder`: Parent folder to search for all folders containing CSV files
- `--folders`: List of specific folders to process (alternative to `-r`)
- `-o` or `--output`: Output folder for results (required, must be separate from data folders)
- All other options are the same as individual processing (see above)

### Using Python Script Directly

If you prefer to run the Python script directly:

1. **Activate the environment first:**
   
   **On macOS/Linux:**
   ```bash
   source init.sh
   ```
   
   **On Windows:**
   ```cmd
   init.bat
   ```

2. **Run the script:**
   
   **On macOS/Linux:**
   ```bash
   python3 scripts/run_individual.py "data/raw/test/cycle_9" \
     --results_folder "data/results/test/cycle_9" \
     --bsline_length 10 \
     --event_length 20
   ```
   
   **On Windows:**
   ```cmd
   python scripts\run_individual.py "data\raw\test\cycle_9" --results_folder "data\results\test\cycle_9" --bsline_length 10 --event_length 20
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

**On macOS/Linux:**
```bash
# 1. Make scripts executable (if needed)
chmod +x setup.sh test_setup.sh init.sh run_individual.sh

# 2. Run the setup
./setup.sh

# 3. Test everything is working
./test_setup.sh
```

**On Windows:**
```cmd
REM 1. Run the setup
setup.bat

REM 2. Test everything is working (if test_setup.bat exists)
test_setup.bat
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

1. **Run the prerequisite checker:**
   
   **On macOS/Linux:**
   ```bash
   ./check_prerequisites.sh
   ```
   
   **On Windows:**
   ```cmd
   check_prerequisites.bat
   ```
   
   This will detect your OS and show specific installation instructions.

2. **Or see the detailed guide:**
   - **macOS/Linux**: See **[INSTALL_PYTHON.md](INSTALL_PYTHON.md)** for comprehensive installation instructions
   - **Windows**: See **[INSTALL_WINDOWS.md](INSTALL_WINDOWS.md)** for Windows-specific installation instructions

3. **Quick installation commands:**

   **On macOS:**
   ```bash
   # Using Homebrew (recommended)
   brew install python3
   
   # Or download from python.org
   # Visit: https://www.python.org/downloads/
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
├── init.sh / init.bat              # Environment initialization script
├── setup.sh / setup.bat / setup.ps1 # First-time setup script
├── run_individual.sh / run_individual.bat  # Individual analysis script
├── run_batch.sh / run_batch.bat            # Batch processing script
├── run_webapp.sh / run_webapp.bat         # Web application script
├── check_prerequisites.sh / check_prerequisites.bat  # Prerequisites checker
├── requirements.txt                # Python dependencies
├── README.md                        # This file
├── INSTALL_PYTHON.md               # Python installation guide (macOS/Linux)
├── INSTALL_WINDOWS.md              # Windows installation guide
├── scripts/
│   ├── run_individual.py           # Main Python analysis script
│   └── run_batch.py                # Batch processing script
├── src/
│   ├── data/
│   │   └── data_loader.py          # Data loading utilities
│   ├── utils/
│   │   ├── data_processing.py
│   │   ├── event_detection.py
│   │   └── utilities.py
│   └── visualization/
│       └── plotter.py
└── data/
    ├── raw/                         # Input data files
    └── results/                     # Output results
```

## Contributing

When contributing to this project:

1. Make sure your code works on macOS, Linux, and Windows
2. Test the setup process from a fresh clone on all supported platforms
3. Update this README if you add new features or change the setup process
4. For Windows-specific changes, test with both batch files (.bat) and PowerShell scripts (.ps1)

## License

[Add your license information here]

## Support

For issues or questions, please [open an issue on GitHub] or contact the project maintainers.

