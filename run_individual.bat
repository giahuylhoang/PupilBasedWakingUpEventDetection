@echo off
REM Run individual analysis script for Windows

setlocal enabledelayedexpansion

REM Run init.bat to set up the environment
call init.bat
if %errorlevel% neq 0 (
    echo [ERROR] Failed to initialize environment
    exit /b 1
)

REM Default values
set "DATA_FOLDER_PATH=data\raw\test\cycle_10"
set "RESULTS_FOLDER=data\results\test\cycle_10"
set BSLINE_LENGTH=10
set EVENT_LENGTH=20
set THRESHOLD_TO_EXCLUDE_FROM_MIN_MAX=1
set THRESHOLD_TO_EXCLUDE_BASE_ON_PUPIL=2
set PLOT_TRACES=false
set SAVE_TRACE_PLOT=true
set CLEAR_OUTPUT=false
set WAKE_UP=true
set INTERACTIVE_PLOTS=true

REM Parse command-line arguments
:parse_args
if "%~1"=="" goto :run_script
if /i "%~1"=="-d" (
    set "DATA_FOLDER_PATH=%~2"
    shift
    shift
    goto :parse_args
)
if /i "%~1"=="--data_folder_path" (
    set "DATA_FOLDER_PATH=%~2"
    shift
    shift
    goto :parse_args
)
if /i "%~1"=="-r" (
    set "RESULTS_FOLDER=%~2"
    shift
    shift
    goto :parse_args
)
if /i "%~1"=="--results_folder" (
    set "RESULTS_FOLDER=%~2"
    shift
    shift
    goto :parse_args
)
if /i "%~1"=="--threshold_min_max" (
    set THRESHOLD_TO_EXCLUDE_FROM_MIN_MAX=%~2
    shift
    shift
    goto :parse_args
)
if /i "%~1"=="--threshold_pupil" (
    set THRESHOLD_TO_EXCLUDE_BASE_ON_PUPIL=%~2
    shift
    shift
    goto :parse_args
)
if /i "%~1"=="--plot_traces" (
    set PLOT_TRACES=%~2
    shift
    shift
    goto :parse_args
)
if /i "%~1"=="--save_trace_plot" (
    set SAVE_TRACE_PLOT=%~2
    shift
    shift
    goto :parse_args
)
if /i "%~1"=="--clear_output" (
    set CLEAR_OUTPUT=%~2
    shift
    shift
    goto :parse_args
)
if /i "%~1"=="--bsline_length" (
    set BSLINE_LENGTH=%~2
    shift
    shift
    goto :parse_args
)
if /i "%~1"=="--event_length" (
    set EVENT_LENGTH=%~2
    shift
    shift
    goto :parse_args
)
if /i "%~1"=="--wake_up" (
    set WAKE_UP=%~2
    shift
    shift
    goto :parse_args
)
if /i "%~1"=="--interactive_plots" (
    set INTERACTIVE_PLOTS=%~2
    shift
    shift
    goto :parse_args
)
if /i "%~1"=="-h" goto :usage
if /i "%~1"=="--help" goto :usage

REM Unknown argument
echo [ERROR] Unknown argument: %~1
goto :usage

:usage
echo Usage: %~nx0 -d ^<data_folder_path^> [-r ^<results_folder^>] [options]
echo.
echo Options:
echo   -d, --data_folder_path      Path to data folder (required)
echo   -r, --results_folder        Path to save results
echo   --threshold_min_max         Percentile threshold for outlier exclusion (default: 1)
echo   --threshold_pupil           Threshold for excluding events based on pupil (default: 2)
echo   --plot_traces               Generate plots of traces (default: false)
echo   --save_trace_plot           Save generated trace plots (default: true)
echo   --clear_output              Clear output after processing (default: false)
echo   --bsline_length             Baseline length in seconds (default: 10)
echo   --event_length              Event length in seconds (default: 20)
echo   --wake_up                   Wake-up detection mode (default: true)
echo   --interactive_plots         Show plots interactively (default: true)
echo.
exit /b 1

:run_script
REM Check if the required argument is provided
if "!DATA_FOLDER_PATH!"=="" (
    echo [ERROR] data_folder_path is required.
    goto :usage
)

REM Convert backslashes to forward slashes for Python (Python handles both, but be consistent)
set "DATA_FOLDER_PATH=!DATA_FOLDER_PATH:\=/!"
set "RESULTS_FOLDER=!RESULTS_FOLDER:\=/!"

REM Run the Python script
python scripts\run_individual.py "!DATA_FOLDER_PATH!" ^
    --results_folder "!RESULTS_FOLDER!" ^
    --threshold_to_exclude_from_min_max !THRESHOLD_TO_EXCLUDE_FROM_MIN_MAX! ^
    --threshold_to_exclude_base_on_pupil !THRESHOLD_TO_EXCLUDE_BASE_ON_PUPIL! ^
    --plot_traces !PLOT_TRACES! ^
    --save_trace_plot !SAVE_TRACE_PLOT! ^
    --clear_output !CLEAR_OUTPUT! ^
    --bsline_length !BSLINE_LENGTH! ^
    --event_length !EVENT_LENGTH! ^
    --wake_up !WAKE_UP! ^
    --interactive_plots !INTERACTIVE_PLOTS!

endlocal

