@echo off
REM Run batch analysis script for Windows

setlocal enabledelayedexpansion

REM Run init.bat to set up the environment
call init.bat
if %errorlevel% neq 0 (
    echo [ERROR] Failed to initialize environment
    exit /b 1
)

REM Default values
set "ROOT_FOLDER=data\raw\test"
set "LIST_OF_FOLDERS="
set "RESULTS_PATH=data\results"
set BSLINE_LENGTH=5
set EVENT_LENGTH=15
set THRESHOLD_TO_EXCLUDE_FROM_MIN_MAX=1
set THRESHOLD_TO_EXCLUDE_BASE_ON_PUPIL=2
set PLOT_TRACES=true
set SAVE_TRACE_PLOT=true
set CLEAR_OUTPUT=false
set WAKE_UP=false
set INTERACTIVE_PLOTS=true

REM Parse command-line arguments
:parse_args
if "%~1"=="" goto :check_args
if /i "%~1"=="-r" (
    set "ROOT_FOLDER=%~2"
    set "LIST_OF_FOLDERS="
    shift
    shift
    goto :parse_args
)
if /i "%~1"=="--root_folder" (
    set "ROOT_FOLDER=%~2"
    set "LIST_OF_FOLDERS="
    shift
    shift
    goto :parse_args
)
if /i "%~1"=="--folders" (
    set "LIST_OF_FOLDERS="
    shift
    :collect_folders
    if "%~1"=="" goto :parse_args
    if /i "%~1"=="-o" goto :parse_args
    if /i "%~1"=="--output" goto :parse_args
    if /i "%~1"=="--results_path" goto :parse_args
    if /i "%~1"=="--default_result_path" goto :parse_args
    if "!LIST_OF_FOLDERS!"=="" (
        set "LIST_OF_FOLDERS=%~1"
    ) else (
        set "LIST_OF_FOLDERS=!LIST_OF_FOLDERS! %~1"
    )
    set "ROOT_FOLDER="
    shift
    goto :collect_folders
)
if /i "%~1"=="-o" (
    set "RESULTS_PATH=%~2"
    shift
    shift
    goto :parse_args
)
if /i "%~1"=="--output" (
    set "RESULTS_PATH=%~2"
    shift
    shift
    goto :parse_args
)
if /i "%~1"=="--results_path" (
    set "RESULTS_PATH=%~2"
    shift
    shift
    goto :parse_args
)
if /i "%~1"=="--default_result_path" (
    set "RESULTS_PATH=%~2"
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

:check_args
REM Check if the required argument is provided
if "!RESULTS_PATH!"=="" (
    echo [ERROR] Output path (-o or --output) is required.
    echo.
    goto :usage
)

REM Check if at least one of the required arguments is provided
if "!ROOT_FOLDER!"=="" (
    if "!LIST_OF_FOLDERS!"=="" (
        echo [ERROR] Either --root_folder (-r) or --folders must be provided.
        echo.
        goto :usage
    )
)

REM Warn if results path might overlap with data folders
if not "!ROOT_FOLDER!"=="" (
    echo !RESULTS_PATH! | findstr /C:"!ROOT_FOLDER!" >nul
    if !errorlevel! equ 0 (
        echo [WARNING] Results path '!RESULTS_PATH!' appears to be inside or overlap with data folder '!ROOT_FOLDER!'
        echo This may cause issues. Consider using a completely separate location (e.g., 'data\results\batch_run')
        echo.
    )
)

REM Convert backslashes to forward slashes for Python (Python handles both, but be consistent)
set "ROOT_FOLDER=!ROOT_FOLDER:\=/!"
set "RESULTS_PATH=!RESULTS_PATH:\=/!"
set "LIST_OF_FOLDERS=!LIST_OF_FOLDERS:\=/!"

REM Run the Python script
if not "!ROOT_FOLDER!"=="" (
    python scripts\run_batch.py --root_folder "!ROOT_FOLDER!" --results_path "!RESULTS_PATH!" ^
        --threshold_to_exclude_from_min_max !THRESHOLD_TO_EXCLUDE_FROM_MIN_MAX! ^
        --threshold_to_exclude_base_on_pupil !THRESHOLD_TO_EXCLUDE_BASE_ON_PUPIL! ^
        --plot_traces !PLOT_TRACES! ^
        --save_trace_plot !SAVE_TRACE_PLOT! ^
        --clear_output !CLEAR_OUTPUT! ^
        --bsline_length !BSLINE_LENGTH! ^
        --event_length !EVENT_LENGTH! ^
        --wake_up !WAKE_UP! ^
        --interactive_plots !INTERACTIVE_PLOTS!
) else (
    python scripts\run_batch.py --folders !LIST_OF_FOLDERS! --results_path "!RESULTS_PATH!" ^
        --threshold_to_exclude_from_min_max !THRESHOLD_TO_EXCLUDE_FROM_MIN_MAX! ^
        --threshold_to_exclude_base_on_pupil !THRESHOLD_TO_EXCLUDE_BASE_ON_PUPIL! ^
        --plot_traces !PLOT_TRACES! ^
        --save_trace_plot !SAVE_TRACE_PLOT! ^
        --clear_output !CLEAR_OUTPUT! ^
        --bsline_length !BSLINE_LENGTH! ^
        --event_length !EVENT_LENGTH! ^
        --wake_up !WAKE_UP! ^
        --interactive_plots !INTERACTIVE_PLOTS!
)

endlocal

:usage
echo Usage: %~nx0 [-r ^<root_folder^>] [--folders ^<folder1 folder2 ...^>] -o ^<results_path^> [options]
echo.
echo Options:
echo   -r, --root_folder       Parent folder to search for all folders containing CSV files
echo   --folders               List of specific folders to process (alternative to -r)
echo   -o, --output            Output folder for results (required, must be separate from data folders)
echo   --threshold_min_max     Percentile threshold for outlier exclusion (default: 1)
echo   --threshold_pupil         Threshold for excluding events based on pupil (default: 2)
echo   --plot_traces           Generate plots of traces (default: true)
echo   --save_trace_plot       Save generated trace plots (default: true)
echo   --clear_output          Clear output after processing (default: false)
echo   --bsline_length         Baseline length in seconds (default: 5)
echo   --event_length          Event length in seconds (default: 15)
echo   --wake_up               Wake-up detection mode (default: false)
echo   --interactive_plots     Show plots interactively (default: true)
echo.
echo Examples:
echo   # Process all folders with CSV files in data\raw\test:
echo   %~nx0 -r data\raw\test -o data\results\batch_run
echo.
echo   # Process specific folders:
echo   %~nx0 --folders data\raw\test\cycle_9 data\raw\test\cycle_10 -o data\results\batch_run
echo.
exit /b 1

