import pandas as pd
import numpy as np
import os
from src.utils.utilities import _load_csv_with_optional_header

def load_data(file_path):
    return pd.read_csv(file_path)

def _find_file_by_pattern(data_folder_path, pattern):
    """
    Helper function to find the first CSV file containing the pattern in its name.
    
    Args:
        data_folder_path: Path to the folder to search
        pattern: String pattern to search for in filenames
    
    Returns:
        Full path to the matching file, or None if not found
    """
    if not os.path.isdir(data_folder_path):
        raise ValueError(f"Data folder path does not exist: {data_folder_path}")
    
    for file in os.listdir(data_folder_path):
        if pattern.lower() in file.lower() and file.endswith('.csv'):
            return os.path.join(data_folder_path, file)
    return None

def load_arteriole_data(data_folder_path):
    # Find file containing 'arteriole' in its name
    arteriole_file = _find_file_by_pattern(data_folder_path, 'arteriole')
    if arteriole_file is None:
        raise FileNotFoundError(f"No file containing 'arteriole' found in {data_folder_path}")
    
    print(f"Loading arteriole data from: {arteriole_file}")
    arteriole_diameter_df = pd.read_csv(arteriole_file)
    arteriole_diameter_df.columns = ['time', 'arteriole_diameter']
    # Convert to numeric types
    arteriole_diameter_df['time'] = pd.to_numeric(arteriole_diameter_df['time'], errors='coerce')
    arteriole_diameter_df['arteriole_diameter'] = pd.to_numeric(arteriole_diameter_df['arteriole_diameter'], errors='coerce')
    arteriole_diameter_df.dropna(inplace=True)
    return arteriole_diameter_df

def load_calcium_data(data_folder_path):
    # Find file containing 'calcium' in its name
    calcium_file = _find_file_by_pattern(data_folder_path, 'calcium')
    if calcium_file is None:
        raise FileNotFoundError(f"No file containing 'calcium' found in {data_folder_path}")
    
    print(f"Loading calcium data from: {calcium_file}")
    calcium_df = pd.read_csv(calcium_file)
    calcium_df.columns = ['time', 'calcium']
    # Convert to numeric types
    calcium_df['time'] = pd.to_numeric(calcium_df['time'], errors='coerce')
    calcium_df['calcium'] = pd.to_numeric(calcium_df['calcium'], errors='coerce')
    calcium_df.dropna(inplace=True)
    return calcium_df

def load_pupil_data(data_folder_path):
    # Find file containing 'pupil' in its name
    pupil_file = _find_file_by_pattern(data_folder_path, 'pupil')
    if pupil_file is None:
        raise FileNotFoundError(f"No file containing 'pupil' found in {data_folder_path}")
    
    print(f"Loading pupil data from: {pupil_file}")
    pupil_size_df = pd.read_csv(pupil_file)
    pupil_size_df.columns = ['time', 'pupil_size']
    # Convert to numeric types
    pupil_size_df['time'] = pd.to_numeric(pupil_size_df['time'], errors='coerce')
    pupil_size_df['pupil_size'] = pd.to_numeric(pupil_size_df['pupil_size'], errors='coerce')
    pupil_size_df.dropna(inplace=True)
    return pupil_size_df

def load_whisker_data(data_folder_path):
    # Find file containing 'whisker' in its name
    whisker_file = _find_file_by_pattern(data_folder_path, 'whisker')
    if whisker_file is None:
        raise FileNotFoundError(f"No file containing 'whisker' found in {data_folder_path}")
    
    print(f"Loading whisker data from: {whisker_file}")
    # Read CSV file without header

    raw_df = _load_csv_with_optional_header(whisker_file)
    
    # Expect two columns: first is time, second is whisker_gradient
    whisker_time = pd.to_numeric(raw_df.iloc[:, 0], errors='coerce').values
    whisker_gradient = pd.to_numeric(raw_df.iloc[:, 1], errors='coerce').values
    
    # Interpolate NaN values in whisker_gradient and whisker_time using average of adjacent rows
    whisker_gradient = whisker_gradient.astype(np.float64)
    whisker_time = whisker_time.astype(np.float64)

    # Helper function to interpolate nan by average of adjacent rows
    def interpolate_adjacent(arr):
        nan_mask = np.isnan(arr)
        if np.any(nan_mask):
            arr = arr.copy()
            for idx in np.where(nan_mask)[0]:
                prev_idx = idx - 1
                next_idx = idx + 1
                while prev_idx >= 0 and np.isnan(arr[prev_idx]):
                    prev_idx -= 1
                while next_idx < len(arr) and np.isnan(arr[next_idx]):
                    next_idx += 1
                prev_val = arr[prev_idx] if prev_idx >= 0 else np.nan
                next_val = arr[next_idx] if next_idx < len(arr) else np.nan
                if not np.isnan(prev_val) and not np.isnan(next_val):
                    arr[idx] = (prev_val + next_val) / 2.0
                elif not np.isnan(prev_val):
                    arr[idx] = prev_val
                elif not np.isnan(next_val):
                    arr[idx] = next_val
                else:
                    arr[idx] = 0.0  # fallback, but should not happen in normal data
        return arr

    whisker_gradient = interpolate_adjacent(whisker_gradient)
    whisker_time = interpolate_adjacent(whisker_time)

    # # Remove any rows where either value is still nan
    # valid_mask = ~np.isnan(whisker_time) & ~np.isnan(whisker_gradient)
    # whisker_time = whisker_time[valid_mask]
    # whisker_gradient = whisker_gradient[valid_mask]

    num_points = len(whisker_gradient)
    if num_points == 0:
        raise ValueError(f"Whisker data file is empty or contains only NaN/non-numeric values after interpolation: {whisker_file}")

    resampled_whisker_angle_df = pd.DataFrame({
        'whisker_gradient': whisker_gradient,
        'time': whisker_time
    })



    return resampled_whisker_angle_df