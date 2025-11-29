import pandas as pd
import numpy as np
import os

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
    raw_df = pd.read_csv(whisker_file, header=None)
    
    # Extract only the first column as whisker_gradient (handles files with multiple columns)
    # Convert to numeric, coercing errors to NaN
    whisker_gradient = pd.to_numeric(raw_df.iloc[:, 0], errors='coerce').values
    
    # Remove NaN values from the gradient data
    mask = ~pd.isna(whisker_gradient)
    whisker_gradient = whisker_gradient[mask]
    num_points = len(whisker_gradient)
    
    if num_points == 0:
        raise ValueError(f"Whisker data file is empty or contains only NaN/non-numeric values: {whisker_file}")
    
    # Ensure it's a numeric array (float64)
    whisker_gradient = whisker_gradient.astype(np.float64)
    
    # Create a new DataFrame with only whisker_gradient and time columns
    resampled_whisker_angle_df = pd.DataFrame({
        'whisker_gradient': whisker_gradient,
        'time': np.linspace(0, 900, num_points)
    })

    return resampled_whisker_angle_df