import pandas as pd
import numpy as np
import os

def load_data(file_path):
    return pd.read_csv(file_path)

def load_arteriole_data(data_folder_path):
    arteriole_diameter_df = pd.read_csv(os.path.join(data_folder_path, 'arteriole_diameter.csv'))
    arteriole_diameter_df.columns = ['time', 'arteriole_diameter']
    arteriole_diameter_df.dropna(inplace=True)
    return arteriole_diameter_df

def load_calcium_data(data_folder_path):
    calcium_df = pd.read_csv(os.path.join(data_folder_path, 'calcium.csv'))
    calcium_df.columns = ['time', 'calcium']
    calcium_df.dropna(inplace=True)
    return calcium_df

def load_pupil_data(data_folder_path):
    pupil_size_df = pd.read_csv(os.path.join(data_folder_path, 'pupil_size.csv'))
    pupil_size_df.columns = ['time', 'pupil_size']
    pupil_size_df.dropna(inplace=True)
    return pupil_size_df

def load_whisker_data(data_folder_path):
    # Find the first file with 'whisker' in its name and ends with .csv
    for file in os.listdir(data_folder_path):
        if 'whisker' in file and file.endswith('.csv'):
            whisker_file_path = os.path.join(data_folder_path, file)
            break
    print(f"Loading whisker data from: {whisker_file_path}")
    resampled_whisker_angle_df = pd.read_csv(whisker_file_path, header=None)
    resampled_whisker_angle_df.dropna(inplace=True)
    resampled_whisker_angle_df['time'] = np.linspace(0, 900, resampled_whisker_angle_df[0].shape[0])
    resampled_whisker_angle_df.columns = ['whisker_angle', 'time']

    return resampled_whisker_angle_df