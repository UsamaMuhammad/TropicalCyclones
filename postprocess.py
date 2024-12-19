import torch
import pandas as pd
import torch.nn.functional as F
from model import prepare_combined_data, Autoencoder
import pickle
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import simps
from scipy.stats import ttest_ind
from scipy.ndimage import gaussian_filter1d
import os
from matplotlib.dates import DateFormatter, MonthLocator, DayLocator, AutoDateLocator



def prepare_combined_data(filtered_data, sequence_length, iqr_threshold=3, smoothing="rolling", window=3):
    """
    Combines sequences from all hurricanes for training, with outlier handling and smoothing.

    Parameters:
        filtered_data (list): List of dictionaries containing filtered data for each hurricane and estuary.
        sequence_length (int): Length of each sequence for LSTM input.
        iqr_threshold (float): Threshold for outlier handling (default is 1.5 IQR).
        smoothing (str): Smoothing method ('rolling', 'exponential', 'gaussian').
        window (int): Window size for smoothing (default is 3).

    Returns:
        combined_sequences (torch.Tensor): Combined sequences for all events.
        scalers (list): List of scalers for each entry.
    """
    combined_sequences = []
    scalers = []

    for entry in filtered_data:
        df = entry["Data"].copy()

        # Handle outliers using the IQR method
        #q1 = df.quantile(0.25)
        #q3 = df.quantile(0.75)
        #iqr = q3 - q1
        #lower_bound = q1 - iqr_threshold * iqr
        #upper_bound = q3 + iqr_threshold * iqr
        #df = df.clip(lower=lower_bound, upper=upper_bound, axis=1)

        # Apply smoothing
        if smoothing == "rolling":
            df = df.rolling(window=window, min_periods=1, center=True).mean()
        elif smoothing == "exponential":
            df = df.ewm(span=window, adjust=False).mean()
        elif smoothing == "gaussian":
            from scipy.ndimage import gaussian_filter1d
            df = pd.DataFrame(
                gaussian_filter1d(df.values, sigma=window),
                index=df.index,
                columns=df.columns,
            )
        elif smoothing is not None:
            raise ValueError(f"Unknown smoothing type: {smoothing}")

        # Scale the data using RobustScaler
        scaler = RobustScaler()
        scaled_data = scaler.fit_transform(df.values)
        scalers.append(scaler)

        # Convert to PyTorch tensor and create sequences
        tensor_data = torch.tensor(scaled_data, dtype=torch.float32).unsqueeze(0)  # Add batch dimension
        sequences = [
            tensor_data[:, i:i + sequence_length, :]
            for i in range(0, tensor_data.size(1) - sequence_length + 1, sequence_length)
        ]
        combined_sequences.extend(sequences)

    # Combine sequences into a single tensor
    combined_sequences = torch.cat(combined_sequences, dim=0)
    return combined_sequences, scalers




# Load the data dictionary
def load_filtered_data(file_path):
    with open(file_path, "rb") as file:
        return pickle.load(file)


#################
#################
#### Data plots
#################
#################

def plot_data_with_customization(sub_data, output_folder, columnslabel):
    """
    Plots raw values of Temperature, Salinity, DO, and Turbidity in 4 separate subplots with custom colors and legends.

    Parameters:
    - sub_data: DataFrame containing 'Temperature', 'Salinity', 'DO', and 'Turbidity' columns.
    - output_folder: Directory where the plots will be saved.
    - columnslabel: String used in the filename for saving the plot.
    """
    features = ['Temperature', 'Salinity', 'DO', 'Turbidity']
    # Features to plot and their corresponding colors
    
    #normalized_data = sub_data.copy()
    for feature in features:
        sub_data[feature] = (
            sub_data[feature] - sub_data[feature].min()
        ) / (sub_data[feature].max() - sub_data[feature].min())

    features = {
        'Temperature': 'blue',
        'Salinity': 'orange',
        'DO': 'green',
        'Turbidity': 'red',
    }

    # Create subplots
    fig, axes = plt.subplots(nrows=4, ncols=1, figsize=(12, 8), sharex=True)
    for i, (feature, color) in enumerate(features.items()):
        ax = axes[i]
        ax.plot(sub_data['datetime'], sub_data[feature], label=feature, color=color, linestyle='-')
        ax.set_ylabel(feature, fontsize=10)
        ax.legend(loc='upper right', fontsize=10)
        ax.grid(True)
        if i == 3:  # Apply x-axis labels to the last subplot only
            ax.set_xlabel("Timeline", fontsize=12)
        else:
            ax.set_xticks([])  # Remove xticks for other subplots

    # Highlight the cyclone period in all subplots
    cyclone_subset = sub_data[sub_data['cyclone_Status'] == 'cyclone']
    if not cyclone_subset.empty:
        cyclone_start = cyclone_subset['datetime'].min()
        cyclone_end = cyclone_subset['datetime'].max()
        for ax in axes:
            ax.axvspan(cyclone_start, cyclone_end, facecolor='red', alpha=0.3, label='Cyclone Period')

    # Customize x-axis ticks for the last subplot
    ax = axes[3]
    total_days = (sub_data['datetime'].max() - sub_data['datetime'].min()).days
    if total_days > 365:  # More than 1 year
        ax.xaxis.set_major_locator(MonthLocator(interval=3))  # One tick every 3 months
        ax.xaxis.set_major_formatter(DateFormatter("%b %Y"))  # Format as "Jan 2021"
    elif total_days > 90:  # More than 3 months
        ax.xaxis.set_major_locator(MonthLocator(interval=1))  # One tick every month
        ax.xaxis.set_major_formatter(DateFormatter("%b %Y"))  # Format as "Jan 2021"
    elif total_days > 30:  # More than 1 month
        ax.xaxis.set_major_locator(DayLocator(interval=15))  # One tick every 15 days
        ax.xaxis.set_major_formatter(DateFormatter("%d %b %Y"))  # Format as "01 Jan 2021"
    else:  # Less than 1 month
        ax.xaxis.set_major_locator(DayLocator(interval=7))  # One tick every 7 days
        ax.xaxis.set_major_formatter(DateFormatter("%d %b %Y"))  # Format as "01 Jan")

    # Ensure the output folder exists
    os.makedirs(output_folder, exist_ok=True)

    # Save the plot
    filename = os.path.join(output_folder, f"customized_plot_{columnslabel}.png")
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()

    print(f"Plot saved to {filename}")


def plot_cyclone_recovery(tot, sub_reconstructedData, global_peak_index, recoveryIndex, columnslabel, output_folder):
    """
    Create and save a plot for cyclone disturbance and recovery phases.

    Parameters:
    - tot: DataFrame containing the 'errors' for the full dataset
    - sub_reconstructedData: DataFrame containing 'errors' and 'serr' (smoothed errors) for the subset
    - global_peak_index: Integer index for the global peak in the subset
    - recoveryIndex: Integer index for the recovery phase in the subset
    - columnslabel: String used in the filename for saving the plot
    - output_folder: Directory where the plot will be saved.
    """
    os.makedirs(output_folder, exist_ok=True)
    
    # Create the figure
    plt.figure(figsize=(12, 6))

    # Plot the reconstruction error
    plt.plot(tot.index, tot['errors'], label='Reconstruction Error', linestyle='-', marker='o', color='blue')

    # Plot the smoothed error
    plt.plot(sub_reconstructedData.index, sub_reconstructedData['serr'], label='Smoothed Error', linestyle='-', marker='x', color='orange')

    # Highlight disturbance phase
    plt.axvspan(
        sub_reconstructedData.index[0],
        sub_reconstructedData.index[global_peak_index],
        color='red', alpha=0.3, label='Disturbance Phase'
    )

    # Highlight recovery phase
    plt.axvspan(
        sub_reconstructedData.index[global_peak_index],
        sub_reconstructedData.index[recoveryIndex],
        color='green', alpha=0.3, label='Recovery Phase'
    )

    # Add title and labels
    plt.xlabel("Timeline")
    plt.ylabel("Reconstruction Error")

    # Format x-axis ticks to show one tick per month
    ax = plt.gca()
    total_days = (tot.index[-1] - tot.index[0]).days
    
    if total_days > 365:  # More than 1 year
        ax.xaxis.set_major_locator(MonthLocator(interval=3))  # One tick every 3 months
        ax.xaxis.set_major_formatter(DateFormatter("%b %Y"))  # Format as "Jan 2021"
    elif total_days > 90:  # More than 3 months
        ax.xaxis.set_major_locator(MonthLocator(interval=1))  # One tick every month
        ax.xaxis.set_major_formatter(DateFormatter("%b %Y"))  # Format as "Jan 2021"
    elif total_days > 30:  # More than 1 month
        ax.xaxis.set_major_locator(DayLocator(interval=15))  # One tick every 15 days
        ax.xaxis.set_major_formatter(DateFormatter("%d %b %Y"))  # Format as "01 Jan 2021"
    else:  # Less than 1 month
        ax.xaxis.set_major_locator(DayLocator(interval=7))  # One tick every 7 days
        ax.xaxis.set_major_formatter(DateFormatter("%d %b %Y"))  # Format as "01 Jan"
    
    # Force matplotlib to display all ticks
    ax.xaxis.set_minor_locator(AutoDateLocator())  # Optional: add minor ticks

    # Rotate x-axis labels for better readability
    plt.xticks(rotation=0)
    # Format x-axis ticks to show one tick per month
    # Show grid
    plt.grid(True)

    # Add legend
    plt.legend(loc='upper right')

    # Save the plot
    filename = os.path.join(output_folder, f"cyclone_disturbance_recovery_{columnslabel}.png")
    plt.savefig(filename, bbox_inches='tight')
    plt.close()

    print(f"Plot saved to {filename}")




def merge_series(arr, threshold):
    iter = 0
    while iter < 50:
        # Find unique numbers in the array (excluding 0)
        unique_numbers = list(set(filter(lambda x: x != 0, arr)))

        # Initialize a dictionary to store the start and end indices for each unique number
        indices_dict = {num: {'start': None, 'end': None} for num in unique_numbers}

        # Iterate through the array to find the start and end indices for each unique number
        for i, num in enumerate(arr):
            if num != 0:
                if indices_dict[num]['start'] is None:
                    indices_dict[num]['start'] = i
                indices_dict[num]['end'] = i

        iter +=  1
        #print(iter)
        merged = False  # Flag to check if any merging occurred

        # Merge series based on the specified threshold
        for i in range(len(unique_numbers) - 1):
            current_num = unique_numbers[i]
            next_num = unique_numbers[i + 1]

            end_index_current = indices_dict[current_num]['end']
            start_index_next = indices_dict[next_num]['start']
            end_index_next = indices_dict[next_num]['end']

            if start_index_next - end_index_current <= threshold:
                # Merge the series by updating the elements in the specified range
                arr[end_index_current + 1:end_index_next+1] = [current_num] * (end_index_next - end_index_current)
                merged = True

        if not merged:
            break  # Break the loop if no merging occurred in this iteration

    return arr

def normalize_and_add_count(df, label_column, min_series_length=96, max_gap_between_series=3*240):
    df['hurricaneCount'] = 0

    # Identify the start of a new series
    start_of_series = (df[label_column] == 1) & (df[label_column].shift(1) == 0)

    # Cumulatively count series
    df['hurricaneCount'] = start_of_series.cumsum()
    df.loc[df[label_column] == 0, 'hurricaneCount'] = 0
    
    series_lengths = df.groupby('hurricaneCount').size()
    short_series = series_lengths[series_lengths <= 5].index
    
    df.loc[df['hurricaneCount'].isin(short_series), 'hurricaneCount'] = 0
    
    
    # Extract unique values from the column
    unique_values = df['hurricaneCount'].unique()
    
    # Create a mapping of unique values to their positions in the sorted array
    value_to_position = {value: position for position, value in enumerate(sorted(unique_values))}
    
    # Replace the values in the column with their positions
    df['hurricaneCount'] = df['hurricaneCount'].map(value_to_position)
    
    
    
    
    df['hurricaneCount'] = merge_series(np.array(df['hurricaneCount']), max_gap_between_series)

    # Calculate the length of each series
    series_lengths = df.groupby('hurricaneCount').size()

    # Identify and drop series with fewer than min_series_length observations
    short_series = series_lengths[series_lengths < min_series_length].index
    # Identify and replace elements of series with fewer than min_series_length observations with 0
    df.loc[df['hurricaneCount'].isin(short_series), 'hurricaneCount'] = 0
    
    # Extract unique values from the column
    unique_values = df['hurricaneCount'].unique()
    
    # Create a mapping of unique values to their positions in the sorted array
    value_to_position = {value: position for position, value in enumerate(sorted(unique_values))}
    
    # Replace the values in the column with their positions
    df['hurricaneCount'] = df['hurricaneCount'].map(value_to_position)
    

    return df[['hurricaneCount']]  # Return only the newLabel column

def is_strictly_concave(vector):
    """
    Check if a given vector is strictly concave.
    
    :param vector: Input vector as a list or a numpy array.
    :return: True if the vector is strictly concave, False otherwise.
    """
    # Calculate second differences (discrete approximation to second derivative)
    second_differences = np.diff(vector, n=2)
    
    # Check if all second differences are strictly negative
    return np.all(second_differences <= 0.0001)




def find_peaks_in_concave_segments(data, sigma=2):
    """
    Process the data to find peaks within concave segments.
    1. Smooth the data with Gaussian smoothing.
    2. Split the smoothed data into concave segments.
    3. Find peaks within each concave segment and count observations after the peak.
    
    :param data: Input data series.
    :param sigma: Standard deviation for Gaussian smoothing.
    :param min_prominence: Minimum prominence for peak detection.
    :return: Maximum peak and observations after the peak across all concave segments.
    
    """
    # Step 1: Smooth the data
    while not is_strictly_concave(data):
        data = gaussian_filter1d(data, sigma)
        sigma+=1
    #plt.plot(data)
    global_peak_index = np.argmax(data)
    afterPeak = data[global_peak_index:]
    recoveryIndex = np.argmin(afterPeak)+global_peak_index
    
    # Calculate the number of observations after the global peak
    observations_after_peak = len(data) - global_peak_index - 1
    
    return global_peak_index, recoveryIndex, observations_after_peak, data




def calculate_disturbance_magnitude(sub_reconstructed_data):
    """
    Calculate the disturbance magnitude based on the area under the error curve.

    Parameters:
    - sub_reconstructed_data: DataFrame with 'datetime', 'Estuary', 'errors' columns.

    Returns:
    - disturbance_magnitude: Magnitude of disturbances.
    """

    # Assuming 'sub_reconstructed_data' is your DataFrame
    sub_reconstructed_data = sub_reconstructed_data.reset_index(drop=True)  # Resetting the index
    
    # Now you can sort the DataFrame by 'datetime'
    sub_reconstructed_data = sub_reconstructed_data.sort_values(by='datetime')

    # Extract errors and corresponding timestamps
    errors = sub_reconstructed_data['errors'].values
    timestamps = sub_reconstructed_data.index.values

    # Calculate the area under the error curve using Simpson's rule
    disturbance_magnitude = simps(errors, timestamps)

    return disturbance_magnitude


def recond_Data(df, sequence_length,AE, threshold=0.1 ):
    

    
    sequences = []
    


    #df = entry["Data"].copy()
    
    # Handle outliers using the IQR method
    #q1 = df.quantile(0.25)
    #q3 = df.quantile(0.75)
    #iqr = q3 - q1
    #lower_bound = q1 - iqr_threshold * iqr
    #upper_bound = q3 + iqr_threshold * iqr
    #df = df.clip(lower=lower_bound, upper=upper_bound, axis=1)
    
    
    
    # Scale the data
    scaler = RobustScaler()
    scaled_data = scaler.fit_transform(df.values)
    scalers.append(scaler)
    
    # Convert to PyTorch tensor
    #tensor_data = torch.tensor(scaled_data, dtype=torch.float32).unsqueeze(0)  # Add batch dimension

    # Create sequences with a shift of 1
    for i in range(len(df) - sequence_length + 1):
        sequences.append(df.iloc[i:i + sequence_length].values)
        # Append metadata for this sequence
        

    model.eval()

    # Iterate through combined sequences and metadata
    labels = []
    errors = []
    with torch.no_grad():
        sequences = torch.tensor(sequences, dtype=torch.float32).to(device)

        # Pass through autoencoder
        reconstructed = model(sequences)

        # Compute per-sequence reconstruction errors
        reconstruction_error = torch.mean((reconstructed - sequences) ** 2, dim=(1, 2)).cpu().numpy()
        reconstruction_error = (reconstruction_error-reconstruction_error.min())/(reconstruction_error.max()-reconstruction_error.min())

        # Perform element-wise comparison to generate labels
        labels = (reconstruction_error > threshold).astype(int)  # 1 for error > threshold, 0 otherwise
        
        # If you need errors returned as well
        return labels, reconstruction_error
        


    return labels, reconstruction_error


########################
########################
########################

result_df = pd.DataFrame()

countEvents = []
disturbance_magnitudes = []
for entry in filtered_data:
    
    data_entry = entry["Data"]
    labels, errors  =  recond_Data(data_entry,168,autoencoder,0.3)
    estuary = entry.get("Estuary", "Unknown")
    hurricane = entry.get("Hurricane", "Unknown")
    year = entry.get("Year", "Unknown")
    columnslabel= estuary+ "_" + hurricane + "_" + str(year)
    
    labels = pd.DataFrame(labels)
    labels.columns = ['Label']
    
    result_df = pd.DataFrame()
    
    # Apply the function for each label
    result_df[columnslabel] = normalize_and_add_count(labels.copy(), 'Label',min_series_length=12, max_gap_between_series=168*3)['hurricaneCount']
    result_df.index = data_entry.index[:result_df.shape[0]]
    df = result_df[[columnslabel]]
    # Step 1: Extract unique values from 'Estuary_18' column excluding 0
    unique_values = df[columnslabel].unique()
    unique_values = unique_values[unique_values != 0]
    
    countEvents.append(unique_values.size)
    
    # Create a new dataframe to store the results
    result_df1 = pd.DataFrame()

    # Iterate over unique values and extract data from 'mainData'
    # Define an empty results DataFrame
    all_results = pd.DataFrame(index=['pre-cyclone', 'cyclone', 'post-cyclone'])
    estmainData = data_entry.iloc[:,0:4]
    #Esturay Reconstruction Data
    
    reconData = pd.DataFrame(errors)
    reconData.columns = ['error']
    
    
    print(estmainData.columns)
    estmainData.columns = ['Temperature', 'Salinity', 'DO', 'Turbidity']
    estmainData['datetime'] = data_entry.index
    reconData.index = df.index
    reconData['datetime'] = df.index
    

    
    
    #Iterate over unique values and extract data
    for value in unique_values:
        subset = df.loc[df[columnslabel] == value]
        subset_reconData = reconData[df[columnslabel] == value]
    
        if not subset.empty:
            start_time =  pd.to_datetime(subset.index[0]) - pd.Timedelta(days=7)
            end_time =  pd.to_datetime(subset.index[-1]) +  pd.Timedelta(days=7)
    
            # Extract data from 'mainData' using time windows
            start_window = start_time - pd.Timedelta(days=14)
            end_window = end_time + pd.Timedelta(days=14)
            
            sub_data = estmainData[(estmainData['datetime'] >= start_window) & (estmainData['datetime'] <= end_window)]
    
            
            sub_data = sub_data.copy()
            sub_data.loc[:, 'cyclone_Status'] = pd.cut(sub_data['datetime'], bins=[start_window, start_time, end_time, end_window],
                                           labels=['pre-cyclone', 'cyclone', 'post-cyclone'], include_lowest=True)

            

    
            #plots
            plot_data_with_customization(sub_data,"featurePlot_subs", columnslabel+"_"+str(value))
                

DM = pd.DataFrame(disturbance_magnitudes)
