# util.py

import torch
import torch.nn as nn
import pickle
from sklearn.preprocessing import RobustScaler
import os
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter, MonthLocator, DayLocator, AutoDateLocator
from scipy.ndimage import gaussian_filter1d
import numpy as np
import pandas as pd
from scipy.signal import find_peaks
from scipy.integrate import simps



class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=4):
        super(Encoder, self).__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )
        self.hidden_size = hidden_size
        self.num_layers = num_layers

    def forward(self, x):
        _, (hidden, _) = self.lstm(x)  # Return only the hidden state
        #print(f"[Encoder] Hidden shape after combining: {hidden.shape}")  # Debug
        hidden = hidden[-1]  # Use the last layer's hidden state
        #print(f"[Encoder] Final hidden shape: {hidden.shape}")  # Debug
        return hidden


class Decoder(nn.Module):
    def __init__(self, hidden_size, output_size, num_layers=4):
        super(Decoder, self).__init__()
        self.lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )
        self.output_layer = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        #print(f"[Decoder] Input shape: {x.shape}")  # Debug
        x, _ = self.lstm(x)
        #print(f"[Decoder] After LSTM shape: {x.shape}")  # Debug
        x = self.output_layer(x)  # Map hidden states to original feature size
        #print(f"[Decoder] After output layer shape: {x.shape}")  # Debug
        return x


class Autoencoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=4):
        super(Autoencoder, self).__init__()
        self.encoder = Encoder(input_size, hidden_size, num_layers)
        self.decoder = Decoder(hidden_size, input_size, num_layers)

    def forward(self, x):
        #print(f"[Autoencoder] Input shape: {x.shape}")  # Debug
        latent = self.encoder(x)  # Shape: [batch_size, hidden_size]
        #print(f"[Autoencoder] Latent shape: {latent.shape}")  # Debug
        latent = latent.unsqueeze(1).repeat(1, x.size(1), 1)  # Repeat for sequence length
        #print(f"[Autoencoder] Latent unsqueezed shape: {latent.shape}")  # Debug
        reconstructed = self.decoder(latent)  # Shape: [batch_size, seq_len, input_size]
        #print(f"[Autoencoder] Reconstructed shape: {reconstructed.shape}")  # Debug
        return reconstructed
    


def load_pkl_file(file_path):
    """
    Loads a .pkl file and returns its content.

    Parameters:
        file_path (str): The path of the .pkl file to load.

    Returns:
        object: The data loaded from the .pkl file.
    """
    with open(file_path, 'rb') as file:
        return pickle.load(file)
        
        
        

def recond_Data(df, sequence_length,model, device, threshold=0.1):
    

    
    sequences = []
    
       
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







def merge_series(arr: np.ndarray, threshold: int) -> np.ndarray:
    """
    Merge series in an array if the gap between consecutive series is below the threshold.

    Parameters:
        arr (np.ndarray): Array of series labels.
        threshold (int): Maximum allowable gap between series to merge.

    Returns:
        np.ndarray: Updated array with merged series.
    """
    unique_numbers = sorted(set(arr) - {0})
    indices = {num: (arr == num).nonzero()[0] for num in unique_numbers}

    for i in range(len(unique_numbers) - 1):
        current, next_ = unique_numbers[i], unique_numbers[i + 1]
        if indices[next_][0] - indices[current][-1] <= threshold:
            arr[indices[current][-1] + 1 : indices[next_][-1] + 1] = current

    return arr


def normalize_and_add_count(
    df: pd.DataFrame, label_column: str, min_series_length: int = 96, max_gap_between_series: int = 720
) -> pd.DataFrame:
    """
    Normalize hurricane series by merging short series and ensuring continuity.

    Parameters:
        df (pd.DataFrame): Input DataFrame with hurricane labels.
        label_column (str): Column name containing binary labels.
        min_series_length (int): Minimum length of a series to retain.
        max_gap_between_series (int): Maximum gap to merge series.

    Returns:
        pd.DataFrame: DataFrame with normalized hurricaneCount.
    """
    df['hurricaneCount'] = 0
    start_of_series = (df[label_column] == 1) & (df[label_column].shift(1, fill_value=0) == 0)
    df['hurricaneCount'] = start_of_series.cumsum()
    df.loc[df[label_column] == 0, 'hurricaneCount'] = 0

    # Remove short series
    series_lengths = df['hurricaneCount'].value_counts()
    short_series = series_lengths[series_lengths < min_series_length].index
    df.loc[df['hurricaneCount'].isin(short_series), 'hurricaneCount'] = 0

    # Merge series with small gaps
    df['hurricaneCount'] = merge_series(df['hurricaneCount'].to_numpy(), max_gap_between_series)

    # Reindex hurricaneCount for sequential numbering
    unique_values = sorted(df['hurricaneCount'].unique())
    reindex_map = {v: i for i, v in enumerate(unique_values)}
    df['hurricaneCount'] = df['hurricaneCount'].map(reindex_map)

    return df[['hurricaneCount']]


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



def calculate_disturbance_magnitude(sub_reconstructed_data: pd.DataFrame) -> float:
    """
    Calculate the disturbance magnitude from error curve.

    Parameters:
        sub_reconstructed_data (pd.DataFrame): DataFrame containing 'datetime' and 'errors'.

    Returns:
        float: Magnitude of disturbance.
    """
    sub_reconstructed_data = sub_reconstructed_data.sort_values(by='datetime')
    errors = sub_reconstructed_data['errors'].to_numpy()
    timestamps = np.arange(len(errors))
    return simps(errors, timestamps)


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
    #plt.savefig(filename, dpi=300)
    plt.show()
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
    plt.plot(tot.index, tot['error'], label='Reconstruction Error', linestyle='-', marker='o', color='blue')

    # Plot the smoothed error
    plt.plot(sub_reconstructedData.index, sub_reconstructedData['smoothed_error'], label='Smoothed Error', linestyle='-', marker='x', color='orange')

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
    #plt.savefig(filename, bbox_inches='tight')
    plt.show()
    plt.close()

    print(f"Plot saved to {filename}")
    
    
    
    
def process_event_data(test_data, autoencoder, device):
    count_events = []
    disturbance_magnitudes = []

    for entry in test_data:
        # Extract entry details
        data_entry = entry["Data"]
        labels, errors = recond_Data(data_entry, 168, autoencoder, device, 0.5)
        estuary = entry.get("Estuary", "Unknown")
        hurricane = entry.get("Hurricane", "Unknown")
        year = entry.get("Year", "Unknown")
        column_label = f"{estuary}_{hurricane}_{year}"

        # Create labels DataFrame
        labels_df = pd.DataFrame(labels, columns=['Label'])
        result_df = normalize_and_add_count(
            labels_df.copy(), 'Label', min_series_length=12, max_gap_between_series=168 * 3
        )[['hurricaneCount']]
        result_df.columns = [column_label]
        result_df.index = data_entry.index[:result_df.shape[0]]

        # Extract unique values excluding 0
        unique_values = result_df[column_label].unique()
        unique_values = unique_values[unique_values != 0]
        count_events.append(len(unique_values))

        # Prepare main data and reconstruction errors
        est_main_data = data_entry.iloc[:, :4].copy()
        est_main_data.columns = ['Temperature', 'Salinity', 'DO', 'Turbidity']
        est_main_data['datetime'] = data_entry.index

        recon_data = pd.DataFrame(errors, columns=['error'], index=result_df.index)
        recon_data['datetime'] = result_df.index

        for value in unique_values:
            # Subset for the current unique value
            subset = result_df[result_df[column_label] == value]

            if not subset.empty:
                start_time = pd.to_datetime(subset.index[0]) - pd.Timedelta(days=7)
                end_time = pd.to_datetime(subset.index[-1]) + pd.Timedelta(days=7)

                # Define time windows
                start_window = start_time - pd.Timedelta(days=14)
                end_window = end_time + pd.Timedelta(days=14)

                # Filter data for reconstruction and analysis
                sub_reconstructed_data = recon_data[
                    (recon_data['datetime'] >= start_time) &
                    (recon_data['datetime'] <= end_time)
                ]

                # Find peaks and recovery indices
                global_peak_index, recovery_index, _, smoothed_data = find_peaks_in_concave_segments(
                    sub_reconstructed_data['error']
                )

                if global_peak_index > 3 * 24:
                    sub_reconstructed_data['smoothed_error'] = smoothed_data

                    full_window_data = recon_data[
                        (recon_data['datetime'] >= start_window) &
                        (recon_data['datetime'] <= end_window)
                    ]

                    # Plot recovery
                    plot_cyclone_recovery(
                        full_window_data, sub_reconstructed_data,
                        global_peak_index, recovery_index,
                        f"{column_label}_{value}", "RecoveryPlots"
                    )

                    # Extract and label data
                    sub_data = est_main_data[
                        (est_main_data['datetime'] >= start_window) &
                        (est_main_data['datetime'] <= end_window)
                    ].copy()

                    sub_data['cyclone_Status'] = pd.cut(
                        sub_data['datetime'],
                        bins=[start_window, start_time, end_time, end_window],
                        labels=['pre-cyclone', 'cyclone', 'post-cyclone'],
                        include_lowest=True
                    )

                    # Plot feature data
                    plot_data_with_customization(
                        sub_data, "featurePlot_subs", f"{column_label}_{value}"
                    )

    #return count_events, disturbance_magnitudes
