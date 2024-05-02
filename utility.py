import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import simps
from scipy.stats import ttest_ind
from scipy.ndimage import gaussian_filter1d
import pandas as pd
# Function to replace outliers with the mean of 5 before and 5 after values
def replace_outliers_with_nearest(df, threshold=15, window_size=5):
    df_copy = df.copy()

    for col in df_copy.columns:
        col_data = df_copy[col].values
        outliers = np.abs(col_data - col_data.mean()) > threshold * col_data.std()

        # Find indices of outliers
        outlier_indices = np.where(outliers)[0]
        threshold = 10
        window_size = 5
        

        while len(outlier_indices) > 0:
            a = len(outlier_indices)
            for index in outlier_indices:
                start_idx = max(0, index - window_size)
                end_idx = min(len(col_data), index + window_size + 1)

                # Replace outlier with the mean of 5 before and 5 after values
                df_copy.at[index, col] = np.mean(col_data[start_idx:end_idx])

            # Check for outliers again
            col_data = df_copy[col].values
            outliers = np.abs(col_data - col_data.mean()) > threshold * col_data.std()
            outlier_indices = np.where(outliers)[0]
            #print(len(outlier_indices))
            if a <= len(outlier_indices):
                window_size +=1
                

    return df_copy



# Function to create sequences from the data
def create_sequences(data, sequence_length):
    sequences = []
    for i in range(len(data) - sequence_length + 1):
        sequence = data[i: i + sequence_length].numpy()
        sequences.append(sequence)
    return np.array(sequences)




def plot_sequence_and_error(original, reconstructed, error, threshold, title="Sequence and Reconstruction Error"):
    num_samples = min(original.shape[0], 5)
    fig, axes = plt.subplots(num_samples, 3, figsize=(12, 2 * num_samples))
    feature_labels = ["Temperature", "Salinity", "DO", "Turbidity"]
    
    for i in range(num_samples):
        # Plot Original
        axes[i, 0].plot(original[i].detach().cpu().numpy(), label=feature_labels)
        axes[i, 0].set_title("Original")
        axes[i, 0].set_xlim([0, len(original[i])])  # Set x-axis limits
        axes[i, 0].set_ylim([original[i].detach().cpu().numpy().min(), original[i].detach().cpu().numpy().max()])
        if i == 0:
            axes[i, 0].legend()

        # Plot Reconstructed
        axes[i, 1].plot(reconstructed[i].detach().cpu().numpy(), label=feature_labels)
        axes[i, 1].set_title("Reconstructed")
        axes[i, 1].set_xlim([0, len(original[i])])  # Set x-axis limits
        axes[i, 1].set_ylim([original[i].detach().cpu().numpy().min(), original[i].detach().cpu().numpy().max()])

        # Plot Reconstruction Error
        axes[i, 2].plot(error[i].detach().cpu().numpy(), label="Reconstruction Error", color='red')
        axes[i, 2].axhline(threshold, color='orange', linestyle='--', label="Threshold")
        axes[i, 2].set_title("Reconstruction Error")
        axes[i, 2].set_xlim([0, len(original[i])])  # Set x-axis limits

    plt.suptitle(title)

    # Add common legend at the bottom of the plot
    lines, labels = axes[0, 2].get_legend_handles_labels()
    fig.legend(lines, labels, loc='lower center', bbox_to_anchor=(0.5, -0.05), ncol=3)

    plt.tight_layout()

    # Save the plot with high quality
    plt.savefig('sequence_and_error_plot.png', dpi=600, bbox_inches='tight')
    
    plt.show()

# Example usage:
# Call the function with your data
# plot_sequence_and_error(original_data, reconstructed_data, error_data, threshold_value)







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






# Assuming df is your original DataFrame

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



def plot_sub_data(sub_data, cid):
    # Features to plot
    features = ['Temp', 'Sal', 'DO_mgl', 'Turb']

    # Time periods
    time_periods = ['pre-cyclone', 'cyclone', 'post-cyclone']

    # Create subplots
    fig, axs = plt.subplots(len(features), 1, figsize=(10, 8), sharex=True)

    # Plot each feature in a subplot
    for i, feature in enumerate(features):
        ax = axs[i]

        # Plot lines for each time period
        for period in time_periods:
            subset = sub_data[sub_data['cyclone_Status'] == period]
            ax.plot(subset['datetime'], subset[feature], label=period)

        # Shade the entire background during the hurricane period
        hurricane_subset = sub_data[sub_data['cyclone_Status'] == 'cyclone']
        hurricane_start = hurricane_subset['datetime'].min()
        hurricane_end = hurricane_subset['datetime'].max()
        ax.axvspan(hurricane_start, hurricane_end, facecolor='red', alpha=0.3, label='Disturbance Window')

        # Add a line for the cumulative mean
        #cum_mean = sub_data.groupby('cyclone_Status')[feature].expanding().mean().reset_index(level=0, drop=True)
        # Plot last 10 values mean
        window_size = 100
        cum_mean_last_10 = sub_data[feature].rolling(window=window_size, min_periods=1).mean()
        ax.plot(sub_data['datetime'], cum_mean_last_10, label= 'Moving Average', linestyle='--', color='red')

        #cum_mean = sub_data[feature].expanding().mean()

        #ax.plot(sub_data['datetime'], cum_mean, label='Cumulative Mean', linestyle='--', color='black')

        ax.set_ylabel(feature)
        ax.legend(loc='upper right')

    plt.tight_layout()

    # Save the plot at 300 DPI
    plt.savefig(f'plot{cid}.png', dpi=300)

    plt.show()


def analyze_sub_data(sub_data):
    # Group by 'Hurricane_Status' and calculate the mean for each feature
    means_by_status = sub_data.groupby('cyclone_Status').mean()

    # Create a table to store results
    results_table = pd.DataFrame(index=['pre-cyclone', 'cyclone', 'post-cyclone'])

    # Loop through each feature and perform t-test
    for feature in ['Temp', 'Sal', 'DO_mgl', 'Turb']:
        pre_hurricane_mean = means_by_status.loc['pre-cyclone', feature]
        hurricane_mean = means_by_status.loc['cyclone', feature]
        post_hurricane_mean = means_by_status.loc['post-cyclone', feature]

        # Perform t-tests
        t_stat_hurricane, p_value_hurricane = ttest_ind(sub_data[sub_data['cyclone_Status'] == 'pre-cyclone'][feature],
                                                         sub_data[sub_data['cyclone_Status'] == 'cyclone'][feature])

        t_stat_post, p_value_post = ttest_ind(sub_data[sub_data['cyclone_Status'] == 'cyclone'][feature],
                                               sub_data[sub_data['cyclone_Status'] == 'post-cyclone'][feature])

        # Store results in the table
        results_table[feature + '_pre_cyclone'] = pre_hurricane_mean
        results_table[feature + '_hcyclone'] = hurricane_mean
        results_table[feature + '_cyclone'] = post_hurricane_mean
        results_table['p_value_cyclone_' + feature] = p_value_hurricane
        results_table['p_value_post_cyclone_' + feature] = p_value_post

    return results_table

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