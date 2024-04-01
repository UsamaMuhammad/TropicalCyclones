import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import joblib
import seaborn as sns
sns.set(color_codes=True)
import matplotlib.pyplot as plt
#matplotlib inline

import h5py
import tensorflow as tf
tf.random.set_seed(10)
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

from matplotlib import rcParams


from keras.layers import Input, Dense, LSTM, TimeDistributed, RepeatVector
from keras.models import Model
from keras import regularizers

from scipy.stats import ttest_ind
from scipy.signal import savgol_filter
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks
###Functions

from scipy.integrate import simps

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
    
    
    
    # Now, df[column_name] contains the positions of the unique values


    #df = df[~df['hurricaneCount'].isin(short_series)]
    #input_array = np.array(df['hurricaneCount'])

    
    #df['newLabel'] = result_array

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




labelData = pd.read_csv('reconstruction_errors.csv')
labelData['datetime'] = pd.to_datetime(labelData['datetime'], format='%Y-%m-%d %H:%M:%S')
reconstruction_errors = labelData.copy()
dateTime = labelData['datetime'] 
labelData = labelData.drop(columns=["datetime"])


# Use the trained model to obtain reconstruction errors for all estuaries
for col in labelData.columns:
    # Extract data for the current estuary
    normalized_errors = labelData[col]
    
    # Vectorized operation to calculate anomalies
    anomalies = (normalized_errors > 3 * normalized_errors.std()).astype(int)
    
    labelData[col] = anomalies







# Define the date range
start_date = '2007-01-01'
end_date = '2023-09-30'

filename = 'noOutlierData.csv'


file_path = os.path.join(folder_path, filename)
        
       
        
# Read the CSV file
mainData = pd.read_csv(file_path)
filename = 'noOutlierData'
#mainData = mainData[['DateTimeStamp', 'Temp', 'Sal','DO_mgl', 'Turb']]
mainData['datetime'] = pd.to_datetime(mainData['datetime'], format='%m/%d/%Y %H:%M')


mainData.index = mainData['datetime']
datetime = mainData[['datetime']]
mainData = mainData.drop(columns=["datetime"])

#labelData = pd.read_csv('abnormallabels.csv')
labelData['datetime'] = pd.to_datetime(dateTime, format='%Y-%m-%d %H:%M:%S')
labelData.index = labelData['datetime']
labelData = labelData.drop(columns = ['datetime'])

result_df = pd.DataFrame()
i = 1

countEvents = []
disturbance_magnitudes = []
for estuary in labelData.columns:
    # Apply the function for each label
    result_df[estuary] = normalize_and_add_count(labelData[[estuary]].copy(), estuary,min_series_length=6, max_gap_between_series=3*240)['hurricaneCount']
    df = result_df[[estuary]]
    # Step 1: Extract unique values from 'Estuary_18' column excluding 0
    unique_values = df[estuary].unique()
    unique_values = unique_values[unique_values != 0]
    
    countEvents.append(unique_values.size)
    
    # Step 2: Create a new dataframe to store the results
    result_df1 = pd.DataFrame()

    # Step 3: Iterate over unique values and extract data from 'mainData'
    # Define an empty results DataFrame
    all_results = pd.DataFrame(index=['pre-cyclone', 'cyclone', 'post-cyclone'])
    #i = 1
    
    estmainData = mainData.iloc[:,(i-1)*4:4*i]
    #Esturay Reconstruction Data
    reconData = reconstruction_errors.iloc[:,i]
    print(estmainData.columns)
    estmainData.columns = ['Temp', 'Sal', 'DO_mgl', 'Turb']
    estmainData['datetime'] = datetime['datetime'].values
    reconData.index = df.index

    
    
    # Step 3: Iterate over unique values and extract data from 'mainData'
    for value in unique_values:
        subset = df.loc[df[estuary] == value]
        subset_reconData = reconData[df[estuary] == value]
    
        if not subset.empty:
            start_time =  pd.to_datetime(subset.index[0]) - pd.Timedelta(days=10)
            end_time =  pd.to_datetime(subset.index[-1]) +  pd.Timedelta(days=10)
    
            # Extract data from 'mainData' using time windows
            start_window = start_time - pd.Timedelta(days=30)
            end_window = end_time + pd.Timedelta(days=30)
            df['errors'] = reconstruction_errors[estuary].values
            df['datetime'] = reconstruction_errors['datetime'].values
         

            sub_reconstructedData = df[(df['datetime'] >= start_time) &(df['datetime'] <= end_time)]
            plt.plot(sub_reconstructedData.index, sub_reconstructedData['errors'].values)
            
            print(f'The value is {value}')
            print(len(sub_reconstructedData['errors']))
            global_peak_index, recoveryIndex, observations_after_peak, datu = find_peaks_in_concave_segments(sub_reconstructedData['errors'])
            sub_reconstructedData['serr'] = datu
            plt.plot(sub_reconstructedData.index, sub_reconstructedData['errors'].values)
            plt.plot(sub_reconstructedData.index, sub_reconstructedData['serr'].values)
            plt.show()
            
            
            import plotly.graph_objs as go
            from plotly.subplots import make_subplots
            
            # Assuming df is your DataFrame and start_time, end_time are defined
            sub_reconstructedData = df[(df['datetime'] >= start_time) & (df['datetime'] <= end_time)]
            
            # Find peaks and recovery indices using your function
            global_peak_index, recoveryIndex, observations_after_peak, datu = find_peaks_in_concave_segments(sub_reconstructedData['errors'])
            
            # Update 'serr' with smoothed data
            sub_reconstructedData['serr'] = datu
            
            # Create subplots for interactive plotting
            fig = make_subplots()
            
            # Add the original error plot
            fig.add_trace(go.Scatter(x=sub_reconstructedData.index, y=sub_reconstructedData['errors'],
                                     mode='lines', name='Error', line=dict(color='blue')))
            
            # Add the smoothed error plot
            fig.add_trace(go.Scatter(x=sub_reconstructedData.index, y=sub_reconstructedData['serr'],
                                     mode='lines', name='Smoothed Error', line=dict(color='orange')))
            
            # Set the disturbance phase with red background
            fig.add_vrect(x0=sub_reconstructedData.index[0], x1=sub_reconstructedData.index[global_peak_index],
                          annotation_text="Disturbance Phase", annotation_position="top left",
                          fillcolor="red", opacity=0.5, line_width=0)
            
            # Set the recovery phase with green background
            fig.add_vrect(x0=sub_reconstructedData.index[global_peak_index], x1=sub_reconstructedData.index[recoveryIndex],
                          annotation_text="Recovery Phase", annotation_position="top left",
                          fillcolor="green", opacity=0.5, line_width=0)
            
            # Update layout with Times New Roman font and plot width
            fig.update_layout(
                title='Cyclone Disturbance and Recovery Phases',
                xaxis_title='Time',
                yaxis_title='Reconstruction Error',
                font=dict(family="Times New Roman", size=12, color="black"),
                width=650  # Width of 6.5 inches (in pixels)
            )
            
            # Show interactive plot
            fig.show()
            import plotly.io as pio
            pio.write_image(fig, 'cyclone_disturbance_recovery.svg')
            
            
            
            
            
            from plotly.subplots import make_subplots
            import plotly.graph_objs as go
            import plotly.io as pio
            import pandas as pd
            
            # Assuming sub_reconstructedData, global_peak_index, and recoveryIndex are defined
            # For demonstration, I'll create a fake datetime
            tot = df[(df['datetime'] >= start_window) & (df['datetime'] <= end_window)]
            
            #sub_reconstructedData.index = pd.date_range(start="2024-01-01", periods=len(sub_reconstructedData), freq='D')
            
            fig = make_subplots()
            
            # Add the original error plot
            fig.add_trace(go.Scatter(x=tot.index, y=tot['errors'],
                                     mode='lines', name='Reconstruction Error', line=dict(color='blue')))
            
            #fig.add_trace(go.Scatter(x=sub_reconstructedData.index, y=sub_reconstructedData['errors'],
             #                        mode='lines', name='Reconstruction Error', line=dict(color='blue')))
            
            # Add the smoothed error plot
            fig.add_trace(go.Scatter(x=sub_reconstructedData.index, y=sub_reconstructedData['serr'],
                                     mode='lines', name='Smoothed Error', line=dict(color='orange')))
            
            # Set the disturbance phase with red background
            fig.add_vrect(x0=sub_reconstructedData.index[0], x1=sub_reconstructedData.index[global_peak_index],
                          annotation_text="Disturbance Phase", annotation_position="top left",
                          fillcolor="red", opacity=0.3, line_width=0)
            
            # Set the recovery phase with green background
            fig.add_vrect(x0=sub_reconstructedData.index[global_peak_index], x1=sub_reconstructedData.index[recoveryIndex],
                          annotation_text="Recovery Phase", annotation_position="top left",
                          fillcolor="green", opacity=0.3, line_width=0)
            
            # Update layout with Times New Roman font, plot width, and x-axis formatting
            fig.update_layout(
                title='Cyclone Disturbance and Recovery Phases',
                xaxis=dict(
                    title='Time',
                    tickmode='auto',
                    nticks=len(sub_reconstructedData) // 30,
                    tickformat='%b %d'  # Format like 'Jan 01'
                ),
                yaxis_title='Error',
                legend=dict(y=0.5, font=dict(size=10), bgcolor='rgba(255,255,255,0.5)'),
                font=dict(family="Times New Roman", size=12, color="black"),
                width=650,  # Width of 6.5 inches (in pixels)
                showlegend=True
            )
            
            # Show interactive plot
            fig.show()
            
            # Save the figure to an SVG file
            pio.write_image(fig, 'cyclone_disturbance_recovery.svg')
            

            
            
            
            
            
            # Calculate disturbance magnitude using the function
            disturbance_magnitude = calculate_disturbance_magnitude(sub_reconstructedData)
            
            if disturbance_magnitude >= 4:
                

               # Append results to the data structure for making plots
               disturbance_magnitudes.append({'Estuary': estuary, 'Value': value, 'Magnitude': disturbance_magnitude, 
                                              'Peak': sub_reconstructedData.index[global_peak_index], 
                                              'Recovery':sub_reconstructedData.index[recoveryIndex],
                                              'RecoveryDays': (recoveryIndex- global_peak_index)/24})
               '''
            
            sub_data = estmainData[(estmainData['datetime'] >= start_window) & (estmainData['datetime'] <= end_window)]
    
            # Add a new column to indicate the time period
            #sub_data['cyclone_Status'] = pd.cut(sub_data['datetime'], bins=[start_window, start_time, end_time, end_window],
            #                                     labels=['pre-cyclone', 'cyclone', 'post-cyclone'], include_lowest=True)
            sub_data = sub_data.copy()
            sub_data.loc[:, 'cyclone_Status'] = pd.cut(sub_data['datetime'], bins=[start_window, start_time, end_time, end_window],
                                           labels=['pre-cyclone', 'cyclone', 'post-cyclone'], include_lowest=True)

            #sub_data.loc[:, 'cyclone_Status'] = pd.cut(sub_data['datetime'], bins=[start_window, start_time, end_time, end_window],
            #                               labels=['pre-cyclone', 'cyclone', 'post-cyclone'], include_lowest=True)

    
            #plots
            plot_sub_data(sub_data,f'{filename}_{estuary}_{value}')
                '''
    #print(i)
    i += 1
    


import pandas as pd
import seaborn as sns


# Assuming 'disturbance_magnitudes' is a list of dictionaries containing disturbance magnitudes
# Example: [{'Estuary': 'Estuary1', 'Value': 'Value1', 'Magnitude': 0.123}, ...]

# Convert the list of dictionaries to a DataFrame
disturbance_df = pd.DataFrame(disturbance_magnitudes)

# Save the DataFrame to an Excel file
disturbance_df.to_excel('Recoverywithmag.xlsx', index=False)
#disturbance_df.to_excel('disturbance_magnitudes1.xlsx', index=False)
