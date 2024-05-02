import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
from torch.nn import functional as F
from sklearn.preprocessing import RobustScaler
import matplotlib.pyplot as plt
from matplotlib import rcParams
from utility import replace_outliers_with_nearest, create_sequences, plot_sequence_and_error, normalize_and_add_count, find_peaks_in_concave_segments, calculate_disturbance_magnitude, plot_sub_data
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
sns.set(color_codes=True)
import tensorflow as tf
tf.random.set_seed(10)
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

# Set the default font to Times New Roman
rcParams['font.family'] = 'serif'
rcParams['font.serif'] = ['Times New Roman']

# Define a single shared encoder for all estuaries
class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(Encoder, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)

    def forward(self, x):
        _, (hidden, _) = self.lstm(x)
        return hidden.squeeze(0)  # Squeeze to remove the batch dimension

class Decoder(nn.Module):
    def __init__(self, hidden_size, output_size):
        super(Decoder, self).__init__()
        self.lstm = nn.LSTM(hidden_size, output_size, batch_first=True)

    def forward(self, x):
        x, _ = self.lstm(x)
        return x

# Define the autoencoder model using separate encoders and decoders

class Autoencoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_estuaries):
        super(Autoencoder, self).__init__()
        self.autoencoders = nn.ModuleList([
            nn.Sequential(
                nn.LSTM(input_size, hidden_size, batch_first=True),
                nn.LSTM(hidden_size, input_size, batch_first=True)
            ) for _ in range(num_estuaries)
        ])

    def forward(self, x, estuary_idx):
        x, _ = self.autoencoders[estuary_idx][0](x)
        x, _ = self.autoencoders[estuary_idx][1](x)
        return x



#initialized 'device' earlier
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load sensor data and labels
data = pd.read_csv("data_df1.csv", parse_dates=["datetime"])

# Convert datetime column to datetime format
data['datetime'] = pd.to_datetime(data['datetime'])

# Extract datetime column
datetime_column = data["datetime"]
# Drop datetime column for processing
data = data.drop(columns=["datetime"])

# Process the data and remove outliers
data = replace_outliers_with_nearest(data)
#data.describe()

# Add datetime column back
data["datetime"] = datetime_column
#data.to_csv("noOutlierData.csv", header = True)
mainData = data.copy()
data = data.set_index('datetime')
# Resample the data at every two hours using mean
resampled_data = data.resample('1H').mean()

# Normalize the resampled data
scaler = RobustScaler()  # RobustScaler is less sensitive to outliers
resampled_data_normalized = scaler.fit_transform(resampled_data)

# Convert data to PyTorch tensor
data_tensor = torch.tensor(resampled_data_normalized, dtype=torch.float32)

# Create sequences
sequence_length = 10  # can adjust this based on needs
sequences = create_sequences(data_tensor, sequence_length)

# Convert sequences to PyTorch tensor
sequences = torch.tensor(sequences, dtype=torch.float32)

# 'device' is defined earlier
sequences = sequences.to(device)


# Initialize and train the autoencoder for each estuary
input_size = 4  # should set this based on actual input size
hidden_size = 64  # can adjust this based on needs
num_estuaries = 19  # Number of estuaries
initial_lr = 0.01  # can adjust this based on needs
num_epochs = 50  # can adjust this based on needs
batch_size = 512

autoencoder = Autoencoder(input_size, hidden_size, num_estuaries).to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(autoencoder.parameters(), lr=initial_lr)

# Create a DataLoader for training
train_dataset = TensorDataset(sequences)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,  drop_last=True)

threshold =0.1

all_losses = {f'Estuary {i + 1}': [] for i in range(num_estuaries)}  # Track losses for each estuary
learning_rates = []

# 'sequences' is input data
for epoch in range(num_epochs):
    epoch_losses = {f'Estuary {i + 1}': [] for i in range(num_estuaries)}  # Track losses for each estuary
    for estuary_idx in range(num_estuaries):
        for batch_inputs in train_loader:
            batch_inputs = batch_inputs[0].to(device)

            optimizer.zero_grad()
            # Extract the current estuary sequence
            current_estuary_sequence = batch_inputs[:, :, estuary_idx * 4: (estuary_idx + 1) * 4]
            outputs = autoencoder(current_estuary_sequence, estuary_idx)
            
            loss = criterion(outputs, current_estuary_sequence)
            loss.backward()
            optimizer.step()

            epoch_losses[f'Estuary {estuary_idx + 1}'].append(loss.item())

        # Print the loss for each epoch and estuary
        print(f'Epoch [{epoch + 1}/{num_epochs}], Estuary {estuary_idx + 1}, Loss: {loss.item():.4f}')
        
        if epoch % 5  == 0:
            with torch.no_grad():
            #for estuary_idx in range(num_estuaries):
                reconstructed_samples = autoencoder(current_estuary_sequence, estuary_idx)
                errors = F.mse_loss(reconstructed_samples, current_estuary_sequence, reduction='none').mean(dim=2)
                mean_errors = errors.mean(dim=1)
    
                normal_indices = torch.where(mean_errors <= mean_errors.mean())[0]
                abnormal_indices = torch.where(mean_errors > mean_errors.mean()*2)[0]
    
                if len(normal_indices) > 1:
                    normal_samples = current_estuary_sequence[normal_indices]
                    normal_reconstructed = reconstructed_samples[normal_indices]
                    normal_errors = errors[normal_indices]
                    plot_sequence_and_error(normal_samples, normal_reconstructed, normal_errors, threshold,
                                            title=f"Epoch {epoch} Estuary: WeeksBay Normal Sequences")
                #{estuary_idx + 1}
                if len(abnormal_indices) > 1:
                    abnormal_samples = current_estuary_sequence[abnormal_indices]
                    abnormal_reconstructed = reconstructed_samples[abnormal_indices]
                    abnormal_errors = errors[abnormal_indices]
                    plot_sequence_and_error(abnormal_samples, abnormal_reconstructed, abnormal_errors, threshold,
                                            title=f"Epoch {epoch} Estuary: WeeksBay  Abnormal Sequences")
# Calculate the average loss for the epoch for each estuary
    for estuary_idx, estuary_loss in epoch_losses.items():
        average_estuary_loss = sum(estuary_loss) / len(estuary_loss)
        all_losses[estuary_idx].append(average_estuary_loss)

    # Dynamic learning rate adjustment
    if epoch % 5 == 0 and epoch > 0:
        for param_group in optimizer.param_groups:
            param_group['lr'] *= 0.9
    current_lr = optimizer.param_groups[0]['lr']
    learning_rates.append(current_lr)
           
            
            
            
# Plot the training losses for each estuary
for estuary_idx, estuary_losses in all_losses.items():
    plt.plot(estuary_losses, label=estuary_idx)

    plt.title("Training Loss Over Epochs for Each Estuary")
    plt.xlabel("Epochs")
    plt.ylabel("Average Loss")
    plt.legend()
    plt.show()

# Plotting the learning rates
plt.plot(learning_rates, marker='o')
plt.title("Learning Rate Adjustment Over Epochs")
plt.xlabel("Epochs")
plt.ylabel("Learning Rate")
plt.show()


# 'all_losses' and 'learning_rates' are defined in environment

# Create a subplot for the training losses
fig_losses = make_subplots(rows=len(all_losses), cols=1, subplot_titles=[f"Estuary {idx}" for idx in all_losses.keys()])

for i, (estuary_idx, estuary_losses) in enumerate(all_losses.items()):
    fig_losses.add_trace(go.Scatter(
        x=np.arange(1, len(estuary_losses) + 1),
        y=estuary_losses,
        mode='lines',
        name=f'Estuary {estuary_idx}',
        line=dict(width=2, backoff=0.7)
    ), row=i+1, col=1)

fig_losses.update_layout(
    title_text="Training Loss Over Epochs for Each Estuary",
    xaxis_title="Epochs",
    yaxis_title="Average Loss",
    showlegend=False,  # Legends are added manually below
    font=dict(family="Times New Roman"),
)


# 'autoencoder' is trained model
torch.save(autoencoder.state_dict(), 'autoencoder_modelV19Estuaries.pth')
# 'Autoencoder' is the class defined for model
model = Autoencoder(input_size, hidden_size, num_estuaries)
model.load_state_dict(torch.load('autoencoder_modelV19Estuaries.pth'))
model.to(device)




# Initialize the dataframe

labelData = pd.DataFrame()
df2 = pd.DataFrame()

# Use the trained model to obtain reconstruction errors for all estuaries
for estuary_idx in range(num_estuaries):
    # Extract data for the current estuary
    estuary_data = sequences[:, :, estuary_idx * 4: (estuary_idx + 1) * 4]

    time_indices = resampled_data.index

    # Use the model to obtain reconstruction errors
    with torch.no_grad():
        reconstructed_samples = model(estuary_data, estuary_idx)
        errors = F.mse_loss(reconstructed_samples, estuary_data, reduction='none').mean(dim=2)
        mean_errors = errors.mean(dim=1)

    # Find disturbance start indices
    #threshold = mean_errors.std() * 5
    
    reconstruction_errors = mean_errors.detach().cpu().numpy()

    # If using NumPy arrays
    #reconstruction_errors = np.array(reconstruction_errors)
        
    normalized_errors = (reconstruction_errors - np.min(reconstruction_errors)) / (np.max(reconstruction_errors) - np.min(reconstruction_errors))
    
    anomalies = [1 if error > 1*normalized_errors.std() else 0 for error in normalized_errors]
    
    labelData[f'Estuary_{estuary_idx + 1}'] = normalized_errors
    df2[f'Estuary_{estuary_idx + 1}'] = anomalies
    
labelData.index = resampled_data.index[sequence_length-1:]
df2.index = labelData.index


labelData['datetime'] = pd.to_datetime(labelData.index, format='%Y-%m-%d %H:%M:%S')
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


mainData['datetime'] = pd.to_datetime(mainData['datetime'],  format='%Y-%m-%d %H:%M:%S')


mainData.index = mainData['datetime']
datetime = mainData[['datetime']]
mainData = mainData.drop(columns=["datetime"])


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
            
            # df is DataFrame and start_time, end_time are defined
            sub_reconstructedData = df[(df['datetime'] >= start_time) & (df['datetime'] <= end_time)]
            
            # Find peaks and recovery indices using function
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
            
            #pio.write_image(fig, 'cyclone_disturbance_recovery.svg')
            
            
            # sub_reconstructedData, global_peak_index, and recoveryIndex are defined
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
            #pio.write_image(fig, 'cyclone_disturbance_recovery.svg')
            
           # Calculate disturbance magnitude using the function
            disturbance_magnitude = calculate_disturbance_magnitude(sub_reconstructedData)
            
            if disturbance_magnitude >= 4:
                

               # Append results to the data structure for making plots
               disturbance_magnitudes.append({'Estuary': estuary, 'Value': value, 'Magnitude': disturbance_magnitude, 
                                              'Peak': sub_reconstructedData.index[global_peak_index], 
                                              'Recovery':sub_reconstructedData.index[recoveryIndex],
                                              'RecoveryDays': (recoveryIndex- global_peak_index)/24})
               
            
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
            plot_sub_data(sub_data,f'Image_{estuary}_{value}')
                
    #print(i)
    i += 1
    

import pandas as pd
import seaborn as sns

# Convert the list of dictionaries to a DataFrame
disturbance_df = pd.DataFrame(disturbance_magnitudes)

# Save the DataFrame to an Excel file
disturbance_df.to_excel('Recoverywithmag.xlsx', index=False)
#disturbance_df.to_excel('disturbance_magnitudes1.xlsx', index=False)
