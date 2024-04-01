import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
from torch.nn import functional as F
from sklearn.preprocessing import RobustScaler
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
from matplotlib import rcParams

# Set the default font to Times New Roman
rcParams['font.family'] = 'serif'
rcParams['font.serif'] = ['Times New Roman']

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



def plot_sequence_and_error(original, reconstructed, error, threshold, title="Sequence and Reconstruction Error"):
    num_samples = min(original.shape[0], 5)
    fig, axes = plt.subplots(num_samples, 3, figsize=(12, 2 * num_samples))
    feature_labels = ["Temperature", "Salinity", "DO", "Turbidity"]
    for i in range(num_samples):
        # Plot Original
        axes[i, 0].plot(original[i].detach().cpu().numpy(), label=feature_labels)
        axes[i, 0].set_title("Original")
        axes[i, 0].set_xlim([0, len(original[i])])  # Set x-axis limits
        axes[i, 0].set_ylim([original[i].detach().cpu().numpy().min(), original[i].detach().cpu().numpy().max()])  # Set y-axis limits
        #axes[i, 0].legend()
        if i==0:
            axes[i, 0].legend()
            

        # Plot Reconstructed
        axes[i, 1].plot(reconstructed[i].detach().cpu().numpy(), label=feature_labels)
        axes[i, 1].set_title("Reconstructed")
        axes[i, 1].set_xlim([0, len(original[i])])  # Set x-axis limits
        axes[i, 1].set_ylim([original[i].detach().cpu().numpy().min(), original[i].detach().cpu().numpy().max()])  # Set y-axis limits
        #axes[i, 1].legend()

        # Plot Reconstruction Error
        axes[i, 2].plot(error[i].detach().cpu().numpy(), label="Reconstruction Error", color='red')
        axes[i, 2].axhline(threshold, color='orange', linestyle='--', label="Threshold")
        axes[i, 2].set_title("Reconstruction Error")
        axes[i, 2].set_xlim([0, len(original[i])])  # Set x-axis limits
        #axes[i, 2].legend()

    plt.suptitle(title)
    plt.tight_layout()

    # Add common legend at the bottom of the plot
    lines, labels = axes[0, 2].get_legend_handles_labels()
    fig.legend(lines, labels, loc='lower center', bbox_to_anchor=(0.5, -0.05), ncol=3)
    
    plt.show()



# Function to plot sequences and reconstruction error
def plot_sequence_and_error(original, reconstructed, error, threshold, title="Sequence and Reconstruction Error"):
    num_samples = min(original.shape[0], 5)
    fig, axes = plt.subplots(num_samples, 3, figsize=(12, 2 * num_samples))

    for i in range(num_samples):
        # Plot Original
        axes[i, 0].plot(original[i].detach().cpu().numpy(), label="Original")
        axes[i, 0].set_title("Original")
        axes[i, 0].set_xlim([0, len(original[i])])  # Set x-axis limits
        axes[i, 0].set_ylim([original[i].detach().cpu().numpy().min(), original[i].detach().cpu().numpy().max()])  # Set y-axis limits

        # Plot Reconstructed
        axes[i, 1].plot(reconstructed[i].detach().cpu().numpy(), label="Reconstructed")
        axes[i, 1].set_title("Reconstructed")
        axes[i, 1].set_xlim([0, len(original[i])])  # Set x-axis limits
        axes[i, 1].set_ylim([original[i].detach().cpu().numpy().min(), original[i].detach().cpu().numpy().max()])  # Set y-axis limits

        # Plot Reconstruction Error
        axes[i, 2].plot(error[i].detach().cpu().numpy(), label="Reconstruction Error", color='red')
        axes[i, 2].axhline(threshold, color='orange', linestyle='--', label="Threshold")
        axes[i, 2].set_title("Reconstruction Error")
        axes[i, 2].set_xlim([0, len(original[i])])  # Set x-axis limits
        
    plt.suptitle(title)
    plt.tight_layout()

    # Add common legend at the bottom of the plot
    lines, labels = axes[0, 2].get_legend_handles_labels()
    fig.legend(lines, labels, loc='lower center', bbox_to_anchor=(0.5, -0.05), ncol=3)
    
    plt.show()
'''
    plt.suptitle(title)
    plt.tight_layout()
    plt.show()
'''
# Function to create sequences from the data
def create_sequences(data, sequence_length):
    sequences = []
    for i in range(len(data) - sequence_length + 1):
        sequence = data[i: i + sequence_length].numpy()
        sequences.append(sequence)
    return np.array(sequences)

# Assuming you have initialized 'device' earlier
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load your sensor data and labels
data = pd.read_csv("data_df1.csv", parse_dates=["datetime"])

# Convert datetime column to datetime format
data['datetime'] = pd.to_datetime(data['datetime'])

# Extract datetime column
datetime_column = data["datetime"]
# Drop datetime column for processing
data = data.drop(columns=["datetime"])

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

# Process the data and remove outliers
data = replace_outliers_with_nearest(data)
#data.describe()

# Add datetime column back
data["datetime"] = datetime_column
#data.to_csv("noOutlierData.csv", header = True)
data = data.set_index('datetime')
# Resample the data at every two hours using mean
resampled_data = data.resample('1H').mean()

# Normalize the resampled data
scaler = RobustScaler()  # RobustScaler is less sensitive to outliers
resampled_data_normalized = scaler.fit_transform(resampled_data)

# Convert data to PyTorch tensor
data_tensor = torch.tensor(resampled_data_normalized, dtype=torch.float32)

# Create sequences
sequence_length = 10  # You can adjust this based on your needs
sequences = create_sequences(data_tensor, sequence_length)

# Convert sequences to PyTorch tensor
sequences = torch.tensor(sequences, dtype=torch.float32)

# Assuming 'device' is defined earlier
sequences = sequences.to(device)

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



# Initialize and train the autoencoder for each estuary
input_size = 4  # You should set this based on your actual input size
hidden_size = 64  # You can adjust this based on your needs
num_estuaries = 19  # Number of estuaries
initial_lr = 0.01  # You can adjust this based on your needs
num_epochs = 50  # You can adjust this based on your needs
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

# Assuming 'sequences' is your input data
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

###
import plotly.graph_objects as go
import plotly.express as px

# Assuming 'all_losses' and 'learning_rates' are defined in your environment

# Plot the training losses for each estuary
fig_losses = go.Figure()

for estuary_idx, estuary_losses in all_losses.items():
    fig_losses.add_trace(go.Scatter(
        x=list(range(1, len(estuary_losses) + 1)),
        y=estuary_losses,
        mode='lines',
        name=str(estuary_idx),
        line=dict(width=2, backoff=0.7)
    ))

fig_losses.update_layout(
    title="Training Loss Over Epochs for Each Estuary",
    xaxis_title="Epochs",
    yaxis_title="Average Loss",
    legend=dict(orientation="h", x=0.5, y=-0.15),
    font=dict(family="Times New Roman"),
)

# Save the interactive plot
fig_losses.write_html("training_losses_plot.html", full_html=False)

# Plotting the learning rates
fig_learning_rates = px.line(
    x=list(range(1, len(learning_rates) + 1)),
    y=learning_rates,
    markers=True,
    labels={"x": "Epochs", "y": "Learning Rate"},
    title="Learning Rate Adjustment Over Epochs",
)

fig_learning_rates.update_layout(
    font=dict(family="Times New Roman"),
)

# Save the interactive plot
fig_learning_rates.write_html("learning_rates_plot.html", full_html=False)


import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import numpy as np

# Assuming 'all_losses' and 'learning_rates' are defined in your environment

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

# Add legends manually at the bottom
fig_losses.add_trace(go.Scatter(), row=len(all_losses), col=1, name=', '.join(all_losses.keys()))

# Create a subplot for the learning rates
fig_learning_rates = go.Figure()

fig_learning_rates.add_trace(go.Scatter(
    x=np.arange(1, len(learning_rates) + 1),
    y=learning_rates,
    mode='markers+lines',
    marker=dict(symbol='circle', size=8),
    line=dict(width=2, backoff=0.7)
))

fig_learning_rates.update_layout(
    title="Learning Rate Adjustment Over Epochs",
    xaxis_title="Epochs",
    yaxis_title="Learning Rate",
    font=dict(family="Times New Roman"),
)

# Save the interactive plots as JPEG
fig_losses.write_image("training_losses_plot.jpg", scale=6)  # 600 DPI
fig_learning_rates.write_image("learning_rates_plot.jpg", scale=6)  # 600 DPI





# Assuming 'autoencoder' is your trained model
torch.save(autoencoder.state_dict(), 'autoencoder_modelV19Estuaries.pth')
# Assuming 'Autoencoder' is the class you defined for your model
model = Autoencoder(input_size, hidden_size, num_estuaries)
model.load_state_dict(torch.load('autoencoder_modelV19Estuaries.pth'))
model.to(device)




# Initialize the dataframe

df = pd.DataFrame()
df2 = pd.DataFrame()

# Use the trained model to obtain reconstruction errors for all estuaries
for estuary_idx in range(num_estuaries):
    # Extract data for the current estuary
    estuary_data = sequences[:, :, estuary_idx * 4: (estuary_idx + 1) * 4]

    # Assuming you have the time indices
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
    
    df[f'Estuary_{estuary_idx + 1}'] = normalized_errors
    df2[f'Estuary_{estuary_idx + 1}'] = anomalies
    
df.index = resampled_data.index[sequence_length-1:]
df2.index = df.index
# Save the dataframe to a CSV file
df.to_csv('reconstruction_errorsv.csv', index=True)
df2.to_csv('abnormallabelsvstd.csv', index=True)