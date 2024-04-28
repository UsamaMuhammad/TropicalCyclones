import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
from torch.nn import functional as F
from sklearn.preprocessing import RobustScaler
import matplotlib.pyplot as plt
from matplotlib import rcParams
import numpy as np
from utils import replace_outliers_with_nearest, create_sequences, plot_sequence_and_error
import plotly.graph_objects as go
import plotly.express as px

from plotly.subplots import make_subplots


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


# Initialize and train the autoencoder for each estuary
input_size = 4  # You should set this based on your actual input size
hidden_size = 64  # You can adjust this based on your needs
num_estuaries = 19  # Number of estuaries
initial_lr = 0.01  # You can adjust this based on your needs
num_epochs = 2  # You can adjust this based on your needs
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
