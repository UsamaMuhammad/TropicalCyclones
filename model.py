
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torch.nn import functional as F
from sklearn.preprocessing import RobustScaler
from torch.optim.lr_scheduler import ReduceLROnPlateau
import random
import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler


##MODEL
#Encoder
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

#Decoder
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

#Auroencoder model
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
    
#random seed
# Function to set the seed for reproducibility
def set_seed(seed=42):
    """
    Sets the random seed for reproducibility.

    Parameters:
        seed (int): Seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False




#Data Preprocessor

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
        '''
        # Handle outliers using the IQR method
        q1 = df.quantile(0.25)
        q3 = df.quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - iqr_threshold * iqr
        upper_bound = q3 + iqr_threshold * iqr
        df = df.clip(lower=lower_bound, upper=upper_bound, axis=1)
        '''
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
            for i in range(0, tensor_data.size(1) - sequence_length + 1, 24)
        ]
        combined_sequences.extend(sequences)

    # Combine sequences into a single tensor
    combined_sequences = torch.cat(combined_sequences, dim=0)
    return combined_sequences, scalers

#Training wrapper 

def train_combined_autoencoder(sequences, model, criterion, optimizer,scheduler, device, epochs=50, batch_size=32, validation_split=0.2,  plot_interval=5, threshold=0.1):
    """
    Trains the autoencoder on combined sequences with training and validation losses.
    """
    sequences = sequences.to(device)

    # Use custom SequenceDataset
    dataset = SequenceDataset(sequences)

    # Split data into training and validation sets
    val_size = int(len(dataset) * validation_split)
    train_size = len(dataset) - val_size
    train_data, val_data = torch.utils.data.random_split(dataset, [train_size, val_size])

    # Initialize DataLoaders
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False, drop_last=False)

    # Initialize loss tracking
    train_losses = []
    val_losses = []
    learning_rates = []

    for epoch in range(epochs):
        # Training phase
        model.train()
        epoch_train_loss = 0

        for batch in train_loader:
            batch = batch.to(device)
            #print(f"Train Batch shape: {batch.shape}")  # Debugging

            optimizer.zero_grad()
            reconstructed = model(batch)
            loss = criterion(reconstructed, batch)
            loss.backward()
            optimizer.step()

            epoch_train_loss += loss.item()

        avg_train_loss = epoch_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # Validation phase
        model.eval()
        epoch_val_loss = 0
        all_validation_data = []
        all_reconstructed_data = []

        with torch.no_grad():
            for batch in val_loader:
                
                batch = batch.to(device)
                #print(f"Validation Batch shape: {batch.shape}")  # Debugging

                reconstructed = model(batch)
                loss = criterion(reconstructed, batch)
                epoch_val_loss += loss.item()
                
                # Save validation and reconstructed data for plotting
                all_validation_data.append(batch.cpu())
                all_reconstructed_data.append(reconstructed.cpu())

        avg_val_loss = epoch_val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        
        
        # Monitor reconstructed outputs every 5 epochs
        if epoch % 5 == 0:
            print(f"Epoch {epoch}: Train Loss = {avg_train_loss}, Validation Loss = {avg_val_loss}")
            with torch.no_grad():
                sample_data = next(iter(val_loader))
                sample_data = sample_data.to(device)
                # Assuming sequence_length = 24
                sequence_length = sample_data.size(1)
                
                # Plotting
                plt.figure(figsize=(6, 3))  # Adjusted plot size for better readability
                plt.plot(
                    range(sequence_length),
                    sample_data[0, :, 0].cpu().numpy(),
                    label="Original",
                    linestyle="--", linewidth=1.5, alpha=0.8,
                )
                plt.plot(
                    range(sequence_length),
                    reconstructed[0, :, 0].cpu().numpy(),
                    label="Reconstructed",
                    linestyle="-", linewidth=1.5, alpha=0.8,
                )
                
                # Adjust x-axis ticks to multiples of 4 (or any suitable interval based on sequence length)
                xtick_interval = 4
                plt.xticks(
                    np.arange(0, sequence_length + 1, step=xtick_interval),
                    fontsize=10,
                )
                
                # Add labels, legend, and title
                plt.xlabel("Sequence", fontsize=12)
                plt.ylabel("Magnitude", fontsize=12)
                plt.title(f"Original and Reconstructed Sequence at Epoch {epoch}", fontsize=12)
                plt.legend(fontsize=10)
                plt.grid(True, linestyle="--", alpha=0.6)
                plt.tight_layout()
                
                plt.savefig(f"reconstruction_loss_epoch_{epoch}_sequence_{sequence_length}.png", dpi=300)
                plt.show()

                            
        
        # Step the scheduler with validation loss
        scheduler.step(avg_val_loss)

        # Log current learning rate
        current_lr = scheduler.optimizer.param_groups[0]['lr']
        learning_rates.append(current_lr)

        # Print epoch losses and learning rate
        print(f"Epoch [{epoch + 1}/{epochs}] - Train Loss: {avg_train_loss:.4f}, Validation Loss: {avg_val_loss:.4f}, LR: {current_lr:.6f}")
        
        
        
        # Plot normal and abnormal sequences every `plot_interval` epochs
        if (epoch + 1) % plot_interval == 0:
            print(f"Analyzing and plotting reconstruction errors at epoch {epoch + 1}...")
            all_validation_data = torch.cat(all_validation_data, dim=0)
            all_reconstructed_data = torch.cat(all_reconstructed_data, dim=0)

            # Compute reconstruction errors
            errors = F.mse_loss(all_reconstructed_data, all_validation_data, reduction="none").mean(dim=2)
            mean_errors = errors.mean(dim=1).numpy()

            # Identify normal and abnormal sequences
            normal_indices = np.where(mean_errors <= mean_errors.mean())[0]
            abnormal_indices = np.where(mean_errors > mean_errors.mean() * 2)[0]

            # Plot normal sequences
            if len(normal_indices) > 0:
                normal_samples = all_validation_data[normal_indices]
                normal_reconstructed = all_reconstructed_data[normal_indices]
                normal_errors = errors[normal_indices]

                plot_sequence_and_error(
                    epoch,
                    "Validation",
                    "Normal",
                    normal_samples,
                    normal_reconstructed,
                    normal_errors,
                    threshold,
                    title=f"Epoch {epoch + 1} - Normal Sequences",
                )

            # Plot abnormal sequences
            if len(abnormal_indices) > 0:
                abnormal_samples = all_validation_data[abnormal_indices]
                abnormal_reconstructed = all_reconstructed_data[abnormal_indices]
                abnormal_errors = errors[abnormal_indices]

                plot_sequence_and_error(
                    epoch,
                    "Validation",
                    "Abnormal",
                    abnormal_samples,
                    abnormal_reconstructed,
                    abnormal_errors,
                    threshold,
                    title=f"Epoch {epoch + 1} - Abnormal Sequences",
                )
                


    # Return loss histories
    return {"train_losses": train_losses, "val_losses": val_losses, "learning_rate": learning_rates}




################################################
################################################
########### MODEL TRAINING #####################
################################################
################################################
set_seed(42)
torch.backends.cudnn.deterministic = True
# Main workflow
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

input_size = 4  # Number of features (Temp, Sal, DO_mgl, Turb)
hidden_size = 64
sequence_length = 24*7 #1 week
num_epochs = 500
threshold = 0.1    #just for visulization


#

# Combine sequences from all hurricanes
print("Preparing combined data...")
combined_sequences, scalers = prepare_combined_data(filtered_data, sequence_length, smoothing="rolling", window=12)

#combined_sequences, scalers,encoder  = prepare_combined_data_with_onehot(filtered_data, sequence_length, smoothing="exponential", window=3)

num_layers = 5  # Number of LSTM layers
bidirectional = True  # Use bidirectional LSTMs

# Initialize the autoencoder
autoencoder = Autoencoder(input_size, hidden_size, num_layers).to(device)

optimizer = torch.optim.Adam(autoencoder.parameters(), lr=0.01)

# Dynamic Learning Rate Scheduler
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)


# Train the model on combined data
print("\nTraining the autoencoder on combined data...")
loss_history = train_combined_autoencoder(combined_sequences, autoencoder, criterion, optimizer, scheduler,
 device, epochs=500, batch_size=64, validation_split=0.1)



model_save_path = "trained_autoencoder_model_500.pth"

torch.save({
    'model_state_dict': autoencoder.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'scheduler_state_dict': scheduler.state_dict(),  # Optional if using a scheduler
    'epoch': 500,  # Add the current epoch if needed
    'loss_history': loss_history  # Save loss history for future reference
}, model_save_path)
