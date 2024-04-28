import numpy as np
import matplotlib.pyplot as plt
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
