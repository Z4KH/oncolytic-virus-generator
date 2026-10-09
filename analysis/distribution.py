import re
import matplotlib.pyplot as plt
import numpy as np
import os
from pathlib import Path

ANALYSIS_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT = ANALYSIS_DIR.parent / 'out.txt'

def extract_losses_from_file(filename=DEFAULT_INPUT):
    """Extract all loss values from the specified file."""
    losses = []

    try:
        with open(filename, 'r') as file:
            for line in file:
                if 'Loss:' in line:
                    # Extract the numeric value after "Loss:"
                    match = re.search(r'Loss:\s*(\d+\.?\d*)', line)
                    if match:
                        loss_value = float(match.group(1))
                        losses.append(loss_value)
    except FileNotFoundError:
        print(f"Error: File '{filename}' not found.")
        return []
    except Exception as e:
        print(f"Error reading file: {str(e)}")
        return []

    print(f"Extracted {len(losses)} loss values from {filename}")
    return losses

def plot_custom_cdf(losses):
    """Create a custom cumulative distribution plot with annotations."""
    if not losses:
        print("No loss values to plot.")
        return

    # Create a figure for the single visualization
    plt.figure(figsize=(10, 6))

    # Basic statistics
    min_loss = np.min(losses)
    mean_loss = np.mean(losses)

    # Cumulative distribution function
    sorted_losses = np.sort(losses)
    cumulative = np.arange(1, len(sorted_losses) + 1) / len(sorted_losses)

    # Plot the CDF
    plt.plot(sorted_losses, cumulative, marker='.', linestyle='none', alpha=0.5)
    plt.title('Cumulative Distribution of Loss Values', fontsize=14)
    plt.xlabel('Loss Value', fontsize=12)
    plt.ylabel('Cumulative Probability', fontsize=12)
    plt.grid(True, alpha=0.3)

    # Add annotation for minimum value
    plt.annotate('Influenza A',
                 xy=(min_loss, 0.01),  # Point to annotate
                 xytext=(min_loss + 100, 0.1),  # Text position
                 arrowprops=dict(facecolor='black', shrink=0.05, width=1.5),
                 fontsize=10)

    # Add annotation for mean value
    # Find the closest point in the data to the mean
    closest_idx_mean = np.argmin(np.abs(sorted_losses - mean_loss))
    closest_value_mean = sorted_losses[closest_idx_mean]
    closest_cumulative_mean = cumulative[closest_idx_mean]

    plt.axvline(x=mean_loss, color='green', linestyle='--')
    plt.annotate(f'Mean: {mean_loss:.2f}',
                 xy=(mean_loss, closest_cumulative_mean),  # Point to annotate
                 xytext=(mean_loss + 100, closest_cumulative_mean - 0.15),  # Text position
                 arrowprops=dict(facecolor='green', shrink=0.05, width=1.5),
                 fontsize=10,
                 color='green')

    # Add vertical line at 5882
    novel_value = 5882
    # Find the closest point in the data to 5882
    closest_idx = np.argmin(np.abs(sorted_losses - novel_value))
    closest_value = sorted_losses[closest_idx]
    closest_cumulative = cumulative[closest_idx]

    plt.axvline(x=novel_value, color='r', linestyle='--')
    plt.annotate('Novel Generated Sequence',
                 xy=(novel_value, closest_cumulative),  # Point to annotate
                 xytext=(novel_value + 100, closest_cumulative + 0.1),  # Text position
                 arrowprops=dict(facecolor='red', shrink=0.05, width=1.5),
                 fontsize=10,
                 color='red')

    plt.tight_layout()
    plt.savefig(ANALYSIS_DIR / 'loss_distribution_cdf.png', dpi=300)
    plt.show()

def main():
    """Main function to extract loss data and plot the distribution."""
    # Check if the output file exists
    if not DEFAULT_INPUT.exists():
        print("Warning: 'out.txt' not found in the current directory.")
        current_dir = os.getcwd()
        print(f"Current directory: {current_dir}")
        files = os.listdir(current_dir)
        print("Available files:", files)

        # Ask the user for the file path
        file_path = input("Enter the path to the output file: ")
        if not file_path:
            print("No file specified. Exiting.")
            return
    else:
        file_path = DEFAULT_INPUT

    # Extract losses and plot distribution
    losses = extract_losses_from_file(file_path)

    if losses:
        plot_custom_cdf(losses)

        # Save the raw loss values to a CSV
        np.savetxt(ANALYSIS_DIR / 'losses.csv', losses, delimiter=',', header='loss_values')
        print("Loss values saved to 'losses.csv'")

if __name__ == "__main__":
    main()
