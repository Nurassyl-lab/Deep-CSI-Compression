import numpy as np
import matplotlib.pyplot as plt
import tensorly as tl
from tensorly.decomposition import parafac
from numpy.fft import fft2, fftshift
tl.set_backend('numpy')


# Function that plots histograms for the dataset
def plot_histogram(x, title='Data', bins=30):
    print('Plotting a histogram...')
    
    x = np.array(x)
    if x.ndim > 1:
        x = x.flatten()

    plt.figure(figsize=(10, 5))
    plt.hist(x, bins=bins, density=True)

    plt.axvline(np.mean(x), color='k', linestyle='dashed', linewidth=1)
    plt.axvline(np.mean(x) + np.std(x), color='r', linestyle='dashed', linewidth=1)
    plt.axvline(np.mean(x) - np.std(x), color='r', linestyle='dashed', linewidth=1)

    plt.legend(['Mean', 'Mean + Std', 'Mean - Std'])

    plt.title(f'Histogram of {title}')
    plt.savefig(f'./plots/{title}_histogram.png')


# Functions that takes tensor
# performs parafac
# plots the parafac components
def plot_parafac(x, rank=3):
    print('Plotting Parafac Components...')
    
    factors = parafac(x, rank=rank)
    fig, axs = plt.subplots(rank, 1, figsize=(10, 10))
    for i, factor in enumerate(factors[1]):
        axs[i].plot(factor)
        axs[i].set_title(f'Parafac Component {i}')
    plt.savefig(f'./plots/parafac_components.png')


def plot_fft(x, y, model_name):
    print('Plotting FFT...')
    
    x = fftshift(fft2(x))
    y = fftshift(fft2(y))

    fig, axs = plt.subplots(1, 2, figsize=(6, 4))  # Adjusted the figure size for better visualization

    # Plot the FFT of the input
    im1 = axs[0].imshow(np.abs(x))
    axs[0].set_title('FFT of Input')
    fig.colorbar(im1, ax=axs[0])  # Add colorbar to the first subplot

    # Plot the FFT of the output
    im2 = axs[1].imshow(np.abs(y))
    axs[1].set_title('FFT of Output')
    fig.colorbar(im2, ax=axs[1])  # Add colorbar to the second subplot

    plt.savefig(f'./models/{model_name}/output_fft_magnitude.png')

def plot_out(x, y, model_name):
    print('Plotting Output...')
    
    fig, axs = plt.subplots(1, 2, figsize=(6, 4))  # Adjusted the figure size for better visualization

    # Plot the input
    im1 = axs[0].imshow(x)
    axs[0].set_title('Input')
    fig.colorbar(im1, ax=axs[0])  # Add colorbar to the first subplot

    # Plot the output
    im2 = axs[1].imshow(y)
    axs[1].set_title('Output')
    fig.colorbar(im2, ax=axs[1])  # Add colorbar to the second subplot

    plt.savefig(f'./models/{model_name}/output.png')
