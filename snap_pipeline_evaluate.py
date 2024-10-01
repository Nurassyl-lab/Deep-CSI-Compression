from data_pre_processing.dataset_sn_RealImag import get_encoded_features

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

import numpy as np
from sklearn.model_selection import train_test_split
import argparse

from tqdm import tqdm
import os,sys

from utils.dataloader_RealImag import create_dataloaders
from train_sn import CustomDataset, ClassificationModel

from models.cnn_autoencoder_RealImag import CNNAutoencoder
from models.model_utils import load_autoencoder, load_classifier

import pandas as pd

def normalize(x, x_max, x_min):
    y = 2 * ((x - x_min) / (x_max - x_min)) - 1
    return y

def denormalize(normalized_data, min_value, max_value):
    # Denormalize the data from range [-1, 1] to original range
    original_data = (normalized_data + 1) / 2 * (max_value - min_value) + min_value
    return original_data

def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='SIREN', help='Model type')
    parser.add_argument('--dataset_train', type=str, default='', help='Train Dataset path')
    parser.add_argument('--dataset_test', type=str, default='', help='Test Dataset path')
    parser.add_argument('--dataset_val', type=str, default='', help='Validation Dataset path')
    parser.add_argument('--het', type=float, default=None, help='Heterogeneity')
    parser.add_argument('--n_class', type=int, default=3, help='Number of classes')
    parser.add_argument('--purpose', type=str, default='train', help='For what purpose')
    return parser.parse_args()

def main():
    # Parse arguments
    args = arg_parser()

    # Get dataset information
    tmp = pd.read_csv("dataset_RealImag.csv")
    
    min_real = tmp['min_real'][0]
    max_real = tmp['max_real'][0]
    min_imag = tmp['min_imag'][0]
    max_imag = tmp['max_imag'][0]

    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    _, _, test_loader = create_dataloaders(args.dataset_train, args.dataset_test, args.dataset_val, 
                                           norm='min-max', batch_size=32, limit=None)
    
    # Load 3 autoencoders
    autoencoders = []
    for class_number in range(args.n_class):
        model = CNNAutoencoder()
        model, _, _ = load_autoencoder(
            model,
            model_path=f"./models/{args.model}_het{int(args.het*100)}_class{class_number+1}/best_autoencoder.pth"
        )
        model.eval()  # Set model to evaluation mode
        autoencoders.append(model)

    # Load classifier
    classifier = ClassificationModel(3*48, 3)
    classifier = load_classifier(
        classifier,
        model_path=f"./models/{args.model}_classification_het{int(args.het*100)}/best_classifier.pth"
    )
    classifier.eval()  # Set classifier to evaluation mode

    # Criterion for reconstruction loss
    criterion = nn.MSELoss()

    # Initialize total loss
    total_loss = 0.0
    total_samples = 0

    total_loss_denorm = 0.0

    # Assume test_loader is provided
    # use tqdm
    selected_labels = []
    for data, labels in tqdm(test_loader, desc='Testing'):
        data = data.to(device)
        batch_size = data.size(0)
        total_samples += batch_size

        # Encode the data using 3 encoders
        encoded_features = []
        for ae in autoencoders:
            encoded = ae.encoder(data)  # (batch, 1, 4, 6)
            encoded = encoded.view(batch_size, 1, -1)  # Flatten to (batch, 1, 24)
            # If necessary, pad or transform to (batch, 1, 48)
            encoded = nn.functional.pad(encoded, (0, 48 - encoded.size(2)))
            encoded_features.append(encoded)

        # Combine encoded data to shape (batch, 3, 48)
        classifier_input = torch.cat(encoded_features, dim=1)  # (batch, 3, 48)

        # Prepare input for classifier
        #classifier_input = combined_encoded.view(batch_size, -1)  # (batch, 3*48)

        # Get classifier predictions
        outputs = classifier(classifier_input)  # (batch, num_classes)

        _, predicted_labels = torch.max(outputs, dim=1)  # (batch,)
        
        # Initialize list to collect losses
        batch_losses = []
        batch_losses_denorm = []
        # Process each sample in the batch
        for idx in range(batch_size):
            predicted_class = predicted_labels[idx].item()
            selected_labels.append(predicted_class)
            # Select the corresponding encoded feature and decoder
            selected_encoded = encoded_features[predicted_class][idx]  # (1, 1, 48)
            
            # make sure that selected_encoded is of shape (1, 2, 4, 6)
            selected_encoded = selected_encoded.view(1, 2, 4, 6)

            decoder = autoencoders[predicted_class].decoder

            # Decode the selected encoded feature
            decoded_output = decoder(selected_encoded)

            # Calculate MSE loss between decoded output and input data
            data_to_compare = data[idx].view(1, 2, 64, 100)

            # display shapes of decoded_output and data
            loss = criterion(decoded_output, data_to_compare)
            batch_losses.append(loss.item())

            # Denormalize the data
            decoded_real, decoded_imag = decoded_output[:, 0], decoded_output[:, 1]
            data_real, data_imag = data_to_compare[:, 0], data_to_compare[:, 1]

            # denormalize the data
            decoded_real = denormalize(decoded_real, min_real, max_real)
            decoder_imag = denormalize(decoded_imag, min_imag, max_imag)

            data_real = denormalize(data_real, min_real, max_real)
            data_imag = denormalize(data_imag, min_imag, max_imag)

            # combine into (1, 2, 64, 100)
            decoded_output = torch.stack((decoded_real, decoded_imag), dim=1)
            data_to_compare = torch.stack((data_real, data_imag), dim=1)

            # Calculate MSE loss between decoded output and input data
            loss_denorm = criterion(decoded_output, data_to_compare)
            batch_losses_denorm.append(loss_denorm.item())

        # Accumulate the total loss
        total_loss += sum(batch_losses)
        total_loss_denorm += sum(batch_losses_denorm)


    # Compute the average loss
    avg_loss = total_loss / total_samples
    avg_loss_denorm = total_loss_denorm / total_samples
    print(f'Average MSE Loss: {avg_loss}')
    print(f'Average MSE Loss (Denormalized): {avg_loss_denorm}')
    
    # count the number of each class in the selected_labels
    unique, counts = np.unique(selected_labels, return_counts=True)
    print(dict(zip(unique, counts)))

if __name__ == '__main__':
    main()
