import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import numpy as np
from sklearn.model_selection import train_test_split

from data_pre_processing.dataset_sn_RealImag import get_encoded_features
from utils.dataloader_RealImag import create_dataloaders
from models.cnn_autoencoder_RealImag import CNNAutoencoder
from models.model_utils import load_autoencoder, load_classifier

import pandas as pd
from tqdm import tqdm
import os, sys
import argparse

# use torch summary
from torchsummary import summary

def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='SIREN', help='Model type')
    parser.add_argument('--dataset_train', type=str, default='', help='Train Dataset path')
    parser.add_argument('--dataset_test', type=str, default='', help='Test Dataset path')
    parser.add_argument('--dataset_val', type=str, default='', help='Validation Dataset path')
    parser.add_argument('--het', type=float, default=None, help='Heterogeneity')
    parser.add_argument('--n_class', type=int, default=3, help='Number of classes')
    parser.add_argument('--purpose', type=str, default='train', help='For what purpose')
    parser.add_argument('--limit', type=int, default=None, help='Limit the number of samples')
    return parser.parse_args()

args = arg_parser()

# create_dataloaders function from data_pre_processing/dataset_sn_RealImag.py
train_loader, val_loader, test_loader = create_dataloaders(args.dataset_train, args.dataset_test, args.dataset_val, 
                                           norm='min-max', batch_size=32, limit=args.limit)


# dataloaders outputing data, labels
# use labels = [] to get all labels from test_loader
# I want to check if labels are uniform or not

all_labels_test = []
for data, labels in test_loader:
    all_labels_test.append(labels)

model1 = CNNAutoencoder()
model2 = CNNAutoencoder()
model3 = CNNAutoencoder()

# Load your pre-trained autoencoders
autoencoder1,_,_ = load_autoencoder(model1, 'models/CNN_het100_class1/best_autoencoder.pth')
autoencoder2,_,_ = load_autoencoder(model2, 'models/CNN_het100_class2/best_autoencoder.pth')
autoencoder3,_,_ = load_autoencoder(model3, 'models/CNN_het100_class3/best_autoencoder.pth')


# Extract encoders and decoders
encoder1 = autoencoder1.encoder
decoder1 = autoencoder1.decoder

encoder2 = autoencoder2.encoder
decoder2 = autoencoder2.decoder

encoder3 = autoencoder3.encoder
decoder3 = autoencoder3.decoder


# Function to flatten encoded features
def flatten_encoded_features(encoded):
    # encoded shape: (batch_size, channels, height, width)
    batch_size = encoded.size(0)
    flat_encoded = encoded.view(batch_size, -1)  # Flatten all dimensions except batch_size
    return flat_encoded

class GatingNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, num_experts=3):
        super(GatingNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, num_experts)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        logits = self.fc2(x)  # Outputs logits for each expert
        return logits

class MixtureOfExperts(nn.Module):
    def __init__(self, encoders, gating_network, decoders):
        super(MixtureOfExperts, self).__init__()
        self.encoders = encoders
        self.gating_network = gating_network
        self.decoders = decoders

    def forward(self, x, hard=True):
        # Encode input with all encoders and flatten
        encoded_outputs = []
        encoded_outputs_non_flat = []
        for encoder in self.encoders:
            encoded = encoder(x)
            flat_encoded = flatten_encoded_features(encoded)
            encoded_outputs.append(flat_encoded)
            encoded_outputs_non_flat.append(encoded)
        
        # Concatenate flattened encoded outputs for gating network
        gating_input = torch.cat(encoded_outputs, dim=1)
        
        # Gating network to compute logits and probabilities
        gating_logits = self.gating_network(gating_input)
        gating_probs = F.softmax(gating_logits, dim=1)  # Shape: (batch_size, num_experts)
        
        if hard:
            # **Hard selection for inference**
            selected_index = torch.argmax(gating_probs, dim=1)  # Shape: (batch_size,)
            
            # Gather the selected encoded feature
            encoded_stack = torch.stack(encoded_outputs_non_flat, dim=1)  # Shape: (batch_size, num_experts, channels, height, width)
            batch_size = x.size(0)
            channels, height, width = encoded_outputs_non_flat[0].size()[1:]
            indices = selected_index.unsqueeze(1).unsqueeze(2).unsqueeze(3).unsqueeze(4).expand(batch_size, 1, channels, height, width)
            selected_encoded = encoded_stack.gather(1, indices).squeeze(1)  # Shape: (batch_size, channels, height, width)
            
            # Decode with the selected decoder
            outputs = []
            for decoder in self.decoders:
                output = decoder(selected_encoded)
                outputs.append(output)
            
            # Gather the output from the selected decoder
            outputs_stack = torch.stack(outputs, dim=1)  # Shape: (batch_size, num_experts, channels, height, width)
            # display shape of outputs_stack

            indices = selected_index.unsqueeze(1).unsqueeze(2).unsqueeze(3).unsqueeze(4).expand(batch_size, 1, channels, 64, 100)
            selected_output = outputs_stack.gather(1, indices).squeeze(1)  # Shape: (batch_size, channels, height, width)
            
            return selected_output, selected_index
        else:
            # **Soft selection for training**
            # Stack encoded features (non-flattened)
            encoded_stack = torch.stack(encoded_outputs_non_flat, dim=1)  # Shape: (batch_size, num_experts, channels, height, width)
            
            # Compute weighted sum of encoded features
            gating_probs_expanded = gating_probs.unsqueeze(2).unsqueeze(3).unsqueeze(4)  # Shape: (batch_size, num_experts, 1, 1, 1)
            selected_encoded = torch.sum(encoded_stack * gating_probs_expanded, dim=1)  # Shape: (batch_size, channels, height, width)
            
            # Decode with all decoders
            outputs = []
            for decoder in self.decoders:
                output = decoder(selected_encoded)
                outputs.append(output)
            
            # Stack outputs
            outputs_stack = torch.stack(outputs, dim=1)  # Shape: (batch_size, num_experts, channels, height, width)
            
            # Compute weighted sum of decoder outputs
            selected_output = torch.sum(outputs_stack * gating_probs_expanded, dim=1)  # Shape: (batch_size, channels, height, width)
            
            return selected_output, gating_probs

# Hyperparameters
input_size = 48 * 3       # Calculated earlier
hidden_size = 128         # Adjust as needed
num_experts = 3           # Number of decoders

# Instantiate the gating network
gating_network = GatingNetwork(input_size=input_size, hidden_size=hidden_size, num_experts=num_experts)

# Collect encoders and decoders
encoders = [encoder1, encoder2, encoder3]
decoders = [decoder1, decoder2, decoder3]


# print('Defined Mixture of Experts model')
# Instantiate the Mixture of Experts model
moe_model = MixtureOfExperts(encoders, gating_network, decoders)

# Move model to device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
moe_model.to(device)

criterion = nn.MSELoss()
optimizer = optim.Adam(gating_network.parameters(), lr=0.001)

if args.purpose == 'train':
    num_epochs = 30

    for epoch in range(num_epochs):
        moe_model.train()
        running_loss = 0.0
        for batch_inputs, _ in tqdm(train_loader):
            batch_inputs = batch_inputs.to(device)
            # Forward pass with soft selection
            outputs, gating_probs = moe_model(batch_inputs, hard=False)
            # Ensure outputs and batch_inputs have the same shape
            assert outputs.shape == batch_inputs.shape, f"Output shape {outputs.shape} does not match input shape {batch_inputs.shape}"
            # Compute loss
            loss = criterion(outputs, batch_inputs)
            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * batch_inputs.size(0)
        # save the model in ./models/moe.pth
        torch.save(moe_model.state_dict(), './models/moe.pth')
        epoch_loss = running_loss / len(train_loader.dataset)
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}')
else:
    # # Load the pre-trained model
    # print(f'Loading pre-trained model from ./models/moe.pth')
    # moe_model.load_state_dict(torch.load('./models/moe.pth'))
    # moe_model.eval()
    # with torch.no_grad():
    #     test_loss = 0.0
    #     for batch_inputs, _ in tqdm(test_loader):
    #         batch_inputs = batch_inputs.to(device)
    #         # Forward pass with hard selection
    #         outputs, selected_indices = moe_model(batch_inputs, hard=True)
    #         # Ensure outputs and batch_inputs have the same shape
    #         assert outputs.shape == batch_inputs.shape, f"Output shape {outputs.shape} does not match input shape {batch_inputs.shape}"
    #         # Compute loss
    #         loss = criterion(outputs, batch_inputs)
    #         test_loss += loss.item() * batch_inputs.size(0)
    #     test_loss /= len(test_loader.dataset)
    #     print(f'Test Loss: {test_loss:.4f}')

    # # After loss.backward()
    # # avg_probs = gating_probs.mean(dim=0)
    # avg_probs = selected_indices.mean(dim=0)
    # print(f"Average gating probabilities: {avg_probs.cpu().detach().numpy()}")

    print(f'Loading pre-trained model from ./models/moe.pth')
    moe_model.load_state_dict(torch.load('./models/moe.pth'))
    moe_model.eval()

    # Initialize counters for each class
    total_selected_indices = []

    with torch.no_grad():
        test_loss = 0.0
        for batch_inputs, _ in tqdm(test_loader):
            batch_inputs = batch_inputs.to(device)
            # Forward pass with hard selection
            outputs, selected_indices = moe_model(batch_inputs, hard=True)
            # Ensure outputs and batch_inputs have the same shape
            assert outputs.shape == batch_inputs.shape, f"Output shape {outputs.shape} does not match input shape {batch_inputs.shape}"
            # Compute loss
            loss = criterion(outputs, batch_inputs)
            test_loss += loss.item() * batch_inputs.size(0)
            # Collect selected indices
            total_selected_indices.append(selected_indices.cpu())
        test_loss /= len(test_loader.dataset)
        print(f'Test Loss: {test_loss:.4f}')

    # After evaluation loop
    # Concatenate all selected indices
    total_selected_indices = torch.cat(total_selected_indices, dim=0)
    # Count occurrences of each class
    num_class1 = (total_selected_indices == 0).sum().item()
    num_class2 = (total_selected_indices == 1).sum().item()
    num_class3 = (total_selected_indices == 2).sum().item()
    total_samples = total_selected_indices.size(0)

    print(f"\nDecoder selection counts:")
    print(f"Class 1 (Decoder 1): {num_class1} times ({num_class1 / total_samples * 100:.2f}%)")
    print(f"Class 2 (Decoder 2): {num_class2} times ({num_class2 / total_samples * 100:.2f}%)")
    print(f"Class 3 (Decoder 3): {num_class3} times ({num_class3 / total_samples * 100:.2f}%)")
