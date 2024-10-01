from data_pre_processing.dataset_sn_RealImag import get_encoded_features

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

import numpy as np
from sklearn.model_selection import train_test_split
import argparse

from tqdm import tqdm
import os

# Define the argument parser
def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='SIREN', help='Model type')
    parser.add_argument('--dataset_train', type=str, default='', help='Train Dataset path')
    parser.add_argument('--dataset_test', type=str, default='', help='Test Dataset path')
    parser.add_argument('--dataset_val', type=str, default='', help='Validation Dataset path')
    parser.add_argument('--het', type=float, default=None, help='Heterogeneity')
    parser.add_argument('--epoch', type=int, default=None, help='Choose epoch to load model from')
    parser.add_argument('--n_class', type=int, default=3, help='Number of classes')
    parser.add_argument('--limit', type=int, default=None, help='Limit the number of samples')
    return parser.parse_args()

class CustomDataset(Dataset):
    def __init__(self, data, labels, transform=None):
        self.data = torch.from_numpy(data.astype(np.float32))
        self.labels = torch.from_numpy(labels.astype(np.long)).squeeze() - 1  # Adjust labels to be 0-based
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        label = self.labels[idx]
        if self.transform:
            sample = self.transform(sample)
        return sample, label


# Define the neural network model
class ClassificationModel(nn.Module):
    def __init__(self, input_size, num_classes=3):
        super(ClassificationModel, self).__init__()
        self.fc1 = nn.Linear(input_size, 256)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(256, 128)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten input
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        return x

def main():
    # Parse arguments
    args = arg_parser()
    # add another parameter to args which is purpose and set it to 'train'
    args.purpose = 'train'
    # Get data and labels
    data, labels, _, _ = get_encoded_features(args)

    # Print the shape of data and labels
    print(f'Shape of data: {data.shape}')
    print(f'Shape of labels: {labels.shape}')

    # Display unique labels
    print(f'Unique labels: {np.unique(labels)}')

    # Split data into training, validation, and test sets
    train_data, temp_data, train_labels, temp_labels = train_test_split(
        data, labels, test_size=0.4, random_state=42
    )
    val_data, test_data, val_labels, test_labels = train_test_split(
        temp_data, temp_labels, test_size=0.5, random_state=42
    )

    # Create datasets
    train_dataset = CustomDataset(train_data, train_labels)
    val_dataset = CustomDataset(val_data, val_labels)
    test_dataset = CustomDataset(test_data, test_labels)

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    # Define the device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Define the model
    input_size = 3 * 48  # Adjust based on your data
    num_classes = args.n_class
    model = ClassificationModel(input_size, num_classes)
    model.to(device)

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Prepare to save the best model
    best_val_loss = float('inf')
    if args.het is None:
        het_int = 0
    else:
        het_int = int(args.het * 100)
    model_path = f'./models/{args.model}_classification_het{het_int}/best_classifier.pth'
    os.makedirs(f'./models/{args.model}_classification_het{het_int}/', exist_ok=True)

    # Training loop
    num_epochs = 10
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0

        train_loader_tqdm = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} - Training")
        for inputs, labels in train_loader_tqdm:
            inputs = inputs.to(device)
            labels = labels.to(device)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # Backward pass and optimize
            loss.backward()
            optimizer.step()

            # Statistics
            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()

            # Update tqdm
            train_loader_tqdm.set_postfix(loss=loss.item())

        epoch_loss = running_loss / len(train_dataset)
        epoch_acc = 100 * correct_train / total_train

        # Validation
        model.eval()
        val_loss = 0.0
        correct_val = 0
        total_val = 0
        val_loader_tqdm = tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} - Validation")
        with torch.no_grad():
            for inputs, labels in val_loader_tqdm:
                inputs = inputs.to(device)
                labels = labels.to(device)

                outputs = model(inputs)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs.data, 1)
                total_val += labels.size(0)
                correct_val += (predicted == labels).sum().item()

                # Update tqdm
                val_loader_tqdm.set_postfix(loss=loss.item())

        val_epoch_loss = val_loss / len(val_dataset)
        val_epoch_acc = 100 * correct_val / total_val

        print(f'Epoch {epoch+1}/{num_epochs}, '
              f'Train Loss: {epoch_loss:.4f}, Train Acc: {epoch_acc:.2f}%, '
              f'Val Loss: {val_epoch_loss:.4f}, Val Acc: {val_epoch_acc:.2f}%')

        # Save the best model
        if val_epoch_loss < best_val_loss:
            best_val_loss = val_epoch_loss
            torch.save(model.state_dict(), model_path)
            print(f'Best model saved with val loss: {best_val_loss:.4f}')

    print('Training complete')

    # Load the best model
    model.load_state_dict(torch.load(model_path))
    model.eval()

    # Test the model
    test_loss = 0.0
    correct_test = 0
    total_test = 0
    test_loader_tqdm = tqdm(test_loader, desc="Testing")
    with torch.no_grad():
        for inputs, labels in test_loader_tqdm:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            test_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total_test += labels.size(0)
            correct_test += (predicted == labels).sum().item()

            # Update tqdm
            test_loader_tqdm.set_postfix(loss=loss.item())

    test_epoch_loss = test_loss / len(test_dataset)
    test_epoch_acc = 100 * correct_test / total_test

    print(f'Test Loss: {test_epoch_loss:.4f}, Test Acc: {test_epoch_acc:.2f}%')


if __name__ == '__main__':
    main()
