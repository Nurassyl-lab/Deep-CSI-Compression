import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
import numpy as np


class CustomToTensor:
    def __call__(self, img):
        return torch.tensor(np.array(img), dtype=torch.float32)


class CSI_Dataset(Dataset):
    def __init__(self, data_info, mean, std, max_val, min_val, norm_method):
        self.data_info = data_info
        self.mean = mean
        self.std = std
        self.max = max_val
        self.min = min_val
        self.norm_method = norm_method

    def __len__(self):
        return len(self.data_info)

    def __getitem__(self, idx):
        # Get the file name from the 'file' column
        # file_name = self.data_info.iloc[idx]['file']
        file_path = self.data_info.iloc[idx]['file']
        
        # Load the .npy file and compute the magnitude
        csi_data = np.load(file_path)
        csi_data = csi_data.reshape(1, 64, 100)
        magnitude = np.abs(csi_data)
        
        # Apply z-score normalization
        if self.norm_method == 'z-score':
            normalized_data = (magnitude - self.mean) / self.std # will normalize the data to have a mean of 0 and std of 1
        elif self.norm_method == 'min-max':
            normalized_data = (magnitude - self.min) / (self.max - self.min) # will normalize the data to be between 0 and 1

        # Convert to PyTorch tensor
        magnitude_tensor = torch.tensor(normalized_data, dtype=torch.float32)
        
        return magnitude_tensor


def create_dataloaders(data_path, norm, batch_size=32, shuffle=True, num_workers=0, test_size=0.2, val_size=0.1, limit=None):
    print('Creating DataLoaders...')

    # Load the full CSV
    df = pd.read_csv('dataset.csv')
    train_df = pd.read_csv(data_path)
    if data_path.startswith('SNR'):
        test_df = pd.read_csv('SNR_balanced_test_dataset.csv')
    else:
        test_df = pd.read_csv('PARAFAC_balanced_test_dataset.csv')

    if limit is None:
        limit = len(df)
    else:
        train_df = train_df[:limit]

    # Extract dataset mean and std for normalization
    dataset_mean = df.loc[0, 'mean']
    dataset_std = df.loc[0, 'std']
    dataset_max = df.loc[0, 'max']
    dataset_min = df.loc[0, 'min']

    # Split the dataset into train, validation, and test sets based on indices
    train_df, val_df = train_test_split(train_df, test_size=val_size, shuffle=shuffle)

    # Create Dataset instances for train, validation, and test
    train_dataset = CSI_Dataset(train_df, dataset_mean, dataset_std, dataset_max, dataset_min, norm_method=norm)
    val_dataset = CSI_Dataset(val_df, dataset_mean, dataset_std, dataset_max, dataset_min, norm_method=norm)
    test_dataset = CSI_Dataset(test_df, dataset_mean, dataset_std, dataset_max, dataset_min, norm_method=norm)

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    print(f'Train DataLoader size: {len(train_loader)} batches')
    print(f'Validation DataLoader size: {len(val_loader)} batches')
    print(f'Test DataLoader size: {len(test_loader)} batches')

    print('DataLoaders created\n')
    return train_loader, val_loader, test_loader


if __name__ == '__main__':
    data_path = 'dataset.csv'
    batch_size = 32
    train_loader, val_loader, test_loader = create_dataloaders(data_path, batch_size=batch_size, limit=100)
    print(f'Train: {len(train_loader.dataset)} samples')
    print(f'Validation: {len(val_loader.dataset)} samples')
    print(f'Test: {len(test_loader.dataset)} samples')
    print(f'Batch size: {batch_size}\n')

    print('Train Loader')
    for i, data in enumerate(train_loader):
        print(f'Batch {i+1}: {data.size()}')
        if i == 5:
            break
        print(f'Mean: {data.mean()}')
        print(f'Std: {data.std()}')
        print(f'Min: {data.min()}')
        print(f'Max: {data.max()}\n')

    print('Validation Loader')
    for i, data in enumerate(val_loader):
        print(f'Batch {i+1}: {data.size()}')
        if i == 5:
            break
        print(f'Mean: {data.mean()}')
        print(f'Std: {data.std()}')
        print(f'Min: {data.min()}')
        print(f'Max: {data.max()}\n')

    print('Test Loader')
    for i, data in enumerate(test_loader):
        print(f'Batch {i+1}: {data.size()}')
        if i == 5:
            break
        print(f'Mean: {data.mean()}')
        print(f'Std: {data.std()}')
        print(f'Min: {data.min()}')
        print(f'Max: {data.max()}\n')
