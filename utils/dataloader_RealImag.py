import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
import numpy as np


class CustomToTensor:
    def __call__(self, img):
        return torch.tensor(np.array(img), dtype=torch.float32)


class CSI_Dataset(Dataset):
    def __init__(self, data_info, max_real, min_real, max_imag, min_imag, norm_method='min-max'):
        self.data_info = data_info
        self.max_real = max_real
        self.min_real = min_real
        self.max_imag = max_imag
        self.min_imag = min_imag
        self.norm_method = norm_method

    def __len__(self):
        return len(self.data_info)

    def __getitem__(self, idx):
        # Get the file name from the 'file' column
        # file_name = self.data_info.iloc[idx]['file']
        file_path = self.data_info.iloc[idx]['file']
        label = self.data_info.iloc[idx]['label']
        
        # Load the .npy file and compute the magnitude
        csi_data = np.load(file_path)
        real = csi_data.real
        imag = csi_data.imag

        real = 2 * ((real - self.min_real) / (self.max_real - self.min_real)) - 1
        # imag = (imag - self.min_imag) / (self.max_imag - self.min_imag)
        imag = 2 * ((imag - self.min_imag) / (self.max_imag - self.min_imag)) - 1


        csi_data_normalized = np.array([real, imag]).reshape(2, 64, 100)
        csi_data_normalized = torch.tensor(csi_data_normalized, dtype=torch.float32)
        
        return csi_data_normalized, label


def create_dataloaders(dataset_train, dataset_test, dataset_val, norm, batch_size=32, shuffle=True, num_workers=0, limit=None):
    print('Creating DataLoaders...')

    # Load the full CSV
    df = pd.read_csv('dataset_RealImag.csv')
    train_df = pd.read_csv(dataset_train)
    #shuffle train
    train_df = train_df.sample(frac=1)

    test_df = pd.read_csv(dataset_test)
    val_df = pd.read_csv(dataset_val)

    if limit is not  None:
        train_df = train_df[:limit]

    # Extract dataset mean and std for normalization
    dataset_max_real = df['max_real'][0]
    dataset_min_real = df['min_real'][0]
    dataset_max_imag = df['max_imag'][0]
    dataset_min_imag = df['min_imag'][0]

    # Create Dataset instances for train, validation, and test
    train_dataset = CSI_Dataset(train_df, dataset_max_real, dataset_min_real, dataset_max_imag, dataset_min_imag, norm_method=norm)
    val_dataset = CSI_Dataset(val_df, dataset_max_real, dataset_min_real, dataset_max_imag, dataset_min_imag, norm_method=norm)
    test_dataset = CSI_Dataset(test_df, dataset_max_real, dataset_min_real, dataset_max_imag, dataset_min_imag, norm_method=norm)

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    print(f'Train DataLoader size: {len(train_loader)} batches')
    print(f'Validation DataLoader size: {len(val_loader)} batches')
    print(f'Test DataLoader size: {len(test_loader)} batches')

    print('DataLoaders created\n')
    return train_loader, val_loader, test_loader

#==================================================================================================

class CSI_Het_Dataset(Dataset):
    def __init__(self, data_info, max_real, min_real, max_imag, min_imag, norm_method='min-max', clus_number=1, het=0.1, dataset_limit=20000):
        self.data_info = data_info
        self.max_real = max_real
        self.min_real = min_real
        self.max_imag = max_imag
        self.min_imag = min_imag
        self.norm_method = norm_method
        self.clus_number = clus_number
        self.het = het
        self.dataset_limit = dataset_limit

        # Adjust the dataset to create bias based on het and clus_number
        self._create_biased_dataset()

    def _create_biased_dataset(self):
        # Total number of samples to include
        num_samples = self.dataset_limit

        # Number of classes
        unique_labels = self.data_info['label'].unique()
        # if zero is in the unique_labels, add + 1 to the whole array
        if 0 in unique_labels:
            unique_labels += 1
        num_classes = len(unique_labels)

        # Classes other than the specified cluster number
        other_classes = [label for label in unique_labels if label != self.clus_number]
        num_other_classes = len(other_classes)

        # Calculate the number of samples for the specified class
        num_class_samples = int(num_samples * self.het)

        # Calculate the number of samples for other classes
        num_other_samples_total = num_samples - num_class_samples

        # Avoid division by zero
        if num_other_classes > 0:
            num_other_samples_per_class = num_other_samples_total // num_other_classes
            remainder = num_other_samples_total % num_other_classes
        else:
            num_other_samples_per_class = 0
            remainder = 0

        # Get samples from the specified class
        class_data = self.data_info[self.data_info['label'] == self.clus_number]
        class_samples = class_data.sample(num_class_samples, replace=True)

        # Get samples from other classes
        other_samples_list = []
        for idx, other_class in enumerate(other_classes):
            other_class_data = self.data_info[self.data_info['label'] == other_class]
            # Distribute the remainder among the first few classes
            samples_to_draw = num_other_samples_per_class + (1 if idx < remainder else 0)
            other_samples = other_class_data.sample(samples_to_draw, replace=True)
            other_samples_list.append(other_samples)

        # Combine all samples into one DataFrame
        self.biased_data_info = pd.concat([class_samples] + other_samples_list).reset_index(drop=True)

    def __len__(self):
        return len(self.biased_data_info)

    def __getitem__(self, idx):
        # Get the file path and label from the biased data
        file_path = self.biased_data_info.iloc[idx]['file']
        label = self.biased_data_info.iloc[idx]['label']
        
        # Load the .npy file
        csi_data = np.load(file_path)
        real = csi_data.real
        imag = csi_data.imag

        # Normalize between 0 and 1 or 1 and -1
        # real = (real - self.min_real) / (self.max_real - self.min_real)
        real = 2 * ((real - self.min_real) / (self.max_real - self.min_real)) - 1
        # imag = (imag - self.min_imag) / (self.max_imag - self.min_imag)
        imag = 2 * ((imag - self.min_imag) / (self.max_imag - self.min_imag)) - 1

        # Stack real and imaginary parts
        csi_data_normalized = np.stack((real, imag), axis=0)
        csi_data_normalized = torch.tensor(csi_data_normalized, dtype=torch.float32)
        
        return csi_data_normalized, label


def create_het_dataloaders(dataset_train, dataset_test, dataset_val, norm, 
                           batch_size=32, shuffle=True, num_workers=0, test_size=0.2, val_size=0.1, limit=None, 
                           dataset_RealImag='dataset_RealImag.csv', 
                           clus_number=1, het=0.1):
    print('Creating Heteregeneous DataLoaders...')

    # Load the full CSV
    df = pd.read_csv('dataset_RealImag.csv')
    train_df = pd.read_csv(dataset_train)
    test_df = pd.read_csv(dataset_test)
    val_df = pd.read_csv(dataset_val)

    if limit is not  None:
        train_df = train_df[:limit]

    # Extract dataset mean and std for normalization
    dataset_max_real = df['max_real'][0]
    dataset_min_real = df['min_real'][0]
    dataset_max_imag = df['max_imag'][0]
    dataset_min_imag = df['min_imag'][0]

    # Create Dataset instances for train, validation, and test
    train_dataset = CSI_Het_Dataset(train_df, dataset_max_real, dataset_min_real, dataset_max_imag, dataset_min_imag, 
                                norm_method=norm, 
                                clus_number=clus_number, het=het, dataset_limit=15000)
    val_dataset = CSI_Het_Dataset(val_df, dataset_max_real, dataset_min_real, dataset_max_imag, dataset_min_imag, 
                              norm_method=norm,
                              clus_number=clus_number, het=het, dataset_limit=1000)
    test_dataset = CSI_Het_Dataset(test_df, dataset_max_real, dataset_min_real, dataset_max_imag, dataset_min_imag, 
                               norm_method=norm,
                               clus_number=clus_number, het=het, dataset_limit=1000)

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
