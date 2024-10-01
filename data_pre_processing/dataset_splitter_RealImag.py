# splits datasets into training and testing datasets
import argparse
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import pandas as pd

def get_argparse():
    parser = argparse.ArgumentParser(description='Split dataset into training and testing datasets')
    parser.add_argument('--dataset', type=str, help='Path to the dataset file')
    parser.add_argument('--clustering', type=str, help='Clustering Method (SNR/PARAFAC)')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    created_datasets = []

    args = get_argparse()
    np.random.seed(42)

    # Load the dataset
    dataset = pd.read_csv(args.dataset)

    # from l2_labels column read all the labels
    if args.clustering == 'SNR':
        labels = dataset['l2_labels'].values
    elif args.clustering == 'PARAFAC':
        labels = dataset['parafac_labels'].values

    labels = np.array(labels)

    # get the unique labels
    print(f'For {args.clustering} clustering method, the unique labels are: {np.unique(labels)}')

    # For each unique label, get the number of samples using np.unique and np.where
    for label in np.unique(labels):
        indices = np.where(labels == label)[0]
        print(f'Label {label} has {len(indices)} samples')
    
    # first 60000 samples are training,
    # create a copy and save it
    train_dataset = dataset.iloc[:60000].copy()
    train_dataset.to_csv(f'{args.clustering}_balanced_train_dataset.csv', index=False)

    # last 5000 samples are testing,
    # create a copy and save it
    test_dataset = dataset.iloc[len(dataset) - 5000:].copy()
    test_dataset.to_csv(f'{args.clustering}_balanced_test_dataset.csv', index=False)

    created_datasets.append('balanced_train_dataset.csv')
    created_datasets.append('balanced_test_dataset.csv')

    hets = [0.1, 0.5, 1.0]
    classes = [0, 1, 2]
    limit = 20000
    # Now using hets, create 3 datasets with 10%, 50%, and 100% heterogeneity level
    for het in hets:
        for class_ in classes:
            # create a dataset so that it has het% of bias towards class_
            # get indices of rows class_
            class_indices = np.where(labels == class_)[0]
            
            # the number of samples to be biased
            # total number of samples should be limit
            # samples of class_ should be het% of limit
            # the rest of the samples should be equally distributed among the other classes
            num_samples = int(limit * het)
            num_samples_per_class = int((limit - num_samples) / 2)
            # get the samples of class_
            class_samples = class_indices[:num_samples]
            # get the samples of other classes
            other_samples = []
            for c in classes:
                if c != class_:
                    other_samples.extend(np.where(labels == c)[0][:num_samples_per_class])
            # combine the samples
            indices = np.concatenate([class_samples, other_samples])
            # shuffle the indices
            np.random.shuffle(indices)
            # get the samples
            biased_dataset = dataset.iloc[indices].copy()
            biased_dataset.to_csv(f'{args.clustering}_biased_{int(het*100)}_{class_+1}_dataset.csv', index=False)
