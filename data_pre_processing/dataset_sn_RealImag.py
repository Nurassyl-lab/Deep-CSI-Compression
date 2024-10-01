import torch
import sys, os
import numpy as np
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# import torch nn
import torch.nn as nn
import pandas as pd
# CNN
from models.cnn_autoencoder_RealImag import CNNAutoencoder

from models.model_utils import load_autoencoder

from utils.dataloader_RealImag import create_dataloaders as cd
from utils.check_dataset_het import denormalize
from tqdm import tqdm
import argparse

def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='SIREN', help='Model type')
    parser.add_argument('--dataset_train', type=str, default='', help='Train Dataset path')
    parser.add_argument('--dataset_test', type=str, default='', help='Train Dataset path')
    parser.add_argument('--dataset_val', type=str, default='', help='Train Dataset path')
    parser.add_argument('--het', type=float, default=None, help='Heterogeneity')
    parser.add_argument('--epoch', type=int, default=None, help='Choose epoch to load model from')
    parser.add_argument('--n_class', type=int, default=3, help='Number of classes')
    parser.add_argument('--limit', type=int, default=None, help='Limit the number of samples')
    parser.add_argument('--purpose', type=str, default='train', help='For what purpose')
    return parser.parse_args()

def get_encoded_features(args):
    # Ensure necessary modules and functions are available
    # Import your custom modules or define them here
    # from your_module import CNNAutoencoder, load_autoencoder, cd

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load the dataset
    train_loader, val_loader, test_loader = cd(
        args.dataset_train, args.dataset_test, args.dataset_val, 'min-max', batch_size=1, limit=args.limit
    )

    if args.purpose == 'train':
        main_loader = train_loader
    elif args.purpose == 'test':
        main_loader = test_loader

    # Total number of samples
    total_samples = len(main_loader.dataset)  # Should be 45000 as per your assumption

    # Number of classes
    n_classes = args.n_class  # Should be 3 as per your assumption

    # Pre-allocate an array to hold all encoded features
    # Shape: (total_samples, n_classes, 48)
    all_encoded_features = np.zeros((total_samples, n_classes, 48))
    all_inputs = np.zeros((total_samples, 2, 64, 100))
    # List to store labels
    all_labels = []

    # Load models for each class and store them in a list
    models = []
    for class_idx in range(n_classes):
        model_path = f'./models/{args.model}_het{int(args.het*100)}_class{class_idx+1}/best_autoencoder.pth'
        model = CNNAutoencoder()
        model, _, _ = load_autoencoder(model, model_path)
        model.to(device)
        model.eval()
        models.append(model)

    # Iterate over the dataset and encode each sample with all class models
    for sample_idx, (inputs, label) in enumerate(tqdm(main_loader, desc="Encoding Samples")):
        inputs = inputs.to(device)
        all_labels.append(label.item())

        # add input into all_inputs
        all_inputs[sample_idx, :, :, :] = inputs.cpu().numpy()

        with torch.no_grad():
            for class_idx, model in enumerate(models):
                # Encode the input
                encoded = model.encoder(inputs)  # Shape: (1, 2, 4, 6)
                # Reshape to (1, 48) and convert to NumPy array
                encoded_np = encoded.view(1, -1).cpu().numpy()
                # Store the encoded features
                all_encoded_features[sample_idx, class_idx, :] = encoded_np
    # convert to numpy array
    all_labels = np.array(all_labels).reshape(-1, 1)
    all_encoded_features = np.array(all_encoded_features)
    all_inputs = np.array(all_inputs)
    return all_encoded_features, all_labels, all_inputs, models

if __name__ == '__main__':
    args = arg_parser()
    all_encoded_features, all_labels, all_inputs = get_encoded_features(args)
    print(all_encoded_features.shape)
    print(all_labels.shape)
    print(all_inputs.shape)

    # check if correct
    tmp = pd.read_csv("dataset_RealImag.csv")
    min_real = tmp['min_real'][0]
    max_real = tmp['max_real'][0]
    min_imag = tmp['min_imag'][0]
    max_imag = tmp['max_imag'][0]

    # denormalize the data
    real = all_inputs[:, 0, :, :]
    imag = all_inputs[:, 1, :, :]

    real = denormalize(real, min_real, max_real)
    imag = denormalize(imag, min_imag, max_imag)

    csi = real + 1j * imag
    # get the l2 norm of the csis
    l2_norm = np.linalg.norm(csi, axis=(1, 2))

    # compare all_labels to l2_norm
    accuracy = 0
    for l2, label in zip(l2_norm, all_labels):
        # find class of l2
        class_ = None
        if l2 >= 40 and l2 <= 45:
            class_ = 1
        elif l2 >= 50 and l2 <= 55:
            class_ = 2
        elif l2 >= 60 and l2 <= 65:
            class_ = 3
        if class_ == label:
            accuracy += 1
    accuracy /= len(all_labels)
    print(f'Accuracy: {accuracy}')
