import torch
from thop import profile
import sys, os
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import mean_squared_error
import numpy as np
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# import torch nn
import torch.nn as nn
# CNN
from models.cnn_autoencoder_RealImag import CNNAutoencoder

from models.model_utils import load_autoencoder

from utils.dataloader_RealImag import create_het_dataloaders as cd_het
from utils.check_dataset_het import evaluate_dataset

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
    return parser.parse_args()

if __name__ == '__main__':
    args = arg_parser()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    all_losses = [[] for _ in range(args.n_class)]
    for class_data in range(args.n_class):
        # Load the dataset for class --> class_data
        train_loader, val_loader, test_loader = cd_het(args.dataset_train, args.dataset_test, args.dataset_val, 'min-max',
                                                       het=args.het, clus_number=class_data+1,
                                                       batch_size=1, limit=None)

        # length of the test loader
        print(f'Length of Test Loader for class {class_data}: {len(test_loader)}')

        print('Information about Test Loader for class {}'.format(class_data+1))
        evaluate_dataset(test_loader)
        print('\n')

        class_losses = [[] for _ in range(args.n_class)]
        for class_model in range(args.n_class):
            model_path = f'./models/{args.model}_het{int(args.het*100)}_class{class_model+1}/best_autoencoder.pth'
            model = CNNAutoencoder()
            model, _, _ = load_autoencoder(model, model_path)
            model.to(device)
            model.eval()
            criterion = nn.MSELoss()
            
            batch_count = 0
            for inputs, _ in test_loader:
                inputs = inputs.to(device)
                with torch.no_grad():
                    encoded = model.encoder(inputs)
                    outputs = model.decoder(encoded)
                    loss = criterion(outputs, inputs)
                    #print(f'Class {class_data+1} - Model {class_model+1} - Batch {batch_count} - Test Loss: {loss.item()}')
                    batch_count += 1
                    class_losses[class_model].append(loss.item())
        
        print('\n')
        for class_model in range(args.n_class):    
            print(f'Model {class_data+1} - Average Test Loss: {np.mean(class_losses[class_model])} on dataset of class {class_data+1}')
            all_losses[class_data].append(np.mean(class_losses[class_model]))
        print('\n')
        print('====================================================\n')
        


    print('ALL LOSSES, each row is a dataset, and each column is a model')
    print(np.array(all_losses))