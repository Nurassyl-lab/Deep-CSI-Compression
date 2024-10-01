'Train torch models'

import torch
import matplotlib.pyplot as plt
import argparse, os, sys
import pandas as pd
import numpy as np

from utils.dataloader_RealImag import create_dataloaders as cd
from utils.dataloader_RealImag import create_het_dataloaders as cd_het

from utils.data_analysis import plot_fft, plot_out
from models.model_utils import save_encoder_decoder, load_encoder, load_decoder

# CNN
from models.cnn_autoencoder_RealImag import CNNAutoencoder
from models.cnn_autoencoder_RealImag import cnn_train, cnn_test

from utils.check_dataset_het import evaluate_dataset, denormalize, normalize

def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='dense', help='Model type')
    parser.add_argument('--dataset_train', type=str, default='dataset.csv', help='Train Dataset path')
    parser.add_argument('--dataset_test', type=str, default='dataset.csv', help='Test Dataset path')
    parser.add_argument('--dataset_val', type=str, default='dataset.csv', help='Validation Dataset path')
    parser.add_argument('--het', type=float, default=None, help='Hetereogeneity level')
    parser.add_argument('--class_', type=int, default=None, help='Class of bias')
    parser.add_argument('--process', type=str, default='train', help='Process type')
    parser.add_argument('--limit', type=int, default=None, help='Limit the number of samples')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    return parser.parse_args()


def show_performance(train_loss, val_loss, model_name):
    plt.figure(figsize=(7, 6), dpi=300)
    plt.plot(train_loss, label='Train loss')
    plt.plot(val_loss, label='Validation loss')
    plt.legend()
    plt.grid()
    plt.title(f'{model_name} Performance')
    plt.savefig(f'plots/{model_name}.png')
    plt.close()
    print(f'Performance plot saved in plots/{model_name}.png')


if __name__ == '__main__':
    args = arg_parser()

    if args.limit == 0:
        args.limit = None

    # Check if GPU is available    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    # Get dataset information
    tmp = pd.read_csv("dataset_RealImag.csv")
    min_real = tmp['min_real'][0]
    max_real = tmp['max_real'][0]
    min_imag = tmp['min_imag'][0]
    max_imag = tmp['max_imag'][0]

    if args.het is None:
        train_loader, val_loader, test_loader = cd(args.dataset_train, args.dataset_test, args.dataset_val, 
                                                               norm='min-max', 
                                                               batch_size=args.batch_size,
                                                               limit=args.limit)
        
        if args.model.startswith('CNN'):
            #* Define the model
            model_cnn = CNNAutoencoder(model_name=args.model)

            #* Train the model
            if args.process == 'train':
                model_cnn, train_loss, val_loss = cnn_train(model_cnn, train_loader, val_loader, 
                        criterion=torch.nn.MSELoss(), 
                        optimizer=torch.optim.Adam(model_cnn.parameters(), lr=args.lr), 
                        num_epochs=args.epochs)
                
                # Save the model
                save_encoder_decoder(model_cnn, f'./models/{args.model}')
                # Show the performance
                show_performance(train_loss, val_loss, model_name=args.model)

            #* Test the model
            if args.process == 'test':
                encoder = load_encoder(model_cnn, f'./models/{args.model}/encoder.pth')
                decoder = load_decoder(model_cnn, f'./models/{args.model}/decoder.pth')
                
                encoder.eval()
                decoder.eval()

                cnn_test(encoder=encoder, decoder=decoder,
                        model_name=args.model, 
                        criterion=torch.nn.MSELoss(),
                        test_loader=test_loader,
                        real_max=max_real, real_min=min_real,
                        imag_max=max_imag, imag_min=min_imag)

                # #* Visualize the output of the model and compare it with the input
                # for i, data in enumerate(test_loader):
                #     sample = data[0].to(device)
                #     break

                # # Encode and Decode the sample
                # encoded = encoder(sample)
                # decoded = decoder(encoded)

                # # Denormalize the data
                # norm = 'min-max'
                # if norm == 'min-max':
                #     decoded_real, decoded_imag = decoded[:, 0], decoded[:, 1]
                #     # denormalize the data
                #     decoded_real = denormalize(decoded_real, min_real, max_real)
                #     decoded_imag = denormalize(decoded_imag, min_imag, max_imag)
                #     decoded = decoded_real + 1j * decoded_imag

                #     sample_real, sample_imag = sample[:, 0], sample[:, 1]
                #     # denormalize the data
                #     sample_real = denormalize(sample_real, min_real, max_real)
                #     sample_imag = denormalize(sample_imag, min_imag, max_imag)
                #     sample = sample_real + 1j * sample_imag

                # # Make sure that its 2D (64, 100)
                # sample = sample.squeeze().detach().cpu().numpy().reshape(64, 100)
                # decoded = decoded.squeeze().detach().cpu().numpy().reshape(64, 100)

                # # Plot the output
                # plot_fft(np.abs(sample), decoded, model_name=args.model)
                # plot_out(np.abs(sample), decoded, model_name=args.model)
    
    elif type(args.het) == float:
        train_loader, val_loader, test_loader = cd_het(args.dataset_train, args.dataset_test, args.dataset_val, 
                                                               norm='min-max', 
                                                               batch_size=args.batch_size,
                                                               limit=args.limit,
                                                               clus_number=args.class_, het=args.het)
        evaluate_dataset(train_loader)
        if args.model.startswith('CNN'):
            #* Define the model
            model_cnn = CNNAutoencoder(model_name=args.model)

            #* Train the model
            if args.process == 'train':
                model_cnn, train_loss, val_loss = cnn_train(model_cnn, train_loader, val_loader, 
                        criterion=torch.nn.MSELoss(), 
                        optimizer=torch.optim.Adam(model_cnn.parameters(), lr=args.lr), 
                        num_epochs=args.epochs)
                
                # Save the model
                save_encoder_decoder(model_cnn, f'./models/{args.model}')
                # Show the performance
                show_performance(train_loss, val_loss, model_name=args.model)

            #* Test the model
            if args.process == 'test':
                encoder = load_encoder(model_cnn, f'./models/{args.model}/encoder.pth')
                decoder = load_decoder(model_cnn, f'./models/{args.model}/decoder.pth')
                
                encoder.eval()
                decoder.eval()

                # Test model
                cnn_test(encoder=encoder, decoder=decoder,
                        model_name=args.model, 
                        criterion=torch.nn.MSELoss(),
                        test_loader=test_loader,
                        real_max=max_real, real_min=min_real,
                        imag_max=max_imag, imag_min=min_imag)
                sys.exit()
                #* Visualize the output of the model and compare it with the input
                for i, data in enumerate(test_loader):
                    sample = data[0].to(device)
                    break
                
                # Encode and Decode the sample
                encoded = encoder(sample)
                decoded = decoder(encoded)

                # Denormalize the data
                norm = 'min-max'
                if norm == 'min-max':
                    decoded = decoded * (max - min) + min
                    sample = sample * (max - min) + min

                # Make sure that its 2D (64, 100)
                sample = sample.squeeze().detach().cpu().numpy().reshape(64, 100)
                decoded = decoded.squeeze().detach().cpu().numpy().reshape(64, 100)

                # Plot the output
                plot_fft(sample, decoded, model_name=args.model)
                plot_out(sample, decoded, model_name=args.model)
