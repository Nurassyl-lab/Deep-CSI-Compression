'Train torch models'

import torch
import matplotlib.pyplot as plt
import argparse
import pandas as pd
from utils.dataloader import create_dataloaders
from utils.data_analysis import plot_fft, plot_out
from models.model_utils import save_encoder_decoder, load_encoder, load_decoder

# CNN
from models.cnn_autoencoder import CNNAutoencoder
# from models.cnn_autoencoder import save_encoder_decoder, load_encoder, load_decoder
from models.cnn_autoencoder import cnn_train, cnn_test

# VAE
from models.vae_autoencoder import VAE
from models.vae_autoencoder import vae_loss, vae_train, vae_test

# SIREN
from models.siren_autoencoder import SIRENAutoencoder
from models.siren_autoencoder import siren_train, siren_test

def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='dense', help='Model type')
    parser.add_argument('--dataset', type=str, default='dataset.csv', help='Train Dataset path')
    parser.add_argument('--process', type=str, default='train', help='Process type')
    
    #? Where do I use directory?
    # parser.add_argument('--directory', type=str, default='test_model', help='Model name')
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

    # Load the dataset
    train_loader, val_loader, test_loader = create_dataloaders(args.dataset, norm='min-max', batch_size=args.batch_size, limit=args.limit)

    # Get dataset information
    tmp = pd.read_csv("dataset.csv")
    mean = tmp.loc[0, 'mean']
    std = tmp.loc[0, 'std']
    max = tmp.loc[0, 'max']
    min = tmp.loc[0, 'min']

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
                    norm='min-max', 
                    test_loader=test_loader, 
                    criterion=torch.nn.MSELoss(), 
                    mean=mean, std=std, max=max, min=min)
            
            #* Visualize the output of the model and compare it with the input
            for i, data in enumerate(test_loader):
                sample = data[0].to(device)
                break

            # Encode and Decode the sample
            encoded = encoder(sample)
            decoded = decoder(encoded)

            # Denormalize the data
            norm = 'min-max'
            if norm == 'z-score':
                decoded = decoded * std + mean
                sample = sample * std + mean
            elif norm == 'min-max':
                decoded = decoded * (max - min) + min
                sample = sample * (max - min) + min

            # Make sure that its 2D (64, 100)
            sample = sample.squeeze().detach().cpu().numpy().reshape(64, 100)
            decoded = decoded.squeeze().detach().cpu().numpy().reshape(64, 100)

            # Plot the output
            plot_fft(sample, decoded, model_name=args.model)
            plot_out(sample, decoded, model_name=args.model)

    elif args.model.startswith('VAE'):
        #* Define the model
        model_vae = VAE(model_name=args.model)

        #* Train the model
        if args.process == 'train':
            model_vae, train_loss, val_loss, mse_train_loss, mse_val_loss = vae_train(model_vae, train_loader, val_loader, 
                                                                                      criterion=vae_loss, 
                                                                                      optimizer=torch.optim.Adam(model_vae.parameters(), lr=args.lr), 
                                                                                      num_epochs=args.epochs)
            
            # Save the model
            save_encoder_decoder(model_vae, f'./models/{args.model}')
            # Show the performance
            show_performance(mse_train_loss, mse_val_loss, model_name=args.model)

        #* Test the model
        if args.process == 'test':
            encoder = load_encoder(model_vae, f'./models/{args.model}/encoder.pth')
            decoder = load_decoder(model_vae, f'./models/{args.model}/decoder.pth')

            encoder.eval()
            decoder.eval()
            
            # Test model
            vae_test(encoder=encoder, decoder=decoder,
                    model_name=args.model, 
                    norm='min-max', 
                    test_loader=test_loader, 
                    criterion=vae_loss, 
                    mean=mean, std=std, max=max, min=min)

            #* Visualize the output of the model and compare it with the input
            for i, data in enumerate(test_loader):
                sample = data.to(device)
                break

            autoencoder = VAE(model_name=None)
            # Encode and Decode the sample
            mu, logvar = encoder(sample)
            encoded = autoencoder.reparameterize(mu, logvar)
            decoded = decoder(encoded)

            sample = sample[0]
            decoded = decoded[0]

            # Denormalize the data
            norm = 'min-max'
            if norm == 'z-score':
                decoded = decoded * std + mean
                sample = sample * std + mean
            elif norm == 'min-max':
                decoded = decoded * (max - min) + min
                sample = sample * (max - min) + min

            # Make sure that its 2D (64, 100)
            sample = sample.squeeze().detach().cpu().numpy().reshape(64, 100)
            decoded = decoded.squeeze().detach().cpu().numpy().reshape(64, 100)
            
            # Plot to compare
            plot_fft(sample, decoded, model_name=args.model)
            plot_out(sample, decoded, model_name=args.model)

    elif args.model.startswith('SIREN'):
        #* Define the model
        model_siren = SIRENAutoencoder(model_name=args.model)

        #* Train the model
        if args.process == 'train':
            model_siren, train_loss, val_loss = siren_train(model_siren, train_loader, val_loader, 
                    criterion=torch.nn.MSELoss(), 
                    optimizer=torch.optim.Adam(model_siren.parameters(), lr=args.lr), 
                    num_epochs=args.epochs)
            
            # Save the model
            save_encoder_decoder(model_siren, f'./models/{args.model}')
            # Show the performance
            show_performance(train_loss, val_loss, model_name=args.model)

        #* Test the model
        if args.process == 'test':
            encoder = load_encoder(model_siren, f'./models/{args.model}/encoder.pth')
            decoder = load_decoder(model_siren, f'./models/{args.model}/decoder.pth')
            
            encoder.eval()
            decoder.eval()

            # Test model
            siren_test(encoder=encoder, decoder=decoder,
                    model_name=args.model, 
                    norm='min-max', 
                    test_loader=test_loader, 
                    criterion=torch.nn.MSELoss(), 
                    mean=mean, std=std, max=max, min=min)
            
            #* Visualize the output of the model and compare it with the input
            for i, data in enumerate(test_loader):
                sample = data[0].to(device)
                break

            # Encode and Decode the sample
            encoded = encoder(sample)
            decoded = decoder(encoded)

            # Denormalize the data
            norm = 'min-max'
            if norm == 'z-score':
                decoded = decoded * std + mean
                sample = sample * std + mean
            elif norm == 'min-max':
                decoded = decoded * (max - min) + min
                sample = sample * (max - min) + min

            # Make sure that its 2D (64, 100)
            sample = sample.squeeze().detach().cpu().numpy().reshape(64, 100)
            decoded = decoded.squeeze().detach().cpu().numpy().reshape(64, 100)

            # Plot the output
            plot_fft(sample, decoded, model_name=args.model)
            plot_out(sample, decoded, model_name=args.model)
