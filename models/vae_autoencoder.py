import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from thop import profile
from torchsummary import summary
from tqdm import tqdm
import copy, os
from sklearn.metrics import mean_squared_error

# Define the Encoder for VAE
class VAE_Encoder(nn.Module):
    def __init__(self, latent_dim=32):
        super(VAE_Encoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1),  # (1, 64, 100) -> (32, 32, 50)
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  # (32, 32, 50) -> (64, 16, 25)
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),  # (64, 16, 25) -> (128, 8, 13)
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=(1, 0))  # (128, 8, 13) -> (256, 4, 6)
        )
        self.flatten = nn.Flatten()
        self.fc_mu = nn.Linear(256 * 4 * 6, latent_dim)
        self.fc_logvar = nn.Linear(256 * 4 * 6, latent_dim)

    def forward(self, x):
        x = self.encoder(x)
        x = self.flatten(x)
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar

# Define the Decoder for VAE
class VAE_Decoder(nn.Module):
    def __init__(self, latent_dim=32):
        super(VAE_Decoder, self).__init__()
        self.fc = nn.Linear(latent_dim, 256 * 4 * 6)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),  # (256, 4, 6) -> (128, 8, 13)
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=(1, 0), output_padding=(1, 0)),  # (128, 8, 13) -> (64, 16, 25)
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),  # (64, 16, 25) -> (32, 32, 50)
            nn.ReLU(),
            nn.ConvTranspose2d(32, 1, kernel_size=3, stride=2, padding=1, output_padding=1),  # (32, 32, 50) -> (1, 64, 100)
            
            #! Here i changed from Tanh to Sigmoid
            nn.Sigmoid()
        )

    def forward(self, z):
        x = self.fc(z)
        x = x.view(-1, 256, 4, 6)
        x = self.decoder(x)
        return x

# Define the VAE Model
class VAE(nn.Module):
    def __init__(self, model_name='vae_autoencoder', latent_dim=32):
        super(VAE, self).__init__()
        self.encoder = VAE_Encoder(latent_dim)
        self.decoder = VAE_Decoder(latent_dim)

        # Model name and directory setup
        if model_name is not None:
            self.model_name = model_name
            self.model_dir = f'models/{self.model_name}'
            if not os.path.exists(self.model_dir):
                os.makedirs(self.model_dir)
        
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        reconstructed_x = self.decoder(z)
        return reconstructed_x, mu, logvar

# Define the VAE Loss
def vae_loss(reconstructed_x, x, mu, logvar):
    #! changed sum to mean for mse_loss
    reconstruction_loss = F.mse_loss(reconstructed_x, x, reduction='mean')
    # KLD: KL divergence between the learned distribution and a unit Gaussian
    kld_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return reconstruction_loss, reconstruction_loss + kld_loss


def vae_train(model, train_loader, val_loader,
              criterion, optimizer, 
              num_epochs=100):
    print('Training VAE model')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    best_val_loss = float('inf')
    best_model_wts = copy.deepcopy(model.state_dict())

    model.to(device)

    train_losses = []
    val_losses = []
    mse_train_losses = []
    mse_val_losses = []
    for epoch in range(num_epochs):
        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
                dataloader = train_loader
            else:
                model.eval()
                dataloader = val_loader

            running_loss = 0.0
            for inputs in tqdm(dataloader, desc=f'{phase.capitalize()} Epoch {epoch+1}'):
                inputs = inputs.to(device)
                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    reconstructed_x, mu, logvar = model(inputs)
                    mse_loss, loss = vae_loss(reconstructed_x, inputs, mu, logvar)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)

            epoch_loss = running_loss / len(dataloader.dataset)
            if phase == 'train':
                train_losses.append(epoch_loss)
                mse_train_losses.append(mse_loss.item())
            else:
                val_losses.append(epoch_loss)
                mse_val_losses.append(mse_loss.item())
            print(f'{phase.capitalize()} Loss: {epoch_loss}')

            if phase == 'val' and epoch_loss < best_val_loss:
                best_val_loss = epoch_loss
                best_model_wts = copy.deepcopy(model.state_dict())
                save_path = os.path.join(model.model_dir, 'best_autoencoder.pth')
                torch.save(best_model_wts, save_path)
                print(f'Saved Best Model with Val Loss: {best_val_loss}')

        print(f'Epoch {epoch+1}/{num_epochs}\n')

    print('Training complete')
    print(f'Best Val Loss: {best_val_loss}')

    # Load the best model weights
    model.load_state_dict(best_model_wts)

    # Save losses
    np.save(os.path.join(model.model_dir, 'train_losses.npy'), np.array(train_losses))
    np.save(os.path.join(model.model_dir, 'val_losses.npy'), np.array(val_losses))
    np.save(os.path.join(model.model_dir, 'mse_train_losses.npy'), np.array(mse_train_losses))
    np.save(os.path.join(model.model_dir, 'mse_val_losses.npy'), np.array(mse_val_losses))

    return model, train_losses, val_losses, mse_train_losses, mse_val_losses


def vae_test(encoder, decoder, model_name, norm, test_loader, mean, std, max, min, criterion):
    print('Testing VAE Model...')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    encoder.to(device)
    decoder.to(device)

    encoder.eval()
    decoder.eval()

    # load model for reparemeterization
    autoencoder = VAE(model_name=None)

    test_losses_norm = []
    test_losses_denorm = []
    for inputs in tqdm(test_loader, desc='Testing'):
        inputs = inputs.to(device)
        with torch.no_grad():
            mu, logvar = encoder(inputs)
            encoded = autoencoder.reparameterize(mu, logvar)
            outputs = decoder(encoded)

            # Use MSE Loss for evaluation
            numpy_inputs = inputs.cpu().numpy().reshape(-1, 6400)
            numpy_outputs = outputs.cpu().numpy().reshape(-1, 6400)
            loss = mean_squared_error(numpy_inputs, numpy_outputs)
            test_losses_norm.append(loss.item())
            
            if norm == 'z-score':
                outputs = outputs * std + mean
                inputs = inputs * std + mean
            elif norm == 'min-max':
                outputs = outputs * (max - min) + min
                inputs = inputs * (max - min) + min
            
            # USE MSE Loss for evaluation
            numpy_inputs = inputs.cpu().numpy().reshape(-1, 6400)
            numpy_outputs = outputs.cpu().numpy().reshape(-1, 6400)
            loss_denorm = mean_squared_error(numpy_inputs, numpy_outputs)
            test_losses_denorm.append(loss_denorm.item())

    # print losses
    print(f'Test Loss (Normalized): {np.mean(test_losses_norm)}')
    print(f'Test Loss (Denormalized): {np.mean(test_losses_denorm)}')
    
    # save losses
    np.save(os.path.join(f'models/{model_name}', 'test_losses_norm.npy'), np.array(test_losses_norm))
    np.save(os.path.join(f'models/{model_name}', 'test_losses_denorm.npy'), np.array(test_losses_denorm))


# Example usage
if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    '''
        - Example Implementation
        - Print out # of parameters and FLOPs using thop
        - Print out the summary of the model using torchsummary
    '''

    # Encoder
    encoder = VAE_Encoder()
    encoder.to(device)
    dummy_input = torch.randn(1, 1, 64, 100).to(device)
    flops_encoder, param_encoder = profile(encoder, inputs=(dummy_input,))

    # Decoder
    decoder = VAE_Decoder()
    decoder.to(device)
    dummy_input = torch.randn(1, 32).to(device)
    flops_decoder, param_decoder = profile(decoder, inputs=(dummy_input,))

    print(f"\n\nEncoder FLOPs: {flops_encoder}")
    print(f"Encoder parameters: {param_encoder}")

    print(f"Decoder FLOPs: {flops_decoder}")
    print(f"Decoder parameters: {param_decoder}")

    # Summary of the model
    model = VAE()
    model.to(device)
    summary(model, (1, 64, 100))
