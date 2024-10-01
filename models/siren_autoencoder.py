import os
import torch
import torch.nn as nn
from thop import profile
from torchsummary import summary
import copy
from tqdm import tqdm
import numpy as np

# Custom sine activation function
class SineActivation(nn.Module):
    def __init__(self, omega_0=30):
        super(SineActivation, self).__init__()
        self.omega_0 = omega_0

    def forward(self, input):
        return torch.sin(self.omega_0 * input)

class SIRENConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, omega_0=30, is_first=False):
        super(SIRENConv2d, self).__init__()
        self.is_first = is_first
        self.omega_0 = omega_0
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.init_weights()
        self.activation = SineActivation(omega_0)

    def init_weights(self):
        with torch.no_grad():
            fan_in = self.conv.in_channels * self.conv.kernel_size[0] * self.conv.kernel_size[1]
            if self.is_first:
                bound = 1 / fan_in
            else:
                bound = np.sqrt(6 / fan_in) / self.omega_0
            self.conv.weight.uniform_(-bound, bound)
            if self.conv.bias is not None:
                self.conv.bias.uniform_(-bound, bound)

    def forward(self, x):
        x = self.conv(x)
        x = self.activation(x)
        return x

class SIRENConvTranspose2d(nn.Module):
    def __init__(
        self, in_channels, out_channels, kernel_size, stride=1, padding=0,
        output_padding=0, omega_0=30, is_first=False
    ):
        super(SIRENConvTranspose2d, self).__init__()
        self.is_first = is_first
        self.omega_0 = omega_0
        self.conv_transpose = nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size, stride=stride,
            padding=padding, output_padding=output_padding
        )
        self.init_weights()
        self.activation = SineActivation(omega_0)

    def init_weights(self):
        with torch.no_grad():
            fan_in = self.conv_transpose.in_channels * self.conv_transpose.kernel_size[0] * self.conv_transpose.kernel_size[1]
            if self.is_first:
                bound = 1 / fan_in
            else:
                bound = np.sqrt(6 / fan_in) / self.omega_0
            self.conv_transpose.weight.uniform_(-bound, bound)
            if self.conv_transpose.bias is not None:
                self.conv_transpose.bias.uniform_(-bound, bound)

    def forward(self, x):
        x = self.conv_transpose(x)
        x = self.activation(x)
        return x

class SIREN_Encoder(nn.Module):
    def __init__(self):
        super(SIREN_Encoder, self).__init__()
        self.encoder = nn.Sequential(
            SIRENConv2d(1, 32, kernel_size=3, stride=2, padding=1, is_first=True),  # Input: 1 x 64 x 100 -> 32 x 32 x 50
            SIRENConv2d(32, 64, kernel_size=3, stride=2, padding=1),  # 32 x 32 x 50 -> 64 x 16 x 25
            SIRENConv2d(64, 128, kernel_size=3, stride=2, padding=1),  # 64 x 16 x 25 -> 128 x 8 x 13
            SIRENConv2d(128, 1, kernel_size=3, stride=2, padding=(1, 0)),  # 128 x 8 x 13 -> 128 x 4 x 6
        )

    def forward(self, x):
        return self.encoder(x)

class SIREN_Decoder(nn.Module):
    def __init__(self):
        super(SIREN_Decoder, self).__init__()
        self.decoder = nn.Sequential(
            SIRENConvTranspose2d(1, 128, kernel_size=3, stride=2, padding=1, output_padding=1),  # 128 x 4 x 6 -> 128 x 8 x 12
            SIRENConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=(1, 0), output_padding=(1, 0)),  # 128 x 8 x 12 -> 64 x 16 x 25
            SIRENConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=(1, 1), output_padding=1),  # 64 x 16 x 25 -> 32 x 32 x 50
            SIRENConvTranspose2d(32, 32, kernel_size=3, stride=2, padding=1, output_padding=1),  # 32 x 32 x 50 -> 32 x 64 x 100
            nn.ConvTranspose2d(32, 1, kernel_size=3, stride=1, padding=1),  # 32 x 64 x 100 -> 1 x 64 x 100
            nn.Sigmoid()  # Assuming input normalized between [0, 1]
        )

    def forward(self, x):
        return self.decoder(x)

class SIRENAutoencoder(nn.Module):
    def __init__(self, model_name='siren_autoencoder'):
        super(SIRENAutoencoder, self).__init__()
        self.encoder = SIREN_Encoder()
        self.decoder = SIREN_Decoder()

        # Model name and directory setup
        if model_name is not None:
            self.model_name = model_name
            self.model_dir = f'models/{self.model_name}'
            if not os.path.exists(self.model_dir):
                os.makedirs(self.model_dir)

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


def siren_train(model, train_loader, val_loader, 
                     criterion, optimizer, 
                     num_epochs=100):
    print('Training SIREN Model...')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 
    best_val_loss = float('inf')
    best_model_wts = copy.deepcopy(model.state_dict())

    model.to(device)
    
    train_losses = []
    val_losses = []
    for epoch in range(num_epochs):
        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
                dataloader = train_loader
            else:
                model.eval()   # Set model to evaluate mode
                dataloader = val_loader
            
            running_loss = 0.0
            for inputs in tqdm(dataloader, desc=f'{phase.capitalize()} Epoch {epoch+1}'):
                inputs = inputs.to(device)
                optimizer.zero_grad()
                
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    loss = criterion(outputs, inputs)
                    
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                
                running_loss += loss.item() * inputs.size(0)
            
            epoch_loss = running_loss / len(dataloader.dataset)
 
            print(f'{phase.capitalize()} Loss: {epoch_loss}')
            
            if phase == 'train':
                current_model_wts = copy.deepcopy(model.state_dict())
                save_path = os.path.join(model.model_dir, f'checkpoint_epoch{epoch+1}.pth')
                train_losses.append(epoch_loss)
            else:
                val_losses.append(epoch_loss)

            if phase == 'val' and epoch_loss < best_val_loss:
                best_val_loss = epoch_loss
                best_model_wts = copy.deepcopy(model.state_dict())
                save_path = os.path.join(model.model_dir, 'best_autoencoder.pth')
                torch.save(best_model_wts, save_path)
                print(f'Saved Best Model with Val Loss: {best_val_loss}')
        
        print(f'Epoch {epoch+1}/{num_epochs}\n')
    
    print('Training complete')
    print(f'Best Val Loss: {best_val_loss}')
    
    # Load best model weights
    model.load_state_dict(best_model_wts)

    # save losses
    np.save(os.path.join(model.model_dir, 'train_losses.npy'), np.array(train_losses))
    np.save(os.path.join(model.model_dir, 'val_losses.npy'), np.array(val_losses))
    return model, train_losses, val_losses


def siren_test(encoder, decoder, model_name, norm, test_loader, mean, std, max, min, criterion):
    print('Testing SIREN Model...')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    encoder.to(device)
    decoder.to(device)

    encoder.eval()
    decoder.eval()

    test_losses_norm = []
    test_losses_denorm = []
    for inputs in tqdm(test_loader, desc='Testing'):
        inputs = inputs.to(device)
        with torch.no_grad():
            encoded = encoder(inputs)
            outputs = decoder(encoded)
            loss = criterion(outputs, inputs)
            test_losses_norm.append(loss.item())

            if norm == 'z-score':
                outputs = outputs * std + mean
                inputs = inputs * std + mean
            elif norm == 'min-max':
                outputs = outputs * (max - min) + min
                inputs = inputs * (max - min) + min
            loss_denorm = criterion(outputs, inputs)
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
    encoder = SIREN_Encoder()
    encoder.to(device)
    dummy_input_encoder = torch.randn(1, 1, 64, 100).to(device) # 1 x 64 x 100 is the input shape
    flops_encoder, param_encoder = profile(encoder, inputs=(dummy_input_encoder,))

    # Decoder
    decoder = SIREN_Decoder()
    decoder.to(device)
    dummy_input_decoder = torch.randn(1, 1, 4, 6).to(device) # 128 x 4 x 6 is the input shape
    flops_decoder, param_decoder = profile(decoder, inputs=(dummy_input_decoder,))

    print(f"\n\nEncoder FLOPs: {flops_encoder}")
    print(f"Encoder parameters: {param_encoder}")

    print(f"Decoder FLOPs: {flops_decoder}")
    print(f"Decoder parameters: {param_decoder}")

    # Summary of the model
    model = SIRENAutoencoder(model_name=None)
    model.to(device)
    summary(model, (1, 64, 100))
