# using torch create a autoencoder model
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from torchsummary import summary
from sklearn.preprocessing import StandardScaler
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import os
from thop import profile

#! This bitch right here gets rewritten and has to be deleted
#TODO: DELETE

# CNN NETWORK===========================================================================================================
class CNNAutoencoder(nn.Module):
    def __init__(self, model_name='cnn_autoencoder'):
        super(CNNAutoencoder, self).__init__()
        
        # Encoder: using convolutional layers
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1), # 1, 64, 100 -> 32, 32, 50
            nn.ReLU(False),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),# 32, 32, 50 -> 64, 16, 25 
            nn.ReLU(False),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1), # 64, 16, 25 -> 128, 8, 13
            nn.ReLU(False),
            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=(1,0)),  # encoding_dim defines the compressed representation
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 128, kernel_size=3, stride=2, padding=1, output_padding=1),  # 4, 7  -> 8, 14
            nn.ReLU(False),
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=(1,0), output_padding=(1,0)),  # 8, 14 -> 16, 28

            nn.ReLU(False),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=(1,1), output_padding=1),  # 16, 28 -> 31, 55
            nn.ReLU(False),
            nn.ConvTranspose2d(32, 32, kernel_size=3, stride=1, padding=1),  # 16 -> 15 (here it reduces)
            nn.ReLU(False),
            # nn.ConvTranspose2d(32, 32, kernel_size=2, stride=1, padding=1),  # 15 -> 14 (here it doesn't reduce)
            nn.ReLU(False),
            nn.ConvTranspose2d(32, 1, kernel_size=3, stride=2, padding=1, output_padding=1),  # 8 -> 16
            nn.Tanh()  # Assuming the input image was normalized between [-1, 1]
        )

        # Model name and directory setup
        self.model_name = model_name
        if not os.path.exists(f'models/{self.model_name}'):
            os.makedirs(f'models/{self.model_name}')

    def info(self, input_shape=(1, 28, 28), encoding_dim=32):
        print(f'Input shape: {input_shape}')
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        dummy_input_encoder = torch.randn(1, *input_shape).to(device)
        dummy_input_decoder = torch.randn(1, encoding_dim, 2, 2).to(device)  # Adjusted for the decoder input

        print('Info about CNNAutoencoder')

        # Display number of parameters
        print("Encoder")
        summary(self.encoder.to(device), input_shape)  # Input shape for encoder
        flops_encoder, _ = profile(self.encoder, inputs=(dummy_input_encoder,))
        print(f"\nEncoder FLOPs: {flops_encoder}\n")

        print("Decoder")
        summary(self.decoder.to(device), (encoding_dim, 4, 6))  # Input shape for decoder
        flops_decoder, _ = profile(self.decoder, inputs=(dummy_input_decoder,))
        print(f"Decoder FLOPs: {flops_decoder}\n")

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
    
    def encode(self, x):
        return self.encoder(x)
    
    def decode(self, x):
        return self.decoder(x)

    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        self.load_state_dict(torch.load(path))
        self.eval()

    def train_model(self, train_dataloader, val_dataloader=None, epochs=10, lr=1e-3):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f'Training the model on {device}')
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        self.to(device)
        
        best_val_loss = float('inf')  # Initialize best validation loss to infinity
        train_loss = []
        val_loss = []

        for epoch in range(epochs):
            # Training phase
            self.train()
            total_loss = 0
            for data in tqdm(train_dataloader, desc=f'Training Epoch {epoch+1}'):
                data = data.to(device)
                optimizer.zero_grad()
                encoded = self.encode(data)
                decoded = self.decode(encoded)
                loss = criterion(decoded, data)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            
            avg_train_loss = total_loss / len(train_dataloader)            
            print(f'Train Epoch [{epoch+1}/{epochs}], Loss: {avg_train_loss}')
            train_loss.append(avg_train_loss)
            
            # Validation phase
            if val_dataloader is not None:
                self.eval()
                total_val_loss = 0
                with torch.no_grad():
                    for data in tqdm(val_dataloader, desc=f'Validating Epoch {epoch+1}'):
                        data = data.to(device)
                        encoded = self.encode(data)
                        decoded = self.decode(encoded)
                        loss = criterion(decoded, data)
                        total_val_loss += loss.item()

                avg_val_loss = total_val_loss / len(val_dataloader)
                print(f'Validation Loss: {avg_val_loss}')
                val_loss.append(avg_val_loss)

                # Save model if validation loss improved
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss

                    # save encoder and decoder separately
                    self.save(os.path.join(f'models/{self.model_name}', 'best_encoder.pth'))
                    self.save(os.path.join(f'models/{self.model_name}', 'best_decoder.pth'))
                    print(f'Models save as best_encoder.pth and best_decoder.pth, in models/{self.model_name}')
            print()
        print()
        return train_loss, val_loss

    def evaluate_model(self, dataloader):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # load encoder and decoder

        self.encoder.load(os.path.join(f'models/{self.model_name}', 'best_encoder.pth'))
        self.decoder.load(os.path.join(f'models/{self.model_name}', 'best_decoder.pth'))

        print(f'Evaluating the model on {device}')
        criterion = nn.MSELoss()
        total_loss = 0
        self.to(device)
        self.eval()
        with torch.no_grad():
            for data in tqdm(dataloader, desc='Evaluating'):
                data = data.to(device)
                encoded = self.encoder(data)
                decoded = self.decoder(encoded)
                loss = criterion(decoded, data)
                total_loss += loss.item()
        avg_loss = total_loss / len(dataloader)
        print(f'Loss: {avg_loss}\n') 


# Test the models
if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Show info about the models
    input_dim = 784
    encoding_dim = 32

    # Dense Autoencoder
    # model_dense = DenseAutoencoder(input_dim, encoding_dim)
    # model_dense.info(input_dim)

    # CNN Autoencoder
    input_dim = (1, 64, 100)
    model_cnn = CNNAutoencoder(encoding_dim)
    model_cnn.info(input_shape=input_dim)
