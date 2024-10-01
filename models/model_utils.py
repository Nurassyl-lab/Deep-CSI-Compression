import torch
import os

def save_encoder_decoder(model, save_dir):
    encoder_path = os.path.join(save_dir, 'encoder.pth')
    decoder_path = os.path.join(save_dir, 'decoder.pth')
    
    torch.save(model.encoder.state_dict(), encoder_path)
    torch.save(model.decoder.state_dict(), decoder_path)
    print(f'Encoder and Decoder saved separately in {save_dir}')


def load_encoder(model, encoder_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.encoder.load_state_dict(torch.load(encoder_path, map_location=device))
    model.encoder.to(device)
    model.encoder.eval()
    print(f'Encoder loaded from {encoder_path}')
    return model.encoder


def load_decoder(model, decoder_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.decoder.load_state_dict(torch.load(decoder_path, map_location=device))
    model.decoder.to(device)
    model.decoder.eval()
    print(f'Decoder loaded from {decoder_path}')
    return model.decoder

def load_autoencoder(model, model_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.to(device)
    model.eval()
    print(f'Model loaded from {model_path}')
    return model, model.encoder, model.decoder

# load classifier
def load_classifier(model, model_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.to(device)
    model.eval()
    print(f'Model loaded from {model_path}')
    return model