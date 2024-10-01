import torch
from thop import profile
import sys, os
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import mean_squared_error
import numpy as np
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# SIREN
from models.siren_autoencoder import SIREN_Encoder, SIREN_Decoder, SIRENAutoencoder

# CNN
from models.cnn_autoencoder import CNN_Encoder, CNN_Decoder, CNNAutoencoder

from models.model_utils import save_encoder_decoder, load_encoder, load_decoder, load_autoencoder
from utils.dataloader import create_dataloaders
import argparse

def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='SIREN', help='Model type')
    parser.add_argument('--datasets', type=str, default='SNR_biased_10_1_dataset.csv, SNR_biased_10_2_dataset.csv, SNR_biased_10_3_dataset.csv', help='Train Dataset path')
    parser.add_argument('--het', type=int, default=None, help='Heterogeneity')
    parser.add_argument('--epoch', type=int, default=None, help='Current epoch')
    return parser.parse_args()

if __name__ == '__main__':
    args = arg_parser()
    n_class = 3

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # load datasets, split using ', ' or ','
    datasets_names = args.datasets.split(', ')

    # Load the dataset and combine them, since then have the same columns
    # df1 = pd.read_csv(datasets_names[0])[:10000]
    # df2 = pd.read_csv(datasets_names[1])[:10000]
    # df3 = pd.read_csv(datasets_names[2])[:10000]
    # merged_df = pd.concat([df1, df2, df3], ignore_index=True)
    
    # # Save the merged dataset
    # merged_df.to_csv('SNR_merged_dataset.csv', index=False)


    # Garbage collection
    # del df1, df2, df3

    #* Heterogeneity

    if args.model == 'CNN' or args.model == 'SIREN':
        if args.model == 'SIREN':
            autoencoder = SIRENAutoencoder(model_name=args.model)
        elif args.model == 'CNN':
            autoencoder = CNNAutoencoder(model_name=args.model)

        dic = {}

        # Load models into dictionary
        for class_ in range(n_class):
            model_path = f'models/{args.model}_het{args.het}_class{class_+1}/checkpoint_epoch{args.epoch}.pth'
            loaded_autoencoder, loaded_encoder, loaded_decoder = load_autoencoder(autoencoder, model_path)
            # dic[f'autoencoder{class_}'] = loading_autoencoder
            dic[f'encoder{class_}'] = loaded_encoder
            dic[f'decoder{class_}'] = loaded_decoder

        # Encode
        dic_encoded = {'class1_encoded':[], 'class2_encoded':[], 'class3_encoded':[]}
        sn_dataset = [[], []] # data, label
        for class_data in range(n_class):
            train_dataloader_for_test, _, _ = create_dataloaders(datasets_names[class_data], norm='min-max', batch_size=1)
    
            stacked_encoded = [[], [], []] # class1, class2, class3
            for class_model in range(n_class):
                mse = []
                for inputs in tqdm(train_dataloader_for_test, desc='Testing'):
                    inputs = inputs.to(device)
                    with torch.no_grad():
                        encoded = dic[f'encoder{class_model}'](inputs) # shape is (1, 128, 4, 6)

                        # since the inner loop runs for 3 times
                        # stacked_encoded shape should be (32, 128*3*4*6)
                        #! Maybe not 128 but 1
                        stacked_encoded[class_model].append(encoded.cpu().numpy().reshape(1, 1*4*6)) # for dataset creation

                        outputs = dic[f'decoder{class_model}'](encoded)
                        outputs, inputs = outputs.cpu().numpy().reshape(-1,6400), inputs.cpu().numpy().reshape(-1,6400)
                        mse.append(mean_squared_error(outputs, inputs))

                print(f'MSE for class {class_model+1} on dataset {class_data+1}: {sum(mse)/len(mse)}, Heterogeneity: {args.het}, Epoch: {args.epoch}')
            stacked_encoded = np.concatenate(stacked_encoded, axis=1)
            print(f'Shape of stacked_encoded: {stacked_encoded.shape}')
            
            # outer loop runs for 3 times
            # sn_dataset shape should be (32*3, 128*3*4*6)
            sn_dataset[0].append(stacked_encoded)
            sn_dataset[1].append([class_data]*len(stacked_encoded))

            del train_dataloader_for_test

        # save the dataset
        sn_data = np.array(sn_dataset[0]).reshape(-1, 3, 1*4*6)
        sn_label = np.array(sn_dataset[1]).reshape(-1, 1)

        np.save(f'current_sn_data.npy', sn_data)
        np.save(f'current_sn_label.npy', sn_label)
        print('Saved SN as current_sn_data.npy and current_sn_label.npy')

        # Garbage collection
        del loaded_decoder, loaded_encoder, loaded_autoencoder, dic
