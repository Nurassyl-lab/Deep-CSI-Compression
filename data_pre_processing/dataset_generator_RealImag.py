import sys, os, re, time, csv

import fcntl
import numpy as np
from tqdm import tqdm
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import tensorly as tl
from tensorly.decomposition import parafac
tl.set_backend('numpy')
# ! IDK why but this code does get executed without a line below
sys.path.append('./')
from utils.data_analysis import plot_histogram, plot_parafac


class Dataset:
    def __init__(self, dataset_name=None, dataset_path=None) -> None:
        self.dataset = dataset_name
        self.dataset_path = dataset_path

    def l2_norm_clustering(self, n_clusters=5):
        # load the dataset
        # csi_filenames = pd.read_csv(self.dataset)['file']
        csi_filenames = os.listdir('DIS_lab_LoS/samples/')

        # l2_norms = []
        # for filename in tqdm(csi_filenames, desc="Calculating L2 Norms"):
        #     sample = np.load(f'DIS_lab_LoS/samples/{filename}')
        #     l2_norm = np.linalg.norm(sample).item()
        #     l2_norms.append(l2_norm)
        #     # garbage collection
        #     del sample

        l2_norms = []
        for filename in tqdm(csi_filenames, desc="Calculating L2 Norms"):
            with open(f'DIS_lab_LoS/samples/{filename}', 'rb') as f:
                # Apply file locking to ensure safe loading
                fcntl.flock(f, fcntl.LOCK_SH)
                try:
                    sample = np.load(f)
                except Exception as e:
                    # File is corrupted
                    print(f"Error: {e}")
                    print(f"File: {filename}")
                fcntl.flock(f, fcntl.LOCK_UN)
                l2_norm = np.linalg.norm(sample).item()
                l2_norms.append(l2_norm)
                # Release the lock



        l2_norms = np.array(l2_norms)
        print()

        # print max and min l2 norms
        print(f"Max L2 Norm: {np.max(l2_norms)}")
        print(f"Min L2 Norm: {np.min(l2_norms)}")

        # calculate how many samples are from
        # CLASS 1: 40-45
        # CLASS 2: 50-55
        # CLASS 3: 60-65
        # using np.where
        class1 = np.where((l2_norms >= 40) & (l2_norms <= 45))[0]
        class2 = np.where((l2_norms >= 50) & (l2_norms <= 55))[0]
        class3 = np.where((l2_norms >= 60) & (l2_norms <= 65))[0]
        
        print(f"Class 1: {len(class1)}")
        print(f"Class 2: {len(class2)}")
        print(f"Class 3: {len(class3)}")

        # Using indices to get the samples of files
        # from the dataset
        class1_files = [csi_filenames[i] for i in class1]
        class2_files = [csi_filenames[i] for i in class2]
        class3_files = [csi_filenames[i] for i in class3]

        # limit to 17000 samples, randomly shuffle them
        class1_files = np.random.choice(class1_files, 17000, replace=False)
        class2_files = np.random.choice(class2_files, 17000, replace=False)
        class3_files = np.random.choice(class3_files, 17000, replace=False)

        # Write above using csv file
        with open('dataset_RealImag.csv', 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['file', 'label'])
            for filename in class1_files:
                writer.writerow([f'./DIS_lab_LoS/samples/{filename}', 1])
            for filename in class2_files:
                writer.writerow([f'./DIS_lab_LoS/samples/{filename}', 2])
            for filename in class3_files:
                writer.writerow([f'./DIS_lab_LoS/samples/{filename}', 3])

        # return labels

    def min_max_finder(self, filename='dataset_RealImag.csv'):
        # using the dataset_RealImag.txt file
        # find the minimum and maximum values
        # for real and imaginary parts of the dataset
        # of the dataset
        
        dataset = pd.read_csv(filename)
        dataset_files = dataset['file']
        min_real = np.inf
        max_real = -np.inf
        min_imag = np.inf
        max_imag = -np.inf        

        # for filename in tqdm(dataset, desc="Finding Min and Max"):
        #     sample = np.load(filename)
        #     real_part = np.real(sample)
        #     imag_part = np.imag(sample)
        #     min_real = min(min_real, np.min(real_part))
        #     max_real = max(max_real, np.max(real_part))
        #     min_imag = min(min_imag, np.min(imag_part))
        #     max_imag = max(max_imag, np.max(imag_part))
        #     del sample, real_part, imag_part

        for filename in tqdm(dataset_files, desc="Finding Min and Max"):
            with open(filename, 'rb') as f:
                # Apply file locking for safe loading of the .npy file
                fcntl.flock(f, fcntl.LOCK_SH)
                sample = np.load(f)
                # Release the lock after loading
                fcntl.flock(f, fcntl.LOCK_UN)

                real_part = np.real(sample)
                imag_part = np.imag(sample)
                min_real = min(min_real, np.min(real_part))
                max_real = max(max_real, np.max(real_part))
                min_imag = min(min_imag, np.min(imag_part))
                max_imag = max(max_imag, np.max(imag_part))

        print('Dataset head before adding min and max')
        print(dataset.head())

        # Add 4 columns to the dataset
        # min_real, max_real, min_imag, max_imag
        dataset['min_real'] = [min_real] + [''] * (len(dataset)-1)
        dataset['max_real'] = [max_real] + [''] * (len(dataset)-1)
        dataset['min_imag'] = [min_imag] + [''] * (len(dataset)-1)
        dataset['max_imag'] = [max_imag] + [''] * (len(dataset)-1)

        print('Dataset head after adding min and max')
        print(dataset.head())

        # save the dataset to a csv file
        dataset.to_csv('dataset_RealImag.csv', index=False)

    # Split the dataset into training and testing
    def split_dataset(self, filename='dataset_RealImag.csv'):
        # load the dataset
        dataset = pd.read_csv(filename)
        # drop min_real, max_real, min_imag, max_imag
        # columns if they exist
        dataset = dataset.drop(columns=['min_real', 'max_real', 'min_imag', 'max_imag'], errors='ignore')
        
        # each class has 17000 samples
        # 15000 for training and 1000 for testing
        # 1000 for validation
        
        class1 = dataset[dataset['label'] == 1]
        class2 = dataset[dataset['label'] == 2]
        class3 = dataset[dataset['label'] == 3]

        # shuffle the dataset
        class1 = class1.sample(frac=1)
        class2 = class2.sample(frac=1)
        class3 = class3.sample(frac=1)

        # split the dataset
        train_class1 = class1[:15000]
        test_class1 = class1[15000:16000]
        val_class1 = class1[16000:]

        train_class2 = class2[:15000]
        test_class2 = class2[15000:16000]
        val_class2 = class2[16000:]

        train_class3 = class3[:15000]
        test_class3 = class3[15000:16000]
        val_class3 = class3[16000:]

        # shuffle train
        train_class1 = train_class1.sample(frac=1)

        # concatenate the datasets
        train = pd.concat([train_class1, train_class2, train_class3])
        test = pd.concat([test_class1, test_class2, test_class3])
        val = pd.concat([val_class1, val_class2, val_class3])

        # save the datasets
        train.to_csv(f'{filename.split(".")[0]}_train.csv', index=False)
        test.to_csv(f'{filename.split(".")[0]}_test.csv', index=False)
        val.to_csv(f'{filename.split(".")[0]}_val.csv', index=False)

if __name__ == '__main__':
    #* if you don't have a dataset
    ds = Dataset()
    ds.l2_norm_clustering()
    # time.sleep(5)
    # ds.min_max_finder(filename='dataset_RealImag.csv')
    # time.sleep(5)
    # ds.split_dataset(filename='dataset_RealImag.csv')
