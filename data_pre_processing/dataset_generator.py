import sys, os, re, time

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

    def create_dataset(self, dataset_path='../DIS_lab_LoS/samples/', limit=10000, dataset_name='dataset.csv'):
        if self.dataset_path == None:
            self.dataset_path = dataset_path
        
        # using numpy set a random seed
        np.random.seed(42)
        
        # using os.listdir get the filenames in the dataset
        # after that shuffle them

        filenames = os.listdir(self.dataset_path)
        np.random.shuffle(filenames)

        # load samples into csv file
        data = []
        dic = {'index': [], 'file': []}
        with open(dataset_name, 'w', newline='') as file:
            for i, filename in enumerate(tqdm(filenames[:limit], desc="Creating Dataset")):
                dic['index'].append(i+1)
                dic['file'].append(os.path.join(self.dataset_path, filename))

                # load the sample
                sample = np.load(os.path.join(self.dataset_path, filename))
                sample = np.abs(sample)
                data.append(sample)
            data = np.array(data)
            dataset_mean = np.mean(data)
            dataset_std = np.std(data)
            dataset_max = np.max(data)
            dataset_min = np.min(data)
        
        # add dataset mean and std to the dictionary
        # and then create a pandas dataframe saving it to a csv file
        dic['mean'] = [dataset_mean] + [''] * (limit-1)
        dic['std'] = [dataset_std] + [''] * (limit-1)
        dic['max'] = [dataset_max] + [''] * (limit-1)
        dic['min'] = [dataset_min] + [''] * (limit-1)
        df = pd.DataFrame(dic)
        df.to_csv(dataset_name, index=False)
        print()

        self.dataset = dataset_name

    def get_statistics(self, plot=False):
        all_filenames = os.listdir(self.dataset_path)
        # shuffle the filenames
        np.random.shuffle(all_filenames)
        filenames = all_filenames[:120000]
        
        # put everything in the dictionary
        data = {
            "l2_norms" : [],
            "magnitudes" : [],
            "phases" : [],
            "real_parts" : [],
            "imag_parts" : []
        }

        for i, filename in enumerate(tqdm(filenames, desc="Preparing Statistics for the whole dataset")):
            path_to_file = os.path.join(self.dataset_path, filename)
            sample = np.load(path_to_file)
            data["l2_norms"].append(np.linalg.norm(sample))
            data["magnitudes"].append(np.abs(sample))
            data["phases"].append(np.angle(sample))
            data["real_parts"].append(np.real(sample))
            data["imag_parts"].append(np.imag(sample))
        print()

        # print statistics, mean, std, min, max
        for key, value in data.items():
            print(f"Statistics for {key}")
            print(f"Mean: {np.mean(value)}")
            print(f"Std: {np.std(value)}")
            print(f"Min: {np.min(value)}")
            print(f"Max: {np.max(value)}")
            print("\n")

        if plot:
            for key, value in data.items():
                # print(f"Plotting Histogram for {key}, value is of shape {np.array(value).shape}")
                plot_histogram(x=value, title=f'{key} Histogram', bins=70)

        self.statistics = data

    def l2_norm_clustering(self, n_clusters=5):
        # load the dataset
        csi_filenames = pd.read_csv(self.dataset)['file']

        l2_norms = []
        for i, filename in enumerate(tqdm(csi_filenames, desc="Calculating L2 Norms")):
            sample = np.load(filename)
            l2_norm = np.linalg.norm(sample).item()
            l2_norms.append(l2_norm)
        l2_norms = np.array(l2_norms)
        print()

        # scale the data
        scaler = StandardScaler()
        l2_norms = scaler.fit_transform(l2_norms.reshape(-1, 1))

        # fit the kmeans model
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        kmeans.fit(l2_norms)

        # get the label for each data point
        labels = kmeans.labels_
        return labels
    
    def get_min_max_RealImag(self, n_clusters=5):
        # load the dataset
        csi_filenames = pd.read_csv(self.dataset)['file']

        l2_norms = []
        for i, filename in enumerate(tqdm(csi_filenames, desc="Calculating L2 Norms")):
            sample = np.load(filename)
            l2_norm = np.linalg.norm(sample).item()
            l2_norms.append(l2_norm)
        l2_norms = np.array(l2_norms)
        print()

        # scale the data
        scaler = StandardScaler()
        l2_norms = scaler.fit_transform(l2_norms.reshape(-1, 1))

        # fit the kmeans model
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        kmeans.fit(l2_norms)

        # get the label for each data point
        labels = kmeans.labels_
        return labels        

    def parafac_clustering(self, n_clusters=5):
        # load the dataset
        csi_filenames = pd.read_csv(self.dataset)['file']

        data = []
        for i, filename in enumerate(tqdm(csi_filenames, desc="Calculating Parafac")):
            sample = np.load(filename)
            data.append(sample)
        parafac = np.array(parafac)
        print()

    def run_clustering(self, method='l2_norms', n_clusters=5):
        df = pd.read_csv(self.dataset)
        if method == 'l2_norms':
            labels = self.l2_norm_clustering(n_clusters=n_clusters)
            df['l2_labels'] = labels
        elif method == 'parafac':
            labels = self.parafac_clustering(n_clusters=n_clusters)
            df['parafac_labels'] = labels

        df.to_csv(self.dataset, index=False)


if __name__ == '__main__':
    #* if you don't have a dataset
    # ds = Dataset()
    # if ds.dataset == None:
    #     ds.create_dataset(dataset_path='./DIS_lab_LoS/samples/', limit=120000, dataset_name='dataset.csv')

    #* if you already have a dataset
    ds = Dataset('dataset.csv', './DIS_lab_LoS/samples/')

    # ds.get_statistics(plot=True)

    ds.run_clustering('l2_norms', n_clusters=3)
