from utils.dataloader_RealImag import create_het_dataloaders as cd_het

import numpy as np

def normalize(x, x_max, x_min):
    y = 2 * ((x - x_min) / (x_max - x_min)) - 1
    return y

def denormalize(normalized_data, min_value, max_value):
    # Denormalize the data from range [-1, 1] to original range
    original_data = (normalized_data + 1) / 2 * (max_value - min_value) + min_value
    return original_data

def evaluate_dataset(te):
    main_norms = []
    labels = []
    for idx in range(len(te.dataset)):
        data, label = te.dataset[idx]
        labels.append(label)

        # convert data to numpy array
        data = data.numpy()
        # get the real and imaginary parts and combine them into complex numbers
        real = data[0]
        imag = data[1]
        # denormalize the data
        real = denormalize(real, -1.9765625, 2.015625)
        imag = denormalize(imag, -2.0273475, 2.14453125)
        csi = real + 1j * imag
        # get the l2 norm of the csis
        l2_norm = np.linalg.norm(csi)
        main_norms.append(l2_norm)

    labels = np.array(labels)
    main_norms = np.array(main_norms)

    print(f'There are in total label_length:{len(labels)} samples')
    # Get the number of samples for each class
    for class_ in range(3):
        class_indices = np.where(labels == class_+1)[0]
        print(f'Label {class_} has {len(class_indices)} samples')

    # find how many samples are there whose norm is between 40 and 45
    class1 = np.where((main_norms >= 40) & (main_norms <= 45))[0]
    # find how many samples are there whose norm is between 50 and 55
    class2 = np.where((main_norms >= 50) & (main_norms <= 55))[0]
    # find how many samples are there whose norm is between 60 and 65
    class3 = np.where((main_norms >= 60) & (main_norms <= 65))[0]

    print(f"Class 1: {len(class1)}")
    print(f"Class 2: {len(class2)}")
    print(f"Class 3: {len(class3)}")
    
if __name__ == "__main__":
    tr, te, va = cd_het('dataset_RealImag_train.csv', 'dataset_RealImag_test.csv', 'dataset_RealImag_val.csv', 'min-max', 
                        het=1.0)
    