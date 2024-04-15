#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Select the GPU index
import scipy.io
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import math
from PIL import Image
from collections import OrderedDict
import torchvision.transforms as T
import matplotlib.pyplot as plt
import time
from torch.optim.lr_scheduler import _LRScheduler
import warnings
import spacy
from scipy.io import savemat
import dill as pickle
import thop
from torch_dataset import Loader
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)


# In[2]:


#Parameters
# onoffdict={'GPS': False, 'CAMERAS': False, 'RADAR': False}
lr=1e-2
num_epochs=100
encoding_features=1024  #keep it in power of 2 
reduction = 12800//encoding_features  #64*100*2
batch_size = 32
weight_path=f'models/'


# In[3]:


#reduction


# In[4]:


#weight_path


# In[5]:


if not os.path.exists(weight_path):
    os.makedirs(weight_path)


# In[6]:


device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'device is {device}')

# # Data Analysis

# In[7]:

train_dataset = Loader(csv_path = 'train_data.csv')
train_loader = DataLoader(train_dataset, batch_size = batch_size, shuffle=True, num_workers=1)
test_dataset = Loader(csv_path = 'validation_data.csv')
test_loader = DataLoader(test_dataset, batch_size = batch_size, shuffle=True, num_workers=1)


# In[8]:


# Get all data in numpy arrays
train_data = []
for batch in train_loader:
    train_data.append(batch.cpu().numpy())
train_data = np.concatenate(train_data, axis=0)


test_data = []
for batch in test_loader:
    test_data.append(batch.cpu().numpy())
test_data = np.concatenate(test_data, axis=0)


# In[9]:


# Calculate mean and standard deviation
train_mean_real = np.mean(np.real(train_data))
train_std_real = np.std(np.real(train_data))
test_mean_real = np.mean(np.real(test_data))
test_std_real = np.std(np.real(test_data))

print(f"Train mean real: {train_mean_real}, Train std real: {train_std_real}")
print(f"Test mean real: {test_mean_real}, Test std real: {test_std_real}")


# In[10]:


# Calculate mean and standard deviation
train_mean_imag = np.mean(np.imag(train_data))
train_std_imag = np.std(np.imag(train_data))
test_mean_imag = np.mean(np.imag(test_data))
test_std_imag = np.std(np.imag(test_data))

print(f"Train mean imag: {train_mean_imag}, Train std imag: {train_std_imag}")
print(f"Test mean imag: {test_mean_imag}, Test std imag: {test_std_imag}")


# In[11]:


print(f"Train max real: {np.max(np.real(train_data))}, Train min real: {np.min(np.real(train_data))}")
print(f"test max real: {np.max(np.real(test_data))}, test min real: {np.min(np.real(test_data))}")


# In[12]:


print(f"Train max imag: {np.max(np.imag(train_data))}, Train min imag: {np.min(np.imag(train_data))}")
print(f"test max imag: {np.max(np.imag(test_data))}, test min imag: {np.min(np.imag(test_data))}")


# In[13]:


#train_data.shape


# In[14]:


# def CSI_reshape( y, csi_std=2.5e-06, target_std=0.015):
#         ry = torch.real(y)
#         iy= torch.imag(y)
#         oy=torch.cat([ry,iy],dim=1)
#         #scaling
#         oy=(oy/csi_std)*target_std+0.5
#         return oy

def CSI_reshape( y, train_mean_real, train_std_real, train_mean_imag, train_std_imag):
        ry = torch.real(y)
        iy= torch.imag(y)
        
        ry = (ry - train_mean_real)/train_std_real
        iy = (iy - train_mean_imag)/train_std_imag
        # oy=torch.cat([ry,iy],dim=1)
        oy = torch.cat([ry.unsqueeze(1), iy.unsqueeze(1)], dim=1)
        return oy

# In[15]:


after_reshape = CSI_reshape( torch.from_numpy(train_data), train_mean_real, train_std_real, train_mean_imag, train_std_imag).numpy() 
# after_reshape = CSI_reshape( torch.from_numpy(train_data)).numpy() 

# In[16]:


# Calculate mean and standard deviation

train_mean_real = np.mean(after_reshape[:,0,:,:])
train_std_real = np.std(after_reshape[:,0,:,:])
test_mean_real = np.mean(after_reshape[:,0,:,:])
test_std_real = np.std(after_reshape[:,0,:,:])

print(f"Train mean real: {train_mean_real}, Train std real: {train_std_real}")
print(f"Test mean real: {test_mean_real}, Test std real: {test_std_real}")
print(f"Train max real: {np.max(after_reshape[:,0,:,:])}, Train min real: {np.min(after_reshape[:,0,:,:])}")
print(f"test max real: {np.max(after_reshape[:,0,:,:])}, test min real: {np.min(after_reshape[:,0,:,:])}")


# In[17]:


# Calculate mean and standard deviation

train_mean_imag = np.mean(after_reshape[:,1,:,:])
train_std_imag = np.std(after_reshape[:,1,:,:])
test_mean_imag = np.mean(after_reshape[:,1,:,:])
test_std_imag = np.std(after_reshape[:,1,:,:])

print(f"Train mean imag: {train_mean_imag}, Train std imag: {train_std_imag}")
print(f"Test mean imag: {test_mean_imag}, Test std imag: {test_std_imag}")
print(f"Train max imag: {np.max(after_reshape[:,1,:,:])}, Train min imag: {np.min(after_reshape[:,1,:,:])}")
print(f"test max imag: {np.max(after_reshape[:,1,:,:])}, test min imag: {np.min(after_reshape[:,1,:,:])}")


# In[ ]:





# # Utils and Models

# In[18]:


#Normalized CSE back to original form
# def CSI_back2original(y, csi_std=2.5e-06, target_std=0.015):
#     y=((y-0.5)*csi_std)/target_std
#     ry=y[:,0,:,:]
#     iy=y[:,1,:,:]
#     original=torch.complex(ry,iy)
#     return original.reshape(-1,1,64,64) 

def CSI_back2original( y, train_mean_real, train_std_real, train_mean_imag, train_std_imag):
    ry=y[:,0,:,:]
    iy=y[:,1,:,:]
    ry = ry * train_std_real + train_mean_real
    ry = ry * train_std_imag + train_mean_imag
    original=torch.complex(ry,iy)
    return original.reshape(-1,64,100) 

# rec = CSI_back2original(after_reshape, train_mean_real, train_std_real, train_mean_imag, train_std_imag)
# In[19]:


#preprocessing output as neural network doesnot understand complex numbers, without normalization
def CSI_complex2real(y):
    ry = torch.real(y)
    iy= torch.imag(y)
    oy=torch.cat([ry,iy],dim=1)
    return oy


# In[20]:


def cal_model_parameters(model):
    total_param  = []
    for p1 in model.parameters():
        total_param.append(int(p1.numel()))
    return sum(total_param)


# In[21]:


# def NMSE_cal(x_hat, x):
#     x_real = torch.reshape(torch.real(x), (x.shape[0], -1))
#     x_imag = torch.reshape(torch.imag(x), (x.shape[0], -1))
#     x_hat_real = torch.reshape(torch.real(x_hat), (x.shape[0], -1))
#     x_hat_imag = torch.reshape(torch.imag(x_hat), (x.shape[0], -1))
#     power = torch.sum(x_real** 2 + x_imag** 2, dim=1)
#     mse = torch.sum((x_real - x_hat_real) ** 2 + (x_imag - x_hat_imag) ** 2, dim=1)
#     nmse = torch.mean(mse / power)
#     return nmse

# def MSE_cal(x_hat, x):
#     x_real = torch.reshape(torch.real(x), (x.shape[0], -1))
#     x_imag = torch.reshape(torch.imag(x), (x.shape[0], -1))
#     x_hat_real = torch.reshape(torch.real(x_hat), (x.shape[0], -1))
#     x_hat_imag = torch.reshape(torch.imag(x_hat), (x.shape[0], -1))
#     mse = torch.mean((x_real - x_hat_real) ** 2 + (x_imag - x_hat_imag) ** 2)
#     return mse
def MSE_cal(x_hat, x):
    x_real = torch.reshape(torch.real(x), (x.shape[0], -1))
    x_imag = torch.reshape(torch.imag(x), (x.shape[0], -1))
    x_hat_real = torch.reshape(torch.real(x_hat), (x.shape[0], -1))
    x_hat_imag = torch.reshape(torch.imag(x_hat), (x.shape[0], -1))
    mse = torch.mean((torch.abs(x_real - x_hat_real)) ** 2 + (torch.abs(x_imag - x_hat_imag)) ** 2)
    return mse

# In[22]:


# fraction to binary
def decimal_to_binary(decimal_matrix, device):
    batch_size,l1 = decimal_matrix.shape
    binary_matrix = torch.zeros((batch_size, l1, 64), dtype=torch.int).to(device)
    for i in range(batch_size):
        for j in range(l1):
            decimal = decimal_matrix[i,j]
            binary = []
            while decimal > 0 and len(binary) < 64:
                decimal *= 2
                if decimal >= 1:
                    binary.append(1)
                    decimal -= 1
                else:
                    binary.append(0)

            while len(binary) < 64:
                binary.append(0)
            binary_matrix[i, j, :] = torch.tensor(binary, dtype=torch.int)
    return binary_matrix
# binary to fraction
def binary_to_decimal(binary_matrix, device):
    batch_size, l0, nb = binary_matrix.shape
    decimal_matrix = torch.zeros((batch_size, l0), dtype=torch.float).to(device)

    for i in range(batch_size):
        for j in range(l0):
            binary = binary_matrix[i, j, :]
            decimal = 0.0
            for k in range(nb):
                decimal += binary[k] * (2 ** (-k-1))

            decimal_matrix[i, j] = decimal

    return decimal_matrix


# In[23]:


class ConvLayer(nn.Sequential):
    def __init__(self, in_planes, out_planes, kernel_size,stride=1, activation="LeakyReLu"):
        padding = (kernel_size - 1) // 2
        dict_activation ={"LeakyReLu":nn.LeakyReLU(negative_slope=0.3,inplace=True),"Sigmoid":nn.Sigmoid(),"Tanh":nn.Tanh()}
        activation_layer = dict_activation[activation]
        super(ConvLayer, self).__init__(OrderedDict([
            ("conv", nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding=padding, bias=False)),
            ("bn", nn.BatchNorm2d(out_planes)),
            ("activation",activation_layer)
        ]))

class RefineNetBlock(nn.Module):
    def __init__(self):
        super(RefineNetBlock, self).__init__()
        self.direct = nn.Sequential(OrderedDict([
            ("conv_7x7", ConvLayer(2, 8, 7, activation="LeakyReLu")),
            ("conv_5x5", ConvLayer(8, 16, 5, activation="LeakyReLu")),
            ("conv_3x3",ConvLayer(16,2,3,activation="Tanh"))
        ]))
        self.identity = nn.Identity()
        self.relu = nn.ReLU()
    def forward(self, x):
        identity = self.identity(x)
        out = self.direct(x)
        out = self.relu(out + identity)
        
        return out


# In[24]:


class task2Encoder(nn.Module):
    
    def __init__(self, reduction=16):
        super(task2Encoder, self).__init__()
        # total_size, in_channel, w, h = 8192, 2, 64, 64
        total_size, in_channel, w, h = 12800, 2, 64, 100
        self.encoder_conv = nn.Sequential(OrderedDict([
            ("conv1_7x7", ConvLayer(2, 2, 7, activation='LeakyReLu')),
            ("conv2_7x7",ConvLayer(2,2,7,activation='LeakyReLu'))
        ]))
        self.encoder_fc = nn.Linear(total_size, total_size // reduction)
        self.output_sig =  nn.Sigmoid()   
    
    def forward(self, x):
        # n,c,h,w = x.detach().size()
        h = x.detach().size()
        x = CSI_reshape(x, train_mean_real, train_std_real, train_mean_imag, train_std_imag) #CSI reshape part is different from original model to convert complex CSI to 2D Normalized real CSI
        out = self.encoder_conv(x.to(torch.float32))
        out = self.encoder_fc(out.view(x.shape[0], -1))
        # have added sigmoid function at the output to keep the values between 0 and 1, 
        # that will be helpul in inference stage's binarization function.
        # only this part in encoder is different from original CSINetplus model.
        # This should not affect the performance in backpropogation and training according to me.
        out = self.output_sig(out)
        return out
       


# In[35]:


class task2Decoder(nn.Module):
    
    def __init__(self, reduction=16):
        super(task2Decoder, self).__init__()
        # total_size, in_channel, w, h = 8192, 2, 64, 64
        total_size, in_channel, w, h = 12800, 2, 64, 100
        self.decoder_fc = nn.Linear(total_size // reduction, total_size)
        self.decoder_conv = ConvLayer(2, 2, 7, activation="Sigmoid")
        self.decoder_refine = nn.Sequential(OrderedDict([
            (f"RefineNet{i+1}",RefineNetBlock()) for i in range(5)
        ]))
        self.decoder_sigmoid = nn.Sigmoid()
        
        
    def forward(self, Hencoded):
        bs = Hencoded.size(0)
        out = self.decoder_fc(Hencoded).view(bs, -1, 64, 100)
        out = self.decoder_conv(out)
        out = self.decoder_refine(out)
        # have added sigmoid function at the output to keep the values between 0 and 1, 
        # this will help in mapping output back to original form
        # only this part in decoder is different from original CSINetplus model.
        out = self.decoder_sigmoid(out)
        output = CSI_back2original(out, train_mean_real, train_std_real, train_mean_imag, train_std_imag)
        return output


# In[36]:


#complete task 2 model including encoder, decoder and channel
class task2model(nn.Module):
    def __init__(self, reduction=16):
        super().__init__()
        
        self.en=task2Encoder(reduction)
        
        self.de=task2Decoder(reduction)
        
    
   
    def forward(self, Hin, device, is_training): 
        
        #Encoder
        Hencoded=self.en(Hin)
        
        if not is_training:
            #convert to 64 bit binary at transmitter
            binary_representation = decimal_to_binary(Hencoded, device)
            
            # At receiver convert back to decimal
            Hreceived = binary_to_decimal(binary_representation, device)
        else:
            Hreceived=Hencoded
        #Decoder   
        Hdecoded=self.de(Hreceived)
        

        return Hdecoded


# In[ ]:





# In[37]:


class WarmUpCosineAnnealingLR(_LRScheduler):
    def __init__(self, optimizer, T_max, T_warmup, eta_min=0, last_epoch=-1):
        self.T_max = T_max
        self.T_warmup = T_warmup
        self.eta_min = eta_min
        super(WarmUpCosineAnnealingLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.T_warmup:
            return [base_lr * self.last_epoch / self.T_warmup for base_lr in self.base_lrs]
        else:
            k = 1 + math.cos(math.pi * (self.last_epoch - self.T_warmup) / (self.T_max - self.T_warmup))
            return [self.eta_min + (base_lr - self.eta_min) * k / 2 for base_lr in self.base_lrs]


# In[38]:


model=task2model(reduction)
print(f'Number of parameters in Task2 Encoder: {cal_model_parameters(model.en)}')
print(f'Number of parameters in Task2 Decoder: {cal_model_parameters(model.de)}')


# In[39]:


# Training


# In[40]:


#Loss

#criterion=nn.BCELoss()
#criterion = nn.CrossEntropyLoss()
criterion= nn.MSELoss().to(device)


# In[41]:


optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.98), eps=1e-9)
#optimizer = torch.optim.Adam(model.parameters(), lr=lr)
#if SGDR == True:
#    sched = CosineWithRestarts(optimizer, T_max=n_batches)

#scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.9, threshold=0.00001, patience=200, verbose=True)


# In[42]:


scheduler = WarmUpCosineAnnealingLR(optimizer=optimizer,
                                            T_max=num_epochs * len(train_loader),
                                            T_warmup=30 * len(train_loader),
                                            eta_min=1e-5)


# In[43]:


model=model.to(device)


# In[44]:


start_time = time.time()
num_train_batches=len(train_loader)
num_test_batches=len(test_loader)
train_losses = []
val_losses = []
train_nmses = []
val_nmses = []
patience_counter = 0
best_val_nmse = float('inf')

for i in range(num_epochs):
    loss1 = 0
    nmse1 = 0
    epoch_time = time.time()
    model.train()
    # Run the training batches
    for b, X_train in enumerate(train_loader):  
        optimizer.zero_grad()
        y_train = torch.clone(X_train)
        # print(f'size is {X_train.shape}')
        y_train=y_train.to(device)
        # Apply the model
        # print(f'{X_train[0].shape}')
        y_pred=model(X_train.to(device), device, is_training=True)
        y_train_reshaped=CSI_reshape(y_train, train_mean_real, train_std_real, train_mean_imag, train_std_imag)
        y_pred_reshaped=CSI_reshape(y_pred, train_mean_real, train_std_real, train_mean_imag, train_std_imag)
        # print(f'loss {y_pred_reshaped.shape} and {y_train_reshaped.shape}')
        loss = criterion(y_pred_reshaped.float(), y_train_reshaped.float()) 
        # Update parameters
        # loss = loss.float()
        loss.backward()
        optimizer.step()
        scheduler.step()
        loss1=loss1+loss
        nmse0 = MSE_cal(y_pred, y_train)
        nmse1+=nmse0 
        #if (b+1)%50==1:
        #    print(f'epoch: {i+1}/{num_epochs} batch:{b+1}/{num_train_batches} loss: {loss.item():10.8f}')      
    train_loss=loss1/num_train_batches  
    train_nmse=nmse1
    train_losses.append(train_loss.item())
    train_nmses.append(train_nmse.item())
    # Update the learning rate scheduler
   
    # Run the testing batches
    model.eval()
    with torch.no_grad():
        loss1=0 
        nmse1=0
        for b, X_test in enumerate(test_loader):
            y_test = torch.clone(X_test)
            y_test=y_test.to(device)
            # Apply the model
            y_pred=model(X_test.to(device), device, is_training=True)
            y_test_reshaped=CSI_reshape(y_test, train_mean_real, train_std_real, train_mean_imag, train_std_imag)
            y_pred_reshaped=CSI_reshape(y_pred, train_mean_real, train_std_real, train_mean_imag, train_std_imag)
            loss = criterion(y_pred_reshaped.float(), y_test_reshaped.float()) 
            # loss = loss.float()
            loss1=loss1+loss   
            nmse0 = MSE_cal(y_pred, y_test)
            nmse1+=nmse0 
        val_loss=loss1/num_test_batches 
        val_nmse=nmse1
        val_losses.append(val_loss.item())
        val_nmses.append(val_nmse.item())
    print(f'epoch:{i+1}/{num_epochs} average reshaped TL:{train_loss.item():10.8f} average reshaped VL:{val_loss.item():10.8f} epoch time:{time.time() - epoch_time:.0f} seconds, lr:{optimizer.param_groups[0]["lr"]:.2e}')               
    print(f' Training MSE:{train_nmse.item():10.8f} Validation MSE:{val_nmse.item():10.8f}')
        # Early stopping
    if val_nmse < best_val_nmse:
        best_val_nmse = val_nmse
        torch.save(model, weight_path+"task2.pth")
        torch.save(model.en, weight_path+"task2Encoder.pth")
        torch.save(model.de, weight_path+"task2Decoder.pth")

print(f'\nDuration: {time.time() - start_time:.0f} seconds') # print the time elapsed            


# In[ ]:





# In[45]:


# Plotting
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()


# In[46]:


flops, params = thop.profile(model, inputs=(X_test[0].to(device), device, True, ), verbose=False)
print(f'Model Flops: {flops}')
print(f'Model Params Num: {params}\n')


# In[ ]:





# In[47]:


np.save(weight_path+'train_loss.npy', train_losses)
np.save(weight_path+'val_loss.npy', val_losses)


# # Inference

# In[49]:


#Create an instance of your model
#model = task2model()

model2=torch.load(weight_path+"task2.pth").to(device)
# Run the testing batches
model2.eval()
with torch.no_grad():
    mse1=0
    nmse1=0
    for b, (X_test, y_test) in enumerate(test_loader):

        y_test=y_test.to(device)
        # Apply the model
        y_pred=model2(X_test[0].to(device),device, is_training=False)
        #y_test_reshaped=CSI_reshape(y_test.to(device))
        y_test_reshaped=CSI_complex2real(y_test)
        y_pred_reshaped=CSI_complex2real(y_pred)
        mse0 = criterion(y_pred_reshaped, y_test_reshaped) 
        mse1+= mse0 
        nmse0 = MSE_cal(y_pred, y_test)
        nmse1+=nmse0 
    avg_mse=mse1/num_test_batches
    avg_nmse=10 * torch.log10(nmse1/num_test_batches)


# In[50]:


print(f'Average MSE:{avg_mse}')
print(f'NMSE:{avg_nmse}')


# In[51]:


dict={'train_losses':train_losses, 'val_losses': val_losses, 'train_nmses':train_nmses, 'val_nmses': val_nmses, 'test_mse': avg_mse.item(), 'test_nmse': avg_nmse.item() }
savemat(weight_path+'losses.mat', dict)
with open(weight_path+'losses.pkl', 'wb') as file:
    
    # A new file will be created
    pickle.dump(dict, file)


# In[ ]:



