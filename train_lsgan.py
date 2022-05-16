import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd.variable import Variable

from data_hub import LoadData
from model import *


import time
import os
import json
import pathlib
from tqdm import tqdm

from sklearn.decomposition import PCA

import seaborn as sb

from torch.utils.tensorboard import SummaryWriter

import warnings
warnings.filterwarnings('ignore')



    
def visualize(ori_data, fake_data, file_name, save_path, seq_len, epoch, mode, writer):
    
    ori_data = np.asarray(ori_data)

    fake_data = np.asarray(fake_data)
    
    ori_data = ori_data[:fake_data.shape[0]]
    
    sample_size = 250
    
    idx = np.random.permutation(len(ori_data))[:sample_size]
    
    randn_num = np.random.permutation(sample_size)[:1]
    
    real_sample = ori_data[idx]

    fake_sample = fake_data[idx]
    
    if mode == 'timegan-extended':
        
        real_sample_2d = real_sample.reshape(-1, seq_len)

        fake_sample_2d = fake_sample.reshape(-1, seq_len)
        
        
    ### PCA
    
    pca = PCA(n_components=2)
    pca.fit(real_sample_2d)
    pca_real = (pd.DataFrame(pca.transform(real_sample_2d))
                .assign(Data='Real'))
    pca_synthetic = (pd.DataFrame(pca.transform(fake_sample_2d))
                     .assign(Data='Synthetic'))
    pca_result = pca_real.append(pca_synthetic).rename(
        columns={0: '1st Component', 1: '2nd Component'})
    
    
    fig, axs = plt.subplots(ncols = 1, nrows=1, figsize=(8, 5))

    sb.scatterplot(x='1st Component', y='2nd Component', data=pca_result,
                    hue='Data', style='Data', ax=axs)
    sb.despine()
    
    axs.set_title('PCA Result')
    
    plt.savefig(os.path.join(f'{save_path}', f'{time.time()}-pca-result-{file_name}-seq-{seq_len}-{mode}-{epoch}.png'))
     
    writer.add_figure(mode, fig, epoch)
    
def train():
    
    
    logs = {
    
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'n_features' : 7,
        'z_dim' : 64,
        'hidden_shape': 192,
        'epoch' : 20000,
        'batch_size' : 2048,
        'seq_len' : 24,
        'num_layers': 8,
        'lr_rate': 0.00003,
        'dataset_name' : 'sports-goal',
        'loss_mode': 'lsgan',
        'architecture': 'GRU'
    }
    
    
    device = logs['device']
    
    n_features = logs['n_features']
    
    z_dim = logs['z_dim']
    
    hidden_shape = logs['hidden_shape']
    
    epoch = logs['epoch']
    
    batch_size = logs['batch_size']
    
    seq_len = logs['seq_len']
    
    num_layers = logs['num_layers']
    
    lr_rate = logs['lr_rate']
    
    dataset_name = logs['dataset_name']
    
    loss_mode = logs['loss_mode']
    
    architecture = logs['architecture']
    
    file_name = f'{architecture}-{dataset_name}-{loss_mode}'
    
    
    Tensor = torch.cuda.FloatTensor if device == 'cuda' else torch.FloatTensor


    folder_name = f'saved_files/{time.time():.4f}-{file_name}'
    
    
    pathlib.Path(folder_name).mkdir(parents=True, exist_ok=True) 
    
    gan_fig_dir_path = f'{folder_name}/output/gan'
    
    
    pathlib.Path(gan_fig_dir_path).mkdir(parents=True, exist_ok=True) 
    
    ####
    
    with open(f'{folder_name}/params.json', 'w') as f:
        
        json.dump(logs, f)
        
        f.close()
        
    train_data, test_data = LoadData(dataset_name, seq_len)
    
    data_loader = torch.utils.data.DataLoader(train_data, batch_size = batch_size)
    
    data_loader_test = torch.utils.data.DataLoader(test_data, batch_size = len(test_data))
    
    data_batch = next(iter(data_loader_test))
    
    print(data_batch.shape)
    
    writer = SummaryWriter(log_dir = folder_name, comment = f'{file_name}', flush_secs = 45)
    
    ## models
    
    if architecture == 'GRU' or architecture == 'LSTM':
    
        gen = CommonGRU(z_dim, hidden_shape, n_features, num_layers, architecture, activation_fn = nn.Sigmoid()).to(device)

        disc = CommonGRU(n_features, hidden_shape, 1, num_layers, architecture, activation_fn = nn.Sigmoid()).to(device)
        

    
    
    gen_optim = torch.optim.AdamW(gen.parameters(), lr = lr_rate)

    disc_optim = torch.optim.AdamW(disc.parameters(), lr = lr_rate)
    
    
    ###
    
    sch_gen = torch.optim.lr_scheduler.LambdaLR(

    gen_optim, lambda step: 1 - step / epoch

    )

    sch_disc = torch.optim.lr_scheduler.LambdaLR(

        disc_optim, lambda step: 1 - step / epoch

    )
    
    


    
    gan_criterion = nn.MSELoss()

    
    start_time = time.time()
    
    
    for running_epoch in tqdm(range(epoch)):

        for i, data in enumerate(data_loader):
        

            data = data.to(device)

            batch_size = data.size(0)

            real_label = Variable(Tensor(data.size(0), data.size(1), data.size(2)).fill_(1.0), requires_grad=False)

            fake_label = Variable(Tensor(data.size(0), data.size(1), data.size(2)).fill_(0.0), requires_grad=False)

            ### Generator

            gen_optim.zero_grad()

            z = torch.randn(batch_size, seq_len, z_dim, dtype = torch.float, device = device)

            fake_data = gen(z)
            
            gan_loss = gan_criterion(disc(fake_data), real_label)

            g_loss = 0.5 * gan_loss

            g_loss.backward()

            gen_optim.step()

            ### Discriminator

            disc_optim.zero_grad()
            
            real_data_loss = gan_criterion(disc(data), real_label)

            fake_data_loss = gan_criterion(disc(fake_data.detach()), fake_label)
    
            d_loss = (0.5 * (real_data_loss + fake_data_loss))

            d_loss.backward()

            disc_optim.step()

            if i%len(data_loader)==0:

                print(f'Epoch : [{running_epoch+1}/{epoch}], d_loss: {d_loss.item()}, g_loss: {g_loss.item()}')

                writer.add_scalar('Discriminator_Loss', d_loss.item(), running_epoch)

                writer.add_scalar('Generator_Loss', g_loss.item(), running_epoch)
            
            if i%len(data_loader)==0 and running_epoch%500==0:

                with torch.no_grad():

                    fixed_noise = torch.randn(len(test_data), seq_len, z_dim, dtype = torch.float, device = device)

                    out_gen = gen(fixed_noise).detach().cpu()

                visualize(data_batch.detach().cpu(), out_gen, file_name, gan_fig_dir_path, seq_len, running_epoch, 'timegan-extended', writer)

                print(running_epoch)
                
                torch.save({

                    'epoch': running_epoch+1,
                    'gen_state_dict_gan': gen.state_dict(),
                    'disc_state_dict_gan': disc.state_dict(),
                    'gen_optim_state_dict': gen_optim.state_dict(),
                    'disc_optim_state_dict': disc_optim.state_dict(),
                    'gen_schedular_state_dict': sch_gen.state_dict(),
                    'disc_schedular_state_dict': sch_disc.state_dict()

                    }, os.path.join(f'{folder_name}', f'{file_name}-ep-{running_epoch+1}.pth'))
                
            
    
    end_time = time.time()
    
    elapsed_time = (end_time - start_time)/60.
    
    print(f'Total time: {elapsed_time:.4f} min')
    
    
    torch.save({

        'epoch': running_epoch+1,
        'gen_state_dict_gan': gen.state_dict(),
        'disc_state_dict_gan': disc.state_dict(),
        'gen_optim_state_dict': gen_optim.state_dict(),
        'disc_optim_state_dict': disc_optim.state_dict(),
        'gen_schedular_state_dict': sch_gen.state_dict(),
        'disc_schedular_state_dict': sch_disc.state_dict()

        }, os.path.join(f'{folder_name}', f'{file_name}-ep-{epoch}-final.pth'))
    

    
    
    print('Weights Saved!!')
    
    print('training done!')
    
    
if __name__ == "__main__":
    
    train()
    
