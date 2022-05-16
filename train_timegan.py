from os import path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from ydata_synthetic.synthesizers import ModelParameters
from ydata_synthetic.preprocessing.timeseries import processed_stock
from ydata_synthetic.synthesizers.timeseries import TimeGAN

from data_hub import LoadData


def train():
    
    
    seq_len=24
    n_seq = 7
    hidden_dim=24
    gamma=1

    noise_dim = 32
    dim = 128
    batch_size = 2048
    dataset_name = 'sports-goal'
    log_step = 100
    learning_rate = 5e-4
    
    training_steps = 20000

    gan_args = ModelParameters(batch_size=batch_size,
                               lr=learning_rate,
                               noise_dim=noise_dim,
                               layers_dim=dim)
    
    train_data, test_data = LoadData(dataset_name, seq_len)
    
    print(len(train_data),train_data[0].shape)
    
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
    tf.config.experimental.set_memory_growth(physical_devices[1], True)
    
    if path.exists('synthesizer_sports_seq_24_goal.pkl'):
        synth = TimeGAN.load('synthesizer_sports_seq_24_goal.pkl')
    else:
        synth = TimeGAN(model_parameters=gan_args, hidden_dim=24, seq_len=seq_len, n_seq=n_seq, gamma=1)
        synth.train(train_data, train_steps=training_steps)
        synth.save('synthesizer_sports_seq_24_goal.pkl')

if __name__ == "__main__":
    
    train()
