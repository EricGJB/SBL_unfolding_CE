import numpy as np
np.random.seed(2022)

import tensorflow as tf
tf.random.set_seed(2022)

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input,Conv2D,Reshape,BatchNormalization,LeakyReLU,Dense,Add
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint

from scipy import io

from functions import Phi_Layer_joint_opt_single_sc

import argparse


#%% Load channel
parser = argparse.ArgumentParser()

parser.add_argument('-Mr')
parser.add_argument('-Mt')
parser.add_argument('-SNR')
parser.add_argument('-data_num')
parser.add_argument('-test_num')
parser.add_argument('-epochs')
parser.add_argument('-gpu_index')

args = parser.parse_args()

import os
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_index
gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)


Mr = int(args.Mr) # number of receive beams at the BS
Mt = int(args.Mt) # number of transmit beams at the user

SNR = int(args.SNR) # SNR
sigma_2 = 1/10**(SNR/10) # noise variance

resolution = 2 # resolution of angle grids
Nt = 32
Nr = 32
G = resolution*np.max([Nt,Nr])

data_num = int(args.data_num)
H_list = io.loadmat('./data/channel.mat')['H_list'][-data_num:]

# fixed parameters
N_r_RF = 4 # number of receive RF chains at the BS

num_sc = 8 # number of subcarriers

noise_list = np.sqrt(sigma_2/2)*\
    (np.random.randn(data_num*num_sc,Mr//N_r_RF,Nr,Mt)\
     +1j*np.random.randn(data_num*num_sc,Mr//N_r_RF,Nr,Mt))
noise_list = np.expand_dims(noise_list, axis=-1)
noise_list = np.concatenate([np.real(noise_list),np.imag(noise_list)], axis=-1)

H_list = np.transpose(H_list,(0,3,1,2,4))
H_list = np.reshape(H_list,(data_num*num_sc,Nr,Nt,2))

print(H_list.shape)
print(noise_list.shape)


#%% construct the network
def SBL_net(Mt, Mr, Nt, Nr, G, N_r_RF, num_layers):
    H_real_imag = Input(shape=(Nr, Nt, 2))
    noise_real_imag = Input(shape=(Mr//N_r_RF, Nr, Mt, 2))

    y_real_imag = Phi_Layer_joint_opt_single_sc(Nt, Nr, Mt, Mr, G, N_r_RF)([H_real_imag,noise_real_imag])

    y = Reshape((Mt*Mr*2,))(y_real_imag)
    coarse_est = Dense(Nt*Nr*2)(y)
    coarse_est = Reshape((Nr,Nt,2))(coarse_est)
    for i in range(num_layers):
        if i==0:
            temp = Conv2D(filters=2**(i+1), kernel_size=3, padding='same')(coarse_est)
        else:
            temp = Conv2D(filters=2**(i+1), kernel_size=3, padding='same')(temp)
        temp = BatchNormalization()(temp)
        temp = LeakyReLU()(temp)

    temp = Conv2D(filters=2, kernel_size=3, padding='same')(temp)
    temp = BatchNormalization()(temp)

    temp = Add()([temp,coarse_est])
    H_hat = LeakyReLU()(temp)

    model = Model(inputs=[H_real_imag, noise_real_imag], outputs=H_hat)
    return model


## only train the matrix with Conv weights frozen
num_layers = 8

model = SBL_net(Mt, Mr, Nt, Nr, G, N_r_RF, num_layers)

model.summary()

epochs = int(args.epochs)
batch_size = 128
best_model_path = './models/best_Mr_%d_Mt_%d_SNR_%d_data_driven_joint_opt.h5'%(Mr,Mt,SNR)

# define callbacks
checkpointer = ModelCheckpoint(best_model_path, verbose=1, save_best_only=True, save_weights_only=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=2, verbose=1, mode='auto',
                              min_delta=1e-5, min_lr=1e-5)
early_stopping = EarlyStopping(monitor='val_loss', min_delta=1e-5, patience=3)

model.compile(loss='mse', optimizer=Adam(lr=1e-4))

model.fit([H_list,noise_list], H_list, epochs=epochs, batch_size=batch_size,
          verbose=1, shuffle=True, \
          validation_split=0.1, callbacks=[checkpointer, reduce_lr, early_stopping])

#  test performance and save weight dict
model.load_weights(best_model_path)

test_num = int(args.test_num)
predictions_H = model.predict([H_list[-test_num:], noise_list[-test_num:]])
predictions_H = predictions_H[:,:,:,0] + 1j*predictions_H[:,:,:,0]
true_H = H_list[-test_num:,:,:,0] + 1j*H_list[-test_num:,:,:,0]

error = 0
error_nmse = 0
for i in range(test_num):
    error = error + np.linalg.norm(predictions_H[i] - true_H[i]) ** 2
    error_nmse = error_nmse + (np.linalg.norm(predictions_H[i] - true_H[i]) / np.linalg.norm(true_H[i])) ** 2
mse = error / (test_num * Nt * Nr)
nmse = error_nmse / test_num

print(mse)
print(nmse)

file_handle=open('./results/Performance_Mr_%d_Mt_%d.txt'%(Mr,Mt),mode='a+')
file_handle.write('Joint opt, data driven, SNR=%d dB:\n'%SNR)
file_handle.write(str([mse,nmse]))
file_handle.write('\n')
file_handle.close()


