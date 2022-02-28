import numpy as np
np.random.seed(2022)

import tensorflow as tf
tf.random.set_seed(2022)

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input,Conv3D,Reshape,Lambda
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint

from scipy import io

from functions import dictionary,Phi_Layer,update_mu_Sigma,circular_padding_2d,A_R_Layer,A_T_Layer

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


A_T = dictionary(Nt, G)
A_T = np.matrix(A_T)
A_R = dictionary(Nr, G)
A_R = np.matrix(A_R)

y_list = np.zeros((data_num, num_sc, Mr * Mt, 1)) + 1j * np.zeros((data_num, num_sc, Mr * Mt, 1))
W_list = np.zeros((data_num, Nr, Mr)) + 1j * np.zeros((data_num, Nr, Mr))
F_list = np.zeros((data_num, Nt, Mt)) + 1j * np.zeros((data_num, Nt, Mt))

for i in range(data_num):
    if i % 1000 == 0:
        print('%d/%d' % (i, data_num))
    # random phase W and F
    random_phases_T = np.random.uniform(0, 2 * np.pi, Nt * Mt)
    random_phases_R = np.random.uniform(0, 2 * np.pi, Nr * Mr)
    F = (np.cos(random_phases_T) + 1j * np.sin(random_phases_T)) / np.sqrt(Nt)
    F = np.reshape(F, (Nt, Mt))
    F = np.matrix(F)
    W = (np.cos(random_phases_R) + 1j * np.sin(random_phases_R)) / np.sqrt(Nr)
    W = np.reshape(W, (Nr, Mr))
    W = np.matrix(W)

    W_list[i] = W
    F_list[i] = F

    Q = np.kron(F.T, W.H)

    # generate the effective noise, and obtain its corresponding covariance matrix
    noise_list = []
    for r in range(Mr // N_r_RF):
        W_r = W[:, r * N_r_RF:(r + 1) * N_r_RF]
        original_noise = np.sqrt(sigma_2 / 2) * (
                    np.random.randn(Nr, Mt * num_sc) + 1j * np.random.randn(Nr, Mt * num_sc))
        noise_list.append(W_r.H.dot(original_noise))
    effective_noise_list = np.concatenate(noise_list, axis=0)
    effective_noise_list = np.reshape(np.array(effective_noise_list), (Mr, Mt, num_sc))

    for n in range(num_sc):
        H_subcarrier = H_list[i, :, :, n, 0] + 1j * H_list[i, :, :, n, 1]  # Nr x Nt
        # notice that, vectorization is to stack column-wise
        H_subcarrier = np.transpose(H_subcarrier)  # Nt x Nr
        h_subcarrier = np.reshape(H_subcarrier, (Nr * Nt, 1))
        # received signal with effective noise added
        effective_noise = np.transpose(effective_noise_list[:, :, n])
        effective_noise = np.reshape(effective_noise, (Mt * Mr, 1))
        y_list[i, n] = Q.dot(h_subcarrier) + effective_noise

y_list = np.concatenate([np.real(y_list), np.imag(y_list)], axis=-1)

W_list = np.expand_dims(W_list, axis=-1)
W_list = np.concatenate([np.real(W_list), np.imag(W_list)], axis=-1)
F_list = np.expand_dims(F_list, axis=-1)
F_list = np.concatenate([np.real(F_list), np.imag(F_list)], axis=-1)

# put the dimension num_sc after to facilitate matrix computation
y_list = np.transpose(y_list, (0, 2, 1, 3))

print(W_list.shape)  # (data_num,Nr,Mr,2)
print(F_list.shape)  # (data_num,Nt,Mt,2)
print(y_list.shape)  # (data_num,Mr*Mt,num_sc,2)
print(H_list.shape)  # (data_num,Nr,Nt,num_sc,2)



#%% construct the network
def SBL_net(Mt, Mr, Nt, Nr, G, num_sc, num_layers, num_filters, kernel_size):
    W_real_imag = Input(shape=(Nr, Mr, 2))
    F_real_imag = Input(shape=(Nt, Mt, 2))
    y_real_imag = Input(shape=(Mt * Mr, num_sc, 2))

    Phi_real_imag = Phi_Layer(Nt, Nr, Mt, Mr, G)([W_real_imag, F_real_imag])

    # of shape (?,G**2,num_sc)
    alpha_list_init = tf.tile(tf.ones_like(W_real_imag[:, 0, 0:1, 0:1]), (1, G ** 2, num_sc))

    # update mu and Sigma
    mu_real, mu_imag, diag_Sigma_real = Lambda(lambda x: update_mu_Sigma(x, num_sc, sigma_2, Mr, Mt))(
        [Phi_real_imag, y_real_imag, alpha_list_init])

    for i in range(num_layers):
        mu_square = Lambda(lambda x: x[0] ** 2 + x[1] ** 2)([mu_real, mu_imag])

        # feature tensor of dim (?,G,G,num_sc,2)
        temp = Lambda(lambda x: tf.concat(x, axis=-1)) \
            ([tf.reshape(mu_square, (-1, G, G, num_sc, 1)), tf.reshape(diag_Sigma_real, (-1, G, G, num_sc, 1))])

        # 3D Convolution, with circular padding, conv padding should use "valid" not "same"
        conv_layer1 = Conv3D(name='SBL_%d1'%i,filters=num_filters,kernel_size=kernel_size,strides=1,padding='valid',activation='relu')
        conv_layer2 = Conv3D(name='SBL_%d2'%i,filters=1,kernel_size=kernel_size,strides=1,padding='valid',activation='relu')

        temp = circular_padding_2d(temp, kernel_size=kernel_size, strides=1)
        temp = conv_layer1(temp)

        temp = circular_padding_2d(temp, kernel_size=kernel_size, strides=1)
        alpha_list = conv_layer2(temp)

        # update mu and Sigma
        mu_real, mu_imag, diag_Sigma_real = Lambda(lambda x: update_mu_Sigma(x, num_sc, sigma_2, Mr, Mt)) \
            ([Phi_real_imag, y_real_imag, tf.reshape(alpha_list, (-1, G ** 2, num_sc))])

    x_hat = Lambda(lambda x: tf.concat([tf.expand_dims(x[0], axis=-1), tf.expand_dims(x[1], axis=-1)], axis=-1))(
        [mu_real, mu_imag])
    X_hat = Reshape((G, G, num_sc, 2))(x_hat)

    # remove the effect of vectorization
    X_hat = Lambda(lambda x: tf.transpose(x, (0, 2, 1, 3, 4)),name='X_hat')(X_hat)

    X_hat = Reshape((G, G * num_sc, 2))(X_hat)
    H_hat = A_R_Layer(Nr, G)(X_hat)
    H_hat = Reshape((Nr, G, num_sc, 2))(H_hat)
    H_hat = Lambda(lambda x: tf.transpose(x, (0, 3, 1, 2, 4)))(H_hat)
    H_hat = Reshape((num_sc * Nr, G, 2))(H_hat)
    H_hat = A_T_Layer(Nt, G)(H_hat)
    H_hat = Reshape((num_sc, Nr, Nt, 2))(H_hat)
    H_hat = Lambda(lambda x: tf.transpose(x, (0, 2, 3, 1, 4)))(H_hat)

    model = Model(inputs=[W_real_imag, F_real_imag, y_real_imag], outputs=H_hat)
    return model

num_layers = 3
num_filters = 8
kernel_size = 5
model = SBL_net(Mt, Mr, Nt, Nr, G, num_sc, num_layers, num_filters, kernel_size)

model.summary()

epochs = int(args.epochs)
batch_size = 16
best_model_path = './models/best_Mr_%d_Mt_%d_SNR_%d_random_init.h5'%(Mr,Mt,SNR)

# weight initialization
Kron2 = np.kron(np.conjugate(A_T),A_R)
Kron2 = np.expand_dims(Kron2,axis=-1)
A_R_0 = np.expand_dims(A_R,axis=-1)
A_T_0 = np.expand_dims(A_T.H,axis=-1)
init_weights_Kron2 = np.concatenate([np.real(Kron2),np.imag(Kron2)],axis=-1)
init_weights_R = np.concatenate([np.real(A_R_0), np.imag(A_R_0)],axis=-1)
init_weights_T = np.concatenate([np.real(A_T_0), np.imag(A_T_0)],axis=-1)
for layer in model.layers:
    if 'phi_' in layer.name:
        print('Set Phi weights')
        layer.set_weights([init_weights_Kron2])
    if 'a_r_' in layer.name:
        print('Set A_R weights')
        layer.set_weights([init_weights_R])
    if 'a_t_' in layer.name:
        print('Set A_T weights')
        layer.set_weights([init_weights_T])

# define callbacks
checkpointer = ModelCheckpoint(best_model_path, verbose=1, save_best_only=True, save_weights_only=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=2, verbose=1, mode='auto',
                              min_delta=1e-5, min_lr=1e-5)
early_stopping = EarlyStopping(monitor='val_loss', min_delta=1e-5, patience=3)

model.compile(loss='mse', optimizer=Adam(lr=1e-4))

model.fit([W_list, F_list, y_list], H_list, epochs=epochs, batch_size=batch_size,
          verbose=1, shuffle=True, \
          validation_split=0.1, callbacks=[checkpointer, reduce_lr, early_stopping])

    
#%% save conv layer weights
model.load_weights(best_model_path)
weight_dict = {}
for layer in model.layers:
    if 'SBL_' in layer.name:
        print('Save Conv weights')
        weight_dict[layer.name] = layer.get_weights()

np.save('./results/weight_dict_Mr_%d_Mt_%d_SNR_%d_random_init.npy'%(Mr,Mt,SNR),weight_dict)
print('Weight dict saved!')


# test performance and save
test_num = int(args.test_num)
predictions_H = model.predict([W_list[-test_num:], F_list[-test_num:], y_list[-test_num:]])
predictions_H = predictions_H[:,:,:,:,0] + 1j*predictions_H[:,:,:,:,0]
true_H = H_list[-test_num:,:,:,:,0] + 1j*H_list[-test_num:,:,:,:,0]

error = 0
error_nmse = 0
for i in range(test_num):
    error = error + np.linalg.norm(predictions_H[i] - true_H[i]) ** 2
    error_nmse = error_nmse + (np.linalg.norm(predictions_H[i] - true_H[i]) / np.linalg.norm(true_H[i])) ** 2
mse = error / (test_num * Nt * Nr * num_sc)
nmse = error_nmse / test_num

print(mse)
print(nmse)

file_handle=open('./results/Performance_Mr_%d_Mt_%d.txt'%(Mr,Mt),mode='a+')
file_handle.write('Random init, SNR=%d dB:\n'%SNR)
file_handle.write(str([mse,nmse]))
file_handle.write('\n')
file_handle.write('\n')
file_handle.close()
