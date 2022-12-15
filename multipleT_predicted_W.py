import numpy as np
np.random.seed(2022)

import tensorflow as tf
tf.random.set_seed(2022)

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input,Conv3D,Reshape,Lambda,GlobalAvgPool2D,Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint

from scipy import io

from functions import dictionary,Phi_Layer_multipleT,update_mu_Sigma,circular_padding_2d,A_R_Layer,A_T_Layer

import math

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
H_list = io.loadmat('./data/channel_current.mat')['H_list'][-data_num:]

# fixed parameters
N_r_RF = 4 # number of receive RF chains at the BS

num_sc = 8 # number of subcarriers

A_T = dictionary(Nt, G)
A_T = np.matrix(A_T)
A_R = dictionary(Nr, G)
A_R = np.matrix(A_R)

noise_list = np.sqrt(sigma_2/2)*\
    (np.random.randn(data_num,Mr//N_r_RF,Nr,Mt*num_sc)\
     +1j*np.random.randn(data_num,Mr//N_r_RF,Nr,Mt*num_sc))
noise_list = np.expand_dims(noise_list, axis=-1)
noise_list = np.concatenate([np.real(noise_list),np.imag(noise_list)], axis=-1)

print(H_list.shape)
print(noise_list.shape)


#%% construct the network
def SBL_net(Mt, Mr, Nt, Nr, G, N_r_RF, num_sc, num_layers, num_filters, kernel_size):
    H_real_imag = Input(shape=(Nr, Nt, num_sc, 2))
    noise_real_imag = Input(shape=(Mr//N_r_RF, Nr, Mt * num_sc, 2))
    X_abs_previous_block = Input(shape=(G, G, num_sc))

    # predict W with previous block's estimation results
    temp = GlobalAvgPool2D(keepdims=False, data_format='channels_first')(X_abs_previous_block)
    temp = Dense(Nr*Mr//2,activation='relu')(temp)

    # temp = Dense(Nr * Mr)(temp)

    temp = Dense(Nr * Mr,activation='sigmoid')(temp)
    temp = Lambda(lambda x:x*2*math.pi)(temp)

    temp = Reshape((Nr,Mr))(temp)
    W_real = Lambda(lambda x: tf.cos(x)/math.sqrt(Nr))(temp)
    W_imag = Lambda(lambda x: tf.sin(x)/math.sqrt(Nr))(temp)
    W_real_imag = Lambda(lambda x: tf.concat(x, axis=-1), name='predicted_W') \
        ([tf.expand_dims(W_real, axis=-1), tf.expand_dims(W_imag, axis=-1)])

    # obtain measurement matrices
    Phi_real_imag, y_real_imag = Phi_Layer_multipleT(Nt, Nr, Mt, Mr, G, N_r_RF, num_sc)([W_real_imag,noise_real_imag,H_real_imag])

    # of shape (?,G**2,num_sc)
    alpha_list_init = tf.tile(tf.ones_like(H_real_imag[:, 0, 0, 0:1, 0:1]), (1, G ** 2, num_sc))

    # update mu and Sigma
    mu_real, mu_imag, diag_Sigma_real = Lambda(lambda x: update_mu_Sigma(x, num_sc, sigma_2, Mr, Mt))(
        [Phi_real_imag, y_real_imag, alpha_list_init])

    for i in range(num_layers):
        mu_square = Lambda(lambda x: x[0] ** 2 + x[1] ** 2)([mu_real, mu_imag])

        # feature tensor of dim (?,G,G,num_sc,2)
        temp = Lambda(lambda x: tf.concat(x, axis=-1)) \
            ([tf.reshape(mu_square, (-1, G, G, num_sc, 1)), tf.reshape(diag_Sigma_real, (-1, G, G, num_sc, 1))])

        # 3D Convolution, with circular padding, conv padding should use "valid" not "same"
        conv_layer1 = Conv3D(trainable=False,name='SBL_%d1'%i,filters=num_filters,kernel_size=kernel_size,strides=1,padding='valid',activation='relu')
        conv_layer2 = Conv3D(trainable=False,name='SBL_%d2'%i,filters=1,kernel_size=kernel_size,strides=1,padding='valid',activation='relu')

        # feature tensor of dim (?,G,G,num_sc,3), 2+1=3
        # notice that two modalities of features have inversed (G,G) shape due to vectorization
        temp = Lambda(lambda x: tf.concat(x, axis=-1))([temp, \
                               tf.expand_dims(tf.transpose(X_abs_previous_block,(0,2,1,3)),axis=-1)])
        temp = circular_padding_2d(temp, kernel_size=kernel_size, strides=1)
        temp = conv_layer1(temp)

        temp = Lambda(lambda x: tf.concat(x, axis=-1))([temp, \
                               tf.expand_dims(tf.transpose(X_abs_previous_block,(0,2,1,3)),axis=-1)])
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

    model = Model(inputs=[H_real_imag, noise_real_imag, X_abs_previous_block], outputs=H_hat)

    return model


## only train the matrix with Conv weights frozen
num_layers = 3
num_filters = 8
kernel_size = 5

model = SBL_net(Mt, Mr, Nt, Nr, G, N_r_RF, num_sc, num_layers, num_filters, kernel_size)

model.summary()

epochs = int(args.epochs)
batch_size = 16
best_model_path = './models/best_Mr_%d_Mt_%d_SNR_%d_multipleT_new.h5'%(Mr,Mt,SNR)

# weight initialization
Kron2 = np.kron(np.conjugate(A_T),A_R)
Kron2 = np.expand_dims(Kron2,axis=-1)
A_R = np.expand_dims(A_R,axis=-1)
A_T = np.expand_dims(A_T.H,axis=-1)
init_weights_Kron2 = np.concatenate([np.real(Kron2),np.imag(Kron2)],axis=-1)
init_weights_R = np.concatenate([np.real(A_R), np.imag(A_R)],axis=-1)
init_weights_T = np.concatenate([np.real(A_T), np.imag(A_T)],axis=-1)

# load the pretrained Conv weights with various random phase matrices
weight_dict = np.load('./results/weight_dict_Mr_%d_Mt_%d_SNR_%d_multipleT_fixed_W.npy'%(Mr,Mt,SNR),allow_pickle=True).item()
for layer in model.layers:
    if 'phi_' in layer.name:
        print('Set Phi weights')
        init_phases_F = weight_dict['F_phases']
        layer.set_weights([init_phases_F,init_weights_Kron2])
    if 'a_r_' in layer.name:
        print('Set A_R weights')
        layer.set_weights([init_weights_R])
    if 'a_t_' in layer.name:
        print('Set A_T weights')
        layer.set_weights([init_weights_T])
    if 'SBL_' in layer.name:
        print('Set Conv weights')
        layer.set_weights(weight_dict[layer.name])

# define callbacks
checkpointer = ModelCheckpoint(best_model_path, verbose=1, save_best_only=True, save_weights_only=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=2, verbose=1, mode='auto',
                              min_delta=1e-5, min_lr=1e-5)
early_stopping = EarlyStopping(monitor='val_loss', min_delta=1e-5, patience=3)

model.compile(loss='mse', optimizer=Adam(lr=1e-4))

# load previous block's estimation results
X_abs_previous_block = io.loadmat('./results/X_abs_previous_block_Mr_%d_Mt_%d_SNR_%d.mat'%(Mr,Mt,SNR))['X_abs_previous_block'][-data_num:]

model.fit([H_list,noise_list,X_abs_previous_block], H_list, epochs=epochs, batch_size=batch_size,
          verbose=1, shuffle=True, \
          validation_split=0.1, callbacks=[checkpointer, reduce_lr, early_stopping])


## defrost the Conv layer and do joint training
model.trainable = True

model.summary()

# define callbacks
checkpointer = ModelCheckpoint(best_model_path, verbose=1, save_best_only=True, save_weights_only=True)
early_stopping = EarlyStopping(monitor='val_loss', min_delta=1e-5, patience=3)

# finetune with a small learning rate
model.compile(loss='mse', optimizer=Adam(lr=1e-5))

model.fit([H_list,noise_list,X_abs_previous_block], H_list, epochs=epochs, batch_size=batch_size,
          verbose=1, shuffle=True, \
          validation_split=0.1, callbacks=[checkpointer, early_stopping])


# test performance and save
model.load_weights(best_model_path)

batch_size = 8

test_num = int(args.test_num)
predictions_H = model.predict([H_list[-test_num:], noise_list[-test_num:], X_abs_previous_block[-test_num:]],batch_size=batch_size)
predictions_H = predictions_H[:,:,:,:,0] + 1j*predictions_H[:,:,:,:,1]
true_H = H_list[-test_num:,:,:,:,0] + 1j*H_list[-test_num:,:,:,:,1]

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
file_handle.write('MultipleT, train matrixnet after SBL net, single SNR training, SNR=%d dB:\n'%(SNR))
file_handle.write(str([mse,nmse]))
file_handle.write('\n')
file_handle.write('\n')
file_handle.close()



#%% observe the learned measurement matrices
init_phases_W = weight_dict['W_phases']
W_optimized = (np.cos(init_phases_W)+1j*np.sin(init_phases_W))/np.sqrt(Nr)
W_optimized = np.matrix(W_optimized)

if SNR==20:
    from matplotlib import pyplot as plt
    import tensorflow.keras.backend as K
    for i in range(len(model.layers)):
        if model.layers[i].name == 'predicted_W':
            W_index = i
    get_W = K.function([model.input],[model.layers[W_index].output])

    A_T = dictionary(Nt, G)
    A_T = np.matrix(A_T)
    A_R = dictionary(Nr, G)
    A_R = np.matrix(A_R)

    for sample_index in [1,2,3,4,5,6,7,8,9]:
        W_sample = get_W([H_list[sample_index:sample_index+1], noise_list[sample_index:sample_index+1],\
                          X_abs_previous_block[sample_index:sample_index+1]])[0]
        W_sample = np.squeeze(W_sample)
        W_sample = W_sample[:,:,0]+1j*W_sample[:,:,1]
        W_sample = np.matrix(W_sample)
        # print(W_sample.H.dot(W_sample))

        X_sample_abs = 0
        for sc in range(num_sc):
            H_sample_sc = H_list[sample_index,:,:,sc,0]+1j*H_list[sample_index,:,:,sc,1]
            X_sample_sc = (Nt*Nr/G**2)*(A_R.H.dot(H_sample_sc).dot(A_T))
            X_sample_abs = X_sample_abs + np.abs(X_sample_sc)
        X_sample_mean_scs = X_sample_abs/num_sc
        X_sample_mean_scs_mean_AoDs = np.mean(X_sample_mean_scs,axis=-1)[:,0]

        measurement_matrix_R = W_sample.H.dot(A_R)
        measurement_energy_R = np.linalg.norm(measurement_matrix_R,axis=0)

        measurement_matrix_R_optimized = W_optimized.H.dot(A_R)
        measurement_energy_R_optimized = np.linalg.norm(measurement_matrix_R_optimized,axis=0)

        plt.figure()
        plt.plot(X_sample_mean_scs_mean_AoDs/np.max(X_sample_mean_scs_mean_AoDs),'b-')
        plt.plot(measurement_energy_R/np.max(measurement_energy_R),'r--')
        plt.plot(measurement_energy_R_optimized / np.max(measurement_energy_R_optimized) *0.7 , 'k--')

        plt.legend(['Mean channel',r'Adaptively predicted $W$',r'Jointly optimized $W$'])
        plt.xlabel('Angular grid index')
        plt.ylabel('Energy')

        plt.savefig('./results/Predicted_matrix%d.png' % sample_index)

        io.savemat('./results/learned_matrix_%d.mat' % sample_index,{'channel_energy': X_sample_mean_scs_mean_AoDs, 'matrix_energy_predicted': measurement_energy_R,'matrix_energy_optimized': measurement_energy_R_optimized})
