import numpy as np
np.random.seed(2022)

from scipy import io

from functions import dictionary,update_mu_Sigma,update_alpha_PC,update_mu_Sigma_PC

import argparse


#%% Load channel
parser = argparse.ArgumentParser()

parser.add_argument('-Mr')
parser.add_argument('-Mt')
parser.add_argument('-SNR')
parser.add_argument('-test_num')
parser.add_argument('-num_layers')
parser.add_argument('-gpu_index')

args = parser.parse_args()

Mr = int(args.Mr) # number of receive beams at the BS
Mt = int(args.Mt) # number of transmit beams at the user

SNR = int(args.SNR) # SNR
sigma_2 = 1/10**(SNR/10) # noise variance

resolution = 2 # resolution of angle grids
Nt = 32
Nr = 32
G = resolution*np.max([Nt,Nr])

test_num = int(args.test_num)
cross_val_num = test_num//10
data_num = test_num + cross_val_num # number of testing samples
H_list = io.loadmat('./data/channel.mat')['H_list'][-data_num:]

# fixed parameters
N_r_RF = 4 # number of receive RF chains at the BS
# OFDM parameters
num_sc = 8 # number of subcarriers
fc = 28 * 1e9 # central frequency
W = 4 * 1e9 # bandwidth
eta = W / num_sc  # subcarrier spacing

A_T = dictionary(Nt, G)
A_T = np.matrix(A_T)
A_R = dictionary(Nr, G)
A_R = np.matrix(A_R)
Kron2 = np.kron(np.conjugate(A_T),A_R)

Phi_list = np.zeros((data_num, Mr*Mt, G**2)) + 1j * np.zeros((data_num, Mr*Mt, G**2))
y_list = np.zeros((data_num, num_sc, Mr*Mt, 1)) + 1j * np.zeros((data_num, num_sc, Mr*Mt, 1))
h_list = np.zeros((data_num, num_sc, Nr*Nt, 1)) + 1j * np.zeros((data_num, num_sc, Nr*Nt, 1))

for i in range(data_num):
    if i % 500 == 0:
        print('%d/%d' % (i, data_num))
    random_phases_T = np.random.uniform(0, 2 * np.pi, Nt*Mt)
    random_phases_R = np.random.uniform(0, 2 * np.pi, Nr*Mr)
    F = (np.cos(random_phases_T) + 1j * np.sin(random_phases_T)) / np.sqrt(Nt)
    F = np.reshape(F, (Nt, Mt))
    F = np.matrix(F)
    W = (np.cos(random_phases_R) + 1j * np.sin(random_phases_R)) / np.sqrt(Nr)
    W = np.reshape(W, (Nr, Mr))
    W = np.matrix(W)    

    Q = np.kron(F.T,W.H)
    Phi = Q.dot(Kron2)
    Phi_list[i] = Phi

    # generate the effective noise, and obtain its corresponding covariance matrix   
    noise_list = []
    for r in range(Mr//N_r_RF):
        W_r = W[:,r*N_r_RF:(r+1)*N_r_RF]
        original_noise = np.sqrt(sigma_2/2)*(np.random.randn(Nr,Mt*num_sc)+1j*np.random.randn(Nr,Mt*num_sc))
        noise_list.append(W_r.H.dot(original_noise))
    effective_noise_list = np.concatenate(noise_list,axis=0)
    effective_noise_list = np.reshape(np.array(effective_noise_list),(Mr,Mt,num_sc))
                 
    for n in range(num_sc):
        H_subcarrier = H_list[i,:,:,n,0]+1j*H_list[i,:,:,n,1] # Nr x Nt
        # notice that, vectorization is to stack column-wise 
        H_subcarrier = np.transpose(H_subcarrier) # Nt x Nr
        h_subcarrier = np.reshape(H_subcarrier,(Nr*Nt,1))
        h_list[i,n] = h_subcarrier    
        # received signal with effective noise added
        effective_noise = np.transpose(effective_noise_list[:,:,n])
        effective_noise = np.reshape(effective_noise,(Mt*Mr,1))
        y_list[i,n] = Q.dot(h_subcarrier) + effective_noise

Phi_list = np.expand_dims(Phi_list, axis=-1)
Phi_real_imag_list = np.concatenate([np.real(Phi_list), np.imag(Phi_list)], axis=-1)

y_real_imag_list = np.concatenate([np.real(y_list), np.imag(y_list)], axis=-1)
h_real_imag_list = np.concatenate([np.real(h_list), np.imag(h_list)], axis=-1)
# put the dimension num_sc after R/Nt, to facilitate matrix computation
y_real_imag_list = np.transpose(y_real_imag_list,(0,2,1,3))
h_real_imag_list = np.transpose(h_real_imag_list,(0,2,1,3))

print(Phi_real_imag_list.shape) # (data_num,Mr*Mt,G**2,2)
print(y_real_imag_list.shape) # (data_num,Mr*Mt,num_sc,2)
print(h_real_imag_list.shape) # (data_num,Nr*Nt,num_sc,2)


#%%
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input,Lambda

import os
os.environ["CUDA_VISIBLE_DEVICES"]=args.gpu_index
gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpus[0],True)



#%%
num_layers = int(args.num_layers)
test_num_layers = num_layers//2

h_list_cross_val = h_list[:cross_val_num]
h_list_test = h_list[-test_num:]

batch_size = 50

## SBL
def SBL_layer(Mt, Mr, G, num_sc, sigma_2):
    Phi_real_imag = Input(shape=(Mt*Mr, G**2, 2))
    y_real_imag = Input(shape=(Mt*Mr, num_sc, 2))
    alpha_list = Input(shape=(G**2,num_sc))
    # update mu and Sigma
    mu_real, mu_imag, diag_Sigma_real = Lambda(lambda x: update_mu_Sigma(x,num_sc,sigma_2,Mr,Mt))(
        [Phi_real_imag, y_real_imag, alpha_list])
    # update alpha_list
    mu_square = Lambda(lambda x: x[0] ** 2 + x[1] ** 2)([mu_real, mu_imag])
    alpha_list_updated = Lambda(lambda x:x[0]+x[1])([mu_square,diag_Sigma_real])
    model = Model(inputs=[Phi_real_imag, y_real_imag, alpha_list], outputs=[alpha_list_updated,mu_real,mu_imag])
    model.compile(loss='mse')
    return model
SBL_single_layer = SBL_layer(Mt,Mr,G,num_sc,sigma_2)

alpha_list = np.ones((test_num,G**2,num_sc)) #initialization
for i in range(num_layers):
    if i % 10 == 0:
        print('SBL iteration %d'%i)
    [alpha_list,mu_real,mu_imag] = SBL_single_layer.predict([Phi_real_imag_list[-test_num:], y_real_imag_list[-test_num:], alpha_list],batch_size=batch_size)

predictions_x = mu_real+1j*mu_imag
predictions_x = np.reshape(predictions_x,(test_num,G,G,num_sc))
# notice that the inverse vectorization operation is also to re-stack column wise
predictions_x = np.transpose(predictions_x,(0,3,2,1))

error = 0
error_nmse = 0
for i in range(test_num):
    prediction_h_list = np.zeros((num_sc,Nr,Nt))+1j*np.zeros((num_sc,Nr,Nt))
    true_h_list = np.zeros((num_sc, Nr, Nt)) + 1j * np.zeros((num_sc, Nr, Nt))
    for j in range(num_sc):
        prediction_h = (A_R.dot(predictions_x[i, j])).dot(A_T.H)
        true_h = np.reshape(h_list_test[i, j], (Nt, Nr))
        true_h = np.transpose(true_h)
        prediction_h_list[j]=prediction_h
        true_h_list[j]=true_h
    error = error + np.linalg.norm(prediction_h_list - true_h_list) ** 2
    error_nmse = error_nmse + (np.linalg.norm(prediction_h_list-true_h_list)/np.linalg.norm(true_h_list)) ** 2
mse_sbl = error / (test_num * Nt * Nr * num_sc)
nmse_sbl = error_nmse / test_num
print(mse_sbl)
print(nmse_sbl)


## M-SBL
def M_SBL_layer(Mt, Mr, G, num_sc, sigma_2):
    Phi_real_imag = Input(shape=(Mt * Mr, G ** 2, 2))
    y_real_imag = Input(shape=(Mt * Mr, num_sc, 2))
    alpha_list = Input(shape=(G ** 2, num_sc))
    # update mu and Sigma
    mu_real, mu_imag, diag_Sigma_real = Lambda(lambda x: update_mu_Sigma(x, num_sc, sigma_2, Mr, Mt))(
        [Phi_real_imag, y_real_imag, alpha_list])
    # update alpha_list
    mu_square = Lambda(lambda x: x[0] ** 2 + x[1] ** 2)([mu_real, mu_imag])
    mu_square_average = Lambda(lambda x:tf.reduce_mean(x,axis=-1,keepdims=True))(mu_square)
    mu_square = tf.tile(mu_square_average,(1,1,num_sc))
    alpha_list_updated = Lambda(lambda x: x[0] + x[1])([mu_square, diag_Sigma_real])
    model = Model(inputs=[Phi_real_imag, y_real_imag, alpha_list], outputs=[alpha_list_updated, mu_real, mu_imag])
    model.compile(loss='mse')
    return model
M_SBL_single_layer = M_SBL_layer(Mt,Mr,G,num_sc,sigma_2)

alpha_list = np.ones((test_num, G ** 2, num_sc))  # initialization
for i in range(num_layers):
    if i % 10 == 0:
        print('M-SBL iteration %d' % i)
    [alpha_list, mu_real, mu_imag] = M_SBL_single_layer.predict([Phi_real_imag_list[-test_num:], y_real_imag_list[-test_num:], alpha_list],batch_size=batch_size)

predictions_x = mu_real + 1j * mu_imag
predictions_x = np.reshape(predictions_x, (test_num, G, G, num_sc))
# notice that the inverse vectorization operation is also to re-stack column wise
predictions_x = np.transpose(predictions_x, (0, 3, 2, 1))

error = 0
error_nmse = 0
for i in range(test_num):
    prediction_h_list = np.zeros((num_sc,Nr,Nt))+1j*np.zeros((num_sc,Nr,Nt))
    true_h_list = np.zeros((num_sc, Nr, Nt)) + 1j * np.zeros((num_sc, Nr, Nt))
    for j in range(num_sc):
        prediction_h = (A_R.dot(predictions_x[i, j])).dot(A_T.H)
        true_h = np.reshape(h_list_test[i, j], (Nt, Nr))
        true_h = np.transpose(true_h)
        prediction_h_list[j]=prediction_h
        true_h_list[j]=true_h
    error = error + np.linalg.norm(prediction_h_list - true_h_list) ** 2
    error_nmse = error_nmse + (np.linalg.norm(prediction_h_list-true_h_list)/np.linalg.norm(true_h_list)) ** 2
mse_msbl = error / (test_num * Nt * Nr * num_sc)
nmse_msbl = error_nmse / test_num
print(mse_msbl)
print(nmse_msbl)


## PC_SBL
def PC_SBL_layer(Mt, Mr, G, num_sc, sigma_2,a,b,beta):
    Phi_real_imag = Input(shape=(Mt * Mr, G ** 2, 2))
    y_real_imag = Input(shape=(Mt * Mr, num_sc, 2))
    alpha_list = Input(shape=(G ** 2, num_sc))
    # update mu and Sigma
    mu_real, mu_imag, diag_Sigma_real = Lambda(lambda x: update_mu_Sigma_PC(x, G, num_sc, sigma_2, Mr, Mt, beta))(
        [Phi_real_imag, y_real_imag, alpha_list])
    # update alpha_list
    alpha_list_updated = update_alpha_PC([mu_real, mu_imag, diag_Sigma_real], G, num_sc, a, b, beta)
    model = Model(inputs=[Phi_real_imag, y_real_imag, alpha_list], outputs=[alpha_list_updated, mu_real, mu_imag])
    model.compile(loss='mse')
    return model

# search the best combination of a and beta using 10 samples with 50 iterations
a_list = [0.2,0.5,0.8,1.1]
beta_list = [0,0.25,0.5,0.75,1]
b = 1e-6

best_a_index = 0
best_beta_index = 0
error_min = 1e8

count = 0
for m in range(len(a_list)):
    for n in range(len(beta_list)):
        a = a_list[m]
        beta = beta_list[n]
        count = count + 1
        print('Trying combination %d'%count)

        PC_SBL_single_layer = PC_SBL_layer(Mt, Mr, G, num_sc, sigma_2,a,b,beta)

        alpha_list = np.ones((cross_val_num, G ** 2, num_sc))  # initialization
        for i in range(test_num_layers):
            if i==0:
                [alpha_list, mu_real, mu_imag] = SBL_single_layer.predict([Phi_real_imag_list[:cross_val_num], y_real_imag_list[:cross_val_num], alpha_list],batch_size=batch_size)
            else:
                [alpha_list, mu_real, mu_imag] = PC_SBL_single_layer.predict([Phi_real_imag_list[:cross_val_num], y_real_imag_list[:cross_val_num], alpha_list],batch_size=batch_size)

        predictions_x = mu_real + 1j * mu_imag
        predictions_x = np.reshape(predictions_x, (cross_val_num, G, G, num_sc))
        # notice that the inverse vectorization operation is also to re-stack column wise
        predictions_x = np.transpose(predictions_x, (0, 3, 2, 1))

        error = 0
        for i in range(cross_val_num):
            prediction_h_list = np.zeros((num_sc,Nr,Nt))+1j*np.zeros((num_sc,Nr,Nt))
            true_h_list = np.zeros((num_sc, Nr, Nt)) + 1j * np.zeros((num_sc, Nr, Nt))
            for j in range(num_sc):
                prediction_h = (A_R.dot(predictions_x[i, j])).dot(A_T.H)
                true_h = np.reshape(h_list_cross_val[i, j], (Nt, Nr))
                true_h = np.transpose(true_h)
                prediction_h_list[j]=prediction_h
                true_h_list[j]=true_h
            error = error + np.linalg.norm(prediction_h_list - true_h_list) ** 2

        if error<error_min:
            best_a_index = m
            best_beta_index = n
            error_min = error
            print('Best combination updated')
            print('a=%.2f,beta=%.2f'%(a,beta))

# use the best combination of a and beta
a = a_list[best_a_index]
beta = beta_list[best_beta_index]

print('Best hyperparameters used:\n')
print('a=%.2f,beta=%.2f'%(a,beta))

PC_SBL_single_layer = PC_SBL_layer(Mt, Mr, G, num_sc, sigma_2, a, b, beta)

alpha_list = np.ones((test_num, G ** 2, num_sc))  # initialization
for i in range(num_layers):
    if i % 10 == 0:
        print('PC-SBL iteration %d' % i)
    if i == 0:
        [alpha_list, mu_real, mu_imag] = SBL_single_layer.predict(
            [Phi_real_imag_list[-test_num:], y_real_imag_list[-test_num:], alpha_list],batch_size=batch_size)
    else:
        [alpha_list, mu_real, mu_imag] = PC_SBL_single_layer.predict(
            [Phi_real_imag_list[-test_num:], y_real_imag_list[-test_num:], alpha_list],batch_size=batch_size)

predictions_x = mu_real + 1j * mu_imag
predictions_x = np.reshape(predictions_x, (test_num, G, G, num_sc))
# notice that the inverse vectorization operation is also to re-stack column wise
predictions_x = np.transpose(predictions_x, (0, 3, 2, 1))

error = 0
error_nmse = 0
for i in range(test_num):
    prediction_h_list = np.zeros((num_sc, Nr, Nt)) + 1j * np.zeros((num_sc, Nr, Nt))
    true_h_list = np.zeros((num_sc, Nr, Nt)) + 1j * np.zeros((num_sc, Nr, Nt))
    for j in range(num_sc):
        prediction_h = (A_R.dot(predictions_x[i, j])).dot(A_T.H)
        true_h = np.reshape(h_list_test[i, j], (Nt, Nr))
        true_h = np.transpose(true_h)
        prediction_h_list[j] = prediction_h
        true_h_list[j] = true_h
    error = error + np.linalg.norm(prediction_h_list - true_h_list) ** 2
    error_nmse = error_nmse + (
                np.linalg.norm(prediction_h_list - true_h_list) / np.linalg.norm(true_h_list)) ** 2
mse_pcsbl = error / (test_num * Nt * Nr * num_sc)
nmse_pcsbl = error_nmse / test_num
print(mse_pcsbl)
print(nmse_pcsbl)



#%% save performance
sbl_performance_list = [mse_sbl,nmse_sbl,mse_msbl,nmse_msbl,mse_pcsbl,nmse_pcsbl]

file_handle=open('./results/Performance_Mr_%d_Mt_%d.txt'%(Mr,Mt),mode='a+')
file_handle.write('SBL with random W and F, SNR=%d dB:\n'%SNR)
file_handle.write(str(sbl_performance_list))
file_handle.write('\n')
file_handle.write('Best hyperparameters used for PC SBL, a and beta:\n')
file_handle.write(str([a_list[best_a_index],beta_list[best_beta_index]]))
file_handle.write('\n')
file_handle.close()
