import numpy as np
np.random.seed(2022)

import tensorflow as tf
tf.random.set_seed(2022)

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input,Conv2D,Conv3D,Reshape,Lambda,Dense,Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint

from scipy import io

from functions import dictionary,Phi_Layer,update_mu_Sigma,circular_padding_2d,circular_padding_2D,A_R_Layer,A_T_Layer

import argparse

#%% Load channel
parser = argparse.ArgumentParser()

parser.add_argument('-data_num')
parser.add_argument('-gpu_index')

args = parser.parse_args()

import os
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_index
gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)

#  To investigate the impact of nn structure, use typical parameters

Mr = 16 # number of receive beams at the BS
Mt = 16 # number of transmit beams at the user

SNR = 20 # SNR
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




import tensorflow.keras.backend as K
def symmetric_pre(feature_map, kernel_size):
    # split into four feature_maps
    G = feature_map.shape[1] + 1 - kernel_size
    sub_feature_map1 = feature_map[:, :G // 2 + kernel_size - 1, :G // 2 + kernel_size - 1]
    sub_feature_map2 = feature_map[:, :G // 2 + kernel_size - 1, G // 2:]
    sub_feature_map2 = K.reverse(sub_feature_map2, axes=2)
    sub_feature_map2 = tf.concat([sub_feature_map2[:,:,:,:,:-1],-sub_feature_map2[:,:,:,:,-1:]],axis=-1)
    sub_feature_map3 = feature_map[:, G // 2:, :G // 2 + kernel_size - 1]
    sub_feature_map3 = K.reverse(sub_feature_map3, axes=1)
    sub_feature_map3 = tf.concat([sub_feature_map3[:, :, :, :, :-2], -sub_feature_map3[:, :, :, :, -2:-1],\
                                  sub_feature_map3[:, :, :, :, -1:]], axis=-1)
    sub_feature_map4 = feature_map[:, G // 2:, G // 2:]
    sub_feature_map4 = K.reverse(sub_feature_map4, axes=[1, 2])
    sub_feature_map4 = tf.concat([sub_feature_map4[:, :, :, :, :-2], -sub_feature_map4[:, :, :, :, -2:]], axis=-1)

    return sub_feature_map1, sub_feature_map2, sub_feature_map3, sub_feature_map4

def symmetric_post(sub_output1, sub_output2, sub_output3, sub_output4):
    output_upper = tf.concat([sub_output1, K.reverse(sub_output2, axes=2)], axis=2)
    output_lower = tf.concat([K.reverse(sub_output3, axes=1), K.reverse(sub_output4, axes=[1, 2])], axis=2)
    output = tf.concat([output_upper, output_lower], axis=1)

    return output


#%% construct the network
mode_list = ['LCConv3D']
def SBL_net(Mt, Mr, Nt, Nr, G, num_sc, num_layers, num_filters, kernel_size, mode):
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

        if mode=='MConv1':
            mu_square_average = Lambda(lambda x:tf.reduce_mean(x,axis=-1,keepdims=True))(temp[:,:,:,:,0])
            diag_Sigma_real_common = temp[:,:,:,0:1,1]
            # (?,G,G,2)
            temp = Lambda(lambda x: tf.concat(x, axis=-1))([mu_square_average,diag_Sigma_real_common])
            conv_layer1 = Conv2D(name='SBL_%d1' % i, filters=num_filters, kernel_size=kernel_size, strides=1,
                                 padding='same', activation='relu')
            conv_layer2 = Conv2D(name='SBL_%d2' % i, filters=1, kernel_size=kernel_size, strides=1, padding='same',
                                 activation='relu')
            # (?,G,G,num_filters)
            temp = conv_layer1(temp)
            # (?,G,G,1)
            temp = conv_layer2(temp)
            alpha_list = Lambda(lambda x:tf.tile(x,(1,1,1,num_sc)))(temp)

        if mode=='MConv2':
            mu_square_average = Lambda(lambda x:tf.reduce_mean(x,axis=-1,keepdims=True))(temp[:,:,:,:,0])
            diag_Sigma_real_common = temp[:,:,:,0:1,1]
            # (?,G,G,2)
            temp = Lambda(lambda x: tf.concat(x, axis=-1))([mu_square_average,diag_Sigma_real_common])
            conv_layer1 = Conv2D(name='SBL_%d1' % i, filters=num_filters, kernel_size=kernel_size, strides=1,
                                 padding='same', activation='relu')
            conv_layer2 = Conv2D(name='SBL_%d2' % i, filters=num_sc, kernel_size=kernel_size, strides=1, padding='same',
                                 activation='relu')
            # (?,G,G,num_filters)
            temp = conv_layer1(temp)
            # (?,G,G,num_sc)
            alpha_list = conv_layer2(temp)

        if mode=='MConv3':
            mu_square_average = Lambda(lambda x:tf.reduce_mean(x,axis=-1,keepdims=True))(temp[:,:,:,:,0])
            diag_Sigma_real_common = temp[:,:,:,0:1,1]
            # (?,G,G,2)
            temp = Lambda(lambda x: tf.concat(x, axis=-1))([mu_square_average,diag_Sigma_real_common])
            conv_layer1 = Conv2D(name='SBL_%d1' % i, filters=num_filters, kernel_size=kernel_size, strides=1,
                                 padding='same', activation='relu')
            conv_layer2 = Conv2D(name='SBL_%d2' % i, filters=num_sc, kernel_size=kernel_size, strides=1, padding='same',
                                 activation='relu')

            # sin values of (G,G) angular grids
            sin_values = tf.cast(tf.linspace(-1 + 1 / G, 1 - 1 / G, G), tf.float32)
            sin_values_1 = tf.tile(tf.expand_dims(sin_values, axis=-1), (1, G))
            sin_values_2 = tf.transpose(sin_values_1)
            # (G,G,2)
            sin_values = tf.concat([tf.expand_dims(sin_values_1, axis=-1), tf.expand_dims(sin_values_2, axis=-1)],
                                   axis=-1)
            # (?,G,G,2)
            sin_values = sin_values * tf.ones_like(temp)

            # expand the information of sine values, feature tensor of dim (?,G,G,4)
            temp = Lambda(lambda x: tf.concat(x, axis=-1))([temp, sin_values])
            # (?,G,G,num_filters)
            temp = conv_layer1(temp)

            temp = Lambda(lambda x: tf.concat(x, axis=-1))([temp, sin_values])
            # (?,G,G,num_sc)
            alpha_list = conv_layer2(temp)

        if mode=='Conv2D':
            conv_layer1 = Conv2D(name='SBL_%d1'%i,filters=num_filters,kernel_size=kernel_size,strides=1,padding='same',activation='relu')
            conv_layer2 = Conv2D(name='SBL_%d2'%i,filters=1,kernel_size=kernel_size,strides=1,padding='same',activation='relu')
            feature_map_sc_list = []
            for k in range(num_sc):
                feature_map_sc = tf.expand_dims(conv_layer1(temp[:,:,:,k]),axis=-2)
                feature_map_sc_list.append(feature_map_sc)
            temp = tf.concat(feature_map_sc_list,axis=-2)

            feature_map_sc_list = []
            for k in range(num_sc):
                feature_map_sc = tf.expand_dims(conv_layer2(temp[:,:,:,k]),axis=-2)
                feature_map_sc_list.append(feature_map_sc)
            alpha_list = tf.concat(feature_map_sc_list,axis=-2)

        if mode=='CConv2D':
            conv_layer1 = Conv2D(name='SBL_%d1'%i,filters=num_filters,kernel_size=kernel_size,strides=1,padding='valid',activation='relu')
            conv_layer2 = Conv2D(name='SBL_%d2'%i,filters=1,kernel_size=kernel_size,strides=1,padding='valid',activation='relu')

            temp = circular_padding_2D(temp, kernel_size=kernel_size, strides=1)

            feature_map_sc_list = []
            for k in range(num_sc):
                feature_map_sc = tf.expand_dims(conv_layer1(temp[:,:,:,k]),axis=-2)
                feature_map_sc_list.append(feature_map_sc)
            temp = tf.concat(feature_map_sc_list,axis=-2)

            temp = circular_padding_2D(temp, kernel_size=kernel_size, strides=1)

            feature_map_sc_list = []
            for k in range(num_sc):
                feature_map_sc = tf.expand_dims(conv_layer2(temp[:,:,:,k]),axis=-2)
                feature_map_sc_list.append(feature_map_sc)
            alpha_list = tf.concat(feature_map_sc_list,axis=-2)

        if mode=='LConv2D':
            conv_layer1 = Conv2D(name='SBL_%d1'%i,filters=num_filters,kernel_size=kernel_size,strides=1,padding='same',activation='relu')
            conv_layer2 = Conv2D(name='SBL_%d2'%i,filters=1,kernel_size=kernel_size,strides=1,padding='same',activation='relu')

            # sin values of (G,G) angular grids
            sin_values = tf.cast(tf.linspace(-1 + 1 / G, 1 - 1 / G, G), tf.float32)
            sin_values_1 = tf.tile(tf.expand_dims(sin_values, axis=-1), (1, G))
            sin_values_2 = tf.transpose(sin_values_1)
            sin_values = tf.concat([tf.expand_dims(sin_values_1, axis=-1), tf.expand_dims(sin_values_2, axis=-1)],
                                   axis=-1)
            sin_values = tf.tile(tf.expand_dims(sin_values, axis=-2), (1, 1, num_sc, 1))
            sin_values = sin_values * tf.ones_like(temp)

            # expand the information of sine values, feature tensor of dim (?,G,G,num_sc,4)
            temp = Lambda(lambda x: tf.concat(x, axis=-1))([temp, sin_values])

            feature_map_sc_list = []
            for k in range(num_sc):
                feature_map_sc = tf.expand_dims(conv_layer1(temp[:,:,:,k]),axis=-2)
                feature_map_sc_list.append(feature_map_sc)
            temp = tf.concat(feature_map_sc_list,axis=-2)

            temp = Lambda(lambda x: tf.concat(x, axis=-1))([temp, sin_values])

            feature_map_sc_list = []
            for k in range(num_sc):
                feature_map_sc = tf.expand_dims(conv_layer2(temp[:,:,:,k]),axis=-2)
                feature_map_sc_list.append(feature_map_sc)
            alpha_list = tf.concat(feature_map_sc_list,axis=-2)

        if mode=='LCConv2D':
            conv_layer1 = Conv2D(name='SBL_%d1'%i,filters=num_filters,kernel_size=kernel_size,strides=1,padding='valid',activation='relu')
            conv_layer2 = Conv2D(name='SBL_%d2'%i,filters=1,kernel_size=kernel_size,strides=1,padding='valid',activation='relu')

            # sin values of (G,G) angular grids
            sin_values = tf.cast(tf.linspace(-1 + 1 / G, 1 - 1 / G, G), tf.float32)
            sin_values_1 = tf.tile(tf.expand_dims(sin_values, axis=-1), (1, G))
            sin_values_2 = tf.transpose(sin_values_1)
            sin_values = tf.concat([tf.expand_dims(sin_values_1, axis=-1), tf.expand_dims(sin_values_2, axis=-1)],
                                   axis=-1)
            sin_values = tf.tile(tf.expand_dims(sin_values, axis=-2), (1, 1, num_sc, 1))
            sin_values = sin_values * tf.ones_like(temp)

            # expand the information of sine values, feature tensor of dim (?,G,G,num_sc,4)
            temp = Lambda(lambda x: tf.concat(x, axis=-1))([temp, sin_values])

            temp = circular_padding_2D(temp, kernel_size=kernel_size, strides=1)

            feature_map_sc_list = []
            for k in range(num_sc):
                feature_map_sc = tf.expand_dims(conv_layer1(temp[:,:,:,k]),axis=-2)
                feature_map_sc_list.append(feature_map_sc)
            temp = tf.concat(feature_map_sc_list,axis=-2)

            temp = Lambda(lambda x: tf.concat(x, axis=-1))([temp, sin_values])

            temp = circular_padding_2D(temp, kernel_size=kernel_size, strides=1)

            feature_map_sc_list = []
            for k in range(num_sc):
                feature_map_sc = tf.expand_dims(conv_layer2(temp[:,:,:,k]),axis=-2)
                feature_map_sc_list.append(feature_map_sc)
            alpha_list = tf.concat(feature_map_sc_list,axis=-2)

        if mode=='Conv3D':
            # 3D Convolution, with circular padding, conv padding should use "valid" not "same"
            conv_layer1 = Conv3D(name='SBL_%d1' % i, filters=num_filters, kernel_size=kernel_size, strides=1,
                                 padding='same', activation='relu')
            conv_layer2 = Conv3D(name='SBL_%d2' % i, filters=1, kernel_size=kernel_size, strides=1, padding='same',
                                 activation='relu')
            temp = conv_layer1(temp)
            alpha_list = conv_layer2(temp)

        if mode=='CConv3D':
            # 3D Convolution, with circular padding, conv padding should use "valid" not "same"
            conv_layer1 = Conv3D(name='SBL_%d1' % i, filters=num_filters, kernel_size=kernel_size, strides=1,
                                 padding='valid', activation='relu')
            conv_layer2 = Conv3D(name='SBL_%d2' % i, filters=1, kernel_size=kernel_size, strides=1, padding='valid',
                                 activation='relu')

            temp = circular_padding_2d(temp, kernel_size=kernel_size, strides=1)
            temp = conv_layer1(temp)

            temp = circular_padding_2d(temp, kernel_size=kernel_size, strides=1)
            alpha_list = conv_layer2(temp)

        if mode=='LConv3D':
            # 3D Convolution, with circular padding, conv padding should use "valid" not "same"
            conv_layer1 = Conv3D(name='SBL_%d1' % i, filters=num_filters, kernel_size=kernel_size, strides=1,
                                 padding='same', activation='relu')
            conv_layer2 = Conv3D(name='SBL_%d2' % i, filters=1, kernel_size=kernel_size, strides=1, padding='same',
                                 activation='relu')

            # sin values of (G,G) angular grids
            sin_values = tf.cast(tf.linspace(-1 + 1 / G, 1 - 1 / G, G), tf.float32)
            sin_values_1 = tf.tile(tf.expand_dims(sin_values, axis=-1), (1, G))
            sin_values_2 = tf.transpose(sin_values_1)
            sin_values = tf.concat([tf.expand_dims(sin_values_1, axis=-1), tf.expand_dims(sin_values_2, axis=-1)],
                                   axis=-1)
            sin_values = tf.tile(tf.expand_dims(sin_values, axis=-2), (1, 1, num_sc, 1))
            sin_values = sin_values * tf.ones_like(temp)

            # expand the information of sine values, feature tensor of dim (?,G,G,num_sc,4)
            temp = Lambda(lambda x: tf.concat(x, axis=-1))([temp, sin_values])

            temp = conv_layer1(temp)

            temp = Lambda(lambda x: tf.concat(x, axis=-1))([temp, sin_values])

            alpha_list = conv_layer2(temp)


        if mode=='LCConv3D':
            # 3D Convolution, with circular padding, conv padding should use "valid" not "same"
            conv_layer1 = Conv3D(name='SBL_%d1' % i, filters=num_filters, kernel_size=kernel_size, strides=1,
                                 padding='valid', activation='relu')
            conv_layer2 = Conv3D(name='SBL_%d2' % i, filters=1, kernel_size=kernel_size, strides=1, padding='valid',
                                 activation='relu')
            conv_layer3 = Conv3D(name='SBL_%d3' % i, filters=1, kernel_size=kernel_size, strides=1, padding='valid',
                                 activation='relu')           
            conv_layer4 = Conv3D(name='SBL_%d4' % i, filters=1, kernel_size=kernel_size, strides=1, padding='valid',
                                 activation='relu')                                 
                                 
            # sin values of (G,G) angular grids
            sin_values = tf.cast(tf.linspace(-1 + 1 / G, 1 - 1 / G, G), tf.float32)
            sin_values_1 = tf.tile(tf.expand_dims(sin_values, axis=-1), (1, G))
            sin_values_2 = tf.transpose(sin_values_1)
            sin_values = tf.concat([tf.expand_dims(sin_values_1, axis=-1), tf.expand_dims(sin_values_2, axis=-1)],
                                   axis=-1)
            sin_values = tf.tile(tf.expand_dims(sin_values, axis=-2), (1, 1, num_sc, 1))
            sin_values = sin_values * tf.ones_like(temp)

            # expand the information of sine values, feature tensor of dim (?,G,G,num_sc,4)
            temp = Lambda(lambda x: tf.concat(x, axis=-1))([temp, sin_values])

            temp = circular_padding_2d(temp, kernel_size=kernel_size, strides=1)
            temp = conv_layer1(temp)

            temp = Lambda(lambda x: tf.concat(x, axis=-1))([temp, sin_values])

            temp = circular_padding_2d(temp, kernel_size=kernel_size, strides=1)
            alpha_list = conv_layer2(temp)

            #temp = Lambda(lambda x: tf.concat(x, axis=-1))([temp, sin_values])

            #temp = circular_padding_2d(temp, kernel_size=kernel_size, strides=1)
            #alpha_list = conv_layer3(temp)
            

        if mode=='LCConv3D_new':
            # 3D Convolution, with circular padding, conv padding should use "valid" not "same"
            conv_layer1 = Conv3D(name='SBL_%d1' % i, filters=num_filters, kernel_size=kernel_size, strides=1,
                                 padding='valid', activation='relu')
            conv_layer2 = Conv3D(name='SBL_%d2' % i, filters=1, kernel_size=kernel_size, strides=1, padding='valid',
                                 activation='relu')

            # include position features
            position_features_angle = tf.cast(tf.linspace(-1,1,G), tf.float32)
            position_features_AoA = tf.tile(tf.expand_dims(position_features_angle, axis=-1), (1, G))
            position_features_AoD = tf.transpose(position_features_AoA)
            position_features_angles = tf.concat([tf.expand_dims(position_features_AoA, axis=-1), tf.expand_dims(position_features_AoD, axis=-1)],
                                   axis=-1)
            # (G,G,num_sc,2)
            position_features_angles = tf.tile(tf.expand_dims(position_features_angles, axis=-2), (1, 1, num_sc, 1))

            position_features_sc = tf.cast(tf.linspace(-1,1,num_sc), tf.float32)
            # (G,G,num_sc,1)
            position_features_sc = tf.tile(tf.reshape(position_features_sc,(1,1,num_sc,1)),(G,G,1,1))

            # (G,G,num_sc,3)
            position_features = tf.concat([position_features_angles,position_features_sc],axis=-1)
            position_features = position_features * tf.ones_like(tf.tile(temp[:,:,:,:,:1],(1,1,1,1,3)))

            # expand the information of sine values, feature tensor of dim (?,G,G,num_sc,4)
            temp = Lambda(lambda x: tf.concat(x, axis=-1))([temp, position_features])

            temp = circular_padding_2d(temp, kernel_size=kernel_size, strides=1)
            temp = conv_layer1(temp)

            temp = Lambda(lambda x: tf.concat(x, axis=-1))([temp, position_features])

            temp = circular_padding_2d(temp, kernel_size=kernel_size, strides=1)
            alpha_list = conv_layer2(temp)

        if mode=='LCConv3D_new2':
            # 3D Convolution, with circular padding, conv padding should use "valid" not "same"
            conv_layer1 = Conv3D(name='SBL_%d1' % i,filters=num_filters,kernel_size=kernel_size,strides=1,padding='valid')
            conv_layer2 = Conv3D(name='SBL_%d2' % i,filters=1,kernel_size=kernel_size,strides=1,padding='valid')

            # include position features
            position_features_angle = tf.cast(tf.linspace(-1,1,G), tf.float32)
            position_features_AoA = tf.tile(tf.expand_dims(position_features_angle, axis=-1), (1, G))
            position_features_AoD = tf.transpose(position_features_AoA)
            position_features_angles = tf.concat([tf.expand_dims(position_features_AoA, axis=-1), tf.expand_dims(position_features_AoD, axis=-1)],
                                   axis=-1)
            # (G,G,num_sc,2)
            position_features_angles = tf.tile(tf.expand_dims(position_features_angles, axis=-2), (1, 1, num_sc, 1))

            position_features_sc = tf.cast(tf.linspace(-1,1,num_sc), tf.float32)
            # (G,G,num_sc,1)
            position_features_sc = tf.tile(tf.reshape(position_features_sc,(1,1,num_sc,1)),(G,G,1,1))

            # (G,G,num_sc,3)
            position_features = tf.concat([position_features_angles,position_features_sc],axis=-1)
            position_features = position_features * tf.ones_like(tf.tile(temp[:,:,:,:,:1],(1,1,1,1,3)))

            temp_position = Conv3D(kernel_size=1,filters=num_filters,strides=1)(position_features)

            temp = circular_padding_2d(temp,kernel_size=kernel_size,strides=1)
            temp = conv_layer1(temp)

            temp = temp + temp_position
            temp = tf.keras.activations.relu(temp)

            temp_position2 = Conv3D(kernel_size=1, filters=1, strides=1)(position_features)

            temp = circular_padding_2d(temp, kernel_size=kernel_size, strides=1)
            temp = conv_layer2(temp)

            temp = temp + temp_position2
            alpha_list = tf.keras.activations.relu(temp)


        if mode=='LConv3D_new':
            # 3D Convolution, with circular padding, conv padding should use "valid" not "same"
            conv_layer1 = Conv3D(name='SBL_%d1' % i, filters=num_filters, kernel_size=kernel_size, strides=1,
                                 padding='same', activation='relu')
            conv_layer2 = Conv3D(name='SBL_%d2' % i, filters=1, kernel_size=kernel_size, strides=1, padding='same',
                                 activation='relu')

            # include position features
            position_features_angle = tf.cast(tf.linspace(-1,1,G), tf.float32)
            position_features_AoA = tf.tile(tf.expand_dims(position_features_angle, axis=-1), (1, G))
            position_features_AoD = tf.transpose(position_features_AoA)
            position_features_angles = tf.concat([tf.expand_dims(position_features_AoA, axis=-1), tf.expand_dims(position_features_AoD, axis=-1)],
                                   axis=-1)
            # (G,G,num_sc,2)
            position_features_angles = tf.tile(tf.expand_dims(position_features_angles, axis=-2), (1, 1, num_sc, 1))

            position_features_sc = tf.cast(tf.linspace(-1,1,num_sc), tf.float32)
            # (G,G,num_sc,1)
            position_features_sc = tf.tile(tf.reshape(position_features_sc,(1,1,num_sc,1)),(G,G,1,1))

            # (G,G,num_sc,3)
            position_features = tf.concat([position_features_angles,position_features_sc],axis=-1)
            position_features = position_features * tf.ones_like(tf.tile(temp[:,:,:,:,:1],(1,1,1,1,3)))

            # expand the information of sine values, feature tensor of dim (?,G,G,num_sc,4)
            temp = Lambda(lambda x: tf.concat(x, axis=-1))([temp, position_features])

            temp = conv_layer1(temp)

            temp = Lambda(lambda x: tf.concat(x, axis=-1))([temp, position_features])

            alpha_list = conv_layer2(temp)


        if mode=='FC':
            FC_layer = Dense(G**2,activation='relu')
            feature_map_sc_list = []
            for k in range(num_sc):
                FC_input = Flatten()(temp[:,:,:,k]) # G*G*2
                FC_output = FC_layer(FC_input)
                FC_output = Reshape((G,G))(FC_output)
                feature_map_sc = tf.expand_dims(FC_output,axis=-2)
                feature_map_sc_list.append(feature_map_sc)
            alpha_list = tf.concat(feature_map_sc_list,axis=-2)


        if mode=='dynamic_1':
            # 3D Convolution, with circular padding, conv padding should use "valid" not "same"
            conv_layer1 = Conv3D(name='SBL_%d1' % i, filters=num_filters, kernel_size=kernel_size, strides=1,
                                 padding='valid', activation='relu')
            conv_layer2 = Conv3D(name='SBL_%d2' % i, filters=1, kernel_size=kernel_size, strides=1, padding='valid',
                                 activation='relu')

            # sin values of (G,G) angular grids
            sin_values = tf.cast(tf.linspace(-1 + 1 / G, 1 - 1 / G, G), tf.float32)
            sin_values_1 = tf.tile(tf.expand_dims(sin_values, axis=-1), (1, G))
            sin_values_2 = tf.transpose(sin_values_1)
            # of dimension (?,G,G,2)
            sin_values = tf.concat([tf.expand_dims(sin_values_1, axis=-1), tf.expand_dims(sin_values_2, axis=-1)],
                                   axis=-1)
            sin_values = sin_values * tf.ones_like(temp[:,:,:,0])

            # predict the position-dependent weights
            temp0 = Conv2D(filters=32,kernel_size=1,strides=1,activation='relu')(sin_values)
            weights = Conv2D(filters=num_filters,kernel_size=1,activation='sigmoid')(temp0)
            weights = Lambda(lambda x:tf.tile(tf.expand_dims(x,axis=-2),(1,1,1,num_sc,1)))(weights)

            temp = circular_padding_2d(temp, kernel_size=kernel_size, strides=1)
            temp = conv_layer1(temp)
            temp = temp*weights

            temp = circular_padding_2d(temp, kernel_size=kernel_size, strides=1)
            alpha_list = conv_layer2(temp)

        if mode=='LCConv3D_symmetric':
            # 3D Convolution, with circular padding, conv padding should use "valid" not "same"
            conv_layer1 = Conv3D(name='SBL_%d1' % i, filters=num_filters, kernel_size=kernel_size, strides=1,
                                 padding='valid', activation='relu')
            conv_layer2 = Conv3D(name='SBL_%d2' % i, filters=1, kernel_size=kernel_size, strides=1, padding='valid',
                                 activation='relu')

            # sin values of (G,G) angular grids
            sin_values = tf.cast(tf.linspace(-1 + 1 / G, 1 - 1 / G, G), tf.float32)
            sin_values_1 = tf.tile(tf.expand_dims(sin_values, axis=-1), (1, G))
            sin_values_2 = tf.transpose(sin_values_1)
            sin_values = tf.concat([tf.expand_dims(sin_values_1, axis=-1), tf.expand_dims(sin_values_2, axis=-1)],
                                   axis=-1)
            sin_values = tf.tile(tf.expand_dims(sin_values, axis=-2), (1, 1, num_sc, 1))
            sin_values = sin_values * tf.ones_like(temp)

            # expand the information of sine values, feature tensor of dim (?,G,G,num_sc,4)
            temp = Lambda(lambda x: tf.concat(x, axis=-1))([temp, sin_values])
            temp = circular_padding_2d(temp, kernel_size=kernel_size, strides=1)

            sub_feature_map1, sub_feature_map2, sub_feature_map3, sub_feature_map4 \
                = symmetric_pre(temp, kernel_size=kernel_size)
            sub_output1 = conv_layer1(sub_feature_map1)
            sub_output2 = conv_layer1(sub_feature_map2)
            sub_output3 = conv_layer1(sub_feature_map3)
            sub_output4 = conv_layer1(sub_feature_map4)

            temp = symmetric_post(sub_output1, sub_output2, sub_output3, sub_output4)

            temp = Lambda(lambda x: tf.concat(x, axis=-1))([temp, sin_values])
            temp = circular_padding_2d(temp, kernel_size=kernel_size, strides=1)

            sub_feature_map1, sub_feature_map2, sub_feature_map3, sub_feature_map4 \
                = symmetric_pre(temp, kernel_size=kernel_size)
            sub_output1 = conv_layer2(sub_feature_map1)
            sub_output2 = conv_layer2(sub_feature_map2)
            sub_output3 = conv_layer2(sub_feature_map3)
            sub_output4 = conv_layer2(sub_feature_map4)

            alpha_list = symmetric_post(sub_output1, sub_output2, sub_output3, sub_output4)

        if mode=='LCConv3D_learnable':
            # 3D Convolution, with circular padding, conv padding should use "valid" not "same"
            conv_layer1 = Conv3D(name='SBL_%d1' % i, filters=num_filters, kernel_size=kernel_size, strides=1,
                                 padding='valid', activation='relu')
            conv_layer2 = Conv3D(name='SBL_%d2' % i, filters=1, kernel_size=kernel_size, strides=1, padding='valid',
                                 activation='relu')

            # include position features
            position_features_angle = tf.cast(tf.linspace(-1,1,G), tf.float32)
            position_features_AoA = tf.tile(tf.expand_dims(position_features_angle, axis=-1), (1, G))
            position_features_AoD = tf.transpose(position_features_AoA)
            position_features_angles = tf.concat([tf.expand_dims(position_features_AoA, axis=-1), tf.expand_dims(position_features_AoD, axis=-1)],
                                   axis=-1)
            # (G,G,num_sc,2)
            position_features_angles = tf.tile(tf.expand_dims(position_features_angles, axis=-2), (1, 1, num_sc, 1))

            position_features_sc = tf.cast(tf.linspace(-1,1,num_sc), tf.float32)
            # (G,G,num_sc,1)
            position_features_sc = tf.tile(tf.reshape(position_features_sc,(1,1,num_sc,1)),(G,G,1,1))

            # (G,G,num_sc,3)
            position_features = tf.concat([position_features_angles,position_features_sc],axis=-1)
            position_features = position_features * tf.ones_like(tf.tile(temp[:,:,:,:,:1],(1,1,1,1,3)))

            # do 1D convolutions to project the position features into embedding spaces
            position_features = Conv3D(filters=16,kernel_size=1,activation='relu')(position_features)
            position_features = Conv3D(filters=8,kernel_size=1,activation='relu')(position_features)

            # expand the information of sine values, feature tensor of dim (?,G,G,num_sc,2+8)
            temp = Lambda(lambda x: tf.concat(x, axis=-1))([temp, position_features])

            temp = circular_padding_2d(temp, kernel_size=kernel_size, strides=1)
            temp = conv_layer1(temp)

            temp = Lambda(lambda x: tf.concat(x, axis=-1))([temp, position_features])

            temp = circular_padding_2d(temp, kernel_size=kernel_size, strides=1)
            alpha_list = conv_layer2(temp)


        if mode=='LCConv3D_learnable2':
            # 3D Convolution, with circular padding, conv padding should use "valid" not "same"
            conv_layer1 = Conv3D(name='SBL_%d1' % i, filters=num_filters, kernel_size=kernel_size, strides=1,padding='valid')
            conv_layer2 = Conv3D(name='SBL_%d2' % i, filters=1, kernel_size=kernel_size, strides=1,padding='valid')

            # include position features
            position_features_angle = tf.cast(tf.linspace(-1,1,G), tf.float32)
            position_features_AoA = tf.tile(tf.expand_dims(position_features_angle, axis=-1), (1, G))
            position_features_AoD = tf.transpose(position_features_AoA)
            position_features_angles = tf.concat([tf.expand_dims(position_features_AoA, axis=-1), tf.expand_dims(position_features_AoD, axis=-1)],
                                   axis=-1)
            # (G,G,num_sc,2)
            position_features_angles = tf.tile(tf.expand_dims(position_features_angles, axis=-2), (1, 1, num_sc, 1))

            position_features_sc = tf.cast(tf.linspace(-1,1,num_sc), tf.float32)
            # (G,G,num_sc,1)
            position_features_sc = tf.tile(tf.reshape(position_features_sc,(1,1,num_sc,1)),(G,G,1,1))

            # (G,G,num_sc,3)
            position_features = tf.concat([position_features_angles,position_features_sc],axis=-1)
            position_features = position_features * tf.ones_like(tf.tile(temp[:,:,:,:,:1],(1,1,1,1,3)))

            position_attention = Conv3D(filters=32,kernel_size=1,activation='relu')(position_features)
            # (G,G,num_sc,num_filters)
            position_attention = Conv3D(filters=num_filters,kernel_size=1,activation='sigmoid')(position_attention)

            temp = circular_padding_2d(temp, kernel_size=kernel_size, strides=1)
            temp = conv_layer1(temp)
            temp = temp*position_attention
            temp = tf.keras.activations.relu(temp)

            temp = circular_padding_2d(temp, kernel_size=kernel_size, strides=1)
            temp = conv_layer2(temp)
            alpha_list = tf.keras.activations.relu(temp)

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

epochs = 1000
batch_size = 16

# weight initialization
Kron2 = np.kron(np.conjugate(A_T),A_R)
Kron2 = np.expand_dims(Kron2,axis=-1)
A_R_0 = np.expand_dims(A_R,axis=-1)
A_T_0 = np.expand_dims(A_T.H,axis=-1)
init_weights_Kron2 = np.concatenate([np.real(Kron2),np.imag(Kron2)],axis=-1)
init_weights_R = np.concatenate([np.real(A_R_0), np.imag(A_R_0)],axis=-1)
init_weights_T = np.concatenate([np.real(A_T_0), np.imag(A_T_0)],axis=-1)

for mode in mode_list:
    model = SBL_net(Mt, Mr, Nt, Nr, G, num_sc, num_layers, num_filters, kernel_size, mode)
    model.summary()

    best_model_path = './models/best_Mr_%d_Mt_%d_SNR_%d_random_init_%s.h5'%(Mr,Mt,SNR,mode)

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
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=2, cooldown=1, verbose=1, mode='auto',
                                  min_delta=1e-5, min_lr=1e-5)
    early_stopping = EarlyStopping(monitor='val_loss', min_delta=1e-5, patience=3)

    model.compile(loss='mse', optimizer=Adam(lr=1e-4))

    loss_history = model.fit([W_list, F_list, y_list], H_list, epochs=epochs, batch_size=batch_size,
              verbose=1, shuffle=True, \
              validation_split=0.1, callbacks=[checkpointer, reduce_lr, early_stopping])

    # save loss history for plotting
    train_loss = np.squeeze(loss_history.history['loss'])
    val_loss = np.squeeze(loss_history.history['val_loss'])
    io.savemat('./results/loss_%s_%d_%d.mat'%(mode,Mr,Mt),{'train_loss':train_loss,'val_loss':val_loss})

    # test performance and save
    model.load_weights(best_model_path)

    test_num = int(0.1*data_num)
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
    file_handle.write('Random init, %s:\n'%mode)
    file_handle.write(str([mse,nmse]))
    file_handle.write('\n')
    file_handle.write('\n')
    file_handle.close()
