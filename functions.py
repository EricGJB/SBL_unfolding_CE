import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input,Conv3D,Reshape,Lambda,Cropping3D,Cropping2D,ZeroPadding3D,Concatenate
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
from tensorflow.keras import backend as K


def dictionary(N, G):
    A = np.zeros((N, G)) + 1j * np.zeros((N, G))
    # 矩阵各列将角度sin值在[-1,1]之间进行 M 等分
    count = 0
    for sin_value in np.linspace(-1 + 1 / G, 1 - 1 / G, G):
        A[:, count] = np.exp(-1j * np.pi * np.arange(N) * sin_value) / np.sqrt(N)
        count = count + 1
    return A


class A_R_Layer(tf.keras.layers.Layer):
    def __init__(self, Nr, G):
        super(A_R_Layer, self).__init__()
        self.Nr = Nr
        self.G = G

    def build(self, input_shape):
        self.kernel = self.add_weight("kernel", shape=[self.Nr, self.G, 2], trainable=False)

    def call(self, input):
        input_real = input[:, :, :, 0]
        input_imag = input[:, :, :, 1]
        batch_zeros = input_real[:, 0:1, 0:1] - input_real[:, 0:1, 0:1]
        # obtain a (?,Nt,Nt*resolution) dimensional zero matrix
        batch_zeros = tf.tile(batch_zeros, (1, self.Nr, self.G))
        # the weights in the kernel is the B matrix
        B_real = self.kernel[:, :, 0] + batch_zeros
        B_imag = self.kernel[:, :, 1] + batch_zeros
        # obtain h = B.dot(x)
        RR = tf.matmul(B_real, input_real)
        RI = tf.matmul(B_real, input_imag)
        IR = tf.matmul(B_imag, input_real)
        II = tf.matmul(B_imag, input_imag)
        output_real = RR - II
        output_imag = RI + IR

        return tf.concat([tf.expand_dims(output_real, axis=-1), tf.expand_dims(output_imag, axis=-1)], axis=-1)


class A_T_Layer(tf.keras.layers.Layer):
    def __init__(self, Nt, G):
        super(A_T_Layer, self).__init__()
        self.Nt = Nt
        self.G = G

    def build(self, input_shape):
        self.kernel = self.add_weight("kernel", shape=[self.G, self.Nt, 2], trainable=False)

    def call(self, input):
        input_real = input[:, :, :, 0]
        input_imag = input[:, :, :, 1]
        batch_zeros = input_real[:, 0:1, 0:1] - input_real[:, 0:1, 0:1]
        # obtain a (?,Nt,Nt*resolution) dimensional zero matrix
        batch_zeros = tf.tile(batch_zeros, (1, self.G, self.Nt))
        # the weights in the kernel is the B matrix
        B_real = self.kernel[:, :, 0] + batch_zeros
        B_imag = self.kernel[:, :, 1] + batch_zeros
        # obtain A = W_rf.dot(B)
        RR = tf.matmul(input_real, B_real)
        RI = tf.matmul(input_real, B_imag)
        IR = tf.matmul(input_imag, B_real)
        II = tf.matmul(input_imag, B_imag)
        output_real = RR - II
        output_imag = RI + IR

        return tf.concat([tf.expand_dims(output_real, axis=-1), tf.expand_dims(output_imag, axis=-1)], axis=-1)


# circular padding function
def circular_padding_2d(x, kernel_size, strides):
    in_height = int(x.get_shape()[1])
    in_width = int(x.get_shape()[2])
    pad_along_height = max(kernel_size - strides, 0)
    pad_along_width = max(kernel_size - strides, 0)
    pad_along_depth = max(kernel_size - strides, 0)

    pad_top = pad_along_height // 2
    pad_bottom = pad_along_height - pad_top
    pad_left = pad_along_width // 2
    pad_right = pad_along_width - pad_left
    pad_shallow = pad_along_depth // 2
    pad_deep = pad_along_depth - pad_shallow

    # top and bottom side padding
    pad_top = Cropping3D(cropping=((in_height - pad_top, 0), (0, 0), (0, 0)))(x)
    pad_bottom = Cropping3D(cropping=((0, in_height - pad_bottom), (0, 0), (0, 0)))(x)
    # add padding to incoming image
    conc = Concatenate(axis=1)([pad_top, x, pad_bottom])

    # top and bottom side padding
    pad_left = Cropping3D(cropping=((0, 0), (in_width - pad_left, 0), (0, 0)))(conc)
    pad_right = Cropping3D(cropping=((0, 0), (0, in_width - pad_right), (0, 0)))(conc)
    # add padding to incoming image
    conc = Concatenate(axis=2)([pad_left, conc, pad_right])

    # zero padding for the third dimension, i.e., subcarrier
    conc = ZeroPadding3D(((0, 0), (0, 0), (pad_shallow, pad_deep)))(conc)

    return conc


def circular_padding_2D(x, kernel_size, strides):
    in_height = int(x.get_shape()[1])
    in_width = int(x.get_shape()[2])
    pad_along_height = max(kernel_size - strides, 0)
    pad_along_width = max(kernel_size - strides, 0)

    pad_top = pad_along_height // 2
    pad_bottom = pad_along_height - pad_top
    pad_left = pad_along_width // 2
    pad_right = pad_along_width - pad_left

    # top and bottom side padding
    pad_top = Cropping3D(cropping=((in_height - pad_top, 0), (0, 0), (0, 0)))(x)
    pad_bottom = Cropping3D(cropping=((0, in_height - pad_bottom), (0, 0), (0, 0)))(x)
    # add padding to incoming image
    conc = Concatenate(axis=1)([pad_top, x, pad_bottom])

    # top and bottom side padding
    pad_left = Cropping3D(cropping=((0, 0), (in_width - pad_left, 0), (0, 0)))(conc)
    pad_right = Cropping3D(cropping=((0, 0), (0, in_width - pad_right), (0, 0)))(conc)
    # add padding to incoming image
    conc = Concatenate(axis=2)([pad_left, conc, pad_right])

    return conc


def circular_padding_single_sc(x, kernel_size, strides):
    in_height = int(x.get_shape()[1])
    in_width = int(x.get_shape()[2])
    pad_along_height = max(kernel_size - strides, 0)
    pad_along_width = max(kernel_size - strides, 0)

    pad_top = pad_along_height // 2
    pad_bottom = pad_along_height - pad_top
    pad_left = pad_along_width // 2
    pad_right = pad_along_width - pad_left

    # top and bottom side padding
    pad_top = Cropping2D(cropping=((in_height - pad_top, 0), (0, 0)))(x)
    pad_bottom = Cropping2D(cropping=((0, in_height - pad_bottom), (0, 0)))(x)
    # add padding to incoming image
    conc = Concatenate(axis=1)([pad_top, x, pad_bottom])

    # top and bottom side padding
    pad_left = Cropping2D(cropping=((0, 0), (in_width - pad_left, 0)))(conc)
    pad_right = Cropping2D(cropping=((0, 0), (0, in_width - pad_right)))(conc)
    # add padding to incoming image
    conc = Concatenate(axis=2)([pad_left, conc, pad_right])

    return conc



def update_mu_Sigma(inputs,num_sc,sigma_2,Mr,Mt):
    if len(inputs[0].shape)==4:
        Phi = tf.cast(inputs[0][:, :, :, 0], tf.complex64) + 1j * tf.cast(inputs[0][:, :, :, 1], tf.complex64)
    if len(inputs[0].shape)==3:
        Phi = tf.cast(inputs[0][:, :, 0], tf.complex64) + 1j * tf.cast(inputs[0][:, :, 1], tf.complex64)
    y_list = tf.cast(inputs[1][:, :, :, 0], tf.complex64) + 1j * tf.cast(inputs[1][:, :, :, 1], tf.complex64)
    alpha_list = tf.cast(inputs[2], tf.complex64)

    mu_real_list = []
    mu_imag_list = []
    diag_Sigma_real_list = []
    for i in range(num_sc):
        y = y_list[:, :, i:i + 1]
        if len(Phi.shape)==3:
            Rx_PhiH = tf.multiply(alpha_list[:, :, i:i + 1], tf.transpose(Phi, (0, 2, 1), conjugate=True))
        if len(Phi.shape)==2:
            Rx_PhiH = tf.multiply(alpha_list[:, :, i:i + 1], tf.transpose(Phi, conjugate=True))
        inv = tf.linalg.inv(
            tf.matmul(Phi, Rx_PhiH) + sigma_2 * tf.eye(Mr * Mt, dtype=tf.complex64))
        z = tf.matmul(Rx_PhiH, inv)
        mu = tf.matmul(z, y)
        diag_Sigma = alpha_list[:, :, i] - tf.reduce_sum(tf.multiply(z, tf.math.conj(Rx_PhiH)), axis=-1)
        # return the updated parameters
        mu_real_list.append(tf.math.real(mu))
        mu_imag_list.append(tf.math.imag(mu))
        diag_Sigma_real_list.append(tf.expand_dims(tf.math.real(diag_Sigma), axis=-1))

    mu_real_list = tf.concat(mu_real_list, axis=-1)
    mu_imag_list = tf.concat(mu_imag_list, axis=-1)
    diag_Sigma_real_list = tf.concat(diag_Sigma_real_list, axis=-1)

    return mu_real_list, mu_imag_list, diag_Sigma_real_list


def update_mu_Sigma_mixed_SNR(inputs,num_sc,Mr,Mt):
    if len(inputs[0].shape)==4:
        Phi = tf.cast(inputs[0][:, :, :, 0], tf.complex64) + 1j * tf.cast(inputs[0][:, :, :, 1], tf.complex64)
    if len(inputs[0].shape)==3:
        Phi = tf.cast(inputs[0][:, :, 0], tf.complex64) + 1j * tf.cast(inputs[0][:, :, 1], tf.complex64)
    y_list = tf.cast(inputs[1][:, :, :, 0], tf.complex64) + 1j * tf.cast(inputs[1][:, :, :, 1], tf.complex64)
    alpha_list = tf.cast(inputs[2], tf.complex64)
    sigma_2 = tf.cast(inputs[3],tf.complex64)
    sigma_2 = tf.reshape(sigma_2,(-1,1,1))

    mu_real_list = []
    mu_imag_list = []
    diag_Sigma_real_list = []
    for i in range(num_sc):
        y = y_list[:, :, i:i + 1]
        if len(Phi.shape)==3:
            Rx_PhiH = tf.multiply(alpha_list[:, :, i:i + 1], tf.transpose(Phi, (0, 2, 1), conjugate=True))
        if len(Phi.shape)==2:
            Rx_PhiH = tf.multiply(alpha_list[:, :, i:i + 1], tf.transpose(Phi, conjugate=True))
        inv = tf.linalg.inv(
            tf.matmul(Phi, Rx_PhiH) + sigma_2 * tf.eye(Mr * Mt, dtype=tf.complex64))
        z = tf.matmul(Rx_PhiH, inv)
        mu = tf.matmul(z, y)
        diag_Sigma = alpha_list[:, :, i] - tf.reduce_sum(tf.multiply(z, tf.math.conj(Rx_PhiH)), axis=-1)
        # return the updated parameters
        mu_real_list.append(tf.math.real(mu))
        mu_imag_list.append(tf.math.imag(mu))
        diag_Sigma_real_list.append(tf.expand_dims(tf.math.real(diag_Sigma), axis=-1))

    mu_real_list = tf.concat(mu_real_list, axis=-1)
    mu_imag_list = tf.concat(mu_imag_list, axis=-1)
    diag_Sigma_real_list = tf.concat(diag_Sigma_real_list, axis=-1)

    return mu_real_list, mu_imag_list, diag_Sigma_real_list



class Phi_Layer(tf.keras.layers.Layer):
    def __init__(self, Nt, Nr, Mt, Mr, G):
        super(Phi_Layer, self).__init__()
        self.Nt = Nt
        self.Nr = Nr
        self.Mt = Mt
        self.Mr = Mr
        self.G = G

    def build(self, input_shape):
        self.kernel_Kron2 = self.add_weight("kernel_Kron2", shape=[self.Nt*self.Nr, (self.G)**2, 2], trainable=False)

    def call(self, input):
        W_real = input[0][:, :, :, 0]
        W_imag = input[0][:, :, :, 1]
        W = tf.cast(W_real, tf.complex64) + 1j * tf.cast(W_imag, tf.complex64)

        F_real = input[1][:, :, :, 0]
        F_imag = input[1][:, :, :, 1]
        F = tf.cast(F_real, tf.complex64) + 1j * tf.cast(F_imag, tf.complex64)

        Kron_1 = tf.linalg.LinearOperatorKronecker \
            ([tf.linalg.LinearOperatorFullMatrix(tf.transpose(F, (0, 2, 1))), \
              tf.linalg.LinearOperatorFullMatrix(tf.transpose(W, (0, 2, 1), conjugate=True))]).to_dense()

        Kron_2_real = self.kernel_Kron2[:, :, 0]
        Kron_2_imag = self.kernel_Kron2[:, :, 1]
        Kron_2 = tf.cast(Kron_2_real, tf.complex64) + 1j * tf.cast(Kron_2_imag, tf.complex64)

        Phi = tf.matmul(Kron_1, Kron_2)
        Phi = tf.expand_dims(Phi, -1)

        return tf.concat([tf.math.real(Phi), tf.math.imag(Phi)], axis=-1)


class Phi_Layer_joint_opt(tf.keras.layers.Layer):
    def __init__(self, Nt, Nr, Mt, Mr, G, N_r_RF, num_sc):
        super(Phi_Layer_joint_opt, self).__init__()
        self.Nt = Nt
        self.Nr = Nr
        self.Mt = Mt
        self.Mr = Mr
        self.G = G
        self.N_r_RF = N_r_RF
        self.num_sc = num_sc

    def build(self, input_shape):
        self.kernel_W = self.add_weight("kernel_W", shape=[self.Nr, self.Mr], trainable=True)
                                        #initializer=tf.random_uniform_initializer(0,2*math.pi))
        self.kernel_F = self.add_weight("kernel_F", shape=[self.Nt, self.Mt], trainable=True)
        self.kernel_Kron2 = self.add_weight("kernel_Kron2", shape=[self.Nt * self.Nr, (self.G) ** 2, 2],
                                            trainable=False)

    def call(self, input):
        W_phase = self.kernel_W
        W_real = tf.cos(W_phase)/tf.sqrt(1.0*self.Nr)
        W_imag = tf.sin(W_phase)/tf.sqrt(1.0*self.Nr)
        W = tf.cast(W_real, tf.complex64) + 1j * tf.cast(W_imag, tf.complex64)
        F_phase = self.kernel_F
        F_real = tf.cos(F_phase)/tf.sqrt(1.0*self.Nt)
        F_imag = tf.sin(F_phase)/tf.sqrt(1.0*self.Nt)
        F = tf.cast(F_real, tf.complex64) + 1j * tf.cast(F_imag, tf.complex64)

        Q = tf.linalg.LinearOperatorKronecker \
            ([tf.linalg.LinearOperatorFullMatrix(tf.transpose(F)), \
              tf.linalg.LinearOperatorFullMatrix(tf.transpose(W,conjugate=True))]).to_dense()

        Kron_2_real = self.kernel_Kron2[:, :, 0]
        Kron_2_imag = self.kernel_Kron2[:, :, 1]
        Kron_2 = tf.cast(Kron_2_real, tf.complex64) + 1j * tf.cast(Kron_2_imag, tf.complex64)

        Phi = tf.matmul(Q, Kron_2)
        Phi = tf.expand_dims(Phi, -1)

        Phi_real_imag = tf.concat([tf.math.real(Phi), tf.math.imag(Phi)], axis=-1)

        # obtain effective noise
        original_noise_real_imag = input[1]
        noise_list = []
        original_noise_list = tf.cast(original_noise_real_imag[:, :, :, :, 0], tf.complex64) + \
                              1j * tf.cast(original_noise_real_imag[:, :, :, :, 1], tf.complex64)
        for r in range(self.Mr // self.N_r_RF):
            W_r = W[:, r * self.N_r_RF:(r + 1) * self.N_r_RF]
            original_noise = original_noise_list[:, r]
            noise_list.append(tf.matmul(tf.transpose(W_r,conjugate=True),original_noise))

        effective_noise_list = tf.concat(noise_list, axis=1)
        effective_noise_list = tf.reshape(effective_noise_list, (-1, self.Mr, self.Mt, self.num_sc))

        # obtain received signal
        H_real_imag = input[0]
        H = tf.cast(H_real_imag[:, :, :, :, 0], tf.complex64) + 1j * tf.cast(H_real_imag[:, :, :, :, 1], tf.complex64)
        y_list = []
        for i in range(self.num_sc):
            # vectorization
            H_subcarrier = tf.transpose(H[:, :, :, i], (0, 2, 1))  # (?,Nt,Nr)
            h_subcarrier = tf.reshape(H_subcarrier, (-1, self.Nr * self.Nt, 1))
            effective_noise = tf.transpose(effective_noise_list[:, :, :, i], (0, 2, 1))  # (?,Mt,Mr)
            effective_noise = tf.reshape(effective_noise, (-1, self.Mt * self.Mr, 1))
            y_list.append(tf.matmul(Q, h_subcarrier) + effective_noise)
        y_list = tf.concat(y_list, axis=-1)
        y_real_imag = tf.concat(
            [tf.expand_dims(tf.math.real(y_list), axis=-1), tf.expand_dims(tf.math.imag(y_list), axis=-1)], axis=-1)

        return Phi_real_imag, y_real_imag



class Phi_Layer_joint_opt_fixed(tf.keras.layers.Layer):
    def __init__(self, Nt, Nr, Mt, Mr, G, N_r_RF, num_sc):
        super(Phi_Layer_joint_opt_fixed, self).__init__()
        self.Nt = Nt
        self.Nr = Nr
        self.Mt = Mt
        self.Mr = Mr
        self.G = G
        self.N_r_RF = N_r_RF
        self.num_sc = num_sc

    def build(self, input_shape):
        self.kernel_W = self.add_weight("kernel_W", shape=[self.Nr, self.Mr], trainable=False)
                                        #initializer=tf.random_uniform_initializer(0,2*math.pi))
        self.kernel_F = self.add_weight("kernel_F", shape=[self.Nt, self.Mt], trainable=False)
        self.kernel_Kron2 = self.add_weight("kernel_Kron2", shape=[self.Nt * self.Nr, (self.G) ** 2, 2],
                                            trainable=False)

    def call(self, input):
        W_phase = self.kernel_W
        W_real = tf.cos(W_phase)/tf.sqrt(1.0*self.Nr)
        W_imag = tf.sin(W_phase)/tf.sqrt(1.0*self.Nr)
        W = tf.cast(W_real, tf.complex64) + 1j * tf.cast(W_imag, tf.complex64)
        F_phase = self.kernel_F
        F_real = tf.cos(F_phase)/tf.sqrt(1.0*self.Nt)
        F_imag = tf.sin(F_phase)/tf.sqrt(1.0*self.Nt)
        F = tf.cast(F_real, tf.complex64) + 1j * tf.cast(F_imag, tf.complex64)

        Q = tf.linalg.LinearOperatorKronecker \
            ([tf.linalg.LinearOperatorFullMatrix(tf.transpose(F)), \
              tf.linalg.LinearOperatorFullMatrix(tf.transpose(W,conjugate=True))]).to_dense()

        Kron_2_real = self.kernel_Kron2[:, :, 0]
        Kron_2_imag = self.kernel_Kron2[:, :, 1]
        Kron_2 = tf.cast(Kron_2_real, tf.complex64) + 1j * tf.cast(Kron_2_imag, tf.complex64)

        Phi = tf.matmul(Q, Kron_2)
        Phi = tf.expand_dims(Phi, -1)

        Phi_real_imag = tf.concat([tf.math.real(Phi), tf.math.imag(Phi)], axis=-1)

        # obtain effective noise
        original_noise_real_imag = input[1]
        noise_list = []
        original_noise_list = tf.cast(original_noise_real_imag[:, :, :, :, 0], tf.complex64) + \
                              1j * tf.cast(original_noise_real_imag[:, :, :, :, 1], tf.complex64)
        for r in range(self.Mr // self.N_r_RF):
            W_r = W[:, r * self.N_r_RF:(r + 1) * self.N_r_RF]
            original_noise = original_noise_list[:, r]
            noise_list.append(tf.matmul(tf.transpose(W_r,conjugate=True),original_noise))

        effective_noise_list = tf.concat(noise_list, axis=1)
        effective_noise_list = tf.reshape(effective_noise_list, (-1, self.Mr, self.Mt, self.num_sc))

        # obtain received signal
        H_real_imag = input[0]
        H = tf.cast(H_real_imag[:, :, :, :, 0], tf.complex64) + 1j * tf.cast(H_real_imag[:, :, :, :, 1], tf.complex64)
        y_list = []
        for i in range(self.num_sc):
            # vectorization
            H_subcarrier = tf.transpose(H[:, :, :, i], (0, 2, 1))  # (?,Nt,Nr)
            h_subcarrier = tf.reshape(H_subcarrier, (-1, self.Nr * self.Nt, 1))
            effective_noise = tf.transpose(effective_noise_list[:, :, :, i], (0, 2, 1))  # (?,Mt,Mr)
            effective_noise = tf.reshape(effective_noise, (-1, self.Mt * self.Mr, 1))
            y_list.append(tf.matmul(Q, h_subcarrier) + effective_noise)
        y_list = tf.concat(y_list, axis=-1)
        y_real_imag = tf.concat(
            [tf.expand_dims(tf.math.real(y_list), axis=-1), tf.expand_dims(tf.math.imag(y_list), axis=-1)], axis=-1)

        return Phi_real_imag, y_real_imag






import math
class Phi_Layer_joint_opt_single_sc(tf.keras.layers.Layer):
    def __init__(self, Nt, Nr, Mt, Mr, G, N_r_RF):
        super(Phi_Layer_joint_opt_single_sc, self).__init__()
        self.Nt = Nt
        self.Nr = Nr
        self.Mt = Mt
        self.Mr = Mr
        self.G = G
        self.N_r_RF = N_r_RF

    def build(self, input_shape):
        self.kernel_W = self.add_weight("kernel_W", shape=[self.Nr, self.Mr], trainable=True,\
                                        initializer=tf.random_uniform_initializer(0,2*math.pi))
        self.kernel_F = self.add_weight("kernel_F", shape=[self.Nt, self.Mt], trainable=True, \
                                        initializer=tf.random_uniform_initializer(0, 2 * math.pi))

    def call(self, input):
        W_phase = self.kernel_W
        W_real = tf.cos(W_phase)/tf.sqrt(1.0*self.Nr)
        W_imag = tf.sin(W_phase)/tf.sqrt(1.0*self.Nr)
        W = tf.cast(W_real, tf.complex64) + 1j * tf.cast(W_imag, tf.complex64)
        F_phase = self.kernel_F
        F_real = tf.cos(F_phase)/tf.sqrt(1.0*self.Nt)
        F_imag = tf.sin(F_phase)/tf.sqrt(1.0*self.Nt)
        F = tf.cast(F_real, tf.complex64) + 1j * tf.cast(F_imag, tf.complex64)

        # obtain effective noise
        original_noise_real_imag = input[1]
        noise_list = []
        original_noise_list = tf.cast(original_noise_real_imag[:, :, :, :, 0], tf.complex64) + \
                              1j * tf.cast(original_noise_real_imag[:, :, :, :, 1], tf.complex64)
        for r in range(self.Mr // self.N_r_RF):
            W_r = W[:, r * self.N_r_RF:(r + 1) * self.N_r_RF]
            original_noise = original_noise_list[:, r]
            noise_list.append(tf.matmul(tf.transpose(W_r,conjugate=True),original_noise))

        effective_noise = tf.concat(noise_list, axis=1)

        # obtain received signal
        H_real_imag = input[0]
        H = tf.cast(H_real_imag[:, :, :, 0], tf.complex64) + 1j * tf.cast(H_real_imag[:, :, :, 1], tf.complex64)

        Y = tf.matmul(tf.matmul(W, H, adjoint_a=True),F)+ effective_noise
        Y_real_imag = tf.concat([tf.math.real(Y),tf.math.imag(Y)], axis=-1)

        return Y_real_imag



class Phi_Layer_multipleT(tf.keras.layers.Layer):
    def __init__(self, Nt, Nr, Mt, Mr, G, N_r_RF, num_sc):
        super(Phi_Layer_multipleT, self).__init__()
        self.Nt = Nt
        self.Nr = Nr
        self.Mt = Mt
        self.Mr = Mr
        self.G = G
        self.N_r_RF = N_r_RF
        self.num_sc = num_sc

    def build(self, input_shape):
        self.kernel_F = self.add_weight("kernel_F", shape=[self.Nt, self.Mt], trainable=False)
        self.kernel_Kron2 = self.add_weight("kernel_Kron2", shape=[self.Nt * self.Nr, (self.G) ** 2, 2],
                                            trainable=False)

    def call(self, input):
        W_real = input[0][:, :, :, 0]
        W_imag = input[0][:, :, :, 1]
        W = tf.cast(W_real, tf.complex64) + 1j * tf.cast(W_imag, tf.complex64)

        F_phase = self.kernel_F
        F_real = tf.cos(F_phase)/tf.sqrt(1.0*self.Nt)
        F_imag = tf.sin(F_phase)/tf.sqrt(1.0*self.Nt)
        F = tf.cast(F_real, tf.complex64) + 1j * tf.cast(F_imag, tf.complex64)

        Q = tf.linalg.LinearOperatorKronecker \
            ([tf.linalg.LinearOperatorFullMatrix(tf.transpose(F)), \
              tf.linalg.LinearOperatorFullMatrix(tf.transpose(W,(0,2,1),conjugate=True))]).to_dense()

        Kron_2_real = self.kernel_Kron2[:, :, 0]
        Kron_2_imag = self.kernel_Kron2[:, :, 1]
        Kron_2 = tf.cast(Kron_2_real, tf.complex64) + 1j * tf.cast(Kron_2_imag, tf.complex64)

        Phi = tf.matmul(Q, Kron_2)
        Phi = tf.expand_dims(Phi, -1)

        Phi_real_imag = tf.concat([tf.math.real(Phi), tf.math.imag(Phi)], axis=-1)

        # obtain effective noise
        original_noise_real_imag = input[1]
        noise_list = []
        original_noise_list = tf.cast(original_noise_real_imag[:, :, :, :, 0], tf.complex64) + \
                              1j * tf.cast(original_noise_real_imag[:, :, :, :, 1], tf.complex64)
        for r in range(self.Mr // self.N_r_RF):
            W_r = W[:,:,r * self.N_r_RF:(r + 1) * self.N_r_RF]
            original_noise = original_noise_list[:, r]
            noise_list.append(tf.matmul(tf.transpose(W_r,(0,2,1),conjugate=True),original_noise))

        effective_noise_list = tf.concat(noise_list, axis=1)
        effective_noise_list = tf.reshape(effective_noise_list, (-1, self.Mr, self.Mt, self.num_sc))

        # obtain received signal
        H_real_imag = input[2]
        H = tf.cast(H_real_imag[:, :, :, :, 0], tf.complex64) + 1j * tf.cast(H_real_imag[:, :, :, :, 1], tf.complex64)
        y_list = []
        for i in range(self.num_sc):
            # vectorization
            H_subcarrier = tf.transpose(H[:, :, :, i], (0, 2, 1))  # (?,Nt,Nr)
            h_subcarrier = tf.reshape(H_subcarrier, (-1, self.Nr * self.Nt, 1))
            effective_noise = tf.transpose(effective_noise_list[:, :, :, i], (0, 2, 1))  # (?,Mt,Mr)
            effective_noise = tf.reshape(effective_noise, (-1, self.Mt * self.Mr, 1))
            y_list.append(tf.matmul(Q, h_subcarrier) + effective_noise)
        y_list = tf.concat(y_list, axis=-1)
        y_real_imag = tf.concat(
            [tf.expand_dims(tf.math.real(y_list), axis=-1), tf.expand_dims(tf.math.imag(y_list), axis=-1)], axis=-1)

        return Phi_real_imag, y_real_imag



#%% symmetric convolution functions
def symmetric_pre(feature_map,kernel_size):
    # split into four feature_maps
    G = feature_map.shape[1]+1-kernel_size
    sub_feature_map1 = feature_map[:,:G//2+kernel_size-1,:G//2+kernel_size-1]
    sub_feature_map2 = feature_map[:,:G//2+kernel_size-1,G//2:]
    sub_feature_map2 = K.reverse(sub_feature_map2,axes=2)
    sub_feature_map3 = feature_map[:,G//2:,:G//2+kernel_size-1]
    sub_feature_map3 = K.reverse(sub_feature_map3,axes=1)
    sub_feature_map4 = feature_map[:,G//2:,G//2:]
    sub_feature_map4 = K.reverse(sub_feature_map4, axes=[1,2])

    return sub_feature_map1,sub_feature_map2,sub_feature_map3,sub_feature_map4


def symmetric_post(sub_output1,sub_output2,sub_output3,sub_output4):
    output_upper = tf.concat([sub_output1,K.reverse(sub_output2,axes=2)],axis=2)
    output_lower = tf.concat([K.reverse(sub_output3,axes=1),K.reverse(sub_output4,axes=[1,2])],axis=2)
    output = tf.concat([output_upper,output_lower],axis=1)

    return output


#%% PC-SBL functions
def update_alpha_PC(input_list,G,num_sc,a,b,beta):
    mu_real,mu_imag,diag_Sigma_real = input_list
    mu_real = tf.reshape(mu_real,(-1,G,G,num_sc))
    mu_imag = tf.reshape(mu_imag,(-1,G,G,num_sc))
    diag_Sigma_real = tf.reshape(diag_Sigma_real,(-1,G,G,num_sc))
    mu_square = mu_real**2+mu_imag**2
    # expand the head and tail of two dimensions
    mu_square = tf.concat([tf.zeros_like(mu_square[:,:,0:1]),mu_square,tf.zeros_like(mu_square[:,:,0:1])],axis=2)
    mu_square = tf.concat([tf.zeros_like(mu_square[:,0:1]),mu_square,tf.zeros_like(mu_square[:,0:1])],axis=1)
    diag_Sigma_real = tf.concat([tf.zeros_like(diag_Sigma_real[:,:,0:1]),diag_Sigma_real,tf.zeros_like(diag_Sigma_real[:,:,0:1])],axis=2)
    diag_Sigma_real = tf.concat([tf.zeros_like(diag_Sigma_real[:,0:1]),diag_Sigma_real,tf.zeros_like(diag_Sigma_real[:,0:1])],axis=1)
    w_list = (mu_square[:,1:-1,1:-1]+diag_Sigma_real[:,1:-1,1:-1])+beta*(mu_square[:,1:-1,:-2]+diag_Sigma_real[:,1:-1,:-2])\
                +beta*(mu_square[:,1:-1,2:]+diag_Sigma_real[:,1:-1,2:])+beta*(mu_square[:,2:,1:-1]+diag_Sigma_real[:,2:,1:-1])\
                +beta*(mu_square[:,:-2,1:-1]+diag_Sigma_real[:,:-2,1:-1])
    w_list = tf.reshape(w_list,(-1,G*G,num_sc))
    alpha_list = (0.5*w_list+b)/a
    return alpha_list


def update_mu_Sigma_PC(inputs,G,num_sc,sigma_2,Mr,Mt,beta):
    Phi = tf.cast(inputs[0][:, :, :, 0], tf.complex64) + 1j * tf.cast(inputs[0][:, :, :, 1], tf.complex64)
    y_list = tf.cast(inputs[1][:, :, :, 0], tf.complex64) + 1j * tf.cast(inputs[1][:, :, :, 1], tf.complex64)

    alpha_list = tf.cast(inputs[2],tf.complex64)
    # 倒数
    alpha_list = 1/alpha_list
    alpha_list = tf.reshape(alpha_list,(-1,G,G,num_sc))
    # expand the head and tail of two dimensions
    alpha_list = tf.concat([tf.zeros_like(alpha_list[:,:,0:1]),alpha_list,tf.zeros_like(alpha_list[:,:,0:1])],axis=2)
    alpha_list = tf.concat([tf.zeros_like(alpha_list[:,0:1]),alpha_list,tf.zeros_like(alpha_list[:,0:1])],axis=1)
    # 错位加权相加
    alpha_list = (alpha_list[:,1:-1,1:-1]+beta*(alpha_list[:,1:-1,:-2]+alpha_list[:,1:-1,2:]+\
                                                alpha_list[:,:-2,1:-1]+alpha_list[:,2:,1:-1]))
    alpha_list = tf.reshape(alpha_list,(-1,G*G,num_sc))
    alpha_list = 1 / alpha_list

    mu_real_list = []
    mu_imag_list = []
    diag_Sigma_real_list = []
    for i in range(num_sc):
        y = y_list[:, :, i:i + 1]
        Rx_PhiH = tf.multiply(alpha_list[:, :, i:i + 1], tf.transpose(Phi, (0, 2, 1), conjugate=True))
        inv = tf.linalg.inv(
            tf.matmul(Phi, Rx_PhiH) + sigma_2 * tf.eye(Mr * Mt, dtype=tf.complex64))
        z = tf.matmul(Rx_PhiH, inv)
        mu = tf.matmul(z, y)
        diag_Sigma = alpha_list[:, :, i] - tf.reduce_sum(tf.multiply(z, tf.math.conj(Rx_PhiH)), axis=-1)
        # return the updated parameters
        mu_real_list.append(tf.math.real(mu))
        mu_imag_list.append(tf.math.imag(mu))
        diag_Sigma_real_list.append(tf.expand_dims(tf.math.real(diag_Sigma), axis=-1))

    mu_real_list = tf.concat(mu_real_list, axis=-1)
    mu_imag_list = tf.concat(mu_imag_list, axis=-1)
    diag_Sigma_real_list = tf.concat(diag_Sigma_real_list, axis=-1)

    return mu_real_list, mu_imag_list, diag_Sigma_real_list


def update_alpha_PC_M(input_list,G,num_sc,a,b,beta):
    mu_real,mu_imag,diag_Sigma_real = input_list
    mu_real = tf.reshape(mu_real,(-1,G,G,num_sc))
    mu_imag = tf.reshape(mu_imag,(-1,G,G,num_sc))
    diag_Sigma_real = tf.reshape(diag_Sigma_real,(-1,G,G,num_sc))
    mu_square = mu_real**2+mu_imag**2

    # averaging here is the only change
    mu_square = tf.reduce_mean(mu_square,axis=-1,keepdims=True)
    mu_square = tf.tile(mu_square,(1,1,1,num_sc))

    # expand the head and tail of two dimensions
    mu_square = tf.concat([tf.zeros_like(mu_square[:,:,0:1]),mu_square,tf.zeros_like(mu_square[:,:,0:1])],axis=2)
    mu_square = tf.concat([tf.zeros_like(mu_square[:,0:1]),mu_square,tf.zeros_like(mu_square[:,0:1])],axis=1)
    diag_Sigma_real = tf.concat([tf.zeros_like(diag_Sigma_real[:,:,0:1]),diag_Sigma_real,tf.zeros_like(diag_Sigma_real[:,:,0:1])],axis=2)
    diag_Sigma_real = tf.concat([tf.zeros_like(diag_Sigma_real[:,0:1]),diag_Sigma_real,tf.zeros_like(diag_Sigma_real[:,0:1])],axis=1)
    w_list = (mu_square[:,1:-1,1:-1]+diag_Sigma_real[:,1:-1,1:-1])+beta*(mu_square[:,1:-1,:-2]+diag_Sigma_real[:,1:-1,:-2])\
                +beta*(mu_square[:,1:-1,2:]+diag_Sigma_real[:,1:-1,2:])+beta*(mu_square[:,2:,1:-1]+diag_Sigma_real[:,2:,1:-1])\
                +beta*(mu_square[:,:-2,1:-1]+diag_Sigma_real[:,:-2,1:-1])
    w_list = tf.reshape(w_list,(-1,G*G,num_sc))
    alpha_list = (0.5*w_list+b)/a
    return alpha_list


def update_alpha_PC_high_order(input_list,G,num_sc,a,b,beta1,beta2):
    mu_real,mu_imag,diag_Sigma_real = input_list
    mu_real = tf.reshape(mu_real,(-1,G,G,num_sc))
    mu_imag = tf.reshape(mu_imag,(-1,G,G,num_sc))
    diag_Sigma_real = tf.reshape(diag_Sigma_real,(-1,G,G,num_sc))
    mu_square = mu_real**2+mu_imag**2
    # expand the head and tail of two dimensions
    mu_square = tf.concat([tf.zeros_like(mu_square[:,:,0:2]),mu_square,tf.zeros_like(mu_square[:,:,0:2])],axis=2)
    mu_square = tf.concat([tf.zeros_like(mu_square[:,0:2]),mu_square,tf.zeros_like(mu_square[:,0:2])],axis=1)
    diag_Sigma_real = tf.concat([tf.zeros_like(diag_Sigma_real[:,:,0:2]),diag_Sigma_real,tf.zeros_like(diag_Sigma_real[:,:,0:2])],axis=2)
    diag_Sigma_real = tf.concat([tf.zeros_like(diag_Sigma_real[:,0:2]),diag_Sigma_real,tf.zeros_like(diag_Sigma_real[:,0:2])],axis=1)
    w_list = (mu_square[:,2:-2,2:-2]+diag_Sigma_real[:,2:-2,2:-2])+beta1*(mu_square[:,2:-2,1:-3]+diag_Sigma_real[:,2:-2,1:-3])\
                +beta1*(mu_square[:,2:-2,3:-1]+diag_Sigma_real[:,2:-2,3:-1])+beta1*(mu_square[:,3:-1,2:-2]+diag_Sigma_real[:,3:-1,2:-2])\
                +beta1*(mu_square[:,1:-3,2:-2]+diag_Sigma_real[:,1:-3,2:-2])\
                +beta2*(mu_square[:,2:-2,:-4]+diag_Sigma_real[:,2:-2,:-4])\
                +beta2*(mu_square[:,2:-2,4:]+diag_Sigma_real[:,2:-2,4:])+beta2*(mu_square[:,4:,2:-2]+diag_Sigma_real[:,4:,2:-2])\
                +beta2*(mu_square[:,:-4,2:-2]+diag_Sigma_real[:,:-4,2:-2])
    w_list = tf.reshape(w_list,(-1,G*G,num_sc))
    alpha_list = (0.5*w_list+b)/a
    return alpha_list


def update_mu_Sigma_PC_high_order(inputs,G,num_sc,sigma_2,Mr,Mt,beta1,beta2):
    Phi = tf.cast(inputs[0][:, :, :, 0], tf.complex64) + 1j * tf.cast(inputs[0][:, :, :, 1], tf.complex64)
    y_list = tf.cast(inputs[1][:, :, :, 0], tf.complex64) + 1j * tf.cast(inputs[1][:, :, :, 1], tf.complex64)

    alpha_list = tf.cast(inputs[2],tf.complex64)
    # 倒数
    alpha_list = 1/alpha_list
    alpha_list = tf.reshape(alpha_list,(-1,G,G,num_sc))
    # expand the head and tail of two dimensions
    alpha_list = tf.concat([tf.zeros_like(alpha_list[:,:,0:2]),alpha_list,tf.zeros_like(alpha_list[:,:,0:2])],axis=2)
    alpha_list = tf.concat([tf.zeros_like(alpha_list[:,0:2]),alpha_list,tf.zeros_like(alpha_list[:,0:2])],axis=1)
    # 错位加权相加
    alpha_list = (alpha_list[:,2:-2,2:-2]+beta1*(alpha_list[:,2:-2,1:-3]+alpha_list[:,2:-2,3:-1]+\
                                                alpha_list[:,1:-3,2:-2]+alpha_list[:,3:-1,2:-2]) \
                                        +beta2 * (alpha_list[:, 2:-2, :-4] + alpha_list[:, 2:-2, 4:] + \
                                                alpha_list[:, :-4, 2:-2] + alpha_list[:, 4:, 2:-2]))
    alpha_list = tf.reshape(alpha_list,(-1,G*G,num_sc))
    alpha_list = 1 / alpha_list

    mu_real_list = []
    mu_imag_list = []
    diag_Sigma_real_list = []
    for i in range(num_sc):
        y = y_list[:, :, i:i + 1]
        Rx_PhiH = tf.multiply(alpha_list[:, :, i:i + 1], tf.transpose(Phi, (0, 2, 1), conjugate=True))
        inv = tf.linalg.inv(
            tf.matmul(Phi, Rx_PhiH) + sigma_2 * tf.eye(Mr * Mt, dtype=tf.complex64))
        z = tf.matmul(Rx_PhiH, inv)
        mu = tf.matmul(z, y)
        diag_Sigma = alpha_list[:, :, i] - tf.reduce_sum(tf.multiply(z, tf.math.conj(Rx_PhiH)), axis=-1)
        # return the updated parameters
        mu_real_list.append(tf.math.real(mu))
        mu_imag_list.append(tf.math.imag(mu))
        diag_Sigma_real_list.append(tf.expand_dims(tf.math.real(diag_Sigma), axis=-1))

    mu_real_list = tf.concat(mu_real_list, axis=-1)
    mu_imag_list = tf.concat(mu_imag_list, axis=-1)
    diag_Sigma_real_list = tf.concat(diag_Sigma_real_list, axis=-1)

    return mu_real_list, mu_imag_list, diag_Sigma_real_list


def update_mu_Sigma_2D(inputs,sigma_2,Mr,Mt):
    if len(inputs[0].shape)==4:
        Phi = tf.cast(inputs[0][:, :, :, 0], tf.complex64) + 1j * tf.cast(inputs[0][:, :, :, 1], tf.complex64)
    if len(inputs[0].shape)==3:
        Phi = tf.cast(inputs[0][:, :, 0], tf.complex64) + 1j * tf.cast(inputs[0][:, :, 1], tf.complex64)
    y = tf.cast(inputs[1][:, :, 0], tf.complex64) + 1j * tf.cast(inputs[1][:, :, 1], tf.complex64)
    alpha_list = tf.cast(inputs[2], tf.complex64)

    if len(Phi.shape)==3:
        Rx_PhiH = tf.multiply(alpha_list, tf.transpose(Phi, (0, 2, 1), conjugate=True))
    if len(Phi.shape)==2:
        Rx_PhiH = tf.multiply(alpha_list, tf.transpose(Phi, conjugate=True))
    inv = tf.linalg.inv(
        tf.matmul(Phi, Rx_PhiH) + sigma_2 * tf.eye(Mr * Mt, dtype=tf.complex64))
    z = tf.matmul(Rx_PhiH, inv)
    mu = tf.matmul(z, tf.expand_dims(y,axis=-1))
    diag_Sigma = alpha_list[:,:,0] - tf.reduce_sum(tf.multiply(z, tf.math.conj(Rx_PhiH)), axis=-1)

    # return the updated parameters
    mu_real = tf.math.real(mu)
    mu_imag = tf.math.imag(mu)
    diag_Sigma_real = tf.math.real(tf.expand_dims(diag_Sigma,axis=-1))

    return mu_real, mu_imag, diag_Sigma_real