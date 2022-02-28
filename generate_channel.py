import numpy as np
np.random.seed(2022)

from scipy import io

import argparse


#%%
parser = argparse.ArgumentParser()

parser.add_argument('-data_num')

args = parser.parse_args()

data_num = int(args.data_num)

Nc = 3 # number of clusters
Np = 10 # number of sub-paths in a cluster
AS = 5 # angular spread of each cluster

Nt = 32 # number of antennas at the user
Nr = 32 # number of antennas at the BS

fc = 28*1e9 # central frequency
W = 16*1e9 # bandwidth
num_sc = 8 # number of subcarriers
Tau_max = 20*1e-9 # max delay

# parameters computed based on input parameters
eta = W / num_sc  # subcarrier spacing
Lp = Nc * Np # number of total paths


#%% Generate channel for single block case
Gains_list = np.sqrt(1/2)*(np.random.randn(data_num,Lp)+1j*np.random.randn(data_num,Lp))
Taus_list = np.random.uniform(0, 1, data_num * Nc) * Tau_max
Taus_list = np.reshape(Taus_list, (data_num, Nc))

H_list = np.zeros((data_num, num_sc, Nr, Nt)) + 1j * np.zeros((data_num, num_sc, Nr, Nt))

for i in range(data_num):
    if i % 500 == 0:
        print('%d/%d' % (i, data_num))
    AoAs = []
    AoDs = []
    for cluster in range(Nc):
        mean_AoA = np.random.uniform(0,360)
        mean_AoD = np.random.uniform(0,360)
        for sub_path in range(Np):
            AoA = np.random.uniform(mean_AoA-AS,mean_AoA+AS)/180*np.pi
            AoAs.append(AoA)
            AoD = np.random.uniform(mean_AoD-AS,mean_AoD+AS)/180*np.pi
            AoDs.append(AoD)
    normalized_AoAs = np.sin(AoAs) / 2
    normalized_AoDs = np.sin(AoDs) / 2

    for n in range(num_sc):
        f = n * eta
        H_subcarrier = np.zeros((Nr, Nt)) + 1j * np.zeros((Nr, Nt))
        for l in range(Lp):
            path_gain = Gains_list[i, l]
            tau = Taus_list[i, l // Np] # sub-paths in a cluster share a common delay
            normalized_AoA = normalized_AoAs[l]
            normalized_AoD = normalized_AoDs[l]
            # frequency dependent steering vectors with beam squint effect
            a_R = np.exp(-1j * 2 * np.pi * np.arange(Nr) * (1 + f / fc) * normalized_AoA)
            a_T = np.exp(-1j * 2 * np.pi * np.arange(Nt) * (1 + f / fc) * normalized_AoD)
            a_R = np.expand_dims(a_R, axis=-1)
            a_T = np.expand_dims(a_T, axis=0)
            H_subcarrier = H_subcarrier + path_gain * a_R.dot(np.conjugate(a_T)) \
                           * np.exp(-1j * 2 * np.pi * f * tau) / np.sqrt(Lp)
        H_list[i, n] = H_subcarrier

H_list = np.expand_dims(H_list, axis=-1)
H_list = np.concatenate([np.real(H_list), np.imag(H_list)], axis=-1)
# put the dimension num_sc after Nt and Nr, to facilitate computation later
H_list = np.transpose(H_list, (0, 2, 3, 1, 4))
io.savemat('./data/channel_large_squint.mat',{'H_list':H_list})
print(H_list.shape)  # (data_num,Nr,Nt,num_sc,2)


