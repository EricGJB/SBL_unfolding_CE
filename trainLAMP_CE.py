import LAMP_CE_Network
import os
import numpy as np
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def dictionary(N, G):
    A = np.zeros((N, G)) + 1j * np.zeros((N, G))
    count = 0
    for sin_value in np.linspace(-1 + 1 / G, 1 - 1 / G, G):
        A[:, count] = np.exp(-1j * np.pi * np.arange(N) * sin_value) / np.sqrt(N)
        count = count + 1
    return A

# beams
Mr = 16
Mt = 16
# antennas
Nr = 32
Nt = 32
MN = '16163232'
# grids
G = 64
# number of subcarriers
K = 8
# SNR in dB
SNRrange = [5]

A_T = dictionary(Nt, G)
A_T = np.matrix(A_T)
A_R = dictionary(Nr, G)
A_R = np.matrix(A_R)
Kron2 = np.kron(np.conjugate(A_T), A_R)

####### CS algorithm settings
# number of unfolding layers
T = 5
shrink = 'bg'

######## Training
for snr in SNRrange:
    savenetworkfilename = 'train_' + str(snr) + 'dB.mat'
    layers, h_, DictT, DictR = LAMP_CE_Network.build_LAMP(Mr=Mr, Nr=Nr, Mt=Mt, Nt=Nt, G=G, K=K, snr=snr, T=T, shrink=shrink, untied=False, initialT = A_T, initialR = A_R, initialU = Kron2)
    training_stages = LAMP_CE_Network.setup_training(layers, h_, DictT, DictR, G, K, Nr, Nt, trinit=1e-3, refinements=(0.1,))
    sess = LAMP_CE_Network.do_training(h_=h_,training_stages=training_stages,savefile=savenetworkfilename,snr=snr,batch_size=16)
