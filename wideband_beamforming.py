import numpy as np
np.random.seed(2022)

from scipy import io

num_antenna_bs = 32
num_antenna_ue = 32
num_sc = 8

num_rf = 4
num_stream = num_rf

rou_dB = 10 # beamforming SNR in dB
rou = 10**(rou_dB/10) # linear

# load testing channel samples
num_samples = 200
SNR_ce = 10 # channel estimation SNR in dB
dataset = io.loadmat('./ce_results_%ddB.mat'%SNR_ce)
H_list_true = dataset['H_list']
H_list_est = dataset['H_hat_list']
    
H_list_true = np.transpose(H_list_true,(0,3,2,1))
H_list_est = np.transpose(H_list_est,(0,3,2,1))

print(H_list_true.shape)  # (data_num, num_sc, num_antenna_ue, num_antenna_bs)
print(H_list_est.shape)  # (data_num, num_sc, num_antenna_ue, num_antenna_bs) 

# perfect CSI
H_list_est = H_list_true


#%% Fully digital beamforming, as upper bound, also the input of matrix decomposition based hybrid precoding algorithms
performance_list = np.zeros(num_samples)
FDBs_list = np.zeros((num_samples,num_sc,num_antenna_bs,num_stream))+1j*np.zeros((num_samples,num_sc,num_antenna_bs,num_stream))

for i in range(num_samples):
    if i % 50 == 0:
        print('Testing sample %d' % i)
    rate_subcarrier_list = []

    for n in range(num_sc):
        # SVD decomposition, rule in numpy: A = U * Sigma * V        
        H_subcarrier_true = H_list_true[i,n]
        H_subcarrier_est = H_list_est[i,n]  
        
        U, Sigma, V = np.linalg.svd(H_subcarrier_est)
            
        fully_digital_beamformer = np.transpose(np.conjugate(V))
        # no need for normalization since already satisfied
        fully_digital_beamformer = fully_digital_beamformer[:,:num_stream]

        eq_noise_0 = num_stream / (rou * Sigma[:num_stream] ** 2)
        eq_noise = num_stream / (rou * Sigma[:num_stream] ** 2)
        flag = 1
        num_subchannels = num_stream
        while flag:
            water_level = (num_stream + np.sum(eq_noise)) / num_subchannels
            if water_level > np.max(eq_noise):
                flag = 0
            else:
                eq_noise = eq_noise[:-1]
                num_subchannels = num_subchannels - 1
        pa_vector = np.maximum(water_level - eq_noise_0, 0)  # (1,num_stream)
        # print(pa_vector)
        fully_digital_beamformer = fully_digital_beamformer * np.sqrt(np.expand_dims(pa_vector, axis=0))
        # print(np.linalg.norm(fully_digital_beamformer)**2)

        # compute the rate, without combiner (won't affect the results)
        temp = H_subcarrier_true.dot(fully_digital_beamformer)
        temp = temp.dot(np.transpose(np.conjugate(temp)))
        rate_subcarrier = np.log2(np.linalg.det(np.eye(num_antenna_ue) + rou / num_stream * temp))
        rate_subcarrier_list.append(np.real(rate_subcarrier))

        FDBs_list[i, n] = fully_digital_beamformer

    performance_list[i] = np.mean(rate_subcarrier_list)

print('Performance of the FDB upper bound: %.4f\n'%np.mean(performance_list))
# print(np.sum(pa_vector))


#%% alternative optimization-based algorithm, based on Linglong dai's paper
def AM(initial_A, FDBs, num_sc, num_antenna_bs, num_stream, num_rf, num_iter):
    object_list = []
    
    A = np.copy(initial_A)
    
    FBBs = np.zeros((num_sc, num_rf, num_stream)) + 1j * np.zeros((num_sc, num_rf, num_stream))
    HBFs = np.zeros((num_sc, num_antenna_bs, num_stream)) + 1j * np.zeros((num_sc, num_antenna_bs, num_stream))
    
    for i in range(num_iter):
        
        #############################     
        FBBs = np.zeros((num_sc, num_rf, num_stream)) + 1j * np.zeros((num_sc, num_rf, num_stream))
        HBFs = np.zeros((num_sc, num_antenna_bs, num_stream)) + 1j * np.zeros((num_sc, num_antenna_bs, num_stream))
        term = 0
        for n in range(num_sc): 
            # update FBBs   
            Dn = np.linalg.pinv(A).dot(FDBs[n])
            FBBs[n] = Dn
            term = term + np.linalg.norm(Dn)**2 * (FDBs[n].dot(np.linalg.pinv(Dn)))
        # update FRF
        A = np.exp(1j * np.angle(term))
                
        for n in range(num_sc):
            HBF = A.dot(FBBs[n])
            # normalization
            HBF = HBF / np.linalg.norm(HBF) * np.sqrt(num_stream)
            HBFs[n] = HBF
            
        object_list.append(np.linalg.norm(HBFs - FDBs) ** 2 / np.product(np.shape(HBFs)))
        
    return A, HBFs, np.array(object_list)

num_iter = 20

performance_list_dll = []
for i in range(num_samples):
    if i % 50 == 0:
        print('Testing sample %d' % i)
    rate_subcarrier_list_dll = np.zeros(num_sc)

    random_A = np.random.uniform(-np.pi, np.pi, num_antenna_bs * num_rf)
    random_A = np.exp(1j * random_A)
    random_A = np.reshape(random_A, (num_antenna_bs, num_rf))
    
    A, HBFs, object_list_dll = AM(random_A, FDBs_list[i], num_sc, num_antenna_bs, num_stream, num_rf, num_iter)
     
    # SVD of equivalent channel with water-filling
    # eigen decomposition
    Sigma,U = np.linalg.eig(np.transpose(np.conjugate(A)).dot(A))
    eigen_values_normalizer = np.sqrt(1/np.real(Sigma))
    normalizer = (U.dot(np.diag(eigen_values_normalizer))).dot(np.transpose(np.conjugate(U)))
    
    sum_rate = 0
    for n in range(num_sc):
        H_subcarrier_est = H_list_est[i, n]
        H_subcarrier_true = H_list_true[i,n]

        H2_temp = H_subcarrier_est.dot(A)
        H2_temp = H2_temp.dot(normalizer)
        Ubb,Sigma,Vbb = np.linalg.svd(H2_temp)
        V = np.transpose(np.conjugate(Vbb))
        Fbb = V[:,:num_rf]

        eq_noise_0 = num_stream / (rou * Sigma[:num_stream] ** 2)
        eq_noise = num_stream / (rou * Sigma[:num_stream] ** 2)
        flag = 1
        num_subchannels = num_stream
        while flag:
            water_level = (num_stream + np.sum(eq_noise)) / num_subchannels
            if water_level > np.max(eq_noise):
                flag = 0
            else:
                eq_noise = eq_noise[:-1]
                num_subchannels = num_subchannels - 1
        pa_vector = np.maximum(water_level - eq_noise_0, 0)  # (1,num_stream)
        
        Fbb = Fbb * np.sqrt(np.expand_dims(pa_vector, axis=0))
        Fbb = normalizer.dot(Fbb)
        
        HBF = A.dot(Fbb)
        
        temp = H_subcarrier_true.dot(HBF)
        temp = temp.dot(np.transpose(np.conjugate(temp)))
        rate_subcarrier = np.log2(np.linalg.det(np.eye(num_antenna_ue) + rou / num_stream * temp))
        sum_rate = sum_rate + np.real(rate_subcarrier)

    performance_list_dll.append(sum_rate/num_sc) 
    
print('Performance of AM, with random initialization: %.4f\n' %np.mean(performance_list_dll))

