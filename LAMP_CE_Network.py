import tensorflow.compat.v1 as tf
# tf.disable_eager_execution()
tf.disable_v2_behavior()

import numpy as np
import myshrinkage
import sys
from scipy import io
import math
import time
pi = math.pi

def build_LAMP(Mr, Nr, Mt, Nt, G, K, snr, T, shrink, untied, initialT, initialR, initialU):
    eta, theta_init = myshrinkage.get_shrinkage_function(shrink)
    layers = []
    var_all = []
    OneOverNr = tf.constant(float(1) / Nr, dtype=tf.float64)
    OneOverNt = tf.constant(float(1) / Nt, dtype=tf.float64)
    Athetar = tf.random_uniform(shape=(Nr, Mr), minval=0, maxval=2*pi, dtype=tf.float64)
    Athetar_ = tf.Variable(initial_value=Athetar, name='Athetar_' + str(snr) + '_0')
    var_all.append(Athetar_)
    Arealr = tf.multiply(tf.cos(Athetar_), tf.sqrt(OneOverNr))
    Aimagr = tf.multiply(tf.sin(Athetar_), tf.sqrt(OneOverNr))
    Ar_ = tf.complex(Arealr, Aimagr, name='Ar') # W

    Athetat = tf.random_uniform(shape=(Nt, Mt), minval=0, maxval=2*pi, dtype=tf.float64)
    Athetat_ = tf.Variable(initial_value=Athetat, name='Athetat_' + str(snr) + '_0')
    var_all.append(Athetat_)
    Arealt = tf.multiply(tf.cos(Athetat_), tf.sqrt(OneOverNt))
    Aimagt = tf.multiply(tf.sin(Athetat_), tf.sqrt(OneOverNt))
    At_ = tf.complex(Arealt, Aimagt, name='At') # F

    # (Mr*Mt,Nr*Nt)
    Q = tf.linalg.LinearOperatorKronecker \
        ([tf.linalg.LinearOperatorFullMatrix(tf.transpose(At_)), \
          tf.linalg.LinearOperatorFullMatrix(tf.transpose(Ar_, conjugate=True))]).to_dense()

    # input original complex channel matrix here, (?,Nr,Nt,K)
    h_ = tf.placeholder(tf.complex128, (None, Nr, Nt, K))

    # add noise
    noise_var = tf.cast(1/(10**(snr/10)), tf.complex128)
    original_noise = tf.complex(real=tf.random_normal(shape=(Nr,Mt*K*tf.shape(h_)[0]), dtype=tf.float64),
                       imag=tf.random_normal(shape=(Nr,Mt*K*tf.shape(h_)[0]), dtype=tf.float64))
    original_noise = tf.sqrt(noise_var/2) * original_noise
    effective_noise = tf.matmul(tf.transpose(Ar_,conjugate=True),original_noise) # (Mr,Mt*K*?)
    effective_noise = tf.reshape(effective_noise,(Mr,Mt,K*tf.shape(h_)[0])) # (Mr,Mt,?*K)
    effective_noise = tf.transpose(effective_noise,(1,0,2)) # (Mt,Mr,?*K)
    effective_noise = tf.reshape(effective_noise,(Mt*Mr,-1)) # (Mt*Mr,?*K)

    h1 = tf.transpose(h_,[0,3,2,1]) # (?,K,Nt,Nr)
    h1 = tf.reshape(h1,(-1,Nt*Nr)) # (?*K,Nt*Nr)
    ytemp1 = tf.matmul(Q,tf.transpose(h1)) # (Mt*Mr,?*K)
    ytemp1 = ytemp1 + effective_noise

    ytemp1 = tf.reshape(ytemp1, (Mr*Mt, tf.shape(h_)[0], K))
    ytemp_ = tf.transpose(ytemp1, [1, 0, 2]) # (?,Mr*Mt,K)

    y_ = ytemp_
    # first layer 初始化v0=0，h0=0，故v1=y
    v_ = y_  # 残差为y

    OneOverMK = tf.constant(float(1) / (Mr*Mt*K), dtype=tf.float64)
    rvar_ = OneOverMK * tf.expand_dims(tf.square(tf.norm(tf.abs(v_), axis=[1, 2])), 1)

    # redundant DFT matrix for both Tx and Rx
    DictT_real = tf.Variable(initial_value=np.real(initialT),trainable=False)
    DictT_imag = tf.Variable(initial_value=np.imag(initialT), trainable=False)
    DictR_real = tf.Variable(initial_value=np.real(initialR),trainable=False)
    DictR_imag = tf.Variable(initial_value=np.imag(initialR), trainable=False)
    U_real = tf.Variable(initial_value=np.real(initialU),trainable=False)
    U_imag = tf.Variable(initial_value=np.imag(initialU), trainable=False)
    DictT = tf.complex(DictT_real, DictT_imag)
    DictR = tf.complex(DictR_real, DictR_imag)
    U_ = tf.complex(U_real, U_imag)

    # equivalent measurment matrix for sparse angular-frequency domain channel
    Phi = tf.matmul(Q,U_)

    # Initialize B as Phi^H
    B = tf.transpose(Phi,conjugate=True)
    Breal_ = tf.Variable(tf.real(B), name='Breal_' + str(snr) + '_1')
    var_all.append(Breal_)
    Bimag_ = tf.Variable(tf.imag(B), name='Bimag_' + str(snr) + '_1')
    var_all.append(Bimag_)
    B_ = tf.complex(Breal_, Bimag_, name='B')
    v1 = tf.transpose(v_, [1, 0, 2])
    v1 = tf.reshape(v1, (Mr*Mt, -1))
    Bvtemp = tf.matmul(B_, v1)
    Bv = tf.reshape(Bvtemp, (G**2, tf.shape(h_)[0], K))
    Bv_ = tf.transpose(Bv, [1, 0, 2])
    theta_ = tf.Variable(theta_init, dtype=tf.float64, name='theta_' + str(snr) + '_1')
    var_all.append(theta_)

    xhat_, dxdr_ = eta(Bv_, rvar_, K, theta_)
    GOverM = tf.constant(float(G**2) / (Mr*Mt), dtype=tf.complex128)
    layers.append(('LAMP-{0} T=1'.format(shrink), xhat_, tuple(var_all), tuple(var_all), (0,)))

    for t in range(2, T+1):
        b_ = tf.expand_dims(GOverM * dxdr_, 1)
        x2 = tf.transpose(xhat_, [1, 0, 2])
        x3 = tf.reshape(x2, (G**2, -1))
        Axhat = tf.matmul(Phi, x3)
        Axhat = tf.reshape(Axhat, (Mr*Mt, tf.shape(h_)[0], K))
        Axhat_ = tf.transpose(Axhat, [1, 0, 2])
        v_ = tf.reshape(v_, [tf.shape(h_)[0], Mr*Mt*K])
        bv = tf.multiply(b_, v_)
        bv_ = tf.reshape(bv, [tf.shape(h_)[0], Mr*Mt, K])
        v_ = y_ - Axhat_ + bv_
        rvar_ = OneOverMK * tf.expand_dims(tf.square(tf.norm(tf.abs(v_), axis=[1, 2])), 1)

        if untied:  # 表明每一层的B都会训练
            Breal_ = tf.Variable(tf.real(B), name='Breal_' + str(snr) + '_' + str(t))
            var_all.append(Breal_)
            Bimag_ = tf.Variable(tf.imag(B), name='Bimag_' + str(snr) + '_' + str(t))
            var_all.append(Bimag_)
            B_ = tf.complex(Breal_, Bimag_, name='B')
            Bv_ = tf.matmul(B_, v_)
            rhat_ = xhat_ + Bv_
        else:
            v3 = tf.transpose(v_, [1, 0, 2])
            v4 = tf.reshape(v3, (Mr*Mt, -1))
            Bv = tf.matmul(B_, v4)
            Bv = tf.reshape(Bv, (G**2, tf.shape(h_)[0], K))
            Bv_ = tf.transpose(Bv, [1, 0, 2])
            rhat_ = xhat_ + Bv_

        xhat_, dxdr_ = eta(rhat_, rvar_, K, theta_)
        layers.append(('LAMP-{0} T={1}'.format(shrink, t), xhat_, tuple(var_all), tuple(var_all), (0,)))

    return layers, h_, DictT, DictR


def setup_training(layers, h_, DictT, DictR, G, K, Nr, Nt, trinit=1e-3, refinements=(.5, .1, .01)):
    training_stages = []
    for name, xhat_, var_list, var_all, flag in layers:
        # recover the original domain channel with angular-frequency channel and redundant DFT matrices
        xhat1 = tf.reshape(xhat_,(-1,G,G,K)) # (?,G,G,K)
        xhat1 = tf.transpose(xhat1, [2,1,0,3]) # (G,G,?,K)
        xhat2 = tf.reshape(xhat1, (G, -1)) #(G,G*?*K)
        hhat1 = tf.matmul(DictR, xhat2) # (Nr,G*?*K)
        hhat1 = tf.reshape(hhat1,(Nr,G,tf.shape(h_)[0],K))  # (Nr,G,?,K)
        hhat1 = tf.transpose(hhat1,(2,3,0,1)) # (?,K,Nr,G)
        hhat1 = tf.reshape(hhat1,(-1,G)) # (?*K*Nr,G)
        hhat2 = tf.matmul(hhat1,tf.transpose(DictT,conjugate=True)) # (?*K*Nr,Nt)
        hhat2 = tf.reshape(hhat2, (-1, K, Nr, Nt))
        hhat2 = tf.transpose(hhat2,(0,2,3,1)) # the same shape as h_, (?,Nr,Nt,K)
        hhat_ = tf.reshape(hhat2,(-1,Nr*Nt*K))
        h = tf.reshape(h_,(-1,Nr*Nt*K))
        nmse_ = tf.reduce_mean(
            tf.square(tf.norm(tf.abs(hhat_ - h), axis=-1)) / tf.square(tf.norm(tf.abs(h), axis=-1)))
        loss_ = nmse_
        #print(var_list)
        if var_list is not None:
            if flag == (0,):
                train_ = tf.train.AdamOptimizer(trinit).minimize(loss_, var_list=var_list)
                training_stages.append((name, hhat_, loss_, nmse_, train_, var_list, var_all, flag))
            elif flag == (1,):
                train_ = tf.train.AdamOptimizer(trinit).minimize(loss_, var_list=var_list)
                training_stages.append((name, hhat_, loss_, nmse_, train_, var_list, var_all, flag))
            else:
                train_ = tf.train.AdamOptimizer(trinit).minimize(loss_, var_list=var_list)
                training_stages.append((name, hhat_, loss_, nmse_, train_, var_list, var_all, flag))
        index = 0
        for fm in refinements:
            train2_ = tf.train.AdamOptimizer(fm * trinit).minimize(loss_, var_list=var_all)
            training_stages.append((name + ' trainrate=' + str(index), hhat_, loss_, nmse_, train2_, (), var_all, flag))
            index = index + 1

    return training_stages


def load_trainable_vars(sess, filename):
    other = {}
    try:
        variables = tf.trainable_variables()
        tv = dict([(str(v.name).replace(':', '_'), v) for v in variables])
        for k, d in io.loadmat(filename).items():  # (k, d)表示字典中的(键，值)
            if k in tv:
                print('restore ' + k)
                sess.run(tf.assign(tv[k], d))
            else:
                if k == 'done':
                    for i in range(0, len(d)):
                        a = d[i]
                        d[i] = a.strip()
                other[k] = d
    except IOError:
        pass
    return other


def save_trainable_vars(sess, filename, snr, **kwargs):
    save = {}
    for v in tf.trainable_variables():
        if str(v.name).split('_')[1] == str(snr):
            save[str(v.name).replace(':', '_')] = sess.run(v)
        continue
    save.update(kwargs)
    io.savemat(filename, save)

def assign_trainable_vars(sess, var_list, var_list_old):
    for i in range(len(var_list)):
        temp = sess.run(var_list_old[i])
        # print(temp)
        sess.run(tf.assign(var_list[i], temp))


def do_training(h_, training_stages, savefile, snr, batch_size, early_stop_patience=3):
    training_size = 9000
    ######### Prepare data for dict feeding
    H_list = io.loadmat('./channel.mat')['H_list'] # (data_num,Nr,Nt,num_sc)
    h_train = H_list[:training_size]
    h_val = H_list[training_size:]
    print(h_train.shape)
    print(h_val.shape)
    validation_size = len(h_val)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)

    sess.run(tf.global_variables_initializer())
    print('Weights initialized\n')

    state = load_trainable_vars(sess, savefile)
    done = state.get('done', [])
    log = state.get('log', [])
    layernmse = state.get('layernmse', [])

    var_list_old0 = ()  # B
    var_list_old1 = ()  # theta
    var_list_old2 = ()  # Atheta
    nmse = None
    stage_count = 0
    for name, xhat_, loss_, nmse_, train_, var_list, var_all, flag in training_stages:
        stage_count = stage_count + 1
        print('############## Training stage %d/%d ################'%(stage_count,len(training_stages)))
        if name in done:
            if name == 'LAMP-gm linear T=5':
                var_list_old0 = var_list
            if name == 'LAMP-gm non-linear T=4':
                var_list_old1 = var_list
            print('Already did  ' + name + ' skipping.')
            continue
        if len(var_list):
            print('')
            print(name + ' ' + 'extending ' + ','.join([v.name for v in var_list]))
            if flag == (0,):  # if linear operation
                if nmse is not None:
                    layernmse = np.append(layernmse, nmse)
                    print(layernmse)
                if len(var_list_old0):
                    # Initialize the training variable to the value of that in previous layer
                    assign_trainable_vars(sess, var_list, var_list_old0)
                var_list_old0 = var_list
            elif flag == (1,):
                if len(var_list_old1):
                    assign_trainable_vars(sess, var_list, var_list_old1)
                var_list_old1 = var_list
            else:
                if len(var_list_old2):
                    assign_trainable_vars(sess, var_list, var_list_old2)
                var_list_old2 = var_list
        else:
            print('')
            print(name + ' ' + 'fine tuning all ' + ','.join([v.name for v in var_all]))

        # do validation per epoch
        validation_space = training_size//batch_size # 每隔多少次进行一次validation
        validation_iters = validation_size // batch_size # 每次validation需多少个iteration
        nmse_history = []
        epoch = 0
        start_time = time.time()
        for i in range(100000):
            # validation
            if i % validation_space == 0:
                nmse = 0
                for j in range(validation_iters):
                    batch_index = np.arange(j*batch_size,(j+1)*batch_size)
                    nmse = nmse + sess.run(nmse_, feed_dict={h_: h_val[batch_index]})  # validation results
                nmse = nmse/validation_iters
                nmse = round(nmse, 5)
                if np.isnan(nmse):
                    raise RuntimeError('nmse is Nan')
                nmse_history.append(nmse)
                nmsebest = np.min(nmse_history)
                sys.stdout.write(
                    '\rEpoch={epoch:<3d} validation nmse={nmse:.6f} (best nmse={best:.6f})\n'.format(epoch=epoch, nmse=nmse, best=nmsebest))
                sys.stdout.flush()
                epoch = epoch + 1
                end_time = time.time()
                print('Time of this epoch: %d seconds'%(end_time-start_time))
                start_time = time.time()
                # early stopping
                no_improvement_epochs = len(nmse_history) - np.argmin(nmse_history) - 1
                if no_improvement_epochs >= early_stop_patience:
                    break
            # training
            rand_index = np.random.choice(training_size, size=batch_size)
            sess.run(train_, feed_dict={h_: h_train[rand_index]})

        done = np.append(done, name)
        result_log = str('{name} nmse={nmse:.6f} in {i} iterations'.format(name=name, nmse=nmse, i=i))
        log = np.append(log, result_log)

        state['done'] = done
        state['log'] = log
        state['layernmse'] = layernmse

        save_trainable_vars(sess, savefile, snr=snr, **state)
    if nmse is None:
        layernmse = layernmse
    else:
        layernmse = np.append(layernmse, nmse)
    print('')
    print(layernmse)
    state['layernmse'] = layernmse
    save_trainable_vars(sess, savefile, snr=snr, **state)
    return sess
