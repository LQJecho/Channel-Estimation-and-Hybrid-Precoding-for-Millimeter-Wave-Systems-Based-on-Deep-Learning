from tensorflow.python.keras import *
from tensorflow.python.keras import backend
import tensorflow as tf
import scipy.io as sio
from tensorflow.python.keras.layers import *
import h5py
import numpy as np
import math


# --------global parameters-------------
Nt = 64
Nr = 2
Nrf = 2
Ns = 2

# test snr range
test_snr_begin=-10
test_snr_end=2
test_snr_array = range(test_snr_begin, test_snr_end+1, 2)
# train snr range
snr_begin=-10
snr_end=10


def load_data_cehbfnet_v(load_trainset_flag, v, L=3, pnr=0):
    def load_from_dir(dir,v):
        feature = h5py.File(dir)
        H = feature['pcsi'][:]
        H = np.transpose(H, (2, 1, 0))
        H = H['real'] + 1j * H['imag']
        data = feature['data'][:]
        data = np.squeeze(data)
        data = np.transpose(data, (1, 0))
        data = data['real'] + 1j * data['imag']
        index_bs = feature['index_bs'][:]
        index_bs = np.transpose(index_bs, (1, 0))
        index_ms = feature['index_ms'][:]
        index_ms = np.transpose(index_ms, (1, 0))

        Nloop = np.int(H.shape[0])
        data = np.reshape(data, [Nloop, -1])
        index_bs = np.reshape(index_bs, [Nloop, -1])
        index_ms = np.reshape(index_ms, [Nloop, -1])
        Noise_power = 1 / np.power(10, np.random.randint(snr_begin, snr_end, [Nloop, 1]) / 10)

        if v==2:
            # Input only SNR, R and Isel in the last inner iteration of every outer iteration (labeled as ‘HCNet-V2’)
            data = np.concatenate([data[:, 54:60], data[:, 114:120], data[:, 174:180]], axis=1)
            data = np.concatenate([np.real(data), np.imag(data)], 1)
            index_bs = np.concatenate([index_bs[:, 12:15], index_bs[:, 27:30], index_bs[:, 42:45]], axis=1)
            index_ms=index_ms
        if v==3:
            # Input only SNR, R, and Isel in the first outer iteration (labeled as ‘HCNet-V3’)
            data=data[:,0:60]
            data = np.concatenate([np.real(data), np.imag(data)], 1)
            index_bs=index_bs[:,0:15]
            index_ms=index_ms[:,0]

        return Nloop, H, data, index_bs, index_ms, Noise_power
    if load_trainset_flag == 0:
        dir = f'../matlab/dataset_L{L}/snr{test_snr_begin}_{test_snr_end}/pnr{pnr}_test.mat'
        Nloop, H, data, index_bs, index_ms, Noise_power = load_from_dir(dir,v)
    if load_trainset_flag == 1:
        # -------------------train set----------------------------
        dir1 = f'../matlab/dataset_L{L}/pnr{pnr}_train1.mat'
        Nloop1, H1, data1, index_bs1, index_ms1, Noise_power1 = load_from_dir(dir1,v)
        dir2 = f'../matlab/dataset_L{L}/pnr{pnr}_train2.mat'
        Nloop2, H2, data2, index_bs2, index_ms2, Noise_power2 = load_from_dir(dir2,v)
        H = np.concatenate([H1, H2], axis=0)
        data = np.concatenate([data1, data2], axis=0)
        index_bs = np.concatenate([index_bs1, index_bs2], axis=0)
        index_ms = np.concatenate([index_ms1, index_ms2], axis=0)
        Noise_power = np.concatenate([Noise_power1, Noise_power2], axis=0)
        Nloop = Nloop1 + Nloop2
    H=H.astype(np.complex64)
    data=data.astype(np.float32)
    index_bs=index_bs.astype(np.float32)
    index_ms=index_ms.astype(np.float32)
    Noise_power=Noise_power.astype(np.float32)
    return Nloop, H, data, index_bs, index_ms, Noise_power

def load_data_from_dir_hbfnet(dir):
        Hest_flag = 'ecsi'
        feature = h5py.File(dir)
        H = feature['pcsi'][:]
        H = np.squeeze(H)
        H = np.transpose(H, (2, 1, 0))
        H = H['real'] + 1j * H['imag']
        Hest = feature[Hest_flag][:]
        Hest = Hest['real'] + 1j * Hest['imag']
        Hest = np.expand_dims(Hest, axis=3)
        Hest = np.concatenate([np.real(Hest), np.imag(Hest)], 3)
        Hest = np.transpose(Hest, (2, 0, 1, 3))
        Nloop = np.int(H.shape[0])
        Noise_power = 1 / np.power(10, np.random.randint(snr_begin, snr_end, [Nloop, 1]) / 10)
        return H, Hest, Noise_power, Nloop
def load_data_hbfnet(pnr, load_trainset_flag, big_trainset_flag=1, L=3):
    if load_trainset_flag == 0:
        dir = f'../matlab/dataset_L{L}/snr{test_snr_begin}_{test_snr_end}/pnr{pnr}_test.mat'
        H, Hest, Noise_power, Nloop = load_data_from_dir_hbfnet(dir)
    if load_trainset_flag == 1:
        if big_trainset_flag == 1:
            dir1 = f'../matlab/dataset_L{L}/pnr{pnr}_train1.mat'
            H1, Hest1, Noise_power1, Nloop1 = load_data_from_dir_hbfnet(dir1)
            dir2 = f'../matlab/dataset_L{L}/pnr{pnr}_train2.mat'
            H2, Hest2, Noise_power2, Nloop2 = load_data_from_dir_hbfnet(dir2)
            H = np.concatenate([H1, H2], axis=0)
            Hest = np.concatenate([Hest1, Hest2], axis=0)
            Noise_power = np.concatenate([Noise_power1, Noise_power2], axis=0)
            Nloop = Nloop1 + Nloop2
        else:
            dir = f'../matlab/dataset_L{L}/pnr{pnr}_train1.mat'
            H, Hest, Noise_power, Nloop = load_data_from_dir_hbfnet(dir)
    H=H.astype(np.complex64)
    Noise_power=Noise_power.astype(np.float32)
    return H, Hest, Noise_power, Nloop

def load_data_from_dir_cehbfnet(dir):
        feature = h5py.File(dir)
        H = feature['pcsi'][:]
        H = np.transpose(H, (2, 1, 0))
        H = H['real'] + 1j * H['imag']
        data = feature['data'][:]
        data = np.squeeze(data)
        data = np.transpose(data, (1, 0))
        data = data['real'] + 1j * data['imag']
        index_bs = feature['index_bs'][:]
        index_bs = np.transpose(index_bs, (1, 0))
        index_ms = feature['index_ms'][:]
        index_ms = np.transpose(index_ms, (1, 0))

        Nloop = np.int(H.shape[0])
        data = np.concatenate([np.real(data), np.imag(data)], 1)
        data = np.reshape(data, [Nloop, -1])
        index_bs = np.reshape(index_bs, [Nloop, -1])
        index_ms = np.reshape(index_ms, [Nloop, -1])
        Noise_power = 1 / np.power(10, np.random.randint(snr_begin, snr_end, [Nloop, 1]) / 10)
        return Nloop, H, data, index_bs, index_ms, Noise_power
def load_data_cehbfnet(pnr, load_trainset_flag, big_trainset_flag=1, L=3):
    if load_trainset_flag == 0:
        dir = f'../matlab/dataset_L{L}/snr{test_snr_begin}_{test_snr_end}/pnr{pnr}_test.mat'
        Nloop, H, data, index_bs, index_ms, Noise_power = load_data_from_dir_cehbfnet(dir)
    if load_trainset_flag == 1:
        # -------------------train set----------------------------
        if big_trainset_flag == 1:
            dir1 = f'../matlab/dataset_L{L}/pnr{pnr}_train1.mat'
            Nloop1, H1, data1, index_bs1, index_ms1, Noise_power1 = load_data_from_dir_cehbfnet(dir1)
            dir2 = f'../matlab/dataset_L{L}/pnr{pnr}_train2.mat'
            Nloop2, H2, data2, index_bs2, index_ms2, Noise_power2 = load_data_from_dir_cehbfnet(dir2)
            H = np.concatenate([H1, H2], axis=0)
            data = np.concatenate([data1, data2], axis=0)
            index_bs = np.concatenate([index_bs1, index_bs2], axis=0)
            index_ms = np.concatenate([index_ms1, index_ms2], axis=0)
            Noise_power = np.concatenate([Noise_power1, Noise_power2], axis=0)
            Nloop = Nloop1 + Nloop2

        else:
            dir = f'../matlab/dataset_L{L}/pnr{pnr}_train1.mat'
            Nloop, H, data, index_bs, index_ms, Noise_power = load_data_from_dir_cehbfnet(dir)
    H=H.astype(np.complex64)
    data=data.astype(np.float32)
    index_bs=index_bs.astype(np.float32)
    index_ms=index_ms.astype(np.float32)
    Noise_power=Noise_power.astype(np.float32)
    return Nloop, H, data, index_bs, index_ms, Noise_power


def phase2vrf(phase):
    v_real = tf.cos(phase)
    v_imag = tf.sin(phase)
    vrf = tf.cast(tf.complex(v_real, v_imag), tf.complex64)
    return vrf

def hbf_func(temp):
    vbb, vrf = temp
    vbb11 = tf.complex(tf.slice(vbb, [0, 0], [-1, 1]), tf.slice(vbb, [0, 1], [-1, 1]))
    vbb12 = tf.complex(tf.slice(vbb, [0, 2], [-1, 1]), tf.slice(vbb, [0, 3], [-1, 1]))
    vbb21 = tf.complex(tf.slice(vbb, [0, 4], [-1, 1]), tf.slice(vbb, [0, 5], [-1, 1]))
    vbb22 = tf.complex(tf.slice(vbb, [0, 6], [-1, 1]), tf.slice(vbb, [0, 7], [-1, 1]))
    vrf1 = tf.slice(vrf, [0, 0], [-1, Nt])
    vrf2 = tf.slice(vrf, [0, Nt], [-1, Nt])
    vhbf1 = vbb11 * vrf1 + vbb12 * vrf2
    vhbf2 = vbb21 * vrf1 + vbb22 * vrf2
    vhbf = tf.concat([vhbf1, vhbf2], axis=1)
    # sum power constraint
    vhbf1 = tf.divide(vhbf1, tf.norm(vhbf, axis=1, keepdims=True))
    vhbf2 = tf.divide(vhbf2, tf.norm(vhbf, axis=1, keepdims=True))
    vhbf = tf.concat([vhbf1, vhbf2], axis=1)
    return vhbf

def rate_func(temp):
    h, vbb, phase, noise_power = temp
    vrf = phase2vrf(phase)
    hbf = hbf_func([vbb, vrf])
    v = tf.reshape(hbf, (-1, 2, Nt))
    noise_power = tf.expand_dims(noise_power, 2)
    noise_power = tf.tile(noise_power, [1, 2, 2])
    noise_power = tf.cast(noise_power, tf.complex64)
    sig_power = tf.matmul(tf.matmul(h, tf.transpose(v, (0, 2, 1))),
                          tf.matmul(tf.conj(v), tf.conj(tf.transpose(h, (0, 2, 1)))))
    ones = tf.eye(num_rows=Ns, num_columns=2, batch_shape=tf.shape(h)[0:1], dtype=tf.complex64)
    rate = tf.math.log(tf.linalg.det(ones + sig_power / noise_power)) / tf.cast(tf.math.log(2.0), tf.complex64)
    rate = tf.cast(rate, tf.float32)
    # rate=tf.reshape(rate,(-1,1))
    return -rate

def loss_func(rate):
    def final_rate(y_t, y_p):
        return rate
    return final_rate


