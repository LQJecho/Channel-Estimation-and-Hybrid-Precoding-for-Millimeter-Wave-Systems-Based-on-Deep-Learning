from model_gen import *

con = tf.ConfigProto()
con.gpu_options.allow_growth = True
tf.keras.backend.set_session(tf.Session(config=con))


def run_hbf_net():
    model = model_hbf_net()
    save_dir = './model/HBF_Net.h5'
    reduce_lr = callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=20, min_lr=0.00005)
    checkpoint = callbacks.ModelCheckpoint(save_dir, monitor='val_loss',
                                           verbose=1, save_best_only=True, mode='min', save_weights_only=True)
    H, Hest, Noise_power, Nloop = load_data_hbfnet(pnr, load_trainset_flag, big_trainset_flag=0, L=L)
    if train_flag == 1:
        model.fit(x=[Hest, H, Noise_power], y=H, batch_size=bs,
                  epochs=epoch, verbose=2, validation_split=0.1, callbacks=[reduce_lr, checkpoint])
    if train_flag == 0:
            rate = []
            model.load_weights(save_dir)
            for snr in test_snr_array:
                Noise_power = 1 / np.power(10, np.ones([Nloop, 1]) * snr / 10)
                y = model.evaluate(x=[Hest, H, Noise_power], y=H, batch_size=10000)
                print(snr, y)
                rate.append(-y)
            print(rate)
def run_ce_hbf_net():
    model = model_ce_hbf_net()
    save_dir = './model/CE_HBF_Net.h5'

    reduce_lr = callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=20, min_lr=0.00005)
    checkpoint = callbacks.ModelCheckpoint(save_dir, monitor='val_loss',
                                           verbose=1, save_best_only=True, mode='min', save_weights_only=True)
    Nloop, H, data, index_bs, index_ms, Noise_power = load_data_cehbfnet(pnr, load_trainset_flag, L=L)
    if train_flag == 1:
        model.fit(x=[data, index_bs, index_ms, H, Noise_power], y=H, batch_size=bs,
                  epochs=epoch, verbose=2, validation_split=0.1, callbacks=[reduce_lr, checkpoint])
    if train_flag == 0:
            rate = []
            model.load_weights(save_dir)
            for snr in test_snr_array:
                Noise_power = 1 / np.power(10, np.ones([Nloop, 1]) * snr / 10)
                y = model.evaluate(x=[data, index_bs, index_ms, H, Noise_power], y=H, batch_size=10000)
                print(snr, y)
                rate.append(-y)
            print(rate)

def run_ce_hbf_net_v1():
    model = model_ce_hbf_net_v1()
    save_dir = './model/CE_HBF_Net_V1.h5'
    reduce_lr = callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=20, min_lr=0.00005)
    checkpoint = callbacks.ModelCheckpoint(save_dir, monitor='val_loss',
                                           verbose=1, save_best_only=True, mode='min', save_weights_only=True)
    Nloop, H, data, index_bs, index_ms, Noise_power = load_data_cehbfnet(pnr, load_trainset_flag, L=L)
    if train_flag == 1:
        model.fit(x=[data, H, Noise_power], y=H, batch_size=bs,
                  epochs=epoch, verbose=2, validation_split=0.1, callbacks=[reduce_lr, checkpoint])
    if train_flag == 0:
            rate = []
            model.load_weights(save_dir)
            for snr in test_snr_array:
                Noise_power = 1 / np.power(10, np.ones([Nloop, 1]) * snr / 10)
                y = model.evaluate(x=[data, H, Noise_power], y=H, batch_size=10000)
                print(snr, y)
                rate.append(-y)
            print(rate)
def run_ce_hbf_net_v2():
    model = model_ce_hbf_net_v2()
    save_dir = './model/CE_HBF_Net_V2.h5'

    reduce_lr = callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=20, min_lr=0.00005)
    checkpoint = callbacks.ModelCheckpoint(save_dir, monitor='val_loss',
                                           verbose=1, save_best_only=True, mode='min', save_weights_only=True)
    Nloop, H, data, index_bs, index_ms, Noise_power = load_data_cehbfnet_v(load_trainset_flag, v=2)

    if train_flag == 1:
        model.fit(x=[data, index_bs, index_ms, H, Noise_power], y=H, batch_size=bs,
                  epochs=epoch, verbose=2, validation_split=0.1, callbacks=[reduce_lr, checkpoint])
    if train_flag == 0:
            rate = []
            model.load_weights(save_dir)
            for snr in test_snr_array:
                Noise_power = 1 / np.power(10, np.ones([Nloop, 1]) * snr / 10)
                y = model.evaluate(x=[data, index_bs, index_ms, H, Noise_power], y=H, batch_size=5000)
                print(snr, y)
                rate.append(-y)
            print(rate)



epoch = 100
bs=2048 * 3
train_flag = 0
load_trainset_flag =train_flag
L=3
pnr= 0

run_hbf_net()
run_ce_hbf_net()
run_ce_hbf_net_v1()
run_ce_hbf_net_v2()


