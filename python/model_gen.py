from utils import *

def channel_attention(input_feature, ratio=1):

    # channel_axis = 1 if K.image_data_format() == "channels_last" else -1
    channel = input_feature.shape[-1]

    shared_layer_one = Dense(channel//ratio,
                             activation='relu',
                             kernel_initializer='he_normal',
                             use_bias=True,
                             bias_initializer='zeros')
    shared_layer_two = Dense(channel,
                             kernel_initializer='he_normal',
                             use_bias=True,
                             bias_initializer='zeros')

    avg_pool = GlobalAveragePooling2D()(input_feature)
    avg_pool = Reshape((1 ,1 ,channel))(avg_pool)
    # assert avg_pool._keras_shape[1:] == (1,1,channel)
    avg_pool = shared_layer_one(avg_pool)
    # assert avg_pool._keras_shape[1:] == (1,1,channel//ratio)
    avg_pool = shared_layer_two(avg_pool)
    # assert avg_pool._keras_shape[1:] == (1,1,channel)

    max_pool = GlobalMaxPooling2D()(input_feature)
    max_pool = Reshape((1 ,1 ,channel))(max_pool)
    # assert max_pool._keras_shape[1:] == (1,1,channel)
    max_pool = shared_layer_one(max_pool)
    # assert max_pool._keras_shape[1:] == (1,1,channel//ratio)
    max_pool = shared_layer_two(max_pool)
    # assert max_pool._keras_shape[1:] == (1,1,channel)

    cbam_feature = Add()([avg_pool ,max_pool])
    cbam_feature = Activation('sigmoid')(cbam_feature)

    #	if K.image_data_format() == "channels_first":
    #		cbam_feature = Permute((3, 1, 2))(cbam_feature)

    return multiply([input_feature, cbam_feature])
def dense_unit_dropout(input_tensor, nn, drop_out_rate):
    out_tensor = Dense(nn)(input_tensor)
    out_tensor = BatchNormalization()(out_tensor)
    out_tensor = tf.nn.relu(out_tensor)
    out_tensor = Dropout(drop_out_rate)(out_tensor)
    return out_tensor

def model_hbf_net():
    perfect_CSI = Input(name='perfect_CSI', shape=(Nr,Nt,), dtype=tf.complex64)
    estimated_CSI=Input(shape=(Nt,Nr,2,), dtype=tf.float32)
    Noise_power_input = Input(name='Noise_power_input', shape=(1,), dtype=tf.float32)

    tmp = BatchNormalization()(estimated_CSI)

    tmp=Conv2D(4*Nt,(3,2),activation='relu')(tmp)
    # tmp=AvgPool2D((2,1),strides=2)(tmp)
    tmp=BatchNormalization()(tmp)
    tmp=channel_attention(tmp)
    tmp=channel_attention(tmp)

    tmp=Conv2D(2*Nt,(3,1),activation='relu')(tmp)
    # tmp=AvgPool2D((2,1),strides=2)(tmp)
    tmp=BatchNormalization()(tmp)
    tmp=channel_attention(tmp)
    tmp=channel_attention(tmp)

    tmp=Flatten()(tmp)
    tmp=concatenate([tmp,Noise_power_input],axis=1)
    tmp=BatchNormalization()(tmp)
    # tmp=dense_unit_dropout(tmp,512,0.3)
    # tmp=BatchNormalization()(tmp)
    phase=Dense(Nrf*Nt,'relu')(tmp)
    vbb =Dense(8,name='vbb')(tmp)
    vrf = Lambda(phase2vrf,name='vrf')(phase)
    hbf = Lambda(hbf_func)([vbb, vrf])
    rate = Lambda(rate_func, dtype=tf.float32, output_shape=(1,))([perfect_CSI, vbb,phase, Noise_power_input])
    model = Model(inputs=[estimated_CSI, perfect_CSI, Noise_power_input], outputs=hbf)
    model.compile(optimizer='adam', loss=loss_func(rate))
    model.summary()
    return model

def model_ce_hbf_net():
    train_data = Input(shape=(180*2,), dtype=tf.float32)
    data_temp = BatchNormalization()(train_data)
    train_index_bs = Input(shape=(45,), dtype=tf.float32)
    index_bs_temp = BatchNormalization()(train_index_bs)
    train_index_ms = Input(shape=(3,), dtype=tf.float32)
    index_ms_temp = BatchNormalization()(train_index_ms)
    perfect_CSI = Input(name='perfect_CSI', shape=(Nr, Nt,), dtype=tf.complex64)
    Noise_power_input = Input(shape=(1,), dtype=tf.float32)

    imperfect_CSI = concatenate([data_temp, index_bs_temp, index_ms_temp, Noise_power_input])
    imperfect_CSI = BatchNormalization()(imperfect_CSI)
    attention= dense_unit_dropout(imperfect_CSI, 180 * 2, 0)
    data_temp=Multiply()([attention,train_data])
    imperfect_CSI = concatenate([data_temp, index_bs_temp, index_ms_temp, Noise_power_input])

    num_channel=6
    tmp = dense_unit_dropout(imperfect_CSI, Nt * Nr * num_channel, 0)
    tmp = Reshape((Nt, Nr, num_channel))(tmp)

    tmp = Conv2D(8 * Nt, (3, 2), activation='relu')(tmp)
    tmp=AvgPool2D((2,1),strides=2)(tmp)
    tmp = BatchNormalization()(tmp)
    tmp = channel_attention(tmp)

    tmp = Conv2D(6 * Nt, (3, 1), activation='relu')(tmp)
    tmp = AvgPool2D((2, 1), strides=2)(tmp)
    tmp = BatchNormalization()(tmp)
    tmp = channel_attention(tmp)

    tmp = Flatten()(tmp)
    tmp = concatenate([tmp, Noise_power_input], axis=1)
    tmp = BatchNormalization()(tmp)
    phase = Dense(Nrf * Nt, 'relu')(tmp)
    vbb = Dense(8,name='vbb')(tmp)
    vrf = Lambda(phase2vrf,name='vrf')(phase)
    hbf = Lambda(hbf_func)([vbb, vrf])
    rate = Lambda(rate_func, dtype=tf.float32, output_shape=(1,))([perfect_CSI, vbb, phase, Noise_power_input])
    model = Model(inputs=[train_data, train_index_bs, train_index_ms, perfect_CSI, Noise_power_input], outputs=hbf)
    model.compile(optimizer='adam', loss=loss_func(rate))
    # optimizer = optimizers.adam(lr=0.00001)
    # model.compile(optimizer=optimizer, loss=loss_func(rate))

    model.summary()
    return model

# Input only SNR and R (labeled as ‘HCNet-V1’)
def model_ce_hbf_net_v1():
    train_data = Input(shape=(180*2,), dtype=tf.float32)
    data_temp = BatchNormalization()(train_data)

    perfect_CSI = Input(name='perfect_CSI', shape=(Nr, Nt,), dtype=tf.complex64)
    Noise_power_input = Input(shape=(1,), dtype=tf.float32)

    imperfect_CSI = concatenate([data_temp, Noise_power_input])
    imperfect_CSI = BatchNormalization()(imperfect_CSI)
    attention= dense_unit_dropout(imperfect_CSI, 180 * 2, 0)
    data_temp=Multiply()([attention,train_data])
    imperfect_CSI = concatenate([data_temp, Noise_power_input])

    num_channel=6
    tmp = dense_unit_dropout(imperfect_CSI, Nt * Nr * num_channel, 0)
    tmp = Reshape((Nt, Nr, num_channel))(tmp)

    tmp = Conv2D(8 * Nt, (3, 2), activation='relu')(tmp)
    tmp=AvgPool2D((2,1),strides=2)(tmp)
    tmp = BatchNormalization()(tmp)
    tmp = channel_attention(tmp)

    tmp = Conv2D(6 * Nt, (3, 1), activation='relu')(tmp)
    tmp = AvgPool2D((2, 1), strides=2)(tmp)
    tmp = BatchNormalization()(tmp)
    tmp = channel_attention(tmp)

    tmp = Flatten()(tmp)
    tmp = concatenate([tmp, Noise_power_input], axis=1)
    tmp = BatchNormalization()(tmp)
    phase = Dense(Nrf * Nt, 'relu')(tmp)
    vbb = Dense(8)(tmp)
    vrf = Lambda(phase2vrf)(phase)
    hbf = Lambda(hbf_func)([vbb, vrf])
    rate = Lambda(rate_func, dtype=tf.float32, output_shape=(1,))([perfect_CSI, vbb, phase, Noise_power_input])
    model = Model(inputs=[train_data, perfect_CSI, Noise_power_input], outputs=hbf)
    model.compile(optimizer='adam', loss=loss_func(rate))
    # optimizer = optimizers.adam(lr=0.00001)
    # model.compile(optimizer=optimizer, loss=loss_func(rate))

    model.summary()
    return model
# Input only SNR, R and Isel in the last inner iteration of every outer iteration (labeled as ‘HCNet-V2’)
def model_ce_hbf_net_v2():
    train_data = Input(shape=(36,), dtype=tf.float32)
    data_temp = BatchNormalization()(train_data)
    train_index_bs = Input(shape=(9,), dtype=tf.float32)
    index_bs_temp = BatchNormalization()(train_index_bs)
    train_index_ms = Input(shape=(3,), dtype=tf.float32)
    index_ms_temp = BatchNormalization()(train_index_ms)
    perfect_CSI = Input(name='perfect_CSI', shape=(Nr, Nt,), dtype=tf.complex64)
    Noise_power_input = Input(shape=(1,), dtype=tf.float32)

    imperfect_CSI = concatenate([data_temp, index_bs_temp, index_ms_temp, Noise_power_input])
    imperfect_CSI = BatchNormalization()(imperfect_CSI)
    attention= dense_unit_dropout(imperfect_CSI, 36, 0)
    data_temp=Multiply()([attention,train_data])
    imperfect_CSI = concatenate([data_temp, index_bs_temp, index_ms_temp, Noise_power_input])

    num_channel=6
    tmp = dense_unit_dropout(imperfect_CSI, Nt * Nr * num_channel, 0)
    tmp = Reshape((Nt, Nr, num_channel))(tmp)

    tmp = Conv2D(8 * Nt, (3, 2), activation='relu')(tmp)#8
    tmp=AvgPool2D((2,1),strides=2)(tmp)
    tmp = BatchNormalization()(tmp)
    tmp = channel_attention(tmp)

    tmp = Conv2D(6 * Nt, (3, 1), activation='relu')(tmp)#6
    tmp = AvgPool2D((2, 1), strides=2)(tmp)
    tmp = BatchNormalization()(tmp)
    tmp = channel_attention(tmp)

    tmp = Flatten()(tmp)
    tmp = concatenate([tmp, Noise_power_input], axis=1)
    tmp = BatchNormalization()(tmp)
    phase = Dense(Nrf * Nt, 'relu')(tmp)
    vbb = Dense(8)(tmp)
    vrf = Lambda(phase2vrf)(phase)
    hbf = Lambda(hbf_func)([vbb, vrf])
    rate = Lambda(rate_func, dtype=tf.float32, output_shape=(1,))([perfect_CSI, vbb, phase, Noise_power_input])
    model = Model(inputs=[train_data, train_index_bs, train_index_ms, perfect_CSI, Noise_power_input], outputs=hbf)
    model.compile(optimizer='adam', loss=loss_func(rate))
    # optimizer = optimizers.adam(lr=0.00001)
    # model.compile(optimizer=optimizer, loss=loss_func(rate))

    model.summary()
    return model




