import keras
import keras.backend as K


def Net():
    # define input
    x = keras.Input(shape=(55, 47, 3), name='input')
    # feature extraction
    conv_1 = keras.layers.Conv2D(32, (3, 3), name='conv_1')(x)
    bn_1 = keras.layers.BatchNormalization(name='bn_1')(conv_1)
    relu_1 = keras.layers.Activation('relu')(bn_1)
    pool_1 = keras.layers.MaxPooling2D((2, 2), name='pool_1')(relu_1)

    conv_2 = keras.layers.Conv2D(64, (3, 3), name='conv_2')(pool_1)
    bn_2 = keras.layers.BatchNormalization(name='bn_2')(conv_2)
    relu_2 = keras.layers.Activation('relu')(bn_2)
    pool_2 = keras.layers.MaxPooling2D((2, 2), name='pool_2')(relu_2)

    conv_3 = keras.layers.Conv2D(128, (3, 3), name='conv_3')(pool_2)
    bn_3 = keras.layers.BatchNormalization(name='bn_3')(conv_3)
    relu_3 = keras.layers.Activation('relu')(bn_3)
    pool_3 = keras.layers.MaxPooling2D((2, 2), name='pool_3')(relu_3)

    conv_4 = keras.layers.Conv2D(128, (3, 3), name='conv_4', padding='same')(pool_3)
    bn_4 = keras.layers.BatchNormalization(name='bn_4')(conv_4)
    relu_4 = keras.layers.Activation('relu')(bn_4)
    conv_5 = keras.layers.Conv2D(128, (3, 3), name='conv_5', padding='same')(relu_4)
    bn_5 = keras.layers.BatchNormalization(name='bn_5')(conv_5)
    relu_5 = keras.layers.Activation('relu')(bn_5)
    conv_6 = keras.layers.Conv2D(128, (3, 3), name='conv_6', padding='same')(relu_5)
    bn_6 = keras.layers.BatchNormalization(name='bn_6')(conv_6)
    add_1 = keras.layers.Add()([pool_3, bn_6])
    act_1 = keras.layers.Activation('relu')(add_1)

    #
    flat_1 = keras.layers.Flatten()(act_1)
    fc_1 = keras.layers.Dense(1284, name='fc_1')(flat_1)
    # second interpretation model
    conv_4 = keras.layers.Conv2D(256, (3, 3), activation='relu', name='conv_7')(act_1)
    flat_2 = keras.layers.Flatten()(conv_4)
    fc_2 = keras.layers.Dense(1284, name='fc_2')(flat_2)
    # merge interpretation
    merge = keras.layers.Concatenate(axis=-1)([fc_1, fc_2])
    add_1 = keras.layers.Activation('relu')(merge)
    y_hat = keras.layers.Dense(1284, activation='softmax', name='output')(add_1)
    model = keras.Model(inputs=x, outputs=y_hat)
    # summarize layers
    print(model.summary())
    return model


K.clear_session()
model = Net()

