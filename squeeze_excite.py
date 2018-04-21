from keras.layers import GlobalAveragePooling2D, Multiply, Dense
from keras import backend as K

def SqueezeExcite(x, ratio=16, name=''):
    nb_chan = K.int_shape(x)[-1]

    y = GlobalAveragePooling2D(name='{}_se_avg'.format(name))(x)
    y = Dense(nb_chan // ratio, activation='relu', name='{}_se_dense1'.format(name))(y)
    y = Dense(nb_chan, activation='sigmoid', name='{}_se_dense2'.format(name))(y)

    y = Multiply(name='{}_se_mul'.format(name))([x, y])
    return y
