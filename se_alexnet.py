from keras.models import Model
from keras.layers import Conv2D, Input, Dense, Flatten, MaxPool2D
from keras.layers import Activation, BatchNormalization

from squeeze_excite import SqueezeExcite

def alexnet_block(x, filters, kernel_size, se, name):
    y = Conv2D(filters, kernel_size, padding='same', name='{}_conv'.format(name))(x)
    if se:
        y = SqueezeExcite(y, ratio=16, name=name)
    y = BatchNormalization(name='{}_bn'.format(name))(y)
    y = Activation('relu', name='{}_act'.format(name))(y)
    y = MaxPool2D(pool_size=(2, 2), padding='same', name='{}_pool'.format(name))(y)
    return y


def dense_block(x, size, name, bn=True):
    y = Dense(size, name='{}_dense'.format(name))(x)
    if bn:
        y = BatchNormalization(name='{}_bn'.format(name))(y)
    y = Activation('relu', name='{}_act'.format(name))(y)
    return y


def SeAlexNet(nb_class, input_shape=(227, 227, 3), include_top=True, weights=None,
              batch_norm=True, se=True):
    """AlexNet without the splitted stream."""

    img_input = Input(shape=input_shape)

    x = alexnet_block(img_input, 64, (11, 11), se=se, name='block1')

    params = [
        (128, (7, 7)),
        (192, (3, 3)),
        (256, (3, 3))
    ]

    for i, (filters, kernel_size) in enumerate(params, start=2):
        x = alexnet_block(x, filters, kernel_size, se=se, name='block{}'.format(i))

    if include_top:
        x = Flatten(name='flatten')(x)
        x = dense_block(x, 4096, 'top1', bn=batch_norm)
        x = dense_block(x, 4096, 'top2', bn=batch_norm)
        x = Dense(nb_class, activation='softmax')(x)

    model = Model(inputs=img_input, outputs=x)

    if weights:
        print('Loading')
        model.load_weights(weights)

    return model
