from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, Activation, MaxPooling2D, UpSampling2D, Concatenate, Add
from tensorflow.keras.models import Model

def conv_block(inputs, filters, kernel_size=3, strides=1):
    x = Conv2D(filters, kernel_size, strides=strides, padding='same')(inputs)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    return x

def encoder_block(inputs, filters, kernel_size=3, strides=1):
    x = conv_block(inputs, filters, kernel_size, strides)
    x = conv_block(x, filters, kernel_size, 1)
    shortcut = Conv2D(filters, kernel_size=1, strides=strides, padding='same')(inputs)
    shortcut = BatchNormalization()(shortcut)
    x = Add()([x, shortcut])
    x = Activation('relu')(x)
    return x

def decoder_block(inputs, filters, kernel_size=3, strides=1):
    x = UpSampling2D(size=(2, 2))(inputs)
    x = conv_block(x, filters, kernel_size, 1)
    x = conv_block(x, filters, kernel_size, 1)
    shortcut = UpSampling2D(size=(2, 2))(inputs)
    shortcut = Conv2D(filters, kernel_size=1, strides=1, padding='same')(shortcut)
    shortcut = BatchNormalization()(shortcut)
    x = Add()([x, shortcut])
    x = Activation('relu')(x)
    return x

def build_model(input_shape=(224, 224, 3), num_classes=1):
    inputs = Input(shape=input_shape)

    # Encoder
    enc1 = encoder_block(inputs, 64, strides=2)
    enc2 = encoder_block(enc1, 128, strides=2)
    enc3 = encoder_block(enc2, 256, strides=2)
    enc4 = encoder_block(enc3, 512, strides=2)

    # Decoder
    dec4 = decoder_block(enc4, 256, strides=2)
    dec3 = decoder_block(dec4, 128, strides=2)
    dec2 = decoder_block(dec3, 64, strides=2)
    dec1 = decoder_block(dec2, 64, strides=2)

    outputs = Conv2D(num_classes, (1, 1), activation='sigmoid')(dec1)

    model = Model(inputs, outputs)
    return model