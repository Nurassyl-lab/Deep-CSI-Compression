'''
    Modified Siren Model
    With More parameters
'''
from keras.layers import Layer
import tensorflow as tf
import numpy as np
import keras
from tensorflow.keras import layers

class Siren(Layer):
    def __init__(self, units, w0=1.0, c=6.0, **kwargs):
        super(Siren, self).__init__(**kwargs)
        self.units = units
        self.w0 = w0
        self.c = c

    def build(self, input_shape):
        self.input_dim = input_shape[-1]
        self.kernel = self.add_weight(shape=(self.input_dim, self.units),
                                      initializer='uniform',
                                      name='kernel')
        self.bias = self.add_weight(shape=(self.units,),
                                    initializer='zeros',
                                    name='bias')
        self.w0_coeff = tf.constant(np.pi * self.w0, dtype=tf.float32)

    def call(self, inputs):
        x = tf.matmul(inputs, self.kernel)
        x = tf.math.sin(self.w0_coeff * x + self.bias) / self.c
        return x

def define_siren():
    encoder_inputs = keras.Input(shape=(2, 64, 100, 1))

    x = layers.Flatten()(encoder_inputs)
    x = Siren(units=12800)(x)
    x = layers.Reshape((2,64,100,1))(x)
    #x = layers.Dropout(0.2)(x)
    x = layers.Conv3D(32, (1, 3, 3), activation='tanh', padding='same', name = 'h1')(x)
    #x = layers.Dropout(0.2)(x)
    x = layers.MaxPooling3D((1,2,2), name = 'h2')(x)#32 50
    x = layers.Conv3D(64, (1, 3,3), activation="tanh", padding='same',name = 'h3')(x)
    #x = layers.Dropout(0.2)(x)
    x = layers.MaxPooling3D((1,1,2), name = 'h4')(x)#32 25
    encoded = layers.Conv3D(64, (1, 3,3), activation="tanh", padding='same',name = 'encoded')(x)

    decoder_inputs = keras.Input(shape = (2, 32, 25, 64))
    x = layers.Conv3D(64, (1, 3, 3), activation="tanh", padding='same',name = 'h5')(decoder_inputs)
    x = layers.UpSampling3D((1, 1,2), name = 'h6')(decoder_inputs)#32 50

    x = layers.Conv3D(64, (1, 3,3), activation="tanh", padding='same',name = 'h7')(x)
    x = layers.UpSampling3D((1,2,2), name = 'h8')(x)#64 100

    decoder_outputs = layers.Conv3D(1, (1,3,3), activation="tanh", padding='same',name = 'h9')(x)

    cnn_encoder = keras.Model(encoder_inputs, encoded, name="encoded")
    cnn_decoder = keras.Model(decoder_inputs, decoder_outputs, name="decoded")

    autoencoder_input = layers.Input(shape=(2,64,100,1), name = 'input')
    autoencoder_encoder_output = cnn_encoder(autoencoder_input)
    autoencoder_decoder_output = cnn_decoder(autoencoder_encoder_output)

    cnn_autoencoder = keras.Model(autoencoder_input, autoencoder_decoder_output, name = 'autoencoder')
    cnn_autoencoder.compile(optimizer=keras.optimizers.Adam(), loss = 'mse')

    return cnn_encoder, cnn_decoder, cnn_autoencoder

if __name__ == "__main__":
    # Instantiate the encoder, decoder, and autoencoder models
    encoder, decoder, autoencoder = define_siren()
    
    # Print the summary of each model to inspect their structures
    print("Encoder Summary:")
    encoder.summary()
    print("\nDecoder Summary:")
    decoder.summary()
    print("\nAutoencoder Summary:")
    autoencoder.summary()
    
    # Generate dummy data to simulate an input to the autoencoder
    sample_data = np.random.random((1, 2, 64, 100, 1))  # Batch size of 1
    
    # Perform a single forward pass with the autoencoder (just for demonstration)
    output = autoencoder.predict(sample_data)
    
    print("\nOutput Shape:", output.shape)
