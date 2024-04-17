import numpy as np
import keras
from tensorflow.keras import layers
from keras.models import Sequential
import tensorflow

def define_cnn(act_function = 'tanh', lr = 0.001):
    model = Sequential()
    model.add(keras.layers.InputLayer(input_shape=(2, 64, 100, 1)))
    model.add(layers.Conv3D(32, (1, 3, 3), activation=act_function, padding='same', name = 'h1'))
    model.add(layers.MaxPooling3D((1, 2, 2), padding='same', name = 'h2'))#32 50
    model.add(layers.Conv3D(64, (1, 3, 3), activation=act_function, padding='same', name = 'h3'))
    model.add(layers.MaxPooling3D((1, 2, 2), padding='same', name = 'h4'))#16 25
    model.add(layers.Conv3D(64, (1, 3, 3), activation=act_function, padding='same', name = 'h5'))
    model.add(layers.MaxPooling3D((1, 2, 5), padding='same', name = 'h6'))#8 5 encoder last layer
    model.add(layers.Conv3D(64, (1, 3, 3), activation=act_function, padding='same', name = 'h7'))

    model.add(layers.Conv3D(64, (1, 3, 3), activation=act_function, padding='same', name = 'h10'))
    model.add(layers.UpSampling3D((1, 2, 5), name = 'h11'))#16, 25 decoder first layer
    model.add(layers.Conv3D(64, (1, 3, 3), activation=act_function, padding='same', name = 'h12'))
    model.add(layers.UpSampling3D((1, 2, 2), name = 'h13'))#32, 50
    model.add(layers.Conv3D(64, (1, 3, 3), activation=act_function, padding = 'same',name = 'h14'))
    model.add(layers.UpSampling3D((1, 2, 2), name = 'h15'))
    model.add(layers.Conv3D(1, (1, 3, 3), activation=act_function, padding='same', name = 'decoded'))
    model.compile(optimizer=tensorflow.keras.optimizers.Adam(learning_rate = lr), loss = 'mse')

    return model

if __name__ == "__main__":
    # Instantiate the encoder, decoder, and autoencoder models
    autoencoder = define_cnn()
    
    # Print the summary of each model to inspect their structures
    print("\nAutoencoder Summary:")
    autoencoder.summary()
    
    # Generate dummy data to simulate an input to the autoencoder
    sample_data = np.random.random((1, 2, 64, 100, 1))  # Batch size of 1
    
    # Perform a single forward pass with the autoencoder (just for demonstration)
    output = autoencoder.predict(sample_data)
    
    print("\nOutput Shape:", output.shape)
