# Define Generator & Discriminator (simplified)
from tensorflow.keras.layers import Dense, LeakyReLU

def build_generator():
    model = Sequential()
    model.add(Dense(128, input_dim=100))
    model.add(LeakyReLU(0.2))
    model.add(Dense(784, activation='tanh'))
    model.add(tf.keras.layers.Reshape((28, 28, 1)))
    return model

def build_discriminator():
    model = Sequential()
    model.add(tf.keras.layers.Flatten(input_shape=(28, 28, 1)))
    model.add(Dense(128))
    model.add(LeakyReLU(0.2))
    model.add(Dense(1, activation='sigmoid'))
    return model

# Training loop can be added on request
