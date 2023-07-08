import os
import numpy as np
import tensorflow as tf
import tensorflow.keras.layers as L
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


data = np.load('lfw_data.npy')
attrs = pd.read_csv('lfw_attributes.csv')
# data = data.astype(np.float32)

X_train = data[:10000].reshape((10000, -1))
print(X_train.shape)
X_val = data[10000:].reshape((-1, X_train.shape[1]))
print(X_val.shape)
X_train = np.float32(X_train)
X_train = X_train/255
X_val = np.float32(X_val)
X_val = X_val/255

image_h = data.shape[1]
image_w = data.shape[2]

dimZ = 100


class VAE(tf.keras.Model):
    def __init__(self, encoder, decoder, **kwargs):
        super(VAE, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder

    def KL_divergence(self, mu, logsigma):
        kl_loss = -0.5 * tf.reduce_sum(1 + logsigma - tf.square(mu) - tf.exp(logsigma), axis=1)
        return tf.reduce_mean(kl_loss)

    def log_likelihood(self, x, z):
        recon_loss = tf.reduce_sum(tf.square(x - self.decoder(z)), axis=[1, 2, 3])
        return tf.reduce_mean(recon_loss)

    @tf.function
    def train_step(self, data):
        x = data
        with tf.GradientTape() as tape:
            mu, logsigma, z = self.encoder(x)
            recon_loss = self.log_likelihood(x, z)
            kl_loss = self.KL_divergence(mu, logsigma)
            total_loss = recon_loss + kl_loss
        gradients = tape.gradient(total_loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        return {'loss': total_loss}

    def call(self, inputs):
        mu, logsigma, _ = self.encoder(inputs)
        return self.decoder(mu + tf.exp(0.5 * logsigma) * tf.random.normal(shape=tf.shape(mu)))



from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Reshape, Conv2DTranspose, Lambda
import tensorflow.keras.backend as K

# Define the encoder model
encoder_inputs = Input(shape=(image_h, image_w, 3))
x = Conv2D(32, (3, 3), activation='relu', padding='same')(encoder_inputs)
x = MaxPooling2D((2, 2))(x)
x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
x = MaxPooling2D((2, 2))(x)
x = Flatten()(x)
z_mean = Dense(dimZ, activation='relu', kernel_initializer='glorot_uniform', bias_initializer='zeros')(x)
z_log_var = Dense(dimZ, activation='relu', kernel_initializer='glorot_uniform', bias_initializer='zeros')(x)

# Define the sampling layer
def sampling(args):
    z_mean, z_log_var = args
    epsilon = K.random_normal(shape=(K.shape(z_mean)[0], dimZ), mean=0., stddev=1.)
    return z_mean + K.exp(0.5 * z_log_var) * epsilon

z = Lambda(sampling)([z_mean, z_log_var])

# Define the encoder model
encoder = tf.keras.Model(encoder_inputs, [z_mean, z_log_var, z])


# Define the decoder model
decoder_inputs = Input(shape=(dimZ,))
x = Dense(11 * 11 * 64, activation='relu', kernel_initializer='glorot_uniform', bias_initializer='zeros')(decoder_inputs)
x = Reshape((11, 11, 64))(x)
x = Conv2DTranspose(32, (3, 3), strides=(2, 2), activation='relu', padding='same')(x)
x = Conv2DTranspose(16, (3, 3), strides=(2, 2), activation='relu', padding='same')(x)
x = Conv2DTranspose(8, (3, 3), strides=(2, 2), activation='relu', padding='same')(x)
x = Conv2DTranspose(3, (3, 3), strides=(1, 1), activation='sigmoid', padding='valid')(x)
decoder_outputs = MaxPooling2D((2, 2))(x)

decoder = tf.keras.Model(decoder_inputs, decoder_outputs)


vae = VAE(encoder, decoder)
vae.compile(optimizer=tf.keras.optimizers.Adam())


history = vae.fit(X_train, epochs=5, batch_size=100)

latent_dim = 100

z = tf.random.normal(shape=(1, latent_dim))

reconstructed_output = vae.decoder(z)

plt.imshow(reconstructed_output[0])
plt.axis('off')
plt.show()
