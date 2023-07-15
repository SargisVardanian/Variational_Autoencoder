import time
import os
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_digits

import tensorflow as tf
import keras
from keras.models import Sequential
from keras import layers as L
from keras import optimizers
from keras.preprocessing.image import ImageDataGenerator


import keras.backend as K
import pandas as pd

data = np.load('lfw_data.npy')
attrs = pd.read_csv('lfw_attributes.csv')

data = np.float32(data[:6000]) / 255.

data = tf.image.resize(data, size=(45, 45))

X_train = tf.reshape(data[:4000], (-1, 45, 45, 3))
print(X_train.shape)

X_val = tf.reshape(data[4000:6000], (-1, 45, 45, 3))
print(X_val.shape)

X_train = np.float32(X_train)
X_val = np.float32(X_val)

IMG_SHAPE = data.shape[1:]

print(f'IMG_SHAPE{IMG_SHAPE}')
hidd = 256

image_h = data.shape[1]
image_w = data.shape[2]

encoder_inputs = L.Input(shape=(image_h, image_w, 3))
x = L.Conv2D(32, (3, 3), activation='relu', padding='same')(encoder_inputs)
x = L.MaxPooling2D((2, 2))(x)
x = L.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
x = L.MaxPooling2D((2, 2))(x)
x = L.Flatten()(x)
z_mean = L.Dense(hidd, activation='relu', kernel_initializer='glorot_uniform', bias_initializer='zeros')(x)
z_log_var = L.Dense(hidd, activation='relu', kernel_initializer='glorot_uniform', bias_initializer='zeros')(x)

# Define the sampling layer
def sampling(args):
    z_mean, z_log_var = args
    epsilon = K.random_normal(shape=(K.shape(z_mean)[0], hidd), mean=0., stddev=1.)
    return z_mean + K.exp(0.5 * z_log_var) * epsilon

z = L.Lambda(sampling)([z_mean, z_log_var])

# Define the encoder model
encoder = tf.keras.Model(encoder_inputs, [z_mean, z_log_var, z])


print("GAN generator")

decoder_inputs = L.Input(shape=(hidd,))
x = L.Dropout(0.5)(x)
x = L.Dense(512, activation='relu')(x)
x = L.Dropout(0.5)(x)
x = L.Dense(256, activation='elu')(decoder_inputs)
x = L.Dropout(0.5)(x)
x = L.BatchNormalization()(x)
x = L.Dense(128, activation='elu')(x)
x = L.BatchNormalization()(x)
x = L.Dense(242, activation='elu')(x)
x = L.BatchNormalization()(x)
x = L.Dense(11 * 11 * 64, activation='relu', kernel_initializer='glorot_uniform')(x)
x = L.Reshape((11, 11, 64))(x)
x = L.BatchNormalization()(x)
x = L.Conv2DTranspose(32, (3, 3), strides=(2, 2), activation='elu')(x)
x = L.BatchNormalization()(x)
x = L.Conv2DTranspose(16, (3, 3), strides=(2, 2), activation='elu')(x)
x = L.BatchNormalization()(x)
x = L.Conv2DTranspose(8, (5, 5), strides=(1, 1), activation='elu')(x)
x = L.BatchNormalization()(x)
x = L.Conv2D(3, (7, 7), strides=(1, 1), activation='sigmoid')(x)
decoder_outputs = x




generator = tf.keras.Model(decoder_inputs, decoder_outputs)


print("GAN discriminator")
discriminator = Sequential()
discriminator.add(L.InputLayer(IMG_SHAPE))
discriminator.add(L.Conv2D(64, kernel_size=3, activation='relu', padding='same'))
discriminator.add(L.BatchNormalization())
discriminator.add(L.Conv2D(128, kernel_size=3, activation='relu', padding='same'))
discriminator.add(L.BatchNormalization())
discriminator.add(L.MaxPooling2D(pool_size=(2, 2)))
discriminator.add(L.Conv2D(64, kernel_size=3, activation='relu', padding='same'))
discriminator.add(L.BatchNormalization())
discriminator.add(L.Flatten())
discriminator.add(L.Dense(256, activation='relu'))
discriminator.add(L.Dropout(0.5))
discriminator.add(L.Dense(1))


print('samples')
def sample_noise_batch(bsize):
    return np.random.normal(size=(bsize, hidd)).astype('float32')

def sample_data_batch(idxs):
    return tf.gather(data, idxs)

def sample_images(nrow, ncol, save_dir, sharp=False):
    images = generator.predict(sample_noise_batch(bsize=nrow * ncol))
    if np.var(images) != 0:
        images = images.clip(np.min(data), np.max(data))

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    for i in range(nrow * ncol):
        if sharp:
            img = images[i].reshape(IMG_SHAPE)
        else:
            img = images[i].reshape(IMG_SHAPE)

        plt.imshow(img, cmap="gray")
        plt.axis('off')
        plt.savefig(os.path.join(save_dir, f'image_{i}.png'))
        plt.close()

def sample_probas(bsize, save_dir, batch_size=50):
    idxs_real = np.random.choice(np.arange(data.shape[0]), size=bsize)
    idxs_gen = np.random.choice(np.arange(batch_size), size=bsize)

    preds_real = discriminator.predict(sample_data_batch(idxs_real))
    preds_gen = discriminator.predict(sample_data_batch(idxs_gen))

    if preds_real.ndim == 1:
        preds_real = np.expand_dims(preds_real, axis=1)
    if preds_gen.ndim == 1:
        preds_gen = np.expand_dims(preds_gen, axis=1)

    plt.title('Generated vs real data')
    plt.hist(np.exp(preds_real)[:, 0], label='D(x)', alpha=0.5, range=[0, 1])
    plt.hist(np.exp(preds_gen)[:, 0], label='D(G(x))', alpha=0.5, range=[0, 1])
    plt.legend(loc='upper center')

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    plt.savefig(os.path.join(save_dir, 'histogram{1{2}}.png'))
    plt.close()

def generator_loss_for_VAE(x, generated_images):
    recon_loss = tf.reduce_mean(tf.square(x - generated_images))
    return recon_loss

# def generator_loss(fake_output):
#     loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(fake_output), logits=fake_output)
#     return tf.reduce_mean(loss)

def discriminator_loss(real_output, fake_output):
    real_loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(real_output), logits=real_output)
    fake_loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros_like(fake_output), logits=fake_output)
    total_loss = tf.reduce_mean(real_loss) + tf.reduce_mean(fake_loss)
    return total_loss
#
#
def generator_loss(fake_output):
    return -tf.reduce_mean(fake_output)
#
#
# def discriminator_loss(real_output, fake_output):
#     return tf.reduce_mean(fake_output) - tf.reduce_mean(real_output)


generator_optimizer = tf.keras.optimizers.Adam(learning_rate=1e-5, clipvalue=0.8)
discriminator_optimizer = tf.keras.optimizers.Adam(learning_rate=1e-6, clipvalue=0.6)

@tf.function
def train_step(images, batch_size, epoch):
    noise = sample_noise_batch(batch_size)
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        z_mean, z_log_var, z = encoder(images, training=True)

        if epoch // 2 == 1:
            generated_images = generator(z, training=True)
        else:
            generated_images = generator(noise, training=True)

        real_output = discriminator(images, training=True)
        fake_output = discriminator(generated_images, training=True)

        gen_loss = generator_loss(fake_output)
        disc_loss = discriminator_loss(real_output, fake_output)

    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))
    return gen_loss, disc_loss

def train(X_train, epochs, batch_size=50):
    history = []
    Max_P_l = 35
    th = X_train.shape[0] // (batch_size * Max_P_l)
    dataset = tf.data.Dataset.from_tensor_slices(X_train)
    dataset = dataset.shuffle(buffer_size=5000).batch(batch_size)

    image_generator = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        brightness_range=(0.8, 1.2)
    )

    checkpoint_dir = 'checkpoints'
    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
    checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                     discriminator_optimizer=discriminator_optimizer,
                                     generator=generator,
                                     discriminator=discriminator)
    manager = tf.train.CheckpointManager(checkpoint, checkpoint_dir, max_to_keep=1)
    latest_checkpoint = manager.latest_checkpoint
    if latest_checkpoint:
        checkpoint.restore(latest_checkpoint)
        print("Loaded weights from the latest checkpoint:", latest_checkpoint)

    for epoch in range(1, epochs + 1):
        epoch_dir = os.path.join('histograms', f'number_of_start_{4}')
        os.makedirs(epoch_dir, exist_ok=True)
        print(f'{epoch}/{epochs}:', end='')
        start = time.time()
        n = 0
        gen_loss_epoch = 0
        disc_loss_epoch = 0
        for image_batch in dataset:
            transformed_images = image_generator.flow(image_batch, batch_size=batch_size, shuffle=False).next()
            gen_loss, disc_loss = train_step(image_batch, batch_size, epoch)
            gen_loss_epoch += gen_loss.numpy()
            disc_loss_epoch += disc_loss.numpy()
            if (n % th == 0): print("~", end='')
            n += 1
        history += [gen_loss_epoch / n]
        print(': ' + str(history[-1]))
        print('epoch: {} Time: {} loss: {}'.format(epoch, time.time() - start, [gen_loss_epoch, disc_loss_epoch]))
        sample_images(2, 5, 'generated_images', sharp=True)
        sample_probas(batch_size, epoch_dir)
        print('Save weights in checkpoint')
        manager.save(checkpoint_number=epoch)
        print('Saved')
        tt = 60
        print(f"Waiting for {tt/60} minutes...")
        time.sleep(tt)

    return history

epochs = 70
history = train(X_train, epochs)

plt.plot(history)
plt.grid(True)
plt.show()