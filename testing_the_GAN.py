from Gan import sample_images, generator
import numpy as np
import os
import tensorflow as tf
import matplotlib.pyplot as plt

data = np.load('lfw_data.npy')
data = np.float32(data[:6000]) / 255.

hidd =256

checkpoint_dir = 'checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(generator=generator)
manager = tf.train.CheckpointManager(checkpoint, checkpoint_dir, max_to_keep=1)
latest_checkpoint = manager.latest_checkpoint
if latest_checkpoint:
    checkpoint.restore(latest_checkpoint)
    print("Loaded weights from the latest checkpoint:", latest_checkpoint)

IMG_SHAPE = data.shape[1:]

def sample_noise_batch(bsize):
    return np.random.normal(size=(bsize, hidd)).astype('float32')

def sample_images(nrow, ncol, save_dir):
    images = generator(sample_noise_batch(bsize=nrow * ncol), training=False)
    if np.var(images) != 0:
        images = images.clip(np.min(data), np.max(data))
    for i in range(nrow*ncol):
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
            img = images[i].reshape(IMG_SHAPE)
            plt.imshow(img, cmap="gray")
            plt.axis('off')
            plt.savefig(os.path.join(save_dir, f'image_{i}.png'))
            plt.close()

sample_images(2, 5, 'generated_images_testing')
