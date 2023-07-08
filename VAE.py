from Dense import DenseLayer, DenseNetwork
from Convolutional import ConvolutionalLayer, ConvolutionalNetwork, ReshapeLayer, LambdaLayer
from Convolutional import MaxPoolingLayer, DenseLayer, UpSamplingLayer, FlattenLayer, BatchNormalizationLayer

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = np.load('lfw_data.npy')
attrs = pd.read_csv('lfw_attributes.csv')
data = data.astype(np.float32)


print(attrs.shape)
y = attrs[attrs.columns[:4]]
print("y", y.shape)

print(data.shape)
print(attrs.head())

data = data[:1000] / 255.0

batch_size = 100

print('Create and train the convolutional autoencoder')
model = ConvolutionalNetwork()

def encoder_function(inputs):
    z_mean_inherits, z_log_var_inherits = inputs[0], inputs[1]
    N = np.random.normal(loc=0., scale=1., size=z_mean_inherits.shape)
    return z_log_var_inherits * N + z_mean_inherits

model.add_layer(ConvolutionalLayer((45, 45, 3), num_filters=4, filter_size=3, padding='same', activation='relu'))
model.add_layer(FlattenLayer())

model.add_layer(DenseLayer(input_size=4*45*45*3, output_size=1000, activation='relu'))
model.add_layer(BatchNormalizationLayer(input_shape=1000))

model.add_layer(DenseLayer(input_size=1000, output_size=500, activation='relu'))
model.add_layer(BatchNormalizationLayer(input_shape=500))

model.add_layer(DenseLayer(input_size=500, output_size=100, activation='relu'))
model.add_layer(BatchNormalizationLayer(input_shape=100))

model.add_layer(DenseLayer(input_size=100, output_size=batch_size, activation='relu'))

z_mean_inherits = model.layers[-1].output
z_log_var_inherits = model.layers[-1].output

print('Добавление lambda-слоев для z_mean_inherits и z_log_var_inherits')
z_mean_lambda_layer = model.add_lambda_layer(lambda x: x, name='z_mean_inherits')
z_log_var_lambda_layer = model.add_lambda_layer(lambda x: x, name='z_log_var_inherits')

print('Использование lambda-слоев в функции encoder_function')
encoder_layer = model.add_lambda_layer(encoder_function, name='encoder_function')

print('Связывание lambda-слоев с z_mean_inherits и z_log_var_inherits')
z_mean_lambda_layer.forward(z_mean_inherits)
z_log_var_lambda_layer.forward(z_log_var_inherits)

model.add_layer(encoder_layer)

decoder_layers = [
    DenseLayer(input_size=batch_size, output_size=100, activation='relu'),
    BatchNormalizationLayer(input_shape=100),
    DenseLayer(input_size=100, output_size=500, activation='relu'),
    BatchNormalizationLayer(input_shape=500),
    DenseLayer(input_size=500, output_size=1000, activation='relu'),
    BatchNormalizationLayer(input_shape=1000),
    DenseLayer(input_size=1000, output_size=45*45*3, activation='sigmoid'),
    ReshapeLayer((batch_size, 45, 45, 3))
]

for layer in decoder_layers:
    model.add_layer(layer)

learning_rate = 0.0004
num_epochs = 3

print("Обучение модели")
model.fit(data, data, learning_rate, num_epochs, batch_size=batch_size)

y_pred = model.forward(data)[-1]

plt.imshow(y_pred[0])
plt.imshow(data[0])
plt.show()