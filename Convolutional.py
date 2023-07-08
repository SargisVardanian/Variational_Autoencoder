import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import pickle
import os
import cv2
import glob
from tqdm import tqdm

class DenseLayer:
    def __init__(self, input_size, output_size, activation='relu', inherits=None, name=None):
        print('DenseLayer', input_size, output_size)
        self.name = name
        if inherits is not None:
            self.weights = np.copy(inherits.weights)
            self.biases = np.copy(inherits.biases)
        else:
            self.weights = np.random.randn(input_size, output_size)
            self.biases = np.ones(output_size)
        self.activation = activation
        self.output = None

    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.dot(inputs, self.weights) + self.biases
        if self.activation == 'relu':
            self.output = self.relu(self.output)
        elif self.activation == 'sigmoid':
            self.output = self.sigmoid(self.output)
        return self.output

    def backward(self, grad_output, learning_rate):
        if self.activation == 'relu':
            grad_output = self.def_relu(self.output) * grad_output
            grad_weights = np.dot(self.inputs.T, grad_output)
            grad_biases = np.sum(grad_output, axis=0)
            grad_weights = self.def_relu(grad_weights)
            grad_biases = self.def_relu(grad_biases)
        elif self.activation == 'sigmoid':  # Добавленная проверка для softmax
            grad_output = self.def_sigmoid(self.output) * grad_output
            grad_weights = np.dot(self.inputs.T, grad_output)
            grad_biases = np.sum(grad_output, axis=0)
            grad_weights = self.def_sigmoid(grad_weights)
            grad_biases = self.def_sigmoid(grad_biases)
        else:
            grad_weights = np.dot(self.inputs.T, grad_output)
            grad_biases = np.sum(grad_output, axis=0)
        # self.t += 1
        # self.m_w = self.beta1 * self.m_w + (1 - self.beta1) * grad_weights
        # self.v_w = self.beta2 * self.v_w + (1 - self.beta2) * (grad_weights ** 2)
        # m_w_hat = self.m_w / (1 - self.beta1 ** self.t)
        # v_w_hat = self.v_w / (1 - self.beta2 ** self.t)
        # self.weights -= learning_rate * m_w_hat / (np.sqrt(v_w_hat) + self.epsilon)
        #
        # self.m_b = self.beta1 * self.m_b + (1 - self.beta1) * grad_biases
        # self.v_b = self.beta2 * self.v_b + (1 - self.beta2) * (grad_biases ** 2)
        # m_b_hat = self.m_b / (1 - self.beta1 ** self.t)
        # v_b_hat = self.v_b / (1 - self.beta2 ** self.t)
        # self.biases -= learning_rate * m_b_hat / (np.sqrt(v_b_hat) + self.epsilon)

        # grad_input = np.dot(grad_output, self.weights.T)
        grad_input = np.dot(grad_output, self.weights.T)
        self.weights -= learning_rate * grad_weights
        self.biases -= learning_rate * grad_biases
        return grad_input

    def relu(self, layer):
        return np.maximum(0.0, layer)

    def def_relu(self, layer):
        return np.where(layer > 0, 1, np.finfo(float).eps)

    def sigmoid(self, layer):
        clipped_layer = np.clip(layer, -500, 500)
        return 1 / (1 + np.exp(-clipped_layer))

    def def_sigmoid(self, layer):
        return self.sigmoid(layer) * (1 - self.sigmoid(layer))

class BatchNormalizationLayer:
    def __init__(self, input_shape, epsilon=1e-8):
        self.epsilon = epsilon
        self.gamma = 1.0
        self.beta = 0.0
        self.running_mean = None
        self.running_var = None
        self.batch_mean = None
        self.batch_var = None
        self.x_norm = None

    def forward(self, x, training=True):
        if self.running_mean is None:
            if x is None:
                return None
            self.running_mean = np.zeros(x.shape[1:])
            self.running_var = np.zeros(x.shape[1:])
        self.x = x
        if training:
            self.batch_mean = np.mean(x, axis=0)
            self.batch_var = np.var(x, axis=0)
            self.x_norm = (x - self.batch_mean) / np.sqrt(self.batch_var + self.epsilon)

            self.running_mean = 0.9 * self.running_mean + 0.1 * self.batch_mean
            self.running_var = 0.9 * self.running_var + 0.1 * self.batch_var
        else:
            self.x_norm = (x - self.running_mean) / np.sqrt(self.running_var + self.epsilon)
        if self.gamma is None or self.beta is None:
            out = None
        else:
            out = self.gamma * self.x_norm + self.beta
        # print('BatchNormalizationLayer output shape:', out.shape)
        return out

    def backward(self, dout, learning_rate):
        dx_norm = dout * self.gamma
        dvar = np.sum(dx_norm * (self.x_norm - self.batch_mean), axis=0) * -0.5 * (self.batch_var + self.epsilon) ** (-1.5)
        dmean = np.sum(dx_norm * -1 / np.sqrt(self.batch_var + self.epsilon),
                       axis=0) + dvar * np.mean(-2 * (self.x_norm - self.batch_mean), axis=0)
        dx = (dx_norm / np.sqrt(self.batch_var + self.epsilon)) + (
                dvar * 2 * (self.x_norm - self.batch_mean) / self.x.shape[0]) + (dmean / self.x.shape[0])

        self.dgamma = np.sum(dout * self.x_norm, axis=0)
        self.dbeta = np.sum(dout, axis=0)

        return dx

    def set_trainable_params(self, d):
        self.gamma = np.ones(d)
        self.beta = np.zeros(d)

    def get_trainable_params(self):
        return self.gamma, self.beta

class FlattenLayer:
    def __init__(self):
        self.input_shape = None
        self.flattened_size = None

    def forward(self, inputs):
        self.input_shape = inputs.shape
        batch_size = inputs.shape[0]
        self.flattened_size = np.prod(inputs.shape[1:])
        flattened_inputs = inputs.reshape(batch_size, self.flattened_size)
        # print('FlattenLayer', inputs.shape, flattened_inputs.shape)
        return flattened_inputs

    def backward(self, grad_output, learning_rate):
        grad_input = grad_output.reshape(self.input_shape)
        return grad_input

class ConvolutionalLayer:
    def __init__(self, input_shape, num_filters, filter_size, stride=1, padding='valid', activation='relu'):
        self.input_shape = input_shape
        print('ConvolutionalLayer\nself.input_shape', self.input_shape)
        self.stride = stride
        self.num_filters = num_filters
        self.filter_size = filter_size
        self.padding = padding
        self.activation = activation
        self.weights = np.random.randn(filter_size, filter_size, input_shape[2], num_filters)
        self.biases = np.ones(num_filters)

    def forward(self, inputs):
        self.inputs = inputs
        batch_size, input_height, input_width, input_channels = inputs.shape

        if self.padding == 'same':
            pad_height = ((input_height - 1) * self.stride + self.filter_size - input_height) // 2
            pad_width = ((input_width - 1) * self.stride + self.filter_size - input_width) // 2
            padded_inputs = np.pad(inputs, ((0, 0), (pad_height, pad_height), (pad_width, pad_width), (0, 0)),
                                   mode='constant')
        else:
            padded_inputs = inputs

        output_height = (padded_inputs.shape[1] - self.filter_size) // self.stride + 1
        output_width = (padded_inputs.shape[2] - self.filter_size) // self.stride + 1
        self.output = np.zeros((batch_size, output_height, output_width, input_channels, self.num_filters), dtype=np.float64)

        for k in range(self.num_filters):
            for c in range(input_channels):
                for i in range(output_height):
                    for j in range(output_width):
                        input_slice = padded_inputs[:, i * self.stride:i * self.stride + self.filter_size,
                                      j * self.stride:j * self.stride + self.filter_size, :]
                        self.output[:, i, j, c, k] = np.sum(input_slice * self.weights[:, :, :, k], axis=(1, 2, 3))
            self.output[:, :, :, :, k] += self.biases[k]
        if self.activation == 'relu':
            self.output = self.relu(self.output)

        # print('ConvolutionalLayer output', self.output.shape)
        return self.output

    def backward(self, grad_output, learning_rate):
        batch_size, output_height, output_width, input_channels, _ = grad_output.shape
        if self.activation == 'relu':
            grad_output = self.relu_derivative(self.output) * grad_output
        grad_inputs = np.zeros_like(self.inputs)
        grad_weights = np.zeros_like(self.weights)
        grad_biases = np.zeros_like(self.biases)
        for k in range(self.num_filters):
            for c in range(input_channels):
                for i in range(output_height):
                    for j in range(output_width):
                        input_slice = self.inputs[:, i * self.stride:i * self.stride + self.filter_size,
                                      j * self.stride:j * self.stride + self.filter_size, :]

                        grad_inputs[:, i * self.stride:i * self.stride + self.filter_size,
                        j * self.stride:j * self.stride + self.filter_size, :] += np.sum(
                            grad_output[:, i, j, :, k][:, np.newaxis, np.newaxis, np.newaxis] *
                            self.weights[:, :, :, k][np.newaxis, :, :, c],
                            axis=3)
                        grad_weights[:, :, :, k] += np.sum(
                            input_slice[:, :, :, c][:, :, :, np.newaxis] *
                            grad_output[:, i, j, :, k][:, np.newaxis, np.newaxis, np.newaxis],
                            axis=(0, 1, 2, 3))
                grad_biases[k] += np.sum(grad_output[:, :, :, :, k])

        if self.padding == 'same':
            pad_height = ((self.input_shape[0] - 1) * self.stride + self.filter_size - self.input_shape[0]) // 2
            pad_width = ((self.input_shape[1] - 1) * self.stride + self.filter_size - self.input_shape[1]) // 2
            grad_inputs = grad_inputs[:, pad_height:self.input_shape[0] + pad_height,
                          pad_width:self.input_shape[1] + pad_width, :]

        self.weights -= learning_rate * grad_weights
        self.biases -= learning_rate * grad_biases
        return grad_inputs

    def relu(self, layer):
        return np.maximum(0.0, layer)

    def relu_derivative(self, layer):
        return np.where(layer > 0, 1, np.finfo(float).eps)

class MaxPoolingLayer:
    def __init__(self, pool_size=2, stride=2):
        print("MaxPoolingLayer")
        self.pool_size = pool_size
        self.stride = stride
        self.inputs = None  # Добавленный атрибут inputs

    def forward(self, inputs):
        self.inputs = inputs  # Установка значения атрибута inputs
        batch_size, input_height, input_width, input_channels = inputs.shape
        output_height = (input_height - self.pool_size) // self.stride + 1
        output_width = (input_width - self.pool_size) // self.stride + 1
        output_channels = input_channels
        output_shape = (batch_size, output_height, output_width, output_channels)
        outputs = np.zeros(output_shape)

        for i in range(output_height):
            for j in range(output_width):
                h_start = i * self.stride
                h_end = h_start + self.pool_size
                w_start = j * self.stride
                w_end = w_start + self.pool_size
                inputs_slice = inputs[:, h_start:h_end, w_start:w_end, :]
                outputs[:, i, j, :] = np.amax(inputs_slice, axis=(1, 2))
        print('MaxPoolingLayer', outputs.shape)
        return outputs

    def backward(self, grad_output, learning_rate):
        batch_size, output_height, output_width, output_channels = grad_output.shape
        grad_inputs = np.zeros_like(self.inputs)

        for i in range(output_height):
            for j in range(output_width):
                h_start = i * self.stride
                h_end = h_start + self.pool_size
                w_start = j * self.stride
                w_end = w_start + self.pool_size
                inputs_slice = self.inputs[:, h_start:h_end, w_start:w_end, :]
                max_values = np.amax(inputs_slice, axis=(1, 2), keepdims=True)
                mask = (inputs_slice == max_values)
                grad_inputs[:, h_start:h_end, w_start:w_end, :] += mask * grad_output[:, i:i+1, j:j+1, :]

        return grad_inputs

class UpSamplingLayer:
    def __init__(self, scale_factor=2):
        print("UpSamplingLayer")
        self.scale_factor = scale_factor
        self.inputs = None  # Добавленный атрибут inputs

    def forward(self, inputs):
        self.inputs = inputs  # Установка значения атрибута inputs
        batch_size, input_height, input_width, input_channels = inputs.shape
        output_height = input_height * self.scale_factor
        output_width = input_width * self.scale_factor
        output_channels = input_channels
        output_shape = (batch_size, output_height, output_width, output_channels)
        outputs = np.zeros(output_shape)

        for i in range(output_height):
            for j in range(output_width):
                outputs[:, i, j, :] = inputs[:, i // self.scale_factor, j // self.scale_factor, :]

        return outputs

    def backward(self, grad_output, learning_rate):
        batch_size, output_height, output_width, output_channels = grad_output.shape
        grad_inputs = np.zeros_like(self.inputs)

        for i in range(output_height):
            for j in range(output_width):
                grad_inputs[:, i // self.scale_factor, j // self.scale_factor, :] += grad_output[:, i, j, :]

        return grad_inputs

class ReshapeLayer:
    def __init__(self, new_shape):
        print("ReshapeLayer")
        self.new_shape = new_shape
        self.input_shape = None

    def forward(self, inputs):
        self.input_shape = inputs.shape
        return np.reshape(inputs, self.new_shape)

    def backward(self, grad_output, learning_rate):
        print("ReshapeLayer")
        return np.reshape(grad_output, self.input_shape)

class LambdaLayer:
    def __init__(self, function):
        self.function = function
        self.output = None

    def forward(self, inputs):
        self.output = self.function(inputs)
        return self.output

    def backward(self, grad_output, learning_rate):
        # Lambda layers do not have any parameters to update,
        # so the backward pass does not involve any computations.
        return grad_output

class ConcatenateLayer:
    def __init__(self, layers):
        self.layers = layers

    def forward(self, inputs):
        layer_outputs = [layer.forward(inputs) for layer in self.layers]
        return np.concatenate(layer_outputs, axis=-1)

    def backward(self, grad_output, learning_rate):
        split_sizes = [layer.output.shape[-1] for layer in self.layers]
        grad_inputs = np.split(grad_output, split_sizes, axis=-1)

        for layer, grad_input in zip(self.layers, grad_inputs):
            layer.backward(grad_input, learning_rate)

        return grad_output

class ConvolutionalNetwork:
    def __init__(self):
        self.layers = []
        self.layer_names = {}

    def add_layer(self, layer, name=None):
        self.layers.append(layer)
        if name is not None:
            self.layer_names[name] = layer

    def add_lambda_layer(self, function, name=None):
        lambda_layer = LambdaLayer(function)
        self.add_layer(lambda_layer, name)
        if name is not None:
            self.layer_names[name] = lambda_layer
        return lambda_layer

    def forward(self, inputs):
        print('forward')
        for layer in self.layers:
            inputs = layer.forward(inputs)
        return inputs

    def backward(self, grad_output, learning_rate):
        for layer in reversed(self.layers):
            grad_output = layer.backward(grad_output, learning_rate)

    def fit(self, X_train, y_train, learning_rate, num_epochs, batch_size):
        total_batches = (len(X_train) + batch_size - 1) // batch_size

        for epoch in range(num_epochs):
            print(f"Epoch {epoch + 1}/{num_epochs}")
            for batch_start in range(0, len(X_train), batch_size):
                batch_end = min(batch_start + batch_size, len(X_train))
                X_batch = X_train[batch_start:batch_end]
                y_batch = y_train[batch_start:batch_end]
                y_pred = self.forward(X_batch)
                print('y_pred', y_pred.shape)
                print('y_batch', y_batch.shape)
                grad_output = (y_pred - y_batch) / len(y_batch)

                self.backward(grad_output, learning_rate)
                loss = self.loss(X_batch, y_batch)
                print(f"Batch {batch_start // batch_size + 1}/{total_batches}, Loss: {loss}")

    def loss(self, X_test, y_test):
        y_pred = self.forward(X_test)
        loss = np.mean(y_pred - y_test)
        return loss

    def save_model(self, file_path):
        model_params = {
            'layers': self.layers,
            'layer_names': self.layer_names,
        }
        with open(file_path, 'wb') as f:
            pickle.dump(model_params, f)

    @staticmethod
    def load_model(file_path):
        with open(file_path, 'rb') as f:
            model_params = pickle.load(f)
        model = ConvolutionalNetwork()
        model.layers = model_params['layers']
        model.layer_names = model_params['layer_names']
        model.encoder_layers = model_params['encoder_layers']
        model.decoder_layers = model_params['decoder_layers']
        return model

    def concatenate(self, layers, name=None):
        concat_layer = ConcatenateLayer(layers)
        self.add_layer(concat_layer, name)

