# coding=utf-8
"""Methods to reproduce normalizing flows paper distributions experiments."""
from keras import activations
from keras import backend
from keras import callbacks
from keras import initializers
from keras import layers
from keras import models
from keras import optimizers
from matplotlib import pyplot
import numpy as np

import flows


def w_1(z_in):
    """Return w1 defined in Table 1."""
    return np.sin(2.0 * np.pi * z_in[:, 0] / 4.0)


def w_1k(z_in):
    """Return w1 defined in Table 1 for backend."""
    return backend.sin(2.0 * np.pi * z_in[:, 0] / 4.0)


def w_2(z_in):
    """Return w2 defined in Table 1."""
    return 3.0 * np.exp(-0.5 * np.square((z_in[:, 0] - 1) / 0.6))


def w_2k(z_in):
    """Return w2 defined in Table 1 for backend."""
    return 3.0 * backend.exp(-0.5 * backend.square((z_in[:, 0] - 1) / 0.6))


def w_3(z_in):
    """Return w3 defined in Table 1."""
    return 3.0 / (1 + np.exp((z_in[:, 0] - 1) / -0.3))


def w_3k(z_in):
    """Return w3 defined in Table 1 for backend."""
    return 3.0 * backend.sigmoid((z_in[:, 0] - 1) / 0.3)


def pot1(z_in):
    """Energy potential 1."""
    return (.5 * np.square((np.sqrt(np.sum(z_in * z_in, axis=-1)) - 2) / 0.4) -
            np.log(np.exp(-0.5 * np.square((z_in[:, 0] - 2) / 0.6)) +
                   np.exp(-0.5 * np.square((z_in[:, 0] + 2) / 0.6))))


def pot1k(z_in):
    """Energy potential 1 for backend."""
    first = backend.square(
        (backend.sqrt(backend.sum(z_in * z_in, axis=-1)) - 2) / 0.4)
    return (0.5 * first - backend.log(
        backend.exp(-0.5 * backend.square((z_in[:, 0] - 2) / 0.6)) +
        backend.exp(-0.5 * backend.square((z_in[:, 0] + 2) / 0.6))))


def pot2(z_in):
    """Energy potential 2."""
    return 0.5 * np.square((z_in[:, 1] - w_1(z_in)) / 0.4)


def pot2k(z_in):
    """Energy potential 2 for backend."""
    return 0.5 * backend.square((z_in[:, 1] - w_1k(z_in)) / 0.4)


def pot3(z_in):
    """Energy potential 3."""
    first = z_in[:, 1] - w_1(z_in)
    return -np.log(np.exp(-0.5 * np.square(first / 0.35)) +
                   np.exp(-0.5 * np.square((first + w_2(z_in)) / 0.35)))


def pot3k(z_in):
    """Energy potential 3 for backend."""
    first = z_in[:, 1] - w_1k(z_in)
    return -backend.log(
        backend.exp(-0.5 * backend.square(first / 0.35)) +
        backend.exp(-0.5 * backend.square((first + w_2k(z_in)) / 0.35)))


def pot4(z_in):
    """Energy potential 4."""
    first = z_in[:, 1] - w_1(z_in)
    return -np.log(np.exp(-0.5 * np.square(first / 0.4)) +
                   np.exp(-0.5 * np.square((first + w_3(z_in)) / 0.35)))


def pot4k(z_in):
    """Energy potential 4 for backend."""
    first = z_in[:, 1] - w_1k(z_in)
    return -backend.log(
        backend.exp(-0.5 * backend.square(first / 0.4)) +
        backend.exp(-0.5 * backend.square((first + w_3k(z_in)) / 0.35)))


class Parameters(layers.Layer):
    """One input layer that gives trainable parameters.

    # Arguments
        flow_steps: Integer, number of flow steps to perform
        latent_dim: Integer, number of dimensions of latent space
        initializer: Initializer for the parameters
    # Input shape
        nD tensor with shape: `(batch_size)`.
    # Output shape
        nD tensor with shape: `(batch_size, flow_steps * (2 * latent_dim + 1) +
                                            2 * latent_dim)`.
    """

    def __init__(self, flow_steps,
                 latent_dim,
                 initializer='glorot_uniform',
                 **kwargs):
        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('input_dim'),)
        super(Parameters, self).__init__(**kwargs)
        self.flow_steps = flow_steps
        self.latent_dim = latent_dim
        self.initializer = initializers.get(initializer)
        self.input_spec = layers.InputSpec(min_ndim=1)
        self.supports_masking = True
        self.parameters = None

    def build(self, input_shape):
        assert len(input_shape) >= 1
        q0_mu = self.add_weight(
            shape=(self.latent_dim,),
            initializer=self.initializer, name='q0_mu')
        q0_logsigma2 = self.add_weight(
            shape=(self.latent_dim,),
            initializer=self.initializer, name='q0_logsigma2')
        q_u = self.add_weight(
            shape=(self.flow_steps, self.latent_dim),
            initializer=self.initializer, name='q_u')
        q_w = self.add_weight(
            shape=(self.flow_steps, self.latent_dim),
            initializer=self.initializer, name='q_w')
        q_b = self.add_weight(
            shape=(self.flow_steps,),
            initializer=self.initializer, name='q_b')
        self.parameters = backend.concatenate([
            q0_mu, q0_logsigma2, backend.flatten(q_u), backend.flatten(q_w),
            q_b], axis=0)
        self.input_spec = layers.InputSpec(min_ndim=1)
        self.built = True

    def call(self, inputs, **kwargs):
        """This is where the layer's logic lives.
        # Arguments
            inputs: Input tensor, or list/tuple of input tensors.
            **kwargs: Additional keyword arguments.
        # Returns
            A tensor or list/tuple of tensors.
        """
        # Copy the parameters for the number of inputs
        return backend.ones_like(inputs) * self.parameters

    def compute_output_shape(self, input_shape):
        assert input_shape and len(input_shape) >= 1
        return (
            input_shape[0],
            self.flow_steps * (2 * self.latent_dim + 1) + 2 * self.latent_dim)

    def get_config(self):
        config = {
            'flow_steps': self.flow_steps,
            'latent_dim': self.latent_dim,
            'initializer': initializers.serialize(self.initializer)
        }
        base_config = super(Parameters, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


def fit_model(flow_steps, potential_fn, learning_rate=1e-5,
              parameter_updates=500000, batch_size=100, optimizer='rmsprop',
              use_temperature=True, min_temperature=0.01, max_temperature=1.0,
              temperature_steps=10000):
    """Performs the building and training of the model to test."""
    input_layer = layers.Input(shape=(1,))  # Random input
    output_layer = Parameters(flow_steps, 2)(input_layer)  # Flow parameters
    output_layer = flows.PlanarFlow(flow_steps, 2,  # 2 latent dimensions
                                    activation=activations.tanh)(output_layer)
    model = models.Model(inputs=input_layer, outputs=output_layer)
    # Add likelihood loss, with temperature for first parameters updates
    temperature = backend.variable(1.0)
    model.add_loss(temperature * backend.mean(potential_fn(output_layer)))
    # Finish model
    model.compile(optimizer=getattr(optimizers, optimizer)(lr=learning_rate))
    model.summary()
    # If using temperature, then add callback to update it
    the_callbacks = None
    if use_temperature:
        def set_temperature(batch, _):
            """Set temperature for this batch."""
            if batch <= temperature_steps:
                temp_step = (max_temperature -
                             min_temperature) / temperature_steps
                backend.set_value(temperature,
                                  min_temperature + temp_step * batch)

        backend.set_value(temperature, min_temperature)
        the_callbacks = [callbacks.LambdaCallback(
            on_batch_begin=set_temperature)]
    # Train the model with random input, that is discarded
    model.fit(x=np.ones(batch_size * parameter_updates), epochs=1,
              batch_size=batch_size, callbacks=the_callbacks)
    return model


def draw_real_distribution(potential_fn, num_samples=300000):
    """Draw chart of real distribution based on potential, similar to paper."""
    sample = np.random.uniform(low=-4.0, high=4.0, size=(num_samples, 2))
    pyplot.figure(figsize=(4, 4))
    pyplot.scatter(sample[:, 0], sample[:, 1],
                   c=(np.exp(-potential_fn(sample))),
                   cmap='jet', marker='.', s=1)
    pyplot.gca().set_ylim(bottom=4.0, top=-4.0)
    pyplot.gca().set_xlim(left=-4.0, right=4.0)
    pyplot.axis('off')
    pyplot.savefig('original_distrib_{}.png'.format(potential_fn.__name__))


def draw_predicted_distribution(local_model, name, num_samples=2000000):
    """Draw heatmap of predicted distribution with given number of samples."""
    sample = local_model.predict(np.ones(num_samples))
    heatmap, xedges, yedges = np.histogram2d(sample[:, 0], sample[:, 1],
                                             bins=(150, 150))
    pyplot.figure(figsize=(4, 4))
    pyplot.imshow(heatmap.T,
                  extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
                  cmap='jet')
    pyplot.gca().set_xlim(left=-4.0, right=4.0)
    pyplot.gca().set_ylim(bottom=4.0, top=-4.0)
    pyplot.axis('off')
    pyplot.savefig('predicted_distrib_{}.png'.format(name))


def main():
    """Main execution point."""
    draw_real_distribution(pot1)
    model = fit_model(8, pot1k, learning_rate=0.0001,
                      parameter_updates=50000, optimizer='adam')
    draw_predicted_distribution(model, 'pot1')


if __name__ == '__main__':
    main()
