# coding=utf-8
"""Keras layer to implement normalizing flows."""
from keras import activations
from keras import backend
from keras import layers


class Flow(layers.Layer):
    """Abstract layer that performs normalizing flow and adds variational loss.
    # Arguments
        flow_steps: Integer, number of flow steps to perform
        latent_dim: Integer, number of dimensions of latent space
        activation: Activation function to use
            If you don't specify anything, no activation is applied
            (ie. "linear" activation: `a(x) = x`).
    # Input shape
        (batch_size,
         parameters for flow flattened as given by kind of transformations).
    # Output shape
        (batch_size, latent_dim).
    """

    def __init__(self, flow_steps, latent_dim, activation=None, **kwargs):
        super(Flow, self).__init__(**kwargs)
        self.activation = activations.get(activation)
        self.flow_steps = flow_steps
        self.latent_dim = latent_dim
        self.supports_masking = True

    def compute_output_shape(self, input_shape):
        assert input_shape and len(input_shape) >= 2
        assert input_shape[-1]
        return tuple(list(input_shape[:-1]) + [self.latent_dim])

    def get_config(self):
        config = {
            'flow_steps': self.flow_steps,
            'latent_dim': self.latent_dim,
            'activation': activations.serialize(self.activation)
        }
        base_config = super(Flow, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class PlanarFlow(Flow):
    """Layer that performs normalizing flow with planar transformations.
    # Arguments
        flow_steps: Integer, number of flow steps to perform
        latent_dim: Integer, number of dimensions of latent space
        activation: Activation function to use (only tanh and None supported!)
            If you don't specify anything, no activation is applied
            (ie. "linear" activation: `a(x) = x`).
    # Input shape
        (batch_size,
         parameters for flow flattened: [q0_mu, q0_logsigma2, q_u, q_w, q_b]
         Total: flow_steps * (2 * latent_dim + 1) + 2 * latent_dim).
    # Output shape
        (batch_size, latent_dim).
    """

    def call(self, inputs, **kwargs):
        """This is where the layer's logic lives.
        # Arguments
            inputs: Input tensor, or list/tuple of input tensors.
            **kwargs: Additional keyword arguments.
        # Returns
            A tensor or list/tuple of tensors.
        """
        # q0_mu, q0_sigma and z_0 have (samples, latent_dim) shape
        q0_mu = inputs[:, 0:self.latent_dim]
        q0_logsigma2 = inputs[:, self.latent_dim:2 * self.latent_dim]
        q0_sigma = backend.exp(0.5 * q0_logsigma2)
        z_shape = backend.shape(q0_mu)
        sample = backend.random_normal(shape=z_shape)
        z_0 = q0_mu + sample * q0_sigma
        # First loss: ln(q0(z0)) = -[sample^2 + ln sigma0^2] / 2 =
        # -[(z0 - mu0)^2 / sigma0^2 + ln sigma0^2] / 2
        self.add_loss(-0.5 * (backend.mean(
            backend.sum(backend.square(sample), axis=-1) +
            backend.sum(q0_logsigma2, axis=-1))))
        for step in range(0, self.flow_steps):
            # Here q_u, q_w have the same shape as z_0, as they are one step
            q_u = inputs[:,
                         (2 + step) * self.latent_dim:
                         (2 + step + 1) * self.latent_dim]
            q_u = backend.reshape(q_u, z_shape)
            q_w = inputs[:,
                         (2 + self.flow_steps + step) * self.latent_dim:
                         (2 + self.flow_steps + step + 1) * self.latent_dim]
            q_w = backend.reshape(q_w, z_shape)
            q_b = inputs[:, (2 + 2 * self.flow_steps) * self.latent_dim + step]
            h_activation = backend.sum(q_w * z_0, axis=-1) + q_b
            # Second loss: -ln abs(det(d_fk/d_z (z_k-1))) =
            # -ln abs(1 + uk^T psi_k(z_k-1)) =
            # -ln abs(1 + uk^T h'(wk^T z_k-1 + bk) wk)
            if self.activation is not None:
                # TODO: Allow activation functions other than tanh
                # = -ln abs(1 + (1 - tanh(wk^T z_k-1 + bk)^2) uk^T wk)
                self.add_loss(-backend.mean(backend.log(backend.abs(
                    (1 - backend.square(backend.tanh(h_activation))) *
                    backend.sum(q_u * q_w, axis=-1) + 1))))
                h_activation = self.activation(h_activation)
            else:
                # = -ln abs(1 + uk^T wk)
                self.add_loss(-backend.mean(backend.log(backend.abs(
                    backend.sum(q_u * q_w, axis=-1) + 1))))
            # Now apply the transformation to z_0
            z_0 += q_u * backend.reshape(backend.repeat_elements(
                backend.expand_dims(h_activation), self.latent_dim,
                1), z_shape)
        return z_0
