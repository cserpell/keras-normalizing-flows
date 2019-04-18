# keras-normalizing-flows
Keras layer that implements normalizing flows.

## Description

A basic implementation of normalizing flows is
available in file `flows.py`. The layer receives
the parameters as flattened input, applies transformations and
outputs latent variables values.

Note that this implementation produces the random samples
within the layer, and it does not receives them as
input. This can be changed easily.

Currently, planar flows are available,
with `tanh` or linear activations.

## Example

In file `reproducing_paper.py`,
there is a reproduction of distribution estimation
of [1] paper using this
implementation. In this case, the same parameters are
used for all samples, as there is no inference going on.

The example allows trying different values for parameters
just changing the call to `fit_model`. For example:

- Optimizer
- Learning rate
- Minibatch size
- Number of parameter updates
- Number of flow steps
- Use of initial increasing temperature (with minimum and maximum value, and number of parameter updates to interpolate)

Nice pictures are generated using many random samples, as
seen below.

![distribution to approximate](https://github.com/cserpell/keras-normalizing-flows/raw/master/pot1_real.png "Distribution to approximate")
![8 steps no temperature](https://github.com/cserpell/keras-normalizing-flows/raw/master/pot1_predicted8_notemp.png "8 steps without temperature")
![8 steps and temperature](https://github.com/cserpell/keras-normalizing-flows/raw/master/pot1_predicted8_temp.png "8 steps with temperature")

[1] Rezende, D. J., Mohamed, S. (2015).
Variational Inference with Normalizing Flows.
In F. Bach &#38; D. Blei (Eds.),
<i>Proceedings of the
32nd International Conference on Machine Learning</i>
(Vol. 37, pp. 1530-1538).
Lille, France: PMLR.
https://doi.org/10.1113/jphysiol.2003.055525

## Questions

Don't hesitate to ask, and contributions are welcome!