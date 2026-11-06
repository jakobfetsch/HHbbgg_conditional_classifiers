import numpy as np
import tensorflow as tf
from tensorflow.keras import layers


class PytorchLinearUniform(tf.keras.initializers.Initializer):
    """
    Reproduces PyTorch's default `nn.Linear` reset_parameters():

        kaiming_uniform_(weight, a=sqrt(5))   -> Uniform(-1/sqrt(fan_in), 1/sqrt(fan_in))
        bias ~ Uniform(-1/sqrt(fan_in), 1/sqrt(fan_in))   (fan_in taken from weight.shape)

    Both the kernel and the bias reduce to the *same* bound, 1/sqrt(fan_in),
    so this one class is used for both `kernel_initializer` and
    `bias_initializer`. `fan_in` is fixed at construction time to the
    layer's input dimension, matching PyTorch (which always computes it
    from the weight tensor, never from the bias tensor's own shape).

    Note: this matches the *distribution*, not the literal random values -
    TF and PyTorch use different RNG algorithms, so seeding both the same
    way will not produce identical initial weights, only the same
    statistics.
    """

    def __init__(self, fan_in, seed=None):
        self.fan_in = fan_in
        self.seed = seed

    def __call__(self, shape, dtype=None):
        limit = 1.0 / np.sqrt(self.fan_in)
        return tf.random.uniform(
            shape, minval=-limit, maxval=limit,
            dtype=dtype or tf.float32, seed=self.seed,
        )

    def get_config(self):
        return {"fan_in": self.fan_in, "seed": self.seed}


# Maps a PyTorch-style activation name (e.g. "ELU", "ReLU", "GELU", "Tanh")
# to a callable TF/Keras activation layer, so callers can keep passing the
# same act_fn_name string they used with `getattr(nn, name)`.
def get_activation_layer(act_fn_name):
    name = act_fn_name.lower()
    mapping = {
        "relu": layers.ReLU,
        "elu": layers.ELU,
        "leakyrelu": layers.LeakyReLU,
        "gelu": lambda: layers.Activation("gelu"),
        "tanh": lambda: layers.Activation("tanh"),
        "sigmoid": lambda: layers.Activation("sigmoid"),
        "softplus": lambda: layers.Activation("softplus"),
    }
    if name not in mapping:
        raise ValueError(f"Unsupported activation function: {act_fn_name}")
    return mapping[name]


class MLP(tf.keras.Model):
    """
    TensorFlow/Keras equivalent of the original PyTorch MLP (mlp.py):

        layers = []
        layers.append(nn.Linear(input_size, num_nodes))
        layers.append(act_fn())
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(num_nodes, num_nodes))
            layers.append(act_fn())
            layers.append(nn.Dropout(p=dropout_prob))
        layers.append(nn.Linear(num_nodes, output_size))

    i.e.: input layer has NO dropout after it; dropout only follows each of
    the (num_layers - 1) hidden layers; the output layer is a plain Dense
    with no activation (raw logits, matching nn.CrossEntropyLoss's
    expectation of unnormalized scores).

    Mirrors the constructor signature used in training_utils.py:
        MLP(input_size, num_layers, num_nodes, output_size, act_fn, dropout_prob)

    seed (optional): if given, every Dense layer's kernel/bias draw from
    their own PytorchLinearUniform(seed=seed + layer_index) instead of TF's
    global random state. This makes the weight initialization reproducible
    from this one seed alone - independent of tf.random.set_seed() and of
    any other seed (data sampling/split, batch shuffling) used elsewhere in
    a run, so it can be varied on its own to try different initial states.
    """

    def __init__(self, input_size, num_layers, num_nodes, output_size, act_fn, dropout_prob, seed=None):
        super().__init__()

        # act_fn can be either an activation name string (e.g. "ELU") or
        # already a Keras activation-layer factory; handle both.
        if isinstance(act_fn, str):
            activation_factory = get_activation_layer(act_fn)
        else:
            activation_factory = act_fn

        self._next_layer_seed = seed

        def pytorch_dense(units, fan_in):
            layer_seed = self._next_layer_seed
            if layer_seed is not None:
                self._next_layer_seed += 1
            init = PytorchLinearUniform(fan_in, seed=layer_seed)
            return layers.Dense(units, kernel_initializer=init, bias_initializer=init)

        self.model_layers = []

        # Input layer (no dropout after it, matching mlp.py)
        self.model_layers.append(pytorch_dense(num_nodes, fan_in=input_size))
        self.model_layers.append(activation_factory())

        # Hidden layers
        for _ in range(num_layers - 1):
            self.model_layers.append(pytorch_dense(num_nodes, fan_in=num_nodes))
            self.model_layers.append(activation_factory())
            self.model_layers.append(layers.Dropout(dropout_prob))

        # Output layer
        self.model_layers.append(pytorch_dense(output_size, fan_in=num_nodes))

        # Build the model *properly* by running one real forward pass,
        # instead of calling self.build(input_shape) (see note below).
        # This creates every sub-layer's variables through the normal
        # __call__ path, so Keras marks the model built correctly.
        self(tf.zeros((1, input_size), dtype=tf.float32), training=False)

    def call(self, x, training=False):
        for layer in self.model_layers:
            if isinstance(layer, layers.Dropout):
                x = layer(x, training=training)
            else:
                x = layer(x)
        return x