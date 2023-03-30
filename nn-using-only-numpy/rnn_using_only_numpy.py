import math
import numpy as np
from computational_utils import Utilities

'''
Implementing RNN with only numpy, not using any other frameworks (such as PyTorch or Tensorflow)

Purpose of doing this:
  * Help myself go back to the ML basics and get better understanding, rather than using wrapped libraries.
  * Using this code to clearly explain what happens in an RNN to help others.
  * Performance optimization is not a goal, code clarity is more important.
'''
class RnnWithNumpy:
    def __init__(self, dim_vocab, dim_hidden):
        self.dim_vocab = dim_vocab
        self.dim_hidden = dim_hidden

        # Matrix U is the one that transforms input one-hot vector into its embedding.
        self.matrix_u = np.random.uniform(-1, 1, dim_hidden * dim_vocab).reshape(dim_hidden, dim_vocab)

        # Transforms post-activation (tanh) state into logits (output embedding)
        self.matrix_v = np.random.uniform(-1, 1, dim_hidden * dim_vocab).reshape(dim_vocab, dim_hidden)

        # Transforms previous state (s[t-1]) into part of next state, before activation.
        self.matrix_w = np.random.uniform(-1, 1, dim_hidden * dim_hidden).reshape(dim_hidden, dim_hidden)

        self.prev_state = np.zeros(dim_hidden)


    @staticmethod
    def forward(input_x_as_integer, dim_vocab, dim_hidden, matrix_u, matrix_w, matrix_v, prev_state,
        check_shapes=True, print_debug=False):
        if check_shapes:
            assert isinstance(input_x_as_integer, int)
            assert matrix_u.ndim == 2
            assert matrix_v.ndim == 2
            assert matrix_w.ndim == 2
            assert matrix_u.shape[0] == dim_hidden and matrix_u.shape[1] == dim_vocab
            assert matrix_v.shape[0] == dim_vocab and matrix_v.shape[1] == dim_hidden
            assert matrix_w.shape[0] == matrix_w.shape[1] == dim_hidden
            assert prev_state.ndim == 1
            assert prev_state.size == dim_hidden
            assert input_x_as_integer >= 0 and input_x_as_integer < dim_vocab

        matrix_u_times_x_onehot = matrix_u[:, input_x_as_integer]
        w_times_prev_state = np.matmul(matrix_w, prev_state)
        current_state_before_activation = matrix_u_times_x_onehot + w_times_prev_state
        current_state = np.tanh(current_state_before_activation)
        logits = np.matmul(matrix_v, current_state)
        probabilities = Utilities.softmax(logits)

        if check_shapes:
            assert probabilities.ndim == 1 and probabilities.size == dim_vocab
            assert current_state.ndim == 1 and current_state.size == dim_hidden

        forward_computation_outputs = (
            matrix_u_times_x_onehot, w_times_prev_state, current_state_before_activation,
            current_state, logits, probabilities)

        return forward_computation_outputs


    @staticmethod
    def predict(dim_vocab, probabilities, check_shapes=True):
        if check_shapes:
            assert probabilities.ndim == 1 and probabilities.size == dim_vocab

        return np.argmax(probabilities)


    @staticmethod
    def loss(dim_vocab, probabilities, y_label_as_integer, check_shapes=True):
        if check_shapes:
            assert probabilities.ndim == 1 and probabilities.size == dim_vocab
            assert isinstance(y_label_as_output, int)
            assert y_label_as_integer >= 0 and y_label_as_integer < dim_vocab

        return - math.log(probabilities[y_label_as_integer])

    def gradient(self):
        pass


    def step(self, gradient, step_size):
        pass