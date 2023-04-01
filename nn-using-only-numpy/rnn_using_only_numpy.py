import math
import numpy as np
from computational_utils import Utilities

'''
Implementing RNN with only numpy, not using any other frameworks (such as PyTorch or Tensorflow)

Purpose of doing this:
  * Help myself go back to the ML basics and get better understanding, rather than using wrapped libraries.
  * Using this code to clearly explain what happens in an RNN to help others.
  * Performance optimization is not a goal, code clarity is more important.

  NOTES: there is no bias term here. Probably I should add bias some time later.
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

        self.prev_state_vector = np.zeros(dim_hidden)


    @staticmethod
    def forward(input_x_as_integer, dim_vocab, dim_hidden, matrix_u, matrix_w, matrix_v, prev_state_vector,
        check_shapes=True, print_debug=False):
        if check_shapes:
            assert isinstance(input_x_as_integer, int)
            assert matrix_u.ndim == 2
            assert matrix_v.ndim == 2
            assert matrix_w.ndim == 2
            assert matrix_u.shape[0] == dim_hidden and matrix_u.shape[1] == dim_vocab
            assert matrix_v.shape[0] == dim_vocab and matrix_v.shape[1] == dim_hidden
            assert matrix_w.shape[0] == matrix_w.shape[1] == dim_hidden
            assert prev_state_vector.ndim == 1
            assert prev_state_vector.size == dim_hidden
            assert input_x_as_integer >= 0 and input_x_as_integer < dim_vocab

        matrix_u_times_x_onehot = matrix_u[:, input_x_as_integer]
        w_times_prev_state = np.matmul(matrix_w, prev_state_vector)
        current_state_before_activation = matrix_u_times_x_onehot + w_times_prev_state
        current_state = np.tanh(current_state_before_activation)
        logits = np.matmul(matrix_v, current_state)
        softmax_probabilities = Utilities.softmax(logits)

        if check_shapes:
            assert softmax_probabilities.ndim == 1 and softmax_probabilities.size == dim_vocab
            assert current_state.ndim == 1 and current_state.size == dim_hidden

        forward_computation_intermediates = (
            input_x_as_integer,
            matrix_u, matrix_v, matrix_w, prev_state_vector
            matrix_u_times_x_onehot, w_times_prev_state, current_state_before_activation,
            current_state, logits, softmax_probabilities)

        return forward_computation_intermediates


    @staticmethod
    def predict_with_softmax_output(dim_vocab, softmax_probabilities, check_shapes=True, print_debug=False):
        if check_shapes:
            assert softmax_probabilities.ndim == 1 and softmax_probabilities.size == dim_vocab

        return np.argmax(softmax_probabilities)


    def forward_and_predict(self, input_x_as_integer, check_shapes=True):
        if check_shapes:
            assert isinstance(input_x_as_integer, int)
            assert input_x_as_integer >= 0 and input_x_as_integer < self.dim_vocab

        forward_computation_intermediates = self.forward(
            input_x_as_integer, self.dim_vocab, self.dim_hidden, self.matrix_u, self.matrix_w, self.matrix_v, self.prev_state_vector,
            check_shapes=check_shapes, print_debug=print_debug)

        (input_x_as_integer,
            matrix_u, matrix_v, matrix_w, prev_state_vector
            matrix_u_times_x_onehot, w_times_prev_state, current_state_before_activation,
            current_state, logits, softmax_probabilities) = forward_computation_intermediates

        return self.predict_with_softmax_output(self.dim_vocab, softmax_probabilities, check_shapes=check_shapes, print_debug=print_debug)


    @staticmethod
    def loss(dim_vocab, softmax_probabilities, y_label_as_integer, check_shapes=True):
        if check_shapes:
            assert softmax_probabilities.ndim == 1 and softmax_probabilities.size == dim_vocab
            assert isinstance(y_label_as_output, int)
            assert y_label_as_integer >= 0 and y_label_as_integer < dim_vocab
            assert (softmax_probabilities >= 0).all() and (softmax_probabilities <= 1).all()

        return - math.log(softmax_probabilities[y_label_as_integer])


    @staticmethod
    def loss_gradient_u_v_w(forward_computation_intermediates, label_y_as_integer, dim_vocab, dim_hidden, check_shapes=True):
        (input_x_as_integer,
         matrix_u, matrix_v, matrix_w, prev_state,
         matrix_u_times_x_onehot, w_times_prev_state, current_state_before_activation,
            current_state, logits, probabilities) = forward_computation_intermediates

        if check_shapes:
            assert matrix_u_times_x_onehot.ndim == 1
            assert matrix_u_times_x_onehot.size == dim_hidden
            assert w_times_prev_state.ndim == 1
            assert w_times_prev_state.ndim.size == dim_hidden
            assert current_state_before_activation.ndim == 1
            assert current_state_before_activation.size == dim_hidden
            assert current_state.ndim == 1
            assert current_state.size == dim_hidden
            assert logits.ndim == 1
            assert logits.size == dim_vocab
            assert probabilities.ndim == 1
            assert probabilities.size == dim_vocab

            assert isinstance(input_x_as_integer, int)
            assert input_x_as_integer >= 0 and input_x_as_integer < dim_vocab
            assert isinstance(y_label_as_output, int)
            assert y_label_as_integer >= 0 and y_label_as_integer < dim_vocab

            assert matrix_u.ndim == 2
            assert matrix_v.ndim == 2
            assert matrix_w.ndim == 2
            assert matrix_u.shape[0] == dim_hidden and matrix_u.shape[1] == dim_vocab
            assert matrix_v.shape[0] == dim_vocab and matrix_v.shape[1] == dim_hidden
            assert matrix_w.shape[0] == matrix_w.shape[1] == dim_hidden
            assert prev_state.ndim == 1
            assert prev_state.size == dim_hidden

        probs_minus_y_one_shot = probabilities.copy()
        probs_minus_y_one_shot[label_y_as_integer] -= 1

        # See computational_utils.py, loss_from_matrix_v_and_hidden_state_derivative_wrt_v() for more details
        partial_loss_partial_v = np.outer(probs_minus_y_one_shot, current_state)
        if check_shapes:
            assert partial_loss_partial_v.shape == matrix_v.shape

        # Computing the Jacobian matrix of: partial(loss)/partial(matrix_u)
        # See comments within computational_utils.py, loss_from_matrix_u_derivative_wrt_u()
        partial_loss_partial_new_state = np.matmul(probs_minus_y_one_shot, matrix_v)
        sech_before_activation_state_vector = 1 / (np.cosh(current_state_before_activation) ** 2)
        partial_loss_partial_before_activation_vector = np.multiply(partial_loss_partial_new_state, sech_before_tanh_vector)

        # According to my notes, partial(loss)/partial(u_at_column_x) equals to partial_loss_partial_before_activation_vector times 1
        partial_loss_partial_u_at_x = partial_loss_partial_before_activation_vector
        partial_loss_partial_u = np.zeros([hidden_dim, vocab_dim])
        partial_loss_partial_u[:, input_x_integer] = partial_loss_partial_u_at_x

        if check_shapes:
            assert partial_loss_partial_u.shape == matrix_u.shape

        # Computing the Jacobian matrix of: partial(loss)/partial(matrix_w)
        # See comments within computational_utils.py, loss_from_matrix_w_derivative_wrt_w()
        partial_loss_partial_w = np.outer(partial_loss_partial_before_tanh_vector, prev_state)

        if check_shapes:
            assert partial_loss_partial_w.shape == matrix_w.shape

        return (partial_loss_partial_u, partial_loss_partial_v, partial_loss_partial_w)


    @staticmethod
    def bptt(forward_computation_intermediates_array, label_y_integer_seq, seq_len, dim_vocab, dim_hidden, truncation_len, trimming_box_radius=5, check_shapes=True):
        if check_shapes:
            assert isinstance(label_y_integer_seq, list)
            assert isinstance(label_y_integer_seq[0], int)
            assert len(label_y_integer_seq) == seq_len

        for t in revsersed(range(seq_len)):
            # Calculate gradients for U, V, W at time t, using input_x[t], label_y[t]
            # but for forward_computation_intermediates_array[t].current_state, it's a function of forward_computation_intermediates_array[t-1].current_state and W


    def step_parameters(self, loss_gradient_u_v_w, step_size, check_shapes=True):
        (partial_loss_partial_u, partial_loss_partial_v, partial_loss_partial_w) = loss_gradient_u_v_w
        if check_shapes:
            assert partial_loss_partial_u.shape == self.matrix_u.shape
            assert partial_loss_partial_v.shape == self.matrix_v.shape
            assert partial_loss_partial_w.shape == self.matrix_w.shape

        self.matrix_u -= partial_loss_partial_u.shape * step_size
        self.matrix_v -= partial_loss_partial_v.shape * step_size
        self.matrix_w -= partial_loss_partial_w.shape * step_size




