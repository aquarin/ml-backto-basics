import logging
import math
import numpy as np
import pickle

from computational_utils import Utilities

logger = logging.getLogger(__name__)

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
    def forward(input_x_as_integer, dim_vocab, dim_hidden, matrix_u, matrix_v, matrix_w, prev_state_vector,
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

        forward_computation_intermediates = {
            'input_x_as_integer': input_x_as_integer,
            'matrix_u': matrix_u,
            'matrix_v': matrix_v,
            'matrix_w': matrix_w,
            'prev_state_vector': prev_state_vector,
            'matrix_u_times_x_onehot': matrix_u_times_x_onehot,
            'w_times_prev_state': w_times_prev_state,
            'current_state_before_activation': current_state_before_activation,
            'current_state': current_state,
            'logits': logits,
            'softmax_probabilities': softmax_probabilities,
        }

        return forward_computation_intermediates


    @staticmethod
    def forward_sequence(input_x_int_array, dim_vocab, dim_hidden, matrix_u, matrix_v, matrix_w,
        start_state_vector=None, check_shapes=True, print_debug=False):
        if check_shapes:
            assert matrix_u.ndim == 2
            assert matrix_v.ndim == 2
            assert matrix_w.ndim == 2
            assert matrix_u.shape[0] == dim_hidden and matrix_u.shape[1] == dim_vocab
            assert matrix_v.shape[0] == dim_vocab and matrix_v.shape[1] == dim_hidden
            assert matrix_w.shape[0] == matrix_w.shape[1] == dim_hidden
            assert start_state_vector.ndim == 1
            assert start_state_vector.size == dim_hidden

        forward_computation_intermediates_array = []
        prev_state_vector = start_state_vector if start_state_vector is None else np.zeros(dim_hidden)

        for input_x_as_integer in input_x_int_array:
            if check_shapes:
                assert isinstance(input_x_as_integer, int)
                assert input_x_as_integer >= 0 and input_x_as_integer < dim_vocab

            forward_computation_intermediates = RnnWithNumpy.forward(
                input_x_as_integer=input_x_as_integer, dim_vocab=dim_vocab, dim_hidden=dim_hidden,
                matrix_u=matrix_u, matrix_w=matrix_w, matrix_v=matrix_v, prev_state_vector=prev_state_vector,
                check_shapes=check_shapes, print_debug=print_debug)

            forward_computation_intermediates_array.append(forward_computation_intermediates)
            prev_state_vector = forward_computation_intermediates['current_state']
            if check_shapes:
                assert prev_state_vector.ndim == 1 and prev_state_vector.size == dim_hidden

        return forward_computation_intermediates_array


    @staticmethod
    def sequence_loss(dim_vocab, probabilities_time_series, label_y_int_array, check_shapes=True):
        assert len(probabilities_time_series) == len(label_y_int_array)
        total_loss = 0

        for t in range(len(label_y_int_array)):
            softmax_probabilities = probabilities_time_series[t]
            loss = RnnWithNumpy.loss(dim_vocab=dim_vocab, softmax_probabilities=softmax_probabilities,
                y_label_as_integer=label_y_int_array[t], check_shapes=check_shapes)
            total_loss += loss

        return total_loss / float(len(label_y_int_array))


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
            matrix_u, matrix_v, matrix_w, prev_state_vector,
            matrix_u_times_x_onehot, w_times_prev_state, current_state_before_activation,
            current_state, logits, softmax_probabilities) = forward_computation_intermediates

        return self.predict_with_softmax_output(self.dim_vocab, softmax_probabilities, check_shapes=check_shapes, print_debug=print_debug)


    @staticmethod
    def loss(dim_vocab, softmax_probabilities, y_label_as_integer, check_shapes=True):
        if check_shapes:
            assert softmax_probabilities.ndim == 1 and softmax_probabilities.size == dim_vocab
            assert isinstance(y_label_as_integer, int)
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
    def sequence_loss_gradient_u_v_w(forward_computation_intermediates_array,
        label_y_int_array, dim_vocab, dim_hidden, bptt_truncation_len=10, check_shapes=True):
        pass


    @staticmethod
    def bptt(forward_computation_intermediates_array, label_y_integer_seq, seq_len, dim_vocab, dim_hidden, truncation_len, trimming_box_radius=5, check_shapes=True):
        if check_shapes:
            assert isinstance(label_y_integer_seq, list)
            assert isinstance(label_y_integer_seq[0], int)
            assert len(label_y_integer_seq) == seq_len

        for t in revsersed(range(seq_len)):
            pass
            # Calculate gradients for U, V, W at time t, using input_x[t], label_y[t]
            # but for forward_computation_intermediates_array[t].current_state, it's a function of forward_computation_intermediates_array[t-1].current_state and W


    # Returns a Jacobian matrix of partial(state_vector@time_t) / partial(matrix_w)
    # See my notes, due to state_vector@time_t is a function of F(W, state_vector@time_t_minus_1), this will be calculated recursively.
    @staticmethod
    def bptt_partial_state_time_t_partial_matrix_w(states_array_indexed_by_time, current_time, dim_hidden, matrix_w, truncation_len=10,
        check_shapes=True, debug_output_dict=None, print_debug=False):
        if truncation_len <= 0 or current_time <= 0:
            logger.debug('Returning 0 due to hitting truncation or current_time == 0, truncation_len=%d, current_time=%d'
                % (truncation_len, current_time))

            # TODO: check if just a simple zero works, or should I need a size-aligned three dimensional zero matrix.
            return 0

        if check_shapes:
            assert matrix_w.shape == (dim_hidden, dim_hidden)

        current_state = states_array_indexed_by_time[current_time]

        if check_shapes:
            assert current_state.ndim == 1 and current_state.size == dim_hidden

        # state = tanh(state_raw)
        # partial(state)/partial(state_raw) = d(tanh)/d(x) = 1 - tanh(x)^2 = 1 - state^2
        diag_matrix_partial_state_partial_state_raw = np.diag(1 - current_state ** 2)

        # in the following code, f_t means a function that takes W, state_vector_t_minus_1 as a parameter, and outputs multiply(W, state_vector_t_minus_1)
        # partial_f_partial_param_1 is the partial derivative wrt to the first parameter (W), and partial_f_partial_2 is the partial derivative wrt to the second parammeter
        # Note that state_vector_t_minus_1 is also a function of (W, state_vector_t_minus_2)
        # So partial(F)/partial(W) = partial(F)/parital(F parameter 1) + partial(F)/partial(F parameter 2) * partial(state_vector_t_minus_1) / partial(W)

        # TODO: this can be replaced with a einsum().
        def _create_partial_f_partial_param_1(prev_state_vector):
            if check_shapes:
                assert prev_state_vector.ndim == 1 and prev_state_vector.size == dim_hidden
            partial_f_partial_param_1 = np.full([dim_hidden, dim_hidden], None, dtype=object)
            for i in range(dim_hidden):
                for j in range(dim_hidden):
                    partial_f_partial_param_1[i][j] = np.zeros(dim_hidden).T
                    partial_f_partial_param_1[i][j][i] = prev_state_vector[j]

            return partial_f_partial_param_1

        # current_time == 0 already checked.
        partial_f_partial_param_1 = _create_partial_f_partial_param_1(states_array_indexed_by_time[current_time - 1])

        # See my notes.
        partial_f_partial_param_2 = matrix_w

        partial_prev_state_partial_matrix_w = RnnWithNumpy.bptt_partial_state_time_t_partial_matrix_w(
            states_array_indexed_by_time, current_time - 1, dim_hidden, matrix_w, truncation_len - 1, check_shapes=check_shapes,
            debug_output_dict=None, print_debug=print_debug)

        partial_state_raw_partial_w = partial_f_partial_param_1 + (
            0 if (
                np.isscalar(partial_prev_state_partial_matrix_w) and partial_prev_state_partial_matrix_w == 0)
            else np.matmul(partial_f_partial_param_2.T, partial_prev_state_partial_matrix_w))
        partial_state_partial_w = np.matmul(diag_matrix_partial_state_partial_state_raw, partial_state_raw_partial_w)

        if print_debug and debug_output_dict == None:
            # Just to avoid duplicate debug tracing code in below.
            debug_output_dict = {}

        if debug_output_dict != None:
            debug_output_dict['current_state'] = current_state
            debug_output_dict['prev_state'] = states_array_indexed_by_time[current_time - 1]
            debug_output_dict['diag_matrix_partial_state_partial_state_raw'] = diag_matrix_partial_state_partial_state_raw
            debug_output_dict['partial_f_partial_param_1'] = partial_f_partial_param_1
            debug_output_dict['partial_f_partial_param_2'] = partial_f_partial_param_2
            debug_output_dict['partial_prev_state_partial_matrix_w'] = partial_prev_state_partial_matrix_w
            debug_output_dict['partial_state_partial_w'] = partial_state_partial_w

            with open('bptt_test_current_time_%d.pkl' % current_time, 'wb') as f:
                pickle.dump(debug_output_dict, f)

        if print_debug:
            logger.debug('==================At current_time = %d:=================' % current_time)
            logger.debug(debug_output_dict)

        return partial_state_partial_w


    def step_parameters(self, loss_gradient_u_v_w, step_size, check_shapes=True):
        (partial_loss_partial_u, partial_loss_partial_v, partial_loss_partial_w) = loss_gradient_u_v_w
        if check_shapes:
            assert partial_loss_partial_u.shape == self.matrix_u.shape
            assert partial_loss_partial_v.shape == self.matrix_v.shape
            assert partial_loss_partial_w.shape == self.matrix_w.shape

        self.matrix_u -= partial_loss_partial_u.shape * step_size
        self.matrix_v -= partial_loss_partial_v.shape * step_size
        self.matrix_w -= partial_loss_partial_w.shape * step_size




