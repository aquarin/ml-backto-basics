'''
This file contains unit tests against the components within class RnnWithNumpy.
The forward propagations are verified by recomputing.
The backward propagation (gradients calculations), especially the BPTT, are verified by comparing the results to the numerical derivatives
    of the loss function, with respect to difference paramter matrices.
'''

import logging
import math
import random
import unittest

import numpy as np

from computational_utils import Utilities
from derivative_verifier import DerivativeVerifier
from rnn_using_only_numpy import RnnWithNumpy

logger = logging.getLogger(__name__)

class RnnUsingOnlyNumpyTest(unittest.TestCase):
    common_test_params = {
        'dim_hidden': 3,
        'dim_vocab': 4,
        'matrix_u': np.array([
            [.5, 0, 0, 0],
            [0, .5, 0, .2],
            [0, 0, .5, .2],
        ]),
        'matrix_v': np.array([
            [9, 1, 2],
            [1, 8, 3],
            [0., 1.5, 7],
            [4, 5, 1],
        ]) / 30,
        'matrix_w': np.array([
            [.1, .2, .3],
            [.4, .5, .6],
            [.7, .8, .9],
        ]) - .5,
        'bias_vector': np.array([.03, .06, .09]),
        'delta_x_scalar': 1e-5
    }

    prediction_test_params = {
        'matrix_u': np.array([
            [.5, 0, 0, 0],
            [0, .5, 0, .2],
            [0, 0, .5, .2],
        ]),
        # Sorted of wanted: embedding first element to be emphasized into second id,
        # embedding second to third id, and no one cared about the fourth id
        'matrix_v': np.array([
            [.2, .2, .8],
            [.8, .2, .2],
            [.2, .8, .2],
            [.2, .3, .3],
        ]),
        # Sorted of wanted: previous state "equally" carried to the next state.
        'matrix_w': np.eye(3),
        'bias_vector': np.array([.03, .06, .09]),
    }

    # A simplified forward process just to verify the bptt. In this forwarding, there is no logits, softmax, matrix V, involved.
    @staticmethod
    def forward_with_only_state_vector(dim_hidden, dim_vocab, state_vector_time_negative_1, matrix_w, matrix_u, input_x_integers_by_time):
        assert matrix_u.ndim == 2 and matrix_w.ndim == 2
        assert matrix_u.shape[0] == matrix_w.shape[0] == matrix_w.shape[1] == dim_hidden
        assert matrix_u.shape[1] == dim_vocab
        assert state_vector_time_negative_1.ndim == 1 and state_vector_time_negative_1.size == dim_hidden

        for x_input_int in input_x_integers_by_time:
            assert isinstance(x_input_int, int)
            assert x_input_int >= 0 and x_input_int < dim_vocab

        state_vectors_by_time = []

        prev_state = state_vector_time_negative_1
        for time in range(len(input_x_integers_by_time)):
            x_input_int = input_x_integers_by_time[time]
            input_x_embedding = matrix_u[:, x_input_int]
            w_mul_prev_state = np.matmul(matrix_w, prev_state)
            assert input_x_embedding.shape == w_mul_prev_state.shape

            state_before_activation = input_x_embedding + w_mul_prev_state
            current_state = np.tanh(state_before_activation)
            state_vectors_by_time.append(current_state)

            prev_state = current_state

        return state_vectors_by_time


    def test_hidden_state_only_forward(self):
        dim_hidden = 3
        dim_vocab = 4

        matrix_w = np.array([
            [.1, .2, .3],
            [.4, .5, .6],
            [.7, .8, .9],
        ]) - .5
        state_vector_time_negative_1 = np.zeros(dim_hidden)
        matrix_u = np.array([
            [.5, 0, 0, 0],
            [0, .5, 0, .2],
            [0, 0, .5, .2],
        ])

        input_x_integers_by_time = [0, 1, 2]
        state_vectors_by_time = self.forward_with_only_state_vector(
            dim_hidden, dim_vocab, state_vector_time_negative_1, matrix_w, matrix_u, input_x_integers_by_time)

        logger.debug(state_vectors_by_time)

        np.testing.assert_almost_equal(
            state_vectors_by_time[0],
            np.tanh(matrix_u[:, 0]),
            7)
        np.testing.assert_almost_equal(
            state_vectors_by_time[1],
            np.tanh(np.matmul(matrix_w, np.tanh(matrix_u[:, 0])) + matrix_u[:, input_x_integers_by_time[1]]),
            7)
        np.testing.assert_almost_equal(
            state_vectors_by_time[2],
            np.tanh(
                np.matmul(matrix_w,
                    np.tanh(np.matmul(matrix_w, np.tanh(matrix_u[:, 0])) + matrix_u[:, input_x_integers_by_time[1]]))
                    + matrix_u[:, input_x_integers_by_time[2]]),
            7)


    # SCENARIO: sequence length = 1
    def test_visual_verify_partial_state_partial_w_seq_length_1(self):
        dim_hidden = 3
        dim_vocab = 4
        matrix_w = np.array([
            [.1, .2, .3],
            [.4, .5, .6],
            [.7, .8, .9],
        ]) - .5
        state_vector_time_negative_1 = np.zeros(dim_hidden)
        matrix_u = np.array([
            [.5, 0, 0, 0],
            [0, .5, 0, .2],
            [0, 0, .5, .2],
        ])
        input_x_integers_by_time = [0]
        delta_x_scalar = 1e-5

        # Only takes a matrix_w parameter so that this one parameter will be diff-ed for numerical derivative.
        def _forward_prop_wrapper(w):
            state_time_series = self.forward_with_only_state_vector(
                dim_hidden, dim_vocab, state_vector_time_negative_1, w, matrix_u, input_x_integers_by_time)
            return state_time_series[-1]

        numerical_jacobian_diff = DerivativeVerifier.numerical_jacobian_diff_matrix(_forward_prop_wrapper, matrix_w, delta_x_scalar)
        logger.debug('numerical_jacobian_diff=\n%s\n' % numerical_jacobian_diff)
        numerical_jacobian_derivative = numerical_jacobian_diff / delta_x_scalar
        logger.debug('numerical_jacobian_derivative=\n%s\n' % numerical_jacobian_derivative)
        logger.debug('numerical_jacobian_derivative shape=%s' % str(numerical_jacobian_derivative.shape))
        logger.debug('numerical_jacobian_derivative[0][0] shape=%s' % str(numerical_jacobian_derivative[0][0].shape))


        debug_output_dict = {}
        state_time_series = self.forward_with_only_state_vector(
                dim_hidden, dim_vocab, state_vector_time_negative_1, matrix_w, matrix_u, input_x_integers_by_time)

        # bptt_partial_state_time_t_partial_matrix_w(states_array_indexed_by_time, current_time, dim_hidden, matrix_w, truncation_len=10, check_shapes=True, debug_output_dict=None):
        bptt_partial_state_partial_w_at_time_2 = RnnWithNumpy.bptt_partial_state_time_t_partial_matrix_w(
            state_time_series, current_time=0, dim_hidden=dim_hidden,
            matrix_w=matrix_w,
            partial_state_partial_w_by_time_cache={},
            truncation_len=10, check_shapes=True, debug_output_dict=debug_output_dict)
        logger.debug('bptt_partial_state_partial_w_at_time_2=\n%s\n' % bptt_partial_state_partial_w_at_time_2)
        logger.debug('debug_output_dict=\n%s\n' % debug_output_dict)

        # seq length = 0, current time = 0, there is such BPTT at all, it should return 0.
        np.testing.assert_equal(bptt_partial_state_partial_w_at_time_2, 0)
        for index, sub_matrix in np.ndenumerate(numerical_jacobian_diff):
            np.testing.assert_equal(sub_matrix, np.zeros([dim_hidden]))


    # SCENARIO: sequence length = 2
    def test_visual_verify_partial_state_partial_w_seq_length_2(self):
        dim_hidden = 3
        dim_vocab = 4
        matrix_w = np.array([
            [.1, .2, .3],
            [.4, .5, .6],
            [.7, .8, .9],
        ]) - .5
        state_vector_time_negative_1 = np.zeros(dim_hidden)
        matrix_u = np.array([
            [.5, 0, 0, 0],
            [0, .5, 0, .2],
            [0, 0, .5, .2],
        ])
        input_x_integers_by_time = [1, 3]
        delta_x_scalar = 1e-5

        # Only takes a matrix_w parameter so that this one parameter will be diff-ed for numerical derivative.
        def _forward_prop_wrapper(w):
            state_time_series = self.forward_with_only_state_vector(
                dim_hidden, dim_vocab, state_vector_time_negative_1, w, matrix_u, input_x_integers_by_time)
            return state_time_series[-1]


        numerical_jacobian_diff = DerivativeVerifier.numerical_jacobian_diff_matrix(_forward_prop_wrapper, matrix_w, delta_x_scalar)
        logger.debug('numerical_jacobian_diff=\n%s\n' % numerical_jacobian_diff)
        numerical_jacobian_derivative = numerical_jacobian_diff / delta_x_scalar
        logger.debug('numerical_jacobian_derivative=\n%s\n' % numerical_jacobian_derivative)
        logger.debug('numerical_jacobian_derivative shape=%s' % str(numerical_jacobian_derivative.shape))
        logger.debug('numerical_jacobian_derivative[0][0] shape=%s' % str(numerical_jacobian_derivative[0][0].shape))

        debug_output_dict = {}
        state_time_series = self.forward_with_only_state_vector(
                dim_hidden, dim_vocab, state_vector_time_negative_1, matrix_w, matrix_u, input_x_integers_by_time)

        # bptt_partial_state_time_t_partial_matrix_w(states_array_indexed_by_time, current_time, dim_hidden, matrix_w, truncation_len=10, check_shapes=True, debug_output_dict=None):
        bptt_partial_state_partial_w_at_time_2 = RnnWithNumpy.bptt_partial_state_time_t_partial_matrix_w(
            state_time_series, current_time=1, dim_hidden=dim_hidden,
            matrix_w=matrix_w,
            partial_state_partial_w_by_time_cache={},
            truncation_len=10, check_shapes=True, debug_output_dict=debug_output_dict)
        logger.debug('bptt_partial_state_partial_w_at_time_2=\n%s\n' % bptt_partial_state_partial_w_at_time_2)
        logger.debug('debug_output_dict=\n%s\n' % debug_output_dict)

        for index, sub_matrix in np.ndenumerate(numerical_jacobian_derivative):
            np.testing.assert_almost_equal(sub_matrix, bptt_partial_state_partial_w_at_time_2[index], 5)


    # SCENARIO: sequence length = 3
    def test_visual_verify_partial_state_partial_w_seq_length_3(self):
        dim_hidden = 3
        dim_vocab = 4
        matrix_w = np.array([
            [.1, .2, .3],
            [.4, .5, .6],
            [.7, .8, .9],
        ]) - .5
        state_vector_time_negative_1 = np.zeros(dim_hidden)
        matrix_u = np.array([
            [.5, 0, 0, 0],
            [0, .5, 0, .2],
            [0, 0, .5, .2],
        ])
        input_x_integers_by_time = [1, 3, 0]
        delta_x_scalar = 1e-5

        # Only takes a matrix_w parameter so that this one parameter will be diff-ed for numerical derivative.
        def _forward_prop_wrapper(w):
            state_time_series = self.forward_with_only_state_vector(
                dim_hidden, dim_vocab, state_vector_time_negative_1, w, matrix_u, input_x_integers_by_time)
            return state_time_series[-1]


        debug_output_dict = {}
        state_time_series = self.forward_with_only_state_vector(
                dim_hidden, dim_vocab, state_vector_time_negative_1, matrix_w, matrix_u, input_x_integers_by_time)

        # bptt_partial_state_time_t_partial_matrix_w(states_array_indexed_by_time, current_time, dim_hidden, matrix_w, truncation_len=10, check_shapes=True, debug_output_dict=None):
        bptt_partial_state_partial_w_at_time_2 = RnnWithNumpy.bptt_partial_state_time_t_partial_matrix_w(
            state_time_series, current_time=2, dim_hidden=dim_hidden,
            matrix_w=matrix_w,
            partial_state_partial_w_by_time_cache={},
            truncation_len=10, check_shapes=True, debug_output_dict=debug_output_dict, print_debug=True)
        logger.debug('debug_output_dict=\n%s\n' % debug_output_dict)
        logger.debug('bptt_partial_state_partial_w_at_time_2=\n%s\n' % bptt_partial_state_partial_w_at_time_2)

        numerical_jacobian_diff = DerivativeVerifier.numerical_jacobian_diff_matrix(_forward_prop_wrapper, matrix_w, delta_x_scalar)
        numerical_jacobian_derivative = numerical_jacobian_diff / delta_x_scalar
        logger.debug('numerical_jacobian_derivative=\n%s\n' % numerical_jacobian_derivative)
        logger.debug('numerical_jacobian_derivative shape=%s' % str(numerical_jacobian_derivative.shape))
        logger.debug('numerical_jacobian_derivative[0][0] shape=%s' % str(numerical_jacobian_derivative[0][0].shape))


        for index, sub_matrix in np.ndenumerate(numerical_jacobian_derivative):
            np.testing.assert_almost_equal(sub_matrix, bptt_partial_state_partial_w_at_time_2[index], 5)

    # SCENARIO: sequence length = n
    def test_visual_verify_partial_state_partial_w_seq_length_n(self):
        dim_hidden = 3
        dim_vocab = 4
        matrix_w = np.array([
            [.1, .2, .3],
            [.4, .5, .6],
            [.7, .8, .9],
        ]) - .5
        state_vector_time_negative_1 = np.zeros(dim_hidden)
        matrix_u = np.array([
            [.5, 0, 0, 0],
            [0, .5, 0, .2],
            [0, 0, .5, .2],
        ])
        input_x_integers_by_time = [1, 3, 0, 1, 2, 0, 2, 2]
        delta_x_scalar = 1e-5

        # Only takes a matrix_w parameter so that this one parameter will be diff-ed for numerical derivative.
        def _forward_prop_wrapper(w):
            state_time_series = self.forward_with_only_state_vector(
                dim_hidden, dim_vocab, state_vector_time_negative_1, w, matrix_u, input_x_integers_by_time)
            return state_time_series[-1]

        debug_output_dict = {}
        state_time_series = self.forward_with_only_state_vector(
                dim_hidden, dim_vocab, state_vector_time_negative_1, matrix_w, matrix_u, input_x_integers_by_time)

        # bptt_partial_state_time_t_partial_matrix_w(states_array_indexed_by_time, current_time, dim_hidden, matrix_w, truncation_len=10, check_shapes=True, debug_output_dict=None):
        bptt_partial_state_partial_w_at_time_n = RnnWithNumpy.bptt_partial_state_time_t_partial_matrix_w(
            state_time_series, current_time=len(input_x_integers_by_time) - 1, dim_hidden=dim_hidden,
            matrix_w=matrix_w,
            partial_state_partial_w_by_time_cache={},
            truncation_len=10, check_shapes=True, debug_output_dict=debug_output_dict, print_debug=True)
        logger.debug('debug_output_dict=\n%s\n' % debug_output_dict)
        logger.debug('bptt_partial_state_partial_w_at_time_n=\n%s\n' % bptt_partial_state_partial_w_at_time_n)

        numerical_jacobian_diff = DerivativeVerifier.numerical_jacobian_diff_matrix(_forward_prop_wrapper, matrix_w, delta_x_scalar)
        numerical_jacobian_derivative = numerical_jacobian_diff / delta_x_scalar
        logger.debug('numerical_jacobian_derivative=\n%s\n' % numerical_jacobian_derivative)
        logger.debug('numerical_jacobian_derivative shape=%s' % str(numerical_jacobian_derivative.shape))
        logger.debug('numerical_jacobian_derivative[0][0] shape=%s' % str(numerical_jacobian_derivative[0][0].shape))

        for index, sub_matrix in np.ndenumerate(numerical_jacobian_derivative):
            np.testing.assert_almost_equal(sub_matrix, bptt_partial_state_partial_w_at_time_n[index], 2)

    def test_forward_sequence_len_1(self):
        dim_hidden = 3
        dim_vocab = 4
        matrix_w = (np.array([
            [.1, .2, .3],
            [.4, .5, .6],
            [.7, .8, .9],
        ]) - .5) / 3
        state_vector_time_negative_1 = np.zeros(dim_hidden)
        matrix_u = np.array([
            [.5, 0, 0, 0],
            [0, .5, 0, .2],
            [0, 0, .5, .2],
        ])
        matrix_v = np.array([
            [9, 1, 2],
            [1, 8, 3],
            [0., 1.5, 7],
            [4, 5, 1],
        ]) / 30
        bias_vector = np.array([.03, 0.06, .09])
        input_x_integers_by_time = [1]

        forward_computation_intermediates_array = RnnWithNumpy.forward_sequence(
            input_x_int_array=input_x_integers_by_time, dim_vocab=dim_vocab, dim_hidden=dim_hidden,
            matrix_u=matrix_u, matrix_v=matrix_v, matrix_w=matrix_w, bias_vector=bias_vector,
            start_state_vector=state_vector_time_negative_1, check_shapes=True, print_debug=True)

        logger.debug(forward_computation_intermediates_array)
        logger.debug('forward_computation_intermediates_array type=%s', type(forward_computation_intermediates_array))

        prev_state = state_vector_time_negative_1
        assert len(forward_computation_intermediates_array) == len(input_x_integers_by_time)
        for t in range(len(input_x_integers_by_time)):
            state_vector = np.tanh(np.matmul(matrix_w, prev_state) + matrix_u[:, input_x_integers_by_time[t]] + bias_vector)

            logger.debug('t=%d, prev_state=%s, state_vector=%s, forward_computation_intermediates[][current_state]=%s, forward_computation_intermediates[][prev_state_vector]=%s'
                %(t, prev_state, state_vector, forward_computation_intermediates_array[t]['current_state'], forward_computation_intermediates_array[t]['prev_state_vector']))

            logits = np.matmul(matrix_v, state_vector)
            softmax_probabilities = Utilities.softmax(logits)

            np.testing.assert_almost_equal(state_vector, forward_computation_intermediates_array[t]['current_state'], 6)
            np.testing.assert_almost_equal(logits, forward_computation_intermediates_array[t]['logits'], 6)
            np.testing.assert_almost_equal(softmax_probabilities, forward_computation_intermediates_array[t]['softmax_probabilities'], 6)
            prev_state = state_vector


    def test_forward_sequence(self):
        dim_hidden = 3
        dim_vocab = 4
        matrix_w = (np.array([
            [.1, .2, .3],
            [.4, .5, .6],
            [.7, .8, .9],
        ]) - .5) / 3
        state_vector_time_negative_1 = np.zeros(dim_hidden)
        matrix_u = np.array([
            [.5, 0, 0, 0],
            [0, .5, 0, .2],
            [0, 0, .5, .2],
        ])
        matrix_v = np.array([
            [9, 1, 2],
            [1, 8, 3],
            [0., 1.5, 7],
            [4, 5, 1],
        ]) / 30
        bias_vector = np.array([.05, .10, .15])
        input_x_integers_by_time = [1, 3, 0, 1, 2, 0, 2, 2]

        forward_computation_intermediates_array = RnnWithNumpy.forward_sequence(
            input_x_int_array=input_x_integers_by_time, dim_vocab=dim_vocab, dim_hidden=dim_hidden,
            matrix_u=matrix_u, matrix_v=matrix_v, matrix_w=matrix_w,
            bias_vector=bias_vector,
            start_state_vector=state_vector_time_negative_1, check_shapes=True, print_debug=True)

        logger.debug(forward_computation_intermediates_array)

        prev_state = state_vector_time_negative_1
        assert len(forward_computation_intermediates_array) == len(input_x_integers_by_time)
        for t in range(len(input_x_integers_by_time)):
            state_vector = np.tanh(np.matmul(matrix_w, prev_state) + matrix_u[:, input_x_integers_by_time[t]] + bias_vector)

            logger.debug('t=%d, prev_state=%s, state_vector=%s, forward_computation_intermediates[][current_state]=%s, forward_computation_intermediates[][prev_state_vector]=%s'
                %(t, prev_state, state_vector, forward_computation_intermediates_array[t]['current_state'], forward_computation_intermediates_array[t]['prev_state_vector']))

            logits = np.matmul(matrix_v, state_vector)
            softmax_probabilities = Utilities.softmax(logits)

            np.testing.assert_almost_equal(state_vector, forward_computation_intermediates_array[t]['current_state'], 6)
            np.testing.assert_almost_equal(logits, forward_computation_intermediates_array[t]['logits'], 6)
            np.testing.assert_almost_equal(softmax_probabilities, forward_computation_intermediates_array[t]['softmax_probabilities'], 6)
            prev_state = state_vector


    def test_sequence_loss(self):
        dim_vocab = 4

        label_y_int_array = [0, 1, 3, 1, 3, 2]

        softmax_probabilities_series = np.array([
            # Doesn't sum to zero. No time to make a very neat test probabilities array.
            [.01, .01, .01, 1.0],
            [.01, .8, .01, .2],
            [.01, .01, .01, 1.0],
            [.01, .3, .01, 1.0],
            [.01, .3, .01, 1.0],
            [.01, .3, .41, .3],
        ])

        expected_total_loss = -(
            math.log(.01) + math.log(.8) + math.log(1.0) + math.log(.3) + math.log(1.0) + math.log(.41)) / 6.0

        np.testing.assert_almost_equal(expected_total_loss,
            RnnWithNumpy.sequence_loss(dim_vocab=dim_vocab, probabilities_time_series=softmax_probabilities_series, label_y_int_array=label_y_int_array,
                check_shapes=True))


    @staticmethod
    def verify_gradient(dim_hidden, dim_vocab, matrix_u,  matrix_v, matrix_w, bias_vector,
        input_x_integers_by_time, label_y_int_by_time):

        state_vector_time_negative_1 = np.zeros(dim_hidden)

        forward_computation_intermediates_array = RnnWithNumpy.forward_sequence(
            input_x_int_array=input_x_integers_by_time, dim_vocab=dim_vocab, dim_hidden=dim_hidden,
            matrix_u=matrix_u, matrix_v=matrix_v, matrix_w=matrix_w, bias_vector=bias_vector,
            start_state_vector=state_vector_time_negative_1, check_shapes=True, print_debug=True)

        (partial_loss_partial_u, partial_loss_partial_v,
            partial_loss_partial_w, partial_loss_partial_b) = RnnWithNumpy.sequence_loss_gradient_u_v_w_b(
            forward_computation_intermediates_array=forward_computation_intermediates_array,
            label_y_int_array=label_y_int_by_time, dim_vocab=dim_vocab, dim_hidden=dim_hidden,
            bptt_truncation_len=10, check_shapes=True)

        delta_x = 1e-5
        def _loss_as_function_of_matrix_u(matrix_u_func):
            computed_intermediaries_array = RnnWithNumpy.forward_sequence(
                input_x_int_array=input_x_integers_by_time, dim_vocab=dim_vocab, dim_hidden=dim_hidden,
                matrix_u=matrix_u_func, matrix_v=matrix_v, matrix_w=matrix_w, bias_vector=bias_vector,
                start_state_vector=state_vector_time_negative_1, check_shapes=False, print_debug=False)
            total_loss = RnnWithNumpy.sequence_loss_from_forward_computations(
                dim_vocab=dim_vocab, forward_computation_intermediates_array=computed_intermediaries_array,
                label_y_int_array=label_y_int_by_time, check_shapes=True)
            return total_loss
        def _loss_as_function_of_matrix_v(matrix_v_func):
            computed_intermediaries_array = RnnWithNumpy.forward_sequence(
                input_x_int_array=input_x_integers_by_time, dim_vocab=dim_vocab, dim_hidden=dim_hidden,
                matrix_u=matrix_u, matrix_v=matrix_v_func, matrix_w=matrix_w, bias_vector=bias_vector,
                start_state_vector=state_vector_time_negative_1, check_shapes=False, print_debug=False)
            total_loss = RnnWithNumpy.sequence_loss_from_forward_computations(
                dim_vocab=dim_vocab, forward_computation_intermediates_array=computed_intermediaries_array,
                label_y_int_array=label_y_int_by_time, check_shapes=True)
            return total_loss
        def _loss_as_function_of_matrix_w(matrix_w_func):
            computed_intermediaries_array = RnnWithNumpy.forward_sequence(
                input_x_int_array=input_x_integers_by_time, dim_vocab=dim_vocab, dim_hidden=dim_hidden,
                matrix_u=matrix_u, matrix_v=matrix_v, matrix_w=matrix_w_func, bias_vector=bias_vector,
                start_state_vector=state_vector_time_negative_1, check_shapes=False, print_debug=False)
            total_loss = RnnWithNumpy.sequence_loss_from_forward_computations(
                dim_vocab=dim_vocab, forward_computation_intermediates_array=computed_intermediaries_array,
                label_y_int_array=label_y_int_by_time, check_shapes=True)
            return total_loss
        def _loss_as_function_of_bias(bias_vector_func):
            computed_intermediaries_array = RnnWithNumpy.forward_sequence(
                input_x_int_array=input_x_integers_by_time, dim_vocab=dim_vocab, dim_hidden=dim_hidden,
                matrix_u=matrix_u, matrix_v=matrix_v, matrix_w=matrix_w, bias_vector=bias_vector_func,
                start_state_vector=state_vector_time_negative_1, check_shapes=False, print_debug=False)
            total_loss = RnnWithNumpy.sequence_loss_from_forward_computations(
                dim_vocab=dim_vocab, forward_computation_intermediates_array=computed_intermediaries_array,
                label_y_int_array=label_y_int_by_time, check_shapes=True)
            return total_loss

        numerical_partial_loss_partial_u = DerivativeVerifier.numerical_jacobian_diff_matrix(
            func=_loss_as_function_of_matrix_u,
            matrix_x_0=matrix_u, delta_x_scalar=delta_x) / delta_x
        logger.debug("partial_loss_partial_u=\n%s\n", partial_loss_partial_u)
        logger.debug("numerical_partial_loss_partial_u=\n%s\n", numerical_partial_loss_partial_u)

        numerical_partial_loss_partial_v = DerivativeVerifier.numerical_jacobian_diff_matrix(
            func=_loss_as_function_of_matrix_v,
            matrix_x_0=matrix_v, delta_x_scalar=delta_x) / delta_x
        logger.debug("partial_loss_partial_v=\n%s\n", partial_loss_partial_v)
        logger.debug("numerical_partial_loss_partial_v=\n%s\n", numerical_partial_loss_partial_v)

        numerical_partial_loss_partial_w = DerivativeVerifier.numerical_jacobian_diff_matrix(
            func=_loss_as_function_of_matrix_w,
            matrix_x_0=matrix_w, delta_x_scalar=delta_x) / delta_x
        logger.debug("partial_loss_partial_w=\n%s\n", partial_loss_partial_w)
        logger.debug("numerical_partial_loss_partial_w=\n%s\n", numerical_partial_loss_partial_w)

        numerical_partial_loss_partial_b = DerivativeVerifier.numerical_jacobian_diff_matrix(
            func=_loss_as_function_of_bias,
            matrix_x_0=bias_vector, delta_x_scalar=delta_x) / delta_x
        logger.debug("partial_loss_partial_b=\n%s\n", partial_loss_partial_b)
        logger.debug("numerical_partial_loss_partial_b=\n%s\n", numerical_partial_loss_partial_b)

        np.testing.assert_almost_equal(partial_loss_partial_u, numerical_partial_loss_partial_u, 2)
        np.testing.assert_almost_equal(partial_loss_partial_v, numerical_partial_loss_partial_v, 2)
        np.testing.assert_almost_equal(partial_loss_partial_w, numerical_partial_loss_partial_w, 2)
        np.testing.assert_almost_equal(partial_loss_partial_b, numerical_partial_loss_partial_b, 2)


    def test_gradient_uvwb_single_step(self):
        dim_hidden = self.common_test_params['dim_hidden']
        dim_vocab = self.common_test_params['dim_vocab']
        matrix_u = self.common_test_params['matrix_u']
        matrix_v = self.common_test_params['matrix_v']
        matrix_w = self.common_test_params['matrix_w']
        bias_vector = self.common_test_params['bias_vector']
        input_x_integers_by_time = [3]
        label_y_int_array = [0]

        for x_int in range(dim_vocab):
            for y_int in range(dim_vocab):
                input_x_integers_by_time = [x_int]
                label_y_int_array = [y_int]
                self.verify_gradient(dim_hidden, dim_vocab, matrix_u,  matrix_v, matrix_w,
                    bias_vector, input_x_integers_by_time, label_y_int_array)


    def test_gradient_uvw_two_steps(self):
        dim_hidden = self.common_test_params['dim_hidden']
        dim_vocab = self.common_test_params['dim_vocab']
        matrix_u = self.common_test_params['matrix_u']
        matrix_v = self.common_test_params['matrix_v']
        matrix_w = self.common_test_params['matrix_w']
        bias_vector = self.common_test_params['bias_vector']
        input_x_integers_by_time = [1, 1]
        label_y_int_array = [0, 0]

        self.verify_gradient(dim_hidden, dim_vocab, matrix_u,  matrix_v, matrix_w,
            bias_vector, input_x_integers_by_time, label_y_int_array)


    # Debug why /partial_u is incorrect when sequence length >= 2, but correct when sequence length == 1
    def test_gradient_uvw_two_steps_debug(self):
        dim_hidden = self.common_test_params['dim_hidden']
        dim_vocab = self.common_test_params['dim_vocab']
        matrix_u = self.common_test_params['matrix_u']
        matrix_v = self.common_test_params['matrix_v']
        matrix_w = np.zeros([dim_hidden, dim_hidden])
        bias_vector = self.common_test_params['bias_vector']
        input_x_integers_by_time = [1, 1]
        label_y_int_array = [0, 0]

        self.verify_gradient(dim_hidden, dim_vocab, matrix_u,  matrix_v, matrix_w,
            bias_vector, input_x_integers_by_time, label_y_int_array)

    def test_gradient_uvw_three_steps(self):
        dim_hidden = self.common_test_params['dim_hidden']
        dim_vocab = self.common_test_params['dim_vocab']
        matrix_u = self.common_test_params['matrix_u']
        matrix_v = self.common_test_params['matrix_v']
        matrix_w = self.common_test_params['matrix_w']
        bias_vector = self.common_test_params['bias_vector']
        input_x_integers_by_time = [1, 0, 3]
        label_y_int_array = [1, 0, 3]

        self.verify_gradient(dim_hidden, dim_vocab, matrix_u,  matrix_v, matrix_w,
            bias_vector, input_x_integers_by_time, label_y_int_array)

    def test_gradient_uvw_sequence_length_n(self):
        dim_hidden = self.common_test_params['dim_hidden']
        dim_vocab = self.common_test_params['dim_vocab']
        matrix_u = self.common_test_params['matrix_u']
        matrix_v = self.common_test_params['matrix_v']
        matrix_w = self.common_test_params['matrix_w']
        bias_vector = self.common_test_params['bias_vector']
        input_x_integers_by_time = [1, 0, 3, 1, 2, 3]
        label_y_int_array = [1, 0, 3, 1, 2, 1]

        self.verify_gradient(dim_hidden, dim_vocab, matrix_u,  matrix_v, matrix_w,
            bias_vector, input_x_integers_by_time, label_y_int_array)


    def test_predict_sequence(self):
        model = RnnWithNumpy(dim_vocab=4, dim_hidden=3)
        model.matrix_u = self.prediction_test_params['matrix_u']
        model.matrix_v = self.prediction_test_params['matrix_v']
        model.matrix_w = self.prediction_test_params['matrix_w']
        model.bias_vector = self.prediction_test_params['bias_vector']

        input_id_seq = [0, 1, 2, 3, 0, 1, 2, 3]
        predicted_ids = model.predict_sequence(input_x_int_sequence=input_id_seq, check_shapes=True)
        logger.info('input ids=%s', input_id_seq)
        logger.info('predicted_ids=%s', predicted_ids)

        prev_state = np.zeros(3)
        for index, input_x_int in enumerate(input_id_seq):
            state = np.tanh(model.matrix_u[:, input_x_int] + np.matmul(model.matrix_w, prev_state) + model.bias_vector)
            logits = np.matmul(model.matrix_v, state)
            logger.info('step %d: state=%s, logits=%s, predicted_id=%d', index, state, logits, predicted_ids[index])
            np.testing.assert_equal(logits[predicted_ids[index]], np.max(logits))

            prev_state = state


    def test_step_parameters(self):
        dim_hidden = 3
        dim_vocab = 4
        model = RnnWithNumpy(dim_vocab=dim_vocab, dim_hidden=dim_hidden)
        model.matrix_u = self.prediction_test_params['matrix_u']
        model.matrix_v = self.prediction_test_params['matrix_v']
        model.matrix_w = self.prediction_test_params['matrix_w']
        model.bias_vector = self.prediction_test_params['bias_vector']
        state_vector_time_negative_1 = np.zeros(dim_hidden)

        input_id_seq = [0, 1, 2, 3, 0, 1, 2, 3]
        expected_output_seq = [2, 2, 2, 2, 2, 2, 2, 2]

        forward_computation_intermediates_array = RnnWithNumpy.forward_sequence(
            input_x_int_array=input_id_seq, dim_vocab=dim_vocab, dim_hidden=dim_hidden,
            matrix_u=model.matrix_u, matrix_v=model.matrix_v, matrix_w=model.matrix_w,
            bias_vector=model.bias_vector,
            start_state_vector=state_vector_time_negative_1, check_shapes=True, print_debug=True)

        (partial_loss_partial_u, partial_loss_partial_v,
            partial_loss_partial_w, partial_loss_partial_b) = RnnWithNumpy.sequence_loss_gradient_u_v_w_b(
            forward_computation_intermediates_array=forward_computation_intermediates_array,
            label_y_int_array=expected_output_seq, dim_vocab=dim_vocab, dim_hidden=dim_hidden,
            bptt_truncation_len=10, check_shapes=True)

        model.step_parameters(
            (partial_loss_partial_u, partial_loss_partial_v, partial_loss_partial_w, partial_loss_partial_b),
            step_size=1e-5, check_shapes=True)


    def test_optimize_bptt_matrix_operations(self):
        dim_hidden = 4

        def _create_partial_f_partial_param_1_version_new(prev_state_vector):
            partial_f_partial_param_1 = np.zeros([dim_hidden, dim_hidden, dim_hidden])
            for i in range(dim_hidden):
                for j in range(dim_hidden):
                    partial_f_partial_param_1[i][j][i] = prev_state_vector[j]

            return partial_f_partial_param_1

        def _create_partial_f_partial_param_1_version_old(prev_state_vector):
            partial_f_partial_param_1 = np.full([dim_hidden, dim_hidden], None, dtype=object)
            for i in range(dim_hidden):
                for j in range(dim_hidden):
                    partial_f_partial_param_1[i][j] = np.zeros(dim_hidden).T
                    partial_f_partial_param_1[i][j][i] = prev_state_vector[j]

            return partial_f_partial_param_1

        prev_state_vector = np.array([.1, .2, .3, .4])

        generated_old = _create_partial_f_partial_param_1_version_old(prev_state_vector)
        generated_new = _create_partial_f_partial_param_1_version_new(prev_state_vector)

        logger.debug("old=\n%s\n, new=\n%s\n", generated_old, generated_new)

        matrix_w = np.arange(dim_hidden * dim_hidden).reshape([dim_hidden, dim_hidden]) / (dim_hidden * dim_hidden)

        mul_old = np.matmul(matrix_w, generated_old)
        mul_new = np.einsum('mn,ijm->ijn', matrix_w, generated_new)

        # Convert the 2d matrix of object (vector) into a 3d matrix.
        mul_old_3d = np.empty([dim_hidden, dim_hidden, dim_hidden])
        for i in range(dim_hidden):
            for j in range(dim_hidden):
                for k in range(dim_hidden):
                    mul_old_3d[i][j][k] = mul_old[i][j][k]

        logger.debug("mul old converted to 3d matrix=\n%s\n, shape=%s, mul new=\n%s\n, shape=%s",
            mul_old_3d, mul_old_3d.shape, mul_new, mul_new.shape)

        np.testing.assert_almost_equal(mul_old_3d, mul_new, 5)


    '''
    I have hand calculated the derivative of the following scenario but need to verify that numerically.

    F(W) = W * S(W)
    W: n by n matrix
    F: n by n -> 1 by n function
    S: n by n -> 1 by n function

    Wanted to calculate: partial(F(W))/partial(W) (a n by n -> (n by n by n) function, and a (n by n by n) tensor, given a fixed W_0)

    My calcualtions gave:
    partial(F(W))/partil(W) = A + B

    A is (n by n by n), given a fixed W_0:
       A(i, j, k) = 0 if k != i
       S(W_0)[j] if k = i

    B is (n by n by n), given a fixed W_0:
        B = einsum('km,ijm->ijk', W_0, partial(S)/partial(W)@W_0)
        partial(S)/partial(W) is a (n by n by n) tensor given fixed W_0

    See derivative_of_W_times_S(S)_1.PNG and _2.PNG for my hand calculations.

    '''
    def test_derivative_of_product_of_two_matrix_functions(self):
        def _s(w):
            assert w.shape == (3,3)
            result = np.zeros(3)
            result[0] = w[0][0] * 3 + w[0][1] * 2 + w[0][2] ** 2
            result[1] = w[1][0] * 4 + w[1][1] * 3 + w[1][2] ** 3
            result[2] = w[2][0] * 5 + w[2][1] * 4 + w[2][2] ** 4

            return result

        def _f(w):
            assert w.shape == (3, 3)
            return np.matmul(w, _s(w))


        def _partial_s_partial_w(w_0):
            assert w_0.shape == (3,3)
            result = np.zeros([3, 3, 3])

            result[0][0][:] = [3, 0, 0]
            result[0][1][:] = [2, 0, 0]
            result[0][2][:] = [2 * w_0[0][2], 0, 0]

            result[1][0][:] = [0, 4, 0]
            result[1][1][:] = [0, 3, 0]
            result[1][2][:] = [0, 3 * w_0[1][2] ** 2, 0]

            result[2][0][:] = [0, 0, 5]
            result[2][1][:] = [0, 0, 4]
            result[2][2][:] = [0, 0, 4 * w_0[2][2] ** 3]

            return result

        def _numerical_derivative(func, w_0, i, j, delta_x_scalar):
            assert w_0.shape == (3, 3)
            assert np.isscalar(delta_x_scalar)

            delta_x_matrix = np.zeros([3, 3])
            delta_x_matrix[i][j] = delta_x_scalar

            output_delta = func(w_0 + delta_x_matrix) - func(w_0)
            return output_delta / delta_x_scalar


        def _compare_numerical_vs_theoretical_derivative(
            func, theoretical_derivative, w_0):
            theoretical = theoretical_derivative(w_0)
            for i in range(3):
                for j in range(3):
                    numerical = _numerical_derivative(func, w_0, i, j, 1e-5)
                    logger.debug("(i,j)=(%d, %d): theoretical derivative=\n%s\n, numerical derivative=\n%s\n",
                        i, j, numerical, theoretical[i][j])
                    np.testing.assert_almost_equal(
                        numerical, theoretical[i][j], 2)


        def _partial_f_partial_w(w_0):
            s_w_0 = _s(w_0)

            # See my notes, image: derivative_of_W_times_S(S)
            sum_A = np.zeros([3, 3, 3])
            for i in range(3):
                for j in range(3):
                    sum_A[i][j][i] = s_w_0[j]

            sum_B = np.einsum('km,ijm->ijk', w_0, _partial_s_partial_w(w_0))

            return sum_A + sum_B

        w_0 = np.zeros(9).reshape([3, 3])
        _compare_numerical_vs_theoretical_derivative(_s, _partial_s_partial_w, w_0)
        w_0 = np.ones(9).reshape([3, 3])
        _compare_numerical_vs_theoretical_derivative(_s, _partial_s_partial_w, w_0)
        w_0 = np.random.randn(9).reshape([3, 3])
        _compare_numerical_vs_theoretical_derivative(_s, _partial_s_partial_w, w_0)

        w_0 = np.zeros(9).reshape([3, 3])
        _compare_numerical_vs_theoretical_derivative(_f, _partial_f_partial_w, w_0)
        w_0 = np.ones(9).reshape([3, 3])
        _compare_numerical_vs_theoretical_derivative(_f, _partial_f_partial_w, w_0)
        w_0 = np.random.randn(9).reshape([3, 3])
        _compare_numerical_vs_theoretical_derivative(_f, _partial_f_partial_w, w_0)


    def test_convert_einsum_to_matmul(self):
        matrix_a = np.random.normal(0, 1, [9, 10])
        matrix_b = np.random.normal(0, 1, [6, 7, 10])

        result1 = np.einsum('km,ijm->ijk', matrix_a, matrix_b)

        result2 = np.matmul(matrix_b, matrix_a.T)

        np.testing.assert_almost_equal(result1, result2)


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    unittest.main()