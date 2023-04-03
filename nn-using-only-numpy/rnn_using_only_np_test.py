import logging
import math
import random
import unittest

import numpy as np

from derivative_verifier import DerivativeVerifier
from rnn_using_only_numpy import RnnWithNumpy

logger = logging.getLogger(__name__)

class RnnUsingOnlyNumpyTest(unittest.TestCase):
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
            matrix_w = matrix_w, truncation_len=10, check_shapes=True, debug_output_dict=debug_output_dict)
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
            matrix_w = matrix_w, truncation_len=10, check_shapes=True, debug_output_dict=debug_output_dict)
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
            matrix_w = matrix_w, truncation_len=10, check_shapes=True, debug_output_dict=debug_output_dict, print_debug=True)
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
            matrix_w = matrix_w, truncation_len=10, check_shapes=True, debug_output_dict=debug_output_dict, print_debug=True)
        logger.debug('debug_output_dict=\n%s\n' % debug_output_dict)
        logger.debug('bptt_partial_state_partial_w_at_time_n=\n%s\n' % bptt_partial_state_partial_w_at_time_n)

        numerical_jacobian_diff = DerivativeVerifier.numerical_jacobian_diff_matrix(_forward_prop_wrapper, matrix_w, delta_x_scalar)
        numerical_jacobian_derivative = numerical_jacobian_diff / delta_x_scalar
        logger.debug('numerical_jacobian_derivative=\n%s\n' % numerical_jacobian_derivative)
        logger.debug('numerical_jacobian_derivative shape=%s' % str(numerical_jacobian_derivative.shape))
        logger.debug('numerical_jacobian_derivative[0][0] shape=%s' % str(numerical_jacobian_derivative[0][0].shape))

        for index, sub_matrix in np.ndenumerate(numerical_jacobian_derivative):
            np.testing.assert_almost_equal(sub_matrix, bptt_partial_state_partial_w_at_time_n[index], 5)


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    unittest.main()