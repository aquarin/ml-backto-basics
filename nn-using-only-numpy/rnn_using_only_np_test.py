import logging
import math
import random
import unittest

import numpy as np

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

if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    unittest.main()