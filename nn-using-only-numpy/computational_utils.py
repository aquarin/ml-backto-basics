# RNN Using only NP, from https://github.com/gy910210/rnn-from-scratch

import math
import numpy as np
import logging

import datetime
import sys

import os

import warnings

# To suppress warning in backwardPreAllocated(), np.dot(W.T, dz, out=pre_allocated_dx). I did use ndarrays, but still a
# "the matrix subclass is not the recommended way to represent matrices or deal with linear algebra" warning came.
warnings.filterwarnings("ignore",category=PendingDeprecationWarning)


logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class Utilities:
    @staticmethod
    def softmax(logits_vector):
        assert logits_vector.ndim == 1

        exp_scores = np.exp(logits_vector)
        return exp_scores / np.sum(exp_scores)


    @staticmethod
    def softmax_derivative(logits_vector_x_0):
        assert logits_vector_x_0.ndim == 1
        size = logits_vector_x_0.size

        softmax_values_0 = Utilities.softmax(logits_vector_x_0)

        # Using a 1d array of objects (which will be arrays), rather than 2-d array, due to 
        # derivative tests relying on array of objects.
        # i-th element of jacobian is the derivative w.r.t. i-th parameter.  
        jacobian = np.full(size, None, dtype=object)

        # Deliberately putting j before i, so that the indices match that of my notes, which
        # wrote d(s_i)/d(a_j), s_i is the softmax result on i-th, and a_j is the logit raw on j-th
        # Deliberately using for loop rather than matrix outer product so that it's easier to understand.
        for j in range(size):
            jacobian[j] = np.zeros(size)
            for i in range(size):
                jacobian[j][i] = (
                    - softmax_values_0[i] * softmax_values_0[j] if i != j
                    else softmax_values_0[j] * (1 - softmax_values_0[i])
                )

        return jacobian

    @staticmethod
    def loss_from_logits(logits_vector, label_y_as_integer):
        assert logits_vector.ndim == 1
        assert isinstance(label_y_as_integer, int)
        assert label_y_as_integer < logits_vector.size and label_y_as_integer >= 0

        softmax_vector = Utilities.softmax(logits_vector)
        loss = - math.log(softmax_vector[label_y_as_integer])
        return loss


    @staticmethod
    def loss_from_logits_derivative_wrt_logits(logits_vector_0, label_y_as_integer):
        assert logits_vector_0.ndim == 1
        assert isinstance(label_y_as_integer, int)
        assert label_y_as_integer < logits_vector_0.size and label_y_as_integer >= 0

        # TODO: optimization chance here. 
        softmax_vector_0 = Utilities.softmax(logits_vector_0)

        # According to my notes, partial(Loss)/partial(logit_j) =
        #    probs_j - Y_j
        # Y_j is the one-hot vector [0, .. 0, 1, 0, ... 0] with only 1 at label_y_as_integer
        jacobian = softmax_vector_0
        jacobian[label_y_as_integer] -= 1

        return jacobian


    @staticmethod
    def loss_from_matrix_v_and_hidden_state(hiddden_state_vector, matrix_v, label_y_as_integer):
        assert hiddden_state_vector.ndim == 1
        assert matrix_v.ndim == 2
        assert matrix_v.shape[1] == hiddden_state_vector.size
        assert isinstance(label_y_as_integer, int)

        output_softmax_size = matrix_v.shape[0]
        assert label_y_as_integer >= 0 and label_y_as_integer < output_softmax_size

        logits = np.matmul(matrix_v, hiddden_state_vector)
        probs = Utilities.softmax(logits)
        loss = - math.log(probs[label_y_as_integer])

        return loss


    @staticmethod
    def loss_from_matrix_v_and_hidden_state_derivative_wrt_v(hiddden_state_vector_0, matrix_v_0, label_y_as_integer):
        assert hiddden_state_vector_0.ndim == 1
        assert matrix_v_0.ndim == 2
        assert matrix_v_0.shape[1] == hiddden_state_vector_0.size
        assert isinstance(label_y_as_integer, int)

        probs_minus_y_one_shot = Utilities.softmax(np.matmul(matrix_v_0, hiddden_state_vector_0))
        probs_minus_y_one_shot[label_y_as_integer] -= 1

        return np.outer(probs_minus_y_one_shot, hiddden_state_vector_0)


    @staticmethod
    def loss_from_matrix_v_and_hidden_state_derivative_wrt_s(hiddden_state_vector_0, matrix_v_0, label_y_as_integer):
        assert hiddden_state_vector_0.ndim == 1
        assert matrix_v_0.ndim == 2
        assert matrix_v_0.shape[1] == hiddden_state_vector_0.size
        assert isinstance(label_y_as_integer, int)

        probs_minus_y_one_shot = Utilities.softmax(np.matmul(matrix_v_0, hiddden_state_vector_0))
        probs_minus_y_one_shot[label_y_as_integer] -= 1

        # is of size (1, hiddden_state_vector_0.size), each element derivative[i] is a derivative w.r.t. hidden_state_vector[i]
        # still supposed to return a Jacobian matrix whose first layer is a matrix of objects. 
        # In this case, this jacboain is a 1 x hiddden_state_vector_0.size object matrix, with each object to be a 1x1 matrix that's
        # the derivative. And I just returned a [1 x hiddden_state_vector_0.size] matrix instead, and the rest of the verification
        # code still worked.
        derivative = np.matmul(probs_minus_y_one_shot, matrix_v_0)

        return derivative


    @staticmethod
    def loss_from_matrix_U(input_x_integer, matrix_u, prev_s_times_w_result_vector, matrix_v, label_y_as_integer):
        assert isinstance(label_y_as_integer, int)
        assert isinstance(input_x_integer, int)
        assert prev_s_times_w_result_vector.ndim == 1

        hidden_dim = matrix_u.shape[0]
        vocab_dim = matrix_u.shape[1]
        assert prev_s_times_w_result_vector.size == hidden_dim
        assert matrix_v.shape[0] == vocab_dim
        assert matrix_v.shape[1] == hidden_dim
