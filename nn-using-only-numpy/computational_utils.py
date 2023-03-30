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
        assert hiddden_state_vector.ndim == 1, logger.debug('hidden_state_vector=%s' % hiddden_state_vector)
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
    def loss_from_matrix_u(input_x_integer, matrix_u, prev_s_times_w_result_vector, matrix_v, label_y_as_integer, print_debug=False):
        assert isinstance(label_y_as_integer, int)
        assert isinstance(input_x_integer, int)
        assert prev_s_times_w_result_vector.ndim == 1

        hidden_dim = matrix_u.shape[0]
        vocab_dim = matrix_u.shape[1]
        assert prev_s_times_w_result_vector.size == hidden_dim
        assert matrix_v.shape[0] == vocab_dim
        assert matrix_v.shape[1] == hidden_dim
        assert label_y_as_integer >= 0 and label_y_as_integer < vocab_dim
        assert input_x_integer >= 0 and input_x_integer < vocab_dim

        u_times_x_one_hot = matrix_u[:, input_x_integer]
        new_state_vector = np.tanh(prev_s_times_w_result_vector + u_times_x_one_hot)
        logits_vector = np.matmul(matrix_v, new_state_vector)

        if print_debug:
            logger.debug('u_times_x_one_hot=\n%s\n' % u_times_x_one_hot)
            logger.debug('loss_from_matrix_u() new_state_vector=\n%s\n' % new_state_vector)
            logger.debug('logits_vector=\n%s\n' % logits_vector)
            logger.debug('softmax_vector=\n%s\n' % Utilities.softmax(logits_vector))

        return Utilities.loss_from_logits(logits_vector, label_y_as_integer)


    @staticmethod
    def loss_from_matrix_u_derivative_wrt_u(
        input_x_integer, matrix_u_0, w_times_prev_state, matrix_v, label_y_as_integer, print_debug=False):

        assert isinstance(label_y_as_integer, int)
        assert isinstance(input_x_integer, int)
        assert w_times_prev_state.ndim == 1

        hidden_dim = matrix_u_0.shape[0]
        vocab_dim = matrix_u_0.shape[1]
        assert w_times_prev_state.size == hidden_dim
        assert matrix_v.shape[0] == vocab_dim
        assert matrix_v.shape[1] == hidden_dim
        assert label_y_as_integer >= 0 and label_y_as_integer < vocab_dim
        assert input_x_integer >= 0 and input_x_integer < vocab_dim

        u_times_x_one_hot = matrix_u_0[:, input_x_integer]
        before_tanh_vector = w_times_prev_state + u_times_x_one_hot

        new_state_vector = np.tanh(before_tanh_vector)
        logits_vector = np.matmul(matrix_v, new_state_vector)

        probs_minus_y_one_shot = Utilities.softmax(logits_vector)
        probs_minus_y_one_shot[label_y_as_integer] -= 1

        # partial(loss) / partial(s[j]), s is new state vector, and iterate j=0...(hidden_dim-1), to make this a vector.
        # this equals (probs_vector - y_one_hot_vector) times matrix_v. See my notes for more detailed calculations.
        partial_loss_partial_new_state = np.matmul(probs_minus_y_one_shot, matrix_v)

        # derivative of tanh is sech
        sech_before_tanh_vector = 1 / np.cosh(before_tanh_vector)

        # partial(loss) / partial(U(j, k)) = 0 when k != x. This is non-zero only for U(j, input_x_integer), a non-zero column.
        # use np.multiply here for a member-wise multiplication.
        partial_loss_partial_u_at_x = np.multiply(partial_loss_partial_new_state, sech_before_tanh_vector)

        # assert ((partial_loss_partial_u_at_x.ndim == 1) and (partial_loss_partial_u_at_x.size == hidden_dim)), 'partial_loss_partial_u_at_x has the wrong shape. It=%s' % partial_loss_partial_u_at_x

        partial_loss_partial_u = np.zeros([hidden_dim, vocab_dim])
        partial_loss_partial_u[:, input_x_integer] = partial_loss_partial_u_at_x

        if print_debug:
            logger.debug('u_times_x_one_hot=\n%s\n' % u_times_x_one_hot)
            logger.debug('before_tanh_vector=\n%s\n' % before_tanh_vector)
            logger.debug('loss_from_matrix_u_derivative_wrt_u() new_state_vector=\n%s\n' % new_state_vector)
            logger.debug('logits_vector=\n%s\n' % logits_vector)
            logger.debug('probs_minus_y_one_shot=\n%s\n' % probs_minus_y_one_shot)
            logger.debug('partial_loss_partial_new_state=\n%s\n' % partial_loss_partial_new_state)
            logger.debug('sech_before_tanh_vector=\n%s\n' % sech_before_tanh_vector)
            logger.debug('partial_loss_partial_u_at_x=\n%s\n' % partial_loss_partial_u_at_x)
            logger.debug('partial_loss_partial_u=\n%s\n' % partial_loss_partial_u)

        return partial_loss_partial_u


    # Quite a duplicate from method loss_from_matrix_u(). Well, just for verifications.
    @staticmethod
    def loss_from_matrix_w_prev_state(matrix_w, prev_state_vector, matrix_u_times_input_x, matrix_v, label_y_as_integer, print_debug=False):
        assert isinstance(label_y_as_integer, int)
        assert prev_state_vector.ndim == 1
        assert matrix_v.ndim == 2
        assert matrix_u_times_input_x.ndim == 1
        assert matrix_w.ndim == 2

        hidden_dim = prev_state_vector.size
        vocab_dim = matrix_v.shape[0]
        assert matrix_v.shape[1] == hidden_dim
        assert matrix_w.shape[0] == matrix_w.shape[1] == hidden_dim
        assert label_y_as_integer >= 0 and label_y_as_integer < vocab_dim

        w_times_prev_state = np.matmul(matrix_w, prev_state_vector)
        new_state_vector = np.tanh(w_times_prev_state + matrix_u_times_input_x)
        logits_vector = np.matmul(matrix_v, new_state_vector)

        if print_debug:
            logger.debug('matrix_u_times_input_x=\n%s\n' % matrix_u_times_input_x)
            logger.debug('matrix_w_prev_state() w_times_prev_state=\n%s\n' % w_times_prev_state)
            logger.debug('matrix_w_prev_state() new_state_vector=\n%s\n' % new_state_vector)
            logger.debug('logits_vector=\n%s\n' % logits_vector)
            logger.debug('softmax_vector=\n%s\n' % Utilities.softmax(logits_vector))

        return Utilities.loss_from_logits(logits_vector, label_y_as_integer)


    # Calculation of this derivative is in my notes. Quite similar to loss_from_matrix_u_derivative_wrt_u().
    @staticmethod
    def loss_from_matrix_u_derivative_wrt_u(matrix_w_0, prev_state_vector, matrix_u_times_input_x, matrix_v, label_y_as_integer, print_debug=False):
        assert isinstance(label_y_as_integer, int)
        assert prev_state_vector.ndim == 1
        assert matrix_v.ndim == 2
        assert matrix_u_times_input_x.ndim == 1
        assert matrix_w_0.ndim == 2

        hidden_dim = prev_state_vector.size
        vocab_dim = matrix_v.shape[0]
        assert matrix_v.shape[1] == hidden_dim
        assert matrix_w_0.shape[0] == matrix_w_0.shape[1] == hidden_dim
        assert label_y_as_integer >= 0 and label_y_as_integer < vocab_dim

        w_times_prev_state = np.matmul(matrix_w, prev_state_vector)
        before_tanh_vector = w_times_prev_state + matrix_u_times_input_x

        new_state_vector = np.tanh(before_tanh_vector)
        logits_vector = np.matmul(matrix_v, new_state_vector)

        probs_minus_y_one_shot = Utilities.softmax(logits_vector)
        probs_minus_y_one_shot[label_y_as_integer] -= 1

        # partial(loss) / partial(s[j]), s is new state vector, and iterate j=0...(hidden_dim-1), to make this a vector.
        # this equals (probs_vector - y_one_hot_vector) times matrix_v. See my notes for more detailed calculations.
        partial_loss_partial_new_state = np.matmul(probs_minus_y_one_shot, matrix_v)

        # derivative of tanh is sech
        sech_before_tanh_vector = 1 / np.cosh(before_tanh_vector)

        # According to my hand calculations,
        # partial(loss)/partial(W[j, k])
        # =  sum_m( (Pm - Ym) * V[m, j]) / cosh(before_tanh_vector[j]) * prev_state_vector[k]
        # Pm is probability[m] obtained by softmax(logit[m])
        # Ym is one element from one-hot representation of label_y_vector, with 1 only at label_y_as_integer
        partial_loss_partial_before_tanh_vector = np.matmul(partial_loss_partial_new_state, sech_before_tanh_vector)
        partial_loss_partial_w = np.outer(partial_loss_partial_before_tanh_vector, prev_state_vector)

        if print_debug:
            logger.debug('matrix_u_times_input_x=\n%s\n' % matrix_u_times_input_x)
            logger.debug('before_tanh_vector=\n%s\n' % before_tanh_vector)
            logger.debug('loss_from_matrix_u_derivative_wrt_u() new_state_vector=\n%s\n' % new_state_vector)
            logger.debug('logits_vector=\n%s\n' % logits_vector)
            logger.debug('probs_minus_y_one_shot=\n%s\n' % probs_minus_y_one_shot)
            logger.debug('partial_loss_partial_new_state=\n%s\n' % partial_loss_partial_new_state)
            logger.debug('sech_before_tanh_vector=\n%s\n' % sech_before_tanh_vector)
            logger.debug('partial_loss_partial_before_tanh_vector=\n%s\n' % partial_loss_partial_before_tanh_vector)
            logger.debug('partial_loss_partial_w=\n%s\n' % partial_loss_partial_w)

        assert partial_loss_partial_w.ndim == 2
        assert partial_loss_partial_w.shape[0] == partial_loss_partial_w.shape[1] == hidden_dim

        return partial_loss_partial_w

