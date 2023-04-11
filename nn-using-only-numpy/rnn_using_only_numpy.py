'''
Implementing RNN with only numpy, not using any other frameworks (such as PyTorch or Tensorflow)

Purpose of doing this:
  * Help myself go back to the ML basics and get better understanding, rather than using wrapped libraries.
  * Using this code to clearly explain what happens in an RNN to help others.
  * Performance optimization is not a goal, code clarity is more important.

  NOTES: there is no bias term in this model. Probably I should add bias some time later.
'''

import concurrent.futures
import logging
import math
import numbers
import numpy as np
import pickle
import threading
import time

from computational_utils import Utilities
from adam_optimizer import AdamOptimizer

logger = logging.getLogger(__name__)

# Can't put this as a member method when using ThreadWorkerPool.
# https://stackoverflow.com/questions/17419879/why-i-cannot-use-python-module-concurrent-futures-in-class-method
def thread_worker_method(args):
    def _get_thread_local_logger():
        local = threading.local()
        logger = getattr(local, 'logger', None)
        if logger is None:
            logger = logging.getLogger()
            logger.setLevel(logging.DEBUG)
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            local.logger = logger

        return logger

    logger = _get_thread_local_logger()

    (model, input_x_int_sequence, y_label_int_sequence) = args
    assert np.ndim(input_x_int_sequence) == 1 and np.ndim(y_label_int_sequence) == 1

    self = model
    forward_computation_intermediates_array = self.forward_sequence(
        input_x_int_array=input_x_int_sequence, dim_vocab=self.dim_vocab, dim_hidden=self.dim_hidden,
        matrix_u=self.matrix_u, matrix_v=self.matrix_v, matrix_w=self.matrix_w,
        start_state_vector=None, check_shapes=False, print_debug=False)

    sequential_loss = self.sequence_loss_from_forward_computations(
        dim_vocab=self.dim_vocab, forward_computation_intermediates_array=forward_computation_intermediates_array,
        label_y_int_array=y_label_int_sequence, check_shapes=False)

    sequential_loss_gradient_uvw = (
        self.sequence_loss_gradient_u_v_w(forward_computation_intermediates_array=forward_computation_intermediates_array,
            label_y_int_array=y_label_int_sequence, dim_vocab=self.dim_vocab, dim_hidden=self.dim_hidden,
            bptt_truncation_len=10, check_shapes=False))

    return (sequential_loss, sequential_loss_gradient_uvw)


class RnnWithNumpy:
    def __init__(self, dim_vocab, dim_hidden):
        self.dim_vocab = dim_vocab
        self.dim_hidden = dim_hidden

        # Glorot initialization, Gaussian distribution version.
        u_variance = math.sqrt(2.0 / dim_vocab)
        # Matrix U is the one that transforms input one-hot vector into its embedding.
        self.matrix_u = np.random.normal(0.0, u_variance, [dim_hidden, dim_vocab])

        # Transforms post-activation (tanh) state into logits (output embedding)
        v_variance = math.sqrt(2.0 / dim_hidden)
        self.matrix_v = np.random.normal(0.0, v_variance, [dim_vocab, dim_hidden])

        # Transforms previous state (s[t-1]) into part of next state, before activation.
        w_variance = math.sqrt(2.0 / dim_hidden)
        self.matrix_w = np.random.normal(0.0, w_variance, [dim_hidden, dim_hidden])

        # No hidden state vector here as member variable. That will be held in each training/prediction's
        # own function call variables.
        self.prev_state_vector = np.zeros(dim_hidden)

        self.adam_optimizer_u = AdamOptimizer(alpha=1) # Set learning rate to 1, later multiply the actual earning rate.
        self.adam_optimizer_v = AdamOptimizer(alpha=1)
        self.adam_optimizer_w = AdamOptimizer(alpha=1)

        self.thread_worker_count = 10 # TODO: parameterize this.


    @staticmethod
    def forward(input_x_as_integer, dim_vocab, dim_hidden, matrix_u, matrix_v, matrix_w, prev_state_vector,
        check_shapes=True, print_debug=False):
        if check_shapes:
            assert isinstance(input_x_as_integer, numbers.Integral)
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
        prev_state_vector = start_state_vector if start_state_vector is not None else np.zeros(dim_hidden)

        if check_shapes:
            assert matrix_u.ndim == 2
            assert matrix_v.ndim == 2
            assert matrix_w.ndim == 2
            assert matrix_u.shape[0] == dim_hidden and matrix_u.shape[1] == dim_vocab
            assert matrix_v.shape[0] == dim_vocab and matrix_v.shape[1] == dim_hidden
            assert matrix_w.shape[0] == matrix_w.shape[1] == dim_hidden
            assert prev_state_vector.ndim == 1
            assert prev_state_vector.size == dim_hidden

        forward_computation_intermediates_array = []

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
    def sequence_loss_from_forward_computations(dim_vocab, forward_computation_intermediates_array, label_y_int_array, check_shapes=True):
        probabilities_time_series = list(map(lambda computed: computed['softmax_probabilities'], forward_computation_intermediates_array))
        assert len(probabilities_time_series) == len(label_y_int_array)
        return RnnWithNumpy.sequence_loss(dim_vocab, probabilities_time_series, label_y_int_array, check_shapes=check_shapes)


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


    def forward_and_predict_carry_state(self, input_x_as_integer, check_shapes=True):
        if check_shapes:
            assert isinstance(input_x_as_integer, numbers.Integral)
            assert input_x_as_integer >= 0 and input_x_as_integer < self.dim_vocab

        forward_computation_intermediates = self.forward(
            input_x_as_integer, dim_vocab=self.dim_vocab, dim_hidden=self.dim_hidden,
            matrix_u=self.matrix_u, matrix_v=self.matrix_v, matrix_w=self.matrix_w,
            prev_state_vector=self.prev_state_vector,
            check_shapes=check_shapes, print_debug=False)

        self.prev_state_vector = forward_computation_intermediates['current_state']

        return self.predict_with_softmax_output(
            self.dim_vocab, forward_computation_intermediates['softmax_probabilities'], check_shapes=check_shapes, print_debug=False)


    def reset_prev_state(self):
        self.prev_state_vector = np.zeros(self.dim_hidden)


    def predict_sequence(self, input_x_int_sequence, check_shapes=True):
        if check_shapes:
            assert np.ndim(input_x_int_sequence) == 1

        forward_computation_intermediates_array = self.forward_sequence(
            input_x_int_array=input_x_int_sequence, dim_vocab=self.dim_vocab, dim_hidden=self.dim_hidden,
            matrix_u=self.matrix_u, matrix_v=self.matrix_v, matrix_w=self.matrix_w,
            start_state_vector=None, check_shapes=check_shapes, print_debug=False)

        probabilities_time_series = list(map(lambda computed: computed['softmax_probabilities'], forward_computation_intermediates_array))

        result_ids = []
        for probabilities in probabilities_time_series:
            predicted_id = self.predict_with_softmax_output(dim_vocab=self.dim_vocab, softmax_probabilities=probabilities, check_shapes=check_shapes,
                print_debug=False)
            result_ids.append(predicted_id)

        return result_ids


    @staticmethod
    def loss(dim_vocab, softmax_probabilities, y_label_as_integer, check_shapes=True):
        if check_shapes:
            assert softmax_probabilities.ndim == 1 and softmax_probabilities.size == dim_vocab
            assert isinstance(y_label_as_integer, int)
            assert y_label_as_integer >= 0 and y_label_as_integer < dim_vocab
            assert (softmax_probabilities >= 0).all() and (softmax_probabilities <= 1).all()

        return - math.log(softmax_probabilities[y_label_as_integer])


    # Computes the gradient of loss function at one input in a sequence. Due to bptt calculation, the whole sequence computation states should be
    # provided. But only current time's label y needs to be provided.
    @staticmethod
    def loss_gradient_u_v_w(
        current_time, forward_computation_intermediates_array, label_y_as_integer, dim_vocab, dim_hidden,
        partial_state_partial_u_by_time_bptt_cache,
        partial_state_partial_w_by_time_bptt_cache,
        bptt_truncation_len=10,
        check_shapes=True):
        assert isinstance(current_time, int) and current_time >= 0 and current_time < len(forward_computation_intermediates_array)
        assert isinstance(partial_state_partial_u_by_time_bptt_cache, dict)
        assert isinstance(partial_state_partial_w_by_time_bptt_cache, dict)

        computed = forward_computation_intermediates_array[current_time]

        input_x_as_integer =  computed['input_x_as_integer']
        matrix_u = computed['matrix_u']
        matrix_v = computed['matrix_v']
        matrix_w = computed['matrix_w']
        prev_state = computed['prev_state_vector']
        matrix_u_times_x_onehot = computed['matrix_u_times_x_onehot']
        w_times_prev_state= computed['w_times_prev_state']
        current_state_before_activation = computed['current_state_before_activation']
        current_state = computed['current_state']
        logits = computed['logits']
        probabilities = computed['softmax_probabilities']

        if check_shapes:
            assert matrix_u_times_x_onehot.ndim == 1
            assert matrix_u_times_x_onehot.size == dim_hidden
            assert w_times_prev_state.ndim == 1
            assert w_times_prev_state.size == dim_hidden
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
            assert isinstance(label_y_as_integer, int)
            assert label_y_as_integer >= 0 and label_y_as_integer < dim_vocab

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

        states_array_indexed_by_time = list(map(lambda computed: computed['current_state'], forward_computation_intermediates_array))
        input_x_time_series = list(map(lambda computed: computed['input_x_as_integer'], forward_computation_intermediates_array))

        partial_state_partial_u = RnnWithNumpy.bptt_partial_state_time_t_partial_matrix_u(
            states_array_indexed_by_time=states_array_indexed_by_time,
            input_x_time_series=input_x_time_series, current_time=current_time, dim_hidden=dim_hidden,
            dim_vocab=dim_vocab, matrix_w=matrix_w, matrix_u=matrix_u,
            partial_state_partial_u_by_time_cache=partial_state_partial_u_by_time_bptt_cache,
            truncation_len=10, check_shapes=True, debug_output_dict=None, print_debug=False)

        if check_shapes:
            # Why not (dim_hidden, dim_vocab, 1, dim_hidden) (a 2d matrix of column vectors)? Because I have messed up the transposition in somewhere.
            assert partial_state_partial_u.shape == (dim_hidden, dim_vocab, dim_hidden)
         
        # Frankly, I don't totally understand the mechanism behind "i,ijkl->il". I want to do a multiplication between a row vector and a 3d matrix.
        # I just figured this out by trial and error. Probably I should learn why this worked, when I have more time.
        partial_loss_partial_u = np.einsum('k,ijk->ij', partial_loss_partial_new_state, partial_state_partial_u)

        if check_shapes:
            assert partial_loss_partial_u.shape == matrix_u.shape

        # Computing the Jacobian matrix of: partial(loss)/partial(matrix_w)
        # See comments within computational_utils.py, loss_from_matrix_w_derivative_wrt_w()

        partial_state_partial_w = RnnWithNumpy.bptt_partial_state_time_t_partial_matrix_w(
            states_array_indexed_by_time=states_array_indexed_by_time, current_time=current_time,
            dim_hidden=dim_hidden, matrix_w=matrix_w,
            partial_state_partial_w_by_time_cache=partial_state_partial_w_by_time_bptt_cache,
            truncation_len=bptt_truncation_len,
            check_shapes=check_shapes, debug_output_dict=None, print_debug=False)

        partial_loss_partial_w = (np.zeros([dim_hidden, dim_hidden]) if np.isscalar(partial_state_partial_w) and partial_state_partial_w == 0
            # My own crafted 3-dimensional "Jacobian" at element(i,j,m), meant partial(F_m)/partial(W_[i,j]). So I need to hand-write this 
            # einsum to do this multiplication. TODO: probably I should fix the order of the indices of higher dimension jacobians later.
            else np.einsum('m,ijm->ij', partial_loss_partial_new_state, partial_state_partial_w))

        if check_shapes:
            assert partial_loss_partial_w.shape == matrix_w.shape

        return (partial_loss_partial_u, partial_loss_partial_v, partial_loss_partial_w)


    @staticmethod
    def sequence_loss_gradient_u_v_w(forward_computation_intermediates_array,
        label_y_int_array, dim_vocab, dim_hidden, bptt_truncation_len=10, check_shapes=True):
        assert len(forward_computation_intermediates_array) == len(label_y_int_array)

        partial_sequential_loss_partial_u = 0
        partial_sequential_loss_partial_v = 0
        partial_sequential_loss_partial_w = 0

        partial_state_partial_u_by_time_bptt_cache = {}
        partial_state_partial_w_by_time_bptt_cache = {}

        for t in range(len(forward_computation_intermediates_array)):
            (partial_loss_partial_u, partial_loss_partial_v, partial_loss_partial_w) = RnnWithNumpy.loss_gradient_u_v_w(
                current_time=t,
                forward_computation_intermediates_array=forward_computation_intermediates_array,
                label_y_as_integer=label_y_int_array[t], dim_vocab=dim_vocab, dim_hidden=dim_hidden,
                partial_state_partial_u_by_time_bptt_cache=partial_state_partial_u_by_time_bptt_cache,
                partial_state_partial_w_by_time_bptt_cache=partial_state_partial_w_by_time_bptt_cache,
                check_shapes=check_shapes)

            partial_sequential_loss_partial_u += partial_loss_partial_u / float(len(forward_computation_intermediates_array))
            partial_sequential_loss_partial_v += partial_loss_partial_v / float(len(forward_computation_intermediates_array))
            partial_sequential_loss_partial_w += partial_loss_partial_w / float(len(forward_computation_intermediates_array))

        return (partial_sequential_loss_partial_u, partial_sequential_loss_partial_v, partial_sequential_loss_partial_w)


    # Returns a Jacobian matrix of partial(state_vector@time_t) / partial(matrix_w)
    # See my notes, due to state_vector@time_t is a function of F(W, state_vector@time_t_minus_1), this will be calculated recursively.
    @staticmethod
    def bptt_partial_state_time_t_partial_matrix_w(states_array_indexed_by_time, current_time, dim_hidden, matrix_w,
        partial_state_partial_w_by_time_cache,
        truncation_len=10,
        check_shapes=True, debug_output_dict=None, print_debug=False):

        assert isinstance(partial_state_partial_w_by_time_cache, dict)

        if truncation_len <= 0 or current_time <= 0:
            # The down-stream code will handle this 0, so that I wouldn't need to return a shape-aligned zero matrix.
            return 0

        if current_time in partial_state_partial_w_by_time_cache:
            return partial_state_partial_w_by_time_cache[current_time]

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
            partial_f_partial_param_1 = np.zeros([dim_hidden, dim_hidden, dim_hidden])
            for i in range(dim_hidden):
                for j in range(dim_hidden):
                    # Also, this equals to (partial(W)/partial(W)) * prev_state_vector. partial(W)/partial(W) is 4 dimensional [dim_hidden, dim_hidden, dim_hidden, dim_hidden] size
                    # matrix, with 1 at only (i, j, i, j), rest 0.
                    partial_f_partial_param_1[i][j][i] = prev_state_vector[j]

            return partial_f_partial_param_1

        # current_time == 0 already checked.
        partial_f_partial_param_1 = _create_partial_f_partial_param_1(states_array_indexed_by_time[current_time - 1])

        # See my notes.
        partial_f_partial_param_2 = matrix_w

        partial_prev_state_partial_matrix_w = RnnWithNumpy.bptt_partial_state_time_t_partial_matrix_w(
            states_array_indexed_by_time, current_time - 1, dim_hidden, matrix_w,
            partial_state_partial_w_by_time_cache=partial_state_partial_w_by_time_cache,
            truncation_len=truncation_len - 1, check_shapes=check_shapes,
            debug_output_dict=None, print_debug=print_debug)

        partial_state_raw_partial_w = partial_f_partial_param_1 + (
            0 if (
                np.isscalar(partial_prev_state_partial_matrix_w) and partial_prev_state_partial_matrix_w == 0)
            else np.einsum('km,ijm->ijk', partial_f_partial_param_2, partial_prev_state_partial_matrix_w))

        partial_state_partial_w = np.einsum('km,ijm->ijk', diag_matrix_partial_state_partial_state_raw, partial_state_raw_partial_w)

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

        partial_state_partial_w_by_time_cache[current_time] = partial_state_partial_w
        return partial_state_partial_w


    # See my notes how the recursive calculation of this one.
    @staticmethod
    def bptt_partial_state_time_t_partial_matrix_u(states_array_indexed_by_time, input_x_time_series, current_time, dim_hidden,
        dim_vocab, matrix_w, matrix_u, partial_state_partial_u_by_time_cache,
        truncation_len=10, check_shapes=True, debug_output_dict=None, print_debug=False):
        assert len(input_x_time_series) == len(states_array_indexed_by_time)
        assert isinstance(partial_state_partial_u_by_time_cache, dict)

        # Note that it's comparing current_time to -1, not 0 (like in the case of W). When t=0, /partial(matrix_u) still has value
        # , unlike the case of matrix W.
        if truncation_len <= 0 or current_time <= -1:
            # The down-stream code will handle this 0, so that I wouldn't need to return a shape-aligned zero matrix.
            return 0

        if current_time in partial_state_partial_u_by_time_cache:
            return partial_state_partial_u_by_time_cache[current_time]

        input_x_as_integer = input_x_time_series[current_time]
        current_state = states_array_indexed_by_time[current_time]

        if check_shapes:
            assert matrix_w.shape == (dim_hidden, dim_hidden)
            assert matrix_u.shape == (dim_hidden, dim_vocab)

            assert current_state.ndim == 1 and current_state.size == dim_hidden
            assert isinstance(input_x_as_integer, int)
            assert input_x_as_integer >= 0 and input_x_as_integer < dim_vocab

        # d(tanh)/d(x) = 1/(cosh(x)^2) = 1 - tanh(x)) ^ 2
        partial_state_partial_raw_state = np.diag(1 - current_state ** 2)


        x_onehot = np.zeros(dim_vocab).reshape(-1, 1)
        x_onehot[input_x_as_integer] = 1.0

        # Can't do this with kronecker product or einsum(). Looked online, asked ChatGPT, answers were wrong.
        # E.g. this doesn't work: partial_u_times_x_onehot_partial_u = np.einsum('ij,kl->ijl', np.eye(dim_hidden, dim_hidden), x_onehot)
        # And this is wrong: partial(M*v)/partial(M) = kron(eye(), v.T)
        # So have to manually construct this 3d matrix by hand.
        partial_u_times_x_onehot_partial_u = np.zeros([dim_hidden, dim_vocab, dim_hidden])
        for i in range(dim_hidden):
            partial_u_times_x_onehot_partial_u[i][input_x_as_integer][i] = 1.0

        # Finding its value recursively (bptt)
        partial_state_t_minus_1_partial_u = RnnWithNumpy.bptt_partial_state_time_t_partial_matrix_u(
            states_array_indexed_by_time, input_x_time_series, current_time - 1, dim_hidden,
            dim_vocab, matrix_w, matrix_u, partial_state_partial_u_by_time_cache=partial_state_partial_u_by_time_cache,
            truncation_len=truncation_len - 1, check_shapes=check_shapes,
            debug_output_dict=debug_output_dict, print_debug=print_debug)

        w_mul_partial_state_minus_1_partial_u = (
            0 if (np.isscalar(partial_state_t_minus_1_partial_u) and partial_state_t_minus_1_partial_u == 0)
            # Why a transposition on matrix W here? I don't know. But putting a .T here makes things right.
            # I am so messed up with the matrix calculus thing now. There is no documents to look up to know the right
            # chain rules...
            else np.einsum('ij,jkl->ikl', matrix_w.T, partial_state_t_minus_1_partial_u))

        # assert np.sahpe(w_mul_partial_state_minus_1_partial_u.shape) in ((), ())
        partial_state_partial_matrix_u = np.einsum('ij,jkl->ikl', 
            partial_state_partial_raw_state,
            partial_u_times_x_onehot_partial_u + w_mul_partial_state_minus_1_partial_u)

        '''
        partial_state_partial_matrix_u = np.matmul(partial_state_partial_raw_state,
            partial_u_times_x_onehot_partial_u + w_mul_partial_state_minus_1_partial_u)
        '''

        partial_state_partial_u_by_time_cache[current_time] = partial_state_partial_matrix_u
        return partial_state_partial_matrix_u


    # Only the very very primitive SGD. Not even having adapting learning rate. Should improve this later.
    # Step size is not used for now. Only relying on adam optimizer.
    def step_parameters(self, loss_gradient_u_v_w, step_size, check_shapes=True):
        (partial_loss_partial_u, partial_loss_partial_v, partial_loss_partial_w) = loss_gradient_u_v_w
        if check_shapes:
            assert partial_loss_partial_u.shape == self.matrix_u.shape
            assert partial_loss_partial_v.shape == self.matrix_v.shape
            assert partial_loss_partial_w.shape == self.matrix_w.shape

        suggested_delta_u = self.adam_optimizer_u.suggest_delta_x_from_graident(partial_loss_partial_u)
        suggested_delta_v = self.adam_optimizer_v.suggest_delta_x_from_graident(partial_loss_partial_v)
        suggested_delta_w = self.adam_optimizer_w.suggest_delta_x_from_graident(partial_loss_partial_w)

        # Multiplying step_size (same as learning rate) here, is equal to using that step_size as "alpha" inside the ADAM.
        self.matrix_u += suggested_delta_u * step_size
        self.matrix_v += suggested_delta_v * step_size
        self.matrix_w += suggested_delta_w * step_size


    def train(self, x_input_int_list_of_sequences, y_label_int_list_of_sequences, learning_rate,
        batch_size, max_epoch, batch_callback):
        assert len(x_input_int_list_of_sequences) == len(y_label_int_list_of_sequences)
        assert isinstance(batch_size, int)

        logger.info("Training started. dim_hidden=%d, dim_vocab=%d, thread_worker_count=%d",
            self.dim_hidden, self.dim_vocab, self.thread_worker_count)

        batch_avg_loss = 0
        batch_avg_loss_history = []
        batch_processed_count = 0
        trained_count = 0
        epoch_count = 0
        start_time = time.time()
        sequential_loss_gradient_uvw_mini_batch = []
        executor = concurrent.futures.ProcessPoolExecutor(max_workers=self.thread_worker_count) # TODO: parameterize this.

        def _update_weights_and_do_callback():
            nonlocal batch_avg_loss, batch_processed_count, trained_count, epoch_count, start_time, sequential_loss_gradient_uvw_mini_batch
            logger.info("Processed %d total training samples, speed=%f samples/sec. Epoch=%d, max epoch=%d, Last batch size = %d, last batch avg loss (rolling calculation) = %f. Calling callback...",
                trained_count, trained_count / (time.time() - start_time), epoch_count, max_epoch, batch_processed_count, batch_avg_loss)

            if (len(sequential_loss_gradient_uvw_mini_batch) > 0):
                sequential_loss_gradient_uvw = _mini_batch_gradient_to_avg_gradient(sequential_loss_gradient_uvw_mini_batch)
                sequential_loss_gradient_uvw_mini_batch = []
                self.step_parameters(loss_gradient_u_v_w=sequential_loss_gradient_uvw, step_size=learning_rate, check_shapes=True)

            if batch_callback is not None:
                batch_callback(self)

            batch_avg_loss = 0
            batch_processed_count = 0

        def _mini_batch_gradient_to_avg_gradient(sequential_loss_gradient_uvw_mini_batch):
            gradient_u = np.average(list(map(lambda tuple: tuple[0], sequential_loss_gradient_uvw_mini_batch)), axis=0)
            gradient_v = np.average(list(map(lambda tuple: tuple[1], sequential_loss_gradient_uvw_mini_batch)), axis=0)
            gradient_w = np.average(list(map(lambda tuple: tuple[2], sequential_loss_gradient_uvw_mini_batch)), axis=0)

            return (gradient_u, gradient_v, gradient_w)


        # TODO: abstract this out and make it a separate learning rate scheduler.
        def _new_learning_rate_if_plataeu(batch_loss_history, learning_rate):
            comparison_moving_window_size = 20
            is_plataeu_criteria = .95
            min_calls_since_last_adjustment = 20
            learning_rate_adjustment_ratio = .7

            _new_learning_rate_if_plataeu.calls_since_last_adjustment = getattr(_new_learning_rate_if_plataeu, 'calls_since_last_adjustment', -1)
            _new_learning_rate_if_plataeu.calls_since_last_adjustment += 1

            if _new_learning_rate_if_plataeu.calls_since_last_adjustment  < min_calls_since_last_adjustment:
                return learning_rate

            if len(batch_loss_history) <= 2 * comparison_moving_window_size:
                return learning_rate

            avg1 = np.average(batch_loss_history[-2 * comparison_moving_window_size : - comparison_moving_window_size])
            avg2 = np.average(batch_loss_history[- comparison_moving_window_size :])

            if avg2 > avg1 * is_plataeu_criteria:
                new_learning_rate = learning_rate * learning_rate_adjustment_ratio
                _new_learning_rate_if_plataeu.calls_since_last_adjustment = 0

                logger.info("Adjusting learning rate to %f", new_learning_rate)
                return new_learning_rate

            return learning_rate


        # Make the callback before any training. Sometimes this is useful to compare the effects, especially to view the text-generation quality throughout the trainings.
        batch_callback(self)

        while epoch_count < max_epoch:
            for batch_start_index in range(0, len(x_input_int_list_of_sequences), batch_size):
                input_x_int_sequence_mini_batch = x_input_int_list_of_sequences[batch_start_index : batch_start_index + batch_size]
                y_label_int_sequence_mini_batch = y_label_int_list_of_sequences[batch_start_index : batch_start_index + batch_size]
                training_thread_inputs = zip(
                    [self] * len(input_x_int_sequence_mini_batch), # Have to pass "self' as a parameter to the worker thread method, as I cannot use a member method to call executor.
                    input_x_int_sequence_mini_batch,
                    y_label_int_sequence_mini_batch)

                training_outputs = list(executor.map(thread_worker_method, list(training_thread_inputs)))

                (sequential_losses, sequential_loss_gradient_uvw_mini_batch) = zip(*training_outputs)

                batch_avg_loss = np.average(sequential_losses)
                batch_avg_loss_history.append(batch_avg_loss)
                batch_processed_count = len(training_outputs)
                trained_count += len(training_outputs)
                learning_rate = _new_learning_rate_if_plataeu(batch_avg_loss_history, learning_rate)
                _update_weights_and_do_callback()

            epoch_count += 1

