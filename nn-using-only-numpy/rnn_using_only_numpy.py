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


    # Computes the gradient of loss function at one input in a sequence. Due to bptt calculation, the whole sequence computation states should be
    # provided. But only current time's label y needs to be provided.
    @staticmethod
    def loss_gradient_u_v_w(current_time, forward_computation_intermediates_array, label_y_as_integer, dim_vocab, dim_hidden, bptt_truncation_len=10,
        check_shapes=True):
        assert isinstance(current_time, int) and current_time >= 0 and current_time < len(forward_computation_intermediates_array)

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
            dim_vocab=dim_vocab, matrix_w=matrix_w, matrix_u=matrix_u, truncation_len=10, check_shapes=True, debug_output_dict=None, print_debug=False)

        logger.debug('#### step=%d, partial_loss_partial_new_state=\n%s\n, shape=%s\n partial_state_partial_u=\n%s\nshape=%s',
            current_time, partial_loss_partial_new_state, np.shape(partial_loss_partial_new_state),
            partial_state_partial_u, np.shape(partial_state_partial_u))
        if check_shapes:
            # Why not (dim_hidden, dim_vocab, 1, dim_hidden) (a 2d matrix of column vectors)? Because I have messed up the transposition in somewhere.
            assert partial_state_partial_u.shape == (dim_hidden, dim_vocab, dim_hidden)
         
        # Frankly, I don't totally understand the mechanism behind "i,ijkl->il". I want to do a multiplication between a row vector and a 3d matrix.
        # I just figured this out by trial and error. Probably I should learn why this worked, when I have more time.
        partial_loss_partial_u = np.einsum('k,ijk->ij', partial_loss_partial_new_state, partial_state_partial_u)
        logger.debug('#### step=%d, partial_state_partial_u=\n%s\n, partial_loss_partial_u=\n%s\n,shape=%s',
            current_time, partial_state_partial_u, partial_loss_partial_u, np.shape(partial_loss_partial_u))

        if check_shapes:
            assert partial_loss_partial_u.shape == matrix_u.shape

        # Computing the Jacobian matrix of: partial(loss)/partial(matrix_w)
        # See comments within computational_utils.py, loss_from_matrix_w_derivative_wrt_w()

        partial_state_partial_w = RnnWithNumpy.bptt_partial_state_time_t_partial_matrix_w(
            states_array_indexed_by_time=states_array_indexed_by_time, current_time=current_time,
            dim_hidden=dim_hidden, matrix_w=matrix_w, truncation_len=bptt_truncation_len,
            check_shapes=check_shapes, debug_output_dict=None, print_debug=True)

        partial_loss_partial_w = (np.zeros([dim_hidden, dim_hidden]) if np.isscalar(partial_state_partial_w) and partial_state_partial_w == 0
            else np.matmul(partial_loss_partial_new_state, partial_state_partial_w))

        # Embarrassingly, I don't fully know the reason I need to do a vstack() then a .T here.
        # Likely due to the following reasons, but I don't have time to drill-down to confirm them, as I am short on time and just want to get
        # this RNN running --
        # 1. In the input, state vector is a 1 x dim_hidden row vector. On my paper calculations, it's treated as a column vector. And np automatically 
        #    transposes them (or the results), when I called np.matmul(matrix, row_vector), treating it as if a matmul(matrix, col_vector)
        # 2. Due to 1, some mis-transpositions happened during the process, I don't know where. If I have time, I can dig them out and correct the code,
        #    rather than doing this patch-up here.
        partial_loss_partial_w = np.vstack(partial_loss_partial_w).T

        if check_shapes:
            pass
            # assert partial_loss_partial_w.shape == matrix_w.shape

        return (partial_loss_partial_u, partial_loss_partial_v, partial_loss_partial_w)


    @staticmethod
    def sequence_loss_gradient_u_v_w(forward_computation_intermediates_array,
        label_y_int_array, dim_vocab, dim_hidden, bptt_truncation_len=10, check_shapes=True):
        assert len(forward_computation_intermediates_array) == len(label_y_int_array)

        partial_sequential_loss_partial_u = 0
        partial_sequential_loss_partial_v = 0
        partial_sequential_loss_partial_w = 0

        for t in range(len(forward_computation_intermediates_array)):
            (partial_loss_partial_u, partial_loss_partial_v, partial_loss_partial_w) = RnnWithNumpy.loss_gradient_u_v_w(
                current_time=t,
                forward_computation_intermediates_array=forward_computation_intermediates_array,
                label_y_as_integer=label_y_int_array[t], dim_vocab=dim_vocab, dim_hidden=dim_hidden,
                check_shapes=check_shapes)

            partial_sequential_loss_partial_u += partial_loss_partial_u / float(len(forward_computation_intermediates_array))
            logger.debug('step=%d, single step partial_loss_partial_u=%s', t, partial_loss_partial_u)
            partial_sequential_loss_partial_v += partial_loss_partial_v / float(len(forward_computation_intermediates_array))
            partial_sequential_loss_partial_w += partial_loss_partial_w / float(len(forward_computation_intermediates_array))

        return (partial_sequential_loss_partial_u, partial_sequential_loss_partial_v, partial_sequential_loss_partial_w)


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


    # See my notes how the recursive calculation of this one.
    @staticmethod
    def bptt_partial_state_time_t_partial_matrix_u(states_array_indexed_by_time, input_x_time_series, current_time, dim_hidden,
        dim_vocab, matrix_w, matrix_u, truncation_len=10, check_shapes=True, debug_output_dict=None, print_debug=False):
        assert len(input_x_time_series) == len(states_array_indexed_by_time)

        # Note that it's comparing current_time to -1, not 0 (like in the case of W). When t=0, /partial(matrix_u) still has value
        # , unlike the case of matrix W.
        if truncation_len <= 0 or current_time <= -1:
            logger.debug('Returning 0 due to hitting truncation or current_time == 0, truncation_len=%d, current_time=%d'
                % (truncation_len, current_time))

            # TODO: check if just a simple zero works, or should I need a size-aligned three dimensional zero matrix.
            return 0

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

        logger.debug('x_onehot=\n%s\nshape=%s', x_onehot, np.shape(x_onehot))

        # Can't do this with kronecker product or einsum(). Looked online, asked ChatGPT, answers were wrong.
        # E.g. this doesn't work: partial_u_times_x_onehot_partial_u = np.einsum('ij,kl->ijl', np.eye(dim_hidden, dim_hidden), x_onehot)
        # And this is wrong: partial(M*v)/partial(M) = kron(eye(), v.T)
        # So have to manually construct this 3d matrix by hand.
        partial_u_times_x_onehot_partial_u = np.zeros([dim_hidden, dim_vocab, dim_hidden])
        for i in range(dim_hidden):
            partial_u_times_x_onehot_partial_u[i][input_x_as_integer][i] = 1.0

        logger.debug('### partial_u_times_x_onehot_partial_u=\n%s\n, shape=%s',
            partial_u_times_x_onehot_partial_u, partial_u_times_x_onehot_partial_u.shape)

        # Finding its value recursively (bptt)
        partial_state_t_minus_1_partial_u = RnnWithNumpy.bptt_partial_state_time_t_partial_matrix_u(
            states_array_indexed_by_time, input_x_time_series, current_time - 1, dim_hidden,
            dim_vocab, matrix_w, matrix_u, truncation_len=truncation_len - 1, check_shapes=check_shapes,
            debug_output_dict=debug_output_dict, print_debug=print_debug)
        logger.debug('### current_time=%d, partial_state_t_minus_1_partial_u=\n%s\n, shape=%s',
            current_time,
            partial_state_t_minus_1_partial_u, np.shape(partial_state_t_minus_1_partial_u))

        w_mul_partial_state_minus_1_partial_u = (
            0 if (np.isscalar(partial_state_t_minus_1_partial_u) and partial_state_t_minus_1_partial_u == 0)
            # Why a transposition on matrix W here? I don't know. But putting a .T here makes things right.
            # I am so messed up with the matrix calculus thing now. There is no documents to look up to know the right
            # chain rules...
            else np.einsum('ij,jkl->ikl', matrix_w.T, partial_state_t_minus_1_partial_u))

        # assert np.sahpe(w_mul_partial_state_minus_1_partial_u.shape) in ((), ())

        logger.debug('### w_mul_partial_state_minus_1_partial_u=\n%s\n',
            w_mul_partial_state_minus_1_partial_u)
        logger.debug('### partial_state_partial_raw_state=\n%s\n, shape=%s',
            partial_state_partial_raw_state, partial_state_partial_raw_state.shape)

        temp = partial_u_times_x_onehot_partial_u + w_mul_partial_state_minus_1_partial_u
        logger.debug('### partial_u_times_x_onehot_partial_u + w_mul_partial_state_minus_1_partial_u=\n%s\n, shape=%s',
            temp, temp.shape)
        partial_state_partial_matrix_u = np.einsum('ij,jkl->ikl', 
            partial_state_partial_raw_state,
            partial_u_times_x_onehot_partial_u + w_mul_partial_state_minus_1_partial_u)

        '''
        partial_state_partial_matrix_u = np.matmul(partial_state_partial_raw_state,
            partial_u_times_x_onehot_partial_u + w_mul_partial_state_minus_1_partial_u)
        '''

        return partial_state_partial_matrix_u


    def step_parameters(self, loss_gradient_u_v_w, step_size, check_shapes=True):
        (partial_loss_partial_u, partial_loss_partial_v, partial_loss_partial_w) = loss_gradient_u_v_w
        if check_shapes:
            assert partial_loss_partial_u.shape == self.matrix_u.shape
            assert partial_loss_partial_v.shape == self.matrix_v.shape
            assert partial_loss_partial_w.shape == self.matrix_w.shape

        self.matrix_u -= partial_loss_partial_u.shape * step_size
        self.matrix_v -= partial_loss_partial_v.shape * step_size
        self.matrix_w -= partial_loss_partial_w.shape * step_size




