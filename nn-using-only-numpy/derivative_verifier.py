import logging
import math
import random
import unittest

import numpy as np

# import computational_utils.Utilities as Utilities
from computational_utils import Utilities

logger = logging.getLogger(__name__)

'''
Verifies the derivative.
'''
class DerivativeVerifier:
    '''
    Tested, x_0, delta_x can both be matrices, and func can be a function of matrix to a scalar
    '''
    @staticmethod
    def getNumericalDiff(func, x_0, delta_x):
        y_0 = func(x_0)
        y_1 = func(x_0 + delta_x)
        return y_1 - y_0


    @staticmethod
    def theoreticalDerivativeEqualsToNumericalDiff(
        func, theoretical_derivative_of_func, x_0, delta_x_scalar=1e-7, error_relative_to_delta_x=1e-6, print_diff=False):

        assert np.isscalar(delta_x_scalar)

        comparison_logical_results = []

        if (np.isscalar(x_0)):
            # Case of returning value as scalar or vector or matrix, all handled.
            numerical_delta_y = DerivativeVerifier.getNumericalDiff(func, x_0, delta_x_scalar)
            theoretical_delta_y = np.dot(theoretical_derivative_of_func(x_0), delta_x_scalar)

            comparison_logical_results = abs(numerical_delta_y - theoretical_delta_y) < abs(delta_x_scalar * error_relative_to_delta_x)

            # TODO: what if this is a scalar to vector/matrix function?
        else:
            numerical_diff_jacobian = DerivativeVerifier.numericalJacobianDiffMatrix(func, x_0, delta_x_scalar)

            theoretical_diff_jacobian = theoretical_derivative_of_func(x_0)
            # Handle a matrix of objects
            theoretical_diff_jacobian *= delta_x_scalar

            assert numerical_diff_jacobian.shape == theoretical_diff_jacobian.shape

            diff_numerical_theoretical = numerical_diff_jacobian - theoretical_diff_jacobian
            comparison_logical_results = np.full(numerical_diff_jacobian.shape, False)

            for index, sub_matrix in np.ndenumerate(diff_numerical_theoretical):
                comparison_logical_results[index] = (np.abs(sub_matrix) < delta_x_scalar * error_relative_to_delta_x).all()

                if print_diff and not comparison_logical_results[index]:
                    logger.info('At index = %s, logical result = %s, diff_matrix=\n%s\n' % (index, comparison_logical_results[index], sub_matrix))

                if not comparison_logical_results[index]:
                    logger.info('numerical_jacobian at same index = \n%s\ntheoretical jacobian diff at same index=\n%s\n, delta_x=%f'
                        % (numerical_diff_jacobian[index], theoretical_diff_jacobian[index], delta_x_scalar))
                    logger.info('diff between numerical and theoretical = %s' % (numerical_diff_jacobian[index] - theoretical_diff_jacobian[index]))
                    logger.info('(diff between theo/numerical) divided by delta_x=%s' %
                        ((numerical_diff_jacobian[index] - theoretical_diff_jacobian[index]) / delta_x_scalar))


        return comparison_logical_results if np.isscalar(comparison_logical_results) else np.all(comparison_logical_results)


    @staticmethod
    def numericalJacobianDiffMatrix(func, matrix_x_0, delta_x_scalar):
        assert np.isscalar(delta_x_scalar)

        diff_jacobian = np.zeros_like(matrix_x_0, dtype=object)
        y_0 = func(matrix_x_0)
        for index, _ in np.ndenumerate(matrix_x_0):
            x_0_plus_delta = matrix_x_0.copy().astype(np.float64)
            x_0_plus_delta[index] += delta_x_scalar
            y_1 = func(x_0_plus_delta)
            output_diff_wrt_index = func(x_0_plus_delta) - y_0
            diff_jacobian[index] = output_diff_wrt_index

        return diff_jacobian


class DerivativeVerifierTests(unittest.TestCase):
    verifier = DerivativeVerifier()

    @staticmethod
    def _matrix_to_matrix_func1(matrix):
        assert matrix.shape == (2, 3)

        result = np.zeros([2, 2])
        result[0][0] = math.sin(matrix[0][0]) * matrix[0][1]
        result[0][1] = matrix[0][1]
        result[1][0] = math.exp(matrix[1][0]) * matrix[1][1]
        result[1][1] = matrix[1][2] ** 3

        return result

    @staticmethod
    def _false_d_fun1(matrix_x_0):
        def _false_d_func1_wrt_1_2(m):
            assert m.shape == (2, 3)
            return_value = np.zeros([2, 2])
            return_value[1][1] = 3 * (m[1][2] ** 4)
            return return_value

        jacobian = DerivativeVerifierTests._theoretical_d_func1(matrix_x_0)
        jacobian[1][2] = _false_d_func1_wrt_1_2(matrix_x_0)
        return jacobian


    @staticmethod
    def _theoretical_d_func1(matrix_x_0):
        assert matrix_x_0.shape == (2, 3)

        # The returned matrix is (2, 3) (parameter shape of func1) not (2, 2) (output shape of func1)
        # Deliberately wrote them in function form (rather than output matrix) for not skipping a step in
        #   the middle, and for better understanding.
        jacobian = np.zeros([2, 3])

        # Derivative with respect to parameter [0][0]
        def _d_func1_wrt_0_0(m):
            assert m.shape == (2, 3)
            return_value = np.zeros([2, 2])
            return_value[0][0] = math.cos(m[0][0]) * m[0][1]

            # Deliberately wrote the zeros (even though initialized as zero) for better understanding.
            return_value[0][1] = 0
            return_value[1][0] = 0
            return_value[1][1] = 0

            return return_value

        # Derivative with respect to parameter [0][1]
        def _d_func1_wrt_0_1(m):
            assert m.shape == (2, 3)
            return_value = np.zeros([2, 2])
            return_value[0][0] = math.sin(m[0][0])
            return_value[0][1] = 1

            # Deliberately wrote the zeros (even though initialized as zero) for better understanding.
            return_value[1][0] = 0
            return_value[1][1] = 0

            return return_value

        def _d_func1_wrt_0_2(m):
            assert m.shape == (2, 3)
            # all zero when deriving w.r.t. (0, 2)
            return_value = np.zeros([2, 2])
            return return_value

        def _d_func1_wrt_1_0(m):
            assert m.shape == (2, 3)
            return_value = np.zeros([2, 2])
            return_value[1][0] = math.exp(m[1][0]) * m[1][1]
            return return_value

        def _d_func1_wrt_1_1(m):
            assert m.shape == (2, 3)
            return_value = np.zeros([2, 2])
            return_value[1][0] = math.exp(m[1][0]) * 1
            return return_value

        def _d_func1_wrt_1_2(m):
            assert m.shape == (2, 3)
            return_value = np.zeros([2, 2])
            return_value[1][1] = 3 * (m[1][2] ** 2)
            return return_value

        def _jacobian(m):
            assert m.shape == (2, 3)
            result = np.full((2, 3), None, dtype=object)

            '''
            Cannot use the following due to creating a shape=(2, 3, 2, 2), 4-dimensional array.
            return np.array([
                [_d_func1_wrt_0_0(m), _d_func1_wrt_0_1(m), _d_func1_wrt_0_2(m)],
                [_d_func1_wrt_1_0(m), _d_func1_wrt_1_1(m), _d_func1_wrt_1_2(m)],
            ], dtype=object)
            '''
            result[0][0] = _d_func1_wrt_0_0(m)
            result[0][1] = _d_func1_wrt_0_1(m)
            result[0][2] = _d_func1_wrt_0_2(m)
            result[1][0] = _d_func1_wrt_1_0(m)
            result[1][1] = _d_func1_wrt_1_1(m)
            result[1][2] = _d_func1_wrt_1_2(m)

            return result

        result = _jacobian(matrix_x_0)

        return result


    def testSimpleFunc(self):
        def _func(x):
            return 2 * x

        delta_y = self.verifier.getNumericalDiff(_func, 0, .1)
        np.testing.assert_equal(delta_y, 0.2)

        np.testing.assert_almost_equal(
            self.verifier.getNumericalDiff(lambda x: x * x, 10, .1),
            2.01, 10)

        # Derivative of cosine at pi is 0. 0 times 1-e12 should still be zero.
        # Verify to the precision of the 20th decimals (1e-20).
        np.testing.assert_almost_equal(
            self.verifier.getNumericalDiff(math.cos, math.pi, 1e-12),
            0.0, 20)


    def testMatrixToScalarFunc(self):
        # m(i, j) -> cox(m(i, j) + n), n = i * size + j
        # Then sum all up to a scalar.
        def _scalar_func_of_matrix1(matrix):
            addition_matrix = np.arange(matrix.size).reshape(matrix.shape)
            cosine_matrix = np.cos(matrix + addition_matrix)
            return np.sum(cosine_matrix)

        x_0 = np.array([
            [math.pi, 0, 0],
            [math.pi, 1, 2],
            [3, 4, 5],
            [7, 7, 7],
        ])

        # derivative to matrix's [0][0] is d(cos(x + 0))/dx @ 0, = 0
        delta_x = np.zeros_like(x_0)
        delta_x[0][0] = 1e-10
        delta_y = self.verifier.getNumericalDiff(_scalar_func_of_matrix1, x_0, delta_x)
        np.testing.assert_almost_equal(delta_y, 0, 20)


        # derivative to matrix's [0][1] is d(cos(x + 3))/dx @pi, = -sin(3 + pi)
        delta_x = np.zeros_like(x_0)
        delta_x[1][0] = 1e-10
        delta_y = self.verifier.getNumericalDiff(_scalar_func_of_matrix1, x_0, delta_x)
        np.testing.assert_almost_equal(delta_y, - math.sin(3 + math.pi) * 1e-10, 15)

        # derivative to matrix's [3][2] is d(cos(x + 11))/dx @7, = - sin(18)
        delta_x = np.zeros_like(x_0)
        delta_x[3][2] = 1e-10
        delta_y = self.verifier.getNumericalDiff(_scalar_func_of_matrix1, x_0, delta_x)
        np.testing.assert_almost_equal(delta_y, - math.sin(18) * 1e-10, 15)


    def testMatrixToMatrixFunc(self):
        def _matrix_to_matrix_func1(matrix):
            addition_matrix = np.arange(matrix.size).reshape(matrix.shape)
            cosine_matrix = np.cos(matrix + addition_matrix)
            # Summing the rows
            return np.sum(cosine_matrix, axis=1)

        x_0 = np.array([
            [math.pi, 0, 0],
            [math.pi, 1, 2],
            [3, 4, 5],
            [7, 7, 7],
        ])

        # Just put this assert here so that this code's reader knows how _matrix_to_matrix_func() output is like
        np.testing.assert_almost_equal(_matrix_to_matrix_func1(x_0),
            np.array([-8.75844531e-01,  2.02755694e+00,  7.42217554e-04, -5.72506110e-01]), 8)

        delta_x = np.zeros_like(x_0)
        delta_x[0][0] = 1e-10
        delta_y = self.verifier.getNumericalDiff(_matrix_to_matrix_func1, x_0, delta_x)
        np.testing.assert_almost_equal(
            delta_y,
            np.array([0 , 0, 0, 0]),
            20)

        delta_x = np.zeros_like(x_0)
        delta_x[1][1] = 1e-10
        delta_y = self.verifier.getNumericalDiff(_matrix_to_matrix_func1, x_0, delta_x)
        np.testing.assert_almost_equal(
            delta_y,
            np.array([0 , math.cos(1 + 4 + 1e-10) - math.cos(1 + 4), 0, 0]),
            15)


    def testTheoreticalVsNumericalDelta(self):
        self.assertTrue(self.verifier.theoreticalDerivativeEqualsToNumericalDiff(
            lambda x: math.cos(x),
            lambda x: -math.sin(x),
            0))


        random_numbers = np.random.rand(100) * 4 * math.pi
        for x_0 in random_numbers:
            self.assertTrue(self.verifier.theoreticalDerivativeEqualsToNumericalDiff(
                lambda x: math.cos(x),
                lambda x: -math.sin(x),
                x_0))

            # Verify the wrong theoretical derivative is wrong
            self.assertFalse(self.verifier.theoreticalDerivativeEqualsToNumericalDiff(
                lambda x: math.cos(x),
                lambda x: math.cos(x),
                x_0))

    def testTheoreticalVsNumericalDeltaScalarToMatrix(self):
        # Scalar to vector function.
        self.assertTrue(self.verifier.theoreticalDerivativeEqualsToNumericalDiff(
            lambda x: np.array([math.cos(x), math.sin(x)]),
            lambda x: np.array([-math.sin(x), math.cos(x)]),
            0))

        random_numbers = np.random.rand(100) * 4 * math.pi
        for x_0 in random_numbers:
            self.assertTrue(self.verifier.theoreticalDerivativeEqualsToNumericalDiff(
                lambda x: math.cos(x),
                lambda x: -math.sin(x),
                x_0))

    def testNumericalJacobianDiffMatrix(self):
        x_0_matrix = np.array([
            [1, 2, 3],
            [11, 12, 13],
        ])

        delta = 1e-4
        numerical_diff_jacobian = self.verifier.numericalJacobianDiffMatrix(
            self._matrix_to_matrix_func1, x_0_matrix, delta)

        # _matrix_to_matrix_func1: result[0][0] = math.sin(matrix[0][0]) * matrix[0][1]
        np.testing.assert_almost_equal(
            numerical_diff_jacobian[0][0][0][0], math.cos(1) * 2 * delta, 6)
        np.testing.assert_almost_equal(
            numerical_diff_jacobian[0][0][0][1], 0, 6)
        np.testing.assert_almost_equal(
            numerical_diff_jacobian[0][0][1][0], 0, 6)
        np.testing.assert_almost_equal(
            numerical_diff_jacobian[0][0][1][1], 0, 6)

        # _matrix_to_matrix_func1: result[0][0] = math.sin(matrix[0][0]) * matrix[0][1]
        np.testing.assert_almost_equal(
            numerical_diff_jacobian[0][1][0][0], math.sin(1) * 1 * delta, 6)
        # _matrix_to_matrix_func1: result[0][1] = matrix[0][1]
        np.testing.assert_almost_equal(
            numerical_diff_jacobian[0][1][0][1], 1 * delta, 6)
        np.testing.assert_almost_equal(
            numerical_diff_jacobian[0][1][1][0], 0, 6)
        np.testing.assert_almost_equal(
            numerical_diff_jacobian[0][1][1][1], 0, 6)

        np.testing.assert_almost_equal(
            numerical_diff_jacobian[0][2][0][0], 0, 6)
        np.testing.assert_almost_equal(
            numerical_diff_jacobian[0][2][0][1], 0, 6)
        np.testing.assert_almost_equal(
            numerical_diff_jacobian[0][2][1][0], 0, 6)
        np.testing.assert_almost_equal(
            numerical_diff_jacobian[0][2][1][1], 0, 6)

        # _matrix_to_matrix_func1: result[1][0] = math.exp(matrix[1][0]) * matrix[1][1]
        np.testing.assert_almost_equal(
            numerical_diff_jacobian[1][0][0][0], 0, 6)
        np.testing.assert_almost_equal(
            numerical_diff_jacobian[1][0][0][1], 0, 6)
        np.testing.assert_almost_equal(
            numerical_diff_jacobian[1][0][1][0], math.exp(11) * 12 * delta, 2)
        np.testing.assert_almost_equal(
            numerical_diff_jacobian[1][0][1][1], 0, 6)

        np.testing.assert_almost_equal(
            numerical_diff_jacobian[1][1][0][0], 0, 6)
        np.testing.assert_almost_equal(
            numerical_diff_jacobian[1][1][0][1], 0, 6)
        np.testing.assert_almost_equal(
            numerical_diff_jacobian[1][1][1][0], math.exp(11) * 1 * delta, 6)
        np.testing.assert_almost_equal(
            numerical_diff_jacobian[1][1][1][1], 0, 6)

        # _matrix_to_matrix_func1: result[1][1] = matrix[1][2] ** 3
        np.testing.assert_almost_equal(
            numerical_diff_jacobian[1][2][0][0], 0, 6)
        np.testing.assert_almost_equal(
            numerical_diff_jacobian[1][2][0][1], 0, 6)
        np.testing.assert_almost_equal(
            numerical_diff_jacobian[1][2][1][0], 0, 6)
        np.testing.assert_almost_equal(
            numerical_diff_jacobian[1][2][1][1], (13 + delta) ** 3 - 13 ** 3, 6)


    def testTheoreticalVsNumericalDeltaMatrixToMatrix(self):
        x_0_matrix = np.array([
            [1, 2, 3],
            [11, 12, 13],
        ])

        delta = 1e-6

        result = self.verifier.theoreticalDerivativeEqualsToNumericalDiff(
            # Used error_relative_to_delta_x=1, as I verified for f(x) = exp(x) * 12 at x=11,
            # the diff(numerical_diff, theoretical_d_times_delta) is 0.3 times delta_x, when delta_x = 1e-6
            self._matrix_to_matrix_func1, self._theoretical_d_func1, x_0_matrix, delta, error_relative_to_delta_x=1,
            print_diff=True)
        self.assertTrue(result)
        self.assertFalse(
            self.verifier.theoreticalDerivativeEqualsToNumericalDiff(
            self._matrix_to_matrix_func1, self._false_d_fun1, x_0_matrix, delta, error_relative_to_delta_x=1,
            print_diff=True))

        x_0_matrix = np.array([
            [100, 200, 300],
            [11, math.pi, 130],
        ])
        result = self.verifier.theoreticalDerivativeEqualsToNumericalDiff(
            # Used error_relative_to_delta_x=1, as I verified for f(x) = exp(x) * 12 at x=11,
            # the diff(numerical_diff, theoretical_d_times_delta) is 0.3 times delta_x, when delta_x = 1e-6
            self._matrix_to_matrix_func1, self._theoretical_d_func1, x_0_matrix, delta, error_relative_to_delta_x=1,
            print_diff=True)
        self.assertTrue(result)
        self.assertFalse(
            self.verifier.theoreticalDerivativeEqualsToNumericalDiff(
            self._matrix_to_matrix_func1, self._false_d_fun1, x_0_matrix, delta, error_relative_to_delta_x=1,
            print_diff=True))


    def testMatrixEnumerator(self):
        a = np.array([
            [1, 2, 3],
            [4, 5, 6],
        ])

        # Only visual verification.
        for index, _ in np.ndenumerate(a):
            b = a.copy().astype(np.float64)
            b[index] += 0.1
            logger.debug('b=%s' % b)


    def __testTheoreticalVsNumericalDeltaForMatrix(self):
        self.assertTrue(self.verifier.theoreticalDerivativeEqualsToNumericalDiff(
            lambda x: np.cos(x),
            lambda x: -np.sin(x),
            0))

        random_numbers_x0 = np.random.rand(100) * 4 * math.pi
        self.assertTrue(self.verifier.theoreticalDerivativeEqualsToNumericalDiff(
            lambda x: np.cos(x),
            lambda x: -np.sin(x),
            random_numbers_x0))


'''
Plans:
  Verify the following --
     1) Theoretical derivative of softmax
     2) Theoretical derivative of the loss function
     3) Theoretical derivative of matrix V
     4) Theoretical derivative of hidden state S
     5) Theoretical derivative of tanh function
     6) Theoretical derivative of matrix U
     7) Theoretical derivative of matrix W
'''
class TestRNNDerivatives(unittest.TestCase):
    def testSoftmax(self):
        logger.debug(Utilities.softmax(np.array([1, 2, 3, 4, 5])))
        logger.debug(Utilities.softmax_derivative(np.array([1, 2, 3, 4, 5])))

        test = DerivativeVerifier.theoreticalDerivativeEqualsToNumericalDiff(
            Utilities.softmax, Utilities.softmax_derivative,
            np.array([1, 2, 3, 4, 5]),
            print_diff=True)
        self.assertTrue(test)

        test = DerivativeVerifier.theoreticalDerivativeEqualsToNumericalDiff(
            Utilities.softmax, Utilities.softmax_derivative,
            np.array([110]), print_diff=True)
        self.assertTrue(test)
        logger.debug('Theoretical derivative of softmax with one element at 110=%f'
            % Utilities.softmax_derivative(np.array([110])))
        logger.debug('Theoretical derivative of softmax with one element at pi=%f'
            % Utilities.softmax_derivative(np.array([math.pi])))

        test_vector_size = 300
        logger.info("Looping %d times to test derivative of softmax of a size %d vector..." % (100, test_vector_size))
        for test_count in range(100):
            logits_x_0 = np.random.rand(test_vector_size)
            test = DerivativeVerifier.theoreticalDerivativeEqualsToNumericalDiff(
                Utilities.softmax, Utilities.softmax_derivative,
                logits_x_0, print_diff=False)
            self.assertTrue(test)


    def testLossFromLogits(self):
        logger.debug("Some visual inspection of loss function from logits:")
        logger.debug(Utilities.loss_from_logits(np.array([1, 2, 3, 4, 5]), 0))
        logger.debug(Utilities.loss_from_logits(np.array([1, 2, 3, 4, 5]), 1))
        logger.debug(Utilities.loss_from_logits(np.array([1, 2, 3, 4, 5]), 2))
        logger.debug(Utilities.loss_from_logits(np.array([1, 2, 3, 4, 5]), 3))
        logger.debug(Utilities.loss_from_logits(np.array([1, 2, 3, 4, 5]), 4))

        logger.debug("Some visual inspection of derivative of (loss function from logits):")
        logger.debug(Utilities.loss_from_logits_derivative_wrt_logits(np.array([1, 2, 3, 4, 5]), 0))
        logger.debug(Utilities.loss_from_logits_derivative_wrt_logits(np.array([1, 2, 3, 4, 5]), 1))
        logger.debug(Utilities.loss_from_logits_derivative_wrt_logits(np.array([1, 2, 3, 4, 5]), 2))
        logger.debug(Utilities.loss_from_logits_derivative_wrt_logits(np.array([1, 2, 3, 4, 5]), 3))
        logger.debug(Utilities.loss_from_logits_derivative_wrt_logits(np.array([1, 2, 3, 4, 5]), 4))


    def testLossFromLogitsDerivatives(self):
        for expected_y in range(5):
            test = DerivativeVerifier.theoreticalDerivativeEqualsToNumericalDiff(
                lambda logits_0: Utilities.loss_from_logits(logits_0, expected_y),
                lambda logits_0: Utilities.loss_from_logits_derivative_wrt_logits(logits_0, expected_y),
                np.array([1, 2, 3, 4, 5]), print_diff=False)
            self.assertTrue(test)

        test_vector_size = 80
        loop_times = 10
        logger.info("Looping %d times to test derivative of loss of a size %d logits vector..." % (loop_times, test_vector_size))
        for test_count in range(loop_times):
            logits_x_0 = np.random.rand(test_vector_size)
            for expected_y in range(test_vector_size):
                test = DerivativeVerifier.theoreticalDerivativeEqualsToNumericalDiff(
                    lambda logits_0: Utilities.loss_from_logits(logits_x_0, expected_y),
                    lambda logits_0: Utilities.loss_from_logits_derivative_wrt_logits(logits_x_0, expected_y),
                    logits_x_0, error_relative_to_delta_x=1, print_diff=True)
                self.assertTrue(test)


    def testLossFromVandS(self):
        s_0 = np.array([1, 2, 3, 4])
        v_0 = np.array([
            [0.11, .12, .13, .14],
            [0.21, .22, .23, .24],
            [0.31, .32, .33, .34],
            [0.41, .42, .43, .44],
            [0.51, .52, .53, .54],
        ])

        for expected_y in range(5):
            logger.debug('y=%d, loss=%f'
                % (expected_y, Utilities.loss_from_matrix_v_and_hidden_state(s_0, v_0, expected_y)))

        for expected_y in range(5):
            logger.debug('y=%d, theoretical derivative of loss w.r.t. v=%s'
                % (expected_y, Utilities.loss_from_matrix_v_and_hidden_state_derivative_wrt_v(s_0, v_0, expected_y)))

        for test_times in range(100):
            v_0 = np.random.rand(4 * 5).reshape([5, 4])
            s_0 = np.random.rand(4)
            for expected_y in range(5):
                # Derivative w.r.t. V
                test = DerivativeVerifier.theoreticalDerivativeEqualsToNumericalDiff(
                    lambda v: Utilities.loss_from_matrix_v_and_hidden_state(s_0, v, expected_y),
                    lambda v: Utilities.loss_from_matrix_v_and_hidden_state_derivative_wrt_v(s_0, v, expected_y),
                    v_0, error_relative_to_delta_x=1, print_diff=True)
                self.assertTrue(test)

                # Derivative w.r.t. S
                test = DerivativeVerifier.theoreticalDerivativeEqualsToNumericalDiff(
                    lambda s: Utilities.loss_from_matrix_v_and_hidden_state(s, v0, expected_y),
                    lambda s: Utilities.loss_from_matrix_v_and_hidden_state_derivative_wrt_v(s, v_0, expected_y),
                    v_0, error_relative_to_delta_x=1, print_diff=True)
                self.assertTrue(test)


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    unittest.main()