'''
Adam Optimizer
'''

import logging
import numpy as np

import unittest

from derivative_verifier import DerivativeVerifier

logger = logging.getLogger(__name__)

class AdamOptimizer:
    # Default parameters copied from the answer from GPT4.
    def __init__(self, alpha=0.1, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.alpha = alpha
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.prev_m = 0
        self.prev_v = 0
        self.iteration = 0


    # Used formula from https://www.analyticsvidhya.com/blog/2021/10/a-comprehensive-guide-on-deep-learning-optimizers/
    def suggest_delta_x_from_graident(self, gradient):
        self.iteration += 1


        m = self.beta1 * self.prev_m + ((1 - self.beta1) * gradient)  # update first moment estimate
        v = self.beta2 * self.prev_v + (1 - self.beta2) * (gradient ** 2)  # update second moment estimate

        m_hat = m / (1 - self.beta1 ** self.iteration)  # correct bias in first moment estimate
        v_hat = v / (1 - self.beta2 ** self.iteration)  # correct bias in second moment estimate
        delta_x = - self.alpha * m_hat / (v_hat ** .5 + self.epsilon)  # update the parameter

        self.prev_m = m
        self.prev_v = v

        return delta_x


class AdamOptizerTest(unittest.TestCase):
    def test_matrix_square(self):
        optimizer = AdamOptimizer(alpha=0.01)

        def _f(matrix):
            return np.sum((matrix - 3) ** 3) ** 2

        x0 = np.array([
            [-10.0, -2],
            [4, 20],
        ])

        for step in range(1000):
            gradient = DerivativeVerifier.numerical_jacobian_diff_matrix(
                _f, x0, 1e-5).astype(np.float64)

            delta_x = optimizer.suggest_delta_x_from_graident(gradient)
            x0 += delta_x

            logger.debug("Step %d, gradient=\n%s\n, adam suggested delta_x=\n%s\n, new x0=\n%s\n" +
                " f(x0)=\n%s\n",
                step, gradient, delta_x, x0, _f(x0))

            if np.isclose(_f(x0), 0, 1e-5, 1e-5):
                logger.debug("Found solution of the equation at step %d.", step)
                break

        np.testing.assert_almost_equal(_f(x0), 0, 5)


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    unittest.main()