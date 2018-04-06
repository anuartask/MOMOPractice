import numpy as np
import scipy
from scipy.special import expit
from scipy.sparse import csr_matrix, diags


class BaseSmoothOracle(object):
    """
    Base class for implementation of oracles.
    """

    def func(self, x):
        """
        Computes the value of function at point x.
        """
        raise NotImplementedError('Func oracle is not implemented.')

    def grad(self, x):
        """
        Computes the gradient at point x.
        """
        raise NotImplementedError('Grad oracle is not implemented.')

    def hess(self, x):
        """
        Computes the Hessian matrix at point x.
        """
        raise NotImplementedError('Hessian oracle is not implemented.')

    def func_directional(self, x, d, alpha):
        """
        Computes phi(alpha) = f(x + alpha*d).
        """
        return np.squeeze(self.func(x + alpha * d))

    def grad_directional(self, x, d, alpha):
        """
        Computes phi'(alpha) = (f(x + alpha*d))'_{alpha}
        """
        return np.squeeze(self.grad(x + alpha * d).dot(d))

    def hess_vec(self, x, v):
        """
        Computes matrix-vector product with Hessian matrix f''(x) v
        """
        return self.hess(x).dot(v)


class QuadraticOracle(BaseSmoothOracle):
    """
    Oracle for quadratic function:
       func(x) = 1/2 x^TAx - b^Tx.
    """

    def __init__(self, A, b):
        if not scipy.sparse.isspmatrix_dia(A) and not np.allclose(A, A.T):
            raise ValueError('A should be a symmetric matrix.')
        self.A = A
        self.b = b

    def func(self, x):
        return 0.5 * np.dot(self.A.dot(x), x) - self.b.dot(x)

    def grad(self, x):
        return self.A.dot(x) - self.b

    def hess(self, x):
        return self.A

    def minimize_directional(self, x, d):
        """
        Minimizes the function with respect to a specific direction:
            Finds alpha = argmin f(x + alpha d)
        """
        return ((self.b - self.A.dot(x)) * d).sum() / (self.A.dot(d) * d).sum() 
        

class LogRegL2Oracle(BaseSmoothOracle):
    """
    Oracle for logistic regression with l2 regularization:
         func(x) = 1/m sum_i log(1 + exp(-b_i * a_i^T x)) + regcoef / 2 ||x||_2^2.

    Let A and b be parameters of the logistic regression (feature matrix
    and labels vector respectively).
    For user-friendly interface use create_log_reg_oracle()

    Parameters
    ----------
        matvec_Ax : function
            Computes matrix-vector product Ax, where x is a vector of size n.
        matvec_ATy : function of y
            Computes matrix-vector product A^Ty, where y is a vector of size m.
        matmat_ATsA : function
            Computes matrix-matrix-matrix product A^T * Diag(s) * A,
    """

    def __init__(self, matvec_Ax, matvec_ATx, matmat_ATsA, b, regcoef):
        self.matvec_Ax = matvec_Ax
        self.matvec_ATx = matvec_ATx
        self.matmat_ATsA = matmat_ATsA
        self.b = b
        self.regcoef = regcoef

    def func(self, x):
        Ax = self.matvec_Ax(x)
        m = Ax.shape[0]
        obj_loss = 1./m * np.logaddexp(np.zeros(m), -self.b * Ax).sum()
        reg_value = 0.5 * self.regcoef * (x ** 2).sum()
        return obj_loss + reg_value

    def grad(self, x):
        Ax = self.matvec_Ax(x)
        m = Ax.shape[0]
        ATx = self.matvec_ATx(self.b * expit(-self.b * Ax))
        grad_value = -1./m * ATx + self.regcoef * x
        return grad_value

    def hess(self, x):
        Ax = self.matvec_Ax(x)
        m = Ax.shape[0]
        n = x.shape[0]
        sigmoid = expit(-self.b * Ax)
        s = sigmoid * (1 - sigmoid)
        ATsA = self.matmat_ATsA(s)
        return 1./m * ATsA + self.regcoef * np.eye(n)
    
    def hess_vec(self, x, v):
        Ax = self.matvec_Ax(x)
        m = Ax.shape[0]
        n = x.shape[0]
        sigmoid = expit(-self.b * Ax)
        s = sigmoid * (1 - sigmoid)
        Av = self.matvec_Ax(v)
        ATsAv = self.matvec_ATx(s * Av)
        return 1./m * ATsAv + self.regcoef * v

class LogRegL2OptimizedOracle(LogRegL2Oracle):
    """
    Oracle for logistic regression with l2 regularization
    with optimized *_directional methods (are used in line_search).

    For explanation see LogRegL2Oracle.
    """

    def __init__(self, matvec_Ax, matvec_ATx, matmat_ATsA, b, regcoef):
        super().__init__(matvec_Ax, matvec_ATx, matmat_ATsA, b, regcoef)
        self.last_x, self.last_x_trial = None, None
        self.last_y = None
        self.last_s = None
        self.last_d = None
        self.Ax, self.ATx, self.ATsA = None, None, None
        self.Ad = None
    
    def _updateAx(self, x):
        if np.array_equal(x, self.last_x):
            return
        if np.array_equal(x, self.last_x_trial):
            self.last_x = self.last_x_trial
            self.Ax = self.Ax_trial
            return
        self.last_x = np.copy(x)
        self.Ax = self.matvec_Ax(x)
        return
    
    def _updateAd(self, d):
        if np.array_equal(d, self.last_d):
            return
        self.last_d = np.copy(d)
        self.Ad = self.matvec_Ax(d)
        return
    
    def _updateATx(self, y):
        if np.array_equal(y, self.last_y):
            return
        self.last_y = np.copy(y)
        self.ATx = self.matvec_ATx(y)
        return
    
    def _updateATsA(self, s):
        if np.array_equal(s, self.last_s):
            return
        self.last_s = np.copy(s)
        self.ATsA = self.matmat_ATsA(s)
        return
        
    def func(self, x):
        self._updateAx(x)
        m = self.Ax.shape[0]
        obj_loss = 1./m * np.logaddexp(np.zeros(m), -self.b * self.Ax).sum()
        reg_value = 0.5 * self.regcoef * (x ** 2).sum()
        return obj_loss + reg_value
    
    def grad(self, x):
        self._updateAx(x)
        m = self.Ax.shape[0]
        n = x.shape[0]
        y = self.b * expit(-self.b * self.Ax)
        self._updateATx(y)
        grad_value = -1./m * self.ATx + self.regcoef * x
        return grad_value
    
    def hess(self, x):
        self._updateAx(x)
        m = self.Ax.shape[0]
        n = x.shape[0]
        sigmoid = expit(-self.b * self.Ax)
        s = sigmoid * (1 - sigmoid)
        self._updateATsA(s)
        return 1./m * self.ATsA + self.regcoef * np.eye(n)
    
    def hess_vec(self, x, v):
        self._updateAx(x)
        n = x.shape[0]
        m = self.Ax.shape[0]
        sigmoid = expit(-self.b * self.Ax)
        s = sigmoid * (1 - sigmoid)
        Av = self.matvec_Ax(v)
        ATsAv = self.matvec_ATx(s * Av)
        return 1./m * ATsAv + self.regcoef * v
        
        
    def func_directional(self, x, d, alpha):
        self._updateAx(x)
        self._updateAd(d)
        self.last_x_trial = x + alpha * d
        self.Ax_trial = self.Ax + alpha * self.Ad
        m = self.Ax_trial.shape[0]
        obj_loss = 1./m * np.logaddexp(np.zeros(m), -self.b * self.Ax_trial).sum()
        reg_value = 0.5 * self.regcoef * (self.last_x_trial ** 2).sum()
        return obj_loss + reg_value

    def grad_directional(self, x, d, alpha):
        self._updateAx(x)
        self._updateAd(d)
        self.last_x_trial = x + alpha * d
        self.Ax_trial = self.Ax + alpha * self.Ad
        m = self.Ax_trial.shape[0]
        y = self.b * expit(-self.b * self.Ax_trial)
        res = -1./m * (y * self.Ad).sum()
        res += self.regcoef * (self.last_x_trial * d).sum()
        return res


def create_log_reg_oracle(A, b, regcoef, oracle_type='usual'):
    """
    Auxiliary function for creating logistic regression oracles.
        `oracle_type` must be either 'usual' or 'optimized'
    """
    matvec_Ax = lambda x: A.dot(x)
    matvec_ATx = lambda x: A.T.dot(x)

    def matmat_ATsA(s):
        if isinstance(A, csr_matrix):
            diag_s = csr_matrix(diags(s))
            return csr_matrix(A.T).dot(diag_s).dot(csr_matrix(A))
        else:
            diag_s = np.diag(s)
            return A.T.dot(diag_s).dot(A)

    if oracle_type == 'usual':
        oracle = LogRegL2Oracle
    elif oracle_type == 'optimized':
        oracle = LogRegL2OptimizedOracle
    else:
        raise 'Unknown oracle_type=%s' % oracle_type
    return oracle(matvec_Ax, matvec_ATx, matmat_ATsA, b, regcoef)


def hess_vec_finite_diff(func, x, v, eps=1e-5):
    """
    Returns approximation of the matrix product 'Hessian times vector'
    using finite differences.
    """
    n = x.shape[0]
    res = np.zeros(n)
    for i in range(n):
        e_i = np.zeros(n)
        e_i[i] = 1.
        res[i] = (func(x + eps * v + eps * e_i) -
                 func(x + eps * v) - 
                 func(x + eps * e_i) +
                 func(x)) / eps ** 2
    return res