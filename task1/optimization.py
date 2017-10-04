import numpy as np
from numpy.linalg import LinAlgError
from math import sqrt
import scipy
from scipy.optimize.linesearch import scalar_search_wolfe2
from scipy.linalg import cho_factor, cho_solve
from datetime import datetime
from collections import defaultdict
from time import clock


class LineSearchTool(object):
    """
    Line search tool for adaptively tuning the step size of the algorithm.

    method : String containing 'Wolfe', 'Armijo' or 'Constant'
        Method of tuning step-size.
        Must be be one of the following strings:
            - 'Wolfe' -- enforce strong Wolfe conditions;
            - 'Armijo" -- adaptive Armijo rule;
            - 'Constant' -- constant step size.
    kwargs :
        Additional parameters of line_search method:

        If method == 'Wolfe':
            c1, c2 : Constants for strong Wolfe conditions
            alpha_0 : Starting point for the backtracking procedure
                to be used in Armijo method in case of failure of Wolfe method.
        If method == 'Armijo':
            c1 : Constant for Armijo rule
            alpha_0 : Starting point for the backtracking procedure.
        If method == 'Constant':
            c : The step size which is returned on every step.
    """
    def __init__(self, method='Wolfe', **kwargs):
        self._method = method
        if self._method == 'Wolfe':
            self.c1 = kwargs.get('c1', 1e-4)
            self.c2 = kwargs.get('c2', 0.9)
            self.alpha_0 = kwargs.get('alpha_0', 1.0)
        elif self._method == 'Armijo':
            self.c1 = kwargs.get('c1', 1e-4)
            self.alpha_0 = kwargs.get('alpha_0', 1.0)
        elif self._method == 'Constant':
            self.c = kwargs.get('c', 1.0)
        else:
            raise ValueError('Unknown method {}'.format(method))

    @classmethod
    def from_dict(cls, options):
        if type(options) != dict:
            raise TypeError('LineSearchTool initializer must be of type dict')
        return cls(**options)

    def to_dict(self):
        return self.__dict__

    def line_search(self, oracle, x_k, d_k, previous_alpha=None):
        """
        Finds the step size alpha for a given starting point x_k
        and for a given search direction d_k that satisfies necessary
        conditions for phi(alpha) = oracle.func(x_k + alpha * d_k).

        Parameters
        ----------
        oracle : BaseSmoothOracle-descendant object
            Oracle with .func_directional() and .grad_directional() methods implemented for computing
            function values and its directional derivatives.
        x_k : np.array
            Starting point
        d_k : np.array
            Search direction
        previous_alpha : float or None
            Starting point to use instead of self.alpha_0 to keep the progress from
             previous steps. If None, self.alpha_0, is used as a starting point.

        Returns
        -------
        alpha : float or None if failure
            Chosen step size
        """
        # Implement line search procedures for Armijo, Wolfe and Constant steps.
        def phi(alpha):
            return oracle.func_directional(x_k, d_k, alpha)
        
        def derphi(alpha):
            return oracle.grad_directional(x_k, d_k, alpha)
        
        def armijo(alpha_0, phi_0, derphi_0):
            if previous_alpha is None:
                new_alpha = self.alpha_0
            else:
                new_alpha = previous_alpha
            while phi(new_alpha) > phi_0 + self.c1 * new_alpha * derphi_0:
                new_alpha /= 2.
            return new_alpha
            
        phi_0 = phi(0.)
        derphi_0 = derphi(0.)
        
        if self._method == 'Constant':
            new_alpha = self.c
        elif self._method == 'Armijo':
            new_alpha = armijo(self.alpha_0, phi_0, derphi_0)
        else:
            new_alpha, _, _, _ = scalar_search_wolfe2(phi, derphi, phi0=phi_0, 
                                             derphi0=derphi_0, c1=self.c1,
                                             c2=self.c2)
            if new_alpha is None:
                new_alpha = armijo(self.alpha_0, phi_0, derphi_0)
        
        return new_alpha


def get_line_search_tool(line_search_options=None):
    if line_search_options:
        if type(line_search_options) is LineSearchTool:
            return line_search_options
        else:
            return LineSearchTool.from_dict(line_search_options)
    else:
        return LineSearchTool()


def gradient_descent(oracle, x_0, tolerance=1e-5, max_iter=10000,
                     line_search_options=None, trace=False, display=False):
    """
    Gradien descent optimization method.

    Parameters
    ----------
    oracle : BaseSmoothOracle-descendant object
        Oracle with .func(), .grad() and .hess() methods implemented for computing
        function value, its gradient and Hessian respectively.
    x_0 : np.array
        Starting point for optimization algorithm
    tolerance : float
        Epsilon value for stopping criterion.
    max_iter : int
        Maximum number of iterations.
    line_search_options : dict, LineSearchTool or None
        Dictionary with line search options. See LineSearchTool class for details.
    trace : bool
        If True, the progress information is appended into history dictionary during training.
        Otherwise None is returned instead of history.
    display : bool
        If True, debug information is displayed during optimization.
        Printing format and is up to a student and is not checked in any way.

    Returns
    -------
    x_star : np.array
        The point found by the optimization procedure
    message : string
        "success" or the description of error:
            - 'iterations_exceeded': if after max_iter iterations of the method x_k still doesn't satisfy
                the stopping criterion.
            - 'computational_error': in case of getting Infinity or None value during the computations.
    history : dictionary of lists or None
        Dictionary containing the progress information or None if trace=False.
        Dictionary has to be organized as follows:
            - history['time'] : list of floats, containing time in seconds passed from the start of the method
            - history['func'] : list of function values f(x_k) on every step of the algorithm
            - history['grad_norm'] : list of values Euclidian norms ||g(x_k)|| of the gradient on every step of the algorithm
            - history['x'] : list of np.arrays, containing the trajectory of the algorithm. ONLY STORE IF x.size <= 2

    Example:
    --------
    >> oracle = QuadraticOracle(np.eye(5), np.arange(5))
    >> x_opt, message, history = gradient_descent(oracle, np.zeros(5), line_search_options={'method': 'Armijo', 'c1': 1e-4})
    >> print('Found optimal point: {}'.format(x_opt))
       Found optimal point: [ 0.  1.  2.  3.  4.]
    """
    
    history = defaultdict(list) if trace else None
    line_search_tool = get_line_search_tool(line_search_options)
    x_k = np.copy(x_0)
    
    norm_grad_0 = (oracle.grad(x_0) ** 2).sum()
    
    alpha = None
    
    start = clock()
   
    for i in range(max_iter + 1):
        
        # Call oracle
        func_k = oracle.func(x_k)
        if (func_k is None) or np.isnan(func_k) or (np.isinf(func_k)):
            message = 'computational_error'
            return x_k, message, history
        grad_k = oracle.grad(x_k)
        if (grad_k is None) or (np.isnan(grad_k).sum() != 0) or (np.isinf(func_k).sum() != 0):
            message = 'computational_error'
            return x_k, message, history
        norm_grad_k = (grad_k ** 2).sum()
        
        # History of iterations
        if trace:
            history['func'].append(oracle.func(x_k))
            history['grad_norm'].append(sqrt(norm_grad_k))
            history['time'].append(clock() - start)
            if x_k.size <= 2:
                history['x'].append(x_k)
        
        # Debug information
        if display:
            print("iteration: ", i, "||grad f(x_k)||=", norm_grad_k)
        
        # Stopping criterion
        if norm_grad_k <= tolerance * norm_grad_0:
            message = 'success'
            break
        
        # Compute direction
        d_k = -grad_k
        
        # Line search alpha
        if alpha is None:
            alpha = line_search_tool.line_search(oracle, x_k, d_k, previous_alpha=None)
        else:
            alpha = line_search_tool.line_search(oracle, x_k, d_k, previous_alpha=alpha)
        
        # Update x_k
        x_k = x_k + alpha * d_k
        
    if norm_grad_k > tolerance * norm_grad_0:
        message = 'iterations_exceeded'
    return x_k, message, history


def newton(oracle, x_0, tolerance=1e-5, max_iter=100,
           line_search_options=None, trace=False, display=False):
    """
    Newton's optimization method.

    Parameters
    ----------
    oracle : BaseSmoothOracle-descendant object
        Oracle with .func(), .grad() and .hess() methods implemented for computing
        function value, its gradient and Hessian respectively. If the Hessian
        returned by the oracle is not positive-definite method stops with message="newton_direction_error"
    x_0 : np.array
        Starting point for optimization algorithm
    tolerance : float
        Epsilon value for stopping criterion.
    max_iter : int
        Maximum number of iterations.
    line_search_options : dict, LineSearchTool or None
        Dictionary with line search options. See LineSearchTool class for details.
    trace : bool
        If True, the progress information is appended into history dictionary during training.
        Otherwise None is returned instead of history.
    display : bool
        If True, debug information is displayed during optimization.

    Returns
    -------
    x_star : np.array
        The point found by the optimization procedure
    message : string
        'success' or the description of error:
            - 'iterations_exceeded': if after max_iter iterations of the method x_k still doesn't satisfy
                the stopping criterion.
            - 'newton_direction_error': in case of failure of solving linear system with Hessian matrix (e.g. non-invertible matrix).
            - 'computational_error': in case of getting Infinity or None value during the computations.
    history : dictionary of lists or None
        Dictionary containing the progress information or None if trace=False.
        Dictionary has to be organized as follows:
            - history['time'] : list of floats, containing time passed from the start of the method
            - history['func'] : list of function values f(x_k) on every step of the algorithm
            - history['grad_norm'] : list of values Euclidian norms ||g(x_k)|| of the gradient on every step of the algorithm
            - history['x'] : list of np.arrays, containing the trajectory of the algorithm. ONLY STORE IF x.size <= 2

    Example:
    --------
    >> oracle = QuadraticOracle(np.eye(5), np.arange(5))
    >> x_opt, message, history = newton(oracle, np.zeros(5), line_search_options={'method': 'Constant', 'c': 1.0})
    >> print('Found optimal point: {}'.format(x_opt))
       Found optimal point: [ 0.  1.  2.  3.  4.]
    """
    history = defaultdict(list) if trace else None
    line_search_tool = get_line_search_tool(line_search_options)
    x_k = np.copy(x_0)
    
    norm_grad_0 = (oracle.grad(x_0) ** 2).sum()
    
    alpha = None

    start = clock()
   
    for i in range(max_iter + 1):
        
        # Call oracle and check func, frad, hess
        func_k = oracle.func(x_k)
        if (func_k is None) or np.isnan(func_k) or (np.isinf(func_k)):
            message = 'computational_error'
            return x_k, message, history
        grad_k = oracle.grad(x_k)
        if (grad_k is None) or (np.isnan(grad_k).sum() != 0) or (np.isinf(func_k).sum() != 0):
            message = 'computational_error'
            return x_k, message, history
        hess_k = oracle.hess(x_k)
        if (hess_k is None) or (np.isnan(hess_k).sum() != 0) or (np.isinf(hess_k).sum() != 0):
            message = 'computational_error'
            return x_k, message, history
        norm_grad_k = (grad_k ** 2).sum()
        
        # History of iterations
        if trace:
            history['func'].append(oracle.func(x_k))
            history['grad_norm'].append(sqrt(norm_grad_k))
            history['time'].append(clock() - start)
            if x_k.size <= 2:
                history['x'].append(x_k)
        
        # Debug information
        if display:
            print("iteration: ", i, "||grad f(x_k)||=", norm_grad_k)
        
        # Stopping criterion
        if norm_grad_k <= tolerance * norm_grad_0:
            message = 'success'
            break
        
        # Compute direction
        try:
            L_cholesky, lower = cho_factor(hess_k)
            d_k = cho_solve((L_cholesky, lower), -grad_k)
        except(LinAlgError):
            message = 'newton_direction_error'
            return x_k, message, history
        
        # Line search alpha
        alpha = line_search_tool.line_search(oracle, x_k, d_k, previous_alpha=1.0)
        
        # Update x_k
        x_k = x_k + alpha * d_k
        
    if norm_grad_k > tolerance * norm_grad_0:
        message = 'iterations_exceeded'
    
    return x_k, message, history
