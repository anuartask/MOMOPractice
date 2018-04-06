import numpy as np
from collections import defaultdict, deque  # Use this for effective implementation of L-BFGS
from utils import get_line_search_tool
from copy import deepcopy
from math import sqrt
from time import clock
from scipy.linalg import norm
from scipy.sparse import csr_matrix

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
    
    norm_grad_0 = norm(oracle.grad(x_0)) ** 2
    
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
        norm_grad_k = norm(grad_k) ** 2
        if norm_grad_k > 1e100:
            message = 'computational_error'
            return x_k, message, history
        
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

def conjugate_gradients(matvec, b, x_0, tolerance=1e-4, max_iter=None, trace=False, display=False):
    """
    Solves system Ax=b using Conjugate Gradients method.

    Parameters
    ----------
    matvec : function
        Implement matrix-vector product of matrix A and arbitrary vector x
    b : 1-dimensional np.array
        Vector b for the system.
    x_0 : 1-dimensional np.array
        Starting point of the algorithm
    tolerance : float
        Epsilon value for stopping criterion.
        Stop optimization procedure and return x_k when:
         ||Ax_k - b||_2 <= tolerance * ||b||_2
    max_iter : int, or None
        Maximum number of iterations. if max_iter=None, set max_iter to n, where n is
        the dimension of the space
    trace : bool
        If True, the progress information is appended into history dictionary during training.
        Otherwise None is returned instead of history.
    display:  bool
        If True, debug information is displayed during optimization.
        Printing format is up to a student and is not checked in any way.

    Returns
    -------
    x_star : np.array
        The point found by the optimization procedure
    message : string
        'success' or the description of error:
            - 'iterations_exceeded': if after max_iter iterations of the method x_k still doesn't satisfy
                the stopping criterion.
    history : dictionary of lists or None
        Dictionary containing the progress information or None if trace=False.
        Dictionary has to be organized as follows:
            - history['time'] : list of floats, containing time in seconds passed from the start of the method
            - history['residual_norm'] : list of values Euclidian norms ||g(x_k)|| of the gradient on every step of the algorithm
            - history['x'] : list of np.arrays, containing the trajectory of the algorithm. ONLY STORE IF x.size <= 2
    """
    history = defaultdict(list) if trace else None
    x_k = np.copy(x_0)
    n = x_k.shape[0]
    if max_iter is None:
        max_iter = n
    if trace:
        if x_k.size <= 2:
            history['x'].append(x_k)
    g_k = np.asarray(matvec(x_k) - b)
    if len(g_k.shape) == 2:
        g_k = g_k[0]
    d_k = -g_k
    norm_g_k = np.linalg.norm(g_k)
    norm_b = np.linalg.norm(b)
    start = clock()
    if trace:
        history['residual_norm'].append(norm_g_k)
        history['time'].append(clock() - start)
    
    for i in range(max_iter):
        Ad_k = np.asarray(matvec(d_k))
        if len(Ad_k.shape) == 2:
            Ad_k = Ad_k[0]
        #compute new x_k
        coef_d_k = norm_g_k ** 2 / (Ad_k * d_k).sum()
        x_k = x_k + coef_d_k * d_k
        
        #compute new direction
        new_g_k = g_k + coef_d_k * Ad_k
        new_norm_g_k = np.linalg.norm(new_g_k)
        d_k = -new_g_k + new_norm_g_k ** 2 / norm_g_k ** 2 * d_k
        g_k = new_g_k.copy()
        norm_g_k = new_norm_g_k.copy()
        
        # Debug information
        if display:
            print("iteration: ", i, "||grad g(x_k)||=", np.linalg.norm(g_k))
        
        if trace:
            history['residual_norm'].append(norm_g_k)
            history['time'].append(clock() - start)
            if x_k.size <= 2:
                history['x'].append(x_k)
        
        # Stopping criterion
        if norm_g_k <= tolerance * norm_b:
            return x_k, 'success', history
                
    if norm_g_k > tolerance * norm_b:
        return x_k, 'iterations_exceeded', history
    else:
        return x_k, 'success', history


def lbfgs(oracle, x_0, tolerance=1e-4, max_iter=500, memory_size=10,
          line_search_options=None, display=False, trace=False):
    """
    Limited-memory Broyden–Fletcher–Goldfarb–Shanno's method for optimization.

    Parameters
    ----------
    oracle : BaseSmoothOracle-descendant object
        Oracle with .func() and .grad() methods implemented for computing
        function value and its gradient respectively.
    x_0 : 1-dimensional np.array
        Starting point of the algorithm
    tolerance : float
        Epsilon value for stopping criterion.
    max_iter : int
        Maximum number of iterations.
    memory_size : int
        The length of directions history in L-BFGS method.
    line_search_options : dict, LineSearchTool or None
        Dictionary with line search options. See LineSearchTool class for details.
    display : bool
        If True, debug information is displayed during optimization.
        Printing format is up to a student and is not checked in any way.
    trace:  bool
        If True, the progress information is appended into history dictionary during training.
        Otherwise None is returned instead of history.

    Returns
    -------
    x_star : np.array
        The point found by the optimization procedure
    message : string
        'success' or the description of error:
            - 'iterations_exceeded': if after max_iter iterations of the method x_k still doesn't satisfy
              the stopping criterion.
    history : dictionary of lists or None
        Dictionary containing the progress information or None if trace=False.
        Dictionary has to be organized as follows:
            - history['func'] : list of function values f(x_k) on every step of the algorithm
            - history['time'] : list of floats, containing time in seconds passed from the start of the method
            - history['grad_norm'] : list of values Euclidian norms ||g(x_k)|| of the gradient on every step of the algorithm
            - history['x'] : list of np.arrays, containing the trajectory of the algorithm. ONLY STORE IF x.size <= 2
    """
    def bfgs_multiply(v, H_history, gamma):
        if len(H_history) == 0:
            return gamma * v
        s, y = H_history.pop()
        v_new = v - (s * v).sum() / (y * s).sum() * y
        z = bfgs_multiply(v_new, H_history, gamma)
        return z + ((s * v).sum() - (y * z).sum()) / (y * s).sum() * s
    
    def lbfgs_direction(grad_k, H_history):
        if len(H_history) == 0:
            gamma_0 = 1.0
            return bfgs_multiply(-grad_k, H_history, gamma_0)
        else:
            s, y = H_history[-1]
            gamma_0 = (y * s).sum() / (y * y).sum()
            return bfgs_multiply(-grad_k, deepcopy(H_history), gamma_0)
    
    history = defaultdict(list) if trace else None
    line_search_tool = get_line_search_tool(line_search_options)
    x_k = np.copy(x_0)
    norm_grad_0 = np.linalg.norm(oracle.grad(x_0))
    alpha = None
    H_history = deque()
    grad_x_k = oracle.grad(x_k)
    
    start = clock()
    
    for i in range(max_iter):
        
        norm_grad_k = np.linalg.norm(grad_x_k)
        
        # Debug information
        if display:
            print("iteration: ", i, "||grad x_k||=", norm_grad_k)
        
        #trace
        if trace:
            history['func'].append(oracle.func(x_k))
            history['grad_norm'].append(norm_grad_k)
            history['time'].append(clock() - start)
            if x_k.size <= 2:
                history['x'].append(x_k)
                
        #Stopping criterion
        if norm_grad_k ** 2 <= tolerance * norm_grad_0 ** 2:
            return x_k, 'success', history
        
        #Direction
        d_k = lbfgs_direction(grad_x_k, H_history)
        
        #Line search
        alpha = line_search_tool.line_search(oracle, x_k, d_k, previous_alpha=alpha)
        
        #Update x_k
        x_new = x_k + alpha * d_k
        s = alpha *  d_k
        grad_x_new = oracle.grad(x_new)
        y = grad_x_new - grad_x_k
        grad_x_k = grad_x_new.copy()
        x_k = x_new.copy()
        
        #Add in history and control size of history
        H_history.append((s, y))
        if i >= memory_size:
            H_history.remove(H_history[0])
    
    if norm_grad_k ** 2 > tolerance * norm_grad_0 ** 2:
        return x_k, 'iterations_exceeded', history
    else:
        return x_k, 'success', history

def hessian_free_newton(oracle, x_0, tolerance=1e-4, max_iter=500, 
                        line_search_options=None, display=False, trace=False):
    """
    Hessian Free method for optimization.

    Parameters
    ----------
    oracle : BaseSmoothOracle-descendant object
        Oracle with .func(), .grad() and .hess_vec() methods implemented for computing
        function value, its gradient and matrix product of the Hessian times vector respectively.
    x_0 : 1-dimensional np.array
        Starting point of the algorithm
    tolerance : float
        Epsilon value for stopping criterion.
    max_iter : int
        Maximum number of iterations.
    line_search_options : dict, LineSearchTool or None
        Dictionary with line search options. See LineSearchTool class for details.
    display : bool
        If True, debug information is displayed during optimization.
        Printing format is up to a student and is not checked in any way.
    trace:  bool
        If True, the progress information is appended into history dictionary during training.
        Otherwise None is returned instead of history.

    Returns
    -------
    x_star : np.array
        The point found by the optimization procedure
    message : string
        'success' or the description of error:
            - 'iterations_exceeded': if after max_iter iterations of the method x_k still doesn't satisfy
              the stopping criterion.
    history : dictionary of lists or None
        Dictionary containing the progress information or None if trace=False.
        Dictionary has to be organized as follows:
            - history['func'] : list of function values f(x_k) on every step of the algorithm
            - history['time'] : list of floats, containing time in seconds passed from the start of the method
            - history['grad_norm'] : list of values Euclidian norms ||g(x_k)|| of the gradient on every step of the algorithm
            - history['x'] : list of np.arrays, containing the trajectory of the algorithm. ONLY STORE IF x.size <= 2
    """
    history = defaultdict(list) if trace else None
    line_search_tool = get_line_search_tool(line_search_options)
    x_k = np.copy(x_0)
    grad_k = oracle.grad(x_k)
    norm_grad_k = norm_grad_0 = np.linalg.norm(oracle.grad(x_0))
    d_k = -grad_k
    
    start = clock()
    
    if trace:
        history['func'].append(oracle.func(x_k))
        history['grad_norm'].append(norm_grad_k)
        history['time'].append(clock() - start)
        if x_k.size <= 2:
            history['x'].append(x_k)
    
    for i in range(max_iter):
        # Debug information
        if display:
            print("iteration: ", i, "||grad x_k||=", norm_grad_k)
        
        eta_k = min(0.5, sqrt(norm_grad_k))
        
        #Conjugate Gradients
        while True:
            d_k, msg, _ = conjugate_gradients(lambda d: oracle.hess_vec(x_k, d), -grad_k, d_k, 
                                              tolerance=eta_k, trace=False)
            # Check
            check = (grad_k * d_k).sum() < 0
            if check:
                break
            eta_k /= 10
            
        # Line search alpha
        alpha = line_search_tool.line_search(oracle, x_k, d_k, previous_alpha=1.0)
        
        # Update x_k
        x_k = x_k + alpha * d_k
        
        #Call oracle
        grad_k = oracle.grad(x_k)
        norm_grad_k = np.linalg.norm(grad_k)
        
        #trace
        if trace:
            history['func'].append(oracle.func(x_k))
            history['grad_norm'].append(norm_grad_k)
            history['time'].append(clock() - start)
            if x_k.size <= 2:
                history['x'].append(x_k)
        
        #Stopping criterion
        if norm_grad_k ** 2 <= tolerance * norm_grad_0 ** 2:
            return x_k, 'success', history
    
    if norm_grad_k ** 2 > tolerance * norm_grad_0 ** 2:
        return x_k, 'iterations_exceeded', history
    else:
        return x_k, 'success', history