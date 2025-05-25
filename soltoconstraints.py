import cvxpy as cp
import sympy as sp

def soltoconstraints(solns: dict, Q: cp.Variable):
    """
    Convert a SymPy solution dictionary into CVXPY constraints using a predefined CVXPY matrix Q.

    Parameters:
        solns (dict): Keys are SymPy symbols like q01, values are SymPy expressions or numbers.
        Q (cp.Variable): A CVXPY variable matrix (must be 2D square, typically PSD).

    Returns:
        list: A list of CVXPY constraints.
    """
    n, m = Q.shape
    assert n == m, "Matrix Q must be square."

    # Step 1: Build a mapping from symbol names like q12 to Q[1, 2]
    cvx_var_map = {}
    for i in range(n):
        for j in range(n):
            sym = sp.Symbol(f'q{i}{j}')
            cvx_var_map[sym] = Q[i, j]

    # Step 2: Create CVXPY constraints
    constraints = []
    for sym, expr in solns.items():
        lhs = cvx_var_map[sym]
        if isinstance(expr, sp.Basic):
            expr_cvx = expr
            for s, v in cvx_var_map.items():
                expr_cvx = expr_cvx.subs(s, v)
            constraints.append(lhs == expr_cvx)
        else:
            constraints.append(lhs == float(expr))

    return constraints