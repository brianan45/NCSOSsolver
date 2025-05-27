import sympy as sp
import cvxpy as cp
import re

def sp_to_cp_constraint(eq, Q):
    """
    Converts a SymPy equation into a CVXPY equation using Q[i,j] variables.
    """
    all_symbols = eq.free_symbols # get the varables used in the equation
    pattern = re.compile(r'q(\d+)(\d+)') # pattern for checking for symbolic variables
    sym_to_cvx = {}

    for s in all_symbols:
        match = pattern.match(str(s)) # boolean for whether variable matches pattern
        if match:
            i, j = int(match.group(1)), int(match.group(2)) # get the indices of the matrix
            if i >= Q.shape[0] or j >= Q.shape[1]:
                raise IndexError(f"Symbol q{i}{j} exceeds dimensions of Q.")
            sym_to_cvx[s] = Q[i, j]  # 0-based indexing

    def sympy_to_cvxpy(expr):
        if expr.is_Number:
            return float(expr)
        elif expr.is_Symbol:
            return sym_to_cvx.get(expr, expr)
        elif expr.is_Add:
            return sum(sympy_to_cvxpy(arg) for arg in expr.args)
        elif expr.is_Mul:
            result = 1
            for arg in expr.args:
                result *= sympy_to_cvxpy(arg)
            return result
        elif expr.is_Pow:
            base, exp = expr.args
            return sympy_to_cvxpy(base) ** sympy_to_cvxpy(exp)
        else:
            raise NotImplementedError(f"Cannot convert expression: {expr}")

    lhs = sympy_to_cvxpy(eq.lhs)
    rhs = sympy_to_cvxpy(eq.rhs)
    return lhs == rhs