import sympy as sp
import cvxpy as cp
import re

# a function that takes a constraint and replaces any
# sympy variables with the corresponding cvxpy variables

def sp_to_cp_constraint(eq, Q, extra_subs=None):
    """
    Converts a SymPy equation into a CVXPY constraint using:
    - Q[i,j] variables for symbolic qij entries
    - optional extra substitutions for other variables like 'lm'
    """
    if extra_subs is None:
        extra_subs = {}

    all_symbols = eq.free_symbols
    pattern = re.compile(r'q(\d+)(\d+)')  # pattern for symbolic matrix entries
    sym_to_cvx = {}

    for s in all_symbols:
        if s in extra_subs:
            sym_to_cvx[s] = extra_subs[s]
        else:
            match = pattern.match(str(s))
            if match:
                i, j = int(match.group(1)), int(match.group(2))
                if i >= Q.shape[0] or j >= Q.shape[1]:
                    raise IndexError(f"Symbol q{i}{j} exceeds dimensions of Q.")
                sym_to_cvx[s] = Q[i, j]

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
