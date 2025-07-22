import sympy as sp
import cvxpy as cp
import re

def sp_to_cp_constraint_multiQ(eq, Q_list, extra_subs=None):
    """
    Converts a SymPy equation into a CVXPY constraint using:
    - Qn_ij symbols corresponding to Q_list[n][i,j]
    - extra_subs for substitutions like lm_cp, etc.

    Args:
        eq (sympy.Eq): Equation with symbolic q-entries.
        Q_list (list of cvxpy.Variable): List of PSD matrices.
        extra_subs (dict): Optional mapping from sympy symbols to cvxpy variables.

    Returns:
        cvxpy constraint (e.g., lhs == rhs)
    """
    if extra_subs is None:
        extra_subs = {}

    pattern = re.compile(r'[qQ](\d+)_(\d+)_(\d+)')

    sym_to_cvx = {}

    for s in eq.free_symbols:
        if s in extra_subs:
            sym_to_cvx[s] = extra_subs[s]
        else:
            match = pattern.match(str(s))
            if match:
                q_idx, i, j = int(match.group(1)), int(match.group(2)), int(match.group(3))
                if q_idx >= len(Q_list):
                    raise IndexError(f"Q index {q_idx} out of range.")
                Q = Q_list[q_idx]
                if i >= Q.shape[0] or j >= Q.shape[1]:
                    raise IndexError(f"Index q{q_idx}_{i}_{j} exceeds dimensions of Q{q_idx}.")
                sym_to_cvx[s] = Q[i, j]
            else:
                raise ValueError(f"Unrecognized symbol format: {s}")

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
