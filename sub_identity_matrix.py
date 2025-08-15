import sympy as sp
from sympy import Add, Mul, Identity, MatMul, Symbol
from sympy.matrices.expressions import MatPow, MatrixSymbol

def get_matrix_exps(expr_list):
    """
    Takes a list of MatrixSymbols or power expressions like A**2,
    and returns a dictionary mapping the base MatrixSymbol to its exponent.
    """
    result = {}
    
    for expr in expr_list:
        if isinstance(expr, sp.MatrixSymbol):
            result[expr] = result.get(expr, 0) + 1
        elif isinstance(expr, sp.MatPow) and isinstance(expr.base, sp.MatrixSymbol):
            result[expr.base] = result.get(expr.base, 0) + expr.exp
        else:
            raise ValueError("Each input must be a MatrixSymbol or a power of one (e.g., A or A**2)")

    return result

# a function that takes a matrix expression and substitutes the identity matrix wherever possible
def sub_identity_matrix(expr, identity_powers):
    def simplify_power(base, exp):
        if base in identity_powers:
            n = identity_powers[base]
            reduced_exp = exp % n
            if reduced_exp == 0:
                return Identity(base.shape[0])
            return base ** reduced_exp
        return base ** exp

    def simplify_term(term):
    # Separate scalar and matrix parts
        if isinstance(term, Mul):
            scalar = sp.S.One
            matrix_factors = []
            for factor in term.args:
                if factor.is_commutative:  # scalar or symbolic
                    scalar *= factor
                elif isinstance(factor, MatPow):
                    matrix_factors.append(simplify_power(factor.base, factor.exp))
                elif isinstance(factor, MatrixSymbol):
                    matrix_factors.append(simplify_power(factor, 1))  # Treat as A**1
                else:
                    matrix_factors.append(factor)

            # Remove identity matrices from matrix part
            matrix_factors = [f for f in matrix_factors if not isinstance(f, Identity)]

            if not matrix_factors:
                return scalar * Identity(1)  # only scalar remains
            return scalar * MatMul(*matrix_factors)

        elif isinstance(term, MatPow):
            return simplify_power(term.base, term.exp)

        elif isinstance(term, MatrixSymbol):
            return simplify_power(term, 1)  # Handle standalone A, B, C

        elif term.is_commutative:
            return term

        else:
            return term


    # Handle sums
    if isinstance(expr, Add):
        simplified_terms = [simplify_term(arg) for arg in expr.args]
        return sp.Add(*simplified_terms)

    # Handle single product or term
    return simplify_term(expr)