import sympy as sp

def get_matrix_exps(var_powers, base_vars):
    """
    Works for scalar or matrix symbols raised to powers.
    Returns a dict mapping base_vars to their exponents.
    """
    result = {}
    for base in base_vars:
        # Search for matching power expression
        exp = 0
        for term in var_powers:
            if isinstance(term, sp.Pow) and term.base == base:
                exp += term.exp
            elif term == base:
                exp += 1
        result[base] = exp
    return result

import sympy as sp
from sympy import Add, Mul, Identity, MatMul, Symbol
from sympy.matrices.expressions import MatPow, MatrixSymbol

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
                else:
                    matrix_factors.append(factor)

            # Remove identity matrices from matrix part
            matrix_factors = [f for f in matrix_factors if not isinstance(f, Identity)]

            if not matrix_factors:
                return scalar * Identity(1)  # only scalar remains
            return scalar * MatMul(*matrix_factors)

        elif isinstance(term, MatPow):
            return simplify_power(term.base, term.exp)

        elif term.is_commutative:
            return term

        else:
            return term

    # Handle sums
    if isinstance(expr, Add):
        simplified_terms = [simplify_term(arg) for arg in expr.args]
        return sp.Add(*simplified_terms)

    # Handle single product or term
#     return simplify_term(expr)
# A, B, C, D = [MatrixSymbol(name, 3, 3) for name in 'ABCD']
# k, m, n = sp.symbols('k m n')

# expr = k * A**4 * B**2 + m*n * C**5 * D**3
# identity_powers = {A: 3, B: 2, C: 4, D: 3}

# simplified = sub_identity_matrix(expr, identity_powers)
# print(simplified)
