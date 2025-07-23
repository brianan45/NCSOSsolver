import sympy as sp

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
    return simplify_term(expr)

# import sympy as sp
# from sympy import MatrixSymbol, symbols, Identity

# # Step 1: Declare symbolic matrix size
# n = symbols('n', integer=True, positive=True)

# # Step 2: Declare symbolic matrix variables of size n×n
# A = MatrixSymbol('A', n, n)
# B = MatrixSymbol('B', n, n)

# # Step 3: Declare scalar symbols
# k, m = symbols('k m')

# # Step 4: Build expression with explicit integer exponents
# expr = k * A**5 * B**7 + m * A**3

# # Step 5: Declare identity powers (known orders of matrices)
# # e.g., A**3 = I, B**4 = I
# identity_powers = {
#     A: 3,
#     B: 4
# }

# # Step 6: Simplify using sub_identity_matrix
# simplified = sub_identity_matrix(expr, identity_powers)

# # Step 7: Display result
# print("Original expression:", expr)

# print("\nSimplified expression:", simplified)
