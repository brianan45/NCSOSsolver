import sympy as sp
from sympy import Adjoint, MatrixSymbol, Identity, MatMul

def is_matrix_factor(factor, matrix_vars):
    """
    Check if a factor is a matrix: matrix itself, transpose, adjoint, power, inverse, or Identity.
    """
    if isinstance(factor, MatrixSymbol):
        return factor in matrix_vars
    if isinstance(factor, sp.MatPow):
        return factor.base in matrix_vars
    if isinstance(factor, sp.Transpose):
        return factor.arg in matrix_vars
    if isinstance(factor, Adjoint):
        return factor.arg in matrix_vars
    if isinstance(factor, Identity):
        return True   # Treat Identity as a matrix factor
    return False

def decompose_term(term, matrix_vars):
    """
    Splits a term into (scalar_part, matrix_part)
    """
    factors = term.as_ordered_factors()
    scalar_part = sp.Integer(1)
    matrix_factors = []
    for factor in factors:
        if is_matrix_factor(factor, matrix_vars):
            matrix_factors.append(factor)
        else:
            scalar_part *= factor

    # If no matrix factors, matrix_part is 1 (scalar)
    if matrix_factors:
        matrix_part = MatMul(*matrix_factors)
    else:
        matrix_part = sp.Integer(1)
    return scalar_part, matrix_part

def get_coeffs(expr, matrix_vars):
    """
    Groups by matrix part and returns equations setting scalar coeffs to zero.
    """
    expr = sp.expand(expr)
    terms = expr.as_ordered_terms()

    coeff_map = {}  # matrix_expr → scalar coefficient sum

    for term in terms:
        scalar, matrix = decompose_term(term, matrix_vars)
        if matrix in coeff_map:
            coeff_map[matrix] += scalar
        else:
            coeff_map[matrix] = scalar

    # Remove trivial term if matrix part is 1 (scalar only term)
    # Usually from constant term, but you can keep if needed
    if sp.Integer(1) in coeff_map:
        # For example, you might want to keep it or remove it
        pass

    return [sp.Eq(coeff, 0) for coeff in coeff_map.values()]