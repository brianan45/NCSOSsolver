import sympy as sp
from sympy import MatrixSymbol

def is_matrix_factor(factor, matrix_vars):
    # Check if factor is matrix or matrix power of a variable in matrix_vars
    if isinstance(factor, MatrixSymbol):
        return factor in matrix_vars
    if isinstance(factor, sp.MatPow):
        return factor.base in matrix_vars
    return False

def extract_coefficient(expr, matrix_vars):
    # Assume expr is a Mul or Pow or simple term involving matrices
    # We want the scalar coefficient for the matrix part that includes any matrix var

    # For terms like -A**2, expr is Mul(-1, MatPow(A, 2))

    factors = expr.as_ordered_factors()
    scalar = sp.Integer(1)
    matrix_factors = []

    for f in factors:
        if is_matrix_factor(f, matrix_vars):
            matrix_factors.append(f)
        else:
            scalar *= f

    # Compose matrix part from matrix factors
    matrix_part = sp.MatMul(*matrix_factors) if matrix_factors else sp.Integer(1)

    return scalar, matrix_part

# Example:

A = MatrixSymbol('A', 2, 2)
expr = -A**2

coeff, matrix_part = extract_coefficient(expr, [A])

print("Coefficient:", coeff)          # Should print -1
print("Matrix part:", matrix_part)   # Should print A**2
