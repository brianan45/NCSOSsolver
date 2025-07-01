import sympy as sp


def is_matrix_factor(factor, matrix_vars):
    """
    Check if a factor is a matrix: matrix itself, transpose, adjoint, power, inverse, or Identity.
    """
    if isinstance(factor, sp.MatrixSymbol):
        return factor in matrix_vars
    if isinstance(factor, sp.MatPow):
        return factor.base in matrix_vars
    if isinstance(factor, sp.Transpose):
        return factor.arg in matrix_vars
    # if HAS_ADJOINT and isinstance(factor, Adjoint):
    #     return factor.arg in matrix_vars
    if isinstance(factor, sp.Identity):
        return True   # ✅ Treat Identity as a matrix factor
    return False


def get_matrix_coeff_equations(expr, matrix_vars):
    """
    For an expression, extract scalar coefficients (i.e., terms not containing matrix_vars or their transforms),
    and return equations setting each to zero.

    Args:
        expr (sympy.Expr): the expression
        matrix_vars (list): list of matrix variables (MatrixSymbol)

    Returns:
        list of sympy.Eq objects
    """
    expr = sp.expand(expr)
    terms = expr.as_ordered_terms()

    equations = []

    for term in terms:
        factors = term.as_ordered_factors()

        scalar_part = sp.Integer(1)

        for factor in factors:
            if is_matrix_factor(factor, matrix_vars):
                continue
            else:
                scalar_part *= factor

        equations.append(sp.Eq(scalar_part, 0))

    return equations

A = sp.MatrixSymbol('A', 2, 2)
B = sp.MatrixSymbol('B', 2, 2)
q1, q2, q3, q4, q5, q6 = sp.symbols('q1 q2 q3 q4 q5 q6')

expr = (
    q1 * A * B +
    q2 * B.T * A +       # transpose
    q3 * A**2 +          # power
    # q4 * Adjoint(A) * B +       # conjugate transpose (Adjoint)
    q5 * A**-1 +         # inverse
    q6 * sp.Identity(2)  # identity matrix
)

eqns = get_matrix_coeff_equations(expr, [A, B])

for eq in eqns:
    print(eq)
