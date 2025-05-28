import sympy as sp

# a function that solves for the coefficents of two equations by comparing them term by term

def get_coeffs(expr1, expr2, vars):
    """
    Match coefficients of two multivariate polynomial expressions and solve for unknowns.
    
    Parameters:
        expr1: sympy expression (e.g., a*x*y + b*y**2 + 1)
        expr2: sympy expression (e.g., x*y + 2*y**2 + 1)
        vars:  list of variables (e.g., [x, y])
    
    Returns:
        dict of solutions, or raises ValueError if polynomials are not compatible
    """
    # Convert both to multivariate Poly objects
    try:
        poly1 = sp.Poly(expr1, *vars)
        poly2 = sp.Poly(expr2, *vars)
    except sp.polys.polyerrors.PolynomialError as e:
        raise ValueError(f"Invalid input polynomial: {e}")
    
    # Get all unique monomials
    monomials = sorted(set(poly1.monoms() + poly2.monoms()))

    # Build coefficient comparison equations
    eqns = [sp.Eq(poly1.coeff_monomial(m), poly2.coeff_monomial(m)) for m in monomials]

    # Solve the system
    sol = sp.solve(eqns)

    if not sol:
        return None
    return sol