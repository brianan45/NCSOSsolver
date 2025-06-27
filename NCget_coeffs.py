import sympy as sp

def get_coeffs(lhs, rhs, vars, unknowns=None):
    """
    Solves scalar coefficients in a symbolic matrix equation of the form:
    lhs == rhs, where lhs and rhs are symbolic expressions involving
    noncommutative matrix symbols.

    Parameters:
        lhs (sympy.Expr): Left-hand side of the expression.
        rhs (sympy.Expr): Right-hand side of the expression.
        vars (dict_values or iterable): The noncommutative variables present in the expression.
        unknowns (list of sympy.Symbol, optional): Scalar unknowns to solve for.

    Returns:
        dict: Solution mapping unknown symbols to values.
    """
    # Convert dict_values to list if needed
    vars_list = list(vars)

    # Move everything to one side
    expr = sp.expand(lhs - rhs)

    # Build equations: coefficient of each noncommutative variable == 0
    equations = []
    for v in vars_list:
        coeff = expr.coeff(v)
        equations.append(sp.Eq(coeff, 0))

    # Include constant term (no noncommutative part)
    constant_term = expr
    for v in vars_list:
        constant_term = constant_term.subs(v, 0)
    if constant_term != 0:
        equations.append(sp.Eq(constant_term, 0))

    # If unknowns aren't specified, use all commutative symbols in the expression
    if unknowns is None:
        unknowns = [s for s in expr.free_symbols if s.is_commutative]

    # Solve the system
    solution = sp.solve(equations, unknowns, dict=True)

    return solution[0] if solution else None