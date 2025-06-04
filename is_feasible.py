import sympy as sp
from sympy.logic.inference import satisfiable

def is_feasible(g_list, vars):
    """
    Check if the set of inequalities g_i(x) >= 0 has any feasible solution.

    Args:
        g_list (list of sympy expressions): Constraints g_i(x) >= 0
        vars (list of sympy Symbols): Variables (e.g. [x, y])

    Returns:
        bool: True if feasible region is non-empty, False otherwise
    """
    # Convert constraints to inequalities
    inequalities = [g >= 0 for g in g_list]

    # Build logical conjunction of constraints
    constraint_expr = sp.And(*inequalities)

    # Use satisfiable with domain='real'
    model = satisfiable(constraint_expr, rational=True)

    return bool(model)