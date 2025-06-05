import sympy as sp
from sympy import symbols, And, solve
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
    all_vars = set()
    for expr in g_list:
        all_vars.update(expr.free_symbols)
    if len(all_vars) == 1:
        inequalities = [g >= 0 for g in g_list]
        constraint_expr = And(*inequalities)

        sol = sp.reduce_inequalities(constraint_expr, vars)
        return sol != False  # reduce_inequalities returns False if infeasible

    """
    Check if the set of inequalities g_i(x) >= 0 has any feasible solution.
    """
    inequalities = [g >= 0 for g in g_list]
    constraint_expr = And(*inequalities)
    model = satisfiable(constraint_expr, algorithm="dpll2")
    return bool(model)