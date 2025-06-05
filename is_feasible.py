import sympy as sp
from sympy.logic.boolalg import And
from sympy.logic.inference import satisfiable

def is_feasible(g_list, vars):
    """
    Check if the set of inequalities g_i(x) >= 0 has any feasible solution.
    """
    inequalities = [g >= 0 for g in g_list]
    constraint_expr = And(*inequalities)
    model = satisfiable(constraint_expr, algorithm="dpll2")
    return bool(model)
