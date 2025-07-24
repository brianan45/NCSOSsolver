import sympy as sp
from sympy.logic.boolalg import BooleanFalse
from sympy import Mul
import cvxpy as cp
import numpy as np
import sys
from NC_monomial_input import get_vars_vec
from NC_poly_input import get_poly
from sp_to_cp_constraint_multiQ import sp_to_cp_constraint_multiQ
from NCget_coeffs import get_coeffs
from NCget_constraints import get_constraints
from matrix_sqrt import matrix_sqrt
from clean_value import clean_value
from sub_identity_matrix import sub_identity_matrix, get_matrix_exps

# An SOS solver that takes a set of variables, a vector v of
# monomials in those variables, a polynomial p(x) in those variables,
# and a set of constraints g(x)=0, and finds the SOS decomposition of
# p(x) or the minimum lambda such that p(x) + lambda is an SOS

def vqvgvqv(p, v, matrix_vars, Q0_sym, g_list, I_list, Qi_syms):
    def reverse_monovec(expr_matrix):
        result = expr_matrix.copy()
        for i in range(expr_matrix.rows):
            entry = expr_matrix[i, 0]
            if isinstance(entry, Mul):
                reversed_entry = Mul(*reversed(entry.args), evaluate=False)
                result[i, 0] = reversed_entry
        return result
    

    sum_vQv = ((reverse_monovec(v)).T * Q0_sym * v)[0, 0]
    for g_sym, Qi_sym in zip(g_list, Qi_syms):
        # print("\n((reverse_monovec(v)).T = ", reverse_monovec(v).T)
        # print("\ntype((reverse_monovec(v)).T * Q0_sym * v) =", type((reverse_monovec(v)).T * Qi_sym * v))
        # print("\ng_sym =", g_sym)
        # print("\ntype(g_sym) =", type(g_sym))
        # print("\n((reverse_monovec(v)).T * Qi_sym * v)[0, 0] * g_sym =", ((reverse_monovec(v)).T * Qi_sym * v)[0, 0] * g_sym)
        sum_vQv += ((reverse_monovec(v)).T * Qi_sym * v)[0, 0] * g_sym  # * ((reverse_monovec(v)).T * Qi_sym * v)[0, 0]
        # poly_expr += (v_adj.T * Qi_sym * v)[0, 0] * g_sym

    # print("poly_expr before expansion =", poly_expr)
    # poly_expr = sum(Q_sym[i, j] * sp.Adjoint(v[i]) * v[j] for i in range(m) for j in range(m))
    sum_vQv = sp.expand(sum_vQv)  
    # print("poly_expr after expansion =", poly_expr)
    # # Convert both to polynomials
    # poly_expr_poly = sp.Poly(poly_expr, *vars)
    # target_poly = sp.Poly(p, *vars)

    # print("type(vars) =", type(matrix_vars))
    # print("poly_expr - p =", poly_expr - p)

    matrix_exps = get_matrix_exps(I_list)
    print("matrix_exps =", matrix_exps)
    simplified = sub_identity_matrix(sub_identity_matrix(sp.expand(sum_vQv - p), matrix_exps),matrix_exps)
    # simplified = sub_identity_matrix(sp.expand(sum_vQv - p), matrix_exps)
    print("sum_vQv - p =", sp.expand(sum_vQv - p))
    print("simplified =", simplified)

    sol = get_coeffs(simplified, matrix_vars)
    return sol